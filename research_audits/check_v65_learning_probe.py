"""Real tiny-Qwen forward-only proof; not real 8B learning validation."""
from pathlib import Path
import tempfile,os,sys,subprocess,zipfile
from wengua_course import data as d
from wengua_course_v65 import runtime
from . import v65_learning_probe as probe,v65_token_diagnostics as tokens


def software_pack():
    rows=[]
    for i in range(4):
        q='question one '+str(i);answer='answer two'
        rows.append({'id':'fixture:'+str(i),'messages':[{'role':'user','content':q},{'role':'assistant','content':answer}],
            'answer':answer,'format':'software_fixture','task':'software_fixture','group':'fixture','source_ids':[],
            'provenance':{'software_fixture_only':True},'components':{'full_target':answer,'automatic_professional_pass':False},'teacher_forcing':i<2})
    episode={'id':'fixture_episode','system':'system one','source_ids':[],'support':[],
        'steps':[{'id':'fixture:turn'+str(i),'question':'question two '+str(i),'answer':'answer one','lesson_contract':{'software_fixture':True}} for i in range(3)]}
    return {'schema':probe.SCHEMA,'records':rows,'pairs':[],'episodes':[episode],'software_fixture_only':True,
            'summary':{'per_model_answers':7,'static_questions':4,'teacher_forced_targets':2,'interactive_turns':3}}


def main():
    runtime.configure()
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig,get_peft_model
    from wengua_course.fixtures import fixture_factory
    from src.wengua.qwen_finetune import prepare_frozen_base
    torch.set_num_threads(2);(runtime.ROOT/'local').mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='v65_learning_cpu_',dir=runtime.ROOT/'local') as tmp:
        root=Path(tmp);sources={};pack=software_pack()
        for i,label in enumerate(probe.LABELS):
            tok,base,_=fixture_factory();base=prepare_frozen_base(base,gradient_checkpointing=False)
            model=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=runtime.data.MODULES,bias='none',task_type='CAUSAL_LM'))
            torch.manual_seed(3407+i)
            for name,p in model.named_parameters():
                if 'lora_B' in name:p.data.copy_(torch.randn_like(p)*.01)
            path=root/label;model.save_pretrained(path)
            files={n:d.digest((path/n).read_bytes()) for n in ['adapter_config.json','adapter_model.safetensors']}
            sources[label]={'adapter':str(path),'adapter_sha256':files['adapter_model.safetensors'],'files':files,'variant':label,'software_fixture_only':True}
        model.eval();model.requires_grad_(False)
        c=d.read(runtime.CONFIG)|{'max_seq_length':512,'inference_max_context':512,'eval_max_new_tokens':8}
        book=runtime.profile(testing=True).book;target=pack['records'][0];encoded=tokens.encoded_target(tok,target,512)
        result=tokens.score(model,tok,target,book,c);ids=torch.tensor([encoded['full'][:-1]])
        with torch.inference_mode():logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0]
        labels=torch.tensor(encoded['full'][len(encoded['prefix']):]);reference=F.cross_entropy(logits[len(encoded['prefix'])-1:].float(),labels,reduction='none')
        assert torch.allclose(torch.tensor([r['nll'] for r in result['tokens']]),reference,atol=1e-6,rtol=1e-6)
        assert all(p.grad is None for p in model.parameters())
        model=base=None;out=root/'probe'
        assert probe.run(out,None,resume=True,testing=True,fixture_pack=pack,fixture_sources=sources,fixture_config=c)==0
        first={p.name:d.digest(p.read_bytes()) for p in (out/'answers').glob('*.jsonl')}
        for label in probe.LABELS:
            rows=probe.read_cache(out/'answers'/(label+'.jsonl'));assert len(rows)==7
            probe.validate_answers(rows,pack,label)
            assert rows[5]['question'][2]['content']==rows[4]['answer']
            assert rows[6]['question'][4]['content']==rows[5]['answer']
            assert len(probe.read_cache(out/'answers'/(label+'.tokens.jsonl')))==2
        with (out/'answers/v65_epoch2.jsonl').open('ab') as f:f.write(b'{"bad":"\xff')
        assert probe.run(out,None,resume=True,testing=True,fixture_pack=pack,fixture_sources=sources,fixture_config=c)==0
        assert all(v['cache_hit_without_model_load'] for v in d.read(out/'run_status.json')['models'].values())
        assert first=={p.name:d.digest(p.read_bytes()) for p in (out/'answers').glob('*.jsonl')}
        for source in sources.values():
            assert all(d.digest((Path(source['adapter'])/name).read_bytes())==sha for name,sha in source['files'].items())
        source=sources['v65_epoch2'];changed=Path(source['adapter'])/'adapter_config.json';original=changed.read_bytes()
        try:
            changed.write_bytes(original+b' ')
            try:probe.generate(out,'v65_epoch2',source,pack,c,__import__('time').time()+3600,testing=True)
            except ValueError:pass
            else:raise AssertionError('Changed source was accepted by completed cache')
        finally:changed.write_bytes(original)
        archive=probe.export(out,root/'review.zip')
        with zipfile.ZipFile(archive) as z:assert len([n for n in z.namelist() if n.endswith('.xlsx')])==2
        if os.name=='nt':
            env=os.environ.copy();env['WENGUA_PY']=sys.executable
            r=subprocess.run(['cmd.exe','/d','/c',str(runtime.ROOT/'check_wengua_v65_learning.cmd'),'--help'],env=env,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=60)
            assert r.returncode==0 and 'training-root' in r.stdout
        proof={'software_fixture_only':True,'two_sources_forward_only':True,'source_files_unchanged':True,
               'teacher_forced_nll_equals_native_ce':True,'no_parameter_gradients_created':True,'model_own_history_verified':True,
               'invalid_utf8_tail_repaired_without_losing_complete_answers':True,'completed_resume_without_model_load':True,
               'excel_and_export_verified':True,'source_mutation_rejected_before_cache_hit':True,
               'native_windows_cmd_verified':os.name=='nt','weight_updates':0,'actual_8b_improvement_verified':False}
        d.write(runtime.ROOT/'local/v65_learning_probe_cpu.json',proof);print(d.dumps(proof))


if __name__=='__main__':main()
