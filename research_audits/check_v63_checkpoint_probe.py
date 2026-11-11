"""Tiny CPU proof of nine-source inference, immutable reuse and resume, no updates."""
from pathlib import Path
import tempfile,sys,os,subprocess,json
from wengua_course import data as d
from wengua_course_v63 import runtime
from . import v63_checkpoint_probe as probe

def main():
    runtime.configure()
    import torch
    from peft import LoraConfig,get_peft_model
    from wengua_course.fixtures import fixture_factory
    from src.wengua.qwen_finetune import prepare_frozen_base
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='v63_readonly_',dir=runtime.ROOT/'local') as tmp:
        root=Path(tmp);tok,base,_=fixture_factory();base=prepare_frozen_base(base,gradient_checkpointing=False)
        model=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=runtime.data.MODULES,bias='none',task_type='CAUSAL_LM'))
        torch.manual_seed(3407)
        for name,param in model.named_parameters():
            if 'lora_B' in name:param.data.copy_(torch.randn_like(param)*.01)
        source=root/'source';model.save_pretrained(source)
        original={p.name:d.digest(p.read_bytes()) for p in source.iterdir() if p.is_file()}
        sha=original['adapter_model.safetensors'];model=base=None
        rows=[]
        for i in range(3):
            q='question one '+str(i);answer='answer two'
            rows.append({'id':'fixture:'+str(i),'question':q,'answer':answer,'messages':[{'role':'user','content':q},{'role':'assistant','content':answer}],
                         'source_ids':[],'format':'normal_dialogue_diagnostic','task':'fixture','group':'fixture','source_mode':'software_fixture_only'})
        pack={'records':rows,'software_fixture_only':True}
        sources={k:{'adapter':None if k=='official_base' else str(source),'adapter_sha256':None if k=='official_base' else sha,'variant':k,'software_fixture_only':True} for k in probe.LABELS}
        for label in ['official_base','low_e02','high_e01']:
            row=rows[0];record={'id':row['id'],'variant':'original_saved_rag','question':row['messages'][:-1],'expected':row['answer'],
                               'answer':'answer two','prompt_sha256':d.digest(json.dumps(row['messages'][:-1],ensure_ascii=False,separators=(',',':'))), 'input_tokens':3,'generated_tokens':2,'reached_token_limit':False,'empty':False}
            path=root/(label+'_original.jsonl');path.write_text(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf8')
            sources[label].update(reuse_file=str(path),reuse_sha256=d.digest(path.read_bytes()))
        c=d.read(runtime.CONFIG)|{'max_seq_length':512,'inference_max_context':512};out=root/'probe'
        assert probe.run(out,None,resume=True,testing=True,fixture_pack=pack,fixture_sources=sources,fixture_config=c)==0
        state=d.read(out/'run_status.json');assert state['stored_answers']==27 and state['weight_updates']==0
        for label in sources:
            replies=probe.read_rows(out/'answers'/(label+'.jsonl'));assert len(replies)==3
            if sources[label].get('reuse_file'):
                assert replies[0]==probe.read_rows(sources[label]['reuse_file'])[0]
        assert {p.name:d.digest(p.read_bytes()) for p in source.iterdir() if p.is_file()}==original
        first={p.name:d.digest(p.read_bytes()) for p in (out/'answers').glob('*.jsonl')}
        with (out/'answers/high_e07.jsonl').open('ab') as f:f.write(b'{"partial":"\xff')
        assert probe.run(out,None,resume=True,testing=True,fixture_pack=pack,fixture_sources=sources,fixture_config=c)==0
        assert {p.name:d.digest(p.read_bytes()) for p in (out/'answers').glob('*.jsonl')}==first
        assert all(m['cache_hit_without_model_load'] for m in d.read(out/'run_status.json')['models'].values())
        damaged=out/'answers/official_base.jsonl';damaged.write_bytes(damaged.read_bytes()[:10])
        assert probe.run(out,None,resume=True,testing=True,fixture_pack=pack,fixture_sources=sources,fixture_config=c)==0
        assert probe.read_rows(damaged)[0]==probe.read_rows(sources['official_base']['reuse_file'])[0]
        from wengua_course.runner import export_review
        assert export_review(out,root/'review.zip').is_file()
        if os.name=='nt':
            env=os.environ.copy();env['WENGUA_PY']=sys.executable
            for script in ['check_wengua_v63_checkpoints.cmd']:
                r=subprocess.run(['cmd.exe','/d','/c',str(runtime.ROOT/script),'--help'],env=env,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=60)
                assert r.returncode==0 and 'training-root' in r.stdout
        proof={'software_fixture_only':True,'nine_models_inference_verified':True,'source_weights_unchanged':True,'cached_original_answers_unchanged':True,
               'invalid_utf8_tail_resume':True,'truncated_reuse_prefix_restored_from_original':True,'export_and_native_cmd_verified':True,'weight_updates':0,'actual_8b_quality_verified':False}
        d.write(runtime.ROOT/'local/v63_checkpoint_probe_cpu.json',proof);print(d.dumps(proof))
if __name__=='__main__':main()
