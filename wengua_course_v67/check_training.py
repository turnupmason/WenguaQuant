"""Tiny real CPU checks of inheritance, view deduplication and exact resume."""
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch
from copy import deepcopy
import tempfile,time,os,sys,subprocess,json
from wengua_course import data as d
from . import runtime,initialization,runner,evaluation,loss


def software_pack():
    from wengua_course_v66.check_training import software_pack as previous
    p=previous();p['schema']=runtime.SCHEMA
    for i,r in enumerate(p['records']):
        r['messages'][1]['content']='question '+('one '*i)+'two';r['question']=r['messages'][1]['content']
    duplicate=deepcopy(p['records'][0]);duplicate['id']='software:duplicate_supervised_source';p['records'].append(duplicate)
    first=deepcopy(p['records'][0]);first['components']={'automatic_professional_pass':False};first['provenance']={}
    extra=deepcopy(first);extra['id']='software:new_static';extra['messages'][1]['content']='question two two'
    episode={'id':'fixture_update','system':'question','source_ids':first['source_ids'],'support':[],
       'steps':[{'id':'fixture:first','question':'question one','answer':'answer one','lesson_contract':{}},
                {'id':'fixture:next','question':'question two','answer':'answer two','lesson_contract':{}}]}
    p['fixture_evaluation_pack']={'schema':runtime.SCHEMA,'retention':{'records':[first],'episodes':[]},
      'new':{'records':[extra],'episodes':[episode]},'summary':{'retention_answers':1,'new_answers':3,'per_model_answers':4}}
    return p


def make_parent(root):
    from wengua_course_v66.check_training import make_v65_parent,software_pack as old_pack
    from wengua_course_v66 import runtime as rt,runner as run,initialization as start
    from wengua_course.fixtures import fixture_factory
    from src.wengua.qwen8b_attention_v1 import attention_route
    source,small=make_v65_parent(root);parent=start.snapshot(root/'v66_parent',source,testing=True)
    small=small|{'num_train_epochs':1,'generation_epochs':[1]};c=run.config(parent,root,'ordinary_ce')|small
    p=rt.profile(testing=True);tok,_,_=fixture_factory();job=p.engine.open_job(root/'v66/trials/ordinary',c,resume=True)
    rt.prepare(job,c,tokenizer=tok,pack=old_pack(),testing=True)
    with attention_route(testing=True),loss.memory_trainer(128,factor=1):result=p.engine.fit(job,c,time.time()+3600,cpu_factory=fixture_factory)
    return Path(result['last_adapter']),small|{'num_train_epochs':2,'generation_epochs':[1,2]}


def main():
    runtime.configure();os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    from peft import PeftModel
    from peft.utils.save_and_load import get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import TrainerCallback
    from wengua_course.fixtures import fixture_factory
    from src.wengua.qwen_finetune import prepare_frozen_base
    from src.wengua.qwen8b_attention_v1 import attention_route
    from .process import launch
    torch.set_num_threads(2);(runtime.ROOT/'local').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='v67_cpu_',dir=runtime.ROOT/'local') as tmp:
        root=Path(tmp);source,small=make_parent(root);before=d.digest((source/'adapter_model.safetensors').read_bytes())
        weights=load_file(str(source/'adapter_model.safetensors'));assert any(bool(v.count_nonzero()) for k,v in weights.items() if 'lora_B' in k)
        parent=initialization.snapshot(root/'parent',source,testing=True);pack=software_pack();tok,_,_=fixture_factory();p=runtime.profile(testing=True)
        original_config=runner.config
        def short(parent,root):return original_config(parent,root)|small
        c=short(parent,root)
        # Generate the actual old source response once, then preserve exact CRLF
        # bytes when seeding the read-only baseline in the full loop.
        target=pack['fixture_evaluation_pack']['retention']['records'][0];tok,base,_=fixture_factory();base=prepare_frozen_base(base,gradient_checkpointing=False)
        model=PeftModel.from_pretrained(base,str(source),is_trainable=False);model.eval();model.requires_grad_(False)
        with attention_route(testing=True):row=evaluation.one_answer(model,tok,target,p.book,c,'ordinary_ce')
        seeded=root/'old_parent.jsonl';seeded.write_bytes((d.dumps(row)+'\r\n').encode('utf8'));pack['fixture_parent_source']=str(seeded);model=base=None
        class Inherited(TrainerCallback):
            seen=False
            def on_train_begin(self,args,state,control,**kw):
                actual=get_peft_model_state_dict(kw['model']);assert all(torch.equal(weights[k],actual[k].detach().cpu()) for k in weights)
                assert not kw['optimizer'].state;self.seen=True
        job=p.engine.open_job(root/'direct/trials/full',c,resume=True);meta=runtime.prepare(job,c,tokenizer=tok,pack=pack,testing=True)
        assert meta['training_rows']==5 and meta['deduplicated_supervised_views']==1 and meta['decision_tokens']==0
        check=Inherited()
        with attention_route(testing=True),loss.memory_trainer(128,factor=1) as audit:result=p.engine.fit(job,c,time.time()+3600,cpu_factory=fixture_factory,extra_callbacks=(check,))
        assert check.seen and result['steps']==6 and [e['steps'] for e in result['epochs']]==[3,6] and audit['weighted_training_calls']==0
        final=load_file(str(Path(result['last_adapter'])/'adapter_model.safetensors'));assert any(not torch.equal(weights[k],final[k]) for k in weights)
        resumed=p.engine.open_job(root/'direct/trials/resumed',c,resume=True);runtime.prepare(resumed,c,tokenizer=tok,pack=pack,testing=True)
        stopped=launch(p,resumed,c,time.time()+3600,'fit',stop_step=1);assert stopped['state']=='paused_training'
        restored=launch(p,resumed,c,time.time()+3600,'fit');actual=load_file(str(Path(restored['last_adapter'])/'adapter_model.safetensors'))
        assert all(torch.equal(final[k],actual[k]) for k in final)
        real_profile=runtime.profile;real_prepare=runtime.prepare
        with patch.object(runtime,'profile',lambda **kw:real_profile(testing=True)),patch.object(runtime,'verify_data',lambda:pack),patch.object(runtime,'prepare',lambda job,c,**kw:real_prepare(job,c,tokenizer=tok,pack=pack,testing=True)),patch.object(initialization,'default_source',lambda:source),patch.object(runner,'config',short):
            args=Namespace(app_root=runtime.app_root(),output=root/'loop',hours=1,resume=True,resume_from_checkpoint=None)
            assert runner.run(args)==0;state=d.read(args.output/'run_status.json');trial=state['trial']
            assert state['state']=='complete_review_required' and trial['fit']['steps']==6 and trial['fit']['parent_weights_reused']
            assert set(trial['epoch_evaluations'])=={'epoch_01','epoch_02'} and state['parent_reuse']['reused_rows']==1
            path=args.output/'trials/parent_baseline/generations/parent.jsonl';assert path.read_bytes().startswith(seeded.read_bytes())
            for tag in ['epoch_01','epoch_02']:
                answers=evaluation.read_cache(args.output/'trials'/runner.TRIAL/'generations'/(tag+'.jsonl'));assert len(answers)==4
                assert answers[-1]['question'][2]['content']==answers[-2]['answer'];assert not answers[-1]['provenance']['gold_prior_answers_injected']
            files={str(f):d.digest(f.read_bytes()) for f in args.output.glob('trials/*/generations/*.jsonl')}
            assert runner.run(args)==0 and all(d.digest(Path(f).read_bytes())==h for f,h in files.items())
            from wengua_course.runner import export_review
            assert export_review(args.output,root/'review.zip').is_file()
        assert d.digest((source/'adapter_model.safetensors').read_bytes())==before
        if os.name=='nt':
            env=os.environ.copy();env['WENGUA_PY']=sys.executable
            child=subprocess.run(['cmd.exe','/d','/c',str(runtime.ROOT/'train_wengua_v67.cmd'),'--help'],env=env,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=60)
            assert child.returncode==0 and 'check' in child.stdout
        proof={'software_fixture_only':True,'loaded_nonzero_v66_parent_exactly':True,'old_source_bytes_unchanged':True,'new_optimizer_empty':True,
             'ordinary_loss_only':True,'duplicate_supervised_prefix_merged':True,'continuous_two_epochs_preserved':True,
             'partial_accumulation_exact_process_resume':True,'parent_actual_answer_crlf_bytes_reused':True,'both_epoch_actual_own_history_saved':True,
             'completed_run_does_not_repeat_training':True,'native_windows_cmd_help':os.name=='nt','actual_8b_improvement_verified':False}
        d.write(runtime.ROOT/'local/course_v67_cpu_verification.json',proof);print(d.dumps(proof))


if __name__=='__main__':main()
