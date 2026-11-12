"""Real tiny-Qwen CPU recovery, not an 8B/GPU quality or memory test."""
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch
import tempfile,time,os,sys,subprocess
from wengua_course import data as d
from wengua_course_v67 import runtime,runner,evaluation,initialization
from wengua_course_v67.check_training import make_parent,software_pack
from . import v67_evaluation as recovery


def main():
    runtime.configure();os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    from peft import PeftModel
    from src.wengua.qwen_finetune import prepare_frozen_base
    from src.wengua.qwen8b_attention_v1 import attention_route
    from wengua_course.fixtures import fixture_factory
    from wengua_course_v67 import loss,process
    torch.set_num_threads(2);(runtime.ROOT/'local').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='v67_eval_recovery_',dir=runtime.ROOT/'local') as tmp:
        root=Path(tmp);source,small=make_parent(root);pack=software_pack();tok,_,_=fixture_factory();p=runtime.profile(testing=True)
        real_config=runner.config;real_prepare=runtime.prepare
        def short(parent,root):return real_config(parent,root)|small
        tok,base,_=fixture_factory();base=prepare_frozen_base(base,gradient_checkpointing=False)
        model=PeftModel.from_pretrained(base,str(source),is_trainable=False);model.eval();model.requires_grad_(False)
        target=pack['fixture_evaluation_pack']['retention']['records'][0]
        with attention_route(testing=True):old=evaluation.one_answer(model,tok,target,p.book,short(initialization.snapshot(root/'first_parent',source,testing=True),root),'ordinary_ce')
        previous=root/'prior.jsonl';previous.write_bytes((d.dumps(old)+'\r\n').encode());pack['fixture_parent_source']=str(previous);base=model=None
        real_one=evaluation.one_answer
        def interrupted_answer(model,tok,target,book,c,tag):
            if tag=='epoch_01' and target['id']=='fixture:next':raise RuntimeError('CUDA error: out of memory (injected software fixture)')
            return real_one(model,tok,target,book,c,tag)
        def direct(p,job,c,deadline,operation,*,probe=False,checkpoint=None,best=None,**kw):
            with attention_route(testing=True),loss.memory_trainer(128,factor=1):
                if operation=='fit':return p.engine.fit(job,c,deadline,probe=probe,cpu_factory=fixture_factory)
                return evaluation.evaluate_adapter(p,job,c,deadline,best=best,factory=fixture_factory)
        args=Namespace(app_root=runtime.app_root(),output=root/'run',hours=1,resume=True,resume_from_checkpoint=None)
        with patch.object(runtime,'profile',lambda **kw:runtime_profile(testing=True)),patch.object(runtime,'verify_data',lambda:pack),patch.object(runtime,'prepare',lambda job,c,**kw:real_prepare(job,c,tokenizer=tok,pack=pack,testing=True)),patch.object(initialization,'default_source',lambda:source),patch.object(runner,'config',short),patch.object(process,'launch',direct),patch.object(evaluation,'one_answer',interrupted_answer):
            try:runner.run(args)
            except RuntimeError as error:assert recovery.is_cuda_oom(error)
            else:raise AssertionError('Injected evaluation OOM did not stop the original run')
        job=args.output/'trials'/runner.TRIAL;path=job/'generations/epoch_01.jsonl';prefix=path.read_bytes();saved=evaluation.read_cache(path)
        assert len(saved)==3 and saved[-1]['id']=='fixture:first'
        prior={str(f):d.digest(f.read_bytes()) for f in args.output.rglob('*') if f.is_file() and ('adapters' in f.parts or 'trainer' in f.parts or f.name=='stage_status.json')}
        with patch.object(runtime,'profile',lambda **kw:runtime_profile(testing=True)):
            assert recovery.run(args.output,testing=True,limit=1)==0
            assert recovery.run(args.output,testing=True,limit=1)==0
        assert path.read_bytes().startswith(prefix)
        for tag in ['epoch_01','epoch_02']:
            rows=evaluation.read_cache(job/'generations'/(tag+'.jsonl'));assert len(rows)==4
            assert rows[-1]['question'][2]['content']==rows[-2]['answer'];assert not rows[-1]['provenance']['gold_prior_answers_injected']
        assert all(d.digest(Path(f).read_bytes())==sha for f,sha in prior.items())
        assert d.read(args.output/'run_status.json')['state']=='complete_review_required'
        requests=[d.read(f) for f in (job/'worker_calls').glob('*.json') if len(f.stem.split('.'))==1]
        assert len(requests)==5 and all(r['operation']=='evaluate' for r in requests)
        assert all(d.read(f)['training_updates']==0 for f in (job/'worker_calls').glob('*.result.json'))
        if os.name=='nt':
            env=os.environ.copy();env['WENGUA_PY']=sys.executable
            result=subprocess.run(['cmd.exe','/d','/c',str(runtime.ROOT/'resume_wengua_v67_evaluation.cmd'),'--help'],env=env,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=60)
            assert result.returncode==0 and 'check' in result.stdout
        proof={'software_fixture_only':True,'original_eval_oom_injected':True,'trained_adapter_and_checkpoint_bytes_unchanged':True,
               'existing_answer_prefix_preserved':True,'actual_own_history_survives_process_boundary':True,'completed_run_no_new_workers':True,
               'fresh_process_per_answer_chunk':True,'same_canonical_generation_function':True,'recovery_training_updates':0,
               'native_windows_cmd_help':os.name=='nt','actual_gpu_memory_fix_verified':False,'actual_8b_improvement_verified':False}
        d.write(runtime.ROOT/'local/v67_evaluation_recovery_cpu_verification.json',proof);print(d.dumps(proof))


runtime_profile=runtime.profile
if __name__=='__main__':main()
