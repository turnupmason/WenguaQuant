"""Actual tiny-CPU zero initialization, two-rate loop and tensor-exact resume."""
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch
from copy import deepcopy
import tempfile,time,os,sys,subprocess
from wengua_course import data as d
from . import runtime,initialization,runner

def software_pack():
    from wengua_course_v61.check_training import software_pack as previous
    p=deepcopy(previous());p['schema']=runtime.SCHEMA
    train=[r for r in p['records'] if r['split']=='train'][:4]
    val=[r for r in p['records'] if r['split']=='validation'][:2]
    p['records']=train+val;p['known_regression_rows']=[];p['actual_rag_rows']=[]
    p['summary']={'software_fixture_only':True};return p

def main():
    runtime.configure()
    import torch
    from transformers import TrainerCallback
    from safetensors.torch import load_file
    from peft.utils.save_and_load import get_peft_model_state_dict
    from wengua_course.fixtures import fixture_factory
    from src.wengua.qwen8b_attention_v1 import attention_route
    from src.wengua.qwen8b_loss_v1 import memory_trainer
    from .process import launch
    torch.set_num_threads(2);(runtime.ROOT/'local').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='v62_cpu_',dir=runtime.ROOT/'local') as tmp:
        root=Path(tmp);tok,_,_=fixture_factory();small=dict(max_seq_length=512,inference_max_context=512,eval_max_new_tokens=8,smoke_steps=2,gradient_accumulation_steps=1,save_steps=1,development_generation_rows=2,num_train_epochs=2)
        p=runtime.profile(testing=True);pack=software_pack()
        seed=d.read(runtime.CONFIG)|small|{'initialization_only':True};job=p.engine.open_job(root/'init/trials/seed',seed,resume=True)
        created=launch(p,job,seed,time.time()+3600,'initialize')
        zero=initialization.verify_zero(created['adapter'],testing=True)
        assert zero['weight_updates']==0 and zero['all_lora_B_zero']
        assert initialization.create(p,job,seed,factory=fixture_factory)['adapter']==created['adapter']
        parent=initialization.snapshot(root/'fresh_parent',created['adapter'],testing=True)
        original_config=runner.config
        def short(parent,root,rate):return original_config(parent,root,rate)|small
        c=short(parent,root,1e-5)
        class Fresh(TrainerCallback):
            seen=False
            def on_train_begin(self,args,state,control,**kw):
                got=get_peft_model_state_dict(kw['model'])
                assert all(not bool(v.count_nonzero()) for k,v in got.items() if 'lora_B' in k)
                assert not kw['optimizer'].state;self.seen=True
        direct=p.engine.open_job(root/'direct/trials/full',c,resume=True);runtime.prepare(direct,c,tokenizer=tok,pack=pack,testing=True);check=Fresh()
        with attention_route(testing=True),memory_trainer(128):completed=p.engine.fit(direct,c,time.time()+3600,cpu_factory=fixture_factory,extra_callbacks=(check,))
        assert check.seen and completed['state']=='complete' and len(completed['epochs'])==2
        assert completed['fresh_official_base'] is True and completed['parent_weights_reused'] is False
        weights=load_file(str(Path(completed['last_adapter'])/'adapter_model.safetensors'))
        assert any(bool(v.count_nonzero()) for k,v in weights.items() if 'lora_B' in k)
        try:initialization.verify_zero(completed['last_adapter'],testing=True)
        except ValueError:pass
        else:raise AssertionError('Trained weights accepted as fresh')
        resumed=p.engine.open_job(root/'direct/trials/resumed',c,resume=True);runtime.prepare(resumed,c,tokenizer=tok,pack=pack,testing=True)
        paused=launch(p,resumed,c,time.time()+3600,'fit',stop_step=1);assert paused['state']=='paused_training'
        restored=launch(p,resumed,c,time.time()+3600,'fit')
        b=load_file(str(Path(restored['last_adapter'])/'adapter_model.safetensors'))
        assert all(torch.equal(weights[k],b[k]) for k in weights)
        real_profile=runtime.profile;real_prepare=runtime.prepare
        with patch.object(runtime,'profile',lambda **kw:real_profile(testing=True)),patch.object(runtime,'verify_data',lambda:pack),patch.object(runtime,'prepare',lambda job,c,**kw:real_prepare(job,c,tokenizer=tok,pack=pack,testing=True)),patch.object(runner,'config',short):
            args=Namespace(app_root=runtime.app_root(),output=root/'loop',hours=1,resume=True,resume_from_checkpoint=None)
            assert runner.run(args)==0
            state=d.read(args.output/'run_status.json');assert state['state']=='complete_review_required' and len(state['trials'])==2
            assert all(t['state']=='complete' and len(t['fit']['epochs'])==2 for t in state['trials'].values())
            assert len({t['config']['book_parent_provenance_sha256'] for t in state['trials'].values()})==1
            assert state['initialization']['zero_update_proof']['all_lora_B_zero']
            assert runner.run(args)==0
            from wengua_course.runner import export_review
            assert export_review(args.output,root/'review.zip').is_file()
        if os.name=='nt':
            env=os.environ.copy();env['WENGUA_PY']=sys.executable
            response=subprocess.run(['cmd.exe','/d','/c',str(runtime.ROOT/'train_wengua_v62.cmd'),'--help'],env=env,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=60)
            assert response.returncode==0 and 'restore' in response.stdout
        proof={'software_fixture_only':True,'official_zero_update_lora_verified':True,'nonzero_trained_weights_rejected_as_fresh':True,'new_optimizer_empty':True,'fresh_process_exact_resume_tensors_equal':True,'two_rates_two_epochs_complete':True,'export_and_rerun_verified':True,'actual_8b_cuda_quality_verified':False}
        d.write(runtime.ROOT/'local/course_v62_cpu_verification.json',proof);print(d.dumps(proof))
if __name__=='__main__':main()
