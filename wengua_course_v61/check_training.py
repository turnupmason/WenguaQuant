"""Actual tiny CPU inheritance/resume and finite loop. Not 8B quality validation."""
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch
from copy import deepcopy
import tempfile,time,os,sys,subprocess
from wengua_course import data as d
from . import runtime,initialization,runner

def software_pack():
    from wengua_course_v60.check_training import software_pack as previous
    p=deepcopy(previous());p['schema']=runtime.SCHEMA
    p['actual_rag_rows']=[r for r in p['records'] if r['split']=='validation' and r['stage']=='application'][:4]
    return p

def main():
    runtime.configure()
    import torch
    from transformers import TrainerCallback
    from safetensors.torch import load_file
    from peft.utils.save_and_load import get_peft_model_state_dict
    from wengua_course.fixtures import fixture_factory
    from wengua_course_v53 import runtime as seed,initialization as si
    from wengua_course_v55 import runtime as old,initialization as oi
    from wengua_course_v55.check_training import software_pack as old_pack
    from src.wengua.qwen8b_attention_v1 import attention_route
    from src.wengua.qwen8b_loss_v1 import memory_trainer
    from .process import launch
    torch.set_num_threads(2);(runtime.ROOT/'local').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='v61_cpu_',dir=runtime.ROOT/'local') as tmp:
        root=Path(tmp);tok,_,_=fixture_factory();small=dict(max_seq_length=512,inference_max_context=512,eval_max_new_tokens=8,smoke_steps=2,gradient_accumulation_steps=1,save_steps=1,development_generation_rows=4,foundation_repeat=1,application_repeat=1,num_train_epochs=2)
        sp=seed.profile(testing=True);sc=d.read(seed.CONFIG)|small|dict(initialization_only=True);sj=sp.engine.open_job(root/'seed/trials/init',sc,resume=True);created=si.create(sp,sj,sc,factory=fixture_factory)
        snap=oi.snapshot(root/'old_parent',created['adapter'],testing=True);op=old.profile(testing=True);oc=d.read(old.CONFIG)|small;oc.update(book_parent_snapshot=str(snap),book_parent_provenance_sha256=d.digest((snap.parent/'parent_provenance.json').read_bytes()))
        oj=op.engine.open_job(root/'old/trials/trained',oc,resume=True);old.prepare(oj,oc,tokenizer=tok,pack=old_pack(),testing=True)
        with attention_route(testing=True),memory_trainer(128):trained=op.engine.fit(oj,oc,time.time()+3600,cpu_factory=fixture_factory)
        source=Path(trained['last_adapter']);weights=load_file(str(source/'adapter_model.safetensors'));assert any(t.count_nonzero() for k,t in weights.items() if 'lora_B' in k)
        p=runtime.profile(testing=True);pack=software_pack();parent=initialization.snapshot(root/'parent',source,testing=True)
        original_config=runner.config
        def short(parent,root,rate):return original_config(parent,root,rate)|small
        c=short(parent,root,1e-5)
        class Inherit(TrainerCallback):
            seen=False
            def on_train_begin(self,args,state,control,**kw):
                got=get_peft_model_state_dict(kw['model']);assert all(torch.equal(got[k].cpu(),v) for k,v in weights.items());assert not kw['optimizer'].state;self.seen=True
        direct=p.engine.open_job(root/'direct/trials/full',c,resume=True);runtime.prepare(direct,c,tokenizer=tok,pack=pack,testing=True);check=Inherit()
        with attention_route(testing=True),memory_trainer(128):completed=p.engine.fit(direct,c,time.time()+3600,cpu_factory=fixture_factory,extra_callbacks=(check,))
        assert check.seen and completed['state']=='complete' and len(completed['epochs'])==2
        resumed=p.engine.open_job(root/'direct/trials/resumed',c,resume=True);runtime.prepare(resumed,c,tokenizer=tok,pack=pack,testing=True)
        paused=launch(p,resumed,c,time.time()+3600,'fit',stop_step=1);assert paused['state']=='paused_training'
        restored=launch(p,resumed,c,time.time()+3600,'fit');a=load_file(str(Path(completed['last_adapter'])/'adapter_model.safetensors'));b=load_file(str(Path(restored['last_adapter'])/'adapter_model.safetensors'));assert all(torch.equal(a[k],b[k]) for k in a)
        assert initialization.inspect(restored['last_adapter'],testing=True)['schema']==runtime.SCHEMA
        real_profile=runtime.profile;real_prepare=runtime.prepare
        with patch.object(runtime,'profile',lambda **kw:real_profile(testing=True)),patch.object(runtime,'verify_data',lambda:pack),patch.object(runtime,'prepare',lambda job,c,**kw:real_prepare(job,c,tokenizer=tok,pack=pack,testing=True)),patch.object(initialization,'default_source',lambda:source),patch.object(runner,'config',short):
            args=Namespace(app_root=runtime.app_root(),output=root/'loop',hours=1,resume=True,resume_from_checkpoint=None)
            assert runner.run(args)==0
            state=d.read(args.output/'run_status.json');assert state['state']=='complete_review_required' and len(state['trials'])==2
            assert all(t['state']=='complete' and len(t['fit']['epochs'])==2 for t in state['trials'].values())
            assert len({t['config']['book_parent_provenance_sha256'] for t in state['trials'].values()})==1
            assert d.read(args.output/'retained_best.json')['adapter']==str(source)
            assert runner.run(args)==0
            from wengua_course.runner import export_review
            assert export_review(args.output,root/'review.zip').is_file()
        if os.name=='nt':
            env=os.environ.copy();env['WENGUA_PY']=sys.executable
            answer=subprocess.run(['cmd.exe','/d','/c',str(runtime.ROOT/'train_wengua_v61.cmd'),'--help'],env=env,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=60)
            assert answer.returncode==0 and 'restore' in answer.stdout and 'export' in answer.stdout
        proof={'software_fixture_only':True,'nonzero_parent_inherited':True,'new_phase_optimizer_empty':True,'fresh_process_exact_resume_tensors_equal':True,'two_rates_two_epochs_complete':True,'original_retained_adapter_preserved':True,'export_and_rerun_verified':True,'actual_8b_cuda_quality_verified':False}
        d.write(runtime.ROOT/'local/course_v61_cpu_verification.json',proof);print(d.dumps(proof))
if __name__=='__main__':main()
