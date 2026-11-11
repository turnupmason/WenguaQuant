"""Actual tiny CPU checks of both sources, parity, training and exact resume."""
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch
import os,tempfile,time,sys,subprocess
from wengua_course import data as d
from . import runtime,initialization,runner


def software_pack():
    from wengua_course_v64.check_training import software_pack as previous
    p=previous();p['schema']=runtime.SCHEMA;p['fixture_parent_rows']={'v63_trained':p['fixture_parent_rows'],'official_zero':[]}
    p['actual_rag_rows']=[];p['known_regression_rows']=[]
    return p


def main():
    runtime.configure();os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    from safetensors.torch import load_file
    from wengua_course.fixtures import fixture_factory
    from wengua_course_v64.check_training import make_parent
    from src.wengua.qwen8b_attention_v1 import attention_route
    from src.wengua.qwen8b_loss_v1 import memory_trainer
    from .process import launch
    torch.set_num_threads(2);(runtime.ROOT/'local').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='v65_cpu_',dir=runtime.ROOT/'local') as tmp:
        root=Path(tmp);trained,small=make_parent(root)
        zero=root/'seed/trials/seed/cycle_01/adapters/step-0';sources={'v63_trained':trained,'official_zero':zero}
        before={v:d.digest((path/'adapter_model.safetensors').read_bytes()) for v,path in sources.items()}
        assert not any(bool(v.count_nonzero()) for k,v in load_file(str(zero/'adapter_model.safetensors')).items() if 'lora_B' in k)
        pack=software_pack();tok,_,_=fixture_factory();p=runtime.profile(testing=True);parent=initialization.snapshot(root/'parent',trained,testing=True)
        original_config=runner.config
        def short(parent,root,variant):return original_config(parent,root,variant)|small|{'learning_rate':2e-5}
        c=short(parent,root,'v63_trained');full=p.engine.open_job(root/'direct/trials/full',c,resume=True)
        runtime.prepare(full,c,tokenizer=tok,pack=pack,testing=True)
        with attention_route(testing=True),memory_trainer(128):finished=p.engine.fit(full,c,time.time()+3600,cpu_factory=fixture_factory)
        assert finished['state']=='complete' and finished['steps']==6
        final=load_file(str(Path(finished['last_adapter'])/'adapter_model.safetensors'))
        resumed=p.engine.open_job(root/'direct/trials/resumed',c,resume=True);runtime.prepare(resumed,c,tokenizer=tok,pack=pack,testing=True)
        stopped=launch(p,resumed,c,time.time()+3600,'fit',stop_step=1);assert stopped['state']=='paused_training'
        restored=launch(p,resumed,c,time.time()+3600,'fit');actual=load_file(str(Path(restored['last_adapter'])/'adapter_model.safetensors'))
        assert all(torch.equal(final[k],actual[k]) for k in final)
        real_profile=runtime.profile;real_prepare=runtime.prepare
        with patch.object(runtime,'profile',lambda **kw:real_profile(testing=True)),patch.object(runtime,'verify_data',lambda:pack),patch.object(runtime,'prepare',lambda job,c,**kw:real_prepare(job,c,tokenizer=tok,pack=pack,testing=True)),patch.object(initialization,'default_source',lambda variant:sources[variant]),patch.object(runner,'config',short):
            args=Namespace(app_root=runtime.app_root(),output=root/'loop',hours=1,resume=True,resume_from_checkpoint=None)
            assert runner.run(args)==0;state=d.read(args.output/'run_status.json')
            assert state['state']=='complete_review_required' and set(state['trials'])==set(runner.VARIANTS)
            assert state['trials']['v63_trained']['fit']['parent_weights_reused']
            assert state['trials']['official_zero']['fit']['fresh_official_base']
            for variant in runner.VARIANTS:
                trial=args.output/'trials'/variant
                assert [e['steps'] for e in state['trials'][variant]['fit']['epochs']]==[3,6]
                assert {x.name for x in (trial/'generations').glob('epoch_*_review.jsonl')}=={'epoch_01_review.jsonl','epoch_02_review.jsonl'}
                proofs=[d.read(f) for f in (trial/'numerical_checks').glob('*.json')]
                assert proofs and all(not f['actual_cuda'] and not f['weights_updated_by_check'] for f in proofs)
                assert all(f['all_non_thinking_prefixes_equal'] for f in [d.read(trial/'prepared/training_generation_protocol.json')])
            assert runner.run(args)==0
            from wengua_course.runner import export_review
            assert export_review(args.output,root/'review.zip').is_file()
        assert all(d.digest((path/'adapter_model.safetensors').read_bytes())==before[v] for v,path in sources.items())
        if os.name=='nt':
            env=os.environ.copy();env['WENGUA_PY']=sys.executable
            child=subprocess.run(['cmd.exe','/d','/c',str(runtime.ROOT/'train_wengua_v65.cmd'),'--help'],env=env,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=60)
            assert child.returncode==0 and 'check' in child.stdout
        proof={'software_fixture_only':True,'matched_trained_and_official_sources_verified':True,'old_source_tensors_unchanged':True,
               'two_epochs_each_source':True,'all_epoch_answers_saved':True,'exact_optimizer_rng_resume_tensors_equal':True,
               'training_generation_prefix_masks_verified':True,'cached_vs_full_prefix_numerics_verified':True,
               'native_windows_cmd_verified':os.name=='nt','actual_8b_cuda_semantic_improvement_verified':False}
        d.write(runtime.ROOT/'local/course_v65_cpu_verification.json',proof);print(d.dumps(proof))


if __name__=='__main__':main()
