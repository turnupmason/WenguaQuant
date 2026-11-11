"""Actual tiny CPU inheritance, same-parent loop and tensor-exact process resume."""
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch
from copy import deepcopy
import tempfile,time,os,sys,subprocess,json
from wengua_course import data as d
from . import runtime,initialization,runner


def software_pack():
    from wengua_course_v63.check_training import software_pack as prior
    p=prior();p['schema']=runtime.SCHEMA
    train=[r for r in p['records'] if r['split']=='train'];val=[r for r in p['records'] if r['split']=='validation']
    extra=deepcopy(train[0]);extra['id']='software:partial_accumulation_tail';extra['messages'][1]['content']+=' one';train.append(extra)
    p['records']=train+val
    p.update(original_recall_rows=train[:1],known_development_rows=[],known_regression_rows=[],dispositions=[])
    question=train[0]['messages'][:-1]
    p['fixture_parent_rows']=[{'id':train[0]['id'],'variant':'original_parent_answer','question':question,'expected':train[0]['answer'],
        'answer':'answer original source','prompt_sha256':d.digest(json.dumps(question,ensure_ascii=False,separators=(',',':'))),
        'input_tokens':4,'generated_tokens':3,'reached_token_limit':False,'empty':False}]
    return p


def make_parent(root):
    """Create a real nonzero seven-projection V63 formal adapter, not a mock load."""
    from wengua_course_v63 import runtime as before,initialization as start,runner as run_before
    from wengua_course_v63.check_training import software_pack as old_pack
    from wengua_course.fixtures import fixture_factory
    from src.wengua.qwen8b_attention_v1 import attention_route
    from src.wengua.qwen8b_loss_v1 import memory_trainer
    p=before.profile(testing=True);tok,_,_=fixture_factory()
    small={'max_seq_length':512,'inference_max_context':512,'eval_max_new_tokens':8,'smoke_steps':2,'gradient_accumulation_steps':2,'save_steps':1,'development_generation_rows':2,'num_train_epochs':2}
    seed=d.read(before.CONFIG)|small|{'initialization_only':True}
    job=p.engine.open_job(root/'seed/trials/seed',seed,resume=True)
    created=start.create(p,job,seed,factory=fixture_factory)
    parent=start.snapshot(root/'fresh_parent',created['adapter'],testing=True)
    c=run_before.config(parent,root,5e-5)|small;job=p.engine.open_job(root/'v63/trials/learn',c,resume=True)
    before.prepare(job,c,tokenizer=tok,pack=old_pack(),testing=True)
    with attention_route(testing=True),memory_trainer(128):result=p.engine.fit(job,c,time.time()+3600,cpu_factory=fixture_factory)
    return Path(result['last_adapter']),small


def main():
    runtime.configure()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    from .dataset import placements
    assert d.digest(d.dumps(placements(runtime.app_root())))==d.read(runtime.RELEASE)['literal_placement_sha256']
    import torch
    from safetensors.torch import load_file
    from peft.utils.save_and_load import get_peft_model_state_dict
    from transformers import TrainerCallback
    from wengua_course.fixtures import fixture_factory
    from src.wengua.qwen8b_attention_v1 import attention_route
    from src.wengua.qwen8b_loss_v1 import memory_trainer
    from .process import launch
    torch.set_num_threads(2);(runtime.ROOT/'local').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='v64_cpu_',dir=runtime.ROOT/'local') as tmp:
        root=Path(tmp);source,small=make_parent(root)
        source_sha=d.digest((source/'adapter_model.safetensors').read_bytes());weights=load_file(str(source/'adapter_model.safetensors'))
        assert any(bool(v.count_nonzero()) for k,v in weights.items() if 'lora_B' in k)
        parent=initialization.snapshot(root/'parent',source,testing=True);tok,_,_=fixture_factory();pack=software_pack();p=runtime.profile(testing=True)
        original_config=runner.config
        def short(parent,root,rate):return original_config(parent,root,rate)|small
        c=short(parent,root,1e-5)
        class Inherited(TrainerCallback):
            seen=False
            def on_train_begin(self,args,state,control,**kw):
                got=get_peft_model_state_dict(kw['model']);assert all(torch.equal(weights[k],got[k].detach().cpu()) for k in weights)
                assert not kw['optimizer'].state;self.seen=True
        full=p.engine.open_job(root/'direct/trials/full',c,resume=True);runtime.prepare(full,c,tokenizer=tok,pack=pack,testing=True);check=Inherited()
        with attention_route(testing=True),memory_trainer(128):result=p.engine.fit(full,c,time.time()+3600,cpu_factory=fixture_factory,extra_callbacks=(check,))
        assert check.seen and result['state']=='complete' and result['parent_weights_reused'] and not result['fresh_official_base']
        assert result['steps']==6 and [e['steps'] for e in result['epochs']]==[3,6]
        final=load_file(str(Path(result['last_adapter'])/'adapter_model.safetensors'))
        assert any(not torch.equal(weights[k],final[k]) for k in weights)
        resumed=p.engine.open_job(root/'direct/trials/resumed',c,resume=True);runtime.prepare(resumed,c,tokenizer=tok,pack=pack,testing=True)
        paused=launch(p,resumed,c,time.time()+3600,'fit',stop_step=1);assert paused['state']=='paused_training'
        restored=launch(p,resumed,c,time.time()+3600,'fit');actual=load_file(str(Path(restored['last_adapter'])/'adapter_model.safetensors'))
        assert all(torch.equal(final[k],actual[k]) for k in final)
        real_profile=runtime.profile;real_prepare=runtime.prepare
        with patch.object(runtime,'profile',lambda **kw:real_profile(testing=True)),patch.object(runtime,'verify_data',lambda:pack),patch.object(runtime,'prepare',lambda job,c,**kw:real_prepare(job,c,tokenizer=tok,pack=pack,testing=True)),patch.object(initialization,'default_source',lambda:source),patch.object(runner,'config',short):
            args=Namespace(app_root=runtime.app_root(),output=root/'loop',hours=1,resume=True,resume_from_checkpoint=None)
            assert runner.run(args)==0;state=d.read(args.output/'run_status.json')
            assert state['state']=='complete_review_required' and len(state['trials'])==2
            assert all(t['state']=='complete' and t['fit']['parent_weights_reused'] for t in state['trials'].values())
            base=runner.read_rows(args.output/'trials/parent_baseline/generations/parent_review.jsonl')
            assert base[0]==pack['fixture_parent_rows'][0] and len(base)==len(runtime.review_rows(pack))
            for trial in state['trials']:
                path=args.output/'trials'/trial/'generations'
                assert not (path/'parent_review.jsonl').exists()
                assert {p.name for p in path.glob('epoch_*_review.jsonl')}=={'epoch_01_review.jsonl','epoch_02_review.jsonl'}
            assert runner.run(args)==0
            from wengua_course.runner import export_review
            assert export_review(args.output,root/'review.zip').is_file()
        assert d.digest((source/'adapter_model.safetensors').read_bytes())==source_sha
        if os.name=='nt':
            env=os.environ.copy();env['WENGUA_PY']=sys.executable
            r=subprocess.run(['cmd.exe','/d','/c',str(runtime.ROOT/'train_wengua_v64.cmd'),'--help'],env=env,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=60)
            assert r.returncode==0 and 'check' in r.stdout
        proof={'software_fixture_only':True,'trained_parent_nonzero_and_unchanged':True,'loaded_parent_tensors_equal_before_update':True,
            'new_optimizer_empty':True,'gas_two_fresh_process_exact_resume_tensors_equal':True,'two_rates_two_epochs_complete':True,
            'parent_answers_reused_verbatim':True,'no_repeated_parent_free_generation_per_trial':True,'all_epoch_reviews_saved':True,
            'partial_gradient_accumulation_tail_verified':True,'literal_1080_locations_only_verified':True,
            'native_windows_cmd_help_verified':os.name=='nt','export_verified':True,'actual_8b_cuda_quality_verified':False}
        d.write(runtime.ROOT/'local/course_v64_cpu_verification.json',proof);print(d.dumps(proof))


if __name__=='__main__':main()
