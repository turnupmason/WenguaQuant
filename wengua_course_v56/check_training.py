"""Genuine tiny CPU tensors and subprocesses; no actual 8B quality claim."""
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch
from copy import deepcopy
import tempfile,time,os,sys
from wengua_course import data as d
from . import runtime,initialization,runner,data,checks


def software_pack():
    from wengua_course_v55.check_training import software_pack as old
    pack=deepcopy(old());pack.update(schema=data.SCHEMA,legacy_v55_prompts=[r for r in pack['records'] if r['split']=='validation'],legacy_actual_rag_rows=[],legacy_diagnostics=[])
    return pack


def windows_delivery_and_launcher(root):
    if os.name!='nt':return {'native_windows':False}
    import subprocess,base64,hmac,hashlib
    from wengua_corpus import delivery
    f=d.read(runtime.app_root()/'tests/fixtures/course_delivery_cipher.json');key=bytes.fromhex(f['key_hex']);enc,iv,_=delivery.authenticate(base64.b64decode(f['envelope_base64']),key)
    plain=bytes(range(256))*98304  # 24 MiB; covers the actual private V56 ZIP.
    payload=d.dumps({'key':base64.b64encode(enc).decode(),'iv':base64.b64encode(iv).decode(),'cipher':base64.b64encode(plain).decode()})
    answer=__import__('subprocess').run(['powershell.exe','-NoProfile','-NonInteractive','-Command',delivery.WINDOWS_DECRYPT.replace('CreateDecryptor','CreateEncryptor')],input=payload,capture_output=True,text=True,timeout=120,check=True)
    cipher=base64.b64decode(answer.stdout.strip(),validate=True);body=delivery.MAGIC+iv+cipher;blob=body+hmac.new(delivery.derive_keys(key)[1],body,hashlib.sha256).digest()
    target=root/'large_windows_delivery_fixture.bin';delivery.install(blob,key,target,expected_sha256=d.digest(plain));assert target.read_bytes()==plain
    env=os.environ.copy();env['WENGUA_PY']=sys.executable
    result=subprocess.run(['cmd.exe','/d','/c',str(runtime.ROOT/'train_wengua_v56.cmd'),'--help'],cwd=runtime.ROOT,env=env,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=60)
    assert result.returncode==0 and 'restore' in result.stdout and 'export' in result.stdout
    return {'native_windows':True,'native_24_mib_aes_sha_verified':True,'actual_cmd_help_entry_verified':True}


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
    with tempfile.TemporaryDirectory(prefix='v56_cpu_',dir=runtime.ROOT/'local') as tmp:
        root=Path(tmp);tok,_,_=fixture_factory();small=dict(max_seq_length=512,inference_max_context=512,eval_max_new_tokens=8,smoke_steps=2,gradient_accumulation_steps=1,save_steps=1,development_generation_rows=4,foundation_repeat=1,application_repeat=1,num_train_epochs=2)
        sp=seed.profile(testing=True);sc=d.read(seed.CONFIG)|small|dict(initialization_only=True);job=sp.engine.open_job(root/'seed/trials/init',sc,resume=True);created=si.create(sp,job,sc,factory=fixture_factory)
        snap=oi.snapshot(root/'old_parent',created['adapter'],testing=True);op=old.profile(testing=True);oc=d.read(old.CONFIG)|small;oc.update(book_parent_snapshot=str(snap),book_parent_provenance_sha256=d.digest((snap.parent/'parent_provenance.json').read_bytes()))
        oj=op.engine.open_job(root/'old/trials/trained',oc,resume=True);old.prepare(oj,oc,tokenizer=tok,pack=old_pack(),testing=True)
        with attention_route(testing=True),memory_trainer(128):trained=op.engine.fit(oj,oc,time.time()+3600,cpu_factory=fixture_factory)
        source=Path(trained['last_adapter']);weights=load_file(str(source/'adapter_model.safetensors'));assert any(t.count_nonzero() for k,t in weights.items() if 'lora_B' in k)
        p=runtime.profile(testing=True);pack=software_pack();data.validate(pack);parent=initialization.snapshot(root/'parent',source,testing=True)
        original_config=runner.config
        def short_config(parent,source_run,rate,phase='full'):return original_config(parent,source_run,rate,phase)|small
        c=short_config(parent,root,2e-5,'repair')
        class Inherit(TrainerCallback):
            seen=False
            def on_train_begin(self,args,state,control,**kw):
                got=get_peft_model_state_dict(kw['model']);assert all(torch.equal(got[k].cpu(),v) for k,v in weights.items());assert not kw['optimizer'].state;self.seen=True
        direct=p.engine.open_job(root/'direct/trials/full',c,resume=True);runtime.prepare(direct,c,tokenizer=tok,pack=pack,testing=True);callback=Inherit()
        with attention_route(testing=True),memory_trainer(128):completed=p.engine.fit(direct,c,time.time()+3600,cpu_factory=fixture_factory,extra_callbacks=(callback,))
        assert callback.seen and completed['state']=='complete'
        resumed=p.engine.open_job(root/'direct/trials/resumed',c,resume=True);runtime.prepare(resumed,c,tokenizer=tok,pack=pack,testing=True)
        paused=launch(p,resumed,c,time.time()+3600,'fit',stop_step=1);assert paused['state']=='paused_training'
        result=launch(p,resumed,c,time.time()+3600,'fit');a=load_file(str(Path(completed['last_adapter'])/'adapter_model.safetensors'));b=load_file(str(Path(result['last_adapter'])/'adapter_model.safetensors'));assert all(torch.equal(a[k],b[k]) for k in a)
        real_profile=runtime.profile;real_prepare=runtime.prepare
        # Tiny random-text output cannot pass professional acceptance. Inject
        # only the decision fixture to exercise the full positive orchestration.
        with patch.object(runtime,'profile',lambda **kw:real_profile(testing=True)),patch.object(runtime,'verify_data',lambda:pack),patch.object(runtime,'prepare',lambda job,c,**kw:real_prepare(job,c,tokenizer=tok,pack=pack,testing=True)),patch.object(runner,'config',short_config),patch.object(checks,'gate',lambda *a,**kw:{'passed':True,'software_fixture_only':True}),patch.object(runner,'retention',lambda *a,**kw:{'passed':True,'software_fixture_only':True}):
            out=root/'loop';args=Namespace(app_root=runtime.app_root(),output=out,hours=1,resume=True,resume_from_checkpoint=None,fixture_parents={'previous_A':source,'previous_B':source})
            assert runner.run(args)==0;state=d.read(out/'run_status.json');assert state['state']=='complete_review_required';assert len(state['trials'])==4;assert len([t for t in state['trials'].values() if t['phase']=='repair'])==2
            assert d.read(out/'repair_selection.json')['adapter'];assert runner.run(args)==0
            from wengua_course.runner import export_review
            assert export_review(out,root/'review.zip').is_file()
        proof={'software_fixture_only':True,'nonzero_v55_weights_inherited':True,'new_phase_optimizer_empty_at_start':True,'separate_process_exact_resume_tensor_equal':True,'two_parent_comparison_then_two_full_rates_exercised':True,'acceptance_injected_only_for_random_tiny_fixture':True,'production_acceptance_not_bypassed':True,'completed_rerun_and_export_verified':True,'windows_delivery_and_launcher':windows_delivery_and_launcher(root),'actual_8b_cuda_training_verified':False,'professional_model_quality_verified':False}
        d.write(runtime.ROOT/'local/course_v56_build/cpu_training_verification.json',proof);print(d.dumps(proof))


if __name__=='__main__':main()
