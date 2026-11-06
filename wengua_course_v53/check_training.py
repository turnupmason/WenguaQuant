"""Tiny random CPU tests for new initialization, checkpoint and finite stages."""
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch
import tempfile,time,os
from . import runtime,initialization,runner,data
from wengua_course import data as d

def software_pack():
 units={'fixture1':{'book_id':'shenqi','original_quote':'software literal source one'},'fixture2':{'book_id':'kaiwu','original_quote':'software literal source two'}};records=[]
 for stage in ['foundation','application']:
  for split in ['train','validation']:
   for i,fmt in enumerate(['foundation_qa','conditional_dialogue','chart_application','complete_case']):
    data.add(records,'question '+stage+' '+split+' '+str(i),'one',['fixture1'],units,task='software_'+fmt,format=fmt,stage=stage,split=split,checks={'kind':'exact_name','answer':'one'},group='software:'+stage+split+str(i))
 return {'schema':data.SCHEMA,'source_books':['shenqi','kaiwu'],'units':units,'records':records,'case_catalog':[],'original_case_splits':{}}

def windows_delivery_fixture(root):
 """Exercise actual Windows stdin/base64 AES with a public synthetic key."""
 if os.name!='nt':return False
 import base64,hashlib,hmac,subprocess
 from wengua_corpus import delivery
 from wengua_course.check_training import windows_delivery_fixture as previous_fixture
 assert previous_fixture(root)
 f=d.read(runtime.app_root()/'tests/fixtures/course_delivery_cipher.json')
 assert f['software_fixture_only']
 key=bytes.fromhex(f['key_hex']);enc_key,iv,_=delivery.authenticate(base64.b64decode(f['envelope_base64']),key)
 plain=bytes(range(256))*65536  # 16 MiB, larger than the reviewed V53 ZIP.
 payload=d.dumps({'key':base64.b64encode(enc_key).decode(),'iv':base64.b64encode(iv).decode(),'cipher':base64.b64encode(plain).decode()})
 output=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',delivery.WINDOWS_DECRYPT.replace('CreateDecryptor','CreateEncryptor')],input=payload,capture_output=True,text=True,timeout=120,check=True)
 cipher=base64.b64decode(output.stdout.strip(),validate=True);body=delivery.MAGIC+iv+cipher;blob=body+hmac.new(delivery.derive_keys(key)[1],body,hashlib.sha256).digest()
 path=root/'v53_large_software_fixture.zip';delivery.install(blob,key,path,expected_sha256=d.digest(plain));assert path.read_bytes()==plain
 return True

def main():
 runtime.configure()
 import torch
 from transformers import TrainerCallback
 from safetensors.torch import load_file
 from peft.utils.save_and_load import get_peft_model_state_dict
 from wengua_course.fixtures import fixture_factory
 from .process import launch
 from src.wengua.qwen8b_attention_v1 import attention_route
 from src.wengua.qwen8b_loss_v1 import memory_trainer
 root=runtime.ROOT/'local';root.mkdir(exist_ok=True);torch.set_num_threads(2)
 with tempfile.TemporaryDirectory(prefix='v53_cpu_',dir=root) as tmp:
  root=Path(tmp);p=runtime.profile(testing=True);pack=software_pack();data.validate(pack);tok,model,_=fixture_factory();inputs=tok('question one',return_tensors='pt')
  inputs.pop('token_type_ids',None);model.eval();a=model.generate(**inputs,do_sample=False,max_new_tokens=4);b=model.generate(**inputs,do_sample=False,max_new_tokens=4,logits_to_keep=1);assert torch.equal(a,b)
  c=d.read(runtime.CONFIG)|dict(max_seq_length=512,inference_max_context=512,eval_max_new_tokens=8,smoke_steps=2,gradient_accumulation_steps=1,save_steps=2,development_generation_rows=4,foundation_repeat=1,application_repeat=1,initialization_only=True)
  d.write(root/'sweep_contract.json',{'schema':runtime.SCHEMA});init=p.engine.open_job(root/'trials/init',c,resume=True);created=launch(p,init,c,time.time()+3600,'initialize');source=Path(created['adapter']);weights=load_file(str(source/'adapter_model.safetensors'));assert all(not t.count_nonzero() for n,t in weights.items() if 'lora_B' in n)
  parent=initialization.snapshot(root/'official_snapshot',source,testing=True);c.update(initialization_only=False,book_parent_snapshot=str(parent),book_parent_provenance_sha256=d.digest((parent.parent/'parent_provenance.json').read_bytes()))
  class Check(TrainerCallback):
   seen=False
   def on_train_begin(self,args,state,control,**kwargs):
    actual=get_peft_model_state_dict(kwargs['model']);assert all(torch.equal(actual[k].cpu(),v) for k,v in weights.items());assert not kwargs['optimizer'].state;self.seen=True
  direct=p.engine.open_job(root/'trials/direct',c,resume=True);runtime.prepare(direct,c,tokenizer=tok,pack=pack,testing=True);check=Check()
  with attention_route(testing=True),memory_trainer(128):completed=p.engine.fit(direct,c,time.time()+3600,cpu_factory=fixture_factory,extra_callbacks=(check,))
  assert check.seen and completed['state']=='complete'
  # Force the real legacy pending-generation branch, where the old fit skipped
  # prepare_frozen_base. V53 must normalize it before attaching the adapter.
  pending=d.read(direct/'cycle_01/stage_status.json');pending['state']='trained_pending_generation';d.write(direct/'cycle_01/stage_status.json',pending)
  generate=p.book.generate_rows;observed=[]
  def mixed_norm_fixture():
   tokenizer,base,compute=fixture_factory()
   for param in base.parameters():
    if param.ndim==1:param.data=param.data.to(torch.bfloat16)
   return tokenizer,base,compute
  def checked_generation(base,*args,**kwargs):
   assert all(param.dtype==torch.float32 for param in base.parameters() if param.ndim==1)
   observed.append(True);return generate(base,*args,**kwargs)
  with attention_route(testing=True),memory_trainer(128),patch.object(p.book,'generate_rows',checked_generation):
   assert p.engine.fit(direct,c,time.time()+3600,cpu_factory=mixed_norm_fixture)['state']=='complete'
  assert observed
  resumed=p.engine.open_job(root/'trials/resumed',c,resume=True);runtime.prepare(resumed,c,tokenizer=tok,pack=pack,testing=True)
  paused=launch(p,resumed,c,time.time()+3600,'fit',stop_step=1);assert paused['state']=='paused_training' and paused['steps']==1
  resumed_status=launch(p,resumed,c,time.time()+3600,'fit');assert resumed_status['state']=='complete'
  a=load_file(str(Path(completed['last_adapter'])/'adapter_model.safetensors'));b=load_file(str(Path(resumed_status['last_adapter'])/'adapter_model.safetensors'));assert all(torch.equal(a[k],b[k]) for k in a)
  ev=launch(p,resumed,c,time.time()+3600,'evaluate',best={'adapter':resumed_status['last_adapter'],'tag':'software_cross_stage','eval_stage':'foundation'});assert ev['state']=='complete'
  # Actual orchestration, real subprocesses and train/evaluate/export, with synthetic gate decisions.
  real_profile=runtime.profile;real_prepare=runtime.prepare;real_configuration=runner.configuration
  def small_config(parent,stage,rate):return real_configuration(parent,stage,rate)|dict(max_seq_length=512,inference_max_context=512,eval_max_new_tokens=8,smoke_steps=2,gradient_accumulation_steps=1,save_steps=2,development_generation_rows=4,foundation_repeat=1,application_repeat=1)
  gate=lambda *args:{'passed':True,'software_fixture_gate_only':True}
  with patch.object(runtime,'profile',lambda **kwargs:real_profile(testing=True)),patch.object(runtime,'verify_data',lambda:pack),patch.object(runtime,'prepare',lambda job,c,**kw:real_prepare(job,c,tokenizer=tok,pack=pack,testing=True)),patch.object(runner,'configuration',small_config),patch.object(runner.checks,'gate',gate),patch('wengua_course_v53.rag.ensure_index',lambda:None):
   out=root/'loop';args=Namespace(app_root=runtime.app_root(),output=out,hours=1,resume=True,resume_from_checkpoint=None);assert runner.run(args)==0
   status=d.read(out/'run_status.json');assert status['state']=='complete_review_required' and len(status['application'])==2
   assert runner.export(out,root/'review.zip').is_file();assert runner.run(args)==0
  result={'software_fixture_only':True,'windows_native_16_mib_restore_fixture_passed':windows_delivery_fixture(root),'pending_generation_resume_uses_fp32_norms':True,'official_fresh_adapter_has_zero_B_and_no_old_weights':True,'first_update_uses_verified_snapshot_and_empty_optimizer':True,'fresh_process_resume_tensor_equal':True,'last_position_generation_equal_to_full_projection':True,'short_eager_forward_reference_exercised':True,'cross_stage_evaluation_exercised':True,'full_finite_loop_two_application_rates_and_export_exercised':True,'real_rag_book_probe_not_claimed':True,'real_8b_cuda_fit_verified':False}
  d.write(runtime.ROOT/'local/course_v53_build/cpu_training_verification.json',result);print(d.dumps(result))
if __name__=='__main__':main()
