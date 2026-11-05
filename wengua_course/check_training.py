"""Exercise inherited weights + fresh optimizer + exact fresh-process resume.
Tiny random CPU Qwen3/PEFT only. No real 8B/GPU performance assertion.
"""
from pathlib import Path
import tempfile,time,os
from argparse import Namespace
from unittest.mock import patch
from . import data,runtime,parents

def pack_fixture():
 units={'fixture1':{'id':'fixture1','book_id':'shenqi','original_quote':'software source one'},'fixture2':{'id':'fixture2','book_id':'kaiwu','original_quote':'software source two'}};rows=[]
 for split in ['train','validation']:
  for i,fmt in enumerate(['foundation_qa','conditional_dialogue','chart_application','complete_case']):
   rows.append(data.row('question '+str(i)+' '+split,'one',['fixture1'],units,task='software_'+fmt,format=fmt,split=split,domain='fixture',group='software:'+split+str(i),checks={'kind':'exact_name','answer':'one'}))
 return {'schema':data.SCHEMA,'source_books':['shenqi','kaiwu'],'records':rows,'units':units,'original_case_splits':{}}

def windows_delivery_fixture(root):
 if os.name!='nt':return False
 # Public known-answer vector, never a user's delivery secret.
 import base64,hashlib,hmac,subprocess
 from wengua_corpus import delivery
 f=data.read(runtime.app_root()/'tests/fixtures/course_delivery_cipher.json');assert f['software_fixture_only'] is True
 key=bytes.fromhex(f['key_hex']);blob=base64.b64decode(f['envelope_base64']);path=root/'synthetic_delivery.zip'
 delivery.install(blob,key,path,expected_sha256=f['plain_sha256']);assert path.read_bytes()==f['plaintext_utf8'].encode('utf8')
 changed=bytearray(blob);changed[-1]^=1
 try:delivery.install(bytes(changed),key,root/'tampered.zip',expected_sha256=f['plain_sha256'])
 except ValueError:pass
 else:raise AssertionError('Tampered cipher accepted')
 # 12 MiB exceeds this release ZIP and exercises actual PowerShell stdin/base64.
 plain=bytes(range(256))*49152;enc_key,iv,_=delivery.authenticate(blob,key)
 payload=data.dumps({'key':base64.b64encode(enc_key).decode(),'iv':base64.b64encode(iv).decode(),'cipher':base64.b64encode(plain).decode()})
 encrypted=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',delivery.WINDOWS_DECRYPT.replace('CreateDecryptor','CreateEncryptor')],input=payload,capture_output=True,text=True,timeout=120,check=True)
 cipher=base64.b64decode(encrypted.stdout.strip(),validate=True);body=delivery.MAGIC+iv+cipher;envelope=body+hmac.new(delivery.derive_keys(key)[1],body,hashlib.sha256).digest()
 output=root/'large_synthetic.zip';delivery.install(envelope,key,output,expected_sha256=data.digest(plain));assert output.read_bytes()==plain
 return True

def main():
 runtime.configure()
 import torch
 from transformers import TrainerCallback
 from peft import LoraConfig,get_peft_model
 from peft.utils.save_and_load import get_peft_model_state_dict
 from safetensors.torch import load_file
 from .fixtures import fixture_factory
 from .process import launch
 from src.wengua.qwen8b_attention_v1 import attention_route
 from src.wengua.qwen8b_loss_v1 import memory_trainer
 torch.set_num_threads(2);root=runtime.ROOT/'local';root.mkdir(exist_ok=True)
 with tempfile.TemporaryDirectory(dir=root,prefix='v52_cpu_') as temp:
  root=Path(temp);source=root/'old/trials/software/cycle_01/adapters/step-486';stage=source.parent.parent;job=stage.parent
  c=data.read(runtime.CONFIG)|dict(max_seq_length=512,inference_max_context=512,eval_max_new_tokens=8,smoke_steps=2,gradient_accumulation_steps=1,save_steps=2,development_generation_rows=4)
  oldconfig=c|{'schema':'wengua_evidence_visible_course_v51'}
  data.write(job/'job_contract.json',{'schema':oldconfig['schema'],'config':oldconfig,'dataset_release_sha256':data.digest((runtime.app_root()/'data/knowledge/course_v51/release.json').read_bytes())})
  data.write(stage/'stage_contract.json',{'schema':oldconfig['schema'],'config':oldconfig,'job_contract_sha256':data.digest((job/'job_contract.json').read_bytes())})
  tok,base,_=fixture_factory();m=get_peft_model(base,LoraConfig(r=c['lora_r'],lora_alpha=c['lora_alpha'],lora_dropout=c['lora_dropout'],target_modules=c['target_modules'],bias='none',task_type='CAUSAL_LM'))
  for name,param in m.named_parameters():
   if 'lora_B' in name:param.data.fill_(0.0125)
  m.save_pretrained(source,safe_serialization=True)
  data.write(source/'book_adapter_receipt.json',{'schema':oldconfig['schema'],'stage_contract_sha256':data.digest((stage/'stage_contract.json').read_bytes()),'files':{f.name:data.digest(f.read_bytes()) for f in source.iterdir() if f.is_file()}})
  expected=load_file(str(source/'adapter_model.safetensors'));phase=root/'phase';snap=parents.snapshot(phase,source,testing=True)
  c.update(book_parent_snapshot=str(snap),book_parent_provenance_sha256=data.digest((phase/'parent_provenance.json').read_bytes()))
  p=runtime.profile(testing=True);pack=pack_fixture();data.validate(pack)
  class Check(TrainerCallback):
   seen=False
   def on_train_begin(self,args,state,control,**kw):
    got=get_peft_model_state_dict(kw['model']);assert all(torch.equal(got[k].cpu(),v) for k,v in expected.items())
    assert state.global_step==0 and not kw['optimizer'].state;assert not kw['model'].config.use_cache;self.seen=True
  direct=p.engine.open_job(phase/'trials/direct',c,resume=True);runtime.prepare(direct,c,tokenizer=tok,pack=pack,testing=True);check=Check()
  with attention_route(testing=True),memory_trainer(128):done=p.engine.fit(direct,c,time.time()+3600,cpu_factory=fixture_factory,extra_callbacks=(check,))
  assert check.seen and done['state']=='complete' and done['parent_weights_reused']
  # Same dataset and parent, new worker each time; preserved optimizer/scheduler/RNG.
  resumed=p.engine.open_job(phase/'trials/resumed',c,resume=True);runtime.prepare(resumed,c,tokenizer=tok,pack=pack,testing=True)
  data.write(phase/'sweep_contract.json',{'schema':data.SCHEMA})
  paused=launch(p,resumed,c,time.time()+3600,'fit',stop_step=1);assert paused['state']=='paused_training' and paused['steps']==1
  restored=launch(p,resumed,c,time.time()+3600,'fit');assert restored['state']=='complete' and restored['steps']==done['steps']
  a=load_file(str(Path(done['last_adapter'])/'adapter_model.safetensors'));b=load_file(str(Path(restored['last_adapter'])/'adapter_model.safetensors'))
  assert all(torch.equal(a[k],b[k]) for k in a),'Resumed tensors differ from uninterrupted CPU fixture'
  evaluated=launch(p,resumed,c,time.time()+3600,'evaluate');assert evaluated['state']=='complete'
  starts=[]
  for f in (resumed/'worker_calls').glob('*.training_start.json'):starts.append(data.read(f))
  assert any(x['step']==1 and set(x['optimizer_steps'])=={1} for x in starts)
  # Exercise the actual finite-loop orchestration, parent comparison and export.
  from . import runner
  actual_profile=runtime.profile;actual_config=runner.config_for;actual_prepare=runtime.prepare
  def tiny_prepare(job,c,**kwargs):return actual_prepare(job,c,tokenizer=tok,pack=pack,testing=True)
  def tiny_config(parent,rate):
   return actual_config(parent,rate)|dict(max_seq_length=512,inference_max_context=512,eval_max_new_tokens=8,smoke_steps=2,gradient_accumulation_steps=1,save_steps=2,development_generation_rows=4)
  with patch.object(runtime,'profile',lambda **kw:actual_profile(testing=True)),patch.object(runtime,'verify_data',lambda:pack),patch.object(runtime,'prepare',tiny_prepare),patch.object(parents,'candidates',lambda:([(source,parents.inspect(source,testing=True))],[])),patch.object(runner,'config_for',tiny_config):
   loop=root/'finite_loop';args=Namespace(app_root=runtime.app_root(),output=loop,hours=1,resume=True,resume_from_checkpoint=None,parent=None)
   assert runner.run(args)==0
   status=data.read(loop/'run_status.json');assert status['state']=='complete_review_required' and len(status['trials'])==2
   exported=runner.export_review(loop,root/'software_review.zip');assert exported.is_file()
  result={'windows_native_12_mib_restore_fixture_passed':windows_delivery_fixture(root),'full_parent_comparison_two_rate_loop_export_exercised':True,'software_fixture_only':True,'steps':done['steps'],'fresh_parent_optimizer_verified':True,'parent_weights_preserved_before_first_update':True,'fresh_worker_resume_tensor_equal':True,'optimizer_step_restored':True,'evaluation_worker_exercised':True,'real_8b_gpu_fit_verified':False}
  data.write(runtime.ROOT/'local/course_v52_build/cpu_training_verification.json',result);print(data.dumps(result))
if __name__=='__main__':main()
