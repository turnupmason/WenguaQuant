"""Real tiny CPU fixtures for V55. No actual 8B/GPU quality claim."""
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch
import tempfile,time
from . import runtime, initialization, runner, data
from wengua_course import data as d


def software_pack():
 from wengua_course_v54.check_training import software_pack as previous
 pack=previous();legacy=pack['legacy_v53_prompts']
 pack.update(schema=data.SCHEMA,legacy_v54_prompts=[r for r in pack['records'] if r['split']=='validation'],legacy_actual_rag_rows=[],extra_units={})
 return pack


def main():
 runtime.configure()
 import torch
 from transformers import TrainerCallback
 from safetensors.torch import load_file
 from peft.utils.save_and_load import get_peft_model_state_dict
 from wengua_course.fixtures import fixture_factory
 from wengua_course_v53 import runtime as seed_runtime,initialization as seed_init
 from wengua_course_v54 import runtime as old,initialization as oi
 from wengua_course_v54.check_training import software_pack as old_pack
 from wengua_course_v53.check_training import windows_delivery_fixture
 from src.wengua.qwen8b_attention_v1 import attention_route
 from src.wengua.qwen8b_loss_v1 import memory_trainer
 from .process import launch
 torch.set_num_threads(2);runtime.ROOT.joinpath('local').mkdir(exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='v55_cpu_',dir=runtime.ROOT/'local') as tmp:
  root=Path(tmp);tok,_,_=fixture_factory()
  # Make a genuinely updated, sealed old-schema adapter as the parent.
  seed_profile=seed_runtime.profile(testing=True);small=dict(max_seq_length=512,inference_max_context=512,eval_max_new_tokens=8,smoke_steps=2,gradient_accumulation_steps=1,save_steps=2,development_generation_rows=4,foundation_repeat=1,application_repeat=1)
  seed_config=d.read(seed_runtime.CONFIG)|small|dict(initialization_only=True);init=seed_profile.engine.open_job(root/'old/trials/init',seed_config,resume=True);created=seed_init.create(seed_profile,init,seed_config,factory=fixture_factory);snap=oi.snapshot(root/'old_source',created['adapter'],testing=True)
  op=old.profile(testing=True);oc=d.read(old.CONFIG)|small|dict(num_train_epochs=1)
  oc.update(initialization_only=False,book_parent_snapshot=str(snap),book_parent_provenance_sha256=d.digest((snap.parent/'parent_provenance.json').read_bytes()))
  oldjob=op.engine.open_job(root/'old/trials/trained',oc,resume=True);old.prepare(oldjob,oc,tokenizer=tok,pack=old_pack(),testing=True)
  with attention_route(testing=True),memory_trainer(128):trained=op.engine.fit(oldjob,oc,time.time()+3600,cpu_factory=fixture_factory)
  source=Path(trained['last_adapter']);weights=load_file(str(source/'adapter_model.safetensors'));assert any(t.count_nonzero() for n,t in weights.items() if 'lora_B' in n)
  p=runtime.profile(testing=True);pack=software_pack();data.validate(pack);parent=initialization.snapshot(root/'source_parent',source,testing=True)
  def small_config(parent,source_run,rate):return config_original(parent,source_run,rate)|dict(max_seq_length=512,inference_max_context=512,eval_max_new_tokens=8,smoke_steps=2,gradient_accumulation_steps=1,save_steps=2,development_generation_rows=4,foundation_repeat=1,application_repeat=1)
  config_original=runner.config;c=small_config(parent,root/'old',1e-5)
  class Check(TrainerCallback):
   seen=False
   def on_train_begin(self,args,state,control,**kw):
    actual=get_peft_model_state_dict(kw['model']);assert all(torch.equal(actual[k].cpu(),v) for k,v in weights.items());assert not kw['optimizer'].state;self.seen=True
  direct=p.engine.open_job(root/'direct/trials/complete',c,resume=True);runtime.prepare(direct,c,tokenizer=tok,pack=pack,testing=True);callback=Check()
  with attention_route(testing=True),memory_trainer(128):completed=p.engine.fit(direct,c,time.time()+3600,cpu_factory=fixture_factory,extra_callbacks=(callback,))
  assert callback.seen and completed['state']=='complete' and len(completed['epochs'])==2
  resumed=p.engine.open_job(root/'direct/trials/resumed',c,resume=True);runtime.prepare(resumed,c,tokenizer=tok,pack=pack,testing=True)
  paused=launch(p,resumed,c,time.time()+3600,'fit',stop_step=1);assert paused['state']=='paused_training' and paused['steps']==1
  restored=launch(p,resumed,c,time.time()+3600,'fit');assert restored['state']=='complete'
  a=load_file(str(Path(completed['last_adapter'])/'adapter_model.safetensors'));b=load_file(str(Path(restored['last_adapter'])/'adapter_model.safetensors'));assert all(torch.equal(a[k],b[k]) for k in a)
  assert initialization.inspect(restored['last_adapter'],testing=True)['schema']==runtime.SCHEMA
  assert initialization.inspect(source,testing=True)['schema']==old.SCHEMA
  result=launch(p,resumed,c,time.time()+3600,'evaluate',best={'adapter':restored['last_adapter'],'tag':'legacy_software_regression','eval_stage':'foundation'});assert result['state']=='complete'
  real_profile=runtime.profile;real_prepare=runtime.prepare
  with patch.object(runtime,'profile',lambda **kw:real_profile(testing=True)),patch.object(runtime,'verify_data',lambda:pack),patch.object(runtime,'prepare',lambda job,c,**kw:real_prepare(job,c,tokenizer=tok,pack=pack,testing=True)),patch.object(runner,'config',small_config):
   out=root/'loop';args=Namespace(app_root=runtime.app_root(),output=out,source_run=root/'old',parent=source,hours=1,resume=True,resume_from_checkpoint=None)
   assert runner.run(args)==0;state=d.read(out/'run_status.json');assert state['state']=='complete_review_required' and len(state['trials'])==2 and all(len(t['epochs'])==2 for t in state['trials'].values())
   from wengua_course.runner import export_review
   assert export_review(out,root/'review.zip').is_file();assert runner.run(args)==0
  result={'software_fixture_only':True,'nonzero_parent_weights_reused':True,'first_update_empty_new_optimizer':True,'fresh_process_resume_tensor_equal':True,'both_rates_two_epochs_finished':True,'legacy_evaluation_transport_exercised':True,'export_and_completed_rerun_verified':True,'windows_native_16_mib_restore':windows_delivery_fixture(root),'actual_8b_cuda_training_verified':False,'real_book_answer_quality_verified':False}
  d.write(runtime.ROOT/'local/course_v55_build/cpu_training_verification.json',result);print(d.dumps(result))
if __name__=='__main__':main()
