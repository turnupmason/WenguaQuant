"""Official fresh LoRA and immutable copies of sealed comparison adapters."""
from pathlib import Path
import shutil
from . import runtime
from wengua_course import data as d
ESSENTIAL=('adapter_config.json','adapter_model.safetensors')

def inspect(path,*,testing=False):
 path=Path(path).resolve();stage=path.parent.parent;job=stage.parent
 if path.parent.name!='adapters' or stage.name!='cycle_01' or not path.name.startswith('step-'):raise ValueError('Formal sealed adapter required; smoke weights forbidden')
 receipt=d.read(path/'book_adapter_receipt.json');sc=d.read(stage/'stage_contract.json');jc=d.read(job/'job_contract.json');c=jc['config']
 if receipt['schema'] not in ('wengua_two_book_conditional_course_v52',runtime.SCHEMA) or sc['schema']!=receipt['schema'] or jc['schema']!=receipt['schema']:raise ValueError('Only V52 comparison or V53 adapters allowed')
 if receipt['stage_contract_sha256']!=d.digest((stage/'stage_contract.json').read_bytes()) or sc['job_contract_sha256']!=d.digest((job/'job_contract.json').read_bytes()):raise ValueError('Adapter provenance mismatch')
 if receipt['schema'].endswith('v52') and d.read(stage/'stage_status.json').get('state')!='complete':raise ValueError('Previous comparison training is not complete')
 if c['model_id']!='Qwen/Qwen3-8B' or c['revision']!=d.read(runtime.CONFIG)['revision']:raise ValueError('Different official base')
 release=runtime.previous.RELEASE if receipt['schema'].endswith('v52') else runtime.RELEASE
 if jc['dataset_release_sha256']!=d.digest(release.read_text()):raise ValueError('Parent course differs')
 for n,h in receipt['files'].items():
  if Path(n).name!=n or d.digest((path/n).read_bytes())!=h:raise ValueError('Adapter bytes changed')
 config=d.read(path/'adapter_config.json')
 if not testing and config['base_model_name_or_path']!='Qwen/Qwen3-8B':raise ValueError('Adapter base is not official 8B')
 if (config['r'],config['lora_alpha'],config['lora_dropout'],set(config['target_modules']))!=(c['lora_r'],c['lora_alpha'],c['lora_dropout'],set(c['target_modules'])):raise ValueError('Adapter structure changed')
 if config['bias']!='none' or config['peft_type']!='LORA':raise ValueError('Unsupported adapter')
 return {'source_path':str(path),'schema':receipt['schema'],'parent_config':c,'files':{n:d.digest((path/n).read_bytes()) for n in ESSENTIAL},'optimizer_inherited':False,'professional_approval':False}

def snapshot(root,path,*,testing=False):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);proof=inspect(path,testing=testing);dest=root/'parent_adapter';receipt=root/'parent_provenance.json'
 if dest.exists():
  if d.read(receipt)!=proof:raise ValueError('Initialization changed')
 else:
  pending=root/'parent_adapter.pending';pending.mkdir()
  for n in ESSENTIAL:shutil.copy2(Path(path)/n,pending/n)
  pending.rename(dest);d.write(receipt,proof)
 for n,h in proof['files'].items():
  if d.digest((dest/n).read_bytes())!=h:raise ValueError('Initialization snapshot bytes changed')
 return dest

def forward_parity(base,tok,book):
 """Short eager reference only; long training still forbids math fallback."""
 import torch
 before=base.config._attn_implementation
 messages=[{'role':'user','content':'请回答一加一等于几。'}]
 tokens=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
 if len(tokens)>128:raise ValueError('Numerical reference must stay short')
 ids=torch.tensor([tokens],device=base.device);base.eval()
 try:
  base.set_attn_implementation('sdpa')
  with torch.inference_mode(),book.generation_autocast(base):fused=base(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1).logits.float()
  base.set_attn_implementation('eager')
  with torch.inference_mode(),book.generation_autocast(base):reference=base(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1).logits.float()
 finally:base.set_attn_implementation(before or 'sdpa')
 if not torch.isfinite(fused).all() or not torch.isfinite(reference).all():raise RuntimeError('Nonfinite short-forward logits before training')
 relative=float(torch.linalg.vector_norm(fused-reference)/torch.linalg.vector_norm(reference).clamp_min(1e-6));difference=float((fused-reference).abs().max())
 result={'short_tokens':len(tokens),'relative_l2_error':relative,'max_logit_abs_error':difference,'next_token_agrees':bool(fused.argmax(-1).eq(reference.argmax(-1)).all()),'reference':'eager_attention_short_forward_only','training_sdpa_restored':base.config._attn_implementation==(before or 'sdpa'),'actual_cuda':base.device.type=='cuda','long_context_training_math_fallback_allowed':False}
 if relative>.03:raise RuntimeError('Short forward differs from eager reference; no training started: '+str(result))
 return result

def create(p,job,c,*,factory=None):
 from transformers import set_seed,AutoTokenizer
 from peft import get_peft_model,LoraConfig
 from src.wengua.qwen_finetune import prepare_frozen_base
 job=Path(job);stage=job/'cycle_01';path=stage/'adapters/step-0'
 if path.exists():inspect(path,testing=p.testing);return {'state':'complete','adapter':str(path),'official_fresh_lora':True,'updates':0}
 set_seed(c['seed'])
 if factory:tok,base,_=factory()
 else:
  tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision'],local_files_only=True,trust_remote_code=False);base,_=p.book.load_book_cuda(c,True)
 base=prepare_frozen_base(base,gradient_checkpointing=False);d.write(job/'short_forward_parity.json',forward_parity(base,tok,p.book));model=get_peft_model(base,LoraConfig(r=c['lora_r'],lora_alpha=c['lora_alpha'],lora_dropout=c['lora_dropout'],target_modules=c['target_modules'],bias='none',task_type='CAUSAL_LM'))
 stage.mkdir(parents=True,exist_ok=True);d.write(stage/'stage_contract.json',{'schema':runtime.SCHEMA,'job_contract_sha256':p.book.file_sha(job/'job_contract.json'),'initialization_only':True,'official_fresh_lora':True,'updates':0})
 adapter=p.book.save_adapter(model,tok,stage,0)
 return {'state':'complete','adapter':str(adapter),'official_fresh_lora':True,'updates':0,'old_weights_loaded':False}
