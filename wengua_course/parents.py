"""Verify genuine sealed book adapters; copy weights, never parent optimizers."""
from pathlib import Path
import shutil
from . import data,runtime
ESSENTIAL=('adapter_config.json','adapter_model.safetensors')
APPROVED={'wengua_reviewed_course_v4':'qwen_course_v4','wengua_evidence_visible_course_v51':'course_v51'}

def inspect(path,*,testing=False):
 path=Path(path).resolve();stage=path.parent.parent
 if path.parent.name!='adapters' or not path.name.startswith('step-') or stage.name!='cycle_01':raise ValueError('Only complete sealed formal epoch adapters can initialize V5.2')
 job=stage.parent;sc=data.read(stage/'stage_contract.json');jc=data.read(job/'job_contract.json');receipt=data.read(path/'book_adapter_receipt.json')
 schema=sc['schema']
 if schema not in APPROVED or jc['schema']!=schema:raise ValueError('Not an approved pure two-book parent')
 if sc['job_contract_sha256']!=data.digest((job/'job_contract.json').read_bytes()):raise ValueError('Parent job contract differs')
 if receipt['schema']!=schema or receipt['stage_contract_sha256']!=data.digest((stage/'stage_contract.json').read_bytes()):raise ValueError('Parent adapter stage differs')
 c=sc['config'];cfg=data.read(path/'adapter_config.json')
 if not testing and (c['lora_r'],c['lora_alpha'],c['lora_dropout'],set(c['target_modules']))!=(16,32,.05,{'q_proj','k_proj','v_proj','o_proj'}):raise ValueError('This repair release requires the inherited r16 Q/K/V/O structure')
 if c['model_id']!='Qwen/Qwen3-8B' or c['revision']!='b968826d9c46dd6066d109eabc6255188de91218':raise ValueError('Wrong parent base')
 if c.get('non_book_teachers_allowed') is not False or c.get('production_enabled') is not False:raise ValueError('Parent policy not book-only')
 release=runtime.app_root()/'data/knowledge'/APPROVED[schema]/'release.json';text=release.read_text(encoding='utf8');allowed={data.digest(text),data.digest(text.replace('\n','\r\n'))}
 if jc['dataset_release_sha256'] not in allowed:raise ValueError('Parent dataset not the pinned book course')
 for n,h in receipt['files'].items():
  if Path(n).name!=n or not (path/n).is_file() or data.digest((path/n).read_bytes())!=h:raise ValueError('Parent file changed')
 for key,ckey in [('lora_r','r'),('lora_alpha','lora_alpha'),('lora_dropout','lora_dropout')]:
  if c[key]!=cfg[ckey]:raise ValueError('Parent LoRA structure mismatch')
 if set(c['target_modules'])!=set(cfg['target_modules']) or cfg['bias']!='none' or cfg['peft_type']!='LORA':raise ValueError('Parent target/type changed')
 if not testing and cfg['base_model_name_or_path']!=c['model_id']:raise ValueError('Parent config base differs')
 for n in ESSENTIAL:
  if n not in receipt['files'] or not (path/n).stat().st_size:raise ValueError('Parent essential weights missing')
 return {'source_path':str(path),'schema':schema,'parent_config':c,'files':{n:data.digest((path/n).read_bytes()) for n in ESSENTIAL},'stage_contract_sha256':data.digest((stage/'stage_contract.json').read_bytes()),'job_contract_sha256':data.digest((job/'job_contract.json').read_bytes()),'receipt_sha256':data.digest((path/'book_adapter_receipt.json').read_bytes()),'optimizer_inherited':False,'quality_selection':'not_yet_assessed_by_v52_development'}

def candidates():
 app=runtime.app_root();found={};failures=[]
 # Prefer the exact V5.1 ancestry and both completed V5.1 epochs, not an arbitrary latest folder.
 for root in (app/'local/course_v51',app/'local/qwen_course_v4'):
  if not root.exists():continue
  for phase in sorted(root.glob('*/phase_status.json')) if root.name=='course_v51' else []:
   state=data.read(phase)
   for r in state.get('epochs',[]):
    path=Path(r['adapter'])
    try:proof=inspect(path);found[str(path.resolve())]=(path,proof)
    except (ValueError,KeyError,OSError) as e:failures.append({'path':str(path),'error':str(e)})
   provenance=phase.parent/'parent_provenance.json'
   if provenance.exists():
    path=Path(data.read(provenance)['source_path'])
    try:found[str(path.resolve())]=(path,inspect(path))
    except (ValueError,KeyError,OSError) as e:failures.append({'path':str(path),'error':str(e)})
 if not found:
  for path in sorted((app/'local/qwen_course_v4').glob('*/trials/*/cycle_01/adapters/step-*')):
   try:found[str(path.resolve())]=(path,inspect(path))
   except (ValueError,KeyError,OSError) as e:failures.append({'path':str(path),'error':str(e)})
 return list(found.values()),failures

def snapshot(root,path,proof=None,*,testing=False):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);dest=root/'parent_adapter';receipt=root/'parent_provenance.json'
 proof=proof or inspect(path,testing=testing)
 if dest.exists():
  if data.read(receipt)!=proof:raise ValueError('Parent snapshot source changed')
  for n,h in proof['files'].items():
   if data.digest((dest/n).read_bytes())!=h:raise ValueError('Snapshot bytes changed')
  return dest
 if receipt.exists():raise ValueError('Partial parent snapshot preserved')
 pending=root/'parent_adapter.pending';pending.mkdir()
 for n in ESSENTIAL:shutil.copy2(Path(path)/n,pending/n)
 for n,h in proof['files'].items():
  if data.digest((pending/n).read_bytes())!=h:raise ValueError('Parent changed during copy')
 pending.rename(dest);data.write(receipt,proof);return dest
