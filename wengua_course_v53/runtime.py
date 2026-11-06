"""New namespace; frozen V52/App computations are reused without modifying them."""
from pathlib import Path
from types import FunctionType
from copy import deepcopy
from collections import Counter,defaultdict
import sys,os
from wengua_course import runtime as previous,data as d
from . import data,checks
ROOT=previous.ROOT;DATA=ROOT/'local/course_v53_data';RELEASE=ROOT/'data/knowledge/course_v53/release.json';CONFIG=ROOT/'configs/course_v53.json';SCHEMA=SWEEP_SCHEMA=data.SCHEMA
configure=previous.configure;app_root=previous.app_root;verify_snapshot=previous.verify_snapshot;ALLOCATOR=previous.ALLOCATOR

def hashes():
 result=previous.hashes()
 for path in sorted((ROOT/'wengua_course_v53').glob('*.py')):result[path.relative_to(ROOT).as_posix()]=d.digest(path.read_text())
 for name in ['configs/course_v53.json','train_wengua_v53.cmd','run_course_v53_worker.py','wengua_corpus/book_rag.py','wengua_corpus/book_rag_editorial.py']:result[name]=d.digest((ROOT/name).read_text())
 result['v53_release']=d.digest(RELEASE.read_text());return result

def verify_data():
 release=d.read(RELEASE)
 for name,h in release['files'].items():
  path=(DATA/name).resolve()
  if not path.is_relative_to(DATA.resolve()) or not path.is_file() or d.digest(path.read_bytes())!=h:raise ValueError('V53 course missing/changed: '+name)
 pack=d.read(DATA/'pack.json');data.validate(pack);return pack

def validate(c,testing=False):
 from src.wengua import qwen8b_book_v1 as old
 if c['schema']!=SCHEMA:raise ValueError('V53 schema required')
 old.validate(c|{'schema':old.BOOK_SCHEMA,'smoke_steps':100},testing=testing)
 if not testing and (c['lora_r'],c['lora_alpha'],c['lora_dropout'],set(c['target_modules']))!=(16,32,.05,{'q_proj','k_proj','v_proj','o_proj'}):raise ValueError('Pinned reviewed adapter structure required')
 if c['curriculum_stage'] not in ('foundation','application'):raise ValueError('Unknown stage')
 if c['learning_rate'] not in (1e-5,2e-5):raise ValueError('Finite learning rate plan required')
 return c

def choose_development(pack,count=128,stage='foundation'):
 rows=[r for r in pack['records'] if r['stage']==stage and r['split']=='validation'];selected={}
 def take(items,n):
  for r in sorted(items,key=lambda x:d.digest('v53-dev:'+x['id']))[:n]:selected[r['id']]=r
 if stage=='foundation':
  take([r for r in rows if r['task']=='element_direction' and r['context']=='closed_book'],25)
  take([r for r in rows if r['task']!='element_direction' and r['context']=='closed_book' and r.get('checks') and r['checks']['kind']!='mentions'],40)
  take([r for r in rows if r['task']!='element_direction' and r['context']=='provided_evidence' and r.get('checks') and r['checks']['kind']!='mentions'],40)
 else:
  take([r for r in rows if r['task']=='chart_positions'],24)
  take([r for r in rows if r['task']=='chart_direction'],40)
  take([r for r in rows if r['context']=='provided_evidence' and r.get('checks') and r['checks']['kind']!='mentions'],40)
  for domain in sorted({r['domain'] for r in rows if r['task'] in ('case_source_chain','author_source_application')}):
   complete=[r for r in rows if r['task']=='case_source_chain' and r['domain']==domain]
   take(complete or [r for r in rows if r['task']=='author_source_application' and r['domain']==domain],1)
 for r in sorted(rows,key=lambda x:d.digest('v53-fill:'+x['id'])):
  if len(selected)>=count:break
  selected.setdefault(r['id'],r)
 return list(selected.values())[:count]

def prepare(job,c,*,tokenizer=None,pack=None,testing=False):
 from src.wengua import qwen_book_loop_v3 as storage,qwen8b_book_v1 as old
 target=Path(job)/'prepared'
 if (target/'receipt.json').exists():
  meta=d.read(target/'receipt.json')
  if meta['config']!=c or meta['dataset_release_sha256']!=d.digest(RELEASE.read_text()):raise ValueError('Prepared curriculum contract changed')
  for n,h in meta['file_sha256'].items():
   if storage.file_sha(target/n)!=h:raise ValueError('Prepared data changed')
  return meta
 if target.exists():raise ValueError('Incomplete preparation preserved')
 pack=pack if testing and pack is not None else verify_data();tokenizer=tokenizer or storage.load_book_tokenizer(c);tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token
 dev=choose_development(pack,c.get('development_generation_rows',128),c['curriculum_stage']);ids={r['id'] for r in dev};train=[];val=[]
 for core in pack['records']:
  if core['stage']!=c['curriculum_stage'] or core['split']=='diagnostic' or core['split']=='validation' and core['id'] not in ids:continue
  for view in storage.target_views(core,0):
   enc,size=storage.encode(view,tokenizer,c['max_seq_length'])
   if enc is None:raise ValueError('Whole target does not fit; revise curriculum: '+core['id'])
   item=view|{'encoded':enc,'tokens':size}
   if core['split']=='train':
    repeat=c['foundation_repeat'] if c['curriculum_stage']=='foundation' else c['application_repeat']
    for i in range(repeat):train.append(item|{'id':item['id']+':review'+str(i)})
   else:val.append(item)
 if not train or not val:raise ValueError('Empty stage train/development')
 gen,skips=storage.generation_rows(dev,tokenizer,c)
 if skips:raise ValueError('Evaluation exceeds whole context budget')
 diagnostics=[r for r in pack['records'] if r['split']=='diagnostic'];diag,skips=storage.generation_rows(diagnostics,tokenizer,c)
 if skips:raise ValueError('Diagnostic exceeds context budget')
 pending=target.with_name('prepared.pending');pending.mkdir(parents=True);tokenizer.save_pretrained(pending/'tokenizer');train.sort(key=lambda r:d.digest('v53-order:'+r['id']))
 for n,rows in [('cycle_01.jsonl.gz',train),('smoke.jsonl.gz',old.stress_rows(train,c['smoke_steps'],c['gradient_accumulation_steps'])),('validation_loss.jsonl.gz',val),('validation_prompts.jsonl.gz',gen),('foundation_development_prompts.jsonl.gz',choose_development(pack,c.get('development_generation_rows',128),'foundation')),('seen_diagnostic_prompts.jsonl.gz',diag),('source_units.jsonl.gz',pack['units'].values())]:storage.write_rows(pending/n,rows)
 meta={'schema':SCHEMA,'config':c,'dataset_release_sha256':d.digest(RELEASE.read_text()),'training_rows':len(train),'repetition_is_not_new_knowledge':True,'max_actual_train_tokens':max(r['tokens'] for r in train),'whole_targets_preserved':True,'original_test_used':False,'file_sha256':{p.relative_to(pending).as_posix():storage.file_sha(p) for p in pending.rglob('*') if p.is_file()}}
 d.write(pending/'receipt.json',meta);pending.rename(target);return meta

def profile(*,testing=False):
 from src.wengua import qwen8b_book_v1 as old,qwen_book_continuous_v1 as engine,qwen_course_training_v4 as integrity
 from wengua_course import fit as continuation
 p=old.profile(testing=testing);b=p.book;b.ROOT=ROOT;b.SCHEMA=SCHEMA;b.normalized_kernel_hashes=hashes;b.prepare=lambda job,c,**kwargs:prepare(job,c,testing=testing,**kwargs)
 native_loader=b.load_book_cuda
 def normalized_loader(c,training):
  from src.wengua.qwen_finetune import prepare_frozen_base
  base,compute=native_loader(c,training)
  return prepare_frozen_base(base,gradient_checkpointing=False),compute
 # The legacy pending-generation resume branch skips its usual training
 # preparation. Normalize at the loader too, before any adapter is attached.
 b.load_book_cuda=normalized_loader
 native_generate=b.generate_rows
 def generate_rows(model,*args,**kwargs):
  from unittest.mock import patch
  method=model.generate
  def last_position_only(*a,**kw):
   kw.setdefault('logits_to_keep',1);return method(*a,**kw)
  with patch.object(model,'generate',last_position_only):return native_generate(model,*args,**kwargs)
 b.generate_rows=generate_rows
 e=old.bind(engine,'wengua_course_v53._engine',SCHEMA=SCHEMA,book=b);e.validate_structure=lambda c:validate(c,testing);e.code_hashes=hashes;e.base_config=lambda:d.read(CONFIG)
 raw=FunctionType(continuation.fit.__code__,e.__dict__,'fit',continuation.fit.__defaults__);raw.__kwdefaults__=continuation.fit.__kwdefaults__
 def fit(job,c,deadline,*,extra_callbacks=(),**kwargs):
  verify_snapshot(c)
  if kwargs.get('cpu_factory'):
   factory=kwargs['cpu_factory']
   def normalized_fixture():
    from src.wengua.qwen_finetune import prepare_frozen_base
    tok,base,compute=factory()
    return tok,prepare_frozen_base(base,gradient_checkpointing=False),compute
   kwargs['cpu_factory']=normalized_fixture
  with integrity.training_integrity() as check:return raw(job,c,deadline,extra_callbacks=(*extra_callbacks,check),**kwargs)
 e.fit=fit
 def open_job(path,c,*,resume):
  if not c.get('initialization_only'):verify_snapshot(c)
  path=b.private_output(path);contract={'schema':SCHEMA,'config':e.validate(c),'code_sha256':hashes(),'dataset_release_sha256':d.digest(RELEASE.read_text()),'test_used_for_selection':False,'initialization':c.get('initialization_label','official_new_adapter')}
  if path.exists():
   if not resume or not (path/'job_contract.json').exists() or d.read(path/'job_contract.json')!=contract:raise ValueError('Job contract changed; preserve previous output')
  else:path.mkdir(parents=True);d.write(path/'job_contract.json',contract)
  return path
 e.open_job=open_job;p.engine=e;return p

def prepared_prompts(job,stage):
 from src.wengua.qwen_book_loop_v3 import read_rows,file_sha
 job=Path(job);meta=d.read(job/'prepared/receipt.json');c=meta['config']
 name='validation_prompts.jsonl.gz' if stage==c['curriculum_stage'] else 'foundation_development_prompts.jsonl.gz'
 path=job/'prepared'/name
 if file_sha(path)!=meta['file_sha256'][name]:raise ValueError('Prepared evaluation source changed')
 return read_rows(path)
