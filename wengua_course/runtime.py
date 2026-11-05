"""V5.2 namespace. Reads frozen App math, writes only Quant local outputs.
Heavy imports are lazy; no old score data or third-book corpus is loaded.
"""
from collections import defaultdict,Counter
from pathlib import Path
from types import FunctionType
import copy,json,math,os,sys
from . import data,checks

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'local/course_v52_data'
CONFIG=ROOT/'configs/course_v52.json'
RELEASE=ROOT/'data/knowledge/course_v52/release.json'
SWEEP_SCHEMA=SCHEMA=data.SCHEMA
ALLOCATOR='backend:native,max_split_size_mb:128'

def app_root():return Path(os.environ.get('WENGUA_APP_ROOT',str(ROOT.parent/'WenguaApp'))).resolve()
def configure(path=None):
 if path is not None:os.environ['WENGUA_APP_ROOT']=str(Path(path).resolve())
 app=app_root();manifest=data.read(ROOT/'configs/course_v52_app_files.json')
 for name,h in manifest.items():
  p=app/name
  if not p.is_file() or data.digest(p.read_text(encoding='utf8'))!=h:raise ValueError('Frozen WenguaApp dependency missing/changed: '+name)
 if str(app) not in sys.path:sys.path.insert(0,str(app))
 return app

def hashes():
 result={'app:'+name:h for name,h in data.read(ROOT/'configs/course_v52_app_files.json').items()}
 for name in ['train_wengua_v52.cmd','run_course_v52_worker.py','configs/course_v52.json','configs/course_v52_app_files.json',*['wengua_course/'+p.name for p in sorted((ROOT/'wengua_course').glob('*.py'))]]:
  result[name]=data.digest((ROOT/name).read_text(encoding='utf8'))
 result['dataset_release']=data.digest(RELEASE.read_text(encoding='utf8'))
 return result

def verify_data():
 release=data.read(RELEASE)
 for name,h in release['files'].items():
  p=(DATA/name).resolve()
  if not p.is_relative_to(DATA.resolve()) or not p.is_file() or data.digest(p.read_bytes())!=h:raise ValueError('V5.2 data missing/changed: '+name)
 pack=data.read(DATA/'pack.json')
 if pack['schema']!=SCHEMA:raise ValueError('Wrong curriculum version')
 data.validate(pack);return pack

def validate(c,testing=False):
 from src.wengua import qwen8b_book_v1 as old
 if c['schema']!=SCHEMA:raise ValueError('V5.2 schema required')
 if not testing and (c['lora_r'],c['lora_alpha'],c['lora_dropout'],set(c['target_modules']))!=(16,32,.05,{'q_proj','k_proj','v_proj','o_proj'}):raise ValueError('Preserve inherited reviewed LoRA structure')
 if not 2<=c['smoke_steps']<=100:raise ValueError('Bounded actual-data pressure steps required')
 old.validate(c|{'schema':old.BOOK_SCHEMA,'smoke_steps':100},testing=testing)
 if c['num_train_epochs'] not in (1,2) or c['learning_rate'] not in (5e-6,1e-5):
  if not testing:raise ValueError('Pinned two-epoch, two-rate repair study')
 return c

def choose_development(pack,count=128):
 rows=[r for r in pack['records'] if r['split']=='validation'];selected={}
 def add(r):
  if len(selected)<count:selected.setdefault(r['id'],r)
 critical={'named_pair_戊戊','named_pair_丁壬','hetu_金','gui_yin_water','jing_method','martial_hit','exam_self','exam_proxy','health_daughter_father','health_father_self','health_parent_child','health_switch_to_self','same_swap_戊戊','same_swap_己己','different_swap_丁壬','different_swap_壬丁'}
 for r in rows:
  if r.get('provenance',{}).get('key') in critical:add(r)
 # First cover every available case domain with actual authored summary review.
 for domain in sorted({r['domain'] for r in rows if r['case_ids']}):
  candidates=[r for r in rows if r['case_ids'] and r['domain']==domain and r['task']=='case_evidence_summary']
  if not candidates:candidates=[r for r in rows if r['case_ids'] and r['domain']==domain and r['task']=='author_evidence_to_conclusion']
  if candidates:add(min(candidates,key=lambda r:data.digest('v52-open-review:'+r['id'])))
 # One declared check per topic first, then expand evenly; no easy reading flood.
 buckets=defaultdict(list)
 for r in sorted(rows,key=lambda r:data.digest('v52-development:'+r['id'])):
  if checks.automatically_checkable(r.get('checks')):buckets[(r['domain'],r['task'])].append(r)
 for offset in range(max(map(len,buckets.values()),default=0)):
  for key in sorted(buckets):
   if offset<len(buckets[key]):add(buckets[key][offset])
 return list(selected.values())

def prepare(job,c,*,tokenizer=None,pack=None,testing=False):
 from src.wengua import qwen_book_loop_v3 as storage,qwen8b_book_v1 as old
 prepared=Path(job)/'prepared'
 if (prepared/'receipt.json').exists():
  meta=data.read(prepared/'receipt.json')
  if meta['config']!=c or meta['dataset_release_sha256']!=data.digest(RELEASE.read_text(encoding='utf8')):raise ValueError('Prepared data contract changed')
  for n,h in meta['file_sha256'].items():
   if storage.file_sha(prepared/n)!=h:raise ValueError('Prepared file changed: '+n)
  return meta
 if prepared.exists():raise ValueError('Incomplete preparation preserved')
 pack=pack if testing and pack is not None else verify_data();data.validate(pack)
 tokenizer=tokenizer or storage.load_book_tokenizer(c);tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token
 pending=prepared.with_name('prepared.pending');pending.mkdir(parents=True)
 tokenizer.save_pretrained(pending/'tokenizer')
 train=[];dev=choose_development(pack,c.get('development_generation_rows',128));dev_ids={r['id'] for r in dev};validation=[]
 for core in pack['records']:
  if core['split']=='diagnostic':continue
  if core['split']=='validation' and core['id'] not in dev_ids:continue
  for view in storage.target_views(core,0):
   enc,size=storage.encode(view,tokenizer,c['max_seq_length'])
   if enc is None:raise ValueError('Complete target exceeds context; curriculum revision required: '+core['id'])
   encoded=view|{'encoded':enc,'tokens':size}
   if core['split']=='train':
    # Explicit review weight, never described as new knowledge or independent samples.
    repeats=3 if core['format']=='conditional_dialogue' else 1
    for repeat in range(repeats):train.append(encoded|{'id':encoded['id']+':review'+str(repeat)})
   else:validation.append(encoded)
 if not train or not validation:raise ValueError('Missing train/development data')
 accepted,skips=storage.generation_rows(dev,tokenizer,c)
 if skips:raise ValueError('Development prompt or target exceeds budget: '+data.dumps(skips))
 diagnostic=[r for r in pack['records'] if r['split']=='diagnostic'][:8]
 accepted_diag,skips=storage.generation_rows(diagnostic,tokenizer,c)
 if skips:raise ValueError('Diagnostic budget changed')
 train.sort(key=lambda r:data.digest('v52-train:'+r['id']))
 storage.write_rows(pending/'cycle_01.jsonl.gz',train)
 smoke=old.stress_rows(train,c['smoke_steps'],c['gradient_accumulation_steps']);storage.write_rows(pending/'smoke.jsonl.gz',smoke)
 storage.write_rows(pending/'validation_loss.jsonl.gz',validation)
 storage.write_rows(pending/'validation_prompts.jsonl.gz',accepted)
 storage.write_rows(pending/'seen_diagnostic_prompts.jsonl.gz',accepted_diag)
 storage.write_rows(pending/'source_units.jsonl.gz',pack['units'].values())
 data.write(pending/'development_manifest.json',{'task_ids':[r['id'] for r in accepted],'case_domains':sorted({r['domain'] for r in accepted if r['case_ids']}),'case_families':sorted({r['group'] for r in accepted if r['case_ids']}),'objective_checks':sum(checks.automatically_checkable(r['checks']) for r in accepted),'manual_case_reviews':sum(not checks.automatically_checkable(r['checks']) for r in accepted),'ancestor_unseen_claimed':False,'original_test_used':False})
 data.write(pending/'stress_plan.json',{'updates':c['smoke_steps'],'rows':len(smoke),'max_actual_tokens':max(r['tokens'] for r in smoke),'longest_and_dense_targets_included':True,'full_context_capacity_proven':False})
 meta={'schema':SCHEMA,'config':c,'dataset_release_sha256':data.digest(RELEASE.read_text(encoding='utf8')),'file_sha256':{p.relative_to(pending).as_posix():storage.file_sha(p) for p in pending.rglob('*') if p.is_file()},'training_rows':len(train),'unique_train_core_records':sum(r['split']=='train' for r in pack['records']),'training_formats_with_explicit_review_weight':dict(Counter(r['format'] for r in train)),'max_actual_train_tokens':max(r['tokens'] for r in train),'whole_targets_preserved':True,'final_test_prepared':False}
 data.write(pending/'receipt.json',meta);pending.rename(prepared);return meta

def verify_snapshot(c):
 path=Path(c['book_parent_snapshot']).resolve()
 if not path.is_relative_to((ROOT/'local').resolve()):raise ValueError('Parent snapshots must be in private Quant local')
 proof=data.read(path.parent/'parent_provenance.json')
 if data.digest((path.parent/'parent_provenance.json').read_bytes())!=c['book_parent_provenance_sha256']:raise ValueError('Parent proof changed')
 for n,h in proof['files'].items():
  if data.digest((path/n).read_bytes())!=h:raise ValueError('Parent snapshot changed')
 for k in ['model_id','revision','lora_r','lora_alpha','lora_dropout','target_modules']:
  if c[k]!=proof['parent_config'][k]:raise ValueError('Inherited structure changed: '+k)
 return path

def profile(*,testing=False):
 configure()
 from src.wengua import qwen8b_book_v1 as old,qwen_book_continuous_v1 as engine,qwen_course_training_v4 as integrity
 from . import fit as continuation
 p=old.profile(testing=testing);b=p.book;b.ROOT=ROOT;b.SCHEMA=SCHEMA;b.normalized_kernel_hashes=hashes
 b.prepare=lambda job,c,**kwargs:prepare(job,c,testing=testing,**kwargs)
 b.load_config=lambda **kwargs:validate(data.read(CONFIG),testing)
 e=old.bind(engine,'wengua_course._v52_engine',SCHEMA=SCHEMA,book=b)
 e.validate_structure=lambda c:validate(c,testing);e.code_hashes=hashes;e.base_config=lambda:data.read(CONFIG)
 rawfit=FunctionType(continuation.fit.__code__,e.__dict__,'fit',continuation.fit.__defaults__);rawfit.__kwdefaults__=continuation.fit.__kwdefaults__
 def checked_fit(job,c,deadline,*,extra_callbacks=(),**kwargs):
  verify_snapshot(c)
  with integrity.training_integrity() as check:return rawfit(job,c,deadline,extra_callbacks=(*extra_callbacks,check),**kwargs)
 e.fit=checked_fit
 def open_job(path,c,*,resume):
  verify_snapshot(c);path=b.private_output(path)
  contract={'schema':SCHEMA,'config':e.validate(c),'code_sha256':hashes(),'dataset_release_sha256':data.digest(RELEASE.read_text(encoding='utf8')),'initial_adapter':c['book_parent_snapshot'],'initialization':'book_parent_with_new_phase_optimizer','parent_provenance_sha256':c['book_parent_provenance_sha256'],'test_used_for_selection':False}
  if path.exists():
   if not resume or data.read(path/'job_contract.json')!=contract:raise ValueError('Run contract changed; preserve prior output and use a new output')
  else:path.mkdir(parents=True);data.write(path/'job_contract.json',contract)
  return path
 e.open_job=open_job;p.engine=e;return p
