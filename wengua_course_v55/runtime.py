"""V55 versioned contracts; immutable App/V53/V54 kernels are reused."""
from pathlib import Path
from types import FunctionType
from copy import deepcopy
import gzip,json,sys,os
from wengua_course import data as d
from wengua_course_v53 import runtime as kernels
from wengua_course_v54 import runtime as previous
from . import data,checks
ROOT=previous.ROOT;DATA=ROOT/'local/course_v55_data';RELEASE=ROOT/'data/knowledge/course_v55/release.json';CONFIG=ROOT/'configs/course_v55.json';SCHEMA=SWEEP_SCHEMA=data.SCHEMA
configure=previous.configure;app_root=previous.app_root;verify_snapshot=previous.verify_snapshot;ALLOCATOR=previous.ALLOCATOR


def hashes():
 result=previous.hashes()
 for path in sorted((ROOT/'wengua_course_v55').glob('*.py')):result[path.relative_to(ROOT).as_posix()]=d.digest(path.read_text())
 for name in ['configs/course_v55.json','run_course_v55_worker.py','train_wengua_v55.cmd']:result[name]=d.digest((ROOT/name).read_text())
 result['v55_release']=d.digest(RELEASE.read_text());return result


def verify_data():
 release=d.read(RELEASE)
 if release['parent_v54_release_sha256']!=d.digest(previous.RELEASE.read_text()):raise ValueError('Frozen V54 release changed')
 for name,expected in release['files'].items():
  path=(DATA/name).resolve()
  if not path.is_relative_to(DATA.resolve()) or not path.is_file() or d.digest(path.read_bytes())!=expected:raise ValueError('Missing/changed V55 private source: '+name)
 parent=previous.verify_data();delta=json.loads(gzip.decompress((DATA/'course_delta.json.gz').read_bytes()));pack=data.compose(parent,delta);data.validate(pack);return pack


def validate(c,testing=False):
 if c['schema']!=SCHEMA:raise ValueError('V55 schema required')
 previous.validate(c|{'schema':previous.SCHEMA},testing)
 if c.get('skill_replay')!=data.REPLAY:raise ValueError('Pinned, transparent curriculum replay required')
 return c


def choose_development(pack,count=192,stage='application'):
 if stage=='foundation':return pack['legacy_v53_prompts'][:count]
 if stage=='v54':return pack['legacy_v54_prompts'][:128]
 rows=[r for r in pack['records'] if r['split']=='validation'];selected={}
 for key in sorted({r['provenance'].get('key') for r in rows if r['task'].startswith('source_star_state_')}):
  candidates=sorted((r for r in rows if r['provenance'].get('key')==key),key=lambda r:(r['provenance'].get('source_label_permutation',False),d.digest(r['id'])))
  if candidates:selected[candidates[0]['id']]=candidates[0]
 quotas={'direction':25,'positions':16,'given_direction':16,'chart_direction':18,'door_relation':16,'property':8,'identity':12,'condition':16,'typed_pair':16,'workflow':12,'author_evidence':12}
 for skill,number in quotas.items():
  candidates=[r for r in rows if r['skill']==skill]
  if skill=='workflow':
   order=['health','lost_property','employment','business_profit','marriage','education','pregnancy','travel_abroad','missing_person','litigation','criminal_case','weather']
   picked=[]
   for domain in order:
    options=sorted((r for r in candidates if r['domain']==domain),key=lambda r:(not r['provenance'].get('actual_production_rag_prompt',False),d.digest(r['id'])))
    if options:picked.append(options[0])
   candidates=picked+sorted((r for r in candidates if r not in picked),key=lambda r:d.digest(r['id']))
   for row in candidates[:number]:selected[row['id']]=row
   continue
  # Controlled missing-year/condition checks must not disappear behind open QA.
  candidates.sort(key=lambda r:(not bool(r.get('checks') and r['checks']['kind'] not in ['mentions']),d.digest('v55-dev:'+r['id'])))
  for row in candidates[:number]:selected[row['id']]=row
 for domain in sorted({r['domain'] for r in rows if r['skill']=='case'}):
  options=sorted((r for r in rows if r['skill']=='case' and r['domain']==domain),key=lambda r:(r['task']!='case_source_chain',d.digest(r['id'])))
  if options:selected[options[0]['id']]=options[0]
 for row in sorted(rows,key=lambda r:d.digest('v55-fill:'+r['id'])):
  if len(selected)>=count:break
  selected.setdefault(row['id'],row)
 return list(selected.values())[:count]


def _bound(f):
 result=FunctionType(f.__code__,globals(),f.__name__,f.__defaults__,f.__closure__);result.__kwdefaults__=f.__kwdefaults__;return result


_prepare=_bound(kernels.prepare)
def prepare(job,c,*,tokenizer=None,pack=None,testing=False):
 pack=pack if testing and pack is not None else verify_data();prepared=dict(pack);records=list(pack['records'])+list(pack.get('legacy_diagnostics',[]))
 if not testing:
  replay=[]
  for r in records:
   repeat=data.REPLAY.get(r.get('skill'),1) if r['split']=='train' else 1
   replay.extend(r|{'id':r['id']+':curriculum_replay'+str(i)} for i in range(1,repeat))
  records+=replay
 prepared['records']=records;meta=_prepare(job,c,tokenizer=tokenizer,pack=prepared,testing=True)
 from src.wengua.qwen_book_loop_v3 import write_rows,file_sha
 target=Path(job)/'prepared'
 for name,rows in [('v54_development_prompts.jsonl.gz',pack.get('legacy_v54_prompts',[])),('legacy_actual_rag_prompts.jsonl.gz',pack.get('legacy_actual_rag_rows',[]))]:
  path=target/name
  if path.exists():
   if name not in meta['file_sha256'] or meta['file_sha256'][name]!=file_sha(path):raise ValueError('Changed regression prompts')
  else:write_rows(path,rows);meta['file_sha256'][name]=file_sha(path)
 meta.update(unique_teacher_cores=sum(r['split']=='train' for r in pack['records']),skill_replay=data.REPLAY,replay_is_not_new_knowledge=True,legacy_actual_rag_rows=len(pack.get('legacy_actual_rag_rows',[])))
 d.write(target/'receipt.json',meta);return meta


profile=_bound(kernels.profile)
_prompts=_bound(kernels.prepared_prompts)
def prepared_prompts(job,stage):
 if stage not in ['v54','legacy_rag']:return _prompts(job,stage)
 from src.wengua.qwen_book_loop_v3 import read_rows,file_sha
 path=Path(job)/'prepared'/('v54_development_prompts.jsonl.gz' if stage=='v54' else 'legacy_actual_rag_prompts.jsonl.gz');meta=d.read(path.parent/'receipt.json')
 if file_sha(path)!=meta['file_sha256'][path.name]:raise ValueError('Regression prompt SHA mismatch')
 return read_rows(path)
