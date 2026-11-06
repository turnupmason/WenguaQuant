"""New contracts and data, using the unchanged tested V53/App kernels."""
from pathlib import Path
from types import FunctionType
import sys,os
from wengua_course import data as d
from wengua_course_v53 import runtime as base
from . import data,checks

ROOT=base.ROOT;DATA=ROOT/'local/course_v54_data';RELEASE=ROOT/'data/knowledge/course_v54/release.json';CONFIG=ROOT/'configs/course_v54.json';SCHEMA=SWEEP_SCHEMA=data.SCHEMA
configure=base.configure;app_root=base.app_root;verify_snapshot=base.verify_snapshot;ALLOCATOR=base.ALLOCATOR

def hashes():
 result=base.hashes()
 for path in sorted((ROOT/'wengua_course_v54').glob('*.py')):result[path.relative_to(ROOT).as_posix()]=d.digest(path.read_text())
 for name in ['configs/course_v54.json','train_wengua_v54.cmd','run_course_v54_worker.py']:result[name]=d.digest((ROOT/name).read_text())
 result['v54_release']=d.digest(RELEASE.read_text());return result

def verify_data():
 release=d.read(RELEASE)
 if release['parent_v53_release_sha256']!=d.digest(base.RELEASE.read_text()):raise ValueError('Frozen V53 source release changed')
 for name,expected in release['files'].items():
  path=(DATA/name).resolve()
  if not path.is_relative_to(DATA.resolve()) or not path.is_file() or d.digest(path.read_bytes())!=expected:raise ValueError('V54 source-use course missing/changed: '+name)
 parent=base.verify_data()
 if d.digest((base.DATA/'pack.json').read_bytes())!=release['parent_v53_pack_sha256']:raise ValueError('Frozen parent corpus differs')
 pack=data.compose(parent,d.read(DATA/'course_delta.json'));data.validate(pack);return pack

def validate(c,testing=False):
 if c['schema']!=SCHEMA:raise ValueError('V54 schema required')
 base.validate(c|{'schema':base.SCHEMA},testing)
 if c['curriculum_stage']!='application':raise ValueError('V54 runs the complete source-use stage, not a 256-step prerequisite')
 return c

def choose_development(pack,count=128,stage='application'):
 if stage=='foundation':return pack['legacy_v53_prompts'][:count]
 rows=[r for r in pack['records'] if r['split']=='validation'];selected={}
 quotas={'direction':25,'door_relation':12,'identity':10,'property':8,'source_fact':16,'typed_pair':12,'condition':6,'chart_direction':16}
 for skill,number in quotas.items():
  candidates=sorted((r for r in rows if r.get('skill')==skill),key=lambda r:d.digest('v54-dev:'+r['id']))
  for row in candidates[:number]:selected[row['id']]=row
 for domain in sorted({r['domain'] for r in rows if r.get('skill')=='case'}):
  candidates=sorted((r for r in rows if r.get('skill')=='case' and r['domain']==domain),key=lambda r:(r['task']!='case_source_chain',d.digest(r['id'])))
  if candidates:selected[candidates[0]['id']]=candidates[0]
 for row in sorted(rows,key=lambda r:d.digest('v54-fill:'+r['id'])):
  if len(selected)>=count:break
  selected.setdefault(row['id'],row)
 return list(selected.values())[:count]

def _bound(function):
 result=FunctionType(function.__code__,globals(),function.__name__,function.__defaults__,function.__closure__);result.__kwdefaults__=function.__kwdefaults__;return result

_prepare=_bound(base.prepare)
def prepare(job,c,*,tokenizer=None,pack=None,testing=False):
 pack=pack if testing and pack is not None else verify_data()
 pack=dict(pack);pack['records']=list(pack['records'])+list(pack.get('legacy_diagnostics',[]))
 meta=_prepare(job,c,tokenizer=tokenizer,pack=pack,testing=True)
 from src.wengua.qwen_book_loop_v3 import write_rows,file_sha
 target=Path(job)/'prepared';extra=target/'legacy_actual_rag_prompts.jsonl.gz'
 if extra.exists():return meta
 legacy=[]
 if not testing:
  source=Path(c['legacy_source_run']);paths=list((source/'trials/foundation_v52_comparison/rag_probe_sources').glob('foundation_final_actual_rag_*.json'))
  paths=[p for p in paths if not p.name.endswith('.checks.json')]
  if not paths:raise FileNotFoundError('Preserve the completed V53 RAG probe sources; no old answers are used as teachers')
  path=max(paths,key=lambda p:int(p.stem.rsplit('_',1)[1]));packets=d.read(path)['packets'];specs={s['key']:s for s in base.verify_data()['reviewed_specs']}
  for rid,packet in packets.items():
   key=rid.split(':')[-1];answer=specs[key]['answer']
   legacy.append({'id':rid,'format':'foundation_qa','task':'legacy_actual_rag','group':'seen_v53_rag:'+key,'source_mode':'seen_regression_only_not_training','source_ids':[x['id'] for x in packet['references']],'messages':packet['messages']+[{'role':'assistant','content':answer}],'answer':answer,'legacy_packet':packet})
  if len(legacy)!=18:raise ValueError('The full18 original V53 RAG regression prompts are required')
 write_rows(extra,legacy);meta['file_sha256'][extra.name]=file_sha(extra);meta['legacy_actual_rag_rows']=len(legacy);meta['seen_regression_not_new_blind_test']=True;d.write(target/'receipt.json',meta);return meta

profile=_bound(base.profile)
_prompts=_bound(base.prepared_prompts)
def prepared_prompts(job,stage):
 if stage!='legacy_rag':return _prompts(job,stage)
 from src.wengua.qwen_book_loop_v3 import read_rows,file_sha
 path=Path(job)/'prepared/legacy_actual_rag_prompts.jsonl.gz';meta=d.read(path.parent/'receipt.json')
 if file_sha(path)!=meta['file_sha256'][path.name]:raise ValueError('Legacy RAG regression source changed')
 return read_rows(path)
