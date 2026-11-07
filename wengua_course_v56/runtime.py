"""Versioned V56 phases reuse frozen CUDA, exact resume and source dependencies."""
from pathlib import Path
from types import FunctionType
from collections import Counter
import gzip,json
from wengua_course import data as d
from wengua_course_v53 import runtime as kernels
from wengua_course_v55 import runtime as previous
from . import data,checks

ROOT=previous.ROOT;DATA=ROOT/'local/course_v56_data';RELEASE=ROOT/'data/knowledge/course_v56/release.json';CONFIG=ROOT/'configs/course_v56.json';SCHEMA=SWEEP_SCHEMA=data.SCHEMA
configure=previous.configure;app_root=previous.app_root;verify_snapshot=previous.verify_snapshot;ALLOCATOR=previous.ALLOCATOR


def hashes():
    result=previous.hashes()
    for path in sorted((ROOT/'wengua_course_v56').glob('*.py')):result[path.relative_to(ROOT).as_posix()]=d.digest(path.read_text())
    for name in ['configs/course_v56.json','run_course_v56_worker.py','train_wengua_v56.cmd']:
        result[name]=d.digest((ROOT/name).read_text())
    result['v56_release']=d.digest(RELEASE.read_text());return result


def verify_data():
    release=d.read(RELEASE)
    if release['parent_v55_release_sha256']!=d.digest(previous.RELEASE.read_text()):raise ValueError('Frozen V55 release changed')
    for name,sha in release['files'].items():
        path=(DATA/name).resolve()
        if not path.is_relative_to(DATA.resolve()) or not path.is_file() or d.digest(path.read_bytes())!=sha:raise ValueError('V56 source delivery missing/changed: '+name)
    pack=data.compose(previous.verify_data(),json.loads(gzip.decompress((DATA/'course_delta.json.gz').read_bytes())))
    data.validate(pack);return pack


def validate(c,testing=False):
    if c['schema']!=SCHEMA:raise ValueError('V56 schema required')
    # Kernel accepts the same frozen 8B structure; extra phase choices are ours.
    kernels.validate(c|{'schema':kernels.SCHEMA},testing)
    if c.get('course_phase') not in ['repair','full'] or c.get('skill_replay')!={}:raise ValueError('Explicit V56 phase required')
    if c['num_train_epochs'] not in [1,2,4]:raise ValueError('Finite phase epochs required')
    return c


def choose_development(pack,count=192,stage='application'):
    if stage=='foundation':return pack['legacy_v53_prompts'][:count]
    if stage=='v54':return pack['legacy_v54_prompts'][:128]
    if stage=='v55':return pack['legacy_v55_prompts'][:192]
    if 'fixed_development_ids' in pack:
        byid={r['id']:r for r in pack['records']};return [byid[i] for i in pack['fixed_development_ids'][:count]]
    return select_development(pack['records'],count,repair=pack.get('active_phase')=='repair')


def select_development(records,count=192,*,repair=False):
    rows=[r for r in records if r['split']=='validation'];selected={}
    quotas={'direction':16,'given_direction':16,'chart_direction':12,'door_relation':12,'identity':8,'condition':8,'typed_pair':8,'property':4,'positions':8,'source_fact':4} if repair else {'direction':25,'given_direction':16,'chart_direction':18,'door_relation':16,'identity':12,'condition':16,'typed_pair':16,'property':8,'positions':12,'source_fact':5,'workflow':12,'author_local':12}
    for skill,n in quotas.items():
        bucket=[r for r in rows if r['skill']==skill]
        bucket.sort(key=lambda r:(not r['provenance'].get('repair_pool',False),d.digest('v56-dev:'+r['id'])))
        if skill=='workflow':
            order=['health','lost_property','employment','business_profit','marriage','education','pregnancy','travel_abroad','missing_person','litigation','criminal_case','weather']
            picked=[next((r for r in bucket if r['domain']==domain),None) for domain in order]
            bucket=[r for r in picked if r]+[r for r in bucket if r not in picked]
        # Deterministic balance across the five relations, not one easy label.
        if skill in ['direction','given_direction','chart_direction']:
            groups={x:[r for r in bucket if (r.get('checks') or {}).get('answer')==x] for x in ['生','受生','克','受克','比和']};bucket=[]
            for i in range(max((len(b) for b in groups.values()),default=0)):
                for b in groups.values():
                    if i<len(b):bucket.append(b[i])
        for r in bucket[:n]:selected[r['id']]=r
    if not repair:
        for domain in sorted({r['domain'] for r in rows if r['skill']=='case'}):
            bucket=sorted((r for r in rows if r['skill']=='case' and r['domain']==domain),key=lambda r:d.digest(r['id']))
            if bucket:selected[bucket[0]['id']]=bucket[0]
    for r in sorted(rows,key=lambda r:d.digest('v56-fill:'+r['id'])):
        if len(selected)>=count:break
        selected.setdefault(r['id'],r)
    return list(selected.values())[:count]


def repair_training(records,count=1600):
    quotas={'positions':160,'direction':250,'given_direction':350,'door_relation':128,'identity':120,'typed_pair':128,'source_fact':96,'condition':128,'property':40,'chart_direction':200}
    result=[]
    for skill,n in quotas.items():
        bucket=sorted((r for r in records if r['split']=='train' and r['skill']==skill),key=lambda r:(not r['provenance'].get('repair_pool',False),d.digest('v56-repair:'+r['id'])))
        if skill in ['typed_pair','source_fact']:
            seen=set();first=[];rest=[]
            for r in bucket:
                key=(r.get('checks') or {}).get('pair') or r['provenance'].get('key') or r['id']
                if key in seen:rest.append(r)
                else:seen.add(key);first.append(r)
            bucket=first+rest
        if not bucket:continue
        for i in range(n):
            r=bucket[i%len(bucket)]
            result.append(r if i<len(bucket) else r|{'id':r['id']+':declared_repair_replay'+str(i)})
    return result[:count]


def _bound(f):
    result=FunctionType(f.__code__,globals(),f.__name__,f.__defaults__,f.__closure__);result.__kwdefaults__=f.__kwdefaults__;return result


_prepare=_bound(kernels.prepare)
def prepare(job,c,*,tokenizer=None,pack=None,testing=False):
    pack=pack if testing and pack is not None else verify_data();prepared=dict(pack)
    if c['course_phase']=='repair' and not testing:
        train=repair_training(pack['records']);dev=select_development(pack['records'],96,repair=True)
        prepared.update(records=train+dev,active_phase='repair',fixed_development_ids=[r['id'] for r in dev])
    prepared['records']=list(prepared['records'])+list(pack.get('legacy_diagnostics',[]))
    meta=_prepare(job,c,tokenizer=tokenizer,pack=prepared,testing=True)
    from src.wengua.qwen_book_loop_v3 import write_rows,file_sha
    for name,rows in [('v55_development_prompts.jsonl.gz',pack.get('legacy_v55_prompts',[])),('v54_development_prompts.jsonl.gz',pack.get('legacy_v54_prompts',[])),('legacy_actual_rag_prompts.jsonl.gz',pack.get('legacy_actual_rag_rows',[]))]:
        path=Path(job)/'prepared'/name
        if path.exists():
            if meta['file_sha256'].get(name)!=file_sha(path):raise ValueError('Regression source changed')
        else:write_rows(path,rows);meta['file_sha256'][name]=file_sha(path)
    meta.update(course_phase=c['course_phase'],replay_is_not_new_knowledge=True,repair_training_unique_cores=len({r['id'].split(':declared_repair_replay')[0] for r in prepared['records'] if r['split']=='train'}))
    d.write(Path(job)/'prepared/receipt.json',meta);return meta


_profile=_bound(kernels.profile)
def profile(*,testing=False):
    p=_profile(testing=testing);native=p.book.generate_rows
    def visible_generate(model,tokenizer,rows,path,c,deadline,tag,*args,**kwargs):
        from unittest.mock import patch
        method=model.generate;number=0
        if Path(path).exists():
            ids={r['id'] for r in rows}
            number=sum(json.loads(line).get('id') in ids for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip())
        def call(*a,**kw):
            nonlocal number
            number+=1;p.book.progress('v56_answer_progress',tag=tag,answer=number,total=len(rows),file=str(path))
            return method(*a,**kw)
        with patch.object(model,'generate',call):return native(model,tokenizer,rows,path,c,deadline,tag,*args,**kwargs)
    p.book.generate_rows=visible_generate;return p


_prompts=_bound(kernels.prepared_prompts)
def prepared_prompts(job,stage):
    names={'v55':'v55_development_prompts.jsonl.gz','v54':'v54_development_prompts.jsonl.gz','legacy_rag':'legacy_actual_rag_prompts.jsonl.gz'}
    if stage not in names:return _prompts(job,stage)
    from src.wengua.qwen_book_loop_v3 import read_rows,file_sha
    path=Path(job)/'prepared'/names[stage];meta=d.read(path.parent/'receipt.json')
    if file_sha(path)!=meta['file_sha256'][path.name]:raise ValueError('Regression prompt hash changed')
    return read_rows(path)
