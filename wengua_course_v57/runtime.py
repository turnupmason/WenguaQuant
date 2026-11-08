"""Frozen CUDA paths; mixed pilot, fixed micro and final review; exact resume."""
from pathlib import Path
from types import FunctionType
import gzip,json,gc,time
from wengua_course import data as d
from wengua_course_v53 import runtime as kernels
from wengua_course_v56 import runtime as previous
from . import data,checks,evaluation
ROOT=previous.ROOT;DATA=ROOT/'local/course_v57_data';RELEASE=ROOT/'data/knowledge/course_v57/release.json';CONFIG=ROOT/'configs/course_v57.json';SCHEMA=SWEEP_SCHEMA=data.SCHEMA
configure=previous.configure;app_root=previous.app_root;verify_snapshot=previous.verify_snapshot;ALLOCATOR=previous.ALLOCATOR


def hashes():
    result=previous.hashes()
    for path in sorted((ROOT/'wengua_course_v57').glob('*.py')):result[path.relative_to(ROOT).as_posix()]=d.digest(path.read_text())
    for name in ['configs/course_v57.json','run_course_v57_worker.py','train_wengua_v57.cmd']:result[name]=d.digest((ROOT/name).read_text())
    result['v57_release']=d.digest(RELEASE.read_text());return result


def verify_data():
    release=d.read(RELEASE)
    if release['parent_v56_release_sha256']!=d.digest(previous.RELEASE.read_text()):raise ValueError('Frozen V56 release changed')
    for name,sha in release['files'].items():
        path=(DATA/name).resolve()
        if not path.is_relative_to(DATA.resolve()) or not path.is_file() or d.digest(path.read_bytes())!=sha:raise ValueError('V57 delivery missing/changed: '+name)
    p=data.compose(previous.verify_data(),json.loads(gzip.decompress((DATA/'course_delta.json.gz').read_bytes())));data.validate(p)
    if len(p.get('v57_actual_rag_rows',[]))!=18 or len(p.get('fixed_micro_ids',[]))!=64 or len(p.get('fixed_final_ids',[]))!=192:raise ValueError('V57 required real retrieval/micro/final coverage missing')
    return p


def validate(c,testing=False):
    if c['schema']!=SCHEMA:raise ValueError('V57 schema required')
    kernels.validate(c|{'schema':kernels.SCHEMA},testing)
    if c.get('course_phase') not in ['pilot','full'] or c.get('skill_replay')!={} or c['num_train_epochs'] not in [1,2]:raise ValueError('Finite mixed V57 phase required')
    if c.get('generate_official_baseline') or c.get('run_seen_diagnostics'):raise ValueError('Redundant answer generation disabled by this contract')
    return c


def choose_development(pack,count=64,stage='application'):
    if stage=='foundation':return pack.get('legacy_v53_prompts',[])[:count]
    if pack.get('fixed_micro_ids'):
        byid={r['id']:r for r in pack['records']};return [byid[k] for k in pack['fixed_micro_ids'][:count]]
    return evaluation.micro(pack['records'])[:count]


def _bound(f):
    g=FunctionType(f.__code__,globals(),f.__name__,f.__defaults__,f.__closure__);g.__kwdefaults__=f.__kwdefaults__;return g


_prepare=_bound(kernels.prepare)
def prepare(job,c,*,tokenizer=None,pack=None,testing=False):
    pack=pack if testing and pack is not None else verify_data();prepared=dict(pack)
    target=Path(job)/'prepared'
    if target.exists() and not (target/'receipt.json').exists():target.rename(target.with_name('prepared.incomplete-'+str(time.time_ns())))
    pending=target.with_name('prepared.pending')
    if pending.exists():pending.rename(pending.with_name('prepared.pending-preserved-'+str(time.time_ns())))
    if not testing:
        micro=choose_development(pack)
        prepared['records']=(evaluation.pilot_training(pack['records']) if c['course_phase']=='pilot' else [r for r in pack['records'] if r['split']=='train'])+micro
    meta=_prepare(job,c,tokenizer=tokenizer,pack=prepared,testing=True)
    from src.wengua.qwen_book_loop_v3 import write_rows,file_sha
    extras=[('final_review_prompts.jsonl.gz',[r for r in pack['records'] if r['id'] in set(pack.get('fixed_final_ids',[]))]),('actual_rag_prompts.jsonl.gz',pack.get('v57_actual_rag_rows',[])),('legacy_v55_prompts.jsonl.gz',pack.get('legacy_v55_prompts',[])),('legacy_actual_rag_prompts.jsonl.gz',evaluation.legacy_rag(pack) if pack.get('legacy_actual_rag_rows') else [])]
    for name,rows in extras:
        path=Path(job)/'prepared'/name
        if path.exists():
            if name in meta['file_sha256'] and meta['file_sha256'][name]!=file_sha(path):raise ValueError('Prepared regression changed')
            if name not in meta['file_sha256']:
                from src.wengua.qwen_book_loop_v3 import read_rows
                if read_rows(path)!=rows:raise ValueError('Interrupted prepared regression differs from source')
                meta['file_sha256'][name]=file_sha(path)
        else:write_rows(path,rows);meta['file_sha256'][name]=file_sha(path)
    meta.update(course_phase=c['course_phase'],unique_training_cores=sum(r['split']=='train' for r in prepared['records']),pilot_contains_cases_and_workflows=True)
    d.write(Path(job)/'prepared/receipt.json',meta);return meta


def profile(*,testing=False):
    from src.wengua import qwen8b_book_v1 as old,qwen_book_continuous_v1 as engine,qwen_course_training_v4 as integrity
    from . import fit as continuation
    p=old.profile(testing=testing);b=p.book;b.ROOT=ROOT;b.SCHEMA=SCHEMA;b.normalized_kernel_hashes=hashes;b.prepare=lambda job,c,**kw:prepare(job,c,testing=testing,**kw)
    loader=b.load_book_cuda
    def normalized_loader(c,training):
        from src.wengua.qwen_finetune import prepare_frozen_base
        model,compute=loader(c,training);return prepare_frozen_base(model,gradient_checkpointing=False),compute
    b.load_book_cuda=normalized_loader;native=b.generate_rows
    def generate_rows(model,tok,rows,path,c,deadline,tag,*args,**kwargs):
        from unittest.mock import patch
        method=model.generate;number=0
        if Path(path).exists():
            ids={r['id'] for r in rows};number=sum(json.loads(line).get('id') in ids for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip())
        def call(*a,**kw):
            nonlocal number
            number+=1;kw.setdefault('logits_to_keep',1);b.progress('v57_answer_progress',phase=tag,answer=number,total=len(rows),file=str(path));return method(*a,**kw)
        with patch.object(model,'generate',call):return native(model,tok,rows,path,c,deadline,tag,*args,**kwargs)
    b.generate_rows=generate_rows
    e=old.bind(engine,'wengua_course_v57._engine',SCHEMA=SCHEMA,book=b);e.validate_structure=lambda c:validate(c,testing);e.code_hashes=hashes;e.base_config=lambda:d.read(CONFIG)
    raw=FunctionType(continuation.fit.__code__,e.__dict__,'fit',continuation.fit.__defaults__);raw.__kwdefaults__=continuation.fit.__kwdefaults__
    def fit(job,c,deadline,*,extra_callbacks=(),**kw):
        verify_snapshot(c)
        if kw.get('cpu_factory'):
            factory=kw['cpu_factory']
            def normalized_fixture():
                from src.wengua.qwen_finetune import prepare_frozen_base
                tok,base,compute=factory();return tok,prepare_frozen_base(base,gradient_checkpointing=False),compute
            kw['cpu_factory']=normalized_fixture
        with integrity.training_integrity() as check:return raw(job,c,deadline,extra_callbacks=(*extra_callbacks,check),**kw)
    e.fit=fit
    def open_job(path,c,*,resume):
        if not c.get('initialization_only'):verify_snapshot(c)
        path=b.private_output(path);contract={'schema':SCHEMA,'config':e.validate(c),'code_sha256':hashes(),'dataset_release_sha256':d.digest(RELEASE.read_text()),'test_used_for_selection':False,'initialization':'reviewed_inherited_weights'}
        if path.exists():
            if not resume or not (path/'job_contract.json').exists() or d.read(path/'job_contract.json')!=contract:raise ValueError('V57 job contract changed; existing output preserved')
        else:path.mkdir(parents=True);d.write(path/'job_contract.json',contract)
        return path
    e.open_job=open_job;p.engine=e;return p


_prompts=_bound(kernels.prepared_prompts)
def prepared_prompts(job,stage):
    names={'final':'final_review_prompts.jsonl.gz','rag':'actual_rag_prompts.jsonl.gz','legacy':'legacy_v55_prompts.jsonl.gz','legacy_rag':'legacy_actual_rag_prompts.jsonl.gz'}
    if stage not in names:return _prompts(job,stage)
    from src.wengua.qwen_book_loop_v3 import read_rows,file_sha
    path=Path(job)/'prepared'/names[stage];meta=d.read(path.parent/'receipt.json')
    if file_sha(path)!=meta['file_sha256'][path.name]:raise ValueError('Prepared prompt bytes changed')
    return read_rows(path)
