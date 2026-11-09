"""Frozen CUDA paths; mixed pilot, fixed micro and final review; exact resume."""
from pathlib import Path
from types import FunctionType
import gzip,json,gc,time
from wengua_course import data as d
from wengua_course_v53 import runtime as kernels
from wengua_course_v60 import runtime as previous
from . import dataset as data
from wengua_course_v60 import checks
ROOT=previous.ROOT;DATA=ROOT/'local/course_v61_data';RELEASE=ROOT/'data/knowledge/course_v61/release.json';CONFIG=ROOT/'configs/course_v61.json';SCHEMA=SWEEP_SCHEMA=data.SCHEMA
FINAL_TOKENS=768
configure=previous.configure;app_root=previous.app_root;verify_snapshot=previous.verify_snapshot;ALLOCATOR=previous.ALLOCATOR


def hashes():
    result=previous.hashes()
    for path in sorted((ROOT/'wengua_course_v61').glob('*.py')):result[path.relative_to(ROOT).as_posix()]=d.digest(path.read_text())
    for name in ['configs/course_v61.json','run_course_v61_worker.py','train_wengua_v61.cmd']:result[name]=d.digest((ROOT/name).read_text())
    for name in ['wengua_recovery/v59_watchdog.py','wengua_recovery/v59_worker.py']:
        result[name]=d.digest((ROOT/name).read_text())
    result['v61_release']=d.digest(RELEASE.read_text());return result


def verify_data():
    from .dataset import validate_pack
    release=d.read(RELEASE)
    if release['parent_v60_release_sha256']!=d.digest(previous.RELEASE.read_bytes()):raise ValueError('Frozen V60 release changed')
    for name,sha in release['files'].items():
        path=(DATA/name).resolve()
        if not path.is_relative_to(DATA.resolve()) or not path.is_file() or d.digest(path.read_bytes())!=sha:raise ValueError('V61 delivery missing/changed: '+name)
    pack=json.loads(gzip.decompress((DATA/'pack.json.gz').read_bytes()))
    validate_pack(pack)
    return pack

def validate(c,testing=False):
    if c['schema']!=SCHEMA:raise ValueError('V61 schema required')
    kernels.validate(c|{'schema':kernels.SCHEMA,'learning_rate':1e-5},testing)
    if c['learning_rate'] not in [3e-6,1e-5] or c['num_train_epochs'] not in [1,2]:raise ValueError('Two finite continuation rates and at most two epochs required')
    if c.get('skill_replay')!={} or c.get('generate_official_baseline') or c.get('run_seen_diagnostics'):raise ValueError('Unplanned replay/control generation')
    if c.get('teacher_replay')!={'natural_dialogue':3,'identity':2}:raise ValueError('Explicit finite teacher replay policy required')
    return c


def choose_development(pack,count=40,stage='application'):
    return [r for r in pack['records'] if r['split']=='validation'][:count]

def _bound(f):
    g=FunctionType(f.__code__,globals(),f.__name__,f.__defaults__,f.__closure__);g.__kwdefaults__=f.__kwdefaults__;return g


_prepare=_bound(kernels.prepare)
def prepare(job,c,*,tokenizer=None,pack=None,testing=False):
    pack=pack if testing and pack is not None else verify_data()
    target=Path(job)/'prepared'
    if target.exists() and not (target/'receipt.json').exists():target.rename(target.with_name('prepared.incomplete-'+str(time.time_ns())))
    pending=target.with_name('prepared.pending')
    if pending.exists():pending.rename(pending.with_name('prepared.pending-preserved-'+str(time.time_ns())))
    meta=_prepare(job,c,tokenizer=tokenizer,pack=pack,testing=True)
    if meta.get('v61_preparation_complete'):return meta
    target.rename(pending);target=pending
    from src.wengua.qwen_book_loop_v3 import write_rows,file_sha,generation_rows,read_rows
    from transformers import AutoTokenizer
    tok=tokenizer or AutoTokenizer.from_pretrained(str(target/'tokenizer'),local_files_only=True,trust_remote_code=False)
    policy=c['teacher_replay']
    if meta.get('dialogue_replay_policy')!=policy:
        if (Path(job)/'cycle_01').exists() or (Path(job)/'smoke').exists():raise ValueError('Cannot reweight prepared data after training began')
        from src.wengua.qwen8b_book_v1 import stress_rows
        cores={r['id']:r for r in pack['records']};weighted=[]
        for row in read_rows(target/'cycle_01.jsonl.gz'):
            for repeat in range(policy.get(cores[row['core_id']]['skill'],1)):
                weighted.append(row|{'id':row['id']+':teacher_replay'+str(repeat)})
        weighted.sort(key=lambda r:d.digest('v61-order:'+r['id']))
        for name,items in [('cycle_01.jsonl.gz',weighted),('smoke.jsonl.gz',stress_rows(weighted,c['smoke_steps'],c['gradient_accumulation_steps']))]:
            write_rows(target/name,items);meta['file_sha256'][name]=file_sha(target/name)
        meta.update(training_rows=len(weighted),dialogue_replay_policy=policy,replays_are_not_new_knowledge=True)
    rows=pack.get('actual_rag_rows',[])
    _,skips=generation_rows(rows,tok,c)
    if skips:raise ValueError('Whole RAG source/teacher exceeds its budget')
    path=target/'actual_rag_prompts.jsonl.gz'
    if path.exists():
        if path.name in meta['file_sha256'] and meta['file_sha256'][path.name]!=file_sha(path):raise ValueError('RAG prompt bytes changed')
        if read_rows(path)!=rows:raise ValueError('RAG prompts differ from frozen source')
    else:write_rows(path,rows)
    meta['file_sha256'][path.name]=file_sha(path)
    meta.update(unique_training_cores=sum(r['split']=='train' for r in pack['records']),natural_dialogue_teacher=True,full_original_curriculum_claimed=False)
    meta['v61_preparation_complete']=True
    d.write(target/'receipt.json',meta)
    target.rename(Path(job)/'prepared')
    return meta

def profile(*,testing=False):
    from src.wengua import qwen8b_book_v1 as old,qwen_book_continuous_v1 as engine,qwen_course_training_v4 as integrity
    from wengua_course_v60 import fit as continuation
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
            number+=1;kw.setdefault('logits_to_keep',1);b.progress('v61_answer_progress',phase=tag,answer=number,total=len(rows),file=str(path));return method(*a,**kw)
        with patch.object(model,'generate',call):return native(model,tok,rows,path,c,deadline,tag,*args,**kwargs)
    b.generate_rows=generate_rows
    e=old.bind(engine,'wengua_course_v61._engine',SCHEMA=SCHEMA,book=b);e.validate_structure=lambda c:validate(c,testing);e.code_hashes=hashes;e.base_config=lambda:d.read(CONFIG)
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
            if not resume or not (path/'job_contract.json').exists() or d.read(path/'job_contract.json')!=contract:raise ValueError('V61 job contract changed; existing output preserved')
        else:path.mkdir(parents=True);d.write(path/'job_contract.json',contract)
        return path
    e.open_job=open_job;p.engine=e;return p


_prompts=_bound(kernels.prepared_prompts)
def prepared_prompts(job,stage):
    if stage!='rag':return _prompts(job,stage)
    from src.wengua.qwen_book_loop_v3 import read_rows,file_sha
    path=Path(job)/'prepared/actual_rag_prompts.jsonl.gz';meta=d.read(path.parent/'receipt.json')
    if file_sha(path)!=meta['file_sha256'][path.name]:raise ValueError('Prepared RAG prompts changed')
    return read_rows(path)
