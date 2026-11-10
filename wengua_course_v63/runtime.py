"""Frozen CUDA paths; mixed pilot, fixed micro and final review; exact resume."""
from pathlib import Path
from types import FunctionType
import gzip,json,gc,time
from wengua_course import data as d
from wengua_course_v53 import runtime as kernels
from wengua_course_v60 import runtime as previous
from wengua_course_v62 import runtime as frozen
from . import dataset as data
from wengua_course_v60 import checks
ROOT=previous.ROOT;DATA=ROOT/'local/course_v63_data';RELEASE=ROOT/'data/knowledge/course_v63/release.json';CONFIG=ROOT/'configs/course_v63.json';SCHEMA=SWEEP_SCHEMA=data.SCHEMA
FINAL_TOKENS=768
configure=previous.configure;app_root=previous.app_root;ALLOCATOR=previous.ALLOCATOR

def verify_snapshot(c,*,testing=False):
    path=previous.verify_snapshot(c)
    proof=d.read(path.parent/'parent_provenance.json')
    if proof['schema']!=SCHEMA or proof['parent_config'].get('initialization_only') is not True:
        raise ValueError('V63 accepts only its official zero-update initializer')
    from .initialization import verify_zero
    zero=verify_zero(proof['source_path'],testing=testing)
    if zero['files']!=proof['files']:raise ValueError('Fresh source differs from initialization snapshot')
    return path


def hashes():
    result=frozen.hashes()
    for path in sorted((ROOT/'wengua_course_v63').glob('*.py')):result[path.relative_to(ROOT).as_posix()]=d.digest(path.read_text())
    for name in ['configs/course_v63.json','run_course_v63_worker.py','train_wengua_v63.cmd']:result[name]=d.digest((ROOT/name).read_text())
    for name in ['wengua_recovery/v59_watchdog.py','wengua_recovery/v59_worker.py']:
        result[name]=d.digest((ROOT/name).read_text())
    result['v63_release']=d.digest(RELEASE.read_text());return result


def verify_data():
    release=d.read(RELEASE)
    if release['source_release_sha256']!=d.digest(frozen.RELEASE.read_bytes()):raise ValueError('V62 source release changed')
    source=frozen.verify_data();pack=data.derive(source)
    # Python 3.12's gzip.compress(mtime=0) writes a platform-dependent OS byte.
    # Freeze the complete canonical JSON, not a platform's compression stream.
    canonical=d.dumps(pack).encode('utf8')
    if d.digest(canonical)!=release['pack_json_sha256']:raise ValueError('Derived V63 source pack changed')
    payload=gzip.compress(canonical,mtime=0)
    DATA.mkdir(parents=True,exist_ok=True);path=DATA/'pack.json.gz'
    if path.exists() and gzip.decompress(path.read_bytes())!=canonical:raise ValueError('Existing V63 source pack differs; preserved')
    if not path.exists():
        pending=path.with_name(path.name+'.pending-'+str(time.time_ns()))
        pending.write_bytes(payload)
        pending.rename(path)
    data.verify_derivation(source,pack)
    return pack

def validate(c,testing=False):
    from src.wengua import qwen8b_book_v1 as old
    if c['schema']!=SCHEMA:raise ValueError('V63 schema required')
    old.validate(c|{'schema':old.BOOK_SCHEMA,'smoke_steps':100},testing=testing)
    if c['learning_rate'] not in (5e-5,1e-4):raise ValueError('Two declared V63 learning rates required')
    if not testing and (c['lora_r'],c['lora_alpha'],c['lora_dropout'],set(c['target_modules']))!=(16,32,.05,set(data.MODULES)):
        raise ValueError('V63 all-seven-projection structure required')
    if type(c['num_train_epochs']) is not int or not 1<=c['num_train_epochs']<=8:raise ValueError('At most eight complete epochs')
    if not testing and c['gradient_accumulation_steps']!=8:raise ValueError('Effective batch eight required')
    if c['generation_epochs']!=[1,2,4,8]:raise ValueError('Declared free-generation boundaries required')
    if c.get('teacher_replay')!={} or c.get('skill_replay')!={}:raise ValueError('No implicit teacher replay')
    if not c.get('initialization_only') and not c.get('book_parent_snapshot'):raise ValueError('Sealed fresh step-zero required')
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
    if meta.get('v63_preparation_complete'):return meta
    target.rename(pending);target=pending
    from src.wengua.qwen_book_loop_v3 import write_rows,file_sha,generation_rows,read_rows
    from transformers import AutoTokenizer
    tok=tokenizer or AutoTokenizer.from_pretrained(str(target/'tokenizer'),local_files_only=True,trust_remote_code=False)
    # Generic and previously exposed raw-board probes are generation-only.
    learning=data.learning_rows(pack,testing=testing)
    _,skips=generation_rows(learning,tok,c)
    if skips:raise ValueError('Whole original learning question exceeds generation context')
    path=target/'learning_prompts.jsonl.gz'
    write_rows(path,learning);meta['file_sha256'][path.name]=file_sha(path)
    meta['learning_prompt_ids']=[r['id'] for r in learning]
    diagnostic=pack.get('known_regression_rows',[])
    _,skips=generation_rows(diagnostic,tok,c)
    if skips:raise ValueError('Whole diagnostic exceeds context')
    path=target/'seen_diagnostic_prompts.jsonl.gz'
    write_rows(path,diagnostic);meta['file_sha256'][path.name]=file_sha(path)
    rows=pack.get('actual_rag_rows',[])
    _,skips=generation_rows(rows,tok,c)
    if skips:raise ValueError('Whole RAG source/teacher exceeds its budget')
    path=target/'actual_rag_prompts.jsonl.gz'
    if path.exists():
        if path.name in meta['file_sha256'] and meta['file_sha256'][path.name]!=file_sha(path):raise ValueError('RAG prompt bytes changed')
        if read_rows(path)!=rows:raise ValueError('RAG prompts differ from frozen source')
    else:write_rows(path,rows)
    meta['file_sha256'][path.name]=file_sha(path)
    meta.update(unique_training_cores=sum(r['split']=='train' for r in pack['records']),typed_source_teacher=True,full_original_curriculum_claimed=False)
    meta['v63_preparation_complete']=True
    d.write(target/'receipt.json',meta)
    target.rename(Path(job)/'prepared')
    return meta

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
            number+=1;kw.setdefault('logits_to_keep',1);b.progress('v63_answer_progress',phase=tag,answer=number,total=len(rows),file=str(path));return method(*a,**kw)
        with patch.object(model,'generate',call):return native(model,tok,rows,path,c,deadline,tag,*args,**kwargs)
    b.generate_rows=generate_rows
    e=old.bind(engine,'wengua_course_v63._engine',SCHEMA=SCHEMA,book=b);e.validate_structure=lambda c:validate(c,testing);e.code_hashes=hashes;e.base_config=lambda:d.read(CONFIG)
    raw=FunctionType(continuation.fit.__code__,e.__dict__,'fit',continuation.fit.__defaults__);raw.__kwdefaults__=continuation.fit.__kwdefaults__
    def fit(job,c,deadline,*,extra_callbacks=(),**kw):
        verify_snapshot(c,testing=testing)
        if kw.get('cpu_factory'):
            factory=kw['cpu_factory']
            def normalized_fixture():
                from src.wengua.qwen_finetune import prepare_frozen_base
                tok,base,compute=factory();return tok,prepare_frozen_base(base,gradient_checkpointing=False),compute
            kw['cpu_factory']=normalized_fixture
        with integrity.training_integrity() as check:return raw(job,c,deadline,extra_callbacks=(*extra_callbacks,check),**kw)
    e.fit=fit
    def open_job(path,c,*,resume):
        if not c.get('initialization_only'):verify_snapshot(c,testing=testing)
        path=b.private_output(path);contract={'schema':SCHEMA,'config':e.validate(c),'code_sha256':hashes(),'dataset_release_sha256':d.digest(RELEASE.read_text()),'test_used_for_selection':False,'initialization':'official_fresh_zero_update_lora'}
        if path.exists():
            if not resume or not (path/'job_contract.json').exists() or d.read(path/'job_contract.json')!=contract:raise ValueError('V63 job contract changed; existing output preserved')
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
