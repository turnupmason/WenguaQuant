"""New data and source contract; reuse the proven bounded-loss CUDA machinery."""
from pathlib import Path
from types import FunctionType
import json,gzip,time,gc
from wengua_course import data as d
from wengua_course_v53 import runtime as kernels
from wengua_course_v60 import runtime as common
from wengua_course_v63 import runtime as frozen
from . import dataset as data
ROOT=common.ROOT;DATA=ROOT/'local/course_v64_data';RELEASE=ROOT/'data/knowledge/course_v64/release.json';CONFIG=ROOT/'configs/course_v64.json'
SCHEMA=SWEEP_SCHEMA=data.SCHEMA;ALLOCATOR=common.ALLOCATOR;FINAL_TOKENS=512
configure=common.configure;app_root=common.app_root


def hashes():
    result=frozen.hashes()
    for path in sorted((ROOT/'wengua_course_v64').glob('*.py')):result[path.relative_to(ROOT).as_posix()]=d.digest(path.read_text())
    for name in ['configs/course_v64.json','train_wengua_v64.cmd','run_course_v64_worker.py']:
        result[name]=d.digest((ROOT/name).read_text())
    result['v64_release']=d.digest(RELEASE.read_text());return result


def verify_data():
    release=d.read(RELEASE)
    if release['source_v63_release_sha256']!=d.digest(frozen.RELEASE.read_bytes()):raise ValueError('Original V63 release changed')
    source=frozen.verify_data();physical=data.placements(app_root())
    if d.digest(d.dumps(physical))!=release['literal_placement_sha256']:raise ValueError('Literal source placements changed')
    pack=data.build(source,physical);canonical=d.dumps(pack).encode('utf8')
    if d.digest(canonical)!=release['pack_json_sha256']:raise ValueError('Derived V64 course changed')
    DATA.mkdir(parents=True,exist_ok=True);path=DATA/'pack.json.gz'
    if path.exists() and gzip.decompress(path.read_bytes())!=canonical:raise ValueError('Old V64 data preserved; changed course refused')
    if not path.exists():
        pending=path.with_name(path.name+'.pending-'+str(time.time_ns()));pending.write_bytes(gzip.compress(canonical,mtime=0));pending.rename(path)
    return json.loads(canonical)


def verify_snapshot(c,*,testing=False):
    path=common.verify_snapshot(c);proof=d.read(path.parent/'parent_provenance.json')
    if proof['schema']!=frozen.SCHEMA or proof['parent_config'].get('initialization_only'):
        raise ValueError('A trained nonzero V63 parent is required')
    if not testing and proof['files']['adapter_model.safetensors']!=d.read(RELEASE)['reviewed_parent']['adapter_sha256']:
        raise ValueError('Only selected V63 high-rate epoch7 can seed V64')
    return path


def validate(c,testing=False):
    from src.wengua import qwen8b_book_v1 as old
    if c['schema']!=SCHEMA:raise ValueError('V64 schema required')
    old.validate(c|{'schema':old.BOOK_SCHEMA,'smoke_steps':100},testing=testing)
    if c['learning_rate'] not in (1e-5,3e-5) or c['num_train_epochs']!=2:raise ValueError('Two declared rates and exactly two epochs required')
    if (c['lora_r'],c['lora_alpha'],c['lora_dropout'],set(c['target_modules']))!=(16,32,.05,set(data.MODULES)):
        raise ValueError('Preserve the inherited seven-projection structure')
    if c['generation_epochs']!=[1,2] or c.get('teacher_replay')!={} or c.get('skill_replay')!={}:raise ValueError('No implicit replay or skipped epoch review')
    if not c.get('book_parent_snapshot'):raise ValueError('Inherited sealed parent required')
    return c


def choose_development(pack,count=64,stage='application'):
    return [r for r in pack['records'] if r['split']=='validation'][:count]


def review_rows(pack):
    # 74 exact prior questions and 39 new task-structure development questions.
    rows=pack.get('original_recall_rows',[])+pack.get('known_regression_rows',[])
    rows += [r for r in pack.get('known_development_rows',[]) if r['id'] not in {q['id'] for q in rows}]
    rows += choose_development(pack)
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Repeated review ids')
    return rows


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
    if meta.get('v64_preparation_complete'):return meta
    target.rename(pending);target=pending
    from src.wengua.qwen_book_loop_v3 import write_rows,file_sha,generation_rows
    from transformers import AutoTokenizer
    tok=tokenizer or AutoTokenizer.from_pretrained(str(target/'tokenizer'),local_files_only=True,trust_remote_code=False)
    rows=review_rows(pack);_,skips=generation_rows(rows,tok,c)
    if skips:raise ValueError('Whole review source or target exceeds context')
    for name,records in [('learning_prompts.jsonl.gz',rows),('seen_diagnostic_prompts.jsonl.gz',[])]:
        write_rows(target/name,records);meta['file_sha256'][name]=file_sha(target/name)
    meta.update(v64_preparation_complete=True,review_prompt_ids=[r['id'] for r in rows],
        unique_training_cores=sum(r['split']=='train' for r in pack['records']),full_original_curriculum_claimed=False)
    d.write(target/'receipt.json',meta);target.rename(Path(job)/'prepared');return meta


def resume_generation_file(path,rows):
    """Preserve complete answers and repair only an incomplete last UTF8/JSON line."""
    path=Path(path)
    if not path.exists():return
    from research_audits.v63_checkpoint_probe import verify_cached
    raw=path.read_bytes();valid=[];lines=raw.splitlines(keepends=True);damaged=False;kept=[]
    for i,line in enumerate(lines):
        if not line.strip():kept.append(line);continue
        try:valid.append(json.loads(line))
        except ValueError:
            if i!=len(lines)-1:raise ValueError('Corrupt answer before final line')
            damaged=True
        else:kept.append(line)
    verify_cached(valid,rows)
    prefix=b''.join(kept)
    # Keep the original bytes, including CRLF and JSON whitespace. A complete
    # last object without its delimiter also needs a newline before appending.
    if prefix and not prefix.endswith(b'\n'):prefix+=b'\n'
    if damaged or prefix!=raw:
        path.rename(path.with_name(path.name+'.partial-'+str(time.time_ns())))
        path.write_bytes(prefix)


def profile(*,testing=False):
    from src.wengua import qwen8b_book_v1 as old,qwen_book_continuous_v1 as engine,qwen_course_training_v4 as integrity
    from . import fit as continuation
    p=old.profile(testing=testing);b=p.book;b.ROOT=ROOT;b.SCHEMA=SCHEMA;b.normalized_kernel_hashes=hashes;b.prepare=lambda job,c,**kw:prepare(job,c,testing=testing,**kw)
    loader=b.load_book_cuda
    def normalized_loader(c,training):
        from src.wengua.qwen_finetune import prepare_frozen_base
        model,compute=loader(c,training);return prepare_frozen_base(model,gradient_checkpointing=False),compute
    b.load_book_cuda=normalized_loader;native=b.generate_rows
    def generate_rows(model,tok,rows,path,c,deadline,tag,*args,**kw):
        from unittest.mock import patch
        resume_generation_file(path,rows);method=model.generate
        count=len([s for s in Path(path).read_text(encoding='utf8').splitlines() if s.strip()]) if Path(path).exists() else 0
        def call(*a,**k):
            nonlocal count
            count+=1;k.setdefault('logits_to_keep',1);b.progress('v64_answer_progress',phase=tag,answer=count,total=len(rows),file=str(path));return method(*a,**k)
        with patch.object(model,'generate',call):return native(model,tok,rows,path,c,deadline,tag,*args,**kw)
    b.generate_rows=generate_rows
    e=old.bind(engine,'wengua_course_v64._engine',SCHEMA=SCHEMA,book=b);e.validate_structure=lambda c:validate(c,testing);e.code_hashes=hashes;e.base_config=lambda:d.read(CONFIG)
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
        verify_snapshot(c,testing=testing);path=b.private_output(path)
        contract={'schema':SCHEMA,'config':e.validate(c),'code_sha256':hashes(),'dataset_release_sha256':d.digest(RELEASE.read_text()),'original_test_used':False,'initialization':'inherited_v63_high_e07_new_optimizer'}
        if path.exists():
            if not resume or not (path/'job_contract.json').exists() or d.read(path/'job_contract.json')!=contract:raise ValueError('V64 job contract changed; old output preserved')
        else:path.mkdir(parents=True);d.write(path/'job_contract.json',contract)
        return path
    e.open_job=open_job;p.engine=e;return p


def prepared_prompts(job,stage):
    from src.wengua.qwen_book_loop_v3 import read_rows,file_sha
    path=Path(job)/'prepared/learning_prompts.jsonl.gz';meta=d.read(path.parent/'receipt.json')
    if stage!='review' or file_sha(path)!=meta['file_sha256'][path.name]:raise ValueError('Whole review contract differs')
    return read_rows(path)
