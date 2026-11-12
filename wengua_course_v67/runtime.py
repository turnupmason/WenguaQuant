"""Versioned source data; keep the proven single-GPU loss and recovery path."""
from pathlib import Path
from types import FunctionType
import gzip,json,time,gc
from wengua_course import data as d
from wengua_course_v66 import runtime as before
from . import dataset as data
ROOT=before.ROOT;DATA=ROOT/'local/course_v67_data';CONFIG=ROOT/'configs/course_v67.json';RELEASE=ROOT/'data/knowledge/course_v67/release.json'
SCHEMA=SWEEP_SCHEMA=data.SCHEMA;ALLOCATOR=before.ALLOCATOR
configure=before.configure;app_root=before.app_root;resume_generation_file=before.resume_generation_file


def hashes():
    result=before.hashes()
    for path in sorted((ROOT/'wengua_course_v67').glob('*.py')):result[path.relative_to(ROOT).as_posix()]=d.digest(path.read_text(encoding='utf8'))
    for n in ['configs/course_v67.json','run_course_v67_worker.py','train_wengua_v67.cmd','export_wengua_v67.cmd','data/reviews/v67_parent_sources.json']:
        result[n]=d.digest((ROOT/n).read_text(encoding='utf8'))
    result['v67_release']=d.digest(RELEASE.read_text(encoding='utf8'));return result


def verify_data():
    release=d.read(RELEASE)
    if d.digest(before.RELEASE.read_bytes())!=release['source_v66_release_sha256']:raise ValueError('Original V66 release changed')
    source=before.verify_data();pack=data.build(source);raw=d.dumps(pack).encode('utf8')
    if d.digest(raw)!=release['pack_json_sha256']:raise ValueError('V67 source-derived course changed')
    DATA.mkdir(parents=True,exist_ok=True);path=DATA/'pack.json.gz'
    if path.exists() and gzip.decompress(path.read_bytes())!=raw:raise ValueError('Existing private course preserved')
    if not path.exists():path.write_bytes(gzip.compress(raw,mtime=0))
    return pack


def verify_snapshot(c,*,testing=False):
    from wengua_course.runtime import verify_snapshot as shared
    path=shared(c);proof=d.read(path.parent/'parent_provenance.json')
    if proof['schema']!=before.SCHEMA or c['source_variant']!='v66_ordinary':raise ValueError('Formal learned V66 ordinary parent required')
    if not testing and proof['files']['adapter_model.safetensors']!=d.read(RELEASE)['parent']['sha256']:raise ValueError('Different V66 source weight')
    return path


def validate(c,testing=False):
    from src.wengua import qwen8b_book_v1 as base
    base.validate(c|{'schema':base.BOOK_SCHEMA,'smoke_steps':100},testing=testing)
    if c['schema']!=SCHEMA or type(c['decision_loss_factor']) is not int or c['decision_loss_factor']!=1:raise ValueError('Ordinary CE only')
    if not testing and (c['learning_rate'],c['num_train_epochs'],c['gradient_accumulation_steps'])!=(1e-5,2,8):raise ValueError('One fixed inherited two-epoch run')
    if not testing and (c['lora_r'],c['lora_alpha'],c['lora_dropout'],set(c['target_modules']))!=(16,32,.05,set(data.old.MODULES)):raise ValueError('Retain learned seven-projection structure')
    if c['generation_epochs']!=[1,2] or c.get('teacher_replay')!={} or c.get('skill_replay')!={}:raise ValueError('Every epoch retained, no hidden repeats')
    if not c.get('book_parent_snapshot'):raise ValueError('Sealed learned parent required')
    return c


def prepare(job,c,*,tokenizer=None,pack=None,testing=False):
    from src.wengua import qwen_book_loop_v3 as b,qwen8b_book_v1 as base
    from wengua_course_v65.protocol import verify
    job=Path(job);target=job/'prepared'
    if (target/'receipt.json').exists():
        meta=d.read(target/'receipt.json')
        if meta['config']!=c or meta['dataset_release_sha256']!=d.digest(RELEASE.read_text(encoding='utf8')):raise ValueError('Prepared V67 contract changed')
        if any(b.file_sha(target/n)!=h for n,h in meta['file_sha256'].items()):raise ValueError('Prepared data changed')
        return meta
    if target.exists():raise ValueError('Incomplete preparation preserved')
    pack=pack if testing and pack is not None else verify_data();tok=tokenizer or b.load_book_tokenizer(c);tok.pad_token=tok.pad_token or tok.eos_token
    pending=target.with_name('prepared.pending')
    if pending.exists():pending.rename(pending.with_name('prepared.incomplete-'+str(time.time_ns())))
    pending.mkdir(parents=True);tok.save_pretrained(pending/'tokenizer');train=[];val=[];seen={};aliases=[]
    for core in pack['records']:
        if core['split'] not in {'train','validation'}:continue
        for view in b.target_views(core,0):
            key=d.dumps(view['messages'][:-1])
            if key in seen:
                existing=seen[key]
                if existing['split']!=view['split'] or existing['messages'][-1]!=view['messages'][-1]:raise ValueError('Conflicting/cross-split supervised view')
                aliases.append({'id':view['id'],'same_supervised_view_as':existing['id']});continue
            encoded,size=b.encode(view,tok,c['max_seq_length'])
            if encoded is None:raise ValueError('Whole source or answer exceeds context: '+core['id'])
            encoded['decision_mask']=[0]*size
            item=view|{'encoded':encoded,'tokens':size};seen[key]=item
            (train if core['split']=='train' else val).append(item)
    if not train or not val:raise ValueError('Missing train/development rows')
    train.sort(key=lambda r:d.digest('v67-train:'+r['id']))
    for name,rows in [('cycle_01.jsonl.gz',train),('smoke.jsonl.gz',base.stress_rows(train,c['smoke_steps'],c['gradient_accumulation_steps'])),
                      ('validation_loss.jsonl.gz',val),('validation_prompts.jsonl.gz',[r for r in pack['records'] if r['split']=='validation']),
                      ('learning_prompts.jsonl.gz',[]),('seen_diagnostic_prompts.jsonl.gz',[]),('source_units.jsonl.gz',pack['units'].values())]:b.write_rows(pending/name,rows)
    protocol=verify(train,tok);d.write(pending/'training_generation_protocol.json',protocol);d.write(pending/'supervised_view_aliases.json',aliases)
    meta={'schema':SCHEMA,'config':c,'dataset_release_sha256':d.digest(RELEASE.read_text(encoding='utf8')),'training_rows':len(train),
          'max_actual_train_tokens':max(r['tokens'] for r in train),'training_generation_protocol':protocol,'deduplicated_supervised_views':len(aliases),
          'decision_tokens':0,'prompt_weighted_tokens':0,'ordinary_training_and_validation':True,'whole_targets_preserved':True,'original_test_used':False,
          'training_tensor_sha256':d.digest(d.dumps([r['encoded'] for r in train])),
          'file_sha256':{p.relative_to(pending).as_posix():b.file_sha(p) for p in pending.rglob('*') if p.is_file()}}
    d.write(pending/'receipt.json',meta);pending.rename(target);return meta


def profile(*,testing=False):
    from src.wengua import qwen8b_book_v1 as base,qwen_book_continuous_v1 as engine,qwen_course_training_v4 as integrity
    from . import fit as continuation
    p=base.profile(testing=testing);b=p.book;b.ROOT=ROOT;b.SCHEMA=SCHEMA;b.normalized_kernel_hashes=hashes;b.prepare=lambda job,c,**kw:prepare(job,c,testing=testing,**kw)
    loader=b.load_book_cuda
    def normalized(c,training):
        from src.wengua.qwen_finetune import prepare_frozen_base
        model,compute=loader(c,training);return prepare_frozen_base(model,gradient_checkpointing=False),compute
    b.load_book_cuda=normalized;native=b.generate_rows
    def generate(model,tok,rows,path,c,deadline,tag,*a,**kw):
        from unittest.mock import patch
        resume_generation_file(path,rows);method=model.generate
        def call(*args,**kwargs):kwargs.setdefault('logits_to_keep',1);return method(*args,**kwargs)
        with patch.object(model,'generate',call):return native(model,tok,rows,path,c,deadline,tag,*a,**kw)
    b.generate_rows=generate
    e=base.bind(engine,'wengua_course_v67._engine',SCHEMA=SCHEMA,book=b);e.validate_structure=lambda c:validate(c,testing);e.code_hashes=hashes;e.base_config=lambda:d.read(CONFIG)
    raw=FunctionType(continuation.fit.__code__,e.__dict__,'fit',continuation.fit.__defaults__);raw.__kwdefaults__=continuation.fit.__kwdefaults__
    def fit(job,c,deadline,*,extra_callbacks=(),**kw):
        verify_snapshot(c,testing=testing)
        from wengua_course_v65.numerics import training_callback
        with integrity.training_integrity() as check:return raw(job,c,deadline,extra_callbacks=(*extra_callbacks,check,training_callback(Path(job),c,b,testing)),**kw)
    e.fit=fit
    def open_job(path,c,*,resume):
        verify_snapshot(c,testing=testing);path=b.private_output(path)
        contract={'schema':SCHEMA,'config':e.validate(c),'code_sha256':hashes(),'dataset_release_sha256':d.digest(RELEASE.read_text(encoding='utf8')),
                  'original_test_used':False,'initialization':'v66_ordinary_step77_new_optimizer'}
        if path.exists():
            if not resume or not (path/'job_contract.json').exists() or d.read(path/'job_contract.json')!=contract:raise ValueError('V67 job changed; preserve old output')
        else:path.mkdir(parents=True);d.write(path/'job_contract.json',contract)
        return path
    e.open_job=open_job;p.engine=e;return p
