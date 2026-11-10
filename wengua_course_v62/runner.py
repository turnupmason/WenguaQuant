"""Two finite fresh-base trials; no inherited-book fallback or automatic promotion."""
from pathlib import Path
from types import FunctionType
import time,html,json,gc
from wengua_course import data as d
from wengua_course_v60 import runner as previous
from . import runtime,initialization
RATES=(1e-5,2e-5)
read_rows=previous.read_rows
evaluate_adapter=FunctionType(previous.evaluate_adapter.__code__,globals(),'evaluate_adapter',previous.evaluate_adapter.__defaults__)
evaluate_adapter.__kwdefaults__=previous.evaluate_adapter.__kwdefaults__

def config(parent,root,rate):
    c=d.read(runtime.CONFIG)
    c.update(learning_rate=rate,book_parent_snapshot=str(Path(parent).resolve()),book_parent_provenance_sha256=d.digest((Path(parent).parent/'parent_provenance.json').read_bytes()),legacy_source_run=str(Path(root).resolve()))
    return c

def report(root,state):
    d.write(root/'run_status.json',state)
    parts=['<!doctype html><meta charset="utf-8"><title>Wengua V6.2</title><style>body{font:16px system-ui;max-width:1100px;margin:24px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>V6.2 官方新LoRA与条件应用</h1><p>两率同一零更新初始化，各两轮；20来源开发题参与loss，24既有诊断只生成，不进入训练或loss。保留每轮回答和完整检查点；不自动替换旧模型。</p><pre>'+html.escape(d.dumps(state))+'</pre>']
    for path in sorted(root.glob('trials/*/generations/*.jsonl')):
        parts.append('<details><summary>'+html.escape(str(path.relative_to(root)))+'</summary>')
        for r in read_rows(path):parts.append('<details><summary>'+html.escape(r['id'])+'</summary><pre>'+html.escape(d.dumps(r['question']))+'</pre><p>'+html.escape(r['answer'])+'</p><pre>'+html.escape(r['expected'])+'</pre></details>')
        parts.append('</details>')
    (root/'results.html').write_text(''.join(parts),encoding='utf8')

def run(args):
    runtime.configure(args.app_root);pack=runtime.verify_data();p=runtime.profile();b=p.book
    root=b.private_output(args.output);root.mkdir(parents=True,exist_ok=True)
    from .process import launch
    deadline=time.time()+3600*args.hours
    contract={'schema':runtime.SCHEMA,'code_sha256':runtime.hashes(),'dataset_release_sha256':d.digest(runtime.RELEASE.read_bytes()),'rates':list(RATES),'epochs':2,'same_zero_update_initialization':True,'old_book_weights_loaded':False,'no_automatic_professional_promotion':True,'original_test_used':False}
    path=root/'sweep_contract.json'
    if path.exists() and d.read(path)!=contract:raise ValueError('V62 run contract changed; previous output preserved')
    if path.exists() and not args.resume:raise ValueError('Use --resume for an existing run')
    if not path.exists():d.write(path,contract)
    state=d.read(root/'run_status.json') if (root/'run_status.json').exists() else {'state':'preparing','trials':{},'curriculum':pack['summary'],'production_enabled':False}
    if state['state']=='complete_review_required':report(root,state);return 0
    with b.keep_awake(),b.job_lock(root/'run.lock'):
        try:
            seed=d.read(runtime.CONFIG)|{'initialization_only':True}
            init=p.engine.open_job(root/'trials/official_step_zero',seed,resume=True)
            created=launch(p,init,seed,deadline,'initialize')
            parent=initialization.snapshot(root/'initialization',created['adapter'],testing=p.testing)
            state['initialization']=created;report(root,state)
            for rate in RATES:
                name='lr_1em05' if rate==1e-5 else 'lr_2em05';c=config(parent,root,rate)
                job=p.engine.open_job(root/'trials'/name,c,resume=True)
                t=state['trials'].setdefault(name,{'config':c,'state':'preparing'})
                if t['config']!=c:raise ValueError('Trial configuration changed')
                state.update(state='running',active_trial=name);report(root,state)
                runtime.prepare(job,c)
                from src.wengua.qwen8b_book_v1 import doctor as assess
                doctor=assess(c) if not p.testing else {'training_preflight':{'status':'ready_for_smoke_test'},'cuda_devices':[]}
                d.write(job/'doctor.json',doctor)
                if doctor['training_preflight']['status']!='ready_for_smoke_test':raise RuntimeError(doctor['training_preflight']['reason'])
                if t.get('probe',{}).get('state')!='complete':
                    t['probe']=launch(p,job,c,deadline,'fit',probe=True);report(root,state)
                    if t['probe']['state']!='complete':state['state']='paused';report(root,state);return 0
                cp=args.resume_from_checkpoint
                if cp and not Path(cp).resolve().is_relative_to((root/'trials').resolve()):raise ValueError('Explicit checkpoint must belong to this V62 run')
                cp=cp if cp and Path(cp).resolve().is_relative_to(job.resolve()) else None
                t['fit']=launch(p,job,c,deadline,'fit',checkpoint=cp);report(root,state)
                if t['fit']['state']!='complete':state['state']='paused';report(root,state);return 0
                if pack.get('actual_rag_rows'):
                    t['rag']=launch(p,job,c,deadline,'evaluate',best={'adapter':t['fit']['adapter'],'tag':'language_nominee_actual_rag','eval_stage':'rag'})
                    if t['rag']['state']!='complete':state['state']='paused';report(root,state);return 0
                t['state']='complete';report(root,state)
            state.update(state='complete_review_required',active_trial=None,selection='none_until_semantic_review',original_weights_and_checkpoints_preserved=True)
            report(root,state);return 0
        except BaseException as error:
            state.update(state='paused' if isinstance(error,(KeyboardInterrupt,TimeoutError)) else 'failed',error={'type':type(error).__name__,'message':str(error)})
            report(root,state);raise
