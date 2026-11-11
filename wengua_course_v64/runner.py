"""Two same-parent continuation trials; all epoch answers remain review-only."""
from pathlib import Path
from types import FunctionType
import time,html,gc
from wengua_course import data as d
from wengua_course_v60 import runner as old
from . import runtime,initialization,baselines
RATES=(1e-5,3e-5)
read_rows=old.read_rows
evaluate_adapter=FunctionType(old.evaluate_adapter.__code__,globals(),'evaluate_adapter',old.evaluate_adapter.__defaults__)
evaluate_adapter.__kwdefaults__=old.evaluate_adapter.__kwdefaults__


def checkpoint_scope(root,path):
    if path is None:return None
    path=Path(path).resolve();root=Path(root).resolve()
    allowed=[root/'trials/lr_1em05/cycle_01/trainer',root/'trials/lr_3em05/cycle_01/trainer']
    if (not path.is_dir() or path.parent not in allowed or not path.name.startswith('checkpoint-')
            or not path.name[len('checkpoint-'):].isdigit()):
        raise ValueError('Explicit checkpoint must be a V64 training checkpoint in one of this run’s two groups')
    return path


def config(parent,root,rate):
    c=d.read(runtime.CONFIG)
    c.update(learning_rate=rate,book_parent_snapshot=str(Path(parent).resolve()),
        book_parent_provenance_sha256=d.digest((Path(parent).parent/'parent_provenance.json').read_bytes()),legacy_source_run=str(Path(root).resolve()))
    return c


def report(root,state):
    d.write(root/'run_status.json',state)
    text=['<!doctype html><meta charset="utf-8"><title>Wengua V6.4</title><style>body{font:16px system-ui;max-width:1150px;margin:24px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:12px}</style><h1>V6.4 条件、对象和纠错短训</h1><p>继承V6.3 step371，两率各两轮，新阶段新优化器；每轮保存新任务和原74题的完整回答。父权重原74答只复用，不重跑；全部候选保留，不自动晋升。</p><pre>'+html.escape(d.dumps(state))+'</pre>']
    for path in sorted(root.glob('trials/*/generations/*.jsonl')):
        text.append('<details><summary>'+html.escape(str(path.relative_to(root)))+'</summary>')
        for row in read_rows(path):text.append('<details><summary>'+html.escape(row['id'])+'</summary><pre>'+html.escape(d.dumps(row['question']))+'</pre><p>'+html.escape(row['answer'])+'</p><pre>'+html.escape(row['expected'])+'</pre></details>')
        text.append('</details>')
    (root/'results.html').write_text(''.join(text),encoding='utf8')


def run(args):
    runtime.configure(args.app_root);pack=runtime.verify_data();p=runtime.profile();b=p.book
    if not p.testing and b.stack_versions()!=d.read(runtime.RELEASE)['original_stack']:
        raise ValueError('Use the pinned V63 Python environment; baseline comparison stack changed')
    from .process import launch
    root=b.private_output(args.output);root.mkdir(parents=True,exist_ok=True);deadline=time.time()+3600*args.hours
    requested_checkpoint=checkpoint_scope(root,args.resume_from_checkpoint)
    source=initialization.default_source();parent=initialization.snapshot(root/'initialization',source,testing=p.testing)
    contract={'schema':runtime.SCHEMA,'code_sha256':runtime.hashes(),'dataset_release_sha256':d.digest(runtime.RELEASE.read_bytes()),
        'rates':list(RATES),'epochs':2,'same_inherited_parent':True,'parent_provenance_sha256':d.digest((parent.parent/'parent_provenance.json').read_bytes()),
        'old_optimizer_inherited':False,'original_test_used':False,'automatic_professional_promotion':False}
    path=root/'sweep_contract.json'
    if path.exists() and (not args.resume or d.read(path)!=contract):raise ValueError('V64 run contract changed; existing run preserved')
    if not path.exists():d.write(path,contract)
    state=d.read(root/'run_status.json') if (root/'run_status.json').exists() else {'state':'preparing','trials':{},'curriculum':pack['summary'],'production_enabled':False}
    if state['state']=='complete_review_required':report(root,state);return 0
    with b.keep_awake(),b.job_lock(root/'run.lock'):
        try:
            c=config(parent,root,RATES[0]);baseline=p.engine.open_job(root/'trials/parent_baseline',c,resume=True);runtime.prepare(baseline,c)
            baselines.seed(baseline,pack,c,d.digest((Path(source)/'adapter_model.safetensors').read_bytes()),testing=p.testing,fixture_rows=pack.get('fixture_parent_rows'))
            state.update(state='parent_review',active_trial='parent_baseline');report(root,state)
            result=launch(p,baseline,c,deadline,'evaluate',best={'adapter':str(source),'tag':'parent_review','eval_stage':'review'})
            state['parent_review']=result;report(root,state)
            if result['state']!='complete':state['state']='paused';report(root,state);return 75
            for rate in RATES:
                name='lr_1em05' if rate==1e-5 else 'lr_3em05';c=config(parent,root,rate);job=p.engine.open_job(root/'trials'/name,c,resume=True)
                t=state['trials'].setdefault(name,{'config':c,'state':'preparing'})
                if t['config']!=c:raise ValueError('Trial settings changed')
                state.update(state='running',active_trial=name);report(root,state);runtime.prepare(job,c)
                if t.get('state')=='complete':continue
                from src.wengua.qwen8b_book_v1 import doctor
                preflight=doctor(c) if not p.testing else {'training_preflight':{'status':'ready_for_smoke_test'},'cuda_devices':[]}
                d.write(job/'doctor.json',preflight)
                if preflight['training_preflight']['status']!='ready_for_smoke_test':raise RuntimeError(preflight['training_preflight']['reason'])
                if t.get('probe',{}).get('state')!='complete':
                    t['probe']=launch(p,job,c,deadline,'fit',probe=True);report(root,state)
                    if t['probe']['state']!='complete':state['state']='paused';report(root,state);return 75
                cp=requested_checkpoint
                cp=cp if cp and Path(cp).resolve().is_relative_to(job.resolve()) else None
                t['fit']=launch(p,job,c,deadline,'fit',checkpoint=cp);report(root,state)
                if t['fit']['state']!='complete':state['state']='paused';report(root,state);return 75
                t['state']='complete';report(root,state)
            state.update(state='complete_review_required',active_trial=None,selection='none_until_full_answer_review',
                old_weights_preserved=True,new_long_training_started=False);report(root,state);return 0
        except BaseException as error:
            state.update(state='paused' if isinstance(error,(KeyboardInterrupt,TimeoutError)) else 'failed',error={'type':type(error).__name__,'message':str(error)})
            report(root,state);raise
