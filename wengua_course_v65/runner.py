"""Two source initializations, identical curriculum/rate, all epoch answers saved."""
from pathlib import Path
from types import FunctionType
import time,html,gc
from wengua_course import data as d
from wengua_course_v60 import runner as old
from . import runtime,initialization,baselines
VARIANTS=('v63_trained','official_zero')
read_rows=old.read_rows
evaluate_adapter=FunctionType(old.evaluate_adapter.__code__,globals(),'evaluate_adapter',old.evaluate_adapter.__defaults__)
evaluate_adapter.__kwdefaults__=old.evaluate_adapter.__kwdefaults__


def checkpoint_scope(root,path):
    if path is None:return None
    path=Path(path).resolve();root=Path(root).resolve()
    if not path.is_dir() or path.parent not in [root/'trials'/v/'cycle_01/trainer' for v in VARIANTS] or not path.name.startswith('checkpoint-') or not path.name[11:].isdigit():
        raise ValueError('Explicit checkpoint must be a formal checkpoint in this V65 run')
    return path


def config(parent,root,variant):
    c=d.read(runtime.CONFIG);c.update(source_variant=variant,initialization_label=variant,book_parent_snapshot=str(Path(parent).resolve()),
        book_parent_provenance_sha256=d.digest((Path(parent).parent/'parent_provenance.json').read_bytes()),legacy_source_run=str(Path(root).resolve()))
    return c


def report(root,state):
    d.write(root/'run_status.json',state)
    lines=['<!doctype html><meta charset="utf-8"><title>Wengua V6.5</title><style>body{font:16px system-ui;max-width:1120px;margin:24px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:10px}</style><h1>V6.5 对象、属性、方向与领域结论</h1><p>同课程、同学习率，对照V6.3已学权重与官方零更新起点，各两轮；保留全部阶段，不自动晋升。</p><pre>'+html.escape(d.dumps(state))+'</pre>']
    for path in sorted(root.glob('trials/*/generations/*.jsonl')):
        lines.append('<details><summary>'+html.escape(str(path.relative_to(root)))+'</summary>')
        for r in read_rows(path):lines.append('<details><summary>'+html.escape(r['id'])+'</summary><pre>'+html.escape(d.dumps(r['question']))+'</pre><p>'+html.escape(r['answer'])+'</p><pre>'+html.escape(r['expected'])+'</pre></details>')
        lines.append('</details>')
    (root/'results.html').write_text(''.join(lines),encoding='utf8')


def run(args):
    runtime.configure(args.app_root);pack=runtime.verify_data();p=runtime.profile();book=p.book
    if not p.testing and book.stack_versions()!=d.read(runtime.RELEASE)['original_stack']:raise ValueError('Use the original pinned V63/V64 environment')
    from .process import launch
    root=book.private_output(args.output);root.mkdir(parents=True,exist_ok=True);deadline=time.time()+3600*args.hours
    cp=checkpoint_scope(root,args.resume_from_checkpoint)
    parents={v:initialization.snapshot(root/'initializations'/v,initialization.default_source(v),testing=p.testing) for v in VARIANTS}
    contract={'schema':runtime.SCHEMA,'code_sha256':runtime.hashes(),'dataset_release_sha256':d.digest(runtime.RELEASE.read_bytes()),
       'source_variants':list(VARIANTS),'learning_rate':2e-5,'epochs':2,'same_course':True,
       'parent_provenance':{v:d.digest((parents[v].parent/'parent_provenance.json').read_bytes()) for v in VARIANTS},
       'old_optimizer_inherited':False,'automatic_professional_promotion':False,'original_test_used':False}
    path=root/'sweep_contract.json'
    if path.exists() and (not args.resume or d.read(path)!=contract):raise ValueError('V65 run contract changed; existing run preserved')
    if not path.exists():d.write(path,contract)
    state=d.read(root/'run_status.json') if (root/'run_status.json').exists() else {'state':'preparing','trials':{},'baselines':{},'curriculum':pack['summary'],'production_enabled':False}
    if state['state']=='complete_review_required':
        print('V6.5此前已完成；此次只导出已有结果，不重复训练。',flush=True);report(root,state);return 0
    with book.keep_awake(),book.job_lock(root/'run.lock'):
        try:
            for variant in VARIANTS:
                c=config(parents[variant],root,variant);source=initialization.default_source(variant)
                baseline=p.engine.open_job(root/'trials'/(variant+'_baseline'),c,resume=True);runtime.prepare(baseline,c)
                baselines.seed(baseline,pack,c,d.digest((Path(source)/'adapter_model.safetensors').read_bytes()),testing=p.testing,fixture_rows=pack.get('fixture_parent_rows'))
                state.update(state='baseline_review',active_trial=variant+'_baseline');report(root,state)
                result=launch(p,baseline,c,deadline,'evaluate',best={'adapter':str(source),'tag':'parent_review','eval_stage':'review'})
                state['baselines'][variant]=result;report(root,state)
                if result['state']!='complete':state['state']='paused';report(root,state);return 75
                job=p.engine.open_job(root/'trials'/variant,c,resume=True);runtime.prepare(job,c)
                t=state['trials'].setdefault(variant,{'config':c,'state':'preparing'})
                if t['config']!=c:raise ValueError('Matched trial configuration changed')
                if t['state']=='complete':continue
                state.update(state='running',active_trial=variant);report(root,state)
                from src.wengua.qwen8b_book_v1 import doctor
                check=doctor(c) if not p.testing else {'training_preflight':{'status':'ready_for_smoke_test'},'cuda_devices':[]}
                d.write(job/'doctor.json',check)
                if check['training_preflight']['status']!='ready_for_smoke_test':raise RuntimeError(check['training_preflight']['reason'])
                if t.get('probe',{}).get('state')!='complete':
                    t['probe']=launch(p,job,c,deadline,'fit',probe=True);report(root,state)
                    if t['probe']['state']!='complete':state['state']='paused';report(root,state);return 75
                requested=cp if cp and cp.is_relative_to(job.resolve()) else None
                t['fit']=launch(p,job,c,deadline,'fit',checkpoint=requested);report(root,state)
                if t['fit']['state']!='complete':state['state']='paused';report(root,state);return 75
                t['state']='complete';report(root,state)
            state.update(state='complete_review_required',active_trial=None,selection='none_until_whole_answer_review',
                         old_weights_preserved=True,additional_long_training_started=False);report(root,state);return 0
        except BaseException as error:
            state.update(state='paused' if isinstance(error,(KeyboardInterrupt,TimeoutError)) else 'failed',error={'type':type(error).__name__,'message':str(error)})
            report(root,state);raise
