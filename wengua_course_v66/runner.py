"""Two fixed loss routes, one learned parent and byte-identical training order."""
from pathlib import Path
import html,time
from wengua_course import data as d
from . import runtime,initialization,evaluation
VARIANTS={'ordinary_ce':1,'decision_ce':4}
evaluate_adapter=evaluation.evaluate_adapter


def config(parent,root,variant):
    c=d.read(runtime.CONFIG);c.update(decision_loss_factor=VARIANTS[variant],book_parent_snapshot=str(Path(parent).resolve()),
        book_parent_provenance_sha256=d.digest((Path(parent).parent/'parent_provenance.json').read_bytes()),legacy_source_run=str(Path(root).resolve()))
    return c


def checkpoint_scope(root,path):
    if path is None:return None
    root=Path(root).resolve();path=Path(path).resolve()
    if not path.is_dir() or path.parent not in [root/'trials'/v/'cycle_01/trainer' for v in VARIANTS] or not path.name.startswith('checkpoint-') or not path.name[11:].isdigit():
        raise ValueError('Resume requires an intact formal V66 trial checkpoint, never V65 or smoke')
    return path


def report(root,state):
    d.write(root/'run_status.json',state)
    parts=['<!doctype html><meta charset="utf-8"><title>Wengua V6.6</title><style>body{font:16px system-ui;max-width:1100px;margin:24px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>V6.6 同父权重的损失对照</h1><p>普通CE与关键结论加权CE；相同可见课程、张量、顺序及学习率。原始回答逐项审阅，不以平均loss或词语命中自动替换旧模型。</p><pre>'+html.escape(d.dumps(state))+'</pre>']
    for path in sorted(root.glob('trials/*/generations/*.jsonl')):
        if path.name.endswith('.tokens.jsonl'):continue
        parts.append('<details><summary>'+html.escape(str(path.relative_to(root)))+'</summary>')
        for r in evaluation.read_cache(path):parts.append('<details><summary>'+html.escape(r['id'])+'</summary><pre>'+html.escape(d.dumps(r['question']))+'</pre><p>'+html.escape(r['answer'])+'</p><pre>参考目标：'+html.escape(r['expected'])+'</pre></details>')
        parts.append('</details>')
    (root/'results.html').write_text(''.join(parts),encoding='utf8')


def freeze(path,value):
    if path.exists() and d.read(path)!=value:raise ValueError('Existing experiment contract changed: '+str(path))
    if not path.exists():d.write(path,value)


def run(args):
    runtime.configure(args.app_root);pack=runtime.verify_data();p=runtime.profile();b=p.book
    if not p.testing and b.stack_versions()!=d.read(runtime.RELEASE)['original_stack']:raise ValueError('Use the existing pinned V65 CUDA environment')
    from .process import launch
    root=b.private_output(args.output);root.mkdir(parents=True,exist_ok=True);deadline=time.time()+3600*args.hours;cp=checkpoint_scope(root,args.resume_from_checkpoint)
    source=initialization.default_source();parent=initialization.snapshot(root/'initialization',source,testing=p.testing)
    ev=evaluation.build(runtime.old.verify_data()) if not p.testing else pack['fixture_evaluation_pack']
    contract={'schema':runtime.SCHEMA,'code_sha256':runtime.hashes(),'dataset_release_sha256':d.digest(runtime.RELEASE.read_bytes()),
              'parent_provenance_sha256':d.digest((parent.parent/'parent_provenance.json').read_bytes()),'variants':VARIANTS,
              'evaluation_pack_sha256':d.digest(d.dumps(ev)),'same_training_tensors_and_order':True,'old_optimizer_inherited':False,
              'automatic_long_training':False,'automatic_professional_promotion':False,'original_test_used':False}
    freeze(root/'sweep_contract.json',contract)
    state=d.read(root/'run_status.json') if (root/'run_status.json').exists() else {'state':'preparing','trials':{},'curriculum':pack['summary'],'production_enabled':False}
    if state['state']=='complete_review_required':
        print('V6.6两组短训与评估已完成；此次只导出结果，不重复训练。',flush=True);report(root,state);return 0
    with b.keep_awake(),b.job_lock(root/'run.lock'):
        try:
            c=config(parent,root,'ordinary_ce');base=p.engine.open_job(root/'trials/baseline',c,resume=True);freeze(base/'evaluation_pack.json',ev)
            state.update(state='baseline_evaluation',active_trial='baseline');report(root,state)
            state['baseline']=launch(p,base,c,deadline,'evaluate',best={'adapter':str(parent),'tag':'parent'})
            if state['baseline']['state']!='complete':state['state']='paused';report(root,state);return 75
            tensor_sha=None
            for variant in VARIANTS:
                c=config(parent,root,variant);job=p.engine.open_job(root/'trials'/variant,c,resume=True);meta=runtime.prepare(job,c)
                freeze(job/'evaluation_pack.json',ev)
                if tensor_sha is not None and meta['training_tensor_sha256']!=tensor_sha:raise ValueError('Loss groups received different training tensors')
                tensor_sha=meta['training_tensor_sha256'];t=state['trials'].setdefault(variant,{'config':c});state.update(state='training',active_trial=variant);report(root,state)
                if t.get('fit',{}).get('state')!='complete':
                    if not p.testing:
                        from src.wengua.qwen8b_book_v1 import doctor
                        check=doctor(c);d.write(job/'doctor.json',check)
                        if check['training_preflight']['status']!='ready_for_smoke_test':raise RuntimeError(check['training_preflight']['reason'])
                    if t.get('probe',{}).get('state')!='complete':
                        t['probe']=launch(p,job,c,deadline,'fit',probe=True);report(root,state)
                        if t['probe']['state']!='complete':state['state']='paused';report(root,state);return 75
                    requested=cp if cp and cp.is_relative_to(job.resolve()) else None
                    t['fit']=launch(p,job,c,deadline,'fit',checkpoint=requested);report(root,state)
                    if t['fit']['state']!='complete':state['state']='paused';report(root,state);return 75
                adapter=t['fit']['last_adapter'];state.update(state='answer_evaluation',active_trial=variant);report(root,state)
                t['evaluation']=launch(p,job,c,deadline,'evaluate',best={'adapter':adapter,'tag':variant})
                if t['evaluation']['state']!='complete':state['state']='paused';report(root,state);return 75
                t['state']='complete';report(root,state)
            state.update(state='complete_review_required',active_trial=None,selection='none_until_whole_answer_review',training_tensor_sha256=tensor_sha,
                         old_weights_preserved=True,additional_long_training_started=False);report(root,state);return 0
        except BaseException as error:
            state.update(state='paused' if isinstance(error,(KeyboardInterrupt,TimeoutError)) else 'failed',error={'type':type(error).__name__,'message':str(error)})
            report(root,state);raise
