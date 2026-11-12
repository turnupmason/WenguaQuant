"""One learned source, continuous two-epoch training, all stages retained."""
from pathlib import Path
import html,time
from wengua_course import data as d
from . import runtime,initialization,evaluation,baselines
TRIAL='object_condition_repair'
evaluate_adapter=evaluation.evaluate_adapter


def config(parent,root):
    c=d.read(runtime.CONFIG);c.update(book_parent_snapshot=str(Path(parent).resolve()),book_parent_provenance_sha256=d.digest((Path(parent).parent/'parent_provenance.json').read_bytes()),legacy_source_run=str(Path(root).resolve()))
    return c


def checkpoint_scope(root,path):
    if path is None:return None
    root=Path(root).resolve();path=Path(path).resolve()
    if not path.is_dir() or path.parent!=root/'trials'/TRIAL/'cycle_01/trainer' or not path.name.startswith('checkpoint-') or not path.name[11:].isdigit():
        raise ValueError('Resume only from an intact formal V67 checkpoint in this run')
    return path


def freeze(path,value):
    if path.exists() and d.read(path)!=value:raise ValueError('Existing V67 experiment changed: '+str(path))
    if not path.exists():d.write(path,value)


def report(root,state):
    d.write(root/'run_status.json',state)
    parts=['<!doctype html><meta charset="utf-8"><title>Wengua V6.7</title><style>body{font:16px system-ui;max-width:1120px;margin:24px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>V6.7 对象与独立条件修复</h1><p>继承V6.6普通组，普通CE连续两轮；复用原同题父回答，只生成新增父题。每个阶段完整回答保留，不以loss自动替换旧模型。</p><pre>'+html.escape(d.dumps(state))+'</pre>']
    for path in sorted(root.glob('trials/*/generations/*.jsonl')):
        parts.append('<details><summary>'+html.escape(str(path.relative_to(root)))+'</summary>')
        for r in evaluation.read_cache(path):parts.append('<details><summary>'+html.escape(r['id'])+'</summary><pre>'+html.escape(d.dumps(r['question']))+'</pre><p>'+html.escape(r['answer'])+'</p><pre>来源参考：'+html.escape(r['expected'])+'</pre></details>')
        parts.append('</details>')
    (root/'results.html').write_text(''.join(parts),encoding='utf8')


def run(args):
    runtime.configure(args.app_root);pack=runtime.verify_data();p=runtime.profile();b=p.book
    if not p.testing and b.stack_versions()!=d.read(runtime.RELEASE)['original_stack']:raise ValueError('Use the existing pinned CUDA environment')
    from .process import launch
    root=b.private_output(args.output);root.mkdir(parents=True,exist_ok=True);deadline=time.time()+3600*args.hours;cp=checkpoint_scope(root,args.resume_from_checkpoint)
    source=initialization.default_source();parent=initialization.snapshot(root/'initialization',source,testing=p.testing);c=config(parent,root)
    ev=evaluation.build(pack,runtime.before.old.verify_data()) if not p.testing else pack['fixture_evaluation_pack']
    contract={'schema':runtime.SCHEMA,'code_sha256':runtime.hashes(),'dataset_release_sha256':d.digest(runtime.RELEASE.read_bytes()),
       'parent_provenance_sha256':d.digest((parent.parent/'parent_provenance.json').read_bytes()),'evaluation_pack_sha256':d.digest(d.dumps(ev)),
       'ordinary_ce':True,'epochs':2,'optimizer_reset_between_epochs':False,'old_optimizer_inherited':False,
       'automatic_professional_promotion':False,'additional_long_training':False,'original_test_used':False}
    freeze(root/'sweep_contract.json',contract)
    state=d.read(root/'run_status.json') if (root/'run_status.json').exists() else {'state':'preparing','curriculum':pack['summary'],'production_enabled':False}
    if state['state']=='complete_review_required':
        print('V6.7此前已完成；此次只导出结果，不重复训练。',flush=True);report(root,state);return 0
    with b.keep_awake(),b.job_lock(root/'run.lock'):
        try:
            base=p.engine.open_job(root/'trials/parent_baseline',c,resume=True);freeze(base/'evaluation_pack.json',ev)
            reuse=baselines.seed(base,ev,testing=p.testing,fixture_source=pack.get('fixture_parent_source'));state['parent_reuse']=reuse
            state.update(state='parent_new_question_evaluation');report(root,state)
            state['baseline']=launch(p,base,c,deadline,'evaluate',best={'adapter':str(parent),'tag':'parent','record_variant':'ordinary_ce'})
            if state['baseline']['state']!='complete':state['state']='paused';report(root,state);return 75
            job=p.engine.open_job(root/'trials'/TRIAL,c,resume=True);meta=runtime.prepare(job,c);freeze(job/'evaluation_pack.json',ev)
            state['prepared_summary']={k:meta[k] for k in ['training_rows','deduplicated_supervised_views','training_tensor_sha256','max_actual_train_tokens']}
            trial=state.setdefault('trial',{'config':c,'state':'preparing'});state.update(state='training');report(root,state)
            if trial.get('fit',{}).get('state')!='complete':
                if not p.testing:
                    from src.wengua.qwen8b_book_v1 import doctor
                    check=doctor(c);d.write(job/'doctor.json',check)
                    if check['training_preflight']['status']!='ready_for_smoke_test':raise RuntimeError(check['training_preflight']['reason'])
                if trial.get('probe',{}).get('state')!='complete':
                    trial['probe']=launch(p,job,c,deadline,'fit',probe=True);report(root,state)
                    if trial['probe']['state']!='complete':state['state']='paused';report(root,state);return 75
                trial['fit']=launch(p,job,c,deadline,'fit',checkpoint=cp);report(root,state)
                if trial['fit']['state']!='complete':state['state']='paused';report(root,state);return 75
            reviews=trial.setdefault('epoch_evaluations',{})
            for epoch in trial['fit']['epochs']:
                tag=f'epoch_{epoch["cycle"]:02d}';state.update(state='answer_evaluation',active_epoch=epoch['cycle']);report(root,state)
                reviews[tag]=launch(p,job,c,deadline,'evaluate',best={'adapter':epoch['adapter'],'tag':tag})
                if reviews[tag]['state']!='complete':state['state']='paused';report(root,state);return 75
            trial['state']='complete';state.update(state='complete_review_required',selection='none_until_full_answer_retention_review',
                old_weights_preserved=True,additional_long_training_started=False);report(root,state);return 0
        except BaseException as error:
            state.update(state='paused' if isinstance(error,(KeyboardInterrupt,TimeoutError)) else 'failed',error={'type':type(error).__name__,'message':str(error)})
            report(root,state);raise
