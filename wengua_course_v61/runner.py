"""Two finite inherited trials; every epoch saved; model quality stays reviewable."""
from pathlib import Path
from types import FunctionType
from argparse import Namespace
import time,html,json,gc
from wengua_course import data as d
from wengua_course_v60 import runner as prior,checks
from . import runtime,initialization
RATES=(3e-6,1e-5)
read_rows=prior.read_rows
# Exact provenanced generation kernel, rebound to this version's prompt files.
evaluate_adapter=FunctionType(prior.evaluate_adapter.__code__,globals(),'evaluate_adapter',prior.evaluate_adapter.__defaults__)
evaluate_adapter.__kwdefaults__=prior.evaluate_adapter.__kwdefaults__

def config(parent,root,rate):
    c=d.read(runtime.CONFIG)
    c.update(learning_rate=rate,book_parent_snapshot=str(Path(parent).resolve()),book_parent_provenance_sha256=d.digest((Path(parent).parent/'parent_provenance.json').read_bytes()),legacy_source_run=str(Path(root).resolve()))
    return c

def report(root,state):
    d.write(root/'run_status.json',state)
    parts=['<!doctype html><meta charset="utf-8"><title>Wengua V6.1</title><style>body{font:16px system-ui;max-width:1100px;margin:24px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>V6.1 正常对话与条件应用</h1><p>两学习率各两轮，继承同一保留权重。保存每轮回答和检查点；自动指标只作诊断，未自动替换原模型。</p><pre>'+html.escape(d.dumps(state))+'</pre>']
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
    contract={'schema':runtime.SCHEMA,'code_sha256':runtime.hashes(),'dataset_release_sha256':d.digest(runtime.RELEASE.read_bytes()),'rates':list(RATES),'epochs':2,'same_parent_each_trial':True,'no_automatic_professional_promotion':True,'original_test_used':False}
    path=root/'sweep_contract.json'
    if path.exists() and d.read(path)!=contract:raise ValueError('V61 output contract changed; existing checkpoints preserved')
    if path.exists() and not args.resume:raise ValueError('Use --resume for an existing run')
    if not path.exists():d.write(path,contract)
    state=d.read(root/'run_status.json') if (root/'run_status.json').exists() else {'state':'preparing','trials':{},'curriculum':pack['summary'],'production_enabled':False}
    if state['state']=='complete_review_required':report(root,state);return 0
    with b.keep_awake(),b.job_lock(root/'run.lock'):
        source=initialization.default_source();source_sha=d.digest((source/'adapter_model.safetensors').read_bytes())
        parent=initialization.snapshot(root/'parents/retained',source,testing=p.testing)
        d.write(root/'retained_best.json',{'adapter':str(source),'adapter_sha256':source_sha,'unchanged_original_parent':True,'professional_approval':False})
        try:
            nominees=[]
            for rate in RATES:
                name='lr_3em06' if rate==3e-6 else 'lr_1em05';c=config(parent,root,rate)
                job=p.engine.open_job(root/'trials'/name,c,resume=True)
                t=state['trials'].setdefault(name,{'config':c,'state':'preparing'})
                if t['config']!=c:raise ValueError('Trial configuration changed')
                state.update(state='running',active_trial=name);report(root,state)
                runtime.prepare(job,c)
                if not (job/'doctor.json').exists():
                    from src.wengua.qwen8b_book_v1 import doctor as assess
                    doctor=assess(c) if not p.testing else {'training_preflight':{'status':'ready_for_smoke_test'},'cuda_devices':[]};d.write(job/'doctor.json',doctor)
                    if doctor['training_preflight']['status']!='ready_for_smoke_test':raise RuntimeError(doctor['training_preflight']['reason'])
                if t.get('probe',{}).get('state')!='complete':
                    t['probe']=launch(p,job,c,deadline,'fit',probe=True);report(root,state)
                    if t['probe']['state']!='complete':state['state']='paused';report(root,state);return 0
                cp=args.resume_from_checkpoint
                if cp and not Path(cp).resolve().is_relative_to((root/'trials').resolve()):raise ValueError('Explicit checkpoint must belong to this V61 run')
                cp=cp if cp and Path(cp).resolve().is_relative_to(job.resolve()) else None
                t['fit']=launch(p,job,c,deadline,'fit',checkpoint=cp)
                report(root,state)
                if t['fit']['state']!='complete':state['state']='paused';report(root,state);return 0
                gold=runtime.choose_development(pack,c['development_generation_rows'])
                scores={}
                parent_answer=job/'generations/validation_parent.jsonl'
                scores['parent']=checks.score(read_rows(parent_answer),gold)
                for epoch in t['fit']['epochs']:
                    tag=f'epoch_{epoch["cycle"]:02d}_validation';path=job/'generations'/(tag+'.jsonl')
                    scores[tag]=checks.score(read_rows(path),gold)
                    nominees.append({'adapter':epoch['adapter'],'trial':name,'epoch':epoch['cycle'],'steps':epoch['steps'],'validation_loss':epoch['validation_loss'],'semantic_approval':False})
                t['finite_checks']=scores
                # RAG generation only for this trial's language-loss nominee,
                # rather than repeating all domain suites at every epoch.
                selected=t['fit']['adapter']
                if name=='lr_3em06':
                    t['parent_rag']=launch(p,job,c,deadline,'evaluate',best={'adapter':str(source),'tag':'parent_actual_rag','eval_stage':'rag'})
                    if t['parent_rag']['state']!='complete':state['state']='paused';report(root,state);return 0
                t['rag']=launch(p,job,c,deadline,'evaluate',best={'adapter':selected,'tag':'selected_actual_rag','eval_stage':'rag'})
                if t['rag']['state']!='complete':state['state']='paused';report(root,state);return 0
                t['state']='complete';report(root,state)
            nominee=min(nominees,key=lambda x:(x['validation_loss'],x['trial'],x['steps']))
            d.write(root/'research_nominee.json',nominee|{'adapter_sha256':d.digest((Path(nominee['adapter'])/'adapter_model.safetensors').read_bytes()),'selection':'language_loss_only_not_professional_best','original_retained_adapter_replaced':False})
            state.update(state='complete_review_required',active_trial=None,research_nominee=nominee,weights_and_optimizer_checkpoints_preserved=True)
            report(root,state);return 0
        except BaseException as error:
            state.update(state='paused' if isinstance(error,(KeyboardInterrupt,TimeoutError)) else 'failed',error={'type':type(error).__name__,'message':str(error)})
            report(root,state);raise
