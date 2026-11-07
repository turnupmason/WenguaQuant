"""Finite inherited repair comparison, factual acceptance, then full courses."""
from pathlib import Path
import argparse,time,json,os,html,gc,re
from types import FunctionType
from wengua_course import data as d
from wengua_course_v53.checks import retention
from wengua_course_v55 import checks as old_checks
from wengua_course_v55.runner import evaluate_adapter as _evaluate_adapter
from . import runtime,initialization,checks

evaluate_adapter=FunctionType(_evaluate_adapter.__code__,globals(),'evaluate_adapter',_evaluate_adapter.__defaults__,_evaluate_adapter.__closure__)
evaluate_adapter.__kwdefaults__=_evaluate_adapter.__kwdefaults__


def config(parent,source_run,rate,phase='full'):
    c=d.read(runtime.CONFIG);c.update(course_phase=phase,learning_rate=rate,book_parent_snapshot=str(parent.resolve()),book_parent_provenance_sha256=d.digest((parent.parent/'parent_provenance.json').read_bytes()),legacy_source_run=str(Path(source_run).resolve()))
    if phase=='repair':c.update(num_train_epochs=4,development_generation_rows=96,eval_max_new_tokens=384)
    else:c.update(num_train_epochs=2,development_generation_rows=192,eval_max_new_tokens=1024)
    return c


def report(root,state):
    root=Path(root);d.write(root/'run_status.json',state)
    value=html.escape(json.dumps(state,ensure_ascii=False,indent=2))
    parts=['<!doctype html><meta charset="utf-8"><title>Wengua V5.6</title><style>body{font:16px system-ui;max-width:1250px;margin:24px auto;padding:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:12px 0}</style><h1>V5.6 课程修订、已有权重短训与验收</h1><p>短训先比较两个已学权重；事实、方向、条件和旧题保留通过才进入完整两学习率训练。loss不单独决定放行。开放案例仍需语义审阅。原教材、起盘、分数和权重保留。</p><details><summary>阶段状态、逐轮指标和验收原因</summary><pre>'+value+'</pre></details>']
    for path in sorted(root.glob('trials/*/generations/*.jsonl')):
        parts.append('<details><summary>'+html.escape(str(path.relative_to(root)))+'</summary>')
        for line in path.read_text(encoding='utf8').splitlines():
            r=json.loads(line);parts.append('<details><summary>'+html.escape(r['id'])+'</summary><h3>完整提问</h3><pre>'+html.escape(d.dumps(r['question']))+'</pre><h3>模型原答</h3><pre>'+html.escape(r['answer'])+'</pre><h3>来源目标</h3><pre>'+html.escape(r['expected'])+'</pre></details>')
        parts.append('</details>')
    (root/'results.html').write_text(''.join(parts),encoding='utf8')


def rows(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf8').splitlines() if s.strip()]


def rag_boundaries(generated):
    required={
      'exam_self':('日干',True),'exam_proxy':('时干',True),
      'health_daughter_father':('年干',True),'health_father_self':('日干',False),
    };items=[]
    for r in generated:
        key=r['id'].split(':actual_rag:')[-1];a=r['answer'];ok=True;reason='open_source_semantics_not_automatically_approved'
        if r.get('reached_token_limit') or not a.strip() or re.search(r'(.{2,24})\1{7,}',re.sub(r'\s+','',a)):ok=False;reason='empty_repeated_or_output_limit'
        elif key in required:
            role,unknown=required[key];g={'messages':r['question']+[{'role':'assistant','content':''}],'checks':{'kind':'role_boundary','answer':role,'unknown_year':unknown},'provenance':{'birth_year_unknown_in_question':unknown}}
            ok=checks.fact_check(r,g)['status']=='pass';reason='patient_or_candidate_identity_and_year_boundary'
            if key=='health_daughter_father':ok=ok and '乾宫' in a
        elif key=='marriage_same_palace':
            ok=all(w in a for w in ['吉门','吉格']) and any(w in a for w in ['不能','不一定','还须','还要求']) and not re.search(r'不(?:需要|必需|必|用)[^。；\n]{0,5}(?:吉门|吉格)',a);reason='both_missing_favorable_conditions_retained'
        elif key=='cross_domain_liuhe':
            ok=bool(re.search(r'婚姻[^。；\n]{0,15}媒人',a)) and bool(re.search(r'诉讼[^。；\n]{0,15}证[人据]',a)) and bool(re.search(r'(?:交易|求财)[^。；\n]{0,15}(?:中介|经纪)',a));reason='explicit_domain_to_role_mapping'
        items.append({'id':r['id'],'passed':ok,'scope':reason})
    critical=[r for r in items if r['scope']!='open_source_semantics_not_automatically_approved']
    needed=set(required)|{'marriage_same_palace','cross_domain_liuhe'}
    present={r['id'].split(':actual_rag:')[-1] for r in items}
    return {'passed':needed.issubset(present) and all(r['passed'] for r in items),'items':items,'critical_tasks':len(critical),'semantic_professional_approval':False}


def compare_rank(summary):
    values=[g['rate'] for g in summary['groups'].values()]
    return (min(values,default=0),sum(values)/max(1,len(values)),summary['citation']['rate'] or 0)


def run(args):
    runtime.configure(args.app_root);pack=runtime.verify_data();p=runtime.profile();b=p.book;root=b.private_output(args.output);root.mkdir(parents=True,exist_ok=True)
    contract={'schema':runtime.SCHEMA,'code_sha256':runtime.hashes(),'dataset_release_sha256':d.digest(runtime.RELEASE.read_text()),'repair_epochs':4,'full_rates':[1e-5,2e-5],'full_epochs':2,'original_test_used':False,'selection':'factual_direction_conditions_retention_then_language_tiebreak'}
    path=root/'sweep_contract.json'
    if path.exists() and d.read(path)!=contract:raise ValueError('V56 contract changed; previous run preserved')
    if not path.exists():d.write(path,contract)
    state=d.read(root/'run_status.json') if (root/'run_status.json').exists() else {'state':'created','trials':{},'professional_approval':False,'production_enabled':False}
    if state['state'] in ['complete_review_required','needs_course_review']:report(root,state);return 0
    deadline=time.time()+3600*args.hours
    from .process import launch
    def evaluate(job,c,adapter,tag,stage):
        result=launch(p,job,c,deadline,'evaluate',best={'adapter':str(adapter),'tag':tag,'eval_stage':stage})
        if result['state']!='complete':raise TimeoutError('Evaluation paused with answers preserved')
        return rows(job/'generations'/(tag+'.jsonl'))
    def trial(label,source,phase,rate):
        snapshot=initialization.snapshot(root/'parents'/label,source,testing=p.testing);c=config(snapshot,root,rate,phase)
        proof=root/'inheritance_proofs'/(label+'.json');proof.parent.mkdir(exist_ok=True)
        proof.write_bytes((snapshot.parent/'parent_provenance.json').read_bytes())
        job=p.engine.open_job(root/'trials'/label,c,resume=True);runtime.prepare(job,c)
        state.update(state='repair_training' if phase=='repair' else 'full_training',active_trial=label);report(root,state)
        oldgold=pack['legacy_v55_prompts'];parent_old=evaluate(job,c,source,'parent_v55_unchanged','v55');before=checks.score(parent_old,oldgold)
        smoke=launch(p,job,c,deadline,'fit',probe=True)
        if smoke['state']!='complete':raise TimeoutError('Pressure phase paused')
        cp=args.resume_from_checkpoint if args.resume_from_checkpoint and args.resume_from_checkpoint.resolve().is_relative_to(job.resolve()) else None
        fitted=launch(p,job,c,deadline,'fit',checkpoint=cp)
        if fitted['state']!='complete':raise TimeoutError('Training/generation paused; intact optimizer checkpoint retained')
        gold=runtime.select_development(pack['records'],96 if phase=='repair' else 192,repair=phase=='repair')
        if p.testing:gold=runtime.choose_development(pack,c['development_generation_rows'])
        baseline=checks.score(rows(job/'generations/validation_parent.jsonl'),gold);epochs=[]
        for e in fitted['epochs']:
            tag=f"epoch_{e['cycle']:02d}";current=checks.score(rows(job/'generations'/(tag+'_validation.jsonl')),gold)
            legacy=evaluate(job,c,e['adapter'],tag+'_v55_unchanged','v55');old=checks.score(legacy,oldgold);keep=retention(old,before['groups']);retain=retention(current,baseline['groups'])
            rag=evaluate(job,c,e['adapter'],tag+'_actual_rag_unchanged','legacy_rag');rb={'passed':True,'software_fixture_only':True} if p.testing and not rag else rag_boundaries(rag)
            if not p.testing:
                from src.wengua.book_rag_query import check_answer
                packets={r['id']:r['legacy_packet'] for r in runtime.prepared_prompts(job,'legacy_rag')}
                if set(packets)!={r['id'] for r in rag}:raise ValueError('Actual RAG regression rows incomplete')
                source_checks=[{'id':r['id'],'checks':check_answer(r['answer'],packets[r['id']])} for r in rag]
                rb['citation_and_chart_checks']=source_checks
                rb['passed']=rb['passed'] and all(r['checks']['status']=='citations_checked_semantics_unverified' for r in source_checks)
            eligible=checks.gate(current,phase=phase)['passed'] and keep['passed'] and retain['passed'] and rb['passed']
            entry={'epoch':e['epoch'],'steps':e['steps'],'adapter':e['adapter'],'validation_language_loss':e['validation_loss'],'source_checks':current,'parent_retention':retain,'v55_with_v56_checks':old,'v55_retention':keep,'original_v55_checks_unchanged':old_checks.score(legacy,oldgold),'rag_boundaries':rb,'eligible_for_next_phase':eligible,'professional_approval':False}
            d.write(job/'generations'/(tag+'_review.json'),entry);epochs.append(entry)
        state['trials'][label]={'state':'complete','phase':phase,'parent_source':str(source),'parent_baseline':baseline,'v55_parent_baseline':before,'epochs':epochs};report(root,state)
        return [(compare_rank(e['source_checks']),label,e) for e in epochs if e['eligible_for_next_phase']]
    try:
        with b.job_lock(root/'orchestrator.lock'):
            if getattr(args,'fixture_parents',None):
                if not p.testing:raise ValueError('Fixture parents forbidden in real training')
                sources=args.fixture_parents
            else:sources=initialization.default_sources()
            candidates=[]
            for label,source in sources.items():candidates+=trial('repair_'+label,Path(source),'repair',2e-5)
            if not candidates:
                state.update(state='needs_course_review',active_trial=None,reason='No inherited short-run checkpoint passed factual/condition/retention acceptance; full training was not started',all_repair_groups_finished=True)
                d.write(root/'review_nominee.json',{'adapter':None,'scope':'no_repair_candidate_passed_old_weights_preserved','professional_approval':False});report(root,state);return 0
            _,label,best=max(candidates,key=lambda x:(x[0],-x[2]['steps'],x[1]));source=Path(best['adapter'])
            d.write(root/'repair_selection.json',{'trial':label,'adapter':str(source),'steps':best['steps'],'basis':'fact_condition_direction_retention_acceptance','professional_approval':False})
            full=[]
            for rate in contract['full_rates']:full+=trial('full_lr_'+format(rate,'.0e').replace('-','m'),source,'full',rate)
            if full:
                _,label,best=max(full,key=lambda x:(x[0],-x[2]['steps'],x[1]));adapter=best['adapter'];scope='factual_acceptance_passed_open_case_review_required'
            else:label='accepted_repair_parent';adapter=str(source);scope='full_phase_did_not_pass_retain_accepted_repair_weights'
            d.write(root/'review_nominee.json',{'trial':label,'adapter':adapter,'scope':scope,'professional_approval':False,'production_enabled':False})
            state.update(state='complete_review_required',active_trial=None,selected_for_review=label,all_planned_groups_finished=True);report(root,state);return 0
    except BaseException as error:
        state.update(state='paused' if isinstance(error,(KeyboardInterrupt,TimeoutError)) else 'failed',error_type=type(error).__name__,error=str(error),outputs_preserved=True);report(root,state);raise
    finally:report(root,state)


def restore_all(key_file=None):
    import re
    from wengua_corpus.__main__ import restore
    from wengua_course_v53 import runtime as v53
    from wengua_course_v54 import runtime as v54
    from wengua_course_v55 import runtime as v55
    value=key_file.read_text(encoding='utf-8-sig').strip() if key_file else input('粘贴此前教材的64位口令（输入可见，只需本次输入一次）：').strip()
    if not re.fullmatch('[0-9a-fA-F]{64}',value):raise ValueError('Expected the existing 64-character delivery key')
    for version in [v53,v54,v55,runtime]:
        release=d.read(version.RELEASE)
        if all((version.DATA/n).is_file() and d.digest((version.DATA/n).read_bytes())==h for n,h in release['files'].items()):continue
        restore({'manifest':release['delivery_manifest'],'destination':version.DATA.relative_to(runtime.ROOT).as_posix(),'files':release['files']},bytes.fromhex(value))
    runtime.verify_data();print('V5.6课程、审阅Excel和数据库已恢复并校验。')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['restore','check','run','export','search','ask']);p.add_argument('--output',type=Path,default=runtime.ROOT/'local/course_v56/run1');p.add_argument('--app-root',type=Path);p.add_argument('--hours',type=float,default=72);p.add_argument('--resume',action='store_true');p.add_argument('--resume-from-checkpoint',type=Path);p.add_argument('--key-file',type=Path);p.add_argument('--archive',type=Path);p.add_argument('--question');p.add_argument('--domain');p.add_argument('--time');p.add_argument('--zone');p.add_argument('--chart',type=Path);a=p.parse_args(argv)
    if a.command=='restore':restore_all(a.key_file);return 0
    if a.command=='export':
        from wengua_course.runner import export_review
        print(export_review(a.output,a.archive or Path.home()/'Downloads/Wengua_V56_review.zip'));return 0
    runtime.configure(a.app_root)
    if a.command=='check':print(d.dumps(runtime.verify_data()['summary']));return 0
    if a.command=='search':
        if not a.question:p.error('--question required')
        from . import corpus
        print(d.dumps(corpus.retrieve(a.question,domain=a.domain)));return 0
    if a.command=='ask':
        if not a.question:p.error('--question required')
        nominee=d.read(a.output/'review_nominee.json')
        if not nominee.get('adapter'):raise ValueError('Short acceptance did not pass; inspect preserved answers before using a V56 model')
        adapter=Path(nominee['adapter']);initialization.inspect(adapter);job=adapter.parent.parent.parent
        if not job.resolve().is_relative_to(a.output.resolve()):raise ValueError('Selected adapter is outside this V56 run')
        c=d.read(job/'job_contract.json')['config']
        from transformers import AutoTokenizer
        from . import corpus
        tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision'],local_files_only=True,trust_remote_code=False)
        packet=corpus.prompt(a.question,tok,domain=a.domain,max_context=c['inference_max_context'],max_new_tokens=c['eval_max_new_tokens'],time_text=a.time,zone=a.zone,chart_file=a.chart)
        if packet['state']!='ready':print('当前来源资料不足以回答。');return 2
        target=a.output/'rag_prompts'/f'{time.time_ns()}.json';d.write(target,packet)
        from .process import launch
        result=launch(runtime.profile(),job,c,time.time()+3600,'ask',best={'adapter':str(adapter),'prompt':str(target)})
        print(result['answer'] if result['checks']['status']=='citations_checked_semantics_unverified' else '回答未通过引用或盘面检查，原答已保留。');print('研究回答记录：'+result['trace_path']);return 0
    if not 0<a.hours<=168:p.error('hours must be in (0,168]')
    if a.resume_from_checkpoint and not a.resume_from_checkpoint.resolve().is_relative_to(a.output.resolve()):p.error('Checkpoint must belong to this V56 run')
    os.environ.update(PYTORCH_CUDA_ALLOC_CONF=runtime.ALLOCATOR,PYTORCH_ALLOC_CONF=runtime.ALLOCATOR,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    return run(a)
