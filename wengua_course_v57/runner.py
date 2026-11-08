"""Mixed inherited pilot, layered assessment and one finite full curriculum."""
from pathlib import Path
from argparse import ArgumentParser
import time,json,os,gc,html,shutil,re
from wengua_course import data as d
from . import runtime,initialization,checks,evaluation


def read_rows(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf8').splitlines() if x.strip()]


def evaluate_adapter(p,job,c,deadline,*,best,factory=None):
    adapter=Path(best['adapter']);initialization.inspect(adapter,testing=p.testing);stage=best.get('eval_stage','application');tag=best['tag']
    if not tag.replace('_','').isalnum():raise ValueError('Unsafe evaluation tag')
    rows=runtime.prepared_prompts(job,stage);inference=c|{'eval_max_new_tokens':best.get('max_new_tokens',c['eval_max_new_tokens'])};path=Path(job)/'generations'/(tag+'.jsonl');path.parent.mkdir(exist_ok=True)
    proof={'adapter_sha256':d.digest((adapter/'adapter_model.safetensors').read_bytes()),'prompt_sha256':d.digest(d.dumps([r['messages'] for r in rows])),'ids':[r['id'] for r in rows],'max_new_tokens':inference['eval_max_new_tokens'],'eval_stage':stage}
    receipt=path.with_suffix('.provenance.json')
    if receipt.exists() and d.read(receipt)!=proof:raise ValueError('Evaluation source, weights or token budget changed')
    if path.exists() and not receipt.exists():
        reuse=path.with_suffix('.reuse.json')
        if not reuse.exists():raise ValueError('Cached answers lack sealed generation or reuse provenance')
        reused=d.read(reuse)
        if reused['adapter_sha256']!=proof['adapter_sha256'] or reused['max_new_tokens']!=proof['max_new_tokens'] or reused['target_sha256']!=d.digest(path.read_bytes()):raise ValueError('Reused answer weights, budget or bytes changed')
    d.write(receipt,proof)
    if path.exists():
        saved=read_rows(path);gold={r['id']:r for r in rows}
        for r in saved:
            if r['id'] not in gold or r['question']!=gold[r['id']]['messages'][:-1] or r['expected']!=gold[r['id']]['answer']:raise ValueError('Cached question/target changed')
        if len(saved)==len(rows) and len({r['id'] for r in saved})==len(rows):return {'state':'complete','requested':len(rows),'completed':len(rows),'cache_hit_without_model_load':True}
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel
    from src.wengua.qwen_finetune import prepare_frozen_base
    if factory:tok,base,_=factory()
    else:tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision'],local_files_only=True,trust_remote_code=False);base,_=p.book.load_book_cuda(c,False)
    base=prepare_frozen_base(base,gradient_checkpointing=False);model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False)
    try:return p.book.generate_rows(model,tok,rows,path,inference,deadline,tag)
    finally:
        model=base=None;gc.collect()
        if not factory:torch.cuda.empty_cache()


def config(parent,root,phase):
    c=d.read(runtime.CONFIG);c.update(course_phase=phase,num_train_epochs=1 if phase=='pilot' else 2,book_parent_snapshot=str(parent.resolve()),book_parent_provenance_sha256=d.digest((parent.parent/'parent_provenance.json').read_bytes()),legacy_source_run=str(Path(root).resolve()))
    return c


def report(root,state):
    root=Path(root);d.write(root/'run_status.json',state);parts=['<!doctype html><meta charset="utf-8"><title>Wengua V5.7</title><style>body{font:16px system-ui;max-width:1250px;margin:24px auto;padding:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:12px 0}</style><h1>V5.7 来源完整、条件对照与分层训练</h1><p>试训同时包含基础、领域与作者案例；小套核对用于训练路径和能力留存，最终门槛与开放案例语义审阅分开。原资料、旧成绩和权重保留。</p><details open><summary>阶段状态与核对</summary><pre>'+html.escape(d.dumps(state))+'</pre></details>']
    for path in sorted(root.glob('trials/*/generations/*.jsonl')):
        parts.append('<details><summary>'+html.escape(str(path.relative_to(root)))+'</summary>')
        for r in read_rows(path):parts.append('<details><summary>'+html.escape(r['id'])+'</summary><h3>问题</h3><pre>'+html.escape(d.dumps(r['question']))+'</pre><h3>原答</h3><pre>'+html.escape(r['answer'])+'</pre><h3>来源目标</h3><pre>'+html.escape(r['expected'])+'</pre></details>')
        parts.append('</details>')
    (root/'results.html').write_text(''.join(parts),encoding='utf8')


def learning_gate(current,before,rag,rag_before):
    """Allow research learning after non-regression, not false expert approval."""
    failures=[]
    if not current['complete'] or not rag['complete'] or not before['complete'] or not rag_before['complete']:failures.append('missing_answers')
    if {r['id'] for r in rag['items']}!={r['id'] for r in rag_before['items']}:failures.append('changed_rag_tasks')
    if set(current['groups'])!=set(before['groups']):failures.append('changed_fact_categories')
    for k,p in before['groups'].items():
        v=current['groups'].get(k,{})
        if v.get('total')!=p['total'] or v.get('rate',0)<p['rate']-.05:failures.append(k+':fact_retention')
    previous={r['id']:r for r in rag_before['items']}
    for r in rag['items']:
        old=previous.get(r['id'])
        if not old:failures.append(r['id']+':changed_rag_task');continue
        if old['fact']['status']=='pass' and r['fact']['status']!='pass':failures.append(r['id']+':fact_regression')
        if old['fact'].get('complete') and not r['fact'].get('complete'):failures.append(r['id']+':condition_regression')
    for item in [*current['items'],*rag['items']]:
        if item['fact']['reason'] in ['empty','output_limit','repeated_output']:failures.append(item['id']+':bad_generation')
    for cur,prior in [(current,before),(rag,rag_before)]:
        if prior['citation']['rate'] is not None and (cur['citation']['rate'] or 0)<prior['citation']['rate']-.05:failures.append('citation_retention')
    return {'passed':not failures,'failed':failures,'scope':'training_path_and_retention_only','professional_approval':False}


def readiness(current,rag):
    base=checks.acceptance(current,current)
    return {'passed':base['passed'] and rag['complete'] and all(r['fact']['status']=='pass' and r['fact']['complete'] for r in rag['items']),'fact_gate':base,'open_case_semantics_approved':False,'production_enabled':False}


def rank(summary):
    groups=list(summary['groups'].values());return (sum(g['pass'] for g in groups)/max(1,sum(g['total'] for g in groups)),sum(g['complete'] for g in groups),summary['citation']['rate'] or 0)


def copy_answers(source,target,gold,*,adapter_sha,scope,max_new_tokens=768):
    """Reuse only identical questions, targets, budgets and sealed adapter."""
    source,target=Path(source),Path(target)
    if target.exists():return
    rows=read_rows(source);byid={r['id']:r for r in gold};valid=[]
    for r in rows:
        if r['id'] in byid:
            g=byid[r['id']]
            if r['question']!=g['messages'][:-1] or r['expected']!=g['answer']:raise ValueError('Cannot reuse changed question')
            valid.append(r)
    target.parent.mkdir(exist_ok=True)
    with target.open('w',encoding='utf8') as f:
        for r in valid:f.write(d.dumps(r)+'\n')
    d.write(target.with_suffix('.reuse.json'),{'source_file':str(source),'source_sha256':d.digest(source.read_bytes()),'adapter_sha256':adapter_sha,'scope':scope,'rows_reused':len(valid),'max_new_tokens':max_new_tokens,'target_sha256':d.digest(target.read_bytes()),'not_training_input':True})


def run(args):
    runtime.configure(args.app_root);pack=runtime.verify_data();p=runtime.profile();b=p.book;root=b.private_output(args.output);root.mkdir(parents=True,exist_ok=True)
    contract={'schema':runtime.SCHEMA,'code_sha256':runtime.hashes(),'dataset_release_sha256':d.digest(runtime.RELEASE.read_text()),'pilot_epochs':1,'full_epochs':2,'rates':[1e-5],'layered_evaluation':True,'original_test_used':False}
    path=root/'sweep_contract.json'
    if path.exists() and d.read(path)!=contract:raise ValueError('V57 run changed; previous results preserved')
    if not path.exists():d.write(path,contract)
    state=d.read(root/'run_status.json') if (root/'run_status.json').exists() else {'state':'created','trials':{},'production_enabled':False,'professional_approval':False}
    if state['state'] in ['complete_review_required','needs_course_review']:report(root,state);return 0
    deadline=time.time()+3600*args.hours
    from .process import launch
    def evaluate(job,c,adapter,tag,stage,cap=None):
        best={'adapter':str(adapter),'tag':tag,'eval_stage':stage}
        if cap and not p.testing:best['max_new_tokens']=cap
        result=launch(p,job,c,deadline,'evaluate',best=best)
        if result['state']!='complete':raise TimeoutError('Evaluation paused; raw answers preserved')
        return read_rows(job/'generations'/(tag+'.jsonl'))
    def prepare(label,source,phase):
        parent=initialization.snapshot(root/'parents'/label,source,testing=p.testing);c=config(parent,root,phase);job=p.engine.open_job(root/'trials'/label,c,resume=True);runtime.prepare(job,c)
        proof=root/'inheritance_proofs'/(label+'.json');proof.parent.mkdir(exist_ok=True);proof.write_bytes((parent.parent/'parent_provenance.json').read_bytes());return job,c
    def fit(job,c):
        state.update(state='training_'+c['course_phase'],active_trial=job.name);report(root,state)
        smoke=launch(p,job,c,deadline,'fit',probe=True)
        if smoke['state']!='complete':raise TimeoutError('Pressure test paused')
        checkpoint=args.resume_from_checkpoint if args.resume_from_checkpoint and args.resume_from_checkpoint.resolve().is_relative_to(job.resolve()) else None
        result=launch(p,job,c,deadline,'fit',checkpoint=checkpoint)
        if result['state']!='complete':raise TimeoutError('Training/generation paused; intact optimizer checkpoint retained')
        return result
    def reviews(job,c,result):
        gold=runtime.choose_development(pack,c['development_generation_rows']);base=checks.score(read_rows(job/'generations/validation_parent.jsonl'),gold);rg=runtime.prepared_prompts(job,'rag')
        rb=evaluate(job,c,Path(d.read(Path(c['book_parent_snapshot']).parent/'parent_provenance.json')['source_path']),'parent_actual_rag','rag');ragbase=checks.score(rb,rg)
        entries=[]
        for e in result['epochs']:
            tag='epoch_'+str(e['cycle']).zfill(2);cur=checks.score(read_rows(job/'generations'/(tag+'_validation.jsonl')),gold)
            raw=evaluate(job,c,Path(e['adapter']),tag+'_actual_rag','rag');rag=checks.score(raw,rg);gate=learning_gate(cur,base,rag,ragbase)
            entry={'epoch':e['epoch'],'steps':e['steps'],'adapter':e['adapter'],'language_loss':e['validation_loss'],'micro_checks':cur,'actual_rag_checks':rag,'training_retention_gate':gate,'factual_readiness':readiness(cur,rag),'professional_approval':False}
            d.write(job/'generations'/(tag+'_review.json'),entry);entries.append(entry)
        state['trials'][job.name]={'state':'complete','phase':c['course_phase'],'parent_micro':base,'parent_rag':ragbase,'epochs':entries};report(root,state);return entries
    try:
        with b.job_lock(root/'orchestrator.lock'):
            source=Path(args.fixture_parent) if getattr(args,'fixture_parent',None) else initialization.default_source()
            if getattr(args,'fixture_parent',None) and not p.testing:raise ValueError('Fixture source forbidden in real run')
            pilot,pc=prepare('mixed_pilot',source,'pilot');result=fit(pilot,pc);candidates=reviews(pilot,pc,result);eligible=[e for e in candidates if e['training_retention_gate']['passed']]
            if not eligible:
                state.update(state='needs_course_review',active_trial=None,reason='Mixed pilot did not retain prior source facts/conditions; no full phase started',case_teaching_was_included_in_pilot=True)
                d.write(root/'review_nominee.json',{'adapter':None,'scope':'pilot_retention_not_passed_old_weights_preserved','professional_approval':False});report(root,state);return 0
            best=max(eligible,key=lambda e:(rank(e['micro_checks']),-e['steps']));pilot_adapter=Path(best['adapter']);d.write(root/'pilot_selection.json',{'adapter':str(pilot_adapter),'steps':best['steps'],'scope':'research_learning_retention_not_professional_approval'})
            full,fc=prepare('full_course',pilot_adapter,'full');micro=runtime.choose_development(pack,fc['development_generation_rows']);sha=d.digest((pilot_adapter/'adapter_model.safetensors').read_bytes())
            tag='epoch_'+str(int(best['epoch'])).zfill(2)
            copy_answers(pilot/'generations'/(tag+'_validation.jsonl'),full/'generations/validation_parent.jsonl',micro,adapter_sha=sha,scope='same_token_budget_same_sealed_pilot_adapter',max_new_tokens=fc['eval_max_new_tokens'])
            copy_answers(pilot/'generations'/(tag+'_actual_rag.jsonl'),full/'generations/parent_actual_rag.jsonl',runtime.prepared_prompts(full,'rag'),adapter_sha=sha,scope='same_actual_rag_packet_and_sealed_pilot_adapter',max_new_tokens=fc['eval_max_new_tokens'])
            result=fit(full,fc);entries=reviews(full,fc,result);passed=[e for e in entries if e['training_retention_gate']['passed']]
            if passed:best=max(passed,key=lambda e:(rank(e['micro_checks']),-e['steps']));chosen=Path(best['adapter']);chosen_job=full;chosen_c=fc;scope='full_research_candidate_open_case_review_required'
            else:chosen=pilot_adapter;chosen_job=pilot;chosen_c=pc;scope='full_retention_failed_accepted_pilot_retained'
            # Complete long-form checks only once, on the selected research candidate.
            final_gold=runtime.prepared_prompts(chosen_job,'final');besttag='epoch_'+str(int(best['epoch'])).zfill(2)
            copy_answers(chosen_job/'generations'/(besttag+'_validation.jsonl'),chosen_job/'generations/final_review.jsonl',final_gold,adapter_sha=d.digest((chosen/'adapter_model.safetensors').read_bytes()),scope='same_candidate_same_budget_micro_subset',max_new_tokens=chosen_c['eval_max_new_tokens'])
            finalrows=evaluate(chosen_job,chosen_c,chosen,'final_review','final');legacygold=runtime.prepared_prompts(chosen_job,'legacy');legacyrows=evaluate(chosen_job,chosen_c,chosen,'final_legacy_review','legacy',1024)
            # Old baseline may be reused only if its exact uploaded bytes, model,
            # source prompts and original 1024-token budget were sealed in release.
            reused=False;entry=d.read(runtime.RELEASE).get('reusable_v55_baseline')
            if entry and not p.testing:
                cache=runtime.ROOT/entry['path']
                if cache.is_file() and d.digest(cache.read_bytes())==entry['sha256']:
                    copy_answers(cache,chosen_job/'generations/parent_legacy_review.jsonl',legacygold,adapter_sha=entry['adapter_sha256'],scope='exact_reviewed_prior_observation_not_training_data',max_new_tokens=1024);reused=True
            before=evaluate(chosen_job,chosen_c,source,'parent_legacy_review','legacy',1024)
            from wengua_course_v55 import checks as old
            final={'new_final_checks':checks.score(finalrows,final_gold),'old_v55_rubric_candidate':old.score(legacyrows,legacygold),'old_v55_rubric_original_parent':old.score(before,legacygold),'prior_baseline_reused':reused,'open_cases_professionally_approved':False}
            d.write(root/'final_review.json',final)
            d.write(root/'review_nominee.json',{'adapter':str(chosen),'scope':scope,'factual_readiness':best['factual_readiness'],'professional_approval':False,'production_enabled':False})
            state.update(state='complete_review_required',active_trial=None,all_planned_groups_finished=True,selected_for_review=str(chosen),final_review=final);report(root,state);return 0
    except BaseException as error:
        state.update(state='paused' if isinstance(error,(KeyboardInterrupt,TimeoutError)) else 'failed',error_type=type(error).__name__,error=str(error),outputs_preserved=True);report(root,state);raise
    finally:report(root,state)


def restore_all(key_file=None):
    from wengua_corpus.__main__ import restore
    from wengua_course_v53 import runtime as v53
    from wengua_course_v54 import runtime as v54
    from wengua_course_v55 import runtime as v55
    from wengua_course_v56 import runtime as v56
    value=key_file.read_text(encoding='utf-8-sig').strip() if key_file else input('粘贴此前教材64位口令（输入可见，只需本次一次）：').strip()
    if not re.fullmatch('[0-9a-fA-F]{64}',value):raise ValueError('Expected the existing 64-character delivery key')
    for version in [v53,v54,v55,v56,runtime]:
        release=d.read(version.RELEASE)
        if all((version.DATA/n).is_file() and d.digest((version.DATA/n).read_bytes())==sha for n,sha in release['files'].items()):continue
        restore({'manifest':release['delivery_manifest'],'destination':version.DATA.relative_to(runtime.ROOT).as_posix(),'files':release['files']},bytes.fromhex(value))
    runtime.verify_data();print('V5.7课程、逐条来源合同、审阅Excel和数据库已恢复并校验。')


def main(argv=None):
    p=ArgumentParser(description=__doc__);p.add_argument('command',choices=['restore','check','run','export','search','ask']);p.add_argument('--output',type=Path,default=runtime.ROOT/'local/course_v57/run1');p.add_argument('--app-root',type=Path);p.add_argument('--hours',type=float,default=24);p.add_argument('--resume',action='store_true');p.add_argument('--resume-from-checkpoint',type=Path);p.add_argument('--key-file',type=Path);p.add_argument('--archive',type=Path);p.add_argument('--question');p.add_argument('--domain');p.add_argument('--time');p.add_argument('--zone');p.add_argument('--chart',type=Path);a=p.parse_args(argv)
    if a.command=='restore':restore_all(a.key_file);return 0
    if a.command=='export':
        from wengua_course.runner import export_review
        print(export_review(a.output,a.archive or Path.home()/'Downloads/Wengua_V57_review.zip'));return 0
    runtime.configure(a.app_root)
    if a.command=='check':print(d.dumps(runtime.verify_data()['summary']));return 0
    if a.command=='search':
        from . import corpus
        if not a.question:p.error('--question required')
        print(d.dumps(corpus.retrieve(a.question,domain=a.domain)));return 0
    if a.command=='ask':
        if not a.question:p.error('--question required')
        nominee=d.read(a.output/'review_nominee.json')
        if not nominee.get('adapter'):raise ValueError('Pilot retention failed; inspect the saved answers first')
        adapter=Path(nominee['adapter']);initialization.inspect(adapter);job=adapter.parent.parent.parent;c=d.read(job/'job_contract.json')['config']
        from transformers import AutoTokenizer
        from . import corpus
        tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision'],local_files_only=True,trust_remote_code=False);packet=corpus.prompt(a.question,tok,domain=a.domain,max_context=4096,max_new_tokens=768,time_text=a.time,zone=a.zone,chart_file=a.chart)
        if packet['state']!='ready':print('当前检索到的原文不足以回答。');return 2
        target=a.output/'rag_prompts'/str(time.time_ns());target=target.with_suffix('.json');d.write(target,packet)
        from .process import launch
        result=launch(runtime.profile(),job,c,time.time()+3600,'ask',best={'adapter':str(adapter),'prompt':str(target)})
        print(result['answer'] if result['checks']['status']=='citations_checked_semantics_unverified' else '回答未通过来源或盘面检查，原答已保留供审阅。');print('研究回答与来源检查：'+result['trace_path']);return 0
    if not 0<a.hours<=168:p.error('hours must be in (0,168]')
    if a.resume_from_checkpoint and not a.resume_from_checkpoint.resolve().is_relative_to(a.output.resolve()):p.error('Checkpoint must belong to this V57 run')
    os.environ.update(PYTORCH_CUDA_ALLOC_CONF=runtime.ALLOCATOR,PYTORCH_ALLOC_CONF=runtime.ALLOCATOR,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    return run(a)
