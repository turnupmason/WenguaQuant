"""Bounded, restartable, same-parent trials. No failed candidate promotion."""
from pathlib import Path
import time
from wengua_course import data as d
from . import runtime,initialization,checks
from .runner import config,report,read_rows,learning_gate,copy_answers,rank

RATES=(1e-6,3e-6,1e-5)
TERMINAL=('complete_review_required','needs_course_review')


def retained(current,before,rag,rag_before,*,require_improvement=True):
    decision=learning_gate(current,before,rag,rag_before)
    if not require_improvement:
        decision['failed']=[x for x in decision['failed'] if x!='no_new_complete_fact_or_condition']
        decision['passed']=not decision['failed']
    return decision


def run(args):
    runtime.configure(args.app_root);pack=runtime.verify_data();p=runtime.profile();b=p.book
    root=b.private_output(args.output);root.mkdir(parents=True,exist_ok=True)
    with b.keep_awake(),b.job_lock(root/'run.lock'):
        return run_locked(args,pack,p,root)


def run_locked(args,pack,p,root):
    b=p.book
    contract={'schema':runtime.SCHEMA,'code_sha256':runtime.hashes(),'dataset_release_sha256':d.digest(runtime.RELEASE.read_bytes()),
              'rates':list(RATES),'pilot_epochs':1,'pilot_cores':768,'full_epochs':1,
              'same_parent_each_trial':True,'retention':'item_fact_condition_citation_no_tradeoffs',
              'official_control':'same_fixed_subset_no_training','original_test_used':False}
    contract['final_max_new_tokens']=runtime.FINAL_TOKENS
    path=root/'sweep_contract.json'
    if path.exists() and d.read(path)!=contract:raise ValueError('V60 run contract changed; prior outputs preserved')
    if not path.exists():d.write(path,contract)
    state=d.read(root/'run_status.json') if (root/'run_status.json').exists() else {'state':'created','trials':{},'production_enabled':False,'professional_approval':False}
    if state['state'] in TERMINAL:report(root,state);return 0
    deadline=time.time()+3600*args.hours
    from .process import launch
    source=initialization.default_source();source_sha=d.digest((source/'adapter_model.safetensors').read_bytes())

    def prepare(label,parent,phase,rate):
        snapshot=initialization.snapshot(root/'parents'/label,parent,testing=p.testing)
        c=config(snapshot,root,phase);c['learning_rate']=rate
        job=p.engine.open_job(root/'trials'/label,c,resume=True);runtime.prepare(job,c)
        proof=root/'inheritance_proofs'/(label+'.json');proof.parent.mkdir(exist_ok=True);proof.write_bytes((snapshot.parent/'parent_provenance.json').read_bytes())
        return job,c

    def evaluate(job,c,adapter,tag,stage,official=False):
        best={'adapter':str(adapter) if adapter else None,'tag':tag,'eval_stage':stage,'official_base':official}
        if stage=='final' and not p.testing:best['max_new_tokens']=runtime.FINAL_TOKENS
        result=launch(p,job,c,deadline,'evaluate',best=best)
        if result['state']!='complete':raise TimeoutError('Evaluation paused; answers and receipts retained')
        return checks.score(read_rows(job/'generations'/(tag+'.jsonl')),runtime.prepared_prompts(job,stage))

    def fit(job,c):
        state.update(state='training_'+c['course_phase'],active_trial=job.name);report(root,state)
        result=launch(p,job,c,deadline,'fit',probe=True)
        if result['state']!='complete':raise TimeoutError('Pressure test paused')
        cp=getattr(args,'resume_from_checkpoint',None)
        if cp and not cp.resolve().is_relative_to(job.resolve()):cp=None
        result=launch(p,job,c,deadline,'fit',checkpoint=cp)
        if result['state']!='complete':raise TimeoutError('Training paused; complete optimizer checkpoints preserved')
        return result

    def score_candidate(job,c,result,before,ragbefore):
        entry=result['epochs'][-1];tag='epoch_'+str(entry['cycle']).zfill(2)
        cur=checks.score(read_rows(job/'generations'/(tag+'_validation.jsonl')),runtime.choose_development(pack))
        rag=evaluate(job,c,Path(entry['adapter']),tag+'_actual_rag','rag')
        gate=retained(cur,before,rag,ragbefore)
        row={**entry,'micro_checks':cur,'rag_checks':rag,'training_retention_gate':gate,'job':str(job),'config':c}
        d.write(job/'paired_assessment.json',row);state['trials'][job.name]=row;report(root,state)
        return row

    try:
        b.progress('v60_plan',pilot_rates=list(RATES),pilot_updates=pack['summary']['pilot_updates'],pilot_epochs=1,
                   domains=data_domains(pack),official_control=pack['summary']['official_base_control'],full_epochs=1)
        baseline,bc=prepare('retained_baseline',source,'pilot',3e-6)
        state.update(state='comparing_base_and_parent',active_trial=baseline.name);report(root,state)
        base_control=evaluate(baseline,bc,None,'official_control','control',official=True)
        parent_control=evaluate(baseline,bc,source,'parent_control','control')
        before=evaluate(baseline,bc,source,'parent_micro','application')
        ragbefore=evaluate(baseline,bc,source,'parent_rag','rag')
        d.write(root/'baseline_comparison.json',{'official_base':base_control,'retained_adapter':parent_control,
                'same_inputs':True,'parent_adapter_sha256':source_sha,'scope':'fixed_development_diagnostic_not_blind_prediction'})
        candidates=[]
        for rate in RATES:
            label='lr_'+format(rate,'.0e').replace('-','m').replace('+','p')
            job,c=prepare(label,source,'pilot',rate)
            copy_answers(baseline/'generations/parent_micro.jsonl',job/'generations/validation_parent.jsonl',runtime.choose_development(pack),
                         adapter_sha=source_sha,scope='identical_fixed_parent_same_budget',max_new_tokens=c['eval_max_new_tokens'])
            candidate=score_candidate(job,c,fit(job,c),before,ragbefore);candidates.append(candidate)
        eligible=[r for r in candidates if r['training_retention_gate']['passed']]
        chosen=None;scope='no_measured_improvement_without_regression'
        if eligible:
            chosen=max(eligible,key=lambda r:(len(r['training_retention_gate']['new_complete_items']),rank(r['micro_checks'])))
            pilot=chosen;adapter=Path(pilot['adapter']);sha=d.digest((adapter/'adapter_model.safetensors').read_bytes())
            job,c=prepare('full_course',adapter,'full',pilot['config']['learning_rate'])
            oldjob=Path(pilot['job']);tag='epoch_'+str(pilot['cycle']).zfill(2)
            copy_answers(oldjob/'generations'/(tag+'_validation.jsonl'),job/'generations/validation_parent.jsonl',runtime.choose_development(pack),
                         adapter_sha=sha,scope='accepted_pilot_exact_questions',max_new_tokens=c['eval_max_new_tokens'])
            full=score_candidate(job,c,fit(job,c),before,ragbefore)
            protection=retained(full['micro_checks'],pilot['micro_checks'],full['rag_checks'],pilot['rag_checks'],require_improvement=False)
            full['accepted_pilot_retention']=protection;d.write(job/'paired_assessment.json',full)
            if full['training_retention_gate']['passed'] and protection['passed']:chosen=full;scope='full_candidate_keeps_parent_and_pilot_measured_facts'
            else:chosen=pilot;scope='full_regressed_accepted_pilot_retained'
        research=chosen or min(candidates,key=lambda r:(len(r['training_retention_gate']['failed']),-len(r['training_retention_gate']['new_complete_items'])))
        job=Path(research['job']);c=research['config'];adapter=Path(research['adapter'])
        state.update(state='final_source_review',active_trial=job.name);report(root,state)
        final=evaluate(job,c,adapter,'final_review','final')
        d.write(root/'final_review.json',{'checks':final,'adapter':str(adapter),'open_author_semantics_approved':False,'not_training_input':True})
        retained_adapter=Path(chosen['adapter']) if chosen else source
        d.write(root/'retained_best.json',{'adapter':str(retained_adapter),'adapter_sha256':d.digest((retained_adapter/'adapter_model.safetensors').read_bytes()),
                 'previous_parent':str(source),'previous_parent_sha256':source_sha,'research_adapter':str(adapter),
                 'training_retention_passed':bool(chosen),'scope':scope,'professional_approval':False,'production_enabled':False})
        state.update(state='complete_review_required' if chosen else 'needs_course_review',active_trial=None,
                     all_three_parameter_trials_finished=True,retained_adapter=str(retained_adapter),scope=scope,
                     full_training_started=bool(eligible),final_review=final)
        report(root,state);return 0
    except BaseException as error:
        state.update(state='paused' if isinstance(error,(TimeoutError,KeyboardInterrupt)) else 'failed',
                     error_type=type(error).__name__,error=str(error),outputs_preserved=True);report(root,state);raise


def data_domains(pack):
    return sorted(k for k,v in pack['summary']['coverage'].items() if v['cases'])
