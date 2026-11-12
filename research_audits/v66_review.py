"""Review the completed V66 matched-loss experiment; never an SFT/RAG source.

All annotations are bounded observations on exposed tasks, not event accuracy.
Rebuild actual training tensors and gold-prefix target masks with the real
Qwen tokenizer; keep every raw answer and byte receipt unchanged.
"""
from pathlib import Path
from collections import Counter
import argparse,gzip,json,tempfile,zipfile
from wengua_course import data as d
from wengua_course_v66 import runtime,evaluation
from .v55_review import inspect_archive,read_rows
from .v59_review import memory_summary
from .v60_review import verify_reply
from .v64_review import verified_notes
from .v63_checkpoint_review import bound_note
from .v65_learning_review import verify_token_row,close,mean
from .v65_token_diagnostics import encoded_target

UPLOAD_SHA='5a91edaf4432f75a97759b314a9091bc2adcc12e6aa6015b720438a83cbfb6d5'
TAGS={'parent':'baseline','ordinary_ce':'ordinary_ce','decision_ce':'decision_ce'}
PREDICATES={'生','克','受','所生','所克'}


def matched_configs(configs):
    values=[{k:v for k,v in configs[label].items() if k!='decision_loss_factor'} for label in ['ordinary_ce','decision_ce']]
    if values[0]!=values[1] or configs['ordinary_ce']['decision_loss_factor']!=1 or configs['decision_ce']['decision_loss_factor']!=4:
        raise ValueError('Training comparison differs beyond decision weighting')


def verify_completion(z,trial,tag,pack_sha,c):
    stem=f'trials/{trial}/generations/{tag}';receipt_raw=z.read(stem+'.receipt.json');receipt=json.loads(receipt_raw)
    complete=json.loads(z.read(stem+'.complete.json'))
    if (receipt['pack_sha256']!=pack_sha or receipt['config']!=c or receipt['history_source']!='model_own_previous_answers'
        or receipt['professional_approved'] or complete['contract_sha256']!=d.digest(receipt_raw)
        or complete['answer_file_sha256']!=d.digest(z.read(stem+'.jsonl'))
        or complete['token_file_sha256']!=d.digest(z.read(stem+'.tokens.jsonl'))
        or complete['answers']!=103 or complete['critical_token_targets']!=38 or complete['adapter_files']!=receipt['adapter_files']):
        raise ValueError('Completed answer/token/source receipt differs')
    return receipt,complete


def verify_critical(row,target,encoded,tokenizer):
    verify_token_row(row,target,encoded)
    spans=target['provenance'].get('decision_spans',[]);start=encoded['answer_start_character'];mask=[];miss=[];values=[]
    for i,t in enumerate(row['tokens']):
        a,b=t['character_offsets'];selected=any(a<start+s['end'] and b>start+s['start'] for s in spans);mask.append(selected)
        if t['target_text']!=tokenizer.decode([t['target_id']]):raise ValueError('Target token text changed')
        if selected:
            values.append(t['nll'])
            if not t['gold_prefix_top1_equals_target']:miss.append(i)
    if (row['critical_spans']!=spans or [t['in_v66_critical_span'] for t in row['tokens']]!=mask
        or row['critical_tokens']!=len(values) or row['critical_gold_prefix_miss_indices']!=miss
        or not row['conditional_score_is_not_free_answer_accuracy']):raise ValueError('Critical target character/token positions differ')
    close(row['critical_mean_nll'],mean(values))


def token_summary(rows,gold,encoded):
    result={}
    for style in ['original','paraphrase','all']:
        selected=[r for r in rows if style=='all' or r['id'].endswith(':'+style)]
        flat=[t for r in selected for t in r['tokens']];crit=[t for t in flat if t['in_v66_critical_span']];predicates=[]
        for r in selected:
            start=encoded[r['id']]['answer_start_character'];spans=[s for s in r['critical_spans'] if s['text'] in PREDICATES]
            predicates += [t for t in r['tokens'] if any(t['character_offsets'][0]<start+s['end'] and t['character_offsets'][1]>start+s['start'] for s in spans)]
        result[style]={'targets':len(selected),'supervised_tokens':len(flat),'token_mean_nll':mean([t['nll'] for t in flat]),
            'gold_prefix_top1_misses':sum(not t['gold_prefix_top1_equals_target'] for t in flat),'critical_tokens':len(crit),
            'critical_mean_nll':mean([t['nll'] for t in crit]),'critical_gold_prefix_misses':sum(not t['gold_prefix_top1_equals_target'] for t in crit),
            'direction_predicate_tokens_excluding_bihe_and_verdict_style':len(predicates),
            'direction_predicate_gold_prefix_misses':sum(not t['gold_prefix_top1_equals_target'] for t in predicates),
            'direction_predicate_mean_nll':mean([t['nll'] for t in predicates]),'is_free_answer_accuracy':False}
    return result


def source_inventory(pack):
    rows=[r for r in pack['records'] if r['pilot_selected']];mar=[r for r in rows if r['skill']=='marriage']
    return {'preserved_train_cores':sum(r.get('split')=='train' for r in pack['records']),'short_training_cores':len(rows),'source_units':len(pack['units']),'original_case_registry':len(pack['case_splits']),
        'marriage_cores':len(mar),'marriage_views':sum(sum(m['role']=='assistant' for m in r['messages']) for r in mar),
        'marriage_first_condition':dict(Counter(r['provenance']['lesson_contract']['palace_relation'] for r in mar)),
        'marriage_second_condition':dict(Counter(r['provenance']['lesson_contract']['state'] for r in mar)),
        'marriage_multiturn_cores':sum(len(r['messages'])>3 for r in mar),'first_condition_always_known_satisfied':all(r['provenance']['lesson_contract']['palace_relation'] in {'相生','比和'} for r in mar),
        'no_first_condition_false_or_unknown_contrast':all(r['provenance']['lesson_contract']['palace_relation'] in {'相生','比和'} for r in mar),'coverage_gap_is_not_proven_unique_failure_cause':True,
        'non_book_teachers_or_new_books_used':False,'full_194_case_training_claimed':False}


def observations(models,remarks):
    groups={'palace_original':list(range(0,12,2)),'palace_paraphrase':list(range(1,12,2)),
            'verify_original':list(range(12,22,2)),'verify_paraphrase':list(range(13,22,2)),
            'marriage_static':list(range(22,28)),'loan':list(range(28,32)),'identity':list(range(32,38)),
            'attributes':list(range(38,44)),'given_elements':list(range(44,50)),
            'star_composition':list(range(50,66)),'flat_latest':list(range(66,70)),
            'rag_context':list(range(70,88)),'ordinary':list(range(88,92)),'own_history':list(range(92,103))}
    out={}
    for tag,rows in models.items():
        n=[remarks[tag][r['id']] for r in rows]
        out[tag]={'groups':{name:{'questions':len(indices),'whole_answer_supported':sum(n[i]['whole_answer_supported'] for i in indices),
            'annotation_verdicts':dict(Counter(n[i]['verdict'] for i in indices))} for name,indices in groups.items()},
            'palace_directions':sum(n[i].get('directed_relation_supported',False) for i in range(12)),
            'verification_verdict_and_direction':sum(n[i].get('verification_verdict_supported',False) and n[i].get('directed_relation_supported',False) for i in range(12,22)),
            'marriage_branch_verdict':sum(n[i].get('branch_verdict_supported',False) for i in range(22,28)),
            'marriage_branch_and_reason':sum(n[i].get('branch_verdict_supported',False) and n[i].get('condition_reason_supported',False) for i in range(22,28)),
            'bank_main_result':sum(n[i].get('bank_main_result_supported',False) for i in range(28,32)),
            'identity_main_field':sum(n[i].get('asked_primary_role_field_supported',False) for i in range(32,38)),
            'star_positions':sum(n[i].get('positions_supported',False) for i in range(50,66)),
            'star_palace_directions':sum(n[i].get('directed_palace_relation_supported',False) for i in range(50,66)),
            'star_intrinsic_attribute_errors':sum(n[i].get('intrinsic_star_error',False) for i in range(50,66)),
            'own_history_main_goal':sum(n[i].get('dialogue_main_goal_supported',False) for i in range(92,103)),
            'own_history_explicit_condition_contradictions':sum(n[i].get('explicit_condition_fact_contradiction',False) for i in range(92,103)),
            'condition_wording_ambiguity_ids':[rows[i]['id'] for i in range(103) if n[i].get('condition_wording_ambiguous')],
            'bounded_known_question_observations_not_case_accuracy':True}
    return out


def audit(archive,output,notes=None,tokenizer_root=None):
    archive=Path(archive)
    if d.digest(archive.read_bytes())!=UPLOAD_SHA:raise ValueError('Exact uploaded V66 result required')
    runtime.configure();pack=runtime.verify_data();gold=evaluation.build(runtime.old.verify_data());remarks=d.read(notes) if notes else {}
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(tokenizer_root) if tokenizer_root else str(runtime.ROOT/'local/course_v66/run1/trials/ordinary_ce/prepared/tokenizer'),local_files_only=True,trust_remote_code=False)
    targets=[r for r in gold['records'] if r.get('teacher_forcing')];encoded={r['id']:encoded_target(tok,r,4096) for r in targets}
    models={};scores={};configs={};stages={};workers=[];details=[];numerics=[]
    with zipfile.ZipFile(archive) as z:
        inspect_archive(z)
        if z.testzip() is not None:raise ValueError('Archive CRC failed')
        names=z.namelist();run=json.loads(z.read('sweep_contract.json'));state=json.loads(z.read('run_status.json'));pack_sha=d.digest(d.dumps(gold))
        if (len(names)!=121 or run['schema']!=runtime.SCHEMA or run['code_sha256']!=runtime.hashes() or len(run['code_sha256'])!=308
            or run['dataset_release_sha256']!=d.digest(runtime.RELEASE.read_bytes()) or run['variants']!={'ordinary_ce':1,'decision_ce':4}
            or run['evaluation_pack_sha256']!=pack_sha or not run['same_training_tensors_and_order'] or run['old_optimizer_inherited']
            or run['automatic_long_training'] or run['automatic_professional_promotion'] or run['original_test_used']
            or state['state']!='complete_review_required' or state['production_enabled'] or state['curriculum']!=pack['summary']
            or state['selection']!='none_until_whole_answer_review' or state['additional_long_training_started']):raise ValueError('Actual frozen plan/course/completion differs')
        execution=json.loads(z.read('process_execution.json'))
        if (execution['schema']!='wengua_source_course_v66_process' or execution['code_sha256']!=run['code_sha256'] or execution['software_fixture']
            or not execution['process_per_unit'] or execution['loss_runtime_route']!='v66_normalized_decision_ce_v1'
            or execution['memory_loss']!='normalized_decision_ce_v1'):raise ValueError('Actual isolated execution contract differs')
        parent_raw=z.read('initialization/parent_provenance.json');parent=json.loads(parent_raw)
        if (d.digest(parent_raw)!=run['parent_provenance_sha256'] or parent['optimizer_inherited'] or parent['schema']!=runtime.old.SCHEMA
            or parent['files']['adapter_model.safetensors']!=d.read(runtime.RELEASE)['parent']['sha256']
            or not parent['source_path'].replace('\\','/').endswith(d.read(runtime.RELEASE)['parent']['relative_path'])):raise ValueError('Different learned V65 E2 source')
        for label,trial in TAGS.items():
            rawjob=z.read(f'trials/{trial}/job_contract.json');job=json.loads(rawjob);c=job['config'];configs[trial]=c
            if (job['schema']!=runtime.SCHEMA or job['code_sha256']!=run['code_sha256'] or job['dataset_release_sha256']!=run['dataset_release_sha256']
                or c['book_parent_provenance_sha256']!=d.digest(parent_raw) or json.loads(z.read(f'trials/{trial}/evaluation_pack.json'))!=gold):raise ValueError('Job source or actual questions differ')
            if label!='parent':
                runtime.validate(c);stage_raw=z.read(f'trials/{trial}/cycle_01/stage_contract.json');sc=json.loads(stage_raw)
                stage=json.loads(z.read(f'trials/{trial}/cycle_01/stage_status.json'));stages[trial]=stage
                if (sc['config']!=c or c!=state['trials'][trial]['config'] or sc['job_contract_sha256']!=d.digest(rawjob) or sc['planned_updates']!=77
                    or sc['parent_provenance_sha256']!=d.digest(parent_raw) or not sc['new_phase_optimizer'] or sc['stack']!=d.read(runtime.RELEASE)['original_stack']
                    or stage!=state['trials'][trial]['fit'] or stage['steps']!=77 or stage['planned_steps']!=77 or stage['state']!='complete'
                    or stage['resume_checkpoint'] is not None or not stage['parent_weights_reused'] or stage['fresh_official_base']
                    or len(stage['epochs'])!=1 or stage['epochs'][0]['steps']!=77 or stage['epochs'][0]['epoch']!=1.):raise ValueError('Actual matched one-epoch fit differs')
            receipt,complete=verify_completion(z,trial,label,pack_sha,c)
            if label=='parent' and receipt['adapter_files']!=parent['files']:raise ValueError('Baseline did not use the same learned parent')
            rows=read_rows(z,f'trials/{trial}/generations/{label}.jsonl');evaluation.validate_cache(rows,gold,label);actual_gold=evaluation.sequence(gold,rows)
            if len(rows)!=103 or len(actual_gold)!=103:raise ValueError('Missing actual free answers or own-history turns')
            if remarks:verified_notes(remarks,label,rows)
            for i,(row,target) in enumerate(zip(rows,actual_gold,strict=True)):
                if row['components']!=target['components'] or row['source_ids']!=target['source_ids'] or row['task']!=target['task'] or row['group']!=target['group']:raise ValueError('Answer source facts differ')
                details.append({'tag':label,'index':i,'row':row,'gold':target|{'skill':target['task']},'frozen_check':None,'annotation':bound_note(remarks,label,row)})
            models[label]=rows;scores[label]=read_rows(z,f'trials/{trial}/generations/{label}.tokens.jsonl')
            if len(scores[label])!=38:raise ValueError('Missing gold-prefix target records')
            for row,target in zip(scores[label],targets,strict=True):verify_critical(row,target,encoded[target['id']],tok)
            actual=state['baseline'] if label=='parent' else state['trials'][trial]['evaluation']
            if actual['state']!='complete' or actual['completed']!=103 or actual['requested']!=103 or actual['critical_token_targets']!=38 or actual['answers_sha256']!=complete['answer_file_sha256']:
                raise ValueError('Final evaluation state differs')
        matched_configs(configs)
        if configs['baseline']!=configs['ordinary_ce']:raise ValueError('Parent/ordinary input generation policy differs')
        for name in sorted(n for n in names if n.endswith('.result.json')):
            raw=z.read(name.replace('.result.json','.json'));request=json.loads(raw);reply=json.loads(z.read(name));verify_reply(reply,request,raw)
            trial=name.split('/')[1]
            if request['config']!=configs[trial] or request['software_fixture'] or not reply['ok'] or reply['software_fixture'] or reply['schema']!=execution['schema']:
                raise ValueError('Actual worker request differs')
            probe=json.loads(z.read(name.replace('.result.json','.attention_probe.json')))
            if probe['state']!='ready' or probe['selected_backend']!='EFFICIENT_ATTENTION':raise ValueError('Actual selected CUDA attention probe differs')
            item={'trial':trial,'operation':request['operation'],'probe':request['probe'],'pid':reply['pid'],'seconds':reply['elapsed_seconds'],'request_sha256':d.digest(raw)}
            mem=name.replace('.result.json','.memory.json');item['memory']=memory_summary(json.loads(z.read(mem)))
            if request['operation']=='fit':
                stage_name='smoke' if request['probe'] else 'cycle_01'
                actual_stage=json.loads(z.read(f'trials/{trial}/{stage_name}/stage_status.json'))
                actual_contract=json.loads(z.read(f'trials/{trial}/{stage_name}/stage_contract.json'))
                if reply['result']!=actual_stage or actual_stage!=state['trials'][trial]['probe' if request['probe'] else 'fit'] or actual_contract['config']!=configs[trial]:
                    raise ValueError('Worker/archived stage/final status differ')
                start=json.loads(z.read(name.replace('.result.json','.training_start.json')));loss=json.loads(z.read(name.replace('.result.json','.loss_path.json')))
                if (start['step'] or start['optimizer_steps'] or reply['result']['steps']!=(2 if request['probe'] else 77)
                    or loss['route']!='v66_normalized_decision_ce_v1' or loss['training_factor']!=configs[trial]['decision_loss_factor'] or loss['evaluation_factor']!=1
                    or not 0<loss['max_projected_positions']<=128 or loss['train_loss_calls']!=(16 if request['probe'] else 615)
                    or loss['weighted_training_calls']!=(loss['train_loss_calls'] if trial=='decision_ce' else 0)):raise ValueError('Actual weighted training/evaluation route differs')
                item.update(training_start=start,loss_path=loss)
                attention=json.loads(z.read(name.replace('.result.json','.attention_path.json')))
                if attention['software_fixture'] or not attention['cuda_calls'] or attention['math_fallback_allowed']:raise ValueError('Fused CUDA path missing')
            workers.append(item)
        for name in (n for n in names if '/numerical_checks/' in n):
            value=json.loads(z.read(name));numerics.append(value)
            if (not value['actual_cuda'] or value['software_fixture_only'] or value['weights_updated_by_check'] or value['training_step']
                or not value['nonzero_lora_B_present'] or value['adapter_max_logit_delta']<=0 or len(value['cached_vs_full_prefix'])!=2
                or any(r['relative_l2_error']>.03 for r in value['cached_vs_full_prefix'])):raise ValueError('Actual inherited adapter activation/parity differs')
        if len(workers)!=7 or len(numerics)!=4:raise ValueError('Missing actual workers/numerical checks')
        smoke=[r for n in names if n.endswith('/smoke_generation.jsonl') for r in read_rows(z,n)]
        if len(smoke)!=2:raise ValueError('Pressure generation count differs')
    # Recreate actual text/tensor order rather than assuming hidden domain fields
    # or inferring supervision from the number of registered source cores.
    with tempfile.TemporaryDirectory(prefix='v66_review_tensor_') as tmp:
        rebuilt=runtime.prepare(Path(tmp)/'job',configs['ordinary_ce'],tokenizer=tok,pack=pack,testing=True)
        if rebuilt['training_rows']!=615 or rebuilt['training_tensor_sha256']!=state['training_tensor_sha256']:raise ValueError('Rebuilt actual training tensor/order digest differs')
    summary={'schema':'wengua_actual_v66_review_v1','upload_sha256':UPLOAD_SHA,'archive_members':121,'frozen_runtime_files':308,
        'formal_updates':154,'pressure_updates':4,'free_answers':309,'smoke_answers':2,'teacher_forced_targets':114,
        'workers':workers,'stages':stages,'numerical_checks':numerics,'source_curriculum':source_inventory(pack),
        'recreated_training':{k:rebuilt[k] for k in ['training_rows','training_tensor_sha256','decision_tokens','prompt_weighted_tokens','max_actual_train_tokens','training_generation_protocol']},
        'token_diagnostics':{tag:token_summary(rows,gold,encoded) for tag,rows in scores.items()},
        'bounded_reading_observations':observations(models,remarks) if remarks else None,'notes_bound_to_answers':sum(bool(x['annotation']) for x in details),
        'shape':{tag:{'empty':sum(r['empty'] for r in rows),'output_limit':sum(r['reached_token_limit'] for r in rows),
            'inline_source_citations':sum(bool(__import__('re').search(r'\[S\d+\]',r['answer'])) for r in rows[70:88])} for tag,rows in models.items()},
        'legacy_teacher_segment_names_not_used_as_semantic_labels':True,'all_token_ids_offsets_masks_and_means_verified':True,
        'unuploaded_weight_tensors_independently_rehashed':False,'actual_gpu_logits_independently_recomputed':False,
        'original_test_used':False,'new_blind_case_accuracy_claimed':False,'professional_promotion':False,'fresh_rag_retrieval_retested':False,
        'responses_allowed_as_sft_or_rag_teachers':False,'per_step_training_curve_uploaded':False}
    output=Path(output);output.mkdir(parents=True,exist_ok=True);d.write(output/'summary.json',summary)
    with gzip.open(output/'answer_observations.jsonl.gz','wt',encoding='utf8') as f:
        for row in details:f.write(d.dumps(row)+'\n')
    return summary,details


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path,nargs='?');p.add_argument('--notes',type=Path);p.add_argument('--tokenizer-root',type=Path)
    p.add_argument('--output',type=Path,default=runtime.ROOT/'local/model_reviews/v66');p.add_argument('--workbook',action='store_true');p.add_argument('--restore',action='store_true');a=p.parse_args(argv)
    if a.restore:
        import re
        from wengua_corpus.__main__ import restore
        value=input('粘贴此前教材的64位口令（输入可见）：').strip()
        if not re.fullmatch('[0-9a-fA-F]{64}',value):raise ValueError('Expected existing textbook key')
        entry=d.read(runtime.ROOT/'docs/model_reviews/20261112_v66/private_review_catalog.json');print(restore(entry,bytes.fromhex(value))/'review/Wengua_V66_两路训练逐答审阅.xlsx');return 0
    if a.archive is None:p.error('archive or --restore required')
    summary,details=audit(a.archive,a.output,a.notes,a.tokenizer_root)
    if a.workbook:
        from .v61_review import workbook
        workbook(a.output/'Wengua_V66_两路训练逐答审阅.xlsx',details)
    print(d.dumps({'free_answers':len(details),'forced_targets':114,'bound_notes':summary['notes_bound_to_answers']}));return 0


if __name__=='__main__':raise SystemExit(main())
