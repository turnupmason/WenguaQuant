"""Read-only review of completed V67 and the exact 198-answer recovery.

Preserve the earlier partial review. Completion is not professional approval;
input-format monitors are known-rule checks, not new independent cases.
"""
from argparse import ArgumentParser
from pathlib import Path
from collections import Counter
import gzip,json,zipfile
from wengua_course import data as d
from wengua_course_v67 import runtime,evaluation
from wengua_recovery import v67_evaluation as recovery
from . import v67_review as previous
from .v55_review import inspect_archive,read_rows
from .v64_review import verified_notes
from .v63_checkpoint_review import bound_note

UPLOAD_SHA='bf3108a2b376246bd05ee92b5dc3fe9fc3c298a2a55cc3b2aed7b7aa07069104'


def verify_original_members(old,new):
    allowed={'run_status.json','results.html','trials/object_condition_repair/generations/epoch_01.jsonl'}
    changed=[]
    for name in old.namelist():
        if name not in new.namelist():raise ValueError('Original run member missing: '+name)
        if old.read(name)!=new.read(name):changed.append(name)
    if set(changed)!=allowed:raise ValueError('Unexpected modification to original training/evaluation records')
    path='trials/object_condition_repair/generations/epoch_01.jsonl'
    if not new.read(path).startswith(old.read(path)):raise ValueError('Original 124-answer byte prefix changed')
    if new.read('evaluation_recovery/original_run_status.json')!=old.read('run_status.json'):raise ValueError('Original failed state was not preserved')
    return {'original_members':len(old.namelist()),'unchanged_original_members':len(old.namelist())-len(changed),
            'changed_members':sorted(changed),'old_parent_and_epoch1_answer_bytes_preserved':True}


def verify_seal(z,stem,c,pack_sha,tag,total=161):
    raw=z.read(stem+'.receipt.json');receipt=json.loads(raw);complete=json.loads(z.read(stem+'.complete.json'))
    if (receipt['config']!=c or receipt['pack_sha256']!=pack_sha or receipt['record_variant']!=tag or receipt['professional_approved']
        or receipt['history_source']!='model_own_previous_answers' or complete['contract_sha256']!=d.digest(raw)
        or complete['answers_sha256']!=d.digest(z.read(stem+'.jsonl')) or complete['answers']!=total
        or complete['adapter_files']!=receipt['adapter_files']):raise ValueError('Completed answer/adapter/contract byte seal differs')
    return receipt


def format_kind(row):
    c=row['components'].get('source_contract',{})
    if c.get('kind')!='typed_objects' or c['goal']!='palaces':return None
    q=row['question'][-1]['content']
    if '程序位置：' in q:return 'fact_card'
    if '下面是程序核定的九宫星位摘录。' in q:return 'raw_full_names'
    raise ValueError('Unknown actual palace comparison input representation')


def curriculum_diagnostics(course):
    train=[r for r in course['records'] if r['split']=='train'];dev=[r for r in course['records'] if r['split']=='validation']
    teachers={m['content'] for r in train for m in r['messages'] if m['role']=='assistant'}
    return {'development_cores':len(dev),'development_unique_teacher_texts':len({r['answer'] for r in dev}),
            'development_teacher_texts_also_in_training':sum(r['answer'] in teachers for r in dev),
            'star_claim_verdicts':dict(Counter('retain' if '可以保留' in r['answer'] else 'correct' for r in train if r['skill']=='star_claim')),
            'old_verify_verdicts':dict(Counter('retain' if '可以保留' in r['answer'] else 'correct' for r in train if r['skill']=='verify')),
            'explicit_current_condition_cards_in_supervised_users':sum('当前核实资料：' in m['content'] for r in train if r['skill'] in ['independent_conditions','condition_claim'] for m in r['messages'] if m['role']=='user'),
            'observed_error_does_not_imply_teacher_is_wrong':True,'development_is_known_source_and_known_board':True,
            'full_194_case_training_claimed':False,'model_answers_are_teachers':False}


def observations(models,notes):
    result=previous.observations(models,notes)
    for label,rows in models.items():
        current=result[label];n=notes[label];groups={}
        for row in rows[112:136]:
            kind=format_kind(row)
            if kind:groups.setdefault(kind,[]).append(row)
        current.update(new_palace_input_paths={k:{'questions':len(v),'whole_clear':sum(n[r['id']]['whole_answer_supported'] for r in v)} for k,v in groups.items()},
            comparison_layer_ambiguity_ids=[r['id'] for r in rows if n[r['id']].get('comparison_layer_ambiguous')],
            conditional_wording_ambiguity_ids=[r['id'] for r in rows if n[r['id']].get('condition_wording_ambiguous')],
            star_claim_clear_verdicts=sum(n[rows[i]['id']].get('claim_verdict_supported') is True for i in range(136,140)),
            old_star_clear_direction_errors=sum(n[rows[i]['id']].get('directed_palace_relation_supported') is False for i in range(50,66)),
            whole_answer_counts_are_bounded_manual_reading_not_accuracy=True)
    return result


def verify_worker(z,name,c,recipe,rows,expected_start):
    raw=z.read(name.replace('.result.json','.json'));q=json.loads(raw);reply=json.loads(z.read(name));tag=q['best']['tag'];res=reply.get('result',{})
    if (q['schema']!=recovery.SCHEMA or reply['schema']!=recovery.SCHEMA or q['operation']!='evaluate' or q['config']!=c
        or q['recipe']!=recipe or q['limit']!=24 or q['software_fixture'] or reply['software_fixture'] or not reply['ok']
        or reply['nonce']!=q['nonce'] or reply['request_sha256']!=d.digest(raw) or reply['recipe_sha256']!=d.digest(d.dumps(recipe))
        or reply['training_updates'] or type(reply['pid']) is not int or reply['pid']<=0):raise ValueError('Actual read-only recovery request/result differs')
    done=res['completed'];made=res['new_answers'];start=done-made
    prefix=b''.join(rows.splitlines(keepends=True)[:start])
    if (start!=expected_start or made!=min(24,161-start) or res['requested']!=161 or res['training_updates'] or res['professional_approved']
        or res['state']!=('complete' if done==161 else 'chunk_complete') or res['previous_prefix_sha256']!=d.digest(prefix)
        or not res['previous_bytes_preserved'] or not res['source_files_unchanged']):raise ValueError('Actual evaluation chunk order/answer prefix differs')
    mem=json.loads(z.read(name.replace('.result.json','.memory.json')));actual=[json.loads(l) for l in rows.splitlines()]
    if len(mem)!=made*2+2 or mem[0]['phase']!='before_model_load' or mem[1]['phase']!='model_loaded':raise ValueError('Missing real per-answer boundary memory evidence')
    for j,i in enumerate(range(start,done)):
        a,b=mem[2+2*j:4+2*j]
        if (a['phase']!='before_answer' or b['phase']!='answer_saved_cache_released' or a['completed']!=i or b['completed']!=i+1
            or a['answer_id']!=actual[i]['id'] or b['answer_id']!=actual[i]['id'] or any(x['pid']!=reply['pid'] or x['tag']!=tag for x in [a,b])):
            raise ValueError('Memory/heartbeat order differs from actual saved answers')
    attention=json.loads(z.read(name.replace('.result.json','.attention_path.json')))
    if attention['software_fixture'] or not attention['cuda_calls'] or attention['math_fallback_allowed'] or attention['selected_backend']!='EFFICIENT_ATTENTION':raise ValueError('Real efficient CUDA inference missing')
    allocator=json.loads(z.read(name.replace('.result.json','.allocator_policy.json')))
    if not allocator['allocator_policy_applied'] or allocator['fraction']!=.8:raise ValueError('Actual recovery allocator cap changed')
    return {'tag':tag,'start':start,'completed':done,'new_answers':made,'seconds':reply['elapsed_seconds'],'pid':reply['pid'],
            'memory_samples':len(mem),'minimum_sampled_free_gib':min(x['free_gib'] for x in mem),
            'maximum_sampled_reserved_gib':max(x['reserved_gib'] for x in mem),'reported_peak_allocated_gib':max(x['peak_allocated_gib'] for x in mem),
            'reported_peak_reserved_gib_including_load':max(x['peak_reserved_gib'] for x in mem),'maximum_allocator_ooms':max(x['allocator_ooms'] for x in mem),
            'minimum_after_answer_free_gib':min(x['free_gib'] for x in mem if x['phase']=='answer_saved_cache_released'),'training_updates':0}


def audit(archive,partial_archive,output,notes=None,partial_notes=None,tokenizer_root=None):
    archive=Path(archive);output=Path(output)
    if d.digest(archive.read_bytes())!=UPLOAD_SHA:raise ValueError('Exact completed V67 upload required')
    original,_=previous.audit(partial_archive,output/'original_partial',partial_notes,tokenizer_root)
    runtime.configure();course=runtime.verify_data();gold=evaluation.build(course,runtime.before.old.verify_data());pack_sha=d.digest(d.dumps(gold));remarks=d.read(notes) if notes else {}
    models={};details=[];workers=[];recipe=recovery.recipe()
    with zipfile.ZipFile(archive) as z,zipfile.ZipFile(partial_archive) as old:
        inspect_archive(z)
        if len(z.namelist())!=150 or z.testzip() is not None:raise ValueError('Completed archive member/CRC mismatch')
        preserved=verify_original_members(old,z);state=json.loads(z.read('run_status.json'));run=json.loads(z.read('sweep_contract.json'))
        c=state['trial']['config'];fit=state['trial']['fit'];src=json.loads(z.read('evaluation_recovery/original_sources.json'));rs=json.loads(z.read('evaluation_recovery/recovery_status.json'))
        if (state['state']!='complete_review_required' or state['production_enabled'] or state['selection']!='none_until_full_answer_retention_review'
            or state['additional_long_training_started'] or not state['old_weights_preserved'] or state['trial']['state']!='complete'
            or fit!=original['formal_fit'] or run['code_sha256']!=runtime.hashes() or run['evaluation_pack_sha256']!=pack_sha):raise ValueError('Completed original experiment/source differs')
        if (json.loads(z.read('evaluation_recovery/recovery_contract.json'))!=recipe or rs['state']!='complete' or rs['training_updates']
            or rs['completed_at_start']!={'epoch_01':124,'epoch_02':0} or state['evaluation_recovery']['recipe_sha256']!=d.digest(d.dumps(recipe))
            or state['evaluation_recovery']['training_updates'] or src['fit']!=fit or src['config']!=c or src['pack_sha256']!=pack_sha):raise ValueError('Read-only recovery source/policy differs')
        for tag,trial in [('parent','parent_baseline'),('epoch_01','object_condition_repair'),('epoch_02','object_condition_repair')]:
            stem=f'trials/{trial}/generations/{tag}';receipt=verify_seal(z,stem,c,pack_sha,'ordinary_ce' if tag=='parent' else tag)
            rows=read_rows(z,stem+'.jsonl');targets=previous.verify_partial_answers(rows,gold,'ordinary_ce' if tag=='parent' else tag,161)
            if tag!='parent':
                source=next(x for x in src['stages'] if x['tag']==tag)
                if receipt['adapter_files']!=source['adapter_files'] or rs['epochs'][tag]!=state['trial']['epoch_evaluations'][tag] or rs['epochs'][tag]['completed']!=161:raise ValueError('Epoch source/sealed final status differs')
            if remarks:verified_notes(remarks,tag,rows)
            for i,(row,target) in enumerate(zip(rows,targets,strict=True)):
                details.append({'tag':tag,'index':i,'row':row,'gold':target|{'skill':target['task']},'frozen_check':None,'annotation':bound_note(remarks,tag,row)})
            models[tag]=rows
        counts={'epoch_01':124,'epoch_02':0}
        for name in sorted(n for n in z.namelist() if n.endswith('.result.json')):
            q=json.loads(z.read(name.replace('.result.json','.json')))
            if q['schema']!=recovery.SCHEMA:continue
            tag=q['best']['tag'];source=next(x for x in src['stages'] if x['tag']==tag)
            if q['best']['adapter']!=source['adapter']:raise ValueError('Recovery loaded a different epoch adapter')
            item=verify_worker(z,name,c,recipe,z.read('trials/object_condition_repair/generations/'+tag+'.jsonl'),counts[tag]);counts[tag]=item['completed'];workers.append(item)
        if len(workers)!=9 or counts!={'epoch_01':161,'epoch_02':161} or sum(w['new_answers'] for w in workers)!=198:raise ValueError('Missing/extra recovery worker or inference count')
    summary={'schema':'wengua_actual_completed_v67_review_v1','upload_sha256':UPLOAD_SHA,'archive_members':150,'original_training_verified_against_partial_upload':True,
        'frozen_training_files':329,'preservation':preserved,'original_formal_updates':226,'new_training_updates':0,'formal_answers':483,'new_inference_answers':198,
        'recovery_workers':workers,'actual_oom_count_after_recovery':sum(w['maximum_allocator_ooms'] for w in workers),
        'epoch_adapter_files':{x['tag']:x['adapter_files'] for x in src['stages']},'fit':fit,'observations':observations(models,remarks) if remarks else None,
        'curriculum_diagnostics':curriculum_diagnostics(course),'notes_bound_to_answers':sum(bool(x['annotation']) for x in details),
        'full_professional_promotion':False,'blind_case_accuracy_claimed':False,'new_case_family_benchmark':False,'original_final_test_used':False,
        'unuploaded_model_tensors_independently_rehashed':False,'source_books_or_original_chart_code_modified':False,
        'model_outputs_may_be_used_as_sft_or_rag':False,'prior_input_format_summary_corrected':True}
    output.mkdir(parents=True,exist_ok=True);d.write(output/'summary.json',summary)
    with gzip.open(output/'answer_observations.jsonl.gz','wt',encoding='utf8') as f:
        for x in details:f.write(d.dumps(x)+'\n')
    return summary,details


def main(argv=None):
    p=ArgumentParser(description=__doc__);p.add_argument('archive',type=Path,nargs='?');p.add_argument('--partial-archive',type=Path);p.add_argument('--notes',type=Path);p.add_argument('--partial-notes',type=Path);p.add_argument('--tokenizer-root',type=Path)
    p.add_argument('--output',type=Path,default=runtime.ROOT/'local/model_reviews/v67_complete');p.add_argument('--workbook',action='store_true');p.add_argument('--restore',action='store_true');a=p.parse_args(argv)
    if a.restore:
        import re
        from wengua_corpus.__main__ import restore
        value=input('粘贴此前教材的64位口令（输入可见）：').strip()
        if not re.fullmatch('[0-9a-fA-F]{64}',value):raise ValueError('Expected existing textbook key')
        print(restore(d.read(runtime.ROOT/'docs/model_reviews/20261112_v67_complete/private_review_catalog.json'),bytes.fromhex(value)));return 0
    if a.archive is None or a.partial_archive is None:p.error('archive and --partial-archive, or --restore, required')
    summary,details=audit(a.archive,a.partial_archive,a.output,a.notes,a.partial_notes,a.tokenizer_root)
    if a.workbook:
        from .v61_review import workbook
        workbook(a.output/'Wengua_V67_三阶段483答完整审阅.xlsx',details)
    print(d.dumps({'formal_answers':len(details),'new_training_updates':0,'new_answers':198,'bound_notes':summary['notes_bound_to_answers']}));return 0


if __name__=='__main__':raise SystemExit(main())
