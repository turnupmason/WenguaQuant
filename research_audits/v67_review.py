"""Audit the actual interrupted V67 evaluation without treating it complete.

Source reading observations are SHA-bound and are never SFT/RAG teachers.
"""
from pathlib import Path
import argparse,gzip,json,tempfile,zipfile
from collections import Counter
from wengua_course import data as d
from wengua_course_v67 import runtime,evaluation
from .v55_review import inspect_archive,read_rows
from .v59_review import memory_summary
from .v64_review import verified_notes
from .v63_checkpoint_review import bound_note

UPLOAD_SHA='97e9fd953946abe4c7379db5646af0bbf104b8c2b8adec6a33d02ff3d41f9591'


def verify_partial_answers(rows,pack,tag,expected_count):
    evaluation.validate_cache(rows,pack,tag)
    if len(rows)!=expected_count:raise ValueError('Actual partial answer count differs')
    targets=evaluation.sequence(pack,rows)
    for row,target in zip(rows,targets):
        if (row['components']!=target['components'] or row['source_ids']!=target['source_ids'] or row['task']!=target['task']
            or row['group']!=target['group'] or row['weight_updates']!=0 or row['semantic_correctness_verified']
            or row['empty']!= (not row['answer'].strip()) or row['reached_token_limit']!=(row['generated_tokens']>=512)):
            raise ValueError('Saved whole answer/source metadata differs')
    return targets


def verify_fit(state,stage):
    if (stage!=state['trial']['fit'] or stage['state']!='complete' or stage['steps']!=226 or stage['planned_steps']!=226
        or stage['resume_checkpoint'] is not None or not stage['parent_weights_reused'] or stage['fresh_official_base']
        or stage['source_variant']!='v66_ordinary' or stage['optimizer_reset_between_epochs']
        or [e['steps'] for e in stage['epochs']]!=[113,226] or [e['epoch'] for e in stage['epochs']]!=[1.,2.]):
        raise ValueError('Both completed continuous epochs must be retained')
    if state['state']!='failed' or state['active_epoch']!=1 or state['trial']['epoch_evaluations']:
        raise ValueError('This upload is not a completed evaluation')
    if 'out of memory' not in state['error']['message'].lower():raise ValueError('Different actual interruption')


def observations(models,notes):
    out={}
    for label,rows in models.items():
        n=[notes[label][r['id']] for r in rows]
        def count(key,ids):return sum(bool(n[i].get(key)) for i in ids if i<len(n))
        groups={'palace':range(12),'verify':range(12,22),'marriage_static':range(22,28),'bank':range(28,32),'identity':range(32,38),
                'attributes':range(38,44),'given_elements':range(44,50),'star_palace':range(50,66),'flat_current':range(66,70),
                'frozen_rag':range(70,88),'ordinary':range(88,92),'old_actual_history':range(92,103),'independent_conditions':range(103,112),
                'new_objects':range(112,min(136,len(rows))),'star_claims':range(136,min(140,len(rows))),
                'identity_referent':range(140,min(147,len(rows))),'new_actual_history':range(147,min(161,len(rows)))}
        out[label]={'available_answers':len(rows),'whole_answer_supported':sum(x['whole_answer_supported'] for x in n),
            'groups':{name:{'available':sum(i<len(n) for i in ids),'whole_answer_supported':count('whole_answer_supported',ids)} for name,ids in groups.items()},
            'verification_verdict_and_direction':sum(n[i].get('verification_verdict_supported',False) and n[i].get('directed_relation_supported',False) for i in range(12,22)),
            'star_positions':count('positions_supported',range(50,66)),'star_palace_direction':count('directed_palace_relation_supported',range(50,66)),
            'star_intrinsic_attribute_errors_in_original_16':count('intrinsic_star_error',range(50,66)),
            'ordinary_unambiguously_supported':count('whole_answer_supported',range(88,92)),
            'explicit_extra_rag_errors':count('explicit_fact_error',range(70,88)),
            'conditional_known_new_nine_main_results':count('branch_verdict_supported',range(103,112)),
            'all_requested_model_answers_present':len(rows)==161,'is_blind_case_accuracy':False}
    return out


def audit(archive,output,notes=None,tokenizer_root=None):
    archive=Path(archive)
    if d.digest(archive.read_bytes())!=UPLOAD_SHA:raise ValueError('Exact uploaded V67 result required')
    runtime.configure();course=runtime.verify_data();gold=evaluation.build(course,runtime.before.old.verify_data());remarks=d.read(notes) if notes else {}
    workers=[];models={};details=[];configs={};numerics=[]
    with zipfile.ZipFile(archive) as z:
        inspect_archive(z)
        if z.testzip() is not None:raise ValueError('Archive CRC failed')
        names=z.namelist();run=json.loads(z.read('sweep_contract.json'));state=json.loads(z.read('run_status.json'));pack_sha=d.digest(d.dumps(gold))
        if (len(names)!=70 or run['schema']!=runtime.SCHEMA or run['code_sha256']!=runtime.hashes() or len(run['code_sha256'])!=329
            or run['dataset_release_sha256']!=d.digest(runtime.RELEASE.read_bytes()) or run['evaluation_pack_sha256']!=pack_sha
            or not run['ordinary_ce'] or run['epochs']!=2 or run['old_optimizer_inherited'] or run['optimizer_reset_between_epochs']
            or run['automatic_professional_promotion'] or run['additional_long_training'] or run['original_test_used'] or state['production_enabled']
            or state['curriculum']!=course['summary']):raise ValueError('Actual frozen V67 contract/source differs')
        parent_raw=z.read('initialization/parent_provenance.json');parent=json.loads(parent_raw)
        if (d.digest(parent_raw)!=run['parent_provenance_sha256'] or parent['optimizer_inherited']
            or parent['files']['adapter_model.safetensors']!=d.read(runtime.RELEASE)['parent']['sha256']):raise ValueError('Wrong learned V66 parent')
        execution=json.loads(z.read('process_execution.json'))
        if execution['code_sha256']!=run['code_sha256'] or execution['software_fixture'] or not execution['process_per_unit']:raise ValueError('Wrong execution receipt')
        job='trials/object_condition_repair';raw_job=z.read(job+'/job_contract.json');jc=json.loads(raw_job);c=jc['config'];runtime.validate(c)
        if (jc['code_sha256']!=run['code_sha256'] or jc['dataset_release_sha256']!=run['dataset_release_sha256']
            or c!=state['trial']['config'] or c['book_parent_provenance_sha256']!=d.digest(parent_raw)):raise ValueError('Actual training job source differs')
        stage=json.loads(z.read(job+'/cycle_01/stage_status.json'));sc=json.loads(z.read(job+'/cycle_01/stage_contract.json'));verify_fit(state,stage)
        if sc['config']!=c or sc['planned_updates']!=226 or sc['job_contract_sha256']!=d.digest(raw_job) or not sc['new_phase_optimizer']:raise ValueError('Original training schedule differs')
        for label,trial,count in [('parent','parent_baseline',161),('epoch_01','object_condition_repair',124)]:
            stem=f'trials/{trial}/generations/{label}';conf=json.loads(z.read(f'trials/{trial}/job_contract.json'))['config'];configs[trial]=conf
            if conf!=c or json.loads(z.read(f'trials/{trial}/evaluation_pack.json'))!=gold:raise ValueError('Model evaluated different prompts/config')
            receipt=json.loads(z.read(stem+'.receipt.json'))
            if (receipt['pack_sha256']!=pack_sha or receipt['config']!=c or receipt['record_variant']!=('ordinary_ce' if label=='parent' else label)
                or receipt['professional_approved'] or receipt['history_source']!='model_own_previous_answers'):raise ValueError('Actual answer receipt differs')
            if label=='parent':
                if receipt['adapter_files']!=parent['files']:raise ValueError('Parent weights differ')
                complete=json.loads(z.read(stem+'.complete.json'))
                if (complete['answers_sha256']!=d.digest(z.read(stem+'.jsonl')) or complete['contract_sha256']!=d.digest(z.read(stem+'.receipt.json'))
                    or complete['answers']!=161 or state['baseline']['answers_sha256']!=complete['answers_sha256']):raise ValueError('Completed baseline seal differs')
                prefix=b''.join(z.read(stem+'.jsonl').splitlines(keepends=True)[:103])
                if (d.digest(prefix)!=state['parent_reuse']['source_sha256'] or state['parent_reuse']['reused_rows']!=103
                    or state['parent_reuse']['old_answers_rewritten']):raise ValueError('Old parent reply bytes were rewritten')
            elif stem+'.complete.json' in names:raise ValueError('Partial evaluation incorrectly sealed complete')
            rows=read_rows(z,stem+'.jsonl');targets=verify_partial_answers(rows,gold,'ordinary_ce' if label=='parent' else label,count)
            if remarks:verified_notes(remarks,label,rows)
            for i,(row,target) in enumerate(zip(rows,targets)):
                details.append({'tag':label,'index':i,'row':row,'gold':target|{'skill':target['task']},'frozen_check':None,'annotation':bound_note(remarks,label,row)})
            models[label]=rows
        for tag in ['epoch_01_review','epoch_02_review']:
            if z.read(job+'/generations/'+tag+'.jsonl'):raise ValueError('Unexpected nonempty auxiliary review file')
        if job+'/generations/epoch_02.jsonl' in names:raise ValueError('Unexpected actual E2 generation file')
        for name in sorted(n for n in names if n.endswith('.result.json')):
            request_raw=z.read(name.replace('.result.json','.json'));request=json.loads(request_raw);reply=json.loads(z.read(name))
            if (reply['request_sha256']!=d.digest(request_raw) or reply['nonce']!=request['nonce'] or reply['schema']!=execution['schema']
                or request['config']!=c or request['software_fixture'] or reply['software_fixture'] or type(reply['pid']) is not int):raise ValueError('Worker request provenance differs')
            item={'operation':request['operation'],'probe':request['probe'],'ok':reply['ok'],'seconds':reply['elapsed_seconds'],
                  'memory':memory_summary(json.loads(z.read(name.replace('.result.json','.memory.json'))))}
            if not reply['ok']:
                if request['operation']!='evaluate' or request['best']['tag']!='epoch_01' or 'out of memory' not in reply['error']['message'].lower():raise ValueError('Different failed worker')
                item['error']=reply['error'];samples=json.loads(z.read(name.replace('.result.json','.memory.json')))
                item['evaluation_memory_samples']=len(samples);item['evaluation_peak_memory_uploaded']=False
            elif request['operation']=='fit':
                if reply['result']!=(state['trial']['probe'] if request['probe'] else stage):raise ValueError('Worker and actual training results differ')
                loss=json.loads(z.read(name.replace('.result.json','.loss_path.json')));item['loss']=loss
                if loss['training_factor']!=1 or loss['weighted_training_calls'] or loss['train_loss_calls']!=(16 if request['probe'] else 1802) or loss['max_projected_positions']>128:raise ValueError('Actual ordinary training loss differs')
                att=json.loads(z.read(name.replace('.result.json','.attention_path.json')))
                if att['software_fixture'] or not att['cuda_calls'] or att['math_fallback_allowed']:raise ValueError('Real fused CUDA training path missing')
                starts=json.loads(z.read(name.replace('.result.json','.training_start.json')))
                if starts['step'] or starts['optimizer_steps']:raise ValueError('New stage inherited optimizer unexpectedly')
                epochs=json.loads(z.read(name.replace('.result.json','.epochs.json')))
                if not request['probe'] and ([x['step'] for x in epochs]!=[0,113] or len({x['optimizer_id'] for x in epochs})!=1):raise ValueError('Optimizer reset between formal epochs')
            workers.append(item)
        if len(workers)!=4 or sum(not x['ok'] for x in workers)!=1:raise ValueError('Unexpected actual worker counts')
        for name in (n for n in names if '/numerical_checks/' in n):
            v=json.loads(z.read(name));numerics.append(v)
            if (not v['actual_cuda'] or v['software_fixture_only'] or not v['nonzero_lora_B_present'] or v['adapter_max_logit_delta']<=0
                or v['weights_updated_by_check'] or v['training_step'] or len(v['cached_vs_full_prefix'])!=2
                or any(x['relative_l2_error']>.03 for x in v['cached_vs_full_prefix'])):raise ValueError('Actual learned source did not activate or match inference')
        if len(numerics)!=2:raise ValueError('Missing actual inherited-source numerical checks')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(tokenizer_root),local_files_only=True,trust_remote_code=False) if tokenizer_root else None
    with tempfile.TemporaryDirectory(prefix='v67_review_tensors_') as tmp:
        rebuilt=runtime.prepare(Path(tmp)/'job',c,tokenizer=tok,pack=course,testing=True)
        if any(rebuilt[k]!=state['prepared_summary'][k] for k in ['training_rows','deduplicated_supervised_views','training_tensor_sha256','max_actual_train_tokens']):raise ValueError('Rebuilt actual training tensors/order differ')
    summary={'schema':'wengua_actual_partial_v67_review_v1','upload_sha256':UPLOAD_SHA,'archive_members':70,'frozen_runtime_files':329,
        'state':'training_complete_evaluation_incomplete','formal_updates':226,'pressure_updates':2,'actual_workers':workers,'numerical_checks':numerics,
        'available_formal_answers':285,'parent_answers':161,'epoch_01_answers':124,'epoch_02_answers':0,'remaining_evaluation_answers':198,
        'epoch_01_complete':False,'epoch_02_complete':False,'formal_fit':stage,'recreated_training':{k:rebuilt[k] for k in ['training_rows','training_tensor_sha256','deduplicated_supervised_views','max_actual_train_tokens','decision_tokens']},
        'source_curriculum':course['summary'],'observations':observations(models,remarks) if remarks else None,'notes_bound_to_answers':sum(bool(x['annotation']) for x in details),
        'no_output_limit_or_empty':all(not r['empty'] and not r['reached_token_limit'] for rows in models.values() for r in rows),
        'unuploaded_epoch2_weight_sha_available':False,'unuploaded_weights_independently_rehashed':False,'evaluation_cache_growth_proven_as_unique_cause':False,
        'actual_gpu_logits_independently_recomputed':False,'professional_promotion':False,'new_blind_case_accuracy_claimed':False,
        'original_test_used':False,'model_replies_allowed_as_sft_or_rag':False,'rag_retrieval_retested':False}
    output=Path(output);output.mkdir(parents=True,exist_ok=True);d.write(output/'summary.json',summary)
    with gzip.open(output/'answer_observations.jsonl.gz','wt',encoding='utf8') as f:
        for row in details:f.write(d.dumps(row)+'\n')
    return summary,details


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path,nargs='?');p.add_argument('--notes',type=Path);p.add_argument('--tokenizer-root',type=Path)
    p.add_argument('--output',type=Path,default=runtime.ROOT/'local/model_reviews/v67');p.add_argument('--workbook',action='store_true');p.add_argument('--restore',action='store_true');args=p.parse_args(argv)
    if args.restore:
        import re
        from wengua_corpus.__main__ import restore
        value=input('粘贴此前教材的64位口令（输入可见）：').strip()
        if not re.fullmatch('[0-9a-fA-F]{64}',value):raise ValueError('Expected existing textbook key')
        print(restore(d.read(runtime.ROOT/'docs/model_reviews/20261112_v67/private_review_catalog.json'),bytes.fromhex(value)));return 0
    if args.archive is None:p.error('archive or --restore required')
    summary,details=audit(args.archive,args.output,args.notes,args.tokenizer_root)
    if args.workbook:
        from .v61_review import workbook
        workbook(args.output/'Wengua_V67_中断前逐答审阅.xlsx',details)
    print(d.dumps({'available_answers':len(details),'bound_notes':summary['notes_bound_to_answers'],'training_complete':True,'evaluation_complete':False}));return 0


if __name__=='__main__':raise SystemExit(main())
