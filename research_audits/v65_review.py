"""Read-only review of six V65 answer sets; never a teacher or promotion gate."""
from pathlib import Path
from collections import Counter
import argparse,gzip,json,zipfile
from wengua_course import data as d
from wengua_course_v65 import runtime,baselines
from .v55_review import inspect_archive,read_rows
from .v59_review import memory_summary
from .v60_review import verify_reply,prompt_sha
from .v64_review import verify_answers as ordered_answers,verified_notes
from .v63_checkpoint_review import bound_note

UPLOAD_SHA='b5d8f4cd7cf3a2a4fc722ead8bfd7480049e74c9fe5e88883a7197a5cc8d27a2'
VARIANTS={'v63_trained':('P','T'),'official_zero':('O','O')}


def align_answers(rows,gold):
    by={r['id']:r for r in rows}
    if len(by)!=len(rows) or set(by)!={r['id'] for r in gold}:raise ValueError('Missing, repeated or foreign answer ID')
    aligned=[by[g['id']] for g in gold];ordered_answers(aligned,gold);return aligned


def source_inventory(pack):
    rows=[r for r in pack['records'] if r['split']=='train'];dev=runtime.choose_development(pack)
    return {'training_cores':len(rows),'distinct_final_teacher_texts':len({r['answer'] for r in rows}),
        'comparison_explanation_phrase_count':sum('这里比较' in r['answer'] for r in rows),
        'source_units':len(pack['units']),'used_source_units':pack['summary']['used_source_units'],
        'case_linked_cores':pack['summary']['case_linked_cores'],'full_two_book_case_training':False,
        'new_development_skills':dict(Counter(r['skill'] for r in dev)),
        'new_development_multiturn':sum(len(r['messages'])>3 for r in dev),
        'review_board_or_location_questions':sum(r.get('skill') in ['board','locations'] for r in runtime.review_rows(pack)),
        'template_frequency_does_not_prove_a_causal_training_defect':True}


def observations(models,gold,remarks):
    groups={'palace_relation':range(40,46),'verification':range(46,51),'three_condition_states':range(51,54),
            'bank_result_with_reason':range(54,56),'named_formation':range(56,58),'ordinary':range(30,34),'original_recall':range(12)}
    result={}
    for name,indices in groups.items():
        result[name]={'denominator':len(indices),'known_development_only':True,'whole_answer_observation':True,
                     'supported':{tag:sum(remarks[tag][gold[i]['id']]['verdict']=='supported_for_asked_fact' for i in indices) for tag in models}}
    result['primary_role_fields']={'denominator':6,'known_development_only':True,'whole_answer_observation':False,
        'supported':{tag:sum(remarks[tag][gold[i]['id']].get('asked_primary_role_field_supported',False) for i in range(34,40)) for tag in models}}
    return result


def audit(archive,output,notes=None,prior_answer_root=None):
    archive=Path(archive)
    if d.digest(archive.read_bytes())!=UPLOAD_SHA:raise ValueError('Exact received V65 archive required')
    runtime.configure();pack=runtime.verify_data();gold=runtime.review_rows(pack);remarks=d.read(notes) if notes else {}
    models={};workers=[];stages={};numerics=[];details=[];configs={};reuse_total=0
    with zipfile.ZipFile(archive) as z:
        inspect_archive(z)
        if z.testzip() is not None:raise ValueError('Archive CRC failed')
        names=z.namelist();run=json.loads(z.read('sweep_contract.json'));state=json.loads(z.read('run_status.json'))
        if (run['schema']!=runtime.SCHEMA or run['code_sha256']!=runtime.hashes()
            or run['dataset_release_sha256']!=d.digest(runtime.RELEASE.read_bytes()) or run['source_variants']!=list(VARIANTS)
            or run['learning_rate']!=2e-5 or run['epochs']!=2 or not run['same_course'] or run['old_optimizer_inherited']
            or run['automatic_professional_promotion'] or run['original_test_used'] or state['state']!='complete_review_required'
            or state['production_enabled'] or state['curriculum']!=pack['summary']):raise ValueError('Frozen actual V65 contract differs')
        for variant,(base,prefix) in VARIANTS.items():
            raw=z.read(f'initializations/{variant}/parent_provenance.json');parent=json.loads(raw)
            expected=d.read(runtime.RELEASE)['sources'][variant]
            if (d.digest(raw)!=run['parent_provenance'][variant] or parent['optimizer_inherited']
                or parent['files']['adapter_model.safetensors']!=expected['sha256']
                or not parent['source_path'].replace('\\','/').endswith(expected['relative_path'])
                or bool(parent['parent_config'].get('initialization_only'))!=(variant=='official_zero')):
                raise ValueError('Actual pinned source kind or weights differ')
            for name in [variant,variant+'_baseline']:
                rawjob=z.read(f'trials/{name}/job_contract.json');jc=json.loads(rawjob);c=jc['config'];configs[name]=c
                if (jc['code_sha256']!=run['code_sha256'] or jc['schema']!=runtime.SCHEMA
                    or jc['dataset_release_sha256']!=run['dataset_release_sha256'] or c['source_variant']!=variant
                    or c['book_parent_provenance_sha256']!=d.digest(raw)):raise ValueError('Actual job source differs')
                if name.endswith('_baseline'):continue
                s=json.loads(z.read(f'trials/{name}/cycle_01/stage_status.json'));stages[name]=s
                sc=json.loads(z.read(f'trials/{name}/cycle_01/stage_contract.json'))
                if (c!=state['trials'][name]['config'] or sc['config']!=c or sc['job_contract_sha256']!=d.digest(rawjob)
                    or sc['planned_updates']!=458 or not sc['new_phase_optimizer'] or sc['parent_provenance_sha256']!=d.digest(raw)
                    or sc['stack']!=d.read(runtime.RELEASE)['original_stack'] or s!=state['trials'][name]['fit']
                    or s['steps']!=458 or s['state']!='complete' or s['resume_checkpoint'] is not None
                    or s['fresh_official_base']!=(variant=='official_zero') or s['parent_weights_reused']!=(variant=='v63_trained')
                    or [e['steps'] for e in s['epochs']]!=[229,458] or [e['epoch'] for e in s['epochs']]!=[1.,2.]):
                    raise ValueError('Actual matched two-epoch training differs')
            baseline=read_rows(z,f'trials/{variant}_baseline/generations/parent_review.jsonl')
            old,source=baselines.source_rows(variant,prior_answer_root or runtime.ROOT);by={r['id']:r for r in old}
            reused=[by[r['id']] for r in gold if r['id'] in by];reuse_total+=len(reused)
            proof=json.loads(z.read(f'trials/{variant}_baseline/generations/parent_review.provenance.json'))
            reuse=json.loads(z.read(f'trials/{variant}_baseline/generations/parent_review.reuse.json'))
            if (baseline[:len(reused)]!=reused or reuse['source']!=source or reuse['reused_ids']!=[r['id'] for r in reused]
                or reuse['original_answers_rewritten'] or reuse['weight_updates'] or reuse['adapter_sha256']!=expected['sha256']
                or proof['adapter_sha256']!=expected['sha256'] or proof['ids']!=[r['id'] for r in gold]
                or proof['prompt_sha256']!=d.digest(d.dumps([r['messages'] for r in gold])) or proof['max_new_tokens']!=512
                or state['baselines'][variant]!={'completed':58,'requested':58,'state':'complete'}):raise ValueError('Old answer reuse differs')
            models[base]=align_answers(baseline,gold)
            for epoch in [1,2]:models[prefix+str(epoch)]=align_answers(read_rows(z,f'trials/{variant}/generations/epoch_0{epoch}_review.jsonl'),gold)
        changing={'source_variant','initialization_label','book_parent_snapshot','book_parent_provenance_sha256'}
        if {k:v for k,v in configs['v63_trained'].items() if k not in changing}!={k:v for k,v in configs['official_zero'].items() if k not in changing}:
            raise ValueError('Comparison differs beyond source initialization')
        for label,rows in models.items():
            if remarks:verified_notes(remarks,label,rows)
            for i,(r,g) in enumerate(zip(rows,gold,strict=True)):
                details.append({'tag':label,'index':i,'row':r,'gold':g|{'skill':g.get('skill','known_dialogue')},'frozen_check':None,'annotation':bound_note(remarks,label,r)})
        for name in sorted(n for n in names if n.endswith('.result.json')):
            raw=z.read(name.replace('.result.json','.json'));request=json.loads(raw);reply=json.loads(z.read(name));verify_reply(reply,request,raw)
            trial=name.split('/')[1]
            if request['config']!=configs[trial]:raise ValueError('Worker settings differ')
            item={'trial':trial,'operation':request['operation'],'probe':request['probe'],'pid':reply['pid'],
                  'seconds':reply['elapsed_seconds'],'request_sha256':d.digest(raw)}
            for suffix,key in [('.memory.json','memory'),('.training_start.json','training_start'),('.epochs.json','epoch_optimizer')]:
                f=name.replace('.result.json',suffix)
                if f in names:
                    v=json.loads(z.read(f));item[key]=memory_summary(v) if key=='memory' else v
            if request['operation']=='fit':
                if item['training_start']['step'] or item['training_start']['optimizer_steps'] or reply['result']['steps']!=(2 if request['probe'] else 458):
                    raise ValueError('Unexpected phase steps or inherited optimizer')
                if not request['probe']:
                    if [e['step'] for e in item['epoch_optimizer']]!=[0,229] or len({e['optimizer_id'] for e in item['epoch_optimizer']})!=1:
                        raise ValueError('Optimizer reset between epochs')
            workers.append(item)
        for name in sorted(n for n in names if '/numerical_checks/' in n):
            value=json.loads(z.read(name));variant=name.split('/')[1]
            if (not value['actual_cuda'] or value['software_fixture_only'] or value['weights_updated_by_check']
                or value['training_step'] or len(value['cached_vs_full_prefix'])!=2
                or any(v['relative_l2_error']>.03 for v in value['cached_vs_full_prefix'])
                or value['nonzero_lora_B_present']!=(variant=='v63_trained')
                or bool(value['adapter_max_logit_delta'])!=(variant=='v63_trained')):raise ValueError('Numerical activation proof differs')
            numerics.append({'trial':variant,'file':name,**value})
        for name in (n for n in names if n.endswith('.attention_path.json')):
            value=json.loads(z.read(name))
            if value['software_fixture'] or not value['cuda_calls'] or value['math_fallback_allowed']:raise ValueError('Actual fused CUDA path missing')
        smoke=[r for n in names if n.endswith('/smoke_generation.jsonl') for r in read_rows(z,n)]
    if len(names)!=109 or len(workers)!=6 or len(details)!=348 or len(smoke)!=2 or reuse_total!=52 or len(numerics)!=4:
        raise ValueError('Actual archive, worker, answer or numerical receipt count differs')
    summary={'schema':'wengua_actual_v65_readonly_review_v1','upload_sha256':UPLOAD_SHA,'archive_members':109,
        'frozen_runtime_files':len(run['code_sha256']),'formal_answers_stored':348,'smoke_answers':2,'unchanged_reused_answers':52,
        'formal_new_answers':296,'new_answers_including_smoke':298,'formal_updates':916,'pressure_updates':4,
        'epochs':stages,'workers':workers,'actual_cuda_numerical_checks':numerics,'source_curriculum':source_inventory(pack),
        'notes_bound_to_answers':sum(bool(r['annotation']) for r in details),'limited_observations':observations(models,gold,remarks) if remarks else None,
        'answer_shapes':{tag:{'answers':58,'token_limit_hits':sum(r['reached_token_limit'] for r in rows),
             'mean_characters':sum(len(r['answer']) for r in rows)/58,'comparison_phrase_count':sum('这里比较' in r['answer'] for r in rows)} for tag,rows in models.items()},
        'all_questions_are_known_development':True,'original_test_used':False,'professional_promotion':False,'per_update_training_loss_uploaded':False,
        'fresh_retrieval_retested':False,'unuploaded_weight_tensors_independently_rehashed':False,'responses_allowed_as_sft_or_rag_teachers':False}
    output=Path(output);output.mkdir(parents=True,exist_ok=True);d.write(output/'summary.json',summary)
    with gzip.open(output/'answer_observations.jsonl.gz','wt',encoding='utf8') as f:
        for r in details:f.write(d.dumps(r)+'\n')
    return summary,details


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path,nargs='?');p.add_argument('--notes',type=Path)
    p.add_argument('--output',type=Path,default=runtime.ROOT/'local/model_reviews/v65');p.add_argument('--prior-answer-root',type=Path)
    p.add_argument('--workbook',action='store_true');p.add_argument('--restore',action='store_true');a=p.parse_args(argv)
    if a.restore:
        import re
        from wengua_corpus.__main__ import restore
        value=input('粘贴此前教材的64位口令（输入可见）：').strip()
        if not re.fullmatch('[0-9a-fA-F]{64}',value):raise ValueError('Expected existing textbook key')
        entry=d.read(runtime.ROOT/'docs/model_reviews/20261111_v65/private_review_catalog.json')
        print(restore(entry,bytes.fromhex(value))/'review/Wengua_V65_逐组逐答审阅.xlsx');return 0
    if a.archive is None:p.error('archive or --restore required')
    summary,details=audit(a.archive,a.output,a.notes,a.prior_answer_root)
    if a.workbook:
        from .v61_review import workbook
        workbook(a.output/'Wengua_V65_逐组逐答审阅.xlsx',details)
    print(d.dumps({'answers':len(details),'bound_notes':summary['notes_bound_to_answers']}));return 0


if __name__=='__main__':raise SystemExit(main())
