"""Read-only audit of the received V64 run and all 570 formal answers.

Annotations are limited post-run observations, never new book or SFT material.
"""
from pathlib import Path
from collections import Counter
import argparse,gzip,json,zipfile
from wengua_course import data as d
from wengua_course_v64 import runtime
from .v55_review import inspect_archive,read_rows
from .v60_review import verify_reply,prompt_sha
from .v59_review import memory_summary
from .v63_checkpoint_review import bound_note

UPLOAD_SHA='2173d2226bd0c66c828726c61ba23b324ea8e65f39d835b74b411c50434622f7'
TRIALS={'lr_1em05':('L',1e-5),'lr_3em05':('H',3e-5)}
VERDICTS={'supported_for_asked_fact','partial_coverage','wrong_target_or_fact','contradictory_whole_answer','unsupported_or_irrelevant_addition'}


def verify_answers(rows,gold):
    if len(rows)!=len(gold):raise ValueError('Actual answer count differs')
    for r,g in zip(rows,gold,strict=True):
        if (r['id']!=g['id'] or r['question']!=g['messages'][:-1] or r['expected']!=g['answer']
                or r['prompt_sha256']!=prompt_sha(r['question']) or r['empty'] or not r['answer']):
            raise ValueError('Actual whole prompt/teacher/answer differs')


def verified_notes(remarks,label,rows):
    if set(remarks.get(label,{}))!={r['id'] for r in rows}:raise ValueError('Every actual answer needs one bound observation')
    for r in rows:
        n=bound_note(remarks,label,r)
        if n['verdict'] not in VERDICTS or not n.get('bounded_manual_observation_only') or not n.get('not_a_training_teacher'):
            raise ValueError('An audit note cannot become professional approval or a teacher')


def source_inventory(pack):
    train=[r for r in pack['records'] if r['split']=='train']
    unclear=[r for r in pack['records'] if (r['provenance'].get('initial_contract') or {}).get('kind')=='identity'
             and r['messages'][1]['content'].startswith('这次关系是')
             and r['provenance']['initial_contract']['role']!='本人']
    dev=[r for r in pack['records'] if r['split']=='validation' and r['skill']=='identity'
         and '这次实际关系是' in r['messages'][1]['content'] and r['provenance']['lesson_contract']['role']!='本人']
    return {'train_cores':len(train),'distinct_final_teacher_texts':len({r['answer'] for r in train}),
            'same_initial_identity_phrase_count':sum(r['answer'].startswith('先取本次起局') for r in train),
            'family_label_without_explicit_proxy_train_ids':[r['id'] for r in unclear],
            'family_label_without_explicit_proxy_development_ids':[r['id'] for r in dev],
            'ambiguity_requires_clarification_not_posthoc_teacher_relabel':True,
            'source_units':len(pack['units']),'actually_used_training_source_units':pack['summary']['training_source_unit_count'],
            'case_linked_cores':pack['summary']['case_linked_training_cores'],'full_two_book_case_training':False}


def cached_parent_rows(root):
    rows=[];receipts=[]
    for entry in d.read(runtime.ROOT/'data/reviews/v64_parent_answer_sources.json').values():
        raw=(Path(root)/entry['path']).read_bytes()
        if d.digest(raw)!=entry['sha256']:raise ValueError('Earlier actual parent answer bytes changed')
        part=list(map(json.loads,raw.splitlines()))
        if len(part)!=entry['count']:raise ValueError('Earlier actual parent answer count changed')
        rows+=part;receipts.append({'path':entry['path'],'sha256':entry['sha256'],'answers':len(part)})
    if len(rows)!=74:raise ValueError('Exactly 74 unchanged earlier answers required')
    return rows,receipts


def audit(archive,output,notes=None,prior_answer_root=None):
    archive=Path(archive)
    if d.digest(archive.read_bytes())!=UPLOAD_SHA:raise ValueError('Exact received V64 archive required')
    runtime.configure();pack=runtime.verify_data();gold=runtime.review_rows(pack);remarks=d.read(notes) if notes else {}
    details=[];workers=[];stages={};configs={};models={}
    with zipfile.ZipFile(archive) as z:
        inspect_archive(z)
        if z.testzip() is not None:raise ValueError('Archive CRC failed')
        names=z.namelist();run=json.loads(z.read('sweep_contract.json'));state=json.loads(z.read('run_status.json'))
        if (run['schema']!=runtime.SCHEMA or run['code_sha256']!=runtime.hashes()
                or run['dataset_release_sha256']!=d.digest(runtime.RELEASE.read_bytes())
                or run['rates']!=[1e-5,3e-5] or run['epochs']!=2 or not run['same_inherited_parent']
                or run['old_optimizer_inherited'] or run['original_test_used'] or run['automatic_professional_promotion']
                or state['state']!='complete_review_required' or state['production_enabled'] or state['curriculum']!=pack['summary']):
            raise ValueError('Actual frozen V64 run/course differs')
        parent_raw=z.read('initialization/parent_provenance.json');parent=json.loads(parent_raw);selected=d.read(runtime.RELEASE)['reviewed_parent']
        if (d.digest(parent_raw)!=run['parent_provenance_sha256'] or parent['optimizer_inherited']
                or parent['files']['adapter_model.safetensors']!=selected['adapter_sha256']
                or not parent['source_path'].replace('\\','/').endswith(selected['relative_path'])):
            raise ValueError('Inherited V63 step371 source differs')
        for label in ['parent_baseline',*TRIALS]:
            raw=z.read(f'trials/{label}/job_contract.json');jc=json.loads(raw);c=jc['config'];configs[label]=c
            if (jc['schema']!=runtime.SCHEMA or jc['code_sha256']!=runtime.hashes()
                    or jc['dataset_release_sha256']!=run['dataset_release_sha256']
                    or c['book_parent_provenance_sha256']!=d.digest(parent_raw)):
                raise ValueError('Sealed job differs from frozen course/parent')
            if label=='parent_baseline':continue
            sc=json.loads(z.read(f'trials/{label}/cycle_01/stage_contract.json'))
            s=json.loads(z.read(f'trials/{label}/cycle_01/stage_status.json'));stages[label]=s
            if (c!=state['trials'][label]['config'] or c['learning_rate']!=TRIALS[label][1]
                    or sc['config']!=c or sc['job_contract_sha256']!=d.digest(raw) or sc['planned_updates']!=152
                    or sc['parent_provenance_sha256']!=d.digest(parent_raw) or not sc['new_phase_optimizer']
                    or sc['stack']!=d.read(runtime.RELEASE)['original_stack'] or s!=state['trials'][label]['fit']
                    or s['state']!='complete' or s['steps']!=152 or s['planned_steps']!=152
                    or not s['parent_weights_reused'] or s['fresh_official_base'] or s['resume_checkpoint'] is not None
                    or [e['steps'] for e in s['epochs']]!=[76,152] or [e['epoch'] for e in s['epochs']]!=[1.,2.]):
                raise ValueError('Actual same-parent continuous formal training differs')
        if {k:v for k,v in configs['lr_1em05'].items() if k!='learning_rate'}!={k:v for k,v in configs['lr_3em05'].items() if k!='learning_rate'}:
            raise ValueError('Trials differ beyond learning rate')
        baseline=read_rows(z,'trials/parent_baseline/generations/parent_review.jsonl')
        old_rows,receipts=cached_parent_rows(prior_answer_root or runtime.ROOT)
        if baseline[:74]!=old_rows:raise ValueError('Original 74 parent answers were changed')
        reuse=json.loads(z.read('trials/parent_baseline/generations/parent_review.reuse.json'))
        proof=json.loads(z.read('trials/parent_baseline/generations/parent_review.provenance.json'))
        if (reuse['source_files']!=receipts or reuse['reused_ids']!=[r['id'] for r in old_rows]
                or reuse['old_answers_rewritten'] or reuse['weight_updates'] or reuse['adapter_sha256']!=selected['adapter_sha256']
                or proof['ids']!=[r['id'] for r in gold] or proof['prompt_sha256']!=d.digest(d.dumps([r['messages'] for r in gold]))
                or proof['adapter_sha256']!=selected['adapter_sha256'] or proof['max_new_tokens']!=512
                or state['parent_review']!={'completed':114,'requested':114,'state':'complete'}):
            raise ValueError('Parent reuse or full-prompt provenance differs')
        models['P']=baseline
        for trial,(prefix,_) in TRIALS.items():
            for epoch in [1,2]:models[prefix+str(epoch)]=read_rows(z,f'trials/{trial}/generations/epoch_0{epoch}_review.jsonl')
        for label,rows in models.items():
            verify_answers(rows,gold)
            if remarks:verified_notes(remarks,label,rows)
            for i,(r,g) in enumerate(zip(rows,gold,strict=True)):
                details.append({'tag':label,'index':i,'row':r,'gold':g|{'skill':g.get('skill','known_dialogue')},
                                'frozen_check':None,'annotation':bound_note(remarks,label,r)})
        for name in sorted(n for n in names if n.endswith('.result.json')):
            raw=z.read(name.replace('.result.json','.json'));request=json.loads(raw);reply=json.loads(z.read(name));verify_reply(reply,request,raw)
            label=name.split('/')[1]
            if request['config']!=configs[label]:raise ValueError('Worker settings differ')
            item={'trial':label,'operation':request['operation'],'probe':request['probe'],'pid':reply['pid'],
                  'seconds':reply['elapsed_seconds'],'request_sha256':d.digest(raw)}
            for suffix,key in [('.memory.json','memory'),('.training_start.json','training_start'),('.epochs.json','epoch_optimizer')]:
                f=name.replace('.result.json',suffix)
                if f in names:
                    value=json.loads(z.read(f));item[key]=memory_summary(value) if key=='memory' else value
            if request['operation']=='fit':
                if item['training_start']['step'] or item['training_start']['optimizer_steps']:raise ValueError('New phase inherited optimizer state')
                if reply['result']['steps']!=(2 if request['probe'] else 152):raise ValueError('Wrong number of actual updates')
                if not request['probe']:
                    events=item['epoch_optimizer']
                    if [v['step'] for v in events]!=[0,76] or len({v['optimizer_id'] for v in events})!=1:
                        raise ValueError('Optimizer reset across epochs')
            workers.append(item)
        for name in (n for n in names if n.endswith('.attention_path.json')):
            a=json.loads(z.read(name))
            if a['software_fixture'] or not a['cuda_calls'] or a['math_fallback_allowed']:raise ValueError('Actual fused CUDA route missing')
        smoke=[r for n in names if n.endswith('/smoke_generation.jsonl') for r in read_rows(z,n)]
    if len(names)!=93 or len(workers)!=5 or len(details)!=570 or len(smoke)!=2:raise ValueError('Actual archive/worker/generation count differs')
    summary={'schema':'wengua_actual_v64_readonly_review_v1','upload_sha256':UPLOAD_SHA,'archive_members':len(names),
             'frozen_runtime_files':len(run['code_sha256']),'formal_answers_stored':570,'smoke_answers':2,
             'parent_old_answers_reused_unchanged':74,'formal_new_answers':496,'new_answers_including_smoke':498,
             'formal_updates':304,'pressure_updates':4,'epochs':stages,'workers':workers,
             'source_curriculum':source_inventory(pack),'notes_bound_to_answers':sum(bool(r['annotation']) for r in details),
             'answer_shapes':{k:{'answers':len(v),'token_limit_hits':sum(r['reached_token_limit'] for r in v),
                                'mean_characters':sum(len(r['answer']) for r in v)/len(v)} for k,v in models.items()},
             'all_questions_are_known_development':True,'original_test_used':False,'professional_promotion':False,
             'per_update_training_loss_uploaded':False,'fresh_retrieval_retested':False,'known_rag_replays_per_stage':18,
             'unuploaded_weight_tensors_independently_rehashed':False,'responses_allowed_as_sft_or_rag_teachers':False}
    output=Path(output);output.mkdir(parents=True,exist_ok=True);d.write(output/'summary.json',summary)
    with gzip.open(output/'answer_observations.jsonl.gz','wt',encoding='utf8') as f:
        for r in details:f.write(d.dumps(r)+'\n')
    return summary,details


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path,nargs='?');p.add_argument('--notes',type=Path)
    p.add_argument('--output',type=Path,default=runtime.ROOT/'local/model_reviews/v64');p.add_argument('--prior-answer-root',type=Path)
    p.add_argument('--workbook',action='store_true');p.add_argument('--restore',action='store_true');a=p.parse_args(argv)
    if a.restore:
        import re
        from wengua_corpus.__main__ import restore
        value=input('粘贴此前教材的64位口令（输入可见）：').strip()
        if not re.fullmatch('[0-9a-fA-F]{64}',value):raise ValueError('Expected the existing textbook key')
        entry=d.read(runtime.ROOT/'docs/model_reviews/20261111_v64/private_review_catalog.json');out=restore(entry,bytes.fromhex(value))
        print(out/'review/Wengua_V64_逐轮逐答审阅.xlsx');return 0
    if a.archive is None:p.error('archive or --restore required')
    summary,details=audit(a.archive,a.output,a.notes,a.prior_answer_root)
    if a.workbook:
        from .v61_review import workbook
        workbook(a.output/'Wengua_V64_逐轮逐答审阅.xlsx',details)
    print(d.dumps({'formal_answers':len(details),'notes':summary['notes_bound_to_answers'],'output':str(a.output)}));return 0


if __name__=='__main__':raise SystemExit(main())
