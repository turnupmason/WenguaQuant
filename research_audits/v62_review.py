"""Read-only review of the actual V62 run; evaluation answers never teach SFT/RAG."""
from pathlib import Path
from collections import Counter
import argparse,gzip,json,zipfile
from wengua_course import data as d
from wengua_course_v62 import runtime
from research_audits.v55_review import inspect_archive,read_rows
from research_audits.v60_review import prompt_sha,verify_reply,verify_generation
from research_audits.v59_review import memory_summary
UPLOAD_SHA='2eb43e103aa832548e7862b634cd1a4b6b9bae02eb0a48a0442c4aca3b55901d'
RATES={'lr_1em05':1e-5,'lr_2em05':2e-5}

def same_prompt(row,target):
    if row['question']!=target['messages'][:-1] or row['expected']!=target['answer'] or row['prompt_sha256']!=prompt_sha(row['question']):raise ValueError('Answer prompt/teacher/SHA mismatch')
    if row['empty'] or not row['answer'].strip():raise ValueError('Empty actual response')

def audit(archive,output,notes=None):
    archive=Path(archive)
    if d.digest(archive.read_bytes())!=UPLOAD_SHA:raise ValueError('Exact uploaded V62 ZIP required')
    runtime.configure();pack=runtime.verify_data();remarks=d.read(notes) if notes else {}
    targets={'validation':runtime.choose_development(pack,20),'diagnostic':pack['known_regression_rows'],'rag':pack['actual_rag_rows']}
    details=[];workers=[];stages={};hardware={};base={};rag_checks={}
    with zipfile.ZipFile(archive) as z:
        inspect_archive(z)
        if z.testzip() is not None:raise ValueError('CRC mismatch')
        names=set(z.namelist());contract=json.loads(z.read('sweep_contract.json'));state=json.loads(z.read('run_status.json'))
        if contract['code_sha256']!=runtime.hashes() or contract['dataset_release_sha256']!=d.digest(runtime.RELEASE.read_bytes()) or contract['rates']!=list(RATES.values()) or contract['epochs']!=2 or contract['old_book_weights_loaded'] or contract['original_test_used']:raise ValueError('V62 frozen runtime/course/plan differs')
        if state['state']!='complete_review_required' or set(state['trials'])!=set(RATES) or state['production_enabled']:raise ValueError('Unexpected actual run state')
        proof=json.loads(z.read('initialization/parent_provenance.json'));zero=json.loads(z.read('trials/official_step_zero/zero_update_proof.json'))
        if proof['schema']!=runtime.SCHEMA or not proof['parent_config']['initialization_only'] or zero['files']!=proof['files'] or not zero['all_lora_B_zero'] or zero['weight_updates']!=0:raise ValueError('Recorded official step-zero provenance differs')
        configs={}
        for trial,rate in RATES.items():
            raw=z.read(f'trials/{trial}/job_contract.json');job=json.loads(raw);c=job['config'];configs[trial]=c
            sc=json.loads(z.read(f'trials/{trial}/cycle_01/stage_contract.json'));stage=json.loads(z.read(f'trials/{trial}/cycle_01/stage_status.json'));stages[trial]=stage
            if job['code_sha256']!=runtime.hashes() or c!=state['trials'][trial]['config'] or c['learning_rate']!=rate or job['dataset_release_sha256']!=contract['dataset_release_sha256']:raise ValueError('Trial configuration differs')
            if c['book_parent_provenance_sha256']!=d.digest(z.read('initialization/parent_provenance.json')) or sc['job_contract_sha256']!=d.digest(raw) or sc['config']!=c or sc['planned_updates']!=54 or not sc['fresh_official_base'] or sc['parent_weights_reused']:raise ValueError('Fresh stage provenance differs')
            if stage!=state['trials'][trial]['fit'] or stage['steps']!=54 or stage['state']!='complete' or stage['resume_checkpoint'] is not None:raise ValueError('Unexpected actual training completion')
            if [(e['cycle'],e['steps'],e['epoch']) for e in stage['epochs']]!=[(1,27,1.0),(2,54,2.0)]:raise ValueError('Epoch boundary differs')
            if stage['base_validation_loss']!=stage['parent_validation_loss']:raise ValueError('Zero-update parent baseline differs from official loss')
        a,b=[{k:v for k,v in c.items() if k!='learning_rate'} for c in configs.values()]
        if a!=b:raise ValueError('Trials differ beyond learning rate')
        for name in sorted(n for n in names if n.endswith('.result.json')):
            raw=z.read(name.replace('.result.json','.json'));req=json.loads(raw);reply=json.loads(z.read(name));verify_reply(reply,req,raw)
            trial=name.split('/')[1]
            if trial in configs and req['config']!=configs[trial]:raise ValueError('Worker config differs from formal job')
            item={'trial':trial,'operation':req['operation'],'probe':req['probe'],'seconds':reply['elapsed_seconds'],'request_sha256':d.digest(raw),'pid':reply['pid'],'steps':reply['result'].get('steps')}
            for suffix,key in [('.training_start.json','training_start'),('.epochs.json','epoch_optimizer'),('.memory.json','memory')]:
                n=name.replace('.result.json',suffix)
                if n in names:
                    value=json.loads(z.read(n));item[key]=memory_summary(value) if key=='memory' else value
            if req['operation']=='fit':
                start=item['training_start']
                if start['step'] or start['optimizer_steps'] or reply['result']['steps']!=(4 if req['probe'] else 54):raise ValueError('New optimizer/training length differs')
                if not req['probe']:
                    epochs=item['epoch_optimizer']
                    if len(epochs)!=2 or [e['step'] for e in epochs]!=[0,27] or len({e['optimizer_id'] for e in epochs})!=1:raise ValueError('Optimizer reset within phase')
            elif req['operation']=='initialize':
                if reply['result']['updates'] or reply['result']['old_weights_loaded']:raise ValueError('Old learned adapter initialized this run')
            workers.append(item)
        for name in sorted(n for n in names if n.endswith('.attention_path.json')):
            v=json.loads(z.read(name))
            if v['software_fixture'] or not v['cuda_calls'] or v['math_fallback_allowed']:raise ValueError('Actual fused CUDA path not recorded')
        for trial in RATES:
            for suite,suffix in [('validation','validation'),('diagnostic','seen_diagnostics')]:
                for phase in ['parent','epoch_01','epoch_02']:
                    filename=('validation_parent' if suite=='validation' else 'parent_seen_diagnostics') if phase=='parent' else phase+'_'+suffix
                    rows=read_rows(z,f'trials/{trial}/generations/{filename}.jsonl');gold=targets[suite];by={r['id']:r for r in gold}
                    if len(rows)!=len(gold) or set(r['id'] for r in rows)!=set(by):raise ValueError('Missing/extra formal responses')
                    if phase=='parent':base[(trial,suite)]=rows
                    for row in rows:
                        same_prompt(row,by[row['id']]);idx=next(i for i,r in enumerate(gold) if r['id']==row['id'])
                        tag=trial+':'+phase+':'+suite;note=remarks.get(tag,{}).get(row['id'])
                        if note and note['answer_sha256']!=d.digest(row['answer']):raise ValueError('Reading note bound to different answer')
                        details.append({'tag':tag,'index':idx,'row':row,'gold':by[row['id']],'annotation':note})
            path=f'trials/{trial}/generations/language_nominee_actual_rag.jsonl';rows=read_rows(z,path);proof=json.loads(z.read(path.replace('.jsonl','.provenance.json')))
            rag_checks[trial]=verify_generation(rows,targets['rag'],proof,adapter_sha=proof['adapter_sha256'],token_budget=512,stage='rag')
            if len(rows)!=18:raise ValueError('18 RAG responses required')
            for i,(row,gold) in enumerate(zip(rows,targets['rag'],strict=True)):
                same_prompt(row,gold);tag=trial+':rag';note=remarks.get(tag,{}).get(row['id'])
                if note and note['answer_sha256']!=d.digest(row['answer']):raise ValueError('RAG note bound to another answer')
                details.append({'tag':tag,'index':i,'row':row,'gold':gold,'annotation':note})
        for suite in ['validation','diagnostic']:
            if base[('lr_1em05',suite)]!=base[('lr_2em05',suite)]:raise ValueError('Same zero-update baseline differs')
        smoke=[(n,r) for n in names if n.endswith('/smoke_generation.jsonl') for r in read_rows(z,n)]
        for n,row in smoke:same_prompt(row,targets['validation'][0])
    if len(details)!=300 or len(smoke)!=2 or len(workers)!=7:raise ValueError('Actual worker/answer count differs')
    totals={tag:dict(Counter(x['annotation']['grade'] for x in details if x['tag']==tag and x['annotation'])) for tag in {x['tag'] for x in details}}
    summary={'schema':'wengua_actual_v62_review_v1','upload_sha256':UPLOAD_SHA,'archive_members':len(names),'state':state['state'],'formal_generations':300,'smoke_generations':2,'distinct_formal_responses_excluding_identical_baseline_copy':256,'verified_workers':workers,'formal_updates_total':108,'pressure_updates_total':8,'stages':stages,'both_official_baselines_identical':True,'zero_update_initializer_checked_from_receipts':True,'unuploaded_weight_tensors_independently_rehashed':False,'old_learned_adapters_loaded':False,'actual_rag_official_baseline_present':False,'frozen_rag_diagnostics':rag_checks,'bounded_semantic_counts':totals,'notes_bound_to_answers':sum(bool(x['annotation']) for x in details),'final_test_used':False,'professional_promotion':False,'model_answers_are_not_rag_or_sft_teachers':True}
    output=Path(output);output.mkdir(parents=True,exist_ok=True);d.write(output/'summary.json',summary)
    with gzip.open(output/'answer_observations.jsonl.gz','wt',encoding='utf8') as f:
        for row in details:f.write(d.dumps(row)+'\n')
    return summary,details

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('archive',type=Path,nargs='?');parser.add_argument('--output',type=Path,default=runtime.ROOT/'local/reviews/v62');parser.add_argument('--notes',type=Path);parser.add_argument('--workbook',action='store_true');parser.add_argument('--restore',action='store_true');parser.add_argument('--key-file',type=Path);a=parser.parse_args(argv)
    if a.restore:
        import re
        from wengua_corpus.__main__ import restore
        value=a.key_file.read_text(encoding='utf-8-sig').strip() if a.key_file else input('粘贴此前教材的64位口令（输入可见）：').strip()
        if not re.fullmatch('[0-9a-fA-F]{64}',value):raise ValueError('Expected existing textbook delivery key')
        entry=d.read(runtime.ROOT/'docs/model_reviews/20261110_v62/private_review_catalog.json');out=restore(entry,bytes.fromhex(value))
        print(out/'review/Wengua_V62_逐轮逐题审阅.xlsx');return 0
    if a.archive is None:parser.error('archive or --restore required')
    summary,details=audit(a.archive,a.output,a.notes)
    if a.workbook:
        from research_audits.v61_review import workbook
        workbook(a.output/'Wengua_V62_逐轮逐题审阅.xlsx',[x|{'frozen_check':None,'gold':x['gold']|{'skill':x['gold'].get('skill',x['gold']['task'])}} for x in details])
    print(d.dumps({'answers':len(details),'workers':len(summary['verified_workers']),'output':str(a.output)}))
if __name__=='__main__':main()
