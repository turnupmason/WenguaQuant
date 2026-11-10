"""Read-only audit of the actual V61 result. Model answers never become teachers."""
from pathlib import Path
from collections import Counter
import argparse,gzip,json,zipfile
from wengua_course import data as d
from wengua_course_v61 import runtime
from wengua_course_v60 import checks
from research_audits.v55_review import inspect_archive,read_rows
from research_audits.v60_review import prompt_sha,verify_reply,verify_generation
from research_audits.v59_review import memory_summary
UPLOAD_SHA='4a34aa3db67d9ef299388fbb6178705233ca7643e742d46409926c700b3b06a9'
RATES={'lr_3em06':3e-6,'lr_1em05':1e-5}

def audit(archive,output,notes=None):
    archive,output=Path(archive),Path(output)
    if d.digest(archive.read_bytes())!=UPLOAD_SHA:raise ValueError('Exact V61 upload required')
    runtime.configure();pack=runtime.verify_data();gold=runtime.choose_development(pack);rag=pack['actual_rag_rows'];remarks=d.read(notes) if notes else {}
    details=[];workers=[];scores={};stages={};annotations=[]
    with zipfile.ZipFile(archive) as z:
        inspect_archive(z)
        if z.testzip() is not None:raise ValueError('Archive CRC mismatch')
        names=set(z.namelist());state=json.loads(z.read('run_status.json'));contract=json.loads(z.read('sweep_contract.json'));retained=json.loads(z.read('retained_best.json'));nominee=json.loads(z.read('research_nominee.json'))
        if contract['code_sha256']!=runtime.hashes() or contract['dataset_release_sha256']!=d.digest(runtime.RELEASE.read_bytes()) or contract['rates']!=list(RATES.values()) or contract['epochs']!=2 or contract['original_test_used']:raise ValueError('Frozen V61 code/release/plan mismatch')
        if state['state']!='complete_review_required' or set(state['trials'])!=set(RATES) or not retained['unchanged_original_parent']:raise ValueError('Actual trial completion/retention differs')
        if retained['adapter_sha256']!=d.read(runtime.RELEASE)['reviewed_parent']['adapter_sha256']:raise ValueError('Original retained adapter differs')
        configs={}
        for trial,rate in RATES.items():
            jc_raw=z.read(f'trials/{trial}/job_contract.json');jc=json.loads(jc_raw);c=jc['config'];configs[trial]=c
            sc=json.loads(z.read(f'trials/{trial}/cycle_01/stage_contract.json'));stage=json.loads(z.read(f'trials/{trial}/cycle_01/stage_status.json'));stages[trial]=stage
            if jc['code_sha256']!=runtime.hashes() or jc['dataset_release_sha256']!=contract['dataset_release_sha256'] or c!=state['trials'][trial]['config'] or c['learning_rate']!=rate:raise ValueError('Job scientific settings differ')
            if sc['job_contract_sha256']!=d.digest(jc_raw) or sc['config']!=c or sc['planned_updates']!=62 or sc['parent_provenance_sha256']!=c['book_parent_provenance_sha256']:raise ValueError('Stage/job provenance differs')
            if stage!=state['trials'][trial]['fit'] or stage['steps']!=62 or stage['planned_steps']!=62 or stage['state']!='complete' or stage['resume_checkpoint'] is not None or not stage['parent_weights_reused'] or stage['fresh_official_base']:raise ValueError('Actual inherited 62-step fit differs')
            if [(e['cycle'],e['steps'],e['epoch']) for e in stage['epochs']]!=[(1,31,1.0),(2,62,2.0)]:raise ValueError('Epoch boundaries differ')
        comparable=[{k:v for k,v in c.items() if k!='learning_rate'} for c in configs.values()]
        if comparable[0]!=comparable[1]:raise ValueError('Trials differ beyond learning rate')
        for name in sorted(n for n in names if n.endswith('.result.json')):
            raw=z.read(name.replace('.result.json','.json'));request=json.loads(raw);reply=json.loads(z.read(name));verify_reply(reply,request,raw);trial=name.split('/')[1]
            if request['config']!=configs[trial]:raise ValueError('Worker and job configs differ')
            item={'trial':trial,'operation':request['operation'],'probe':request['probe'],'tag':(request.get('best') or {}).get('tag'),'seconds':reply['elapsed_seconds'],'state':reply['result']['state'],'request_sha256':d.digest(raw)}
            for suffix,key in [('.memory.json','memory'),('.training_start.json','training_start'),('.epochs.json','epoch_optimizer')]:
                n=name.replace('.result.json',suffix)
                if n in names:
                    value=json.loads(z.read(n));item[key]=memory_summary(value) if key=='memory' else value
            cache=name.replace('.result.json','.cache_releases.jsonl')
            if cache in names:item['cache_releases']=len(read_rows(z,cache))
            if request['operation']=='fit':
                start=item['training_start']
                if start['step']!=0 or start['optimizer_steps']!=[]:raise ValueError('New stage inherited an optimizer')
                if reply['result']['steps']!=(4 if request['probe'] else 62):raise ValueError('Wrong actual fit length')
                if not request['probe']:
                    epochs=item['epoch_optimizer']
                    if len(epochs)!=2 or len({e['optimizer_id'] for e in epochs})!=1 or [e['step'] for e in epochs]!=[0,31]:raise ValueError('Optimizer reset across continuous epochs')
            workers.append(item)
        for name in [n for n in names if n.endswith('.attention_path.json')]:
            a=json.loads(z.read(name))
            if not a['cuda_calls'] or a['math_fallback_allowed'] or a['software_fixture']:raise ValueError('Real fused CUDA route missing')
        sets=[]
        for trial in RATES:
            for tag,filename in [('parent','validation_parent'),('epoch_01','epoch_01_validation'),('epoch_02','epoch_02_validation')]:
                sets.append((trial+':'+tag,trial,filename,gold,None))
        sets += [('parent_rag','lr_3em06','parent_actual_rag',rag,retained['adapter_sha256']),('low_rag','lr_3em06','selected_actual_rag',rag,nominee['adapter_sha256']),('high_rag','lr_1em05','selected_actual_rag',rag,None)]
        for tag,trial,filename,targets,sha in sets:
            path=f'trials/{trial}/generations/{filename}.jsonl';rows=read_rows(z,path)
            if len(rows)!=len(targets):raise ValueError('Missing or extra answers')
            if 'rag' in tag:
                proof=json.loads(z.read(path.replace('.jsonl','.provenance.json')))
                if sha is None:sha=proof['adapter_sha256']
                scores[tag]=verify_generation(rows,targets,proof,adapter_sha=sha,token_budget=512,stage='rag')
            else:
                scores[tag]=checks.score(rows,targets)
                expected=state['trials'][trial]['finite_checks']['parent' if filename=='validation_parent' else filename]
                if scores[tag]!=expected:raise ValueError('Frozen finite score replay differs')
            targets_byid={r['id']:r for r in targets}
            for i,row in enumerate(rows):
                if row['prompt_sha256']!=prompt_sha(row['question']) or row['empty'] or row['variant']!=('parent' if filename=='validation_parent' else filename.removesuffix('_validation') if filename.startswith('epoch_') else filename):raise ValueError('Generation question/protocol differs')
                note=remarks.get(tag,{}).get(row['id'])
                if note and note['answer_sha256']!=d.digest(row['answer']):raise ValueError('Annotation bound to another answer')
                details.append({'tag':tag,'index':i,'row':row,'gold':targets_byid[row['id']],'frozen_check':checks.fact_check(row,targets_byid[row['id']]),'annotation':note})
        low=read_rows(z,'trials/lr_3em06/generations/validation_parent.jsonl');high=read_rows(z,'trials/lr_1em05/generations/validation_parent.jsonl')
        if low!=high:raise ValueError('Same retained parent baselines are not identical')
        allrows=[(n,r) for n in sorted(names) if n.endswith('.jsonl') and ('/generations/' in n or n.endswith('/smoke_generation.jsonl')) for r in read_rows(z,n)]
        selected=min((e|{'trial':t} for t in RATES for e in stages[t]['epochs']),key=lambda e:(e['validation_loss'],e['trial'],e['steps']))
        if (nominee['adapter']!=selected['adapter'] or nominee['semantic_approval'] or nominee['original_retained_adapter_replaced'] or nominee['trial']!='lr_3em06' or nominee['steps']!=31):raise ValueError('Nominee or preservation differs')
    if len(workers)!=7 or len(details)!=294 or len(allrows)!=296:raise ValueError('Unexpected actual worker/generation count')
    totals={k:{'pass':sum(g['pass'] for g in s['groups'].values()),'complete':sum(g['complete'] for g in s['groups'].values()),'total':sum(g['total'] for g in s['groups'].values())} for k,s in scores.items()}
    summary={'schema':'wengua_readonly_v61_review_20261109','upload_sha256':UPLOAD_SHA,'members':len(names),'state':state['state'],'formal_updates_total':124,'updates_per_trial':62,'formal_generations':294,'smoke_generations':2,'same_parent_generated_twice_identical':True,'same_parent_recorded_sha256':retained['adapter_sha256'],'unuploaded_weights_independently_rehashed':False,'verified_workers':workers,'stages':stages,'frozen_scores':scores,'score_totals':totals,'stored_scores_replayed':6,'additional_rag_summaries':3,'limited_semantic_annotations':sum(bool(x['annotation']) for x in details),'semantic_counts':{tag:dict(Counter(x['annotation']['verdict'] for x in details if x['tag']==tag and x['annotation'])) for tag in scores},'original_retained_unchanged':True,'automatic_professional_promotion':False,'natural_conversation_suite_present':False,'complete_case_evaluation_present':False,'not_rag_or_sft_input':True,'production_enabled':False}
    output.mkdir(parents=True,exist_ok=True);d.write(output/'summary.json',summary)
    with gzip.open(output/'answer_observations.jsonl.gz','wt',encoding='utf8') as f:
        for x in details:f.write(d.dumps(x)+'\n')
    return summary,details

def workbook(path,details):
    from openpyxl import Workbook,load_workbook
    wb=Workbook(write_only=True);ws=wb.create_sheet('逐组逐题');header=['组别','序号','ID','任务','完整问题JSON','原始回答','来源参考答案','原检查JSON','限定读答注释JSON'];ws.append(header);expected=[]
    for x in details:
        r=x['row'];values=[x['tag'],x['index'],r['id'],x['gold']['skill'],d.dumps(r['question']),r['answer'],r['expected'],d.dumps(x['frozen_check']),d.dumps(x['annotation'])]
        if any(len(str(v))>32767 for v in values):raise ValueError('Excel cell limit; refuse truncation')
        ws.append(values);expected.append(values)
    wb.save(path);reader=load_workbook(path,read_only=True,data_only=False);actual=iter(reader.active.values)
    if list(next(actual))!=header:raise ValueError('Workbook header differs')
    for a,b in zip(expected,actual,strict=True):
        if a!=list(b):raise ValueError('Workbook full-cell readback differs')
    reader.close()

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path,nargs='?');p.add_argument('--output',type=Path,default=runtime.ROOT/'local/reviews/v61');p.add_argument('--notes',type=Path);p.add_argument('--workbook',action='store_true');p.add_argument('--restore',action='store_true');p.add_argument('--key-file',type=Path);a=p.parse_args(argv)
    if a.restore:
        import re
        from wengua_corpus.__main__ import restore
        entry=d.read(runtime.ROOT/'docs/model_reviews/20261109_v61/private_review_catalog.json')
        value=a.key_file.read_text(encoding='utf-8-sig').strip() if a.key_file else input('粘贴此前教材的64位口令（输入可见）：').strip()
        if not re.fullmatch('[0-9a-fA-F]{64}',value):raise ValueError('Expected the existing textbook delivery key')
        output=restore(entry,bytes.fromhex(value));print('逐组逐题审阅已恢复：'+str(output));return 0
    if a.archive is None:p.error('archive or --restore required')
    summary,details=audit(a.archive,a.output,a.notes)
    if a.workbook:workbook(a.output/'Wengua_V61_逐组逐题审核.xlsx',details)
    print(d.dumps({'formal_generations':len(details),'stored_scores_replayed':6,'output':str(a.output)}))
if __name__=='__main__':main()
