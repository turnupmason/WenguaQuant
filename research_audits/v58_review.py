"""Frozen V58 audit: replay every stored score and identify scope limitations.

Read-only observations are post hoc, never new blind validation or teachers.
"""
from pathlib import Path
from collections import Counter
import argparse,gzip,json,re,zipfile
from wengua_course import data as d
from wengua_course_v58 import runtime,checks,runner
from research_audits.v55_review import inspect_archive,read_rows
UPLOAD_SHA='f07188e336c9f01a4a59466c82e4d32710358cb81c45b402a89d7447a08c0495'


def validate_note(note,row):
    if note['id']!=row['id'] or note['answer_sha256']!=d.digest(row['answer']):raise ValueError('Manual note detached from original answer')
    if note['verdict'] not in ['correct','partial','wrong','unusable'] or not note['reason']:raise ValueError('Incomplete bounded note')
    return note


def observation(row,gold):
    text=row['answer'];prompt=d.dumps(row['question']);c=gold.get('checks') or {}
    # A word boundary matters: 土生土长 is an idiom, not a five-element law.
    invalid=[]
    for m in re.finditer(r'([金木水火土])([生克])([金木水火土])',text):
        if text[m.start():m.start()+4]=='土生土长':continue
        if checks.asserted(text,m.start(),m.end()) and (d.GENERATES if m[2]=='生' else d.CONTROLS)[m[1]]!=m[3]:invalid.append(m[0])
    labels=re.findall(r'应改(?:记|写)?(?:为|成)?(受生|受克|比和|生|克)',text)
    mismatched=[x for x in labels if c.get('answer') in ['生','受生','克','受克','比和'] and x!=c['answer']]
    dates=re.findall(r'\d{4}年\d{1,2}月\d{1,2}日',text)
    return {'literal_law_conflicts_for_review':invalid,'correction_head_conflicts':mismatched,'placeholder_retained':'待补' in text,'unprovided_absolute_dates_for_review':[s for s in dates if s not in prompt],'output_limit':row.get('reached_token_limit',False),'exact_gold_copy':text==gold['answer'],'posthoc_not_new_score':True,'professional_approval':False}


def audit(archive,output,*,notes=None,workbook=False):
    archive,output=Path(archive),Path(output);output.mkdir(parents=True,exist_ok=True)
    if d.digest(archive.read_bytes())!=UPLOAD_SHA:raise ValueError('Bound actual V58 upload required')
    runtime.configure();pack=runtime.verify_data();micro=runtime.choose_development(pack);by={r['id']:r for r in pack['records']};final=[by[k] for k in pack['fixed_final_ids']];rag=pack['v58_actual_rag_rows'];details=[];workers=[];allrows=[];scores={}
    annotations=d.read(notes) if notes else {}
    with zipfile.ZipFile(archive) as z:
        inspect_archive(z)
        if z.testzip() is not None:raise ValueError('Archive CRC failed')
        state=json.loads(z.read('run_status.json'));contract=json.loads(z.read('sweep_contract.json'));trial=state['trials']['mixed_pilot']
        if contract['code_sha256']!=runtime.hashes() or contract['dataset_release_sha256']!=d.digest(runtime.RELEASE.read_text()):raise ValueError('Frozen code/course differs')
        if state['state']!='needs_course_review' or set(state['trials'])!={'mixed_pilot'} or contract['original_test_used']:raise ValueError('Unexpected phases/test use')
        for name in z.namelist():
            if name.endswith('.result.json'):
                reply=json.loads(z.read(name));request_name=name.replace('.result.json','.json');request=json.loads(z.read(request_name))
                if reply['request_sha256']!=d.digest(z.read(request_name)) or reply['nonce']!=request['nonce'] or not reply['ok'] or reply['software_fixture']:raise ValueError('Unverified worker receipt')
                if reply['result']['state']!='complete':raise ValueError('Incomplete worker')
                workers.append({'operation':request['operation'],'probe':request['probe'],'tag':(request.get('best') or {}).get('tag'),'seconds':reply['elapsed_seconds'],'steps':reply['result'].get('steps')})
        prefix='trials/mixed_pilot/generations/'
        sets=[('parent_micro','validation_parent',micro),('parent_rag','parent_actual_rag',rag)]
        for i in [1,2]:sets.extend([(f'epoch_{i:02d}_micro',f'epoch_{i:02d}_validation',micro),(f'epoch_{i:02d}_rag',f'epoch_{i:02d}_actual_rag',rag)])
        sets.append(('final','final_review',final))
        for tag,name,gold in sets:
            rows=read_rows(z,prefix+name+'.jsonl');score=checks.score(rows,gold);scores[tag]=score;targets={r['id']:r for r in gold}
            if not score['complete']:raise ValueError('Incomplete answers')
            for r in rows:
                if r['question']!=targets[r['id']]['messages'][:-1] or r['expected']!=targets[r['id']]['answer']:raise ValueError('Question/teacher differs')
                # Windows reuse serializes existing dicts with sorted keys. The
                # generation SHA was originally computed with role/content order.
                messages=[{'role':m['role'],'content':m['content']} for m in r['question']]
                if r['prompt_sha256']!=d.digest(json.dumps(messages,ensure_ascii=False,separators=(',',':'))):raise ValueError('Prompt SHA differs')
                note=annotations.get(tag,{}).get(r['id'])
                if note:validate_note(note,r)
                details.append({'tag':tag,'row':r,'gold':targets[r['id']],'frozen_check':checks.fact_check(r,targets[r['id']]),'observation':observation(r,targets[r['id']]),'read_answer_note':note})
        if scores['parent_micro']!=trial['parent_micro'] or scores['parent_rag']!=trial['parent_rag']:raise ValueError('Baseline score differs')
        for i,e in enumerate(trial['epochs'],1):
            cur,rg=scores[f'epoch_{i:02d}_micro'],scores[f'epoch_{i:02d}_rag']
            if cur!=e['micro_checks'] or rg!=e['actual_rag_checks']:raise ValueError('Epoch score differs')
            gate=runner.learning_gate(cur,scores['parent_micro'],rg,scores['parent_rag'])
            if gate!=e['training_retention_gate'] or gate['passed']:raise ValueError('Frozen stopping decision differs')
        selected=json.loads(z.read('pilot_full_review.json'))
        if selected['checks']!=scores['final']:raise ValueError('Final review score differs')
        # The 80 saved copies are not another 80 actual generations.
        finalrows=read_rows(z,prefix+'final_review.jsonl');epochrows=read_rows(z,prefix+'epoch_02_validation.jsonl');reuse=json.loads(z.read(prefix+'final_review.reuse.json'));proof=json.loads(z.read(prefix+'final_review.provenance.json'))
        if finalrows[:80]!=epochrows or reuse['source_sha256']!=d.digest(z.read(prefix+'epoch_02_validation.jsonl')):raise ValueError('Reused micro answers differ')
        serialized=b''.join(z.read(prefix+'final_review.jsonl').splitlines(keepends=True)[:80])
        if reuse['target_sha256']!=d.digest(serialized) or reuse['adapter_sha256']!=proof['adapter_sha256'] or reuse['max_new_tokens']!=proof['max_new_tokens']:raise ValueError('Reuse weights/budget proof differs')
        for name in sorted(n for n in z.namelist() if n.endswith('.jsonl')):
            for r in read_rows(z,name):allrows.append((name,r))
        stage=json.loads(z.read('trials/mixed_pilot/cycle_01/stage_status.json'));parent=json.loads(z.read('inheritance_proofs/mixed_pilot.json'));backend=json.loads(z.read('attention_backend.json'))
        if stage['steps']!=194 or stage['planned_steps']!=194 or stage['state']!='complete' or not stage['parent_weights_reused'] or stage['resume_checkpoint'] is not None:raise ValueError('Unexpected formal run')
        if parent['schema']!=runtime.previous.SCHEMA or backend['software_fixture']:raise ValueError('Wrong parent/fixture backend')
        parent_rag_proof=json.loads(z.read(prefix+'parent_actual_rag.provenance.json'))
        if parent['files']['adapter_model.safetensors']!=parent_rag_proof['adapter_sha256'] or parent['optimizer_inherited']:raise ValueError('Parent fit/evaluation weight receipts differ')
        if not parent['source_path'].replace('\\','/').endswith(d.read(runtime.RELEASE)['reviewed_parent']['path']):raise ValueError('Parent source path differs from declared stage')
        for name in z.namelist():
            if name.endswith('.attention_path.json'):
                a=json.loads(z.read(name))
                if a['cuda_calls']<=0 or a['math_fallback_allowed'] or a['software_fixture']:raise ValueError('Unverified actual attention route')
        requests=[json.loads(z.read(n)) for n in z.namelist() if n.endswith('.training_start.json')]
        if any(r['optimizer_steps'] for r in requests):raise ValueError('Unexpected inherited optimizer')
    summary={'schema':'wengua_readonly_v58_review_20261108','upload_sha256':UPLOAD_SHA,'members':len(z.namelist()),'state':state['state'],'formal_updates':194,'full_650_started':False,'generation_rows_stored':len(allrows),'reused_micro_copies':80,'distinct_generated_answer_records_including_smoke':len(allrows)-80,'verified_actual_workers':workers,'parent_recorded_model_sha256':parent['files']['adapter_model.safetensors'],'weights_not_uploaded_hash_not_independently_recomputed':True,'trainer_runtime_seconds':stage['training_metrics']['train_runtime'],'fit_stage_seconds':stage['runtime_seconds'],'peak_reserved_gib':stage['peak_reserved_gib'],'parent_language_loss':stage['parent_validation_loss'],'epochs':trial['epochs'],'frozen_scores':scores,'final_case_reading_scope':dict(Counter(by[r['row']['id']]['skill'] for r in details if r['tag']=='final')),'not_rag_or_sft_input':True,'production_enabled':False,'posthoc_read_answer_review_not_new_blind_validation':True}
    d.write(output/'aggregate.json',summary);d.write(output/'upload_receipt.json',{'sha256':UPLOAD_SHA,'audit_source_sha256':d.digest(Path(__file__).read_bytes()),'frozen_runtime_or_scores_changed':False,'uploaded_instructions_executed':False})
    with gzip.open(output/'answer_observations.jsonl.gz','wt',encoding='utf8') as f:
        for v in details:f.write(d.dumps(v)+'\n')
    if workbook:
        from openpyxl import Workbook,load_workbook
        wb=Workbook(write_only=True);ws=wb.create_sheet('全部原答含复用');ws.append(['文件','ID','完整问题JSON','目标','原答','tokens','输出上限'])
        for name,r in allrows:
            values=[name,r['id'],d.dumps(r['question']),r['expected'],r['answer'],r['generated_tokens'],r['reached_token_limit']]
            if any(isinstance(v,str) and len(v)>32767 for v in values):raise ValueError('Workbook needs complete-text chunking')
            ws.append(values)
        ws=wb.create_sheet('原检查及读答观察');ws.append(['阶段','技能','领域','ID','原检查JSON','观察非新分数','读答注释'])
        for v in details:ws.append([v['tag'],v['gold']['skill'],v['gold']['domain'],v['row']['id'],d.dumps(v['frozen_check']),d.dumps(v['observation']),d.dumps(v['read_answer_note'])])
        ws=wb.create_sheet('边界说明');ws.append(['项目','说明'])
        for k,v in {'运行':'两轮97/194更新；完整650更新未启动。','原答':'487存储记录中80为第二轮复用副本，实际407含1短测。','读答注释':'限定所问主结论、依据与条件的事后审阅；不是专业认证或新盲测。','冻结检查':'保留原分数。表达误判另注，不覆盖原分数。','案例':'192仅第二轮研究候选有答，不能作父权重和第一轮整案进步对照。','用途':'仅评估，禁止作为古籍RAG或SFT教师。'}.items():ws.append([k,v])
        dest=output/'Wengua_V58_逐轮逐题审核.xlsx';wb.save(dest);reader=load_workbook(dest,read_only=True,data_only=True);rows=iter(reader['全部原答含复用'].values);next(rows)
        for (_,expected),got in zip(allrows,rows,strict=True):assert json.loads(got[2])==expected['question'] and got[3]==expected['expected'] and got[4]==expected['answer']
        reader.close()
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path);p.add_argument('--output',type=Path);p.add_argument('--notes',type=Path);p.add_argument('--workbook',action='store_true');p.add_argument('--restore',action='store_true');p.add_argument('--key-file',type=Path);a=p.parse_args()
    if a.restore:
        from wengua_corpus.__main__ import restore
        key=a.key_file.read_text(encoding='utf-8-sig').strip() if a.key_file else input('输入此前教材64位口令（可见）：').strip()
        if not re.fullmatch('[0-9a-fA-F]{64}',key):raise ValueError('Existing delivery key required')
        print(restore(d.read(runtime.ROOT/'docs/model_reviews/20261108_v58/private_review_catalog.json'),bytes.fromhex(key)));return
    if a.archive is None or a.output is None:p.error('--archive and --output required')
    s=audit(a.archive,a.output,notes=a.notes,workbook=a.workbook);print(d.dumps({k:s[k] for k in ['state','formal_updates','generation_rows_stored','reused_micro_copies','distinct_generated_answer_records_including_smoke']}))

if __name__=='__main__':main()
