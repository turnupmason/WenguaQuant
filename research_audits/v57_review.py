"""Read-only V57 run audit. Never changes the frozen course, gate or scores.

Bounded wording observations are post hoc diagnostics, not a replacement
professional evaluator, new blind test, RAG document or training teacher.
"""
from pathlib import Path
from collections import Counter
import argparse,gzip,json,re,zipfile
from wengua_course import data as d
from wengua_course_v57 import runtime,checks,runner,evaluation
from research_audits.v55_review import inspect_archive,read_rows

UPLOAD_SHA='84db863383f43baaccd52307577b093a25c9116c8541b2a630f323ee13f97333'


def wording_observation(row,gold):
    text=row['answer'];c=gold.get('checks') or {};labels=[]
    pattern=r'(?:应改(?:写)?(?:为|成)?|门与宫|门和宫)(受生|受克|比和|生|克)(?=[。；，,：:\s]|$)'
    for m in re.finditer(pattern,text):
        if checks.asserted(text,m.start(),m.end()):labels.append(m[1])
    expected=c.get('relation') if c.get('kind')=='door_facts' else c.get('answer')
    result={'new_explicit_relation_claims':labels,'explicit_relation_conflicts':[x for x in labels if isinstance(expected,str) and x!=expected],
            'output_limit':bool(row.get('reached_token_limit')),'professional_approval':False,'posthoc_not_new_score':True}
    if c.get('kind')=='property':
        symbol,element=c['symbol'],c['element'];before=re.split('依据|\\[S\\d+\\]',text)[0].strip('。；; 、')
        simple=before in [element,'属'+element,'属于'+element]
        named=bool(re.search(re.escape(symbol)+r'[^。；;]{0,20}(?:归于|同是|亦是|属|属于|为|是)'+element,text))
        result['natural_property_candidate']=simple or named
    if c.get('kind')=='number_pair':
        # The two numbers may be right while an added heaven/earth claim is wrong.
        result['inverted_heaven_earth_explanation']=bool(re.search(r'天[二四六八十][生生成]|地[一三五七九][成生成]',text))
    clauses=[s.strip() for s in re.split('[。\n]',text) if len(s.strip())>=20]
    result['maximum_repeated_long_clause']=max(Counter(clauses).values(),default=0)
    return result


def validate_annotation(note,row):
    if note['id']!=row['id'] or note['answer_sha256']!=d.digest(row['answer']):raise ValueError('Read-answer annotation no longer matches the original answer')
    if note['verdict'] not in ['correct','partial','wrong','unusable']:raise ValueError('Unknown bounded review verdict')
    if not note['reason']:raise ValueError('Missing review reason')
    return note


def audit(archive,output,*,annotations=None,workbook=False):
    archive,output=Path(archive),Path(output);output.mkdir(parents=True,exist_ok=True)
    if d.digest(archive.read_bytes())!=UPLOAD_SHA:raise ValueError('This audit is bound to the actual V57 upload')
    runtime.configure();pack=runtime.verify_data();gold=runtime.choose_development(pack);rag=pack['v57_actual_rag_rows'];details=[];allrows=[];workers=[]
    notes=d.read(annotations) if annotations else {};summaries={}
    with zipfile.ZipFile(archive) as z:
        inspect_archive(z)
        if z.testzip() is not None:raise ValueError('ZIP CRC differs')
        state=json.loads(z.read('run_status.json'));contract=json.loads(z.read('sweep_contract.json'));trial=state['trials']['mixed_pilot']
        if contract['code_sha256']!=runtime.hashes() or contract['dataset_release_sha256']!=d.digest(runtime.RELEASE.read_text()):raise ValueError('Frozen source/release differs')
        if contract['original_test_used'] or set(state['trials'])!={'mixed_pilot'} or state['state']!='needs_course_review':raise ValueError('Unexpected run scope')
        for name in z.namelist():
            if name.endswith('.result.json'):
                reply=json.loads(z.read(name));request_name=name.replace('.result.json','.json');request=json.loads(z.read(request_name))
                if reply['request_sha256']!=d.digest(z.read(request_name)) or reply['nonce']!=request['nonce'] or not reply['ok'] or reply['software_fixture']:raise ValueError('Nonactual or invalid worker receipt')
                if reply['result']['state']!='complete':raise ValueError('Incomplete worker')
                workers.append({'operation':request['operation'],'probe':request['probe'],'seconds':reply['elapsed_seconds'],'steps':reply['result'].get('steps')})
        prefix='trials/mixed_pilot/generations/'
        sets=[('parent_micro','validation_parent.jsonl',gold),('epoch_micro','epoch_01_validation.jsonl',gold),('parent_rag','parent_actual_rag.jsonl',rag),('epoch_rag','epoch_01_actual_rag.jsonl',rag)]
        for tag,name,targets in sets:
            values=read_rows(z,prefix+name);summaries[tag]=checks.score(values,targets);by={r['id']:r for r in targets}
            if len(values)!=len(targets):raise ValueError('Missing answer')
            for r in values:
                g=by[r['id']]
                if r['prompt_sha256']!=d.digest(json.dumps(r['question'],ensure_ascii=False,separators=(',',':'))):raise ValueError('Prompt SHA differs')
                note=validate_annotation(notes[tag][r['id']],r) if annotations else None
                details.append({'tag':tag,'gold':g,'original':r,'frozen_check':checks.fact_check(r,g),'wording_observation':wording_observation(r,g),'read_answer_annotation':note})
        e=trial['epochs'][0]
        for saved,new in [(trial['parent_micro'],summaries['parent_micro']),(trial['parent_rag'],summaries['parent_rag']),(e['micro_checks'],summaries['epoch_micro']),(e['actual_rag_checks'],summaries['epoch_rag'])]:
            if saved!=new:raise ValueError('Stored frozen score differs on replay')
        gate=runner.learning_gate(summaries['epoch_micro'],summaries['parent_micro'],summaries['epoch_rag'],summaries['parent_rag'])
        if gate!=e['training_retention_gate'] or gate['passed']:raise ValueError('Stored stopping decision differs')
        for name in sorted(n for n in z.namelist() if n.endswith('.jsonl')):
            for r in read_rows(z,name):allrows.append({'file':name,'row':r})
        stage=json.loads(z.read('trials/mixed_pilot/cycle_01/stage_status.json'));proof=json.loads(z.read('inheritance_proofs/mixed_pilot.json'))
        if proof['files']['adapter_model.safetensors']!=d.read(runtime.RELEASE)['reviewed_parent']['adapter_sha256']:raise ValueError('Wrong recorded parent')
        if stage['state']!='complete' or stage['steps']!=stage['planned_steps'] or stage['steps']!=65 or not stage['parent_weights_reused']:raise ValueError('Unexpected formal training stage')
        backend=json.loads(z.read('attention_backend.json'))
        if backend['software_fixture'] or backend['selected_backend']!='EFFICIENT_ATTENTION':raise ValueError('Expected actual fused CUDA route')
        for name in z.namelist():
            if name.endswith('.attention_path.json'):
                a=json.loads(z.read(name))
                if a['software_fixture'] or a['cuda_calls']<=0 or a['math_fallback_allowed']:raise ValueError('Unverified attention execution')
    pilot=evaluation.pilot_training(pack['records']);ids={r['id'] for r in pilot};by={r['id']:r for r in pilot};tok=d.read(runtime.DATA/'token_audit.json');tokens=Counter()
    for r in tok['rows']:
        if r['id'] in ids:tokens[by[r['id']]['skill']]+=r['assistant_tokens']
    summary={'schema':'wengua_readonly_v57_review_20261108','source_upload_sha256':UPLOAD_SHA,'run_state':state['state'],'formal_steps':stage['steps'],'full_phase_started':False,'raw_generation_rows_including_smoke':len(allrows),'reviewed_comparable_answers':len(details),'verified_actual_workers':len(workers),'recorded_gpu':backend['device'],'peak_reserved_gib':stage['peak_reserved_gib'],'train_runtime_seconds':stage['training_metrics']['train_runtime'],'parent_validation_loss':stage['parent_validation_loss'],'epoch_validation_loss':stage['validation_loss'],'frozen_retention_gate':gate,'frozen_scores':summaries,'pilot_teacher_tokens':dict(tokens),'pilot_teacher_tokens_total':sum(tokens.values()),'posthoc_review_not_new_model_score':True,'professional_approval':False,'not_rag_or_sft_input':True,'weights_not_uploaded_actual_parent_hash_is_provenance_record_only':True}
    d.write(output/'aggregate.json',summary);d.write(output/'verified_workers.json',workers)
    d.write(output/'upload_receipt.json',{'upload_sha256':UPLOAD_SHA,'files':len(z.namelist()),'audit_code_sha256':d.digest(Path(__file__).read_bytes()),'uploaded_instructions_executed':False,'frozen_runtime_and_old_scores_changed':False})
    with gzip.open(output/'all_answer_observations.jsonl.gz','wt',encoding='utf8') as f:
        for v in details:f.write(d.dumps(v)+'\n')
    if workbook:
        from openpyxl import Workbook,load_workbook
        wb=Workbook(write_only=True);ws=wb.create_sheet('全部原答含短测');ws.append(['原文件','ID','完整问题JSON','原目标','原答','输出到上限'])
        for v in allrows:
            r=v['row'];ws.append([v['file'],r['id'],d.dumps(r['question']),r['expected'],r['answer'],r['reached_token_limit']])
        ws=wb.create_sheet('164逐答复核');ws.append(['阶段','技能','ID','原检查','有限复核结论','读答说明','补充观察不是新分数'])
        for v in details:
            n=v['read_answer_annotation'] or {};ws.append([v['tag'],v['gold']['skill'],v['original']['id'],d.dumps(v['frozen_check']),n.get('verdict'),n.get('reason'),d.dumps(v['wording_observation'])])
        ws=wb.create_sheet('运行与原门槛');ws.append(['完整运行及门槛JSON']);ws.append([d.dumps({k:v for k,v in summary.items() if k!='frozen_scores'})])
        dest=output/'Wengua_V57_全部原答及逐题复核.xlsx';wb.save(dest);reader=load_workbook(dest,read_only=True,data_only=True)
        rows=iter(reader['全部原答含短测'].values);next(rows)
        for expected,got in zip(allrows,rows,strict=True):
            r=expected['row'];assert json.loads(got[2])==r['question'] and got[3]==r['expected'] and got[4]==r['answer']
        assert len(list(reader['164逐答复核'].values))==len(details)+1;reader.close()
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path);p.add_argument('--output',type=Path);p.add_argument('--annotations',type=Path);p.add_argument('--workbook',action='store_true');p.add_argument('--restore',action='store_true');p.add_argument('--key-file',type=Path);a=p.parse_args()
    if a.restore:
        from wengua_corpus.__main__ import restore
        value=a.key_file.read_text(encoding='utf-8-sig').strip() if a.key_file else input('输入此前教材64位口令（可见）：').strip()
        if not re.fullmatch('[0-9a-fA-F]{64}',value):raise ValueError('Expected the existing delivery key')
        print(restore(d.read(runtime.ROOT/'docs/model_reviews/20261108_v57/private_review_catalog.json'),bytes.fromhex(value)));return
    if a.archive is None or a.output is None:p.error('--archive and --output required for audit')
    r=audit(a.archive,a.output,annotations=a.annotations,workbook=a.workbook);print(d.dumps({k:r[k] for k in ['run_state','formal_steps','full_phase_started','raw_generation_rows_including_smoke','reviewed_comparable_answers','verified_actual_workers']}))

if __name__=='__main__':main()
