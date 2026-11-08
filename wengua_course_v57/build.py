"""Compile, audit literal premises and whole token budgets, then seal V57."""
from pathlib import Path
from collections import Counter
import argparse,gzip,json,io,zipfile,sqlite3,hmac,hashlib,os,re
from wengua_course import data as d
from wengua_course_v56 import runtime as previous
from . import data,checks,runtime,evaluation,corpus


def compile_course(out):
    runtime.configure()
    from transformers import AutoTokenizer
    from src.wengua.qwen_book_loop_v3 import target_views,encode,generation_rows
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-8B',revision='b968826d9c46dd6066d109eabc6255188de91218',local_files_only=True,trust_remote_code=False)
    from wengua_course_v55.corpus import r as retriever
    parent=previous.verify_data();delta=data.build(parent);pack=data.compose(parent,delta);units=pack['units'];specs={s['key']:s for s in pack['reviewed_specs']}
    missing=[];new_teachers=[]
    print('V57 source records built',len(pack['records']),flush=True)
    # Actually query the production retriever; never append missing teacher text.
    for r in list(pack['records']):
        if r['provenance'].get('kind')!='explicit_person_condition':continue
        q=r['messages'][-2]['content'].split('\n\n原文依据：',1)[0]
        packet=corpus.prompt(q,tok,domain=r['domain'],max_new_tokens=768)
        citations=corpus.covering_citations(packet,r['provenance']['support'],units) if packet['state']=='ready' else None
        if not citations:missing.append({'id':r['id'],'action':'not_a_teacher_required_source_not_retrieved'});continue
        target=r['answer'].rsplit(' [S',1)[0]+' '+'、'.join('['+s+']' for s in citations)
        rr=dict(r);rr['messages']=packet['messages']+[{'role':'assistant','content':target}];rr.update(id='v57:rag:'+d.digest(d.dumps(rr['messages']))[:24],question=packet['messages'][-1]['content'],answer=target)
        rr['provenance']=r['provenance']|{'kind':'actual_v57_rag_person_teacher','actual_production_rag_prompt':True,'required_citations':citations,'references':[{'id':x['id'],'citation':x['citation'],'sources':x['sources']} for x in packet['references']],'prompt_budget':packet['prompt_budget'],'repair_priority':True};new_teachers.append(rr)
    delta['records'].extend(new_teachers);pack=data.compose(parent,delta)
    probes=[]
    for old in pack['legacy_actual_rag_rows']:
        key=old['id'].split(':')[-1];spec=specs[key];q=spec['question'];print('V57 actual source probe',key,flush=True)
        if key=='health_daughter_father':q='女儿代问父亲疾病，父亲出生年未知；按本章先取什么主用神、兼看哪一宫，年命如何处理？'
        if key=='health_father_self':q='父亲本人问自己的健康；应先按本人求测还是代问父亲的规则取用？'
        packet=corpus.prompt(q,tok,domain=spec['domain'] if spec['domain'] in retriever.DOMAINS else None,max_new_tokens=768)
        required=spec['support'];citations=corpus.covering_citations(packet,required,units) if packet['state']=='ready' else None
        if not citations:raise ValueError('Actual RAG probe lacks literal source: '+key)
        target=re.sub(r'(?:\s|、)*\[S\d+\]','',old['answer']).strip()+' '+'、'.join('['+s+']' for s in citations)
        rr=deep_row(old);rr.update(id='v57:actual_rag:'+key,messages=packet['messages']+[{'role':'assistant','content':target}],question=packet['messages'][-1]['content'],answer=target,checks=evaluation.rag_contract(key),skill='identity' if key.startswith(('health_','exam_')) else 'source_fact',domain=spec['domain'],provenance={'required_citations':citations,'actual_production_rag_prompt':True,'packet':packet,'support':required})
        probes.append(rr)
    delta['v57_actual_rag_rows']=probes
    data.validate(pack)
    failures=[];token_rows=[];oversized=[];source_contract=[]
    for i,r in enumerate(pack['records']):
        if i%1000==0:print('V57 whole source/target audit',i,'/',len(pack['records']),flush=True)
        c=r.get('checks')
        if c and c['kind']!='mentions':
            result=checks.fact_check({'answer':r['answer']},r)
            if result['status']!='pass':failures.append({'id':r['id'],'kind':'teacher_fact','result':result})
        contracts=r['provenance'].get('turn_contracts')
        if not contracts and r['provenance'].get('role_transition'):
            transition=r['provenance']['role_transition'];contracts=[{'kind':'person_contract','answer':transition['before']},c]
        if contracts:
            turns=[m for m in r['messages'] if m['role']=='assistant']
            if len(turns)!=len(contracts):raise ValueError('Turn contract count differs')
            for turn,contract in zip(turns,contracts):
                result=checks.fact_check({'answer':turn['content']},{'checks':contract})
                if result['status']!='pass' or not result['complete']:failures.append({'id':r['id'],'kind':'teacher_turn','result':result})
        for view in target_views(r,0):
            enc,size=encode(view,tok,4096)
            if enc is None or size>4096:oversized.append({'id':r['id'],'skill':r['skill'],'tokens':size});continue
            token_rows.append({'id':r['id'],'tokens':size,'assistant_tokens':sum(v!=-100 for v in enc['labels'])})
        source_contract.append({'id':r['id'],'split':r['split'],'skill':r['skill'],'case_ids':r['case_ids'],'question_sha256':d.digest(d.dumps(r['messages'][:-1])),'answer_sha256':d.digest(r['answer']),'source_fragments':[{'source_id':s['source_id'],'quote_sha256':d.digest(s['quote'])} for s in r['provenance'].get('support',[])],'check':c,'semantic_scope':'limited_source_facts' if c and c['kind']!='mentions' else 'literal_source_reading_not_independent_prediction'})
    for r in probes:
        result=checks.fact_check({'answer':r['answer']},r)
        if result['status']!='pass' or not result['complete']:failures.append({'id':r['id'],'kind':'rag_teacher','result':result})
    out.mkdir(parents=True,exist_ok=True);d.write(out/'build_failures.json',{'fact_or_turn_failures':failures,'oversized':oversized,'missing_rag_teachers':missing})
    if failures or oversized:raise ValueError('V57 complete-source teacher audit needs review: '+str((len(failures),len(oversized))))
    small=evaluation.micro(pack['records']);eligible=[];excluded=[];c=d.read(runtime.CONFIG)
    for r in pack['records']:
        if r['split']!='validation':continue
        _,skips=generation_rows([r],tok,c)
        if skips or len(tok.encode(r['answer'],add_special_tokens=False))>c['eval_max_new_tokens']:excluded.append(r['id'])
        else:eligible.append(r)
    if not {r['id'] for r in small}.issubset({r['id'] for r in eligible}):raise ValueError('Micro exceeds complete generation budget')
    final=evaluation.full(eligible,small);delta['fixed_micro_ids']=[r['id'] for r in small];delta['fixed_final_ids']=[r['id'] for r in final]
    # Expected answers are not truncated for either training or generation.
    for label,rows in [('micro',small),('actual_rag',probes),('final',final)]:
        _,skips=generation_rows(rows,tok,c)
        if skips:raise ValueError('Whole '+label+' prompt budget exceeded')
    pilot=evaluation.pilot_training(pack['records']);views=sum(sum(1 for _ in target_views(r,0)) for r in pilot);train=[r for r in pack['records'] if r['split']=='train'];full_views=sum(sum(1 for _ in target_views(r,0)) for r in train)
    delta['summary']={'records':len(pack['records']),'splits':dict(Counter(r['split'] for r in pack['records'])),'skills':dict(Counter(r['skill'] for r in pack['records'])),'materials_preserved':len(pack['units']),'cases_preserved':len(pack['case_catalog']),'author_steps_preserved':sum(len(c['author_steps']) for c in pack['case_catalog']),'complete_author_reading_courses':sum(r['provenance'].get('kind')=='complete_original_author_step' for r in pack['records']),'actual_new_rag_teachers':len(new_teachers),'actual_rag_probes':len(probes),'micro_questions':len(small),'final_questions':len(final),'pilot_unique_cores':len(pilot),'pilot_assistant_views':views,'pilot_updates':(views+15)//16,'pilot_skills':dict(Counter(r['skill'] for r in pilot)),'full_assistant_views_per_epoch':full_views,'full_updates_two_epochs':2*((full_views+15)//16),'old_manual_scores_used':False,'final_test_used':False,'model_answers_as_teachers':False,'whole_source_and_target_preserved':True}
    with gzip.open(out/'course_delta.json.gz','wt',encoding='utf8') as f:json.dump(delta,f,ensure_ascii=False,separators=(',',':'))
    with gzip.open(out/'source_contracts.jsonl.gz','wt',encoding='utf8') as f:
        for r in source_contract:f.write(d.dumps(r)+'\n')
    d.write(out/'summary.json',delta['summary']);d.write(out/'token_audit.json',{'teacher_views':len(token_rows),'max_tokens':max(r['tokens'] for r in token_rows),'whole_targets_preserved':True,'oversized':oversized,'failures':failures,'evaluation_reading_exclusions_not_source_deletions':excluded,'rows':token_rows})
    print(d.dumps(delta['summary']),flush=True)


def deep_row(row):
    from copy import deepcopy
    return deepcopy(row)


def seal(out,key_file):
    from openpyxl import Workbook,load_workbook
    from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes
    from cryptography.hazmat.primitives import padding
    from wengua_corpus import delivery
    pack=data.compose(previous.verify_data(),json.loads(gzip.decompress((out/'course_delta.json.gz').read_bytes())));data.validate(pack)
    path=out/'Wengua_两书课程_V57_完整对照.sqlite3'
    if path.exists():raise FileExistsError('Preserve previous build DB')
    db=sqlite3.connect(path);db.executescript('PRAGMA foreign_keys=ON; CREATE TABLE lessons(id TEXT PRIMARY KEY,skill TEXT,split TEXT,payload_json TEXT); CREATE TABLE materials(id TEXT PRIMARY KEY,payload_json TEXT); CREATE TABLE teacher_sources(lesson_id TEXT REFERENCES lessons(id),source_id TEXT REFERENCES materials(id),PRIMARY KEY(lesson_id,source_id)); CREATE TABLE cases(case_id TEXT PRIMARY KEY,payload_json TEXT); CREATE TABLE author_steps(case_id TEXT,step TEXT,payload_json TEXT,PRIMARY KEY(case_id,step)); CREATE TABLE coverage(kind TEXT,id TEXT,payload_json TEXT,PRIMARY KEY(kind,id)); CREATE TABLE dispositions(id INTEGER PRIMARY KEY,payload_json TEXT);')
    db.executemany('INSERT INTO lessons VALUES(?,?,?,?)',[(r['id'],r['skill'],r['split'],d.dumps(r)) for r in pack['records']]);db.executemany('INSERT INTO materials VALUES(?,?)',[(k,d.dumps(v)) for k,v in pack['units'].items()]);db.executemany('INSERT INTO teacher_sources VALUES(?,?)',[(r['id'],sid) for r in pack['records'] for sid in set(r['source_ids'])]);db.executemany('INSERT INTO cases VALUES(?,?)',[(c['case_id'],d.dumps(c)) for c in pack['case_catalog']]);db.executemany('INSERT INTO author_steps VALUES(?,?,?)',[(c['case_id'],str(s['step']),d.dumps(s)) for c in pack['case_catalog'] for s in c['author_steps']]);db.executemany('INSERT INTO dispositions(payload_json) VALUES(?)',[(d.dumps(r),) for r in pack['dispositions']])
    for kind,items in [('rules',pack['rule_inventory']),('workflow',pack['workflow_plan'])]:
        db.executemany('INSERT INTO coverage VALUES(?,?,?)',[(kind,str(i),d.dumps(v)) for i,v in enumerate(items)])
    db.commit();assert not db.execute('PRAGMA foreign_key_check').fetchall();assert [json.loads(x) for x, in db.execute('SELECT payload_json FROM lessons ORDER BY rowid')]==pack['records']
    assert dict((k,json.loads(v)) for k,v in db.execute('SELECT id,payload_json FROM materials'))==pack['units']
    assert [json.loads(v) for v, in db.execute('SELECT payload_json FROM cases ORDER BY rowid')]==pack['case_catalog']
    assert [json.loads(v) for v, in db.execute('SELECT payload_json FROM author_steps ORDER BY rowid')]==[s for c in pack['case_catalog'] for s in c['author_steps']]
    assert db.execute('SELECT count(*) FROM teacher_sources').fetchone()[0]==sum(len(set(r['source_ids'])) for r in pack['records']);db.close()
    wb=Workbook(write_only=True);ws=wb.create_sheet('全部课程');ws.append(['ID','划分','技能','领域','完整对话JSON','完整答案','完整来源及前提合同JSON'])
    for r in pack['records']:
        values=[r['id'],r['split'],r['skill'],r['domain'],d.dumps(r['messages']),r['answer'],d.dumps(r['provenance'])]
        if any(isinstance(x,str) and len(x)>32767 for x in values):raise ValueError('Workbook text needs source-safe chunking')
        ws.append(values)
    ws=wb.create_sheet('全部案例JSON分段');ws.append(['案例ID','序号','原案例JSON按序拼接'])
    for case in pack['case_catalog']:
        text=d.dumps(case)
        for i in range(0,len(text),30000):ws.append([case['case_id'],i//30000+1,text[i:i+30000]])
    ws=wb.create_sheet('1008作者分析');ws.append(['案例ID','原阶段','来源ID','原文'])
    for case in pack['case_catalog']:
        for s in case['author_steps']:ws.append([case['case_id'],s['step'],s['source_id'],s['original']])
    ws=wb.create_sheet('材料课程关联');ws.append(['材料ID','课程ID'])
    for r in pack['records']:
        for sid in r['source_ids']:ws.append([sid,r['id']])
    ws=wb.create_sheet('课程修订处置');ws.append(['完整处置JSON'])
    for r in pack['dispositions']:ws.append([d.dumps(r)])
    wb.save(out/'Wengua_两书课程_V57_全量审阅.xlsx')
    reader=load_workbook(out/'Wengua_两书课程_V57_全量审阅.xlsx',read_only=True,data_only=True)
    course_rows=iter(reader['全部课程'].values);next(course_rows)
    for expected,actual in zip(pack['records'],course_rows,strict=True):
        assert list(actual[:4])==[expected[k] for k in ['id','split','skill','domain']]
        assert json.loads(actual[4])==expected['messages'] and actual[5]==expected['answer'] and json.loads(actual[6])==expected['provenance']
    chunks={}
    for cid,number,text in list(reader['全部案例JSON分段'].values)[1:]:chunks.setdefault(cid,[]).append((number,text))
    assert [json.loads(''.join(t for _,t in sorted(chunks[c['case_id']]))) for c in pack['case_catalog']]==pack['case_catalog']
    for actual,expected in zip(list(reader['1008作者分析'].values)[1:],[(c['case_id'],s['step'],s['source_id'],s['original']) for c in pack['case_catalog'] for s in c['author_steps']],strict=True):assert tuple(actual)==tuple(expected)
    reader.close()
    d.write(out/'database_workbook_audit.json',{'lessons_readback_equal':len(pack['records']),'materials_readback_equal':len(pack['units']),'cases_readback_equal':len(pack['case_catalog']),'author_steps_readback_equal':sum(len(c['author_steps']) for c in pack['case_catalog']),'foreign_key_errors':0,'workbook_complete_messages_answers_provenance_readback_equal':True,'workbook_full_cases_and_author_steps_readback_equal':True})
    recipe=out/'source_recipe';recipe.mkdir(exist_ok=True)
    for name in ['data.py','checks.py','build.py','evaluation.py']:(recipe/name).write_bytes((Path(__file__).parent/name).read_bytes())
    (out/'README.txt').write_text('V5.7：具体摘录去重、双方前提覆盖、完整同阶段作者阅读、人物条件及资料修订多轮。\n只用两书；原194案、1008分析保留，原test家族封存。阅读复盘不是独立预测。\n试训含案例与领域课；小套检查筛选后再完整训练，最终开放案例需审阅。\n',encoding='utf8')
    mem=io.BytesIO()
    with zipfile.ZipFile(mem,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out).as_posix())
    plain=mem.getvalue();master=bytes.fromhex(key_file.read_text().strip());key,mac=delivery.derive_keys(master);iv=os.urandom(16);pad=padding.PKCS7(128).padder();raw=pad.update(plain)+pad.finalize();enc=Cipher(algorithms.AES(key),modes.CBC(iv)).encryptor();body=delivery.MAGIC+iv+enc.update(raw)+enc.finalize();blob=body+hmac.new(mac,body,hashlib.sha256).digest()
    dest=runtime.ROOT/'data/delivery/course_v57';dest.mkdir(parents=True,exist_ok=True);cipher=dest/('course_v57_'+d.digest(blob)[:12]+'.enc');cipher.write_bytes(blob);d.write(dest/'manifest.json',{'file':cipher.relative_to(runtime.ROOT).as_posix(),'ciphertext_sha256':d.digest(blob),'archive_name':'Wengua_V57_source_contract.zip','archive_sha256':d.digest(plain),'private_book_materials':True})
    d.write(runtime.RELEASE,{'schema':data.SCHEMA,'version':'5.7','parent_v56_release_sha256':d.digest(previous.RELEASE.read_text()),'delivery_manifest':'data/delivery/course_v57/manifest.json','reviewed_parent':d.read(previous.RELEASE)['reviewed_parents']['v55'],'reusable_v55_baseline':{'path':'local/course_v56/run1/trials/repair_v55/generations/parent_v55_unchanged.jsonl','sha256':'1ab8d1616e83ec55485ed0118db0d8459af74264dcbbd56d05861149224804d9','adapter_sha256':'fe22c4ba1efbf6aba198eb3f68725e4f2d155387f3de619f138578db99ce93f5','max_new_tokens':1024,'source_upload_sha256':'aa042c22c14fe4d7caf10efdadfc2fd12caddaacee01f0b2bc0ecadd54fa0ac4','not_training_input':True},'summary':pack['summary'],'files':{p.relative_to(out).as_posix():d.digest(p.read_bytes()) for p in sorted(out.rglob('*')) if p.is_file()}})
    print('V57 private source delivery sealed',len(plain),'bytes')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['compile','seal']);p.add_argument('--output',type=Path,default=runtime.DATA);p.add_argument('--key-file',type=Path);a=p.parse_args()
    if a.command=='compile':compile_course(a.output)
    elif a.key_file:seal(a.output,a.key_file)
    else:p.error('--key-file required for sealing')

if __name__=='__main__':main()
