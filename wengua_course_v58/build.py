"""Compile, audit literal premises and whole token budgets, then seal V58."""
from pathlib import Path
from collections import Counter
import argparse,gzip,json,io,zipfile,sqlite3,hmac,hashlib,os,re
from wengua_course import data as d
from wengua_course_v57 import runtime as previous
from . import data,checks,runtime,evaluation,corpus


def compile_course(out):
    runtime.configure()
    from transformers import AutoTokenizer
    from src.wengua.qwen_book_loop_v3 import target_views,encode,generation_rows
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-8B',revision='b968826d9c46dd6066d109eabc6255188de91218',local_files_only=True,trust_remote_code=False)
    from wengua_course_v55.corpus import r as retriever
    parent=previous.verify_data();delta=data.build(parent);delta['old_fixed_micro_ids']=parent['fixed_micro_ids'];pack=data.compose(parent,delta);units=pack['units'];specs={s['key']:s for s in pack['reviewed_specs']}
    data.validate(pack)
    missing=[];new_teachers=[]
    print('V58 source records built',len(pack['records']),flush=True)
    # Actually query the production retriever; never append missing teacher text.
    for r in list(pack['records']):
        if r['provenance'].get('kind')!='explicit_person_condition':continue
        q=r['messages'][-2]['content'].split('\n\n原文依据：',1)[0]
        packet=corpus.prompt(q,tok,domain=r['domain'],max_new_tokens=768)
        citations=corpus.covering_citations(packet,r['provenance']['support'],units) if packet['state']=='ready' else None
        if not citations:missing.append({'id':r['id'],'action':'not_a_teacher_required_source_not_retrieved'});continue
        target=r['answer'].rsplit(' [S',1)[0]+' '+'、'.join('['+s+']' for s in citations)
        rr=dict(r);rr['messages']=packet['messages']+[{'role':'assistant','content':target}];rr.update(id='v58:rag:'+d.digest(d.dumps(rr['messages']))[:24],question=packet['messages'][-1]['content'],answer=target)
        rr['provenance']=r['provenance']|{'kind':'actual_v58_rag_person_teacher','actual_production_rag_prompt':True,'required_citations':citations,'references':[{'id':x['id'],'citation':x['citation'],'sources':x['sources']} for x in packet['references']],'prompt_budget':packet['prompt_budget'],'repair_priority':True};new_teachers.append(rr)
    teacher_lookup={d.dumps(r['messages']):r for r in delta['records']};refreshed=0;added=0
    for teacher in new_teachers:
        existing=teacher_lookup.get(d.dumps(teacher['messages']))
        if existing:
            existing['provenance'].update(actual_production_rag_prompt=True,fresh_v58_retrieval_confirmed=True,fresh_reference_sha256=d.digest(d.dumps(teacher['provenance']['references'])));refreshed+=1
        else:delta['records'].append(teacher);teacher_lookup[d.dumps(teacher['messages'])]=teacher;added+=1
    pack=data.compose(parent,delta)
    probes=[]
    for old in pack['legacy_actual_rag_rows']:
        key=old['id'].split(':')[-1];spec=specs[key];q=spec['question'];print('V58 actual source probe',key,flush=True)
        if key=='health_daughter_father':q='女儿代问父亲疾病，父亲出生年未知；按本章先取什么主用神、兼看哪一宫，年命如何处理？'
        if key=='health_father_self':q='父亲本人问自己的健康；应先按本人求测还是代问父亲的规则取用？'
        packet=corpus.prompt(q,tok,domain=spec['domain'] if spec['domain'] in retriever.DOMAINS else None,max_new_tokens=768)
        required=spec['support'];citations=corpus.covering_citations(packet,required,units) if packet['state']=='ready' else None
        if not citations:raise ValueError('Actual RAG probe lacks literal source: '+key)
        target=re.sub(r'(?:\s|、)*\[S\d+\]','',old['answer']).strip()+' '+'、'.join('['+s+']' for s in citations)
        rr=deep_row(old);rr.update(id='v58:actual_rag:'+key,messages=packet['messages']+[{'role':'assistant','content':target}],question=packet['messages'][-1]['content'],answer=target,checks=evaluation.rag_contract(key),skill='identity' if key.startswith(('health_','exam_')) else 'source_fact',domain=spec['domain'],provenance={'required_citations':citations,'actual_production_rag_prompt':True,'packet':packet,'support':required})
        rr['provenance']['citation_rule_groups']=corpus.citation_groups(packet,key);probes.append(rr)
    delta['v58_actual_rag_rows']=probes
    data.validate(pack)
    failures=[];token_rows=[];oversized=[];source_contract=[]
    for i,r in enumerate(pack['records']):
        if i%1000==0:print('V58 whole source/target audit',i,'/',len(pack['records']),flush=True)
        c=r.get('checks')
        if c and c['kind']!='mentions':
            result=checks.fact_check({'answer':r['answer']},r)
            if result['status']!='pass' or not result['complete']:failures.append({'id':r['id'],'kind':'teacher_fact','result':result})
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
    if failures or oversized:raise ValueError('V58 complete-source teacher audit needs review: '+str((len(failures),len(oversized))))
    small=evaluation.micro(pack);eligible=[];excluded=[];c=d.read(runtime.CONFIG)
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
    pilot=evaluation.pilot_training(pack['records']);views=sum(sum(1 for _ in target_views(r,0)) for r in pilot);train=[r for r in pack['records'] if data.active(r)];full_views=sum(sum(1 for _ in target_views(r,0)) for r in train)
    pilot_by={r['id']:r['skill'] for r in pilot};pilot_tokens=Counter()
    for token_row in token_rows:
        if token_row['id'] in pilot_by:pilot_tokens[pilot_by[token_row['id']]]+=token_row['assistant_tokens']
    delta['summary']={'records':len(pack['records']),'active_training_cores':len(train),'reading_archive_cores':sum(r['skill'] in data.READING for r in pack['records']),'splits':dict(Counter(r['split'] for r in pack['records'])),'skills':dict(Counter(r['skill'] for r in pack['records'])),'materials_preserved':len(pack['units']),'cases_preserved':len(pack['case_catalog']),'author_steps_preserved':sum(len(c['author_steps']) for c in pack['case_catalog']),'complete_author_reading_courses':sum(r['provenance'].get('kind')=='complete_original_author_step' for r in pack['records']),'actual_new_rag_teachers':added,'actual_rag_teacher_refreshes_without_duplicate_rows':refreshed,'actual_rag_probes':len(probes),'micro_questions':len(small),'final_questions':len(final),'pilot_unique_cores':len(pilot),'pilot_assistant_views':views,'pilot_updates':2*((views+15)//16),'pilot_skills':dict(Counter(r['skill'] for r in pilot)),'full_assistant_views_per_epoch':full_views,'full_updates_two_epochs':2*((full_views+15)//16),'pilot_supervised_tokens_by_skill':dict(pilot_tokens),'old_manual_scores_used':False,'final_test_used':False,'model_answers_as_teachers':False,'whole_source_and_target_preserved':True}
    with gzip.open(out/'course_delta.json.gz','wt',encoding='utf8') as f:json.dump(delta,f,ensure_ascii=False,separators=(',',':'))
    with gzip.open(out/'source_contracts.jsonl.gz','wt',encoding='utf8') as f:
        for r in source_contract:f.write(d.dumps(r)+'\n')
    d.write(out/'summary.json',delta['summary']);d.write(out/'token_audit.json',{'teacher_views':len(token_rows),'max_tokens':max(r['tokens'] for r in token_rows),'whole_targets_preserved':True,'oversized':oversized,'failures':failures,'evaluation_reading_exclusions_not_source_deletions':excluded,'rows':token_rows})
    print(d.dumps(delta['summary']),flush=True)


def deep_row(row):
    from copy import deepcopy
    return deepcopy(row)


def seal(out,key_file):
    import zlib
    from openpyxl import Workbook,load_workbook
    from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes
    from cryptography.hazmat.primitives import padding
    from wengua_corpus import delivery
    pack=data.compose(previous.verify_data(),json.loads(gzip.decompress((out/'course_delta.json.gz').read_bytes())));data.validate(pack)
    path=out/'Wengua_两书课程_V58_完整对照.sqlite3'
    if not path.exists():
        db=sqlite3.connect(path);db.executescript('PRAGMA foreign_keys=ON; CREATE TABLE lessons(id TEXT PRIMARY KEY,skill TEXT,split TEXT,active INTEGER,payload_zlib BLOB); CREATE TABLE materials(id TEXT PRIMARY KEY,payload_zlib BLOB); CREATE TABLE teacher_sources(lesson_id TEXT REFERENCES lessons(id),source_id TEXT REFERENCES materials(id),PRIMARY KEY(lesson_id,source_id)); CREATE TABLE cases(case_id TEXT PRIMARY KEY,payload_zlib BLOB); CREATE TABLE author_steps(case_id TEXT,step TEXT,payload_zlib BLOB,PRIMARY KEY(case_id,step)); CREATE TABLE coverage(kind TEXT,id TEXT,payload_zlib BLOB,PRIMARY KEY(kind,id));')
        enc=lambda x:zlib.compress(d.dumps(x).encode(),9);dec=lambda x:json.loads(zlib.decompress(x))
        db.executemany('INSERT INTO lessons VALUES(?,?,?,?,?)',[(r['id'],r['skill'],r['split'],data.active(r),enc(r)) for r in pack['records']]);db.executemany('INSERT INTO materials VALUES(?,?)',[(k,enc(v)) for k,v in pack['units'].items()]);db.executemany('INSERT INTO teacher_sources VALUES(?,?)',[(r['id'],sid) for r in pack['records'] for sid in set(r['source_ids'])]);db.executemany('INSERT INTO cases VALUES(?,?)',[(c['case_id'],enc(c)) for c in pack['case_catalog']]);steps=[(c['case_id'],str(s['step']),s) for c in pack['case_catalog'] for s in c['author_steps']];db.executemany('INSERT INTO author_steps VALUES(?,?,?)',[(cid,n,enc(s)) for cid,n,s in steps])
        for kind,items in [('rules',pack['rule_inventory']),('workflows',pack['workflow_plan'])]:db.executemany('INSERT INTO coverage VALUES(?,?,?)',[(kind,str(i),enc(v)) for i,v in enumerate(items)])
        db.commit();assert not db.execute('PRAGMA foreign_key_check').fetchall()
        assert [dec(v) for v, in db.execute('SELECT payload_zlib FROM lessons ORDER BY rowid')]==pack['records']
        assert {k:dec(v) for k,v in db.execute('SELECT id,payload_zlib FROM materials')}==pack['units']
        assert [dec(v) for v, in db.execute('SELECT payload_zlib FROM cases ORDER BY rowid')]==pack['case_catalog']
        assert [dec(v) for v, in db.execute('SELECT payload_zlib FROM author_steps ORDER BY rowid')]==[s for _,_,s in steps];db.close()
    else:
        db=sqlite3.connect(path);dec=lambda x:json.loads(zlib.decompress(x));steps=[(c['case_id'],str(t['step']),t) for c in pack['case_catalog'] for t in c['author_steps']]
        assert [dec(v) for v, in db.execute('SELECT payload_zlib FROM lessons ORDER BY rowid')]==pack['records']
        assert {k:dec(v) for k,v in db.execute('SELECT id,payload_zlib FROM materials')}==pack['units']
        assert [dec(v) for v, in db.execute('SELECT payload_zlib FROM cases ORDER BY rowid')]==pack['case_catalog']
        assert [dec(v) for v, in db.execute('SELECT payload_zlib FROM author_steps ORDER BY rowid')]==[t for _,_,t in steps]
        assert not db.execute('PRAGMA foreign_key_check').fetchall();db.close()
    wb=Workbook(write_only=True);ws=wb.create_sheet('全部课程');ws.append(['ID','划分','技能','领域','参与训练','完整对话JSON','完整答案','来源合同JSON'])
    for r in pack['records']:
        values=[r['id'],r['split'],r['skill'],r['domain'],data.active(r),d.dumps(r['messages']),r['answer'],d.dumps(r['provenance'])]
        if any(isinstance(x,str) and len(x)>32767 for x in values):raise ValueError('Complete workbook cell exceeds boundary')
        ws.append(values)
    ws=wb.create_sheet('194案例JSON分段');ws.append(['案例ID','序号','按序拼接的原案例JSON'])
    for case in pack['case_catalog']:
        text=d.dumps(case)
        for i in range(0,len(text),30000):ws.append([case['case_id'],i//30000+1,text[i:i+30000]])
    ws=wb.create_sheet('1008作者分析');ws.append(['案例ID','原阶段','来源ID','原文'])
    for case in pack['case_catalog']:
        for s in case['author_steps']:ws.append([case['case_id'],s['step'],s['source_id'],s['original']])
    dest=out/'Wengua_两书课程_V58_全量审阅.xlsx';wb.save(dest);reader=load_workbook(dest,read_only=True,data_only=True);rows=iter(reader['全部课程'].values);next(rows)
    for expected,actual in zip(pack['records'],rows,strict=True):assert json.loads(actual[5])==expected['messages'] and actual[6]==expected['answer'] and json.loads(actual[7])==expected['provenance'] and actual[4]==data.active(expected)
    chunks={}
    for cid,n,text in list(reader['194案例JSON分段'].values)[1:]:chunks.setdefault(cid,[]).append((n,text))
    assert [json.loads(''.join(t for _,t in sorted(chunks[c['case_id']]))) for c in pack['case_catalog']]==pack['case_catalog']
    assert list(reader['1008作者分析'].values)[1:]==[(cid,s['step'],s['source_id'],s['original']) for cid,_,s in steps];reader.close()
    d.write(out/'storage_audit.json',{'lessons_equal':len(pack['records']),'materials_equal':len(pack['units']),'cases_equal':len(pack['case_catalog']),'author_steps_equal':len(steps),'sqlite_payload_encoding':'zlib UTF-8 JSON; lossless; query metadata uncompressed','foreign_key_errors':0,'workbook_full_dialogues_answers_sources_cases_steps_equal':True})
    recipe=out/'source_recipe';recipe.mkdir(exist_ok=True)
    for name in ['data.py','checks.py','build.py','evaluation.py','corpus.py']:(recipe/name).write_bytes((Path(__file__).parent/name).read_bytes())
    mem=io.BytesIO()
    with zipfile.ZipFile(mem,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out).as_posix())
    plain=mem.getvalue();master=bytes.fromhex(key_file.read_text().strip());key,mac=delivery.derive_keys(master);iv=os.urandom(16);pad=padding.PKCS7(128).padder();raw=pad.update(plain)+pad.finalize();encryptor=Cipher(algorithms.AES(key),modes.CBC(iv)).encryptor();body=delivery.MAGIC+iv+encryptor.update(raw)+encryptor.finalize();blob=body+hmac.new(mac,body,hashlib.sha256).digest()
    folder=runtime.ROOT/'data/delivery/course_v58';folder.mkdir(parents=True,exist_ok=True);cipher=folder/('course_v58_'+d.digest(blob)[:12]+'.enc');cipher.write_bytes(blob);d.write(folder/'manifest.json',{'file':cipher.relative_to(runtime.ROOT).as_posix(),'ciphertext_sha256':d.digest(blob),'archive_name':'Wengua_V58_question_contract.zip','archive_sha256':d.digest(plain),'private_book_materials':True})
    d.write(runtime.RELEASE,{'schema':data.SCHEMA,'version':'5.8','parent_v57_release_sha256':d.digest(previous.RELEASE.read_text()),'delivery_manifest':'data/delivery/course_v58/manifest.json','reviewed_parent':{'path':'local/course_v57/run1/trials/mixed_pilot/cycle_01/adapters/step-65','steps':65,'ancestor_v55_sha256':'fe22c4ba1efbf6aba198eb3f68725e4f2d155387f3de619f138578db99ce93f5','research_continuation_not_professional_approval':True,'adapter_sha256_recorded_at_local_snapshot':True},'summary':pack['summary'],'files':{p.relative_to(out).as_posix():d.digest(p.read_bytes()) for p in sorted(out.rglob('*')) if p.is_file()}})
    print('V58 exact storage and encrypted delivery verified before install; bytes',len(blob))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['compile','seal']);p.add_argument('--output',type=Path,default=runtime.DATA);p.add_argument('--key-file',type=Path);a=p.parse_args()
    if a.command=='compile':compile_course(a.output)
    elif a.key_file:seal(a.output,a.key_file)
    else:p.error('--key-file required')

if __name__=='__main__':main()
