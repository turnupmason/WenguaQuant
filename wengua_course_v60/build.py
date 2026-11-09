"""Compile exact source contracts, whole-token audit, database and readable Excel."""
from collections import Counter
from pathlib import Path
import gzip, io, json, zipfile, zlib, sqlite3, hmac, hashlib, os
from wengua_course import data as d
from wengua_course_v59 import runtime as previous
from . import data, runtime, evaluation, checks


def compile_course(out):
    runtime.configure()
    from transformers import AutoTokenizer
    from src.wengua.qwen_book_loop_v3 import target_views, encode
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-8B',revision='b968826d9c46dd6066d109eabc6255188de91218',local_files_only=True,trust_remote_code=False)
    parent=previous.verify_data(); delta=data.build(parent); pack=data.compose(parent,delta)
    micro=evaluation.micro(pack); final=evaluation.final(pack)
    delta.update(fixed_micro_ids=[r['id'] for r in micro],fixed_final_ids=[r['id'] for r in final])
    # Actual RAG packets keep their literal retrieved evidence; teaching uses
    # the new response instruction without changing original retention inputs.
    for r in pack['records']:
        if r['provenance'].get('v60_training_allowed') and r['provenance'].get('actual_production_rag_prompt'):
            r['provenance']['v60_retrieval_sources_retained']=True
    data.validate(pack)
    audit=[]; failures=[]; oversized=[]; turn_count=0
    for i,r in enumerate(pack['records']):
        if i%1000==0:print('V60 source and full token audit',i,'/',len(pack['records']),flush=True)
        if r['provenance'].get('v60_training_allowed'):
            c=r.get('checks') or {}
            if c and c['kind']!='mentions':
                result=checks.fact_check({'answer':r['answer']},r)
                if result['status']!='pass' or not result['complete']:failures.append({'id':r['id'],'result':result})
            contracts=r['provenance'].get('turn_contracts',[])
            if contracts:
                answers=[m['content'] for m in r['messages'] if m['role']=='assistant']
                if len(answers)!=len(contracts):raise ValueError('Turn contract mismatch')
                for a,c in zip(answers,contracts):
                    turn_count+=1;res=checks.fact_check({'answer':a},{'checks':c})
                    if res['status']!='pass' or not res['complete']:failures.append({'id':r['id'],'turn_result':res})
        sizes=[]
        for view in target_views(r,0):
            encoded,size=encode(view,tok,4096); sizes.append(size)
            if encoded is None or size>4096:
                if r['provenance'].get('v60_training_allowed'):
                    oversized.append({'id':r['id'],'tokens':size,'action':'archive_whole_example_not_truncated'})
                    r['provenance'].update(v60_training_allowed=False,v60_disposition='archive_whole_example_over_context')
        audit.append({'id':r['id'],'sizes':sizes,'training':data.active(r),'answer_sha256':d.digest(r['answer']),
                      'dialogue_sha256':d.digest(d.dumps(r['messages'])),'source_fragments':[
                          {'source_id':s['source_id'],'sha256':d.digest(s['quote'])} for s in r['provenance'].get('support',[])]})
    if failures:
        out.mkdir(parents=True,exist_ok=True);d.write(out/'teacher_failures.json',failures)
        raise ValueError(str(len(failures))+' finite teacher/turn contract failures; see local report')
    pilot=evaluation.pilot_training(pack['records']); coverage=data.coverage(pack)
    pilot_views=sum(sum(1 for _ in target_views(r,0)) for r in pilot)
    full_views=sum(sum(1 for _ in target_views(r,0)) for r in pack['records'] if data.active(r))
    delta['summary']={'schema':data.SCHEMA,'registered':len(pack['records']),'active_train':sum(data.active(r) for r in pack['records']),
                      'materials':len(pack['units']),'cases':len(pack['case_catalog']),
                      'author_steps':len(pack['v60_author_ledger']),'original_rules':len(pack['rule_inventory']),
                      'source_workflows':len(pack['workflow_plan']),'pilot_cores':len(pilot),'pilot_supervised_views':pilot_views,'pilot_updates':(pilot_views+15)//16,
                      'full_supervised_views':full_views,'full_updates_one_epoch':(full_views+15)//16,
                      'pilot_rates':[1e-6,3e-6,1e-5],'pilot_epochs':1,'full_epochs':1,
                      'closed_micro':len(micro),'actual_rag':len(pack['v60_actual_rag_rows']),'final_review':len(final),
                      'official_base_control':len(runtime.control_rows(pack|delta)),
                      'masked_inference_active':0,'whole_over_budget_archived':len(oversized),'multi_turn_contracts_checked':turn_count,
                      'condition_decision_pairs':sum(r['provenance'].get('kind')=='source_condition_decision_pair' for r in pack['records']),
                      'case_replay_train':sum(data.active(r) and r['skill']=='case_replay' for r in pack['records']),
                      'coverage':coverage,'source_only':True,'professional_semantics_certified':False,'production_enabled':False}
    out.mkdir(parents=True,exist_ok=True)
    (out/'course_delta.json.gz').write_bytes(gzip.compress(d.dumps(delta).encode(),mtime=0))
    d.write(out/'summary.json',delta['summary']);d.write(out/'source_token_audit.json',{'rows':audit,'oversized':oversized,'teacher_contract_failures':failures})
    pack.update(delta);store(out,pack);return pack


def store(out,pack):
    """Compressed, lossless objects; normalised IDs and source links are queryable."""
    db=out/'Wengua_两书课程_V60_来源对照.sqlite3'
    if db.exists():raise FileExistsError('Preserve existing reviewed database')
    con=sqlite3.connect(db)
    con.executescript('CREATE TABLE objects(kind TEXT,id TEXT,payload BLOB,sha256 TEXT,PRIMARY KEY(kind,id));'
                      'CREATE TABLE lessons(id TEXT PRIMARY KEY,domain TEXT,split TEXT,skill TEXT,training INTEGER);'
                      'CREATE TABLE support(lesson_id TEXT,source_id TEXT,quote_sha256 TEXT);'
                      'CREATE TABLE cases(id TEXT PRIMARY KEY,domain TEXT,event_family TEXT,phase_split TEXT,author_step_count INTEGER);'
                      'CREATE TABLE author_analysis(case_id TEXT,step INTEGER,source_id TEXT,original TEXT,source_sha256 TEXT,PRIMARY KEY(case_id,step));')
    expected={}
    def put(kind,id,obj):
        raw=d.dumps(obj).encode();sha=d.digest(raw);expected[(kind,str(id))]=sha
        con.execute('INSERT INTO objects VALUES (?,?,?,?)',(kind,str(id),zlib.compress(raw),sha))
    for r in pack['records']:
        put('lesson',r['id'],r);con.execute('INSERT INTO lessons VALUES (?,?,?,?,?)',(r['id'],r['domain'],r['split'],r['skill'],int(data.active(r))))
        for s in r['provenance'].get('support',[]):con.execute('INSERT INTO support VALUES (?,?,?)',(r['id'],s['source_id'],d.digest(s['quote'])))
    for sid,u in pack['units'].items():put('material',sid,u)
    for c in pack['case_catalog']:
        put('case',c['case_id'],c)
        con.execute('INSERT INTO cases VALUES (?,?,?,?,?)',(c['case_id'],c['domain'],c['original_event_family'],c['phase_split'],len(c['author_steps'])))
        for s in c['author_steps']:con.execute('INSERT INTO author_analysis VALUES (?,?,?,?,?)',(c['case_id'],s['step'],s['source_id'],s['original'],d.digest(s['original'])))
    for i,r in enumerate(pack['rule_inventory']):put('rule',i,r)
    for i,r in enumerate(pack['workflow_plan']):put('workflow',i,r)
    for i,r in enumerate(pack['v60_author_ledger']):put('author_disposition',i,r)
    put('manifest','sealed_test',pack['sealed_original_test_manifest']);con.commit()
    for kind,id,payload,sha in con.execute('SELECT kind,id,payload,sha256 FROM objects'):
        if d.digest(zlib.decompress(payload))!=sha or sha!=expected[(kind,id)]:raise ValueError('SQLite roundtrip differs')
    if con.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('SQLite integrity check failed')
    counts=dict(con.execute('SELECT kind,count(*) FROM objects GROUP BY kind'))
    normalized={name:con.execute('SELECT count(*) FROM '+name).fetchone()[0] for name in ['cases','author_analysis']}
    if normalized!={'cases':len(pack['case_catalog']),'author_analysis':len(pack['v60_author_ledger'])}:raise ValueError('Normalized source tables incomplete')
    for text,sha in con.execute('SELECT original,source_sha256 FROM author_analysis'):
        if d.digest(text)!=sha:raise ValueError('Normalized author source differs')
    con.close()
    from openpyxl import Workbook,load_workbook
    wb=Workbook(write_only=True);ws=wb.create_sheet('课程审阅');headers=['ID','划分','领域','类型','参与训练','完整问题','完整回答','来源合同','处置'];ws.append(headers)
    for r in pack['records']:
        values=[r['id'],r['split'],d.DOMAIN_ZH.get(r['domain'],r['domain']),r['skill'],data.active(r),
                r['messages'][-2]['content'],r['answer'],d.dumps(r['provenance'].get('support',[])),r['provenance'].get('v60_disposition','')]
        if any(isinstance(v,str) and len(v)>32767 for v in values):raise ValueError('Excel cell would truncate; database remains authoritative')
        ws.append(values)
    ws=wb.create_sheet('1008作者分析处置');ws.append(['案例','领域','分析步','划分','处置','来源ID','原文SHA','课程ID'])
    for r in pack['v60_author_ledger']:ws.append([r['case_id'],r['domain'],r['step'],r['split'],r['action'],r['source_id'],r['source_sha256'],d.dumps(r['lesson_ids'])])
    ws=wb.create_sheet('领域覆盖');ws.append(['领域','案例','分析步','训练','验证','封存案例','作者复盘训练'])
    for k,v in data.coverage(pack).items():ws.append([k,*[v[x] for x in ['cases','source_steps','train','validation','sealed_cases','replay_train']]])
    ws=wb.create_sheet('全部194案例总览');ws.append(['案例ID','领域','原事件家族','原划分','作者分析步数','主盘来源JSON','相关课程视图数'])
    for c in pack['case_catalog']:
        chart=d.dumps(c['primary_printed_chart'])
        if len(chart)>32767:raise ValueError('Chart overview needs explicitly split storage')
        ws.append([c['case_id'],d.DOMAIN_ZH.get(c['domain'],c['domain']),c['original_event_family'],c['phase_split'],len(c['author_steps']),chart,sum(c['case_id'] in r.get('case_ids',[]) for r in pack['records'])])
    ws=wb.create_sheet('案例完整来源JSON分段');ws.append(['案例ID','分段号','总段数','完整来源数据（按序拼接）'])
    for c in pack['case_catalog']:
        text=d.dumps(c);parts=[text[i:i+20000] for i in range(0,len(text),20000)]
        for i,part in enumerate(parts):ws.append([c['case_id'],i+1,len(parts),part])
    ws=wb.create_sheet('1008条作者原文');ws.append(['案例ID','领域','原分析步','来源ID','原文分段号','总段数','原书分析全文','原文SHA256'])
    for c in pack['case_catalog']:
        for s in c['author_steps']:
            text=s['original'];parts=[text[i:i+20000] for i in range(0,len(text),20000)]
            for i,part in enumerate(parts):ws.append([c['case_id'],d.DOMAIN_ZH.get(c['domain'],c['domain']),s['step'],s['source_id'],i+1,len(parts),part,d.digest(text)])
    excel=out/'Wengua_V60_课程与领域审阅.xlsx';wb.save(excel)
    check=load_workbook(excel,read_only=True,data_only=True);rows=iter(check['课程审阅'].values);next(rows)
    for r,values in zip(pack['records'],rows,strict=True):
        if values[0]!=r['id'] or values[5]!=r['messages'][-2]['content'] or values[6]!=r['answer']:raise ValueError('Excel roundtrip differs')
    restored={};restored_steps={}
    overview=list(check['全部194案例总览'].values)[1:]
    if len(overview)!=len(pack['case_catalog']) or len({r[0] for r in overview})!=len(overview):raise ValueError('Case overview incomplete')
    for id,index,total,part in list(check['案例完整来源JSON分段'].values)[1:]:restored.setdefault(id,[]).append((index,part))
    for id,domain,step,sid,index,total,part,sha in list(check['1008条作者原文'].values)[1:]:restored_steps.setdefault((id,step),[]).append((index,part))
    for c in pack['case_catalog']:
        if ''.join(s for _,s in sorted(restored[c['case_id']]))!=d.dumps(c):raise ValueError('Complete case source Excel differs')
        for s in c['author_steps']:
            if ''.join(t for _,t in sorted(restored_steps[(c['case_id'],s['step'])]))!=s['original']:raise ValueError('Author original Excel differs')
    check.close();d.write(out/'storage_audit.json',{'sqlite_roundtrip':True,'excel_question_answer_roundtrip':True,'excel_all_case_json_roundtrip':len(restored),'excel_all_author_originals_roundtrip':len(restored_steps),'counts':counts,'normalized_source_tables':normalized,'all_source_bytes_retained':True})


def seal(out,key_file):
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes
    from wengua_corpus import delivery
    files={p.relative_to(out).as_posix():d.digest(p.read_bytes()) for p in sorted(out.rglob('*')) if p.is_file()}
    mem=io.BytesIO()
    with zipfile.ZipFile(mem,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(out/name,name)
    plain=mem.getvalue();master=bytes.fromhex(key_file.read_text().strip());key,mac=delivery.derive_keys(master);iv=os.urandom(16)
    pad=padding.PKCS7(128).padder();raw=pad.update(plain)+pad.finalize();enc=Cipher(algorithms.AES(key),modes.CBC(iv)).encryptor()
    body=delivery.MAGIC+iv+enc.update(raw)+enc.finalize();blob=body+hmac.new(mac,body,hashlib.sha256).digest()
    folder=runtime.ROOT/'data/delivery/course_v60';folder.mkdir(parents=True,exist_ok=True)
    cipher=folder/('course_v60_'+d.digest(blob)[:12]+'.enc');cipher.write_bytes(blob)
    manifest={'file':cipher.relative_to(runtime.ROOT).as_posix(),'ciphertext_sha256':d.digest(blob),'archive_name':'Wengua_V60_multidomain_course.zip','archive_sha256':d.digest(plain),'private_book_materials':True}
    d.write(folder/'manifest.json',manifest)
    release={'schema':data.SCHEMA,'version':'6.0','parent_v59_release_sha256':d.digest(previous.RELEASE.read_text()),
             'reviewed_parent':d.read(previous.RELEASE)['reviewed_parent'],'delivery_manifest':'data/delivery/course_v60/manifest.json',
             'summary':d.read(out/'summary.json'),'files':files}
    d.write(runtime.RELEASE,release)
    # Authenticate/decrypt the published bytes before telling the user to run.
    k,iv,ciphertext=delivery.authenticate(blob,master);dec=Cipher(algorithms.AES(k),modes.CBC(iv)).decryptor();unpad=padding.PKCS7(128).unpadder()
    recovered=unpad.update(dec.update(ciphertext)+dec.finalize())+unpad.finalize()
    if recovered!=plain:raise ValueError('Delivery encryption roundtrip differs')
    return release


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--key-file',type=Path,required=True);a=p.parse_args()
    compile_course(a.output);release=seal(a.output,a.key_file);print(d.dumps(release['summary']))

if __name__=='__main__':main()
