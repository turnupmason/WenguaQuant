"""Compile private V56 source repairs, audit every target, then seal delivery."""
from pathlib import Path
from collections import Counter
import argparse,gzip,json,sqlite3,zipfile,hashlib,hmac,os
from wengua_course import data as d
from wengua_course_v55 import runtime as old
from . import data,runtime,checks


def compile_course(out):
    runtime.configure()
    from transformers import AutoTokenizer
    from src.wengua.qwen_book_loop_v3 import encode,target_views,generation_rows
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-8B',revision='b968826d9c46dd6066d109eabc6255188de91218',local_files_only=True,trust_remote_code=False)
    parent=old.verify_data();delta=data.compile_data(parent);pack=data.compose(parent,delta);print('V56 source repairs compiled; preparing actual RAG teachers',flush=True)
    # These are actual V56 production retrieval prompts, not legacy prompt
    # copies relabelled as new retrieval.
    from . import corpus
    extra=[];missing=[]
    for r in list(pack['records']):
        if r['provenance'].get('kind') not in ['condition_contrast_identity','reviewed_rule_premise_contrast']:
            continue
        q=r['messages'][-2]['content'].split('\n\n本题来源：')[0]
        packet=corpus.prompt(q,tok,domain=r['domain'],max_context=3328,max_new_tokens=512)
        citations=corpus.covering_citations(packet,r['provenance']['support'],parent['units']) if packet['state']=='ready' else None
        if not citations:
            missing.append({'id':r['id'],'reason':'required_literal_source_not_in_retrieved_packet'});continue
        target=r['answer'].split(' [S',1)[0]+' '+'、'.join('['+c+']' for c in citations)
        row=dict(r);row['messages']=packet['messages']+[{'role':'assistant','content':target}];row['question']=row['messages'][-2]['content'];row['answer']=target;row['id']='v56:rag:'+d.digest(d.dumps(row['messages']))[:24]
        row['provenance']=r['provenance']|{'actual_production_rag_prompt':True,'required_citations':citations,'references':[{'id':x['id'],'citation':x['citation'],'sources':x['sources']} for x in packet['references']],'prompt_budget':packet['prompt_budget'],'runtime_prompt_version':'v56'};extra.append(row)
    delta['records'].extend(extra);pack=data.compose(parent,delta);data.validate(pack)
    token_rows=[];failures=[];oversized=[]
    for index,r in enumerate(pack['records']):
        if index%1000==0:print('V56 complete-target audit',index,'/',len(pack['records']),flush=True)
        for view in target_views(r,0):
            enc,size=encode(view,tok,4096)
            if enc is None or size>3328:oversized.append({'id':r['id'],'tokens':size,'source_retained':True});continue
            token_rows.append({'id':r['id'],'tokens':size,'assistant_tokens':sum(x!=-100 for x in enc['labels'])})
        if r.get('checks') and r['checks']['kind']!='mentions':
            result=checks.fact_check({'answer':r['answer'],'reached_token_limit':False},r)
            if result['status']!='pass':failures.append({'id':r['id'],'skill':r['skill'],'check':r['checks'],'result':result,'answer':r['answer']})
    # Oversized originals are kept for reading/indexing, never cut mid-condition.
    if oversized:raise ValueError('Whole teacher context needs review: '+d.dumps(oversized))
    if failures:
        out.mkdir(parents=True,exist_ok=True);d.write(out/'teacher_failures.json',failures)
        raise ValueError('Teacher slot/condition audit failed: '+str(len(failures)))
    full=runtime.select_development(pack['records'],192);short=runtime.select_development(pack['records'],96,repair=True)
    config=d.read(runtime.CONFIG)
    for label,rows,limit in [('full',full,1024),('repair',short,384)]:
        _,skips=generation_rows(rows,tok,config|{'eval_max_new_tokens':limit})
        if skips:raise ValueError(label+' generation budget requires review: '+d.dumps(skips))
        long=[r['id'] for r in rows if len(tok.encode(r['answer'],add_special_tokens=False))>limit]
        if long:raise ValueError('Full target exceeds generation response budget: '+d.dumps(long))
    delta['fixed_development_ids']=[r['id'] for r in full]
    delta['summary']={'records':len(pack['records']),'splits':dict(Counter(r['split'] for r in pack['records'])),'skills':dict(Counter(r['skill'] for r in pack['records'])),'cases_preserved':len(pack['case_catalog']),'author_steps_preserved':sum(len(c['author_steps']) for c in pack['case_catalog']),'complete_author_reading':sum(r['skill']=='author_reading' for r in pack['records']),'author_context_without_conclusion':sum(r['skill']=='author_context' for r in pack['records']),'all_258_author_inputs_restored':sum(r['skill'] in ['author_reading','author_context'] for r in pack['records'])==258,'reviewed_local_judgment_tasks':sum(r['skill']=='author_local' for r in pack['records']),'actual_new_rag_teachers':len(extra),'actual_rag_missing_sources':len(missing),'final_test_used':False,'old_manual_scores_used':False,'no_whole_source_or_target_truncated':True}
    out.mkdir(parents=True,exist_ok=True)
    with gzip.open(out/'course_delta.json.gz','wt',encoding='utf8') as f:json.dump(delta,f,ensure_ascii=False,separators=(',',':'))
    d.write(out/'summary.json',delta['summary']);d.write(out/'token_audit.json',{'teacher_views':len(token_rows),'max_tokens':max(r['tokens'] for r in token_rows),'target_or_context_truncated':False,'teacher_slot_failures':failures,'oversized':oversized,'rows':token_rows,'rag_missing_sources':missing})
    print(d.dumps(delta['summary']))


def seal(out,key_file):
    from openpyxl import Workbook
    from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes
    from cryptography.hazmat.primitives import padding
    from wengua_corpus import delivery
    parent=old.verify_data();delta=json.loads(gzip.decompress((out/'course_delta.json.gz').read_bytes()));pack=data.compose(parent,delta);data.validate(pack)
    # Source texts keep their existing immutable IDs and SHA. This DB adds a
    # complete lesson/phase/case/material index, not a new truth scoring table.
    path=out/'Wengua_两书课程_V56_完整对照.sqlite3'
    if path.exists():raise FileExistsError('Existing build DB preserved')
    db=sqlite3.connect(path);db.executescript('PRAGMA foreign_keys=ON; CREATE TABLE lessons(id TEXT PRIMARY KEY,domain TEXT,skill TEXT,split TEXT,payload_json TEXT NOT NULL); CREATE TABLE materials(id TEXT PRIMARY KEY,payload_json TEXT NOT NULL); CREATE TABLE teacher_sources(lesson_id TEXT REFERENCES lessons(id),source_id TEXT REFERENCES materials(id),PRIMARY KEY(lesson_id,source_id)); CREATE TABLE cases(case_id TEXT PRIMARY KEY,payload_json TEXT); CREATE TABLE author_steps(case_id TEXT,step TEXT,payload_json TEXT,PRIMARY KEY(case_id,step)); CREATE TABLE dispositions(id INTEGER PRIMARY KEY,payload_json TEXT); CREATE TABLE meta(key TEXT PRIMARY KEY,value_json TEXT);')
    db.executemany('INSERT INTO lessons VALUES(?,?,?,?,?)',[(r['id'],r['domain'],r['skill'],r['split'],d.dumps(r)) for r in pack['records']]);db.executemany('INSERT INTO materials VALUES(?,?)',[(k,d.dumps(v)) for k,v in pack['units'].items()]);db.executemany('INSERT INTO teacher_sources VALUES(?,?)',[(r['id'],sid) for r in pack['records'] for sid in set(r['source_ids'])]);db.executemany('INSERT INTO cases VALUES(?,?)',[(c['case_id'],d.dumps(c)) for c in pack['case_catalog']]);db.executemany('INSERT INTO author_steps VALUES(?,?,?)',[(c['case_id'],str(s['step']),d.dumps(s)) for c in pack['case_catalog'] for s in c['author_steps']]);db.executemany('INSERT INTO dispositions(payload_json) VALUES(?)',[(d.dumps(r),) for r in pack['dispositions']]);db.execute('INSERT INTO meta VALUES(?,?)',('summary',d.dumps(pack['summary'])))
    db.execute('CREATE TABLE coverage(kind TEXT,id TEXT,payload_json TEXT,PRIMARY KEY(kind,id))')
    for kind,items in [('rules',pack['rule_inventory']),('workflow',pack['workflow_plan']),('prior_author_bridge_dispositions',pack['author_bridge_inventory'])]:
        for i,r in enumerate(items):db.execute('INSERT INTO coverage VALUES(?,?,?)',(kind,str(i),d.dumps(r)))
    db.execute('INSERT INTO meta VALUES(?,?)',('original_case_splits',d.dumps(pack['original_case_splits'])));db.commit();assert not db.execute('PRAGMA foreign_key_check').fetchall();assert [json.loads(s) for s, in db.execute('SELECT payload_json FROM lessons ORDER BY rowid')]==pack['records'];db.close()
    wb=Workbook(write_only=True);ws=wb.create_sheet('全量课程');ws.append(['ID','划分','领域','技能','完整对话JSON','完整答案','来源ID','来源及边界JSON'])
    for r in pack['records']:ws.append([r['id'],r['split'],r['domain'],r['skill'],d.dumps(r['messages']),r['answer'],d.dumps(r['source_ids']),d.dumps(r['provenance'])])
    ws=wb.create_sheet('作者前提修订');ws.append(['旧题ID','具体处置','完整处置JSON'])
    for r in pack['dispositions']:ws.append([r['id'],r['action'],d.dumps(r)])
    ws=wb.create_sheet('全部案例目录');ws.append(['案例ID','领域','家族','作者分析数'])
    for c in pack['case_catalog']:ws.append([c['case_id'],c['domain'],c['group'],len(c['author_steps'])])
    ws=wb.create_sheet('完整案例JSON分段');ws.append(['案例ID','片段序号','完整JSON按序拼接'])
    for c in pack['case_catalog']:
        text=d.dumps(c)
        for i in range(0,len(text),30000):ws.append([c['case_id'],i//30000+1,text[i:i+30000]])
    ws=wb.create_sheet('1008作者分析');ws.append(['案例ID','分析序号','来源ID','完整原段'])
    for c in pack['case_catalog']:
        for s in c['author_steps']:ws.append([c['case_id'],s['step'],s['source_id'],s['original']])
    ws=wb.create_sheet('2049规则索引');ws.append(['原索引ID','领域','原文选择坐标','处置'])
    for r in pack['rule_inventory']:ws.append([r['id'],d.dumps(r['domains']),d.dumps(r['parts']),r['disposition']])
    links={sid:[] for sid in pack['units']}
    for r in pack['records']:
        for sid in r['source_ids']:links[sid].append(r['id'])
    ws=wb.create_sheet('材料课程对应');ws.append(['材料ID','书','种类','关联课程数'])
    for sid,u in pack['units'].items():ws.append([sid,u['book_id'],u.get('kind'),len(links[sid])])
    ws=wb.create_sheet('材料课程明细');ws.append(['材料ID','课程ID'])
    for sid,linked in links.items():
        for lesson in linked:ws.append([sid,lesson])
    wb.save(out/'Wengua_两书课程_V56_全量审阅.xlsx')
    recipe=out/'source_recipe';recipe.mkdir(exist_ok=True)
    for name in ['data.py','checks.py','build.py']:(recipe/name).write_bytes((Path(__file__).parent/name).read_bytes())
    (out/'README.txt').write_text('V5.6：完整原阶段阅读、限定局部结论、去重规则组、条件对照。\n全部原材料、194案及1008分析保留。作者阅读和局部填补均为历史教学，非新盘独立预测。短训比较两个已有权重，事实与条件门槛通过后才继续两学习率完整训练。\n',encoding='utf8')
    archive=out.parent/'Wengua_V56_premise_repair.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out).as_posix())
    key=bytes.fromhex(key_file.read_text().strip());enc,mac=delivery.derive_keys(key);iv=os.urandom(16);plain=archive.read_bytes();pad=padding.PKCS7(128).padder();raw=pad.update(plain)+pad.finalize();encryptor=Cipher(algorithms.AES(enc),modes.CBC(iv)).encryptor();body=delivery.MAGIC+iv+encryptor.update(raw)+encryptor.finalize();blob=body+hmac.new(mac,body,hashlib.sha256).digest()
    dest=runtime.ROOT/'data/delivery/course_v56';dest.mkdir(parents=True,exist_ok=True);p=dest/('course_v56_'+d.digest(blob)[:12]+'.enc');p.write_bytes(blob);d.write(dest/'manifest.json',{'file':p.relative_to(runtime.ROOT).as_posix(),'ciphertext_sha256':d.digest(blob),'archive_name':archive.name,'archive_sha256':d.digest(plain),'private_book_materials':True})
    parents={'v54':{'path':'local/course_v54/run1/trials/source_lr_2em05/cycle_01/adapters/step-134','adapter_sha256':'9052361fc7e72a34610fb8e2ca45f9c71c821dacdbedaf4b92773e639bf74f88'},'v55':{'path':'local/course_v55/run1/trials/source_lr_2em05/cycle_01/adapters/step-397','adapter_sha256':'fe22c4ba1efbf6aba198eb3f68725e4f2d155387f3de619f138578db99ce93f5'}}
    release={'schema':data.SCHEMA,'version':'5.6','parent_v55_release_sha256':d.digest(old.RELEASE.read_text()),'delivery_manifest':'data/delivery/course_v56/manifest.json','reviewed_parents':parents,'summary':pack['summary'],'files':{p.relative_to(out).as_posix():d.digest(p.read_bytes()) for p in sorted(out.rglob('*')) if p.is_file()}}
    d.write(runtime.RELEASE,release);print(d.dumps(release['summary']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['compile','seal']);p.add_argument('--output',type=Path,default=runtime.DATA);p.add_argument('--key-file',type=Path);a=p.parse_args()
    if a.command=='compile':compile_course(a.output)
    elif a.key_file:seal(a.output,a.key_file)
    else:p.error('--key-file required for encrypted delivery')


if __name__=='__main__':main()
