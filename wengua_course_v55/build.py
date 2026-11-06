"""Private V55 data/SQLite/Excel delivery. Never publish book plaintext or keys."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,gzip,json,sqlite3,zipfile,shutil,os,hmac,hashlib
from wengua_course import data as d
from wengua_course_v54 import runtime as parent_runtime
from . import runtime,data


def coverage(pack):
 links=defaultdict(list)
 for row in pack['records']:
  for sid in row['source_ids']:links[sid].append(row['id'])
 materials=[{'source_id':sid,'book_id':u['book_id'],'kind':u.get('kind'),'linked_lessons':links[sid],'disposition':'linked_to_active_course' if links[sid] else 'source_retained_not_replayed_in_this_course'} for sid,u in pack['units'].items()]
 cases=[];steps=[]
 for c in pack['case_catalog']:
  related=[r for r in pack['records'] if c['case_id'] in r['case_ids']];split=pack['original_case_splits'][c['case_id']]
  cases.append({'case_id':c['case_id'],'domain':c['domain'],'group':c['group'],'split':split,'linked_lessons':[r['id'] for r in related],'author_steps':len(c['author_steps']),'disposition':'sealed_original_test' if split=='test' else 'source_and_course_links_recorded','independent_initial_forecast_task_verified':False})
  for s in c['author_steps']:steps.append({'case_id':c['case_id'],'step':s['step'],'source_id':s['source_id'],'original':s['original'],'linked_lessons':links[s['source_id']],'disposition':'sealed_original_test' if split=='test' else 'linked_source_in_course' if links[s['source_id']] else 'source_catalog_only','exact_full_step_is_teacher_target':any(d.compact(s['original']) in d.compact(r['answer']) for r in related)})
 domains=[]
 for domain in sorted({c['domain'] for c in pack['case_catalog']}|set(pack['summary']['domain_rule_coverage'])):
  domains.append({'domain':domain,'source_cases':sum(c['domain']==domain for c in pack['case_catalog']),'source_author_steps':sum(len(c['author_steps']) for c in pack['case_catalog'] if c['domain']==domain),'source_rules_in_this_domain':sum(domain in r['domains'] for r in pack['rule_inventory']),'literal_rule_lessons':sum(r['domain']==domain and r['skill']=='domain_rule' for r in pack['records']),'workflow_groups':sum(f['domain']==domain for f in pack['workflow_plan']),'train_lessons':sum(r['domain']==domain and r['split']=='train' for r in pack['records']),'development_lessons':sum(r['domain']==domain and r['split']=='validation' for r in pack['records']),'all_workflow_steps_expert_approved':False})
 return {'materials':materials,'cases':cases,'author_steps':steps,'domains':domains,'rules':pack['rule_inventory'],'author_bridges':pack['author_bridge_inventory']}


def build(a):
 parent=parent_runtime.verify_data();delta=json.loads(gzip.decompress(a.delta.read_bytes()));pack=data.compose(parent,delta);data.validate(pack);audit=d.read(a.token_audit)
 if audit['summary']!=delta['summary'] or audit['oversized'] or audit['teacher_slot_failures'] or audit['generation_skips']:raise ValueError('Teacher/token audit is incomplete or inconsistent')
 if a.output.exists():raise FileExistsError('Developer output preserved: '+str(a.output))
 out=a.output;out.mkdir(parents=True);(out/'course_delta.json.gz').write_bytes(a.delta.read_bytes());d.write(out/'token_audit.json',audit);d.write(out/'summary.json',delta['summary']);register=coverage(pack);d.write(out/'coverage.json',register)
 # Every material/rule/case/author step has a disposition. Presence in a source
 # registry, teacher input and teacher target are deliberately distinct fields.
 db=sqlite3.connect(out/'Wengua_两书课程_V55_完整对照.sqlite3');db.execute('PRAGMA foreign_keys=ON');db.executescript('CREATE TABLE lessons(id TEXT PRIMARY KEY,domain TEXT,skill TEXT,split TEXT,payload_json TEXT NOT NULL); CREATE TABLE materials(id TEXT PRIMARY KEY,payload_json TEXT NOT NULL); CREATE TABLE teacher_sources(lesson_id TEXT REFERENCES lessons(id),source_id TEXT REFERENCES materials(id),PRIMARY KEY(lesson_id,source_id)); CREATE TABLE cases(case_id TEXT PRIMARY KEY,payload_json TEXT NOT NULL); CREATE TABLE author_steps(case_id TEXT,step TEXT,payload_json TEXT NOT NULL,PRIMARY KEY(case_id,step)); CREATE TABLE coverage(kind TEXT,id TEXT,payload_json TEXT NOT NULL,PRIMARY KEY(kind,id)); CREATE TABLE meta(key TEXT PRIMARY KEY,value_json TEXT NOT NULL);')
 db.executemany('INSERT INTO materials VALUES(?,?)',[(sid,d.dumps(u)) for sid,u in pack['units'].items()]);db.executemany('INSERT INTO lessons VALUES(?,?,?,?,?)',[(r['id'],r['domain'],r['skill'],r['split'],d.dumps(r)) for r in pack['records']]);db.executemany('INSERT INTO teacher_sources VALUES(?,?)',[(r['id'],sid) for r in pack['records'] for sid in set(r['source_ids'])]);db.executemany('INSERT INTO cases VALUES(?,?)',[(c['case_id'],d.dumps(c)) for c in pack['case_catalog']]);db.executemany('INSERT INTO author_steps VALUES(?,?,?)',[(c['case_id'],str(s['step']),d.dumps(s)) for c in pack['case_catalog'] for s in c['author_steps']])
 for kind,records in register.items():
  for i,row in enumerate(records):db.execute('INSERT INTO coverage VALUES(?,?,?)',(kind,str(i),d.dumps(row)))
 db.execute('INSERT INTO meta VALUES(?,?)',('source_parent_v54_release_sha256',d.dumps(d.digest(parent_runtime.RELEASE.read_text()))));db.commit()
 assert db.execute('SELECT count(*) FROM lessons').fetchone()[0]==len(pack['records']) and not db.execute('PRAGMA foreign_key_check').fetchall();assert [json.loads(s) for s, in db.execute('SELECT payload_json FROM lessons ORDER BY rowid')]==pack['records'];db.close()
 from openpyxl import Workbook
 from openpyxl.cell import WriteOnlyCell
 from openpyxl.styles import Font,PatternFill
 wb=Workbook(write_only=True)
 def sheet(name,headers):
  ws=wb.create_sheet(name);ws.freeze_panes='A2'
  for i,width in enumerate([30,18,22,28,70,85,55,55,75,75],1):ws.column_dimensions[chr(64+i)].width=width
  cells=[]
  for text in headers:
   c=WriteOnlyCell(ws,value=text);c.font=Font(bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='244B5A');cells.append(c)
  ws.append(cells);return ws
 ws=sheet('全部核心课程',['ID','划分','技能','领域','当前提问预览','完整答案','来源ID','完整对话JSON','来源证明JSON'])
 for r in pack['records']:
  q=r['messages'][-2]['content'];preview=q.split('用户问题：',1)[-1].split('\n本题',1)[0]
  ws.append([r['id'],r['split'],r['skill'],r['domain'],preview,r['answer'],'\n'.join(r['source_ids']),d.dumps(r['messages']),d.dumps(r['provenance'])])
 for title,key,columns in [('领域课程覆盖','domains',['domain','source_cases','source_author_steps','source_rules_in_this_domain','literal_rule_lessons','workflow_groups','train_lessons','development_lessons','all_workflow_steps_expert_approved']),('2049规则处置','rules',['id','domains','disposition','parts']),('全部材料处置','materials',['source_id','book_id','kind','disposition','linked_lessons']),('194案例处置','cases',['case_id','domain','split','author_steps','disposition','linked_lessons','independent_initial_forecast_task_verified']),('1008作者分析','author_steps',['case_id','step','source_id','disposition','original','linked_lessons','exact_full_step_is_teacher_target']),('作者证据任务处置','author_bridges',['source_core_id','case_id','author_step','disposition','not_independent_fresh_chart_forecast'])]:
  ws=sheet(title,columns)
  for r in register[key]:ws.append([d.dumps(r.get(c)) if isinstance(r.get(c),(list,dict)) else r.get(c) for c in columns])
 ws=sheet('原书规则组',['领域','书','组ID','本组完整原文与来源JSON'])
 for f in pack['workflow_plan']:ws.append([f['domain'],f['book_id'],f['id'],d.dumps(f['rules'])])
 wb.save(out/'Wengua_两书课程_V55_全量覆盖审阅.xlsx')
 recipe=out/'source_recipe';recipe.mkdir()
 for name in ['data.py','checks.py','sourceplan.py','corpus.py','build.py']:shutil.copy2(Path(__file__).parent/name,recipe/name)
 d.write(recipe/'compiler_hashes.json',{p.name:d.digest(p.read_bytes()) for p in recipe.glob('*.py')})
 (out/'README.txt').write_text('V5.5：分步基础、领域规则组、作者证据推结论与案例复盘。\n默认继承已审核V5.4高学习率第1轮step134，新教材新优化器。两学习率各两轮，逐轮检查新192题、旧128题与原18题实际RAG。\n原194案、1008分析、所有材料和2049规则逐项存库并有处置。source保留、教师输入和教师目标分别记录；不声称全部领域流程已获专家核准。原最终test封存。\n',encoding='utf8')
 archive=out.parent/'Wengua_V55_domain_reasoning.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for p in sorted(out.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(out).as_posix())
 from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes
 from cryptography.hazmat.primitives import padding
 from wengua_corpus import delivery
 master=bytes.fromhex(a.key_file.read_text().strip());enc,mac=delivery.derive_keys(master);iv=os.urandom(16);plain=archive.read_bytes();pad=padding.PKCS7(128).padder();raw=pad.update(plain)+pad.finalize();encryptor=Cipher(algorithms.AES(enc),modes.CBC(iv)).encryptor();body=delivery.MAGIC+iv+encryptor.update(raw)+encryptor.finalize();blob=body+hmac.new(mac,body,hashlib.sha256).digest();dest=runtime.ROOT/'data/delivery/course_v55';dest.mkdir(parents=True,exist_ok=True);path=dest/('course_v55_'+d.digest(blob)[:12]+'.enc');path.write_bytes(blob)
 d.write(dest/'manifest.json',{'file':path.relative_to(runtime.ROOT).as_posix(),'ciphertext_sha256':d.digest(blob),'archive_name':archive.name,'archive_sha256':d.digest(plain),'private_book_materials':True})
 parent_proof=d.read(a.parent_proof)
 release={'schema':data.SCHEMA,'version':'5.5','delivery_manifest':'data/delivery/course_v55/manifest.json','parent_v54_release_sha256':d.digest(parent_runtime.RELEASE.read_text()),'reviewed_parent_adapter_sha256':parent_proof['adapter_sha256'],'reviewed_parent':'V54/source_lr_2em05/step-134_research_initialization_only','source_books':pack['source_books'],'summary':pack['summary'],'files':{p.relative_to(out).as_posix():d.digest(p.read_bytes()) for p in sorted(out.rglob('*')) if p.is_file()}};d.write(runtime.RELEASE,release);return release


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--delta',type=Path,required=True);p.add_argument('--token-audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--key-file',type=Path,required=True);p.add_argument('--parent-proof',type=Path,required=True);a=p.parse_args();print(d.dumps(build(a)['summary']))
if __name__=='__main__':main()
