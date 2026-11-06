"""Reproducible private delta delivery. Never publish plaintext book material.

Developer entry: --delta is an already source-checked generated delta. Without
it this builds against the restored frozen V53 corpus and cached tokenizer.
The master delivery key is read from a local file, never logged or embedded.
"""
from pathlib import Path
import argparse, gzip, hashlib, hmac, json, os, shutil, sqlite3, zipfile
from . import runtime, data, corpus
from wengua_course import data as d
from wengua_course_v53 import runtime as prior


def build(a):
 if a.review_workbook and not a.review_workbook.is_file():raise FileNotFoundError(a.review_workbook)
 prior.configure();parent=prior.verify_data()
 if a.delta:delta=d.read(a.delta)
 else:
  from transformers import AutoTokenizer
  c=d.read(runtime.CONFIG);tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision'],local_files_only=True,trust_remote_code=False)
  delta=data.build(parent,tok)
 pack=data.compose(parent,delta);data.validate(pack)
 if delta['missing_retrieval']:raise ValueError('Review required: not all proposed lessons have visible literal sources')
 out=a.output.resolve()
 if out.exists():raise FileExistsError('Developer build preserved: '+str(out))
 out.mkdir(parents=True)
 d.write(out/'course_delta.json',delta)
 audit=d.read(a.token_audit)
 if audit['summary']!=delta['summary'] or audit['oversized'] or audit['teacher_slot_failures'] or audit['generation_skips']:raise ValueError('Course/token audit does not agree')
 d.write(out/'token_audit.json',audit)
 d.write(out/'summary.json',delta['summary'])
 d.write(out/'literal_property_sources.json',corpus.properties(parent,corpus.source_library()))
 # Full original books remain in the immutable parent database. This database
 # stores the new lessons, all original material coordinates and all cases.
 db=sqlite3.connect(out/'Wengua_两书课程_V54_增补.sqlite3')
 db.executescript('CREATE TABLE lessons(id TEXT PRIMARY KEY,payload_json TEXT NOT NULL); CREATE TABLE materials(id TEXT PRIMARY KEY,payload_json TEXT NOT NULL); CREATE TABLE cases(case_id TEXT PRIMARY KEY,payload_json TEXT NOT NULL); CREATE TABLE author_steps(case_id TEXT,step_id TEXT,payload_json TEXT NOT NULL,PRIMARY KEY(case_id,step_id)); CREATE TABLE meta(key TEXT PRIMARY KEY,value_json TEXT NOT NULL);')
 db.executemany('INSERT INTO lessons VALUES(?,?)',[(r['id'],d.dumps(r)) for r in delta['records']]);db.executemany('INSERT INTO materials VALUES(?,?)',[(k,d.dumps(v)) for k,v in parent['units'].items()]);db.executemany('INSERT INTO cases VALUES(?,?)',[(r['case_id'],d.dumps(r)) for r in parent['case_catalog']])
 db.executemany('INSERT INTO author_steps VALUES(?,?,?)',[(c['case_id'],str(s['step']),d.dumps(s)) for c in parent['case_catalog'] for s in c['author_steps']])
 db.executemany('INSERT INTO meta VALUES(?,?)',[('parent_release_sha256',d.dumps(d.digest(prior.RELEASE.read_text()))),('parent_full_database',d.dumps('local/course_v53_data/Wengua_两书RAG课程_V53.sqlite3')),('source_books',d.dumps(parent['source_books']))]);db.commit()
 assert [json.loads(p) for p, in db.execute('SELECT payload_json FROM lessons ORDER BY rowid')]==delta['records']
 assert db.execute('SELECT count(*) FROM cases').fetchone()[0]==194 and db.execute('SELECT count(*) FROM author_steps').fetchone()[0]==1008
 db.close()
 from openpyxl import Workbook
 wb=Workbook(write_only=True);sheet=wb.create_sheet('全部增补课程');sheet.append(['ID','划分','技能','领域','题型','完整对话JSON','最终答案','来源ID','条件及来源证明JSON'])
 for r in delta['records']:sheet.append([r['id'],r['split'],r['skill'],r['domain'],r['format'],d.dumps(r['messages']),r['answer'],'\n'.join(r['source_ids']),d.dumps(r['provenance'])])
 sheet=wb.create_sheet('194原案例');sheet.append(['案例','家族','领域','划分','原作者断语','后来反馈','资料类型'])
 for c in parent['case_catalog']:sheet.append([c['case_id'],c['group'],c['domain'],parent['original_case_splits'][c['case_id']],c.get('author_original_judgment'),c.get('later_feedback_separate'),'完整资料保存在数据库；原反馈不当作初断已知'])
 sheet=wb.create_sheet('1008作者分析');sheet.append(['案例','步骤','来源ID','作者原文'])
 for c in parent['case_catalog']:
  for s in c['author_steps']:sheet.append([c['case_id'],s['step'],s['source_id'],s['original']])
 sheet=wb.create_sheet('检查说明');sheet.append(['项目','数据']);sheet.append(['完整来源保留','原7808材料及原两书索引保存在父包，增补数据库保留材料表、194案和1008分析']);sheet.append(['自动检查边界','事实槽位、引用成员和原文坐标通过，不等于全部案例语义获专家核准']);sheet.append(['统计',d.dumps(delta['summary'])]);sheet.append(['原最终test','不训练、不评估，继续封存'])
 wb.save(out/'Wengua_两书课程_V54_全量审阅.xlsx')
 if a.review_workbook:shutil.copy2(a.review_workbook,out/'Wengua_V53_逐题检查与课程错误审核.xlsx')
 recipe=out/'source_recipe';recipe.mkdir()
 for name in ['data.py','corpus.py','checks.py','build.py']:shutil.copy2(Path(__file__).parent/name,recipe/name)
 d.write(recipe/'compiler_hashes.json',{p.name:d.digest(p.read_bytes()) for p in recipe.glob('*.py')})
 (out/'README.txt').write_text('V5.4 两书来源使用修订。默认继承本机V5.3 step256；新阶段新优化器，两学习率各两轮。\n先看全量审阅Excel和V53逐题审核表。原两书、旧模型、原始分数和封存test不改。\n课程、坐标、案例和作者分析逐项入库；自动槽位检查不代表全部语义百分之百正确。\n',encoding='utf8')
 archive=out.parent/'Wengua_V54_source_repair.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for f in sorted(out.rglob('*')):
   if f.is_file():z.write(f,f.relative_to(out).as_posix())
 from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes
 from cryptography.hazmat.primitives import padding
 from wengua_corpus import delivery
 master=bytes.fromhex(a.key_file.read_text().strip());enc,mac=delivery.derive_keys(master);iv=os.urandom(16);pad=padding.PKCS7(128).padder();plain=archive.read_bytes();padded=pad.update(plain)+pad.finalize();encryptor=Cipher(algorithms.AES(enc),modes.CBC(iv)).encryptor();body=delivery.MAGIC+iv+encryptor.update(padded)+encryptor.finalize();blob=body+hmac.new(mac,body,hashlib.sha256).digest()
 dest=runtime.ROOT/'data/delivery/course_v54';dest.mkdir(parents=True,exist_ok=True);name='course_v54_'+d.digest(blob)[:12]+'.enc';(dest/name).write_bytes(blob)
 manifest={'file':(dest/name).relative_to(runtime.ROOT).as_posix(),'ciphertext_sha256':d.digest(blob),'archive_sha256':d.digest(plain),'archive_name':archive.name,'private_book_materials':True,'encryption':'AES-256-CBC+HMAC-SHA256'};d.write(dest/'manifest.json',manifest)
 release={'schema':data.SCHEMA,'version':'5.4','delivery_manifest':'data/delivery/course_v54/manifest.json','parent_v53_release_sha256':d.digest(prior.RELEASE.read_text()),'parent_v53_pack_sha256':d.digest((prior.DATA/'pack.json').read_bytes()),'source_books':parent['source_books'],'summary':delta['summary'],'files':{p.relative_to(out).as_posix():d.digest(p.read_bytes()) for p in sorted(out.rglob('*')) if p.is_file()}}
 d.write(runtime.RELEASE,release);return release


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--delta',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--token-audit',type=Path,required=True);p.add_argument('--review-workbook',type=Path);p.add_argument('--key-file',type=Path,required=True);a=p.parse_args();print(d.dumps(build(a)['summary']))
if __name__=='__main__':main()
