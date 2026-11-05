"""Restore and search the independent private book corpus. No model is loaded."""
import argparse,gzip,json,os,re,zipfile,sqlite3
from pathlib import Path
from contextlib import closing
from . import book_rag as r,delivery

def library(path):return json.loads(gzip.decompress(Path(path).read_bytes()))
def decrypt(key,iv,cipher):
 if os.name=='nt':return delivery.windows_decrypt(key,iv,cipher)
 try:
  from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes
  from cryptography.hazmat.primitives import padding
 except ImportError as ex:raise RuntimeError('Mac/Linux restoration requires cryptography; Windows uses built-in .NET') from ex
 d=Cipher(algorithms.AES(key),modes.CBC(iv)).decryptor();p=d.update(cipher)+d.finalize();u=padding.PKCS7(128).unpadder();return u.update(p)+u.finalize()

def restore(entry,key):
 m=r.read(r.ROOT/entry['manifest']);blob=(r.ROOT/m['file']).read_bytes()
 if r.digest(blob)!=m['ciphertext_sha256']:raise ValueError('Cipher changed')
 archive=delivery.install(blob,key,r.ROOT/'local/downloads'/m['archive_name'],expected_sha256=m['archive_sha256'],decrypt=decrypt)
 out=r.private(r.ROOT/entry['destination'])
 with zipfile.ZipFile(archive) as z:
  if len(set(z.namelist()))!=len(z.namelist()) or set(z.namelist())!=set(entry['files']):raise ValueError('Archive members differ')
  payload={n:z.read(n) for n in z.namelist()}
  for n,h in entry['files'].items():
   if Path(n).is_absolute() or '..' in Path(n).parts or '\\' in n or r.digest(payload[n])!=h:raise ValueError('Unsafe/changed member')
  for n,raw in payload.items():
   p=out/n
   if p.exists() and p.read_bytes()!=raw:raise ValueError('Existing source preserved: '+str(p))
   p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 return out

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
 q=s.add_parser('restore');q.add_argument('--key-file',type=Path);q.add_argument('--app-root',type=Path,default=r.ROOT.parent/'WenguaApp');q.add_argument('--include-course-v51',action='store_true')
 q=s.add_parser('search');q.add_argument('--question',required=True);q.add_argument('--book');q.add_argument('--study',action='store_true')
 a=p.parse_args(argv);catalog=r.read(r.ROOT/'data/corpus/catalog.json')
 if a.command=='restore':
  keytext=a.key_file.read_text().strip() if a.key_file else input('输入此前教材口令（可见）：').strip()
  if not re.fullmatch('[0-9a-fA-F]{64}',keytext):raise ValueError('Expected existing 64-character delivery key')
  libraries=[];directories={}
  for entry in catalog['entries']:
   if entry.get('optional') and not a.include_course_v51:continue
   out=restore(entry,bytes.fromhex(keytext));directories[entry['id']]=out
   if entry.get('library'):libraries.append(library(out/entry['library']))
  index=r.ROOT/'local/rag/batch02.sqlite3'
  if not index.exists():r.build_index(libraries,index)
  else:
   with r.Index(index) as i:
    actual=json.loads(i.db.execute("SELECT value FROM meta WHERE key='source_libraries'").fetchone()[0]);assert actual==r.merge_libraries(libraries)['source_libraries']
  from .book_rag_editorial import append_database
  parent=a.app_root/'archives/sqlite/WenguaApp_双书194案例_V391纯断词十二长生_完整研究库.sqlite3'
  if r.digest(parent.read_bytes())!='21362e6f0266c03dcdc4a7ef35f45954ba274ae6458de0ba341132abca7d0405':raise ValueError('Wrong original source database')
  for name in ('preface','batch02'):
   out=r.ROOT/'local/databases'/('three_books_'+name+'.sqlite3');receipt=out.with_suffix('.receipt.json')
   if not out.exists():
    proof=append_database(parent,out,library(directories[name]/'sources.json.gz'),r.read(directories[name]/'review_units.json'))
    if name=='batch02':
     course=r.read(directories[name]/'core_course.json')['records']
     with closing(sqlite3.connect(out)) as db:
      db.execute('CREATE TABLE corpus_core_lessons(id TEXT PRIMARY KEY,unit_id TEXT NOT NULL,payload_json TEXT NOT NULL)')
      db.executemany('INSERT INTO corpus_core_lessons VALUES(?,?,?)',[(x['id'],x['material_unit_id'],r.dumps(x)) for x in course]);db.commit()
     proof['output_database_sha256']=r.digest(out.read_bytes())
    r.write(receipt,proof)
   elif r.digest(out.read_bytes())!=r.read(receipt)['output_database_sha256']:raise ValueError('Database receipt changed')
   parent=out
  r.activate_index(index);print('资料与课程对照已恢复；数据库：'+str(parent));print('未启动训练，未改WenguaApp起盘和V5.1。')
 else:
  with r.Index(r.active_index()) as i:
   found=i.search(a.question,book=a.book,mode='study' if a.study else 'knowledge')
   for d in found['documents']:
    print(d['id'],d.get('editorial_notice',''));print(d.get('reading_content',d['content']))
if __name__=='__main__':main()
