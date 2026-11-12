"""Private curriculum database and lossless Excel, using the standard library."""
from pathlib import Path
import json,sqlite3,zipfile
from xml.etree import ElementTree as ET
from wengua_course import data as d
NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main'


def xlsx(path,sheets):
    payload={};expected=[]
    for number,(title,rows) in enumerate(sheets,1):
        root=ET.Element('worksheet',xmlns=NS);data=ET.SubElement(root,'sheetData');saved=[]
        for i,row in enumerate(rows,1):
            element=ET.SubElement(data,'row',r=str(i));saved.append([str(v) for v in row])
            for j,value in enumerate(row):
                value=str(value)
                if len(value)>32767:raise ValueError('Excel cell exceeds limit; refuse truncation')
                cell=ET.SubElement(element,'c',r=chr(65+j)+str(i),t='inlineStr');text=ET.SubElement(ET.SubElement(cell,'is'),'t')
                text.set('{http://www.w3.org/XML/1998/namespace}space','preserve');text.text=value
        payload[f'xl/worksheets/sheet{number}.xml']=ET.tostring(root,encoding='utf-8',xml_declaration=True);expected.append(saved)
    names=''.join(f'<sheet name="{title}" sheetId="{i}" r:id="rId{i}"/>' for i,(title,_) in enumerate(sheets,1))
    payload['xl/workbook.xml']=f'<workbook xmlns="{NS}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>{names}</sheets></workbook>'.encode()
    rels=''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1,len(sheets)+1))
    payload['xl/_rels/workbook.xml.rels']=f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rels}</Relationships>'.encode()
    payload['_rels/.rels']=b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'
    overrides=''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1,len(sheets)+1))
    payload['[Content_Types].xml']=f'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>{overrides}</Types>'.encode()
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for name,value in payload.items():z.writestr(name,value)
    with zipfile.ZipFile(path) as z:
        for i,gold in enumerate(expected,1):
            root=ET.fromstring(z.read(f'xl/worksheets/sheet{i}.xml'));rows=[]
            for row in root.findall('.//{'+NS+'}row'):rows.append([''.join(c.itertext()) for c in row])
            if rows!=gold:raise ValueError('Excel full-cell readback differs')


def write_review(pack,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);stamp=d.digest(d.dumps(pack));receipt=output/'review_receipt.json'
    if receipt.exists():
        old=d.read(receipt)
        if old['pack_json_sha256']!=stamp or any(d.digest((output/n).read_bytes())!=h for n,h in old['files'].items()):
            raise ValueError('Private course review changed; original preserved')
        return old
    x=output/'Wengua_V66_条件修复课程.xlsx';dbfile=output/'course.sqlite3'
    if x.exists() or dbfile.exists():raise ValueError('Incomplete course review preserved; move it aside before retrying')
    rows=[['ID','划分','任务','完整对话JSON','末回合参考答','来源ID','条件与出处JSON','本轮短训选中']]
    for r in pack['records']:rows.append([r['id'],r['split'],r['skill'],d.dumps(r['messages']),r['answer'],d.dumps(r['source_ids']),d.dumps(r['provenance']),r['pilot_selected']])
    units=[['ID','书籍','章节','原文行号','完整原文']]+[[u['id'],u['book_id'],u.get('chapter',''),d.dumps(u.get('lines')),u['original_quote']] for u in pack['units'].values()]
    xlsx(x,[('课程',rows),('来源',units)])
    with sqlite3.connect(dbfile) as db:
        db.execute('PRAGMA foreign_keys=ON')
        db.executescript('CREATE TABLE source_units(id TEXT PRIMARY KEY,payload_json TEXT NOT NULL); CREATE TABLE lessons(id TEXT PRIMARY KEY,split TEXT NOT NULL,payload_json TEXT NOT NULL); CREATE TABLE lesson_sources(lesson_id TEXT REFERENCES lessons(id),source_id TEXT REFERENCES source_units(id),PRIMARY KEY(lesson_id,source_id)); CREATE TABLE original_dispositions(id TEXT PRIMARY KEY,payload_json TEXT NOT NULL); CREATE TABLE material_usage(source_id TEXT PRIMARY KEY REFERENCES source_units(id),payload_json TEXT NOT NULL); CREATE TABLE original_case_splits(case_id TEXT PRIMARY KEY,split TEXT NOT NULL);')
        db.executemany('INSERT INTO source_units VALUES(?,?)',[(k,d.dumps(v)) for k,v in pack['units'].items()])
        db.executemany('INSERT INTO lessons VALUES(?,?,?)',[(r['id'],r['split'],d.dumps(r)) for r in pack['records']])
        db.executemany('INSERT INTO lesson_sources VALUES(?,?)',[(r['id'],s) for r in pack['records'] for s in r['source_ids']])
        db.executemany('INSERT INTO original_dispositions VALUES(?,?)',[(r['old_id'],d.dumps(r)) for r in pack['dispositions']]);db.commit()
        db.executemany('INSERT INTO material_usage VALUES(?,?)',[(r['source_id'],d.dumps(r)) for r in pack.get('material_usage',[])])
        db.executemany('INSERT INTO original_case_splits VALUES(?,?)',list(pack.get('case_splits',{}).items()));db.commit()
        if dict(db.execute('SELECT id,payload_json FROM source_units'))!={k:d.dumps(v) for k,v in pack['units'].items()}:raise ValueError('Source SQLite readback differs')
        if dict(db.execute('SELECT id,payload_json FROM lessons'))!={r['id']:d.dumps(r) for r in pack['records']}:raise ValueError('Lesson SQLite readback differs')
        if dict(db.execute('SELECT source_id,payload_json FROM material_usage'))!={r['source_id']:d.dumps(r) for r in pack.get('material_usage',[])}:raise ValueError('Material coverage readback differs')
    proof={'pack_json_sha256':stamp,'files':{p.name:d.digest(p.read_bytes()) for p in [x,dbfile]},'full_excel_and_sqlite_readback':True,'model_answers_included':False,'source_book_rag_index_modified':False}
    d.write(receipt,proof);return proof
