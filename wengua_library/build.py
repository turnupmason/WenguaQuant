"""Compile a private source extension and seal it using the existing delivery key."""
from pathlib import Path
from collections import Counter
import gzip,json,sqlite3,io,zipfile,os,hmac,hashlib
from . import intake
from wengua_corpus import book_rag as r,delivery


def bind_external_evidence(pack,old):
    """Make every cross-source teaching claim visible; never import case outcomes."""
    available={p['id']:p for p in old}
    for course in pack['courses']:
        ids=course.pop('prior_document_ids',[])
        allow_reference=course.pop('allow_reference_sources',False)
        if not ids:continue
        evidence=[]
        for id in ids:
            prior=available[id]
            if prior['kind'] not in ({'rule','reference'} if allow_reference else {'rule'}) or prior.get('case_ids'):
                raise ValueError('Historical cases or unapproved reference text cannot become external course evidence')
            evidence.append({k:v for k,v in prior.items() if k!='grams'})
        course['external_source_document_ids']=ids;course['external_source_documents']=evidence
        known={r.dumps(p) for p in course['source_parts']}
        for e in evidence:
            for part in e['parts']:
                if r.dumps(part) not in known:course['source_parts'].append(part);known.add(r.dumps(part))
        course['source_family_ids']=sorted(set(course.get('source_family_ids',[]))|{'source_document:'+id for id in ids})
        course['external_evidence_is_source_study_not_outcome']=True


def build(source,annotations,output,prior_entries):
    output=r.private(output);output.mkdir(parents=True,exist_ok=True)
    original=Path(source).read_bytes();raw=original.decode('utf-8-sig')
    prior=[intake.load_library(r.ROOT/e['destination']/e['library']) for e in prior_entries]
    book=next(lib['books']['jinhan'] for lib in prior if 'jinhan' in lib['books'])
    spec=r.read(annotations);pack=intake.compile_source(raw,book,spec)
    old,duplicates=intake.compare_sources(pack,prior);comparisons=intake.curated_comparisons(pack,old,spec)
    bind_external_evidence(pack,old)
    pack.update(comparisons=comparisons,duplicate_links=duplicates,output_profile=spec.get('output_profile',{}))
    for name in pack['output_profile'].values():
        if not isinstance(name,str) or Path(name).name!=name or '\\' in name:raise ValueError('Unsafe output filename')
    pack['alignment']=[dict(material_unit_id=u['id'],source_lines=u['source_lines'],state=u['course_state'],
        course_candidate_ids=[c['id'] for c in pack['courses'] if u['id'] in c['source_unit_ids']],
        logic_rule_ids=[x['id'] for x in pack['logic_rules'] if u['id'] in x.get('source_unit_ids',[x['source_unit_id']])],
        workflow_ids=[x['id'] for x in pack['workflows'] if u['id'] in x['source_unit_ids']],
        rag_document_ids=[x['id'] for x in pack['library']['documents'] if x['id']==u['rag_document_id'] or x.get('parent_material_id')==u['id'] or u['id'] in x.get('source_unit_ids',[])],
        case_ids=[],author_case_analysis_ids=[],current_training_eligible=False) for u in pack['units']]
    for structure in pack['source_structures']:
        line=structure['source_line'];text=raw.splitlines()[line-1]
        for fragment in structure.get('literal_fragments',[]):
            if fragment not in text:raise ValueError('Structure cell not in literal source: '+structure['id'])
    (output/'original.txt').write_bytes(original)
    (output/'reading.txt').write_text(raw,encoding='utf8')
    r.write(output/'editorial_audit.json',{'received_bytes_sha256':r.digest(original),'decoded_text_sha256':r.digest(raw),
                 'additional_professional_word_replacements':[], 'source_already_contains_supplied_ocr_edits':True,
                 'source_notes_are_not_facsimile_verification':True,'original_bytes_preserved':True})
    (output/'sources.json.gz').write_bytes(gzip.compress(r.dumps(pack['library']).encode(),mtime=0))
    for name,key in [('materials.json','units'),('logic_rules.json','logic_rules'),('workflows.json','workflows'),
                     ('course_candidates.json','courses'),('worked_examples.json','illustrations'),
                     ('duplicate_links.json','duplicate_links'),('reviewed_comparisons.json','comparisons'),
                     ('gaps.json','gaps'),('source_structures.json','source_structures'),
                     ('material_alignment.json','alignment')]:r.write(output/name,pack[key])
    r.write(output/'curation_recipe.json',spec)
    # The private input's notes are evidence, never executable instructions.
    report=pack['report']|{'schema':intake.SCHEMA,'source_bytes':len(original),'source_characters':len(raw),
             'source_lines':len(raw.splitlines()),'source_file_sha256':r.digest(original),'book_complete':False,
             'material_units':len(pack['units']),'course_candidates':len(pack['courses']),
             'course_formats':dict(Counter(c['format'] for c in pack['courses'])),
             'logic_rules':len(pack['logic_rules']),'source_workflows':len(pack['workflows']),
             'worked_illustrations':len(pack['illustrations']),'observed_cases_added':0,'observed_author_case_analyses_added':0,
             'quarantined_materials':sum(u['course_state'].startswith(('quarantined','historical')) for u in pack['units']),
             'material_dispositions':dict(Counter(u['course_state'] for u in pack['units'])),
             'duplicate_relation_counts':dict(Counter(d['relation'] for d in duplicates)),
             'reviewed_comparison_groups':len(comparisons),'gap_items':len(pack['gaps']),
             'old_manual_scores_used':False,'model_answers_used_as_sources':False,'current_v60_training_changed':False,
             'all_new_courses_current_training_eligible':False,'v60_release_unchanged':True,
             'source_comparison_is_not_observed_predictive_validation':True}
    if spec.get('source_profile'):report['courses_with_visible_external_source_evidence']=sum(bool(c.get('external_source_documents')) for c in pack['courses'])
    r.write(output/'coverage.json',report)
    write_database(output,pack,prior);write_excel(output,pack)
    return pack,report


def write_database(output,pack,prior):
    path=output/pack.get('output_profile',{}).get('database','Wengua_金函第三批_材料规则课程对照.sqlite3')
    if path.exists():raise FileExistsError('Existing source snapshot preserved')
    con=sqlite3.connect(path)
    con.executescript('CREATE TABLE source_sections(id TEXT PRIMARY KEY,original_text TEXT,sha256 TEXT,payload TEXT);'
                     'CREATE TABLE materials(id TEXT PRIMARY KEY,section TEXT,mode TEXT,line_start INTEGER,line_end INTEGER,state TEXT,payload TEXT);'
                     'CREATE TABLE logic_rules(id TEXT PRIMARY KEY,source_unit_id TEXT,payload TEXT);'
                     'CREATE TABLE workflows(id TEXT PRIMARY KEY,title TEXT,payload TEXT);'
                     'CREATE TABLE core_course_candidates(id TEXT PRIMARY KEY,format TEXT,eligible INTEGER,payload TEXT);'
                     'CREATE TABLE comparisons(id TEXT PRIMARY KEY,relation TEXT,payload TEXT);'
                     'CREATE TABLE duplicates(seq INTEGER PRIMARY KEY,payload TEXT);'
                     'CREATE TABLE gaps(id TEXT PRIMARY KEY,payload TEXT);'
                     'CREATE TABLE source_structures(id TEXT PRIMARY KEY,payload TEXT);'
                     'CREATE TABLE worked_examples(id TEXT PRIMARY KEY,observed_case INTEGER,payload TEXT);'
                     'CREATE TABLE material_alignment(id TEXT PRIMARY KEY,payload TEXT);'
                     'CREATE TABLE inherited_case_registry(id TEXT PRIMARY KEY,payload TEXT);'
                     'CREATE TABLE rag_documents(id TEXT PRIMARY KEY,kind TEXT,payload TEXT);')
    chapter=pack['library']['chapters'][0]
    con.execute('INSERT INTO source_sections VALUES (?,?,?,?)',(r.chapter_key(chapter),chapter['original_text'],chapter['source_sha256'],r.dumps(chapter)))
    for u in pack['units']:con.execute('INSERT INTO materials VALUES(?,?,?,?,?,?,?)',(u['id'],u['section'],u['source_mode'],*u['source_lines'],u['course_state'],r.dumps(u)))
    for x in pack['logic_rules']:con.execute('INSERT INTO logic_rules VALUES(?,?,?)',(x['id'],x['source_unit_id'],r.dumps(x)))
    for x in pack['workflows']:con.execute('INSERT INTO workflows VALUES(?,?,?)',(x['id'],x['title'],r.dumps(x)))
    for x in pack['courses']:con.execute('INSERT INTO core_course_candidates VALUES(?,?,?,?)',(x['id'],x['format'],0,r.dumps(x)))
    for x in pack['comparisons']:con.execute('INSERT INTO comparisons VALUES(?,?,?)',(x['id'],x['relation'],r.dumps(x)))
    for i,x in enumerate(pack['duplicate_links']):con.execute('INSERT INTO duplicates VALUES(?,?)',(i,r.dumps(x)))
    for table,key in [('gaps','gaps'),('source_structures','source_structures')]:
        for x in pack[key]:con.execute('INSERT INTO '+table+' VALUES(?,?)',(x['id'],r.dumps(x)))
    for x in pack['illustrations']:con.execute('INSERT INTO worked_examples VALUES(?,?,?)',(x['id'],0,r.dumps(x)))
    for x in pack['alignment']:con.execute('INSERT INTO material_alignment VALUES(?,?)',(x['material_unit_id'],r.dumps(x)))
    cases={cid:c for lib in prior for cid,c in lib['cases'].items()}
    for cid,c in cases.items():con.execute('INSERT INTO inherited_case_registry VALUES(?,?)',(cid,r.dumps(c)))
    for doc in pack['library']['documents']:con.execute('INSERT INTO rag_documents VALUES(?,?,?)',(doc['id'],doc['kind'],r.dumps(doc)))
    con.commit();checks={}
    for table,key in [('materials','units'),('logic_rules','logic_rules'),('workflows','workflows'),('core_course_candidates','courses'),('comparisons','comparisons'),('gaps','gaps'),('source_structures','source_structures'),('worked_examples','illustrations')]:
        restored={id:json.loads(payload) for id,payload in con.execute('SELECT id,payload FROM '+table)}
        if restored!={x['id']:x for x in pack[key]}:raise ValueError('Database object roundtrip mismatch: '+table)
        checks[table]=len(restored)
    if con.execute('SELECT original_text FROM source_sections').fetchone()[0]!=chapter['original_text']:raise ValueError('Original source changed in database')
    assert {id:json.loads(p) for id,p in con.execute('SELECT id,payload FROM material_alignment')}=={x['material_unit_id']:x for x in pack['alignment']}
    checks['material_alignment']=len(pack['alignment'])
    if con.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Database integrity failure')
    checks['inherited_cases']=con.execute('SELECT count(*) FROM inherited_case_registry').fetchone()[0]
    assert checks['inherited_cases']==len(cases)==194
    assert con.execute('SELECT count(*) FROM core_course_candidates WHERE eligible!=0').fetchone()[0]==0
    con.close();r.write(output/'database_audit.json',{'roundtrip':checks,'source_unchanged':True,'new_observed_cases':0,'v60_tables_modified':False})


def write_excel(output,pack):
    from openpyxl import Workbook,load_workbook
    from openpyxl.styles import Alignment,Font,PatternFill
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.utils import get_column_letter
    wb=Workbook(write_only=True);expected={}
    def sheet(name,headers,widths):
        ws=wb.create_sheet(name);ws.freeze_panes='A2'
        for i,width in enumerate(widths,1):ws.column_dimensions[get_column_letter(i)].width=width
        cells=[]
        for text in headers:
            c=WriteOnlyCell(ws,value=text);c.font=Font(bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='24476B');cells.append(c)
        ws.append(cells);expected[name]=[];return ws
    def add(ws,values):
        if any(isinstance(v,str) and len(v)>32767 for v in values):raise ValueError('Excel needs split full-source storage')
        cells=[]
        for v in values:
            c=WriteOnlyCell(ws,value=v);c.data_type='s' if isinstance(v,str) else c.data_type;c.alignment=Alignment(wrap_text=True,vertical='top');cells.append(c)
        ws.append(cells);expected[ws.title].append(values)
    ws=sheet('全部材料与处置',['材料ID','卷','篇目','原文译文层','行起','行止','处置','完整文字','疑点与限制','所需上下文','关联候选课程'],[35,10,23,18,9,9,31,100,70,45,45])
    for u in pack['units']:
        linked=[c['id'] for c in pack['courses'] if u['id'] in c['source_unit_ids']]
        add(ws,[u['id'],u['volume'],u['section'],u['source_mode'],*u['source_lines'],u['course_state'],u['text'],r.dumps(u['findings']),r.dumps(u['required_context_ids']),r.dumps(linked)])
    ws=sheet('断案逻辑',['规则ID','专题','条件和分支','原文结论','证据片段','完整结构JSON'],[36,24,90,65,95,120])
    for x in pack['logic_rules']:add(ws,[x['id'],x['source_section'],r.dumps(x['conditions']),x['conclusion'],'\n'.join(x['literal_fragments']),r.dumps(x)])
    ws=sheet('全部占事流程',['流程ID','篇目','领域','按来源顺序的材料','对应逻辑规则','疑点限制'],[40,24,30,80,80,100])
    for x in pack['workflows']:add(ws,[x['id'],x['title'],r.dumps(x['domains']),r.dumps(x['source_unit_ids']),r.dumps(x['logic_rule_ids']),r.dumps(x['restrictions'])])
    external=any(x.get('external_source_documents') for x in pack['courses'])
    ws=sheet('核心课程候选',['课程ID','类型','问题或对话','参考答案','完整来源材料','状态','当前训练可用']+(['跨来源完整证据JSON'] if external else []),[40,24,100,100,70,45,15]+([120] if external else []))
    for x in pack['courses']:add(ws,[x['id'],x['format'],x.get('question',r.dumps(x.get('messages',[]))),x.get('answer',''),r.dumps(x['source_unit_ids']),x['status'],False]+([r.dumps(x.get('external_source_documents',[]))] if external else []))
    ws=sheet('新旧来源对照',['组ID','关系','专题发现','新增来源','旧来源完整对照','合并限制'],[40,30,90,70,120,90])
    for x in pack['comparisons']:add(ws,[x['id'],x['relation'],x['finding'],r.dumps(x['new_unit_ids']),r.dumps(x['prior_evidence']),x['merge_policy']])
    ws=sheet('查重候选',['新材料','旧材料','关系','相似度','可自动合并','说明'],[40,55,50,15,18,90])
    for x in pack['duplicate_links']:add(ws,[x['new_unit_id'],x['prior_document_id'],x['relation'],x.get('jaccard_3gram'),x['automatically_mergeable'],'所有出处保留；相似不等于条件、方向或结论一致。'])
    ws=sheet('缺文异说及补证',['ID','范围','状态','问题和依据JSON'],[40,40,35,140])
    for x in pack['gaps']:add(ws,[x['id'],x.get('scope',''),x['status'],r.dumps(x)])
    ws=sheet('结构表与论例',['ID','类型','完整结构JSON'],[40,40,150])
    for x in pack['source_structures']+pack['illustrations']:add(ws,[x['id'],x.get('type',x.get('example_type','')),r.dumps(x)])
    path=output/pack.get('output_profile',{}).get('workbook','金函第三批_全材料断案逻辑课程RAG对照.xlsx');wb.save(path)
    restored=load_workbook(path,read_only=True,data_only=False)
    for title,values in expected.items():
        rows=list(restored[title].values)[1:]
        if len(rows)!=len(values):raise ValueError('Excel row count mismatch')
        for got,want in zip(rows,values):
            if [None if v=='' else v for v in want]!=list(got):raise ValueError('Excel full-cell roundtrip mismatch: '+title)
    restored.close();r.write(output/'excel_audit.json',{'all_full_cells_read_back_equal':True,'sheets':{k:len(v) for k,v in expected.items()},'truncated_cells':0,'formula_cells_created':0})


def seal(output,key_file,release_path,prior_entries,*,batch_id='jinhan_batch03',version='jinhan_batch03_20261109',source_scope='partial_juan04_06_not_complete_book',archive_name='Wengua_金函第三批_五假三诈九遁与占事.zip'):
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes
    output=Path(output);release_path=Path(release_path)
    files={p.name:r.digest(p.read_bytes()) for p in output.iterdir() if p.is_file() and p.name!='paragraph_inventory.json'}
    mem=io.BytesIO()
    with zipfile.ZipFile(mem,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):z.write(output/name,name)
    plain=mem.getvalue();master=bytes.fromhex(Path(key_file).read_text().strip());key,mac=delivery.derive_keys(master);iv=os.urandom(16)
    pad=padding.PKCS7(128).padder();padded=pad.update(plain)+pad.finalize();enc=Cipher(algorithms.AES(key),modes.CBC(iv)).encryptor()
    body=delivery.MAGIC+iv+enc.update(padded)+enc.finalize();blob=body+hmac.new(mac,body,hashlib.sha256).digest()
    if not batch_id.replace('_','').isalnum() or not version.replace('_','').isalnum():raise ValueError('Unsafe release identifier')
    folder=r.ROOT/'data/delivery'/batch_id;folder.mkdir(parents=True,exist_ok=True)
    target=folder/(batch_id+'_'+r.digest(blob)[:12]+'.enc');target.write_bytes(blob)
    manifest=dict(file=target.relative_to(r.ROOT).as_posix(),ciphertext_sha256=r.digest(blob),archive_name=archive_name,archive_sha256=r.digest(plain))
    r.write(folder/'manifest.json',manifest)
    entry=dict(id=batch_id,manifest=(folder/'manifest.json').relative_to(r.ROOT).as_posix(),destination=output.relative_to(r.ROOT).as_posix(),files=files,library='sources.json.gz')
    release=dict(schema=intake.SCHEMA,version=version,data_owner='WenguaQuant',source_scope=source_scope,
                 entries=[*prior_entries,entry],new_entry_id=entry['id'],index='local/rag/'+version+'.sqlite3',
                 current_v60_training_changed=False,automatic_training_inclusion=False,coverage=r.read(output/'coverage.json'))
    r.write(release_path,release)
    k,iv,cipher=delivery.authenticate(blob,master);dec=Cipher(algorithms.AES(k),modes.CBC(iv)).decryptor();unpad=padding.PKCS7(128).unpadder()
    assert unpad.update(dec.update(cipher)+dec.finalize())+unpad.finalize()==plain
    return release
