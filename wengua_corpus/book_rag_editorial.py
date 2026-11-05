"""Auditable prose cleanup; the pasted original and all casting code stay intact."""
from pathlib import Path
from contextlib import closing
import bisect,json,re,sqlite3
from . import book_rag as r

SCHEMA='wengua_prose_editorial_v1'


def _step(text,mapping,spec):
    kind=spec['operation']
    if kind=='remove_artifact_lines':
        out=[];marks=[];offset=0
        for line in text.splitlines(keepends=True):
            if not line.lstrip().startswith('image'):
                out.append(line);marks.extend(mapping[offset:offset+len(line)])
            offset+=len(line)
        return ''.join(out),marks
    if kind=='remove_prose_whitespace':
        keep=[i for i,c in enumerate(text) if not c.isspace()]
        return ''.join(text[i] for i in keep),[mapping[i] for i in keep]
    if kind=='replace':
        a=spec['start'];b=a+len(spec['before'])
        if not spec.get('reason') or not spec.get('basis') or not spec['before'] or text[a:b]!=spec['before']:
            raise ValueError('Editorial replacement does not match its exact prior text')
        if not 0<=a<b<=len(text):raise ValueError('Editorial span outside source')
        span=(min(x[0] for x in mapping[a:b]),max(x[1] for x in mapping[a:b]))
        return text[:a]+spec['after']+text[b:],mapping[:a]+[span]*len(spec['after'])+mapping[b:]
    if kind=='sentence_lines':
        out=[];marks=[]
        for char,span in zip(text,mapping):
            out.append(char);marks.append(span)
            if char in '。！':out.append('\n');marks.append((span[1],span[1]))
        if out and out[-1]=='\n':out.pop();marks.pop()
        return ''.join(out),marks
    raise ValueError('Unknown editorial operation')


def make_edition(raw,corrections,*,prose_only=False):
    if not prose_only:raise ValueError('Prose-only cleanup must not be applied blindly to charts or tables')
    text=raw;mapping=[(i,i+1) for i in range(len(raw))];steps=[]
    def apply(spec):
        nonlocal text,mapping
        spec=dict(spec);spec['input_sha256']=r.digest(text);text,mapping=_step(text,mapping,spec)
        spec['output_sha256']=r.digest(text);steps.append(spec)
    apply({'operation':'remove_artifact_lines'});apply({'operation':'remove_prose_whitespace'})
    for correction in corrections:
        before=correction['before'];positions=[m.start() for m in re.finditer(re.escape(before),text)]
        if len(positions)!=correction.get('expected_occurrences',1):raise ValueError('Correction occurrence count differs: '+before)
        for a in reversed(positions):apply({'operation':'replace','start':a,**correction})
    for before,after in [(',', '，'),('!', '！')]:
        for a in reversed([i for i,c in enumerate(text) if c==before]):
            apply({'operation':'replace','start':a,'before':before,'after':after,'reason':'中文标点排版','basis':'user_authorized_formatting','category':'format'})
    apply({'operation':'sentence_lines'})
    return {'schema':SCHEMA,'prose_only':True,'original_sha256':r.digest(raw),'cleaned_sha256':r.digest(text),
            'cleaned_reading_text':text,'steps':steps,'original_not_overwritten':True}


def replay(raw,edition):
    if edition.get('schema')!=SCHEMA or edition.get('prose_only') is not True or r.digest(raw)!=edition['original_sha256']:
        raise ValueError('Wrong source/editorial edition')
    text=raw;mapping=[(i,i+1) for i in range(len(raw))]
    for step in edition['steps']:
        if r.digest(text)!=step['input_sha256']:raise ValueError('Editorial input hash changed')
        text,mapping=_step(text,mapping,step)
        if r.digest(text)!=step['output_sha256']:raise ValueError('Editorial output hash changed')
    if text!=edition['cleaned_reading_text'] or r.digest(text)!=edition['cleaned_sha256']:
        raise ValueError('Cleaned source differs from replayed edits')
    return text,mapping


def parts_for_reading(chapter,line_ranges):
    text,mapping=replay(chapter['original_text'],chapter['editorial_edition']);lines=text.splitlines(keepends=True)
    offsets=[0]
    for line in lines:offsets.append(offsets[-1]+len(line))
    raw=chapter['original_text'];raw_starts=[0]+[m.end() for m in re.finditer('\n',raw)];parts=[]
    for lo,hi in line_ranges:
        if not 1<=lo<=hi<=len(lines):raise ValueError('Cleaned reading line range invalid')
        marks=[p for p in mapping[offsets[lo-1]:offsets[hi]] if p[0]<p[1]]
        if not marks:raise ValueError('Empty source reading range')
        a=min(p[0] for p in marks);b=max(p[1] for p in marks);first=bisect.bisect_right(raw_starts,a);last=bisect.bisect_right(raw_starts,b-1)
        start=raw_starts[first-1];quote=raw[a:b]
        part={'chapter':r.chapter_key(chapter),'source_sha256':chapter['source_sha256'],'lines':[first,last],
              'chars':[a-start,b-start],'quote_sha256':r.digest(quote)}
        if r.quote_part({r.chapter_key(chapter):chapter},part)!=quote:raise ValueError('Raw line/character projection mismatch')
        parts.append(part)
    return parts


def reading_for_document(chapters,doc):
    edition=doc['editorial_reading'];key=edition['chapter'];chapter=chapters[key]
    if edition['cleaned_sha256']!=chapter['editorial_edition']['cleaned_sha256']:raise ValueError('Wrong cleaned edition')
    parts=parts_for_reading(chapter,edition['line_ranges'])
    if doc['parts']!=parts:raise ValueError('Reading quote and raw provenance differ')
    if doc['kind']=='rule' and doc.get('source_uncertainties'):raise ValueError('Unresolved text must not enter positive rules')
    text,_=replay(chapter['original_text'],chapter['editorial_edition']);lines=text.splitlines()
    return '\n'.join('\n'.join(lines[a-1:b]) for a,b in edition['line_ranges'])


def build_library(raw,edition,book,chapter_id,title,units):
    replay(raw,edition);chapter={'book_id':book['id'],'book_title':book['title'],'chapter_id':chapter_id,
        'title':title,'source_revision':1,'source_sha256':r.digest(raw),'original_text':raw,
        'method_id':book['method_id'],'editorial_edition':edition,'source_completeness':'excerpt_not_complete_book'}
    key=r.chapter_key(chapter);docs=[]
    for unit in units:
        ranges=unit['cleaned_lines'];uncertain=unit.get('uncertainties',[]);kind=unit.get('kind','reference' if uncertain else 'rule')
        docs.append({'id':unit['id'],'kind':kind,'parts':parts_for_reading(chapter,ranges),'title':unit['title'],
            'cues':unit.get('search_cues',[]),'domains':unit.get('domains',[]),'case_ids':[],
            'phase':'source_text_requires_collation' if uncertain else 'general_rule',
            'topic':unit.get('topic','rule'),'parent_source_ids':[],
            'review_status':'reviewed_source_selector','source_uncertainties':uncertain,
            'interpretation_limits':unit.get('interpretation_limits',[]),
            'editorial_reading':{'chapter':key,'cleaned_sha256':edition['cleaned_sha256'],'line_ranges':ranges}})
        # Searchable corrections are explicit editorial cues, never passed
        # off as literal raw-source text in the FTS source column.
        docs[-1]['cues']=[*docs[-1]['cues'],reading_for_document({key:chapter},docs[-1])]
    # Always retain every raw character, including removed watermarks, solely
    # as source context. It cannot be retrieved as a fresh-question rule.
    from .book_rag_sources import literal_selector
    docs.append({'id':book['id']+'.'+chapter_id+'.raw_context','kind':'reference',
        'parts':[literal_selector(chapter,1,len(raw.splitlines()))],'title':'用户原贴及版面痕迹',
        'cues':[],'domains':[],'case_ids':[],'phase':'immutable_raw_transcription',
        'topic':'source','parent_source_ids':[],'review_status':'reviewed_source_selector'})
    library={'schema':r.SCHEMA,'books':{book['id']:{k:v for k,v in book.items() if k!='id'}},
        'chapters':[chapter],'documents':docs,'cases':{},'qa_answers_indexed':False}
    r.validate_library(library);return library


def append_database(source,output,library,units):
    """New SQLite snapshot using existing books/sections/rules; no V1 importer edits."""
    output=r.private(output)
    if output.exists():raise ValueError('Preserve prior database; choose a new version')
    source=Path(source).resolve();before=r.digest(source.read_bytes());output.parent.mkdir(parents=True,exist_ok=True)
    with closing(sqlite3.connect(source.as_uri()+'?mode=ro',uri=True)) as old,closing(sqlite3.connect(output)) as db:
        old.backup(db);db.execute('PRAGMA foreign_keys=ON')
        db.executescript((r.ROOT/'schema/migrations/072_source_editorial_reading_versions.sql').read_text(encoding='utf8'))
        chapter=library['chapters'][0];bid=chapter['book_id'];book=library['books'][bid]
        row=db.execute('SELECT title FROM books WHERE book_id=?',(bid,)).fetchone()
        if row and row[0]!=book['title']:raise ValueError('Conflicting existing book title')
        db.execute('INSERT OR IGNORE INTO books(book_id,title,source_status) VALUES(?,?,?)',(bid,book['title'],'partial_user_source'))
        version=db.execute('SELECT COALESCE(MAX(content_version),0)+1 FROM sections WHERE book_id=? AND section_key=?',(bid,chapter['chapter_id'])).fetchone()[0]
        row=db.execute('SELECT 1 FROM sections WHERE book_id=? AND section_key=? AND sha256=?',(bid,chapter['chapter_id'],chapter['source_sha256'])).fetchone()
        if row:raise ValueError('Identical source already present; do not duplicate')
        sid=db.execute('INSERT INTO sections(book_id,section_key,title,original_text,sha256,content_version,reviewed) VALUES(?,?,?,?,?,?,?)',
            (bid,chapter['chapter_id'],chapter['title'],chapter['original_text'],chapter['source_sha256'],version,1)).lastrowid
        e=chapter['editorial_edition'];db.execute('INSERT INTO source_editorial_editions VALUES(?,?,?,?,?)',(sid,1,e['cleaned_reading_text'],e['cleaned_sha256'],r.dumps(e)))
        for unit in units:db.execute('INSERT INTO source_collation_units VALUES(?,?,?)',(unit['id'],sid,r.dumps(unit)))
        # Editorial review does not create executable/approved rule rows or
        # add fictional cases. Source rules remain in the independent RAG.
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        db.commit()
    if r.digest(source.read_bytes())!=before:raise ValueError('Original database changed')
    with closing(sqlite3.connect(output)) as db:
        if db.execute('SELECT original_text FROM sections WHERE section_id=?',(sid,)).fetchone()[0]!=chapter['original_text']:raise ValueError('Stored original differs')
        if db.execute('SELECT cleaned_text FROM source_editorial_editions WHERE section_id=?',(sid,)).fetchone()[0]!=e['cleaned_reading_text']:raise ValueError('Stored reading differs')
    return {'source_database_sha256':before,'output_database_sha256':r.digest(output.read_bytes()),
            'section_id':sid,'original_database_unchanged':True,'new_cases':0}
