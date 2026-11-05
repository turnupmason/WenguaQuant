"""WenguaApp source-only retrieval. No model or old health scoring is imported.

An immutable source library stores full chapters and reviewed selectors. SQLite
FTS5 indexes literal evidence plus editorial search cues; Q&A target answers
are never indexed. Historical passages are only retrievable in study mode.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import gzip,hashlib,json,math,re,sqlite3,unicodedata
from pathlib import Path

SCHEMA='wengua_book_rag_v1'
ROOT=Path(__file__).resolve().parents[1]
METHOD='wengua_shijia_zhuanpan'
INDEX_FORMAT='cjk_bigrams_reviewed_scopes_ordered_pairs_v1'
KINDS={'rule','case','reference'}
DOMAINS={
 'health':('疾病','看病','生病','病情','康复','医生','医药','治疗','天芮'),
 'marriage':('婚姻','结婚','婚恋','恋爱','配偶','离婚','复合'),
 'pregnancy':('怀孕','孕妇','分娩','生育','胎儿','生产'),
 'education':('考试','考研','考生','成绩','录取','考学','高考','升学'),
 'employment':('求职','工作','就业','升职','晋升','辞职','调动','职位'),
 'business_profit':('求财','投资','经营','生意','利润','合伙','财运','赚钱'),
 'travel_abroad':('出行','出国','签证','移民','旅行','远行'),
 'missing_person':('走失','失踪','行人','寻人','孩子不见','人没回来'),
 'lost_property':('失物','丢失','遗失','丢了','找东西','钱物','东西不见'),
 'criminal_case':('刑事','盗窃','凶手','嫌犯','破案','抢劫','犯罪'),
 'litigation':('官司','诉讼','起诉','法院','判刑','律师'),
 'sports_competition':('体育','球赛','足球','篮球','比赛','运动员'),
 'military_history_source':('军事','战役','作战','行军'),
 'weather':('天气','下雨','降雨','晴天','阴天','风雨','气象','下雪'),
 'geography_environment_historical_source':('风水','地理','住宅','环境','墓地'),
 'life_opportunity_book_study':('终身','命运','人生','近期运气'),
 'competition_evaluation':('竞争评比','评比','竞聘','评选'),
 'politics_elections':('选举','竞选','总统','市长'),
 'industrial_safety':('安全事故','工厂事故','生产安全'),
 'world_affairs':('国际局势','世界形势'),
 'miscellaneous_historical_source':('杂项','杂事')}
DOMAIN_EQUIVALENTS={'weather_historical_source':'weather'}
ALIASES={'爸爸':'父亲','老爸':'父亲','妈妈':'母亲','老妈':'母亲','老公':'丈夫','老婆':'妻子',
         '找工作':'求职','找丢的东西':'失物','丢的东西':'失物','东西找不到':'失物','丢东西':'失物','怎么选':'如何取','什么时候好':'康复应期'}
SYMBOLS=('天蓬','天任','天冲','天辅','天英','天芮','天禽','天柱','天心','乙奇','丙奇','丁奇',
 '休门','生门','伤门','杜门','景门','死门','惊门','开门','值符','值使','伏吟','反吟','门迫',
 '空亡','入墓','六仪击刑','长生','河图','后天','年干','年命','日干','时干','乾宫','坤宫')


def digest(x):return hashlib.sha256(x if isinstance(x,bytes) else x.encode('utf8')).hexdigest()
def dumps(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def private(path):
    path=Path(path).expanduser().resolve()
    if not path.is_relative_to((ROOT/'local').resolve()):raise ValueError('Private sources/index/answers must stay under project local/')
    return path

def normalize(text):
    text=unicodedata.normalize('NFKC',text).lower()
    for a,b in ALIASES.items():text=text.replace(a,b)
    return text

def ordered_pairs(text):
    text=normalize(text)
    stem='甲乙丙丁戊己庚辛壬癸'
    patterns=[r'天盘?(['+stem+r'])(?:[，,、/加与和\s]*)地盘?(['+stem+r'])',
              r'(['+stem+r'])(?:加|/)(['+stem+r'])']
    return set(pair for pattern in patterns for pair in re.findall(pattern,text))


def terms(text):
    text=normalize(text);result=[]
    for token in re.findall(r'[a-z0-9_]+|[\u3400-\u9fff]+',text):
        if token.isascii():result.append(token)
        else:result.extend(token[i:i+2] for i in range(max(1,len(token)-1)))
    result.extend('ordered'+str('甲乙丙丁戊己庚辛壬癸'.index(a))+'d'+str('甲乙丙丁戊己庚辛壬癸'.index(b)) for a,b in ordered_pairs(text))
    result.extend('symbol'+str(i) for i,s in enumerate(SYMBOLS) if s in text)
    return result

def domains_for(question,explicit=None):
    if explicit:
        explicit=DOMAIN_EQUIVALENTS.get(explicit,explicit)
        if explicit not in DOMAINS:raise ValueError('Unknown domain: '+explicit)
        return [explicit]
    text=normalize(question)
    return [domain for domain,words in DOMAINS.items() if any(w in text for w in words)]

def chapter_key(chapter):return chapter['book_id']+'/'+chapter['chapter_id']

def quote_part(chapters,part):
    chapter=chapters[part['chapter']]
    if part['source_sha256']!=chapter['source_sha256']:raise ValueError('Part source revision mismatch')
    lo,hi=part['lines'];lines=chapter['original_text'].splitlines()
    if type(lo)is not int or type(hi)is not int or not 1<=lo<=hi<=len(lines):raise ValueError('Part source lines invalid')
    raw='\n'.join(lines[lo-1:hi]);a,b=part.get('chars',[0,len(raw)])
    if type(a)is not int or type(b)is not int or not 0<=a<b<=len(raw):raise ValueError('Part character selector invalid')
    text=raw[a:b]
    if digest(text)!=part['quote_sha256']:raise ValueError('Part quote changed')
    return text

def validate_library(library):
    if library.get('schema')!=SCHEMA:raise ValueError('Unknown source library schema')
    books=library['books'];chapters={chapter_key(c):c for c in library['chapters']}
    if len(chapters)!=len(library['chapters']):raise ValueError('Duplicate chapter version; select replacement explicitly')
    for key,c in chapters.items():
        if c['book_id'] not in books or not c['method_id'] or not c['source_revision']:raise ValueError('Incomplete bibliography/method')
        if digest(c['original_text'])!=c['source_sha256']:raise ValueError('Full chapter changed: '+key)
    ids=set();covered=defaultdict(set);counts=Counter();cases=set()
    for doc in library['documents']:
        if doc['id'] in ids or doc['kind'] not in KINDS or not doc['parts']:raise ValueError('Duplicate/invalid source document')
        ids.add(doc['id']);counts[doc['kind']]+=1
        if doc.get('review_status')!='reviewed_source_selector':raise ValueError('Unreviewed selectors cannot enter a live index')
        if doc['kind']=='rule' and (doc['case_ids'] or doc['phase']!='general_rule'):raise ValueError('Historical case cannot be promoted to a general rule')
        if doc['kind']=='case' and not doc['case_ids']:raise ValueError('Case evidence must name its event')
        for part in doc['parts']:
            quote_part(chapters,part)
            matching=[]
            for cid,case in library['cases'].items():
                for span in case.get('spans',[]):
                    if span['chapter']==part['chapter'] and max(part['lines'][0],span['lines'][0])<=min(part['lines'][1],span['lines'][1]):matching.append(cid)
            if doc['kind']=='rule' and matching:raise ValueError('Case source range cannot enter fresh-question rules')
            if doc['kind']=='case' and not set(matching).intersection(doc['case_ids']):raise ValueError('Case evidence is outside its registered source range: '+doc['id']+' '+repr(doc['case_ids'])+' '+repr(part))
            covered[part['chapter']].update(range(part['lines'][0],part['lines'][1]+1))
        cases.update(doc['case_ids'])
        if doc.get('editorial_reading'):
            from .book_rag_editorial import reading_for_document
            reading_for_document(chapters,doc)
    expected=set(library['cases'])
    if cases!=expected:raise ValueError('Case registry/evidence coverage mismatch')
    missing={k:[i for i,line in enumerate(c['original_text'].splitlines(),1) if line.strip() and i not in covered[k]] for k,c in chapters.items()}
    if any(missing.values()):raise ValueError('Source lines without an explicit index disposition')
    return {'schema':SCHEMA,'books':len(books),'chapters':len(chapters),'documents':len(ids),
        'document_kinds':dict(counts),'cases':len(cases),'unaccounted_nonblank_lines':0,
        'source_characters':sum(len(c['original_text']) for c in chapters.values()),
        'all_chapter_hashes_and_literal_parts_verified':True,'qa_answers_indexed':False,
        'model_quality_or_predictive_accuracy_verified':False}

def merge_libraries(libraries):
    if not libraries:raise ValueError('At least one reviewed source library required')
    result={'schema':SCHEMA,'books':{},'chapters':[],'documents':[],'cases':{},'source_libraries':[]}
    chapter_seen={};doc_seen={}
    for library in libraries:
        validate_library(library);result['source_libraries'].append(digest(dumps(library)))
        for key,book in library['books'].items():
            if key in result['books'] and result['books'][key]!=book:raise ValueError('Conflicting book metadata')
            result['books'][key]=book
        for c in library['chapters']:
            key=chapter_key(c)
            if key in chapter_seen:
                if chapter_seen[key]!=c:raise ValueError('Different active chapter revisions; build a reviewed replacement bundle')
            else:chapter_seen[key]=c;result['chapters'].append(c)
        for d in library['documents']:
            if d['id'] in doc_seen:
                if doc_seen[d['id']]!=d:raise ValueError('Conflicting document ID')
            else:doc_seen[d['id']]=d;result['documents'].append(d)
        for cid,case in library['cases'].items():
            if cid in result['cases'] and result['cases'][cid]!=case:raise ValueError('Conflicting event provenance')
            result['cases'][cid]=case
    validate_library(result);return result

def document_text(chapters,doc):
    return '\n\n'.join(quote_part(chapters,p) for p in doc['parts'])

def build_index(libraries,output):
    output=private(output)
    if output.exists():raise ValueError('Index snapshots are immutable; select a new output path')
    library=merge_libraries(libraries);report=validate_library(library)
    pending=output.with_name(output.name+'.pending')
    if pending.exists():raise ValueError('Incomplete index preserved')
    pending.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(pending)
    try:
        db.executescript('''CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE chapters(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
CREATE TABLE cases(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
CREATE TABLE documents(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
CREATE VIRTUAL TABLE source_fts USING fts5(id UNINDEXED,cues,text,tokenize='unicode61');''')
        chapters={chapter_key(c):c for c in library['chapters']}
        for key,c in chapters.items():db.execute('INSERT INTO chapters VALUES(?,?)',(key,dumps(c)))
        for key,c in library['cases'].items():db.execute('INSERT INTO cases VALUES(?,?)',(key,dumps(c)))
        for doc in library['documents']:
            literal=document_text(chapters,doc)
            db.execute('INSERT INTO documents VALUES(?,?)',(doc['id'],dumps(doc)))
            phase_cues={'book_feedback':'后来反馈 回访结果 应验结果','later_disclosure':'后来披露 中途补充','author_correction':'作者更正 纠错 错断'}
            cues=' '.join(doc.get('cues',[]))+' '+doc.get('title','')+' '+phase_cues.get(doc['phase'],'')
            db.execute('INSERT INTO source_fts VALUES(?,?,?)',(doc['id'],' '.join(terms(cues)),' '.join(terms(literal))))
        for key,value in {'schema':SCHEMA,'index_format':INDEX_FORMAT,'sqlite_version':sqlite3.sqlite_version,'source_libraries':library['source_libraries'],'coverage':report}.items():
            db.execute('INSERT INTO meta VALUES(?,?)',(key,dumps(value)))
        db.commit()
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Index integrity failure')
    finally:db.close()
    pending.rename(output)
    write(output.with_suffix(output.suffix+'.receipt.json'),{'schema':SCHEMA,'index_sha256':digest(output.read_bytes()),'coverage':report})
    return report

class Index:
    def __init__(self,path):
        self.path=private(path)
        receipt=read(self.path.with_suffix(self.path.suffix+'.receipt.json'))
        if receipt['schema']!=SCHEMA or digest(self.path.read_bytes())!=receipt['index_sha256']:raise ValueError('Index snapshot/receipt changed')
        self.db=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True)
        version=self.db.execute("SELECT value FROM meta WHERE key='index_format'").fetchone()
        if not version or json.loads(version[0])!=INDEX_FORMAT:self.db.close();raise ValueError('Index tokenizer format differs; build a new snapshot')
        self.chapters={r[0]:json.loads(r[1]) for r in self.db.execute('SELECT * FROM chapters')}
        self.coverage=receipt['coverage']
        self.cases={key:json.loads(raw) for key,raw in self.db.execute('SELECT * FROM cases')}
    def close(self):self.db.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def get(self,doc_id,*,mode):
        row=self.db.execute('SELECT payload FROM documents WHERE id=?',(doc_id,)).fetchone()
        if not row:raise KeyError('Unknown source document')
        doc=json.loads(row[0])
        if doc['kind']!='rule' and mode!='study':raise ValueError('Historical/reference evidence requires source-study mode')
        return self.expand(doc)
    def expand(self,doc):
        expanded=doc|{'content':document_text(self.chapters,doc),'sources':[{
            'book_id':self.chapters[p['chapter']]['book_id'],
            'book_title':self.chapters[p['chapter']]['book_title'],
            'chapter':self.chapters[p['chapter']]['title'],
            'source_revision':self.chapters[p['chapter']]['source_revision'],
            'method_id':self.chapters[p['chapter']]['method_id'],
            **p} for p in doc['parts']]}
        if doc.get('editorial_reading'):
            from .book_rag_editorial import reading_for_document
            expanded['reading_content']=reading_for_document(self.chapters,doc)
            expanded['editorial_notice']='校订阅读稿，非逐字原文；原贴、逐项改动及原始坐标均保留。'
        return expanded
    def case(self,cid):
        row=self.db.execute('SELECT payload FROM cases WHERE id=?',(cid,)).fetchone()
        if not row:raise KeyError('Unknown case')
        docs=[json.loads(r[0]) for r in self.db.execute('SELECT payload FROM documents')]
        return {'case':json.loads(row[0]),'documents':[self.expand(d) for d in docs if cid in d['case_ids']]}
    def search(self,question,*,mode='knowledge',domain=None,book=None,case_id=None,limit=6,exclude_case_ids=(),exclude_event_families=(),method_id=None):
        if mode not in ('knowledge','reading','study'):raise ValueError('Unknown retrieval mode')
        if not isinstance(question,str) or not question.strip() or len(question)>4000:raise ValueError('Question required, maximum 4000 characters')
        if type(limit)is not int or not 1<=limit<=20:raise ValueError('Retrieval limit 1..20')
        if case_id and mode!='study':raise ValueError('Case retrieval is retrospective study only')
        if book and book not in {c['book_id'] for c in self.chapters.values()}:raise ValueError('Unknown book ID')
        requested=domains_for(question,domain);query=list(dict.fromkeys(terms(question)))[:128]
        if not query:return {'documents':[],'domains':requested,'mode':mode,'reason':'no_search_terms'}
        match=' OR '.join('"'+t.replace('"','""')+'"' for t in query)
        # Filter by provenance before ranking, so a pool of historical matches
        # cannot crowd general rules out of a small candidate shortlist.
        candidates=[];query_years=set(re.findall(r'(?<!\d)(?:18|19|20)\d{2}(?!\d)',question))
        for raw,score in self.db.execute('SELECT d.payload,bm25(source_fts,0,2,1) FROM source_fts JOIN documents d ON d.id=source_fts.id WHERE source_fts MATCH ? ORDER BY bm25(source_fts,0,2,1)',(match,)):
            d=json.loads(raw)
            if mode!='study' and d['kind']!='rule':continue
            if book and not any(self.chapters[p['chapter']]['book_id']==book for p in d['parts']):continue
            if method_id and any(self.chapters[p['chapter']]['method_id']!=method_id for p in d['parts']):continue
            if set(d['case_ids']).intersection(exclude_case_ids):continue
            if exclude_event_families and d['kind']=='reference':continue
            if any(self.cases[cid].get('event_family_id') in exclude_event_families or self.cases[cid].get('course_split_group') in exclude_event_families for cid in d['case_ids']):continue
            if case_id and case_id not in d['case_ids']:continue
            if mode=='study' and query_years and not case_id:
                if d['kind']!='case' or not any(query_years.issubset(set(self.cases[cid].get('literal_years',[]))) for cid in d['case_ids']):continue
            scopes={DOMAIN_EQUIVALENTS.get(x,x) for x in d.get('domains',[])}
            if requested and scopes and not scopes.intersection(requested):continue
            literal=document_text(self.chapters,d)
            text=normalize(d.get('title','')+' '+' '.join(d.get('cues',[]))+' '+literal)
            pair_missing=len(ordered_pairs(question)-ordered_pairs(literal))
            symbols=[s for s in (*SYMBOLS,'父亲','母亲','丈夫','妻子','兄弟','同事','考生','本人','年命') if s in normalize(question)]
            missing=sum(s not in text for s in symbols)
            priority=int(bool(requested and scopes.intersection(requested)))
            # Mechanical 排盘 passages cannot outrank role definitions merely
            # by mentioning the same stars, unless the question asks casting.
            mechanical=any(x in question for x in ('排盘','起局','转动','布盘'))
            irrelevant_cast=int(d.get('topic')=='casting' and not mechanical)
            needs_timing=any(x in question for x in ('应期','什么时候','何时','多久'))
            missing_topic=int(needs_timing and 'timing' not in d.get('topics',[d.get('topic')]))
            candidates.append(((irrelevant_cast,pair_missing,missing,missing_topic,-priority,score,d['id']),d))
        if not candidates and case_id and case_id not in exclude_case_ids:
            # Explicit event lookup is allowed even when a generic question
            # has no lexical overlap. Its source identity was supplied by user.
            for raw, in self.db.execute('SELECT payload FROM documents'):
                d=json.loads(raw)
                if d['kind']!='case' or case_id not in d['case_ids']:continue
                if book and not any(self.chapters[p['chapter']]['book_id']==book for p in d['parts']):continue
                if method_id and any(self.chapters[p['chapter']]['method_id']!=method_id for p in d['parts']):continue
                if any(self.cases[cid].get('event_family_id') in exclude_event_families or self.cases[cid].get('course_split_group') in exclude_event_families for cid in d['case_ids']):continue
                candidates.append(((0,0,0,0,0,d['parts'][0]['lines'][0],d['id']),d))
        chosen=[];seen=set();seen_parts=set()
        for rank,d in sorted(candidates,key=lambda x:x[0]):
            key=digest(document_text(self.chapters,d))
            parts={dumps(p) for p in d['parts']}
            if key in seen or parts.issubset(seen_parts):continue
            seen.add(key);seen_parts.update(parts);chosen.append(self.expand(d)|{'retrieval_rank':len(chosen)+1})
            if len(chosen)==limit:break
        return {'documents':chosen,'domains':requested,'domain_ambiguous':len(requested)>1,
            'mode':mode,'index_sha256':digest(self.path.read_bytes()),'case_outcomes_allowed':mode=='study',
            'retrieval_method':'CPU SQLite FTS5 Chinese bigrams + reviewed scope filters; not dense embeddings',
            'retrieval_scores_are_not_event_probabilities':True}


def activate_index(path,pointer=None):
    """Atomically switch the retrieval snapshot; trained weights stay untouched."""
    import os,time
    path=private(path);pointer=private(pointer or ROOT/'local/wengua_rag/active_index.json')
    with Index(path) as index:coverage=index.coverage
    value={'schema':SCHEMA,'path':str(path),'index_sha256':digest(path.read_bytes()),'coverage':coverage}
    pointer.parent.mkdir(parents=True,exist_ok=True)
    if pointer.exists():
        previous=read(pointer)
        if previous==value:return value
        history=pointer.parent/'index_history';history.mkdir(exist_ok=True)
        write(history/(str(time.time_ns())+'.json'),previous)
    pending=pointer.with_name(pointer.name+'.pending-'+str(time.time_ns()));write(pending,value);os.replace(pending,pointer)
    return value


def active_index(pointer=None):
    pointer=private(pointer or ROOT/'local/wengua_rag/active_index.json')
    value=read(pointer)
    if value.get('schema')!=SCHEMA:raise ValueError('Unknown active retrieval snapshot')
    path=private(value['path'])
    if digest(path.read_bytes())!=value['index_sha256']:raise ValueError('Active index changed; refusing an unverified replacement')
    return path
