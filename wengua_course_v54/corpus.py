"""Literal property selectors and the frozen TWO-book retrieval implementation.

Seventeen existing property definitions get focused source records. No text,
case outcome, year of birth or traditional outcome score is generated here.
"""
from copy import deepcopy
from pathlib import Path
from functools import lru_cache
import gzip,json,re,sys,importlib.util
from wengua_course import data as d
from wengua_course_v53 import runtime as prior,rag as old
from wengua_corpus import book_rag as original

ROOT=prior.ROOT
_spec=importlib.util.spec_from_file_location('wengua_corpus._v54_retriever',original.__file__)
r=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=r;_spec.loader.exec_module(r)
r.INDEX_FORMAT+=':v54_literal_property_selectors'
r.SYMBOLS=old.r.SYMBOLS;r.DOMAINS=deepcopy(old.r.DOMAINS);r.terms=old.lexical_terms
DOORS={'休':'水','生':'土','伤':'木','杜':'木','景':'火','死':'土','惊':'金','开':'金'}

@lru_cache(maxsize=1)
def source_library():return json.loads(gzip.decompress((prior.DATA/'rag_sources.json.gz').read_bytes()))

def part_for(unit,quote,chapters):
 chapter=unit['book_id']+'/'+str(unit['section_id']);c=chapters[chapter];lo,hi=unit['lines'];raw='\n'.join(c['original_text'].splitlines()[lo-1:hi]);start=raw.find(quote)
 if start<0:
  indices=[i for i,x in enumerate(raw) if not x.isspace()];normalized=''.join(raw[i] for i in indices);needle=d.compact(quote);at=normalized.find(needle)
  if at<0 or normalized.find(needle,at+1)>=0:raise ValueError('Property quote has no unique literal source selector')
  start,end=indices[at],indices[at+len(needle)-1]+1
 else:
  if raw.find(quote,start+1)>=0:raise ValueError('Ambiguous property source selector')
  end=start+len(quote)
 return {'chapter':chapter,'lines':[lo,hi],'chars':[start,end],'source_sha256':c['source_sha256'],'quote_sha256':d.digest(raw[start:end])}

def properties(parent,library):
 chapters={r.chapter_key(c):c for c in library['chapters']};units=parent['units'];items={}
 matrix=units['book:22:L61-64:a7bc478e0fe9']
 grids=[u for sid,u in units.items() if sid.startswith('book:22:') and '[图6 后天八卦九宫格]' in u['original_quote']]
 if len(grids)!=1:raise ValueError('Original numbered palace table is ambiguous')
 for n,name in d.PALACE.items():
  quotes=[name if n!=5 else '中五',name[0]+d.ELEMENT[n] if n!=5 else '中五土'];supports=[{'source_id':grids[0]['id'],'quote':quotes[0]},{'source_id':matrix['id'],'quote':quotes[1]}]
  items['palace:'+str(n)]={'element':d.ELEMENT[n],'label':name+'宫','support':supports,'parts':[part_for(grids[0],quotes[0],chapters),part_for(matrix,quotes[1],chapters)]}
 for door,element in DOORS.items():
  candidates=[(sid,u) for sid,u in units.items() if sid.startswith('book:26:L46-46:') and door+'门' in u['original_quote']]
  sid,u=min(candidates,key=lambda x:len(x[1]['original_quote']))
  if element not in u['original_quote']:raise ValueError('Door element differs from literal source')
  support={'source_id':sid,'quote':u['original_quote']}
  items['door:'+door]={'element':element,'label':door+'门','support':[support],'parts':[part_for(u,support['quote'],chapters)]}
 element=next(row for row in parent['records'] if row['task']=='element_direction')
 for key,sid in zip(['rule:generation','rule:control'],element['source_ids']):
  u=units[sid];items[key]={'label':key,'support':[{'source_id':sid,'quote':u['original_quote']}],'parts':[part_for(u,u['original_quote'],chapters)]}
 sid='book:3:L697-711:1130fce7c96b';u=units[sid]
 items['rule:door_relation']={'label':'门宫命名规则','support':[{'source_id':sid,'quote':u['original_quote']}],'parts':[part_for(u,u['original_quote'],chapters)]}
 return items

def indexed_library(parent,library):
 indexed,changes=old.classified_library(library);facts=properties(parent,library)
 for key,fact in facts.items():
  parts=fact['parts'];identity='rag:v54:property:'+d.digest(d.dumps(parts))[:24]
  doc={'id':identity,'kind':'rule','phase':'general_rule','case_ids':[],'parts':parts,'review_status':'reviewed_source_selector','title':'原书符号五行属性','topic':'literal_property','topics':['literal_property'],'domains':[],'cues':['v54_property:'+key],'parent_source_ids':[s['source_id'] for s in fact['support']],'case_context_complete':False}
  indexed['documents'].append(doc);fact['document_id']=identity
 return indexed,facts,changes

def property_keys(question):
 if not any(w in question for w in ('五行','金木水火土','属性','归于','属','门宫','生克','关系','迫','制','和','义')):return []
 found=['door:'+door for door in DOORS if door+'门' in question]
 for n,name in d.PALACE.items():
  if re.search(re.escape(name[0])+r'(?:'+re.escape(name[1:])+'|'+str(n)+r')?宫',question):found.append('palace:'+str(n))
 door_relation='门宫' in question or bool(any(x.startswith('door:') for x in found) and any(x.startswith('palace:') for x in found))
 if any(x in question for x in ('五行关系','生克','受生','受克','前者','后者')) or door_relation:found+=['rule:generation','rule:control']
 if door_relation:found.append('rule:door_relation')
 return found

def search_query(question):
 text=old.search_query(question)
 text=re.sub(r'上下层(?:都|均)(?:是|为)\s*([甲乙丙丁戊己庚辛壬癸])',lambda m:'天盘'+m[1]+'地盘'+m[1],text)
 for label in ['天盘','地盘']:
  text=re.sub(label+r'(?:干)?\s*(?:为|是|：|:)?\s*([甲乙丙丁戊己庚辛壬癸])',label+r'\1',text)
 heaven=set(re.findall(r'天盘([甲乙丙丁戊己庚辛壬癸])',text));earth=set(re.findall(r'地盘([甲乙丙丁戊己庚辛壬癸])',text))
 if len(heaven)==len(earth)==1:
  pair=(next(iter(heaven)),next(iter(earth)))
  if pair not in r.ordered_pairs(text):text+=' 天盘'+pair[0]+'地盘'+pair[1]
 return text

_Base=r.Index
class Index(_Base):
 def search(self,question,**kwargs):
  limit=kwargs.get('limit',8)
  if type(limit)is not int or not 1<=limit<=20:raise ValueError('Retrieval limit 1..20')
  found=super().search(question,**(kwargs|{'limit':20}));docs=found['documents'];domains=r.domains_for(question,kwargs.get('domain'))
  if len(domains)>1:
   groups=[super(Index,self).search(question,**(kwargs|{'domain':domain,'limit':3}))['documents'] for domain in domains]
   docs=[group[n] for n in range(3) for group in groups if len(group)>n]+docs
  symbols=old.explicit_symbols(question)
  if symbols and any(w in question for w in ('阴','阳','五行','属于','属什么','方向')) and not r.ordered_pairs(question):
   docs.sort(key=lambda x:(sum(s not in x['content'] for s in symbols),x['retrieval_rank']))
  # The lookup selects original passages, never a generated property answer.
  wanted=set('v54_property:'+key for key in property_keys(question));property_docs=[]
  if wanted and found.get('mode')!='study':
   for payload, in self.db.execute("SELECT payload FROM documents WHERE payload LIKE '%v54_property:%'"):
    doc=json.loads(payload)
    if wanted.intersection(doc['cues']):
     if kwargs.get('book') and any(self.chapters[p['chapter']]['book_id']!=kwargs['book'] for p in doc['parts']):continue
     property_docs.append(self.expand(doc))
  docs=list({x['id']:x for x in [*property_docs,*docs]}.values())[:limit]
  for n,doc in enumerate(docs,1):doc['retrieval_rank']=n
  return found|{'documents':docs}
r.Index=Index

def ensure_index():
 path=ROOT/'local/rag/two_books_v54.sqlite3'
 if not path.exists():
  library,facts,changes=indexed_library(prior.verify_data(),source_library());report=r.build_index([library],path)
  d.write(path.with_suffix('.properties.json'),{'facts':facts,'metadata_changes':changes,'index_report':report,'all_property_text_is_literal':True,'source_text_changed':False})
 return path

def retrieve(question,*,domain=None,book=None,mode='knowledge',limit=8):
 if book is None:
  if '开悟之门' in question and '神奇之门' not in question:book='kaiwu'
  elif '神奇之门' in question and '开悟之门' not in question:book='shenqi'
 with r.Index(ensure_index()) as index:return index.search(search_query(question),domain=domain,book=book,mode=mode,limit=limit)

def prompt(question,tokenizer,*,domain=None,book=None,max_new_tokens=1024,max_context=4096,time_text=None,zone=None,chart_file=None,facts=None):
 prior.configure()
 from src.wengua.book_rag_query import make_prompt,cast_time,checked_chart
 if chart_file and time_text:raise ValueError('Supply either time or a checked program chart')
 packet=checked_chart(d.read(chart_file)) if chart_file else None
 if time_text:
  if not zone:raise ValueError('The existing calendar needs an explicit IANA timezone')
  packet=cast_time(time_text,zone)
 found=retrieve(question,domain=domain,book=book,mode='reading' if packet else 'knowledge')
 with r.Index(ensure_index()) as index:
  for doc in found['documents']:
   for source in doc['sources']:
    source['chapter_id']=source['chapter'];source['chapter']=index.chapters[source['chapter_id']]['title']
 return make_prompt(found,question,tokenizer,packet=packet,facts=facts,max_context=max_context,max_new_tokens=max_new_tokens)

def covering_citations(packet,support,units):
 """Every required literal quote must occur in an actually visible source."""
 required=[];chapters={r.chapter_key(c):c for c in source_library()['chapters']}
 def bounds(part):
  key=part.get('chapter_id',part['chapter']);raw=chapters[key]['original_text'];lo,hi=part['lines'];prefix=sum(len(x)+1 for x in raw.splitlines()[:lo-1]);slice_text='\n'.join(raw.splitlines()[lo-1:hi]);a,b=part.get('chars',[0,len(slice_text)]);return prefix+a,prefix+b
 for fragment in support:
  u=units[fragment['source_id']];quote=d.compact(fragment['quote']);matches=[]
  needed=part_for(u,fragment['quote'],chapters);needed_lo,needed_hi=bounds(needed)
  for doc in packet['references']:
   if quote not in d.compact(doc.get('reading_content',doc['content'])):continue
   for s in doc['sources']:
    if s['book_id']!=u['book_id'] or s.get('chapter_id',s['chapter'])!=u['book_id']+'/'+str(u['section_id']) or s['source_sha256']!=u['source_sha256']:continue
    source_lo,source_hi=bounds(s)
    if source_lo<=needed_lo and source_hi>=needed_hi:matches.append(doc['citation']);break
  if not matches:return None
  required.append(matches[0])
 return list(dict.fromkeys(required))
