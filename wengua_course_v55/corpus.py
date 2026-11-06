"""Existing literal two-book RAG plus focused original workflow selectors."""
from functools import lru_cache
from wengua_course_v54 import corpus as base
from wengua_course_v53 import runtime as prior
from . import sourceplan
r=base.r

@lru_cache(maxsize=1)
def source_plan():return sourceplan.plan(base.source_library())

def retrieve(question,*,domain=None,book=None,mode='knowledge',limit=8):
 if book is None:
  if '开悟之门' in question and '神奇之门' not in question:book='kaiwu'
  elif '神奇之门' in question and '开悟之门' not in question:book='shenqi'
 found=base.retrieve(question,domain=domain,book=book,mode=mode,limit=limit)
 if mode=='knowledge' and any(w in question for w in ['规则组','流程','步骤','判断顺序']):
  domains=r.domains_for(question,domain);flows=[f for f in source_plan()['workflows'] if f['domain'] in domains and (book is None or f['book_id']==book)]
  # A caller can ask for a particular source group. Its original ID is a source
  # selection key, not an answer appended after an unsuccessful search.
  specific=[f for f in flows if f['id'] in question]
  if specific:flows=specific
  else:
   from wengua_course import data as d
   compact=d.compact(question)
   flows.sort(key=lambda f:(-sum(d.compact(x['question'].rstrip('？')) in compact for x in f['rules']),f['number']))
  with r.Index(base.ensure_index()) as index:docs=[index.expand(f['document']) for f in flows[:3]]
  found['documents']=list({x['id']:x for x in [*docs,*found['documents']]}.values())[:limit]
 return found

def prompt(question,tokenizer,*,domain=None,book=None,max_new_tokens=1024,max_context=4096,time_text=None,zone=None,chart_file=None,facts=None):
 prior.configure()
 from src.wengua.book_rag_query import make_prompt,cast_time,checked_chart
 from wengua_course import data as d
 if chart_file and time_text:raise ValueError('Supply time or a program chart')
 packet=checked_chart(d.read(chart_file)) if chart_file else None
 if time_text:
  if not zone:raise ValueError('The existing program calendar requires an IANA timezone')
  packet=cast_time(time_text,zone)
 found=retrieve(question,domain=domain,book=book,mode='reading' if packet else 'knowledge')
 with r.Index(base.ensure_index()) as index:
  for doc in found['documents']:
   for source in doc['sources']:
    source['chapter_id']=source['chapter'];source['chapter']=index.chapters[source['chapter_id']]['title']
 return make_prompt(found,question,tokenizer,packet=packet,facts=facts,max_context=max_context,max_new_tokens=max_new_tokens)

covering_citations=base.covering_citations
