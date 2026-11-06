"""Literal source selections and explicit dispositions; never generated rules."""
from collections import defaultdict
from wengua_course import data as d
from wengua_course_v54 import corpus as base


def plan(library):
 chapters={base.r.chapter_key(c):c for c in library['chapters']};selected=[];inventory=[];seen=set()
 for doc in library['documents']:
  if doc['kind']!='rule':continue
  reason=None;domains=doc.get('domains',[]);parts=doc['parts'];text=base.r.document_text(chapters,doc)
  cues=[q.split('\n')[-1] for q in doc.get('cues',[]) if '？' in q and '具体条件和原意' not in q]
  if not domains:reason='general_knowledge_in_existing_courses_or_rag'
  elif doc.get('case_ids'):reason='case_scoped_not_promoted_to_general_rule'
  elif not cues:reason='no_reviewed_specific_application_question'
  elif len(text)<25 or len(text)>900 or '[图' in text or '+---' in text:reason='structure_or_context_requires_separate_review'
  key=d.dumps(parts)
  if key in seen:reason='same_literal_selector_already_accounted_for'
  if reason is None:
   seen.add(key);selected.append({'id':doc['id'],'domain':domains[0],'book_id':chapters[parts[0]['chapter']]['book_id'],'question':cues[0],'parts':parts,'text':text,'source_rule_kinds':doc.get('source_rule_kinds',[]),'source_concepts':doc.get('source_concepts',[]),'parent_source_ids':doc.get('parent_source_ids',[])})
  inventory.append({'id':doc['id'],'domains':domains,'parts':parts,'disposition':reason or 'source_rule_lesson_and_workflow_component','not_a_new_observed_case':True})
 groups=defaultdict(list)
 for item in selected:groups[(item['domain'],item['book_id'])].append(item)
 flows=[]
 for (domain,book),items in sorted(groups.items()):
  ordered=sorted(items,key=lambda x:(x['parts'][0]['chapter'],x['parts'][0]['lines'],x['id']))
  # Complete original fragments stay intact; never split a condition/conclusion.
  chunks=[];chunk=[];size=0
  for item in ordered:
   if chunk and (size+len(item['text'])>700 or len(chunk)>=3):chunks.append(chunk);chunk=[];size=0
   chunk.append(item);size+=len(item['text'])
  if chunk:chunks.append(chunk)
  for number,chunk in enumerate(chunks):
   if len(chunk)<2:continue
   parts=list({d.dumps(p):p for x in chunk for p in x['parts']}.values());identity='rag:v55:workflow:'+d.digest(d.dumps(parts))[:24]
   flows.append({'id':identity,'domain':domain,'book_id':book,'number':number+1,'rules':chunk,'document':{'id':identity,'kind':'rule','phase':'general_rule','case_ids':[],'parts':parts,'review_status':'literal_source_workflow_selection','title':d.DOMAIN_ZH.get(domain,domain)+'原书规则组'+str(number+1),'topic':'domain_workflow','topics':['domain_workflow'],'domains':[domain],'cues':[],'parent_source_ids':sorted({s for x in chunk for s in x['parent_source_ids']}),'case_context_complete':False}})
 return {'rules':selected,'rule_inventory':inventory,'workflows':flows,'workflow_order':'literal_source_order_not_an_invented_universal_method'}
