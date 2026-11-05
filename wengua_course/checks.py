"""Conservative checks for declared factual slots, not a general semantic judge.

Free-form case reasoning is always sent to human review. Matching a keyword
is not called professional correctness or real-world predictive accuracy.
"""
from collections import defaultdict
import re

def first_clause(s):return re.split(r'[。！!\n；;]',s.strip(),maxsplit=1)[0].strip(' **：:')
def clean(s):return re.sub(r'[\s，,、。；;：:！!？?*`“”「」]','',s)
def automatically_checkable(check):return bool(check) and check.get('kind') in {'exact_name','state','role','number_pair','relation','identity_swap'}

def evaluate(answer,check,*,empty=False,capped=False):
 if empty or not answer.strip():return {'status':'fail','reason':'empty'}
 if capped:return {'status':'fail','reason':'generation_limit_reached'}
 if check is None:return {'status':'needs_review','reason':'free_case_reasoning_requires_source_review'}
 kind=check['kind'];head=clean(first_clause(answer));body=clean(answer)
 if kind in ('exact_name','state'):
  accepted=[check['answer'],*check.get('aliases',[])];accepted={clean(a) for a in accepted}
  ok=head in accepted or any(head in {f'为{x}',f'是{x}',f'称{x}',f'本书称{x}'} for x in accepted)
 elif kind=='identity_swap':
  ok=any(x in head for x in ('不变','没有变化','不改变')) and check['answer'] in body and not any(x in body for x in ('不是'+check['answer'],'不为'+check['answer']))
 elif kind=='role':
  # Inspect the answer's conclusion, not a later quotation of both possibilities.
  correct=check['answer'];wrong=check['opposite'];ok=correct in head and not any(x in head for x in ['日干','时干','年干','月干'] if x!=correct) and wrong not in head
  ok=ok and all(clean(x) in body for x in check.get('required',[]))
 elif kind=='number_pair':
  chinese={'一':'1','二':'2','三':'3','四':'4','五':'5','六':'6','七':'7','八':'8','九':'9','十':'10'}
  text=first_clause(answer)
  for k,v in chinese.items():text=text.replace(k,v)
  values=[int(n) for n in re.findall(r'\d+',text)];ok=len(values)==2 and sorted(values)==sorted(check['answer'])
 elif kind=='relation':
  # Natural answer must identify a direction, not merely repeat the supplied elements.
  candidates={'受生':['受生','被后者所生','后者生前者','后者生前者'],'受克':['受克','被后者克','后者克前者'],'生':['前者生后者','生'],'克':['前者克后者','克'],'比和':['比和','同属','相同五行']}
  expected=check['answer'];found=set()
  for label,terms in candidates.items():
   for term in terms:
    if head==term or term!='生' and term!='克' and term in head:found.add(label)
  ok=found=={expected} and not any(w in head for w in ('不是','并非','不应','不为'))
 elif kind=='mentions':
  # A limited rubric-constraint screen; positive output is not an expert approval.
  ok=all(any(clean(t) in body for t in options) for options in check['required']) and not any(clean(t) in body for t in check.get('forbidden',[]))
 else:raise ValueError('Unknown factual check '+kind)
 if kind=='mentions':return {'status':'needs_review','reason':'rubric_terms_are_not_semantic_truth','constraints_satisfied':ok,'kind':kind,'professional_semantics_verified':False}
 return {'status':'pass' if ok else 'needs_review','reason':'declared_slot_constraints_satisfied' if ok else 'slot_not_unambiguously_satisfied','kind':kind,'professional_semantics_verified':False}

def score(rows,prompts):
 gold={r['id']:r for r in prompts};seen=set();items=[];topics=defaultdict(list)
 for r in rows:
  if r['id'] not in gold or r['id'] in seen:raise ValueError('Unknown or duplicate evaluated task')
  seen.add(r['id']);g=gold[r['id']]
  if r['question']!=g['messages'][:-1] or r['expected']!=g['answer']:raise ValueError('Evaluation prompt/target provenance changed')
  check=g.get('checks');result=evaluate(r['answer'],check,empty=r.get('empty',False),capped=r.get('reached_token_limit',False))
  item={'id':g['id'],'domain':g['domain'],'task':g['task'],'group':g['group'],'result':result,'automatically_checkable':automatically_checkable(check)};items.append(item)
  if automatically_checkable(check):topics[g['domain']+'/'+g['task']].append(int(result['status']=='pass'))
 macro=sum(sum(v)/len(v) for v in topics.values())/len(topics) if topics else 0.
 return {'outputs':len(rows),'requested':len(gold),'complete':len(seen)==len(gold),'checkable_outputs':sum(i['automatically_checkable'] for i in items),'topic_macro_constraint_pass_rate':macro,'per_topic':{k:{'pass':sum(v),'total':len(v)} for k,v in topics.items()},'unreviewed_case_reasoning':sum(not i['automatically_checkable'] for i in items),'failures_or_review_required':sum(i['result']['status']!='pass' for i in items),'semantic_professional_approval':False,'independent_prediction_approval':False,'items':items}

def rank(summary):
 if not summary['complete']:raise ValueError('Cannot select an incomplete evaluation')
 # All candidates get the same predeclared source-seen development questions.
 # Used only to choose an initialization / review candidate, never deployment.
 return (summary['topic_macro_constraint_pass_rate'],-sum(i['result']['reason'] in ('empty','generation_limit_reached') for i in summary['items']))
