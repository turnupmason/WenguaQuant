"""Source-only repair: worked relations, typed symbols and actual RAG prompts.

Old V53 development answers are already exposed. General knowledge can be
rehearsed as repair, but its old scores remain seen regression diagnostics.
Held-out CASE families and the original final test never become training.
"""
from copy import deepcopy
from collections import Counter
import re
from wengua_course import data as d
from wengua_course_v53 import data as old
from . import corpus

SCHEMA='wengua_actual_rag_worked_source_course_v54'

def bridge(a,b):
 pa={v:k for k,v in d.ELEMENT.items()};relation=d.relation(pa[a],pa[b])
 if relation=='比和':return '两者同属'+a+'，所以为比和。'
 if relation=='生':return a+'生'+b+'，是前者生后者。'
 if relation=='克':return a+'克'+b+'，是前者克后者。'
 if relation=='受生':return b+'生'+a+'，是后者生前者，所以前者受生。'
 return b+'克'+a+'，是后者克前者，所以前者受克。'

def plain_answer(spec):
 answer=spec['answer']
 if spec['key'].startswith('role_'):
  target=re.search(r'这一条规则，(.+?)以什么符号代表',spec['question'])
  if target:return target[1]+'以'+answer.rstrip('。')+'为代表符号。'
 if spec['key'].startswith('named_pair_'):
  pair=spec['key'].split('_')[-1]
  return '天盘'+pair[0]+'加地盘'+pair[1]+'，本条称为'+answer.rstrip('。')+'。'
 if spec.get('conditions') and spec['key'].startswith(('bundle_','premise_completeness_')):
  if 'False' not in spec['key']:
   conclusion=answer.split('；')[0].split('。')[0]
   return conclusion+'。这里已核实'+spec['conditions'][0]+'，应保留这些适用条件。'
 return answer

def record(question,answer,support,units,*,skill,split='train',domain='foundation',checks=None,task=None,group=None,case_ids=(),context='provided_evidence',messages=None,provenance=None):
 row=d.row(question,answer,[x['source_id'] for x in support],units,task=task or skill,format='chart_application' if skill in ('direction','door_relation','property') else 'conditional_dialogue' if skill in ('identity','condition') else 'foundation_qa',split=split,domain=domain,checks=checks,group=group,case_ids=case_ids,messages=messages,provenance=provenance or {})
 row.update(id='v54:'+d.digest(d.dumps([skill,split,row['messages'],context]))[:24],stage='application',skill=skill,context=context)
 return row

def add_rag(rows,missing,question,answer,support,units,tok,**kw):
 # A real retrieval call is used; no missing gold passage is appended.
 domain=kw.get('domain','foundation');packet=corpus.prompt(question,tok,domain=domain if domain in corpus.r.DOMAINS else None)
 citations=corpus.covering_citations(packet,support,units) if packet['state']=='ready' else None
 if not citations:
  missing.append({'skill':kw['skill'],'question':question,'support_ids':[s['source_id'] for s in support],'action':'not_a_teacher_required_literal_source_not_visible'});return
 target=answer.rstrip()+' '+'、'.join('['+x+']' for x in citations)
 messages=packet['messages']+[{'role':'assistant','content':target}]
 row=record(question,target,support,units,messages=messages,**kw)
 # d.row supplies source scope for ordinary rows; actual RAG messages stay exact.
 row['messages']=messages;row['question']=messages[1]['content']
 row['provenance'].update(actual_production_rag_prompt=True,required_citations=citations,support=deepcopy(support),references=[{'id':x['id'],'citation':x['citation'],'sources':x['sources']} for x in packet['references']],prompt_budget=packet['prompt_budget'],teacher_is_model_output=False)
 rows.append(row)

def add_dialogue(rows,missing,spec,units,tok,*,split='train'):
 raw=spec['messages'];users=[m['content'] for m in raw if m['role']=='user'];first_answer=raw[1]['content'];support=spec['support']
 # These identity turns use the general relationship rule. A separate birth-year
 # definition is not needed when neither turn identifies a concrete year stem.
 if spec['key'].startswith('health_switch_'):support=[s for s in support if s['source_id']!='book:4:L180-180:cd8e3175a112']
 scope=d.source_scope([x['source_id'] for x in support],units);first_packet=corpus.prompt('按'+scope+'，'+users[0],tok,domain=spec['domain'],max_context=1536,max_new_tokens=256)
 final_question=users[-1] if split=='train' else '接着上一轮，请按当前实际提问者重新取用：'+spec['development_question']
 final_packet=corpus.prompt('按'+scope+'，'+final_question,tok,domain=spec['domain'],max_context=1536,max_new_tokens=256)
 first_cites=corpus.covering_citations(first_packet,support,units);final_cites=corpus.covering_citations(final_packet,support,units)
 if not first_cites or not final_cites:
  missing.append({'skill':'identity','question':final_question,'action':'dialogue_not_taught_missing_literal_source'});return
 suffix=lambda labels:' '+'、'.join('['+x+']' for x in labels)
 answer=spec['answer']+suffix(final_cites);messages=[first_packet['messages'][0],first_packet['messages'][1],{'role':'assistant','content':first_answer+suffix(first_cites)},final_packet['messages'][1],{'role':'assistant','content':answer}]
 before='时干' if spec['key'].startswith('exam_switch_') else '年干' if spec['key']=='health_switch_to_self' else '日干';after=spec['checks']['answer']
 row=record(final_question,answer,support,units,skill='identity',split=split,domain=spec['domain'],checks=spec['checks'],task='dialogue_'+spec['key'],messages=messages,provenance={'kind':'source_verified_actual_role_transition','role_transition':{'before':before,'after':after},'source_user_turns':users,'required_citations':final_cites,'first_turn_citations':first_cites,'actual_production_rag_prompt':True,'support':deepcopy(support),'turn_references':[[{'id':r['id'],'citation':r['citation'],'sources':r['sources']} for r in p['references']] for p in (first_packet,final_packet)],'turn_budgets':[first_packet['prompt_budget'],final_packet['prompt_budget']],'teacher_is_model_output':False})
 row['messages']=messages;row['question']=messages[-2]['content'];rows.append(row)

def worked_specs(parent):
 specs=deepcopy(parent['reviewed_specs']);facts=corpus.properties(parent,corpus.source_library())
 for spec in specs:
  key=spec['key']
  if key=='health_reliable_birth_stem':spec['development_question']='女儿代问父亲，已经确认父亲出生年干为癸；兼看年命时能否换成别的干或另造出生年？'
  if key=='health_daughter_father':spec['answer']='女儿代问父亲，患者身份先看年干，兼看乾宫；天芮用于看疾病，天心星和乙奇用于看医生、医药。出生年未知，不补造年命。'
  if key=='health_father_self':spec['answer']='父亲本人问自己的病，患者身份取日干。天芮代表疾病；家庭中的父亲身份不能使本人的主用神自动改成年干。'
  if key=='health_sibling':spec['answer']='代问同辈兄弟，患者身份先看月干。长幼和出生年未提供时，不补相应兼用宫或年命。'
  if key=='health_parent_child':spec['answer']='父母代问子女，以时干看患者身份；兼用宫应按已知子女性别、长幼另定，这些条件未知时先不补。'
  if key.startswith('star_state_'):
   n=[3,1,9,6,2][int(key.rsplit('_',1)[1])];state=spec['checks']['answer'];element=d.ELEMENT[n]
   spec['question']=spec['question'].replace('水星天蓬','天蓬星（五行属水）')+'请同时说明星与宫的五行生克。';spec['development_question']=spec['development_question'].replace('水星天蓬','天蓬星（五行属水）')+'请同时说明星与宫的五行生克。'
   spec['support']+=facts['palace:'+str(n)]['support']+facts['rule:generation']['support']+facts['rule:control']['support']
   spec['answer']=state+'。天蓬属水，'+d.PALACE[n]+'宫属'+element+'；'+bridge('水',element)+'这里只按本条九星落宫法取'+state+'。';spec['worked_answer']=True
  if key.startswith('door_direction_'):
   _,_,door,n=key.split('_');n=int(n);a=corpus.DOORS[door];b=d.ELEMENT[n];rel=d.relation({v:k for k,v in d.ELEMENT.items()}[a],n)
   spec['answer']=spec['checks']['answer']+'。'+door+'门属'+a+'，'+d.PALACE[n]+'宫属'+b+'；'+bridge(a,b)+'本条以门对宫的方向命名。'
   spec['support']=[*facts['door:'+door]['support'],*facts['palace:'+str(n)]['support'],*facts['rule:control' if rel in ('克','受克') else 'rule:generation']['support'],*facts['rule:door_relation']['support']]
   spec['question']='根据两书属性定义和门宫命名规则，'+door+'门落'+d.PALACE[n]+'宫。请先认清五行，再说明门对宫的生克及迫、制、和、义、比和名称。'
   spec['development_question']='请用原文说明'+d.PALACE[n]+'宫中的'+door+'门属于哪一种门宫关系，并解释两者五行和作用方向。'
   spec['worked_answer']=True;spec['skill']='door_relation'
  elif key.startswith('health_') or key.startswith('exam_') or key.startswith('missing_proxy_'):spec['skill']='identity'
  elif spec['format']=='conditional_dialogue':spec['skill']='condition'
  else:spec['skill']='source_fact'
  if key.startswith('role_'):
   match=re.search(r'这一条规则，(.+?)以什么符号代表',spec['question'])
   if match:spec['checks']['role_label']=match[1]
  spec['answer']=spec['answer'] if spec.get('worked_answer') else plain_answer(spec)
 return specs,facts

def build(parent,tok,*,progress=lambda **kw:None):
 units=parent['units'];rows=[];missing=[];specs,facts=worked_specs(parent)
 # Actual retrieved book use. Old noncase knowledge paraphrases are known repair,
 # while new alternate wording is a seen-knowledge development check.
 for n,spec in enumerate(specs):
  if spec.get('case_ids'):continue
  scope=d.source_scope([x['source_id'] for x in spec['support']],units);question=spec['question']
  history=spec.get('messages')
  if history:
   users=[m['content'] for m in history if m['role']=='user'];question='此前问题：'+users[0]+'\n当前问题：'+users[-1]
  variants=[question]
  if spec.get('development_question'):
   variants.append(spec['development_question'])
  for q in dict.fromkeys(variants):
   q='按'+scope+'，'+q
   if '上层' in q or '下层' in q:q+='这里的上层、下层明确指同宫天盘干与地盘干。'
   add_rag(rows,missing,q,spec['answer'],spec['support'],units,tok,skill=spec['skill'],domain=spec['domain'],checks=spec.get('checks'),task='source_'+spec['key'],provenance={'kind':'source_derived_repair_not_unseen_knowledge','key':spec['key']})
  q='依据'+scope+'的原文，请回答并解释必要条件：'+question
  add_rag(rows,missing,q,spec['answer'],spec['support'],units,tok,skill=spec['skill'],split='validation',domain=spec['domain'],checks=spec.get('checks'),task='source_'+spec['key'],provenance={'kind':'same_knowledge_alternate_wording_not_new_event','key':spec['key']})
  if history:
   add_dialogue(rows,missing,spec,units,tok)
   add_dialogue(rows,missing,spec,units,tok,split='validation')
  if n%30==0:progress(spec=n)
 # Exact ordered-pair definitions, with layer types taught explicitly.
 names={}
 for row in parent['records']:
  if row['task']=='ordered_name' and row['split']=='train' and row['stage']=='foundation' and row['context']=='closed_book':names[row['provenance']['pair']]=row
 for n,(pair,source) in enumerate(sorted(names.items())):
  support=[{'source_id':sid,'quote':units[sid]['original_quote']} for sid in source['source_ids']];label=source['checks']['answer'];answer='天盘'+pair[0]+'加地盘'+pair[1]+'，本条称为'+label+'。'
  for split,q in [('train',f'同宫的天盘干为{pair[0]}、地盘干为{pair[1]}，按《开悟之门》本条怎么命名？'),('train',f'《开悟之门》中，上层明确是天盘{pair[0]}，下层明确是地盘{pair[1]}。请说明格名和上下层条件。'),('validation',f'请按《开悟之门》核对：天盘{pair[0]}加地盘{pair[1]}，其名称是什么？这两个干属于天地盘，不能改作年干或时干。')]:
   add_rag(rows,missing,q,answer,support,units,tok,skill='typed_pair',split=split,domain='formation',checks=source['checks']|{'pair':pair},provenance={'kind':'literal_typed_pair_definition','pair':pair})
  # Logical type contrast, not a made-up historical chart or outcome.
  q=f'这里只给年干{pair[0]}和时干{pair[1]}，没有给出天地盘同宫位置。能据此认定天盘{pair[0]}加地盘{pair[1]}的{label}格吗？按《开悟之门》定义说明。'
  answer=f'不能据此认定。年干、时干是时间柱资料；本条格局要求天盘{pair[0]}与地盘{pair[1]}同宫，还缺天地盘位置证据。'
  add_rag(rows,missing,q,answer,support,units,tok,skill='typed_pair',domain='formation',checks={'kind':'insufficient'},provenance={'kind':'time_stem_is_not_a_printed_same_palace_pair','pair':pair})
  if n%20==0:progress(pair=n)
 # Literal attributes, followed by transparent directed reasoning.
 for key,fact in facts.items():
  if not key.startswith(('door:','palace:')):continue
  answer=fact['label']+'属'+fact['element']+'。这是本题采用的符号五行属性。'
  for split,q in [('train',fact['label']+'是什么五行？'),('train','请说明'+fact['label']+'的五行属性。'),('validation','原书中'+fact['label']+'归于金木水火土的哪一类？')]:
   add_rag(rows,missing,q,answer,fact['support'],units,tok,skill='property',split=split,checks={'kind':'property','symbol':fact['label'],'element':fact['element']},provenance={'kind':'literal_property','property_key':key})
 representative={'水':1,'土':2,'木':3,'金':6,'火':9};support=facts['rule:generation']['support']+facts['rule:control']['support']
 for a,pa in representative.items():
  for b,pb in representative.items():
   rel=d.relation(pa,pb);answer=rel+'。'+bridge(a,b)
   for split,q in [('train',f'五行关系练习：前者属{a}，后者属{b}。从前者对后者的方向，是生、克、受生、受克还是比和？请说明原因。'),('train',f'对象甲属{a}，对象乙属{b}。只判断甲对乙的五行生克关系，解释谁生谁或谁克谁；不据此预测事件结果。'),('validation',f'请按原书五行生克说明：以属{a}的前者为观察主体，它对于属{b}的后者是什么关系？')]:
    add_rag(rows,missing,q,answer,support,units,tok,skill='direction',split=split,checks={'kind':'relation','answer':rel,'elements':[a,b]},provenance={'kind':'book_element_cycles_worked_direction','elements':[a,b]})
 # The real source charts and author chains stay in their existing CASE split.
 keep={'chart_direction','case_source_chain','author_source_application','case_author_supplement'}
 for source in parent['records']:
  if source['task'] not in keep or source['split']=='diagnostic':continue
  row=deepcopy(source);row['id']='v54:case:'+source['id'];row.update(stage='application',skill='chart_direction' if source['task']=='chart_direction' else 'case')
  if row['task']=='chart_direction':
   pos=row['provenance']['positions'];(a,pa),(b,pb)=list(pos.items());rel=d.relation(pa,pb)
   answer=rel+'。'+a+'星在'+d.PALACE[pa]+'宫属'+d.ELEMENT[pa]+'，'+b+'星在'+d.PALACE[pb]+'宫属'+d.ELEMENT[pb]+'；'+bridge(d.ELEMENT[pa],d.ELEMENT[pb])
   row['answer']=answer;row['messages'][-1]['content']=answer
  rows.append(row)
 # Preserve every original question for honest before/after regression reports.
 from wengua_course_v53.runtime import choose_development
 unique={}
 for row in rows:
  key=(row['split'],d.dumps(row['messages'][:-1]))
  if key in unique and unique[key]['answer']!=row['answer']:raise ValueError('Conflicting teaching prompts')
  unique.setdefault(key,row)
 rows=list(unique.values());result={'schema':SCHEMA,'records':rows,'missing_retrieval':missing,'legacy_v53_prompts':choose_development(parent,128,'foundation'),'legacy_diagnostics':[x for x in parent['records'] if x['split']=='diagnostic'],'source_books':parent['source_books'],'original_final_test_used':False,'teacher_model_answers_used':False,'new_books_used':False,'old_manual_scores_used':False,'source_parent_schema':parent['schema']}
 result['summary']={'records':len(rows),'splits':dict(Counter(x['split'] for x in rows)),'skills':dict(Counter(x['skill'] for x in rows)),'actual_rag_rows':sum(x['provenance'].get('actual_production_rag_prompt',False) for x in rows),'retrieval_missing_not_taught':len(missing),'cases_preserved':len(parent['case_catalog']),'author_steps_preserved':sum(len(c['author_steps']) for c in parent['case_catalog']),'known_legacy_questions':128,'original_final_test_used':False};return result

def compose(parent,delta):
 result=deepcopy(parent);result.update(delta);result['case_stage_index']=parent.get('case_stage_index',{});return result

def validate(pack):
 d.validate(pack)
 if pack['schema']!=SCHEMA:raise ValueError('V54 schema required')
 for row in pack['records']:
  if row['stage']!='application':raise ValueError('One explicit source-use stage required')
  roles=[m['role'] for m in row['messages']]
  if roles[0]!='system' or roles[1:]!=['user' if i%2==0 else 'assistant' for i in range(len(roles)-1)]:raise ValueError('Teacher conversation must start with system and alternate user/assistant')
  if any(not isinstance(m['content'],str) or not m['content'].strip() for m in row['messages']):raise ValueError('Empty teacher message')
  if row['provenance'].get('role_transition'):
   from wengua_course.checks import first_clause
   transition=row['provenance']['role_transition']
   if transition['before'] not in first_clause(row['messages'][2]['content']) or transition['after'] not in first_clause(row['messages'][-1]['content']):raise ValueError('Dialogue answer is attached to the wrong identity turn')
  if row['provenance'].get('actual_production_rag_prompt'):
   if not row['provenance']['required_citations'] or '[S' not in row['answer']:raise ValueError('Uncited teacher')
   for support in row['provenance']['support']:
    if d.compact(support['quote']) not in d.compact(pack['units'][support['source_id']]['original_quote']):raise ValueError('Teacher changed source quote')
 return True
