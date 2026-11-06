"""V55 declared-slot and contradiction checks; open semantics remain reviewed.

Saved V53/V54 scores are never overwritten. Natural sentences and explicit
reasoning are inspected separately, including reversed subject directions.
"""
import re
from collections import defaultdict
from wengua_course import data as d,checks as elementary
from wengua_course_v54 import checks as old
from wengua_course_v53 import checks as previous


def positive_occurrences(text,label):
 for m in re.finditer(re.escape(label),text):
  if re.search(r'(?:不是|并非|不为|不取|不用|不应|不能|不可|不)(?:称为|叫作|取|用|说|认为|保证)?$',text[max(0,m.start()-9):m.start()]):continue
  yield m


def question_text(gold):
 text=gold['messages'][-2]['content']
 if '用户问题：' in text:text=text.split('用户问题：',1)[1].split('\n本题',1)[0]
 return text.split('本题检索证据：',1)[0]


def unknown_year_misuse(answer,gold):
 q=question_text(gold);unknown=gold.get('provenance',{}).get('birth_year_unknown_in_question') or bool(re.search(r'(?:不知|不知道|未知|未提供|没提供|没有|没给)[^。；\n]{0,12}出生年|出生年[^。；\n]{0,6}(?:未知|不知|没提供)',q))
 if not unknown:return False
 for m in re.finditer(r'(?:兼看|兼取|取|看|计算|使用)年命',answer):
  before=answer[max(0,m.start()-40):m.start()]
  after=answer[m.end():m.end()+36]
  if re.search(r'(?:不|不能|暂不|先不|不宜|不要|不可)$',before):continue
  if any(w in before for w in ['可靠出生年','提供出生年','补充出生年','有出生年','出生年明确','确认出生年']):continue
  if before.endswith('需要') and any(w in after for w in ['补充可靠出生年','提供出生年','补充出生年']):continue
  return True
 return False


def reasoning_conflicts(answer,check):
 conflicts=old.reasoning_conflicts(answer);elements=check.get('elements')
 if check.get('positions') and not elements:elements=[d.ELEMENT[n] for n in check['positions'].values()]
 if elements and len(elements)==2:
  a,b=elements;subjects={'前者':a,'后者':b,'A':a,'B':b,'对象A':a,'对象B':b}
  if check.get('symbols'):subjects.update(zip(check['symbols'],elements))
  if check.get('positions'):subjects.update((star+'宫',d.ELEMENT[n]) for star,n in check['positions'].items())
  if check.get('door_relation'):subjects.update({'门':a,'宫':b})
  token='|'.join(sorted(map(re.escape,subjects),key=len,reverse=True))
  for m in re.finditer('('+token+r')(生|克)('+token+')',answer):
   if any(x in answer[max(0,m.start()-5):m.start()] for x in ['不是','并非','不能','不应']):continue
   s,operation,t=m.groups();cycle=d.GENERATES if operation=='生' else d.CONTROLS
   if cycle[subjects[s]]!=subjects[t]:conflicts.append({'claim':m[0],'reason':'subject_direction_conflicts_with_given_elements'})
  representative={'水':1,'土':2,'木':3,'金':6,'火':9}
  for m in re.finditer('('+token+r')(受生|受克)',answer):
   if any(x in answer[max(0,m.start()-5):m.start()] for x in ['不是','并非','不能','不应']):continue
   subject,state=m.groups();other=b if subjects[subject]==a else a
   if d.relation(representative[subjects[subject]],representative[other])!=state:conflicts.append({'claim':m[0],'reason':'received_relation_has_wrong_subject'})
 if check.get('pair'):
  stem_elements=dict(zip('甲乙丙丁戊己庚辛壬癸','木木火火土土金金水水'))
  for m in re.finditer(r'天盘([甲乙丙丁戊己庚辛壬癸])([生克])地盘([甲乙丙丁戊己庚辛壬癸])',answer):
   a,op,b=m.groups();cycle=d.GENERATES if op=='生' else d.CONTROLS
   if cycle[stem_elements[a]]!=stem_elements[b]:conflicts.append({'claim':m[0],'reason':'explicit_stem_element_cycle_conflict'})
 return conflicts


def name_ok(answer,check):
 if old.name_slot(answer,check):return True
 labels=[check['answer'],*check.get('aliases',[])];head=old.first(answer)
 if check['answer'].endswith('星'):labels.append(check['answer'][:-1])
 if any(w in head for w in ['或者','或是','都可以','不确定']):return False
 if check.get('pair'):
  # A list of alternative names is not a unique answer, even if one is right.
  if re.search(r'(?:命名取|称为|名为|格名为)[^。\n]{0,40}、',answer):return False
  for name in labels:
   for m in positive_occurrences(answer,name):
    prefix=answer[max(0,m.start()-18):m.start()]
    if re.search(r'(?:称为|名为|叫作|称|是|为)(?:本条中(?:的)?|本条格局中(?:的)?|格局|格)?[“「]?$',prefix) or old.normalized(head)==old.normalized(name):return True
  return False
 if check.get('role_label'):
  for name in labels:
   for m in positive_occurrences(head,name):
    if re.search(r'(?:以|取|看|用|是|为|代表)(?:天盘)?$',head[:m.start()]) or '代表' in head[m.end():]:return True
 return False


def relation_ok(answer,check):
 expected=check['answer'];head=elementary.first_clause(answer)
 claims=[]
 for m in re.finditer(r'(?:受生|受克|比和|生|克)',head):
  before=head[:m.start()];after=head[m.end():]
  if re.search(r'(?:不是|并非|不为|不应)$',before):continue
  if (not before.strip() or re.search(r'(?:是|为|改为|关系为|关系是|关系属)$',before)) and (not after or after[0] in '。，,；;：: '):claims.append(m[0])
 if claims:return set(claims)=={expected}
 return previous.relation_ok(answer,expected,elements=check.get('elements'))


def fact_check(row,gold):
 answer=row['answer'];check=gold.get('checks')
 if not answer.strip() or row.get('empty'):return {'status':'fail','reason':'empty'}
 if row.get('reached_token_limit'):return {'status':'fail','reason':'output_limit'}
 if re.search(r'(.{2,24})\1{7,}',re.sub(r'\s+','',answer)):return {'status':'fail','reason':'repeated_output'}
 if unknown_year_misuse(answer,gold):return {'status':'fail','reason':'unknown_birth_year_used_as_known'}
 if not check or check['kind']=='mentions':return {'status':'needs_review','reason':'open_semantics'}
 conflicts=reasoning_conflicts(answer,check)
 if conflicts:return {'status':'fail','reason':'explicit_reasoning_conflict','conflicts':conflicts}
 kind=check['kind'];ok=None
 if kind=='exact_name':
  # Retain V54's explicit layer-stem mismatch rejection before allowing prose.
  prior=old.fact_check(row,gold)
  if prior['status']=='fail':return prior
  ok=name_ok(answer,check)
 elif kind=='positions':ok=previous.positions_ok(answer,check['positions'])
 elif kind=='relation':ok=relation_ok(answer,check)
 elif kind=='joint_relation':ok=previous.positions_ok(answer,check['positions']) and relation_ok(answer,check)
 elif kind=='birth_boundary':
  ok=check['answer'] in elementary.first_clause(answer) and check['opposite'] not in elementary.first_clause(answer)
  ok=ok and any(w in answer for w in ['暂不取年命','先不取年命','不计算年命','不能取年命','暂不计算年命'])
 elif kind=='condition_boundary':
  body=old.normalized(answer);ok=all(any(old.normalized(t) in body for t in alternatives) for alternatives in check['required'])
  ok=ok and not any(list(positive_occurrences(answer,t)) for t in check.get('forbidden',[]))
 else:return old.fact_check(row,gold)
 return {'status':'pass' if ok else 'needs_review','reason':'declared_slots_and_explicit_contradictions_only','semantic_professional_approval':False}


citation_check=old.citation_check


def score(rows,gold):
 gold={r['id']:r for r in gold};seen=set();items=[];groups=defaultdict(list);citation=[];domains=defaultdict(list)
 for row in rows:
  if row['id'] not in gold or row['id'] in seen:raise ValueError('Unknown or duplicate output')
  g=gold[row['id']];seen.add(row['id'])
  if row['question']!=g['messages'][:-1] or row['expected']!=g['answer']:raise ValueError('Evaluation question or target changed')
  fact=fact_check(row,g);cite=citation_check(row,g);automatic=bool(g.get('checks') and g['checks']['kind']!='mentions');skill=g.get('skill',g['task'])
  if automatic:groups[skill].append(fact['status']=='pass');domains[g['domain']].append(fact['status']=='pass')
  if cite['status']!='not_required':citation.append(cite['ok'])
  items.append({'id':row['id'],'skill':skill,'domain':g['domain'],'fact':fact,'citation':cite,'automatic_fact_slots':automatic})
 def aggregate(values):return {k:{'pass':sum(v),'total':len(v),'rate':sum(v)/len(v)} for k,v in values.items()}
 return {'complete':set(gold)==seen,'outputs':len(rows),'groups':aggregate(groups),'per_domain_fact_slots':aggregate(domains),'citation':{'pass':sum(citation),'total':len(citation),'rate':sum(citation)/len(citation) if citation else None},'items':items,'semantic_professional_approval':False,'real_prediction_accuracy':None}


def gate(summary):
 required={'positions':.9,'given_direction':.85,'direction':.9,'door_relation':.8,'identity':.85,'condition':.85,'typed_pair':.8,'property':.9,'source_fact':.8,'chart_direction':.8};missing=[k for k in required if not summary['groups'].get(k,{}).get('total')]
 return {'passed':summary['complete'] and not missing and all(summary['groups'][k]['rate']>=v for k,v in required.items()) and (summary['citation']['rate'] or 0)>=.9,'thresholds':required,'missing_categories':missing,'professional_approval':False}


rank=old.rank
