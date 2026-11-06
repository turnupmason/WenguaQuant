"""Fact slots, reasoning contradictions and citations are separate checks.

This is a limited reproducible rubric. It never automatically approves open
case semantics or real-world prediction, and never changes saved V53 scores.
"""
import re
from collections import defaultdict
from wengua_course import checks as legacy,data as d
from wengua_course_v53 import checks as prior

def first(text):return legacy.first_clause(text)
def normalized(text):return legacy.clean(text)

def positive_label(text,label):
 text=normalized(first(text));label=normalized(label)
 for match in re.finditer(re.escape(label),text):
  before=text[max(0,match.start()-6):match.start()]
  if re.search(r'(?:不是|并非|不为|不取|不用|不能(?:取|用|看|称为)|不(?:称为|代表)|不应(?:取|用|称为))$',before):continue
  return True
 return False

def name_slot(answer,check):
 text=first(answer);labels=[check['answer'],*check.get('aliases',[])]
 if check['answer'].startswith('天') and check['answer'].endswith('星'):labels.append(check['answer'][:-1])
 text=re.split(r'[，,](?=因为|这|对应|要|可|并|兼|再)',text,maxsplit=1)[0];clean=normalized(text)
 if check.get('role_label'):
  # A list of alternatives is not an unambiguous role assignment.
  if any(w in text for w in ('或者','或是','都可以','不确定')):return False
  role=normalized(check['role_label'])
  for label in labels:
   value=normalized(label);prefix=r'(?:(?:本章|本书|本条)(?:一般)?)?(?:应|可|主要|先|改)?(?:取|用|以|看)?(?:天盘)?'
   if re.fullmatch(prefix+re.escape(value)+r'(?:落宫)?(?:(?:为|代表|是)'+re.escape(role)+')?',clean):return True
   if re.fullmatch(re.escape(role)+r'(?:以|用|取|是|为)'+re.escape(value)+r'(?:为代表符号)?',clean):return True
  return False
 for label in labels:
  value=normalized(label)
  if clean in {value,'本条称为'+value,'本书称为'+value,'本章称为'+value,'名称为'+value,'格名为'+value}:return True
  if re.search(r'(?:称为|名为|格名为)'+re.escape(value)+r'(?:[，,。；;]|$)',text) and positive_label(text,label):return True
 return False

def reasoning_conflicts(answer):
 conflicts=[]
 for match in re.finditer(r'([金木水火土])([生克])([金木水火土])',answer):
  if any(x in answer[max(0,match.start()-5):match.start()] for x in ('不是','并非','不能','不应')):continue
  a,operation,b=match.groups();mapping=d.GENERATES if operation=='生' else d.CONTROLS
  if mapping[a]!=b:conflicts.append({'claim':match[0],'reason':'contradicts_literal_element_cycle'})
 if re.search(r'(?:上层|下层|天盘|地盘)(?:应|要|先)?取(?:年干|月干|日干|时干)',answer):conflicts.append({'reason':'time_stem_substituted_for_heaven_earth_layer'})
 return conflicts

def fact_check(row,gold):
 answer=row['answer'];check=gold.get('checks')
 if not answer.strip() or row.get('empty'):return {'status':'fail','reason':'empty'}
 if row.get('reached_token_limit'):return {'status':'fail','reason':'output_limit'}
 if re.search(r'(.{2,24})\1{7,}',re.sub(r'\s+','',answer)):return {'status':'fail','reason':'repeated_output'}
 if not check or check['kind']=='mentions':return {'status':'needs_review','reason':'open_semantics'}
 conflicts=reasoning_conflicts(answer)
 if conflicts:return {'status':'fail','reason':'explicit_reasoning_conflict','conflicts':conflicts}
 kind=check['kind']
 if check.get('pair'):
  typed=answer
  for full,stem in [('甲子戊','戊'),('甲戌己','己'),('甲申庚','庚'),('甲午辛','辛'),('甲辰壬','壬'),('甲寅癸','癸')]:typed=typed.replace(full,stem)
  for aliases,expected in [(r'(?:天盘|上层)',check['pair'][0]),(r'(?:地盘|下层)',check['pair'][1])]:
   values=re.findall(aliases+r'(?:干)?(?:为|是)?([甲乙丙丁戊己庚辛壬癸])',typed)
   if values and set(values)!={expected}:return {'status':'fail','reason':'explicit_layer_stem_mismatch'}
 if kind=='exact_name':ok=name_slot(answer,check)
 elif kind=='property':
  text=normalized(first(answer));symbol=normalized(check['symbol']);element=check['element']
  ok=bool(re.search(re.escape(symbol)+r'(?:的五行)?(?:属性)?(?:是|为|属|属于|归于)'+element,text))
 elif kind=='state':
  aliases={'迫':['迫','门迫','门克宫'],'制':['制','宫克门'],'和':['和','门生宫'],'义':['义','宫生门'],'旺':['旺','最旺'],'相':['相','次旺'],'比和':['比和','五行相同']}
  text=normalized(re.split(r'[，,]',first(answer),maxsplit=1)[0]);ok=False
  if not any(x in text for x in ('不是','并非','不属','不为','不能','不应')):
   ok=any(text==x or re.fullmatch(r'.{1,30}(?:为|是|称为)'+re.escape(x)+r'(?:关系|状态)?',text) for x in aliases.get(check['answer'],[check['answer']]))
 elif kind=='relation':ok=prior.relation_ok(answer,check['answer'],elements=check.get('elements',gold.get('provenance',{}).get('elements')))
 elif kind=='joint_relation':ok=prior.positions_ok(answer,check['positions']) and prior.relation_ok(answer,check['answer'],elements=[d.ELEMENT[p] for p in check['positions'].values()])
 elif kind=='insufficient':
  ok=any(x in answer for x in ('不足','不能','无法','缺少','还需','需要补充')) and any(x in answer for x in ('证据','原文','组合','格名','天地盘','位置','条件'))
 else:ok=legacy.evaluate(answer,check)['status']=='pass'
 return {'status':'pass' if ok else 'needs_review','reason':'declared_fact_slots_only','semantic_professional_approval':False}

def citation_check(row,gold):
 if gold.get('context')!='provided_evidence':return {'status':'not_required','ok':True}
 available=set(re.findall(r'\[(S\d+)\]',gold['messages'][-2]['content']));used=set(re.findall(r'\[(S\d+)\]',row['answer']))
 # Printed case-study extracts have source coordinates, but no S-labels.
 # Only the actual RAG/source-packet interface teaches these citation labels.
 if not available and not gold.get('provenance',{}).get('actual_production_rag_prompt'):return {'status':'not_required','ok':True}
 if not used:return {'status':'missing','ok':False}
 if not used.issubset(available):return {'status':'unknown','ok':False}
 required=set(gold.get('provenance',{}).get('required_citations',[]))
 if required and not required.issubset(used):return {'status':'required_source_not_cited','ok':False}
 return {'status':'membership_and_required_sources_checked','ok':True,'entailment_verified':False}

def score(rows,gold):
 gold={x['id']:x for x in gold};seen=set();items=[];groups=defaultdict(list);citation=[]
 for row in rows:
  if row['id'] not in gold or row['id'] in seen:raise ValueError('Unknown/duplicate output')
  g=gold[row['id']];seen.add(row['id'])
  if row['question']!=g['messages'][:-1] or row['expected']!=g['answer']:raise ValueError('Changed evaluation prompt/target')
  fact=fact_check(row,g);cite=citation_check(row,g);automatic=bool(g.get('checks') and g['checks']['kind']!='mentions');skill=g.get('skill',g['task'])
  if automatic:groups[skill].append(fact['status']=='pass')
  if cite['status']!='not_required':citation.append(cite['ok'])
  items.append({'id':row['id'],'skill':skill,'fact':fact,'citation':cite,'automatic_fact_slots':automatic})
 return {'complete':set(gold)==seen,'outputs':len(rows),'groups':{k:{'pass':sum(v),'total':len(v),'rate':sum(v)/len(v)} for k,v in groups.items()},'citation':{'pass':sum(citation),'total':len(citation),'rate':sum(citation)/len(citation) if citation else None},'items':items,'semantic_professional_approval':False,'real_prediction_accuracy':None}

def gate(summary):
 required={'direction':.90,'door_relation':.80,'identity':.85,'source_fact':.80,'typed_pair':.80,'property':.90,'chart_direction':.80};missing=[k for k in required if not summary['groups'].get(k,{}).get('total')]
 ok=summary['complete'] and not missing and all(summary['groups'][k]['rate']>=v for k,v in required.items()) and (summary['citation']['rate'] or 0)>=.9
 return {'passed':ok,'fact_thresholds':required,'citation_threshold':.9,'missing_categories':missing,'closed_book_legacy_is_separate_diagnostic':True,'professional_approval':False}

def rank(summary):
 values=[x['rate'] for x in summary['groups'].values()]
 return (min(values,default=0),sum(values)/len(values) if values else 0,summary['citation']['rate'] or 0)
