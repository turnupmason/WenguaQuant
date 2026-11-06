"""Source-slot checks and stage gates, not Qimen scores or expert approval."""
import re
from collections import defaultdict
from wengua_course import checks as old,data as d

def head(answer):return old.clean(old.first_clause(answer))
def name_ok(answer,check):
 text=head(answer);accepted=[check['answer'],*check.get('aliases',[])];forms=set()
 for raw in accepted:
  label=old.clean(raw)
  forms.add(label)
  if label.startswith('天') and label.endswith('星'):forms.add(label[:-1])
  for prefix in ['本条称为','本书称为','本条格名为','名称为','格名为','该格为','这一组合称为','本章取','本章用','取','用','以','看','为','是','称为']:
   forms.add(prefix+label)
 if check.get('role_label'):
  role=old.clean(check['role_label']);sentences=set()
  for raw in accepted:
   label=old.clean(raw)
   sentences.update({prefix+label+'代表'+role for prefix in ('本章以','以','本章用','用','')})
   sentences.update(role+verb+label for verb in ('取','用','是','为','以'))
  if text in sentences:return True
 for form in forms:
  if text==form:return True
  if re.fullmatch(re.escape(form)+r'(?:为|代表)(?:失主|考生|本人|原告|被告|医生|医药|男方|女方|媒人|经纪人|资本|利息|试卷|文章|同事|上级领导|下级或单位职工)',text):return True
 return False

def positions_ok(answer,positions):
 for star,palace in positions.items():
  found=[]
  for n,name in d.PALACE.items():
   gua,number=name[0],name[1:];places=rf'(?:{gua}(?:{number}|{n})?宫|(?<![乾坎艮震巽离坤兑中])(?:{number}|{n})宫)'
   pattern=rf'(?:天)?{re.escape(star)}(?:星)?(?:落(?:在|于|入)?|位于|在)?\s*{places}'
   for match in re.finditer(pattern,answer):
    found.append(n)
    tail=answer[match.end():match.end()+12];element=re.match(r'[，,、\s]*(?:五行(?:为|属)?|属|为)([金木水火土])',tail)
    if element and element[1]!=d.ELEMENT[n]:return False
  if set(found)!={palace}:return False
 return True

def relation_ok(answer,expected,*,elements=None):
 """Explicit direction with a short grammatical wrapper, never keyword-only."""
 text=head(answer)
 if any(w in text for w in ('不是','并非','不应','不为','不能认定')):return False
 explicit=set(re.findall(r'受生|受克|比和',text))
 if explicit and explicit!={expected}:return False
 if old.evaluate(answer,{'kind':'relation','answer':expected})['status']=='pass':return True
 match=re.fullmatch(r'.{1,40}?(?:关系为|关系是|关系属|为|是)(受生|受克|比和|生|克)(?:关系)?',text)
 if match and match[1]==expected:return True
 if elements:
  a,b=elements
  literal=a+({'生':'生','克':'克','受生':'被','受克':'被'}.get(expected,'相同'))+b
  # Directed element statements are only accepted for the two stated elements.
  if expected in ('生','克') and text==literal:return True
  if expected=='受生' and text==b+'生'+a:return True
  if expected=='受克' and text==b+'克'+a:return True
 return False

def evaluate(row,gold):
 answer=row['answer'];check=gold.get('checks')
 if row.get('empty') or not answer.strip():return {'status':'fail','reason':'empty'}
 if row.get('reached_token_limit'):return {'status':'fail','reason':'output_limit'}
 if re.search(r'(.{2,24})\1{7,}',re.sub(r'\s+','',answer)):return {'status':'fail','reason':'repetitive_output'}
 if gold['context']=='provided_evidence':
  available=set(re.findall(r'\[(S\d+)\]',gold['messages'][-2]['content']));cited=set(re.findall(r'\[(S\d+)\]',answer))
  if not cited or not cited.issubset(available) or bool(gold.get('provenance',{}).get('required_citations')) and set(gold['provenance']['required_citations'])!=cited:return {'status':'fail','reason':'missing_or_unknown_citation'}
 if check is None or check['kind']=='mentions':return {'status':'needs_review','reason':'open_semantics_not_auto_approved'}
 if check['kind']=='insufficient':
  ok=any(x in answer for x in ('不足','没有','不能确认','无法确认','未提供')) and any(x in answer for x in ('组合','格名','原文','证据'))
 elif check['kind'] in ('positions','joint_relation'):
  positions=check['positions'];ok=positions_ok(answer,positions)
  if check['kind']=='joint_relation':ok=ok and relation_ok(answer,check['answer'],elements=[d.ELEMENT[p] for p in positions.values()])
 elif check['kind']=='exact_name':ok=name_ok(answer,check)
 elif check['kind']=='relation':ok=relation_ok(answer,check['answer'],elements=gold.get('provenance',{}).get('elements'))
 else:ok=old.evaluate(answer,check)['status']=='pass'
 return {'status':'pass' if ok else 'needs_review','reason':'declared_source_slots_only','professional_approval':False}

def category(g):
 if g['task']=='element_direction':return 'element_direction'
 if g['task']=='chart_direction':return 'chart_direction'
 if g['task']=='chart_positions':return 'chart_positions'
 return 'provided_evidence' if g['context']=='provided_evidence' else 'closed_book'

def score(rows,gold):
 gold={r['id']:r for r in gold};seen=set();groups=defaultdict(list);items=[]
 for row in rows:
  if row['id'] in seen or row['id'] not in gold:raise ValueError('Unknown or duplicate output')
  seen.add(row['id']);g=gold[row['id']]
  if row['question']!=g['messages'][:-1] or row['expected']!=g['answer']:raise ValueError('Evaluation changed source question/answer')
  result=evaluate(row,g);auto=g.get('checks') is not None and g['checks']['kind']!='mentions'
  if auto:groups[category(g)].append(result['status']=='pass')
  items.append({'id':g['id'],'category':category(g),'task':g['task'],'result':result,'auto_slots_only':auto})
 return {'complete':seen==set(gold),'outputs':len(rows),'groups':{k:{'pass':sum(v),'total':len(v),'rate':sum(v)/len(v)} for k,v in groups.items()},'items':items,'professional_approval':False,'real_prediction_accuracy':None}

def gate(summary,stage):
 required={'provided_evidence':.80,'closed_book':.50,'element_direction':.80} if stage=='foundation' else {'provided_evidence':.80,'chart_positions':.85,'chart_direction':.75}
 missing=[k for k in required if not summary['groups'].get(k,{}).get('total')]
 passed=summary['complete'] and not missing and all(summary['groups'][k]['rate']>=v for k,v in required.items())
 return {'passed':passed,'thresholds':required,'missing_categories':missing,'purpose':'training_progress_screen_not_professional_approval'}

def rank(summary):
 rates=[v['rate'] for v in summary['groups'].values()]
 return (min(rates,default=0),sum(rates)/len(rates) if rates else 0)

def retention(summary,before,*,maximum_drop=.05):
 """Compare the SAME declared tasks, not just a low absolute pass floor."""
 rows={};same_tasks=bool(before) and summary['complete']
 for key,prior in before.items():
  after=summary['groups'].get(key,{})
  same=bool(prior.get('total')) and after.get('total')==prior['total'];same_tasks&=same
  rows[key]={'same_declared_task_count':same,'before':prior['rate'],'after':after.get('rate'),'drop':prior['rate']-after['rate'] if same else None}
 passed=same_tasks and all(row['drop']<=maximum_drop+1e-12 for row in rows.values())
 return {'passed':passed,'maximum_absolute_group_rate_drop':maximum_drop,'groups':rows,'professional_approval':False}
