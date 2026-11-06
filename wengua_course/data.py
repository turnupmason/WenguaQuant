"""Source-bound V5.2 lessons. No manual Qimen scores or generated outcomes.

Original V5.1 test questions are fingerprinted, never turned into lessons.
Original materials remain intact; each old non-test task receives a disposition.
"""
from collections import Counter,defaultdict
from copy import deepcopy
import hashlib,json,re
from pathlib import Path

SCHEMA='wengua_two_book_conditional_course_v52'
BOOKS={'shenqi','kaiwu'}
DOMAIN_ZH={'foundation':'基础知识','formation':'格局规则','mixed_domain':'跨领域取用','health':'疾病','education':'考试求学','employment':'工作就业','business_profit':'经营求财','marriage':'恋爱婚姻','pregnancy':'怀孕分娩','lost_property':'钱物丢失','missing_person':'行人走失','litigation':'官司诉讼','criminal_case':'刑事案件','travel_abroad':'出行出国','sports_competition':'体育竞赛','life_opportunity_book_study':'人生机遇','weather_historical_source':'天时天气','weather':'天气','geography_environment_historical_source':'地理环境','military_history_source':'军事史例','politics_elections':'政治选举','competition_evaluation':'竞争评比','miscellaneous_historical_source':'其他史例','world_affairs':'时事史例','industrial_safety':'生产安全'}
SYSTEM='直接回答问题。先说结果，再说必要条件和具体证据；不重复结论。不补造宫位、身份、年份、应期或反馈。按指定书和领域取用，不混用异说，不使用人工分数。'
ELEMENT={1:'水',2:'土',3:'木',4:'木',5:'土',6:'金',7:'金',8:'土',9:'火'}
PALACE={1:'坎一',2:'坤二',3:'震三',4:'巽四',5:'中五',6:'乾六',7:'兑七',8:'艮八',9:'离九'}
GENERATES={'木':'火','火':'土','土':'金','金':'水','水':'木'}
CONTROLS={'木':'土','土':'水','水':'火','火':'金','金':'木'}
COMPARISONS={'kaiwu.health.19950219.brother_paralysis':{'A':'kaiwu.health.19950219.brother_paralysis','B':'shenqi.health.19960727.father_illness'}}
RETIRE_TASKS={'supplementary_attributed_case_source','complete_source_rule_reading','unparsed_case_table_literal_cell','literal_source_table_cell','all_fields_literal_chart_cell','author_step_attributed_review','author_analysis_review','chronology_review','case_error_correction','missing_feedback_boundary','identity_switch_multiturn','ordered_pair_condition_followup','ordered_name_error_correction','context_required_fragment','real_chart_directed_relation'}
BAD=('HEALTH_V9_V11','STRENGTH_200','candidate_five_grade_quantization','以下是来源资料，尚非现实预测验证')

def dumps(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def digest(x):return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def compact(x):return re.sub(r'\s+','',x)
def title_only(x):return bool(re.fullmatch(r'[\W\d_]*(九宫盘面|九宫格局|奇门格局|九宫格|分析|实例[一二三四五六七八九十\d]*|[一二三四五六七八九十\d]+)[：:。\s]*',x.strip()))
def relation(a,b):
 x,y=ELEMENT[a],ELEMENT[b]
 return '比和' if x==y else '生' if GENERATES[x]==y else '克' if CONTROLS[x]==y else '受生' if GENERATES[y]==x else '受克'
def evidence(ids,units):return '\n'.join(units[s]['original_quote'] for s in ids)
def source_scope(ids,units):
 books=sorted({units[s]['book_id'] for s in ids});return '、'.join('《神奇之门》' if b=='shenqi' else '《开悟之门》' for b in books)
def row(question,answer,ids,units,*,task,format,split='train',group=None,domain='foundation',case_ids=(),messages=None,checks=None,provenance=None):
 if not ids or any(s not in units or units[s]['book_id'] not in BOOKS for s in ids):raise ValueError('Missing or non-two-book source')
 content=f'{source_scope(ids,units)}；{DOMAIN_ZH.get(domain,domain)}。\n'+question
 for code,label in DOMAIN_ZH.items():content=content.replace(code,label)
 if messages:
  messages=deepcopy(messages);messages[1]['content']=source_scope(ids,units)+'。\n'+messages[1]['content']
 else:messages=[{'role':'system','content':SYSTEM},{'role':'user','content':content},{'role':'assistant','content':answer}]
 key=digest(dumps([task,split,messages,ids]))[:24]
 return {'id':'v52:'+key,'question':messages[1]['content'],'answer':answer,'messages':messages,'source_ids':sorted(set(ids)),'case_id':next(iter(case_ids),None),'case_ids':list(case_ids),'group':group or 'knowledge:'+digest(dumps(sorted(ids)))[:16],'split':split,'format':format,'task':task,'domain':domain,'source_mode':'book_supported_conditional_teaching' if not case_ids else 'historical_retrospective_not_blind_prediction','checks':checks,'provenance':provenance or {'kind':'reviewed_source_bound_instruction'}}

def case_ownership(pack):
 owners=defaultdict(set)
 for r in pack['records']:
  for c in r['case_ids']:owners[c].add((r['group'],r['split']))
 if any(len(v)!=1 for v in owners.values()):raise ValueError('Original case split/family conflicts')
 return {cid:next(iter(v)) for cid,v in owners.items()}

def development_groups(pack,owner,exclude_groups=()):
 """New-phase family holdout; ancestor/source exposure is explicitly acknowledged.
 Known failed examples become repair teaching; new development families are
 withdrawn completely from this phase, including any related examples.
 """
 domains={d['case_id']:d['domain'] for d in pack['dossiers']};dev=set();excluded=set(exclude_groups)
 for domain in sorted(set(domains.values())):
  available={g for cid,(g,split) in owner.items() if split!='test' and domains[cid]==domain and g not in excluded}
  total={g for cid,(g,split) in owner.items() if split!='test' and domains[cid]==domain}
  if len(total)<2 or not available:continue
  count=min(2 if len(total)>=5 else 1,len(available),len(total)-1)
  dev.update(sorted(available,key=lambda g:digest('v52-case-development:'+domain+g))[:count])
 return dev

def chart_text(d):
 rows=d['primary_printed_chart']
 return '\n'.join(f"{PALACE[int(c['palace'])]}宫：神{c['deity_raw']}；门行{c['door_position_raw']}；星行{c['star_position_raw']}" for c in rows) if rows else ''

def step_board(d,step,units):
 """Choose only a preceding transcription in the same source chapter.
 Cross-chart arguments retain author facts without attaching a false main board.
 """
 if d['case_id'] in COMPARISONS:return [],[],{'status':'paired_case_comparison_no_single_board'}
 source=units[step['source_id']];candidates=[]
 if source.get('section_id') is None or not source.get('lines'):return [],[],{'status':'missing_source_chapter_scope_author_evidence_only'}
 for key,cells in [('primary',d['primary_printed_chart']),*d['additional_printed_charts'].items()]:
  ids=[f"chart:{d['case_id']}:{key}:p{c['palace']}" for c in cells]
  if len(cells)!=9 or any(sid not in units for sid in ids):continue
  if any(units[sid].get('book_id')!=source.get('book_id') or units[sid].get('section_id')!=source.get('section_id') for sid in ids):continue
  if any(json.loads(units[sid]['original_quote'])!=cell for sid,cell in zip(ids,cells)):raise ValueError('Board source cell differs')
  end=max(c['source_line'] for c in cells)
  if end<source['lines'][0]:candidates.append((end,key,cells,ids))
 if not candidates:return [],[],{'status':'no_unambiguous_preceding_source_board'}
 latest=max(candidates,key=lambda x:x[0]);end,key,cells,ids=latest
 if sum(c[0]==end for c in candidates)>1:return [],[],{'status':'tied_source_board_scopes_author_evidence_only'}
 if d['additional_printed_charts'] and any(t in step['original'] for t in ('上次','两局','前局','前一局','原局','第一局','第二局','两张','前后两')):
  return [],[],{'status':'explicit_cross_chart_reasoning_author_evidence_only','source_cast_keys':[x[1] for x in candidates]}
 # If an explicit star position disagrees, preserve both sources but do not mix
 # the narrative's facts with a conflicting physical board as one gold input.
 violations=[]
 for match in re.finditer(r'天([蓬任冲辅英芮柱心])星[^。；]{0,12}?(?:落|在)(?:入|于)?[乾坎艮震巽离坤兑]?([1-9])宫',step['original']):
  if '地盘' in step['original'][max(0,match.start()-8):match.start()]:continue
  star,pal=match[1],int(match[2]);actual=[int(c['palace']) for c in cells if star in c['star_position_raw']]
  if actual and pal not in actual:violations.append({'star':star,'author_palace':pal,'transcribed_palaces':actual})
 if violations:return [],[],{'status':'source_position_difference_author_evidence_only','candidate_cast_key':key,'differences':violations}
 return cells,ids,{'status':'source_chronology_bound','cast_key':key,'chart_last_source_line':end,'analysis_source_line':source['lines'][0],'book_id':source['book_id'],'section_id':source['section_id']}

def source_bound_cases(pack):
 """Each author step remains in its own event. Never copy a multi-case review block.
 Case dossiers retain original metadata; normalized split comes from actual tasks.
 """
 units=pack['units'];owner=case_ownership(pack);out=[]
 for d in pack['dossiers']:
  c=deepcopy(d);c['original_dossier_metadata']={'group':c['group'],'split':c['split']};c['group'],c['split']=owner[c['case_id']]
  c['scope']='author_retrospective_only';c['cross_case_full_review_blocks_used']=False
  for s in c['author_steps']:
   if compact(s['original']) not in compact(units[s['source_id']]['original_quote']):raise ValueError('Author step not literal at declared source')
  out.append(c)
 return out

def validate(pack):
 units=pack['units'];ids=set();prompts={};case_prompts=defaultdict(set);split_groups=defaultdict(set);case_splits=defaultdict(set)
 for r in pack['records']:
  if r['id'] in ids:raise ValueError('Duplicate lesson id')
  ids.add(r['id'])
  if r['split'] not in ('train','validation','diagnostic'):raise ValueError('Final test must stay out of active lessons')
  if r['split']!='diagnostic':split_groups[r['group']].add(r['split'])
  if not r['source_ids'] or any(s not in units or units[s]['book_id'] not in BOOKS for s in r['source_ids']):raise ValueError('Wrong teacher/source')
  if not r['answer'].strip() or title_only(r['answer']) or any(x in r['answer'] for x in BAD):raise ValueError('Empty/title/legacy target')
  if r['messages'][-1]!={'role':'assistant','content':r['answer']}:raise ValueError('Wrong assistant target')
  prompt=dumps(r['messages'][:-1])
  if prompt in prompts and prompts[prompt]!=compact(r['answer']):raise ValueError('Conflicting targets')
  prompts[prompt]=compact(r['answer'])
  if r['case_ids'] and r['split']!='diagnostic':case_prompts[prompt].add(r['split'])
  for cid in r['case_ids']:
   case_splits[cid].add(r['split'])
   if pack['original_case_splits'][cid]=='test':raise ValueError('Sealed test case used for active teaching')
 if any(len(s)>1 for g,s in split_groups.items() if g.startswith('family:')):raise ValueError('Case family crosses current-phase split')
 if any('train' in s and 'validation' in s for s in case_splits.values()):raise ValueError('Case crosses current-phase split')
 if any(len(s)>1 for s in case_prompts.values()):raise ValueError('Identical case question crosses train/development')
 if set(pack['source_books'])!=BOOKS:raise ValueError('Exactly the two books are required')
 return {'records':len(ids),'conflicting_prompts':0,'title_targets':0,'case_train_validation_leakage':0,'original_test_used':False,'third_book_used':False}

def build(old,manual,*,revealed_groups=()):
 units=old['units'];owner=case_ownership(old);dev=development_groups(old,owner,revealed_groups);records=[];dispositions=[]
 # Final test prompts/answers never enter the new teaching pipeline.
 sealed=[{'id':r['id'],'group':r['group'],'sha256':digest(dumps(r))} for r in old['records'] if r['split']=='test']
 for r in old['records']:
  if r['split']=='test':continue
  reason=None
  if title_only(r['answer']):reason='structure_only_not_content_target'
  elif r['task'] in RETIRE_TASKS:reason='replaced_with_explicit_scoped_or_conditional_lesson'
  elif r['format']=='complete_case':reason='replaced_case_boundary_and_author_stage'
  elif len(r['answer'])>700:reason='long_source_retained_for_index_not_repeated_SFT'
  elif r['task']=='printed_star_location':
   if int(digest(r['id'])[:8],16)%5:reason='literal_reading_downsampled_no_material_deleted'
  if reason:
   dispositions.append({'old_id':r['id'],'source_ids':r['source_ids'],'action':'source_preserved_not_replayed_as_old_QA','reason':reason});continue
  x=deepcopy(r);x['id']='v52:replay:'+r['id'].removeprefix('v51:');x['messages'][0]['content']=SYSTEM
  x.update(domain='foundation',checks=None,provenance={'kind':'unchanged_reviewed_replay','parent_id':r['id']})
  if x['task'] in {'ordered_stem_conditions','ordered_door_rule','formation_complete_conditions','three_zha_conditions','five_jia_common_conditions','five_jia_complete_conditions','outcome_conditioned_timing'}:x['format']='conditional_dialogue'
  if x['case_ids']:
   domains={d['case_id']:d['domain'] for d in old['dossiers']};x['domain']=domains[x['case_ids'][0]]
   x['split']='validation' if x['group'] in dev else 'train'
  elif x['split']=='validation':
   x['split']='diagnostic'
  records.append(x);dispositions.append({'old_id':r['id'],'new_id':x['id'],'source_ids':r['source_ids'],'action':'replay','split':x['split']})
 # Reviewer-authored conditional contrasts with literal support fragments.
 for spec in manual:
  if any(compact(s['quote']) not in compact(units[s['source_id']]['original_quote']) for s in spec['support']):raise ValueError('Manual lesson support mismatch: '+spec['key'])
  for split,question in [('train',spec['question']),('validation',spec.get('development_question'))]:
   if not question:continue
   ids=[s['source_id'] for s in spec['support']]
   messages=None
   if split=='train' and spec.get('messages'):
    messages=[{'role':'system','content':SYSTEM},*spec['messages']]
   cids=spec.get('case_ids',[]);split=('validation' if owner[cids[0]][0] in dev else ('diagnostic' if split=='validation' else 'train')) if cids else split
   records.append(row(question,spec['answer'],ids,units,group=owner[cids[0]][0] if cids else 'controlled:'+spec['key']+':'+split,case_ids=cids,task='condition_contrast' if spec['format']=='conditional_dialogue' else 'scoped_rule',format=spec['format'],split=split,domain=spec['domain'],messages=messages,checks=spec.get('checks'),provenance={'kind':'assistant_source_reviewed_not_independent_expert','key':spec['key'],'support':spec['support'],'conditions':spec.get('conditions',[]),'book_method_scoped':True}))
 # Scoped ordered names and swaps, using the book's actual names rather than a universal table.
 names={}
 for r in old['records']:
  if r['split']=='test' or r['task']!='ordered_pair_name':continue
  m=re.search(r'天盘(.)加地盘(.)',r['question'])
  if not m:raise ValueError('Unknown ordered-pair syntax')
  names[m.groups()]=(r['answer'].rstrip('。').split('、')[0],r['source_ids'])
 for (a,b),(name,sids) in sorted(names.items()):
  if not any(name in units[s]['original_quote'] for s in sids):raise ValueError('Name not supported literally')
  q=f'天盘{a}、地盘{b}，按本书名称是什么？'
  records.append(row(q,name+'。',sids,units,task='ordered_name',format='foundation_qa',checks={'kind':'exact_name','answer':name},provenance={'kind':'literal_name_slots'}))
  swapped=names.get((b,a));answer=f'不变。两层都是{a}，交换后仍为{a}加{b}，本书称{name}。' if a==b else f'组合改变。原为{a}加{b}的{name}，交换后为{b}加{a}'+('，本书称'+swapped[0]+('，两组合虽同名，层次和条件仍要分别看。' if swapped[0]==name else '。') if swapped else '，不能直接沿用原组合名称。')
  records.append(row(f'天盘{a}、地盘{b}；交换上下两层，组合和名称是否改变？',answer,sids+(swapped[1] if swapped else []),units,task='same_vs_different_swap',format='conditional_dialogue',provenance={'kind':'ordered_symbol_identity_no_outcome_inference'}))
  # Wrong-name example uses an actually different book name, not a fabricated rule.
  wrong=next(n for (x,y),(n,_) in sorted(names.items()) if not any(n in units[s]['original_quote'] for s in sids))
  records.append(row(f'有人把天盘{a}加地盘{b}称为“{wrong}”。按本书校正名称。',f'应为{name}。{wrong}不是本书这一干对的名称。',sids,units,task='ordered_name_correction',format='conditional_dialogue',provenance={'kind':'source_name_correction'}))
 # Same real printed board, changed question direction; never mutate a historical chart.
 relation_sources=['book:22:L61-64:a7bc478e0fe9','book:22:L155-155:b7acca36ca77']
 case_catalog=source_bound_cases(old)
 for case in case_catalog:case['phase_split']='sealed_test' if case['split']=='test' else ('validation' if case['group'] in dev else 'train')
 prior_case={r['case_id']:r for r in old['records'] if r['task']=='author_analysis_review' and r['split']!='test'}
 step_dispositions=[]
 for d in case_catalog:
  cid=d['case_id'];group,oldsplit=owner[cid];split='validation' if group in dev else 'train'
  if oldsplit=='test':
   step_dispositions.extend({'case_id':cid,'step':s['step'],'source_id':s['source_id'],'action':'sealed_original_test_source_not_active_lesson'} for s in d['author_steps']);continue
  board=chart_text(d);steps=d['author_steps'];sourceids=[s['source_id'] for s in steps];comparison=COMPARISONS.get(cid);lesson_cases=list(comparison.values()) if comparison else [cid]
  if comparison and any(owner[c][0]!=group for c in lesson_cases):raise ValueError('Paired source cases must stay in one conservative family')
  # Do not train an identical paragraph as both input and output. Where the
  # author supplies an explicit causal bridge, teach evidence -> that conclusion.
  # Remaining steps are source/index only, with an explicit disposition.
  for st in steps:
   text=st['original'];matches=list(re.finditer(r'故我断|所以断|故断|故可断|因此断|所以说|由此可见|据此|表明|说明|故',text));chosen=None
   for match in reversed(matches):
    tail=text[match.end():];sentence=re.search(r'[。！？]',tail);end=match.end()+(sentence.end() if sentence else len(tail));target=text[match.end():end].strip('，,：:；; \n')
    negated=any(text[max(0,match.start()-3):match.start()].endswith(n) for n in ('不','未','不能','并非','并不'))
    if not negated and target not in text[:match.start()] and match.start()>=24 and 4<=len(target)<=180:
     chosen=(match,target,end);break
   if chosen is None:
    step_dispositions.append({'case_id':cid,'step':st['step'],'source_id':st['source_id'],'action':'source_preserved_not_identical_input_output_SFT'});continue
   match,conclusion,target_end=chosen;prefix=text[:match.start()].strip()
   bound_cells,bound_ids,binding=step_board(d,st,units);bound_board=chart_text(d|{'primary_printed_chart':bound_cells})
   evidence_start=max(text.rfind('。',0,match.start()),text.rfind('；',0,match.start()))+1
   causal_evidence=text[evidence_start:match.start()].strip('，,：:；; \n')
   target=conclusion+('\n依据：'+causal_evidence+'。' if 6<=len(causal_evidence)<=260 else '')
   scope=None
   if comparison:
    tags=list(re.finditer(r'([AB])卦',text[:match.start()]));scope=tags[-1][1] if tags else None
    if scope:target=scope+'卦结论：'+target
   q=('原作者比较A、B两个不同病例。必须保留A、B标签，不把B的结论归给A。根据下列证据，本步对哪一例作了什么判断？\n' if comparison else '历史案例分析练习。根据作者已经核对的下列证据，本步作出了什么具体判断？只答本步，不补后来反馈。\n')+('原书抄盘（盘次'+binding['cast_key']+'）：\n'+bound_board+'\n' if bound_board else '')+'作者已核对的证据：\n'+prefix
   x=row(q,target,[st['source_id'],*bound_ids],units,task='author_evidence_to_conclusion',format='complete_case',split=split,group=group,domain=d['domain'],case_ids=lesson_cases,provenance={'board_binding':binding,'comparison_cases':comparison,'conclusion_scope':scope,'kind':'literal_causal_bridge_not_blind_prediction','author_step':st['step'],'source_case_boundary':cid,'evidence_chars':[0,match.start()],'target_chars':[match.end(),target_end],'short_evidence_chars':[evidence_start,match.start()],'full_original_step_sha256':digest(text)})
   records.append(x);step_dispositions.append({'case_id':cid,'step':st['step'],'source_id':st['source_id'],'action':'author_evidence_to_conclusion','new_id':x['id']})
  # Whole-case summary with only this case's author evidence. No multi-case retrospective block.
  if cid in prior_case and sourceids:
   r=prior_case[cid]
   excerpt='\n'.join(f'{s["step"]}. '+(('盘次：'+step_board(d,s,units)[2].get('cast_key','跨盘或仅作者证据')+'；') if d['additional_printed_charts'] else '')+s['original'] for s in steps)
   # Long dense cases get complete contiguous analysis groups, not arbitrary character cuts.
   if len('\n'.join(s['original'] for s in steps))<=1800:
    records.append(row(('作者对照A、B两个不同病例。分清两例的条件差异与原断，不把B的结论转给A。\n' if comparison else ('本例多次排盘。按各步盘次分别归纳作者判断和依据，不把不同盘合并。\n' if d['additional_printed_charts'] else '历史复盘。归纳本例作者的主要判断与盘面依据；后续反馈没有列出时，不写成已经验证。\n'))+('原书抄盘：\n'+board+'\n' if board and not comparison and not d['additional_printed_charts'] else '')+'本例作者分析：\n'+excerpt,r['answer'],sorted(set(sourceids+r['source_ids'])),units,task='case_evidence_summary',format='complete_case',split=split,group=group,domain=d['domain'],case_ids=lesson_cases,provenance={'comparison_cases':comparison,'multiple_source_casts_separated':bool(d['additional_printed_charts']),'kind':'single_case_evidence_to_summary','parent_summary_id':r['id'],'unparsed_chart_not_fabricated':not bool(board)}))
  # Distinct source-printed positions are read independently from methodological interpretation.
  stars={}
  for c in d['primary_printed_chart']:
   match=re.match(r'([蓬任冲辅英芮柱心])',c['star_position_raw'])
   if match and match[1] not in stars:stars[match[1]]=int(c['palace'])
  pairs=[(a,b) for a in sorted(stars) for b in sorted(stars) if a!=b]
  pairs=sorted(pairs,key=lambda p:digest(cid+''.join(p)))[:(6 if split=='train' else 2)]
  chartids=sorted(sid for sid,unit in units.items() if unit.get('kind')=='original_printed_chart_cell' and unit.get('case_ids')==[cid] and ':primary:' in sid)
  for sid in chartids:
   cell=json.loads(units[sid]['original_quote'])
   if cell not in d['primary_printed_chart']:raise ValueError('Chart source cell not from this case')

  for a,b in pairs:
   pa,pb=stars[a],stars[b];rel=relation(pa,pb)
   sids=chartids+relation_sources
   if not chartids:continue
   q=f'只判断宫间五行方向：{a}星所在宫对{b}星所在宫，是生、克、受生、受克还是比和？先给关系，再给两个宫的五行。\n原书星行：\n'+board
   answer=f'{rel}。{a}在{PALACE[pa]}宫属{ELEMENT[pa]}，{b}在{PALACE[pb]}宫属{ELEMENT[pb]}。'
   records.append(row(q,answer,sids,units,task='directed_relation',format='chart_application',split=split,group=group,domain=d['domain'],case_ids=[cid],checks={'kind':'relation','answer':rel},provenance={'kind':'deterministic_relation_from_book_elements_and_literal_chart','positions':{a:pa,b:pb},'no_strength_or_outcome_score':True}))
 # Add manual single-case stage tasks supplied with exact narrow sources (no fallback full review).
 result={'schema':SCHEMA,'revision':2,'source_books':sorted(BOOKS),'parent_source_sha256':old['source_sha256'],'source_parent_course_sha256':digest(dumps(old)),'units':deepcopy(units),'records':records,'case_catalog':case_catalog,'original_case_splits':old['case_splits'],'sealed_original_test_manifest':sealed,'old_task_dispositions':dispositions,'author_step_dispositions':step_dispositions,'development_policy':{'event_families':sorted(dev),'ancestor_unseen_claimed':False,'new_phase_withdrawn_families':sorted(dev),'revealed_case_families_kept_for_repair':sorted(revealed_groups),'purpose':'multi_domain_development_diagnostic_not_blind_prediction','final_test_opened_for_repairs':False}}
 # Identical replays become one target with explicit deduplication, never different labels.
 unique={}
 for r in result['records']:
  k=(r['split'],dumps(r['messages'][:-1]))
  if k in unique:
   if compact(unique[k]['answer'])!=compact(r['answer']):raise ValueError('Conflicting rebuilt duplicate')
   unique[k]['source_ids']=sorted(set(unique[k]['source_ids']+r['source_ids']))
   if unique[k]['case_ids']!=r['case_ids']:
    if unique[k]['group']!=r['group']:raise ValueError('Identical case prompt from different event groups')
    unique[k]['case_ids']=sorted(set(unique[k]['case_ids']+r['case_ids']));unique[k]['provenance']['deduplicated_case_ids']=unique[k]['case_ids']
  else:unique[k]=r
 result['records']=list(unique.values());result['validation']=validate(result)
 result['summary']={'records':len(result['records']),'splits':dict(Counter(r['split'] for r in result['records'])),'formats':dict(Counter((r['format'] for r in result['records'] if r['split']=='train'))),'tasks':dict(Counter(r['task'] for r in result['records'])),'case_catalog':len(case_catalog),'original_author_steps':len(step_dispositions),'old_non_test_tasks_accounted':len(dispositions),'sealed_test_tasks':len(sealed),'old_materials_preserved':len(units),'train_case_domains':sorted({r['domain'] for r in result['records'] if r['split']=='train' and r['case_ids']}),'development_case_domains':sorted({r['domain'] for r in result['records'] if r['split']=='validation' and r['case_ids']}),'trainable_outcome_probability':False,'independent_prediction_validated':False}
 return result
