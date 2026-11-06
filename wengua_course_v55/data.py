"""Source-only V55: restored foundations, directed reasoning and domain chains."""
from collections import Counter,defaultdict
from copy import deepcopy
import re
from wengua_course import data as d
from wengua_course_v53 import data as v53
from wengua_course_v54 import data as v54,corpus as old_corpus
from . import sourceplan
SCHEMA='wengua_domain_reasoning_source_course_v55'
SYSTEM=v53.SYSTEM
REPLAY={'direction':3,'given_direction':2,'door_relation':2,'condition':2,'workflow':2}


def usable_author_phase(phase):
 return bool(phase) and not phase.startswith('更正或补充说明之前') and '本案有按语或更正' not in phase

def add(rows,question,answer,support,units,*,skill,split='train',domain='foundation',checks=None,task=None,group=None,case_ids=(),context='closed_book',provenance=None):
 row=d.row(question,answer,[s['source_id'] for s in support],units,task=task or skill,format='chart_application' if skill in ['positions','given_direction','direction','chart_direction','door_relation'] else 'complete_case' if skill=='author_evidence' else 'conditional_dialogue' if skill in ['condition','identity'] else 'foundation_qa',split=split,domain=domain,checks=checks,group=group,case_ids=case_ids,provenance=provenance or {})
 row['messages'][0]['content']=SYSTEM;row.update(stage='application',skill=skill,context=context);row['id']='v55:'+d.digest(d.dumps([row['messages'],split,skill]))[:24];rows.append(row);return row

def support_for(ids,units):return [{'source_id':sid,'quote':units[sid]['original_quote']} for sid in ids]

def supplied(rows,question,answer,support,units,**kw):
 refs=list({s['source_id']:s for s in support}.values());question=v53.supplied(question,[s['quote'] for s in refs]);target=answer+' '+ '、'.join('[S'+str(i)+']' for i in range(1,len(refs)+1));row=add(rows,question,target,refs,units,context='provided_evidence',**kw);row['provenance'].update(support=refs,required_citations=['S'+str(i) for i in range(1,len(refs)+1)],source_packet_is_literal_not_claimed_actual_search=True);return row

def rule_unit(part,library,units):
 chapters={old_corpus.r.chapter_key(c):c for c in library['chapters']};chapter=chapters[part['chapter']];text=old_corpus.r.quote_part(chapters,part)
 sid='v55:literal:'+d.digest(d.dumps(part))[:24]
 if sid not in units:units[sid]={'id':sid,'book_id':chapter['book_id'],'section_id':int(chapter['chapter_id']),'chapter':chapter['title'],'lines':part['lines'],'source_sha256':chapter['source_sha256'],'original_quote':text,'kind':'literal_rule_selector','case_ids':[],'selector':deepcopy(part)}
 return {'source_id':sid,'quote':text}

def word_relation(a,b,subject_a='前者',subject_b='后者'):
 representative={'水':1,'土':2,'木':3,'金':6,'火':9};rel=d.relation(representative[a],representative[b]);explain=v54.bridge(a,b)
 if subject_a!='前者':explain=explain.replace('前者',subject_a).replace('后者',subject_b)
 return rel,explain

def build(parent53,parent54,library,*,tokenizer=None,progress=lambda **kw:None):
 units=deepcopy(parent53['units']);rows=deepcopy(parent54['records']);dispositions=[]
 for row in rows:
  row['id']='v55:replay:'+row['id'];row['provenance']['v54_course_replay']=True
  # Minimal numerical definitions become ordinary complete statements.
  key=row['provenance'].get('key','')
  if key.startswith('hetu_') and key!='hetu_not_luoshu':
   element=key.split('_')[-1];body=row['answer'].split(' [S',1)[0];new='河图中，'+element+'对应的生成数是'+body
   row['answer']=new+row['answer'][len(body):];row['messages'][-1]['content']=row['answer']
  elif key=='gui_yin_water':
   body=row['answer'].split(' [S',1)[0];row['answer']='癸属阴水，方位为北方。'+row['answer'][len(body):];row['messages'][-1]['content']=row['answer']
  if row['checks'] and row['checks']['kind'] in ['relation','joint_relation']:
   c=row['checks'];elements=c.get('elements') or [d.ELEMENT[n] for n in c.get('positions',{}).values()]
   if elements:c['elements']=elements
  if row['skill']=='door_relation':
   key=row['provenance'].get('key','');match=re.fullmatch(r'door_direction_(.)_(\d)',key)
   if match:
    door,n=match[1],int(match[2]);row['checks'].update(elements=[old_corpus.DOORS[door],d.ELEMENT[n]],symbols=[door+'门',d.PALACE[n]+'宫'],door_relation=True)
  if key in ['exam_self','health_father_self']:row['provenance']['birth_year_unknown_in_question']=True
 cycles=old_corpus.properties(parent53,library);cycle_support=cycles['rule:generation']['support']+cycles['rule:control']['support']
 # Restore the TWO omitted task types, keeping case-family ownership intact.
 for old in parent53['records']:
  if old['stage']!='application' or old['task'] not in ['chart_positions','verified_positions_direction']:continue
  pos=old['provenance']['positions'];(a,pa),(b,pb)=list(pos.items());support=support_for(old['source_ids'],units);q=old['question'].split('\n',1)[1]
  if old['task']=='chart_positions':answer=f'{a}星在{d.PALACE[pa]}宫，宫属{d.ELEMENT[pa]}；{b}星在{d.PALACE[pb]}宫，宫属{d.ELEMENT[pb]}。';skill='positions';check=old['checks']
  else:
   rel,explain=word_relation(d.ELEMENT[pa],d.ELEMENT[pb],a+'星所在宫',b+'星所在宫');q=q.replace('只写关系词。','请说明双方宫位五行和作用方向。');answer=rel+'。'+a+'星所在宫属'+d.ELEMENT[pa]+'，'+b+'星所在宫属'+d.ELEMENT[pb]+'；'+explain;skill='given_direction';check=old['checks']|{'elements':[d.ELEMENT[pa],d.ELEMENT[pb]],'positions':pos,'symbols':[a+'星所在宫',b+'星所在宫']};support+=cycle_support
  add(rows,q,answer,support,units,skill=skill,split=old['split'],domain=old['domain'],checks=check,task=old['task'],group=old['group'],case_ids=old['case_ids'],provenance={'kind':'restored_source_task_with_worked_direction','positions':pos,'source_core_id':old['id']})
  if old['task']=='chart_positions' and old['split']=='validation':
   rel,explain=word_relation(d.ELEMENT[pa],d.ELEMENT[pb],a+'星所在宫',b+'星所在宫');answer=rel+'。'+a+'星所在宫属'+d.ELEMENT[pa]+'，'+b+'星所在宫属'+d.ELEMENT[pb]+'；'+explain
   q=f'位置已经核对：{a}星在{d.PALACE[pa]}宫，{b}星在{d.PALACE[pb]}宫。不再读取星行，只判断前一星所在宫对后一星所在宫的关系并说明方向。'
   check={'kind':'relation','answer':rel,'elements':[d.ELEMENT[pa],d.ELEMENT[pb]],'symbols':[a+'星所在宫',b+'星所在宫']}
   add(rows,q,answer,support+cycle_support,units,skill='given_direction',split='validation',domain=old['domain'],checks=check,task='given_direction_holdout',group=old['group'],case_ids=old['case_ids'],provenance={'kind':'heldout_source_family_given_positions_direction','positions':pos,'source_core_id':old['id']})
 # Every directed elemental relation, its inverse, and an explicit wrong-claim
 # correction. Generated exercises are labelled rule exercises, never events.
 for a in ['水','土','木','金','火']:
  for b in ['水','土','木','金','火']:
   rel,explain=word_relation(a,b);c={'kind':'relation','answer':rel,'elements':[a,b]}
   variants=[('train',f'对象A属{a}，对象B属{b}。A对B是什么五行关系？请分清施力与受力方。',rel+'。对象A属'+a+'，对象B属'+b+'；'+word_relation(a,b,'A','B')[1]),('train',f'前者属{a}，后者属{b}。请用五行循环核对方向，不把前后倒置。',rel+'。'+explain),('validation',f'以属{a}的对象A为观察主体，面对属{b}的对象B，它是生、受生、克、受克还是比和？请解释。',rel+'。'+word_relation(a,b,'A','B')[1])]
   for split,q,answer in variants:supplied(rows,q,answer,cycle_support,units,skill='direction',split=split,checks=c,provenance={'kind':'source_cycle_subject_direction_exercise','elements':[a,b]})
   wrong=next(x for x in ['生','克','受生','受克','比和'] if x!=rel)
   for split,q in [('train',f'有人说：前者{a}、后者{b}，前者对后者为{wrong}。请核对并纠正关系和方向。'),('validation',f'纠错题：对象A属{a}，对象B属{b}，A对B被写成{wrong}。按原书生克循环应怎样改？')]:
    supplied(rows,q,'应改为'+rel+'。'+explain,cycle_support,units,skill='direction',split=split,checks=c,provenance={'kind':'explicit_false_claim_is_user_input_not_teacher','elements':[a,b]})
 # Information boundaries: asking for a missing year differs from using one.
 specs={s['key']:s for s in parent53['reviewed_specs']};boundary=[
 ('education','exam_self','考生本人问考试，不知道出生年。能先按日干分析吗？年命现在能取吗？','先取日干代表考生本人。出生年未知，暂不取年命；需要兼看时再补充可靠出生年。','日干','时干'),
 ('education','exam_proxy','母亲代问女儿考试，未知女儿出生年。考生与年命分别怎样处理？','取时干代表女儿这位考生。出生年未知，暂不取年命；不能把起局年干当成她的年命。','时干','日干'),
 ('health','health_daughter_father','女儿代问父亲疾病，没有父亲出生年。年干和年命是不是一回事？','患者身份先看起局年干，兼看乾宫。年命是父亲出生年的年干，需要可靠出生年；未知时先不取年命。','年干','日干'),
 ('health','health_father_self','父亲本人问自己的病，出生年没提供。患者先取谁？','取日干代表患者本人。出生年未知，暂不取年命；父亲的家庭身份不能把本人用神改成年干。','日干','年干')]
 year='book:4:L180-180:cd8e3175a112'
 for domain,key,q,answer,role,opposite in boundary:
  support=deepcopy(specs[key]['support'])+[{'source_id':year,'quote':units[year]['original_quote']}]
  for split,prefix in [('train',''),('validation','请核实当前条件：')]:supplied(rows,prefix+q,answer,support,units,skill='condition',split=split,domain=domain,checks={'kind':'birth_boundary','answer':role,'opposite':opposite},provenance={'kind':'known_question_identity_unknown_birth_year','birth_year_unknown_in_question':True})
 marriage=specs['marriage_same_palace']
 for split,q in [('train','乙、庚两宫相生，但吉门吉格尚未核实，能直接说婚姻一定成功吗？'),('validation','只确认男女用神两宫相生，其他条件未给出。原书还要求什么，能否保证婚成？')]:
  supplied(rows,q,'不能保证婚成。本条还要求吉门、吉格，不能只凭两宫相生下结论。',marriage['support'],units,skill='condition',split=split,domain='marriage',checks={'kind':'condition_boundary','required':[['不能保证','不能只凭','不一定'],['吉门'],['吉格']],'forbidden':['一定成功','必定结婚']},provenance={'kind':'source_rule_premise_not_filled'})
 # Preserve a modest set of source-labelled citation permutations. All literal
 # source contents are unchanged and every answer label is remapped together.
 picked=set()
 for old in parent54['records']:
  p=old['provenance'];key=p.get('key');slot=(old['split'],key)
  if not key or slot in picked or old['skill'] not in ['identity','source_fact','condition','typed_pair'] or len(old['messages'])!=3 or not p.get('actual_production_rag_prompt'):continue
  labels=sorted(set(re.findall(r'\[(S\d+)\]',old['messages'][-2]['content'])),key=lambda s:int(s[1:]))
  if len(labels)<2:continue
  picked.add(slot);mapping=dict(zip(labels,labels[1:]+labels[:1]));remap=lambda text:re.sub(r'\[(S\d+)\]',lambda m:'['+mapping.get(m[1],m[1])+']',text)
  row=deepcopy(old);row['messages'][-2]['content']=remap(row['messages'][-2]['content']);row['messages'][-1]['content']=remap(row['messages'][-1]['content']);row['question']=row['messages'][-2]['content'];row['answer']=row['messages'][-1]['content'];row['id']='v55:source_labels:'+d.digest(d.dumps(row['messages']))[:24]
  row['provenance'].update(actual_production_rag_prompt=False,source_label_permutation=True,source_content_unchanged=True,required_citations=[mapping[s] for s in p['required_citations']],label_mapping=mapping,reference_labels_remapped=True)
  for ref in row['provenance'].get('references',[]):ref['citation']=mapping[ref['citation']]
  rows.append(row)
 # Domain rules and workflow groups use only existing reviewed literal selectors.
 plan=sourceplan.plan(library);rule_support={}
 for item in plan['rules']:
  support=[rule_unit(p,library,units) for p in item['parts']];rule_support[item['id']]=support
  answer='\n'.join(s['quote'] for s in support)
  for split,prefix in [('train',''),('validation','请按原书核对：')]:
   supplied(rows,prefix+item['question']+'请保留原文中的适用条件，不把本条当成所有领域的统一规则。',answer,support,units,skill='domain_rule',split=split,domain=item['domain'],provenance={'kind':'existing_reviewed_specific_question_literal_answer','source_document':item['id'],'literal_answer_not_new_paraphrase':True})
 rag_missing=[]
 for flow in plan['workflows']:
  support=[s for rule in flow['rules'] for s in rule_support[rule['id']]];answer='\n'.join(str(i+1)+'、'+rule['text'] for i,rule in enumerate(flow['rules']));q='按'+d.DOMAIN_ZH.get(flow['domain'],flow['domain'])+'本组规则，依次说明：'+'；'.join(rule['question'].rstrip('？') for rule in flow['rules'])+'。这是所给原书规则组，不补未列出的断案步骤。'
  for split,prefix in [('train',''),('validation','请把本组前提与结论分清：')]:
   supplied(rows,prefix+q,answer,support,units,skill='workflow',split=split,domain=flow['domain'],provenance={'kind':'source_ordered_domain_rule_group','workflow_id':flow['id'],'source_documents':[x['id'] for x in flow['rules']]})
   if tokenizer is not None:
    from . import corpus
    query=prefix+q;packet=corpus.prompt(query,tokenizer,domain=flow['domain'],book=flow['book_id'],max_context=3328,max_new_tokens=1024);citations=corpus.covering_citations(packet,support,units)
    if citations:
     target=answer+' '+ '、'.join('['+c+']' for c in citations);r=add(rows,query,target,support,units,skill='workflow',split=split,domain=flow['domain'],context='provided_evidence',provenance={'kind':'actual_domain_workflow_rag','workflow_id':flow['id'],'support':support,'actual_production_rag_prompt':True,'required_citations':citations,'prompt_budget':packet['prompt_budget'],'references':[{'id':x['id'],'citation':x['citation'],'sources':x['sources']} for x in packet['references']]});r['messages']=packet['messages']+[{'role':'assistant','content':target}];r['question']=r['messages'][-2]['content']
    else:rag_missing.append({'workflow_id':flow['id'],'split':split,'disposition':'not_taught_as_actual_rag_missing_visible_source'})
 progress(step='domain_rules_and_restored_foundations',rows=len(rows))
 # Author-selected evidence -> withheld literal judgment, without later feedback.
 cases={c['case_id']:c for c in parent53['case_catalog']};bridge_inventory=[]
 for old in parent53['records']:
  if old['task']!='author_source_application':continue
  p=old['provenance'];cid=p['source_case_boundary'];case=cases[cid];step=next(s for s in case['author_steps'] if s['step']==p['author_step']);text=step['original'];span=p.get('evidence_chars');target_span=p.get('target_chars');reason=None
  if not span or not target_span or p.get('source_phase','').startswith('更正或补充说明之前'):reason='missing_safe_span_or_pre_correction_analysis'
  elif not usable_author_phase(p.get('source_phase')):reason='unresolved_correction_phase_source_review_only'
  else:
   evidence=text[slice(*span)];target=text[slice(*target_span)]
   if len(evidence)<18 or len(target)<3 or d.compact(target) in d.compact(evidence):reason='judgment_already_visible_or_insufficient_context'
  bridge_inventory.append({'source_core_id':old['id'],'case_id':cid,'author_step':step['step'],'disposition':reason or 'author_selected_evidence_to_literal_judgment','not_independent_fresh_chart_forecast':True})
  if reason:continue
  support=[{'source_id':step['source_id'],'quote':evidence}];q='历史案例教学：根据作者本段已经列出的证据，推导本段所述判断并说明依据。此处未提供后续反馈，不补结局。\n作者证据：'+evidence
  short=text[slice(*p['short_evidence_chars'])] if p.get('short_evidence_chars') else evidence
  if not short or d.compact(short) not in d.compact(evidence):short=evidence
  answer='作者在本段据此判断：'+target.rstrip('。')+'。依据：'+short
  row=add(rows,q,answer,support,units,skill='author_evidence',split=old['split'],domain=old['domain'],group=old['group'],case_ids=old['case_ids'],provenance={'kind':'author_evidence_to_withheld_literal_judgment','author_step':step['step'],'evidence_chars':span,'target_chars':target_span,'source_step_sha256':d.digest(text),'source_phase':p.get('source_phase'),'support':support,'later_feedback_in_input':False,'not_independent_prediction':True})
  row['provenance']['literal_target']=target
 unique={}
 for row in rows:
  key=(row['split'],d.dumps(row['messages'][:-1]))
  if key in unique and unique[key]['answer']!=row['answer']:raise ValueError('Conflicting teacher prompts')
  unique.setdefault(key,row)
 rows=list(unique.values());result={'schema':SCHEMA,'records':rows,'extra_units':{k:v for k,v in units.items() if k not in parent53['units']},'source_books':parent53['source_books'],'legacy_v53_prompts':parent54['legacy_v53_prompts'],'legacy_v54_prompts':None,'legacy_diagnostics':parent54['legacy_diagnostics'],'rule_inventory':plan['rule_inventory'],'workflow_plan':plan['workflows'],'workflow_rag_missing':rag_missing,'author_bridge_inventory':bridge_inventory,'source_model_answers_as_teachers':False,'old_manual_scores_used':False,'original_final_test_used':False}
 from wengua_course_v54.runtime import choose_development
 result['legacy_v54_prompts']=choose_development(parent54,128)
 result['summary']={'records':len(rows),'splits':dict(Counter(r['split'] for r in rows)),'skills':dict(Counter(r['skill'] for r in rows)),'restored_positions':sum(r['task']=='chart_positions' for r in rows),'restored_given_direction':sum(r['task']=='verified_positions_direction' for r in rows),'given_direction_validation_added':sum(r['task']=='given_direction_holdout' for r in rows),'literal_domain_rule_documents':len(plan['rules']),'literal_workflow_groups':len(plan['workflows']),'source_rule_inventory':len(plan['rule_inventory']),'domain_rule_coverage':sorted({x['domain'] for x in plan['rules']}),'workflow_domains':sorted({x['domain'] for x in plan['workflows']}),'actual_rag_rows':sum(r['provenance'].get('actual_production_rag_prompt',False) for r in rows),'citation_permutation_rows':sum(r['provenance'].get('source_label_permutation',False) for r in rows),'author_evidence_lessons':sum(r['skill']=='author_evidence' for r in rows),'cases_preserved':len(parent53['case_catalog']),'author_steps_preserved':sum(len(c['author_steps']) for c in parent53['case_catalog']),'new_material_selectors':len(result['extra_units']),'original_final_test_used':False};result['summary']['unresolved_author_bridge_candidates_not_taught']=sum(x['disposition']=='unresolved_correction_phase_source_review_only' for x in bridge_inventory);return result

def compose(parent,delta):
 pack=deepcopy(parent);pack.update(delta);pack['units'].update(delta.get('extra_units',{}));return pack

def validate(pack):
 # Reuse the corrected conversation/source/family checks, never V53's missing
 # system-role construction. The actual schema is still separately required.
 v54.validate(pack|{'schema':v54.SCHEMA})
 if pack['schema']!=SCHEMA:raise ValueError('V55 schema required')
 for row in pack['records']:
  for s in row['provenance'].get('support',[]):
   if d.compact(s['quote']) not in d.compact(pack['units'][s['source_id']]['original_quote']):raise ValueError('Nonliteral support')
  if row['skill']=='author_evidence':
   p=row['provenance']
   if d.compact(p['literal_target']) in d.compact(row['messages'][-2]['content']):raise ValueError('Withheld judgment leaked into input')
   if p['later_feedback_in_input']:raise ValueError('Later feedback in author-evidence exercise')
 return True
