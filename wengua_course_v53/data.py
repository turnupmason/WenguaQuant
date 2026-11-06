"""V53 source-supported tasks; no model answer or manual outcome scores.

Memory, supplied-evidence reading, symbolic direction and historical author
review are distinct tasks. Original test cases are never teaching examples.
"""
from copy import deepcopy
from collections import Counter,defaultdict
import re
from wengua_course import data as d
SCHEMA='wengua_source_grounded_course_v53'
SYSTEM='用自然、完整的中文回答当前问题。先说明判断，再解释必要条件与具体依据；正常交流不只报一个词，也不重复结论。给出原文时以原文为依据，区分原断、后来反馈和事后复盘；缺少条件就指出缺少什么，不补造身份、宫位、日期、金额或结果。用程序提供的真实盘面，按当前领域取用，不使用人工吉凶分数。'

def add(records,question,answer,ids,units,*,task,format='foundation_qa',stage='foundation',split='train',domain='foundation',group=None,case_ids=(),checks=None,context='closed_book',provenance=None,messages=None):
 if messages:
  messages=deepcopy(messages);messages[0]['content']=SYSTEM;messages[-1]['content']=answer
  messages[1]['content']=d.DOMAIN_ZH.get(domain,domain)+'。\n'+messages[1]['content']
 r=d.row(question,answer,ids,units,task=task,format=format,split=split,domain=domain,group=group,case_ids=case_ids,checks=checks,messages=messages,provenance=provenance)
 r['messages'][0]['content']=SYSTEM
 key=d.digest(d.dumps([task,split,r['messages'],ids,stage]))[:24]
 r.update(id='v53:'+key,question=r['messages'][1]['content'],stage=stage,context=context);records.append(r);return r

def supplied(question,quotes):return question+'\n\n本题检索证据：\n'+'\n'.join('[S'+str(i)+'] '+q for i,q in enumerate(quotes,1))
def cite(answer,count=1,numbers=None):return answer+'\n依据：'+'、'.join('[S'+str(i)+']' for i in (numbers or range(1,count+1)))+'。'

def reviewed_specs(parent,manual):
 manual=deepcopy(manual);general_id='book:4:L180-180:cd8e3175a112'
 for spec in manual:
  if spec['key']=='exam_proxy':spec['answer']='取时干代表实际考生。父母代问子女考试时按时干取用；出生年未知，不补造年命。'
  if spec['domain']=='health':
   for item in spec['support']:
    if item['source_id']=='book:8:L6-6:c231e5a18b12':
     item.update(source_id=general_id,quote=parent['units'][general_id]['original_quote'])
 return manual

CORRECTION_MARKER=r'(?:事后又作了?如下分析|正确判断应该如此分析|正确的分析|张志春按[：:]|(?:^|\n)其实[，,])'

def case_stages(parent,library):
 """Existing reviewed annotations + literal correction boundaries, not a verdict judge."""
 from wengua_corpus import book_rag as original_rag
 chapters={original_rag.chapter_key(c):c for c in (library or {}).get('chapters',[])};result={}
 for doc in (library or {}).get('documents',[]):
  if doc['kind']!='case' or len(doc['case_ids'])!=1:continue
  cid=doc['case_ids'][0]
  if parent['original_case_splits'].get(cid)=='test':continue
  entry=result.setdefault(cid,{'review_annotation_requires_correction':False,'markers':[]})
  entry['review_annotation_requires_correction']|=any(a.get('correction_required') for a in doc.get('analysis_annotations',[]))
  for part in doc['parts']:
   text=original_rag.quote_part(chapters,part);lo,hi=part['lines'];raw='\n'.join(chapters[part['chapter']]['original_text'].splitlines()[lo-1:hi]);offset=part.get('chars',[0,len(raw)])[0]
   for match in re.finditer(CORRECTION_MARKER,text):
    marker={'chapter':part['chapter'],'line':lo+raw[:offset+match.start()].count('\n'),'source_document':doc['id']}
    if marker not in entry['markers']:entry['markers'].append(marker)
 return result

def step_phase(cid,source,stages):
 entry=stages.get(cid,{})
 chapter=source['book_id']+'/'+str(source.get('section_id'));markers=[m['line'] for m in entry.get('markers',[]) if m['chapter']==chapter]
 if markers and source.get('lines'):
  if source['lines'][0]<min(markers):return '更正或补充说明之前的分析（不能单独当作最终判断）',bool(entry.get('review_annotation_requires_correction'))
  return '原书后段的更正或补充分析',False
 return ('原书记录中的分析（本案有按语或更正，人物及阶段不能混用）' if entry.get('review_annotation_requires_correction') else '原书记录的分析'),False

def case_chains(parent,units,records,library=None,stages=None):
 """Complete author analyses plus SOURCE-LOCATED initial/feedback excerpts.
 No former summary target is reused, and no unseen feedback is inferred.
 """
 stages=stages if stages is not None else case_stages(parent,library);owners=defaultdict(set)
 for r in parent['records']:
  if r['split']!='diagnostic':
   for cid in r['case_ids']:owners[cid].add((r['group'],r['split']))
 normalized={s:d.compact(u['original_quote']) for s,u in units.items()};by_case=defaultdict(list)
 for sid,u in units.items():
  for cid in u.get('case_ids',[]):by_case[cid].append(sid)
 bridges=defaultdict(list)
 for r in parent['records']:
  if r['task']=='author_evidence_to_conclusion':bridges[r['provenance']['source_case_boundary']].append(r)
 outcomes=[];chapters={}
 for chapter in (library or {}).get('chapters',[]):
  raw=chapter['original_text'];indices=[i for i,ch in enumerate(raw) if not ch.isspace()];norm=''.join(raw[i] for i in indices);chapters[(chapter['book_id'],str(chapter['chapter_id']))]=(chapter,norm,indices)
 def located(cid,text):
  if not text or len(d.compact(text))<12:return None
  candidates=[sid for sid in by_case[cid] if d.compact(text) in normalized[sid]]
  if candidates:return min(candidates,key=lambda sid:len(normalized[sid]))
  allowed={(units[s]['book_id'],str(units[s]['section_id'])) for s in by_case[cid] if units[s].get('section_id') is not None}
  case=next(c for c in parent['case_catalog'] if c['case_id']==cid)
  allowed.update((units[s['source_id']]['book_id'],str(units[s['source_id']]['section_id'])) for s in case['author_steps'] if units[s['source_id']].get('section_id') is not None)
  matches=[];needle=d.compact(text)
  for key,(chapter,norm,indices) in chapters.items():
   if allowed and key not in allowed:continue
   if not allowed and chapter['book_id']!=cid.split('.')[0]:continue
   start=norm.find(needle)
   if start>=0 and norm.find(needle,start+1)<0:matches.append((chapter,indices[start],indices[start+len(needle)-1]+1))
  if len(matches)!=1:return None
  chapter,start,end=matches[0];raw=chapter['original_text'];quote=raw[start:end];sid='v53phase:'+chapter['book_id']+':'+str(chapter['chapter_id'])+':'+str(start)+'-'+str(end)+':'+d.digest(quote)[:12]
  units[sid]={'id':sid,'book_id':chapter['book_id'],'chapter':chapter['title'],'section_id':int(chapter['chapter_id']),'source_sha256':chapter['source_sha256'],'lines':[raw[:start].count('\n')+1,raw[:end].count('\n')+1],'chars':[start,end],'original_quote':quote,'kind':'source_located_case_phase_excerpt','case_ids':[cid],'literal_match':True,'source_library_unchanged':True}
  normalized[sid]=d.compact(quote);by_case[cid].append(sid);return sid
 def excerpt(text):
  if len(text)<=180:return text
  sentences=re.findall(r'[^。！？]+[。！？]?',text)
  eligible=[s.strip() for s in sentences if 12<=len(s.strip())<=180 and not s.strip().endswith('？')]
  return eligible[-1] if eligible else None
 for c in parent['case_catalog']:
  cid=c['case_id']
  if parent['original_case_splits'][cid]=='test':continue
  own=owners.get(cid,set())
  if len(own)>1:raise ValueError('Case chain family/split ambiguous')
  group,split=next(iter(own)) if own else (c['group'],'validation' if c['group'] in parent['development_policy']['new_phase_withdrawn_families'] else 'train')
  refs=[];ids=[];steps={};bindings=[]
  for step in c['author_steps']:
   sid=step['source_id'];text=step['original']
   if d.compact(text) not in normalized[sid]:raise ValueError('Case author step is not literal')
   binding=d.step_board(c,step,units)[2];bindings.append(binding);number=len(refs)+1;steps[step['step']]=number;label,_=step_phase(cid,units[sid],stages);refs.append(label+'第'+str(step['step'])+'步，盘次'+binding.get('cast_key','跨盘或仅作者证据')+'：'+text);ids.append(sid)
  selected=[];phase_proofs={};phase_targets={};supplement_targets=[];supplements=[]
  for field,label in [('author_original_judgment','书中初断及按语选段'),('later_feedback_separate','书中后来反馈选段')]:
   full=c.get(field,'');sentences=[s.strip() for s in re.findall(r'[^。！？]+[。！？]?',full) if 12<=len(s.strip())<=180 and not s.strip().endswith('？')]
   candidates=([full] if full and len(full)<=180 else [])+(sentences if field=='author_original_judgment' else list(reversed(sentences)));sid=None
   for part in dict.fromkeys(candidates):
    sid=located(cid,part)
    if sid:break
   if sid:
    number=len(refs)+1;refs.append(label+'：'+part);ids.append(sid);phase_proofs[field]={'citation':'S'+str(number),'source_id':sid,'literal_quote':part};phase_targets[field]=label+'：'+part+' [S'+str(number)+']'
  choices=[r for r in bridges[cid] if not step_phase(cid,units[next(s['source_id'] for s in c['author_steps'] if s['step']==r['provenance']['author_step'])],stages)[1]]
  if choices:
   ordered=sorted(choices,key=lambda r:r['provenance']['author_step']);chosen=[ordered[0]] if len(ordered)==1 else [ordered[0],ordered[-1]]
   for r in chosen:
    n=steps[r['provenance']['author_step']];sid=next(s['source_id'] for s in c['author_steps'] if s['step']==r['provenance']['author_step']);label,_=step_phase(cid,units[sid],stages);selected.append(label+'中提出：'+r['answer']+' [S'+str(n)+']')
  elif c['author_steps']:
   safe=[s for s in c['author_steps'] if not step_phase(cid,units[s['source_id']],stages)[1]]
   for step in (safe[:1]+safe[-1:] if len(safe)>1 else safe):
    part=excerpt(step['original'])
    if part:selected.append(step_phase(cid,units[step['source_id']],stages)[0]+'中列出的依据：'+part+' [S'+str(steps[step['step']])+']')
  if library:
   from wengua_corpus import book_rag as original_rag
   source_chapters={original_rag.chapter_key(ch):ch for ch in library['chapters']};seen_supp=set();known=''.join(d.compact(step['original']) for step in c['author_steps'])
   pattern=r'(?:[^\n]{0,12}事后又作了如下分析[：:]|(?:^|\n)其实[，,]|张志春按[：:]|(?:^|\n)[^\n]{0,40}正确(?:判断应该如此|的)?分析[：:])'
   for doc in library['documents']:
    if doc['kind']!='case' or doc['case_ids']!=[cid]:continue
    raw=original_rag.document_text(source_chapters,doc);match=re.search(pattern,raw)
    if not match:continue
    block=raw[match.start():].strip();normalized_block=d.compact(block)
    if normalized_block in seen_supp or normalized_block in known:continue
    sid=located(cid,block)
    if not sid:continue
    seen_supp.add(normalized_block);number=len(refs)+1;refs.append('原书另列的解释或按语（按文内先后理解）：'+block);ids.append(sid)
    sentences=[x.strip() for x in re.findall(r'[^。！？]+[。！？]?',block) if 12<=len(x.strip())<=200 and any(w in x for w in ('宫','用神','门迫','年命','入墓','入库'))]
    picked=([sentences[0],sentences[-1]] if len(sentences)>1 else sentences) or [excerpt(block)]
    picked=[x for x in dict.fromkeys(picked) if x]
    if picked and any(any(w in x for w in ('宫','用神','门迫','年命','入墓','入库')) for x in picked):
     fragment='原书补充解释：'+' '.join(picked);supplement_targets.append(fragment+' [S'+str(number)+']')
     add(records,supplied('请说明这段原书解释列出的条件及对判断的限制，不把后来的说明提前当作初断已经掌握。',[block]),cite(fragment),[sid],units,task='case_author_supplement',format='complete_case',stage='application',split=split,domain=c['domain'],group=group,case_ids=[cid],context='provided_evidence',provenance={'kind':'literal_case_explanation_not_new_prediction','source_document':doc['id'],'target_literal_extracts':picked,'citation_required':True})
    supplements.append({'source_id':sid,'source_document':doc['id'],'citation':'S'+str(number),'literal_quote':block,'target_extracts':picked})
  selected=([phase_targets['author_original_judgment']] if 'author_original_judgment' in phase_targets else [])+selected+([phase_targets['later_feedback_separate']] if 'later_feedback_separate' in phase_targets else [])+supplement_targets
  if not refs or not selected:
   outcomes.append({'case_id':cid,'disposition':'source_catalog_only_no_safe_complete_target'});continue
  # Phase records first, then the causal author claims; all remain explicitly attributed.
  question='请复盘这一个原书案例：说明主要判断、作者采用的依据，并分清初断、作者按语和后来反馈。只使用所给材料，未列出的结局不补。'
  if c['additional_printed_charts']:question+='本例多次起盘，按各步盘次分别理解，不能把不同盘合并。'
  elif c['primary_printed_chart'] and cid not in d.COMPARISONS:question+='\n原书明确抄盘（历史资料，不重新起局）：\n'+d.chart_text(c)
  if cid in d.COMPARISONS:question+='本例比较A、B两个病例，保留两例标签。'
  question=supplied(question,refs)
  answer='\n'.join(dict.fromkeys(selected))
  case_ids=list(d.COMPARISONS[cid].values()) if cid in d.COMPARISONS else [cid]
  add(records,question,answer,ids,units,task='case_source_chain',format='complete_case',stage='application',split=split,domain=c['domain'],group=group,case_ids=case_ids,context='provided_evidence',provenance={'kind':'source_located_author_chain_not_blind_prediction','author_steps_visible':len(c['author_steps']),'phase_sources':phase_proofs,'board_bindings':bindings,'no_old_summary_target_reused':True,'supplemental_source_sections':supplements,'source_stage_index':stages.get(cid,{})})
  outcomes.append({'case_id':cid,'disposition':'author_chain_with_visible_phase_sources','author_steps':len(c['author_steps']),'initial_quote_visible':'author_original_judgment' in phase_proofs,'feedback_quote_visible':'later_feedback_separate' in phase_proofs,'supplemental_source_sections':len(supplements)})
 return outcomes

def build(parent,manual,*,library=None):
 if library is not None:
  from wengua_corpus import book_rag as original_rag
  original_rag.validate_library(library)
  if set(library['books'])!=d.BOOKS:raise ValueError('Only the two original books are allowed')
 manual=reviewed_specs(parent,manual);stages=case_stages(parent,library)
 units=deepcopy(parent['units']);records=[];dispositions=[]
 # Only literal, explicitly selected support fragments are used as rule teachers.
 for spec in manual:
  if spec.get('case_ids'):continue
  support=spec['support'];ids=[s['source_id'] for s in support]
  for s in support:
   if d.compact(s['quote']) not in d.compact(units[s['source_id']]['original_quote']):raise ValueError('Manual support differs from source')
  for split,qkey in [('train','question'),('validation','development_question')]:
   q=spec.get(qkey)
   if not q:continue
   q=q.replace('只报两个数。','并说明采用的数系。')
   answer=spec['answer'];role_label=None
   if spec['key'].startswith('role_'):
    match=re.search(r'这一条规则，(.+?)以什么符号代表',spec['question']);role_label=match[1] if match else None
    if not role_label:raise ValueError('Rule role label absent')
    answer='本章以'+spec['answer'].rstrip('。')+'代表'+role_label+'。'
   if spec['key'].startswith('named_pair_'):
    pair=spec['key'].split('_')[-1];answer='本条称为'+spec['answer'].rstrip('。')+'。对应天盘'+pair[0]+'加地盘'+pair[1]+'，要保留这个组合条件。'
   if spec['key'].startswith('hetu_') and spec['key']!='hetu_not_palace':
    answer=spec['answer']+'这里采用的是河图天地生成数，不能与后天九宫数混用。'
   if spec['key']=='exam_proxy':answer='取时干代表实际考生。父母代问子女考试时按时干取用；出生年未知，不补造年命。'
   # Preserve multi-turn identity changes; replace only the final user question.
   messages=deepcopy(spec.get('messages'))
   if messages:
    messages[-2]['content']=q;messages[-1]['content']=answer
   for context in ('closed_book','provided_evidence'):
    quotes=[s['quote'] for s in support];used_ids=list(ids);numbers=list(range(1,len(support)+1))
    donors=[s for s in manual if not s.get('case_ids') and s['domain']!=spec['domain'] and s['key'].startswith('role_' if spec['domain']=='foundation' else 'hetu_')]
    if context=='provided_evidence' and donors:
     donor=donors[int(d.digest(spec['key'])[:8],16)%len(donors)]['support'][0]
     if int(d.digest(spec['key']+split)[:8],16)%2:
      quotes.insert(0,donor['quote']);numbers=[n+1 for n in numbers]
     else:quotes.append(donor['quote'])
     used_ids.append(donor['source_id'])
    scoped_q='按'+d.source_scope(ids,units)+'的'+d.DOMAIN_ZH.get(spec['domain'],spec['domain'])+'规则，'+q
    qs=q if context=='closed_book' else supplied(scoped_q,quotes)
    ms=deepcopy(messages)
    if ms:ms[-2]['content']=qs
    a=answer if context=='closed_book' else cite(answer,numbers=numbers)
    check=deepcopy(spec.get('checks'))
    if check and role_label:check['role_label']=role_label
    add(records,qs,a,used_ids,units,task='reviewed_'+spec['key'],format=spec['format'],split=split,domain=spec['domain'],context=context,checks=check,messages=ms,provenance={'kind':'literal_reviewed_rule','key':spec['key'],'support':support,'citation_required':context=='provided_evidence','required_citations':['S'+str(i) for i in numbers] if context=='provided_evidence' else [],'distractor_is_source_not_teacher':len(used_ids)>len(ids)})
 # Ordered names with source-backed short explanations, not a bare-word house style.
 for old in parent['records']:
  if old['task']!='ordered_name' or old['split']!='train':continue
  pair=old['provenance'].get('pair')
  if not pair:
   text=old['messages'][-2]['content'];m=re.search(r'天盘([甲乙丙丁戊己庚辛壬癸]).*?地盘([甲乙丙丁戊己庚辛壬癸])',text);pair=''.join(m.groups()) if m else None
  if not pair:raise ValueError('Ordered pair metadata missing')
  for split,q in [('train',f'天盘{pair[0]}、地盘{pair[1]}，本条叫什么？说明组合条件。'),('validation',f'按所限定原书，上层{pair[0]}、下层{pair[1]}的格名与条件是什么？')]:
   label=old['checks']['answer'];answer=f'本条称为{label}。对应的是天盘{pair[0]}加地盘{pair[1]}，要保留这个上下层顺序。'
   ids=old['source_ids'];quote='\n'.join(units[s]['original_quote'] for s in ids)
   for context in ('closed_book','provided_evidence'):
    add(records,q if context=='closed_book' else supplied(q,[quote]),answer if context=='closed_book' else cite(answer),ids,units,task='ordered_name',split=split,domain='formation',context=context,checks={'kind':'exact_name','answer':label},provenance={'kind':'ordered_pair_literal_name','pair':pair,'citation_required':context=='provided_evidence'})
 # Source insufficiency is taught explicitly with a different literal pair, never a fake rule.
 names=[r for r in records if r['task']=='ordered_name' and r['split']=='train' and r['context']=='provided_evidence']
 for i,base in enumerate(names[:24]):
  other=names[(i+17)%len(names)]
  if base['checks']['answer'] in '\n'.join(units[s]['original_quote'] for s in other['source_ids']):continue
  pair=base['provenance']['pair'];q=f'只根据所给这段原文，能否确认天盘{pair[0]}、地盘{pair[1]}这一组合的格名？材料没有写的不要用记忆补充。'
  proof='\n'.join(units[s]['original_quote'] for s in other['source_ids'])
  for split in ['train','validation']:
   query=q if split=='train' else f'本题只允许使用所列证据。它是否足以支持上层{pair[0]}、下层{pair[1]}的具体格名？'
   add(records,supplied(query,[proof]),cite('这段证据不足以确认所问组合的格名。它给的是另一个上下层组合，需要补充本题对应的原文后再判断。'),other['source_ids'],units,task='insufficient_source',split=split,context='provided_evidence',checks={'kind':'insufficient'},provenance={'kind':'different_literal_pair_not_a_fake_rule','asked_pair':pair,'provided_pair':other['provenance']['pair'],'citation_required':True})
 # Teach the five directed element relations independently from chart reading.
 element_ids=[s for s,u in units.items() if '木生火' in d.compact(u['original_quote']) and '金生水' in d.compact(u['original_quote']) and s.startswith('book:')]
 if not element_ids:raise ValueError('No literal five-element generation source')
 control_ids=[s for s,u in units.items() if '木克土' in d.compact(u['original_quote']) and '金克木' in d.compact(u['original_quote']) and s.startswith('book:')]
 if not control_ids:raise ValueError('No literal five-element control source')
 ids=[min(element_ids,key=lambda s:len(units[s]['original_quote'])),min(control_ids,key=lambda s:len(units[s]['original_quote']))]
 representative={'水':1,'土':2,'木':3,'金':6,'火':9}
 for x,pa in representative.items():
  for y,pb in representative.items():
   rel=d.relation(pa,pb)
   for split,version in [('train','从前者看后者'),('validation','关系方向固定为前者对后者')]:
    for context in ('closed_book','provided_evidence'):
     q=f'前者属{x}，后者属{y}。{version}，是生、克、受生、受克还是比和？这是一道关系检查，只写关系词。'
     proof='五行相生：木生火，火生土，土生金，金生水，水生木。五行相克：木克土，土克水，水克火，火克金，金克木。相同五行为比和。'
     add(records,q if context=='closed_book' else supplied(q,[proof]),rel+'。' if context=='closed_book' else cite(rel+'。'),ids,units,task='element_direction',format='chart_application',split=split,context=context,checks={'kind':'relation','answer':rel},provenance={'kind':'deduced_from_book_element_cycles','elements':[x,y],'citation_required':context=='provided_evidence'})
 # Every selected literal chart pair has BOTH query directions; no physical chart mutation.
 seen=set()
 for old in parent['records']:
  if old['task']!='directed_relation':continue
  pos=old['provenance']['positions'];a,b=list(pos);pa,pb=pos[a],pos[b];base_question=old['messages'][-2]['content'];board=base_question.split('原书星行：\n',1)[1]
  for a,b,pa,pb in [(a,b,pa,pb),(b,a,pb,pa)]:
   key=(old['group'],board,a,b)
   if key in seen:continue
   seen.add(key);rel=d.relation(pa,pb);evidence=f'{a}在{d.PALACE[pa]}宫属{d.ELEMENT[pa]}，{b}在{d.PALACE[pb]}宫属{d.ELEMENT[pb]}。'
   common=dict(stage='application',split=old['split'],domain=old['domain'],group=old['group'],case_ids=old['case_ids'],format='chart_application')
   add(records,f'只读取原盘：{a}星和{b}星分别落在哪个宫？\n'+board,evidence,old['source_ids'],units,task='chart_positions',checks={'kind':'positions','positions':{a:pa,b:pb}},provenance={'kind':'literal_chart_reading','positions':{a:pa,b:pb}},**common)
   if old['split']=='train':
    add(records,f'已核对盘面：{evidence}\n只检查方向，{a}宫对{b}宫是什么关系？只写关系词。',rel+'。',old['source_ids'],units,task='verified_positions_direction',checks={'kind':'relation','answer':rel},provenance={'kind':'deduced_from_literal_chart_and_book_cycles','positions':{a:pa,b:pb}},**common)
   add(records,f'根据原盘判断{a}星所在宫对{b}星所在宫的五行关系，并给出两宫作为依据；不据此单独判断事情成败。\n'+board,rel+'。'+evidence,old['source_ids'],units,task='chart_direction',checks={'kind':'joint_relation','answer':rel,'positions':{a:pa,b:pb}},provenance={'kind':'literal_chart_and_exact_book_relation','positions':{a:pa,b:pb}},**common)
 # Rule rehearsal belongs in application too, with immutable source evidence.
 rehearsal=[deepcopy(r) for r in records if r['stage']=='foundation' and r['context']=='provided_evidence']
 for r in rehearsal:
  r['stage']='application';r['id']=r['id']+':application';records.append(r)
 # The old broad case summaries are NOT replayed. All original materials remain stored.
 for old in parent['records']:
  if old['task']=='case_evidence_summary':dispositions.append({'old_id':old['id'],'action':'quarantined_until_question_source_stage_alignment'})
  if old['task']!='author_evidence_to_conclusion':continue
  source=units[old['source_ids'][0]]
  # Identify the author step source explicitly; chart cell IDs may sort first.
  cid=old['provenance']['source_case_boundary'];case=next(c for c in parent['case_catalog'] if c['case_id']==cid);step=next(s for s in case['author_steps'] if s['step']==old['provenance']['author_step']);sid=step['source_id'];text=step['original']
  label,before_correction=step_phase(cid,units[sid],stages)
  if before_correction:
   dispositions.append({'old_id':old['id'],'action':'pre_correction_analysis_retained_in_full_case_not_standalone_teacher','source_id':sid});continue
  start,end=old['provenance']['target_chars'];literal=text[start:end].strip('，,：:；; \n')
  if not literal or d.compact(literal) not in d.compact(text):raise ValueError('Author conclusion not source-literal')
  q='阅读这段历史案例中'+label+'，说明本段提出的判断及其已经列明的依据。这是分段复盘，不把本段自动当作作者最终结论，也不补后来反馈。\n'+supplied('',[text])
  answer=cite('这段记录提出：'+old['answer'])
  add(records,q,answer,[sid],units,task='author_source_application',format='complete_case',stage='application',split=old['split'],domain=old['domain'],group=old['group'],case_ids=old['case_ids'],context='provided_evidence',provenance=old['provenance']|{'kind':'historical_author_source_reading','literal_target':literal,'citation_required':True,'source_phase':label,'source_stage_index':stages.get(cid,{})})
 chain_coverage=case_chains(parent,units,records,library,stages)
 unique={}
 for r in records:
  key=(r['stage'],r['split'],d.dumps(r['messages'][:-1]))
  if key in unique and unique[key]['answer']!=r['answer']:raise ValueError('Conflicting exact teaching prompts')
  unique.setdefault(key,r)
 records=list(unique.values())
 for old in parent['records']:
  if old['split']=='diagnostic':
   r=deepcopy(old);r.update(id='v53:diagnostic:'+old['id'],stage='application',context='closed_book');r['messages'][0]['content']=SYSTEM;records.append(r)
 result={'schema':SCHEMA,'source_books':parent['source_books'],'units':units,'records':records,'case_catalog':parent['case_catalog'],'original_case_splits':parent['original_case_splits'],'sealed_original_test_manifest':parent['sealed_original_test_manifest'],'dispositions':dispositions,'parent_pack_sha256':d.digest(d.dumps(parent)),'model_outputs_used_as_teachers':False,'reviewed_specs':manual,'case_chain_coverage':chain_coverage,'case_stage_index':stages}
 validate(result);result['summary']={'records':len(records),'stages':dict(Counter(r['stage'] for r in records)),'stage_splits':dict(Counter(r['stage']+'/'+r['split'] for r in records)),'tasks':dict(Counter(r['task'].split('reviewed_')[-1] if r['task'].startswith('reviewed_') else r['task'] for r in records)),'source_materials':len(units),'original_materials_preserved':len(parent['units']),'source_located_phase_excerpts_added':len(units)-len(parent['units']),'cases':len(result['case_catalog']),'author_steps':sum(len(c['author_steps']) for c in result['case_catalog']),'old_case_summaries_quarantined':sum(x['action']=='quarantined_until_question_source_stage_alignment' for x in dispositions),'pre_correction_bridges_quarantined':sum(x['action']=='pre_correction_analysis_retained_in_full_case_not_standalone_teacher' for x in dispositions),'test_used':False,'real_prediction_validated':False}
 return result

def validate(pack):
 d.validate(pack)
 if pack['schema']!=SCHEMA:raise ValueError('Wrong course schema')
 for r in pack['records']:
  if r['stage'] not in ('foundation','application'):raise ValueError('Unknown stage')
  if '本例核对：' in r['answer'] or re.fullmatch(r'\[图[^\]\n]*\]',r['answer'].strip()):raise ValueError('Audit note or figure caption used as target')
  if r['context']=='provided_evidence' and (not re.search(r'\[S\d+\]',r['messages'][-2]['content']) or not re.search(r'\[S\d+\]',r['answer'])):raise ValueError('Source citation boundary missing')
 return True
