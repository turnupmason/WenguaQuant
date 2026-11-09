"""Source-only evidence application; source bytes and original families freeze."""
from copy import deepcopy
from collections import Counter
import re
from wengua_course import data as d
from wengua_course_v58 import data as base
from wengua_course_v57 import data as source
SCHEMA='wengua_evidence_application_course_v59'
SYSTEM=base.SYSTEM.replace('先回答所问结论，再说明必要事实、条件和原文依据；','按题目指定顺序回答；未指定时先给所问结论，再说明必要事实、条件和原文依据；')+'已给作者分析证据的局部推断题，应回答局部判断并解释依据，不把待答位置当答案；历史复盘分清当时分析与后来反馈。先核对两个对象及作用方向，再让结论和理由一致。'
READING=base.READING
clean=base.clean
compose=base.compose
active=base.active


def dedupe_answer(text):
    """Exact repeated sentences in rule-group teachers only, not source text."""
    parts=re.split(r'(?<=[。！？\n])',text);seen=set();out=[];removed=[]
    for part in parts:
        key=re.sub(r'\[S\d+\]|\s+','',part)
        key=re.sub(r'^(?:(?:\d+[、.．]|[（(]\d+[）)])\s*)+','',key).strip('。！？')
        if len(key)>=20 and key in seen:removed.append(part);continue
        seen.add(key);out.append(part)
    result=''.join(out)
    result=re.sub(r'(?m)^\s*\d+、\s*(?=\[S|$)','',result)
    index=iter(range(1,1000))
    result=re.sub(r'(?m)^\d{1,2}、',lambda m:str(next(index))+'、',result)
    return result.strip(),removed


def normalize_person_contract(c,question):
    c=deepcopy(c);question=question.split('原文依据：',1)[0].split('原书证据：',1)[0].split('本题来源：',1)[0];required=str(c.get('required',[]))
    if not c.get('helper'):
        for name in ['乾','坤']:
            if name in required:c['helper']=name+'宫'
    if not c.get('known_year_stem') and not c.get('unknown_year'):
        m=re.search(r'(?:出生年干|出生年|年命)(?:为|是|已给为)?([甲乙丙丁戊己庚辛壬癸])',question)
        if m:c['known_year_stem']=m[1]
    if not c.get('cast_year_stem'):
        m=re.search(r'(?:本次起局|起局|本局)年干(?:为|是|已给为)?([甲乙丙丁戊己庚辛壬癸])',question)
        if m:c['cast_year_stem']=m[1]
    if not c.get('known_year_stem') and re.search(r'出生年[^。；]{0,6}(?:未知|未提供|没给)|(?:不知|不知道|未知)[^。；]{0,8}出生年',question):c['unknown_year']=True
    return c


def person_answer(c):
    role=c['answer'];birth=c.get('known_year_stem');cast=c.get('cast_year_stem')
    answer={'年干':'代问长辈，主用神取本次起局年干'+(cast or ''),'日干':'本人求测，主用神取日干','时干':'父母代问子女考试，主用神取时干','月干':'主用神取月干'}[role]
    if c.get('helper'):answer+='，兼看'+c['helper']
    answer+='。'
    if birth:answer+='已知出生年干'+birth+'，另兼看出生年命'+birth+'；出生年命与本次起局年干分开使用。'
    elif c.get('unknown_year') and c.get('require_year_boundary') is not False:answer+='出生年未知，暂不取年命。'
    if cast and role!='年干':answer+='起局年干'+cast+'与'+role+'是不同类别，不能据此补造'+role+'的具体字。'
    elif role=='年干' and not cast:answer+='这里可确定年干这一类别，具体天干尚未给出，不能用出生年干替填。'
    return answer


def row(q,a,refs,u,**kw):
    r=source.row(q,a,refs,u,**kw);r['messages'][0]['content']=SYSTEM;r['id']='v59:'+d.digest(d.dumps([r['split'],r['messages']]))[:24]
    r['provenance'].update(source_contract_version='v59',repair_priority=True)
    return r


def anchors(text):
    """Visible literal concepts; coverage is a diagnostic, not semantic proof."""
    named=re.findall(r'天[蓬任冲辅英芮柱心禽]|年命[甲乙丙丁戊己庚辛壬癸]|日干|时干|年干|月干|开门|休门|生门|伤门|杜门|景门|死门|惊门|白虎|玄武|六合|值符|天乙|庚格|月格|日格|年格|时格|庚下临丁|空亡|入墓|伏吟|反吟|旺相|休囚|相生|相克|不能克|生被告|克宫|[金木水火土][生克][金木水火土]|[一二三四五六七八九1-9]宫',text)
    return list(dict.fromkeys(named))[:5]


def trim_evidence(text):
    text=text.strip('。；，：: \n')
    pattern=r'(?:^|(?<=[，；。,:;：\s]))(?:(?:据此|故断|故而|所以断|所以|说明|表明|可见|因此|故)(?:我断|而断|而|说|断)?|我断|而断)[，、\s]*$'
    while True:
        result=re.sub(pattern,'',text).rstrip('。；，：: ')
        if result==text:return result
        text=result


def inference(original,judgment,template,units,*,kind):
    """Keep all evidence outside the one explicitly selected source judgment."""
    if judgment not in original or len(judgment)<3:raise ValueError('Judgment must be literal source')
    if original.count(judgment)!=1:return None
    if kind=='literal_connective_author_inference' and (re.search(r'(?:落|临|在)[^。；，]{0,6}(?:[坎坤震巽乾兑艮离][1-9一二三四五六七八九]?|[1-9一二三四五六七八九])宫',judgment) or any(x in judgment for x in ['上乘','下临','月令','太岁','出生'])):return None
    before,after=original.split(judgment,1);sid=template['source_ids'][0]
    if any(year not in before+after for year in re.findall(r'\d{4}年',judgment)):return None
    refs=[{'source_id':sid,'quote':t} for t in [before,after] if t.strip('。；， \n')]
    if not refs:return None
    pieces=[]
    for i,s in enumerate(refs):
        piece=s['quote'].strip('。；，：: \n')
        if i==0:piece=trim_evidence(piece)
        if piece:pieces.append(piece)
    evidence='；'.join(pieces)
    facts=anchors(evidence)
    if not facts:return None
    answer='按原作者本段，'+judgment.strip('。；，')+'。\n依据：'+evidence+'。'
    q='本题领域：'+d.DOMAIN_ZH.get(template['domain'],template['domain'])+'。原文在[S1]之后省去了一项局部判断，只问紧接这段证据的判断及依据；后续摘录如有，作为同段补充条件。请回答这一项，不补其他阶段或后来结局。'
    r=row(q,answer,refs,units,skill='author_local',split=template['split'],domain=template['domain'],case_ids=template['case_ids'],group=template['group'],check={'kind':'source_inference','judgment':judgment.strip('。；，'),'evidence_terms':facts,'minimum_evidence_terms':min(2,len(facts))},provenance={'kind':kind,'source_core_id':template['id'],'source_case_id':template['case_ids'][0],'author_step':template['provenance']['author_step'],'source_step_sha256':d.digest(original),'judgment':judgment,'evidence_before':before,'evidence_after':after,'not_independent_prediction':True,'explicit_author_inference_not_new_outcome':True,'answer_goal':'judgment_plus_source_evidence'})
    return r


def enrich_case(answer,question):
    # Restore the full cited analysis behind selected conclusions. Original
    # phase labels and feedback lines stay in their own original positions.
    refs={m[1]:m[2] for m in re.finditer(r'(?ms)^\[(S\d+)\]\s*(.*?)(?=^\[S\d+\]|\Z)',question)}
    lines=answer.splitlines();changed=[]
    for i,line in enumerate(lines):
        if not line.startswith('依据：'):continue
        cites=re.findall(r'\[(S\d+)\]',line)
        if len(cites)!=1 or cites[0] not in refs:continue
        text=refs[cites[0]]
        if not re.match(r'.*?分析第\d+步，盘次[^：]+：',text):continue
        body=re.sub(r'^.*?分析第\d+步，盘次[^：]+：','',text,count=1).strip()
        if body not in line:lines[i]='依据：'+body+' ['+cites[0]+']';changed.append(cites[0])
    return '\n'.join(lines),changed



def rewrite_person_turns(r):
    p=r['provenance'];c=r.get('checks') or {}
    if c.get('kind') in ['person_contract','role_boundary','birth_boundary','role'] and c.get('answer') in ['日干','年干','时干','月干']:
        c=normalize_person_contract(c,source.question(r));r['checks']=c
    contracts=p.get('turn_contracts')
    if not contracts and p.get('role_transition'):contracts=[{'kind':'person_contract','answer':p['role_transition']['before']},c]
    assistant=[m for m in r['messages'] if m['role']=='assistant']
    if contracts and len(contracts)!=len(assistant):raise ValueError('Original assistant-turn contract count differs')
    normalized_contracts=[]
    for turn,(m,ct) in enumerate(zip(assistant,contracts or [c]*len(assistant))):
        if ct.get('kind') in ['person_contract','role_boundary','birth_boundary','role'] and ct.get('answer') in ['日干','年干','时干','月干']:
            ct=normalize_person_contract(ct,source.question(r|{'messages':r['messages'][:2*turn+3]}))
            cites=re.findall(r'\[S\d+\]',m['content']);m['content']=person_answer(ct)+' '+'、'.join(dict.fromkeys(cites))
        normalized_contracts.append(ct)
    if contracts:p['turn_contracts']=normalized_contracts
    if normalized_contracts and normalized_contracts[-1]:r['checks']=normalized_contracts[-1]
    r['answer']=assistant[-1]['content']


def build(parent):
    records=[];mapping={};changes=[];u=parent['units'];inference_keys=set()
    for old in parent['records']:
        r=deepcopy(old);r['id']='v59:retained:'+old['id'];mapping[old['id']]=r['id'];r['messages'][0]['content']=SYSTEM
        p=r['provenance'];p.update(previous_v58_id=old['id'],actual_production_rag_prompt=False)
        before=r['answer'];rewrite_person_turns(r)
        if r['skill'] in ['workflow','domain_rule']:
            r['answer'],removed=dedupe_answer(r['answer'])
            if removed:p['removed_repeated_sentences']=removed
        if r['skill']=='case':r['answer'],p['restored_analysis_citations']=enrich_case(r['answer'],r['messages'][-2]['content'])
        if r['skill']=='author_local':
            original=next(s['original'] for ca in parent['case_catalog'] if ca['case_id']==p['source_case_id'] for s in ca['author_steps'] if s['step']==p['author_step'])
            new=inference(original,p['judgment'],r,u,kind='reviewed_source_inference')
            if new:
                r=new;mapping[old['id']]=r['id'];inference_keys.add((p['source_case_id'],p['author_step']))
        r['messages'][-1]['content']=r['answer'];r['question']=r['messages'][-2]['content']
        if before!=r['answer']:r['provenance']['v59_repaired_teacher']=True;changes.append({'id':r['id'],'old_id':old['id'],'skill':r['skill'],'old_answer_sha256':d.digest(before),'new_answer_sha256':d.digest(r['answer']),'source_unchanged':True,'restored_citations':p.get('restored_analysis_citations',[]),'removed_repeated_sentences':p.get('removed_repeated_sentences',[])})
        records.append(r)
    # New inference teachers come only from literal, complete authored steps.
    # Reject narrative feedback and statements depending on an unshown dialogue.
    excluded=[]
    for original in parent['records']:
        pr=original['provenance'];key=(pr.get('source_case_id'),pr.get('author_step'))
        if pr.get('kind')!='complete_original_author_step' or original['split'] not in ['train','validation'] or key in inference_keys:continue
        text=pr['support'][0]['quote']
        if re.search('反馈|告知|得知|告诉|果然|答复|更正|后来|实际|听说|据查',text):continue
        matches=list(re.finditer(r'[，；。,:;：]\s*(?:故断|故而|所以断|所以|说明|表明|可见|因此|据此|故)',text))
        if not matches:continue
        m=matches[-1];start=m.end();tail=re.split('[。！？]',text[start:].strip(' ：:，'),maxsplit=1)[0]
        tail=re.sub(r'^(?:我断[，,]?|而断|说[，,]?|而)','',tail).strip(' ，,：:')
        if re.match(r'^[了的地得着过)）]',tail) or tail.count('(')!=tail.count(')') or tail.count('（')!=tail.count('）'):continue
        if not 3<=len(tail)<=90 or len(text[:m.start()])<20 or len(text)>650 or '？' in tail:continue
        if any(v in tail for v in ['\n','“','”']):continue
        derived=inference(text,tail,original,u,kind='literal_connective_author_inference')
        if derived:
            records.append(derived);inference_keys.add(key)
    # Direct and reversed questions share explicitly provided facts, never
    # fabricated observed cases. Each answer states the applicable edge first.
    from wengua_course_v54 import corpus as literal
    from wengua_course_v53 import runtime as original_runtime
    facts=literal.properties(original_runtime.verify_data(),literal.source_library());cycle=source.property_bundle(facts,['rule:generation','rule:control'],u)
    for a in '金木水火土':
        for b in '金木水火土':
            pa={'水':1,'土':2,'木':3,'金':6,'火':9}[a];pb={'水':1,'土':2,'木':3,'金':6,'火':9}[b]
            for i,q in enumerate([f'甲方属{a}，乙方属{b}。先说清谁作用于谁，再给甲方对乙方的关系，不列其他五行的循环。',f'对象A是{a}、对象B是{b}。从A看B，哪一方是施力者？用法则和主客方向解释。',f'对象B属{b}，对象A属{a}。按指定的A对B方向，给出关系和原因。']):
                sa,sb=('甲方','乙方') if i==0 else ('对象A','对象B') if i==1 else ('对象A','对象B')
                from .facts import checked_prompt
                packet=checked_prompt(q,{'label':sa,'element':a},{'label':sb,'element':b})
                if packet['relation']!=d.relation(pa,pb):raise ValueError('Input-only relation computation differs')
                answer=source.relation_answer(a,b,sa,sb);parts=answer.split('。');answer='。'.join(parts[1:]).strip('。')+'。结论：'+d.relation(pa,pb)+'。'
                records.append(row(q,answer,cycle,u,skill='direction',split='validation' if i==2 else 'train',check={'kind':'relation_facts','answer':d.relation(pa,pb),'elements':[a,b],'symbols':[sa,sb]},provenance={'kind':'edge_then_label','input_only_derivation':packet,'synthetic_given_facts_not_outcome':True,'answer_goal':'consistent_ordered_relation'}))
    # Genuine condition changes, not paraphrases: each turn keeps its own
    # reviewed rule contract and the same source excerpts/family ownership.
    buckets={}
    for r in parent['records']:
        if r['skill']!='condition' or len(r['messages'])!=3 or (r.get('checks') or {}).get('kind')!='condition_slots':continue
        key=(r['split'],r['group'],d.dumps(r['provenance'].get('support',[])))
        buckets.setdefault(key,{})[d.dumps(r['checks'])]=r
    for bucket in buckets.values():
        choices=list(bucket.values())
        if len(choices)<2:continue
        for i,a in enumerate(choices):
            b=choices[(i+1)%len(choices)];refs=a['provenance']['support']
            if not refs:continue
            q1=source.question(a);q2='现在替换条件，上一问事实不沿用，只按本问判断：'+source.question(b)
            aa=re.sub(r'(?:\s|、)*\[S\d+\]','',a['answer']).strip();bb=b['answer']
            r=row(q1,aa,refs,u,skill='condition',split=a['split'],domain=a['domain'],group=a['group'],check=deepcopy(b['checks']),provenance={'kind':'same_rule_changed_conditions','turn_contracts':[deepcopy(a['checks']),deepcopy(b['checks'])],'source_teachers':[a['id'],b['id']],'hypothetical_premises_not_observed_outcomes':True,'answer_goal':'changed_conditions_changed_judgment'})
            r['messages'].extend([{'role':'user','content':q2},{'role':'assistant','content':bb}]);r.update(answer=bb,question=q2,id='v59:'+d.digest(d.dumps([r['split'],r['messages']]))[:24]);records.append(r)
    # The previous 80 questions keep their exact original messages/targets.
    # They are evaluation-only; corrected teaching views use the new system.
    prior_by={r['id']:r for r in parent['records']}
    for rid in parent['fixed_micro_ids']:
        old=deepcopy(prior_by[rid]);old['provenance']['v59_retention_only']=True;records.append(old)
    # Exact aliases are merged; source originals and all archived views remain.
    unique={};aliases=[]
    for r in records:
        key=d.dumps(r['messages'])
        if key in unique:
            prior=unique[key]
            if prior['split']!=r['split']:raise ValueError('Dialogue collision crosses original split')
            aliases.append({'id':r['id'],'canonical':prior['id']});continue
        unique[key]=r
    aliases_by={r['id']:r['canonical'] for r in aliases}
    mapping={k:aliases_by.get(v,v) for k,v in mapping.items()}
    author_dispositions=[]
    for ca in parent['case_catalog']:
        for step in ca['author_steps']:
            linked=[r for r in unique.values() if r['provenance'].get('source_case_id')==ca['case_id'] and r['provenance'].get('author_step')==step['step']]
            made=any((r.get('checks') or {}).get('kind')=='source_inference' for r in linked)
            action='bounded_source_inference_and_original_retained' if made else 'complete_source_reading_or_context_only' if linked else 'sealed_original_test' if ca.get('phase_split') not in ['train','validation'] else 'original_catalog_retained_no_safe_derived_task'
            author_dispositions.append({'case_id':ca['case_id'],'author_step':step['step'],'source_id':step['source_id'],'source_step_sha256':d.digest(step['original']),'original_phase_split':ca.get('phase_split'),'action':action,'linked_lesson_ids':[r['id'] for r in linked],'no_model_output_or_new_outcome_label':True})
    return {'schema':SCHEMA,'records':list(unique.values()),'v58_to_v59_ids':mapping,'old_fixed_micro_ids':parent['fixed_micro_ids'],'v59_retention_ids':parent['fixed_micro_ids'],'course_repairs':changes,'author_derivation_dispositions':author_dispositions,'exact_dialogue_aliases':aliases,'old_manual_scores_used':False,'model_outputs_as_teachers':False,'original_final_test_used':False}


def validate(pack):
    base.validate(pack|{'schema':base.SCHEMA})
    if pack['schema']!=SCHEMA:raise ValueError('V59 schema required')
    cases={c['case_id']:c for c in pack['case_catalog']}
    for r in pack['records']:
        p=r['provenance']
        if r['skill'] in ['workflow','domain_rule'] and dedupe_answer(r['answer'])[1]:raise ValueError('Internal repeated teacher sentence')
        if (r.get('checks') or {}).get('kind')=='source_inference':
            step=next(t['original'] for t in cases[p['source_case_id']]['author_steps'] if t['step']==p['author_step'])
            if d.digest(step)!=p['source_step_sha256'] or p['evidence_before']+p['judgment']+p['evidence_after']!=step:raise ValueError('Inference not bound to one original authored step')
            if p['judgment'] in r['messages'][-2]['content']:raise ValueError('Selected judgment leaked into inference prompt')
            if r['group']!=cases[p['source_case_id']]['group'] or r['split']!=cases[p['source_case_id']].get('phase_split',cases[p['source_case_id']]['split']):raise ValueError('Case family moved')
    return True
