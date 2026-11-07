"""V56 source-only repairs. Originals and exposed old scores stay immutable.

Reading a complete author paragraph and solving a local masked judgment are
different tasks. No source feedback is converted into a fresh prediction label.
"""
from copy import deepcopy
from collections import Counter
import re
from wengua_course import data as d
from wengua_course_v54 import data as validation
from wengua_course_v55 import data as prior, runtime as previous

SCHEMA = 'wengua_premise_complete_source_course_v56'
REPLAY = {}
SYSTEM = '按问题限定的领域、人物和来源回答。说明相关事实、必要条件及结论，使用正常中文，不重复段落。关系题先核对双方五行和作用方向，再给关系名称；不把星本身五行换成落宫五行。不补造出生年、盘面、日期或反馈，不使用人工吉凶分数。'

# These local conclusions were read against all 258 source targets. Only the
# listed literal judgment is hidden; the rest of its full original step stays
# visible, including conditions that used to be hidden inside a long target.
CLOZE = {
 '66883ebd1ee831755e2a72e3': ('原作者在本段怎样判断病位及阶段？只复盘该项。', 99, 109, '0b701d52dac5697139c61d1972c20cc903f5aefd35b8be76096f6a8b177cdc7b'),
 '2343bd07d40d862f0147c9a9': ('本段对医生治疗力量作了哪项判断？', 38, 46, '8228b7d6b3c2427424836fe27578cb6774aed00bfd537708e710f19b5655e0c9'),
 '392e372d3a6bab128461c39a': ('本段对心脏状况作何局部判断？不扩展其他病位。', 49, 53, 'fab4e0359262a43827fde0c1b35fd697da97b3274b1e3effeb4d3d826b347705'),
 'c0cd22b8d6a8a18a87ed5094': ('只问证据对女儿录取是否有利，按宫间关系补全本段判断。', 80, 90, '174d1a61c70b38f0d4f47603e5d2daed5cfaf1e44300eb78c975d69b89d6f097'),
 'decf1f3c6c23c3a1bb353003': ('天盘己土克地盘壬水，本段的主客作用方向是什么？', 34, 37, '5b736ecfe8b890e391a240390ffa49f569bf088aafaf9ec70409521dbaeeaff5'),
 'f3381234e2bdf409b6ad042e': ('本段只判断原告的调解意愿，不要求宣布胜诉方。', 42, 53, '2b5051b96a7a64d42b05504d846b236f50ffe8d914abb34ac97d6182665e6183'),
 '6e4aab7a157cac754e5beb77': ('本段惊门所代表的律师对哪一方有利？', 32, 40, 'dd219941cbb7d4845d73fb10fadaa0713156bb630db79fea5dc7ccde5ad056f4'),
 '541189cb5d2772791aaf3635': ('本段怎样区分失盗和本人忘记放置？', 63, 80, '5f507c866c55d3ea7404f8785b3104e2d553085a9c4dd8c22a6279c7c127bf59'),
 'cf8fd4cbae886acd46d99f81': ('白虎等凶象还受到本段哪些条件限制？补全孩子安全这项判断。', 105, 116, '4e34f41cb5ab457b656645369a2fe6015d94e0d89e5e115e75576e2e6392641c'),
 '5cf4cc7b3b6e82502afbcc7d': ('只补全原作者的相对破案时间，不换成未给出的公历日期。', 30, 38, '5e593afdcd43f56fad5db4c64c3b3ceac53be216e86fcb2e682b0c96daf6f3d1'),
 '53892b34f96fffc55525b039': ('本段是在讨论兄弟姐妹可能损伤多少，不是重报兄弟姐妹总数。', 34, 40, '5791186983e5e7032725e87013e8c748c503c71d84bff0943d205e73af658489'),
 'aeda37339bbc8c4017c0005d': ('只问一旦成婚后的夫妻关系，保留成婚这个前提。', 81, 94, '4f84ec8d2b2c93557703bac716279d94d40f22fb3ed32bedc3dab9632ee2f932'),
 '5b2e7efb0cdb185b3584ceca': ('本段六合对应哪项辅助判断？不扩大为所有事情均顺利。', 79, 83, '96db36104008ddda8f4ede26a7bce9c1a2c8c24cc4cf0bd2945055f713706604'),
 '76a7641ea4c8d50fff90ff04': ('天芮星生宫为旺，本段据此对天时怎样判断？', 31, 34, '54162dd90858875db74fc98ab9d482df5557f586bd40e0f6ac7a8b4195358d4a'),
 '1d38ab81dae5fa6274c57b04': ('只问本段当下处境，不替换为最终大选结果。', 141, 152, 'de7e5cdf4dd2001bcc78958736292fb70eca7244c4533d9bced8921e5669964e'),
 'e66b1edeb3396fa5b070ff45': ('年命落宫为死地，本段对当下地利作何判断？', 50, 56, 'b22467bc7f8e35b3f9a128ec70c40fe850629be33e0ba9d3a4da73f2e55d66b6'),
 '94fca59282bb679420b16efd': ('本段怎样判断申办手续本身？不问能否最终出国。', 66, 77, '1f9ecf2171a142a83dee077c29c06b87d54508b34035a346ca4a75d33d848072'),
 '5eff383c0e7863981dfc99f3': ('本段两颗雨星的条件对降雨判断有何作用？', 56, 60, '8f327e82b6ba2099876829a714fe725af5c214da48ed6b68452a7910d239dd5e'),
 '2174bafee6b7ebb9e7099be7': ('只补全本段女方对男方的态度。', 107, 116, '6d7b713700063983a386aa3e0ca59f81049374cc093f1af58bb7b89d346e304a'),
 'c00f12528e599fd2a0a3e0d3': ('本段是被告想支付的数额，不是法院最后判额。', 60, 71, '878205b4c4ac3bf52f411002901696a76dead49b1c636c3842246d7929cdc14b'),
 '0ad9d357905a9298c25ee286': ('本段问出租方想要的租金，不等于实际成交额。', 7, 18, '8a37d3dbf60562578ab85d6c07cdfcf26a75e1d37a261f46de26f06e2c5ac50d'),
 '3d56d5c58b4e87a747b8f6d7': ('本段对这件事的后果怎样作局部判断？', 109, 113, '80756c6f5e1dd710db00afe6bcab425bfdfc92f3713ad19786add24be1dd914b'),
 '7e7f4e3dc21bdceb59a6b161': ('仅复盘本段录取去向的判断，保留原作者范围。', 41, 54, 'd759b900a355ee4668d26bff01b127a309a06b7d2c0170175e0a61720a6f7a9b'),
 'f183b4af15aeba5804d0fb5e': ('本段怎样判断调动工作？不引入其他阶段反馈。', 57, 65, 'c937da7cf446f488d415e786923bef26ee8ee4a94617aad781f79a7912a02f59'),
}


def dedupe_paragraphs(text):
    """Remove exact repeated paragraphs only, preserving unique conditions."""
    seen=set();result=[]
    for paragraph in re.split(r'\n+', text.strip()):
        key=re.sub(r'^\s*\d+[、.]', '', paragraph).strip()
        if not key or key in seen:continue
        seen.add(key);result.append(paragraph.strip())
    return '\n'.join(result)


def set_answer(row, answer):
    row['answer']=answer;row['messages'][-1]['content']=answer


def complete_sentence_span(text,start,end):
    """Expand a source selector to grammatical sentence boundaries, never cut
    a dangling conjunction, opening parenthesis or causal premise as a target.
    No original character is rewritten.
    """
    left=max(text.rfind(mark,0,start) for mark in ['。','！','？','\n'])+1
    if end and text[end-1] in '。！？':right=end
    else:
        ends=[text.find(mark,end) for mark in ['。','！','？']]
        right=min((x+1 for x in ends if x>=0),default=len(text))
    return left,right


def new_row(question, answer, support, units, *, skill, split='train', domain='foundation', checks=None, tag='', case_ids=(), group=None, provenance=None):
    q=question+'\n\n本题来源：\n'+'\n\n'.join('[S'+str(i+1)+'] '+s['quote'] for i,s in enumerate(support))
    target=answer.rstrip()+' '+'、'.join('[S'+str(i+1)+']' for i in range(len(support)))
    row=prior.add([],q,target,support,units,skill=skill,split=split,domain=domain,checks=checks,case_ids=case_ids,group=group,provenance=provenance or {})
    if skill in ['author_reading','author_context','author_local']:row['format']='complete_case'
    row['messages'][0]['content']=SYSTEM;row['id']='v56:'+d.digest(d.dumps([tag,split,row['messages']]))[:24]
    row['provenance'].update(required_citations=['S'+str(i+1) for i in range(len(support))],teacher_is_model_output=False)
    row['provenance']['support']=deepcopy(support)
    return row


def chain(a,b,sa='前者',sb='后者'):
    if a==b:return f'{sa}属{a}，{sb}也属{b}；两者五行相同。结论：比和。'
    representative={'水':1,'土':2,'木':3,'金':6,'火':9};rel=d.relation(representative[a],representative[b])
    op='生' if rel in ['生','受生'] else '克';x,y=(a,b) if rel in ['生','克'] else (b,a);source,target=(sa,sb) if rel in ['生','克'] else (sb,sa)
    return f'{sa}属{a}，{sb}属{b}；{x}{op}{y}，是{source}{op}{target}。结论：{rel}。'


def compile_data(parent):
    units=parent['units'];rows=[];dispositions=[];cases={c['case_id']:c for c in parent['case_catalog']}
    cloze_rows=[]
    for old in parent['records']:
        row=deepcopy(old);row['id']='v56:retained:'+old['id'];row['messages'][0]['content']=SYSTEM
        row['provenance']['v55_source_core_id']=old['id']
        if row['provenance'].get('actual_production_rag_prompt'):
            row['provenance'].update(actual_production_rag_prompt=False,inherited_actual_rag_prompt_with_revised_system=True)
        if row['provenance'].get('key') in ['cross_domain_liuhe','health_daughter_father','health_father_self','exam_self','exam_proxy','marriage_same_palace']:
            row['provenance']['repair_pool']=True
        if old['skill']=='author_evidence':
            p=old['provenance'];case=cases[old['case_ids'][0]];step=next(s for s in case['author_steps'] if s['step']==p['author_step']);text=step['original'];target=p['literal_target'].strip('，。；,; ')
            if d.digest(text)!=p['source_step_sha256']:raise ValueError('Author source step changed')
            js,je=complete_sentence_span(text,*p['target_chars']);complete_judgment=text[js:je].strip()
            es,ee=complete_sentence_span(text,min(p['evidence_chars'][0],js),max(p['evidence_chars'][1],je))
            q=f'原案局部阅读：本题只复盘这条完整原句“{complete_judgment}”。请解释其中的条件、规则应用及局部含义，不扩展到整案其他结论。下面提供同一分析阶段的完整原段。'
            evidence=text[slice(*p['evidence_chars'])].strip();support=[{'source_id':step['source_id'],'quote':text}]
            answer=text[es:ee].strip()
            row=new_row(q,answer,support,units,skill='author_reading',split=old['split'],domain=old['domain'],case_ids=old['case_ids'],group=old['group'],provenance={'kind':'full_original_step_local_reading','source_core_id':old['id'],'source_case_id':case['case_id'],'author_step':step['step'],'source_step_sha256':d.digest(text),'literal_target':p['literal_target'],'full_step_in_input':True,'not_independent_prediction':True,'source_phase':p['source_phase']})
            dispositions.append({'id':old['id'],'action':'restored_complete_original_stage_as_explicit_reading','all_original_facts_visible':True})
            row['provenance'].update(complete_judgment_span=[js,je],complete_answer_span=[es,ee],answer_is_complete_original_causal_passage=True)
            if target.endswith(('：',':')):
                # A speaker introduction is not a judgment. Keep its evidence
                # as an explicitly different reading task, without inventing
                # the speech that the source step does not include.
                q='原段只列出证据和讲话提示，未给完整后续断语。请说明本段已提供的证据，不补后续结果。'
                stop=text.rfind('。')+1;body=text[:stop].strip() if stop else evidence
                row=new_row(q,body,support,units,skill='author_context',split=old['split'],domain=old['domain'],case_ids=old['case_ids'],group=old['group'],provenance=row['provenance']|{'kind':'complete_author_context_no_result_in_source_step','complete_answer_span':[0,stop] if stop else p['evidence_chars'],'context_only_not_a_conclusion_teacher':True,'answer_is_complete_original_causal_passage':False})
                dispositions.append({'id':old['id'],'action':'speaker_heading_not_used_as_a_judgment','source_retained':True})
            key=old['id'].split(':')[-1]
            if key in CLOZE:
                question,start,end,judgment_sha=CLOZE[key]
                judgment=text[start:end]
                if d.digest(judgment)!=judgment_sha:raise ValueError('Reviewed judgment selector changed')
                if judgment not in text:raise ValueError('Reviewed local judgment not literal: '+key)
                # All factual clauses outside this specifically reviewed judgment
                # survive, including facts formerly hidden in a long target.
                masked=text.replace(judgment,'〔待补局部判断〕')
                masked=re.sub(r'^.*?为什么[^\n]*[？?]\s*','',masked,count=1)
                if judgment in masked:raise ValueError('Masked judgment still visible')
                cq=question+'\n原作者本阶段分析（只遮去本题局部判断）：\n'+masked
                r=new_row(cq,judgment+'。\n依据：'+evidence,support,units,skill='author_local',split=old['split'],domain=old['domain'],case_ids=old['case_ids'],group=old['group'],provenance={'kind':'reviewed_local_judgment_cloze','source_core_id':old['id'],'source_case_id':case['case_id'],'source_step_sha256':d.digest(text),'author_step':step['step'],'judgment':judgment,'masked_original_step':masked,'not_independent_prediction':True,'source_phase':p['source_phase'],'full_factual_context_restored':True})
                # The complete unmasked paragraph is a reading source, but must
                # not also appear as a second copy in the cloze user prompt.
                r['messages'][-2]['content']=cq+'\n来源坐标：'+step['source_id'];r['question']=r['messages'][-2]['content'];r['provenance']['required_citations']=[]
                set_answer(r,judgment+'。\n依据：'+evidence)
                if judgment in evidence:
                    dispositions.append({'id':old['id'],'action':'local_cloze_not_used_judgment_in_reason'})
                else:cloze_rows.append(r)
        else:
            answer=dedupe_paragraphs(old['answer'])
            if answer!=old['answer']:dispositions.append({'id':old['id'],'action':'exact_duplicate_paragraphs_removed','source_originals_unchanged':True})
            set_answer(row,answer)
            check=row.get('checks') or {}
            if old['skill'] in ['direction','given_direction','chart_direction'] and check.get('elements'):
                a,b=check['elements'];symbols=check.get('symbols') or ['前者','后者']
                body=chain(a,b,*symbols)
                if check.get('positions'):
                    body='；'.join(f'{s}星在{d.PALACE[n]}宫' for s,n in check['positions'].items())+'。'+body
                citations='、'.join(re.findall(r'\[S\d+\]',old['answer']))
                set_answer(row,body+(' '+citations if citations else ''))
                row['checks']=check|{'kind':'reasoning_chain','require_positions':bool(check.get('positions'))}
                row['provenance']['teacher_uses_fact_law_conclusion_order']=True
            if old['skill']=='workflow':
                # Keep source provenance and real retrieval prompts; compact only
                # the answer's duplicated source paragraphs.
                row['provenance']['deduplicated_source_group_answer']=True
                n=0;parts=[]
                for line in row['answer'].splitlines():
                    if re.match(r'^\d+[、.]',line):
                        n+=1;line=re.sub(r'^\d+[、.]',str(n)+'、',line)
                    parts.append(line)
                set_answer(row,'\n'.join(parts))
        rows.append(row)
    rows.extend(cloze_rows)
    # Explicit pair and direction exercises use only source property/cycle facts.
    from wengua_course_v54 import corpus
    from wengua_course_v53 import runtime as source_runtime
    facts=corpus.properties(source_runtime.verify_data(),corpus.source_library())
    cycle=facts['rule:generation']['support']+facts['rule:control']['support']
    exercise=[]
    for a in ['水','土','木','金','火']:
        for b in ['水','土','木','金','火']:
            representative={'水':1,'土':2,'木':3,'金':6,'火':9};rel=d.relation(representative[a],representative[b]);wrong=next(x for x in ['生','克','受生','受克','比和'] if x!=rel)
            questions=[f'前者属{a}、后者属{b}。逐步说明前者对后者的五行关系。',f'对象A属{a}，对象B属{b}。请核对谁作用于谁，再给A对B的关系。',f'有人把属{a}的前者对属{b}的后者写成“{wrong}”，请指出并改正。',f'已知A五行为{a}、B五行为{b}，请以A为受观察对象解释方向。',f'比较双方：甲对象属{a}，乙对象属{b}。甲乙只是对象标签。请说明甲对象对乙对象的关系。',f'判定关系之前先确认：第一对象为{a}，第二对象为{b}。从第一对象看第二对象，如何推导？']
            for i,q in enumerate(questions+['请重新推导：以'+a+'方为观察主体，面对'+b+'方，哪一方生或克哪一方？最终关系词是什么？']):
                sa,sb=('对象A','对象B') if i in [1,3] else ('甲对象','乙对象') if i==4 else ('前者','后者')
                exercise.append(new_row(q,chain(a,b,sa,sb),cycle,units,skill='direction',split='validation' if i==6 else 'train',checks={'kind':'reasoning_chain','answer':rel,'elements':[a,b],'symbols':[sa,sb]},tag=str(i),provenance={'kind':'book_cycle_exercise','synthetic_not_observed_event':True,'elements':[a,b],'repair_pool':True}))
    for pa in [1,2,3,4,6,7,8,9]:
        for pb in [1,2,3,4,6,7,8,9]:
            a,b=d.ELEMENT[pa],d.ELEMENT[pb];support=facts['palace:'+str(pa)]['support']+facts['palace:'+str(pb)]['support']+cycle
            support=list({s['source_id']:s for s in support}.values());rel=d.relation(pa,pb)
            for i in range(4):
                q=[f'A宫在{d.PALACE[pa]}宫，B宫在{d.PALACE[pb]}宫。这里判断落宫五行，不是某颗星自身五行。A宫对B宫是什么关系？',f'已经给定两宫：前宫为{d.PALACE[pa]}，后宫为{d.PALACE[pb]}。请按宫位五行推导前宫对后宫的方向。',f'规则练习：假设某用神在{d.PALACE[pa]}宫，另一用神在{d.PALACE[pb]}宫。只求前一宫对后一宫的五行关系，不断事情吉凶。',f'换一种问法：观察{d.PALACE[pa]}宫这一方，它面对{d.PALACE[pb]}宫，是施生、受生、施克、受克还是比和？请写推导。'][i]
                exercise.append(new_row(q,chain(a,b,'A宫','B宫'),support,units,skill='given_direction',split='validation' if i==3 else 'train',checks={'kind':'reasoning_chain','answer':rel,'elements':[a,b],'symbols':['A宫','B宫']},tag=str(i),provenance={'kind':'source_palace_rule_exercise','synthetic_not_observed_event':True,'repair_pool':True}))
    rows.extend(exercise)
    rows.extend(condition_contrasts(parent,units))
    # No row silently disappears. Any later context exception is recorded by the
    # build/token audit and excluded explicitly, while source material survives.
    for r in rows:
        r['messages'][0]['content']=SYSTEM
    result={'schema':SCHEMA,'records':rows,'dispositions':dispositions,'repair_extra_ids':[r['id'] for r in rows if r['provenance'].get('repair_pool')],
            'legacy_v55_prompts':previous.choose_development(parent,192),'original_final_test_used':False,
            'old_manual_scores_used':False,'model_outputs_as_teachers':False}
    return result


def condition_contrasts(parent,units):
    specs={s['key']:s for s in parent['reviewed_specs']};rows=[]
    pairs=[
      ('exam_self','education','考生本人问考试，出生年未知','取日干代表考生本人；出生年未知，暂不取年命。','日干'),
      ('exam_proxy','education','母亲代问女儿考试，出生年未知','取时干代表女儿这位考生；出生年未知，暂不取年命。','时干'),
      ('health_daughter_father','health','女儿代问父亲疾病，没有父亲出生年','取起局年干代表父亲，兼看乾宫；暂不取父亲年命。','年干'),
      ('health_father_self','health','父亲本人问自己的健康，没有出生年','取日干代表患者本人；暂不取年命。','日干'),
      ('health_sibling','health','代问同辈兄弟疾病，长幼及出生年未知','取月干代表同辈患者；暂不取年命，长幼宫待相应资料明确。','月干'),
      ('health_parent_child','health','父母代问子女健康，性别长幼与出生年未知','取时干代表子女患者；暂不取年命，兼用宫待性别长幼资料明确。','时干'),
    ]
    for key,domain,condition,answer,role in pairs:
        support=deepcopy(specs[key]['support'])
        for i,prefix in enumerate(['条件题：','请按现在的人物关系分析：','先分清问事人与被问者：','请只回答患者或考生身份及未知年命的处理：']):
            q=prefix+condition+'。主用神取谁？未知条件怎样处理？'
            rows.append(new_row(q,answer,support,units,skill='identity',split='validation' if i==3 else 'train',domain=domain,checks={'kind':'role_boundary','answer':role,'unknown_year':True},tag=str(i),provenance={'kind':'condition_contrast_identity','repair_pool':True,'birth_year_unknown_in_question':True}))
    # The two sides are kept together in training, with different held-out
    # wording. They are logical teaching contrasts, not extra observed cases.
    rules={r['id']:r for f in parent['workflow_plan'] for r in f['rules']}
    definitions=[
      ('rag:44f6cefbc261d9d5cbb52b8e','employment','杜门生日干','领导有意留用，不准退役','留用','准退'),
      ('rag:44f6cefbc261d9d5cbb52b8e','employment','杜门克日干','准予退役','准退','不准'),
      ('rag:312e1faa0b3ecc6bb442c49d','business_profit','站在银行放贷人立场，值符克天乙','按本条对放贷人吉','吉','凶'),
      ('rag:312e1faa0b3ecc6bb442c49d','business_profit','站在银行放贷人立场，天乙克值符','按本条对放贷人凶','凶','吉'),
      ('rag:ff1d59b30fe9012908ac4b6b','litigation','值符宫生天乙宫','原告可能主动求和','原告','被告'),
      ('rag:ff1d59b30fe9012908ac4b6b','litigation','天乙宫生值符宫','被告可能主动求和','被告','原告'),
      ('rag:3a4ba64659bdcdc6de2027a0','business_profit','卖货专题以日干为卖方，时干为货物，日干生时干','人恋货，不肯卖','不肯卖','卖得快'),
      ('rag:3a4ba64659bdcdc6de2027a0','business_profit','卖货专题以日干为卖方，时干为货物，时干克日干','卖得快','卖得快','不肯卖'),
      ('rag:d6ee057cb5c1247772a08bfc','lost_property','只确认失物时干临玄武','可能被偷，也可能自己忘记；凭这一项不能确认失盗','忘记','一定被偷'),
      ('rag:d6ee057cb5c1247772a08bfc','lost_property','又核实时干宫被玄武宫或天蓬宫所克','本条增加了可能被盗的依据，仍应保留可能这一判断范围','可能被盗','必定被盗'),
    ]
    for rid,domain,premise,conclusion,required,forbidden in definitions:
        rule=rules[rid];support=[]
        for part in rule['parts']:
            sid='v55:literal:'+d.digest(d.dumps(part))[:24]
            support.append({'source_id':sid,'quote':units[sid]['original_quote']})
        for i,prefix in enumerate(['只按本条已给条件：','条件对照：','请不要套用别的专题：','换条件后重新判断：']):
            q=prefix+premise+'。'+('在退役专题中怎样判断？' if domain=='employment' else '本条具体怎样判断？')
            alternatives=[required,'准予退役','允许退役'] if required=='准退' else [required]
            rows.append(new_row(q,conclusion+'。适用前提是'+premise+'。',support,units,skill='condition',split='validation' if i==3 else 'train',domain=domain,checks={'kind':'condition_slots','required':[alternatives],'forbidden':[forbidden]},tag=str(i),provenance={'kind':'reviewed_rule_premise_contrast','rule_id':rid,'repair_pool':True,'hypothetical_condition_not_observed_outcome':True}))
    return rows


def compose(parent,delta):
    result=deepcopy(parent);result.update(delta);return result


def validate(pack):
    validation.validate(pack|{'schema':validation.SCHEMA})
    if pack['schema']!=SCHEMA or pack.get('old_manual_scores_used') or pack.get('model_outputs_as_teachers'):
        raise ValueError('V56 source-only course required')
    for row in pack['records']:
        if dedupe_paragraphs(row['answer'])!=row['answer'] and row['skill']=='workflow':raise ValueError('Repeated workflow teacher paragraph')
        if row['skill'] in ['author_reading','author_context']:
            p=row['provenance']
            source=next(s['original'] for c in pack['case_catalog'] if c['case_id']==p['source_case_id'] for s in c['author_steps'] if s['step']==p['author_step'])
            if source not in row['messages'][-2]['content']:raise ValueError('Complete author premise lost')
            if row['answer']!=source[slice(*p['complete_answer_span'])].strip()+' [S1]':
                raise ValueError('Author answer no longer matches the complete literal source selector')
        if row['skill']=='author_local':
            p=row['provenance']
            if p['judgment'] in row['messages'][-2]['content']:raise ValueError('Local judgment leaked')
            source=next(s['original'] for c in pack['case_catalog'] if c['case_id']==p['source_case_id'] for s in c['author_steps'] if s['step']==p['author_step'])
            masked=re.sub(r'^.*?为什么[^\n]*[？?]\s*','',source.replace(p['judgment'],'〔待补局部判断〕'),count=1)
            if masked!=p['masked_original_step'] or masked not in row['messages'][-2]['content']:
                raise ValueError('Local factual context changed beyond the reviewed judgment mask')
    return True
