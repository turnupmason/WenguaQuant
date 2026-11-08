"""V58 source-bound question goals, contrast groups, and explicit reading archive."""
from copy import deepcopy
import re
from wengua_course import data as d
from wengua_course_v57 import data as base
SCHEMA='wengua_question_contract_course_v58'
SYSTEM='按当前问题的领域、求测人和所问对象回答。先回答所问结论，再说明必要事实、条件和原文依据；多项问题逐项回答。区分起局年干与出生年命、宫位五行与宫内符号属性。结论必须与理由一致。原文已足够回答规则问题时直接解释；实际预测缺少必要资料时只追问缺的资料。正常中文，不照录整段无关原文，不编造具体日期。'
GUARDS=['本题是知识问答，没有本次盘面，不补造具体宫位或应期。\n','本题是知识问答，没有本次盘面，不补造具体宫位或应期。']
READING={'author_reading','author_context'}


def clean(text):
    for guard in GUARDS:text=text.replace(guard,'')
    return text


def active(r):return r['split']=='train' and r['skill'] not in READING


def compose(parent,delta):return base.compose(parent,delta)


def row(q,a,refs,u,**kw):
    r=base.row(q,a,refs,u,**kw);r['messages'][0]['content']=SYSTEM;r['id']='v58:'+d.digest(d.dumps([r['split'],r['messages']]))[:24]
    r['provenance'].update(source_contract_version='v58',answer_goal=kw.get('provenance',{}).get('answer_goal',r['skill']),repair_priority=True)
    return r


def build(parent):
    from wengua_course_v54 import corpus as literal
    from wengua_course_v53 import runtime as original
    u=parent['units'];facts=literal.properties(original.verify_data(),literal.source_library());specs={s['key']:s for s in parent['reviewed_specs']};records=[];old_to_new={}
    for old in parent['records']:
        r=deepcopy(old)
        for m in r['messages']:m['content']=clean(m['content'])
        r['messages'][0]['content']=SYSTEM;r.update(question=r['messages'][-2]['content'],id='v58:retained:'+old['id'])
        r['provenance'].update(previous_v57_id=old['id'],answer_goal=r['skill'],training_disposition='reading_archive_not_active_sft' if r['skill'] in READING else 'active_source_application')
        r['provenance']['actual_production_rag_prompt']=False
        if r['provenance'].get('required_source_fragments'):
            assert all(s['quote'] in r['messages'][-2]['content'] for s in r['provenance']['required_source_fragments'])
        records.append(r);old_to_new[old['id']]=r['id']
    cycle=base.property_bundle(facts,['rule:generation','rule:control'],u)
    for a in '金木水火土':
        for b in '金木水火土':
            pa={'水':1,'土':2,'木':3,'金':6,'火':9}[a];pb={'水':1,'土':2,'木':3,'金':6,'火':9}[b];rel=d.relation(pa,pb)
            questions=[f'以{a}方为主体，面对{b}方，它是生对方、受生、克对方、受克还是比和？说明这两个对象的关系。',f'只比较本题的{a}方与{b}方：第一方对第二方怎样作用？不要换到第三个五行。',f'A属{a}，B属{b}。请核对A对B的结论及依据是否一致，应怎样表述？',f'对象甲的五行是{a}，对象乙的五行是{b}。从甲看乙，请用自然中文解释。',f'比较方向固定为前者到后者：前者属{a}，后者属{b}。给出本题关系和一条适用法则。',f'若{a}与{b}分别是所问两方的属性，应如何从第一方判断第二方？']
            for i,q in enumerate(questions):records.append(row(q,base.relation_answer(a,b),cycle,u,skill='direction',split='train' if i<4 else 'validation',check={'kind':'relation_facts','answer':rel,'elements':[a,b],'symbols':['前者','后者']},provenance={'kind':'source_relation_style_contrast','contrast_group':'element:'+a+b,'answer_goal':'ordered_element_relation'}))
    for pa in [1,2,3,4,6,7,8,9]:
        for pb in [1,2,3,4,6,7,8,9]:
            refs=base.property_bundle(facts,['palace:'+str(pa),'palace:'+str(pb),'rule:generation','rule:control'],u);a,b=d.ELEMENT[pa],d.ELEMENT[pb];sa,sb=d.PALACE[pa]+'宫',d.PALACE[pb]+'宫'
            for i,q in enumerate([f'以{sa}为第一对象，以{sb}为第二对象。比较宫位本身的五行，第一宫对第二宫是什么关系？',f'现在问{sa}对{sb}的宫位关系。请分别列出两个宫的五行，再回答生克方向。',f'只问宫的属性：从{sa}看{sb}，哪一方生克哪一方？不要以宫内天干代替宫本身。']):
                records.append(row(q,base.relation_answer(a,b,sa,sb),refs,u,skill='given_direction',split='validation' if i==2 else 'train',check={'kind':'relation_facts','answer':d.relation(pa,pb),'elements':[a,b],'palaces':[pa,pb],'symbols':[sa,sb]},provenance={'kind':'palace_object_contrast','contrast_group':f'palace:{pa}:{pb}','required_source_fragments':refs,'answer_goal':'palace_not_occupant_element'}))
    # All definitions are given by the two books; concrete stems below are
    # explicit teaching premises, never inferred historical birth/outcome data.
    for who,key,role,helper in [('女儿代问父亲','health_daughter_father','年干','乾宫'),('父亲本人问自己的病','health_father_self','日干',None),('儿子代问母亲','health_daughter_father','年干','坤宫'),('母亲本人问自己的病','health_father_self','日干',None),('考生本人问考试','exam_self','日干',None),('母亲代问女儿考试','exam_proxy','时干',None)]:
        for birth in [None,'癸','甲']:
            refs=deepcopy(specs[key]['support'])+(deepcopy(specs['health_reliable_birth_stem']['support']) if birth else [])
            known=f'已可靠提供出生年干{birth}' if birth else '出生年未知'
            for cast in [None,'乙']:
                premise=who+'；'+known+('；本次起局年干已给为乙' if cast else '；本次起局年干的具体字未给出')
                answer=('代问长辈先取本局年干'+(cast if cast else '') if role=='年干' else '本人求测取日干' if role=='日干' else '父母代问子女考试取时干')+('，兼看'+helper if helper else '')+'。'
                answer+=f'另兼看已提供的出生年命{birth}，它与起局年干是不同资料。' if birth else '出生年未知，暂不取年命。'
                if not cast and role=='年干':answer+='这里可以确定用神类别为年干，不能把出生年干填成本局年干的具体字。'
                for i,prefix in enumerate(['请按这些已知条件回答：','资料分层核对：','请解释本题人物、时间资料与用神的对应：']):
                    records.append(row(prefix+premise+'。主用神、兼用宫及年命各怎样处理？',answer,refs,u,skill='identity',split='validation' if i==2 else 'train',domain='education' if '考试' in who else 'health',check={'kind':'person_contract','answer':role,'helper':helper,'unknown_year':not birth,'known_year_stem':birth,'cast_year_stem':cast},provenance={'kind':'cast_year_birth_year_contrast','contrast_group':who+str(birth)+str(cast),'answer_goal':'person_role_and_typed_year_facts','synthetic_given_stems_not_observed_outcomes':True}))
    mapping={'婚姻':('媒人','role_20'),'诉讼':('证人、证据','role_20'),'交易求财':('经纪人、中介人','role_20')}
    refs=specs['cross_domain_liuhe']['support']
    domains=list(mapping)
    for asked in [[x] for x in domains]+[domains,domains[::-1],[domains[1],domains[0]],[domains[2],domains[1]]]:
        a='；'.join(f'{x}中，六合为{mapping[x][0]}' for x in asked)+'。不同领域按各章身份解释。'
        for i,prefix in enumerate(['请逐项回答：','只回答实际列出的领域：','核对所问范围：']):
            records.append(row(prefix+'六合在'+ '、'.join(asked)+'中分别代表什么？',a,refs,u,skill='condition',split='validation' if i==2 else 'train',domain='mixed_domain',check={'kind':'domain_map'} if len(asked)==3 else {'kind':'literal_slots','required':[[x] for x in asked]+[[mapping[x][0].split('、')[0]] for x in asked]},provenance={'kind':'domain_scope_contrast','answer_goal':'all_and_only_requested_domains','contrast_group':'liuhe:'+','.join(asked)}))
    marriage=specs['marriage_same_palace']['support']
    questions=[('乙庚两宫相生，但尚未说明是否逢吉门吉格。按本条能直接保证结婚吗？','不能只凭相生保证结婚。本条还要求逢吉门吉格，应先核对这些条件。','marriage_conditions'),('问的是结婚能否成立，并非结婚日期。除了两宫相生，本条还需什么条件？','还需逢吉门吉格。相生或比和与吉门吉格是本条要合看的条件。','marriage_conditions'),('只给乙庚两宫相生这一条，没有实际局与时间资料。能按这一条列出具体结婚日期吗？','不能给出具体日期。这条说明恋爱婚姻成否的条件，没有提供可换算为日期的应期依据；还需实际盘面和应期时间资料。','timing_scope'),('这里不问日期，只问乙庚相生是否已经满足本条全部条件。应该先核对什么？','先核对是否还逢吉门吉格。不能把单独相生等同于本条全部条件。','marriage_conditions')]
    for q,a,kind in questions:
        for i,prefix in enumerate(['所问目标核对：','请先回答这一问：','请区分成否与日期：']):records.append(row(prefix+q,a,marriage,u,skill='condition',split='validation' if i==2 else 'train',domain='marriage',check={'kind':kind},provenance={'kind':'marriage_outcome_vs_timing','contrast_group':kind,'answer_goal':kind}))
    # Existing source-backed rule applications get explicit question-goal
    # contrasts; answers remain the approved literal-source teachers.
    for key in ['health_not_one_star','health_doctor_relation','health_shared_palace_roles','health_unknown_real_querent','marriage_mediator_woman','marriage_mediator_man','exam_ask_identity']:
        s=specs[key]
        linked=[c for c in parent['case_catalog'] if c['case_id'] in s.get('case_ids',[])]
        ownership=next((r['split'] for r in parent['records'] if linked and r['group']==linked[0]['group']),None)
        if linked and ownership not in ['train','validation']:continue
        for i,prefix in enumerate(['请回答所问并解释必要条件：','按本条应用规则，直接说明理由：','请根据原文说明这一问题：']):
            records.append(row(prefix+s['question'],s['answer'],s['support'],u,skill='condition',split=ownership or ('validation' if i==2 else 'train'),domain=s['domain'],check=s.get('checks'),case_ids=s.get('case_ids',[]),group=linked[0]['group'] if linked else s.get('group'),provenance={'kind':'source_goal_application','answer_goal':key}))
    # Preserve family ownership and all originals. No model observations enter.
    result={'schema':SCHEMA,'records':records,'old_manual_scores_used':False,'model_outputs_as_teachers':False,'original_final_test_used':False,'v57_to_v58_ids':old_to_new}
    return result


def validate(pack):
    base.validate(pack|{'schema':base.SCHEMA})
    if pack['schema']!=SCHEMA:raise ValueError('V58 schema required')
    ids=set();messages={}
    for r in pack['records']:
        if r['id'] in ids:raise ValueError('Duplicate lesson ID')
        ids.add(r['id']);key=d.dumps(r['messages'])
        if key in messages:raise ValueError('Exact dialogue duplicate or cross-split leakage')
        messages[key]=r['split']
        if any(g in m['content'] for m in r['messages'] for g in GUARDS):raise ValueError('Irrelevant missing-chart instruction in knowledge curriculum')
    return True
