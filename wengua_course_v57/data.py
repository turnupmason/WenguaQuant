"""Source-first V57 lessons. Originals, family boundaries and old scores freeze."""
from copy import deepcopy
from collections import Counter
import re
from wengua_course import data as d
from wengua_course_v54 import corpus as literal, data as structural
from wengua_course_v56 import data as previous

SCHEMA='wengua_source_contract_course_v57'
SYSTEM='根据本次问题的领域、实际求测人与所问对象，使用给出的原文和盘面回答。先说明所问结论，再给必要事实、条件和依据；正常中文，不重复免责声明。严格区分本人求测与代问、起局年干与出生年命、星本身五行与落宫五行。未知信息不补造；若题目缺决定取用的条件，明确追问。没有证据不编日期，不使用人工吉凶评分。'
RELATIONS=['生','受生','克','受克','比和']


def supports(items,units):
    """A material can legitimately supply several DIFFERENT literal excerpts."""
    result=[];seen=set()
    for s in items:
        sid=s['source_id'];quote=s['quote']
        if sid not in units or d.compact(quote) not in d.compact(units[sid]['original_quote']):raise ValueError('Nonliteral source excerpt')
        key=(sid,quote)
        if key not in seen:seen.add(key);result.append(deepcopy(s))
    return result


def question(row):
    text=row['messages'][-2]['content']
    if '用户问题：' in text:text=text.split('用户问题：',1)[1].split('\n本题',1)[0]
    for marker in ['\n\n本题来源：','\n本题来源：','\n\n本题检索证据：','\n本题检索证据：','\n原书证据：']:
        text=text.split(marker,1)[0]
    return text.strip().replace('施生、受生、施克','生、受生、克')


def row(q,answer,support,units,*,skill,split='train',domain='foundation',check=None,case_ids=(),group=None,provenance=None):
    source=supports(support,units)
    messages=[{'role':'system','content':SYSTEM},{'role':'user','content':q+'\n\n原文依据：\n'+'\n\n'.join('[S'+str(i+1)+'] '+s['quote'] for i,s in enumerate(source))}]
    answer=answer.rstrip()+' '+'、'.join('[S'+str(i+1)+']' for i in range(len(source)))
    messages.append({'role':'assistant','content':answer})
    item=d.row(q,answer,[s['source_id'] for s in source],units,task=skill,format='complete_case' if skill in ['author_reading','author_local','case'] else 'conditional_dialogue' if skill in ['identity','condition'] else 'chart_application' if skill in ['chart_direction','given_direction','positions'] else 'foundation_qa',split=split,domain=domain,checks=check,group=group,case_ids=case_ids,messages=messages,provenance=provenance or {})
    item.update(id='v57:'+d.digest(d.dumps([split,messages]))[:24],messages=messages,question=messages[-2]['content'],answer=answer,stage='application',skill=skill,context='provided_evidence')
    item['provenance'].update(support=source,required_citations=['S'+str(i+1) for i in range(len(source))],teacher_is_model_output=False,source_contract_version='v57')
    return item


def relation_answer(a,b,sa='前者',sb='后者'):
    rel=d.relation({'水':1,'土':2,'木':3,'金':6,'火':9}[a],{'水':1,'土':2,'木':3,'金':6,'火':9}[b])
    if a==b:return f'两者比和。{sa}和{sb}同属{a}，五行相同。'
    x,y=(a,b) if rel in ['生','克'] else (b,a);s,t=(sa,sb) if rel in ['生','克'] else (sb,sa);op='生' if rel in ['生','受生'] else '克'
    return f'从{sa}看{sb}，关系为{rel}。{sa}属{a}，{sb}属{b}；{x}{op}{y}，即{s}{op}{t}。'


def palace_pair(r):
    c=r['checks']
    if c.get('positions'):return list(c['positions'].values())
    names={v:k for k,v in d.PALACE.items()};found=re.findall('|'.join(map(re.escape,names)),question(r))
    if len(found)<2:raise ValueError('Missing explicit palace pair: '+r['id'])
    return [names[x] for x in found[:2]]


def property_bundle(facts,keys,units):
    return supports([s for key in keys for s in facts[key]['support']],units)


def graph_positions(text,expected):
    rows={}
    for line in text.splitlines():
        m=re.match(r'([坎坤震巽中乾兑艮离])([1-9一二三四五六七八九])宫：.*?星行(.*)',line)
        if m:
            number='一二三四五六七八九'.find(m[2])+1 if m[2] not in '123456789' else int(m[2]);rows[number]=m[3]
    if len(rows)!=9:raise ValueError('Incomplete nine-palace task input')
    for star,n in expected.items():
        raw=star.removesuffix('星').removeprefix('天')
        found=[p for p,cell in rows.items() if raw in cell]
        if found!=[n]:raise ValueError('Teacher position does not match visible board: '+str((star,n,found)))
    return True


def build(parent):
    from wengua_course_v53 import runtime as original
    units=parent['units'];facts=literal.properties(original.verify_data(),literal.source_library());rows=[];dispositions=[]
    cycle=property_bundle(facts,['rule:generation','rule:control'],units)
    for original in parent['records']:
        skill=original['skill'];c=deepcopy(original.get('checks') or {});r=deepcopy(original)
        if skill in ['author_reading','author_context'] or skill=='case' and r['provenance'].get('kind')=='historical_author_source_reading':
            dispositions.append({'id':r['id'],'action':'superseded_by_complete_same_stage_source_reading','source_ids':r['source_ids']});continue
        if skill in ['direction','given_direction','chart_direction'] and c.get('elements'):
            a,b=c['elements'];q=question(r);refs=cycle;sa,sb='前者','后者'
            if skill in ['given_direction','chart_direction']:
                pa,pb=palace_pair(r)
                if [d.ELEMENT[pa],d.ELEMENT[pb]]!=[a,b]:raise ValueError('Gold elements/palaces inconsistent')
                refs=property_bundle(facts,['palace:'+str(pa),'palace:'+str(pb),'rule:generation','rule:control'],units)
                if c.get('positions'):
                    names=list(c['positions']);sa=names[0]+'星所在宫';sb=names[1]+'星所在宫'
                    if skill=='chart_direction':graph_positions(q,c['positions'])
                else:sa=d.PALACE[pa]+'宫';sb=d.PALACE[pb]+'宫'
                c['palaces']=[pa,pb]
            target=relation_answer(a,b,sa,sb)
            if c.get('positions'):target='；'.join(s+'星在'+d.PALACE[n]+'宫' for s,n in c['positions'].items())+'。'+target
            c.update(kind='relation_facts',symbols=[sa,sb],require_positions=bool(c.get('positions')))
            nr=row(q,target,refs,units,skill=skill,split=r['split'],domain=r['domain'],check=c,case_ids=r['case_ids'],group=r['group'],provenance={'kind':'complete_source_relation','old_id':r['id'],'required_source_fragments':refs,'question_comparison_level':'palace' if skill!='direction' else 'given_element'})
            rows.append(nr);continue
        if skill=='door_relation':
            key=r['provenance'].get('key','');m=re.fullmatch(r'door_direction_([休生伤杜景死惊开])_([1-9])',key)
            if not m:raise ValueError('Missing door source identity')
            door,n=m[1],int(m[2]);a=literal.DOORS[door];b=d.ELEMENT[n];rel=d.relation({'水':1,'土':2,'木':3,'金':6,'火':9}[a],n)
            state={'克':'迫','受克':'制','生':'和','受生':'义','比和':'比和'}[rel]
            if c['answer']!=state:raise ValueError('Door naming differs from source')
            refs=property_bundle(facts,['door:'+door,'palace:'+str(n),'rule:generation','rule:control','rule:door_relation'],units)
            target='本条为'+state+'。'+relation_answer(a,b,door+'门',d.PALACE[n]+'宫')
            rows.append(row(question(r),target,refs,units,skill=skill,split=r['split'],domain=r['domain'],check=c|{'kind':'door_facts','door':door,'palaces':[n],'relation':rel},provenance={'kind':'source_door_direction','old_id':r['id'],'required_source_fragments':refs}));continue
        if skill=='property':
            key=r['provenance']['property_key'];fact=facts[key]
            rows.append(row(question(r),fact['label']+'属'+fact['element']+'。',fact['support'],units,skill=skill,split=r['split'],domain=r['domain'],check=c,provenance={'kind':'literal_property','property_key':key,'old_id':r['id']}));continue
        if skill=='positions':graph_positions(question(r),c['positions'])
        r['messages'][0]['content']=SYSTEM;r['id']='v57:retained:'+r['id'];r['provenance']['v56_source_id']=original['id']
        if r['provenance'].get('actual_production_rag_prompt'):r['provenance']['actual_production_rag_prompt']=False
        r['provenance']['inherited_packet_not_claimed_new_retrieval']=True
        rows.append(r)
    # Preserve the already validated family-level phase ownership; a catalog
    # also keeps its original split, which is not the active teaching split.
    ownership={r['group']:r['split'] for r in parent['records'] if r['group'].startswith('family:')}
    sealed_groups={c['group'] for c in parent['case_catalog'] if parent['original_case_splits'][c['case_id']]=='test' or c.get('phase_split')=='test'}
    # Every original author step is indexed, with original test families sealed.
    for case in parent['case_catalog']:
        for step in case['author_steps']:
            if case['group'] in sealed_groups:
                dispositions.append({'case_id':case['case_id'],'author_step':step['step'],'action':'original_final_test_family_sealed'});continue
            text=step['original'];source=units[step['source_id']];book='神奇之门' if source['book_id']=='shenqi' else '开悟之门';q=f'《{book}》第{source["section_id"]}节原文第{source["lines"][0]}—{source["lines"][1]}行，原案第{step["step"]}条分析。历史原案同一阶段阅读：请复盘下列完整原段已经写明的依据和局部判断；保留各条件，不把该段当成其他阶段或整案最终结果，也不补后来反馈。'
            nr=row(q,text,[{'source_id':step['source_id'],'quote':text}],units,skill='author_reading',split=ownership.get(case['group'],case['phase_split']),domain=case['domain'],case_ids=[case['case_id']],group=case['group'],provenance={'kind':'complete_original_author_step','source_case_id':case['case_id'],'author_step':step['step'],'source_step_sha256':d.digest(text),'complete_source_in_input':True,'not_independent_prediction':True})
            if re.search(r'[：:][”」]?\s*$',text):
                nr['skill']='author_context';nr['provenance']['kind']='complete_original_author_context'
                nr['messages'][-2]['content']=nr['messages'][-2]['content'].replace('请复盘下列完整原段已经写明的依据和局部判断','此段止于分析引入或讲话提示，只复盘原段已有文字和条件')
                nr['question']=nr['messages'][-2]['content']
                target='本段给出的原文是：“'+text.rstrip()+'” 本段未给完整后续断语，不补后续结果。 [S1]'
                nr['answer']=target;nr['messages'][-1]['content']=target
            rows.append(nr)
    rows.extend(identity_lessons(parent,facts))
    for pair in ['戊戊','丁壬','庚戊','乙辛']:
        original=next(r for r in parent['records'] if r['skill']=='typed_pair' and (r.get('checks') or {}).get('pair')==pair and r['checks']['kind']=='exact_name')
        refs=[{'source_id':sid,'quote':units[sid]['original_quote']} for sid in original['source_ids']]
        rows.append(row(f'这里只知道年干{pair[0]}和时干{pair[1]}，天地盘同宫资料没有给出。能把它当作天盘{pair[0]}加地盘{pair[1]}的格局吗？请说明缺的条件。',f'不能据此认定。这里给的是年干、时干，还缺天盘{pair[0]}和地盘{pair[1]}同宫的位置证据。',refs,units,skill='typed_pair',split='validation',domain='formation',check={'kind':'insufficient'},provenance={'kind':'explicit_type_boundary_development','pair':pair}))
    # Additional concise literal properties, not more independent source facts.
    for key,fact in facts.items():
        if not key.startswith(('palace:','door:')):continue
        for i,q in enumerate([fact['label']+'的五行是什么？说明这个属性。','只核对原书定义：'+fact['label']+'属哪一行？']):
            rows.append(row(q,fact['label']+'属'+fact['element']+'。',fact['support'],units,skill='property',check={'kind':'property','symbol':fact['label'],'element':fact['element']},provenance={'kind':'literal_property','property_key':key}))
    # Literal exact teacher duplicates are redundant; source materials stay.
    for r in rows:
        if r['provenance'].get('support'):r['provenance']['support']=supports(r['provenance']['support'],units)
    unique={}
    for r in rows:
        key=d.dumps(r['messages'])
        if key in unique:
            if unique[key]['split']!=r['split']:raise ValueError('Exact dialogue crosses split')
            unique[key]['case_ids']=sorted(set(unique[key]['case_ids'])|set(r['case_ids']))
            dispositions.append({'id':r['id'],'action':'exact_dialogue_duplicate','kept':unique[key]['id']});continue
        unique[key]=r
    rows=list(unique.values())
    result={'schema':SCHEMA,'records':rows,'dispositions':dispositions,'old_manual_scores_used':False,'model_outputs_as_teachers':False,'original_final_test_used':False}
    return result


def identity_lessons(parent,facts):
    specs={s['key']:s for s in parent['reviewed_specs']};u=parent['units'];rows=[]
    definitions=[
      ('father_proxy','health_daughter_father','女儿替父亲问病，父亲出生年未知','先以起局年干看父亲，兼看乾宫。出生年未知，暂不取父亲年命。','年干','乾宫',True,None),
      ('father_proxy_known','health_daughter_father','女儿替父亲问病，已可靠确认父亲出生年干为癸','先以起局年干看父亲，兼看乾宫；另兼看父亲年命癸。癸是已提供的出生年干，不能与本局年干混同。','年干','乾宫',False,'癸'),
      ('father_self','health_father_self','父亲本人来问自己的健康，出生年未知','以日干看患者本人；暂不取年命。本人求测按本人规则，家庭中的父亲身份不改变这个主用神。','日干',None,True,None),
      ('father_self_known','health_father_self','父亲本人来问健康，已可靠确认出生年干为癸','以日干看患者本人，另兼看已确认的年命癸。','日干',None,False,'癸'),
      ('mother_proxy','health_daughter_father','儿子替母亲问病，母亲出生年未知','以起局年干看母亲，兼看坤宫；暂不取母亲年命。','年干','坤宫',True,None),
      ('mother_self','health_father_self','母亲本人问自己的健康，出生年未知','以日干看患者本人；出生年未知，暂不取年命。','日干',None,True,None),
      ('sibling','health_sibling','代问同辈兄弟疾病，长幼与出生年均未提供','以月干看同辈患者；暂不取年命。相应长幼兼用宫待资料明确，不猜长幼。','月干',None,True,None),
      ('child','health_parent_child','父母代问子女健康，性别长幼和出生年均未知','以时干看子女患者；暂不取年命，相关兼用宫待性别长幼明确。','时干',None,True,None),
      ('exam_self','exam_self','考生本人问考试，出生年未知','以日干代表考生本人；暂不取年命。','日干',None,True,None),
      ('exam_proxy','exam_proxy','母亲代问女儿考试，女儿出生年未知','以时干代表女儿这位考生；出生年未知，暂不取年命。','时干',None,True,None),
    ]
    bykey={}
    for key,spec,condition,answer,role,helper,unknown,known in definitions:
        refs=deepcopy(specs[spec]['support'])
        if known:
            extra=next(s for s in parent['reviewed_specs'] if s['key']=='health_reliable_birth_stem')
            refs+=extra['support']
        bykey[key]=(condition,answer,refs,role,helper,unknown,known)
        variants=[condition+'。本题主用神、必要兼用符号与年命怎样处理？',condition+'。请区分谁在求测、谁是患者或考生，再回答取用。',condition+'。只按目前已知条件回答，不增加其他人物。',condition+'。请解释主用神为什么这样取，已知和未知的资料分别怎样使用？']
        for i,q in enumerate(variants):
            rows.append(row(q,answer,refs,u,skill='identity',split='validation' if i==3 else 'train',domain='education' if key.startswith('exam') else 'health',check={'kind':'person_contract','answer':role,'helper':helper,'unknown_year':unknown,'known_year_stem':known,'person_key':key},provenance={'kind':'explicit_person_condition','person_key':key,'birth_year_unknown_in_question':unknown,'repair_priority':True}))
    # Genuine multi-turn transitions: the last answer follows the CURRENT asker.
    pairs=[('father_proxy','father_self'),('father_self','father_proxy'),('mother_proxy','mother_self'),('exam_proxy','exam_self'),('father_proxy','father_proxy_known'),('father_proxy_known','father_proxy')]
    for first,last in pairs:
        x,y=bykey[first],bykey[last];refs=supports(x[2]+y[2],u)
        for i in range(2):
            prefix=(('更正：刚才提供的出生年资料不可靠，应撤回。'+('现在请回答：' if i==0 else '请重新按修订资料判断：')) if first=='father_proxy_known' and last=='father_proxy' else '条件改变：' if i==0 else '请按新条件重新回答：')
            q=prefix+y[0]+'。现在应怎样取用？请重新按当前人物和资料判断。'
            item=row(q,y[1],refs,u,skill='identity',split='validation' if i==1 else 'train',domain='education' if last.startswith('exam') else 'health',check={'kind':'person_contract','answer':y[3],'helper':y[4],'unknown_year':y[5],'known_year_stem':y[6],'person_key':last},provenance={'kind':'actual_role_or_information_transition','person_key':last,'birth_year_unknown_in_question':y[5],'repair_priority':True,'turn_contracts':[{'kind':'person_contract','answer':x[3],'helper':x[4],'unknown_year':x[5],'known_year_stem':x[6],'person_key':first},{'kind':'person_contract','answer':y[3],'helper':y[4],'unknown_year':y[5],'known_year_stem':y[6],'person_key':last}]})
            source=item['messages'][-2]['content'].split('\n\n原文依据：',1)[1]
            first_answer=x[1]+' '+'、'.join('[S'+str(j+1)+']' for j in range(len(refs)))
            item['messages']=[{'role':'system','content':SYSTEM},{'role':'user','content':x[0]+'。应怎样取用？\n\n原文依据：'+source},{'role':'assistant','content':first_answer},item['messages'][-2],item['messages'][-1]]
            item['id']='v57:'+d.digest(d.dumps([first,last,i,item['messages']]))[:24]
            rows.append(item)
    cases=[
      ('health','给爸爸看病，应该先取哪个主用神？未说明由谁求测。','先确认是父亲本人来问，还是子女代问。本人求测取日干；子女代问父亲取起局年干，兼看乾宫。未给出生年时不补年命。',['health_daughter_father','health_father_self'],{'kind':'clarification','options':[['本人','日干'],['代问','年干']],'helper':'乾宫'}),
      ('education','孩子考试该取日干吗？没有说明谁来求测。','先问是考生本人求测，还是父母代问。本人求测以日干看考生；父母代问以时干看考生。可靠出生年资料另用于年命，不猜出生年。',['exam_self','exam_proxy'],{'kind':'clarification','options':[['本人','日干'],['代问','时干']]}),
      ('health','朋友让我替他看看他父亲的病，由我起局；直接把朋友父亲当作我的父亲取用吗？','需要先确认本局实际求测主体与患者的关系，以及是否有可靠出生年资料。儿女代问长辈取年干这条，不能直接把朋友的父亲等同于你的父亲；现有信息不足以唯一指定主用神。',['health_daughter_father'],{'kind':'scope_clarification'}),
      ('health','患者、病神和医生的代表符号同在乾宫，是否必须合并成一个用神？','同宫也要分别识别患者、病神和医生。它们承担的判断身份不同，不能合并成一个用神；具体治疗判断仍按原文相应条件分析。',['health_daughter_father','role_6','role_7'],{'kind':'literal_slots','required':[['分别'],['患者'],['病神'],['医生']],'forbidden':['必须合并','只能取一个']})]
    for domain,q,a,keys,contract in cases:
        refs=[s for key in keys for s in specs[key]['support']]
        for i,prefix in enumerate(['条件核对：','请先确定适用条件：','换一种问法，请解释取用前提：']):
            rows.append(row(prefix+q,a,refs,u,skill='identity' if contract['kind'] in ['clarification','scope_clarification'] else 'condition',split='validation' if i==2 else 'train',domain=domain,check=contract,provenance={'kind':'explicit_missing_or_shared_role_condition','repair_priority':True}))
    return rows


def compose(parent,delta):
    p=deepcopy(parent);p.update(delta);return p


def validate(pack):
    structural.validate(pack|{'schema':structural.SCHEMA})
    if pack['schema']!=SCHEMA or pack.get('old_manual_scores_used') or pack.get('model_outputs_as_teachers'):raise ValueError('V57 source-only schema required')
    for r in pack['records']:
        prov=r['provenance'];refs=prov.get('support',[])
        if refs:
            if refs!=supports(refs,pack['units']):raise ValueError('Exact support duplicates remain')
            for s in refs:
                if d.compact(s['quote']) not in d.compact(pack['units'][s['source_id']]['original_quote']):raise ValueError('Changed source quote')
        if prov.get('required_source_fragments'):
            visible=r['messages'][-2]['content']
            for s in prov['required_source_fragments']:
                if s['quote'] not in visible:raise ValueError('Required factual premise not visible')
        if prov.get('kind') in ['complete_original_author_step','complete_original_author_context']:
            c=next(c for c in pack['case_catalog'] if c['case_id']==prov['source_case_id']);s=next(s for s in c['author_steps'] if s['step']==prov['author_step']);text=s['original']
            if d.digest(text)!=prov['source_step_sha256'] or text not in r['messages'][-2]['content']:raise ValueError('Incomplete or cross-stage author reading')
            if prov['kind']=='complete_original_author_step' and r['answer'].removesuffix(' [S1]')!=text.rstrip():raise ValueError('Changed complete author target')
            if prov['kind']=='complete_original_author_context' and text.rstrip() not in r['answer']:raise ValueError('Changed source-only author context')
    return True
