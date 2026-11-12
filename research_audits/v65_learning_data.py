"""Known-source learning diagnostics, never SFT rows or unseen outcome cases."""
from copy import deepcopy
from collections import Counter
from wengua_course import data as d
from wengua_course_v65 import dataset as course

SCHEMA='wengua_v65_paired_learning_probe_v1'


def build(pack):
    train=[r for r in pack['records'] if r['split']=='train'];rows=[];pairs=[]
    def find(kind,predicate):
        candidates=[r for r in train if r['skill']==kind and len(r['messages'])==3
                    and predicate(r['provenance']['lesson_contract'])]
        return min(candidates,key=lambda r:'参考原文' in r['messages'][1]['content'])
    def add(key,q,c,anchor,category,*,original=False,pair=None,score=False):
        answer=course.answer(c);messages=deepcopy(anchor['messages']) if original else [{'role':'system','content':course.SYSTEM},{'role':'user','content':q},{'role':'assistant','content':answer}]
        if original and messages[-1]['content']!=answer:raise ValueError('Original target changed')
        record={'id':'v65learn:'+key,'messages':messages,'answer':answer,'question':messages[-2]['content'],
          'format':'known_source_learning_diagnostic','task':category,'group':pair or key,
          'source_mode':'evaluation_only_book_rule_generated_not_outcome_case','source_ids':anchor['source_ids'],
          'provenance':{'support':deepcopy(anchor['provenance']['support']),'lesson_contract':deepcopy(c),
             'anchor_training_id':anchor['id'],'exact_v65_training_messages':original,'pair_id':pair,
             'source_book_teacher_not_model_output':True,'inference_only':True},'teacher_forcing':score,
          'components':{'current_objects_and_properties':deepcopy(c),'decision_sentence':answer.split('。')[0]+'。',
                        'full_target':answer,'automatic_professional_pass':False}}
        rows.append(record);return record
    def pair(key,anchor,q):
        if '参考原文：' in anchor['messages'][1]['content']:
            q+='\n参考原文：'+anchor['messages'][1]['content'].split('参考原文：',1)[1]
        c=anchor['provenance']['lesson_contract'];left=add(key+':original','',c,anchor,'training_original',original=True,pair=key,score=True)
        right=add(key+':paraphrase',q,c,anchor,'same_facts_paraphrase',pair=key,score=True)
        pairs.append({'id':key,'original':left['id'],'paraphrase':right['id'],'same_facts_and_target':True,'anchor_training_id':anchor['id']})
    palace_pairs=[('艮','乾'),('乾','艮'),('震','坎'),('坎','震'),('离','兑'),('兑','离')]
    for a,b in palace_pairs:
        anchor=find('objects',lambda c:c['a']==a+'宫' and c['b']==b+'宫' and c['goal']=='relation')
        pair('palace:'+a+b,anchor,f'本问的两个对象依次为{a}宫与{b}宫。请确认各自五行，再明确前一个对后一个是生、克、受生、受克还是比和，理由与结论要一致。')
    for x,y,claim in [('水','火','生'),('金','木','受生'),('木','金','受克'),('水','水','比和'),('土','火','克')]:
        anchor=find('verify',lambda c:c['x']==x and c['y']==y and c['claim']==claim)
        c=anchor['provenance']['lesson_contract']
        pair('verify:'+x+y,anchor,f'请检查一条记录：{c["a"]}属{x}，{c["b"]}属{y}，现把前者对后者写为“{claim}”。保留这个标注还是改正？请说出正确方向和理由。')
    for state in ['已具备','未提供','明确不具备']:
        anchor=next(r for r in train if r['skill']=='marriage' and len(r['messages'])==3
            and r['provenance']['lesson_contract']['state']==state and r['provenance']['lesson_contract']['goal']=='can'
            and '乙庚所在宫相生，' in r['messages'][1]['content'] and '参考原文' not in r['messages'][1]['content'])
        pair('marriage:'+state,anchor,f'乙庚两宫相生，关于吉门吉格，目前的核实状态是“{state}”。就这条可成规则能下什么结论？请保留未知与明确没有的区别。')
    for rel in ['值符生天乙','天乙生值符']:
        anchor=find('loan',lambda c:c['relation']==rel)
        pair('loan:'+rel,anchor,f'银行由值符代表，借贷人由天乙代表。已给的关系是{rel}，请回答这条放贷规则对银行有利还是不利，并把作用方向说清。')
    for role in ['本人','父亲','同事']:
        anchor=find('identity',lambda c:c['role']==role and c.get('birth') is not None and c['fields'].get(course.ROLES[role][0]))
        c=anchor['provenance']['lesson_contract']
        pair('identity:'+role,anchor,course.ROLE_PHRASE[role]+'。本次起局字段'+d.dumps(c['fields'])+f'，患者可靠出生年干{c["birth"]}。哪个字段是患者主用神？已给天干字和出生资料各怎么用？')
    # Atomic controls separate property reading from a directed comparison.
    for palace in ['艮','乾','震','坎','离','兑']:
        anchor=find('attribute',lambda c:c['object']==palace+'宫')
        add('attribute:'+palace,f'只核对固定属性，{palace}宫属什么五行？',anchor['provenance']['lesson_contract'],anchor,'atomic_property')
    relation_anchor=find('relation',lambda c:c['x']=='土' and c['y']=='金')
    for a,b in palace_pairs:
        c={'kind':'relation','a':'前者','b':'后者','x':course.PALACES[a],'y':course.PALACES[b]}
        add('given_elements:'+a+b,f'已给定前者属{c["x"]}、后者属{c["y"]}。只判前者对后者的方向并说明依据。',c,relation_anchor,'given_property_control')
    # Every line is literal physical star data, not a historical patient case.
    boards=pack['physical_star_boards'];number=course.PALACE_NUMBER
    for i,board in enumerate(boards):
        positions=[['天心星',board['stars']['天心星']],['天芮星',board['stars']['天芮星']]]
        anchor=next(r for r in train if r['skill']=='board' and r['provenance']['lesson_contract']['board_id']==board['base_id'])
        c={'kind':'board','board_id':board['base_id'],'positions':positions,'goal':'relation'}
        for compact in [False,True]:
            lines=[]
            for palace in ['巽','离','坤','震','中','兑','艮','坎','乾']:
                stars=[s for s,p in board['stars'].items() if p==palace]
                values=[s[1] for s in stars] if compact else stars
                lines.append(f'{palace}{number[palace]}宫：'+(('星行 ' if compact else '')+'、'.join(values) if stars else '未列星位'))
            q='以下只摘录程序已核定的九宫星位'+('，星行使用九星单字简称' if compact else '')+'。先找天心、天芮的当前宫位，再比较天心所在宫对天芮所在宫的关系并说明两宫五行。\n'+'\n'.join(lines)
            add(f'board:{i}:{int(compact)}',q,c,anchor,'physical_star_reading')
    # Interactive histories are populated with the model's own stored replies.
    episodes=[]
    def episode(key,steps,anchor,flattened):
        episodes.append({'id':key,'system':course.SYSTEM,'source_ids':anchor['source_ids'],
          'support':deepcopy(anchor['provenance']['support']),'steps':[{'id':f'v65learn:dialogue:{key}:{i}',
             'question':q,'answer':course.answer(c),'lesson_contract':c} for i,(q,c) in enumerate(steps)],
          'history_source':'model_own_previous_answers','gold_prior_answers_injected':False})
        add('latest_only:'+key,flattened,steps[-1][1],anchor,'latest_facts_without_history',pair=key)
    fields={'年干':'壬','月干':'己','日干':'丁','时干':'乙'}
    anchor=find('identity',lambda c:c['role']=='父亲' and c.get('birth') is not None)
    father={'kind':'identity','role':'父亲','fields':fields,'birth':None};self_=father|{'role':'本人'}
    episode('patient_takeover',[(f'子女代问自己的父亲，起局字段{d.dumps(fields)}，患者出生年未核实。先怎么取用？',father),
        ('现在由这位父亲亲自问自己的健康，不再由子女代问；还是同一个患者，起局字段不变。现在怎么取？',self_),
        ('人物、起局字段都不变，刚补充患者可靠出生年干癸。主用神和出生资料各怎么处理？',self_|{'birth':'癸'})],anchor,
        f'患者本人问自己的健康，起局字段{d.dumps(fields)}，患者可靠出生年干癸。主用神和出生资料各怎么处理？')
    anchor=find('marriage',lambda c:c['state']=='未提供' and c['goal']=='can')
    states=['未提供','明确不具备','已具备'];steps=[]
    for i,state in enumerate(states):steps.append((('乙庚两宫相生，' if not i else '两宫仍是相生，现在更新核实结果：')+f'吉门吉格的状态是“{state}”。按最新资料，本条可成分支能否成立？',{'kind':'marriage','state':state,'goal':'can'}))
    episode('marriage_update',steps,anchor,'乙庚两宫相生，吉门吉格已核实具备。就这条规则的可成分支能否成立？')
    board=boards[0];p,q=board['stars']['天心星'],board['stars']['天芮星']
    anchor=next(r for r in train if r['skill']=='board' and r['provenance']['lesson_contract']['board_id']==board['base_id'])
    star={'kind':'objects','a':'天心星','b':p+'宫','x':'金','y':course.PALACES[p],'goal':'relation','board_id':board['base_id']}
    forward={'kind':'objects','a':'天心星所在的'+p+'宫','b':'天芮星所在的'+q+'宫','x':course.PALACES[p],'y':course.PALACES[q],'goal':'relation','board_id':board['base_id']}
    reverse=forward|{'a':forward['b'],'b':forward['a'],'x':forward['y'],'y':forward['x']}
    episode('comparison_target',[(f'天心在{p}宫，天芮在{q}宫。先只比较天心星自身与它所在宫的关系。',star),
       ('星位不变，现在比较天心所在宫对天芮所在宫；两个对象都是宫，方向如何？',forward),
       ('星位不变，把比较方向反过来：天芮所在宫对天心所在宫。请重判并说明理由。',reverse)],anchor,
       f'已核定天心在{p}宫、天芮在{q}宫。只比较天芮所在宫对天心所在宫，说明双方五行和方向。')
    anchor=find('identity',lambda c:c['role']=='父亲' and c.get('birth') is not None)
    initial=father|{'birth':'癸'};updated=deepcopy(initial);updated['fields']=fields|{'年干':'辛'}
    episode('calendar_correction',[(f'子女代问自己的父亲，起局字段{d.dumps(fields)}，患者可靠出生年干癸。主用神怎么取？',initial),
       ('人物关系和出生年干癸都不变，刚核实起局年干应为辛，之前壬是录入错误。请按最新字段回答主用神、兼看宫和年命。',updated)],anchor,
       f'子女代问自己的父亲，本次已核定起局字段{d.dumps(updated["fields"])}，患者可靠出生年干癸。主用神、兼看宫和年命各怎么取？')
    result={'schema':SCHEMA,'records':rows,'pairs':pairs,'episodes':episodes,'inference_only':True,'weight_updates':0,
        'source_models_used_as_teachers':False,'new_blind_benchmark':False,'physical_boards_are_not_outcome_cases':True,
        'summary':{'static_questions':len(rows),'pairs':len(pairs),'teacher_forced_targets':sum(r['teacher_forcing'] for r in rows),
           'interactive_turns':sum(len(e['steps']) for e in episodes),'episodes':len(episodes),'per_model_answers':len(rows)+sum(len(e['steps']) for e in episodes),
           'categories':dict(Counter(r['task'] for r in rows))}}
    validate(result,pack);return result


def validate(result,pack):
    by={r['id']:r for r in pack['records']};ids=set();boards={b['base_id']:b['stars'] for b in pack['physical_star_boards']}
    for row in result['records']:
        if row['id'] in ids:raise ValueError('Repeated diagnostic ID')
        ids.add(row['id']);p=row['provenance'];anchor=by[p['anchor_training_id']]
        if anchor['split']!='train' or row['answer']!=course.answer(p['lesson_contract']):raise ValueError('Diagnostic target is not source-bound')
        if p['exact_v65_training_messages'] and row['messages']!=anchor['messages']:raise ValueError('Training original was rewritten')
        if p['support']!=anchor['provenance']['support'] or row['source_ids']!=anchor['source_ids']:raise ValueError('Original support changed')
        c=p['lesson_contract']
        if c['kind']=='board':
            actual=boards[c['board_id']]
            if any(actual[s]!=palace for s,palace in c['positions']):raise ValueError('Diagnostic star position is not literal')
            compact=row['id'].endswith(':1');expected=[]
            for palace in ['巽','离','坤','震','中','兑','艮','坎','乾']:
                stars=[s for s,pal in actual.items() if pal==palace];values=[s[1] for s in stars] if compact else stars
                expected.append(f'{palace}{course.PALACE_NUMBER[palace]}宫：'+(('星行 ' if compact else '')+'、'.join(values) if stars else '未列星位'))
            if row['question'].splitlines()[-9:]!=expected:raise ValueError('Full literal nine-palace transcription changed')
    for pair in result['pairs']:
        a,b=[next(r for r in result['records'] if r['id']==pair[k]) for k in ['original','paraphrase']]
        if a['provenance']['lesson_contract']!=b['provenance']['lesson_contract'] or a['answer']!=b['answer'] or a['messages']==b['messages']:
            raise ValueError('Pair changes facts, target or has no changed wording')
    for episode in result['episodes']:
        if episode['history_source']!='model_own_previous_answers' or episode['gold_prior_answers_injected']:raise ValueError('Teacher history forbidden')
        for step in episode['steps']:
            if step['id'] in ids or step['answer']!=course.answer(step['lesson_contract']):raise ValueError('Dialogue target differs')
            ids.add(step['id'])
    return True


def sequence(pack,answers):
    """Materialize only reachable turns; no future answer can enter a prompt."""
    result=deepcopy(pack['records']);by={r['id']:r for r in answers}
    for ep in pack['episodes']:
        history=[{'role':'system','content':ep['system']}];dependencies=[]
        for step in ep['steps']:
            row={'id':step['id'],'messages':deepcopy(history)+[{'role':'user','content':step['question']},{'role':'assistant','content':step['answer']}],
                'answer':step['answer'],'format':'own_history_dialogue_diagnostic','task':'interactive_condition_update','group':ep['id'],
                'source_mode':'evaluation_only_model_own_history','source_ids':ep['source_ids'],
                'provenance':{'lesson_contract':step['lesson_contract'],'support':ep['support'],'history_source':'model_own_previous_answers',
                    'previous_answers_sha256':list(dependencies),'gold_prior_answers_injected':False},'teacher_forcing':False,
                'components':{'current_objects_and_properties':step['lesson_contract'],'full_target':step['answer'],'automatic_professional_pass':False}}
            result.append(row)
            if step['id'] not in by:break
            actual=by[step['id']];history.extend([{'role':'user','content':step['question']},{'role':'assistant','content':actual['answer']}])
            dependencies.append(d.digest(actual['answer']))
    return result
