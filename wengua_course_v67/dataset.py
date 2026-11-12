"""Source-only typed objects and independently varied conjunction premises.

No saved model answer is read here. The existing chart engine supplies literal
positions; two books supply properties and conditional interpretations. Old
records, source units, case families and final-test precedence stay preserved.
"""
from copy import deepcopy
from itertools import product
from collections import Counter
from wengua_course import data as d
from wengua_course_v66 import dataset as previous
old=previous.old
SCHEMA='wengua_typed_objects_independent_conditions_v67'
SYSTEM=old.SYSTEM
RELATIONS=['相生','比和','相克','未提供'];STATES=['已具备','明确不具备','未提供']
STAR_ELEMENTS={'天心星':'金','天芮星':'土'}
GOALS=['stars','heart_to_palace','illness_to_palace','palaces','reverse']



def visible_question(c,q):
    kind=c['kind']
    if kind in ['independent_conditions','condition_claim']:
        prefix=f'当前核实资料：乙庚两宫关系={c["palace_relation"]}；吉门吉格={c["gate_state"]}。'
    elif kind=='typed_objects':
        selectors={'stars':'天心星本身 → 天芮星本身','heart_to_palace':'天心星本身 → 天心星所在宫',
                   'illness_to_palace':'天芮星本身 → 天芮星所在宫','palaces':'天心星所在宫 → 天芮星所在宫','reverse':'天芮星所在宫 → 天心星所在宫'}
        prefix='本次比较对象：'+selectors[c['goal']]+'。'
    else:return q
    return q if q.startswith(prefix+'\n') else prefix+'\n'+q

def condition_state(c):
    if c['palace_relation'] not in RELATIONS or c['gate_state'] not in STATES:raise ValueError('Unregistered condition state')
    a=True if c['palace_relation'] in ['相生','比和'] else False if c['palace_relation']=='相克' else None
    b={'已具备':True,'明确不具备':False,'未提供':None}[c['gate_state']]
    return {'palace_requirement':a,'auspicious_requirement':b,'joint':False if a is False or b is False else True if a is True and b is True else None}


def condition_facts(c):
    a={'相生':'乙庚两宫相生，符合宫间关系要求','比和':'乙庚两宫比和，符合宫间关系要求',
       '相克':'乙庚两宫相克，不符合相生或比和要求','未提供':'乙庚的两宫关系尚未提供，这一项还没有核实'}[c['palace_relation']]
    b={'已具备':'吉门吉格已核实具备','明确不具备':'吉门吉格已明确不具备','未提供':'吉门吉格尚未提供，这一项还没有核实'}[c['gate_state']]
    return a,b


def condition_answer(c):
    state=condition_state(c);a,b=condition_facts(c)
    if state['joint'] is True:result='两项前提齐全，可以据此判断恋爱可成。'
    elif state['joint'] is False:
        failed=[]
        if state['palace_requirement'] is False:failed.append('两宫关系')
        if state['auspicious_requirement'] is False:failed.append('吉门吉格')
        result='已知不符合的是'+'和'.join(failed)+'，这一组条件不能支持判成。'
    else:
        unknown=[]
        if state['palace_requirement'] is None:unknown.append('两宫关系')
        if state['auspicious_requirement'] is None:unknown.append('吉门吉格')
        result='还需核实'+'和'.join(unknown)+'，暂时不能据此判成；未提供不等于已经不符合。'
    text=a+'；'+b+'。'+result
    if c.get('date'):text+='这些条件没有给出具体结婚日期，不能据此定到某一天。'
    return text


def object_facts(board):
    return {'symbols':{s:{'category':'九星','intrinsic_element':e,'palace':board['stars'][s]} for s,e in STAR_ELEMENTS.items()},
            'palaces':{p:{'element':old.PALACES[p]} for p in set(board['stars'].values())}}


def objects(board,goal):
    if goal not in GOALS:raise ValueError('Unregistered comparison objects')
    heart,illness=board['stars']['天心星'],board['stars']['天芮星']
    if goal=='stars':return ('天心星','天芮星','金','土')
    if goal=='heart_to_palace':return ('天心星',heart+'宫','金',old.PALACES[heart])
    if goal=='illness_to_palace':return ('天芮星',illness+'宫','土',old.PALACES[illness])
    a,b=(heart,illness) if goal=='palaces' else (illness,heart)
    return (a+'宫',b+'宫',old.PALACES[a],old.PALACES[b])


def object_answer(c,boards):
    b=boards[c['board_id']];a,t,x,y=objects(b,c['goal']);text=previous.direction(a,t,x,y)[0]
    if c['goal'] in ['palaces','reverse']:
        h,i=b['stars']['天心星'],b['stars']['天芮星'];text=f'天心星在{h}宫，天芮星在{i}宫。'+text
    return text


def literal_board(board,style):
    if style=='typed':
        h,i=board['stars']['天心星'],board['stars']['天芮星']
        return f'程序位置：天心星在{h}宫，天芮星在{i}宫。\n符号本体属性：天心星属金，天芮星属土。\n宫的属性：{h}宫属{old.PALACES[h]}，{i}宫属{old.PALACES[i]}。'
    lines=[]
    for p in ['巽','离','坤','震','中','兑','艮','坎','乾']:
        stars=[s for s,g in board['stars'].items() if g==p]
        values=[s[1] for s in stars] if style=='compact' else stars
        lines.append(f'{p}{old.PALACE_NUMBER[p]}宫：'+('、'.join(values) if values else '未列星位'))
    return ('下面是程序核定的九宫星位摘录。' if style=='full' else '下面星行用九星单字简称，仍是程序核定的星位。')+'\n'+'\n'.join(lines)


def object_question(board,goal,style,*,development=False):
    q={'stars':'只比较天心星与天芮星本身的五行关系，说明前者对后者。',
       'heart_to_palace':'只比较天心星自身与它所在的宫，分别说明属性和作用方向。',
       'illness_to_palace':'只比较天芮星自身与它所在的宫，分别说明属性和作用方向。',
       'palaces':'只比较天心所在宫对天芮所在宫；请说明两个宫的属性和关系。',
       'reverse':'只比较天芮所在宫对天心所在宫；请说明两个宫的属性和关系。'}[goal]
    if development:q='请按本次所问对象重新组织回答：'+q
    return literal_board(board,style)+'\n'+q


RULE='本问是恋爱婚姻，只检查这一条可成分支：乙庚两宫相生或比和，并且逢吉门吉格，才可按本条判可成。其他结局与日期需另有依据。'
def condition_question(c,style=0,*,development=False):
    rel=c['palace_relation'];gate=c['gate_state']
    text=[f'已核对两宫关系为“{rel}”，吉门吉格为“{gate}”。哪项已经满足、哪项不满足或尚未知？能使用这条分支吗？',
          '当前资料：'+d.dumps({'乙庚两宫关系':rel,'吉门吉格核实结果':gate})+'。请说明条件和本条所能给的结论。',
          f'先分别确认两项资料：乙庚两宫={rel}；吉门吉格={gate}。有人只看一个条件就判成，你怎样按完整条件判断？',
          f'我现在只知道这些：乙庚两宫关系是{rel}，吉门吉格是{gate}。请直接说条件齐不齐，缺的究竟是哪一项。'][style]
    if development:text=f'本次最新核实：两宫关系“{rel}”，吉门吉格“{gate}”。请逐项处理已知与未知，再回答本条可成条件。'
    if c.get('date'):text+='这些资料能确定具体哪一天结婚吗？'
    return RULE+'\n'+text


def answer(c,boards):
    if c['kind']=='typed_objects':return object_answer(c,boards)
    if c['kind']=='independent_conditions':return condition_answer(c)
    if c['kind']=='star_claim':
        star=c['star'];expected=STAR_ELEMENTS[star];p=boards[c['board_id']]['stars'][star]
        verdict='这个标注可以保留。' if c['claim']==expected else '这个标注需要改正。'
        return verdict+f'{star}自身属{expected}，当前在{p}宫；{p}宫属{old.PALACES[p]}。星本体属性与所在宫属性分别记录。'
    if c['kind']=='condition_claim':
        verdict=condition_state(c)['joint'];claim=c['claim']
        ok=claim==('成立' if verdict is True else '不具备' if verdict is False else '待核实')
        return ('这个标注可以保留。' if ok else '这个标注需要改正。')+condition_answer(c)
    if c['kind']=='identity_referent':return old.identity_answer(c)
    if c['kind']=='source_scope':return f'{c["symbol"]}代表{c["role"]}。这是一条取用规则；本问未给具体排盘及月令，不能据此确认当前落宫或已经与月令比和。'
    raise ValueError('Unregistered V67 source contract')


def build(source):
    if source['schema']!=previous.SCHEMA:raise ValueError('Frozen V66 curriculum required')
    boards={b['base_id']:b for b in source['physical_star_boards']};units=deepcopy(source['units'])
    train=[r for r in source['records'] if r['pilot_selected']]
    def family(kind):return deepcopy(next(r['provenance']['support'] for r in train if r['skill']==kind))
    def literal(fragment):
        u=min((u for u in units.values() if fragment in u['original_quote']),key=lambda u:len(u['original_quote']))
        return {'source_id':u['id'],'quote':u['original_quote']}
    object_support=family('objects')+[literal('天心星属金'),literal('死门、天芮俱是土')]
    unique={s['source_id']:s for s in object_support};object_support=list(unique.values())
    marriage_support=family('marriage');identity_support=family('identity');rows=[];dispositions=[]
    def add(c,q,support,*,split='train',history=None,initial=None,initial_chain=None):
        text=answer(c,boards);messages=deepcopy(history) if history else [{'role':'system','content':SYSTEM}]
        messages += [{'role':'user','content':q},{'role':'assistant','content':text}]
        contracts=deepcopy(initial_chain) if initial_chain else [deepcopy(initial)] if initial else []
        contracts.append(deepcopy(c));users=[m for m in messages if m['role']=='user']
        if len(users)!=len(contracts):raise ValueError('One explicit contract for each supervised turn required')
        for message,contract in zip(users,contracts,strict=True):message['content']=visible_question(contract,message['content'])
        key='v67:'+d.digest(d.dumps(messages))[:24]
        rows.append({'id':key,'split':split,'domain':'marriage' if c['kind'] in ['independent_conditions','condition_claim'] else 'health' if c['kind']=='identity_referent' else 'source_objects',
            'format':'source_object_and_condition_dialogue','stage':'application','task':'typed_objects_and_independent_conditions','skill':c['kind'],
            'group':'rule:'+c['kind'],'question':messages[1]['content'],'answer':text,'messages':messages,'source_ids':sorted({s['source_id'] for s in support}),
            'case_id':None,'case_ids':[],'source_mode':'book_rule_generated_teaching_not_observed_event','checks':None,
            'provenance':{'support':deepcopy(support),'lesson_contract':deepcopy(c),'initial_contract':deepcopy(initial),
              'initial_chain':deepcopy(initial_chain),'turn_contracts':contracts,'visible_user_sha256':d.digest(d.dumps([m['content'] for m in users])),'teacher_is_model_output':False,'not_observed_outcome':True}})
    # The whole V66 archive is preserved, while only its declared short-course
    # replays are used. Do not silently train all 1780 rows again.
    for original in source['records']:
        action='preserved_not_in_short_training'
        if original['pilot_selected']:
            if original['skill']=='marriage':action='replace_with_independent_premises'
            elif original['skill']=='board':action='replace_with_typed_and_raw_object_tasks'
            else:
                row=deepcopy(original);row['id']='v67:replay:'+row['id'];row['provenance']['unchanged_v66_source_replay']=original['id'];rows.append(row);action='replay_unchanged'
        dispositions.append({'old_id':original['id'],'action':action})
    for board in boards.values():
        for goal,style in product(GOALS,['full','compact','typed']):
            c={'kind':'typed_objects','board_id':board['base_id'],'goal':goal,'facts':object_facts(board)}
            add(c,object_question(board,goal,style),object_support)
        for goals in [['stars','heart_to_palace','palaces','reverse'],['palaces','illness_to_palace','stars','reverse']]:
            history=[{'role':'system','content':SYSTEM}];chain=[]
            for index,goal in enumerate(goals):
                c={'kind':'typed_objects','board_id':board['base_id'],'goal':goal,'facts':object_facts(board)}
                q=object_question(board,goal,'full') if not index else '星位不变，现在变更所问对象。'+object_question(board,goal,'typed').split('\n')[-1]
                if index==len(goals)-1:add(c,q,object_support,history=history,initial_chain=chain)
                history.extend([{'role':'user','content':q},{'role':'assistant','content':answer(c,boards)}]);chain.append(deepcopy(c))
        for star,correct in product(STAR_ELEMENTS,[False,True]):
            expected=STAR_ELEMENTS[star];palace=board['stars'][star];wrong=old.PALACES[palace] if old.PALACES[palace]!=expected else next(e for e in old.ELEMENTS if e!=expected)
            c={'kind':'star_claim','board_id':board['base_id'],'star':star,'claim':expected if correct else wrong}
            add(c,f'程序已核定{star}在{palace}宫，有人据此把星本身五行标成{c["claim"]}。这个标注保留还是改正？请把星与宫分别说清。',object_support)
    for rel,gate in product(RELATIONS,STATES):
        c={'kind':'independent_conditions','palace_relation':rel,'gate_state':gate}
        for style in range(4):add(c,condition_question(c,style),marriage_support)
        add(c|{'date':True},condition_question(c|{'date':True},1),marriage_support)
        joint=condition_state(c)['joint'];correct='成立' if joint is True else '不具备' if joint is False else '待核实'
        for claim in [correct,next(v for v in ['成立','不具备','待核实'] if v!=correct)]:
            cc=c|{'kind':'condition_claim','claim':claim}
            add(cc,condition_question(c,2)+f'已有标注是“{claim}”，保留还是改正？',marriage_support)
    # Change one slot while preserving the other. All nine combinations exist;
    # unknown remains distinct from false even if the other slot is already false.
    for rel,gate in product(['相生','相克','未提供'],STATES):
        first={'kind':'independent_conditions','palace_relation':rel,'gate_state':gate}
        first_q=condition_question(first,0);history=[{'role':'system','content':SYSTEM},{'role':'user','content':first_q},{'role':'assistant','content':answer(first,boards)}]
        for field,options in [('palace_relation',['相生','相克','未提供']),('gate_state',STATES)]:
            for value in options:
                if value==first[field]:continue
                final=first|{field:value}
                q=(f'只更新乙庚两宫关系为“{value}”，吉门吉格仍为“{gate}”。' if field=='palace_relation' else f'只更新吉门吉格为“{value}”，两宫关系仍为“{rel}”。')
                add(final,q+'请分别说明两项当前状态和可成分支。',marriage_support,history=history,initial=first)
    for states in [[('相生','未提供'),('相生','明确不具备'),('相生','已具备')],
                   [('相克','已具备'),('未提供','已具备'),('比和','已具备')],
                   [('未提供','未提供'),('相生','未提供'),('相生','已具备'),('相克','已具备')]]:
        history=[{'role':'system','content':SYSTEM}];chain=[]
        for index,(rel,gate) in enumerate(states):
            c={'kind':'independent_conditions','palace_relation':rel,'gate_state':gate}
            q=condition_question(c,3) if not index else f'现在只按最新核实资料：两宫关系为{rel}，吉门吉格为{gate}。请重新逐项判断可成分支。'
            if index==len(states)-1:add(c,q,marriage_support,history=history,initial_chain=chain)
            history.extend([{'role':'user','content':q},{'role':'assistant','content':answer(c,boards)}]);chain.append(deepcopy(c))
    for original in [r for r in train if r['skill']=='literal_fact'][:8]:
        c=deepcopy(original['provenance']['lesson_contract']);c['kind']='source_scope'
        add(c,f'在{c["domain_name"]}规则里，{c["role"]}取什么？当前只给了这一条原文，没有排盘与月令。能不能进一步确认它现在在哪一宫、与月令已经比和？',original['provenance']['support'])
    fields={'年干':'甲','月干':'丁','日干':'庚','时干':'壬'}
    for role,birth,missing in product(old.ROLES,[None,'癸'],[False,True]):
        f=deepcopy(fields)
        if missing:f.pop(old.ROLES[role][0])
        c={'kind':'identity_referent','role':role,'fields':f,'birth':birth}
        q=old.ROLE_PHRASE[role]+'，本题只有这一位患者，患者就是上述同一人。'+d.dumps({'起局字段':f,'患者可靠出生年干':birth})+'。主用神、必要兼看宫和年命各怎样处理？'
        add(c,q,identity_support)
    # Development wording is separate; knowledge and physical boards are known.
    for rel,gate in product(['相生','相克','未提供'],STATES):
        c={'kind':'independent_conditions','palace_relation':rel,'gate_state':gate}
        add(c,condition_question(c,development=True),marriage_support,split='validation')
    for i,board in enumerate(boards.values()):
        for j,goal in enumerate(['stars','palaces']):
            c={'kind':'typed_objects','board_id':board['base_id'],'goal':goal,'facts':object_facts(board)}
            add(c,object_question(board,goal,'typed' if (i+j)%2 else 'full',development=True),object_support,split='validation')
    for i,board in enumerate(boards.values()):
        c={'kind':'typed_objects','board_id':board['base_id'],'goal':'palaces','facts':object_facts(board)}
        # Complete an exact same-board/goal pair: raw full names versus the
        # program/source fact card. Never mix their results into raw-read scores.
        add(c,object_question(board,'palaces','full' if (i+1)%2 else 'typed',development=True),object_support,split='validation')
    for star,claim in [('天心星','金'),('天心星','木'),('天芮星','土'),('天芮星','火')]:
        board=next(iter(boards.values()));c={'kind':'star_claim','board_id':board['base_id'],'star':star,'claim':claim}
        add(c,f'请核对一项记录：{star}落{board["stars"][star]}宫，记录写星自身属{claim}。依据固定属性，这项记录应保留还是修正？',object_support,split='validation')
    for role in ['本人','父亲','同事']:
        c={'kind':'identity_referent','role':role,'fields':fields,'birth':'癸'}
        add(c,old.ROLE_PHRASE[role]+'。这里所说的患者就是这一位人。'+d.dumps(fields)+'，可靠出生年干癸。请把主用神与出生资料分开回答。',identity_support,split='validation')
    for i,role in enumerate(['父亲','同事','本人','女儿']):
        c={'kind':'identity_referent','role':role,'fields':fields,'birth':None}
        add(c,'我想确认你会怎样直接回复：'+old.ROLE_PHRASE[role]+'，不知道患者出生年，起局资料'+d.dumps(fields)+'。请用自然的一段话给取用结论与必要原因。',identity_support,split='validation')
    unique={}
    for r in rows:
        key=d.dumps(r['messages'][:-1])
        if key in unique:
            if unique[key]['answer']!=r['answer'] or unique[key]['split']!=r['split']:raise ValueError('Conflicting or cross-split literal prompt')
        else:unique[key]=r
    rows=list(unique.values());out=deepcopy(source)
    out.update(schema=SCHEMA,records=rows,prior_records=deepcopy(source['records']),dispositions=dispositions,
               model_outputs_used_as_teachers=False,new_books_used=False)
    usage=Counter(s for r in rows if r['split']=='train' for s in r['source_ids'])
    out['material_usage']=[{'source_id':sid,'new_train_lessons':usage[sid],'disposition':'used' if usage[sid] else 'preserved'} for sid in units]
    out['summary']={'schema':SCHEMA,'train_cores':sum(r['split']=='train' for r in rows),'development':sum(r['split']=='validation' for r in rows),
        'training_skills':dict(Counter(r['skill'] for r in rows if r['split']=='train')),'unchanged_v66_replays':sum(bool(r['provenance'].get('unchanged_v66_source_replay')) for r in rows),
        'preserved_v66_cores':len(source['records']),'source_units':len(units),'original_case_registry':len(source['case_splits']),
        'first_and_second_conditions_independently_varied':True,'model_outputs_are_teachers':False,'old_manual_scores_used':False,
        'new_blind_benchmark':False,'full_case_training_claimed':False}
    validate(out,source);return out


def validate(pack,source):
    if pack['units']!=source['units'] or pack['case_splits']!=source['case_splits'] or pack['prior_records']!=source['records'] or pack['physical_star_boards']!=source['physical_star_boards']:raise ValueError('Old source/record/test boundary changed')
    if pack['schema']!=SCHEMA or pack['model_outputs_used_as_teachers'] or pack.get('old_manual_scores_used') or pack.get('new_books_used'):raise ValueError('Forbidden training track')
    if pack['source_books']!=source['source_books'] or set(pack['source_books'])!={'shenqi','kaiwu'}:raise ValueError('Only the original two source books are allowed')
    boards={b['base_id']:b for b in pack['physical_star_boards']};before={r['id']:r for r in source['records']};seen=set()
    for r in pack['records']:
        key=d.dumps(r['messages'][:-1])
        if key in seen:raise ValueError('Repeated core prompt')
        seen.add(key);p=r['provenance']
        if p.get('unchanged_v66_source_replay'):
            original=before[p['unchanged_v66_source_replay']]
            if r['messages']!=original['messages'] or r['answer']!=original['answer'] or original['split']!='train':raise ValueError('Source replay changed')
            continue
        c=p['lesson_contract']
        users=[m for m in r['messages'] if m['role']=='user'];contracts=p['turn_contracts']
        if contracts[-1]!=c or len({cc['board_id'] for cc in contracts if cc['kind']=='typed_objects'})>1:raise ValueError('Current contract or unchanged board differs')
        if len(users)!=len(contracts) or p['visible_user_sha256']!=d.digest(d.dumps([m['content'] for m in users])):raise ValueError('Visible input changed')
        for message,cc in zip(users,contracts,strict=True):
            if visible_question(cc,message['content'])!=message['content']:raise ValueError('Conditions or object selector only exist in hidden metadata')
            if cc['kind']=='typed_objects':
                board=boards[cc['board_id']]
                if cc['facts']!=object_facts(board):raise ValueError('Wrong intrinsic or palace fact')
                for marker,style in [('程序核定的九宫星位摘录','full'),('星行用九星单字简称','compact'),('程序位置：','typed')]:
                    if marker in message['content'] and literal_board(board,style) not in message['content']:raise ValueError('Visible program chart transcription differs')
        if r['messages'][-1]['content']!=r['answer'] or r['answer']!=answer(c,boards):raise ValueError('Teacher contradicts source contract')
        for item in p['support']:
            if d.compact(item['quote']) not in d.compact(pack['units'][item['source_id']]['original_quote']):raise ValueError('Source support changed')
        if set(r['source_ids'])!={s['source_id'] for s in p['support']}:raise ValueError('Source ids differ')
        if c['kind']=='typed_objects' and c['facts']!=object_facts(boards[c['board_id']]):raise ValueError('Intrinsic and palace facts conflated')
        initial=p.get('initial_chain') or ([p['initial_contract']] if p.get('initial_contract') else [])
        historical=[m for m in r['messages'][:-1] if m['role']=='assistant']
        if len(historical)!=len(initial) or any(m['content']!=answer(cc,boards) for m,cc in zip(historical,initial,strict=True)):raise ValueError('Historical teacher changed')
        if c['kind'] in ['independent_conditions','condition_claim'] and '恋爱婚姻' not in r['messages'][1]['content']:raise ValueError('Domain is not visible')
    return True
