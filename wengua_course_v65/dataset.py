"""Work explicit bindings through to a complete, ordinary Chinese answer.

No saved model response is read. The physical library contributes literal
star locations only; generated drills are not observed historical cases.
"""
from pathlib import Path
from copy import deepcopy
from itertools import product
from collections import Counter
import json
from wengua_course import data as d
from wengua_course_v64 import dataset as before

SCHEMA='wengua_binding_and_goal_course_v65'
SYSTEM=before.SYSTEM;MODULES=before.MODULES;PALACES=before.PALACES;DOORS=before.DOORS;ROLES=before.ROLES
ELEMENTS=['木','火','土','金','水'];RELATIONS=['生','克','受生','受克','比和']
relation=before.relation
PALACE_NUMBER={'坎':1,'坤':2,'震':3,'巽':4,'中':5,'乾':6,'兑':7,'艮':8,'离':9}
ROLE_PHRASE={'本人':'患者本人问自己的健康','父亲':'子女代问自己的父亲','母亲':'子女代问自己的母亲',
             '同事':'求测人代问与自己同辈的同事','兄弟':'求测人代问自己的兄弟','儿子':'父母代问自己的儿子','女儿':'父母代问自己的女儿'}


def physical_star_boards(app_root):
    raw=Path(app_root)/before.ATLAS;unique={};ids=set()
    for line in raw.read_text(encoding='utf-8-sig').splitlines():
        row=json.loads(line);ids.add(row['base_id']);stars={s['star']:p['gua'] for p in row['palaces'] for s in p.get('heaven_stars',[])}
        if not row.get('physical_only') or len(stars)!=9:raise ValueError('Literal physical star board required')
        key=d.dumps(stars);unique.setdefault(key,{'base_id':row['base_id'],'stars':stars})
    if len(unique)!=8 or len(ids)!=1080:raise ValueError('Expected 1080 distinct physical boards and eight literal star rotations')
    return sorted(unique.values(),key=lambda x:d.dumps(x['stars']))


def binding_answer(a,b,x,y):
    r=relation(x,y)
    if r=='比和':return f'{a}与{b}是比和。两者都属{x}，这一层不判相生或相克。'
    rule=x+('生' if r=='生' else '克')+y if r in ['生','克'] else y+('生' if r=='受生' else '克')+x
    result={'生':f'{a}生{b}','克':f'{a}克{b}','受生':f'{a}受{b}所生','受克':f'{a}受{b}所克'}[r]
    return f'{result}。这里比较{a}的{x}与{b}的{y}，依据{rule}确定方向。'


def identity_answer(c):
    field,who,palace=ROLES[c['role']];value=c['fields'].get(field)
    result=f'{who}的主用神取本次起局{field}'+(value if value else '')+'。'
    result+=f'因为当前是{ROLE_PHRASE[c["role"]]}，'+(f'兼看{palace}宫。' if palace else f'本条按{field}取用。')
    if not value:result+=f'{field}的具体字未提供，先确定字段，不补造天干。'
    result+=f'出生年命{c["birth"]}另作参看。' if c.get('birth') else '出生年未核实，年命留空。'
    return result


def answer(c):
    kind=c['kind']
    if kind=='attribute':return f'{c["object"]}属{c["element"]}。'
    if kind in ['relation','verify','repair']:
        text=binding_answer(c['a'],c['b'],c['x'],c['y'])
        if kind=='verify':text=('这个标注成立。' if c['claim']==relation(c['x'],c['y']) else '这个标注需要改正。')+text
        return text
    if kind=='objects':
        if c['goal']=='extract':return f'比较对象是{c["a"]}和{c["b"]}，分别属{c["x"]}、{c["y"]}。'
        return binding_answer(c['a'],c['b'],c['x'],c['y'])
    if kind=='locations':return '；'.join(f'{s}在{p}宫' for s,p in c['positions'])+'。'
    if kind=='board':
        lead='；'.join(f'{s}在{p}宫' for s,p in c['positions'])+'。'
        if c['goal']=='locations':return lead
        (s,p),(t,q)=c['positions'];return lead+binding_answer(f'{s}所在的{p}宫',f'{t}所在的{q}宫',PALACES[p],PALACES[q])
    if kind=='identity':return identity_answer(c)
    if kind=='identity_verify':return ('这次取用正确。' if c['claim']==ROLES[c['role']][0] else '这次取用需要改正。')+identity_answer(c)
    if kind=='identity_ambiguous':return '还需要确认谁在求测、谁是患者，以及两人的实际关系。患者本人问取日干，子女代问父母取年干；不能仅凭“父亲”这个称谓固定主用神。'
    if kind=='exam':return ('考生本人来问，主用神取日干。' if c['self'] else '父母代问子女考试，主用神取时干代表实际考生。')+(f'可靠出生年命{c["birth"]}另作参看。' if c.get('birth') else '出生年未核实，年命留空，不影响先确定考生用神。')
    if kind=='door':
        r=relation(DOORS[c['door']],PALACES[c['palace']]);name={'克':'门迫','受克':'门受制','生':'门生宫','受生':'宫生门','比和':'门宫比和'}[r]
        text=f'这一层叫{name}。'+binding_answer(c['door']+'门',c['palace']+'宫',DOORS[c['door']],PALACES[c['palace']])
        if 'claim' in c:text=('这个名称可以保留。' if c['claim']==name else '这个名称应改正。')+text
        return text
    if kind=='marriage':
        if c['state']=='已具备':text='本条可成分支成立。两宫相生或比和与吉门吉格这两项条件均已具备。'
        elif c['state']=='未提供':text='现在还不能按本条判成。已知两宫相生或比和，但吉门吉格没有提供，需要核实；未知不等于明确不具备。'
        else:text='本条这一可成分支不成立。两宫关系虽满足，但题目明确不具备吉门吉格；这一个分支未成立，也不能单独断整件婚事必败。'
        if c['goal']=='date':text+='这些信息没有给出具体结婚日期，不能据此编出某一天。'
        return text
    if kind in ['loan','sale','retirement']:
        if kind=='loan':
            explanations={'值符克天乙':('有利','银行克借贷人'),'天乙克值符':('不利','借贷人克银行'),'天乙生值符':('有利','借贷人生银行'),'值符生天乙':('不利','银行生借贷人')}
            result,why=explanations[c['relation']];return f'按题中放贷规则，对银行{result}。值符代表银行，天乙代表借贷人，这里是{why}。'
        if kind=='sale':
            results={'日干生时干':'卖方恋货，不肯卖','时干生日干':'货恋人，卖不出去','日干克时干':'想卖快，但成交迟缓','时干克日干':'卖得快'}
            text=results[c['relation']]+'。本题日干代表卖方，时干代表货物，按'+c['relation']+'这一分支判断销售情况。'
            if c.get('profit'):text+='仅有这条销售关系，还不能认定利润高。'
            return text
        return ('不准退役。杜门代表领导，日干代表退役者；杜门生日干对应工作需要、领导有意留用。' if c['relation']=='杜门生日干'
                else '本条准予退役。杜门代表领导，日干代表退役者；题中'+c['relation']+'属于这条准退分支。')
    if kind=='formation':
        if c['goal']=='pair':return f'本段讨论的是天盘{c["upper"]}加地盘{c["lower"]}。'
        if c['goal']=='claim':return f'这个格名要改成本条的“{c["name"]}”。五行关系和原书给出的格名是两个问题，不能相互替换。'
        return f'本条叫“{c["name"]}”，对应天盘{c["upper"]}加地盘{c["lower"]}这个组合。'
    if kind=='literal_fact':return f'{c["role"]}取{c["symbol"]}。原文直接写明“{c["literal"]}”，这里按{c["domain_name"]}这一条取用。'
    if kind=='types':return '不能直接复制。天心是九星，乙奇是天干；星主要结合月令节气和落宫，天干主要结合生旺死绝并兼月令，同宫也要分别核对状态。'
    if kind=='liuhe':return '；'.join({'婚姻':'婚姻取媒人','诉讼':'诉讼取证人或证据','交易求财':'交易求财取经纪人或中介人'}[s] for s in c['domains'])+'。同一符号按所问领域取用。'
    raise ValueError('Unknown explicit teaching contract')


def build(source,boards):
    if source['schema']!=before.SCHEMA:raise ValueError('Frozen V64 material source required')
    units=deepcopy(source['units']);records=[];train=[r for r in source['records'] if r['split']=='train']
    def family(skill):return deepcopy(next(r['provenance']['support'] for r in train if r['skill']==skill))
    supports={'identity':family('identity'),'relation':family('direction'),'door':family('door'),'objects':family('objects'),
              'marriage':family('marriage'),'loan':family('loan'),'sale':family('sale'),'retirement':family('retirement'),
              'types':family('types'),'liuhe':family('liuhe'),'exam':family('exam')}
    def literal(fragment):
        candidates=[u for u in units.values() if fragment in u['original_quote']]
        if not candidates:raise ValueError('Missing original literal source '+fragment)
        u=min(candidates,key=lambda u:len(u['original_quote']));return [{'source_id':u['id'],'quote':u['original_quote']}]
    supports['attribute']=literal('[图7 后天九宫的五行]')
    def add(c,q,fam,*,style=0,split='train',history=None,initial=None,domain='foundation',support=None):
        support=deepcopy(support or supports[fam]);text=answer(c)
        if style==2:q+='\n参考原文：\n'+'\n'.join(s['quote'] for s in support)
        messages=deepcopy(history) if history else [{'role':'system','content':SYSTEM}]
        messages += [{'role':'user','content':q},{'role':'assistant','content':text}]
        key=d.digest(d.dumps([messages,split]))[:24]
        records.append({'id':'v65:'+key,'split':split,'domain':domain,'format':'source_binding_dialogue','stage':'application','task':'binding_and_goal',
          'skill':c['kind'],'group':'rule:'+fam,'question':messages[1]['content'],'answer':text,'messages':messages,
          'source_ids':sorted({s['source_id'] for s in support}),'case_id':None,'case_ids':[],
          'source_mode':'book_rule_generated_teaching_not_historical_outcome','checks':None,
          'provenance':{'support':support,'lesson_contract':c,'initial_contract':initial,'teacher_is_model_output':False,
             'rule_generated_not_observed_case':True,'source_book_definition_only':True}})
    for p,e in PALACES.items():
        for j,q in enumerate([f'{p}宫本身是什么五行？',f'先提取属性：{p}宫属什么？',f'请核对{p}宫的固定属性，不先判断吉凶。',f'只查九宫属性表，{p}这一宫怎么归类？']):
            add({'kind':'attribute','object':p+'宫','element':e},q,'attribute',style=j%3)
    for door,e in DOORS.items():
        for j in range(3):add({'kind':'attribute','object':door+'门','element':e},[f'{door}门的五行是什么？',f'只查八门属性：{door}门属什么？',f'请先说明{door}门自身属性。'][j],'door',style=j)
    for q in ['天心星本身是什么五行？','先不看落宫，天心这颗星属什么？','天心在震宫时，星自身和震宫是不是都属木？']:
        add({'kind':'attribute','object':'天心星','element':'金'},q,'objects',support=literal('天心星属金'))
    for i,(x,y) in enumerate(product(ELEMENTS,repeat=2)):
        for j in range(4):
            c={'kind':'relation','a':'甲方' if j%2==0 else '前者','b':'乙方' if j%2==0 else '后者','x':x,'y':y}
            q=[f'甲方属{x}、乙方属{y}。只比较两者，甲方对乙方是什么关系？',f'前者{x}、后者{y}。请把属性和作用方向说明白。',
               '核对这两项：'+d.dumps({'甲方':x,'乙方':y})+'。给出关系和依据。',f'两个对象依次是{x}与{y}，从前者看后者，应怎么读？'][j]
            add(c,q,'relation',style=j%3)
            for correct in [True,False]:
                claim=relation(x,y) if correct else [s for s in RELATIONS if s!=relation(x,y)][(i+j)%4]
                q=[f'有人把甲方{x}对乙方{y}标成“{claim}”。请核对标注，并说明依据。',f'前者{x}、后者{y}，现标注前者对后者“{claim}”。这行怎么处理？',
                   '待核对记录：'+d.dumps({'甲方':x,'乙方':y,'甲方对乙方':claim})+'。保留还是修改？',f'前者属{x}，后者属{y}。有人认为“{claim}”，请给出你的核对结论和正确方向。'][j]
                add(c|{'kind':'verify','claim':claim},q,'relation',style=j%3)
        for j in range(2):
            wrong=[s for s in RELATIONS if s!=relation(x,y)][i%4]
            c={'kind':'repair','a':'甲方','b':'乙方','x':x,'y':y}
            first,second=(relation(x,y),wrong) if j==0 else (wrong,relation(x,y))
            add(c,f'甲方{x}、乙方{y}。一段回答先说“{first}”，后面又说“{second}”。请改成方向和理由一致的一段话。'+('别只纠正首句。' if j else ''),'relation')
    outer=[p for p in PALACES if p!='中']
    for p,q in product(outer,repeat=2):
        for j in range(3):
            c={'kind':'objects','a':p+'宫','b':q+'宫','x':PALACES[p],'y':PALACES[q],'goal':'relation'}
            text=[f'只比较{p}宫与{q}宫，从前一个宫看后一个宫，关系是什么？',f'对象A是{p}宫，对象B是{q}宫。先识别属性，再说明A对B的关系。',
                  '当前比较对象：'+d.dumps({'A宫':p,'B宫':q})+'。请给出两宫关系与理由。'][j]
            add(c,text,'objects',style=j)
            add(c|{'goal':'extract'},text+'这一步先只提取比较对象和属性，不计算关系。','objects',style=j)
    for i,(door,palace) in enumerate(product(DOORS,outer)):
        c={'kind':'door','door':door,'palace':palace}
        for style in range(2):
            q=(f'{door}门落{palace}宫，按本条叫门迫、门受制还是其他关系？请说明依据。' if style==0
               else '门宫资料：'+d.dumps({'门':door,'宫':palace})+'。请先给这一层名称，再说明双方属性和方向。')
            add(c,q,'door',style=style)
        if i%2==0:
            correct={'克':'门迫','受克':'门受制','生':'门生宫','受生':'宫生门','比和':'门宫比和'}[relation(DOORS[door],PALACES[palace])]
            for claim in [correct,'门迫' if correct!='门迫' else '门受制']:
                add(c|{'claim':claim},f'有人把{door}门落{palace}宫称为“{claim}”。请核对名称、属性和方向，整段要一致。','door',style=2)
    for n,board in enumerate(boards):
        heart,illness=board['stars']['天心星'],board['stars']['天芮星']
        for j in range(4):
            prefix=(f'已核定天心在{heart}宫、天芮在{illness}宫。' if j%2==0 else d.dumps({'当前星位':{'天心星':heart,'天芮星':illness}}))
            for goal in ['palaces','reverse','star','locations']:
                if goal=='locations':c={'kind':'locations','positions':[['天心星',heart],['天芮星',illness]],'board_id':board['base_id']};query='只复述当前落宫。'
                elif goal=='star':c={'kind':'objects','a':'天心星','b':heart+'宫','x':'金','y':PALACES[heart],'goal':'relation','board_id':board['base_id']};query='这次只比较天心星自身与它所在的宫。'
                else:
                    p,q=(heart,illness) if goal=='palaces' else (illness,heart);s,t=('天心星','天芮星') if goal=='palaces' else ('天芮星','天心星')
                    c={'kind':'objects','a':s+'所在的'+p+'宫','b':t+'所在的'+q+'宫','x':PALACES[p],'y':PALACES[q],'goal':'relation','board_id':board['base_id']}
                    query=f'这次只比较{s}所在宫对{t}所在宫的关系。'
                add(c,prefix+query,'objects',style=j%3)
        first={'kind':'objects','a':'天心星','b':heart+'宫','x':'金','y':PALACES[heart],'goal':'relation','board_id':board['base_id']}
        last={'kind':'objects','a':'天心星所在的'+heart+'宫','b':'天芮星所在的'+illness+'宫','x':PALACES[heart],'y':PALACES[illness],'goal':'relation','board_id':board['base_id']}
        for initial,final in [(first,last),(last,first)]:
            query=f'天心在{heart}宫，天芮在{illness}宫。先比较'+('天心星与本宫。' if initial is first else '两颗星所在的两个宫。')
            history=[{'role':'system','content':SYSTEM},{'role':'user','content':query},{'role':'assistant','content':answer(initial)}]
            change='现在星位不变，改问'+('天心星自身与它所在宫的关系。' if final is first else '天心所在宫对天芮所在宫的关系；两个比较对象都是宫。')
            add(final,change,'objects',history=history,initial=initial)
        lines=[];compact=[]
        for p in ['巽','离','坤','震','中','兑','艮','坎','乾']:
            names=[s for s,g in board['stars'].items() if g==p]
            lines.append(f'{p}{PALACE_NUMBER[p]}宫：'+('、'.join(names) if names else '未列星位'))
            compact.append(f'{p}{PALACE_NUMBER[p]}宫：星行'+('、'.join(s[1] for s in names) if names else '未列'))
        for s,t in [('天任星','天英星'),('天辅星','天柱星'),('天心星','天芮星'),('天蓬星','天冲星')]:
            for goal in ['locations','relation']:
                c={'kind':'board','positions':[[s,board['stars'][s]],[t,board['stars'][t]]],'goal':goal,'board_id':board['base_id']}
                query=f'从下面的程序星位摘录找出{s}和{t}，'+('只说各在哪一宫。' if goal=='locations' else '比较它们所在两个宫的关系并给理由。')
                add(c,query+'\n'+'\n'.join(lines),'objects')
                add(c,query+'以下星行用九星单字简称，仅摘录星位：\n'+'\n'.join(compact),'objects')
    contexts=[]
    stems='甲乙丙丁戊己庚辛壬癸'
    for j in range(12):contexts.append(({field:stems[(j+offset)%10] for field,offset in zip(['年干','月干','日干','时干'],[0,3,6,8])},stems[(j+5)%10] if j%2==0 else None))
    for role in ROLES:
        for j,(fields,birth) in enumerate(contexts):
            for style in range(3):
                f=deepcopy(fields);field=ROLES[role][0]
                if style==1 and j%3==0:f[field]=None
                if style==2 and j%3==1:f.pop(field)
                c={'kind':'identity','role':role,'fields':f,'birth':birth}
                who=ROLE_PHRASE[role];data=d.dumps({'起局字段':f,'患者可靠出生年干':birth})
                q=[f'实际关系：{who}。{data}。主用神取谁，出生资料怎么用？',f'请处理这一次{who}：{data}。已给和未给的字段分别怎样使用？',
                   f'{who}，不要把家庭称谓代替求测关系。{data}。说明本问的主用神和必要依据。'][style]
                add(c,q,'identity',style=style,domain='health')
        for j,(fields,birth) in enumerate(contexts[:3]):
            for correct in [True,False]:
                field=ROLES[role][0];claim=field if correct else next(f for f in ['日干','年干','月干','时干'] if f!=field)
                c={'kind':'identity_verify','role':role,'fields':fields,'birth':birth,'claim':claim}
                add(c,ROLE_PHRASE[role]+'，资料'+d.dumps(fields)+f'，患者出生年干{birth or "未核实"}。有人建议取{claim}代表患者，这个建议应保留还是改正？','identity',domain='health')
        first={'kind':'identity','role':role,'fields':contexts[2][0],'birth':None};last=first|{'birth':'庚'}
        history=[{'role':'system','content':SYSTEM},{'role':'user','content':ROLE_PHRASE[role]+'，起局字段'+d.dumps(first['fields'])+'，患者出生年未核实。先怎么取？'}, {'role':'assistant','content':answer(first)}]
        add(last,'求测人和患者都没变，只补充患者可靠出生年干庚。主用神是否改变，出生资料怎么用？','identity',history=history,initial=first,domain='health')
        if role!='本人':
            for j in range(2):
                first={'kind':'identity','role':role,'fields':contexts[j][0],'birth':'壬' if j else None};last=first|{'role':'本人'}
                history=[{'role':'system','content':SYSTEM},{'role':'user','content':ROLE_PHRASE[role]+'，起局字段'+d.dumps(first['fields'])+f'，这位患者出生年干{first["birth"] or "未知"}。先取谁？'},{'role':'assistant','content':answer(first)}]
                add(last,f'现在由这位{role}亲自问自己的健康，不再由我代问。还是同一个患者，起局字段不变，年命资料不变。这次主用神怎样改？','identity',history=history,initial=first,domain='health')
        first={'kind':'identity','role':role,'fields':deepcopy(contexts[0][0]),'birth':'壬'}
        last=deepcopy(first);last['fields'][ROLES[role][0]]='辛'
        history=[{'role':'system','content':SYSTEM},{'role':'user','content':ROLE_PHRASE[role]+'，资料'+d.dumps(first['fields'])+'，患者出生年干壬。先怎么取？'},{'role':'assistant','content':answer(first)}]
        add(last,f'人物关系和出生年干壬都不变，刚核实起局{ROLES[role][0]}应该是辛，原记录录错了。现在主用神和年命各怎样处理？','identity',history=history,initial=first,domain='health')
    for q in ['只知道家里有位爸爸生病，尚未说明谁来求测，能固定用神吗？','我帮别人转述他父亲的病，但实际求测人还没确认，先问什么？','只说患者是母亲，还没说明本人问还是代问，取用怎样确认？']:
        add({'kind':'identity_ambiguous'},q,'identity',domain='health')
    for self,birth,style in product([True,False],[None,'癸','庚'],range(2)):
        c={'kind':'exam','self':self,'birth':birth};who='考生亲自问自己的考试' if self else '父母代问自己的子女考试'
        add(c,who+f'，考生出生年干{birth or "还未核实"}。这一章先以什么代表实际考生，出生资料怎样使用？'+('请说明关系和取用。' if style else ''),'exam',style=style,domain='education')
    for rel,state,goal,style in product(['相生','比和'],['已具备','未提供','明确不具备'],['can','date'],range(4)):
        c={'kind':'marriage','state':state,'goal':goal};condition={'已具备':'已经核实有吉门吉格','未提供':'吉门吉格的信息尚未提供','明确不具备':'已经核实没有吉门吉格'}[state]
        question=f'乙庚所在宫{rel}，{condition}。'+('就本条分支能否判成，请说明条件怎样生效。' if goal=='can' else '这些资料能否确定具体哪一天结婚？先说明本条条件，再回答日期。')
        if style%2:question=d.dumps({'乙庚宫关系':rel,'吉门吉格状态':state})+'。'+question
        add(c,question,'marriage',style=style%3,domain='marriage')
    for first_state,last_state in product(['已具备','未提供','明确不具备'],repeat=2):
        if first_state==last_state:continue
        first={'kind':'marriage','state':first_state,'goal':'can'};last=first|{'state':last_state}
        history=[{'role':'system','content':SYSTEM},{'role':'user','content':f'乙庚两宫比和，吉门吉格的状态是{first_state}。本条能否判成？'},{'role':'assistant','content':answer(first)}]
        add(last,f'刚核实需要更新资料：两宫还是比和，但吉门吉格现在确认的状态是{last_state}。请按最新资料重判这条分支。','marriage',history=history,initial=first,style=2,domain='marriage')
    for kind,relations in [('loan',['值符克天乙','天乙克值符','天乙生值符','值符生天乙']),('sale',['日干生时干','时干生日干','日干克时干','时干克日干']),('retirement',['杜门生日干','杜门克日干','杜门与日干比和'])]:
        for rel,style in product(relations,range(6)):
            c={'kind':kind,'relation':rel,'profit':style%2==1}
            q={'loan':f'值符是银行、天乙是借贷人，现在{rel}。请从银行放贷方说有利还是不利，并解释是谁作用于谁。',
               'sale':f'日干是卖方、时干是货物，现在{rel}。卖货情况怎样？'+('能据此认定利润高吗？' if c['profit'] else '请说明这一层判断依据。'),
               'retirement':f'杜门代表领导、日干代表申请退役者，现在{rel}。本条准退还是不准退，原因是什么？'}[kind]
            q=[q,'这次只处理本条关系。'+q,q+'请把结论和依据一起给出。',d.dumps({'领域':kind,'当前关系':rel})+'。'+q,'我重新确认一下：'+q,q+'不要只复述题面关系，需回答所问结果。'][style]
            add(c,q,kind,style=style%3,domain=kind)
    named=[('戊','戊','伏吟','戊/戊，甲甲比肩，名谓“伏吟”'),('丁','壬','奇仪相合','丁/壬，天盘丁，地盘壬，名“奇仪相合”'),('庚','戊','值符伏宫','故名“值符伏宫”')]
    for upper,lower,name,fragment in named:
        support=literal(fragment)
        for goal,style in product(['name','pair','claim'],range(4)):
            c={'kind':'formation','upper':upper,'lower':lower,'name':name,'goal':goal}
            q={'name':f'请查这一段的具体格名：天盘{upper}加地盘{lower}叫什么？','pair':'先只从原文提取天盘与地盘的天干分别是什么，这一步不解释格局名称。',
               'claim':f'有人把天盘{upper}加地盘{lower}的具体格名写成“比和格”。请依据本段核对这个名称。'}[goal]
            add(c,(['按这一段回答：','现在问的是具体目标：','','请把概念层次分开：'][style])+q,'relation',style=2,support=support)
    for r in [r for r in train if r['skill']=='source_fact']:
        c=deepcopy(r['provenance']['lesson_contract']);c['kind']='literal_fact'
        add(c,f'在{c["domain_name"]}这一条中，应当用什么代表{c["role"]}？给出直接取用和对应原句。','relation',support=r['provenance']['support'],style=2,domain='mixed_domain')
    for palace in outer:
        add({'kind':'types'},f'天心星和乙奇同在{palace}宫，能否把一个的旺衰或墓态直接给另一个？分别依据什么判断？','types',domain='health')
    for a in ['婚姻','诉讼','交易求财']:
        add({'kind':'liuhe','domains':[a]},f'这次只问{a}，六合应代表谁？','liuhe',domain=a)
        for b in ['婚姻','诉讼','交易求财']:
            if b!=a:add({'kind':'liuhe','domains':[a,b]},f'请把{a}与{b}的取用分开：六合在两者分别代表谁？','liuhe',domain='mixed_domain')
    for original in [r for r in train if r['skill']=='source_dialogue']:
        r=deepcopy(original);r['id']='v65:replay:'+r['id'];r['provenance']['replayed_original_source_dialogue_unchanged']=True;records.append(r)
    # New wording is development only. The knowledge and old monitors are exposed.
    for role in ['本人','父亲','同事','兄弟','儿子','女儿']:
        c={'kind':'identity','role':role,'fields':{'年干':'壬','月干':'己','日干':'丁','时干':'乙'},'birth':'癸'}
        add(c,'请重新按实际关系处理：'+ROLE_PHRASE[role]+'。已知'+d.dumps(c['fields'])+'，患者可靠出生年干癸。该先用哪个字段，哪些资料另看？','identity',split='validation',domain='health')
    for p,q in [('艮','乾'),('震','坎'),('离','兑')]:
        for reverse in [False,True]:
            a,b=(q,p) if reverse else (p,q);c={'kind':'objects','a':a+'宫','b':b+'宫','x':PALACES[a],'y':PALACES[b],'goal':'relation'}
            add(c,f'本问只比较两个宫：第一个是{a}宫，第二个是{b}宫。第一个对第二个怎样作用？请把属性和理由一起说清。','objects',split='validation')
    for i,(x,y) in enumerate([('水','火'),('金','木'),('木','金'),('水','水'),('土','火')]):
        c={'kind':'verify','a':'前项','b':'后项','x':x,'y':y,'claim':['生','受生','受克','比和','克'][i]}
        add(c,'请审核这条记录：'+d.dumps({'前项':x,'后项':y,'前项对后项':c['claim']})+'。给出审核结论及正确方向，不遗漏后半句理由。','relation',split='validation')
    for state in ['已具备','未提供','明确不具备']:
        add({'kind':'marriage','state':state,'goal':'can'},f'核对婚姻规则分支：乙庚两宫相生，吉门吉格的状态是“{state}”。本条结论能给到哪一步？','marriage',style=2,split='validation',domain='marriage')
    for rel in ['值符生天乙','天乙生值符']:
        add({'kind':'loan','relation':rel},f'银行方由值符代表，借款人由天乙代表，本次{rel}。请同时回答银行方结果和关系理由。','loan',style=2,split='validation')
    for upper,lower,name,fragment in named[:2]:
        add({'kind':'formation','upper':upper,'lower':lower,'name':name,'goal':'name'},f'看这一段，{upper}在天盘、{lower}在地盘，原文给出的专门名称是什么？不要改成一般五行关系。','relation',style=2,split='validation',support=literal(fragment))
    unique={}
    for r in records:
        key=d.dumps(r['messages'][:-1])
        if key in unique:
            if r['answer']!=unique[key]['answer'] or r['split']!=unique[key]['split']:raise ValueError('Conflicting or cross-split exact prompt')
        else:unique[key]=r
    out=deepcopy(source);out.update(schema=SCHEMA,records=list(unique.values()),physical_star_boards=boards,new_books_used=False,model_outputs_used_as_teachers=False)
    out['dispositions']=[{'old_id':r['id'],'action':'replay_unchanged_source_dialogue' if r['skill']=='source_dialogue' else 'replace_with_explicit_bindings','source_ids':r['source_ids']} for r in train]
    usage=Counter(s for r in out['records'] if r['split']=='train' for s in r['source_ids'])
    out['material_usage']=[{'source_id':s,'new_train_lessons':usage[s],'disposition':'used' if usage[s] else 'preserved'} for s in units]
    out['summary']={'schema':SCHEMA,'train':sum(r['split']=='train' for r in out['records']),'development':sum(r['split']=='validation' for r in out['records']),
        'skills':dict(Counter(r['skill'] for r in out['records'] if r['split']=='train')),'original_source_units':len(units),'used_source_units':len(usage),
        'source_case_registry_entries':len(source['case_splits']),'case_linked_cores':sum(bool(r.get('case_ids')) for r in out['records'] if r['split']=='train'),
        'all_old_case_splits_preserved':True,'full_two_book_case_training':False,'new_blind_benchmark':False,'source_models_used_as_teachers':False}
    validate(out,source);return out


def validate(pack,source):
    if pack['units']!=source['units'] or pack['case_splits']!=source['case_splits'] or pack['schema']!=SCHEMA:raise ValueError('Original source or event boundaries changed')
    if pack.get('model_outputs_used_as_teachers') or pack.get('new_books_used') or pack.get('old_manual_scores_used'):raise ValueError('Forbidden teacher source')
    boards={b['base_id']:b['stars'] for b in pack['physical_star_boards']};seen=set()
    for b in pack['physical_star_boards']:
        if set(b)!={'base_id','stars'}:raise ValueError('Only literal star positions allowed')
    for r in pack['records']:
        key=d.dumps(r['messages'][:-1])
        if key in seen:raise ValueError('Repeated exact full prompt')
        seen.add(key);p=r['provenance'];c=p.get('lesson_contract')
        if p.get('replayed_original_source_dialogue_unchanged'):
            original=next(q for q in source['records'] if q['id']==r['id'].removeprefix('v65:replay:'))
            if original['messages']!=r['messages'] or original['split']!=r['split']:raise ValueError('Original case dialogue changed')
            continue
        if r['answer']!=answer(c) or r['messages'][-1]['content']!=r['answer'] or r['messages'][0]['content']!=SYSTEM:raise ValueError('Teacher contradicts explicit binding')
        if set(r['source_ids'])!={s['source_id'] for s in p['support']}:raise ValueError('Source list differs')
        for s in p['support']:
            if d.compact(s['quote']) not in d.compact(pack['units'][s['source_id']]['original_quote']):raise ValueError('Original quote changed')
        if c.get('board_id'):
            actual=boards[c['board_id']]
            if c['kind'] in ['board','locations'] and any(actual[s]!=g for s,g in c['positions']):raise ValueError('Wrong literal position')
            if c['kind']=='objects':
                if c['a']=='天心星' and (c['b']!=actual['天心星']+'宫' or c['x']!='金' or c['y']!=PALACES[actual['天心星']]):raise ValueError('Wrong star-to-palace binding')
                for key in ['a','b']:
                    for star in actual:
                        if c[key].startswith(star+'所在的') and c[key]!=star+'所在的'+actual[star]+'宫':raise ValueError('Object differs from current physical palace')
        if c['kind']=='objects':
            for key,element in [('a','x'),('b','y')]:
                obj=c[key];expected='金' if obj=='天心星' else PALACES.get(obj[-2]) if obj.endswith('宫') else None
                if expected is None or c[element]!=expected:raise ValueError('Object property differs from the source table')
        if c['kind']=='identity' and not p.get('initial_contract') and not any(ROLE_PHRASE[c['role']] in m['content'] for m in r['messages'] if m['role']=='user'):
            raise ValueError('Explicit actual questioner-patient relationship required')
        if p.get('initial_contract') and r['messages'][2]['content']!=answer(p['initial_contract']):raise ValueError('Wrong historical assistant turn')
        if c['kind']=='formation' and not any(c['name'] in s['quote'] for s in p['support']):raise ValueError('Named formation lacks original literal support')
        if c['kind']=='formation' and not any(s['quote'].lstrip().startswith(c['upper']+'/'+c['lower']) for s in p['support']):raise ValueError('Formation name is bound to a different stem pair')
        if c['kind']=='literal_fact' and not any(c['literal'] in s['quote'] for s in p['support']):raise ValueError('Source fact lacks its quoted role-symbol binding')
    if len([r for r in pack['records'] if r['split']=='validation'])!=24:raise ValueError('Exactly 24 new known-development checks required')
    return True
