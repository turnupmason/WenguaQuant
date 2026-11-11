"""Rule-derived teaching, explicit objects, balanced identity and correction tasks.

Only original TRAIN evidence is a teacher. Saved model answers are never read.
"""
from copy import deepcopy
from collections import Counter
from itertools import product, combinations
from pathlib import Path
import json,re
from wengua_course import data as d
from wengua_course_v63 import dataset as old

SCHEMA='wengua_source_conditional_repair_v64'
SYSTEM=old.SYSTEM
MODULES=old.MODULES
PALACES={'坎':'水','坤':'土','震':'木','巽':'木','中':'土','乾':'金','兑':'金','艮':'土','离':'火'}
DOORS={'休':'水','生':'土','伤':'木','杜':'木','景':'火','死':'土','惊':'金','开':'金'}
ROLES={'本人':('日干','患者本人',None),'父亲':('年干','父亲','乾'),'母亲':('年干','母亲','坤'),
       '同事':('月干','同事',None),'兄弟':('月干','兄弟',None),'儿子':('时干','儿子',None),'女儿':('时干','女儿',None)}
ATLAS='data/atlas/shijia_zhuanpan_physical_1080_v1.jsonl'


def relation(a,b):
    if a==b:return '比和'
    if d.GENERATES[a]==b:return '生'
    if d.CONTROLS[a]==b:return '克'
    return '受生' if d.GENERATES[b]==a else '受克'


def direction(a,b,x,y):
    r=relation(x,y)
    if r=='比和':return f'{a}与{b}同属{x}，是比和。'
    if r=='生':return f'{x}生{y}，所以{a}生{b}。'
    if r=='克':return f'{x}克{y}，所以{a}克{b}。'
    if r=='受生':return f'{y}生{x}，所以{a}受{b}所生。'
    return f'{y}克{x}，所以{a}受{b}所克。'


def answer(c):
    kind=c['kind']
    if kind in ('identity','identity_verify'):
        field,who,palace=ROLES[c['role']];value=c['fields'].get(field)
        result=f'先取本次起局的{field}'+(f'{value}' if value else '')+f'代表{who}'
        result+=f'，兼看{palace}宫。' if palace else '。'
        if value is None:result+=f'具体{field}还没提供，暂不填入某个天干。'
        if kind=='identity_verify':return ('这个取用对。' if c['claim']==field else '这个取用不对。')+result
        if c.get('focus'):return result
        birth=c['birth'];result+=f'出生年命{birth}另作参看。' if birth else '出生年未核实，年命先留空。'
        return result
    if kind=='identity_ambiguous':return '先确认是患者本人问，还是别人代问，以及实际求测人与患者的关系。本人取日干，子女代问父母取年干；出生年命另按可靠出生资料参看。'
    if kind in ('direction','verify_direction','consistent_direction'):
        result=direction(c['a'],c['b'],c['x'],c['y'])
        if kind=='verify_direction':result=('这句判断对。' if c['claim']==relation(c['x'],c['y']) else '这句判断不对。')+result
        if kind=='consistent_direction':result+='同一层比较中，不能把它同时判成另一个方向。'
        return result
    if kind=='door':
        door,palace=c['door'],c['palace'];r=relation(DOORS[door],PALACES[palace]);name={'克':'门迫','受克':'门受制','生':'门生宫','受生':'宫生门','比和':'比和'}[r]
        x,y=DOORS[door],PALACES[palace]
        if r=='比和':result=f'{door}门与{palace}宫同属{x}，是门宫比和。'
        else:
            rule={'生':f'{x}生{y}','克':f'{x}克{y}','受生':f'{y}生{x}','受克':f'{y}克{x}'}[r]
            phrase={'克':'门克宫，称门迫','受克':'宫克门，称门受制','生':'门生宫','受生':'宫生门'}[r]
            result=f'{door}门属{x}，{palace}宫属{y}；{rule}，所以{phrase}。'
        if 'claim' in c:result=('这句判断对。' if c['claim']==name else '这句判断不对。')+result
        return result
    if kind=='objects':
        a,b=c['heart'],c['illness'];scope=c['scope']
        if scope=='locations':return f'天心星当前在{a}宫，天芮星当前在{b}宫。'
        if scope=='heart_and_palace':return direction('天心星',f'{a}宫','金',PALACES[a])
        if scope=='palaces':return direction(f'天心所在的{a}宫',f'天芮所在的{b}宫',PALACES[a],PALACES[b])
        if scope=='reverse_palaces':return direction(f'天芮所在的{b}宫',f'天心所在的{a}宫',PALACES[b],PALACES[a])
        raise ValueError('Unknown comparison objects')
    if kind=='marriage':
        if c['condition']=='已具备':
            return ('这符合本条恋爱可成、婚姻美满的条件，但还没有给出结婚日期。' if c['goal']=='日期'
                    else '两宫相生或比和，又逢吉门吉格，符合本条可成分支；双方年命可以另外参看。')
        prefix='目前还不能确定结婚日期。' if c['goal']=='日期' else ''
        if c['condition']=='未提供':return prefix+'还需要核实吉门吉格。仅凭两宫相生或比和，尚不能按本条判成；资料未给，也不等于婚事必败。'
        return prefix+'既然明确不具备吉门吉格，就不满足本条这一可成分支；单凭这一处不足，还不能断整件婚事必败。'
    if kind=='loan':
        rel=c['relation'];good=rel in ('值符克天乙','天乙生值符');why={'值符克天乙':'银行克借贷人','天乙克值符':'借贷人克银行','天乙生值符':'借贷人生银行','值符生天乙':'银行生借贷人'}[rel]
        return f'按这条放贷规则，对银行'+('有利' if good else '不利')+f'。值符是银行、天乙是借贷人，这里是{why}。'
    if kind=='sale':
        reasons={'日干生时干':'卖方恋货、不肯卖','时干生日干':'货恋人，卖不出去','日干克时干':'虽想卖快，成交仍迟缓','时干克日干':'卖得快'}
        return f'按这条卖货规则，{reasons[c["relation"]]}。'+('这条关系只给了销售情况，还不足以判断利润。' if c.get('profit') else '')
    if kind=='retirement':
        return ('领导有意留用、工作需要，按本条不准退役。' if c['relation']=='杜门生日干' else '按本条准予退役。'+('这里是领导宫克退役者日干。' if c['relation']=='杜门克日干' else '这里杜门与日干比和。'))
    if kind=='liuhe':return '；'.join({'婚姻':'在婚姻这一条里，六合代表媒人','诉讼':'在诉讼这一条里，六合代表证人或证据','交易求财':'在交易求财这一条里，六合代表经纪人或中介人'}[s] for s in c['domains'])+'。'
    if kind=='types':return '不能直接复制。天心是九星，乙奇是天干；九星主要看月令节气和落宫，天干主要看生旺死绝并兼月令，同宫也要分别核对状态。'
    if kind=='exam':return ('考生本人来问，以日干为考生。' if c['self'] else '父母代问子女考试，以时干为考生。')+('出生年未核实，先不取年命。' if not c['birth'] else f'可靠出生年命{c["birth"]}可另作参看。')
    if kind=='source_fact':
        if c['quote_requested']:return f'有明确规定。原文写的是“{c["literal"]}”。'
        if c['role']=='被告':return '按这条官司诉讼规则，天乙代表被告；这里的天乙指值符落宫的地盘之星。'
        return f'按这条{c["domain_name"]}规则，{c["symbol"]}代表{c["role"]}。'
    if kind=='source_absence':
        return f'这段{c["domain_name"]}原文没有给出{c["role"]}。仅凭这一段不能作为对应结论的依据，还需要查{c["needed"]}。'
    raise ValueError('Unknown teaching contract')


def placements(app_root):
    """Extract literal locations only; discard every score, void and 81-table field."""
    path=Path(app_root)/ATLAS;raw=[json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    if len(raw)!=1080 or len({x['base_id'] for x in raw})!=1080:raise ValueError('Original physical 1080 source required')
    selected={}
    for board in raw:
        if not board.get('physical_only') or {p['palace'] for p in board['palaces']}!=set(range(1,10)):
            raise ValueError('Literal nine-palace physical templates required')
        for star in ['天心星','天芮星']:
            if sum(s['star']==star for p in board['palaces'] for s in p.get('heaven_stars',[]))!=1:
                raise ValueError('Ambiguous or missing literal star location')
        positions={s['star']:p['gua'] for p in board['palaces'] for s in p.get('heaven_stars',[])}
        pair=(positions['天心星'],positions['天芮星'])
        selected.setdefault(pair,{'base_id':board['base_id'],'heart':pair[0],'illness':pair[1]})
    if len(selected)!=8:raise ValueError('Expected eight literal star rotations')
    return sorted(selected.values(),key=lambda x:(x['heart'],x['illness']))


def build(source,physical):
    if source['schema']!=old.SCHEMA:raise ValueError('Frozen V63 source required')
    pool=[r for r in source['records'] if r['split']=='train'];units=deepcopy(source['units']);records=[]
    donors={k:next(r for r in pool if r['skill']==k) for k in ['identity','direction','door_relation','typed_palace','branch_conditions','symbol_types','domain_roles']}
    for name,prefix in [('loan','v62:loan_'),('sale','v62:sale_'),('retirement','v62:retire_')]:donors[name]=next(r for r in pool if r['id'].startswith(prefix))
    donors['exam']=next(r for r in pool if (r['provenance'].get('conditions') or {}).get('实际求测关系')=='考生本人')
    donors['domain_roles']=max((r for r in pool if r['skill']=='domain_roles'),key=lambda r:len(r['source_ids']))
    def add(c,q,family,*,split='train',context=False,messages=None,initial=None,domain='foundation'):
        donor=donors[family];support=deepcopy(donor['provenance']['support']);target=answer(c)
        if context:q+='\n参考原文：\n'+'\n'.join(x['quote'] for x in support)
        chat=deepcopy(messages) if messages else [{'role':'system','content':SYSTEM}]
        chat += [{'role':'user','content':q},{'role':'assistant','content':target}]
        row=deepcopy(donor);key=d.digest(d.dumps([c,chat,split]))[:24]
        row.update(id='v64:'+key,question=chat[1]['content'],answer=target,messages=chat,source_ids=sorted({x['source_id'] for x in support}),
            case_id=None,case_ids=[],split=split,format='conditional_source_dialogue',stage='application',task='conditional_repair',
            skill=c['kind'],domain=domain,group='rule:'+family,source_mode='source_rule_generated_not_observed_case',checks=None,
            provenance={'support':support,'source_train_ids':[donor['id']],'teacher_is_model_output':False,'lesson_contract':c,
                'initial_contract':initial,'source_was_train':True,'rule_generated_teaching_scene':True,'not_observed_outcome':True})
        records.append(row)
    contexts=[({'年干':'甲','月干':'丙','日干':'辛','时干':'丁'},'庚'),({'年干':'庚','月干':'戊','日干':'壬','时干':'乙'},None),
              ({'年干':'丙','月干':'辛','日干':'戊','时干':'癸'},'乙'),({'年干':'壬','月干':'甲','日干':'己','时干':'丙'},None)]
    for n,(role,(field,who,pal)) in enumerate(ROLES.items()):
        for j,(times,birth) in enumerate(contexts):
            for style in range(3):
                fields=deepcopy(times)
                if style==2:fields[field]=None
                c={'kind':'identity','role':role,'fields':fields,'birth':birth}
                rel='我自己问自己的健康' if role=='本人' else f'我替{who}问健康'+('，我是患者的子女' if role in ('父亲','母亲') else '，我是患者的父母' if role in ('儿子','女儿') else '，我们是同辈')
                if style==0:q=f'{rel}。起局资料是'+d.dumps(fields)+f'；出生年干是{birth or "未核实"}。先看哪个人物用神？这些字段分别怎么用？'
                elif style==1:q='请帮我把这次问病的人物关系分清。'+d.dumps({'实际求测关系':rel,'起局字段':fields,'可靠出生年干':birth})+'应该先取谁？'
                else:q=f'{rel}，但相应的{field}具体是什么字还没拿到。其他已知资料：'+d.dumps(fields)+f'；出生年干{birth or "不详"}。现在能先确定取用字段吗？'
                add(c,q,'identity',context=(n+j+style)%4==0,domain='health')
    for role,(field,who,pal) in ROLES.items():
        for j,(times,birth) in enumerate(contexts):
            fields=deepcopy(times)
            if j==1:fields[field]=None
            if j==2:fields.pop(field)
            if j==3:fields={'年干':times['年干']}
            c={'kind':'identity','role':role,'fields':fields,'birth':birth,'focus':True}
            relationship={'本人':'患者本人求测','父亲':'子女代问父亲','母亲':'子女代问母亲','同事':'同事之间代问','兄弟':'兄弟之间代问','儿子':'父母代问儿子','女儿':'父母代问女儿'}[role]
            q='这次只确定首要人物用神和兼看宫位。已知资料：'+d.dumps({'关系':relationship,'患者':who,'起局字段':fields,'出生年干':birth})+'。主用神取哪个字段？没有具体字的字段怎样处理？'
            add(c,q,'identity',domain='health')
        for correct in [False,True]:
            c={'kind':'identity_verify','role':role,'fields':contexts[0][0],'birth':contexts[0][1],
               'claim':field if correct else next(s for s in ['年干','日干','月干','时干'] if s!=field)}
            q=f'按这次{relationship}的关系，有人建议用{c["claim"]}代表患者。已给起局字段'+d.dumps(c['fields'])+f'，出生年干{c["birth"]}。这个建议对吗？请说明主用神。'
            add(c,q,'identity',domain='health')
    for role in list(ROLES)[1:]:
        for known in [False,True]:
            for reverse in [False,True]:
                fields={'年干':'戊','月干':'乙','日干':'庚','时干':'壬'};birth='丙' if known else None
                first={'kind':'identity','role':'本人' if reverse else role,'fields':fields,'birth':birth}
                last=first|{'role':role if reverse else '本人'}
                q1=('先问我的病' if reverse else f'刚才由我代问{role}的病')+'，起局字段'+d.dumps(fields)+f'，出生年干{birth or "还没核实"}。先取谁？'
                history=[{'role':'system','content':SYSTEM},{'role':'user','content':q1},{'role':'assistant','content':answer(first)}]
                q2=(f'换一个患者：现在我代问{role}，不是问我自己的病。' if reverse else f'现在改由这位{role}本人问自己的病，我不再代问。')+f'这位患者的出生年干{birth or "还没核实"}，仍用这些起局字段。这次主用神怎么改？'
                add(last,q2,'identity',messages=history,initial=first,domain='health')
    for role in ROLES:
        first={'kind':'identity','role':role,'fields':{'年干':'丁','月干':'乙','日干':'辛','时干':'戊'},'birth':None}
        last=first|{'birth':'己'}
        history=[{'role':'system','content':SYSTEM},{'role':'user','content':f'这次关系是{role}，起局字段'+d.dumps(first['fields'])+'，出生年还不知道，先取谁？'},{'role':'assistant','content':answer(first)}]
        add(last,'人物和代问关系都没有变，现在只补充可靠的出生年干己。主用神需要因此换掉吗？','identity',messages=history,initial=first,domain='health')
    for q in ['爸爸生病了，先看哪一个用神？','有一个人的父亲看病，谁来求测还没说明，能直接取年干吗？','只说患者在家里是父亲，能确定这次取用吗？','我还没说是自己问还是代问，人物用神能先定死吗？']:
        add({'kind':'identity_ambiguous'},q,'identity',domain='health')
    elements=['木','火','土','金','水'];claims=['生','克','受生','受克','比和']
    for i,(x,y) in enumerate(product(elements,repeat=2)):
        c={'kind':'direction','a':'甲方','b':'乙方','x':x,'y':y}
        add(c,f'甲方是{x}，乙方是{y}。甲方对乙方的五行关系怎么读？','direction',context=i%4==0)
        correct=relation(x,y);wrong=[z for z in claims if z!=correct][i%4]
        for claim in [correct,wrong]:
            add(c|{'kind':'verify_direction','claim':claim},f'有人把甲方{x}对乙方{y}写成“{claim}”。请核对这句，说明正确方向。','direction')
    for x,y in [('土','金'),('金','水'),('水','木'),('火','金'),('火','火')]:
        for first_claim in [relation(x,y),next(z for z in claims if z!=relation(x,y))]:
            second=next(z for z in claims if z!=first_claim)
            c={'kind':'consistent_direction','a':'甲方','b':'乙方','x':x,'y':y,'claims':[first_claim,second]}
            q=f'同一层只比较甲方{x}和乙方{y}。有一段回答先判甲方对乙方“{first_claim}”，后面又判“{second}”。请改成一段方向一致的回答。'
            add(c,q,'direction')
    for i,(door,palace) in enumerate(product(DOORS,[s for s in PALACES if s!='中'])):
        c={'kind':'door','door':door,'palace':palace}
        add(c,f'{door}门现在落{palace}宫。门和宫分别是什么五行，这一层是什么关系、叫什么？','door_relation',context=i%5==0)
        if i%3==0:
            correct={'克':'门迫','受克':'门受制','生':'门生宫','受生':'宫生门','比和':'比和'}[relation(DOORS[door],PALACES[palace])]
            claim=correct if i%2==0 else ('门受制' if correct!='门受制' else '门迫')
            add(c|{'claim':claim},f'我把{door}门落{palace}宫标作“{claim}”，这个标注要改吗？','door_relation')
    scopes=['palaces','heart_and_palace','reverse_palaces','locations']
    for j,p in enumerate(physical):
        split='validation' if j>=6 else 'train'
        for scope in scopes:
            c={'kind':'objects','heart':p['heart'],'illness':p['illness'],'scope':scope,'physical_base_id':p['base_id']}
            query={'palaces':'请比较天心所在宫与天芮所在宫的五行关系。','heart_and_palace':'这次只问天心星自身和它所临地宫的关系。','reverse_palaces':'从天芮所在宫看，它对天心所在宫是什么关系？','locations':'先只说这两颗星的当前落宫。'}[scope]
            for style in range(2):
                facts=(f'程序已核定：天心星在{p["heart"]}宫，天芮星在{p["illness"]}宫。' if style==0 else d.dumps({'当前宫位':{'天心星':p['heart'],'天芮星':p['illness']}}))
                add(c,facts+query,'typed_palace',context=style==1,split=split)
        if split=='train':
            first={'kind':'objects','heart':p['heart'],'illness':p['illness'],'scope':'heart_and_palace','physical_base_id':p['base_id']}
            last=first|{'scope':'palaces'}
            history=[{'role':'system','content':SYSTEM},{'role':'user','content':f'程序给出天心在{p["heart"]}宫，天芮在{p["illness"]}宫。先看天心星自身和它所在宫的关系。'},{'role':'assistant','content':answer(first)}]
            add(last,'落宫不变，但换一个比较对象：现在只比较两颗星所在的两个宫，不比较星自身。关系是什么？','typed_palace',messages=history,initial=first)
    for r,state,extra,goal in product(['相生','比和'],['已具备','未提供','明确不具备'],['双方年命已知','双方年命未知'],['能否','日期']):
        c={'kind':'marriage','condition':state,'goal':goal}
        q=f'乙奇所在宫与庚所在宫{r}，吉门吉格{state}，{extra}。'+('按书中这一分支，现在能怎样判断婚事？' if goal=='能否' else '仅凭这些，已经能确定哪一天结婚了吗？')
        add(c,q,'branch_conditions',context=True,domain='marriage')
    for family,relations in [('loan',['值符克天乙','天乙克值符','天乙生值符','值符生天乙']),('sale',['日干生时干','时干生日干','日干克时干','时干克日干']),('retirement',['杜门生日干','杜门克日干','杜门与日干比和'])]:
        for j,rel in enumerate(relations):
            for style in range(3):
                c={'kind':family,'relation':rel}
                if family=='loan':q=f'在放贷这条里，值符代表银行，天乙代表借贷人。现在{rel}，站在银行一方怎么判断？'
                elif family=='sale':
                    c['profit']=style==2;q=f'这一段卖货，日干是卖方，时干是货物。现在{rel}，'+('销售和利润都已经知道了吗？' if style==2 else '货卖得怎样？')
                else:q=f'申请退役的人取日干，领导取杜门。现在{rel}，按这一条领导会准退还是留用？'
                if style==1:q='已知角色：'+d.dumps({'专题':{'loan':'放贷','sale':'卖货','retirement':'退役'}[family],'当前关系':rel})+'。'+q
                add(c,q,family,context=style==2,domain='business_profit' if family!='retirement' else 'employment')
    for n in range(1,4):
        for doms in combinations(['婚姻','诉讼','交易求财'],n):
            for contextual in [False,True]:add({'kind':'liuhe','domains':list(doms)},'六合用于'+'和'.join(doms)+'时，各代表谁？','domain_roles',context=contextual,domain='mixed_domain')
    for palace in [s for s in PALACES if s!='中']:
        add({'kind':'types'},f'天心星和乙奇这次同在{palace}宫，能因为同宫就把天心的入墓或旺衰状态直接给乙奇吗？为什么？','symbol_types',context=True,domain='health')
    for own,birth in product([True,False],[None,'甲','己']):
        add({'kind':'exam','self':own,'birth':birth},('我本人来问考试' if own else '我作为家长代孩子问考试')+f'，出生年干{birth or "还不知道"}。谁代表实际考生？','exam',context=bool(birth),domain='education')
    facts=[('疾病预测','疾病','天芮星','天芮星为疾病的主要代表符号'),
        ('疾病预测','医生与医药','天心星和乙奇','以天心星和乙奇为医生和医药的代表符号'),
        ('经营求财','资本','甲子戊','甲子戊为资本'),('放贷','利息','生门','生门为利息'),
        ('钱物丢失','失主','日干落宫','以日干落宫为失主'),('钱物丢失','丢失的钱物','时干落宫','以时干落宫为丢失之钱物'),
        ('官司诉讼','律师','惊门','以惊门为律师'),('官司诉讼','法官','开门','以开门为法官'),
        ('官司诉讼','原告','值符','以值符为原告'),('官司诉讼','被告','天乙','以天乙(值符落宫地盘之星)为被告'),
        ('官司诉讼','证人证据','六合','以六合为证人证据'),('官司诉讼','诉状','景门','以景门为诉状'),
        ('官司诉讼','传票','丁奇','以丁奇为传票'),('恋爱婚姻','女方','天盘乙奇','以天盘乙奇为女方'),
        ('恋爱婚姻','男方','天盘庚','天盘庚为男方'),('交易求财','经纪人和中介人','六合','六合为经纪人、中介人')]
    for i,(domain_name,role,symbol,literal) in enumerate(facts):
        candidates=[r for r in pool if any(literal in s['quote'] and symbol in s['quote'] for s in r['provenance']['support'])]
        if role=='利息':candidates=[r for r in candidates if r['id'].startswith('v62:loan_')]
        if not candidates:raise ValueError('Missing original TRAIN literal for '+role)
        family='literal_fact_'+str(i);donors[family]=min(candidates,key=lambda r:sum(len(s['quote']) for s in r['provenance']['support']))
        for quote_requested in [False,True]:
            c={'kind':'source_fact','domain_name':domain_name,'role':role,'symbol':symbol,'literal':literal,'quote_requested':quote_requested}
            q=(f'在{domain_name}这一段里，{role}通常用哪个符号来代表？' if not quote_requested else f'这段原文有没有明确给出{role}的代表符号？请引用对应的那一句。')
            add(c,q,family,context=True,domain='mixed_domain')
    for i,(context_literal,domain_name,role,missing,needed) in enumerate([
        ('以天盘乙奇为女方','婚姻','律师的代表符号','律师','诉讼专题的取用原文'),
        ('以开门为法官','诉讼','资本的代表符号','资本','经营求财专题的原文'),
        ('以日干落宫为失主','钱物丢失','媒人的代表符号','媒人','婚姻专题的取用原文'),
        ('天芮星为疾病的主要代表符号','疾病符号','康复的具体日期','具体日期','实际盘和相应应期规则')]):
        matches=[(r,s) for r in pool for s in r['provenance']['support'] if context_literal in s['quote'] and missing not in s['quote']]
        if not matches:raise ValueError('Missing bounded source-absence contrast')
        parent,support=min(matches,key=lambda x:len(x[1]['quote']));donor=deepcopy(parent)
        donor['provenance']['support']=[deepcopy(support)];family='absence_'+str(i);donors[family]=donor
        c={'kind':'source_absence','domain_name':domain_name,'role':role,'missing_term':missing,'needed':needed}
        add(c,f'只看给出的这段{domain_name}材料，它有没有明确给出{role}？能直接引用它来下这个结论吗？',family,context=True,domain='mixed_domain')
    # Replay the previously curated source dialogues, including four attributed cases.
    for donor in [r for r in pool if r['skill']=='source_dialogue']:
        r=deepcopy(donor);r['id']='v64:source:'+donor['id'];r['provenance']['source_train_ids']=[donor['id']];r['provenance']['replayed_source_dialogue_unchanged']=True;records.append(r)
    # Fresh wording/field combinations: known sources, not a new blind benchmark.
    for i,role in enumerate(ROLES):
        c={'kind':'identity','role':role,'fields':{'年干':'辛','月干':'丁','日干':'乙','时干':'己'},'birth':'壬' if i%2 else None}
        add(c,'先不要沿用上一问：这次实际关系是'+role+'，资料'+d.dumps(c['fields'])+f'，可靠出生年干{c["birth"] or "不详"}。应该用哪个起局字段代表患者，其他字段如何处理？','identity',split='validation',domain='health')
    for i,(x,y) in enumerate([('水','火'),('土','火'),('金','木'),('木','金'),('水','水')]):
        c={'kind':'verify_direction','a':'前者','b':'后者','x':x,'y':y,'claim':claims[i]}
        add(c,f'关系核查：前者{x}，后者{y}，标注前者对后者“{claims[i]}”。这行要怎么处理？','direction',split='validation')
    for s in ['已具备','未提供','明确不具备']:
        add({'kind':'marriage','condition':s,'goal':'能否'},f'只核对原书可成分支：两宫比和，吉门吉格{s}。你能给到哪一层结论？','branch_conditions',context=True,split='validation',domain='marriage')
    for family,rel in [('loan','值符生天乙'),('loan','天乙生值符'),('sale','日干克时干'),('sale','时干克日干'),('retirement','杜门克日干')]:
        q={'loan':f'银行仍取值符、借款人仍取天乙，改成{rel}。请只按这条说明银行一方的结果和方向。','sale':f'卖方日干与货物时干这次是{rel}。按原段应怎样说销售情况？','retirement':f'领导杜门与申请退役人日干这次是{rel}。本条准不准退出？'}[family]
        add({'kind':family,'relation':rel},q,family,context=True,split='validation')
    for role in ['父亲','同事','女儿']:
        c={'kind':'identity','role':'本人','fields':{'年干':'己','月干':'辛','日干':'丁','时干':'甲'},'birth':None}
        initial=c|{'role':role};history=[{'role':'system','content':SYSTEM},{'role':'user','content':f'先按代问{role}取用，起局字段'+d.dumps(c['fields'])+'，年命没核实。'},{'role':'assistant','content':answer(initial)}]
        add(c,f'现在{role}本人接过来，亲自问自己的健康。起局资料不变，年命仍不知道，这次怎么取？','identity',messages=history,initial=initial,split='validation',domain='health')
    initial={'kind':'identity','role':'母亲','fields':{'年干':'辛','月干':'丙','日干':'甲','时干':'庚'},'birth':None}
    history=[{'role':'system','content':SYSTEM},{'role':'user','content':'我代妈妈问病，起局字段'+d.dumps(initial['fields'])+'，出生年不详。怎么取？'},{'role':'assistant','content':answer(initial)}]
    add(initial|{'birth':'丁'},'仍然是我代妈妈问，只核实了她出生年干丁。原主用神是否改变，出生资料怎么用？','identity',messages=history,initial=initial,split='validation',domain='health')
    # Exact duplicate questions with the same teacher are merged; no row is split or truncated.
    unique={}
    for r in records:
        key=d.dumps(r['messages'][:-1])
        if key in unique:
            if unique[key]['answer']!=r['answer'] or unique[key]['split']!=r['split']:raise ValueError('Conflicting or leaking teaching dialogue')
            continue
        unique[key]=r
    records=list(unique.values());old_eval=source['known_regression_rows']+source['actual_rag_rows']+[r for r in source['records'] if r['split']=='validation']
    frozen_prompts={d.dumps(r['messages'][:-1]) for r in old_eval}
    if any(d.dumps(r['messages'][:-1]) in frozen_prompts for r in records if r['split']=='train'):raise ValueError('Exposed evaluation prompt copied into teaching')
    out=deepcopy(source);out.update(schema=SCHEMA,records=records,physical_contexts=physical,fresh_official_lora_required=False,
        known_development_rows=old_eval,original_recall_rows=old.learning_rows(source),new_books_used=False,model_outputs_used_as_teachers=False)
    out['dispositions']=[{'old_id':r['id'],'action':'replay_source_dialogue' if r['skill']=='source_dialogue' else 'source_preserved_replaced_by_conditional_tasks','source_ids':r['source_ids']} for r in pool]
    old_used=Counter(s for r in pool for s in r['source_ids']);new_used=Counter(s for r in records if r['split']=='train' for s in r['source_ids'])
    out['material_usage']=[{'source_id':s,'old_train_lessons':old_used[s],'new_train_lessons':new_used[s],
        'disposition':'source_used_in_new_training' if new_used[s] else 'source_preserved_not_repeated_as_training'} for s in units]
    out['summary']={'schema':SCHEMA,'cores':len(records),'train':sum(r['split']=='train' for r in records),'new_development':sum(r['split']=='validation' for r in records),
        'training_skills':dict(Counter(r['skill'] for r in records if r['split']=='train')),'original_source_units':len(units),'original_case_registry_entries':len(source['case_splits']),
        'training_source_unit_count':len(new_used),'case_linked_training_cores':sum(bool(r['case_ids']) for r in records if r['split']=='train'),
        'true_generated_identity_changes':sum(r['split']=='train' and r['provenance'].get('initial_contract',{}).get('kind')=='identity' and r['provenance']['initial_contract']['role']!=r['provenance']['lesson_contract']['role'] for r in records if r['provenance'].get('initial_contract')),
        'same_identity_birth_updates':7,'generated_comparison_layer_changes':6,
        'physical_layouts_used':len(physical),'physical_layouts_are_not_outcome_cases':True,'source_dialogues_replayed':60,'original_training_cores_dispositioned':len(pool),
        'full_two_book_case_training_claimed':False,'new_blind_benchmark':False}
    validate(out,source);return out


def validate(pack,source):
    if pack['schema']!=SCHEMA or pack['units']!=source['units'] or pack['case_splits']!=source['case_splits']:raise ValueError('Original book material/splits changed')
    if set(pack['source_books'])!=d.BOOKS or any(pack.get(k) for k in ['old_manual_scores_used','model_outputs_used_as_teachers','original_final_test_used','new_books_used']):
        raise ValueError('Only the two source books, no scores/model-output/final-test teachers')
    allowed={r['id']:r for r in source['records'] if r['split']=='train'};seen={};ids=set()
    physical={p['base_id']:p for p in pack['physical_contexts']}
    if any(set(p)!={'base_id','heart','illness'} for p in physical.values()):raise ValueError('Only literal locations may enter physical teaching')
    for r in pack['records']:
        if r['id'] in ids:raise ValueError('Duplicate lesson id')
        ids.add(r['id']);p=r['provenance'];parents=p['source_train_ids']
        if not parents or not set(parents)<=set(allowed):raise ValueError('Teacher derived from non-train/model output')
        if p.get('teacher_is_model_output'):raise ValueError('Model output cannot become a teacher')
        support_allowed={(s['source_id'],d.compact(s['quote'])) for parent in parents for s in allowed[parent]['provenance']['support']}
        if {s['source_id'] for s in p['support']}!=set(r['source_ids']):raise ValueError('Lesson source list differs from complete evidence')
        if not {(s['source_id'],d.compact(s['quote'])) for s in p['support']}<=support_allowed:raise ValueError('Source is not bound to the declared original train donor')
        if any(u not in pack['units'] for u in r['source_ids']):raise ValueError('Unbound source id')
        for s in p['support']:
            if d.compact(s['quote']) not in d.compact(pack['units'][s['source_id']]['original_quote']):raise ValueError('Source quote changed')
        if p.get('lesson_contract'):
            c=p['lesson_contract']
            if r['answer']!=answer(c):raise ValueError('Answer contradicts explicit conditions/objects')
            if c['kind']=='objects':
                actual=physical[c['physical_base_id']]
                if (c['heart'],c['illness'])!=(actual['heart'],actual['illness']):raise ValueError('Current placement changed from the source program')
            if c['kind']=='source_fact' and not any(c['literal'] in s['quote'] and c['symbol'] in s['quote'] for s in p['support']):
                raise ValueError('A positive source claim requires its visible literal support')
            if c['kind']=='source_fact':
                normalize=lambda text:re.sub(r'[、，,和与的之()（）]','',text)
                if normalize(c['role']) not in normalize(c['literal']) or normalize(c['symbol']) not in normalize(c['literal']):
                    raise ValueError('The cited span must bind the requested role to its symbol')
            if c['kind']=='source_absence' and any(c['missing_term'] in s['quote'] for s in p['support']):
                raise ValueError('Cannot call visible source content absent')
            if p.get('initial_contract') and r['messages'][2]['content']!=answer(p['initial_contract']):raise ValueError('Multi-turn initial answer wrong')
        elif r['messages']!=allowed[parents[0]]['messages']:raise ValueError('Unreviewed modification to source dialogue')
        if r['messages'][-1]['content']!=r['answer'] or r['messages'][0]['content']!=SYSTEM:raise ValueError('Wrong final answer/system')
        if '这里只判断门与宫的关系' in r['answer']:raise ValueError('Legacy scope tail replayed')
        key=d.dumps(r['messages'][:-1])
        if key in seen:raise ValueError('Duplicate or conflicting complete prompt')
        seen[key]=r['split']
        for case_id in r.get('case_ids',[]):
            if source['case_splits'][case_id] in ('test','sealed_test') or source['case_splits'][case_id]!=r['split']:
                raise ValueError('Original case boundary changed')
    if not any(r['split']=='validation' for r in pack['records']):raise ValueError('Missing explicit development set')
    return True
