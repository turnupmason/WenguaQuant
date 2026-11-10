"""Controlled normal-language diagnostics. No training, no final-test cases."""
from copy import deepcopy
import re
from wengua_course import data as d
SCHEMA='wengua_v61_normal_dialogue_diagnostic_v1'
SYSTEM=('按当前用户问题正常交流，直接说明结果及有用的原因，不用固定开场白或备课式标题。'
        '先核对人物、所问目标和已给条件；没有给出的事实不补造。'
        '有参考资料时按其适用范围回答，不照抄无关段落。普通问题按普通问题回答。')

def build(pack):
    train=[r for r in pack['records'] if r['split']=='train']
    def find(text):
        rows=[r for r in train if r['skill']=='natural_dialogue' and text in r['question']]
        if not rows:raise ValueError('Missing frozen source teacher: '+text)
        return rows[0]
    father=find('不知道他的出生年份');known=find('起局的年干是乙');selfrule=find('是我自己来看病');same=find('治疗力量就一样');loan=find('银行放贷');sell=find('卖货，日干是我');liuhe=find('六合是不是');marry=find('婚事是不是');door=find('景门落兑');wood=find('伤门落乾')
    direction=next(r for r in train if r['skill']=='direction')
    boards=[r for r in pack['records'] if r['split']=='validation' and r['skill']=='chart_direction' and '任星所在宫对英星所在宫' in r['question']]
    if len(boards)!=1:raise ValueError('Expected exact pre-existing development board')
    board=boards[0]
    if any(pack['case_splits'][cid]!='validation' for cid in board['case_ids']):raise ValueError('A sealed/original-test chart cannot enter this diagnostic')
    board_text='\n'.join(s for s in board['question'].splitlines() if re.match(r'^[巽离坤震中兑艮坎乾][一二三四五六七八九]宫：',s))
    if len(board_text.splitlines())!=9:raise ValueError('Whole nine-palace transcription required')
    rows=[]
    def add(key,q,answer,anchors=(),*,context='plain',history=(),pair=None):
        support=[];seen=set();case_ids=set()
        for r in anchors:
            case_ids.update(r['case_ids'])
            refs=r['provenance'].get('support') or r['provenance'].get('required_source_fragments') or [{'source_id':sid,'quote':pack['units'][sid]['original_quote']} for sid in r['source_ids']]
            for ref in refs:
                k=(ref['source_id'],ref['quote'])
                if k not in seen:seen.add(k);support.append(deepcopy(ref))
        for ref in support:
            u=pack['units'][ref['source_id']]
            if u['book_id'] not in d.BOOKS or d.compact(ref['quote']) not in d.compact(u['original_quote']):raise ValueError('Probe evidence is not literal at source')
        content=q
        if context=='provided_source':content+='\n\n参考资料：\n'+'\n\n'.join(f'[S{i}] '+r['quote'] for i,r in enumerate(support,1))
        messages=[{'role':'system','content':SYSTEM},*deepcopy(history),{'role':'user','content':content},{'role':'assistant','content':answer}]
        rows.append({'id':'v61_probe:'+key,'question':content,'answer':answer,'messages':messages,'source_ids':sorted({r['source_id'] for r in support}),'case_id':next(iter(sorted(case_ids)),None),'case_ids':sorted(case_ids),'group':'v61_probe:'+key,'format':'normal_dialogue_diagnostic','task':key,'source_mode':'eval_only_no_training','context':context,'pair':pair,'support':support,'source_anchor_ids':[r['id'] for r in anchors],'not_training_or_rag_teacher':True,'not_unseen_concept_claimed':True,'controlled_history_is_authored':bool(history)})
    rule='这里沿用“本人求测先取日干，子女代问长辈先取起局年干”的取法。'
    add('self_unknown',rule+'我自己来问健康，但出生年份没核实，先用什么？不知道出生年会不会就没法看了？','本人先取日干；缺出生年只是不取年命，不影响先确定本人用神。',[selfrule,father])
    add('father_unknown',rule+'女儿替爸爸问病，爸爸哪年出生还没问清。现在先取什么，哪些资料要先留空？','起局年干代表代问的父亲，兼看乾宫；未知出生年先不取年命。',[father])
    add('father_known',rule+'爸爸的出生年干癸，起局年干乙；我女儿替他问。乙和癸分别怎么用？不要补出题里没给的日干。','代问父亲用起局年干乙，兼乾宫；出生年命癸另看，不互换，也不填入未知日干。',[known])
    history=[{'role':'user','content':'女儿替父亲问事，出生年没核实，先取谁？'},{'role':'assistant','content':'代问父亲先取起局年干，兼看乾宫；出生年未核实，暂不取年命。'}]
    add('switch_to_self','现在换成父亲本人来问，出生年干已确认是癸。这一问还沿用上一问的主用神吗？','改按本人求测，主用神取日干，兼可靠出生年命癸；不沿用代问父亲的年干作主用神。',[selfrule,known],history=history)
    add('metal_water','甲方属金，乙方属水。到底谁生谁？请只讲这两方的直接关系，不借第三种五行绕一圈。','金生水，所以甲方生乙方；乙方受甲方所生，不把间接链当直接克制。',[direction])
    add('fire_fire','前后两个对象都是火，它们之间还要判谁克谁吗？','两者同属火，为比和；这两者之间不取相生相克的方向。',[direction])
    add('qian_kun','乾宫属金，坤宫属土。有人说“乾金生坤土”，这句话对吗？错在哪里？','不对，应土生金，也就是坤宫生乾宫；从乾宫看是受生。',[direction,wood])
    add('jing_dui','景门属火，兑宫属金。景门落兑，是门克宫还是宫克门？本条名称是什么？','火克金，是门克宫，称门迫。',[door])
    add('shang_qian','伤门属木，乾宫属金。有人把伤门落乾也说成门迫，你怎么纠正？','金克木，是乾宫克伤门，称门受制；门迫指门克宫。',[wood])
    add('read_board','盘给你：\n'+board_text+'\n只比较天任和天英所在的宫：它们各在哪，谁生谁？不用断整件事情的吉凶。','天任在坎一宫属水，天英在巽四宫属木；水生木，所以天任所在宫生天英所在宫。',[board],context='whole_board')
    add('given_board_positions','位置已经核实：天任在坎宫，天英在巽宫。坎属水，巽属木。只问两颗星所在宫的关系，谁生谁？','坎水生巽木，即天任所在宫生天英所在宫；不用两颗星本身的五行替换宫位五行。',[board],context='given_positions')
    add('same_palace_different_symbols','天心星和乙奇同宫，它们的旺衰、入墓状态就一定一样吗？还需要分别看什么？','不能因同宫就认定同强或处于同一状态。星与天干需分别按相应时令、落宫及长生墓绝等条件判断。',[same])
    pairs=[('loan', '我站在银行放贷人的立场问，值符代表银行、天乙代表借贷人；现在天乙克值符。按这条放贷规则，对银行有利吗？','本条对银行放贷人不利；是借贷人一方克放贷人，不能倒读成值符克天乙。',loan),
           ('sale','这题问卖货速度：日干是卖方，时干是货物。现在时干克日干，按这条规则卖得快还是卖不动？能顺便认定利润高吗？','按这条卖货取用，时干克日干对应卖得快；它判断销售速度，不能直接推出利润高。',sell),
           ('liuhe','只说六合分别代表谁：婚姻、官司、交易求财这三种场景各怎么取？别把三章都重新讲一遍。','婚姻看媒人，官司看证人或证据，交易求财看经纪人或中介；按领域分别取用。',liuhe),
           ('marriage','问的是婚事能否成，不是问哪天结婚。只知道乙庚两宫相生，按这条还欠哪个关键条件？','还要逢吉门吉格，不能只凭相生认定成婚；本题无需补造结婚日期。',marry)]
    for key,q,a,anchor in pairs:
        for context in ['plain','provided_source']:add(key+'_'+context,q,a,[anchor],context=context,pair=key)
    add('ordinary_rewrite','把“今晚不行，改明天”改成一句给同事的自然礼貌消息，保留改约明天的意思。不要加入占卜内容。','今晚我不太方便，我们改到明天可以吗？',context='ordinary_non_book_eval')
    add('ordinary_extract','只复述这句话的两项已知事实，不解卦：“女儿替爸爸问事；爸爸出生年干癸，本次起局年干乙。”谁的出生年是癸，哪个年干是乙？','爸爸的出生年干是癸；本次起局的年干是乙。',context='ordinary_non_book_eval')
    add('ordinary_arithmetic','库存18本，卖出5本后又退回2本，现在有多少本？用一句话说明计算。','18－5＋2＝15，现在有15本。',context='ordinary_non_book_eval')
    history=[{'role':'user','content':'给我三个操作步骤。'},{'role':'assistant','content':'第一，备份资料。第二，更新程序。第三，检查结果。'}]
    add('ordinary_followup','现在只保留第二步，写成一句话，不加解释，也别重新列三步。','更新程序。',context='ordinary_non_book_eval',history=history)
    if len(rows)!=24 or len({r['id'] for r in rows})!=24:raise ValueError('Exactly 24 unique controlled prompts required')
    for key,_,_,_ in pairs:
        a,b=[r for r in rows if r['pair']==key]
        if a['answer']!=b['answer'] or b['question'].split('\n\n参考资料：')[0]!=a['question']:raise ValueError('Evidence contrast also changed the question or target')
    return {'schema':SCHEMA,'records':rows,'only_evaluation':True,'training_enabled':False,'ordinary_non_book_eval_rows':4,'source_concepts_previously_exposed':True,'original_test_used':False,'new_unseen_chart_or_real_outcome_benchmark':False,'full_case_prediction_test':False}
