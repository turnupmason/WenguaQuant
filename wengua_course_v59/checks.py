"""Typed, context-bound facts; explanation and open semantics are separate.

No lexical overlap score certifies an open case. Original V56 rubrics freeze.
"""
import re
from collections import defaultdict,Counter
from wengua_course import data as d
from wengua_course_v55 import checks as legacy
from wengua_course_v53.checks import positions_ok

REL=r'(受生|受克|比和|施生|施克|生|克)'
INVERSE={'生':'受生','受生':'生','克':'受克','受克':'克','比和':'比和'}


def asserted(text,start,end):
    before=text[max(0,start-30):start];after=text[end:end+14]
    if re.search(r'(?:不是|并非|非|不能(?:说|认定|视为|断为)?|不应|不)(?:[“「\s]|称为|叫作|认为|保证|代表|取|看|用){0,3}$',before):return False
    if re.search(r'^[”」\s]*(?:是错|不对|错误|不成立)',after):return False
    if re.search(r'(?:错误说法|误称|误说)[：:为“「\s]*$',before):return False
    return True


def relation_labels(text,c=None):
    """Unqualified target labels, with explicit reverse viewpoint preserved."""
    result=[];pattern=r'(?:^\s*|(?:结论|关系)(?:为|是|属)?[：:为是\s]*|这里(?:称为|为|是)?|称为|应改(?:记|写)?(?:为|成)?|改为|两者|双方|两宫|前者与后者|A与B|门与宫|门和宫|是|为)'+REL+r'(?=[。；，,：:\s]|$)'
    for m in re.finditer(pattern,text):
        if not asserted(text,m.start(),m.end()):continue
        label=m[1].replace('施生','生').replace('施克','克');prefix=re.split('[。；;\n]',text[:m.start()])[-1]
        reverse=bool(re.search(r'(?:后者对前者|从后者|对后者而言)[^。；;\n]{0,18}$',prefix))
        if label in ['生','克'] and re.search(r'两者之间[^。；;\n]{0,6}$',prefix):continue
        if c and len(c.get('symbols',[]))==2:
            sa,sb=c['symbols'];reverse=reverse or bool(re.search('从'+re.escape(sb)+'看'+re.escape(sa)+r'[^。；;\n]{0,10}$',prefix))
        result.append((label,reverse,m[0]))
    return result


def relation_facts(answer,c):
    a,b=c['elements'];expected=c['answer'];conflicts=[]
    subjects={'门':a,'宫':b,'前者':a,'后者':b,'A':a,'B':b,'A宫':a,'B宫':b,'对象A':a,'对象B':b,'第一对象':a,'第二对象':b,'第一方':a,'第二方':b,'第一宫':a,'第二宫':b,'对象甲':a,'对象乙':b,'甲方':a,'乙方':b}
    subjects.update(zip(c.get('symbols',[]),[a,b]));subjects.update({x+'方':x for x in '金木水火土'})
    subjects.update({name+'宫':d.ELEMENT[n] for n,name in d.PALACE.items()})
    for s,n in c.get('positions',{}).items():subjects[s+'星所在宫']=d.ELEMENT[n]
    for m in re.finditer(r'([金木水火土])([生克])([金木水火土])',answer):
        if asserted(answer,m.start(),m.end()) and (d.GENERATES if m[2]=='生' else d.CONTROLS)[m[1]]!=m[3]:conflicts.append(m[0])
    tokens='|'.join(map(re.escape,sorted(subjects,key=len,reverse=True)))
    for m in re.finditer('('+tokens+r')(?:的五行|属性)?(?:也属|属于|属|代表属|代表|为|是)([金木水火土])',answer):
        if asserted(answer,m.start(),m.end()) and subjects[m[1]]!=m[2]:conflicts.append(m[0])
    applied_edges=[];applied_received=False
    front={a+'方','前者','A','A宫','对象A','第一对象','第一方','第一宫','对象甲','甲方',*c.get('symbols',[])[:1]};rear={b+'方','后者','B','B宫','对象B','第二对象','第二方','第二宫','对象乙','乙方',*c.get('symbols',[])[1:]}
    for m in re.finditer('('+tokens+r')(生|克)('+tokens+')',answer):
        if asserted(answer,m.start(),m.end()):
            if (d.GENERATES if m[2]=='生' else d.CONTROLS)[subjects[m[1]]]!=subjects[m[3]]:conflicts.append(m[0])
            if m[1] in front and m[3] in rear or m[1] in rear and m[3] in front:applied_edges.append(m[0])
    if not c.get('positions'):
        for m in re.finditer(r'(?:对象A|对象B|A|B|前者|后者)星?(?:为|是)(?:[甲乙丙丁戊己庚辛壬癸]|天[蓬任冲辅英芮柱心禽])',answer):
            if asserted(answer,m.start(),m.end()):conflicts.append('invented_object_identity:'+m[0])
    labels=relation_labels(answer,c)
    for label,reverse,claim in labels:
        if label!=(INVERSE[expected] if reverse else expected):conflicts.append(claim)
    for subject,element in subjects.items():
        if subject not in front|rear:continue
        for m in re.finditer(re.escape(subject)+r'(受生|受克|比和)',answer):
            if not asserted(answer,m.start(),m.end()):continue
            other=b if element==a else a
            if d.relation({'水':1,'土':2,'木':3,'金':6,'火':9}[element],{'水':1,'土':2,'木':3,'金':6,'火':9}[other])!=m[1]:conflicts.append(m[0])
            elif subject in front:applied_received=True
    pos=not c.get('require_positions') or positions_ok(answer,c['positions'])
    if conflicts:return {'correct':False,'complete':False,'conflicts':conflicts,'positions':pos}
    if a==b:
        applied=any(x=='比和' and not rev for x,rev,_ in labels) or bool(re.search(r'五行相同|同属'+a+'|均属'+a+'|都属'+a+'|都是'+a,answer))
        return {'correct':applied and pos,'complete':applied and pos,'conflicts':[],'positions':pos}
    x,y=(a,b) if expected in ['生','克'] else (b,a);op='生' if expected in ['生','受生'] else '克';law=x+op+y
    law_present=any(asserted(answer,m.start(),m.end()) for m in re.finditer(law,answer))
    law_set={m[0] for m in re.finditer(r'[金木水火土][生克][金木水火土]',answer) if asserted(answer,m.start(),m.end())}
    label_present=any(label==(INVERSE[expected] if rev else expected) for label,rev,_ in labels)
    # Quoting every cycle is not selection/application of this one relation.
    selected_law=law_present and (law_set=={law} or label_present)
    return {'correct':pos and (label_present or selected_law or bool(applied_edges) or applied_received),'complete':pos and selected_law,'conflicts':[],'positions':pos}


def year_use_conflicts(answer,unknown):
    if not unknown:return []
    result=[]
    for m in re.finditer(r'(?:兼看|兼取|取|看|计算|使用|暂用|以|用)(?:父亲|母亲|患者|考生|子女|本人的)?(?:出生)?年命(?:[为是取：:]?[甲乙丙丁戊己庚辛壬癸])?',answer):
        if not asserted(answer,m.start(),m.end()):continue
        clause=re.split('[。；;\n]',answer[:m.start()])[-1];after=answer[m.end():m.end()+28]
        conditional=bool(re.search(r'(?:有|提供|补充|待|确认|可靠)[^。；;\n]{0,12}出生年[^。；;\n]{0,12}(?:再|后|才|时|则)?$',clause))
        conditional=conditional or bool(re.search(r'^(?:时|前)?[^。；;\n]{0,6}(?:需|要|应)先(?:补充|确认|提供)[^。；;\n]{0,8}出生年',after))
        if not conditional:result.append(m[0])
    return result


def person_facts(answer,c):
    unknown=c.get('unknown_year',False);conflicts=year_use_conflicts(answer,unknown);roles=[]
    if unknown:
        for m in re.finditer(r'年命(?:为|是|取|用|：)?[甲乙丙丁戊己庚辛壬癸]',answer):
            if asserted(answer,m.start(),m.end()):conflicts.append('invented_unknown_birth_stem')
    # Resolve explicit actors within the answer before comparing their role.
    # Correct alternatives are allowed, wrong conditional mappings are not.
    patterns=[(r'本人(?:来问|来求测|求测|询问|问|取|看|用)|考生自己(?:来问|求测|问)|父亲自己(?:来问|求测|问)|母亲自己(?:来问|求测|问)','日干'),
              (r'(?:儿女|女儿|儿子|子女|晚辈|你).{0,5}(?:代问|问|替).{0,5}(?:父母|父亲|母亲|长辈)','年干'),
              (r'(?:父母|父亲|母亲|妈妈|爸爸).{0,5}(?:代问|问).{0,5}(?:子女|儿子|女儿|孩子|考学|考试)','时干')]
    def scoped_role(part,start):
        found=[(m.start(),role) for pattern,role in patterns for m in re.finditer(pattern,part[:start])]
        return max(found,key=lambda x:x[0])[1] if found else c['answer']
    def observe(part,m,role):
        scope=re.split(r'但是|但|然而|不过|而应|而是|仍(?:然)?(?:应|要)?|却',part[:m.start()])[-1]
        negative=bool(re.search(r'(?:不能|不可|不应|不要|不因|不因为|不会)[^。；;]{0,24}$',scope))
        if not asserted(part,m.start(),m.end()) or negative:return
        actor=scoped_role(part,m.start())
        if role!=actor:conflicts.append('explicit_actor_role_mismatch:'+m[0])
        if actor==c['answer']:roles.append(role)
    for part in re.split('[。；;\n]',answer):
        for m in re.finditer(r'(?:取|看|以|用|先看|先取|改用|改取)(?:本次起局|起局|本局)?(年干|月干|日干|时干)',part):observe(part,m,m[1])
        for m in re.finditer(r'(年干|月干|日干|时干)(?:落宫)?(?:为|代表)(?:父亲|母亲|患者|本人|考生|子女)',part):observe(part,m,m[1])
        head=re.match(r'^\s*(年干|月干|日干|时干)[，,：:]?\s*$',part)
        if head:roles.append(head[1])
    role_ok=c['answer'] in roles and all(r==c['answer'] for r in roles)
    helper=c.get('helper') or next((x+'宫' for x in ['乾','坤'] if x in str(c.get('required',[]))),None)
    helper_ok=not helper or helper in answer or helper.removesuffix('宫') in answer
    unknown_ok=not unknown or bool(re.search(r'(?:暂不|不|不能|不应|不补造|不补|无从|待)[^。；;\n]{0,9}(?:取|加|补|用|使用|确定|造|计算)?(?:相应|父亲|母亲|患者|考生|子女)?年命|年命[^。；;\n]{0,14}(?:未知|不取|不加|暂不|待|不使用)|出生年未知[^。；;\n]{0,10}(?:不补造|不补|不猜)',answer))
    known=c.get('known_year_stem')
    if re.search(r'(?:取|以|将)(?:本局|起局)?年干(?:为|作为)(?:可靠)?出生年命',answer):conflicts.append('cast_year_and_birth_year_conflated')
    if known and c.get('cast_year_stem')!=known and re.search(r'(?:取|看|用|以|本局|起局)年干'+known,answer):conflicts.append('birth_stem_used_as_ungiven_cast_year_stem')
    for m in re.finditer(r'(日干|时干|月干)(?:为|是|取)?([甲乙丙丁戊己庚辛壬癸])',answer):
        if asserted(answer,m.start(),m.end()) and c.get('role_stem')!=m[2]:conflicts.append('ungiven_role_stem:'+m[0])
    if known and re.search(r'(?:(?:缺|未知|未给)[^。；;]{0,12}起局年干|起局年干[^。；;]{0,8}(?:未知|缺|未给))[^。；;]{0,20}(?:不能|不应)[^。；;]{0,9}年命',answer):conflicts.append('known_birth_wrongly_rejected_for_missing_cast')
    unknown_ok=unknown_ok or c.get('require_year_boundary') is False
    known_ok=not known or bool(re.search(r'年命(?:为|是|取|用)?'+known,answer)) or bool(known and re.search(r'出生年(?:干)?'+known+r'[^。；;]{0,10}(?:为|是|作|看|兼看)年命',answer)) or bool(known and re.search(r'已提供出生年干'+known+r'[^。；;]{0,12}兼看年命宫',answer))
    if known:
        wrong=re.findall(r'年命(?:为|是|取|用)?([甲乙丙丁戊己庚辛壬癸])',answer)
        if any(x!=known for x in wrong):conflicts.append('wrong_given_birth_stem')
    return {'correct':role_ok and not conflicts,'complete':role_ok and helper_ok and unknown_ok and known_ok and not conflicts,'roles':roles,'helper':helper_ok,'year_boundary':unknown_ok and known_ok,'conflicts':conflicts}


def source_slots(answer,c):
    normalized=re.sub(r'[\s、，,；;]', '',answer)
    ok=all(any(any(asserted(answer,m.start(),m.end()) for m in re.finditer(re.escape(x),answer)) for x in choices) for choices in c.get('required',[]))
    for label in c.get('forbidden',[]):
        for m in re.finditer(re.escape(label),answer):
            if asserted(answer,m.start(),m.end()):ok=False
    return ok


def fact_check(record,gold):
    a=record['answer'];c=gold.get('checks') or {};kind=c.get('kind');detail={};complete=True
    prov=gold.get('provenance',{})
    if kind=='literal_slots' and prov.get('kind')=='domain_scope_contrast':
        c={'kind':'domain_map','requested_domains':prov['contrast_group'].split(':',1)[1].split(',')};kind='domain_map'
    if not a.strip() or record.get('empty'):return {'status':'fail','reason':'empty','complete':False}
    if record.get('reached_token_limit'):return {'status':'fail','reason':'output_limit','complete':False}
    if max(Counter(x.strip() for x in re.split('[。\n]',a) if len(x.strip())>=20).values(),default=0)>=5 or re.search(r'(.{2,24})\1{7,}',re.sub(r'\s+','',a)):return {'status':'fail','reason':'repeated_output','complete':False}
    if not kind or kind=='mentions':return {'status':'needs_review','reason':'open_source_semantics','complete':False,'professional_approval':False}
    if kind in ['relation_facts','reasoning_chain','relation','joint_relation'] and c.get('elements'):
        detail=relation_facts(a,c|{'require_positions':kind=='joint_relation' or c.get('require_positions',False)});ok=detail['correct'];complete=detail['complete']
    elif kind=='door_facts':
        detail=relation_facts(a,c|{'answer':c['relation'],'symbols':[c['door']+'门',d.PALACE[c['palaces'][0]]+'宫']});expected=c['answer'];names=[]
        for m in re.finditer(r'(?:^\s*|门与宫|门和宫|为|是|称为|名为|属于)(门迫|比和|迫|制|和|义)(?=[。；，,：:\s]|$)',a):
            if asserted(a,m.start(),m.end()):names.append(m[1].replace('门迫','迫'))
        aliases={'迫':['门克宫'],'制':['宫克门'],'和':['门生宫'],'义':['宫生门'],'比和':['五行相同','同属']}
        name_conflict=bool(names and set(names)!={expected});ok=detail['correct'] and not name_conflict;complete=ok and detail['complete'] and bool(names);detail['name_complete']=bool(names) and not name_conflict
    elif kind in ['person_contract','role_boundary','birth_boundary','role'] and c.get('answer') in ['年干','月干','日干','时干']:
        detail=person_facts(a,c);ok=detail['correct'];complete=detail['complete']
    elif kind=='condition_slots':
        if any('可能被盗' in group for group in c.get('required',[])):
            c=dict(c,required=[['可能被盗','可能被人盗走'] if '可能被盗' in group else group for group in c['required']]);gold=dict(gold,checks=c)
        ok=source_slots(a,c);detail={'scope':'source_condition_slots_not_full_case_semantics'};complete=ok
    elif kind=='property':
        symbol=c['symbol'];element=c['element'];claims=[]
        for m in re.finditer(re.escape(symbol)+r'[^。；;\n]{0,20}?(?:归于|同是|亦是|属于|属|为|是)([金木水火土])',a):
            if asserted(a,m.start(),m.end()):claims.append(m[1])
        lead=re.split(r'依据|\[S\d+\]',a)[0].strip('。；; 、\n')
        short=lead in [element,'属'+element,'属于'+element,'五行属'+element]
        ok=bool(claims) and set(claims)=={element} or short
        complete=ok
    elif kind=='clarification':
        ok=any(w in a for w in ['先确认','先问','需要确认','请确认','要确认','需明确'])
        for actor,role in c['options']:
            other='|'.join(re.escape(x) for x,_ in c['options'] if x!=actor)
            forward=re.escape(actor)+r'(?:(?!(?:'+other+r'|年干|月干|日干|时干)).){0,22}'+role
            inverse=role+r'(?:代表|用于|对应|看)(?:(?!(?:年干|月干|日干|时干)).){0,18}'+re.escape(actor)
            ok=ok and bool(re.search(forward+'|'+inverse,a))
        ok=ok and (not c.get('helper') or c['helper'] in a)
    elif kind=='scope_clarification':
        ok=all(w in a for w in ['求测','关系']) and any(w in a for w in ['不足','先确认','需要确认']) and any(w in a for w in ['不能直接','不可直接','不等同'])
    elif kind=='number_pair':
        clean=re.sub(r'\[S\d+\]', '',a);first=re.split('[。；;\n]',clean)[0];mapping=dict(zip('一二三四五六七八九十',range(1,11)));values=[int(x) if x.isdigit() else mapping[x] for x in re.findall(r'10|[1-9]|[一二三四五六七八九十]',first)]
        pair_ok=set(values)==set(c['answer']);inverted=any(asserted(clean,m.start(),m.end()) for m in re.finditer(r'天[二四六八十][生生成]|地[一三五七九][成生成]',clean));ok=pair_ok and not inverted;complete=ok;detail={'pair_correct':pair_ok,'inverted_heaven_earth':inverted}
    elif kind=='literal_slots':ok=source_slots(a,c)
    elif kind=='insufficient':
        positive=re.search(r'(?:^|[。；;\n])\s*(?:这里(?:已|只)?(?:列出|给出|是)|本题(?:是|为)|已确认)?\s*天盘[甲乙丙丁戊己庚辛壬癸][^。；;\n]{0,12}地盘[甲乙丙丁戊己庚辛壬癸]',a)
        if positive:
            clause=re.split('[。；;\n]',a[positive.start():])[0]
            actual=any(w in positive[0] for w in ['这里','本题','已确认'])
            if not actual and any(w in clause for w in ['名','称','叫','格','若','如果','只有','才']):positive=None
        ok=positive is None and any(w in a for w in ['不能','不可','不足','缺少','未提供','未给']) and '天盘' in a and '地盘' in a
        detail={'positive_ungiven_pair':positive[0] if positive else None};complete=ok
    elif kind=='source_inference':
        normalized=re.sub(r'[\s。；，、：:]', '',a);judgment=re.sub(r'[\s。；，、：:]', '',c['judgment'])
        main=judgment in normalized;terms=[x for x in c['evidence_terms'] if x in a]
        bad_date=[x for x in re.findall(r'\d{4}年\d{1,2}月\d{1,2}日',a) if x not in gold.get('question','') and x not in gold.get('answer','')]
        placeholder=bool(re.search('待补|待答|需要回答的判断|尚待推断',a));ok=main and not placeholder and not bad_date
        complete=ok and len(terms)>=c['minimum_evidence_terms'] and any(x in a for x in ['依据','因为','由于','所以'])
        detail={'judgment_present':main,'evidence_terms':terms,'placeholder':placeholder,'unsupported_dates':bad_date,'scope':'bounded_literal_coverage_not_semantic_approval'}
    elif kind=='domain_map':
        mappings={'婚姻':['媒人'],'诉讼':['证人','证据','证物'],'交易':['经纪人','中介']};wanted={'交易' if x in ['交易求财','求财'] else x for x in c.get('requested_domains',mappings)};found={};mentions=list(re.finditer(r'婚姻|诉讼|交易|求财',a));ok=True
        for i,m in enumerate(mentions):
            domain='交易' if m[0]=='求财' else m[0];end=mentions[i+1].start() if i+1<len(mentions) else len(a);segment=a[m.end():end]
            roles=[r[0] for r in re.finditer(r'媒人|证人|证据|证物|经纪人|中介',segment) if asserted(segment,r.start(),r.end())]
            if roles:
                if any(role not in mappings[domain] for role in roles):ok=False
                found[domain]=True
        ok=ok and wanted.issubset(found)
        if not ok and len(wanted)>1 and not mentions:
            raw=re.sub(r'\[S\d+\]','',a);ordered=c.get('requested_domains_ordered',c.get('requested_domains',['婚姻','诉讼','交易']))
            parts=[t for t in re.split(r'[。；;\n]',raw) if t.strip()];ordered_roles=[]
            for part in parts:
                roles=re.findall(r'媒人|证人(?:证据)?|证据|证物|经纪人|中介',part)
                if roles:ordered_roles.append(roles)
            ok=len(ordered_roles)==len(ordered) and all(all(role in mappings['交易' if domain in ['交易求财','求财'] else domain] or role=='证人证据' and domain=='诉讼' for role in roles) for domain,roles in zip(ordered,ordered_roles))
        if not ok and '分别' in a:
            domains=re.findall(r'婚姻|诉讼|交易|求财',a);roles=re.findall(r'媒人|证人|证据|证物|经纪人|中介',a)
            if len(domains)==len(roles)==len(wanted):ok=wanted.issubset({'交易' if x=='求财' else x for x in domains}) and all(role in mappings['交易' if domain=='求财' else domain] for domain,role in zip(domains,roles))
    elif kind=='timing_scope':
        ok=any(w in a for w in ['不能','不足','不够']) and any(w in a for w in ['日期','应期','时间']) and any(w in a for w in ['实际盘面','实际局','盘面','成否条件','成否的条件','只能说明','不能单独推出'])
    elif kind=='marriage_conditions':
        ok=all(w in a for w in ['吉门','吉格'])
        for w in ['不需要吉门','不用吉门','不需要吉格','不用吉格']:ok=ok and w not in a
        for m in re.finditer(r'(?:一定|必定|肯定)(?:能|会)?(?:结婚|成功|成婚)',a):
            before=re.split('但是|但|然而|不过',re.split('[。；;\n]',a[:m.start()])[-1])[-1]
            negated_inference=bool(re.search(r'(?:不能|不可|不足以|不应|无法)[^。；;\n]{0,35}(?:断|认定|保证|认为|推出)',before))
            if asserted(a,m.start(),m.end()) and not negated_inference:ok=False
    else:
        old=legacy.fact_check(record,gold);ok=old['status']=='pass';detail={'legacy_limited_check':old}
        if kind=='exact_name' and not ok and c.get('answer'):
            label=c['answer'];alternatives=c.get('aliases',[]);labels=[label,*alternatives]
            if label.endswith('星'):labels.append(label[:-1])
            ok=not legacy.reasoning_conflicts(a,c) and any(any(asserted(a,m.start(),m.end()) for m in re.finditer(re.escape(x),a)) for x in labels)
        complete=ok
    return {'status':'pass' if ok else 'needs_review','reason':'typed_source_fact_contract','complete':bool(complete and ok),'detail':detail,'professional_approval':False}


def citation_check(record,gold):
    visible=set(re.findall(r'\[(S\d+)\]','\n'.join(m['content'] for m in gold['messages'] if m['role']=='user')));used=set(re.findall(r'\[(S\d+)\]',record['answer']))
    if not visible:return {'ok':not used,'required':False}
    groups=gold['provenance'].get('citation_rule_groups',[])
    covered=bool(used) and used.issubset(visible) and all(set(g)&used for g in groups)
    return {'ok':covered,'required':True,'scope':'curated_rule_group_coverage' if groups else 'visible_membership_only','semantic_support_verified':False}


def score(rows,gold):
    targets={r['id']:r for r in gold};seen=set();groups=defaultdict(list);items=[];cite=[]
    for r in rows:
        if r['id'] not in targets or r['id'] in seen:raise ValueError('Duplicate or unknown evaluation ID')
        g=targets[r['id']];seen.add(r['id'])
        if r['question']!=g['messages'][:-1] or r['expected']!=g['answer']:raise ValueError('Changed source question or gold')
        fact=fact_check(r,g);c=citation_check(r,g);automatic=bool(g.get('checks') and g['checks']['kind']!='mentions')
        if automatic:groups[g['skill']].append(fact)
        if c['required']:cite.append(c['ok'])
        items.append({'id':r['id'],'skill':g['skill'],'fact':fact,'citation':c,'automatic':automatic})
    agg={k:{'pass':sum(x['status']=='pass' for x in v),'complete':sum(x['complete'] for x in v),'total':len(v),'rate':sum(x['status']=='pass' for x in v)/len(v),'complete_rate':sum(x['complete'] for x in v)/len(v)} for k,v in groups.items()}
    return {'complete':set(targets)==seen,'groups':agg,'items':items,'citation':{'pass':sum(cite),'total':len(cite),'rate':sum(cite)/len(cite) if cite else None},'professional_approval':False,'real_prediction_accuracy':None}


def acceptance(current,before):
    """Fixed factual safety floor plus matched baseline retention, not loss."""
    floors={'direction':.9,'given_direction':.85,'chart_direction':.8,'door_relation':.8,'identity':.85,'property':.9,'typed_pair':.8,'source_fact':.8}
    missing=[k for k in floors if not current['groups'].get(k,{}).get('total')];failures=[]
    for k,minimum in floors.items():
        g=current['groups'].get(k,{});p=before['groups'].get(k,{})
        if not g or g['rate']<minimum:failures.append(k+':floor')
        if p and (p['total']!=g.get('total') or g['rate']<p['rate']-.05):failures.append(k+':retention')
    return {'passed':current['complete'] and not missing and not failures and (current['citation']['rate'] or 0)>=.9,'missing':missing,'failed':failures,'thresholds':floors,'professional_approval':False}
