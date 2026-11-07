"""Limited source slots and explicit contradictions, with natural chain answers.

No keyword check is promoted to expert approval of an open historical case.
"""
import re
from types import FunctionType
from collections import defaultdict
from wengua_course import data as d
from wengua_course_v55 import checks as old
from wengua_course_v53 import checks as legacy

citation_check=old.citation_check
positive_occurrences=old.positive_occurrences
unknown_year_misuse=old.unknown_year_misuse


def relation_chain_ok(answer,check):
    if old.reasoning_conflicts(answer,check):return False
    a,b=check['elements'];rel=check['answer']
    head=re.match(r'^\s*(受生|受克|比和|生|克)[。；，,]',answer)
    if head and head[1]!=rel:return False
    subjects=dict(zip(check.get('symbols') or ['前者','后者'],[a,b]))
    subjects.update({'前者':a,'后者':b,'A':a,'B':b,'对象A':a,'对象B':b})
    subjects.update({d.PALACE[n]+'宫':element for n,element in d.ELEMENT.items()})
    for subject,element in subjects.items():
        for m in re.finditer(re.escape(subject)+r'(?:的五行|属性)?(?:也属|属|为|属于)([金木水火土])',answer):
            if m[1]!=element:return False
    if check.get('require_positions') and not legacy.positions_ok(answer,check['positions']):return False
    conclusions=re.findall(r'(?:结论|关系)(?:为|是)?[：:为是\s]*(受生|受克|比和|生|克)',answer)
    if conclusions:
        if set(conclusions)!={rel}:return False
    elif not old.relation_ok(answer,check):return False
    if a==b:
        return any(w in answer for w in ['同属'+a,'五行相同','均属'+a,'都属'+a])
    x,y=(a,b) if rel in ['生','克'] else (b,a);op='生' if rel in ['生','受生'] else '克'
    if not list(positive_occurrences(answer,x+op+y)):return False
    # Both input elements must be explicitly retained, not silently replaced.
    return bool(re.search('(?:属|为)'+a,answer)) and bool(re.search('(?:属|为)'+b,answer))


def fact_check(row,gold):
    answer=row['answer'];c=gold.get('checks') or {};kind=c.get('kind')
    if not answer.strip():return {'status':'fail','reason':'empty'}
    if row.get('reached_token_limit'):return {'status':'fail','reason':'output_limit'}
    if re.search(r'(.{2,24})\1{7,}',re.sub(r'\s+','',answer)):return {'status':'fail','reason':'repeated_output'}
    if old.unknown_year_misuse(answer,gold):return {'status':'fail','reason':'unknown_birth_year_used_as_known'}
    # Old gold and scores remain immutable. This is an explicitly new rubric
    # accepting a correct fact-law-conclusion order on the same exposed tasks.
    if kind in ['reasoning_chain','relation','joint_relation'] and c.get('elements'):
        check=c|{'require_positions':kind=='joint_relation' or c.get('require_positions',False)}
        ok=relation_chain_ok(answer,check)
    elif kind=='role_boundary':
        matches=[]
        for m in re.finditer(r'(?:取|以|用|先看|改取|看|按)(?:起局)?(年干|月干|日干|时干)',answer):
            if re.search(r'(?:不|不能|暂不|不要)$',answer[max(0,m.start()-4):m.start()]):continue
            matches.append(m[1])
        for m in re.finditer(r'(年干|月干|日干|时干)(?:落宫)?(?:代表|为)(?:考生|患者|本人|父亲|子女|问事人)',answer):
            if re.search(r'(?:不|不是|不能取)$',answer[max(0,m.start()-4):m.start()]):continue
            matches.append(m[1])
        matches.extend(re.findall(r'主(?:要)?用神(?:是|为)(?:起局)?(年干|月干|日干|时干)',answer))
        ok=c['answer'] in matches and len(set(matches))==1
        if c.get('unknown_year'):
            ok=ok and bool(re.search(r'(?:不补造|暂不取|不取|不计算|暂不计算|不能取|不补|暂不使用|不能确定)[^。；\n]{0,10}年命|年命[^。；\n]{0,12}(?:待|未知|暂缺|暂不使用|暂不能|不能确定|无从)',answer))
    elif kind=='condition_slots':
        ok=all(any(list(positive_occurrences(answer,w)) for w in words) for words in c['required'])
        ok=ok and not any(list(positive_occurrences(answer,w)) for w in c.get('forbidden',[]))
        rule=gold.get('provenance',{}).get('rule_id')
        if rule=='rag:ff1d59b30fe9012908ac4b6b':
            side=c['required'][0][0]
            ok=ok and bool(re.search(re.escape(side)+r'(?:可能|或许|会|有意)?(?:主动)?(?:求和|寻求和解|有调解意愿)',answer))
            ok=ok and not re.search(re.escape(side)+r'[^。；\n]{0,5}(?:不会|不愿|没有)[^。；\n]{0,5}(?:求和|和解|调解)',answer)
        elif rule=='rag:312e1faa0b3ecc6bb442c49d':
            ok=ok and any(w in answer for w in ['放贷人','银行'])
        elif rule=='rag:d6ee057cb5c1247772a08bfc':
            ok=ok and not any(w in answer for w in ['一定被盗','肯定被盗','必定被盗','一定被偷','肯定被偷','必定被偷','确定失盗'])
            if '忘记' in c['required'][0]:ok=ok and bool(re.search(r'(?:也)?可能(?:自己)?忘记',answer))
    else:
        result=old.fact_check(row,gold)
        if result['status']!='pass' and kind=='exact_name' and '名曰'+c['answer'] in answer and not old.reasoning_conflicts(answer,c):
            return {'status':'pass','reason':'explicit_name_slot','semantic_professional_approval':False}
        return result
    return {'status':'pass' if ok else 'needs_review','reason':'declared_facts_direction_conditions_only','semantic_professional_approval':False}


score=FunctionType(old.score.__code__,globals(),'score',old.score.__defaults__)


def gate(summary,*,phase='repair'):
    required={'direction':.9,'given_direction':.85,'chart_direction':.8,'positions':.9,'door_relation':.8,'identity':.85,'condition':.8,'typed_pair':.8,'property':.9,'source_fact':.8}
    missing=[k for k in required if not summary['groups'].get(k,{}).get('total')]
    failures=[k for k,v in required.items() if summary['groups'].get(k,{}).get('rate',0)<v]
    return {'passed':summary['complete'] and not missing and not failures and (summary['citation']['rate'] or 0)>=.9,
            'thresholds':required,'missing_categories':missing,'failed_categories':failures,'phase':phase,'professional_approval':False}


rank=old.rank
