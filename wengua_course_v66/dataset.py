"""Versioned source-only repair: visible scope and fact-first natural answers.

Old books, teachers, splits and model answers are never rewritten. This module
reads V65 *curriculum*, not its generated answers or the learning-review notes.
"""
from copy import deepcopy
from collections import Counter,defaultdict
from wengua_course import data as d
from wengua_course_v65 import dataset as old
SCHEMA='wengua_visible_scope_decision_loss_v66'
SYSTEM=old.SYSTEM
RELATIONAL={'relation','verify','repair','objects','board','door'}


def direction(a,b,x,y):
    r=old.relation(x,y)
    law=(x+'生'+y if r=='生' else x+'克'+y if r=='克' else y+'生'+x if r=='受生' else y+'克'+x if r=='受克' else '两者五行相同')
    clause={'生':a+'生'+b,'克':a+'克'+b,'受生':a+'受'+b+'所生','受克':a+'受'+b+'所克','比和':a+'与'+b+'比和'}[r]
    return f'{a}属{x}，{b}属{y}。{law}，所以{clause}。',clause


def answer(c):
    k=c['kind'];spans=[]
    if k in {'relation','verify','repair','objects'} and c.get('goal')!='extract':
        text,clause=direction(c['a'],c['b'],c['x'],c['y']);spans=[clause]
        if k=='verify':
            verdict='标注可以保留' if c['claim']==old.relation(c['x'],c['y']) else '标注需要改正'
            text+=verdict+'。';spans.append(verdict)
    elif k=='board' and c['goal']=='relation':
        (s,p),(t,q)=c['positions'];text='；'.join(f'{s}在{p}宫' for s,p in c['positions'])+'。'
        body,clause=direction(p+'宫',q+'宫',old.PALACES[p],old.PALACES[q]);text+=body;spans=[clause]
    elif k=='door':
        r=old.relation(old.DOORS[c['door']],old.PALACES[c['palace']]);name={'克':'门迫','受克':'门受制','生':'门生宫','受生':'宫生门','比和':'门宫比和'}[r]
        text,clause=direction(c['door']+'门',c['palace']+'宫',old.DOORS[c['door']],old.PALACES[c['palace']]);spans=[clause]
        text+='这一层叫'+name+'。';spans.append('这一层叫'+name)
        if 'claim' in c:
            verdict='名称可以保留' if c['claim']==name else '名称需要改正';text+=verdict+'。';spans.append(verdict)
    elif k=='marriage':
        rel=c['palace_relation']
        if c['state']=='已具备':
            text=f'按这条婚姻规则，恋爱可成的分支成立。乙庚两宫{rel}，又已核实具备吉门吉格，两项前提都满足。';spans=['恋爱可成的分支成立']
        elif c['state']=='未提供':
            text=f'暂时不能按这条婚姻规则判成。乙庚两宫{rel}这一项已满足，吉门吉格尚未核实，需要补查；不能把未知当作没有。';spans=['暂时不能按这条婚姻规则判成','吉门吉格尚未核实']
        else:
            text=f'这条婚姻可成分支不成立。乙庚两宫虽{rel}，但已明确不具备吉门吉格；缺少其中一项前提，不能使用该分支，也不能仅因此断定婚事必败。';spans=['婚姻可成分支不成立','已明确不具备吉门吉格']
        if c['goal']=='date':text+='这条规则没有提供具体结婚日期，现有资料不能定到某一天。';spans.append('现有资料不能定到某一天')
    else:
        text=old.answer(c)
        if k in {'identity','identity_verify'}:
            f=old.ROLES[c['role']][0];spans=[f+c['fields'][f] if c['fields'].get(f) else f]
            if k=='identity_verify':spans.append(text.split('。')[0])
        elif k in {'loan','sale','retirement'}:spans=[text.split('。')[0]]
        elif k in {'attribute','locations'}:spans=[text.rstrip('。')]
        elif k=='objects':spans=[text.rstrip('。')]
    if any(s not in text for s in spans):raise ValueError('Decision span is not literal target text')
    intervals=[]
    for s in spans:
        start=text.index(s)
        if k in RELATIONAL and s==locals().get('clause'):
            # Weight the actual directed predicates, not long repeated names.
            if k=='board':a,b=p+'宫',q+'宫';x,y=old.PALACES[p],old.PALACES[q]
            elif k=='door':a,b=c['door']+'门',c['palace']+'宫';x,y=old.DOORS[c['door']],old.PALACES[c['palace']]
            else:a,b,x,y=c['a'],c['b'],c['x'],c['y']
            rel=old.relation(x,y)
            offsets=([(len(a),len(a)+1),(len(s)-2,len(s))] if rel in {'受生','受克'}
                     else [(len(s)-2,len(s))] if rel=='比和' else [(len(a),len(a)+1)])
            for left,right in offsets:intervals.append({'start':start+left,'end':start+right,'text':s[left:right]})
        else:intervals.append({'start':start,'end':start+len(s),'text':s})
    return text,intervals


def visible_question(question,c):
    k=c['kind']
    if k=='marriage':
        # The only relevant implication, not other chapters or post-outcome facts.
        return ('本问是恋爱婚姻预测。采用的本条规则是：乙庚两宫相生或比和，并且逢吉门吉格，才可据此判恋爱可成。这里只检查这一可成分支。\n'+question)
    if k=='loan':return '本问是银行放贷，所问结果针对银行方。\n'+question
    if k=='retirement':return '本问是退役申请，所问结论是领导准退还是留用。\n'+question
    if k=='sale':return '本问是卖货，所问销售情况，利润另需条件。\n'+question
    return question


def contract(c,question):
    c=deepcopy(c)
    if c['kind']=='marriage':
        question=question.split('参考原文：',1)[0]
        # V65 records omitted this known fact from metadata. Read it from the
        # actual user prompt, never infer from the assistant conclusion.
        if '比和' in question and '相生' not in question:c['palace_relation']='比和'
        elif '相生' in question and '比和' not in question:c['palace_relation']='相生'
        else:raise ValueError('Marriage actual palace relation is ambiguous')
    return c


def rewrite(row):
    r=deepcopy(row);p=r['provenance'];c=p.get('lesson_contract')
    if not c:return r
    c=contract(c,r['messages'][-2]['content'] if len(r['messages'])==3 else r['messages'][1]['content'])
    if p.get('initial_contract'):
        first=contract(p['initial_contract'],r['messages'][1]['content'])
        r['messages'][1]['content']=visible_question(r['messages'][1]['content'],first)
        r['messages'][2]['content']=answer(first)[0];p['initial_contract']=first
    r['messages'][-2]['content']=visible_question(r['messages'][-2]['content'],c)
    text,intervals=answer(c);r['messages'][-1]['content']=text;r['answer']=text;r['question']=r['messages'][1]['content']
    p.update(lesson_contract=c,decision_spans=intervals,v65_original_id=r['id'],visible_scope_repaired=True)
    r['id']='v66:'+d.digest(d.dumps(r['messages']))[:24];r['format']='visible_scope_fact_first_dialogue'
    return r


def bucket(r):
    c=r['provenance'].get('lesson_contract') or {};k=c.get('kind',r['skill'])
    if k in {'relation','repair','verify'}:return (c['x'],c['y'],str(c.get('claim')==old.relation(c['x'],c['y'])) if k=='verify' else '')
    if k=='objects':return (c['a'],c['b'],c['goal'])
    if k=='board':return (c['board_id'],c['goal'])
    if k=='door':return (c['door'],c['palace'])
    if k in {'identity','identity_verify'}:return (c['role'],bool(c.get('birth')),bool(r['provenance'].get('initial_contract')))
    if k=='marriage':return (c['state'],c['goal'],c['palace_relation'],bool(r['provenance'].get('initial_contract')))
    return (d.dumps(c),)


def pilot_selection(rows):
    # Finite stratified curriculum, not thousands of spelling-only repeats.
    limits={'source_dialogue':60,'relation':25,'verify':50,'repair':25,'objects':72,'board':48,'identity':56,'door':64,'marriage':24,'attribute':18}
    selected=[]
    for skill in sorted({r['skill'] for r in rows}):
        if skill=='objects':
            # All 8x8 ordered outer-palace pairs plus 8 intrinsic-star contrasts.
            grouped=defaultdict(list)
            for r in rows:
                c=r['provenance'].get('lesson_contract',{})
                if r['skill']==skill and c['goal']=='relation' and ((c['a'] in [p+'宫' for p in old.PALACES] and c['b'] in [p+'宫' for p in old.PALACES]) or c['a']=='天心星'):
                    grouped[(c['a'],c['b'])].append(r)
            selected += [min(v,key=lambda r:d.digest('v66-object:'+r['id'])) for _,v in sorted(grouped.items())]
            continue
        group=defaultdict(list)
        for r in rows:
            if r['skill']==skill:group[bucket(r)].append(r)
        for v in group.values():v.sort(key=lambda r:d.digest('v66-order:'+r['id']))
        chosen=[];limit=limits.get(skill,16)
        for depth in range(max(map(len,group.values()),default=0)):
            for key in sorted(group):
                if depth<len(group[key]) and len(chosen)<limit:chosen.append(group[key][depth])
        selected+=chosen
    return {r['id'] for r in selected}


def build(source):
    if source['schema']!=old.SCHEMA:raise ValueError('Frozen V65 source required')
    out=deepcopy(source);rows=[rewrite(r) for r in source['records']];ids=pilot_selection([r for r in rows if r['split']=='train'])
    for r in rows:
        r['pilot_selected']=r['id'] in ids
    out.update(schema=SCHEMA,records=rows,dispositions=[{'old_id':r['id'],'action':'source_replay_unchanged' if r['skill']=='source_dialogue' else 'visible_scope_fact_first_target'} for r in source['records']],
        pilot_ids=sorted(ids),model_outputs_used_as_teachers=False)
    out['summary']={'schema':SCHEMA,'preserved_train_cores':sum(r['split']=='train' for r in rows),'pilot_train_cores':len(ids),
        'pilot_skills':dict(Counter(r['skill'] for r in rows if r['id'] in ids)),'development':sum(r['split']=='validation' for r in rows),
        'source_units':len(out['units']),'original_case_registry':len(out['case_splits']),'new_books':False,'old_manual_scores':False,
        'full_case_training_claimed':False,'new_blind_benchmark':False,'ordinary_vs_decision_weighted_control':True}
    validate(out,source);return out


def validate(pack,source):
    if pack['units']!=source['units'] or pack['case_splits']!=source['case_splits']:raise ValueError('Original source or final-test split changed')
    if pack['schema']!=SCHEMA or pack.get('old_manual_scores_used') or pack.get('model_outputs_used_as_teachers'):raise ValueError('Wrong teacher track')
    expected=[rewrite(r) for r in source['records']]
    for a,b in zip(pack['records'],expected,strict=True):
        if a!={**b,'pilot_selected':a['id'] in pack['pilot_ids']}:raise ValueError('Visible input, source, target or decision span differs')
    actual=pilot_selection([r for r in expected if r['split']=='train'])
    if set(pack['pilot_ids'])!=actual:raise ValueError('Balanced short training selection changed')
    for r in pack['records']:
        for s in r['provenance'].get('support',[]):
            if d.compact(s['quote']) not in d.compact(pack['units'][s['source_id']]['original_quote']):raise ValueError('Unsupported teacher citation')
    return True
