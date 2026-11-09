"""Strictly sourced, direct-answer practice. No model answers or legacy scores."""
from collections import Counter
from wengua_course import data as d
SCHEMA='wengua_natural_dialogue_v61'
SYSTEM=('按用户当前问题正常交流，先回答他实际问的事，再解释有用的原因。'
        '不用固定开场白或备课式标题，不重复结论，不逐段复述参考材料。'
        '按当前领域、人物和条件选规则；改变条件时重新判断。'
        '盘由程序提供，不自行补造宫位、出生年、日期、结局或百分比。'
        '复盘书中案例时分清作者分析与后来反馈。')

def validate_pack(pack):
    if pack['schema']!=SCHEMA or set(pack['source_books'])!=d.BOOKS:raise ValueError('Wrong two-book curriculum')
    if any(pack.get(k) for k in ['model_outputs_used_as_teachers','old_manual_scores_used','original_final_test_used','new_books_used']):raise ValueError('Forbidden teacher or test source')
    ids=set();prompts={};owners={};counts=Counter();units=pack['units']
    for r in pack['records']:
        if r['id'] in ids:raise ValueError('Duplicate lesson')
        ids.add(r['id']);counts[r['split']]+=1
        if r['split'] not in ('train','validation'):raise ValueError('No diagnostic or final test in active lesson pack')
        if r['messages'][0]['role']!='system' or r['messages'][-1]!={'role':'assistant','content':r['answer']}:raise ValueError('Teacher/message mismatch')
        if not r['answer'].strip() or d.title_only(r['answer']):raise ValueError('Empty teacher')
        if any(t in r['answer'] for t in d.BAD):raise ValueError('Legacy template or score teacher')
        for i,m in enumerate(r['messages'][1:]):
            if m['role']!=('user' if i%2==0 else 'assistant'):raise ValueError('Broken conversation turns')
            if r['split']=='train' and m['role']=='assistant' and len(m['content'])<12:raise ValueError('Teacher is a bare token')
        key=d.dumps(r['messages'][:-1])
        if key in prompts:raise ValueError('Duplicate or conflicting prompt')
        prompts[key]=r['answer']
        support=r['provenance']['support']
        if set(r['source_ids'])!={s['source_id'] for s in support}:raise ValueError('Missing evidence reference')
        if not support or r['provenance'].get('teacher_is_model_output'):raise ValueError('No source or generated-output teacher')
        for s in support:
            unit=units.get(s['source_id'])
            if not unit or unit['book_id'] not in d.BOOKS or not s['quote'].strip() or d.compact(s['quote']) not in d.compact(unit['original_quote']):raise ValueError('Evidence not literal at source')
        for cid in r['case_ids']:
            phase=pack['case_splits'][cid]
            if phase=='sealed_test' or phase!=r['split']:raise ValueError('Original case boundary changed')
            owners.setdefault(cid,set()).add(r['split'])
        if r['split']=='train' and not r['provenance'].get('source_was_train'):raise ValueError('Validation source moved into teachers')
    if not counts['train'] or not counts['validation']:raise ValueError('Train/development required')
    if any(len(x)>1 for x in owners.values()):raise ValueError('Case family crosses splits')
    return dict(counts)
