"""Book-defined five-element facts. No scores, outcome labels or model text."""
from wengua_course import data as d


def ordered(first,second):
    for item in [first,second]:
        if set(item)!={'label','element'} or item['element'] not in '金木水火土' or not item['label']:raise ValueError('Two explicitly typed objects required')
    a,b=first['element'],second['element']
    if a==b:relation='比和';edge=None
    elif d.GENERATES[a]==b:relation='生';edge=[first['label'],'生',second['label']]
    elif d.GENERATES[b]==a:relation='受生';edge=[second['label'],'生',first['label']]
    elif d.CONTROLS[a]==b:relation='克';edge=[first['label'],'克',second['label']]
    else:relation='受克';edge=[second['label'],'克',first['label']]
    return {'ordered_objects':[first,second],'relation':relation,'directed_edge':edge,'basis':'two_book_five_element_definitions','old_manual_scores_used':False,'source_rule_not_observed_outcome':True}


def checked_prompt(question,first,second):
    # Input-only derivation: no expected answer or gold relation is accepted.
    for obj in [first,second]:
        if obj['label']+'属'+obj['element'] not in question and obj['label']+'是'+obj['element'] not in question:raise ValueError('Typed premise is not explicit in this question')
    return ordered(first,second)
