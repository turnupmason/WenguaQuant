"""Fixed question-scope coverage and application-balanced finite training."""
from wengua_course import data as d
from wengua_course_v57 import evaluation as old
from . import data
ordered=old.ordered;full=old.full;legacy_rag=old.legacy_rag

def rag_contract(key):
    c=old.rag_contract(key)
    if key in ['exam_self','exam_proxy']:c['require_year_boundary']=False
    return c

def micro(pack):
    by={r['id']:r for r in pack['records']};mapping=pack['v57_to_v58_ids'];picked=[by[mapping[k]] for k in pack['old_fixed_micro_ids']]
    val=[r for r in pack['records'] if r['split']=='validation'];extra=[]
    people=ordered([r for r in val if r['provenance'].get('kind')=='cast_year_birth_year_contrast'])
    for role in ['年干','日干','时干']:
        for known in [False,True]:extra.append(next(r for r in people if r['checks']['answer']==role and bool(r['checks']['known_year_stem'])==known))
    for role in ['年干','日干']:extra.append(next(r for r in people if r['checks']['answer']==role and r['checks'].get('cast_year_stem') and r['checks'].get('known_year_stem') and r not in extra))
    for kind,n in [('marriage_outcome_vs_timing',2),('domain_scope_contrast',2),('source_relation_style_contrast',2),('palace_object_contrast',2)]:
        rows=ordered([r for r in val if r['provenance'].get('kind')==kind])
        if kind=='marriage_outcome_vs_timing':rows=sorted(rows,key=lambda r:r['checks']['kind']!='timing_scope')
        if kind=='domain_scope_contrast':rows=sorted(rows,key=lambda r:r['checks']['kind']!='domain_map')
        extra.extend(rows[:n])
    result=picked+extra
    if len(result)!=80 or len({r['id'] for r in result})!=80:raise ValueError('80 distinct scoped development questions required')
    return result

def pilot_training(records,count=1536):
    quotas={'positions':64,'direction':144,'given_direction':192,'chart_direction':160,'door_relation':96,'typed_pair':64,'property':64,'source_fact':64,'identity':256,'condition':224,'case':64,'workflow':48,'domain_rule':64,'author_local':10};chosen={}
    for skill,n in quotas.items():
        bucket=sorted((r for r in records if data.active(r) and r['skill']==skill),key=lambda r:(0 if r['provenance'].get('fresh_v58_retrieval_confirmed') or r['id'].startswith('v58:') and not r['id'].startswith('v58:retained') else 1 if r['provenance'].get('repair_priority') else 2,d.digest('v58-pilot:'+r['id'])))
        for r in bucket[:n]:chosen[r['id']]=r
    for r in ordered([r for r in records if data.active(r)]):
        if len(chosen)>=count:break
        chosen.setdefault(r['id'],r)
    if len(chosen)!=count:raise ValueError('Unique mixed pilot count differs')
    return list(chosen.values())
