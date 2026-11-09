"""Fixed paired retention and additional authored-evidence application checks."""
from wengua_course import data as d
from wengua_course_v58 import evaluation as old
from . import data
ordered=old.ordered;full=old.full;legacy_rag=old.legacy_rag;rag_contract=old.rag_contract


def micro(pack):
    by={r['id']:r for r in pack['records']};picked=[by[k] for k in pack['v59_retention_ids']]
    inferred=ordered([r for r in pack['records'] if r['split']=='validation' and (r.get('checks') or {}).get('kind')=='source_inference' and r['id'] not in {x['id'] for x in picked}])
    # Development belongs to its original family; no final-test family is read.
    chosen=inferred[:16]
    transitions=ordered([r for r in pack['records'] if r['split']=='validation' and r['provenance'].get('kind')=='same_rule_changed_conditions'])
    chosen.extend(transitions[:8])
    extras=ordered([r for r in pack['records'] if r['split']=='validation' and r['provenance'].get('kind') in ['edge_then_label','cast_year_birth_year_contrast','marriage_outcome_vs_timing','domain_scope_contrast'] and r['id'] not in {x['id'] for x in picked+chosen}])
    chosen.extend(extras[:32-len(chosen)])
    if len(chosen)!=32:raise ValueError('32 additional inference/scope development tasks required')
    result=picked+chosen
    if len(result)!=112 or len({r['id'] for r in result})!=112:raise ValueError('112 distinct source-bound questions required')
    return result


def pilot_training(records,count=1536):
    quotas={'positions':32,'direction':128,'given_direction':144,'chart_direction':144,'door_relation':80,'typed_pair':64,'property':32,'source_fact':48,'identity':224,'condition':192,'case':96,'workflow':64,'domain_rule':48,'author_local':128};chosen={}
    for skill,n in quotas.items():
        bucket=sorted((r for r in records if data.active(r) and r['skill']==skill),key=lambda r:(-1 if r['provenance'].get('kind')=='reviewed_source_inference' else 0 if r['provenance'].get('kind') in ['literal_connective_author_inference','same_rule_changed_conditions','edge_then_label','cast_year_birth_year_contrast'] or r['provenance'].get('v59_repaired_teacher') else 1,d.digest('v59-pilot:'+r['id'])))
        for r in bucket[:n]:chosen[r['id']]=r
    for r in ordered([r for r in records if data.active(r)]):
        if len(chosen)>=count:break
        chosen.setdefault(r['id'],r)
    if len(chosen)!=count:raise ValueError('Unique mixed repair pilot count differs')
    return list(chosen.values())
