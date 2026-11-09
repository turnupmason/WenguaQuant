"""Fixed old retention plus explicit rule tasks across source-supported domains."""
from collections import defaultdict
from wengua_course import data as d
from wengua_course_v59 import evaluation as old
from . import data
ordered=old.ordered
legacy_rag=old.legacy_rag


def micro(pack):
    by={r['id']:r for r in pack['records']}
    # Preserve the oldest 80 stable questions, including their exact system.
    original=pack.get('v59_retention_ids',[])[:80]
    picked=[by[k] for k in original]
    groups=defaultdict(list)
    for r in pack['records']:
        c=r.get('checks') or {}
        if r['split']=='validation' and c.get('kind') in ('condition_slots','exact_name','state','role','person_contract','birth_boundary','role_boundary','marriage_conditions','timing_scope','domain_map'):
            if r['id'] not in {x['id'] for x in picked} and r['provenance'].get('v60_training_allowed'):
                groups[r['domain']].append(r)
    buckets={k:ordered(v) for k,v in groups.items()}
    extra=[]
    while len(extra)<32:
        changed=False
        for domain in sorted(buckets):
            if buckets[domain] and len(extra)<32:extra.append(buckets[domain].pop(0));changed=True
        if not changed:break
    if len(extra)!=32 or len(picked)!=80:raise ValueError('Fixed 80+32 unambiguous development tasks required')
    return picked+extra


def pilot_training(records,count=768):
    """Balance domains before filling with basic symbolic drills."""
    candidates=[r for r in records if data.active(r)]
    groups=defaultdict(list)
    for r in candidates:
        if r['skill'] not in ('positions','direction','given_direction','chart_direction','door_relation','typed_pair','property'):
            groups[r['domain']].append(r)
    chosen={}
    for domain in sorted(groups):
        bucket=sorted(groups[domain],key=lambda r:(0 if r['provenance'].get('kind')=='source_condition_decision_pair' else 1 if r['provenance'].get('kind')=='explicit_condition_application' else 2 if r['skill']=='author_replay' else 3,d.digest(r['id'])))
        for r in bucket[:12]:chosen[r['id']]=r
    for skill,n in [('identity',64),('condition',64),('author_replay',64),('case_replay',32),('domain_rule',32),('workflow',32),('direction',40),('given_direction',40),('chart_direction',40),('door_relation',24),('typed_pair',24),('property',16),('source_fact',24)]:
        for r in ordered([r for r in candidates if r['skill']==skill])[:n]:
            if len(chosen)<count:chosen.setdefault(r['id'],r)
    for r in ordered(candidates):
        if len(chosen)>=count:break
        chosen.setdefault(r['id'],r)
    if len(chosen)!=count:raise ValueError('Mixed pilot size mismatch')
    return list(chosen.values())


def final(pack,count=192):
    # Open author semantics stay separate from finite automatic rule contracts.
    available=[r for r in pack['records'] if r['split']=='validation' and r['provenance'].get('v60_training_allowed')]
    groups=defaultdict(list)
    for r in available:groups[r['domain']].append(r)
    chosen={}
    for domain in sorted(groups):
        whole=ordered([r for r in groups[domain] if r['skill']=='case_replay'])
        for r in whole[:1]:chosen[r['id']]=r
        replay=ordered([r for r in groups[domain] if r['skill']=='author_replay'])
        for r in replay[:2]:chosen[r['id']]=r
        for r in ordered(groups[domain])[:3]:chosen.setdefault(r['id'],r)
    for r in ordered(available):
        if len(chosen)>=count:break
        chosen.setdefault(r['id'],r)
    if len(chosen)!=count:raise ValueError('Final review size mismatch')
    return list(chosen.values())
