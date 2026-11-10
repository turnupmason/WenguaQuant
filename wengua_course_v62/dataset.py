"""Typed source curriculum; model diagnostics stay generation-only."""
from copy import deepcopy
from collections import Counter
from wengua_course import data as d
from wengua_course_v61 import dataset as previous
SCHEMA='wengua_typed_source_application_v62'

def validate_pack(pack):
    if pack['schema']!=SCHEMA or pack.get('fresh_official_lora_required') is not True:raise ValueError('Fresh official V62 source pack required')
    checked=deepcopy(pack);checked['schema']=previous.SCHEMA
    counts=previous.validate_pack(checked)
    ids={r['id'] for r in pack['records']}
    for r in pack.get('known_regression_rows',[]):
        if r['id'] in ids or not r['not_training_or_rag_teacher']:raise ValueError('Diagnostic entered curriculum')
        if r['messages'][-1]['content']!=r['answer']:raise ValueError('Diagnostic answer mismatch')
    if any(r['source_mode']=='eval_only_no_training' for r in pack['records']):raise ValueError('Evaluation-only teacher forbidden')
    for r in pack['records']:
        evidence=r['provenance']['support']
        if not evidence or not all(d.compact(s['quote']) in d.compact(pack['units'][s['source_id']]['original_quote']) for s in evidence):raise ValueError('Unbound book evidence')
        if r['provenance'].get('rule_structure'):
            rule=r['provenance']['rule_structure']
            if not rule.get('本条可成分支') or set(rule['本条可成分支']) & set(rule.get('辅助观察',[])):raise ValueError('Branch and auxiliary conditions overlap')
    return counts
