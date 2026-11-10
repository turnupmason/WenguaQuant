"""V63 changes only SYSTEM; literal book evidence, users and teachers stay intact."""
from copy import deepcopy
from wengua_course import data as d
from wengua_course_v62 import dataset as source

# Identical to the neutral control, but no evaluation-answer module is imported
# by training. This constant is itself inside the sealed V63 runtime hashes.
SYSTEM = '按当前用户问题正常交流，直接说明结果及有用的原因，不用固定开场白或备课式标题。先核对人物、所问目标和已给条件；没有给出的事实不补造。有参考资料时按其适用范围回答，不照抄无关段落。普通问题按普通问题回答。'

SCHEMA = 'wengua_source_learning_repair_v63'
MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
RECALL_SKILLS = ['identity', 'direction', 'door_relation', 'branch_conditions', 'symbol_types', 'source_dialogue']


def derive(pack):
    source.validate_pack(pack)
    out = deepcopy(pack); out['schema'] = SCHEMA
    for key in ['records', 'known_regression_rows', 'actual_rag_rows']:
        for row in out[key]:
            if row['messages'][0]['role'] != 'system':
                raise ValueError('Explicit SYSTEM turn required')
            row['messages'][0]['content'] = SYSTEM
    out['summary'] = pack['summary'] | {
        'rates': [5e-5, 1e-4], 'updates_per_epoch': 53, 'updates_per_trial': 424,
        'epochs': 8, 'max_train_tokens': None, 'max_tokens_recomputed_before_training': True,
        'initialization': 'official_fresh_all_seven_projections',
        'source_users_and_teachers_changed': False, 'one_system_in_training_loss_generation_and_rag': True,
        'training_recall_prompts': 12, 'free_generation_epochs': [1, 2, 4, 8]}
    return out


def verify_derivation(before, after):
    """Compare every byte of every user/assistant turn and all book provenance."""
    expected = derive(before)
    if after != expected:
        raise ValueError('V63 may change SYSTEM and declared metadata only')
    if before['units'] != after['units'] or before['case_splits'] != after['case_splits']:
        raise ValueError('Source or split changed')
    for key in ['records', 'known_regression_rows', 'actual_rag_rows']:
        for a, b in zip(before[key], after[key], strict=True):
            if a['messages'][1:] != b['messages'][1:] or a['answer'] != b['answer']:
                raise ValueError('Source user or teacher changed')
    return True


def learning_rows(pack, *, testing=False):
    """Select only original TRAIN records; never import diagnostic answer targets."""
    rows = [r for r in pack['records'] if r['split'] == 'train']
    if testing:
        return rows[:2]
    selected = []
    for skill in RECALL_SKILLS:
        subset = [r for r in rows if r['skill'] == skill]
        first = min(subset, key=lambda r: d.digest('v62-training-recall:' + r['id']))
        selected.append(first)
        selected.append(min((r for r in subset if r['id'] != first['id']),
                            key=lambda r: d.digest('v63-second-training-recall:' + r['id'])))
    if len({r['id'] for r in selected}) != 12:
        raise ValueError('Twelve distinct original learning prompts required')
    return selected
