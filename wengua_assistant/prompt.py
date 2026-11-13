"""Civil-time adapter and whole-source, actual-history inference budgeting."""
from copy import deepcopy
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import json

EXTRA_SYSTEM = ('按用户现在的问题回答。用正常对话长度先说判断，再解释关键原因和证据；'
                '不要抄整段教材或重复结论。区分星本体五行、落宫五行、人物用神；'
                '同宫不表示力量相同。已给的出生年干不得改成未知。'
                '用户更改人物或条件时重新核对；前一轮模型回答不是新的事实或原书证据。'
                '不能由一个吉凶符号推出确诊、确定寿命或精确应期。')


def aware_time(value, zone):
    """Attach a real timezone to local input; DST gaps/folds need an offset."""
    clock = datetime.fromisoformat(value.strip())
    tz = ZoneInfo(zone)
    if clock.tzinfo is not None:
        local = clock.astimezone(tz)
        if clock.replace(tzinfo=None) != local.replace(tzinfo=None):
            raise ValueError('所填时差与所选地点的当地时间不一致')
        return local.isoformat()
    candidates = {}
    for fold in (0, 1):
        candidate = clock.replace(tzinfo=tz, fold=fold)
        back = candidate.astimezone(timezone.utc).astimezone(tz)
        if back.replace(tzinfo=None) == clock:
            candidates[candidate.utcoffset()] = candidate
    if len(candidates) != 1:
        raise ValueError('此当地时间不存在或重复，请给出带明确时差的完整时间')
    return next(iter(candidates.values())).isoformat()


def cast(value, zone):
    from wengua_course.runtime import configure
    configure()
    from src.wengua.book_rag_query import cast_time
    return cast_time(aware_time(value, zone), zone)


def chart_fields(packet):
    """Only positions from the existing chart; no traditional scoring."""
    from src.wengua.casting_shenqi import xun_shou
    chart = packet['chart']; fields = {}
    for kind, pillar in packet['calendar']['four_pillars'].items():
        stem = pillar[0]; locator = xun_shou(pillar)[2] if stem == '甲' else stem
        fields[kind] = {'pillar': pillar, 'stem': stem, 'located_as': locator,
                        'heaven_palaces': [int(p) for p, c in chart['palaces'].items()
                                            if any(s['carried_stem'] == locator for s in c['heaven_stars'])],
                        'earth_palaces': [int(p) for p, c in chart['palaces'].items() if c['earth_stem'] == locator]}
    return fields


def validate_facts(facts):
    if not isinstance(facts, dict) or set(facts) - {'background', 'asker_relation', 'birth_stem', 'birth_status'}:
        raise ValueError('资料字段不正确')
    clean = {}
    for key in ('background', 'asker_relation'):
        value = facts.get(key, '')
        if not isinstance(value, str) or len(value) > 1500:
            raise ValueError('资料过长或格式不正确')
        if value.strip(): clean[key] = value.strip()
    status = facts.get('birth_status', 'not_entered')
    if status not in {'not_entered', 'known', 'unknown'}:
        raise ValueError('年命资料状态不正确')
    if status == 'known':
        if facts.get('birth_stem') not in '甲乙丙丁戊己庚辛壬癸' or len(facts.get('birth_stem', '')) != 1:
            raise ValueError('请填写明确的出生年干')
        clean.update(birth_status='known', birth_stem=facts['birth_stem'])
    elif status == 'unknown':
        clean['birth_status'] = 'unknown'
    # Unfilled form is NOT a claim that the question supplies no birth data.
    return clean


def build(question, tokenizer, *, domain=None, packet=None, facts=None, history=(), previous_refs=(), max_new_tokens=768):
    from wengua_course.runtime import configure
    configure()
    from src.wengua.book_rag_query import SYSTEM, checked_chart, compact_chart
    from .sources import retrieve
    if not isinstance(question, str) or not question.strip() or len(question) > 4000:
        raise ValueError('请输入问题，最多4000字')
    question = question.strip()
    if len(history) % 2 or any(x.get('role') != ('user' if i % 2 == 0 else 'assistant')
                              or not isinstance(x.get('content'), str) for i, x in enumerate(history)):
        raise ValueError('对话历史必须是真实的用户问题和本模型回答')
    # Retrieval uses the actual question history for omitted subjects; quoted
    # model answers are deliberately not used as retrieval facts.
    query = '\n'.join([x['content'] for x in history if x['role'] == 'user'][-3:] + [question])
    found = retrieve(query[-4000:], domain, 'reading' if packet else 'knowledge')
    if packet and len(found['domain_candidates']) != 1:
        raise ValueError('请在领域选项中选定当前问事的领域，避免把不同领域的取用混在一起')
    offset = max([int(x['citation'][1:]) for x in previous_refs] + [0])
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens < 4096:
        raise ValueError('回答长度必须小于4096上下文')
    facts = validate_facts(facts or {})
    base = '任务：' + found['mode'] + '。\n用户问题：' + question
    if facts: base += '\n用户明确资料：' + json.dumps(facts, ensure_ascii=False)
    if packet:
        checked_chart(packet)
        base += '\nWenguaApp程序盘面：\n' + compact_chart(packet)
    else: base += '\n本题是知识问答，没有本次盘面，不补造具体宫位或应期。'
    def messages(blocks):
        return [{'role': 'system', 'content': SYSTEM + EXTRA_SYSTEM}, *deepcopy(list(history)),
                {'role': 'user', 'content': base + '\n原书证据：\n' + '\n\n'.join(blocks)}]
    def size(blocks):
        return len(tokenizer.apply_chat_template(messages(blocks), tokenize=True,
                                                add_generation_prompt=True, enable_thinking=False))
    if size([]) + max_new_tokens > 4096:
        raise ValueError('完整问题、盘面和对话已超过4096上下文，请新开对话；没有截掉已有条件')
    refs = []; blocks = []; skips = []
    for doc in found['documents']:
        if doc['kind'] != 'rule' or doc.get('case_ids'):
            raise ValueError('新问事不能混入历史案例和反馈')
        if packet and any(s['method_id'] != packet['method_id'] for s in doc['sources']):
            raise ValueError('来源方法与本局方法不一致')
        label = 'S' + str(offset + len(refs) + 1)
        bibliography = '；'.join('《' + s['book_title'] + '》' + s['chapter'] + ' 原文第' +
                                 str(s['lines'][0]) + '—' + str(s['lines'][1]) + '行' for s in doc['sources'])
        # Set labels before constructing blocks. Never run a replace/regex
        # over quoted text, even if an original passage itself contains [S1].
        block = '[' + label + '] ' + bibliography + '\n资料性质：rule；阶段：' + doc['phase'] + '。\n'
        if doc.get('editorial_notice'): block += doc['editorial_notice'] + '\n'
        for key in ('source_uncertainties', 'interpretation_limits'):
            if doc.get(key): block += key + '（整理批注，非原书断语）：' + json.dumps(doc[key], ensure_ascii=False) + '\n'
        block += doc.get('reading_content', doc['content'])
        if size([*blocks, block]) + max_new_tokens > 4096:
            skips.append({'id': doc['id'], 'reason': 'whole_evidence_bundle_does_not_fit; no condition truncated'})
            continue
        refs.append(deepcopy(doc) | {'citation': label}); blocks.append(block)
    prompt = {'messages': messages(blocks), 'references': refs, 'current_references': deepcopy(refs),
              'prompt_budget': {'context_limit': 4096, 'max_new_tokens': max_new_tokens, 'input_tokens': size(blocks),
                                'evidence_bundles_kept': len(refs), 'whole_bundles_skipped': skips,
                                'question_chart_or_source_truncated': False},
              'state': 'ready' if refs else 'insufficient_retrieved_evidence', 'mode': found['mode'],
              'chart_receipt': packet, 'question': question, 'index_sha256': found.get('index_sha256'),
              'professional_correctness_verified': False}
    # Old citations retain their identity in the saved conversation, but only
    # sources actually present THIS round may support a new assertion.
    prompt['previous_references_not_current_evidence'] = deepcopy(list(previous_refs))
    prompt['retrieval_plan'] = {k: found[k] for k in ('domain_candidates', 'workflow_candidates', 'all_workflow_rules_loaded')}
    prompt['history_turns'] = len(history) // 2
    prompt['history_source'] = 'actual_same_model_answers'
    prompt['previous_source_blocks_retrieved_again_for_current_question'] = True
    prompt['training_updates'] = 0
    return prompt
