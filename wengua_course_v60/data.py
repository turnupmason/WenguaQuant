"""Versioned source curriculum: explicit tasks, full evidence, no masked cloze.

All original material and case families survive in the parent pack. A lesson
being source-bound does not make it an independently validated prediction.
"""
from collections import Counter,defaultdict
from copy import deepcopy
import re
from wengua_course import data as d
from wengua_course_v59 import data as old
from wengua_course_v57 import data as source

SCHEMA = 'wengua_grounded_multidomain_v60'
SYSTEM = ('根据实际所问的领域、人物和已知条件回答。先给明确答案，再解释必要原因和来源依据；'
          '通常用二至五句话，多项问题逐项回答，不重复结论、不照抄无关原文。'
          '区分宫位与宫内符号、起局年干与出生年命，并核对生克方向。'
          '规则、作者复盘与现实预测分开；资料不足就指出具体缺项，不编造宫位、出生年、日期或结局。'
          '作者复盘只解释本段已记载的分析，不把后续反馈当成事前已知。')
clean = old.clean
compose = old.compose
READING = old.READING


def active(r):
    return r['split'] == 'train' and r['provenance'].get('v60_training_allowed', False)


def row(q, answer, refs, units, **kw):
    answer=re.sub(r'(?:\s*、?\s*\[S\d+\])+\s*$','',answer)
    r = source.row(q, answer, refs, units, **kw)
    r['messages'][0]['content'] = SYSTEM
    r['id'] = 'v60:' + d.digest(d.dumps([r['split'], r['messages']]))[:24]
    r['provenance'].update(source_contract_version='v60', v60_training_allowed=True)
    return r


def segments(text):
    return [s.strip() for s in re.split(r'(?<=[。！？；])', text) if s.strip()]


def author_tasks(parent):
    """Explicit source replay, with the conclusion visible and correctly labelled.

    No suffix is hidden and treated as inferable. Dates/amounts remain author's
    reported observations; they are never generated from an incomplete chart.
    Long or narrative-only steps stay in the complete source archive.
    """
    records, ledger = [], []
    for ca in parent['case_catalog']:
        for step in ca['author_steps']:
            text = step['original'].strip()
            entry = dict(case_id=ca['case_id'], step=step['step'], domain=ca['domain'],
                         source_id=step['source_id'], source_sha256=d.digest(step['original']),
                         split=ca['phase_split'], action='archive_complete_author_step', lesson_ids=[])
            if ca['phase_split'] not in ('train', 'validation'):
                entry['action'] = 'sealed_original_test'; ledger.append(entry); continue
            # The whole single step must fit; never cut its conditions to fit.
            if len(text) > 650 or not re.search(r'为|落|临|生|克|说明|判断|断', text):
                ledger.append(entry); continue
            parts = segments(text)
            causal = [i for i, s in enumerate(parts) if re.search(r'所以|因此|故断|故而|据此|说明|表明|断[他她其为]|综合', s)]
            selected = ''.join(parts[max(0,causal[-1]-1):causal[-1]+1]) if causal else text
            # Long evidence remains in input. The target quotes a complete,
            # literal sentence, not a guessed paraphrase or numeric calculation.
            if len(selected) > 420:
                ledger.append(entry); continue
            q = (f"《{ '神奇之门' if ca['case_id'].startswith('shenqi') else '开悟之门'}》"
                 f"{d.DOMAIN_ZH.get(ca['domain'], ca['domain'])}案例，作者分析第{step['step']}步。"
                 '这是已给分析的来源复盘，不是让你预测后来结局。'
                 '请指出作者在本段作出的判断及其相连条件，保持本段的适用范围。')
            answer = selected + ' [S1]'
            refs = [{'source_id': step['source_id'], 'quote': text}]
            if d.compact(text) not in d.compact(parent['units'][step['source_id']]['original_quote']):
                entry['action'] = 'archive_cross_unit_context_required'; ledger.append(entry); continue
            r = row(q, answer, refs, parent['units'], skill='author_replay',
                    split=ca['phase_split'], domain=ca['domain'], case_ids=[ca['case_id']],
                    group=ca['group'], check=None,
                    provenance={'kind': 'explicit_complete_author_replay', 'source_case_id': ca['case_id'],
                                'author_step': step['step'], 'source_step_sha256': d.digest(step['original']),
                                'literal_target': selected, 'phase': 'attributed_author_replay_only',
                                'posthoc_marker': bool(re.search(r'事后|复盘|事后验证', text)),
                                'not_independent_prediction': True, 'required_source_fragments': refs})
            records.append(r); entry.update(action='explicit_source_replay', lesson_ids=[r['id']]); ledger.append(entry)
    return records, ledger


def build(parent):
    records, dispositions, aliases = [], [], []
    protected = set(parent['fixed_micro_ids'])
    for original in parent['records']:
        r = deepcopy(original); p = r['provenance']
        unsafe = (r['skill'] in READING or r['skill'] == 'author_local'
                  or (r.get('checks') or {}).get('kind') == 'source_inference')
        reason = ('source_replay_replaces_underspecified_cloze' if r['skill'] == 'author_local'
                  else 'source_archive' if r['skill'] in READING else 'retained_source_task')
        p['v60_training_allowed'] = not unsafe and original['id'] not in protected
        p['v60_disposition'] = reason
        if original['id'] not in protected:
            r['messages'][0]['content'] = SYSTEM
            r['id'] = 'v60:retained:' + original['id']
        if original['skill']=='case':
            # These are explicit historical source replays, not fresh forecasts.
            # The old generic phase label was not evidence that every excerpt
            # was an initial prediction. Correct the generated headings only;
            # literal chart, source sentences and reported feedback are intact.
            changes={'书中初断及按语选段':'原书分析或叙述选段（阶段以原文明示为准）',
                     '书中后来反馈选段':'原书反馈记载选段'}
            p['previous_dialogue_sha256']=d.digest(d.dumps(original['messages']))
            p.update(v60_disposition='explicit_case_replay_neutral_phase_labels',
                     phase_uncertainty_preserved=True,not_independent_prediction=True)
            for m in r['messages']:
                for a,b in changes.items():m['content']=m['content'].replace(a,b)
                if m['role']=='user':m['content']=m['content'].replace(
                    '请复盘这一个原书案例：说明主要判断、作者采用的依据，并分清初断、作者按语和后来反馈。只使用所给材料，未列出的结局不补。',
                    '请复盘所给原书案例片段，说明作者判断及其依据。阶段只按原文明示区分，反馈仅作来源复盘，不纳入事前推断；未列出的内容不补。')
            r.update(skill='case_replay',checks=None,question=r['messages'][-2]['content'],answer=r['messages'][-1]['content'])
            r['id']='v60:case:'+d.digest(d.dumps([r['split'],r['messages']]))[:24]
            reason='explicit_case_replay_neutral_phase_labels'
            bycase={c['case_id']:c for c in parent['case_catalog']}
            if not r['case_ids'] or not any(bycase[k]['author_steps'] for k in r['case_ids']):
                p.update(v60_training_allowed=False,v60_disposition='archive_narrative_without_author_reasoning')
                reason='archive_narrative_without_author_reasoning'
        dispositions.append({'previous_id': original['id'], 'id': r['id'], 'action': reason,
                             'training_allowed': p['v60_training_allowed'], 'domain': r['domain']})
        records.append(r)
    made, ledger = author_tasks(parent); records += made
    # Four perspectives are used where source prerequisites are already explicit:
    # application, condition changes, correction, and source replay. No bulk
    # surface paraphrases are presented as independent evidence.
    candidates = [r for r in records if r['split'] in ('train', 'validation')
                  and (r.get('checks') or {}).get('kind') == 'condition_slots'
                  and len(r['messages']) == 3 and r['provenance'].get('support')]
    for r in candidates:
        question = source.question(r).split('\n\n原文依据：',1)[0]
        q = ('请先核对问题所属专题和题内已给条件，再给本题结论及必要依据。\n' + question)
        refs = r['provenance']['support']
        answer = r['answer']
        nr = row(q, answer, refs, parent['units'], skill='condition', split=r['split'],
                 domain=r['domain'], case_ids=r['case_ids'], group=r['group'], check=deepcopy(r['checks']),
                 provenance={'kind': 'explicit_condition_application', 'source_teacher': r['id'],
                             'hypothetical_premises_not_observed_outcomes': True})
        records.append(nr)
    # Same rule, genuinely different stated condition and expected decision.
    # Both turns remain in the same source/split. Do not pair unrelated case
    # observations, different fields, or mere alternative wording.
    groups=defaultdict(list)
    for r in candidates:
        if not r['case_ids']:
            groups[(r['domain'],r['split'],d.dumps(r['provenance']['support']))].append(r)
    for bucket in groups.values():
        seen_decisions={}
        for r in bucket:
            key=d.dumps(r['checks'].get('required',[]))
            seen_decisions.setdefault(key,r)
        choices=list(seen_decisions.values())
        if len(choices)<2:continue
        for i,a in enumerate(choices[:2]):
            b=choices[(i+1)%len(choices)]
            qa=source.question(a).split('\n\n原文依据：',1)[0]
            qb=source.question(b).split('\n\n原文依据：',1)[0]
            if qa==qb:continue
            r=row(qa,a['answer'],a['provenance']['support'],parent['units'],skill='condition',
                  split=a['split'],domain=a['domain'],group=a['group'],check=deepcopy(b['checks']),
                  provenance={'kind':'source_condition_decision_pair','source_teachers':[a['id'],b['id']],
                              'turn_contracts':[deepcopy(a['checks']),deepcopy(b['checks'])],
                              'conditions_reset_between_turns':True,'not_independent_observations':True})
            r['messages'].extend([{'role':'user','content':'现在按以下条件重新判断，上一问额外条件不沿用：'+qb},
                                  {'role':'assistant','content':b['answer']}])
            r.update(question=r['messages'][-2]['content'],answer=b['answer'],id='v60:pair:'+d.digest(d.dumps(r['messages']))[:24])
            records.append(r)
    unique = {}
    for r in records:
        key = d.dumps(r['messages'])
        if key in unique:
            previous = unique[key]
            if previous['split'] != r['split']:
                raise ValueError('Same source dialogue crosses original split')
            aliases.append({'id': r['id'], 'canonical': previous['id']}); continue
        unique[key] = r
    return dict(schema=SCHEMA, records=list(unique.values()), v60_dispositions=dispositions,
                v60_author_ledger=ledger, v60_aliases=aliases,
                old_manual_scores_used=False, model_outputs_as_teachers=False,
                original_final_test_used=False, old_fixed_micro_ids=parent['fixed_micro_ids'],
                v60_actual_rag_rows=deepcopy(parent['v59_actual_rag_rows']))


def validate(pack):
    if pack['schema'] != SCHEMA or pack.get('old_manual_scores_used') or pack.get('model_outputs_as_teachers'):
        raise ValueError('Source-only V60 required')
    ids=set(); cases={c['case_id']:c for c in pack['case_catalog']}
    for r in pack['records']:
        if r['id'] in ids: raise ValueError('Duplicate lesson ID')
        ids.add(r['id']); p=r['provenance']
        if r['messages'][-1]['content'] != r['answer']: raise ValueError('Teacher differs from final turn')
        if active(r) and ((r.get('checks') or {}).get('kind')=='source_inference' or r['skill']=='author_local'):
            raise ValueError('Underspecified cloze cannot enter training')
        if active(r) and r['split'] != 'train': raise ValueError('Non-train family in training')
        for case_id in r.get('case_ids',[]):
            if active(r) and cases[case_id]['phase_split']!='train':raise ValueError('Held-out case used for training')
        for s in p.get('support', []):
            if d.compact(s['quote']) not in d.compact(pack['units'][s['source_id']]['original_quote']):
                raise ValueError('Teacher support absent from original source')
        if p.get('kind') == 'explicit_complete_author_replay':
            ca=cases[p['source_case_id']]; step=next(s for s in ca['author_steps'] if s['step']==p['author_step'])
            if r['group']!=ca['group'] or r['split']!=ca['phase_split']: raise ValueError('Case family moved')
            if step['original'].strip() not in r['messages'][-2]['content']: raise ValueError('Incomplete author step')
            if d.compact(p['literal_target']) not in d.compact(step['original']) or d.digest(step['original'])!=p['source_step_sha256']:
                raise ValueError('Author target changed')
        if r['skill']=='case_replay':
            if not p.get('phase_uncertainty_preserved') or '书中初断及按语选段' in r['question']:
                raise ValueError('Uncertain case chronology relabelled as initial prediction')
    if len(pack['v60_author_ledger']) != sum(len(c['author_steps']) for c in cases.values()):
        raise ValueError('Author step disposition coverage incomplete')
    return True


def coverage(pack):
    result={}
    for ca in pack['case_catalog']:
        v=result.setdefault(ca['domain'], {'cases':0,'source_steps':0,'train':0,'validation':0,'sealed_cases':0,'replay_train':0})
        v['cases']+=1;v['source_steps']+=len(ca['author_steps']);v['sealed_cases']+=ca['phase_split']=='sealed_test'
    for r in pack['records']:
        v=result.setdefault(r['domain'],{'cases':0,'source_steps':0,'train':0,'validation':0,'sealed_cases':0,'replay_train':0})
        if active(r):v['train']+=1;v['replay_train']+=r['skill']=='author_replay'
        if r['split']=='validation' and r['provenance'].get('v60_training_allowed'):v['validation']+=1
    return result
