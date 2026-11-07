"""Read-only V55 ZIP review; never changes frozen training scores or sources.

Uses the restored private V55 course. A ZIP is read in place to avoid a second
copy of private book text. Overlays identify explicit defects, not expert
semantic accuracy, and must never become RAG documents or SFT targets.
"""
from pathlib import Path, PurePosixPath
from collections import Counter
import argparse
import hashlib
import json
import re
import zipfile
from wengua_course import data as d
from wengua_course_v55 import runtime, checks
from wengua_course_v53 import checks as legacy
from wengua_course_v54 import checks as v54_checks

TRIALS = ('source_lr_1em05', 'source_lr_2em05')


def normalized_content(text):
    text = re.sub(r'(?m)^\s*\d+[、.]', '', text)
    return re.sub(r'\s|\[S\d+\]', '', text)


def answer_overlay(row, gold):
    """Finite observed properties; no modification of the original rubric."""
    answer = row['answer']
    check = gold.get('checks') or {}
    positions = check.get('positions')
    result = {
        'content_equals_target_ignoring_citations_whitespace_numbering':
            normalized_content(answer) == normalized_content(gold['answer']),
        'literal_or_subject_conflicts': checks.reasoning_conflicts(answer, check),
        'repeated_output': bool(re.search(r'(.{2,24})\1{7,}', re.sub(r'\s+', '', answer))),
        'reached_output_limit': bool(row.get('reached_token_limit')),
        'semantic_professional_approval': False,
    }
    if positions:
        result['positions_match'] = legacy.positions_ok(answer, positions)
    # Additional wording observation only. In particular a correct first label
    # never cancels a contradictory explanation.
    if check.get('kind') in ('relation', 'joint_relation'):
        match = re.match(r'(?:应改为|应改|改为|关系为|关系是)?(受生|受克|比和|生|克)(?=[。；，,.\s]|$)', answer)
        result['headline_relation'] = match[1] if match else None
        result['headline_matches_target'] = bool(match and match[1] == check['answer'])
    return result


def bridge_boundary_candidates(gold):
    """Textual flags for review, not a count of conclusively invalid lessons."""
    if gold.get('skill') != 'author_evidence':
        return []
    target = gold['provenance']['literal_target']
    evidence = gold['messages'][-2]['content'].split('作者证据：', 1)[-1]
    patterns = [
        r'[开休生伤杜景死惊]门(?:伏吟|反吟|生宫|克宫|逢空|空亡|入墓)',
        r'上乘(?:九天|九地|值符|六合|白虎|玄武|太阴|螣蛇)',
        r'[甲乙丙丁戊己庚辛壬癸][+＋][甲乙丙丁戊己庚辛壬癸]',
    ]
    return sorted({m[0] for pattern in patterns for m in re.finditer(pattern, target)
                   if m[0] not in evidence})


def read_rows(z, path):
    return [json.loads(line) for line in z.read(path).splitlines() if line.strip()]


def inspect_archive(z):
    names = z.namelist()
    if len(names) != len(set(names)):
        raise ValueError('Duplicate ZIP member')
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name:
            raise ValueError('Unsafe ZIP member name')
    if sum(x.file_size for x in z.infolist()) > 128 * 2**20:
        raise ValueError('Review ZIP exceeds declared size limit')


def audit(archive, output, *, workbook=False):
    archive, output = Path(archive), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pack = runtime.verify_data()
    gold = runtime.choose_development(pack)
    byid = {r['id']: r for r in gold}
    details = []
    with zipfile.ZipFile(archive) as z:
        inspect_archive(z)
        state = json.loads(z.read('run_status.json'))
        contract = json.loads(z.read('sweep_contract.json'))
        if contract.get('original_test_used') is not False:
            raise ValueError('Unexpected original test use')
        if z.testzip() is not None:
            raise ValueError('ZIP CRC failed')
        if contract['code_sha256'] != runtime.hashes():
            raise ValueError('Uploaded training code does not match frozen V55')
        if contract['dataset_release_sha256'] != d.digest(runtime.RELEASE.read_text()):
            raise ValueError('Uploaded curriculum release mismatch')
        parent = json.loads(z.read('source_parent/parent_provenance.json'))
        if parent['files']['adapter_model.safetensors'] != d.read(runtime.RELEASE)['reviewed_parent_adapter_sha256']:
            raise ValueError('Reviewed parent adapter SHA differs')
        workers = []
        for name in z.namelist():
            if not name.endswith('.result.json'):
                continue
            reply = json.loads(z.read(name))
            request_path = name[:-len('.result.json')] + '.json'
            request = json.loads(z.read(request_path))
            if (reply['request_sha256'] != d.digest(z.read(request_path))
                    or reply['nonce'] != request['nonce'] or not reply['ok']
                    or reply['software_fixture'] or reply['result'].get('state') != 'complete'):
                raise ValueError('Worker request/receipt mismatch or incomplete result')
            workers.append(reply)
        allrows = []
        for name in sorted(n for n in z.namelist() if n.endswith('.jsonl')):
            for row in read_rows(z, name):
                actual = hashlib.sha256(json.dumps(row['question'], ensure_ascii=False,
                    separators=(',', ':')).encode()).hexdigest()
                if row['prompt_sha256'] != actual:
                    raise ValueError('Prompt SHA mismatch: ' + name)
                allrows.append((name, row))
        summary = {
            'schema': 'wengua_readonly_v55_review_20261107',
            'run_state': state['state'],
            'original_nominee': json.loads(z.read('review_nominee.json'))['trial'],
            'generation_files': len({n for n, _ in allrows}),
            'generation_rows': len(allrows),
            'code_release_prompt_hashes_match': True,
            'reviewed_parent_adapter_sha_matches': True,
            'completed_nonfixture_worker_receipts': len(workers),
            'original_final_test_used': False,
            'original_scores_rewritten': False,
            'posthoc_observations_are_not_blind_validation': True,
            'trials': {},
        }
        for trial in TRIALS:
            t = state['trials'][trial]
            items = []
            for tag in ['validation_base', 'validation_parent', 'epoch_01_validation', 'epoch_02_validation']:
                name = f'trials/{trial}/generations/{tag}.jsonl'
                rows = read_rows(z, name)
                scored = checks.score(rows, gold)
                if tag != 'validation_base':
                    expected = t['parent_baseline'] if tag == 'validation_parent' else t['epochs'][int(tag[6:8])-1]['source_checks']
                    if scored != expected:
                        raise ValueError('Stored V55 checks do not reproduce: ' + name)
                byskill = {}
                for row in rows:
                    g = byid[row['id']]
                    overlay = answer_overlay(row, g)
                    details.append((trial, tag, g, row, checks.fact_check(row, g), overlay))
                    byskill.setdefault(g['skill'], []).append(overlay)
                observed = {}
                for skill, values in byskill.items():
                    observed[skill] = {
                        'total': len(values),
                        'target_content_exact': sum(v['content_equals_target_ignoring_citations_whitespace_numbering'] for v in values),
                        'explicit_reasoning_conflicts': sum(bool(v['literal_or_subject_conflicts']) for v in values),
                        'repeated': sum(v['repeated_output'] for v in values),
                        'output_limit': sum(v['reached_output_limit'] for v in values),
                        'correct_headline_but_explicit_conflict': sum(v.get('headline_matches_target', False) and bool(v['literal_or_subject_conflicts']) for v in values),
                    }
                    if any('positions_match' in v for v in values):
                        observed[skill]['positions_matching'] = sum(v.get('positions_match', False) for v in values)
                item = {'tag': tag, 'original_groups': scored['groups'], 'citation': scored['citation'], 'observations': observed}
                if tag.startswith('epoch'):
                    index = int(tag[6:8])-1
                    e = t['epochs'][index]
                    # Reconstruct every retained old-score branch without changing it.
                    for oldtag, oldgold, scorer, saved in [
                        ('v53', pack['legacy_v53_prompts'], legacy.score, 'original_v53_legacy_checks'),
                        ('v54', pack['legacy_v54_prompts'], v54_checks.score, 'original_v54_legacy_checks'),
                        ('v54', pack['legacy_v54_prompts'], checks.score, 'v54_with_v55_checks'),
                    ]:
                        oldrows = read_rows(z, f'trials/{trial}/generations/epoch_{index+1:02d}_{oldtag}_unchanged.jsonl')
                        if scorer(oldrows, oldgold) != e[saved]:
                            raise ValueError('Legacy rubric does not reproduce: ' + saved)
                    rag = read_rows(z, f'trials/{trial}/generations/epoch_{index+1:02d}_v53_actual_rag_unchanged.jsonl')
                    rg = {r['id']: r for r in pack['legacy_actual_rag_rows']}
                    if len(rag) != 18 or {r['id'] for r in rag} != set(rg):
                        raise ValueError('Actual RAG packet incomplete')
                    for r in rag:
                        if r['question'] != rg[r['id']]['messages'][:-1] or r['expected'] != rg[r['id']]['answer']:
                            raise ValueError('Actual RAG question changed')
                    item.update(epoch=e['epoch'], steps=e['steps'], validation_language_loss=e['validation_language_loss'],
                        original_eligible=e['eligible_for_review'], original_v53_groups=e['original_v53_legacy_checks']['groups'],
                        original_v54_groups=e['original_v54_legacy_checks']['groups'],
                        v54_with_v55_groups=e['v54_with_v55_checks']['groups'],
                        retention={k: e[k] for k in ['relative_retention','v54_retention','old_direction_retention']},
                        actual_rag_unknown_year_flags=[r['id'] for r in e['original_v53_rag_checks'] if r['unknown_birth_year_misuse']],
                        actual_rag_output_limits=sum(r.get('reached_token_limit', False) for r in rag))
                items.append(item)
            stage = json.loads(z.read(f'trials/{trial}/cycle_01/stage_status.json'))
            if stage['state'] != 'complete' or stage['steps'] != stage['planned_steps'] or stage['steps'] != 794:
                raise ValueError('Formal run incomplete')
            summary['trials'][trial] = {'stages': items, 'formal_steps': stage['steps'],
                'epochs': len(stage['epochs']), 'parent_weights_reused': stage['parent_weights_reused'],
                'peak_reserved_gib': stage['peak_reserved_gib'], 'runtime_seconds': stage['runtime_seconds'],
                'train_runtime_seconds': stage['training_metrics']['train_runtime']}
        teacher_flags = [{'id': r['id'], 'split': r['split'], 'case_ids': r['case_ids'],
                          'candidate_missing_conditions': bridge_boundary_candidates(r)}
                         for r in pack['records'] if r['skill'] == 'author_evidence']
        summary['teacher_boundary_screen'] = {
            'examined_author_evidence_cores': len(teacher_flags),
            'candidate_rows': sum(bool(r['candidate_missing_conditions']) for r in teacher_flags),
            'scope': 'finite literal patterns only; candidates require source review, not complete semantic approval',
        }
        duplicate_workflows = []
        for r in pack['records']:
            if r['skill'] != 'workflow':
                continue
            lines = [re.sub(r'^\d+、', '', s.strip()) for s in r['answer'].splitlines()]
            repeated = [s for s, n in Counter(lines).items() if n > 1 and len(s) > 35]
            if repeated:
                duplicate_workflows.append({'id':r['id'], 'split':r['split'], 'workflow_id':r['provenance']['workflow_id'],
                                            'repeated_paragraph_sha256':[d.digest(s) for s in repeated]})
        train_workflow_targets = {normalized_content(r['answer']) for r in pack['records'] if r['skill']=='workflow' and r['split']=='train'}
        selected_flows = [r for r in gold if r['skill']=='workflow']
        summary['workflow_teacher_screen'] = {
            'duplicate_paragraph_rows': len(duplicate_workflows),
            'duplicate_paragraph_groups': len({r['workflow_id'] for r in duplicate_workflows}),
            'split_counts': dict(Counter(r['split'] for r in duplicate_workflows)),
            'selected_development_workflows': len(selected_flows),
            'selected_workflow_target_content_also_in_training': sum(normalized_content(r['answer']) in train_workflow_targets for r in selected_flows),
            'scope': 'same-rule extraction retention, not unseen-rule or independent-board reasoning',
        }
        receipt = {'archive_sha256': d.digest(archive.read_bytes()), 'members': len(z.infolist()),
                   'uncompressed_bytes': sum(x.file_size for x in z.infolist()),
                   'audit_source_sha256': d.digest(Path(__file__).read_bytes()),
                   'read_without_extraction': True, 'uploaded_code_executed': False,
                   'original_run_and_curriculum_unchanged': True, 'not_rag_or_sft_input': True}
        d.write(output/'aggregate.json', summary)
        d.write(output/'upload_receipt.json', receipt)
        d.write(output/'teacher_boundary_candidates.json', teacher_flags)
        d.write(output/'workflow_duplicate_candidates.json', duplicate_workflows)
        if workbook:
            from openpyxl import Workbook
            wb = Workbook(write_only=True)
            ws = wb.create_sheet('全部原答与提示')
            ws.append(['原文件','题目ID','完整提问JSON','原回答','原目标','tokens','达到长度上限'])
            for name, r in allrows:
                ws.append([name, r['id'], d.dumps(r['question']), r['answer'], r['expected'], r['generated_tokens'], r['reached_token_limit']])
            ws = wb.create_sheet('新192逐轮分解')
            ws.append(['参数组','阶段','技能','领域','ID','完整提示JSON','目标','原答','原检查JSON','补充观察非语义批准'])
            for trial, tag, g, r, fact, overlay in details:
                ws.append([trial, tag, g['skill'], g['domain'], g['id'], d.dumps(r['question']), g['answer'], r['answer'], d.dumps(fact), d.dumps(overlay)])
            ws = wb.create_sheet('作者证据边界')
            ws.append(['ID','划分','案例','需复查条件','实际输入','目标','候选不等于全错'])
            flags = {r['id']: r for r in teacher_flags}
            for r in pack['records']:
                if r['id'] in flags:
                    ws.append([r['id'], r['split'], d.dumps(r['case_ids']), d.dumps(flags[r['id']]['candidate_missing_conditions']), r['messages'][-2]['content'], r['answer'], True])
            manual = output/'semantic_review.json'
            if manual.exists():
                ws = wb.create_sheet('语义审阅与边界说明')
                notes = d.read(manual)
                ws.append(['类别','完整审阅记录JSON','人工读答不等于专家批准'])
                for kind, values in notes.items():
                    for value in values:
                        ws.append([kind, d.dumps(value), True])
            wb.save(output/'Wengua_V55_逐轮逐题审核.xlsx')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--workbook', action='store_true')
    p.add_argument('--restore', action='store_true')
    p.add_argument('--key-file', type=Path)
    a = p.parse_args()
    if a.restore:
        from wengua_corpus.__main__ import restore
        value = a.key_file.read_text(encoding='utf-8-sig').strip() if a.key_file else input('输入此前教材64位口令（可见）：').strip()
        if not re.fullmatch('[0-9a-fA-F]{64}', value):
            raise ValueError('Expected the existing 64-character delivery key')
        entry = d.read(runtime.ROOT/'docs/model_reviews/20261107_v55/private_review_catalog.json')
        print(restore(entry, bytes.fromhex(value)))
        return
    if a.archive is None or a.output is None:
        p.error('--archive and --output required for read-only audit')
    result = audit(a.archive, a.output, workbook=a.workbook)
    print(d.dumps({k: result[k] for k in ['run_state','generation_files','generation_rows','code_release_prompt_hashes_match','teacher_boundary_screen']}))


if __name__ == '__main__':
    main()
