"""Read-only review of the actual nine-model V63 checkpoint comparison."""
from pathlib import Path
from collections import Counter
import argparse, gzip, json, zipfile
from wengua_course import data as d
from wengua_course_v63 import runtime
from . import v63_checkpoint_probe as probe
from .v55_review import inspect_archive, read_rows

UPLOAD_SHA = '970a63c80fbec89a01e3576aad7f6cb20092514c880a8f1d0cd7109f4a192af5'
PARENT_LABEL = 'high_e07'


def source_inventory(pack):
    """Count actual source lessons; no assertion of professional completeness."""
    rows = [r for r in pack['records'] if r['split'] == 'train']
    switches = [r for r in rows if r['skill'] == 'identity_switch']
    same_self = [r for r in switches if r['messages'][1]['content'] == '按本人取谁？']
    return {
        'training_cores': len(rows), 'distinct_teacher_texts': len({r['answer'] for r in rows}),
        'skills': dict(Counter(r['skill'] for r in rows)),
        'case_linked_cores': sum(bool(r.get('case_id') or r.get('case_ids')) for r in rows),
        'distinct_linked_case_ids': sorted({c for r in rows for c in r.get('case_ids', [])}),
        'same_scope_tail_occurrences': sum('这里只判断门与宫的关系' in r['answer'] for r in rows),
        'identity_switch_cores': len(switches), 'same_self_to_self_cores': len(same_self),
        'same_self_to_self_is_not_a_role_change': True,
        'all_original_sources_and_case_splits_preserved': True,
        'not_a_full_two_book_case_curriculum': True,
    }


def check_activation(value, source):
    if (value['software_fixture'] or not value['parameters_frozen'] or value['weight_updates'] != 0
            or value['adapter_sha256'] != source['adapter_sha256'] or value['lora_B_matrices'] != 252
            or value['nonzero_lora_B_matrices'] != 252 or value['max_last_position_logit_delta'] <= 0):
        raise ValueError('Actual loaded adapter has not passed the numerical activation check')


def check_sources(contract):
    if set(contract['sources']) != set(probe.LABELS): raise ValueError('Nine source models required')
    if contract['sources']['official_base']['adapter'] is not None or contract['sources']['official_base']['adapter_sha256'] is not None:
        raise ValueError('Official baseline has an adapter')
    for label, (trial, epoch) in probe.SELECTION.items():
        source = contract['sources'][label]; proof = source['source_proof']
        expected_end = '/trials/' + trial + '/cycle_01/adapters/step-' + str(53 * epoch)
        if not source['adapter'].replace('\\', '/').endswith(expected_end): raise ValueError('Different source epoch')
        if (proof['schema'] != runtime.SCHEMA or proof['files']['adapter_model.safetensors'] != source['adapter_sha256']
                or set(proof['parent_config']['target_modules']) != set(runtime.data.MODULES)
                or source['professional_approval']):
            raise ValueError('Source seven-projection provenance differs')
    expected = d.read(runtime.ROOT/'data/reviews/v63_checkpoint_sources.json')
    for label, info in expected['reuse'].items():
        if contract['sources'][label]['reuse_sha256'] != info['file_sha256']: raise ValueError('Original reuse SHA differs')
    if contract['stack'] != expected['original_stack']: raise ValueError('Actual package stack differs')


def bound_note(remarks, label, row):
    if not remarks: return None
    note = remarks.get(label, {}).get(row['id'])
    if not note or note.get('answer_sha256') != d.digest(row['answer']):
        raise ValueError('Annotation missing or belongs to a different answer')
    return note


def audit(archive, output, notes=None, original_v63=None):
    archive = Path(archive)
    if d.digest(archive.read_bytes()) != UPLOAD_SHA: raise ValueError('Exact uploaded V63 checkpoint archive required')
    runtime.configure(); pack = runtime.verify_data(); expected = probe.build_prompts(pack)
    remarks = d.read(notes) if notes else {}; details, workers, activation, lengths = [], [], {}, {}
    reuse_rows = {}; rows_by_model = {}
    with zipfile.ZipFile(archive) as z:
        inspect_archive(z)
        if z.testzip() is not None: raise ValueError('Archive CRC mismatch')
        names = z.namelist(); contract = json.loads(z.read('run_contract.json')); state = json.loads(z.read('run_status.json'))
        prompts = json.loads(z.read('prompts.json')); c = contract['config']
        if (contract['schema'] != probe.SCHEMA or contract['code_sha256'] != probe.hashes()
                or contract['software_fixture'] or not contract['inference_only'] or contract['weight_updates'] != 0
                or contract['model_quality_verified'] or prompts != expected
                or d.digest(z.read('prompts.json')) != contract['prompts_sha256']
                or c != d.read(runtime.CONFIG) | {'eval_max_new_tokens': 512}):
            raise ValueError('Frozen source, prompt or inference contract differs')
        if (state['state'] != 'complete_review_required' or state['stored_answers'] != 342
                or state['additional_answers'] != 288 or state['reused_original_answers'] != 54
                or state['weight_updates'] != 0 or state['professional_promotion']):
            raise ValueError('Actual nine-model comparison incomplete')
        check_sources(contract)
        byid = {r['id']: r for r in prompts['records']}
        for label in probe.LABELS:
            source = contract['sources'][label]; path = 'answers/' + label + '.jsonl'; rows = read_rows(z, path)
            if not probe.verify_cached(rows, prompts['records']) or len(rows) != 38: raise ValueError('Missing original prompts')
            proof = json.loads(z.read(path.replace('.jsonl', '.provenance.json')))
            if proof != probe.result_contract(c, source, prompts['records']): raise ValueError('Inference source/budget differs')
            attention = json.loads(z.read(path.replace('.jsonl', '.attention.json')))
            if (attention['software_fixture'] or not attention['cuda_calls'] or attention['math_fallback_allowed']
                    or attention['selected_backend'] != 'EFFICIENT_ATTENTION'): raise ValueError('Actual CUDA route missing')
            if label != 'official_base':
                a = json.loads(z.read(path.replace('.jsonl', '.activation.json'))); check_activation(a, source); activation[label] = a
            if source.get('reuse_file'):
                receipt = json.loads(z.read(path.replace('.jsonl', '.original_rag.json')))
                saved = rows[:18]
                if (receipt['source_sha256'] != source['reuse_sha256'] or receipt['source_answers_rewritten']
                        or receipt['weight_updates'] or receipt['answer_ids'] != [r['id'] for r in saved]
                        or receipt['adapter_sha256'] != source['adapter_sha256']): raise ValueError('Reuse receipt differs')
                reuse_rows[label] = saved
            for row in rows[18:] if source.get('reuse_file') else rows:
                if row['variant'] != label: raise ValueError('New answer belongs to another model label')
            rows_by_model[label] = rows
            lengths[label] = {'stored_answers': len(rows), 'token_limit_ids': [r['id'] for r in rows if r['reached_token_limit']],
                              'generated_tokens': sum(r['generated_tokens'] for r in rows),
                              'mean_answer_characters': sum(len(r['answer']) for r in rows)/len(rows)}
            for index, row in enumerate(rows):
                note = bound_note(remarks, label, row)
                if row['empty'] or not row['answer']: raise ValueError('Empty actual answer')
                details.append({'tag': label, 'index': index, 'row': row, 'gold': byid[row['id']] | {'skill': byid[row['id']].get('skill', 'rag')},
                                'frozen_check': None, 'annotation': note})
        for name in sorted(n for n in names if n.endswith('.result.json')):
            raw = z.read(name.replace('.result.json', '.json')); request = json.loads(raw); reply = json.loads(z.read(name))
            probe.verify_reply(reply, request, raw); label = request['label']
            if request['source'] != contract['sources'][label] or request['config'] != c or request['software_fixture']:
                raise ValueError('Actual worker source/config differs')
            if reply['result'] != state['models'][label] or reply['result']['completed'] != 38: raise ValueError('Worker receipt differs from result')
            workers.append({'label': label, 'pid': reply['pid'], 'request_sha256': d.digest(raw), 'weight_updates': 0})
    reused_verified = False
    if original_v63:
        from .v63_review import UPLOAD_SHA as ORIGINAL_SHA
        if d.digest(Path(original_v63).read_bytes()) != ORIGINAL_SHA: raise ValueError('Exact earlier V63 upload required')
        with zipfile.ZipFile(original_v63) as original:
            for label, (trial, epoch) in probe.SELECTION.items():
                prior = json.loads(original.read('trials/'+trial+'/cycle_01/stage_status.json'))
                source = contract['sources'][label]
                event = next(e for e in prior['epochs'] if e['cycle'] == epoch)
                job = json.loads(original.read('trials/'+trial+'/job_contract.json'))
                if source['adapter'] != event['adapter'] or source['source_proof']['parent_config'] != job['config']:
                    raise ValueError('Source stage/config differs from original formal training')
            for label, info in d.read(runtime.ROOT/'data/reviews/v63_checkpoint_sources.json')['reuse'].items():
                raw = original.read(info['file'])
                if d.digest(raw) != info['file_sha256'] or list(map(json.loads, raw.splitlines())) != reuse_rows[label]:
                    raise ValueError('Original 54 stored answers have changed')
        reused_verified = True
    if len(names) != 69 or len(workers) != 9 or len({w['label'] for w in workers}) != 9 or len(details) != 342:
        raise ValueError('Actual archive/worker/answer count differs')
    source = contract['sources'][PARENT_LABEL]
    summary = {'schema': 'wengua_actual_v63_checkpoint_review_v1', 'upload_sha256': UPLOAD_SHA,
               'archive_members': len(names), 'frozen_runtime_files': len(contract['code_sha256']),
               'stored_answers': 342, 'reused_answers': 54, 'new_answers': 288, 'weight_updates': 0,
               'reused_answers_match_earlier_upload': reused_verified, 'workers': workers, 'activation': activation,
               'answer_lengths': lengths, 'source_curriculum': source_inventory(pack),
               'notes_bound_to_actual_answers': sum(bool(x['annotation']) for x in details),
               'research_continuation_candidate': {'label': PARENT_LABEL, 'relative_path': 'local/course_v63/run1/trials/lr_1em04/cycle_01/adapters/step-371',
                                                   'adapter_sha256': source['adapter_sha256'], 'production_approved': False,
                                                   'not_uniformly_best_on_known_tasks': True},
               'all_questions_are_known_development_not_blind': True, 'professional_promotion': False,
               'unuploaded_weight_tensors_independently_rehashed': False, 'model_answers_allowed_as_teachers': False}
    output = Path(output); output.mkdir(parents=True, exist_ok=True); d.write(output/'summary.json', summary)
    with gzip.open(output/'answer_observations.jsonl.gz', 'wt', encoding='utf8') as f:
        for row in details: f.write(d.dumps(row)+'\n')
    return summary, details


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('archive', type=Path, nargs='?')
    p.add_argument('--original-v63', type=Path); p.add_argument('--notes', type=Path); p.add_argument('--workbook', action='store_true')
    p.add_argument('--output', type=Path, default=runtime.ROOT/'local/model_reviews/v63_checkpoints')
    p.add_argument('--restore', action='store_true'); p.add_argument('--key-file', type=Path); a = p.parse_args(argv)
    if a.restore:
        import re
        from wengua_corpus.__main__ import restore
        value = a.key_file.read_text(encoding='utf-8-sig').strip() if a.key_file else input('粘贴此前教材的64位口令（输入可见）：').strip()
        if not re.fullmatch('[0-9a-fA-F]{64}', value): raise ValueError('Expected existing textbook delivery key')
        entry = d.read(runtime.ROOT/'docs/model_reviews/20261111_v63_checkpoints/private_review_catalog.json')
        out = restore(entry, bytes.fromhex(value)); print(out/'review/Wengua_V63_九模型逐答审阅.xlsx'); return 0
    if a.archive is None: p.error('archive or --restore required')
    summary, details = audit(a.archive, a.output, a.notes, a.original_v63)
    if a.workbook:
        from .v61_review import workbook
        workbook(a.output/'Wengua_V63_九模型逐答审阅.xlsx', details)
    print(d.dumps({'stored_answers': len(details), 'notes': summary['notes_bound_to_actual_answers'], 'output': str(a.output)})); return 0


if __name__ == '__main__': raise SystemExit(main())
