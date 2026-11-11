"""Audit the actual V63 run; all model responses remain evaluation-only."""
from pathlib import Path
import argparse, gzip, json, zipfile
from collections import Counter
from wengua_course import data as d
from wengua_course_v63 import runtime, dataset
from research_audits.v55_review import inspect_archive, read_rows
from research_audits.v60_review import verify_reply, verify_generation
from research_audits.v62_review import same_prompt
from research_audits.v59_review import memory_summary

UPLOAD_SHA = 'ae32ff3d8c17954d23ad01bdcc8370210a9fd5a4e9705519aed2ce450dd71a7c'
TRIALS = {'lr_5em05': (5e-5, 8, [1, 2, 4, 8]), 'lr_1em04': (1e-4, 7, [1, 2, 4, 7])}


def audit(archive, output, notes=None):
    archive = Path(archive)
    if d.digest(archive.read_bytes()) != UPLOAD_SHA:
        raise ValueError('Exact uploaded V63 archive required')
    runtime.configure(); pack = runtime.verify_data()
    targets = {'recall': dataset.learning_rows(pack), 'seen': pack['known_regression_rows'], 'rag': pack['actual_rag_rows']}
    remarks = d.read(notes) if notes else {}; details, workers, stages, baseline, exact = [], [], {}, {}, {}
    with zipfile.ZipFile(archive) as z:
        inspect_archive(z)
        if z.testzip() is not None: raise ValueError('Archive CRC differs')
        names = z.namelist(); contract = json.loads(z.read('sweep_contract.json')); state = json.loads(z.read('run_status.json'))
        if (contract['code_sha256'] != runtime.hashes() or contract['dataset_release_sha256'] != d.digest(runtime.RELEASE.read_bytes())
                or contract['rates'] != [5e-5, 1e-4] or contract['epochs'] != 8 or contract['old_book_weights_loaded']
                or contract['original_test_used'] or state['state'] != 'complete_review_required' or state['production_enabled']):
            raise ValueError('Frozen V63 runtime/course/run differs')
        parent_raw=z.read('initialization/parent_provenance.json');parent=json.loads(parent_raw)
        if parent['schema']!=runtime.SCHEMA or not parent['parent_config']['initialization_only']:
            raise ValueError('Original official initializer provenance missing')
        configs = {}
        for trial, (rate, epochs, landmarks) in TRIALS.items():
            raw = z.read(f'trials/{trial}/job_contract.json'); job = json.loads(raw); c = job['config']; configs[trial] = c
            sc = json.loads(z.read(f'trials/{trial}/cycle_01/stage_contract.json'))
            stage = json.loads(z.read(f'trials/{trial}/cycle_01/stage_status.json')); stages[trial] = stage
            if (job['code_sha256'] != runtime.hashes() or job['dataset_release_sha256']!=contract['dataset_release_sha256']
                    or c['book_parent_provenance_sha256']!=d.digest(parent_raw)
                    or c != state['trials'][trial]['config'] or c['learning_rate'] != rate
                    or sc['job_contract_sha256'] != d.digest(raw) or sc['config'] != c or sc['planned_updates'] != 424
                    or not sc['fresh_official_base'] or sc['parent_weights_reused'] or stage != state['trials'][trial]['fit']
                    or stage['state'] != 'complete' or stage['steps'] != epochs * 53 or stage['resume_checkpoint'] is not None
                    or stage['trainable_parameters'] != 43646976):
                raise ValueError('Actual seven-projection trial differs')
            if [(e['cycle'], e['steps'], e['epoch']) for e in stage['epochs']] != [(i, i*53, float(i)) for i in range(1, epochs+1)]:
                raise ValueError('Epoch boundaries differ')
            nominated = min(stage['epochs'], key=lambda e: (e['validation_loss'], e['steps']))
            if stage['adapter'] != nominated['adapter'] or stage['selected_epoch'] != nominated['cycle']:
                raise ValueError('Saved language-loss nominee differs')
            from src.wengua.qwen_book_continuous_v1 import update_history
            h, early = [], False
            for e in stage['epochs']: h, early = update_history(h, e, c)
            if early != stage['early_stopped']: raise ValueError('Patience replay differs')
            for suite in ['recall', 'seen']:
                for epoch in [0] + landmarks:
                    filename = ('learning_parent' if suite == 'recall' else 'parent_seen_diagnostics') if not epoch else f'epoch_{epoch:02d}_' + ('training_recall' if suite == 'recall' else 'seen_diagnostics')
                    rows = read_rows(z, f'trials/{trial}/generations/{filename}.jsonl')
                    if not epoch: baseline[(trial, suite)] = rows
                    add_details(details, rows, targets[suite], trial + ':' + filename, remarks)
                    if suite == 'recall': exact[trial + ':' + filename] = sum(r['answer'] == r['expected'] for r in rows)
            path = f'trials/{trial}/generations/language_nominee_actual_rag.jsonl'
            rows = read_rows(z, path); proof = json.loads(z.read(path.replace('.jsonl', '.provenance.json')))
            verify_generation(rows, targets['rag'], proof, adapter_sha=proof['adapter_sha256'], token_budget=512, stage='rag')
            add_details(details, rows, targets['rag'], trial + ':rag', remarks)
        if baseline[('lr_5em05', 'recall')] != baseline[('lr_1em04', 'recall')] or baseline[('lr_5em05', 'seen')] != baseline[('lr_1em04', 'seen')]:
            raise ValueError('Same official zero-update answers differ')
        official_path='trials/lr_5em05/generations/official_actual_rag.jsonl'
        official_rows=read_rows(z,official_path);official_proof=json.loads(z.read(official_path.replace('.jsonl','.provenance.json')))
        verify_generation(official_rows,targets['rag'],official_proof,adapter_sha=None,token_budget=512,stage='rag')
        add_details(details,official_rows, targets['rag'], 'official:rag', remarks)
        for name in sorted(n for n in names if n.endswith('.result.json')):
            raw = z.read(name.replace('.result.json', '.json')); req = json.loads(raw); reply = json.loads(z.read(name)); verify_reply(reply, req, raw)
            trial = name.split('/')[1]
            if trial in configs and req['config'] != configs[trial]: raise ValueError('Worker config differs')
            item = {'trial': trial, 'operation': req['operation'], 'probe': req['probe'], 'seconds': reply['elapsed_seconds'], 'steps': reply['result'].get('steps'), 'request_sha256': d.digest(raw)}
            for suffix, key in [('.memory.json', 'memory'), ('.training_start.json', 'training_start'), ('.epochs.json', 'epoch_optimizer')]:
                n = name.replace('.result.json', suffix)
                if n in names:
                    value = json.loads(z.read(n)); item[key] = memory_summary(value) if key == 'memory' else value
            if req['operation'] == 'fit':
                if item['training_start']['step'] or item['training_start']['optimizer_steps']: raise ValueError('New trial did not start at zero')
                if not req['probe']:
                    history = item['epoch_optimizer']
                    if [e['step'] for e in history] != list(range(0, stages[trial]['steps'], 53)) or len({e['optimizer_id'] for e in history}) != 1:
                        raise ValueError('Optimizer reset across epochs')
            workers.append(item)
        for name in (n for n in names if n.endswith('.attention_path.json')):
            a = json.loads(z.read(name))
            if a['software_fixture'] or not a['cuda_calls'] or a['math_fallback_allowed']: raise ValueError('Actual fused CUDA path missing')
        parity = json.loads(z.read('trials/official_step_zero/short_forward_parity.json'))
        if not parity['actual_cuda'] or len(parity['comparisons']) != 6 or max(r['relative_l2_error'] for r in parity['comparisons']) > .03:
            raise ValueError('Actual preflight parity differs')
        zero = json.loads(z.read('trials/official_step_zero/zero_update_proof.json'))
        if not zero['all_lora_B_zero'] or zero['weight_updates'] or zero['files']!=parent['files']: raise ValueError('Fresh zero update proof missing')
        smoke = [(n, r) for n in names if n.endswith('/smoke_generation.jsonl') for r in read_rows(z, n)]
    if len(details) != 414 or len(workers) != 8 or len(smoke) != 2: raise ValueError('Actual answer/worker count differs')
    summary = {'schema': 'wengua_actual_v63_review_v1', 'upload_sha256': UPLOAD_SHA, 'archive_members': len(names),
               'frozen_runtime_files': len(contract['code_sha256']), 'formal_answers': 414, 'smoke_answers': 2,
               'distinct_answers_excluding_identical_baseline_copy': 378, 'formal_updates': 795, 'pressure_updates': 8,
               'epochs': stages, 'workers': workers, 'short_actual_cuda_parity': parity,
               'training_recall_literal_exact_counts_out_of_12': exact,
               'token_limit_hits': sum(x['row']['reached_token_limit'] for x in details),
               'notes_bound_to_answers': sum(bool(x['annotation']) for x in details),
               'per_step_train_loss_log_uploaded': False, 'later_checkpoint_rag_measured': False,
               'unuploaded_weight_tensors_independently_rehashed': False, 'original_test_used': False,
               'professional_promotion': False, 'responses_allowed_as_sft_or_rag_teachers': False}
    output = Path(output); output.mkdir(parents=True, exist_ok=True); d.write(output / 'summary.json', summary)
    with gzip.open(output / 'answer_observations.jsonl.gz', 'wt', encoding='utf8') as f:
        for row in details: f.write(d.dumps(row) + '\n')
    return summary, details


def add_details(details, rows, gold, tag, remarks):
    if len(rows) != len(gold): raise ValueError('Missing or extra responses')
    by = {r['id']: r for r in gold}
    if set(by) != {r['id'] for r in rows}: raise ValueError('Response IDs differ')
    for i, row in enumerate(rows):
        target = by[row['id']]; same_prompt(row, target); note = remarks.get(tag, {}).get(row['id'])
        if note and note['answer_sha256'] != d.digest(row['answer']): raise ValueError('Note bound to another answer')
        details.append({'tag': tag, 'index': i, 'row': row, 'gold': target | {'skill': target.get('skill', 'rag')}, 'frozen_check': None, 'annotation': note})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('archive', type=Path,nargs='?')
    p.add_argument('--output', type=Path,default=runtime.ROOT/'local/model_reviews/v63'); p.add_argument('--notes', type=Path); p.add_argument('--workbook', action='store_true')
    p.add_argument('--restore',action='store_true');p.add_argument('--key-file',type=Path)
    a = p.parse_args(argv)
    if a.restore:
        import re
        from wengua_corpus.__main__ import restore
        text=a.key_file.read_text(encoding='utf-8-sig').strip() if a.key_file else input('粘贴此前教材的64位口令（输入可见）：').strip()
        if not re.fullmatch('[0-9a-fA-F]{64}',text):raise ValueError('Expected existing textbook delivery key')
        entry=d.read(runtime.ROOT/'docs/model_reviews/20261110_v63/private_review_catalog.json')
        out=restore(entry,bytes.fromhex(text));print(out/'review/Wengua_V63_逐轮逐答审阅.xlsx');return 0
    if a.archive is None:p.error('archive or --restore required')
    summary, details = audit(a.archive, a.output, a.notes)
    if a.workbook:
        from research_audits.v61_review import workbook
        workbook(a.output / 'Wengua_V63_逐轮逐答审阅.xlsx', details)
    print(d.dumps({'formal_answers': len(details), 'output': str(a.output)}))


if __name__ == '__main__': main()
