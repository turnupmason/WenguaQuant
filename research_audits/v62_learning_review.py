"""Read-only audit of the actual 108-answer V62 learning probe, never a teacher."""
from pathlib import Path
import argparse, gzip, json, zipfile
from wengua_course import data as d
from wengua_course_v62 import runtime
from research_audits import v62_probe as probe, v62_probe_data
from research_audits.v55_review import inspect_archive, read_rows

UPLOAD_SHA = '9685b496af836a4e14f47f8827b4c6c9bb9b6d612cb062684c5e47a6ffda13d1'


def audit(archive, output, notes=None):
    archive = Path(archive)
    if d.digest(archive.read_bytes()) != UPLOAD_SHA:
        raise ValueError('Exact uploaded learning-probe archive required')
    runtime.configure()
    expected = v62_probe_data.build(runtime.verify_data())
    remarks = d.read(notes) if notes else {}
    details, workers, activation, lengths = [], [], {}, {}
    with zipfile.ZipFile(archive) as z:
        inspect_archive(z)
        if z.testzip() is not None:
            raise ValueError('Archive CRC mismatch')
        names = z.namelist()
        contract = json.loads(z.read('run_contract.json'))
        state = json.loads(z.read('run_status.json'))
        pack = json.loads(z.read('prompts.json'))
        if pack != expected or d.digest(z.read('prompts.json')) != contract['prompts_sha256']:
            raise ValueError('Frozen source prompts differ')
        if contract['code_sha256'] != probe.hashes() or contract['software_fixture']:
            raise ValueError('Frozen actual probe code differs')
        if not contract['inference_only'] or contract['weight_updates'] != 0 or contract['config']['eval_max_new_tokens'] != 512:
            raise ValueError('Probe must be inference-only with the declared budget')
        if state['state'] != 'complete_review_required' or state['generated_answers'] != 108:
            raise ValueError('Actual probe incomplete')
        for label in probe.LABELS:
            source = contract['sources'][label]
            if source['adapter_sha256'] != probe.EXPECTED_ADAPTER_SHA.get(label):
                raise ValueError('Adapter source differs from the actual V62 run')
            targets = pack['records'] + (pack['official_rag_rows'] if label == 'official_base' else [])
            rows = read_rows(z, 'answers/' + label + '.jsonl')
            if not probe.verify_cached(rows, targets, label):
                raise ValueError('Missing answers')
            proof = json.loads(z.read('answers/' + label + '.provenance.json'))
            if proof != probe.result_contract(contract['config'], source, targets):
                raise ValueError('Prompt/budget/source provenance differs')
            attention = json.loads(z.read('answers/' + label + '.attention.json'))
            if not attention['cuda_calls'] or attention['software_fixture'] or attention['math_fallback_allowed']:
                raise ValueError('Actual fused CUDA path missing')
            lengths[label] = {'answers': len(rows), 'token_limit_hits': sum(r['reached_token_limit'] for r in rows)}
            if label != 'official_base':
                a = json.loads(z.read('answers/' + label + '.activation.json'))
                if (a['adapter_sha256'] != source['adapter_sha256'] or a['software_fixture'] or
                        not a['parameters_frozen'] or a['weight_updates'] or a['lora_B_matrices'] != 144 or
                        a['nonzero_lora_B_matrices'] != 144 or a['max_last_position_logit_delta'] <= 0):
                    raise ValueError('Numerical adapter activation failed')
                activation[label] = a
            by = {r['id']: r for r in targets}
            for i, row in enumerate(rows):
                note = remarks.get(label, {}).get(row['id'])
                if note and note['answer_sha256'] != d.digest(row['answer']):
                    raise ValueError('Reading note bound to another answer')
                details.append({'tag': label, 'index': i, 'row': row,
                                'gold': by[row['id']] | {'skill': by[row['id']].get('task', 'rag')},
                                'frozen_check': None, 'annotation': note})
        for name in sorted(n for n in names if n.endswith('.result.json')):
            raw = z.read(name.replace('.result.json', '.json'))
            request, reply = json.loads(raw), json.loads(z.read(name))
            probe.verify_reply(reply, request, raw)
            if request['source'] != contract['sources'][request['label']] or request['config'] != contract['config']:
                raise ValueError('Worker request differs from run contract')
            workers.append({'label': request['label'], 'completed': reply['result']['completed'],
                            'request_sha256': d.digest(raw), 'weight_updates': 0})
    if len(names) != 24 or len(workers) != 3 or len(details) != 108:
        raise ValueError('Unexpected archive/worker/answer count')
    summary = {'schema': 'wengua_actual_v62_learning_probe_review_v1', 'upload_sha256': UPLOAD_SHA,
               'archive_members': len(names), 'frozen_runtime_files': len(contract['code_sha256']),
               'actual_answers': len(details), 'weight_updates': 0, 'workers': workers,
               'activation': activation, 'answer_lengths': lengths,
               'notes_bound_to_answers': sum(bool(x['annotation']) for x in details),
               'official_same_prompt_rag_baseline_answers': 18,
               'original_training_questions_per_model': 6, 'unseen_case_benchmark': False,
               'unuploaded_weights_independently_rehashed': False, 'professional_promotion': False,
               'model_answers_allowed_as_teachers': False}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    d.write(output / 'summary.json', summary)
    with gzip.open(output / 'answer_observations.jsonl.gz', 'wt', encoding='utf8') as f:
        for x in details:
            f.write(d.dumps(x) + '\n')
    return summary, details


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path, nargs='?')
    p.add_argument('--output', type=Path, default=runtime.ROOT / 'local/model_reviews/v62_learning')
    p.add_argument('--notes', type=Path); p.add_argument('--workbook', action='store_true')
    p.add_argument('--restore', action='store_true'); p.add_argument('--key-file', type=Path)
    a = p.parse_args(argv)
    if a.restore:
        import re
        from wengua_corpus.__main__ import restore
        text = a.key_file.read_text(encoding='utf-8-sig').strip() if a.key_file else input('粘贴此前教材的64位口令（输入可见）：').strip()
        if not re.fullmatch('[0-9a-fA-F]{64}', text):
            raise ValueError('Expected the existing textbook delivery key')
        entry = d.read(runtime.ROOT / 'docs/model_reviews/20261110_v62_learning/private_review_catalog.json')
        out = restore(entry, bytes.fromhex(text))
        print(out / 'review/Wengua_V62_学习与输入逐答审阅.xlsx')
        return 0
    if a.archive is None: p.error('archive or --restore required')
    summary, details = audit(a.archive, a.output, a.notes)
    if a.workbook:
        from research_audits.v61_review import workbook
        workbook(a.output / 'Wengua_V62_学习与输入逐答审阅.xlsx', details)
    print(d.dumps({'answers': summary['actual_answers'], 'workers': len(summary['workers']),
                   'notes': summary['notes_bound_to_answers'], 'output': str(a.output)}))


if __name__ == '__main__':
    main()
