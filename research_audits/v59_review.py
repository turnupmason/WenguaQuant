"""Read-only replay of the V59 upload, recovery receipts and bounded answer review."""
from collections import Counter
from pathlib import Path
import argparse
import gzip
import json
import re
import zipfile
from wengua_course import data as d
from wengua_course_v59 import runtime, checks, runner
from research_audits.v55_review import inspect_archive, read_rows
from research_audits.v58_review import observation, validate_note

UPLOAD_SHA = '5e097ef89e920b7746305718145041b80496b3d9eb1d5226fc3f89ed3bc06ffd'


def prompt_sha(messages):
    ordered = [{'role': m['role'], 'content': m['content']} for m in messages]
    return d.digest(json.dumps(ordered, ensure_ascii=False, separators=(',', ':')))


def verify_worker(reply, request, raw_request):
    if (reply['request_sha256'] != d.digest(raw_request) or reply['nonce'] != request['nonce']
            or type(reply.get('pid')) is not int or reply['pid'] <= 0 or request['software_fixture']):
        raise ValueError('Worker/request provenance differs')
    if request.get('recovery'):
        from wengua_recovery.v59_watchdog import canonical_sha, recipe
        if (reply.get('recovery_sha256') != canonical_sha(request['recovery'])
                or request['recovery']['code_sha256'] != recipe()['code_sha256']
                or request['recovery']['policy'] != recipe()['policy']):
            raise ValueError('Operational recovery receipt differs')
    if not reply['ok']:
        if reply['error']['type'] != 'KeyboardInterrupt' or reply['error']['cuda_oom']:
            raise ValueError('Unexpected actual failed worker')
    elif reply['result']['state'] != 'complete':
        raise ValueError('Incomplete successful worker')


def memory_summary(samples):
    rows = [r for r in samples if r['phase'] == 'train']
    if not rows:
        return {'train_samples': 0}
    return {'train_samples': len(rows), 'zero_free_samples': sum(r['free_gib'] == 0 for r in rows),
            'minimum_free_gib': min(r['free_gib'] for r in rows),
            'max_sampled_reserved_gib': max(r['reserved_gib'] for r in rows),
            'max_reported_peak_allocated_gib': max(r.get('peak_allocated_gib', 0) for r in rows),
            'max_allocator_ooms': max(r.get('allocator_ooms', 0) for r in rows),
            'first_step': min(r['step'] for r in rows), 'last_sampled_step': max(r['step'] for r in rows)}


def audit(archive, output, *, notes=None, workbook=False):
    archive, output = Path(archive), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if d.digest(archive.read_bytes()) != UPLOAD_SHA:
        raise ValueError('Exact actual V59 upload required')
    runtime.configure()
    pack = runtime.verify_data()
    by = {r['id']: r for r in pack['records']}
    micro = runtime.choose_development(pack)
    final = [by[k] for k in pack['fixed_final_ids']]
    rag = pack['v59_actual_rag_rows']
    annotations = d.read(notes) if notes else {}
    details, workers, allrows, scores = [], [], [], {}
    with zipfile.ZipFile(archive) as z:
        inspect_archive(z)
        if z.testzip() is not None:
            raise ValueError('Archive CRC differs')
        state = json.loads(z.read('run_status.json'))
        contract = json.loads(z.read('sweep_contract.json'))
        if (contract['code_sha256'] != runtime.hashes()
                or contract['dataset_release_sha256'] != d.digest(runtime.RELEASE.read_text())
                or state['state'] != 'needs_course_review' or set(state['trials']) != {'mixed_pilot'}
                or contract['original_test_used']):
            raise ValueError('Frozen execution/course/phase differs')
        trial = state['trials']['mixed_pilot']
        job_raw = z.read('trials/mixed_pilot/job_contract.json')
        job_contract = json.loads(job_raw)
        stage_contract = json.loads(z.read('trials/mixed_pilot/cycle_01/stage_contract.json'))
        if (job_contract['code_sha256'] != runtime.hashes()
                or stage_contract['job_contract_sha256'] != d.digest(job_raw)
                or stage_contract['config'] != job_contract['config']
                or job_contract['config']['learning_rate'] != 3e-6):
            raise ValueError('Actual job/stage configuration differs')
        for name in z.namelist():
            if not name.endswith('.result.json'):
                continue
            reply = json.loads(z.read(name))
            request_name = name.replace('.result.json', '.json')
            raw_request = z.read(request_name)
            request = json.loads(raw_request)
            verify_worker(reply, request, raw_request)
            if request['config'] != job_contract['config']:
                raise ValueError('Worker config differs from sealed job')
            entry = {'operation': request['operation'], 'probe': request['probe'],
                     'tag': (request.get('best') or {}).get('tag'), 'ok': reply['ok'],
                     'recovery': bool(request.get('recovery')), 'seconds': reply.get('elapsed_seconds'),
                     'steps': reply.get('result', {}).get('steps'), 'error': reply.get('error')}
            mem_name = name.replace('.result.json', '.memory.json')
            if mem_name in z.namelist():
                entry['memory'] = memory_summary(json.loads(z.read(mem_name)))
            releases_name = name.replace('.result.json', '.cache_releases.jsonl')
            if releases_name in z.namelist():
                releases = read_rows(z, releases_name)
                entry['cache_releases'] = len(releases)
                entry['after_release_reserved_range_gib'] = [min(r['after_reserved_gib'] for r in releases), max(r['after_reserved_gib'] for r in releases)]
            start_name = name.replace('.result.json', '.training_start.json')
            if start_name in z.namelist():
                started = json.loads(z.read(start_name))
                entry['training_start_step'] = started['step']
                entry['optimizer_state_entries'] = len(started['optimizer_steps'])
                entry['optimizer_step_values'] = sorted(set(started['optimizer_steps']))
            workers.append(entry)
        prefix = 'trials/mixed_pilot/generations/'
        sets = [('parent_micro', 'validation_parent', micro), ('parent_rag', 'parent_actual_rag', rag)]
        for i in (1, 2):
            sets += [(f'epoch_{i:02d}_micro', f'epoch_{i:02d}_validation', micro),
                     (f'epoch_{i:02d}_rag', f'epoch_{i:02d}_actual_rag', rag)]
        sets.append(('final', 'final_review', final))
        for tag, name, gold in sets:
            rows = read_rows(z, prefix+name+'.jsonl')
            score = checks.score(rows, gold)
            scores[tag] = score
            if not score['complete']:
                raise ValueError('Incomplete answers')
            targets = {r['id']: r for r in gold}
            for row in rows:
                g = targets[row['id']]
                if row['question'] != g['messages'][:-1] or row['expected'] != g['answer'] or row['prompt_sha256'] != prompt_sha(row['question']):
                    raise ValueError('Question/target/prompt SHA differs')
                note = annotations.get(tag, {}).get(row['id'])
                if note:
                    validate_note(note, row)
                details.append({'tag': tag, 'row': row, 'gold': g,
                                'frozen_check': checks.fact_check(row, g),
                                'observation': observation(row, g), 'read_answer_note': note})
        if scores['parent_micro'] != trial['parent_micro'] or scores['parent_rag'] != trial['parent_rag']:
            raise ValueError('Parent score differs')
        for i, epoch in enumerate(trial['epochs'], 1):
            cur, rg = scores[f'epoch_{i:02d}_micro'], scores[f'epoch_{i:02d}_rag']
            gate = runner.learning_gate(cur, scores['parent_micro'], rg, scores['parent_rag'])
            if cur != epoch['micro_checks'] or rg != epoch['actual_rag_checks'] or gate != epoch['training_retention_gate'] or gate['passed']:
                raise ValueError('Frozen score/stopping gate differs')
        if json.loads(z.read('pilot_full_review.json'))['checks'] != scores['final']:
            raise ValueError('Final score differs')
        finalrows = read_rows(z, prefix+'final_review.jsonl')
        e1rows = read_rows(z, prefix+'epoch_01_validation.jsonl')
        reuse = json.loads(z.read(prefix+'final_review.reuse.json'))
        proof = json.loads(z.read(prefix+'final_review.provenance.json'))
        if finalrows[:112] != e1rows or reuse['source_sha256'] != d.digest(z.read(prefix+'epoch_01_validation.jsonl')):
            raise ValueError('112 reused answers differ from selected epoch1')
        serialized = b''.join(z.read(prefix+'final_review.jsonl').splitlines(keepends=True)[:112])
        if (reuse['target_sha256'] != d.digest(serialized) or reuse['adapter_sha256'] != proof['adapter_sha256']
                or reuse['max_new_tokens'] != proof['max_new_tokens']):
            raise ValueError('Reuse weights/budget proof differs')
        for name in sorted(n for n in z.namelist() if n.endswith('.jsonl') and ('/generations/' in n or n.endswith('/smoke_generation.jsonl'))):
            allrows.extend((name, row) for row in read_rows(z, name))
        stage = json.loads(z.read('trials/mixed_pilot/cycle_01/stage_status.json'))
        if stage['state'] != 'complete' or stage['steps'] != 196 or stage['planned_steps'] != 196 or not stage['resume_checkpoint'].replace('\\', '/').endswith('/checkpoint-150'):
            raise ValueError('Formal run/resume differs')
        resumed = [w for w in workers if w['operation'] == 'fit' and w['recovery']]
        if len(resumed) != 1 or resumed[0]['training_start_step'] != 150 or resumed[0]['optimizer_step_values'] != [150] or resumed[0]['optimizer_state_entries'] != 288:
            raise ValueError('Actual optimizer resume differs')
        parent = json.loads(z.read('inheritance_proofs/mixed_pilot.json'))
        if parent['files']['adapter_model.safetensors'] != json.loads(z.read(prefix+'parent_actual_rag.provenance.json'))['adapter_sha256'] or parent['optimizer_inherited']:
            raise ValueError('Parent fit and evaluation weights differ')
        nominee = json.loads(z.read('review_nominee.json'))
        if (nominee['adapter'] is not None or nominee['retained_parent_sha256'] != parent['files']['adapter_model.safetensors']
                or not nominee['research_adapter'].replace('\\','/').endswith('/step-98')):
            raise ValueError('Failed-pilot weight preservation or research selection differs')
        for name in z.namelist():
            if name.endswith('.attention_path.json'):
                a = json.loads(z.read(name))
                if a['cuda_calls'] <= 0 or a['math_fallback_allowed'] or a['software_fixture']:
                    raise ValueError('Unverified actual attention route')
    summary = {'schema': 'wengua_readonly_v59_review_20261109', 'upload_sha256': UPLOAD_SHA,
               'members': len(z.namelist()), 'state': state['state'], 'formal_updates': 196,
               'resume_step': 150, 'post_resume_new_updates': 46, 'full_346_started': False,
               'generation_rows_stored': len(allrows), 'reused_micro_copies': 112,
               'distinct_generated_answer_records_including_smoke': len(allrows)-112,
               'verified_workers': workers, 'parent_recorded_model_sha256': parent['files']['adapter_model.safetensors'],
               'weights_not_uploaded_hash_not_independently_recomputed': True,
               'resumed_trainer_runtime_seconds': stage['training_metrics']['train_runtime'],
               'resumed_fit_stage_seconds': stage['runtime_seconds'], 'post_resume_peak_reserved_gib': stage['peak_reserved_gib'],
               'parent_language_loss': stage['parent_validation_loss'], 'epochs': trial['epochs'], 'frozen_scores': scores,
               'final_reading_scope': dict(Counter(v['gold']['skill'] for v in details if v['tag'] == 'final')),
               'not_rag_or_sft_input': True, 'production_enabled': False,
               'posthoc_read_answer_review_not_new_blind_validation': True}
    d.write(output/'aggregate.json', summary)
    d.write(output/'upload_receipt.json', {'sha256': UPLOAD_SHA, 'audit_source_sha256': d.digest(Path(__file__).read_bytes()),
                                        'frozen_runtime_or_scores_changed': False, 'uploaded_instructions_executed': False})
    with gzip.open(output/'answer_observations.jsonl.gz', 'wt', encoding='utf8') as f:
        for value in details:
            f.write(d.dumps(value)+'\n')
    if workbook:
        from openpyxl import Workbook, load_workbook
        from openpyxl.cell import WriteOnlyCell
        from openpyxl.styles import Alignment
        wb = Workbook(write_only=True)
        ws = wb.create_sheet('逐轮逐题审阅'); ws.freeze_panes = 'E2'
        for col, width in [('A',22),('B',20),('C',20),('D',24),('E',75),('F',75),('G',85),('H',45),('I',85)]:
            ws.column_dimensions[col].width = width
        ws.auto_filter.ref = 'A1:I'+str(len(details)+1)
        ws.append(['阶段', '技能', '领域', 'ID', '问题及来源完整JSON', '原目标', '模型原答', '冻结检查', '限定审阅注释'])
        for index, value in enumerate(details, 2):
            r, g = value['row'], value['gold']
            vals = [value['tag'], g['skill'], g['domain'], r['id'], d.dumps(r['question']), r['expected'], r['answer'], d.dumps(value['frozen_check']), d.dumps(value['read_answer_note'])]
            if any(len(x) > 32767 for x in vals):
                raise ValueError('Workbook complete text boundary exceeded')
            cells = []
            for text in vals:
                cell = WriteOnlyCell(ws, value=text); cell.alignment = Alignment(wrap_text=True, vertical='top'); cells.append(cell)
            ws.row_dimensions[index].height = 100; ws.append(cells)
        ws = wb.create_sheet('全部637原答含复用与短测')
        ws.append(['文件', 'ID', '完整输入JSON', '原目标', '模型原答', '生成tokens', '达到上限'])
        for name, row in allrows:
            ws.append([name,row['id'],d.dumps(row['question']),row['expected'],row['answer'],row['generated_tokens'],row['reached_token_limit']])
        path = output/'Wengua_V59_逐轮逐题审核.xlsx'; wb.save(path)
        reader = load_workbook(path, read_only=True, data_only=True); rows = iter(reader['逐轮逐题审阅'].values); next(rows)
        for expected, got in zip(details, rows, strict=True):
            assert json.loads(got[4]) == expected['row']['question'] and got[5] == expected['row']['expected'] and got[6] == expected['row']['answer']
        rows = iter(reader['全部637原答含复用与短测'].values); next(rows)
        for (_, expected), got in zip(allrows, rows, strict=True):
            assert json.loads(got[2]) == expected['question'] and got[3] == expected['expected'] and got[4] == expected['answer']
        reader.close()
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path); p.add_argument('--output', type=Path); p.add_argument('--notes', type=Path)
    p.add_argument('--workbook', action='store_true'); p.add_argument('--restore', action='store_true'); p.add_argument('--key-file', type=Path)
    a = p.parse_args()
    if a.restore:
        from wengua_corpus.__main__ import restore
        key = a.key_file.read_text(encoding='utf-8-sig').strip() if a.key_file else input('输入此前教材64位口令（可见）：').strip()
        if not re.fullmatch('[0-9a-fA-F]{64}', key):
            raise ValueError('Existing delivery key required')
        print(restore(d.read(runtime.ROOT/'docs/model_reviews/20261109_v59/private_review_catalog.json'), bytes.fromhex(key)))
        return
    if a.archive is None or a.output is None:
        p.error('--archive and --output required')
    s = audit(a.archive, a.output, notes=a.notes, workbook=a.workbook)
    print(d.dumps({k:s[k] for k in ['state','formal_updates','resume_step','generation_rows_stored','reused_micro_copies']}))


if __name__ == '__main__':
    main()
