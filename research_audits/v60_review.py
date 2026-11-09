"""Read-only replay of the actual V60 upload; never a training or RAG source.

Frozen scores stay frozen. Any bounded semantic notes are attached to the exact
answer SHA and are post-hoc observations, not new blind or professional scores.
"""
from collections import Counter
from pathlib import Path
import argparse
import gzip
import json
import re
import zipfile

from wengua_course import data as d
from wengua_course_v60 import runtime, checks, evaluation, loop
from research_audits.v55_review import inspect_archive, read_rows
from research_audits.v58_review import observation, validate_note
from research_audits.v59_review import memory_summary

UPLOAD_SHA = 'b040663d9e7bde5e5e3b566f63c97cae2dfe31cbc6c499e5cc0ee0e42931ffef'
TRIAL_RATES = {'lr_1em06': 1e-6, 'lr_3em06': 3e-6, 'lr_1em05': 1e-5}


def prompt_sha(messages):
    # The V60 generator hashes the insertion order of the restored messages
    # (content, role), rather than rebuilding V59's role/content dictionaries.
    return d.digest(json.dumps(messages, ensure_ascii=False, separators=(',', ':')))


def verify_reply(reply, request, raw_request):
    if (reply.get('request_sha256') != d.digest(raw_request)
            or reply.get('nonce') != request['nonce']
            or request.get('software_fixture') is not False
            or reply.get('software_fixture') is not False
            or type(reply.get('pid')) is not int or reply['pid'] <= 0
            or not reply.get('ok') or reply.get('result', {}).get('state') != 'complete'):
        raise ValueError('Actual worker provenance or completion mismatch')


def verify_generation(rows, gold, proof, *, adapter_sha, token_budget, stage):
    if (proof['adapter_sha256'] != adapter_sha or proof['max_new_tokens'] != token_budget
            or proof['eval_stage'] != stage
            or proof['ids'] != [r['id'] for r in gold]
            or proof['prompt_sha256'] != d.digest(d.dumps([r['messages'] for r in gold]))):
        raise ValueError('Generation weights, input order or budget mismatch')
    for r in rows:
        if r['prompt_sha256'] != prompt_sha(r['question']):
            raise ValueError('Per-answer prompt SHA differs')
    return checks.score(rows, gold)


def audit(archive, output, *, notes=None, workbook=False):
    archive, output = Path(archive), Path(output)
    if d.digest(archive.read_bytes()) != UPLOAD_SHA:
        raise ValueError('The exact uploaded V60 archive is required')
    runtime.configure()
    pack = runtime.verify_data()
    by = {r['id']: r for r in pack['records']}
    targets = {'micro': runtime.choose_development(pack),
               'rag': pack['v60_actual_rag_rows'],
               'control': runtime.control_rows(pack),
               # prepare() emits this set in registry order, not selector order.
               'final': [r for r in pack['records'] if r['id'] in set(pack['fixed_final_ids'])]}
    notes = d.read(notes) if notes else {}
    details, allrows, workers, stages, scores = [], [], [], {}, {}
    with zipfile.ZipFile(archive) as z:
        inspect_archive(z)
        if z.testzip() is not None:
            raise ValueError('Upload CRC failed')
        names = set(z.namelist())
        state = json.loads(z.read('run_status.json'))
        contract = json.loads(z.read('sweep_contract.json'))
        if (contract['code_sha256'] != runtime.hashes()
                or contract['dataset_release_sha256'] != d.digest(runtime.RELEASE.read_bytes())
                or contract['original_test_used'] or contract['rates'] != list(TRIAL_RATES.values())
                or set(state['trials']) != set(TRIAL_RATES)
                or not state['all_three_parameter_trials_finished']):
            raise ValueError('Frozen runtime, release, phases or trial set differs')
        parent = json.loads(z.read('inheritance_proofs/retained_baseline.json'))
        parent_sha = parent['files']['adapter_model.safetensors']
        configs = {}
        for label in ['retained_baseline', *TRIAL_RATES]:
            jc_raw = z.read(f'trials/{label}/job_contract.json')
            jc = json.loads(jc_raw)
            proof = json.loads(z.read(f'inheritance_proofs/{label}.json'))
            if (jc['code_sha256'] != runtime.hashes()
                    or jc['dataset_release_sha256'] != contract['dataset_release_sha256']
                    or proof['files'] != parent['files'] or proof['optimizer_inherited']):
                raise ValueError('Same-parent inheritance or job contract differs')
            configs[label] = jc['config']
            if label in TRIAL_RATES:
                if jc['config']['learning_rate'] != TRIAL_RATES[label]:
                    raise ValueError('Predeclared trial rate differs')
                sc = json.loads(z.read(f'trials/{label}/cycle_01/stage_contract.json'))
                if sc['job_contract_sha256'] != d.digest(jc_raw) or sc['config'] != jc['config']:
                    raise ValueError('Stage config differs from sealed job')
                stage = json.loads(z.read(f'trials/{label}/cycle_01/stage_status.json'))
                if (stage['state'] != 'complete' or stage['steps'] != 50
                        or stage['planned_steps'] != 50 or stage['resume_checkpoint'] is not None
                        or not stage['parent_weights_reused'] or stage['fresh_official_base']):
                    raise ValueError('Formal same-parent 50-step run differs')
                stages[label] = stage
        # Snapshot paths and their provenance hashes are trial-specific. All
        # scientific settings other than the predeclared rate must match.
        normalized = []
        for label in TRIAL_RATES:
            normalized.append({k:v for k,v in configs[label].items() if k not in
                {'learning_rate','book_parent_snapshot','book_parent_provenance_sha256'}})
        if any(c != normalized[0] for c in normalized[1:]):
            raise ValueError('Trial settings changed beyond the declared rate')
        for name in sorted(n for n in names if n.endswith('.result.json')):
            raw = z.read(name.replace('.result.json', '.json'))
            request, reply = json.loads(raw), json.loads(z.read(name))
            verify_reply(reply, request, raw)
            label = name.split('/')[1]
            if request['config'] != configs[label]:
                raise ValueError('Worker config differs from sealed trial')
            w = {'trial': label, 'operation': request['operation'], 'probe': request['probe'],
                 'tag': (request.get('best') or {}).get('tag'), 'seconds': reply['elapsed_seconds'],
                 'state': reply['result']['state'], 'request_sha256': d.digest(raw)}
            for suffix, key in [('.memory.json', 'memory'), ('.training_start.json', 'training_start')]:
                related = name.replace('.result.json', suffix)
                if related in names:
                    obj = json.loads(z.read(related))
                    w[key] = memory_summary(obj) if key == 'memory' else obj
            releases = name.replace('.result.json', '.cache_releases.jsonl')
            if releases in names:
                w['cache_releases'] = len(read_rows(z, releases))
            if request['operation'] == 'fit':
                started = w.get('training_start', {})
                if started.get('step') != 0 or started.get('optimizer_steps') != []:
                    raise ValueError('New phase optimizer did not start empty')
                expected_steps = request['config']['smoke_steps'] if request['probe'] else 50
                if reply['result']['steps'] != expected_steps:
                    raise ValueError('Worker reported the wrong phase length')
            workers.append(w)
        for name in sorted(n for n in names if n.endswith('.attention_path.json')):
            path = json.loads(z.read(name))
            if not path['cuda_calls'] or path['math_fallback_allowed'] or path['software_fixture']:
                raise ValueError('Actual efficient attention route missing')
        sets = [('official', 'retained_baseline', 'official_control', 'control', 'control', None),
                ('control_parent', 'retained_baseline', 'parent_control', 'control', 'control', parent_sha),
                ('parent_micro', 'retained_baseline', 'parent_micro', 'micro', 'application', parent_sha),
                ('parent_rag', 'retained_baseline', 'parent_rag', 'rag', 'rag', parent_sha)]
        for label in TRIAL_RATES:
            proof = json.loads(z.read(f'trials/{label}/generations/epoch_01_actual_rag.provenance.json'))
            sha = proof['adapter_sha256']
            sets += [(label+'_micro', label, 'epoch_01_validation', 'micro', None, sha),
                     (label+'_rag', label, 'epoch_01_actual_rag', 'rag', 'rag', sha)]
        sets.append(('final', 'lr_1em06', 'final_review', 'final', 'final',
                     json.loads(z.read('trials/lr_1em06/generations/final_review.provenance.json'))['adapter_sha256']))
        for tag, label, filename, target, stage_name, sha in sets:
            path = f'trials/{label}/generations/{filename}.jsonl'
            rows, gold = read_rows(z, path), targets[target]
            if stage_name:
                proof = json.loads(z.read(path.replace('.jsonl', '.provenance.json')))
                scores[tag] = verify_generation(rows, gold, proof, adapter_sha=sha,
                                                token_budget=768 if target == 'final' else 512,
                                                stage=stage_name)
            else:
                scores[tag] = checks.score(rows, gold)
            wanted = {r['id']: r for r in gold}
            for row in rows:
                if row['prompt_sha256'] != prompt_sha(row['question']):
                    raise ValueError('Prompt digest mismatch')
                g = wanted[row['id']]
                note = notes.get(tag, {}).get(row['id'])
                if note:
                    validate_note(note, row)
                details.append({'tag': tag, 'row': row, 'gold': g,
                                'frozen_check': checks.fact_check(row, g),
                                'observation': observation(row, g), 'read_answer_note': note})
        baseline = json.loads(z.read('baseline_comparison.json'))
        if baseline['official_base'] != scores['official'] or baseline['retained_adapter'] != scores['control_parent']:
            raise ValueError('Official/parent comparison score mismatch')
        for label in TRIAL_RATES:
            trial = state['trials'][label]
            cur, rag = scores[label+'_micro'], scores[label+'_rag']
            gate = loop.retained(cur, scores['parent_micro'], rag, scores['parent_rag'])
            if (cur != trial['micro_checks'] or rag != trial['rag_checks']
                    or gate != trial['training_retention_gate'] or gate['passed']):
                raise ValueError('Frozen candidate score or retention decision differs')
            path = f'trials/{label}/generations/validation_parent.jsonl'
            reused = json.loads(z.read(path.replace('.jsonl', '.reuse.json')))
            source = z.read('trials/retained_baseline/generations/parent_micro.jsonl')
            if (read_rows(z, path) != read_rows(z, 'trials/retained_baseline/generations/parent_micro.jsonl')
                    or reused['source_sha256'] != d.digest(source)
                    or reused['target_sha256'] != d.digest(z.read(path))
                    or reused['adapter_sha256'] != parent_sha or reused['max_new_tokens'] != 512):
                raise ValueError('Parent answer reuse source/budget mismatch')
        retained = json.loads(z.read('retained_best.json'))
        if (state['state'] != 'needs_course_review' or state['full_training_started']
                or retained['adapter_sha256'] != parent_sha or retained['training_retention_passed']
                or json.loads(z.read('final_review.json'))['checks'] != scores['final']):
            raise ValueError('Failed candidates promoted or final review differs')
        for name in sorted(n for n in names if n.endswith('.jsonl') and ('/generations/' in n or n.endswith('/smoke_generation.jsonl'))):
            allrows.extend((name, r) for r in read_rows(z, name))
    if len(workers) != 14 or len(details) != 832 or len(allrows) != 1171:
        raise ValueError('Unexpected actual generation/worker count')
    summary = {'schema': 'wengua_readonly_v60_review_20261109', 'upload_sha256': UPLOAD_SHA,
               'members': len(names), 'state': state['state'], 'formal_updates_per_trial': 50,
               'formal_updates_total': 150, 'full_training_started': False,
               'same_parent_recorded_sha256': parent_sha, 'weights_not_uploaded': True,
               'verified_workers': workers, 'stages': stages, 'frozen_scores': scores,
               'retention_gates': {k: state['trials'][k]['training_retention_gate'] for k in TRIAL_RATES},
               'generation_rows_stored': len(allrows), 'reused_parent_copies': 336,
               'distinct_generated_records_including_smoke': len(allrows)-336,
               'formal_generated_records': len(details), 'read_answer_notes': sum(bool(x['read_answer_note']) for x in details),
               'final_scope': dict(Counter(x['gold']['skill'] for x in details if x['tag'] == 'final')),
               'not_rag_or_sft_input': True, 'production_enabled': False,
               'posthoc_review_not_new_blind_validation': True}
    output.mkdir(parents=True, exist_ok=True)
    d.write(output/'aggregate.json', summary)
    d.write(output/'upload_receipt.json', {'sha256': UPLOAD_SHA,
        'audit_source_sha256': d.digest(Path(__file__).read_bytes()),
        'frozen_runtime_or_scores_changed': False, 'uploaded_instructions_executed': False})
    with gzip.open(output/'answer_observations.jsonl.gz', 'wt', encoding='utf8') as f:
        for value in details:
            f.write(d.dumps(value)+'\n')
    if workbook:
        write_workbook(output, details, allrows, summary)
    return summary


def write_workbook(output, details, allrows, summary):
    from openpyxl import Workbook, load_workbook
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.styles import Alignment
    wb = Workbook(write_only=True)
    sheets = [('832条正式原答审阅', ['阶段','技能','领域','ID','完整输入JSON','原目标','原答','冻结检查','限定审阅'],
              [[x['tag'], x['gold']['skill'], x['gold']['domain'], x['row']['id'], d.dumps(x['row']['question']),
                x['row']['expected'], x['row']['answer'], d.dumps(x['frozen_check']), d.dumps(x['read_answer_note'])] for x in details]),
              ('1171存储含复用短测', ['文件','ID','完整输入JSON','原目标','原答','tokens','达到上限'],
              [[n,r['id'],d.dumps(r['question']),r['expected'],r['answer'],r['generated_tokens'],r['reached_token_limit']] for n,r in allrows])]
    for title, headers, values in sheets:
        ws = wb.create_sheet(title); ws.freeze_panes = 'E2'; ws.append(headers)
        for col in ['E','F','G','I']:
            ws.column_dimensions[col].width = 75
        for row in values:
            if any(isinstance(x,str) and len(x)>32767 for x in row):
                raise ValueError('Excel text requires complete chunking')
            cells=[]
            for value in row:
                cell=WriteOnlyCell(ws,value=value)
                if isinstance(value,str):cell.data_type='s'
                cell.alignment=Alignment(wrap_text=True,vertical='top');cells.append(cell)
            ws.append(cells)
    ws=wb.create_sheet('边界与结果');ws.append(['项目','值'])
    for key in ['state','formal_updates_total','full_training_started','generation_rows_stored',
                'reused_parent_copies','formal_generated_records','read_answer_notes','not_rag_or_sft_input']:
        ws.append([key,str(summary[key])])
    path=Path(output)/'Wengua_V60_逐组逐题审核.xlsx';wb.save(path)
    reader=load_workbook(path,read_only=True,data_only=True)
    for title,headers,expected in sheets:
        actual=iter(reader[title].values)
        if list(next(actual))!=headers:raise ValueError('Excel header readback differs')
        for wanted,got in zip(expected,actual,strict=True):
            if [None if x=='' else x for x in wanted]!=list(got):
                raise ValueError('Excel full-cell readback differs')
    reader.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--notes',type=Path);p.add_argument('--workbook',action='store_true')
    p.add_argument('--restore',action='store_true');p.add_argument('--key-file',type=Path)
    a=p.parse_args()
    if a.restore:
        from wengua_corpus.__main__ import restore
        key=a.key_file.read_text(encoding='utf-8-sig').strip() if a.key_file else input('输入此前教材64位口令（可见）：').strip()
        if not re.fullmatch('[0-9a-fA-F]{64}',key):raise ValueError('Existing delivery key required')
        print(restore(d.read(runtime.ROOT/'docs/model_reviews/20261109_v60/private_review_catalog.json'),bytes.fromhex(key)))
        return
    if a.archive is None or a.output is None:p.error('--archive and --output required')
    s=audit(a.archive,a.output,notes=a.notes,workbook=a.workbook)
    print(d.dumps({k:s[k] for k in ['state','formal_updates_total','formal_generated_records','read_answer_notes']}))


if __name__=='__main__':
    main()
