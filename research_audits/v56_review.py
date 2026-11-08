"""Read V56 receipts and every answer in place; never change frozen scores.

Supplementary wording observations are explicitly post hoc, not a new model
selection rubric, independent validation, RAG document or training target.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import argparse, gzip, json, re, zipfile
from wengua_course import data as d
from wengua_course_v56 import runtime, checks, runner
from wengua_course_v55 import checks as original_v55
from wengua_course_v53.checks import retention
from research_audits.v55_review import inspect_archive, read_rows

TRIALS = ('repair_v54', 'repair_v55')


def collapsed_selectors(support):
    """Different quotes within one source unit must not be deduped by ID alone."""
    merged=list({s['source_id']:s for s in support}.values())
    return [s for s in support if s not in merged]


def source_selector_observation():
    from wengua_course_v54 import corpus
    from wengua_course_v53 import runtime as source
    facts=corpus.properties(source.verify_data(),corpus.source_library());affected=[]
    for a in [1,2,3,4,6,7,8,9]:
        for b in [1,2,3,4,6,7,8,9]:
            support=facts['palace:'+str(a)]['support']+facts['palace:'+str(b)]['support']+facts['rule:generation']['support']+facts['rule:control']['support']
            lost=collapsed_selectors(support)
            if lost:affected.append({'palaces':[a,b],'lost_distinct_quotes':len(lost),'lost_source_ids':[s['source_id'] for s in lost]})
    return {'affected_palace_pairs':len(affected),'new_course_rows_affected':len(affected)*4,
            'training_rows_affected':len(affected)*3,'validation_rows_affected':len(affected),
            'source_originals_unchanged':True,'cause':'different literal selectors sharing a source_id were deduplicated using source_id alone',
            'interpretation':'missing visible attribute excerpts, not proof that book rules are wrong or sole cause of failed answers','cases':affected}


def strip_citations(text):
    return re.sub(r'\[S\d+\]', '', text).strip()


def explicit_relation_labels(text):
    """All unqualified relationship labels, including contradictory tails.

    A law can be right while 'here it is received-control' is wrong. Do not
    infer an unspoken correct conclusion over an explicitly wrong one.
    """
    normalized=text.replace('施生','生').replace('施克','克')
    labels=[]
    head=re.match(r'^\s*(受生|受克|比和|生|克)(?=[。；，,\s]|$)',normalized)
    if head:labels.append(head[1])
    for m in re.finditer(r'(?:(?:结论|关系)(?:为|是)?[：:为是\s]*|这里(?:称为|为|是)?|称为|改为|是|为)(受生|受克|比和|生|克)(?=[。；，,：:\s]|$)',normalized):
        before=normalized[max(0,m.start()-5):m.start()]
        if re.search(r'(?:不|不是|并非|不能|不应)$',before):continue
        labels.append(m[1])
    return labels


def wording_observation(row, gold):
    """Observe bounded fact chains and short properties without semantic claims.

    A correct chain without a canonical label is distinguished from a wrong
    label. Neither an incorrect headline nor an incorrect law is excused.
    """
    text = row['answer']; c = gold.get('checks') or {}
    result = {'explicit_conflicts': checks.old.reasoning_conflicts(text, c),
              'output_limit': bool(row.get('reached_token_limit')),
              'repeated': bool(re.search(r'(.{2,24})\1{7,}', re.sub(r'\s+', '', text))),
              'semantic_professional_approval': False}
    if c.get('positions'):
        result['positions_match'] = checks.legacy.positions_ok(text, c['positions'])
    if result['explicit_conflicts'] or result['output_limit'] or result['repeated']:
        return result
    if c.get('kind') == 'property':
        short = re.sub(r'[。；;\s]', '', strip_citations(text))
        result['unambiguous_short_property'] = short in [c['element'], '属'+c['element'], '属于'+c['element']]
    if c.get('elements') and c.get('kind') in ['reasoning_chain', 'relation', 'joint_relation']:
        claims=explicit_relation_labels(text)
        result['explicit_relation_label_conflict']=any(x!=c['answer'] for x in claims)
        if result['explicit_relation_label_conflict']:return result
        normalized = text.replace('施生', '生').replace('施克', '克')
        cc = c | {'require_positions': c.get('kind') == 'joint_relation' or c.get('require_positions', False)}
        result['alias_chain_passes'] = checks.relation_chain_ok(normalized, cc)
        result['uses_explicit_alias'] = normalized != text
        # Preserve the original words in the audit; this temporary prefix only
        # tests whether all remaining law/binding/position constraints pass.
        has_label = bool(re.search(r'^\s*(?:施生|施克|受生|受克|比和|生|克)[。；，,]', text)
                         or re.search(r'(?:结论|关系)(?:为|是)?[：:为是\s]*(?:施生|施克|受生|受克|比和|生|克)', text))
        result['complete_chain_without_canonical_label'] = not has_label and checks.relation_chain_ok('结论：'+c['answer']+'。'+normalized, cc)
        # The question already supplies elements or verifiable palace facts.
        # Do not require the answer to repeat those words if its law and
        # direction are correct. Check the original headline FIRST so adding
        # given facts cannot hide a contradictory initial relation.
        head = re.match(r'^\s*(受生|受克|比和|生|克)[。；，,]', normalized)
        if not head or head[1] == c['answer']:
            a,b=c['elements']
            with_facts='前者属'+a+'，后者属'+b+'。'+normalized
            if not has_label:with_facts+='结论：'+c['answer']+'。'
            result['bounded_chain_with_question_facts'] = checks.relation_chain_ok(with_facts,cc)
        # For two identical elements/palaces, an explicit consistent 比和
        # answer need not repeat the same given attribute twice.
        if c['elements'][0]==c['elements'][1] and claims and set(claims)=={'比和'}:
            pos_ok=not cc.get('require_positions') or checks.legacy.positions_ok(text,cc['positions'])
            result['explicit_same_element_relation']=pos_ok
    return result


def duration(z, request_name, result_name):
    """ZIP wall-clock difference, 2-second precision; timezone not inferred."""
    start = datetime(*z.getinfo(request_name).date_time)
    end = datetime(*z.getinfo(result_name).date_time)
    seconds = (end-start).total_seconds()
    if seconds < 0: raise ValueError('ZIP member clock order reversed')
    return seconds


def audit(archive, output, *, workbook=False):
    archive, output = Path(archive), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runtime.configure(); pack = runtime.verify_data(); gold = runtime.select_development(pack['records'], 96, repair=True)
    legacy = pack['legacy_v55_prompts']; rg = pack['legacy_actual_rag_rows']
    details = []; timings = []; allrows = []
    with zipfile.ZipFile(archive) as z:
        inspect_archive(z)
        if z.testzip() is not None: raise ValueError('ZIP CRC failed')
        state = json.loads(z.read('run_status.json')); contract = json.loads(z.read('sweep_contract.json'))
        if contract['code_sha256'] != runtime.hashes(): raise ValueError('Frozen V56 code differs')
        if contract['dataset_release_sha256'] != d.digest(runtime.RELEASE.read_text()): raise ValueError('V56 release differs')
        if contract['original_test_used'] is not False: raise ValueError('Unexpected final test use')
        if set(state['trials']) != set(TRIALS): raise ValueError('Unexpected phases; audit this run explicitly')
        reply_count = 0
        for name in z.namelist():
            if name.endswith('.result.json'):
                reply = json.loads(z.read(name)); request_name = name.replace('.result.json', '.json'); request = json.loads(z.read(request_name))
                if (reply['request_sha256'] != d.digest(z.read(request_name)) or reply['nonce'] != request['nonce']
                    or not reply['ok'] or reply['software_fixture'] or reply['result']['state'] != 'complete'):
                    raise ValueError('Unverified or incomplete worker result')
                timings.append({'trial':name.split('/')[1], 'operation':request['operation'], 'probe':request['probe'],
                                'tag':(request.get('best') or {}).get('tag'), 'seconds_zip_precision_2s':duration(z, request_name, name)})
                reply_count += 1
        for name in sorted(n for n in z.namelist() if n.endswith('.jsonl')):
            for row in read_rows(z, name):
                if row['prompt_sha256'] != d.digest(json.dumps(row['question'],ensure_ascii=False,separators=(',',':'))): raise ValueError('Prompt SHA differs')
                allrows.append((name, row))
        summary = {'schema':'wengua_readonly_v56_review_20261107', 'run_state':state['state'],
                   'full_training_started':False, 'original_nominee':json.loads(z.read('review_nominee.json')),
                   'generation_files':len({n for n,_ in allrows}), 'generation_rows':len(allrows),
                   'verified_nonfixture_worker_receipts':reply_count, 'frozen_code_release_prompts_verified':True,
                   'original_final_test_used':False, 'original_scores_rewritten':False,
                   'posthoc_observations_are_not_blind_validation':True, 'trials':{}}
        release = d.read(runtime.RELEASE)
        for trial in TRIALS:
            t = state['trials'][trial]; label = trial.split('_')[-1]
            proof = json.loads(z.read('inheritance_proofs/'+trial+'.json'))
            if proof['files']['adapter_model.safetensors'] != release['reviewed_parents'][label]['adapter_sha256']:
                raise ValueError('Inherited adapter differs')
            prefix = 'trials/'+trial+'/'
            stage = json.loads(z.read(prefix+'cycle_01/stage_status.json'))
            if stage['state'] != 'complete' or stage['steps'] != stage['planned_steps'] or stage['steps'] != 404 or len(stage['epochs']) != 4:
                raise ValueError('Incomplete short training')
            if not stage['parent_weights_reused'] or stage['fresh_official_base'] or stage['optimizer_reset_between_epochs']:
                raise ValueError('Weight/optimizer continuation differs')
            baseline = checks.score(read_rows(z, prefix+'generations/validation_parent.jsonl'), gold)
            oldbase = checks.score(read_rows(z, prefix+'generations/parent_v55_unchanged.jsonl'), legacy)
            if baseline != t['parent_baseline'] or oldbase != t['v55_parent_baseline']: raise ValueError('Parent score differs')
            stages = []
            sets = [('validation_base', gold, None), ('validation_parent', gold, baseline), ('parent_v55_unchanged', legacy, oldbase)]
            for i,e in enumerate(t['epochs'], 1):
                tag = f'epoch_{i:02d}'
                current = checks.score(read_rows(z,prefix+'generations/'+tag+'_validation.jsonl'),gold)
                oldrows = read_rows(z,prefix+'generations/'+tag+'_v55_unchanged.jsonl'); oldscore = checks.score(oldrows,legacy)
                if current != e['source_checks'] or oldscore != e['v55_with_v56_checks']:
                    raise ValueError('Frozen V56 scoring does not reproduce')
                if original_v55.score(oldrows,legacy) != e['original_v55_checks_unchanged']: raise ValueError('Original V55 rubric differs')
                if retention(current,baseline['groups']) != e['parent_retention'] or retention(oldscore,oldbase['groups']) != e['v55_retention']:
                    raise ValueError('Retention differs')
                rag = read_rows(z,prefix+'generations/'+tag+'_actual_rag_unchanged.jsonl')
                if {r['id'] for r in rag} != {r['id'] for r in rg}: raise ValueError('RAG regression incomplete')
                bg = {r['id']:r for r in rg}
                for r in rag:
                    if r['question'] != bg[r['id']]['messages'][:-1] or r['expected'] != bg[r['id']]['answer']: raise ValueError('RAG prompt/target changed')
                rb = runner.rag_boundaries(rag)
                for k in ['items','critical_tasks','semantic_professional_approval']:
                    if rb[k] != e['rag_boundaries'][k]: raise ValueError('RAG boundary checks differ')
                from src.wengua.book_rag_query import check_answer
                sourcechecks = [{'id':r['id'],'checks':check_answer(r['answer'],bg[r['id']]['legacy_packet'])} for r in rag]
                if sourcechecks != e['rag_boundaries']['citation_and_chart_checks']: raise ValueError('RAG source checks differ')
                passed = rb['passed'] and all(r['checks']['status']=='citations_checked_semantics_unverified' for r in sourcechecks)
                if passed != e['rag_boundaries']['passed']: raise ValueError('RAG status differs')
                eligible = checks.gate(current,phase='repair')['passed'] and e['parent_retention']['passed'] and e['v55_retention']['passed'] and passed
                if eligible != e['eligible_for_next_phase']: raise ValueError('Acceptance differs')
                stages.append({'epoch':e['epoch'],'steps':e['steps'],'loss':e['validation_language_loss'],
                               'new_groups':current['groups'],'old_groups':oldscore['groups'],
                               'new_citation':current['citation'],'old_citation':oldscore['citation'],
                               'original_gate':checks.gate(current),'parent_retention':e['parent_retention']['passed'],
                               'old_retention':e['v55_retention']['passed'],'rag_critical_failures':[r['id'] for r in rb['items'] if not r['passed']],
                               'original_eligible':eligible})
                sets += [(tag+'_validation',gold,current),(tag+'_v55_unchanged',legacy,oldscore)]
                for r in rag: details.append({'trial':trial,'tag':tag+'_actual_rag','gold':bg[r['id']],'row':r,'original':None,'observation':wording_observation(r,bg[r['id']])})
            observed = {}
            for tag, gg, score in sets:
                values = read_rows(z,prefix+'generations/'+tag+'.jsonl'); byid = {r['id']:r for r in gg}
                if score is None: score=checks.score(values,gg)
                buckets=defaultdict(Counter)
                for r in values:
                    g=byid[r['id']];obs=wording_observation(r,g);fact=checks.fact_check(r,g)
                    details.append({'trial':trial,'tag':tag,'gold':g,'row':r,'original':fact,'observation':obs})
                    b=buckets[g['skill']];b['total']+=1;b['original_pass']+=fact['status']=='pass'
                    for key in ['positions_match','uses_explicit_alias','alias_chain_passes','complete_chain_without_canonical_label','bounded_chain_with_question_facts','explicit_same_element_relation','explicit_relation_label_conflict','unambiguous_short_property','output_limit','repeated']:
                        b[key]+=bool(obs.get(key))
                    b['explicit_conflicts']+=bool(obs['explicit_conflicts'])
                    b['supplementary_wording_only_misses']+=fact['status']!='pass' and any(obs.get(k) for k in ['alias_chain_passes','complete_chain_without_canonical_label','bounded_chain_with_question_facts','explicit_same_element_relation','unambiguous_short_property'])
                observed[tag]={k:dict(v) for k,v in buckets.items()}
            summary['trials'][trial]={'parent_groups':baseline['groups'],'old_parent_groups':oldbase['groups'],'epochs':stages,
                                      'formal_updates':stage['steps'],'parent_weights_reused':stage['parent_weights_reused'],
                                      'peak_reserved_gib':stage['peak_reserved_gib'],'fit_worker_seconds':stage['runtime_seconds'],
                                      'trainer_runtime_seconds_includes_callbacks':stage['training_metrics']['train_runtime'],'observations':observed}
        assert not any(e['original_eligible'] for t in summary['trials'].values() for e in t['epochs'])
        if state['state']!='needs_course_review': raise ValueError('Terminal state differs')
        totals=defaultdict(float)
        for item in timings:
            category='regression_workers' if item['operation']=='evaluate' else 'pressure_workers' if item['probe'] else 'formal_fit_workers'
            totals[category]+=item['seconds_zip_precision_2s']
        summary['worker_wall_seconds_zip_precision_2s']=dict(totals)
        summary['repair_course_skills']=dict(Counter(r['skill'] for r in runtime.repair_training(pack['records'])))
        summary['author_case_workflow_lessons_in_short_training']=sum(r['skill'] in ['case','author_reading','author_context','author_local','workflow','domain_rule'] for r in runtime.repair_training(pack['records']))
        selector=source_selector_observation()
        d.write(output/'source_selector_observation.json',selector)
        summary['source_selector_observation']={k:v for k,v in selector.items() if k!='cases'}
        summary['not_rag_or_sft_input']=True
        d.write(output/'aggregate.json',summary);d.write(output/'worker_timings.json',timings)
        d.write(output/'upload_receipt.json',{'archive_sha256':d.digest(archive.read_bytes()),'members':len(z.infolist()),'uncompressed_bytes':sum(i.file_size for i in z.infolist()),'audit_source_sha256':d.digest(Path(__file__).read_bytes()),'read_without_extraction':True,'uploaded_code_executed':False,'original_run_and_scores_unchanged':True,'not_rag_or_sft_input':True})
        with gzip.open(output/'answer_observations.jsonl.gz','wt',encoding='utf8') as f:
            for r in details:f.write(d.dumps(r)+'\n')
        if workbook:
            from openpyxl import Workbook
            wb=Workbook(write_only=True);ws=wb.create_sheet('全部原答')
            ws.append(['原文件','ID','完整输入JSON','原目标','原答','tokens','达到上限'])
            for name,r in allrows:
                values=[name,r['id'],d.dumps(r['question']),r['expected'],r['answer'],r['generated_tokens'],r['reached_token_limit']]
                if any(isinstance(v,str) and len(v)>32767 for v in values):raise ValueError('Workbook cell needs chunking')
                ws.append(values)
            ws=wb.create_sheet('逐题原检查与补充观察');ws.append(['组','阶段','技能','领域','ID','原检查JSON','补充观察非新分数'])
            for v in details:ws.append([v['trial'],v['tag'],v['gold'].get('skill',v['gold'].get('task','actual_rag')),v['gold'].get('domain','actual_rag'),v['row']['id'],d.dumps(v['original']),d.dumps(v['observation'])])
            ws=wb.create_sheet('逐轮汇总');ws.append(['组','轮','步','loss','原新题分项','原旧题分项','原门槛','RAG未过项'])
            for t,v in summary['trials'].items():
                for e in v['epochs']:ws.append([t,e['epoch'],e['steps'],e['loss'],d.dumps(e['new_groups']),d.dumps(e['old_groups']),d.dumps(e['original_gate']),d.dumps(e['rag_critical_failures'])])
            ws=wb.create_sheet('运行耗时');ws.append(['完整计时记录JSON','ZIP精度2秒、不推断时区'])
            for v in timings:ws.append([d.dumps(v),True])
            ws=wb.create_sheet('来源选择器边界');ws.append(['不同摘录被ID去重的记录JSON','原文未改'])
            for v in selector['cases']:ws.append([d.dumps(v),True])
            manual=output/'semantic_review.json'
            if manual.exists():
                ws=wb.create_sheet('读答审阅与限制');ws.append(['类别','完整记录JSON','不作教师、不自动专业晋升'])
                for key,values in d.read(manual).items():
                    for value in values:ws.append([key,d.dumps(value),True])
            wb.save(output/'Wengua_V56_逐轮逐题审核.xlsx')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--archive',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--workbook',action='store_true');parser.add_argument('--restore',action='store_true');parser.add_argument('--key-file',type=Path);a=parser.parse_args()
    if a.restore:
        from wengua_corpus.__main__ import restore
        value=a.key_file.read_text(encoding='utf-8-sig').strip() if a.key_file else input('输入此前教材64位口令（可见）：').strip()
        if not re.fullmatch('[0-9a-fA-F]{64}',value):raise ValueError('Expected the existing delivery key')
        print(restore(d.read(runtime.ROOT/'docs/model_reviews/20261107_v56/private_review_catalog.json'),bytes.fromhex(value)));return
    if a.archive is None or a.output is None:parser.error('--archive and --output required for audit')
    result=audit(a.archive,a.output,workbook=a.workbook)
    print(d.dumps({k:result[k] for k in ['run_state','generation_files','generation_rows','verified_nonfixture_worker_receipts','worker_wall_seconds_zip_precision_2s']}))


if __name__=='__main__':main()
