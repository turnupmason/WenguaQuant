"""Read-only V54 upload audit. All overlays are post hoc, not new model scores.

Run from repository root with restored V53/V54 private sources. Original run
files, gates, model weights and frozen training code are never modified.
"""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, re
from wengua_course import data as d
from wengua_course_v54 import runtime, checks
from wengua_course_v53 import checks as previous

TRIALS=('source_lr_1em05','source_lr_2em05')

def reasoning_overlay(answer,elements):
 """Explicit cycle/subject contradictions only; not a semantic expert grader."""
 issues=list(checks.reasoning_conflicts(answer));subjects=dict(zip(['前者','后者'],elements))
 for m in re.finditer(r'(前者|后者)(生|克)(前者|后者)',answer):
  if any(w in answer[max(0,m.start()-5):m.start()] for w in ['不是','并非','不能','不应']):continue
  a,op,b=m.groups();mapping=d.GENERATES if op=='生' else d.CONTROLS
  if mapping[subjects[a]]!=subjects[b]:issues.append({'claim':m[0],'reason':'explicit_subject_direction_conflicts_with_given_elements'})
 return issues

def name_overlay(row,gold,known):
 answer=row['answer'];expected=gold['checks']['answer'];found=[]
 for name in known:
  for m in re.finditer(re.escape(name),answer):
   if any(w in answer[max(0,m.start()-5):m.start()] for w in ['不是','并非','不能','不应']):continue
   found.append(name);break
 alternative=bool(re.search(r'(?:命名取|称为|名为)[^。\n]{0,40}、',answer))
 return {'expected_name_present':expected in found,'other_known_names':sorted(set(found)-{expected}),'explicit_name_alternatives':alternative,'only_expected_among_known_names':set(found)=={expected} and not alternative,'semantic_approval':False,'original_score_unchanged':True}

def audit(root,out,*,workbook=False):
 root=Path(root);out=Path(out)
 if out.resolve().is_relative_to(root.resolve()):raise ValueError('Audit must be outside the immutable original upload')
 out.mkdir(parents=True,exist_ok=True);pack=runtime.verify_data();gold=runtime.choose_development(pack,128);byid={r['id']:r for r in gold};state=d.read(root/'run_status.json');contract=d.read(root/'sweep_contract.json')
 assert contract['dataset_release_sha256']==d.digest(runtime.RELEASE.read_text());assert contract['code_sha256']==runtime.hashes()
 allrows=[]
 for path in sorted(root.glob('trials/**/*.jsonl')):
  for line in path.read_text(encoding='utf8').splitlines():
   row=json.loads(line);assert row['prompt_sha256']==hashlib.sha256(json.dumps(row['question'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest();allrows.append((path.relative_to(root).as_posix(),row))
 names={r['checks']['answer'] for r in pack['records'] if r.get('checks',{} ) and r['checks']['kind']=='exact_name' and (r['task']=='typed_pair' or r['task'].startswith('source_named_pair_'))}
 summary={'schema':'wengua_readonly_v54_review_20261106','run_state':state['state'],'original_selected_for_review':state['selected_for_review'],'generation_files':len(list(root.glob('trials/**/*.jsonl'))),'generation_rows':len(allrows),'contract_and_prompt_hashes_match':True,'original_final_test_used':False,'original_scores_rewritten':False,'trials':{},'posthoc_overlays_are_not_new_blind_validation':True};details=[];typed=[];cases=[];rag=[]
 for trial in TRIALS:
  t=state['trials'][trial];baseline=None;official=None;epochs=[]
  for label in ['validation_base','validation_parent','epoch_01_validation','epoch_02_validation']:
   path=root/'trials'/trial/'generations'/(label+'.jsonl');rows=[json.loads(s) for s in path.read_text(encoding='utf8').splitlines()];scored=checks.score(rows,gold)
   if label!='validation_base':
    expected=t['parent_baseline'] if label=='validation_parent' else t['epochs'][int(label[6:8])-1]['source_checks'];assert scored==expected
   chart=[];names_out=[];directions=[]
   for row in rows:
    g=byid[row['id']];overlay={};kind=g.get('checks',{}).get('kind') if g.get('checks') else None
    if g['skill']=='chart_direction':
     pos=g['checks']['positions'];elements=[d.ELEMENT[p] for p in pos.values()];overlay={'positions_match':previous.positions_ok(row['answer'],pos),'explicit_reasoning_conflicts':reasoning_overlay(row['answer'],elements)};chart.append(overlay)
    if g['skill']=='direction':overlay={'explicit_reasoning_conflicts':reasoning_overlay(row['answer'],g['checks']['elements'])};directions.append((row,g,overlay))
    if g['skill']=='typed_pair':overlay=name_overlay(row,g,names);names_out.append(overlay);typed.append((trial,label,g,row,overlay))
    if g['skill']=='case':
     lines=[x.strip() for x in row['answer'].splitlines() if x.strip()];repeats=[x for x,n in Counter(lines).items() if n>1]
     overlay={'exact_repeated_lines':repeats,'only_feedback_no_analysis':('后来反馈' in row['answer'] and not any(w in row['answer'] for w in ['初断','分析中','原书解释','原书补充','书中解释'])),'source_study_prompt_already_contains_author_analysis_and_feedback':True,'independent_chart_reasoning_verified':False};cases.append((trial,label,g,row,overlay))
    details.append((trial,label,g,row,overlay))
   item={'label':label,'original_groups':scored['groups'],'original_citation_check':scored['citation'],'chart_positions_matching':{'pass':sum(x['positions_match'] for x in chart),'total':len(chart)},'chart_explicit_reasoning_conflicts':sum(bool(x['explicit_reasoning_conflicts']) for x in chart),'direction_explicit_reasoning_conflicts':sum(bool(x[2]['explicit_reasoning_conflicts']) for x in directions),'typed_name_overlay':{'expected_name_present':sum(x['expected_name_present'] for x in names_out),'only_expected_among_known_names':sum(x['only_expected_among_known_names'] for x in names_out),'total':len(names_out),'fact_name_only_not_full_semantic_approval':True}}
   if label=='validation_base':official=item
   elif label=='validation_parent':baseline=item
   else:
    e=t['epochs'][int(label[6:8])-1];item.update(epoch=e['epoch'],steps=e['steps'],validation_language_loss=e['validation_language_loss'],original_eligible=e['eligible_for_review'],original_legacy_groups=e['original_v53_legacy_checks']['groups']);epochs.append(item)
  # All original 18 RAG prompts remain exactly the exposed V53 source packets.
  original=d.read(runtime.ROOT/'local/reviews/v53_20261106/trials/foundation_v52_comparison/rag_probe_sources/foundation_final_actual_rag_256.json')['packets']
  for epoch in [1,2]:
   rows=[json.loads(l) for l in (root/'trials'/trial/'generations'/f'epoch_{epoch:02d}_v53_actual_rag_unchanged.jsonl').read_text(encoding='utf8').splitlines()];assert len(rows)==18
   for row in rows:assert row['question']==original[row['id']]['messages'];rag.append((trial,epoch,row))
   epochs[epoch-1]['original_18_rag_citation_membership_pass']=sum(x['check']['status']=='citations_checked_semantics_unverified' for x in t['epochs'][epoch-1]['original_v53_rag_checks'])
  status=d.read(root/'trials'/trial/'cycle_01/stage_status.json');assert status['steps']==status['planned_steps']==268 and status['state']=='complete';assert len(status['epochs'])==2
  summary['trials'][trial]={'official_base_diagnostic':official,'parent_baseline':baseline,'epochs':epochs,'actual_formal_steps':status['steps'],'peak_reserved_gib':status['peak_reserved_gib'],'formal_runtime_seconds':status['runtime_seconds']}
 # Knowledge/task inventory, not an assertion that all workflows are structured.
 coverage=[];old=d.read(runtime.ROOT/'local/course_v53_data/pack.json')
 for domain in sorted({c['domain'] for c in pack['case_catalog']}):
  catalog=[c for c in pack['case_catalog'] if c['domain']==domain];rows=[r for r in pack['records'] if r['domain']==domain]
  coverage.append({'domain':domain,'source_cases':len(catalog),'source_author_steps':sum(len(c['author_steps']) for c in catalog),'explicit_reviewed_rule_specs':sum(s['domain']==domain for s in old['reviewed_specs']),'v54_train_lessons':sum(r['split']=='train' for r in rows),'v54_development_lessons':sum(r['split']=='validation' for r in rows),'workflow_step_coverage_verified':False})
 d.write(out/'aggregate.json',summary);d.write(out/'domain_inventory.json',coverage)
 if workbook:
  from openpyxl import Workbook
  wb=Workbook(write_only=True)
  sheet=wb.create_sheet('全部1670原答');sheet.append(['原文件','ID','完整原问题JSON','原答','来源目标','tokens','达到长度上限'])
  for path,r in allrows:sheet.append([path,r['id'],d.dumps(r['question']),r['answer'],r['expected'],r['generated_tokens'],r['reached_token_limit']])
  sheet=wb.create_sheet('新128题分解');sheet.append(['参数组','阶段','领域','技能','ID','原目标','原答','原事实检查JSON','检查补充分解JSON'])
  for trial,label,g,r,overlay in details:sheet.append([trial,label,g['domain'],g['skill'],r['id'],g['answer'],r['answer'],d.dumps(checks.fact_check(r,g)),d.dumps(overlay)])
  sheet=wb.create_sheet('格名与句式');sheet.append(['组','阶段','天地盘','应答格名','原答','格名检查补充JSON'])
  for trial,label,g,r,overlay in typed:sheet.append([trial,label,g['checks']['pair'],g['checks']['answer'],r['answer'],d.dumps(overlay)])
  sheet=wb.create_sheet('实际RAG72答');sheet.append(['组','轮','题目','原答','自动引用检查不等于语义正确'])
  for trial,epoch,r in rag:sheet.append([trial,epoch,r['id'],r['answer'],True])
  sheet=wb.create_sheet('案例复盘');sheet.append(['组','阶段','领域','案例ID','原目标','原答','结构观察JSON','独立断盘已验证'])
  for trial,label,g,r,overlay in cases:sheet.append([trial,label,g['domain'],'\n'.join(g['case_ids']),g['answer'],r['answer'],d.dumps(overlay),False])
  sheet=wb.create_sheet('领域覆盖清单');sheet.append(list(coverage[0]))
  for row in coverage:sheet.append(list(row.values()))
  wb.save(out/'Wengua_V54_逐轮逐题审核.xlsx')
 return summary

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path);p.add_argument('--output',type=Path);p.add_argument('--workbook',action='store_true');p.add_argument('--restore',action='store_true');p.add_argument('--key-file',type=Path);a=p.parse_args()
 if a.restore:
  from wengua_corpus.__main__ import restore
  value=a.key_file.read_text(encoding='utf-8-sig').strip() if a.key_file else input('输入此前教材64位口令（可见）：').strip()
  if not re.fullmatch('[0-9a-fA-F]{64}',value):raise ValueError('Expected the existing 64-character delivery key')
  entry=d.read(runtime.ROOT/'docs/model_reviews/20261106_v54/private_review_catalog.json');print(restore(entry,bytes.fromhex(value)))
 else:
  if a.root is None or a.output is None:p.error('--root and --output required for a read-only local audit')
  print(d.dumps(audit(a.root,a.output,workbook=a.workbook)))
