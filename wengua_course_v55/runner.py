"""Continue reviewed V54 weights with source-aligned teaching; finish and review both rates."""
from pathlib import Path
import argparse,time,json,html,os,gc
from . import runtime,initialization,checks
from wengua_course import data as d
from wengua_course_v53 import checks as legacy_checks
from wengua_course_v54 import checks as v54_checks

def config(parent,source_run,rate):
 c=d.read(runtime.CONFIG);c.update(book_parent_snapshot=str(parent.resolve()),book_parent_provenance_sha256=d.digest((parent.parent/'parent_provenance.json').read_bytes()),legacy_source_run=str(Path(source_run).resolve()))
 c['learning_rate']=rate;return c

def evaluate_adapter(p,job,c,deadline,*,best,factory=None):
 import torch
 from transformers import AutoTokenizer
 from peft import PeftModel
 from src.wengua.qwen_finetune import prepare_frozen_base
 adapter=Path(best['adapter']);initialization.inspect(adapter,testing=p.testing);stage=best.get('eval_stage','application');tag=best['tag']
 if not tag.replace('_','').isalnum():raise ValueError('Unsafe tag')
 rows=runtime.prepared_prompts(job,stage);proof={'adapter_sha256':d.digest((adapter/'adapter_model.safetensors').read_bytes()),'stage':stage,'prompt_ids':[r['id'] for r in rows]};gen=Path(job)/'generations';gen.mkdir(exist_ok=True);receipt=gen/(tag+'.provenance.json')
 if receipt.exists() and d.read(receipt)!=proof:raise ValueError('Evaluation source/adapter changed')
 if not receipt.exists():d.write(receipt,proof)
 if factory:tok,base,_=factory()
 else:tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision'],local_files_only=True,trust_remote_code=False);base,_=p.book.load_book_cuda(c,False)
 base=prepare_frozen_base(base,gradient_checkpointing=False);model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False)
 try:return p.book.generate_rows(model,tok,rows,gen/(tag+'.jsonl'),c,deadline,tag)
 finally:
  model=base=None;gc.collect()
  if not factory:torch.cuda.empty_cache()

def report(root,state):
 root=Path(root);d.write(root/'run_status.json',state);esc=lambda x:html.escape(str(x))
 parts=['<!doctype html><meta charset="utf-8"><title>Wengua V5.5</title><style>body{font:16px system-ui;max-width:1300px;margin:24px auto;padding:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>V5.5 分步推导、领域流程与案例带练</h1><p>原V5.3/V5.4分数不改；定位、属性、方向、条件与引用分别检查。新教材继承已学权重并建立新优化器。同组两轮连续，未自动批准专业案例语义。</p><pre>'+esc(json.dumps(state,ensure_ascii=False,indent=2))+'</pre>']
 for path in sorted(root.glob('trials/*/generations/*.jsonl')):
  parts.append('<h2>'+esc(path.relative_to(root))+'</h2>')
  for line in path.read_text().splitlines():
   r=json.loads(line);parts.append('<details><summary>'+esc(r['id'])+'</summary><h3>问题</h3><pre>'+esc(json.dumps(r['question'],ensure_ascii=False))+'</pre><h3>原答</h3><pre>'+esc(r['answer'])+'</pre><h3>来源目标</h3><pre>'+esc(r['expected'])+'</pre></details>')
 (root/'results.html').write_text(''.join(parts),encoding='utf8')

def _rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]

def run(args):
 runtime.configure(args.app_root);pack=runtime.verify_data();p=runtime.profile();b=p.book;root=b.private_output(args.output);root.mkdir(parents=True,exist_ok=True)
 contract={'schema':runtime.SCHEMA,'code_sha256':runtime.hashes(),'dataset_release_sha256':d.digest(runtime.RELEASE.read_text()),'source_run':str(args.source_run.resolve()),'rates':[1e-5,2e-5],'epochs_each':2,'legacy_v53_scores_unchanged':True,'original_test_used':False}
 file=root/'sweep_contract.json'
 if file.exists():
  if not args.resume or d.read(file)!=contract:raise ValueError('Existing V55 contract preserved; do not overwrite a different course/run')
 else:d.write(file,contract)
 state=d.read(root/'run_status.json') if (root/'run_status.json').exists() else {'state':'preparing','trials':{},'professional_approval':False,'production_enabled':False}
 if state['state']=='complete_review_required':report(root,state);return 0
 deadline=time.time()+3600*args.hours
 from .process import launch
 from wengua_course_v53.checks import retention
 def evaluate(job,c,adapter,tag,stage):
  result=launch(p,job,c,deadline,'evaluate',best={'adapter':str(adapter),'tag':tag,'eval_stage':stage})
  if result['state']!='complete':raise TimeoutError('Evaluation paused with outputs preserved')
  return _rows(job/'generations'/(tag+'.jsonl'))
 try:
  with b.job_lock(root/'orchestrator.lock'):
   source=args.parent or initialization.default_source(args.source_run);parent=initialization.snapshot(root/'source_parent',source,testing=p.testing);candidates=[]
   for rate in contract['rates']:
    tag='source_lr_'+format(rate,'.0e').replace('-','m');c=config(parent,args.source_run,rate);job=p.engine.open_job(root/'trials'/tag,c,resume=True);runtime.prepare(job,c)
    state.update(state='training_or_resuming',active_trial=tag);report(root,state)
    old54_gold=runtime.choose_development(pack,128,'v54');old54_parent=evaluate(job,c,source,'parent_v54_unchanged','v54');old54_baseline=checks.score(old54_parent,old54_gold)
    old53_parent=evaluate(job,c,source,'parent_v53_unchanged','foundation');old53_baseline=legacy_checks.score(old53_parent,runtime.choose_development(pack,128,'foundation'))
    result=launch(p,job,c,deadline,'fit',probe=True)
    if result['state']!='complete':raise TimeoutError('Pressure test paused')
    result=launch(p,job,c,deadline,'fit',checkpoint=args.resume_from_checkpoint if args.resume_from_checkpoint and args.resume_from_checkpoint.resolve().is_relative_to(job.resolve()) else None)
    if result['state']!='complete':raise TimeoutError('Training or generation paused; complete checkpoint retained')
    gold=runtime.choose_development(pack,192);baseline=checks.score(_rows(job/'generations/validation_parent.jsonl'),gold);epochs=[]
    for e in result['epochs']:
     scored=checks.score(_rows(job/'generations'/f"epoch_{e['cycle']:02d}_validation.jsonl"),gold)
     legacy=evaluate(job,c,e['adapter'],f"epoch_{e['cycle']:02d}_v53_unchanged",'foundation');old=legacy_checks.score(legacy,runtime.choose_development(pack,128,'foundation'))
     old54_rows=evaluate(job,c,e['adapter'],f"epoch_{e['cycle']:02d}_v54_unchanged",'v54');old54_original=v54_checks.score(old54_rows,old54_gold);old54_strict=checks.score(old54_rows,old54_gold);old54_retention=retention(old54_strict,old54_baseline['groups'])
     rag=evaluate(job,c,e['adapter'],f"epoch_{e['cycle']:02d}_v53_actual_rag_unchanged",'legacy_rag')
     from src.wengua.book_rag_query import check_answer
     packets={r['id']:r['legacy_packet'] for r in runtime.prepared_prompts(job,'legacy_rag')};rag_checks=[{'id':r['id'],'check':check_answer(r['answer'],packets[r['id']]),'unknown_birth_year_misuse':checks.unknown_year_misuse(r['answer'],{'messages':packets[r['id']]['messages']+[{'role':'assistant','content':''}]})} for r in rag]
     preserve=retention(scored,baseline['groups']);old_direction_retention=retention(old,{'element_direction':old53_baseline['groups']['element_direction']}) if 'element_direction' in old53_baseline['groups'] else {'passed':True,'software_fixture_only':True};eligible=checks.gate(scored)['passed'] and preserve['passed'] and old54_retention['passed'] and old_direction_retention['passed'] and not any(x['unknown_birth_year_misuse'] for x in rag_checks)
     record={'epoch':e['epoch'],'steps':e['steps'],'adapter':e['adapter'],'validation_language_loss':e['validation_loss'],'source_checks':scored,'relative_retention':preserve,'original_v53_legacy_checks':old,'original_v54_legacy_checks':old54_original,'v54_with_v55_checks':old54_strict,'v54_retention':old54_retention,'old_direction_retention':old_direction_retention,'original_v53_rag_checks':rag_checks,'eligible_for_review':eligible,'professional_approval':False};epochs.append(record)
     d.write(job/'generations'/f"epoch_{e['cycle']:02d}_review.json",record)
     if eligible:candidates.append((checks.rank(scored),tag,e))
    state['trials'][tag]={'state':'complete','parent_baseline':baseline,'prior_v54_parent_baseline':old54_baseline,'prior_v53_parent_baseline':old53_baseline,'epochs':epochs};report(root,state)
   if candidates:
    _,tag,best=max(candidates,key=lambda x:(x[0],-x[2]['steps'],x[1]));adapter=best['adapter'];scope='source_rubric_passed_open_case_review_required'
   else:tag='preserved_v54_research_parent';adapter=str(source);scope='no_v55_candidate_accepted_old_weights_retained'
   d.write(root/'review_nominee.json',{'adapter':adapter,'trial':tag,'scope':scope,'professional_approval':False,'production_enabled':False})
   state.update(state='complete_review_required',active_trial=None,selected_for_review=tag,all_planned_groups_finished=True);report(root,state);return 0
 except BaseException as error:
  state.update(state='paused' if isinstance(error,(KeyboardInterrupt,TimeoutError)) else 'failed',error_type=type(error).__name__,error=str(error),outputs_preserved=True);report(root,state);raise
 finally:report(root,state)

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['restore','check','run','export','search','ask']);p.add_argument('--output',type=Path,default=runtime.ROOT/'local/course_v55/run1');p.add_argument('--source-run',type=Path,default=runtime.ROOT/'local/course_v54/run1');p.add_argument('--app-root',type=Path);p.add_argument('--parent',type=Path);p.add_argument('--hours',type=float,default=48);p.add_argument('--resume',action='store_true');p.add_argument('--resume-from-checkpoint',type=Path);p.add_argument('--key-file',type=Path);p.add_argument('--archive',type=Path);p.add_argument('--question');p.add_argument('--domain');p.add_argument('--time');p.add_argument('--zone');p.add_argument('--chart',type=Path);a=p.parse_args(argv)
 if a.command=='restore':
  import re
  from wengua_corpus.__main__ import restore
  value=a.key_file.read_text(encoding='utf-8-sig').strip() if a.key_file else input('输入此前教材64位口令（可见）：').strip()
  if not re.fullmatch('[0-9a-fA-F]{64}',value):raise ValueError('Expected the existing 64-character delivery key')
  release=d.read(runtime.RELEASE);restore({'manifest':release['delivery_manifest'],'destination':runtime.DATA.relative_to(runtime.ROOT).as_posix(),'files':release['files']},bytes.fromhex(value));runtime.verify_data();print('V5.5增补课程已恢复；原两书、V5.3/V5.4权重和结果保留。');return 0
 if a.command=='export':
  from wengua_course.runner import export_review
  print(export_review(a.output,a.archive or Path.home()/'Downloads/Wengua_V55_review.zip'));return 0
 runtime.configure(a.app_root)
 if a.command=='check':print(json.dumps(runtime.verify_data()['summary'],ensure_ascii=False,indent=2));return 0
 if not 0<a.hours<=168:p.error('hours must be in (0,168]')
 if a.resume_from_checkpoint and not a.resume_from_checkpoint.resolve().is_relative_to(a.output.resolve()):p.error('Checkpoint must belong to the current V55 output')
 os.environ.update(PYTORCH_CUDA_ALLOC_CONF=runtime.ALLOCATOR,PYTORCH_ALLOC_CONF=runtime.ALLOCATOR,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 if a.command in ('search','ask'):
  if not a.question:p.error('--question required')
  from . import corpus
  if a.command=='search':print(json.dumps(corpus.retrieve(a.question,domain=a.domain),ensure_ascii=False,indent=2));return 0
  from transformers import AutoTokenizer
  c=d.read(runtime.CONFIG);tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision'],local_files_only=True,trust_remote_code=False);packet=corpus.prompt(a.question,tok,domain=a.domain,time_text=a.time,zone=a.zone,chart_file=a.chart)
  if packet['state']!='ready':print('当前检索证据不足。');return 2
  nominee=d.read(a.output/'review_nominee.json');adapter=Path(nominee['adapter']);initialization.inspect(adapter);job=adapter.parent.parent.parent;c=d.read(job/'job_contract.json')['config']
  if not str(job.resolve()).startswith(str(a.output.resolve())):raise ValueError('No V55 candidate accepted; inspect the report before using the preserved V53 parent')
  path=a.output/'rag_prompts'/f'{time.time_ns()}.json';d.write(path,packet)
  from .process import launch
  result=launch(runtime.profile(),job,c,time.time()+3600,'ask',best={'adapter':str(adapter),'prompt':str(path)});print(result['answer'] if result['checks']['status']=='citations_checked_semantics_unverified' else '本次回答未通过引用或盘面核对，原答已保存在记录中。');print('来源与检查记录：'+result['trace_path']);return 0
 return run(a)
