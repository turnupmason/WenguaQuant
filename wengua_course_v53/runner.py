"""Finite initialization screen -> gated source applications -> cited RAG."""
from pathlib import Path
import argparse,gc,html,json,math,os,time,zipfile
from . import runtime,checks,initialization
from wengua_course import data as d

def configuration(parent,stage,rate):
 c=d.read(runtime.CONFIG);c.update(curriculum_stage=stage,learning_rate=rate,book_parent_snapshot=str(Path(parent).resolve()),book_parent_provenance_sha256=d.digest((Path(parent).parent/'parent_provenance.json').read_bytes()),initialization_only=False)
 return c

def evaluate_adapter(p,job,c,deadline,*,best,factory=None):
 import torch
 from transformers import AutoTokenizer
 from peft import PeftModel
 from src.wengua.qwen_finetune import prepare_frozen_base
 adapter=Path(best['adapter']);initialization.inspect(adapter,testing=p.testing);stage=best.get('eval_stage',c['curriculum_stage']);tag=best['tag']
 if not tag.replace('_','').isalnum():raise ValueError('Unsafe evaluation tag')
 gold=runtime.prepared_prompts(job,stage)
 receipt={'adapter_sha256':d.digest((adapter/'adapter_model.safetensors').read_bytes()),'prompt_ids':[r['id'] for r in gold],'stage':stage,'normalized_base':'frozen_norm_fp32_both_training_and_evaluation'}
 gen=Path(job)/'generations';gen.mkdir(exist_ok=True);proof=gen/(tag+'.provenance.json')
 if proof.exists() and d.read(proof)!=receipt:raise ValueError('Evaluation adapter or source changed')
 if not proof.exists():d.write(proof,receipt)
 if factory:tok,base,_=factory()
 else:
  tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision'],trust_remote_code=False,local_files_only=True);base,_=p.book.load_book_cuda(c,False)
 # Exactly the same frozen norm preparation as the fit path.
 base=prepare_frozen_base(base,gradient_checkpointing=False);model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False)
 try:return p.book.generate_rows(model,tok,gold,gen/(tag+'.jsonl'),c,deadline,tag)
 finally:
  model=base=None;gc.collect()
  if not factory:torch.cuda.empty_cache()

def score_path(path,pack,stage,count):
 rows=list(map(json.loads,Path(path).read_text().splitlines()));result=checks.score(rows,runtime.choose_development(pack,count,stage));d.write(Path(path).with_suffix('.source_checks.json'),result);return result

def report(root,state):
 root=Path(root);d.write(root/'run_status.json',state)
 esc=lambda x:html.escape(str(x));parts=['<!doctype html><meta charset="utf-8"><title>Wengua V5.3</title><style>body{font:16px system-ui;max-width:1300px;margin:24px auto;padding:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}</style><h1>V5.3 来源应用与初始化对照</h1><p>程序起盘；两书RAG提供依据；新LoRA与旧权重同课程短训比较。检查门槛不等于专业正确率，最终测试仍封存。</p><pre>'+esc(json.dumps(state,ensure_ascii=False,indent=2))+'</pre>']
 for path in sorted(root.glob('trials/*/generations/*.jsonl')):
  parts.append('<h2>'+esc(str(path.relative_to(root)))+'</h2>')
  for line in path.read_text().splitlines():
   r=json.loads(line);parts.append('<details><summary>'+esc(r['id'])+'</summary><h3>问题</h3><pre>'+esc(json.dumps(r['question'],ensure_ascii=False))+'</pre><h3>实际回答</h3><pre>'+esc(r['answer'])+'</pre><h3>来源参考</h3><pre>'+esc(r['expected'])+'</pre></details>')
 (root/'results.html').write_text(''.join(parts),encoding='utf8')

def checkpoint_argument(root,path):
 root=Path(root).resolve();cp=Path(path).resolve();job=cp.parent.parent.parent
 if not cp.is_dir() or cp.parent.name!='trainer' or not cp.name.startswith('checkpoint-') or not cp.name[11:].isdigit() or job.parent!=root/'trials':raise ValueError('Resume requires a real checkpoint in this V53 trial')
 if not job.name.startswith(('foundation_','application_lr_')) or d.read(job/'job_contract.json')['schema']!=runtime.SCHEMA:raise ValueError('Wrong checkpoint stage/version')
 status_file=cp.parent.parent/'stage_status.json';status=d.read(status_file) if status_file.exists() else {}
 if status.get('state')=='complete':raise ValueError('That stage is already complete; omit the explicit checkpoint flag')
 if int(cp.name[11:])<status.get('steps',0):raise ValueError('Older checkpoint would roll back reviewed outputs; choose the last complete checkpoint')
 if (root/'selected_foundation.json').exists() and job.name.startswith('foundation_'):raise ValueError('Foundation already selected; do not roll its optimizer back inside the application phase')
 return job,cp

def run(args):
 runtime.configure(args.app_root);pack=runtime.verify_data();p=runtime.profile();b=p.book;root=b.private_output(args.output);root.mkdir(parents=True,exist_ok=True)
 contract={'schema':runtime.SCHEMA,'code_sha256':runtime.hashes(),'dataset_sha256':d.digest(runtime.RELEASE.read_text()),'foundation_milestones':[64,128,256],'application_rates':[1e-5,2e-5],'new_books':False,'original_test_used':False}
 old=root/'sweep_contract.json'
 if old.exists():
  if not args.resume or d.read(old)!=contract:raise ValueError('Existing plan/source preserved; use the unchanged --resume plan or a new output')
 else:d.write(old,contract)
 deadline=time.time()+3600*args.hours;state=d.read(root/'run_status.json') if (root/'run_status.json').exists() else {'state':'preparing','foundation':{},'application':{},'original_test_used':False,'production_enabled':False,'professional_approval':False}
 if args.resume_from_checkpoint:
  cp_job,cp=checkpoint_argument(root,args.resume_from_checkpoint);p.engine.explicit_checkpoint(cp_job,cp)
 if state['state']=='complete_review_required':report(root,state);return 0
 from .process import launch
 from .rag import ensure_index
 ensure_index()
 def evaluate(job,c,adapter,tag,stage):
  result=launch(p,job,c,deadline,'evaluate',best={'adapter':str(adapter),'tag':tag,'eval_stage':stage})
  if result['state']!='complete':raise TimeoutError('Evaluation paused at the run budget; outputs preserved')
  return score_path(job/'generations'/(tag+'.jsonl'),pack,stage,c['development_generation_rows'])
 def checkpoint(job):
  cp=args.resume_from_checkpoint
  return cp if cp and Path(cp).resolve().is_relative_to(job.resolve()) else None
 def fit_until(job,c,stop):
  status=job/'cycle_01/stage_status.json'
  if status.exists():
   old=d.read(status)
   if old.get('state')=='complete' or old.get('steps',0)>=stop and old.get('state')=='paused_training':
    b.checked_adapter(old['last_adapter']);return old
  return launch(p,job,c,deadline,'fit',stop_step=stop,checkpoint=checkpoint(job))
 try:
  with b.job_lock(root/'orchestrator.lock'):
   selected=root/'selected_foundation.json'
   if not selected.exists():
    c=d.read(runtime.CONFIG)|{'initialization_only':True,'initialization_label':'official_new_lora_no_V4_V51'}
    init=p.engine.open_job(root/'trials/official_initialization',c,resume=True);fresh=launch(p,init,c,deadline,'initialize')['adapter'];initializers=[('official_fresh',fresh)]
    # Preserve previous progress as a comparison; do not require its existence.
    nominee=runtime.ROOT/'local/course_v52/run2/review_nominee.json'
    if nominee.exists():
     previous=Path(d.read(nominee)['adapter'])
     try:initialization.inspect(previous,testing=p.testing);initializers.append(('v52_comparison',str(previous)))
     except (OSError,ValueError,KeyError) as e:d.write(root/'previous_adapter_not_adopted.json',{'reason':str(e),'old_weights_modified':False})
    scored=[]
    for name,source in initializers:
     parent=initialization.snapshot(root/'initializers'/name,source,testing=p.testing);c=configuration(parent,'foundation',2e-5);c['initialization_label']=name
     job=p.engine.open_job(root/'trials'/('foundation_'+name),c,resume=True);meta=runtime.prepare(job,c);planned=math.ceil(meta['training_rows']/c['gradient_accumulation_steps'])*c['num_train_epochs'];stop=min(64,planned)
     state.update(state='foundation_short_comparison',active_trial=job.name);report(root,state)
     pressure=launch(p,job,c,deadline,'fit',probe=True)
     if pressure['state']!='complete':raise TimeoutError('Pressure run paused; preserve output')
     result=fit_until(job,c,stop)
     if not result.get('last_adapter'):raise TimeoutError('Foundation paused before first saved update')
     summary=evaluate(job,c,result['last_adapter'],'foundation_screen_'+str(result['steps']),'foundation');gate=checks.gate(summary,'foundation')
     state['foundation'][name]={'steps':result['steps'],'score':summary['groups'],'gate':gate,'adapter':result['last_adapter']};report(root,state)
     scored.append((checks.rank(summary),name,job,c,result,summary))
    # Same curriculum and LR for initializer comparison. A tie prefers the official fresh route.
    _,name,job,c,result,summary=max(scored,key=lambda x:(x[0],x[1]=='official_fresh'))
    for stop in (128,256):
     if checks.gate(summary,'foundation')['passed']:break
     if result['state']=='complete':break
     state.update(state='foundation_bounded_repair',active_trial=job.name);report(root,state)
     result=fit_until(job,c,stop);summary=evaluate(job,c,result['last_adapter'],'foundation_screen_'+str(result['steps']),'foundation')
     state['foundation'][name]={'steps':result['steps'],'score':summary['groups'],'gate':checks.gate(summary,'foundation'),'adapter':result['last_adapter']};report(root,state)
    actual=launch(p,job,c,deadline,'rag_probe',best={'adapter':result['last_adapter'],'tag':'foundation_final_actual_rag_'+str(result['steps'])})
    if actual['state'] not in ('complete','software_fixture_skipped_real_book_probe'):raise TimeoutError('Foundation real RAG check paused')
    state['foundation'][name]['actual_rag_probe']={'state':actual['state'],'executed':actual['real_book_probe_executed'],'semantic_review_required':True};report(root,state)
    if not checks.gate(summary,'foundation')['passed']:
     state.update(state='needs_course_review',active_trial=None,reason='Source application or elementary directions below the declared screen; long training not started')
     report(root,state);return 2
    parent=initialization.snapshot(root/'foundation_selected',result['last_adapter'],testing=p.testing)
    d.write(selected,{'route':name,'adapter':str(parent),'source_adapter':result['last_adapter'],'steps':result['steps'],'gate':checks.gate(summary,'foundation'),'score':summary['groups'],'optimizer_reused_in_next_stage':False})
   parent=Path(d.read(selected)['adapter']);candidates=[]
   for rate in contract['application_rates']:
    tag='application_lr_'+format(rate,'.0e').replace('-','m');c=configuration(parent,'application',rate);job=p.engine.open_job(root/'trials'/tag,c,resume=True);runtime.prepare(job,c)
    state.update(state='source_and_chart_application',active_trial=tag);report(root,state)
    pressure=launch(p,job,c,deadline,'fit',probe=True)
    if pressure['state']!='complete':raise TimeoutError('Application pressure paused')
    result=launch(p,job,c,deadline,'fit',checkpoint=checkpoint(job))
    if result['state']!='complete':raise TimeoutError('Application training paused with full checkpoint')
    epochs=[]
    baseline=score_path(job/'generations/validation_parent.jsonl',pack,'application',c['development_generation_rows'])
    for e in result['epochs']:
     score=score_path(job/'generations'/f"epoch_{e['cycle']:02d}_validation.jsonl",pack,'application',c['development_generation_rows'])
     retained=evaluate(job,c,e['adapter'],f"epoch_{e['cycle']:02d}_foundation_regression",'foundation')
     rag_result=launch(p,job,c,deadline,'rag_probe',best={'adapter':e['adapter'],'tag':f"epoch_{e['cycle']:02d}_actual_rag"})
     if rag_result['state'] not in ('complete','software_fixture_skipped_real_book_probe'):raise TimeoutError('Actual RAG probe paused')
     allowed=checks.gate(score,'application')['passed'] and checks.gate(retained,'foundation')['passed'] and rag_operations_ok(rag_result,software_fixture=p.testing)
     row={'epoch':e['epoch'],'steps':e['steps'],'adapter':e['adapter'],'language_loss':e['validation_loss'],'application_checks':score['groups'],'foundation_checks':retained['groups'],'actual_rag_probe':{'state':rag_result['state'],'executed':rag_result['real_book_probe_executed'],'semantic_review_required':True},'eligible_for_review':allowed,'professional_approval':False};epochs.append(row)
     if allowed:candidates.append((checks.rank(score),tag,row))
    state['application'][tag]={'state':'complete','epochs':epochs,'foundation_before_application':baseline['groups']};report(root,state)
   if candidates:
    _,tag,row=max(candidates,key=lambda x:(x[0],-x[2]['steps'],x[1]));adapter=row['adapter'];scope='both_source_and_foundation_screens_passed_case_review_pending'
   else:tag='foundation_only';adapter=d.read(selected)['source_adapter'];scope='application_not_accepted_keep_foundation_only'
   d.write(root/'review_nominee.json',{'adapter':adapter,'route':d.read(selected)['route'],'trial':tag,'scope':scope,'professional_approval':False,'production_enabled':False,'old_weights_modified':False})
   state.update(state='complete_review_required',active_trial=None,selected_for_review=tag,automatic_retraining=False);report(root,state);return 0
 except BaseException as e:
  state.update(state='paused' if isinstance(e,(KeyboardInterrupt,TimeoutError)) else 'failed',error_type=type(e).__name__,error=str(e),outputs_preserved=True);report(root,state);raise
 finally:report(root,state)

def rag_operations_ok(result,*,software_fixture=False):
 if software_fixture:return result.get('state')=='software_fixture_skipped_real_book_probe'
 if result.get('state')!='complete' or not result.get('real_book_probe_executed'):return False
 rows=result.get('citation_and_chart_checks',[]);coverage=list(result.get('source_coverage',{}).values())
 return bool(rows and coverage) and sum(r['checks']['status']=='citations_checked_semantics_unverified' and not r['reached_token_limit'] for r in rows)/len(rows)>=.8 and sum(r['all_declared_source_spans_in_visible_prompt'] for r in coverage)/len(coverage)>=.8

def export(root,archive):
 from wengua_course.runner import export_review
 return export_review(root,archive)

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['restore','check','run','export','search','ask']);parser.add_argument('--app-root',type=Path);parser.add_argument('--output',type=Path,default=runtime.ROOT/'local/course_v53/run1');parser.add_argument('--hours',type=float,default=48);parser.add_argument('--resume',action='store_true');parser.add_argument('--resume-from-checkpoint',type=Path);parser.add_argument('--archive',type=Path);parser.add_argument('--key-file',type=Path);parser.add_argument('--question');parser.add_argument('--domain');parser.add_argument('--book',choices=['shenqi','kaiwu']);parser.add_argument('--mode',choices=['knowledge','reading','study'],default='knowledge');parser.add_argument('--case-id');parser.add_argument('--time');parser.add_argument('--zone');parser.add_argument('--chart',type=Path);parser.add_argument('--facts',type=Path);args=parser.parse_args(argv)
 if args.command=='restore':
  from wengua_corpus.__main__ import restore
  release=d.read(runtime.RELEASE);value=args.key_file.read_text().strip() if args.key_file else input('粘贴此前V4/V5.1教材的64位口令（输入可见）：').strip()
  import re
  if not re.fullmatch('[0-9a-fA-F]{64}',value):raise ValueError('Use the existing 64-character delivery key, not a ZIP path')
  restore({'manifest':release['delivery_manifest'],'destination':runtime.DATA.relative_to(runtime.ROOT).as_posix(),'files':release['files']},bytes.fromhex(value));runtime.verify_data();print('V5.3教材、审阅表和两书RAG已恢复并校验。');return 0
 if args.command=='export':print(export(args.output,args.archive or Path.home()/'Downloads/Wengua_V53_review.zip'));return 0
 runtime.configure(args.app_root);pack=runtime.verify_data()
 if args.command=='check':print(json.dumps(pack['summary'],ensure_ascii=False,indent=2));return 0
 os.environ.update(PYTORCH_CUDA_ALLOC_CONF=runtime.ALLOCATOR,PYTORCH_ALLOC_CONF=runtime.ALLOCATOR,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 if args.command in ('search','ask'):
  if not args.question:parser.error('--question required')
  from . import rag
  if args.command=='search':print(json.dumps(rag.retrieve(args.question,domain=args.domain,mode=args.mode,case_id=args.case_id,book=args.book),ensure_ascii=False,indent=2));return 0
  from transformers import AutoTokenizer
  c=d.read(runtime.CONFIG);tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision'],local_files_only=True,trust_remote_code=False);prompt=rag.prompt(args.question,tok,domain=args.domain,mode=args.mode,case_id=args.case_id,time_text=args.time,zone=args.zone,chart_file=args.chart,facts=d.read(args.facts) if args.facts else None,book=args.book)
  if prompt['state']!='ready':print('检索材料不足，请查看检索结果并补充问题条件。');return 2
  choice=d.read(args.output/'review_nominee.json');adapter=Path(choice['adapter']);initialization.inspect(adapter);c=d.read(adapter.parent.parent.parent/'job_contract.json')['config'];job=adapter.parent.parent.parent;p=runtime.profile();path=args.output/'rag_prompts'/f'{time.time_ns()}.json';d.write(path,prompt)
  from .process import launch
  result=launch(p,job,c,time.time()+3600,'ask',best={'adapter':str(adapter),'prompt':str(path)});print(result['answer'] if result['checks']['status']=='citations_checked_semantics_unverified' else '本次回答未通过引用或盘面检查，原答已保留在核对记录中。');print('来源与核对记录：'+result['trace_path']);return 0
 if not 0<args.hours<=168:parser.error('--hours must be >0 and <=168')
 if args.resume_from_checkpoint and not args.resume_from_checkpoint.resolve().is_relative_to(args.output.resolve()):parser.error('Checkpoint must belong to this V53 run')
 return run(args)
