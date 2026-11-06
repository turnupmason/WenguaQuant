"""Finite V5.2 repair study: compare sealed parents, then two learning rates.

Model selection uses explicitly limited source-seen constraint checks; full
case reasoning stays review-required. No automatic production promotion.
"""
from pathlib import Path
import argparse,gc,html,json,os,shutil,time,zipfile
from . import data,runtime,checks,parents

def config_for(parent,rate):
 c=data.read(runtime.CONFIG);proof=data.read(Path(parent).parent/'parent_provenance.json')
 for key in ['model_id','revision','lora_r','lora_alpha','lora_dropout','target_modules']:c[key]=proof['parent_config'][key]
 c.update(learning_rate=rate,book_parent_snapshot=str(Path(parent).resolve()),book_parent_provenance_sha256=data.digest((Path(parent).parent/'parent_provenance.json').read_bytes()))
 return c

def evaluate_parent(p,job,c,deadline,*,factory=None):
 import torch
 from transformers import AutoTokenizer
 from peft import PeftModel
 if factory:tok,base,_=factory()
 else:
  tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision'],trust_remote_code=False,local_files_only=True)
  base,_=p.book.load_book_cuda(c,False)
 model=None
 try:
  model=PeftModel.from_pretrained(base,str(runtime.verify_snapshot(c)),is_trainable=False)
  prompts=p.book.read_rows(Path(job)/'prepared/validation_prompts.jsonl.gz')
  return p.book.generate_rows(model,tok,prompts,Path(job)/'generations/parent_comparison.jsonl',c,deadline,'parent_comparison')
 finally:
  model=base=None;gc.collect()
  if not factory:torch.cuda.empty_cache()

def report(root,state,pack=None):
 root=Path(root);state=state|{'production_enabled':False,'professional_semantic_approval':False,'original_test_run':False}
 data.write(root/'run_status.json',state)
 esc=lambda x:html.escape(str(x));parts=['<!doctype html><meta charset="utf-8"><title>Wengua V5.2</title><style>body{font:16px system-ui;max-width:1400px;margin:24px auto;padding:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:10px;vertical-align:top}summary{cursor:pointer}</style><h1>V5.2 两书条件课程</h1><p>事实约束检查与整案人工审阅分开；本报告不批准独立预测、现实概率或上线。已知古籍/旧权重接触不被称作新盲测。</p><pre>'+esc(json.dumps(state,ensure_ascii=False,indent=2))+'</pre>']
 if pack is not None:
  gold={r['id']:r for r in runtime.choose_development(pack)}
  for path in sorted(root.glob('trials/*/generations/*.jsonl')):
   if 'seen_diagnostic' in path.name:continue
   parts.append('<h2>'+esc(str(path.relative_to(root)))+'</h2>')
   for record in map(json.loads,path.read_text(encoding='utf8').splitlines()):
    g=gold.get(record['id']);parts.append('<details><summary>'+esc(record['id']+' / '+record['task'])+'</summary><h3>问题</h3><pre>'+esc(json.dumps(record['question'],ensure_ascii=False))+'</pre><h3>模型实答</h3><pre>'+esc(record['answer'])+'</pre><h3>来源参考</h3><pre>'+esc(record['expected'])+'</pre><p>来源：'+esc(', '.join(record['source_ids']))+'</p><pre>'+esc(checks.evaluate(record['answer'],g.get('checks') if g else None,capped=record.get('reached_token_limit',False)))+'</pre></details>')
 (root/'results.html').write_text(''.join(parts),encoding='utf8')

def score_file(path,prompts):
 path=Path(path);rows=[json.loads(s) for s in path.read_text(encoding='utf8').splitlines()]
 result=checks.score(rows,prompts);data.write(path.with_suffix('.constraints.json'),result);return result

def run(args):
 runtime.configure(args.app_root);pack=runtime.verify_data()
 p=runtime.profile();b=p.book;root=b.private_output(args.output);root.mkdir(parents=True,exist_ok=True)
 if (root/'run_status.json').exists() and not args.resume:raise ValueError('Existing run preserved; use --resume')
 contract={'schema':runtime.SCHEMA,'code_sha256':runtime.hashes(),'data_release_sha256':data.digest(runtime.RELEASE.read_text(encoding='utf8')),'finite_rates':[5e-6,1e-5],'epochs_per_rate':2,'selection':'declared_constraint_development_not_language_loss_or_professional_approval','original_test_used':False}
 path=root/'sweep_contract.json'
 if path.exists() and data.read(path)!=contract:raise ValueError('V5.2 code/data/plan changed; keep old output and use a new output directory')
 if not path.exists():data.write(path,contract)
 if args.resume_from_checkpoint:
  cp=args.resume_from_checkpoint.resolve()
  allowed=[root/'trials'/('lr_'+format(rate,'.0e').replace('-','m')) for rate in (5e-6,1e-5)]
  if not cp.is_dir() or cp.parent.name!='trainer' or not cp.name.startswith('checkpoint-') or not any(cp.is_relative_to(p) for p in allowed):raise ValueError('Resume checkpoint must be a real checkpoint in one rate of this V5.2 run')
 deadline=time.time()+args.hours*3600;state={'state':'preparing','baselines':{},'trials':{}}
 if (root/'run_status.json').exists():state=data.read(root/'run_status.json')
 from .process import launch
 try:
  selected_path=root/'selected_initialization.json'
  if selected_path.exists() and args.parent and str(args.parent.resolve())!=data.read(selected_path)['source_path']:raise ValueError('Parent already fixed for this run; preserve it and use a new output to change parent')
  if not selected_path.exists():
   candidates,failed=parents.candidates();data.write(root/'rejected_parent_candidates.json',failed)
   if args.parent:candidates=[(args.parent,parents.inspect(args.parent))]
   if not candidates:raise FileNotFoundError('No intact sealed r16 V4/V5.1 epoch weights found in WenguaApp local; do not use a smoke adapter')
   nominees=[]
   for src,proof in candidates:
    tag=('v51_' if proof['schema'].endswith('v51') else 'v4_')+src.name+'_'+data.digest(data.dumps(proof['files']))[:8]
    parent=parents.snapshot(root/'parents'/tag,src,proof);c=config_for(parent,5e-6);job=p.engine.open_job(root/'trials'/('baseline_'+tag),c,resume=True)
    runtime.prepare(job,c);(job/'generations').mkdir(exist_ok=True)
    state.update(state='comparing_existing_book_weights',active_trial=job.name);report(root,state,pack)
    result=launch(p,job,c,deadline,'evaluate')
    if result['state']!='complete':return 2
    prompts=b.read_rows(job/'prepared/validation_prompts.jsonl.gz');scored=score_file(job/'generations/parent_comparison.jsonl',prompts)
    state['baselines'][tag]={'constraint_macro':scored['topic_macro_constraint_pass_rate'],'objective_constraints_only':True,'open_case_reviews':scored['unreviewed_case_reasoning']}
    nominees.append((checks.rank(scored),tag,src,proof));report(root,state,pack)
   # Ties stay deterministic. This chooses a repair starting point, not a certified master.
   _,tag,src,proof=max(nominees,key=lambda t:(t[0],t[1]))
   chosen=parents.snapshot(root,src,proof)
   data.write(selected_path,{'tag':tag,'source_path':str(src),'snapshot':str(chosen),'selection_scope':'limited_source_seen_constraints_only','professional_approval':False,'parent_files':proof['files']})
  chosen=Path(data.read(selected_path)['snapshot'])
  for rate in contract['finite_rates']:
   tag='lr_'+format(rate,'.0e').replace('-','m');c=config_for(chosen,rate);job=p.engine.open_job(root/'trials'/tag,c,resume=True);runtime.prepare(job,c)
   state.update(state='pressure_then_training',active_trial=tag);report(root,state,pack)
   probe=launch(p,job,c,deadline,'fit',probe=True)
   if probe['state']!='complete':return 2
   checkpoint=args.resume_from_checkpoint if args.resume_from_checkpoint and Path(args.resume_from_checkpoint).resolve().is_relative_to(job.resolve()) else None
   result=launch(p,job,c,deadline,'fit',checkpoint=checkpoint)
   data.write(job/'language_loss_phase_status.json',result)
   if result['state']!='complete':state['trials'][tag]=result;return 2
   prompts=b.read_rows(job/'prepared/validation_prompts.jsonl.gz');epochs=[]
   for epoch in result['epochs']:
    f=job/'generations'/f"epoch_{epoch['cycle']:02d}_validation.jsonl";scored=score_file(f,prompts)
    epochs.append({'epoch':epoch['epoch'],'steps':epoch['steps'],'adapter':epoch['adapter'],'language_loss':epoch['validation_loss'],'constraint_macro':scored['topic_macro_constraint_pass_rate'],'open_case_reviews':scored['unreviewed_case_reasoning'],'constraint_result_path':str(f.with_suffix('.constraints.json')),'professional_approval':False})
   state['trials'][tag]={'state':'complete','epochs':epochs,'parent_weights_reused':True,'new_phase_optimizer':True,'optimizer_reset_between_epochs':False};report(root,state,pack)
  epoch_candidates=[(tag,epoch) for tag,trial in state['trials'].items() for epoch in trial.get('epochs',[])]
  tag,nominee=max(epoch_candidates,key=lambda item:(item[1]['constraint_macro'],-item[1]['steps'],item[0]))
  initial=data.read(selected_path);parent_constraint=state['baselines'][initial['tag']]['constraint_macro']
  data.write(root/'review_nominee.json',{'adapter':nominee['adapter'] if nominee['constraint_macro']>=parent_constraint else str(chosen),'trial':tag,'selection_scope':'declared_source_seen_factual_constraints_only','parent_constraint_macro':parent_constraint,'candidate_constraint_macro':nominee['constraint_macro'],'case_reasoning_requires_review':True,'professional_approval':False,'production_enabled':False})
  state.update(state='complete_review_required',active_trial=None,auto_training_repeated=False)
  # Keep every candidate; no overwrite / merge / deployment of a numeric winner.
  report(root,state,pack);return 0
 except BaseException as error:
  state.update(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error_type=type(error).__name__,error=str(error),all_outputs_preserved=True)
  report(root,state,pack);raise
 finally:report(root,state,pack)

def export_review(root,out):
 root=Path(root).resolve();out=Path(out).resolve();out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists():raise FileExistsError('Existing review archive preserved')
 allowed={'.json','.jsonl','.html','.csv','.tsv','.log'};files=[]
 for p in root.rglob('*'):
  if not p.is_file() or p.suffix not in allowed:continue
  if any(part in ('prepared','parents','parent_adapter','trainer','adapters') for part in p.relative_to(root).parts):continue
  files.append(p)
 with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for p in sorted(files):z.write(p,p.relative_to(root).as_posix())
 return out

def main(argv=None):
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('command',choices=['restore','check','run','export']);a.add_argument('--app-root',type=Path);a.add_argument('--output',type=Path,default=runtime.ROOT/'local/course_v52/run2');a.add_argument('--hours',type=float,default=48);a.add_argument('--resume',action='store_true');a.add_argument('--resume-from-checkpoint',type=Path);a.add_argument('--parent',type=Path);a.add_argument('--archive',type=Path);a.add_argument('--key-file',type=Path);args=a.parse_args(argv)
 if args.command=='restore':
  from wengua_corpus.__main__ import restore
  release=data.read(runtime.RELEASE);value=args.key_file.read_text(encoding='utf-8-sig').strip() if args.key_file else input('粘贴此前V4/V5.1教材的64位口令（输入可见）：').strip()
  import re
  if not re.fullmatch('[0-9a-fA-F]{64}',value):raise ValueError('请使用此前教材的64位十六进制口令，不要输入ZIP文件路径')
  restore({'manifest':release['delivery_manifest'],'destination':runtime.DATA.relative_to(runtime.ROOT).as_posix(),'files':release['files']},bytes.fromhex(value));runtime.verify_data();print('V5.2课程、审阅Excel与两书课程数据库已恢复并校验。');return 0
 if args.command=='export':print(export_review(args.output,args.archive or Path.home()/'Downloads/Wengua_V52_review.zip'));return 0
 runtime.configure(args.app_root);pack=runtime.verify_data()
 if not 0<args.hours<=168:a.error('hours must be >0 and <=168')
 if args.command=='check':print(json.dumps(pack['summary'],ensure_ascii=False,indent=2));return 0
 os.environ['PYTORCH_CUDA_ALLOC_CONF']=runtime.ALLOCATOR;os.environ['PYTORCH_ALLOC_CONF']=runtime.ALLOCATOR
 # No downloads should be necessary after the completed user 8B run.
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 return run(args)
