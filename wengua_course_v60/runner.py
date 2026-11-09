"""Finite same-parent learning-rate comparison with source/condition retention."""
from pathlib import Path
from argparse import ArgumentParser
import time,json,os,gc,html,re
from wengua_course import data as d
from . import runtime,initialization,checks,evaluation

def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf8").splitlines() if x.strip()]

def evaluate_adapter(p,job,c,deadline,*,best,factory=None):
    official=best.get('official_base') is True
    adapter=None if official else Path(best['adapter'])
    if adapter is not None:initialization.inspect(adapter,testing=p.testing)
    stage=best.get('eval_stage','application');tag=best['tag']
    if not tag.replace('_','').isalnum():raise ValueError('Unsafe evaluation tag')
    rows=runtime.prepared_prompts(job,stage);inference=c|{'eval_max_new_tokens':best.get('max_new_tokens',c['eval_max_new_tokens'])};path=Path(job)/'generations'/(tag+'.jsonl');path.parent.mkdir(exist_ok=True)
    proof={'adapter_sha256':d.digest((adapter/'adapter_model.safetensors').read_bytes()) if adapter else None,'variant':'official_base' if official else 'inherited_adapter','model_id':c['model_id'],'revision':c['revision'],'prompt_sha256':d.digest(d.dumps([r['messages'] for r in rows])),'ids':[r['id'] for r in rows],'max_new_tokens':inference['eval_max_new_tokens'],'eval_stage':stage}
    receipt=path.with_suffix('.provenance.json')
    if receipt.exists() and d.read(receipt)!=proof:raise ValueError('Evaluation source, weights or token budget changed')
    if path.exists() and not receipt.exists():
        reuse=path.with_suffix('.reuse.json')
        if not reuse.exists():raise ValueError('Cached answers lack sealed generation or reuse provenance')
        reused=d.read(reuse)
        if reused['adapter_sha256']!=proof['adapter_sha256'] or reused['max_new_tokens']!=proof['max_new_tokens'] or reused['target_sha256']!=d.digest(path.read_bytes()):raise ValueError('Reused answer weights, budget or bytes changed')
    d.write(receipt,proof)
    if path.exists():
        saved=read_rows(path);gold={r['id']:r for r in rows}
        for r in saved:
            if r['id'] not in gold or r['question']!=gold[r['id']]['messages'][:-1] or r['expected']!=gold[r['id']]['answer']:raise ValueError('Cached question/target changed')
        if len(saved)==len(rows) and len({r['id'] for r in saved})==len(rows):return {'state':'complete','requested':len(rows),'completed':len(rows),'cache_hit_without_model_load':True}
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel
    from src.wengua.qwen_finetune import prepare_frozen_base
    if factory:tok,base,_=factory()
    else:tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision'],local_files_only=True,trust_remote_code=False);base,_=p.book.load_book_cuda(c,False)
    base=prepare_frozen_base(base,gradient_checkpointing=False);model=base if official else PeftModel.from_pretrained(base,str(adapter),is_trainable=False)
    try:return p.book.generate_rows(model,tok,rows,path,inference,deadline,tag)
    finally:
        model=base=None;gc.collect()
        if not factory:torch.cuda.empty_cache()


def config(parent,root,phase):
    c=d.read(runtime.CONFIG);c.update(course_phase=phase,num_train_epochs=1,book_parent_snapshot=str(parent.resolve()),book_parent_provenance_sha256=d.digest((parent.parent/'parent_provenance.json').read_bytes()),legacy_source_run=str(Path(root).resolve()))
    return c


def report(root,state):
    root=Path(root);d.write(root/'run_status.json',state);parts=['<!doctype html><meta charset="utf-8"><title>Wengua V6.0</title><style>body{font:16px system-ui;max-width:1250px;margin:24px auto;padding:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:12px 0}</style><h1>V6.0 来源完整、条件对照与分层训练</h1><p>试训同时包含基础、领域与作者案例；小套核对用于训练路径和能力留存，最终门槛与开放案例语义审阅分开。原资料、旧成绩和权重保留。</p><details open><summary>阶段状态与核对</summary><pre>'+html.escape(d.dumps(state))+'</pre></details>']
    for path in sorted(root.glob('trials/*/generations/*.jsonl')):
        parts.append('<details><summary>'+html.escape(str(path.relative_to(root)))+'</summary>')
        for r in read_rows(path):parts.append('<details><summary>'+html.escape(r['id'])+'</summary><h3>问题</h3><pre>'+html.escape(d.dumps(r['question']))+'</pre><h3>原答</h3><pre>'+html.escape(r['answer'])+'</pre><h3>来源目标</h3><pre>'+html.escape(r['expected'])+'</pre></details>')
        parts.append('</details>')
    (root/'results.html').write_text(''.join(parts),encoding='utf8')


def learning_gate(current,before,rag,rag_before):
    """Itemwise protection: gains elsewhere cannot pay for a lost known answer."""
    failures=[];improved=[]
    for label,cur,old in [('micro',current,before),('rag',rag,rag_before)]:
        if not cur['complete'] or not old['complete']:failures.append(label+':missing_answers')
        prior={r['id']:r for r in old['items']};present={r['id']:r for r in cur['items']}
        if set(prior)!=set(present):failures.append(label+':changed_tasks');continue
        for rid,row in present.items():
            previous=prior[rid];fact=row['fact'];was=previous['fact']
            if was['status']=='pass' and fact['status']!='pass':failures.append(rid+':lost_known_fact')
            if was.get('complete') and not fact.get('complete'):failures.append(rid+':lost_known_conditions')
            if previous['citation']['ok'] and not row['citation']['ok']:failures.append(rid+':lost_source_coverage')
            if fact['reason'] in ['empty','output_limit','repeated_output']:failures.append(rid+':bad_generation')
            if fact.get('complete') and not was.get('complete'):improved.append(rid)
    if not improved:failures.append('no_new_complete_fact_or_condition')
    return {'passed':not failures,'failed':failures,'new_complete_items':improved,'scope':'paired_itemwise_training_retention_not_unseen_or_professional_guarantee','professional_approval':False}


def readiness(current,rag):
    base=checks.acceptance(current,current)
    return {'passed':base['passed'] and rag['complete'] and all(r['fact']['status']=='pass' and r['fact']['complete'] for r in rag['items']),'fact_gate':base,'open_case_semantics_approved':False,'production_enabled':False}


def rank(summary):
    groups=list(summary['groups'].values());return (sum(g['pass'] for g in groups)/max(1,sum(g['total'] for g in groups)),sum(g['complete'] for g in groups),summary['citation']['rate'] or 0)


def copy_answers(source,target,gold,*,adapter_sha,scope,max_new_tokens=768):
    """Reuse only identical questions, targets, budgets and sealed adapter."""
    source,target=Path(source),Path(target)
    if target.exists():return
    rows=read_rows(source);byid={r['id']:r for r in gold};valid=[]
    for r in rows:
        if r['id'] in byid:
            g=byid[r['id']]
            if r['question']!=g['messages'][:-1] or r['expected']!=g['answer']:raise ValueError('Cannot reuse changed question')
            valid.append(r)
    target.parent.mkdir(exist_ok=True)
    with target.open('w',encoding='utf8') as f:
        for r in valid:f.write(d.dumps(r)+'\n')
    d.write(target.with_suffix('.reuse.json'),{'source_file':str(source),'source_sha256':d.digest(source.read_bytes()),'adapter_sha256':adapter_sha,'scope':scope,'rows_reused':len(valid),'max_new_tokens':max_new_tokens,'target_sha256':d.digest(target.read_bytes()),'not_training_input':True})
