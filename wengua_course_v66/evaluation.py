"""Fixed known-source questions and actual model-own-history retention checks.

No reviewed model answer is a teacher. These are exposed development checks,
not new independent events. Keep full raw answers; no keyword promotion.
"""
from pathlib import Path
from copy import deepcopy
import time,gc
from wengua_course import data as d
from research_audits import v65_learning_data as prior
from research_audits.v65_learning_probe import read_cache,one_answer,append
from . import dataset as data


def critical_score(model,tok,target,book,c):
    from research_audits.v65_token_diagnostics import score
    row=score(model,tok,target,book,c)
    before=tok.apply_chat_template(target['messages'][:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    spans=target['provenance'].get('decision_spans',[])
    values=[];misses=[]
    for i,t in enumerate(row['tokens']):
        a,b=t['character_offsets']
        critical=any(a<len(before)+s['end'] and b>len(before)+s['start'] for s in spans)
        t['in_v66_critical_span']=critical
        if critical:
            values.append(t['nll'])
            if not t['gold_prefix_top1_equals_target']:misses.append(i)
    return row|{'critical_spans':spans,'critical_tokens':len(values),'critical_mean_nll':sum(values)/len(values) if values else None,
                'critical_gold_prefix_miss_indices':misses,'conditional_score_is_not_free_answer_accuracy':True}



def build(source):
    pack=prior.build(source)
    for r in pack['records']:
        c=data.contract(r['provenance']['lesson_contract'],r['messages'][-2]['content'])
        q=data.visible_question(r['messages'][-2]['content'],c);text,spans=data.answer(c)
        r['messages'][-2]['content']=q;r['messages'][-1]['content']=text;r['answer']=text;r['question']=q
        if r['task']=='training_original':r['task']='known_v65_question_revised_teacher'
        r['provenance'].update(lesson_contract=c,decision_spans=spans,exact_v65_training_messages=False,v66_visible_scope=True)
        r['components']={'current_objects_and_properties':c,'full_target':text,'automatic_professional_pass':False}
    for ep in pack['episodes']:
        for i,s in enumerate(ep['steps']):
            q=s['question'];c=deepcopy(s['lesson_contract'])
            if c['kind']=='marriage':c['palace_relation']='相生'
            s['question']=data.visible_question(q,c);s['lesson_contract']=c;s['answer']=data.answer(c)[0]
    extra=list(source.get('actual_rag_rows',[]))+[r for r in source.get('known_regression_rows',[]) if r['id'].startswith('v61_probe:ordinary_')]
    for r in extra:
        item=deepcopy(r);item['provenance']={**item.get('provenance',{}),'inference_only':True}
        item['components']={'full_target':item['answer'],'automatic_professional_pass':False};pack['records'].append(item)
    pack['schema']=data.SCHEMA;pack['summary']['static_questions']=len(pack['records']);pack['summary']['per_model_answers']=len(pack['records'])+sum(len(e['steps']) for e in pack['episodes'])
    from collections import Counter
    pack['summary']['categories']=dict(Counter(r['task'] for r in pack['records']))
    pack['weight_updates']=0;pack['new_blind_benchmark']=False;return pack


def sequence(pack,answers):
    rows=deepcopy(pack['records']);by={r['id']:r for r in answers}
    for ep in pack['episodes']:
        history=[{'role':'system','content':ep['system']}];deps=[]
        for step in ep['steps']:
            row={'id':step['id'],'messages':deepcopy(history)+[{'role':'user','content':step['question']},{'role':'assistant','content':step['answer']}],
                 'answer':step['answer'],'task':'interactive_condition_update','group':ep['id'],'source_ids':ep['source_ids'],
                 'provenance':{'lesson_contract':step['lesson_contract'],'support':ep['support'],'history_source':'model_own_previous_answers',
                    'previous_answers_sha256':list(deps),'gold_prior_answers_injected':False},'components':{'full_target':step['answer'],'automatic_professional_pass':False}}
            rows.append(row)
            if step['id'] not in by:break
            actual=by[step['id']];history.extend([{'role':'user','content':step['question']},{'role':'assistant','content':actual['answer']}]);deps.append(d.digest(actual['answer']))
    return rows


def validate_cache(saved,pack,tag):
    rows=sequence(pack,saved)
    if len(saved)>len(rows):raise ValueError('Unexpected cached extra answer')
    for a,b in zip(saved,rows):
        if (a['id']!=b['id'] or a['question']!=b['messages'][:-1] or a['expected']!=b['answer'] or a['variant']!=tag
            or a['answer_sha256']!=d.digest(a['answer']) or a['prompt_sha256']!=d.digest(d.dumps(a['question'])) or a['provenance']!=b['provenance']):
            raise ValueError('Answer source, conditions or actual history changed')
    return rows


def evaluate_adapter(p,job,c,deadline,*,best,factory=None):
    from . import runtime
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel
    from src.wengua.qwen_finetune import prepare_frozen_base
    job=Path(job);folder=job/'generations';folder.mkdir(exist_ok=True);tag=best['tag'];path=folder/(tag+'.jsonl')
    pack=d.read(job/'evaluation_pack.json');stamp=d.digest(d.dumps(pack));receipt=path.with_suffix('.receipt.json')
    saved=read_cache(path);validate_cache(saved,pack,tag);total=pack['summary']['per_model_answers'];adapter=Path(best['adapter'])
    forced=[r for r in pack['records'] if r.get('teacher_forcing')]
    token_path=folder/(tag+'.tokens.jsonl');scores=read_cache(token_path)
    from research_audits.v65_learning_probe import validate_token_rows
    validate_token_rows(scores,forced)
    files={n:d.digest((adapter/n).read_bytes()) for n in ['adapter_config.json','adapter_model.safetensors']}
    contract={'adapter_files':files,'pack_sha256':stamp,'config':c,'history_source':'model_own_previous_answers','professional_approved':False}
    if receipt.exists() and d.read(receipt)!=contract:raise ValueError('Evaluation source changed')
    if not receipt.exists():d.write(receipt,contract)
    completion=path.with_suffix('.complete.json')
    def seal():
        return {'answer_file_sha256':d.digest(path.read_bytes()),'token_file_sha256':d.digest(token_path.read_bytes()) if token_path.exists() else d.digest(b''),
                'contract_sha256':d.digest(receipt.read_bytes()),'answers':total,'critical_token_targets':len(forced),'adapter_files':files}
    if completion.exists() and d.read(completion)!=seal():raise ValueError('Completed free answers or token records changed')
    if len(saved)==total and len(scores)==len(forced):
        if not completion.exists():d.write(completion,seal())
        return {'state':'complete','completed':total,'critical_token_targets':len(scores),'cache_hit_without_model_load':True,'professional_approved':False}
    model=base=None
    try:
        if factory:tok,base,_=factory()
        else:
            tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision'],local_files_only=True,trust_remote_code=False)
            base,_=p.book.load_book_cuda(c,False)
        base=prepare_frozen_base(base,gradient_checkpointing=False);model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False)
        model.eval();model.requires_grad_(False)
        while len(saved)<total:
            if time.time()>deadline-120:return {'state':'paused','completed':len(saved),'requested':total}
            target=sequence(pack,saved)[len(saved)]
            p.book.progress('v66_answer_progress',phase=tag,answer=len(saved)+1,total=total,file=str(path))
            row=one_answer(model,tok,target,p.book,c,tag);append(path,row);saved.append(row)
        while len(scores)<len(forced):
            if time.time()>deadline-120:return {'state':'paused','completed':len(saved),'critical_token_targets':len(scores)}
            target=forced[len(scores)];p.book.progress('v66_critical_token_progress',phase=tag,target=len(scores)+1,total=len(forced))
            row=critical_score(model,tok,target,p.book,c);append(token_path,row);scores.append(row)
        if any(d.digest((adapter/n).read_bytes())!=h for n,h in files.items()):raise ValueError('Source adapter modified during inference')
        d.write(completion,seal())
        return {'state':'complete','completed':total,'requested':total,'actual_own_history_turns':sum(len(e['steps']) for e in pack['episodes']),
                'answers_sha256':d.digest(path.read_bytes()),'critical_token_targets':len(scores),'critical_token_scores_are_gold_prefix_conditional':True,'source_files_unchanged':True,'professional_approved':False}
    finally:
        model=base=None;gc.collect()
        if not factory:torch.cuda.empty_cache()
