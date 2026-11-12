"""Retain the exact V66 questions, add paired object and condition monitors.

Actual model-own-history answers are used in all interactive evaluations.
Known source questions are development checks, not blind outcome cases.
"""
from pathlib import Path
from copy import deepcopy
import gc,time
from wengua_course import data as d
from wengua_course_v66 import evaluation as previous
from research_audits.v65_learning_probe import one_answer,read_cache,append
from . import dataset as data


def build(course,source65):
    retention=previous.build(source65);records=deepcopy([r for r in course['records'] if r['split']=='validation'])
    for r in records:r['components']={'source_contract':r['provenance']['lesson_contract'],'automatic_professional_pass':False}
    boards={b['base_id']:b for b in course['physical_star_boards']};episodes=[]
    def episode(name,steps,anchor):
        episodes.append({'id':name,'system':data.SYSTEM,'source_ids':anchor['source_ids'],'support':anchor['provenance']['support'],
          'steps':[{'id':f'v67eval:{name}:{i}','question':data.visible_question(c,q),'answer':data.answer(c,boards),'lesson_contract':c} for i,(q,c) in enumerate(steps)],
          'history_source':'model_own_previous_answers','gold_prior_answers_injected':False})
    anchor=next(r for r in records if r['skill']=='independent_conditions')
    first=[]
    for i,rel in enumerate(['相生','相克','未提供','比和']):
        c={'kind':'independent_conditions','palace_relation':rel,'gate_state':'已具备'}
        q=data.condition_question(c,development=True) if not i else f'只更新两宫关系为“{rel}”，吉门吉格仍已具备。现在本条可成条件怎样？'
        first.append((q,c))
    episode('first_condition_changes',first,anchor)
    steps=[]
    states=[('相克','已具备'),('相克','明确不具备'),('相生','明确不具备'),('相生','已具备')]
    for i,(rel,gate) in enumerate(states):
        c={'kind':'independent_conditions','palace_relation':rel,'gate_state':gate}
        q=data.condition_question(c,development=True) if not i else f'只按当前新资料：两宫关系“{rel}”，吉门吉格“{gate}”。分别核实两项后，这条分支能否使用？'
        steps.append((q,c))
    episode('which_condition_failed',steps,anchor)
    anchor=next(r for r in records if r['skill']=='typed_objects');board=boards[anchor['provenance']['lesson_contract']['board_id']];steps=[]
    for i,goal in enumerate(['stars','palaces','heart_to_palace','reverse']):
        c={'kind':'typed_objects','board_id':board['base_id'],'goal':goal,'facts':data.object_facts(board)}
        q=data.object_question(board,goal,'full',development=True) if not i else '所有位置不变；现在换比较对象。'+data.object_question(board,goal,'typed').split('\n')[-1]
        steps.append((q,c))
    episode('object_layers_change',steps,anchor)
    anchor=next(r for r in records if r['skill']=='identity_referent');f={'年干':'甲','月干':'丁','日干':'庚','时干':'壬'}
    c={'kind':'identity_referent','role':'同事','fields':f,'birth':'癸'}
    episode('same_patient_reference',[(data.old.ROLE_PHRASE['同事']+'。患者就是这位同事。'+d.dumps(f)+'，可靠出生年干癸。主用神怎么取？',c),
       ('仍是同一个患者，现在这位同事亲自问自己的健康，不再由别人代问。起局字段和出生资料都不变，主用神如何处理？',c|{'role':'本人'})],anchor)
    retained=retention['summary']['per_model_answers']
    new={'records':records,'episodes':episodes};total=len(records)+sum(len(e['steps']) for e in episodes)
    return {'schema':data.SCHEMA,'retention':retention,'new':new,'summary':{'retention_answers':retained,'new_static_questions':len(records),
      'new_own_history_turns':sum(len(e['steps']) for e in episodes),'new_answers':total,'per_model_answers':retained+total},
      'source_models_used_as_teachers':False,'new_blind_benchmark':False,'original_test_used':False}


def sequence(pack,answers):
    old_count=pack['summary']['retention_answers'];old_answers=answers[:old_count]
    old=previous.sequence(pack['retention'],old_answers)
    if len(old_answers)<old_count:return old
    rows=old+deepcopy(pack['new']['records']);by={r['id']:r for r in answers[old_count:]}
    for ep in pack['new']['episodes']:
        history=[{'role':'system','content':ep['system']}];deps=[]
        for step in ep['steps']:
            target={'id':step['id'],'messages':deepcopy(history)+[{'role':'user','content':step['question']},{'role':'assistant','content':step['answer']}],
              'answer':step['answer'],'task':'v67_actual_own_history','group':ep['id'],'source_ids':ep['source_ids'],
              'provenance':{'support':ep['support'],'lesson_contract':step['lesson_contract'],'history_source':'model_own_previous_answers',
                            'previous_answers_sha256':list(deps),'gold_prior_answers_injected':False},
              'components':{'source_contract':step['lesson_contract'],'automatic_professional_pass':False}}
            rows.append(target)
            if step['id'] not in by:break
            reply=by[step['id']];history.extend([{'role':'user','content':step['question']},{'role':'assistant','content':reply['answer']}]);deps.append(d.digest(reply['answer']))
    return rows


def validate_cache(saved,pack,variant):
    rows=sequence(pack,saved)
    if len(saved)>len(rows):raise ValueError('Unexpected saved answer')
    for a,b in zip(saved,rows):
        if (a['id']!=b['id'] or a['variant']!=variant or a['question']!=b['messages'][:-1] or a['expected']!=b['answer']
            or a['prompt_sha256']!=d.digest(d.dumps(a['question'])) or a['answer_sha256']!=d.digest(a['answer']) or a['provenance']!=b['provenance']):
            raise ValueError('Saved target/source/actual own-history dependency differs')
    return rows


def evaluate_adapter(p,job,c,deadline,*,best,factory=None):
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel
    from src.wengua.qwen_finetune import prepare_frozen_base
    job=Path(job);folder=job/'generations';folder.mkdir(exist_ok=True);tag=best['tag'];variant=best.get('record_variant',tag)
    path=folder/(tag+'.jsonl');pack=d.read(job/'evaluation_pack.json');saved=read_cache(path);validate_cache(saved,pack,variant)
    total=pack['summary']['per_model_answers'];adapter=Path(best['adapter']);files={n:d.digest((adapter/n).read_bytes()) for n in ['adapter_config.json','adapter_model.safetensors']}
    contract={'adapter_files':files,'pack_sha256':d.digest(d.dumps(pack)),'config':c,'record_variant':variant,'history_source':'model_own_previous_answers','professional_approved':False}
    receipt=path.with_suffix('.receipt.json');completion=path.with_suffix('.complete.json')
    if receipt.exists() and d.read(receipt)!=contract:raise ValueError('Evaluation contract changed')
    if not receipt.exists():d.write(receipt,contract)
    def seal():return {'answers_sha256':d.digest(path.read_bytes()),'contract_sha256':d.digest(receipt.read_bytes()),'answers':total,'adapter_files':files}
    if completion.exists() and d.read(completion)!=seal():raise ValueError('Completed evaluation bytes changed')
    if len(saved)==total:
        if not completion.exists():d.write(completion,seal())
        return {'state':'complete','completed':total,'cache_hit_without_model_load':True,'professional_approved':False}
    model=base=None
    try:
        if factory:tok,base,_=factory()
        else:
            tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision'],local_files_only=True,trust_remote_code=False)
            base,_=p.book.load_book_cuda(c,False)
        base=prepare_frozen_base(base,gradient_checkpointing=False);model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False);model.eval();model.requires_grad_(False)
        while len(saved)<total:
            if time.time()>deadline-120:return {'state':'paused','completed':len(saved),'requested':total}
            target=sequence(pack,saved)[len(saved)];p.book.progress('v67_answer_progress',phase=tag,answer=len(saved)+1,total=total,file=str(path))
            reply=one_answer(model,tok,target,p.book,c,variant);append(path,reply);saved.append(reply)
        if any(d.digest((adapter/n).read_bytes())!=h for n,h in files.items()):raise ValueError('Adapter files changed during evaluation')
        d.write(completion,seal())
        return {'state':'complete','completed':total,'requested':total,'answers_sha256':d.digest(path.read_bytes()),'source_files_unchanged':True,'professional_approved':False}
    finally:
        model=base=None;gc.collect()
        if not factory:torch.cuda.empty_cache()
