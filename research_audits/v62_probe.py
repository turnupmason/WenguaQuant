"""Inference-only fit/input controls: official8B, V62 low54 and high54.

No Trainer, optimizer, weight saving or curriculum update is permitted here.
"""
from pathlib import Path
from argparse import ArgumentParser
from copy import deepcopy
import gc,html,json,os,subprocess,sys,time,uuid
from wengua_course import data as d
from wengua_course_v62 import runtime,initialization
from . import v62_probe_data
ROOT=runtime.ROOT
SCHEMA='wengua_v62_readonly_dialogue_probe_v1'
LABELS=('official_base','v62_low54','v62_high54')
EXPECTED_ADAPTER_SHA={'v62_low54': '4472f943c6efac2802e57c768e28c9f6a2049f4c90837d6537ed54520c303a72', 'v62_high54': '84d62205af930d24e74780d858ced36d76f1ccc4a245c5379db135d50dcbc58c'}


def private(path):
    path=Path(path).resolve()
    if not path.is_relative_to((ROOT/'local').resolve()):raise ValueError('Probe outputs must stay under Quant local')
    return path


def selected_sources(training_root):
    root=private(training_root);state=d.read(root/'run_status.json')
    if state['state']!='complete_review_required':raise ValueError('Completed V62 run required')
    sources={'official_base':{'adapter':None,'adapter_sha256':None,'variant':'official_pinned_base','professional_approval':False}}
    for label,trial,rate in [('v62_low54','lr_1em05',1e-5),('v62_high54','lr_2em05',2e-5)]:
        t=state['trials'][trial]
        if t['state']!='complete' or t['config']['learning_rate']!=rate or t['fit']['steps']!=54:raise ValueError('Expected complete V62 two-epoch source')
        path=private(t['fit']['last_adapter']);proof=initialization.inspect(path)
        jc=d.read(path.parent.parent.parent/'job_contract.json')
        if jc['code_sha256']!=runtime.hashes() or proof['schema']!=runtime.SCHEMA or not path.name=='step-54':raise ValueError('Source weights differ from frozen V62')
        if proof['files']['adapter_model.safetensors']!=EXPECTED_ADAPTER_SHA[label]:raise ValueError('Adapter differs from the uploaded V62 result')
        sources[label]={'adapter':str(path),'adapter_sha256':proof['files']['adapter_model.safetensors'],'source_proof':proof,'variant':'v62_second_epoch_candidate','professional_approval':False}
    return sources


def hashes():
    names=['research_audits/v62_probe.py','research_audits/v62_probe_data.py','check_wengua_v62_learning.cmd','export_wengua_v62_learning.cmd']
    return runtime.hashes()|{n:d.digest((ROOT/n).read_text()) for n in names}


def read_rows(path,allow_partial=False):
    rows=[];lines=Path(path).read_bytes().splitlines()
    for i,line in enumerate(lines):
        if not line.strip():continue
        try:rows.append(json.loads(line))
        except ValueError:
            if allow_partial and i==len(lines)-1:break
            raise
    return rows


def verify_cached(rows,targets,variant=None):
    by={r['id']:r for r in targets};seen=set()
    for r in rows:
        if variant is not None and r.get('variant')!=variant:raise ValueError('Cached model variant differs')
        if r['id'] not in by or r['id'] in seen:raise ValueError('Cached answer ID missing or duplicated')
        target=by[r['id']];seen.add(r['id'])
        if r['question']!=target['messages'][:-1] or r['expected']!=target['answer']:raise ValueError('Cached question or target changed')
        prompt_sha=d.digest(json.dumps(r['question'],ensure_ascii=False,separators=(',',':')))
        if r['prompt_sha256']!=prompt_sha:raise ValueError('Cached prompt SHA differs')
    return len(seen)==len(targets)


def result_contract(c,source,rows):
    return {'schema':SCHEMA,'adapter_sha256':source['adapter_sha256'],'variant':source['variant'],'model_id':c['model_id'],'revision':c['revision'],'max_new_tokens':c['eval_max_new_tokens'],'max_context':c['inference_max_context'],'messages_and_targets_sha256':d.digest(d.dumps([r['messages'] for r in rows])),'ids':[r['id'] for r in rows],'inference_only':True,'weight_updates':0}


def generate(root,label,source,rows,c,deadline,*,testing=False):
    root=private(root);dest=root/'answers'/(label+'.jsonl');dest.parent.mkdir(exist_ok=True);receipt=dest.with_suffix('.provenance.json');proof=result_contract(c,source,rows)
    if receipt.exists() and d.read(receipt)!=proof:raise ValueError('Cached model, budget or source differs')
    if dest.exists() and not receipt.exists():raise ValueError('Answer file lacks frozen provenance')
    d.write(receipt,proof)
    if dest.exists():
        valid=[];lines=dest.read_bytes().splitlines()
        for i,line in enumerate(lines):
            try:valid.append(json.loads(line))
            except ValueError:
                if i!=len(lines)-1:raise ValueError('Corrupt cached answer before last line')
                verify_cached(valid,rows,label)
                dest.rename(dest.with_name(dest.name+'.partial-'+str(time.time_ns())))
                dest.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in valid),encoding='utf8')
        if verify_cached(valid,rows,label):return {'state':'complete','completed':len(rows),'weight_updates':0,'cache_hit_without_model_load':True}
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel
    from src.wengua.qwen_finetune import prepare_frozen_base
    from src.wengua.qwen8b_attention_v1 import cuda_probe,attention_route
    from wengua_recovery.v59_watchdog import allocator_limit
    profile=runtime.profile(testing=testing);book=profile.book
    if testing:
        from wengua_course.fixtures import fixture_factory
        tok,base,compute=fixture_factory();attention=None
    else:
        tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision'],local_files_only=True,trust_remote_code=False)
        d.write(root/'answers'/(label+'.allocator.json'),allocator_limit(torch,testing=False))
        attention=cuda_probe()
        if attention['state']!='ready':raise RuntimeError(attention['reason'])
        base,compute=book.load_book_cuda(c,training=False)
    base=prepare_frozen_base(base,gradient_checkpointing=False);model=base
    if source['adapter']:
        if d.digest((Path(source['adapter'])/'adapter_model.safetensors').read_bytes())!=source['adapter_sha256']:raise ValueError('Read-only source weights changed')
        model=PeftModel.from_pretrained(base,source['adapter'],is_trainable=False)
    model.eval()
    if any(p.requires_grad for p in model.parameters()):raise RuntimeError('Inference probe requires all parameters frozen')
    try:
        with attention_route(testing=testing,report=attention) as audit:
            if source['adapter']:
                activation=activation_check(model,tok,rows[0],book,testing=testing)
                d.write(root/'answers'/(label+'.activation.json'),activation|{'adapter_sha256':source['adapter_sha256']})
            result=book.generate_rows(model,tok,rows,dest,c,deadline,label)
            if not testing and not audit['cuda_calls']:raise RuntimeError('No actual efficient CUDA inference recorded')
            d.write(root/'answers'/(label+'.attention.json'),dict(audit))
        verify_cached(read_rows(dest),rows,label)
        if source['adapter'] and d.digest((Path(source['adapter'])/'adapter_model.safetensors').read_bytes())!=source['adapter_sha256']:raise ValueError('Source adapter bytes changed during read-only inference')
        return dict(result,weight_updates=0)
    finally:
        model=base=None;gc.collect()
        if not testing:torch.cuda.empty_cache()


def activation_check(model,tok,row,book,*,testing=False):
    """Numerically verify the loaded adapter changes a real forward, without updates."""
    import torch
    matrices=[p for n,p in model.named_parameters() if 'lora_B.' in n]
    nonzero=sum(bool(p.count_nonzero()) for p in matrices)
    ids=tok.apply_chat_template(row['messages'][:-1],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    if len(ids)>4096:raise ValueError('Activation prompt exceeds context')
    prompt=torch.tensor([ids],device=model.device)
    with torch.inference_mode(),book.generation_autocast(model):
        enabled=model(input_ids=prompt,attention_mask=torch.ones_like(prompt),use_cache=False,logits_to_keep=1).logits.float()
        with model.disable_adapter():
            disabled=model(input_ids=prompt,attention_mask=torch.ones_like(prompt),use_cache=False,logits_to_keep=1).logits.float()
    delta=float((enabled-disabled).abs().max())
    if not torch.isfinite(enabled).all() or not torch.isfinite(disabled).all():raise RuntimeError('Nonfinite probe logits')
    if not testing and (not matrices or not nonzero or delta<=0):raise RuntimeError('Trained adapter has zero numerical effect; no training is started')
    return {'software_fixture':testing,'lora_B_matrices':len(matrices),'nonzero_lora_B_matrices':nonzero,'max_last_position_logit_delta':delta,'next_token_changed':bool(enabled.argmax(-1).ne(disabled.argmax(-1)).any()),'parameters_frozen':not any(p.requires_grad for p in model.parameters()),'weight_updates':0}


def worker(request):
    request=private(request);raw=request.read_bytes();payload=json.loads(raw);root=private(payload['root']);label=payload['label']
    if request.parent!=root/'worker_calls' or payload['schema']!=SCHEMA or label not in LABELS:raise ValueError('Invalid read-only probe request')
    contract=d.read(root/'run_contract.json')
    if contract['code_sha256']!=hashes() or contract['sources'][label]!=payload['source'] or contract['config']!=payload['config'] or contract['software_fixture']!=payload['software_fixture']:raise ValueError('Worker source/config differs from frozen probe')
    prompt_pack=d.read(root/'prompts.json');rows=prompt_pack['records']+(prompt_pack.get('official_rag_rows',[]) if label=='official_base' else [])
    if d.digest((root/'prompts.json').read_bytes())!=contract['prompts_sha256']:raise ValueError('Probe prompts changed')
    reply={'schema':SCHEMA,'nonce':payload['nonce'],'request_sha256':d.digest(raw),'pid':os.getpid(),'software_fixture':payload['software_fixture']}
    try:
        result=generate(root,label,payload['source'],rows,payload['config'],payload['deadline'],testing=payload['software_fixture'])
        reply.update(ok=True,result=result);code=0
    except BaseException as error:
        import traceback
        traceback.print_exc();reply.update(ok=False,error={'type':type(error).__name__,'message':str(error)});code=130 if isinstance(error,KeyboardInterrupt) else 1
    d.write(request.with_suffix('.result.json'),reply);return code


def verify_reply(reply,request,raw):
    if reply.get('schema')!=SCHEMA or reply.get('nonce')!=request['nonce'] or reply.get('request_sha256')!=d.digest(raw) or reply.get('software_fixture')!=request['software_fixture'] or type(reply.get('pid')) is not int or reply['pid']<=0:raise ValueError('Probe worker receipt/request mismatch')
    if not reply.get('ok'):raise RuntimeError('Read-only probe worker failed: '+reply['error']['message'])
    if reply['result'].get('weight_updates')!=0:raise ValueError('Probe performed weight updates')
    return reply['result']


def launch(root,label,source,c,deadline,*,testing=False):
    from src.wengua.qwen_book_process_v1 import stop_child
    root=private(root);calls=root/'worker_calls';calls.mkdir(exist_ok=True);request=calls/(str(time.time_ns())+'-'+uuid.uuid4().hex+'.json')
    payload={'schema':SCHEMA,'root':str(root),'label':label,'source':source,'config':c,'deadline':deadline,'nonce':uuid.uuid4().hex,'software_fixture':testing}
    d.write(request,payload);raw=request.read_bytes();answer=root/'answers'/(label+'.jsonl')
    proc=subprocess.Popen([sys.executable,'-X','utf8','-m','research_audits.v62_probe','--worker-request',str(request)],cwd=ROOT,env=os.environ.copy(),creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
    print('正在比较：'+label,flush=True);changed=time.monotonic();seen=None
    try:
        while proc.poll() is None:
            stamp=answer.stat().st_mtime_ns if answer.exists() else 0
            if stamp!=seen:seen=stamp;changed=time.monotonic()
            if time.monotonic()-changed>1800 or time.time()>deadline+60:stop_child(proc);raise TimeoutError('Inference paused; partial answers preserved for resume')
            time.sleep(.5)
        code=proc.wait()
    except BaseException:stop_child(proc);raise
    out=request.with_suffix('.result.json')
    if not out.exists():raise RuntimeError('Probe worker exited without receipt: '+hex(code & 0xffffffff))
    result=verify_reply(d.read(out),payload,raw)
    if code:raise RuntimeError('Probe worker exit code '+str(code))
    return result


def report(root,state,rows):
    d.write(root/'run_status.json',state);parts=['<!doctype html><meta charset="utf-8"><title>V6.2 学习与输入对照</title><style>body{font:16px system-ui;max-width:1150px;margin:24px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}section{border-top:1px solid #ddd;padding:12px}</style><h1>V6.2 原题、提示与输入格式对照</h1><p>仅推理，权重更新0次。三模型各30题，官方另补18题既有RAG基线。原题及概念已见，不是未知案例或专业准确率。</p><pre>'+html.escape(d.dumps(state))+'</pre>'];answers={}
    for label in LABELS:
        path=root/'answers'/(label+'.jsonl');answers[label]={r['id']:r for r in read_rows(path,allow_partial=True)} if path.exists() else {}
    for r in rows:
        parts+=['<section><h2>'+html.escape(r['id'])+'</h2><pre>'+html.escape(d.dumps(r['messages'][:-1]))+'</pre>']
        for label in LABELS:
            a=answers[label].get(r['id'],{});missing='原V6.2已有本模型RAG回答，本次不重复' if label!='official_base' and r['id'].startswith('v58:actual_rag:') else '尚未生成'
            parts.append('<h3>'+label+'</h3><pre>'+html.escape(a.get('answer',missing))+'</pre>')
        parts.append('</section>')
    (root/'results.html').write_text(''.join(parts),encoding='utf8')


def run(output,training_root,*,resume=False,hours=4,testing=False,fixture_pack=None,fixture_sources=None,fixture_config=None):
    runtime.configure();root=private(output);root.mkdir(parents=True,exist_ok=True)
    from src.wengua.qwen_book_loop_v3 import job_lock,keep_awake,generation_rows,stack_versions
    from transformers import AutoTokenizer
    if testing:
        from wengua_course.fixtures import fixture_factory
        tok,_,_=fixture_factory();pack=fixture_pack;sources=fixture_sources
    else:
        pack=v62_probe_data.build(runtime.verify_data());sources=selected_sources(training_root)
        c=d.read(runtime.CONFIG);tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision'],local_files_only=True,trust_remote_code=False)
    c=(fixture_config or d.read(runtime.CONFIG))|{'eval_max_new_tokens':8 if testing else 512}
    if set(sources)!=set(LABELS):raise ValueError('Exactly three declared model sources required')
    _,skips=generation_rows(pack['records']+pack.get('official_rag_rows',[]),tok,c)
    if skips:raise ValueError('Whole diagnostic source/target exceeds budget: '+d.dumps(skips))
    with keep_awake(),job_lock(root/'probe.lock'):
        prompts=root/'prompts.json'
        if prompts.exists() and d.read(prompts)!=pack:raise ValueError('Existing probe prompts differ')
        if not prompts.exists():d.write(prompts,pack)
        contract={'schema':SCHEMA,'code_sha256':hashes(),'sources':sources,'config':c,'prompts_sha256':d.digest(prompts.read_bytes()),'software_fixture':testing,'stack':stack_versions(),'inference_only':True,'weight_updates':0,'model_quality_verified':False}
        path=root/'run_contract.json'
        if path.exists() and (not resume or d.read(path)!=contract):raise ValueError('Existing probe source/model contract changed')
        if not path.exists():d.write(path,contract)
        state=d.read(root/'run_status.json') if (root/'run_status.json').exists() else {'state':'created','models':{},'inference_only':True,'weight_updates':0,'professional_promotion':False}
        deadline=time.time()+hours*3600
        try:
            for label in LABELS:
                state.update(state='running',active_model=label);report(root,state,pack['records']+pack.get('official_rag_rows',[]))
                state['models'][label]=launch(root,label,sources[label],c,deadline,testing=testing)
                if state['models'][label]['state']!='complete':state['state']='paused';report(root,state,pack['records']+pack.get('official_rag_rows',[]));return 75
            state.update(state='complete_review_required',active_model=None,generated_answers=sum(m['completed'] for m in state['models'].values()));report(root,state,pack['records']+pack.get('official_rag_rows',[]));return 0
        except BaseException as error:
            state.update(state='paused' if isinstance(error,(KeyboardInterrupt,TimeoutError)) else 'failed',error={'type':type(error).__name__,'message':str(error)});report(root,state,pack['records']+pack.get('official_rag_rows',[]));raise


def main(argv=None):
    p=ArgumentParser(description=__doc__);p.add_argument('command',nargs='?',choices=['run','export']);p.add_argument('--output',type=Path,default=ROOT/'local/v62_learning_probe/run1');p.add_argument('--training-root',type=Path,default=ROOT/'local/course_v62/run1');p.add_argument('--resume',action='store_true');p.add_argument('--hours',type=float,default=4);p.add_argument('--worker-request',type=Path)
    a=p.parse_args(argv)
    if not 0<a.hours<=24:p.error('--hours must be in (0,24]')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['PYTORCH_CUDA_ALLOC_CONF']=runtime.ALLOCATOR;os.environ['PYTORCH_ALLOC_CONF']=runtime.ALLOCATOR
    if a.worker_request:
        runtime.configure();return worker(a.worker_request)
    if a.command=='export':
        from wengua_course.runner import export_review
        target=Path.home()/'Downloads'/('Wengua_V62_learning_probe_'+time.strftime('%Y%m%d_%H%M%S')+'.zip');print(export_review(a.output,target));return 0
    if a.command!='run':p.error('run or export required')
    return run(a.output,a.training_root,resume=a.resume,hours=a.hours)
if __name__=='__main__':raise SystemExit(main())
