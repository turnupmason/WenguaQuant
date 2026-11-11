"""Complete the missing V63 development/RAG matrix without updating weights.

No Trainer, optimizer, weight saving or curriculum update is permitted here.
"""
from pathlib import Path
from argparse import ArgumentParser
from copy import deepcopy
import gc,html,json,os,subprocess,sys,time,uuid
from wengua_course import data as d
from wengua_course_v63 import runtime,initialization
from . import v63_review
ROOT=runtime.ROOT
SCHEMA='wengua_v63_complete_checkpoint_probe_v1'
LABELS=('official_base', 'low_e01', 'low_e02', 'low_e04', 'low_e08', 'high_e01', 'high_e02', 'high_e04', 'high_e07')
SELECTION={'low_e01':('lr_5em05',1),'low_e02':('lr_5em05',2),'low_e04':('lr_5em05',4),'low_e08':('lr_5em05',8),
           'high_e01':('lr_1em04',1),'high_e02':('lr_1em04',2),'high_e04':('lr_1em04',4),'high_e07':('lr_1em04',7)}


def private(path):
    path=Path(path).resolve()
    if not path.is_relative_to((ROOT/'local').resolve()):raise ValueError('Probe outputs must stay under Quant local')
    return path


def build_prompts(pack):
    rows=pack['actual_rag_rows']+runtime.choose_development(pack,20)
    if len(rows)!=38 or len({r['id'] for r in rows})!=38:raise ValueError('18 exact RAG and20 original loss-development prompts required')
    return {'schema':SCHEMA,'records':rows,'inference_only':True,'weight_updates':0,
            'source_book_targets_only':True,'unseen_case_or_prediction_benchmark':False}


def selected_sources(training_root):
    root=private(training_root);state=d.read(root/'run_status.json');run=d.read(root/'sweep_contract.json')
    expected=d.read(ROOT/'data/reviews/v63_checkpoint_sources.json')
    from src.wengua.qwen_book_loop_v3 import stack_versions
    if stack_versions()!=expected['original_stack']:raise ValueError('Use the original pinned V63 environment to compare cached and new answers')
    if run['code_sha256']!=runtime.hashes() or run['dataset_release_sha256']!=d.digest(runtime.RELEASE.read_bytes()):raise ValueError('Frozen V63 source run differs')
    if state['state']!='complete_review_required' or state['production_enabled']:raise ValueError('Completed original V63 run required')
    for trial,files in expected['trials'].items():
        for name,sha in files.items():
            if d.digest((root/'trials'/trial/name).read_bytes())!=sha:raise ValueError('Source differs from uploaded V63: '+trial+'/'+name)
    sources={'official_base':{'adapter':None,'adapter_sha256':None,'variant':'official_pinned_base','professional_approval':False}}
    for label,(trial,epoch) in SELECTION.items():
        t=state['trials'][trial];e=next(e for e in t['fit']['epochs'] if e['cycle']==epoch)
        if t['state']!='complete' or e['steps']!=53*epoch:raise ValueError('Missing formal source epoch')
        path=private(e['adapter']);proof=initialization.inspect(path)
        if proof['schema']!=runtime.SCHEMA or path.name!='step-'+str(e['steps']):raise ValueError('Different adapter version or step')
        jc=d.read(path.parent.parent.parent/'job_contract.json')
        if jc['code_sha256']!=runtime.hashes() or set(proof['parent_config']['target_modules'])!=set(runtime.data.MODULES):raise ValueError('Not the original seven-projection source')
        sources[label]={'adapter':str(path),'adapter_sha256':proof['files']['adapter_model.safetensors'],'source_proof':proof,'variant':'formal_v63_epoch','professional_approval':False}
    for label,info in expected['reuse'].items():
        source=private(root/info['file']);receipt=private(root/info['provenance'])
        if d.digest(source.read_bytes())!=info['file_sha256'] or d.digest(receipt.read_bytes())!=info['provenance_sha256']:raise ValueError('Original RAG bytes changed')
        if d.read(receipt)['adapter_sha256']!=sources[label]['adapter_sha256']:raise ValueError('Reused RAG belongs to different weights')
        sources[label].update(reuse_file=str(source),reuse_sha256=info['file_sha256'])
    return sources


def hashes():
    names=['research_audits/v63_checkpoint_probe.py','research_audits/v63_review.py','data/reviews/v63_checkpoint_sources.json','check_wengua_v63_checkpoints.cmd','export_wengua_v63_checkpoints.cmd']
    return runtime.hashes()|{n:d.digest((ROOT/n).read_text()) for n in names}


def stage_reuse(root,label,source,rows,c):
    if not source.get('reuse_file'):return 0
    raw=Path(source['reuse_file']).read_bytes()
    if d.digest(raw)!=source['reuse_sha256']:raise ValueError('Original reuse bytes changed')
    prior=read_rows(source['reuse_file']);by={r['id']:r for r in rows}
    if not verify_cached(prior,[by[r['id']] for r in prior]):raise ValueError('Reused RAG prompts changed')
    dest=root/'answers'/(label+'.jsonl');dest.parent.mkdir(exist_ok=True)
    proof=result_contract(c,source,rows);receipt=dest.with_suffix('.provenance.json')
    if receipt.exists() and d.read(receipt)!=proof:raise ValueError('Reuse source/model contract changed')
    if not receipt.exists():d.write(receipt,proof)
    copy=dest.with_suffix('.original_rag.json')
    reuse={'source_sha256':source['reuse_sha256'],'source_file':source['reuse_file'],'adapter_sha256':source['adapter_sha256'],
           'answer_ids':[r['id'] for r in prior],'source_answers_rewritten':False,'weight_updates':0}
    if copy.exists() and d.read(copy)!=reuse:raise ValueError('Reuse receipt changed')
    d.write(copy,reuse)
    if not dest.exists():
        pending=dest.with_suffix('.reuse.pending');pending.write_bytes(raw);pending.rename(dest)
    else:
        saved=read_rows(dest,allow_partial=True)
        verify_cached(saved,rows)
        present={r['id']:r for r in saved}
        for r in prior:
            if r['id'] in present and present[r['id']]!=r:raise ValueError('Existing original RAG answer changed')
        if any(r['id'] not in present for r in prior):
            # Recover a truncated cached prefix using the immutable source.
            # Every newly generated, complete answer stays intact.
            dest.rename(dest.with_name(dest.name+'.partial-reuse-'+str(time.time_ns())))
            ids={r['id'] for r in prior}
            dest.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in prior+[r for r in saved if r['id'] not in ids]),encoding='utf8')
    return len(prior)


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
        if r['prompt_sha256']!=prompt_sha or prompt_sha!=d.digest(json.dumps(target['messages'][:-1],ensure_ascii=False,separators=(',',':'))):raise ValueError('Cached prompt serialization SHA differs')
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
                verify_cached(valid,rows)
                dest.rename(dest.with_name(dest.name+'.partial-'+str(time.time_ns())))
                dest.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in valid),encoding='utf8')
        if verify_cached(valid,rows):return {'state':'complete','completed':len(rows),'weight_updates':0,'cache_hit_without_model_load':True}
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
        verify_cached(read_rows(dest),rows)
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
    # PEFT 0.17.1 re-enables adapter gradients when leaving disable_adapter(),
    # including adapters originally loaded as inference-only. Restore freezing.
    model.requires_grad_(False)
    delta=float((enabled-disabled).abs().max())
    if not torch.isfinite(enabled).all() or not torch.isfinite(disabled).all():raise RuntimeError('Nonfinite probe logits')
    if not testing and (not matrices or not nonzero or delta<=0):raise RuntimeError('Trained adapter has zero numerical effect; no training is started')
    return {'software_fixture':testing,'lora_B_matrices':len(matrices),'nonzero_lora_B_matrices':nonzero,'max_last_position_logit_delta':delta,'next_token_changed':bool(enabled.argmax(-1).ne(disabled.argmax(-1)).any()),'parameters_frozen':not any(p.requires_grad for p in model.parameters()),'weight_updates':0}


def worker(request):
    request=private(request);raw=request.read_bytes();payload=json.loads(raw);root=private(payload['root']);label=payload['label']
    if request.parent!=root/'worker_calls' or payload['schema']!=SCHEMA or label not in LABELS:raise ValueError('Invalid read-only probe request')
    contract=d.read(root/'run_contract.json')
    if contract['code_sha256']!=hashes() or contract['sources'][label]!=payload['source'] or contract['config']!=payload['config'] or contract['software_fixture']!=payload['software_fixture']:raise ValueError('Worker source/config differs from frozen probe')
    prompt_pack=d.read(root/'prompts.json');rows=prompt_pack['records']
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
    proc=subprocess.Popen([sys.executable,'-X','utf8','-m','research_audits.v63_checkpoint_probe','--worker-request',str(request)],cwd=ROOT,env=os.environ.copy(),creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
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
    d.write(root/'run_status.json',state);parts=['<!doctype html><meta charset="utf-8"><title>V6.3 全检查点对照</title><style>body{font:16px system-ui;max-width:1150px;margin:24px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}section{border-top:1px solid #ddd;padding:12px}</style><h1>V6.3 来源开发题与RAG全检查点对照</h1><p>仅推理，权重更新0次。官方与8个已保存阶段，各20开发题和18RAG；复用已有54份原答，补生成288答。原题和概念已见，不是未知案例或专业准确率。</p><pre>'+html.escape(d.dumps(state))+'</pre>'];answers={}
    for label in LABELS:
        path=root/'answers'/(label+'.jsonl');answers[label]={r['id']:r for r in read_rows(path,allow_partial=True)} if path.exists() else {}
    for r in rows:
        parts+=['<section><h2>'+html.escape(r['id'])+'</h2><pre>'+html.escape(d.dumps(r['messages'][:-1]))+'</pre>']
        for label in LABELS:
            a=answers[label].get(r['id'],{});missing='原V6.2已有本模型RAG回答，本次保留' if label!='official_base' and r['id'].startswith('v58:actual_rag:') else '尚未生成'
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
        pack=build_prompts(runtime.verify_data());sources=selected_sources(training_root)
        c=d.read(runtime.CONFIG);tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision'],local_files_only=True,trust_remote_code=False)
    c=(fixture_config or d.read(runtime.CONFIG))|{'eval_max_new_tokens':8 if testing else 512}
    if set(sources)!=set(LABELS):raise ValueError('All nine original model sources required')
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
                stage_reuse(root,label,sources[label],pack['records'],c)
                state['models'][label]=launch(root,label,sources[label],c,deadline,testing=testing)
                if state['models'][label]['state']!='complete':state['state']='paused';report(root,state,pack['records']+pack.get('official_rag_rows',[]));return 75
            total=sum(m['completed'] for m in state['models'].values())
            reused=sum(len(read_rows(s['reuse_file'])) for s in sources.values() if s.get('reuse_file'))
            state.update(state='complete_review_required',active_model=None,stored_answers=total,reused_original_answers=reused,
                         additional_answers=total-reused,weight_updates=0)
            report(root,state,pack['records']+pack.get('official_rag_rows',[]));return 0
        except BaseException as error:
            state.update(state='paused' if isinstance(error,(KeyboardInterrupt,TimeoutError)) else 'failed',error={'type':type(error).__name__,'message':str(error)});report(root,state,pack['records']+pack.get('official_rag_rows',[]));raise


def main(argv=None):
    p=ArgumentParser(description=__doc__);p.add_argument('command',nargs='?',choices=['run','export']);p.add_argument('--output',type=Path,default=ROOT/'local/v63_checkpoint_probe/run1');p.add_argument('--training-root',type=Path,default=ROOT/'local/course_v63/run1');p.add_argument('--resume',action='store_true');p.add_argument('--hours',type=float,default=4);p.add_argument('--worker-request',type=Path)
    a=p.parse_args(argv)
    if not 0<a.hours<=24:p.error('--hours must be in (0,24]')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['PYTORCH_CUDA_ALLOC_CONF']=runtime.ALLOCATOR;os.environ['PYTORCH_ALLOC_CONF']=runtime.ALLOCATOR
    if a.worker_request:
        runtime.configure();return worker(a.worker_request)
    if a.command=='export':
        from wengua_course.runner import export_review
        target=Path.home()/'Downloads'/('Wengua_V63_checkpoint_probe_'+time.strftime('%Y%m%d_%H%M%S')+'.zip');print(export_review(a.output,target));return 0
    if a.command!='run':p.error('run or export required')
    return run(a.output,a.training_root,resume=a.resume,hours=a.hours)
if __name__=='__main__':raise SystemExit(main())
