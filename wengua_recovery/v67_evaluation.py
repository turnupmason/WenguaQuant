"""Finish a frozen V67 evaluation; this module never calls a training routine.

Keep the exact weights, prompts, decoding and already appended answers. Bound
the lifetime of each CUDA process and release inactive cache between answers.
"""
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
import gc, json, os, subprocess, sys, time, traceback, uuid
from wengua_course import data as d
from wengua_course_v67 import runtime, evaluation, runner

SCHEMA='wengua_v67_evaluation_recovery_v1'
POLICY={'answers_per_worker':24,'allocator_fraction':.8,'empty_cache_after_every_answer':True,
        'same_model_weights_prompts_and_decoding':True,'training_updates':0,'oom_attempts_per_position':2}
FILES=('wengua_recovery/v67_evaluation.py','resume_wengua_v67_evaluation.cmd')
MANIFEST=runtime.ROOT/'data/knowledge/course_v67/evaluation_recovery_v1.json'


def recipe():
    return {'schema':SCHEMA,'policy':POLICY,'code_sha256':{n:d.digest((runtime.ROOT/n).read_text(encoding='utf8')) for n in FILES}}


def frozen_check():
    expected=d.read(MANIFEST)
    if runtime.hashes()!=expected['original_code_sha256'] or POLICY!=expected['policy']:
        raise ValueError('Frozen V67 source or recovery policy changed; original run preserved')
    return expected


def freeze(path,value):
    if path.exists() and d.read(path)!=value:raise ValueError('Recovery receipt changed: '+str(path))
    if not path.exists():d.write(path,value)


def is_cuda_oom(error):
    # PyTorch 2.8 can report an asynchronous CUDA OOM as AcceleratorError,
    # rather than torch.OutOfMemoryError. Do not classify every CUDA error OOM.
    value=(str(error.get('message','')) if isinstance(error,dict) else str(error)).lower()
    return 'out of memory' in value and ('cuda' in value or 'gpu' in value)


def verify_original_answer_prefix(path,proof):
    raw=Path(path).read_bytes();lines=raw.splitlines(keepends=True)
    if len(lines)<proof['rows'] or d.digest(b''.join(lines[:proof['rows']]))!=proof['sha256']:
        raise ValueError('Original uploaded answer prefix changed; preserve this run')


def snapshot(torch,testing):
    if testing:return {'software_fixture_only':True}
    try:
        free,total=torch.cuda.mem_get_info();stats=torch.cuda.memory_stats()
        return {'free_gib':free/2**30,'total_gib':total/2**30,'allocated_gib':torch.cuda.memory_allocated()/2**30,
                'reserved_gib':torch.cuda.memory_reserved()/2**30,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'allocator_retries':stats.get('num_alloc_retries'),
                'allocator_ooms':stats.get('num_ooms')}
    except Exception as error:return {'cuda_snapshot_error':type(error).__name__}


def stage_contract(job,c,best,pack):
    adapter=Path(best['adapter']);tag=best['tag']
    files={n:d.digest((adapter/n).read_bytes()) for n in ('adapter_config.json','adapter_model.safetensors')}
    contract={'adapter_files':files,'pack_sha256':d.digest(d.dumps(pack)),'config':c,'record_variant':tag,
              'history_source':'model_own_previous_answers','professional_approved':False}
    folder=job/'generations';folder.mkdir(exist_ok=True);path=folder/(tag+'.jsonl')
    freeze(path.with_suffix('.receipt.json'),contract)
    saved=evaluation.read_cache(path);evaluation.validate_cache(saved,pack,tag)
    return path,saved,contract


def evaluate_chunk(p,job,c,deadline,best,*,testing=False,limit=None,observe=None,factory=None):
    """Same canonical one_answer, with CPU-only saved history and finite lifetime."""
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel
    from src.wengua.qwen_finetune import prepare_frozen_base
    job=Path(job);pack=d.read(job/'evaluation_pack.json');tag=best['tag']
    path,saved,contract=stage_contract(job,c,best,pack);start=len(saved);total=pack['summary']['per_model_answers']
    if limit is None:limit=POLICY['answers_per_worker']
    if type(limit) is not int or limit<=0:raise ValueError('A positive bounded answer count is required')
    prefix=path.read_bytes() if path.exists() else b'';model=base=None
    def beat(phase,**extra):
        if observe:observe({'phase':phase,'time':time.time(),'tag':tag,'completed':len(saved),'total':total,**snapshot(torch,testing),**extra})
    def seal():
        return {'answers_sha256':d.digest(path.read_bytes()),'contract_sha256':d.digest(path.with_suffix('.receipt.json').read_bytes()),'answers':total,'adapter_files':contract['adapter_files']}
    complete=path.with_suffix('.complete.json')
    if complete.exists() and d.read(complete)!=seal():raise ValueError('Completed V67 answers changed')
    try:
        if start<total:
            beat('before_model_load')
            if factory:tok,base,_=factory()
            else:
                tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision'],local_files_only=True,trust_remote_code=False)
                base,_=p.book.load_book_cuda(c,False)
            base=prepare_frozen_base(base,gradient_checkpointing=False)
            model=PeftModel.from_pretrained(base,str(best['adapter']),is_trainable=False);model.eval();model.requires_grad_(False)
            beat('model_loaded')
        while len(saved)<total and len(saved)-start<limit:
            if time.time()>deadline-120:break
            target=evaluation.sequence(pack,saved)[len(saved)];beat('before_answer',answer_id=target['id'])
            p.book.progress('v67_recovery_answer_progress',phase=tag,answer=len(saved)+1,total=total,file=str(path))
            reply=evaluation.one_answer(model,tok,target,p.book,c,tag)
            evaluation.append(path,reply);saved.append(reply)
            # one_answer returned only CPU strings/integers. All temporary CUDA
            # tensors and its KV cache are out of scope before empty_cache.
            gc.collect()
            if not testing:torch.cuda.synchronize();torch.cuda.empty_cache()
            beat('answer_saved_cache_released',answer_id=target['id'])
        evaluation.validate_cache(saved,pack,tag)
        if not path.read_bytes().startswith(prefix):raise ValueError('Existing answer prefix was rewritten')
        if any(d.digest((Path(best['adapter'])/n).read_bytes())!=h for n,h in contract['adapter_files'].items()):raise ValueError('Read-only source adapter changed')
        if len(saved)==total:
            freeze(complete,seal());state='complete'
        else:state='paused' if time.time()>deadline-120 else 'chunk_complete'
        return {'state':state,'completed':len(saved),'requested':total,'new_answers':len(saved)-start,'previous_prefix_sha256':d.digest(prefix),
                'previous_bytes_preserved':True,'source_files_unchanged':True,'training_updates':0,'professional_approved':False}
    finally:
        model=base=None;gc.collect()
        if not testing:
            try:torch.cuda.empty_cache()
            except Exception:pass


def verify_run(root,*,testing=False,pack=None):
    frozen_check();root=Path(root).resolve();contract=d.read(root/'sweep_contract.json');state=d.read(root/'run_status.json')
    if contract['code_sha256']!=runtime.hashes() or contract['schema']!=runtime.SCHEMA or contract['epochs']!=2 or not contract['ordinary_ce']:
        raise ValueError('Original V67 experiment contract differs')
    if contract['dataset_release_sha256']!=d.digest(runtime.RELEASE.read_bytes()):raise ValueError('Original course release differs')
    if not testing:
        from src.wengua import qwen_book_loop_v3 as b
        if b.stack_versions()!=d.read(runtime.RELEASE)['original_stack']:raise ValueError('Use the original pinned CUDA environment')
        for name,proof in d.read(MANIFEST)['uploaded_answer_prefixes'].items():verify_original_answer_prefix(root/name,proof)
    job=root/'trials'/runner.TRIAL;c=state['trial']['config'];runtime.validate(c,testing=testing);fit=state['trial'].get('fit',{})
    if fit.get('state')!='complete' or fit['steps']!=fit['planned_steps'] or len(fit['epochs'])!=2 or not fit['parent_weights_reused'] or fit['fresh_official_base']:
        raise ValueError('Both formal V67 training epochs must already be complete; this entry cannot train')
    if not testing and (fit['steps']!=226 or [x['steps'] for x in fit['epochs']]!=[113,226]):raise ValueError('Unexpected formal training steps')
    if d.read(job/'cycle_01/stage_status.json')!=fit:raise ValueError('Original completed fit receipt differs')
    if not testing and d.read(job/'cycle_01/stage_contract.json')['config']!=c:raise ValueError('Original training configuration differs')
    ev=d.read(job/'evaluation_pack.json')
    if d.digest(d.dumps(ev))!=contract['evaluation_pack_sha256']:raise ValueError('Original evaluation prompts changed')
    if not testing:
        course=runtime.verify_data();rebuilt=evaluation.build(course,runtime.before.old.verify_data())
        if ev!=rebuilt:raise ValueError('Source-derived evaluation differs')
    elif pack is not None and ev!=pack:raise ValueError('Software evaluation fixture differs')
    parent_raw=(root/'initialization/parent_provenance.json').read_bytes()
    if d.digest(parent_raw)!=contract['parent_provenance_sha256'] or c['book_parent_provenance_sha256']!=d.digest(parent_raw):raise ValueError('Original learned parent receipt differs')
    runtime.verify_snapshot(c,testing=testing)
    baseline=root/'trials/parent_baseline/generations/parent.jsonl';saved=evaluation.read_cache(baseline)
    evaluation.validate_cache(saved,ev,'ordinary_ce')
    if len(saved)!=ev['summary']['per_model_answers'] or state['baseline']['completed']!=len(saved) or state['baseline']['state']!='complete':
        raise ValueError('Completed original parent baseline required')
    if d.digest(baseline.read_bytes())!=state['baseline']['answers_sha256']:raise ValueError('Original baseline answers changed')
    stages=[]
    for epoch in fit['epochs']:
        tag=f'epoch_{epoch["cycle"]:02d}';adapter=Path(epoch['adapter']).resolve()
        if adapter.parent!=job/'cycle_01/adapters' or adapter.name!=f'step-{epoch["steps"]}':raise ValueError('Epoch source outside formal adapters')
        best={'tag':tag,'adapter':str(adapter)};path,answers,receipt=stage_contract(job,c,best,ev)
        stages.append(best|{'completed':len(answers),'adapter_files':receipt['adapter_files']})
    return job,c,ev,state,stages


def worker(request):
    request=Path(request).resolve();payload=d.read(request);testing=payload['software_fixture']
    if payload['schema']!=SCHEMA or payload['operation']!='evaluate':raise ValueError('Evaluation-only recovery request required')
    runtime.configure(payload.get('app_root'));frozen_check()
    if payload['recipe']!=recipe():raise ValueError('Recovery worker code changed')
    job=Path(payload['job']).resolve()
    if request.parent!=job/'worker_calls':raise ValueError('Recovery request outside original trial')
    p=runtime.profile(testing=testing);p.engine.open_job(job,payload['config'],resume=True)
    import torch
    from wengua_recovery.v59_watchdog import allocator_limit
    from src.wengua.qwen8b_attention_v1 import cuda_probe,attention_route,POLICY as ATTN
    from wengua_recovery.v59_worker import diagnostics
    import threading
    reply={'schema':SCHEMA,'nonce':payload['nonce'],'request_sha256':d.digest(request.read_bytes()),'pid':os.getpid(),
           'software_fixture':testing,'recipe_sha256':d.digest(d.dumps(payload['recipe'])),'training_updates':0}
    samples=[]
    def observe(sample):
        sample=sample|{'pid':os.getpid()};samples.append(sample);d.write(request.with_suffix('.heartbeat.json'),sample);d.write(request.with_suffix('.memory.json'),samples)
    stopped=threading.Event();thread=threading.Thread(target=diagnostics,args=(request,stopped),daemon=True);thread.start();started=time.monotonic()
    try:
        with p.book.job_lock(job.parent.parent/'isolated_gpu_worker.lock'):
            d.write(request.with_suffix('.allocator_policy.json'),allocator_limit(torch,testing=testing))
            report=None if testing else cuda_probe();d.write(request.with_suffix('.attention_probe.json'),report or {'software_fixture':True})
            if report is not None:
                if report['state']!='ready':raise RuntimeError(report['reason'])
                expected={'policy':ATTN,'software_fixture':False,'selected_backend':report['selected_backend'],'device':report['device'],'dtype':report['dtype']}
                if d.read(job.parent.parent/'attention_backend.json')!=expected:raise ValueError('Original CUDA backend changed')
            factory=None
            if testing:
                from wengua_course.fixtures import fixture_factory
                factory=fixture_factory
            with attention_route(testing=testing,report=report) as audit:
                result=evaluate_chunk(p,job,payload['config'],payload['deadline'],payload['best'],testing=testing,
                                      limit=payload['limit'],observe=observe,factory=factory)
                d.write(request.with_suffix('.attention_path.json'),audit)
                if not testing and result['new_answers'] and not audit['cuda_calls']:raise ValueError('Actual efficient CUDA attention not recorded')
            reply.update(ok=True,result=result);code=0
    except BaseException as error:
        trace=traceback.format_exc();request.with_suffix('.exception.log').write_text(trace,encoding='utf8');print(trace,file=sys.stderr,flush=True)
        reply.update(ok=False,error={'type':type(error).__name__,'message':str(error),'cuda_oom':is_cuda_oom(error)});code=130 if isinstance(error,KeyboardInterrupt) else 1
    finally:stopped.set();thread.join(timeout=3)
    reply['elapsed_seconds']=time.monotonic()-started;d.write(request.with_suffix('.result.json'),reply);return code


def launch(p,job,c,deadline,best,*,testing=False,limit=None):
    from wengua_recovery.v59_watchdog import ProgressClock,gpu_snapshot
    from src.wengua.qwen_book_process_v1 import stop_child
    calls=job/'worker_calls';calls.mkdir(exist_ok=True);nonce=uuid.uuid4().hex
    request=calls/(str(time.time_ns())+'-'+nonce+'.json');out=request.with_suffix('.result.json')
    value={'schema':SCHEMA,'operation':'evaluate','job':str(job),'config':c,'deadline':deadline,'best':best,'nonce':nonce,
           'software_fixture':testing,'recipe':recipe(),'app_root':str(runtime.app_root()),'limit':limit or POLICY['answers_per_worker']}
    d.write(request,value);sha=d.digest(request.read_bytes())
    child=subprocess.Popen([sys.executable,'-X','utf8','-m','wengua_recovery.v67_evaluation','worker',str(request)],cwd=runtime.ROOT,
                           env=os.environ.copy(),creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
    p.book.progress('v67_evaluation_recovery_worker_started',tag=best['tag'],launcher_pid=child.pid,max_new_answers=value['limit'])
    clock=ProgressClock(time.monotonic());diagnosed=False
    try:
        while child.poll() is None:
            paths=[request.with_suffix('.heartbeat.json'),job/'generations'/(best['tag']+'.jsonl')];quiet=clock.observe(paths,time.monotonic())
            if quiet>=300 and not diagnosed:
                diagnosed=True;request.with_suffix('.dump_requested').touch();d.write(request.with_suffix('.stall_diagnostic.json'),{'quiet_seconds':quiet,'gpu':{} if testing else gpu_snapshot()})
            if quiet>=1800 or time.time()>deadline+60:
                stop_child(child);raise TimeoutError('Evaluation paused; existing answers and both adapters preserved')
            time.sleep(.5)
        code=child.wait()
    except BaseException:stop_child(child);raise
    if not out.exists():raise RuntimeError('Evaluation worker exited without receipt; status '+hex(code&0xffffffff)+'; preserved '+str(request))
    reply=d.read(out)
    if (reply.get('schema')!=SCHEMA or reply.get('nonce')!=nonce or reply.get('request_sha256')!=sha or d.digest(request.read_bytes())!=sha
        or reply.get('recipe_sha256')!=d.digest(d.dumps(value['recipe'])) or reply.get('training_updates')!=0 or reply.get('software_fixture')!=testing
        or type(reply.get('pid')) is not int or reply['pid']<=0 or (code!=0 and reply.get('ok'))):
        raise ValueError('Recovery request/result provenance differs')
    p.book.progress('v67_evaluation_recovery_worker_exited',tag=best['tag'],launcher_pid=child.pid,worker_pid=reply['pid'],exit_code=code)
    return reply


def run(root,*,hours=24,testing=False,limit=None):
    frozen_check();root=Path(root).resolve();p=runtime.profile(testing=testing);deadline=time.time()+hours*3600
    with p.book.keep_awake(),p.book.job_lock(root/'run.lock'):
        job,c,ev,original,stages=verify_run(root,testing=testing);folder=root/'evaluation_recovery';folder.mkdir(exist_ok=True)
        archive=folder/'original_run_status.json'
        if not archive.exists():archive.write_bytes((root/'run_status.json').read_bytes())
        proof=recipe();freeze(folder/'recovery_contract.json',proof)
        frozen={'fit':original['trial']['fit'],'config':c,'pack_sha256':d.digest(d.dumps(ev)),
                'baseline_sha256':d.digest((root/'trials/parent_baseline/generations/parent.jsonl').read_bytes()),
                'stages':[{k:v for k,v in s.items() if k!='completed'} for s in stages]}
        freeze(folder/'original_sources.json',frozen)
        state={'schema':SCHEMA,'state':'evaluating','training_updates':0,'completed_at_start':{s['tag']:s['completed'] for s in stages},'epochs':{}}
        d.write(folder/'recovery_status.json',state)
        try:
            for stage in stages:
                best={k:stage[k] for k in ['adapter','tag']};failures={}
                while True:
                    count=len(evaluation.read_cache(job/'generations'/(best['tag']+'.jsonl')))
                    if count==ev['summary']['per_model_answers']:
                        # Recheck byte seal without creating a GPU process.
                        result=evaluate_chunk(p,job,c,deadline,best,testing=testing,limit=1)
                        state['epochs'][best['tag']]=result;break
                    if time.time()>deadline-120:state['state']='paused';d.write(folder/'recovery_status.json',state);return 75
                    reply=launch(p,job,c,deadline,best,testing=testing,limit=limit)
                    if not reply['ok']:
                        if reply['error']['type']=='KeyboardInterrupt':raise KeyboardInterrupt()
                        after=len(evaluation.read_cache(job/'generations'/(best['tag']+'.jsonl')))
                        failures[after]=failures.get(after,0)+1
                        if reply['error']['cuda_oom'] and failures[after]<POLICY['oom_attempts_per_position']:
                            p.book.progress('v67_evaluation_retry_fresh_process',tag=best['tag'],completed=after,training_updates=0);continue
                        raise RuntimeError('Evaluation recovery stopped: '+reply['error']['message'])
                    result=reply['result'];state['epochs'][best['tag']]=result;d.write(folder/'recovery_status.json',state)
                    if result['state']=='paused':state['state']='paused';d.write(folder/'recovery_status.json',state);return 75
                    if result['completed']<=count:raise RuntimeError('Recovery made no answer progress')
            if d.read(job/'cycle_01/stage_status.json')!=frozen['fit']:raise ValueError('Original fit changed during evaluation')
            state['state']='complete';d.write(folder/'recovery_status.json',state)
            final=deepcopy(original);final['trial']['epoch_evaluations']=state['epochs'];final['trial']['state']='complete'
            final.update(state='complete_review_required',selection='none_until_full_answer_retention_review',old_weights_preserved=True,
                         additional_long_training_started=False,evaluation_recovery={'schema':SCHEMA,'training_updates':0,'recipe_sha256':d.digest(d.dumps(proof))})
            final.pop('error',None);final.pop('active_epoch',None);runner.report(root,final);return 0
        except BaseException as error:
            state.update(state='paused' if isinstance(error,(KeyboardInterrupt,TimeoutError)) else 'failed',error={'type':type(error).__name__,'message':str(error)})
            d.write(folder/'recovery_status.json',state);raise


def main(argv=None):
    parser=ArgumentParser(description=__doc__);parser.add_argument('command',choices=['run','check','worker']);parser.add_argument('request',type=Path,nargs='?')
    parser.add_argument('--output',type=Path,default=runtime.ROOT/'local/course_v67/run1');parser.add_argument('--hours',type=float,default=24);parser.add_argument('--app-root',type=Path)
    args=parser.parse_args(argv)
    if not 0<args.hours<=168:parser.error('hours must be in (0,168]')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTORCH_CUDA_ALLOC_CONF=runtime.ALLOCATOR,PYTORCH_ALLOC_CONF=runtime.ALLOCATOR)
    if args.command=='worker':
        if args.request is None:parser.error('worker request required')
        return worker(args.request)
    runtime.configure(args.app_root)
    if args.command=='check':
        _,_,ev,_,stages=verify_run(args.output);print(d.dumps({'training_updates':0,'answers_per_stage':ev['summary']['per_model_answers'],'completed':{s['tag']:s['completed'] for s in stages}}));return 0
    return run(args.output,hours=args.hours)


if __name__=='__main__':raise SystemExit(main())
