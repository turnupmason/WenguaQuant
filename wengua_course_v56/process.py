"""8B-only worker transport, pressure receipts, progress heartbeat and timeouts."""
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

from .runtime import profile,hashes,SWEEP_SCHEMA
from src.wengua.qwen8b_loss_v1 import memory_trainer
from src.wengua.qwen8b_attention_v1 import cuda_probe,attention_route,POLICY
from src.wengua.qwen_book_process_v1 import stop_child

SCHEMA='wengua_source_course_v56_process'
MEMORY_KEYS=('model_id','revision','max_seq_length','inference_max_context','lora_r','lora_alpha','target_modules',
    'gradient_accumulation_steps','gradient_checkpointing','loss_mode','loss_chunk_tokens','attention_mode','allocator_config','smoke_steps','seed')


def execution(root,p):
    b=p.book;root=b.private_output(root)
    contract={'schema':SCHEMA,'code_sha256':hashes(),'software_fixture':p.testing,'process_per_unit':True,
        'receipt_protocol':'request_sha_and_nonce','memory_loss':'checkpointed_supervised_ce_v1',
        'loss_runtime_route':'canonical_trainer_compute_loss_v2','attention_runtime_route':POLICY}
    if (root/'sweep_contract.json').exists() and b.read_json(root/'sweep_contract.json')['schema']!=SWEEP_SCHEMA:
        raise ValueError('8B cannot adopt a 14B/legacy experiment')
    path=root/'process_execution.json'
    if path.exists():
        if b.read_json(path)!=contract:raise ValueError('8B execution contract changed; choose a new output')
    elif (root/'sweep_contract.json').exists():b.atomic_json(path,contract)


def pressure_signature(job,c,p):
    b=p.book;doctor=b.read_json(job/'doctor.json') if (job/'doctor.json').exists() else {}
    devices=[{k:v for k,v in d.items() if k!='free_gib'} for d in doctor.get('cuda_devices',[])]
    backend=job.parent.parent/'attention_backend.json'
    return {'config':{k:c[k] for k in MEMORY_KEYS},'code_sha256':hashes(),'software_fixture':p.testing,
        'attention_backend_sha256':b.file_sha(backend) if backend.exists() else None,
        'smoke_data_sha256':b.file_sha(job/'prepared/smoke.jsonl.gz'),
        'validation_data_sha256':b.file_sha(job/'prepared/validation_loss.jsonl.gz'),
        'devices':devices,'stack':b.stack_versions()}


def launch(p,job,c,deadline,operation,*,probe=False,checkpoint=None,best=None,stop_step=None):
    b=p.book;job=b.private_output(job);root=job.parent.parent;execution(root,p)
    pressure=pressure_signature(job,c,p) if probe else None
    cache=root/'pressure_proofs'/(b.digest(b.dumps(pressure))+'.json') if probe else None
    if cache is not None and cache.exists():
        proof=b.read_json(cache);source=b.private_output(proof['job'])
        if not source.is_relative_to(root/'trials') or proof['signature']!=pressure:raise ValueError('Pressure proof scope mismatch')
        if b.file_sha(source/'smoke/stage_status.json')!=proof['status_sha256']:raise ValueError('Pressure proof changed')
        result=b.read_json(source/'smoke/stage_status.json')
        b.checked_adapter(result['adapter'])
        if result['state']!='complete' or result['steps']<c['smoke_steps']:raise ValueError('Incomplete pressure proof')
        b.atomic_json(job/'pressure_reuse.json',{'source_job':str(source),'proof_sha256':b.file_sha(cache),
            'memory_only':True,'adapter_not_used_for_training':True})
        b.progress('pressure_proof_reused',trial=job.name,source=source.name,updates=result['steps'])
        return result
    calls=job/'worker_calls';calls.mkdir(exist_ok=True);nonce=uuid.uuid4().hex
    request=calls/(str(time.time_ns())+'-'+nonce+'.json');out=request.with_suffix('.result.json');heart=request.with_suffix('.heartbeat.json')
    b.atomic_json(request,{'schema':SCHEMA,'job':str(job),'config':c,'deadline':deadline,'operation':operation,
        'probe':probe,'checkpoint':str(checkpoint) if checkpoint else None,'best':best,'nonce':nonce,
        'software_fixture':p.testing,'fixture_stop_step':stop_step})
    digest=b.file_sha(request)
    child=subprocess.Popen([sys.executable,'-X','utf8',str(b.ROOT/'run_course_v56_worker.py'),str(request)],cwd=b.ROOT,
        env=os.environ.copy(),creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
    b.progress('qwen8b_worker_started',trial=job.name,operation=operation,probe=probe,launcher_pid=child.pid)
    last_change=time.monotonic();seen=None
    try:
        while child.poll() is None:
            paths=[heart,job/'smoke/stage_status.json',job/'cycle_01/stage_status.json']+list((job/'generations').glob('*.jsonl'))
            stamp=max((f.stat().st_mtime_ns for f in paths if f.exists()),default=0)
            if stamp!=seen:seen=stamp;last_change=time.monotonic()
            if time.monotonic()-last_change>600:
                stop_child(child)
                raise RuntimeError('8B worker made no heartbeat/output progress for 10 minutes; preserved '+str(request))
            time.sleep(0.5)
        code=child.wait()
    except BaseException:stop_child(child);raise
    if not out.exists():
        b.atomic_json(request.with_suffix('.process_exit.json'),{'launcher_pid':child.pid,'returncode':code,'windows_status_hex':hex(code & 0xffffffff),'request_sha256':digest,'all_outputs_preserved':True})
        raise RuntimeError('8B worker exited without receipt (status '+hex(code & 0xffffffff)+'): '+str(request))
    reply=b.read_json(out)
    if (reply.get('schema')!=SCHEMA or reply.get('nonce')!=nonce or reply.get('request_sha256')!=digest
        or b.file_sha(request)!=digest or type(reply.get('pid')) is not int or reply['pid']<=0):
        raise ValueError('8B worker receipt/request mismatch')
    b.progress('qwen8b_worker_exited',launcher_pid=child.pid,worker_pid=reply.get('pid'),exit_code=code)
    if code or not reply['ok']:
        error=reply['error']
        if error['type']=='KeyboardInterrupt':raise KeyboardInterrupt()
        raise RuntimeError(('CUDA out of memory: ' if error['cuda_oom'] else '8B worker failed: ')+error['message'])
    result=reply['result']
    if cache is not None and result['state']=='complete':
        # The first real worker establishes the immutable backend contract.
        pressure=pressure_signature(job,c,p);cache=root/'pressure_proofs'/(b.digest(b.dumps(pressure))+'.json')
        b.atomic_json(cache,{'signature':pressure,'job':str(job),'status_sha256':b.file_sha(job/'smoke/stage_status.json')})
    return result


def worker(request):
    request=Path(request).resolve()
    from src.wengua.qwen_book_loop_v3 import read_json
    payload=read_json(request);p=profile(testing=payload.get('software_fixture') is True);b=p.book
    if payload['schema']!=SCHEMA or payload['operation'] not in ('fit','evaluate','ask','ping'):raise ValueError('V5.6 worker permits fit/evaluate/ask/ping; final test stays sealed')
    job=b.private_output(payload['job']);c=payload['config']
    if request.parent!=job/'worker_calls':raise ValueError('8B request outside trial')
    execution(job.parent.parent,p);p.engine.open_job(job,c,resume=True)
    if not p.testing:
        os.environ['PYTORCH_CUDA_ALLOC_CONF']=c['allocator_config'];os.environ['PYTORCH_ALLOC_CONF']=c['allocator_config']
    heart=request.with_suffix('.heartbeat.json')
    def beat(phase,**extra):b.atomic_json(heart,{'phase':phase,'pid':os.getpid(),'time':time.time(),**extra})
    beat('load')
    def interrupt(signum,frame):raise KeyboardInterrupt()
    if os.name=='nt':signal.signal(signal.SIGBREAK,interrupt)
    if payload['operation']=='ping':
        b.atomic_json(request.with_suffix('.result.json'),{'schema':SCHEMA,'nonce':payload['nonce'],
            'request_sha256':b.file_sha(request),'pid':os.getpid(),'ok':True,
            'result':{'pid':os.getpid(),'torch_imported':'torch' in sys.modules,'offline':os.environ.get('HF_HUB_OFFLINE')}})
        return 0
    factory=None;callbacks=()
    if p.testing:
        from wengua_course.fixtures import fixture_factory
        factory=fixture_factory
    from transformers import TrainerCallback
    loss_audit=attention_audit=None
    class Heartbeat(TrainerCallback):
        count=0;printed=0;epochs=[];last_train_calls=0;last_eval_calls=0;last_attention_calls=0
        def checked_loss(self,phase,state):
            key='train_loss_calls' if phase=='train' else 'eval_loss_calls'
            previous=self.last_train_calls if phase=='train' else self.last_eval_calls
            if (loss_audit is None or loss_audit[key]<=previous
                or not 0<loss_audit['max_projected_positions']<=c['loss_chunk_tokens']):
                raise RuntimeError('Actual Trainer bypassed bounded loss; refusing unverified memory results')
            if (attention_audit is None or attention_audit['calls']<=self.last_attention_calls
                or (not p.testing and not attention_audit['cuda_calls'])):
                raise RuntimeError('Actual model bypassed verified SDPA route')
            self.last_attention_calls=attention_audit['calls']
            if previous==0:
                b.progress('qwen8b_bounded_loss_verified',phase=phase,route=loss_audit['route'],
                    max_projected_positions=loss_audit['max_projected_positions'],chunk_tokens=c['loss_chunk_tokens'])
                b.progress('qwen8b_attention_verified',phase=phase,backend=attention_audit['selected_backend'],
                    native_gqa=False,math_fallback_allowed=attention_audit['math_fallback_allowed'])
            if phase=='train':self.last_train_calls=loss_audit[key]
            else:self.last_eval_calls=loss_audit[key]
            b.atomic_json(request.with_suffix('.loss_path.json'),dict(loss_audit,checked_phase=phase,step=state.global_step))
            b.atomic_json(request.with_suffix('.attention_path.json'),dict(attention_audit,checked_phase=phase,step=state.global_step))
        def on_train_begin(self,args,state,control,**kwargs):
            steps=[int(v['step']) for v in kwargs['optimizer'].state.values() if 'step' in v]
            b.atomic_json(request.with_suffix('.training_start.json'),{'step':state.global_step,'optimizer_steps':steps,'pid':os.getpid()})
            beat('train',step=state.global_step);b.progress('qwen8b_training_started',probe=payload['probe'],updates=state.max_steps,accumulation=args.gradient_accumulation_steps)
        def on_epoch_begin(self,args,state,control,**kwargs):
            self.epochs.append({'step':state.global_step,'optimizer_id':id(kwargs['optimizer']),'scheduler_step':kwargs['lr_scheduler'].last_epoch})
            b.atomic_json(request.with_suffix('.epochs.json'),self.epochs)
        def on_substep_end(self,args,state,control,**kwargs):
            self.checked_loss('train',state)
            self.count+=1;beat('train',step=state.global_step,micro_batches=self.count)
            if time.time()-self.printed>30:self.printed=time.time();b.progress('qwen8b_microbatch_progress',step=state.global_step,micro_batches=self.count)
        def on_step_end(self,args,state,control,**kwargs):
            self.checked_loss('train',state)
            beat('train',step=state.global_step)
            if payload.get('fixture_stop_step') and state.global_step>=payload['fixture_stop_step']:
                control.should_save=True;control.should_training_stop=True
        def on_prediction_step(self,args,state,control,**kwargs):
            self.checked_loss('validation',state);beat('validation',step=state.global_step)
        def on_train_end(self,args,state,control,**kwargs):beat('generation',step=state.global_step)
    callbacks=(Heartbeat(),)
    reply={'schema':SCHEMA,'nonce':payload['nonce'],'request_sha256':b.file_sha(request),'pid':os.getpid(),'software_fixture':p.testing}
    try:
        with b.job_lock(job.parent.parent/'isolated_gpu_worker.lock'):
            report=None if p.testing else cuda_probe()
            b.atomic_json(request.with_suffix('.attention_probe.json'),report or {'software_fixture':True,'cuda_verified':False})
            if report is not None and report['state']!='ready':raise RuntimeError(report['reason']+'; see '+str(request.with_suffix('.attention_probe.json')))
            backend={'policy':POLICY,'software_fixture':p.testing,'selected_backend':report['selected_backend'] if report else 'cpu_fixture',
                'device':report['device'] if report else 'cpu_fixture','dtype':report['dtype'] if report else 'torch.float32'}
            backend_path=job.parent.parent/'attention_backend.json'
            if backend_path.exists():
                if b.read_json(backend_path)!=backend:raise RuntimeError('Attention backend changed within experiment; preserve it and choose a new output')
            else:b.atomic_json(backend_path,backend)
            with attention_route(testing=p.testing,report=report) as attention_audit,memory_trainer(c['loss_chunk_tokens']) as loss_audit:
                b.progress('qwen8b_bounded_loss_installed',route=loss_audit['route'],chunk_tokens=loss_audit['chunk_tokens'])
                if report is not None:b.progress('qwen8b_fused_attention_probe_passed',backend=report['selected_backend'],math_fallback_allowed=False,test_lengths=[64,3866,4096,1])
                if payload['operation']=='evaluate':
                    from wengua_course_v56.runner import evaluate_adapter
                    result=evaluate_adapter(p,job,c,payload['deadline'],best=payload['best'],factory=factory)
                elif payload['operation']=='ask':
                    from wengua_course_v53.rag import answer
                    result=answer(p,job,c,payload['deadline'],b.read_json(payload['best']['prompt']),payload['best']['adapter'],factory=factory)
                elif payload['operation']=='fit':result=p.engine.fit(job,c,payload['deadline'],probe=payload['probe'],cpu_factory=factory,extra_callbacks=callbacks,resume_from_checkpoint=payload['checkpoint'])
                else:raise ValueError('Unknown 8B worker operation')
        reply.update(ok=True,result=result);code=0
    except BaseException as error:
        if attention_audit is not None:b.atomic_json(request.with_suffix('.attention_path.json'),dict(attention_audit,checked_phase='failed',error_type=type(error).__name__))
        traceback.print_exc();reply.update(ok=False,error={'type':type(error).__name__,'message':str(error),'cuda_oom':b.is_cuda_oom(error)})
        code=130 if isinstance(error,KeyboardInterrupt) else 1
    b.atomic_json(request.with_suffix('.result.json'),reply);return code
