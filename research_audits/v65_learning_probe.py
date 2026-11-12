"""Two saved models: originals/paraphrases, actual star data, own-history turns.

Forward-only token diagnostics run separately from free answers. No training,
optimizer, adapter saving, source repair or automatic promotion is permitted.
"""
from pathlib import Path
from argparse import ArgumentParser
import gc,html,json,os,subprocess,sys,time,uuid
from wengua_course import data as d
from wengua_course_v65 import runtime,initialization
from . import v65_learning_data as data
from . import v65_token_diagnostics as token_diagnostics

ROOT=runtime.ROOT;SCHEMA=data.SCHEMA;LABELS=('v63_parent','v65_epoch2')
PINS=ROOT/'data/reviews/v65_learning_sources.json'


def private(path):
    path=Path(path).resolve()
    if not path.is_relative_to((ROOT/'local').resolve()):raise ValueError('Learning probe outputs must stay under Quant local')
    return path


def hashes():
    names=['research_audits/v65_learning_data.py','research_audits/v65_learning_probe.py','research_audits/v65_token_diagnostics.py',
           'check_wengua_v65_learning.cmd','export_wengua_v65_learning.cmd','data/reviews/v65_learning_sources.json']
    return runtime.hashes()|{n:d.digest((ROOT/n).read_text()) for n in names}


def verify_source_receipts(training_root):
    root=private(training_root);pins=d.read(PINS)
    for name,sha in pins['files'].items():
        if d.digest((root/name).read_bytes())!=sha:raise ValueError('Actual uploaded V65 source changed: '+name)
    run=d.read(root/'sweep_contract.json');state=d.read(root/'run_status.json')
    if run['schema']!=runtime.SCHEMA or run['code_sha256']!=runtime.hashes() or state['state']!='complete_review_required' or state['production_enabled']:
        raise ValueError('Completed frozen V65 run required')
    return root,state


def selected_sources(training_root):
    from src.wengua.qwen_book_loop_v3 import stack_versions
    if stack_versions()!=d.read(runtime.RELEASE)['original_stack']:raise ValueError('Use the original pinned V65 CUDA environment')
    root,state=verify_source_receipts(training_root)
    after=private(root/'trials/v63_trained/cycle_01/adapters/step-458');proof=initialization.inspect(after)
    if proof['schema']!=runtime.SCHEMA or proof['parent_config']!=state['trials']['v63_trained']['config']:
        raise ValueError('Expected the existing inherited V65 second epoch, not the official branch')
    before=initialization.default_source('v63_trained');sources={}
    for label,path in [('v63_parent',before),('v65_epoch2',after)]:
        proof=initialization.inspect(path)
        sources[label]={'adapter':str(path),'adapter_sha256':proof['files']['adapter_model.safetensors'],
            'files':proof['files'],'source_proof':proof,'variant':label,'professional_approval':False}
    return sources


def freeze(path,value):
    if path.exists():
        if d.read(path)!=value:raise ValueError('Existing probe contract changed: '+str(path))
    else:d.write(path,value)


def read_cache(path):
    if not path.exists():return []
    rows=[];kept=[];lines=path.read_bytes().splitlines(keepends=True);damaged=False
    for i,line in enumerate(lines):
        if not line.strip():raise ValueError('Unexpected blank cached record')
        try:row=json.loads(line)
        except ValueError:
            if i!=len(lines)-1:raise ValueError('Corrupt record before final line')
            damaged=True
        else:rows.append(row);kept.append(line)
    prefix=b''.join(kept)
    if prefix and not prefix.endswith(b'\n'):prefix+=b'\n'
    if damaged or prefix!=b''.join(lines):
        path.rename(path.with_name(path.name+'.partial-'+str(time.time_ns())));path.write_bytes(prefix)
    return rows


def validate_answers(answers,pack,label):
    targets=data.sequence(pack,answers)
    if len(answers)>len(targets):raise ValueError('Too many saved answers')
    for row,target in zip(answers,targets):
        if (row['id']!=target['id'] or row['variant']!=label or row['question']!=target['messages'][:-1]
            or row['expected']!=target['answer'] or row['prompt_sha256']!=d.digest(d.dumps(row['question']))
            or row['answer_sha256']!=d.digest(row['answer']) or row['provenance']!=target['provenance']):
            raise ValueError('Saved answer or own-history dependency changed')
    return targets


def validate_token_rows(rows,targets):
    if len(rows)>len(targets):raise ValueError('Too many forced targets')
    for row,target in zip(rows,targets):
        if (row['id']!=target['id'] or row['target_sha256']!=d.digest(target['answer'])
            or row['prompt_sha256']!=d.digest(d.dumps(target['messages'][:-1])) or row['weight_updates']!=0
            or row['free_generation_correctness_verified'] or not row['conditional_on_gold_prefix']):
            raise ValueError('Saved teacher-forced diagnostic target changed')


def progress(path,phase,done,total):
    value={'phase':phase,'completed':done,'total':total,'time':time.time()};d.write(path,value)
    print('v65_learning_progress '+d.dumps(value),flush=True)


def append(path,row):
    with path.open('ab') as f:f.write((d.dumps(row)+'\n').encode('utf8'));f.flush()


def one_answer(model,tok,target,book,c,label):
    import torch
    from transformers import set_seed
    messages=target['messages'][:-1];tokens=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
    if len(tokens)+c['eval_max_new_tokens']>c['inference_max_context']:raise ValueError('Whole model-own-history prompt exceeds context; no truncation allowed')
    ids=torch.tensor([tokens],device=model.device);set_seed(c['seed']);model.config.use_cache=True
    with torch.inference_mode(),book.generation_autocast(model):
        output=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,max_new_tokens=c['eval_max_new_tokens'],
            pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True,logits_to_keep=1)
    generated=output[0,len(tokens):];answer=tok.decode(generated,skip_special_tokens=True)
    return {'id':target['id'],'variant':label,'question':messages,'expected':target['answer'],'answer':answer,
        'answer_sha256':d.digest(answer),'prompt_sha256':d.digest(d.dumps(messages)),'input_tokens':len(tokens),
        'generated_tokens':len(generated),'reached_token_limit':len(generated)>=c['eval_max_new_tokens'],'empty':not answer.strip(),
        'task':target['task'],'group':target['group'],'source_ids':target['source_ids'],'provenance':target['provenance'],
        'components':target['components'],'semantic_correctness_verified':False,'weight_updates':0}


def generate(root,label,source,pack,c,deadline,*,testing=False):
    root=private(root);folder=root/'answers';folder.mkdir(exist_ok=True);dest=folder/(label+'.jsonl');forced=folder/(label+'.tokens.jsonl')
    proof={'schema':SCHEMA,'source':source,'config':c,'prompts_sha256':d.digest(d.dumps(pack)),'inference_only':True,'weight_updates':0}
    freeze(folder/(label+'.provenance.json'),proof)
    answers=read_cache(dest);validate_answers(answers,pack,label);metrics=read_cache(forced)
    targets=[r for r in pack['records'] if r['teacher_forcing']];validate_token_rows(metrics,targets)
    expected=pack['summary']['per_model_answers'];done=folder/(label+'.complete.json')
    for name,sha in source['files'].items():
        if d.digest((Path(source['adapter'])/name).read_bytes())!=sha:raise ValueError('Frozen source file changed')
    def completion():return {'answers':expected,'teacher_forced_targets':len(targets),'answer_sha256':d.digest(dest.read_bytes()),
                             'tokens_sha256':d.digest(forced.read_bytes()) if forced.exists() else d.digest(b''),'source_weights_unchanged':True,'weight_updates':0}
    if done.exists():
        if len(answers)!=expected or len(metrics)!=len(targets) or d.read(done)!=completion():raise ValueError('Completed probe output changed')
        return {'state':'complete',**d.read(done),'cache_hit_without_model_load':True}
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    from src.wengua.qwen_finetune import prepare_frozen_base
    from src.wengua.qwen8b_attention_v1 import cuda_probe,attention_route
    from wengua_recovery.v59_watchdog import allocator_limit
    from .v63_checkpoint_probe import activation_check
    profile=runtime.profile(testing=testing);book=profile.book
    if testing:
        from wengua_course.fixtures import fixture_factory
        tok,base,_=fixture_factory();attention=None
    else:
        tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision'],local_files_only=True,trust_remote_code=False)
        d.write(folder/(label+'.allocator.json'),allocator_limit(torch,testing=False));attention=cuda_probe()
        if attention['state']!='ready':raise RuntimeError(attention['reason'])
        base,_=book.load_book_cuda(c,training=False)
    base=prepare_frozen_base(base,gradient_checkpointing=False);model=PeftModel.from_pretrained(base,source['adapter'],is_trainable=False)
    model.eval();model.requires_grad_(False)
    heartbeat=folder/(label+'.progress.json')
    try:
        with attention_route(testing=testing,report=attention) as audit:
            activation=activation_check(model,tok,pack['records'][0],book,testing=testing)
            d.write(folder/(label+'.activation.json'),activation|{'adapter_sha256':source['adapter_sha256']})
            while len(answers)<expected:
                if time.time()>deadline-120:return {'state':'paused','answers':len(answers),'teacher_forced_targets':len(metrics),'weight_updates':0}
                target=data.sequence(pack,answers)[len(answers)];progress(heartbeat,'free_answers',len(answers)+1,expected)
                row=one_answer(model,tok,target,book,c,label);append(dest,row);answers.append(row)
            while len(metrics)<len(targets):
                if time.time()>deadline-120:return {'state':'paused','answers':len(answers),'teacher_forced_targets':len(metrics),'weight_updates':0}
                target=targets[len(metrics)];progress(heartbeat,'teacher_forced_tokens',len(metrics)+1,len(targets))
                result=token_diagnostics.score(model,tok,target,book,c);append(forced,result);metrics.append(result)
            if not testing and not audit['cuda_calls']:raise RuntimeError('Efficient CUDA inference was not recorded')
            d.write(folder/(label+'.attention.json'),dict(audit))
        if any(p.requires_grad for p in model.parameters()):raise RuntimeError('Source parameters unexpectedly became trainable')
        for name,sha in source['files'].items():
            if d.digest((Path(source['adapter'])/name).read_bytes())!=sha:raise ValueError('Source files changed during inference')
        freeze(done,completion());return {'state':'complete',**completion(),'cache_hit_without_model_load':False}
    finally:
        model=base=None;gc.collect()
        if not testing:torch.cuda.empty_cache()


def verify_reply(reply,payload,raw):
    if (reply.get('schema')!=SCHEMA or reply.get('nonce')!=payload['nonce'] or reply.get('request_sha256')!=d.digest(raw)
        or reply.get('software_fixture')!=payload['software_fixture'] or type(reply.get('pid')) is not int or reply['pid']<=0):
        raise ValueError('Learning-probe request/receipt mismatch')
    if not reply.get('ok'):raise RuntimeError('Learning-probe worker failed: '+reply['error']['message'])
    if reply['result'].get('weight_updates')!=0:raise ValueError('Inference probe cannot update weights')
    return reply['result']


def worker(request):
    request=private(request);raw=request.read_bytes();p=json.loads(raw);root=private(p['root']);contract=d.read(root/'run_contract.json')
    if (request.parent!=root/'worker_calls' or p['schema']!=SCHEMA or p['label'] not in LABELS
        or contract['code_sha256']!=hashes() or contract['sources'][p['label']]!=p['source'] or contract['config']!=p['config']
        or contract['software_fixture']!=p['software_fixture'] or d.digest((root/'prompts.json').read_bytes())!=contract['prompts_sha256']):
        raise ValueError('Worker frozen source or question set changed')
    reply={'schema':SCHEMA,'nonce':p['nonce'],'request_sha256':d.digest(raw),'pid':os.getpid(),'software_fixture':p['software_fixture']}
    try:
        result=generate(root,p['label'],p['source'],d.read(root/'prompts.json'),p['config'],p['deadline'],testing=p['software_fixture'])
        reply.update(ok=True,result=result);code=0
    except BaseException as e:
        import traceback
        traceback.print_exc();reply.update(ok=False,error={'type':type(e).__name__,'message':str(e)});code=130 if isinstance(e,KeyboardInterrupt) else 1
    d.write(request.with_suffix('.result.json'),reply);return code


def launch(root,label,source,c,deadline,testing):
    from src.wengua.qwen_book_process_v1 import stop_child
    calls=root/'worker_calls';calls.mkdir(exist_ok=True);request=calls/(str(time.time_ns())+'-'+uuid.uuid4().hex+'.json')
    payload={'schema':SCHEMA,'root':str(root),'label':label,'source':source,'config':c,'deadline':deadline,'nonce':uuid.uuid4().hex,'software_fixture':testing}
    d.write(request,payload);raw=request.read_bytes()
    proc=subprocess.Popen([sys.executable,'-X','utf8','-m','research_audits.v65_learning_probe','--worker-request',str(request)],
        cwd=ROOT,env=os.environ.copy(),creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
    changed=time.monotonic();stamp=None;heartbeat=root/'answers'/(label+'.progress.json')
    try:
        while proc.poll() is None:
            now=heartbeat.stat().st_mtime_ns if heartbeat.exists() else 0
            if now!=stamp:changed=time.monotonic();stamp=now
            if time.monotonic()-changed>1800 or time.time()>deadline+60:stop_child(proc);raise TimeoutError('Inference paused; complete answers retained')
            time.sleep(.5)
        code=proc.wait()
    except BaseException:stop_child(proc);raise
    path=request.with_suffix('.result.json')
    if not path.exists():raise RuntimeError('Learning probe worker exited without receipt: '+hex(code&0xffffffff))
    result=verify_reply(d.read(path),payload,raw)
    if code:raise RuntimeError('Learning probe exit code '+str(code))
    return result


def report(root,state):
    d.write(root/'run_status.json',state);parts=['<!doctype html><meta charset="utf-8"><title>V6.5学习定位</title><style>body{font:16px system-ui;max-width:1120px;margin:24px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:12px}</style><h1>V6.5 原题、改问、星位与实际多轮</h1><p>只推理，更新0次；逐token指标以标准前缀为条件，不是自由断案正确率。</p><pre>'+html.escape(d.dumps(state))+'</pre>']
    for label in LABELS:
        path=root/'answers'/(label+'.jsonl')
        if not path.exists():continue
        for r in read_cache(path):parts.append('<details><summary>'+html.escape(label+' / '+r['id'])+'</summary><pre>'+html.escape(d.dumps(r['question']))+'</pre><p>'+html.escape(r['answer'])+'</p><pre>'+html.escape(r['expected'])+'</pre></details>')
    (root/'results.html').write_text(''.join(parts),encoding='utf8')


def workbook(root,pack,*,answers=False):
    """Private full-cell review only, never a training data input."""
    from wengua_course_v65.review import xlsx
    path=root/('Wengua_V65_学习定位原答.xlsx' if answers else 'Wengua_V65_学习定位题目.xlsx')
    if path.exists():return
    if answers:
        rows=[['模型','ID','任务','完整实际对话JSON','原始回答','来源参考答','条件JSON','逐token指标JSON']]
        for label in LABELS:
            token_file=root/'answers'/(label+'.tokens.jsonl');token_rows={r['id']:r for r in read_cache(token_file)}
            for r in read_cache(root/'answers'/(label+'.jsonl')):
                rows.append([label,r['id'],r['task'],d.dumps(r['question']),r['answer'],r['expected'],d.dumps(r['components']),d.dumps(token_rows.get(r['id']))])
        sheets=[('完整原答',rows)]
    else:
        rows=[['ID','任务','完整对话JSON','来源参考答','条件JSON','来源ID']]
        rows += [[r['id'],r['task'],d.dumps(r['messages']),r['answer'],d.dumps(r['components']),d.dumps(r['source_ids'])] for r in pack['records']]
        episodes=[['对话','回合ID','本轮问题','本轮参考答','本轮条件','历史来源']]
        for ep in pack['episodes']:
            episodes += [[ep['id'],s['id'],s['question'],s['answer'],d.dumps(s['lesson_contract']),'模型自己此前的实际回答，标准答不注入'] for s in ep['steps']]
        sheets=[('固定题目',rows),('多轮计划',episodes)]
    pending=path.with_name(path.name+'.pending-'+str(time.time_ns()));xlsx(pending,sheets);pending.rename(path)


def run(output,training_root,*,resume=False,hours=4,testing=False,fixture_pack=None,fixture_sources=None,fixture_config=None):
    runtime.configure();root=private(output);root.mkdir(parents=True,exist_ok=True)
    from src.wengua.qwen_book_loop_v3 import job_lock,keep_awake,generation_rows
    from transformers import AutoTokenizer
    if testing:
        from wengua_course.fixtures import fixture_factory
        tok,_,_=fixture_factory();pack=fixture_pack;sources=fixture_sources
    else:
        pack=data.build(runtime.verify_data());sources=selected_sources(training_root)
        if d.digest(d.dumps(pack))!=d.read(PINS)['diagnostic_pack_sha256']:raise ValueError('Source-bound diagnostic pack changed')
        tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-8B',revision=d.read(runtime.CONFIG)['revision'],local_files_only=True,trust_remote_code=False)
    c=(fixture_config or d.read(runtime.CONFIG))|{'eval_max_new_tokens':8 if testing else 512}
    if set(sources)!=set(LABELS):raise ValueError('Both selected existing sources required')
    _,skips=generation_rows(pack['records'],tok,c)
    if skips:raise ValueError('Whole static prompt or target exceeds context')
    for row in pack['records']:
        if row['teacher_forcing']:token_diagnostics.encoded_target(tok,row,c['inference_max_context'])
    with keep_awake(),job_lock(root/'probe.lock'):
        freeze(root/'prompts.json',pack)
        workbook(root,pack)
        contract={'schema':SCHEMA,'code_sha256':hashes(),'sources':sources,'config':c,'prompts_sha256':d.digest((root/'prompts.json').read_bytes()),
            'software_fixture':testing,'inference_only':True,'weight_updates':0,'professional_promotion':False}
        if (root/'run_contract.json').exists() and not resume:raise ValueError('Use --resume for an existing probe')
        freeze(root/'run_contract.json',contract)
        state=d.read(root/'run_status.json') if (root/'run_status.json').exists() else {'state':'created','models':{},'summary':pack['summary'],'weight_updates':0,'professional_promotion':False}
        deadline=time.time()+hours*3600
        try:
            for label in LABELS:
                state.update(state='running',active_model=label);report(root,state)
                print('只读检查模型：'+label+'；本轮不训练。',flush=True)
                state['models'][label]=launch(root,label,sources[label],c,deadline,testing);report(root,state)
                if state['models'][label]['state']!='complete':state['state']='paused';report(root,state);return 75
            state.update(state='complete_review_required',active_model=None);report(root,state);workbook(root,pack,answers=True);return 0
        except BaseException as error:
            state.update(state='paused' if isinstance(error,(KeyboardInterrupt,TimeoutError)) else 'failed',error={'type':type(error).__name__,'message':str(error)})
            report(root,state);raise


def export(root,out):
    from wengua_course.runner import export_review
    import zipfile
    export_review(root,out)
    with zipfile.ZipFile(out,'a',zipfile.ZIP_DEFLATED) as z:
        for path in Path(root).glob('*.xlsx'):z.write(path,path.name)
    return out


def main(argv=None):
    p=ArgumentParser(description=__doc__);p.add_argument('command',nargs='?',choices=['run','check','export']);p.add_argument('--output',type=Path,default=ROOT/'local/v65_learning_probe/run1')
    p.add_argument('--training-root',type=Path,default=ROOT/'local/course_v65/run1');p.add_argument('--resume',action='store_true');p.add_argument('--hours',type=float,default=4);p.add_argument('--worker-request',type=Path)
    a=p.parse_args(argv)
    if not 0<a.hours<=24:p.error('--hours must be in (0,24]')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTORCH_CUDA_ALLOC_CONF=runtime.ALLOCATOR,PYTORCH_ALLOC_CONF=runtime.ALLOCATOR)
    runtime.configure()
    if a.worker_request:return worker(a.worker_request)
    if a.command=='export':
        out=Path.home()/'Downloads'/('Wengua_V65_learning_probe_'+time.strftime('%Y%m%d_%H%M%S')+'.zip');print(export(a.output,out));return 0
    if a.command=='check':
        pack=data.build(runtime.verify_data());print(d.dumps(pack['summary']));return 0
    if a.command!='run':p.error('run, check or export required')
    return run(a.output,a.training_root,resume=a.resume,hours=a.hours)


if __name__=='__main__':raise SystemExit(main())
