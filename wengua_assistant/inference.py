"""One independent GPU process per answer; never invoke a training engine."""
from pathlib import Path
import hashlib, json, os, sys, time, traceback

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'Qwen/Qwen3-8B'
REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
ADAPTER = ROOT / 'local/course_v67/run1/trials/object_condition_repair/cycle_01/adapters/step-113'
ADAPTER_SHA = {'adapter_model.safetensors': '06ae23d3a9130f062da400eab67bf280c425705f5771f5596d203b42ccd6d38a',
               'adapter_config.json': 'cab45dfe09408657b4c9397bf7aa8441c6bd12d7a1b4c629cafbe36954992b83'}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): result.update(chunk)
    return result.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    pending.replace(path)


def adapter_receipt(variant):
    if variant not in {'official', 'v67_e1'}: raise ValueError('未知模型')
    if variant == 'official': return {'variant': variant, 'model_id': MODEL, 'revision': REVISION}
    actual = {name: sha(ADAPTER / name) for name in ADAPTER_SHA}
    if actual != ADAPTER_SHA: raise ValueError('V6.7第一轮权重缺失或已变化，不能悄悄换用其他权重')
    return {'variant': variant, 'model_id': MODEL, 'revision': REVISION, 'adapter_files': actual,
            'adapter': str(ADAPTER), 'professional_approved': False}


def code_receipt():
    folder = Path(__file__).parent
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(folder.rglob('*'))
            if path.is_file() and path.suffix in {'.py', '.js', '.html', '.css'}}


def decode(model, tokenizer, prompt, *, testing=False):
    import torch
    from src.wengua.qwen_book_loop_v3 import generation_autocast
    sent = tokenizer.apply_chat_template(prompt['messages'], tokenize=True, add_generation_prompt=True,
                                         enable_thinking=False, return_tensors='pt').to(model.device)
    if sent.shape[-1] != prompt['prompt_budget']['input_tokens']:
        raise ValueError('实际输入与整段预算不同，停止而不截断')
    budget = prompt['prompt_budget']['max_new_tokens']
    if sent.shape[-1] + budget > 4096: raise ValueError('超过4096上下文')
    model.eval(); model.requires_grad_(False); model.config.use_cache = True
    with torch.inference_mode(), generation_autocast(model):
        answer = model.generate(input_ids=sent, attention_mask=torch.ones_like(sent), do_sample=False,
                                max_new_tokens=budget, use_cache=True, logits_to_keep=1,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    new = answer[0, sent.shape[-1]:].detach().cpu()
    return {'raw_answer': tokenizer.decode(new, skip_special_tokens=True).strip(),
            'generated_tokens': len(new), 'reached_token_limit': len(new) >= budget,
            'training_updates': 0, 'all_parameters_frozen': all(not p.requires_grad for p in model.parameters()),
            'software_fixture_only': testing}


def assess(answer, prompt):
    from src.wengua.book_rag_query import check_answer
    check = check_answer(answer, prompt)
    # This limited check does not certify full domain reasoning or medical
    # predictions. Preserve the raw response even when it detects a conflict.
    check['professional_correctness_verified'] = False
    check['whole_answer_checked_by_expert'] = False
    return check


def worker(request):
    from wengua_course.runtime import configure
    configure()
    request = Path(request).resolve(); payload = json.loads(request.read_text(encoding='utf8'))
    if request.name != 'request.json' or request.parent.parent != (ROOT / 'local/assistant/jobs').resolve():
        raise ValueError('模型任务路径不正确')
    proof = sha(request); folder = request.parent
    reply = {'nonce': payload['nonce'], 'request_sha256': proof, 'worker_pid': os.getpid(), 'training_updates': 0}
    def beat(phase, **extra): write(folder / 'progress.json', {'phase': phase, 'time': time.time(), **extra})
    try:
        import torch
        from transformers import AutoTokenizer, set_seed
        from peft import PeftModel
        from src.wengua import qwen8b_book_v1 as base
        from src.wengua.qwen_finetune import prepare_frozen_base
        from src.wengua.qwen8b_attention_v1 import cuda_probe, attention_route
        if not torch.cuda.is_available(): raise RuntimeError('当前Python没有CUDA；请使用已经跑通训练的qwen_env')
        torch.cuda.set_per_process_memory_fraction(.8)
        receipt = adapter_receipt(payload['variant'])
        if payload['model'] != receipt or payload['code_sha256'] != code_receipt():
            raise ValueError('模型或问答代码在任务开始后发生变化')
        set_seed(3407); beat('检查缓存与注意力')
        probe = cuda_probe(); write(folder / 'attention_probe.json', probe)
        if probe['state'] != 'ready': raise RuntimeError(probe['reason'])
        config = json.loads((ROOT / 'configs/course_v67.json').read_text(encoding='utf8'))
        with base.old_book.job_lock(ROOT / 'local/assistant/gpu.lock'), attention_route(report=probe) as audit:
            beat('加载本地模型')
            tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True, trust_remote_code=False)
            model, _ = base.load_cuda(config, False)
            model = prepare_frozen_base(model, gradient_checkpointing=False)
            if payload['variant'] == 'v67_e1': model = PeftModel.from_pretrained(model, str(ADAPTER), is_trainable=False)
            beat('生成回答')
            result = decode(model, tokenizer, payload['prompt'])
            result['dependencies'] = base.old_book.stack_versions()
            result['checks'] = assess(result['raw_answer'], payload['prompt'])
            if adapter_receipt(payload['variant']) != receipt: raise ValueError('推理期间权重发生变化')
            write(folder / 'attention_path.json', audit)
            result['model'] = receipt
        reply.update(ok=True, result=result); code = 0
    except BaseException as error:
        traceback.print_exc()
        reply.update(ok=False, error=str(error), error_type=type(error).__name__); code = 1
    write(folder / 'result.json', reply); beat('完成' if code == 0 else '失败')
    return code


if __name__ == '__main__': raise SystemExit(worker(sys.argv[1]))
