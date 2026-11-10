"""Bounded preflight: fused/eager and cached/full logits on the actual model."""
from .dataset import SYSTEM


def forward_parity(base, tok, book):
    import torch
    before = base.config._attn_implementation
    rows = [
        '已核实条件：甲方属木，乙方属水。谁生谁？只问直接五行关系。',
        '已核实条件：{"甲方五行":"木","乙方五行":"水"}。谁生谁？只问直接五行关系。']
    comparisons = []
    base.eval()

    def check(a, b, label, tokens):
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise RuntimeError('Nonfinite preflight logits; no training started')
        relative = float(torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(b).clamp_min(1e-6))
        record = {'kind': label, 'prompt_tokens': tokens, 'relative_l2_error': relative,
                  'max_logit_abs_error': float((a-b).abs().max()),
                  'next_token_agrees': bool(a.argmax(-1).eq(b.argmax(-1)).all())}
        comparisons.append(record)
        if relative > .03:
            raise RuntimeError('Fused/cache forward differs from reference; no training started: ' + str(record))

    try:
        with torch.inference_mode(), book.generation_autocast(base):
            for user in rows:
                ids = tok.apply_chat_template([{'role': 'system', 'content': SYSTEM},
                                                {'role': 'user', 'content': user}],
                                               tokenize=True, add_generation_prompt=True, enable_thinking=False)
                if not 1 <= len(ids) <= 512:
                    raise ValueError('Preflight reference must stay bounded to 512 prompt tokens')
                prompt = torch.tensor([ids], device=base.device)
                base.set_attn_implementation('sdpa')
                first = base(input_ids=prompt, attention_mask=torch.ones_like(prompt), use_cache=True, logits_to_keep=1)
                fused = first.logits.float()
                next_id = fused[:, -1].argmax(-1).reshape(1, 1)
                full_ids = torch.cat([prompt, next_id], dim=1)
                cached = base(input_ids=next_id, attention_mask=torch.ones_like(full_ids),
                              past_key_values=first.past_key_values, use_cache=True, logits_to_keep=1).logits.float()
                full = base(input_ids=full_ids, attention_mask=torch.ones_like(full_ids),
                            use_cache=False, logits_to_keep=1).logits.float()
                check(cached, full, 'cached_vs_full_sdpa', len(ids))
                base.set_attn_implementation('eager')
                reference = base(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                                 use_cache=False, logits_to_keep=1).logits.float()
                check(fused, reference, 'sdpa_vs_eager_prefill', len(ids))
                ref_next = base(input_ids=full_ids, attention_mask=torch.ones_like(full_ids),
                                use_cache=False, logits_to_keep=1).logits.float()
                check(cached, ref_next, 'sdpa_cached_vs_eager_full', len(ids))
                del first, fused, cached, full, reference, ref_next
    finally:
        base.set_attn_implementation(before or 'sdpa')
    return {'comparisons': comparisons, 'actual_cuda': base.device.type == 'cuda',
            'reference_scope': 'two_short_inputs_not_full_context_validation',
            'training_sdpa_restored': base.config._attn_implementation == (before or 'sdpa'),
            'long_context_training_math_fallback_allowed': False}
