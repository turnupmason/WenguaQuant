"""Check actual training inputs, adapter activation and incremental generation."""
from wengua_course import data as d


def compare(model,ids,book):
    import torch
    was_training=model.training;cache=model.config.use_cache;model.eval();model.config.use_cache=True
    try:
        with torch.inference_mode(),book.generation_autocast(model):
            generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,
                max_new_tokens=2,min_new_tokens=2,pad_token_id=model.config.pad_token_id,
                use_cache=True,return_dict_in_generate=True,output_scores=True,repetition_penalty=1.,logits_to_keep=1)
            comparisons=[]
            for i,scores in enumerate(generated.scores):
                prefix=generated.sequences[:,:ids.shape[1]+i]
                reference=model(input_ids=prefix,attention_mask=torch.ones_like(prefix),use_cache=False,logits_to_keep=1).logits[:,-1,:].float()
                # min_new_tokens suppresses EOS in generation scores; compare
                # all other logits, not the deliberately blocked EOS entries.
                finite=torch.isfinite(scores)&torch.isfinite(reference)
                a=reference[finite];b=scores.float()[finite]
                relative=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a).clamp_min(1e-6))
                if not bool(finite.any()) or relative>.03:raise RuntimeError('Cached generation differs from full-prefix forward')
                comparisons.append({'generated_step':i,'relative_l2_error':relative,'max_abs_error':float((a-b).abs().max())})
            active=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1).logits[:,-1,:].float()
            with model.disable_adapter():base=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=1).logits[:,-1,:].float()
            nonzero=any(bool(p.count_nonzero()) for n,p in model.named_parameters() if 'lora_B' in n)
            delta=float((active-base).abs().max())
        return {'actual_cuda':ids.device.type=='cuda','cached_vs_full_prefix':comparisons,'nonzero_lora_B_present':nonzero,
                'adapter_max_logit_delta':delta,'weights_updated_by_check':False,'actual_8b_quality_verified':False}
    finally:model.train(was_training);model.config.use_cache=cache


def training_callback(job,c,book,testing):
    from transformers import TrainerCallback
    class NumericalAudit(TrainerCallback):
        def on_train_begin(self,args,state,control,**kw):
            import torch
            rows=book.read_rows(job/'prepared/cycle_01.jsonl.gz');r=min(rows,key=lambda r:r['tokens']);encoded=r['encoded']
            first=next(i for i,v in enumerate(encoded['labels']) if v!=-100)
            if first<1:raise ValueError('Assistant prefix must have masked prompt tokens')
            model=kw['model'];ids=torch.tensor([encoded['input_ids'][:first]],device=model.device)
            devices=[model.device.index or 0] if ids.device.type=='cuda' else []
            with torch.random.fork_rng(devices=devices):proof=compare(model,ids,book)
            proof.update(training_step=state.global_step,source_variant=c['source_variant'],software_fixture_only=testing,
                         actual_training_prompt_id=r['id'],masked_prefix_tokens=first,
                         prompt_tokens_sha256=d.digest(d.dumps(encoded['input_ids'][:first])))
            output=job/'numerical_checks';output.mkdir(exist_ok=True)
            d.write(output/('start_'+str(state.global_step)+'_'+str(__import__('time').time_ns())+'.json'),proof)
            book.progress('v65_training_inference_parity_verified',source_variant=c['source_variant'],step=state.global_step,
                          actual_cuda=proof['actual_cuda'],max_relative_error=max(v['relative_l2_error'] for v in proof['cached_vs_full_prefix']))
    return NumericalAudit()
