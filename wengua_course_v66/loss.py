"""Bounded exact CE; redistribute each answer's mass towards declared spans.

Factor one is native token CE, factor four is the controlled experiment. Both
use the same total target-token mass per answer and Trainer accumulation
normalizer. Evaluation always reports *unweighted* CE. No fortune scores.
"""
from contextlib import contextmanager
from unittest.mock import patch


def encode_decisions(view,tokenizer,encoded,spans):
    messages=view['messages'];text=messages[-1]['content']
    before=tokenizer.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,enable_thinking=False)
    offsets=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
    prefix=tokenizer.apply_chat_template(messages[:-1],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    if not rendered.startswith(before+text) or offsets['input_ids']!=encoded['input_ids'] or encoded['labels'][:len(prefix)]!=[-100]*len(prefix):
        raise ValueError('Actual prefix, target or tokenizer offsets differ')
    for s in spans:
        if text[s['start']:s['end']]!=s['text'] or not 0<=s['start']<s['end']<=len(text):raise ValueError('Nonliteral decision span')
    start=len(before);mask=[]
    for i,(a,b) in enumerate(offsets['offset_mapping']):
        mask.append(int(i>=len(prefix) and any(a<start+s['end'] and b>start+s['start'] for s in spans)))
    if spans and not any(mask):raise ValueError('Declared decision has no supervised positions')
    if any(v and label==-100 for v,label in zip(mask,encoded['labels'],strict=True)):raise ValueError('Prompt was weighted')
    return {**encoded,'decision_mask':mask}


def collator(pad_id):
    def collate(rows):
        import torch
        size=max(len(r['input_ids']) for r in rows)
        return {key:torch.tensor([r[key]+[pad]*(size-len(r[key])) for r in rows],dtype=torch.long)
                for key,pad in [('input_ids',pad_id),('attention_mask',0),('labels',-100),('decision_mask',0)]}
    return collate


def supervised_loss(model,inputs,*,factor,chunk_tokens=128,num_items_in_batch=None,audit=None):
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    base=model.get_base_model() if hasattr(model,'get_base_model') else model
    if base.config.model_type!='qwen3' or any(p.requires_grad for p in base.lm_head.parameters()):raise ValueError('Frozen Qwen3 vocabulary head required')
    labels=inputs['labels'][:,1:];mask=inputs['decision_mask'][:,1:]
    if labels.shape[0]!=1:raise ValueError('Matched single-device microbatch one required')
    if mask.shape!=labels.shape or bool(((mask!=0)&(mask!=1)).any()) or bool((mask.bool()&labels.eq(-100)).any()):raise ValueError('Invalid decision mask')
    if factor not in (1,4):raise ValueError('Predeclared ordinary or fourfold decision loss required')
    count=labels.ne(-100).sum();weights=(1+(factor-1)*mask.float())*labels.ne(-100)
    mass=weights.sum()
    if not bool(count>0) or not bool(mass>0):raise ValueError('Empty supervised answer')
    # Normalized answer mass: neither extra tokens nor a changed effective LR.
    weights=weights*(count/mass)
    positions=labels.ne(-100).any(0).nonzero(as_tuple=False).flatten()
    data={k:v for k,v in inputs.items() if k not in {'labels','decision_mask'}}
    hidden=base.model(**data,use_cache=False,return_dict=True).last_hidden_state
    def ce(h,t,w):
        if h.shape[1]>chunk_tokens:raise RuntimeError('Unbounded vocabulary projection')
        if audit is not None:audit['max_projected_positions']=max(audit['max_projected_positions'],int(h.shape[1]))
        logits=base.lm_head(h).float()
        losses=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),t.reshape(-1),ignore_index=-100,reduction='none')
        return (losses*w.reshape(-1)).sum()
    total=None
    for idx in positions.split(chunk_tokens):
        args=(hidden.index_select(1,idx),labels.index_select(1,idx),weights.index_select(1,idx))
        val=checkpoint(ce,*args,use_reentrant=False,preserve_rng_state=False) if torch.is_grad_enabled() and args[0].requires_grad else ce(*args)
        total=val if total is None else total+val
    if audit is not None:
        audit['last_supervised_tokens']=int(count);audit['last_decision_tokens']=int(mask.sum());audit['last_factor']=factor
    denominator=count if num_items_in_batch is None else num_items_in_batch
    return total/torch.as_tensor(denominator,device=total.device,dtype=total.dtype)


@contextmanager
def memory_trainer(chunk_tokens,*,factor):
    from transformers.trainer import Trainer
    audit={'route':'v66_normalized_decision_ce_v1','chunk_tokens':chunk_tokens,'training_factor':factor,'evaluation_factor':1,
           'compute_loss_calls':0,'train_loss_calls':0,'eval_loss_calls':0,'max_projected_positions':0,'weighted_training_calls':0}
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        if self.accelerator.num_processes!=1:raise ValueError('Single-GPU normalization required')
        train=model.training;audit['compute_loss_calls']+=1;audit['train_loss_calls' if train else 'eval_loss_calls']+=1
        if train and factor>1:audit['weighted_training_calls']+=1
        with self.accelerator.autocast():
            value=supervised_loss(model,inputs,factor=factor if train else 1,chunk_tokens=chunk_tokens,num_items_in_batch=num_items_in_batch,audit=audit)
        return (value,{'loss':value}) if return_outputs else value
    with patch.object(Trainer,'compute_loss',compute_loss):yield audit
