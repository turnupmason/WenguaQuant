"""Frozen forward-only teacher-forced NLL, split at the decision sentence.

This conditions each token on the gold prefix. It is not free-generation
correctness, a clinical probability, a training objective or an update.
"""
from wengua_course import data as d


def target_segments(row):
    kind=(row.get('provenance',{}).get('lesson_contract') or {}).get('kind')
    names={'verify':['verification_verdict','directed_relation','properties_and_law'],
           'objects':['directed_relation','properties_and_law'],'relation':['directed_relation','properties_and_law'],
           'identity':['patient_binding','caller_condition','birth_information'],
           'loan':['bank_result','role_bindings_and_direction'],
           'marriage':['branch_verdict','condition_state_and_scope']}.get(kind,['decision','explanation'])
    text=row['answer'];out=[];start=0
    while start<len(text):
        end=text.find('。',start);end=len(text) if end<0 else end+1
        i=len(out);out.append({'name':names[min(i,len(names)-1)],'start':start,'end':end,'text':text[start:end]});start=end
    return out


def encoded_target(tokenizer,row,context_limit):
    messages=row['messages'];prefix=tokenizer.apply_chat_template(messages[:-1],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    full=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,enable_thinking=False)
    rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,enable_thinking=False)
    before=tokenizer.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    offsets=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
    if (full[:len(prefix)]!=prefix or len(full)<=len(prefix) or len(full)>context_limit
        or offsets['input_ids']!=full or not rendered.startswith(before)
        or not rendered[len(before):].startswith(row['answer'])):raise ValueError('Full teacher prefix/offsets differ; truncation forbidden')
    start=len(before);end=start+len(row['answer'].split('。')[0]+'。')
    end=min(end,start+len(row['answer']))
    selected=[bool(b>start and a<end) for a,b in offsets['offset_mapping'][len(prefix):]]
    if not any(selected):raise ValueError('Decision sentence has no supervised tokens')
    segments=target_segments(row);indices=[]
    for a,b in offsets['offset_mapping'][len(prefix):]:
        indices.append(next((i for i,s in enumerate(segments) if start+s['start']<=a<start+s['end'] and b>start),None))
    return {'prefix':prefix,'full':full,'decision_mask':selected,'answer_start_character':start,
            'decision_end_character':end,'target_offsets':offsets['offset_mapping'][len(prefix):],
            'target_segments':segments,'token_segment_indices':indices}


def score(model,tokenizer,row,book,c):
    import torch
    import torch.nn.functional as F
    if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Teacher-forced diagnostic requires a frozen eval model')
    e=encoded_target(tokenizer,row,c['inference_max_context']);n=len(e['prefix']);target=e['full'][n:];length=len(target)
    ids=torch.tensor([e['full'][:-1]],device=model.device);labels=torch.tensor(target,device=model.device)
    with torch.inference_mode(),book.generation_autocast(model):
        logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=length).logits[0].float()
        if logits.shape[0]!=length or not bool(torch.isfinite(logits).all()):raise RuntimeError('Invalid target-only diagnostic logits')
        losses=F.cross_entropy(logits,labels,reduction='none');predicted=logits.argmax(-1)
    values=losses.cpu().tolist();choices=predicted.cpu().tolist();decision=[v for v,m in zip(values,e['decision_mask'],strict=True) if m]
    rest=[v for v,m in zip(values,e['decision_mask'],strict=True) if not m]
    segments=[]
    for i,segment in enumerate(e['target_segments']):
        matched=[v for v,j in zip(values,e['token_segment_indices'],strict=True) if j==i]
        segments.append(segment|{'supervised_tokens':len(matched),'mean_nll':sum(matched)/len(matched) if matched else None})
    return {'id':row['id'],'target_sha256':d.digest(row['answer']),'prompt_sha256':d.digest(d.dumps(row['messages'][:-1])),
        'all_supervised_mean_nll':sum(values)/length,'decision_sentence_mean_nll':sum(decision)/len(decision),
        'remaining_target_mean_nll':sum(rest)/len(rest) if rest else None,'supervised_tokens':length,'decision_tokens':len(decision),
        'teacher_segments':segments,'segment_boundary_tokens_assigned_by_start_offset':True,
        'tokens':[{'target_id':t,'target_text':tokenizer.decode([t]),'nll':v,'gold_prefix_top1_id':a,
                   'gold_prefix_top1_equals_target':a==t,'in_decision_sentence':m,'character_offsets':list(offset),'teacher_segment_index':j}
                  for t,v,a,m,offset,j in zip(target,values,choices,e['decision_mask'],e['target_offsets'],e['token_segment_indices'],strict=True)],
        'conditional_on_gold_prefix':True,'free_generation_correctness_verified':False,'professional_probability':False,'weight_updates':0}
