"""Verify saved SFT tensors against the independently rendered generation prefix."""
from wengua_course import data as d


def verify(rows,tokenizer):
    count=targets=0;receipts=[]
    for row in rows:
        messages=row['messages'];encoded=row['encoded']
        prefix=tokenizer.apply_chat_template(messages[:-1],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        full=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,enable_thinking=False)
        expected=[-100]*len(prefix)+full[len(prefix):]
        if full[:len(prefix)]!=prefix or encoded['input_ids']!=full or encoded['labels']!=expected or not full[len(prefix):]:
            raise ValueError('Training mask or target differs from the actual generation prefix: '+row['id'])
        count+=1;targets+=len(full)-len(prefix);receipts.append([row['id'],d.digest(d.dumps(prefix)),d.digest(d.dumps(expected))])
    return {'complete_supervised_views':count,'supervised_tokens':targets,'all_non_thinking_prefixes_equal':True,
            'all_prompt_tokens_masked':True,'whole_assistant_targets_preserved':True,'view_receipts_sha256':d.digest(d.dumps(receipts))}
