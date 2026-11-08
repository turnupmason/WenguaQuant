"""Actual two-book retrieval and the existing App calendar, with V57 wording."""
from wengua_course_v55.corpus import retrieve,covering_citations
from wengua_course_v55.corpus import prompt as original_prompt
from .data import SYSTEM

def prompt(*args,**kwargs):
    packet=original_prompt(*args,**kwargs)
    if packet.get('messages'):
        packet['messages'][0]['content']=SYSTEM
        tok=args[1] if len(args)>1 else kwargs['tokenizer']
        length=len(tok.apply_chat_template(packet['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False))
        if length+packet['prompt_budget']['max_new_tokens']>kwargs.get('max_context',4096):raise ValueError('V57 system recount exceeds whole prompt budget')
        packet['prompt_budget']['input_tokens']=length;packet['prompt_budget']['v57_system_recounted']=True
    return packet
