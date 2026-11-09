"""Literal retrieval, question-specific scope, equivalent source coverage."""
from wengua_course_v55.corpus import retrieve,covering_citations
from wengua_course_v55.corpus import prompt as original_prompt
from .data import SYSTEM,clean

def prompt(*args,**kwargs):
    packet=original_prompt(*args,**kwargs)
    if packet.get('messages'):
        packet['messages'][0]['content']=SYSTEM
        if not any(kwargs.get(k) for k in ['time_text','chart_file','facts']):
            for m in packet['messages'][1:]:m['content']=clean(m['content'])
        tok=args[1] if len(args)>1 else kwargs['tokenizer'];length=len(tok.apply_chat_template(packet['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False))
        if length+packet['prompt_budget']['max_new_tokens']>kwargs.get('max_context',4096):raise ValueError('Whole question and sources exceed budget')
        packet['prompt_budget'].update(input_tokens=length,v60_question_scope_recounted=True)
    return packet

def citation_groups(packet,key):
    anchors={'gui_yin_water':[['癸','阴水','北方']],'hetu_金':[['地四生金','天九成之']],'cross_domain_liuhe':[['六合','媒人'],['六合','证人'],['六合','经纪人']],'marriage_same_palace':[['相生','吉门吉格']],'role_6':[['天芮','疾病']],'role_7':[['天心','乙奇','医药']],'role_18':[['甲子戊','资本']],'role_20':[['六合','经纪人']],'role_21':[['日干','失主']],'role_22':[['时干','丢失之钱物']],'role_29':[['惊门','律师']],'named_pair_戊戊':[['戊/戊','伏吟']],'named_pair_丁壬':[['丁/壬','奇仪相合']],'named_pair_庚戊':[['庚/戊','值符伏宫']],'exam_self':[['考生本人求测','日干']],'exam_proxy':[['父母求测子女','时干']],'health_daughter_father':[['儿女辈求测长辈','年干','乾宫']],'health_father_self':[['患者自己求测','日干']]}
    result=[]
    for terms in anchors[key]:
        choices=[r['citation'] for r in packet['references'] if all(x in r['content'] for x in terms)]
        if not choices:raise ValueError('Retrieved sources lack a required rule group: '+key+str(terms))
        result.append(choices)
    return result
