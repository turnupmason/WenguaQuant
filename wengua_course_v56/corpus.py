"""V56 uses the existing two-book literal RAG and deterministic App casting."""
from wengua_course_v55.corpus import retrieve,covering_citations
from wengua_course_v55.corpus import prompt as original_prompt
from .data import SYSTEM

def prompt(*args,**kwargs):
 packet=original_prompt(*args,**kwargs)
 if packet.get('messages'):packet['messages'][0]['content']=SYSTEM
 return packet
