"""Literal source selectors migrated from WenguaApp; no model imports."""
from . import book_rag as rag

def compact(x):return ''.join(x.split())

def literal_selector(chapter,lo,hi,quote=None):
    lines=chapter['original_text'].splitlines();raw='\n'.join(lines[lo-1:hi])
    if not raw:raise ValueError('Empty source selector')
    if quote is None:a,b=0,len(raw)
    else:
        normalized=[];positions=[]
        for i,char in enumerate(raw):
            if not char.isspace():normalized.append(char);positions.append(i)
        key=compact(quote);text=''.join(normalized);offset=text.find(key)
        if not key or offset<0:raise ValueError('Reviewed excerpt not literal at its source coordinates')
        a,b=positions[offset],positions[offset+len(key)-1]+1
    return {'chapter':rag.chapter_key(chapter),'source_sha256':chapter['source_sha256'],'lines':[lo,hi],
            'chars':[a,b],'quote_sha256':rag.digest(raw[a:b])}
