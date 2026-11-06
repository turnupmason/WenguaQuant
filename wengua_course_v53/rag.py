"""Use the already curated TWO-book index and the original App casting code."""
from pathlib import Path
from copy import deepcopy
import gzip,json,os,time,re,importlib.util,sys
from . import runtime
from wengua_course import data as d
from wengua_corpus import book_rag as original
# Independent tokenizer namespace; existing corpus indexes remain untouched.
_spec=importlib.util.spec_from_file_location('wengua_corpus._v53_source_retriever',original.__file__)
r=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=r;_spec.loader.exec_module(r)
r.INDEX_FORMAT+=':explicit_symbols_and_source_rule_classification_v53_v4'
r.SYMBOLS=(*r.SYMBOLS,'六合','太阴','九天','九地','螣蛇','玄武','白虎','勾陈','太常')
r.DOMAINS['business_profit']=(*r.DOMAINS['business_profit'],'买卖','交易')
r.DOMAINS.update(formation=(),foundation=())  # Explicit filters, no broad automatic trigger.
_native_terms=r.terms

def explicit_symbols(text):
 stems='甲乙丙丁戊己庚辛壬癸';branches='子丑寅卯辰巳午未申酉戌亥';found=set()
 for pattern in [rf'(?:天干|天盘|地盘|上层|下层|年命|年干|月干|日干|时干)\s*([{stems}])',rf'(?:^|[，、：:;；\s(（])([{stems}])(?=$|[，、。；;\s]|是|属|为|的|在|落|奇)',rf'地支\s*([{branches}])',rf'([{branches}])(?:时|月|日|宫)']:
  found.update(re.findall(pattern,text))
 return sorted(found)

def lexical_terms(text):return list(dict.fromkeys([*_native_terms(text),*explicit_symbols(text)]))
r.terms=lexical_terms
_BaseIndex=r.Index
class ContextIndex(_BaseIndex):
 def search(self,question,**kwargs):
  limit=kwargs.get('limit',6)
  if type(limit)is not int or not 1<=limit<=20:raise ValueError('Retrieval limit must be between 1 and 20')
  kwargs['limit']=20;found=super().search(question,**kwargs);symbols=explicit_symbols(question)
  domains=r.domains_for(question,kwargs.get('domain'))
  if len(domains)>1:
   # A question may explicitly change domains on the same board. Preserve
   # evidence from each requested domain instead of letting one fill the pool.
   groups=[super(ContextIndex,self).search(question,**(kwargs|{'domain':domain,'limit':3}))['documents'] for domain in domains]
   diversified=[group[n] for n in range(3) for group in groups if len(group)>n]
   found['documents']=list({doc['id']:doc for doc in [*diversified,*found['documents']]}.values())
  attribute_question=any(w in question for w in ('阴','阳','五行','属于','属什么','方向'))
  if symbols and attribute_question and not r.ordered_pairs(question):
   found['documents'].sort(key=lambda doc:(sum(s not in doc.get('content','') for s in symbols),doc['retrieval_rank']))
  found['documents']=found['documents'][:limit]
  for n,doc in enumerate(found['documents'],1):doc['retrieval_rank']=n
  return found
r.Index=ContextIndex

def classified_library(library):
 """Versioned metadata correction; literal book text and selectors unchanged."""
 library=deepcopy(library);chapters={r.chapter_key(c):c for c in library['chapters']};changes=[]
 for doc in library['documents']:
  pair_cues=[x for x in doc['cues'] if x.startswith('stem_pair:')]
  if doc['kind']!='reference' or doc['case_ids'] or len(pair_cues)!=1:continue
  _,a,b=pair_cues[0].split(':');text=r.document_text(chapters,doc).lstrip()
  if not re.match(re.escape(a)+r'\s*[/＋+加]\s*'+re.escape(b),text):continue
  changes.append({'id':doc['id'],'previous_kind':'reference','kind':'rule','reason':'literal_stem_pair_definition_with_matching_explicit_source_cue','parts':deepcopy(doc['parts'])})
  doc.update(kind='rule',phase='general_rule',domains=['formation'],topic='formation',v53_classification_reason=changes[-1]['reason'])
 return library,changes

def index_path():return runtime.ROOT/'local/rag/two_books_v53_v4.sqlite3'
def ensure_index():
 library=json.loads(gzip.decompress((runtime.DATA/'rag_sources.json.gz').read_bytes()))
 if set(library['books'])!={'shenqi','kaiwu'}:raise ValueError('Only the two approved books belong in this run')
 path=index_path()
 if not path.exists():
  indexed,changes=classified_library(library)
  proof=r.build_index([indexed],path)
  d.write(path.with_suffix('.classification.json'),{'source_library_sha256':r.digest(r.dumps(library)),'changes':changes,'original_text_and_source_selectors_changed':False,'index_report':proof})
 with r.Index(path) as idx:
  if {c['book_id'] for c in idx.chapters.values()}!={'shenqi','kaiwu'}:raise ValueError('Non-two-book index')
 return path

def search_query(question):
 query=re.sub(r'上层\s*([甲乙丙丁戊己庚辛壬癸])',r'天盘\1',question)
 query=re.sub(r'下层\s*([甲乙丙丁戊己庚辛壬癸])',r'地盘\1',query)
 if any(word in question for word in ('互换','倒换','交换')):
  for a,b in sorted(r.ordered_pairs(query)):query+=' 天盘'+b+'地盘'+a
 return query

def retrieve(question,*,domain=None,mode='knowledge',case_id=None,limit=8,book=None):
 if book is None:
  if '开悟之门' in question and '神奇之门' not in question:book='kaiwu'
  elif '神奇之门' in question and '开悟之门' not in question:book='shenqi'
 with r.Index(ensure_index()) as index:
  found=index.search(search_query(question),domain=domain,mode=mode,case_id=case_id,limit=limit,book=book)
  found['original_question']=question;found['retrieval_query']=search_query(question);return found

def prompt(question,tokenizer,*,domain=None,mode='knowledge',case_id=None,time_text=None,zone=None,chart_file=None,facts=None,max_new_tokens=1024,book=None):
 runtime.configure()
 from src.wengua.book_rag_query import cast_time,checked_chart,make_prompt
 if chart_file and time_text:raise ValueError('Choose an existing program receipt or a time')
 packet=checked_chart(d.read(chart_file)) if chart_file else None
 if time_text:
  if not zone:raise ValueError('Explicit casting timezone is required for the existing calendar')
  packet=cast_time(time_text,zone)
 if packet is not None and mode=='knowledge':mode='reading'
 found=retrieve(question,domain=domain,mode=mode,case_id=case_id,limit=20,book=book)
 with r.Index(ensure_index()) as index:
  for doc in found['documents']:
   for source in doc['sources']:
    source['chapter_id']=source['chapter'];source['chapter']=index.chapters[source['chapter_id']]['title']
 result=make_prompt(found,question,tokenizer,packet=packet,facts=facts,max_context=4096,max_new_tokens=max_new_tokens)
 result['two_books_only']=True;return result

def answer(p,job,c,deadline,packet,adapter,*,factory=None):
 import torch
 from transformers import AutoTokenizer
 from peft import PeftModel
 from src.wengua.qwen_finetune import prepare_frozen_base
 from src.wengua.book_rag_query import check_answer
 if packet['state']!='ready':return {'state':'insufficient_retrieved_evidence','prompt':packet,'GPU_answer_generated':False}
 if factory:tok,base,_=factory()
 else:
  tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision'],local_files_only=True,trust_remote_code=False);base,_=p.book.load_book_cuda(c,False)
 base=prepare_frozen_base(base,gradient_checkpointing=False);model=PeftModel.from_pretrained(base,adapter,is_trainable=False).eval()
 tokens=tok.apply_chat_template(packet['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False)
 if len(tokens)+packet['prompt_budget']['max_new_tokens']>c['inference_max_context']:raise ValueError('Unbudgeted RAG prompt')
 ids=torch.tensor([tokens],device=model.device)
 with torch.inference_mode(),p.book.generation_autocast(model):
  output=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,max_new_tokens=packet['prompt_budget']['max_new_tokens'],logits_to_keep=1,pad_token_id=tok.pad_token_id or tok.eos_token_id)
 raw=tok.decode(output[0,len(tokens):],skip_special_tokens=True)
 trace={'state':'complete','question':packet['question'],'answer':raw,'prompt':packet,'checks':check_answer(raw,packet),'adapter':str(adapter),'generated_tokens':int(output.shape[-1]-len(tokens)),'production_enabled':False,'semantic_support_verified':False}
 path=Path(job)/'rag_answers'/f'{time.time_ns()}.json';d.write(path,trace);return trace|{'trace_path':str(path)}

def actual_probe(p,job,c,deadline,adapter,tag,*,factory=None):
 """Real retrieval + generation, separate from supplied-gold-source development.
 Reference coverage and citation membership never certify answer semantics.
 """
 import torch
 from transformers import AutoTokenizer
 from peft import PeftModel
 from src.wengua.qwen_finetune import prepare_frozen_base
 from src.wengua.book_rag_query import check_answer
 if factory:return {'state':'software_fixture_skipped_real_book_probe','real_book_probe_executed':False}
 tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision'],local_files_only=True,trust_remote_code=False)
 specs=d.read(runtime.DATA/'manual_source_specs.json');pack=runtime.verify_data();units=pack['units'];wanted={'hetu_金','gui_yin_water','named_pair_戊戊','named_pair_丁壬','exam_self','exam_proxy','health_daughter_father','health_father_self','role_6','role_7','role_18','role_20','role_21','role_22','role_29','cross_domain_liuhe','marriage_same_palace','named_pair_庚戊'};prompts=[];packets={};coverage={}
 for spec in specs:
  if spec['key'] not in wanted:continue
  books={units[s['source_id']]['book_id'] for s in spec['support']};book=next(iter(books)) if len(books)==1 else None;domain=spec['domain'] if spec['domain'] in r.DOMAINS else None
  packet=prompt(d.DOMAIN_ZH.get(spec['domain'],spec['domain'])+'：'+spec['question'],tok,domain=domain,book=book,max_new_tokens=c['eval_max_new_tokens'])
  rid='v53:actual_rag:'+spec['key'];packets[rid]=packet
  covered=[]
  for support in spec['support']:
   unit=units[support['source_id']];covered.append(any(s['book_id']==unit['book_id'] and s.get('chapter_id',s['chapter'])==unit['book_id']+'/'+str(unit['section_id']) and s['lines'][0]<=unit['lines'][0] and s['lines'][1]>=unit['lines'][1] and s['source_sha256']==unit['source_sha256'] for doc in packet['references'] for s in doc['sources']))
  coverage[rid]={'all_declared_source_spans_in_visible_prompt':all(covered),'source_fragments':len(covered),'covered_fragments':sum(covered),'prompt_state':packet['state']}
  if packet['state']!='ready':continue
  prompts.append({'id':rid,'format':'foundation_qa','task':'actual_rag_reference_application','group':'actual_rag:'+spec['key'],'source_mode':'actual_retrieval_not_oracle; semantic_review_required','source_ids':[x['id'] for x in packet['references']],'messages':packet['messages']+[{'role':'assistant','content':spec['answer']}],'answer':spec['answer']})
 gen=Path(job)/'generations';gen.mkdir(exist_ok=True);traces=Path(job)/'rag_probe_sources';traces.mkdir(exist_ok=True);meta={'adapter_sha256':d.digest((Path(adapter)/'adapter_model.safetensors').read_bytes()),'index_sha256':r.digest(index_path().read_bytes()),'coverage':coverage,'packets':packets};path=traces/(tag+'.json')
 if path.exists() and d.read(path)!=meta:raise ValueError('RAG source or adapter changed within this probe')
 if not path.exists():d.write(path,meta)
 base,_=p.book.load_book_cuda(c,False);base=prepare_frozen_base(base,gradient_checkpointing=False);model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False)
 result=p.book.generate_rows(model,tok,prompts,gen/(tag+'.jsonl'),c,deadline,tag)
 outputs=[json.loads(line) for line in (gen/(tag+'.jsonl')).read_text().splitlines()];checked=[{'id':x['id'],'checks':check_answer(x['answer'],packets[x['id']]),'reached_token_limit':x['reached_token_limit']} for x in outputs]
 final=result|{'source_coverage':coverage,'citation_and_chart_checks':checked,'real_book_probe_executed':True,'semantic_support_verified':False,'professional_approval':False};d.write(traces/(tag+'.checks.json'),final);return final
