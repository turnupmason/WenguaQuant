"""Synthetic nested-source routing: identical headings are not identical roles."""
import tempfile,unittest
from pathlib import Path
from wengua_corpus import book_rag as r
from wengua_library import intake as i
from wengua_library.__main__ import query

BOOK={'title':'合成同名篇','method_id':'synthetic_not_runtime'}
def profile(contexts):return {'namespace':'jinhan.batch05','chapter_id':'synthetic_nested','title':'合成嵌套材料','main_headings':[],
 'subheadings':[],'structural_headings':['论值符','论值使','论干支'],
 'domains':{'兵占·论值符':['military_history_source'],'兵占·论值使':['military_history_source'],
            '兵占·论干支':['military_history_source'],'官禄占·论干支':['employment']},
 'heading_contexts':contexts,'study_sections':[],'connected_evidence_required':True,'source_ordered_contexts':True}

class NestedSourceTests(unittest.TestCase):
 def setUp(self):(r.ROOT/'local').mkdir(exist_ok=True)
 def test_identical_headings_keep_domain_specific_role_bindings(self):
  source='论干支\n日干为我方，时干为他方。\n论干支\n日干为官，时干为印。\n'
  p=i.compile_source(source,BOOK,{'source_profile':profile({'1':{'section':'兵占·论干支','source_mode':'original'},'3':{'section':'官禄占·论干支','source_mode':'original'}})})
  bodies=[u for u in p['units'] if u['scope']=='source_rule']
  self.assertEqual([u['section'] for u in bodies],['兵占·论干支','官禄占·论干支'])
  self.assertEqual(len(p['workflows']),2)
  with tempfile.TemporaryDirectory(dir=r.ROOT/'local') as directory:
   path=Path(directory)/'index.sqlite3';r.build_index([p['library']],path)
   with r.Index(path) as index:result=query(index,'日干时干',section='官禄占·论干支',book='jinhan')
   self.assertTrue(result['documents']);self.assertTrue(all('我方' not in d['content'] for d in result['documents']))
 def test_inline_heading_tail_is_preserved_while_next_body_changes_topic(self):
  source='论值符\n原文甲。论值使\n原文乙。\n【白话译释】\n译文甲。论值使\n译文乙。\n'
  contexts={'1':{'section':'兵占·论值符','source_mode':'original'},'3':{'section':'兵占·论值使','source_mode':'original'},
            '5':{'section':'兵占·论值符','source_mode':'translation'},'6':{'section':'兵占·论值使','source_mode':'translation'}}
  p=i.compile_source(source,BOOK,{'source_profile':profile(contexts)})
  byline={u['source_lines'][0]:u for u in p['units']}
  self.assertEqual(byline[2]['text'],'原文甲。论值使')
  self.assertEqual(byline[3]['section'],'兵占·论值使')
  self.assertEqual(byline[5]['source_mode'],'translation')
  self.assertEqual(byline[6]['source_mode'],'translation')
  self.assertEqual(p['library']['chapters'][0]['original_text'],source)
  self.assertEqual(p['report']['unaccounted_nonblank_lines'],0)
 def test_translation_section_override_does_not_reset_to_original(self):
  source='论干支\n原句甲。\n【白话译释】\n论干支\n译句乙。\n'
  contexts={'1':{'section':'兵占·论干支','source_mode':'original'},'4':{'section':'兵占·论干支','source_mode':'translation'}}
  p=i.compile_source(source,BOOK,{'source_profile':profile(contexts)})
  u=next(u for u in p['units'] if u['source_lines'][0]==5)
  self.assertEqual(u['source_mode'],'translation');self.assertTrue(u['required_context_ids'])
 def test_missing_override_anchor_is_rejected_instead_of_losing_topic(self):
  with self.assertRaisesRegex(ValueError,'does not start a material'):
   i.compile_source('论干支\n合成原文。\n',BOOK,{'source_profile':profile({'999':{'section':'兵占·论干支'}})})
 def test_same_topic_in_different_chapters_can_be_selected_without_mixing(self):
  a=profile({});a.update(chapter_id='first',domains={'占借贷':['business_profit']})
  b=profile({});b.update(namespace='jinhan.batch06',chapter_id='second',domains={'占借贷':['business_profit']})
  first=i.compile_source('占借贷\n甲为求者，乙为出借者。\n',BOOK,{'source_profile':a})
  second=i.compile_source('占借贷\n丙为求者，丁为出借者。\n',BOOK,{'source_profile':b})
  with tempfile.TemporaryDirectory(dir=r.ROOT/'local') as tmp:
   path=Path(tmp)/'index.sqlite3';r.build_index([first['library'],second['library']],path)
   with r.Index(path) as index:
    result=query(index,'求者出借者',book='jinhan',section='占借贷',chapter='jinhan/second')
    self.assertTrue(result['documents'])
    self.assertTrue(all('丙为求者' in d['content'] and '甲为求者' not in d['content'] for d in result['documents']))
 def test_invalid_chapter_is_not_silently_replaced_by_a_different_method(self):
  p=i.compile_source('论干支\n甲乙合成材料。\n',BOOK,{'source_profile':profile({'1':{'section':'兵占·论干支'}})})
  with tempfile.TemporaryDirectory(dir=r.ROOT/'local') as tmp:
   path=Path(tmp)/'index.sqlite3';r.build_index([p['library']],path)
   with r.Index(path) as index:
    with self.assertRaisesRegex(ValueError,'Unknown source chapter'):query(index,'甲乙',chapter='jinhan/absent')

if __name__=='__main__':unittest.main()
