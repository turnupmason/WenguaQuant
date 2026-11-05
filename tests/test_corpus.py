import unittest,tempfile
from pathlib import Path
from wengua_corpus import book_rag as r
from wengua_corpus.book_rag_editorial import make_edition,replay,build_library
class Corpus(unittest.TestCase):
 def setUp(self):
  (r.ROOT/'local').mkdir(exist_ok=True);self.tmp=tempfile.TemporaryDirectory(dir=r.ROOT/'local')
 def tearDown(self):self.tmp.cleanup()
 def source(self):
  raw='软件规 则甲，须乙并不得丙。\n软件疑字。';e=make_edition(raw,[],prose_only=True)
  return build_library(raw,e,{'id':'fixture','title':'软件夹具','method_id':'fixture_only'},'1','测试章',[{'id':'one','title':'软件规则','cleaned_lines':[[1,1]],'search_cues':['软件规则条件']},{'id':'two','title':'软件疑字','cleaned_lines':[[2,2]],'uncertainties':['未核']}])
 def test_literal_correction_projection_and_raw_preserved(self):
  v=self.source();c=v['chapters'][0];text,_=replay(c['original_text'],c['editorial_edition']);self.assertIn('规 则',c['original_text']);self.assertIn('规则',text);r.validate_library(v)
 def test_new_index_cpu_and_unknown_text_kept_out_of_rules(self):
  p=Path(self.tmp.name)/'index.sqlite3';r.build_index([self.source()],p)
  with r.Index(p) as i:
   self.assertTrue(i.search('软件规则条件')['documents']);self.assertTrue(all(d['kind']=='rule' for d in i.search('软件疑字')['documents']));self.assertTrue(any(d['kind']=='reference' for d in i.search('软件疑字',mode='study')['documents']))
 def test_prior_index_cannot_be_overwritten(self):
  p=Path(self.tmp.name)/'index.sqlite3';r.build_index([self.source()],p)
  with self.assertRaises(ValueError):r.build_index([self.source()],p)
 def test_edited_content_tamper_rejected(self):
  v=self.source();v['chapters'][0]['editorial_edition']['cleaned_reading_text']+='假';self.assertRaises(ValueError,r.validate_library,v)
 def test_layout_sensitive_source_needs_explicit_prose_scope(self):
  self.assertRaises(ValueError,make_edition,'图表',[])
