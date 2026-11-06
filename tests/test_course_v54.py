"""Public synthetic regressions; no private book passages or model weights."""
import unittest
from copy import deepcopy
from unittest.mock import patch
from wengua_course import data as d
from wengua_course_v54 import corpus, data, checks


class WorkedDirections(unittest.TestCase):
 def test_all_directions_have_consistent_worked_cycles(self):
  representatives={'水':1,'土':2,'木':3,'金':6,'火':9}
  reverse={'生':'受生','受生':'生','克':'受克','受克':'克','比和':'比和'}
  for a,pa in representatives.items():
   for b,pb in representatives.items():
    rel=d.relation(pa,pb);answer=rel+'。'+data.bridge(a,b)
    self.assertEqual(d.relation(pb,pa),reverse[rel])
    self.assertFalse(checks.reasoning_conflicts(answer))
    self.assertEqual(checks.fact_check({'answer':answer},{'checks':{'kind':'relation','answer':rel,'elements':[a,b]}})['status'],'pass')
 def test_wrong_explanation_rejects_even_correct_first_word(self):
  self.assertEqual(checks.fact_check({'answer':'生。金生土。'},{'checks':{'kind':'relation','answer':'生','elements':['金','水']}})['status'],'fail')


class SourceSelectors(unittest.TestCase):
 def test_selector_preserves_original_spaces_and_coordinates(self):
  raw='前文\n乾 金\n后文';chapter={'original_text':raw,'source_sha256':d.digest(raw)}
  unit={'book_id':'shenqi','section_id':1,'lines':[2,2]}
  part=corpus.part_for(unit,'乾金',{'shenqi/1':chapter})
  self.assertEqual(part['chars'],[0,3]);self.assertEqual(part['quote_sha256'],d.digest('乾 金'))
 def test_ambiguous_or_missing_source_is_rejected(self):
  unit={'book_id':'shenqi','section_id':1,'lines':[1,1]}
  for text,quote in [('甲甲','甲'),('甲','乙')]:
   with self.assertRaises(ValueError):corpus.part_for(unit,quote,{'shenqi/1':{'original_text':text,'source_sha256':d.digest(text)}})
 def test_visible_word_without_actual_coordinate_coverage_is_not_evidence(self):
  text='甲乙\n丙丁';sha=d.digest(text);chap={'book_id':'shenqi','chapter_id':'1','original_text':text,'source_sha256':sha}
  unit={'book_id':'shenqi','section_id':1,'lines':[1,1],'original_quote':'甲乙','source_sha256':sha}
  ref={'citation':'S1','content':'甲乙','sources':[{'book_id':'shenqi','chapter_id':'shenqi/1','chapter':'标题','lines':[2,2],'source_sha256':sha}]}
  with patch.object(corpus,'source_library',lambda:{'chapters':[chap]}):
   self.assertIsNone(corpus.covering_citations({'references':[ref]},[{'source_id':'u','quote':'甲乙'}],{'u':unit}))
   ref['sources'][0]['lines']=[1,1]
   self.assertEqual(corpus.covering_citations({'references':[ref]},[{'source_id':'u','quote':'甲乙'}],{'u':unit}),['S1'])
 def test_typed_pair_variants_retrieve_ordered_pair(self):
  for q in ['天盘干为丁、地盘干为壬','上层明确是天盘丁，下层明确是地盘壬']:
   self.assertIn(('丁','壬'),corpus.r.ordered_pairs(corpus.search_query(q)))
  self.assertIn(('戊','戊'),corpus.r.ordered_pairs(corpus.search_query('上下层都是戊')))
 def test_property_queries_find_both_attributes_and_literal_relation_rules(self):
  self.assertIn('palace:6',corpus.property_keys('乾六宫归于金木水火土的哪一类'))
  keys=corpus.property_keys('景门落兑七宫的门宫关系')
  self.assertTrue({'door:景','palace:7','rule:generation','rule:control','rule:door_relation'}.issubset(keys))
  self.assertNotIn('rule:door_relation',corpus.property_keys('天芮星的含义'))


class FactAndCitationChecks(unittest.TestCase):
 def result(self,a,kind='exact_name',**kw):return checks.fact_check({'answer':a},{'checks':{'kind':kind,**kw}})['status']
 def test_normal_role_sentence_is_not_a_bare_label_requirement(self):
  self.assertEqual(self.result('本章取开门为法官。',answer='开门',role_label='法官'),'pass')
  self.assertEqual(self.result('取丁奇，因为本章以丁奇为文章。',answer='丁奇',role_label='文章'),'pass')
  self.assertNotEqual(self.result('开门或者惊门都可以。',answer='开门',role_label='法官'),'pass')
 def test_name_plus_reason_and_negative_names(self):
  self.assertEqual(self.result('天盘壬加地盘壬，本条称为天狱自刑。',answer='天狱自刑'),'pass')
  self.assertNotEqual(self.result('不是天狱自刑。',answer='天狱自刑'),'pass')
 def test_state_aliases_do_not_confuse_he_and_bihe(self):
  self.assertEqual(self.result('门迫。景门克兑宫。','state',answer='迫'),'pass')
  self.assertEqual(self.result('最旺。','state',answer='旺'),'pass')
  self.assertNotEqual(self.result('比和。','state',answer='和'),'pass')
 def test_correct_name_with_wrong_layer_is_rejected(self):
  self.assertEqual(self.result('本条称为伏吟。上层取年干。',answer='伏吟',pair='戊戊'),'fail')
  self.assertEqual(self.result('天盘戊加地盘己，本条称为伏吟。',answer='伏吟',pair='戊戊'),'fail')
 def test_citations_are_checked_separately_from_fact(self):
  g={'context':'provided_evidence','messages':[{'role':'user','content':'[S1] wrong [S2] source'},{'role':'assistant','content':'x'}],'provenance':{'required_citations':['S2']}}
  self.assertFalse(checks.citation_check({'answer':'正确事实。'},g)['ok'])
  self.assertFalse(checks.citation_check({'answer':'正确事实。[S1]'},g)['ok'])
  self.assertTrue(checks.citation_check({'answer':'待人工检查的事实。[S2]'},g)['ok'])


class DialogueAssembly(unittest.TestCase):
 def fixture(self):
  units={'u':{'book_id':'shenqi','original_quote':'synthetic source'}}
  spec={'key':'exam_switch_fixture','domain':'education','messages':[{'role':'user','content':'母亲代问女儿考试取谁？'},{'role':'assistant','content':'取时干代表考生。'},{'role':'user','content':'改为我本人问考试呢？'},{'role':'assistant','content':'改取日干代表本人。'}],'development_question':'现在是考生本人提问','answer':'改取日干代表本人。','support':[{'source_id':'u','quote':'synthetic source'}],'checks':{'kind':'role','answer':'日干','opposite':'时干'}}
  def prompt(q,tok,**kwargs):return {'messages':[{'role':'system','content':'natural source-only answers'},{'role':'user','content':q+' [S1] synthetic source'}],'references':[{'id':'s','citation':'S1','sources':[]}],'prompt_budget':{'max_context':kwargs['max_context']}}
  rows=[]
  with patch.object(corpus,'prompt',prompt),patch.object(corpus,'covering_citations',lambda *args:['S1']):data.add_dialogue(rows,[],spec,units,None)
  pack={'schema':data.SCHEMA,'source_books':['shenqi','kaiwu'],'units':units,'records':rows,'case_catalog':[],'original_case_splits':{}}
  return pack
 def test_first_and_current_identity_answers_preserved(self):
  pack=self.fixture();data.validate(pack);r=pack['records'][0]
  self.assertEqual([m['role'] for m in r['messages']],['system','user','assistant','user','assistant'])
  self.assertIn('母亲代问女儿',r['messages'][1]['content']);self.assertIn('时干',r['messages'][2]['content']);self.assertIn('我本人',r['messages'][3]['content']);self.assertIn('日干',r['messages'][4]['content'])
  self.assertTrue(all(b['max_context']==1536 for b in r['provenance']['turn_budgets']))
 def test_swapped_target_or_missing_system_rejected(self):
  for change in ['target','role']:
   pack=deepcopy(self.fixture());r=pack['records'][0]
   if change=='target':r['messages'][-1]['content']=r['answer']='取时干代表本人。'
   else:r['messages'][0]['role']='user'
   with self.assertRaises(ValueError):data.validate(pack)
 def test_audit_metadata_is_not_added_as_a_rule_condition(self):
  s={'key':'named_pair_戊戊','answer':'伏吟。','conditions':['规则练习，非新增历史案例']}
  self.assertNotIn('已核实',data.plain_answer(s))

if __name__=='__main__':unittest.main()
