"""Synthetic regressions, including the actual V54 failure *shapes*."""
import unittest
from copy import deepcopy
from unittest.mock import patch
from pathlib import Path
import tempfile
from wengua_course import data as d
from wengua_course_v55 import data,checks,sourceplan


def gold(kind='relation',**kw):
 return {'checks':{'kind':kind,**kw},'messages':[{'role':'system','content':'source only'},{'role':'user','content':'question'},{'role':'assistant','content':''}],'provenance':{}}


class DirectedReasoningV55(unittest.TestCase):
 def test_all_five_relations_and_named_subjects(self):
  representative={'水':1,'土':2,'木':3,'金':6,'火':9}
  for a,pa in representative.items():
   for b,pb in representative.items():
    rel,why=data.word_relation(a,b,'A','B');self.assertEqual(rel,d.relation(pa,pb))
    self.assertEqual(checks.fact_check({'answer':rel+'。'+why},gold(answer=rel,elements=[a,b]))['status'],'pass')
 def test_correct_headline_and_valid_cycle_do_not_hide_wrong_subject(self):
  g=gold(answer='受克',elements=['水','土'])
  self.assertEqual(checks.fact_check({'answer':'受克。土克水，是前者克后者。'},g)['status'],'fail')
  self.assertEqual(checks.fact_check({'answer':'受克。土克水，是后者克前者。'},g)['status'],'pass')
 def test_received_relationship_with_wrong_subject_is_rejected(self):
  g=gold(answer='生',elements=['水','木'])
  self.assertEqual(checks.fact_check({'answer':'生。水生木，所以前者受生。'},g)['status'],'fail')
 def test_negated_error_and_positive_correction_accepted(self):
  self.assertEqual(checks.fact_check({'answer':'不是生，是受克。土克水，后者克前者。'},gold(answer='受克',elements=['水','土']))['status'],'pass')
 def test_invalid_element_cycle_rejected(self):
  self.assertEqual(checks.fact_check({'answer':'受生。木生金。'},gold(answer='受生',elements=['金','木']))['status'],'fail')
 def test_correct_positions_with_wrong_direction_fails(self):
  g=gold('joint_relation',answer='克',positions={'心':7,'辅':3},elements=['金','木'])
  self.assertEqual(checks.fact_check({'answer':'受生。心星在兑七宫属金，辅星在震三宫属木；木生金。'},g)['status'],'fail')
  self.assertEqual(checks.fact_check({'answer':'克。心星在兑七宫属金，辅星在震三宫属木；金克木。'},g)['status'],'pass')


class NaturalNamesAndBoundariesV55(unittest.TestCase):
 def test_second_sentence_name_and_is_wrapper_accepted(self):
  g=gold('exact_name',answer='网盖天牢',pair='癸辛')
  for a in ['天盘癸加地盘辛，是网盖天牢。','天盘癸加地盘辛，前者为天盘，后者为地盘。本条称为网盖天牢。']:
   self.assertEqual(checks.fact_check({'answer':a},g)['status'],'pass')
 def test_correct_name_with_false_explanation_rejected(self):
  g=gold('exact_name',answer='fixture',pair='壬壬')
  self.assertEqual(checks.fact_check({'answer':'本条称为fixture。天盘壬克地盘壬。'},g)['status'],'fail')
 def test_alternative_names_not_unique_answer(self):
  self.assertNotEqual(checks.fact_check({'answer':'命名取fixture、other。'},gold('exact_name',answer='fixture',pair='甲乙'))['status'],'pass')
 def test_explicit_wrong_layer_still_rejected(self):
  self.assertEqual(checks.fact_check({'answer':'天盘戊加地盘己，是伏吟。'},gold('exact_name',answer='伏吟',pair='戊戊'))['status'],'fail')
 def test_unknown_birth_year_cannot_be_used(self):
  g=gold('role',answer='日干',opposite='时干');g['messages'][-2]['content']='考生本人问考试，不知出生年，先取谁？'
  self.assertEqual(checks.fact_check({'answer':'取日干；不知道出生年也可兼看年命。'},g)['status'],'fail')
  self.assertEqual(checks.fact_check({'answer':'取日干；有可靠出生年再兼看年命。'},g)['status'],'pass')
  self.assertEqual(checks.fact_check({'answer':'取日干代表考生。'},g)['status'],'pass')
 def test_requesting_missing_information_is_not_fabricating_it(self):
  g=gold('role',answer='年干',opposite='日干');g['messages'][-2]['content']='女儿代问父亲，未提供出生年。'
  self.assertEqual(checks.fact_check({'answer':'先看年干，兼乾宫；需要兼看年命时，补充可靠出生年。'},g)['status'],'pass')
 def test_marriage_needs_explicit_auspicious_conditions(self):
  g=gold('condition_boundary',required=[['不能保证'],['吉门'],['吉格']],forbidden=[])
  self.assertNotEqual(checks.fact_check({'answer':'不能保证，应综合其他条件。'},g)['status'],'pass')
  self.assertEqual(checks.fact_check({'answer':'不能保证。本条还要求吉门、吉格。'},g)['status'],'pass')
 def test_negated_overclaim_is_not_a_positive_overclaim(self):
  g=gold('condition_boundary',required=[['吉门'],['吉格']],forbidden=['一定成功'])
  self.assertEqual(checks.fact_check({'answer':'有吉门吉格还须结合本题条件，不能说一定成功。'},g)['status'],'pass')
  self.assertNotEqual(checks.fact_check({'answer':'有吉门吉格，所以一定成功。'},g)['status'],'pass')


class SourceCoverageV55(unittest.TestCase):
 def test_explicit_book_scope_reaches_base_search_and_workflow_selection(self):
  from wengua_course_v55 import corpus
  from unittest.mock import MagicMock
  for title,book in [('开悟之门','kaiwu'),('神奇之门','shenqi')]:
   with patch.object(corpus.base,'retrieve',return_value={'documents':[]}) as search,patch.object(corpus,'source_plan',return_value={'workflows':[]}),patch.object(corpus.base,'ensure_index',return_value=None),patch.object(corpus.r,'Index',return_value=MagicMock()):
    corpus.retrieve('按《'+title+'》疾病流程',domain='health')
    self.assertEqual(search.call_args.kwargs['book'],book)
 def test_unclear_correction_phase_is_review_only(self):
  self.assertFalse(data.usable_author_phase('原书记录中的分析（本案有按语或更正，人物及阶段不能混用）'))
  self.assertFalse(data.usable_author_phase('更正或补充说明之前的分析'))
  self.assertTrue(data.usable_author_phase('原书后段的更正或补充分析'))
 def library(self):
  raw='用神取法需先辨所问的具体对象，不能脱离当前问题领域和实际条件直接套用。\n关系判断还要核对双方的已知条件，不能只看单一符号名称便直接决定结论。';sha=d.digest(raw);parts=[]
  for i,line in enumerate(raw.splitlines(),1):parts.append({'chapter':'shenqi/1','lines':[i,i],'chars':[0,len(line)],'quote_sha256':d.digest(line),'source_sha256':sha})
  docs=[{'id':'r'+str(i),'kind':'rule','case_ids':[],'domains':['health'],'parts':[p],'cues':['来源范围。\n本段怎样保留适用条件？'],'parent_source_ids':['s'+str(i)],'review_status':'reviewed_source_selector'} for i,p in enumerate(parts)]
  return {'chapters':[{'book_id':'shenqi','chapter_id':'1','title':'synthetic','original_text':raw,'source_sha256':sha}],'documents':docs}
 def test_every_rule_has_disposition_and_exact_source_selectors(self):
  lib=self.library();lib['documents'].append({'id':'general','kind':'rule','case_ids':[],'domains':[],'parts':lib['documents'][0]['parts'],'cues':[]})
  p=sourceplan.plan(lib);self.assertEqual(len(p['rule_inventory']),3);self.assertEqual(len(p['rules']),2);self.assertEqual(len(p['workflows']),1)
  self.assertEqual(p['workflows'][0]['document']['parts'],[lib['documents'][0]['parts'][0],lib['documents'][1]['parts'][0]])
 def test_case_rule_cannot_be_promoted_to_general_workflow(self):
  lib=self.library();lib['documents'][0]['case_ids']=['event'];p=sourceplan.plan(lib);self.assertEqual(len(p['rules']),1);self.assertIn('case_scoped',p['rule_inventory'][0]['disposition'])
 def test_title_or_missing_reviewed_question_not_turned_into_teacher(self):
  lib=self.library();lib['documents'][0]['cues']=[];p=sourceplan.plan(lib);self.assertIn('no_reviewed',p['rule_inventory'][0]['disposition'])


class ReviewedParentV55(unittest.TestCase):
 def test_five_star_states_are_not_lost_in_large_development_pool(self):
  from wengua_course_v55 import runtime
  rows=[{'id':'s'+str(i),'task':'source_star_state_'+str(i),'skill':'source_fact','split':'validation','provenance':{'key':'star_state_'+str(i)},'checks':{'kind':'state','answer':'fixture'}} for i in range(5)]
  rows+=[{'id':'w'+str(i),'task':'workflow','skill':'workflow','split':'validation','domain':'foundation','provenance':{},'checks':None} for i in range(600)]
  selected=runtime.choose_development({'records':rows},192)
  self.assertTrue({'s0','s1','s2','s3','s4'}.issubset({r['id'] for r in selected}))
 def test_default_uses_reviewed_first_epoch_and_checks_hash(self):
  from wengua_course_v55 import initialization,runtime
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);d.write(root/'run_status.json',{'state':'complete_review_required','all_planned_groups_finished':True,'trials':{'source_lr_2em05':{'epochs':[{'steps':134,'epoch':1.0},{'steps':268,'epoch':2.0}]}}});reader=d.read
   def read(path):return {'reviewed_parent_adapter_sha256':'expected'} if path==runtime.RELEASE else reader(path)
   with patch.object(d,'read',read),patch.object(initialization,'inspect',lambda path:{'files':{'adapter_model.safetensors':'expected'}}):
    self.assertEqual(initialization.default_source(root),root/'trials/source_lr_2em05/cycle_01/adapters/step-134')
   with patch.object(d,'read',read),patch.object(initialization,'inspect',lambda path:{'files':{'adapter_model.safetensors':'different'}}):
    with self.assertRaises(ValueError):initialization.default_source(root)


if __name__=='__main__':unittest.main()
