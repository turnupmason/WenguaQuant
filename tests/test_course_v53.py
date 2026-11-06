import unittest
from copy import deepcopy
from wengua_course import data as old
from wengua_course_v53 import data,checks,runtime

def fixture():
 units={'book:one':{'book_id':'shenqi','original_quote':'水生木、木生火、火生土、土生金、金生水。水克火、火克金、金克木、木克土、土克水。父母代问子女考试，以时干为考生。'},'book:two':{'book_id':'kaiwu','original_quote':'戊/戊，甲甲比肩，名谓伏吟。'}}
 parent={'units':units,'records':[],'source_books':['shenqi','kaiwu'],'case_catalog':[],'original_case_splits':{},'sealed_original_test_manifest':[]}
 spec={'key':'exam_proxy','domain':'education','format':'conditional_dialogue','question':'母亲代问女儿考试，不知出生年，取谁？','development_question':'父亲代问儿子考试，取谁？','answer':'取时干代表女儿，不补造年命。','support':[{'source_id':'book:one','quote':'父母代问子女考试，以时干为考生。'}],'checks':{'kind':'role','answer':'时干','opposite':'日干'},'case_ids':[],'messages':None}
 return parent,[spec]

def gold(kind='exact_name',answer='天芮星',context='closed_book'):
 return {'id':'x','checks':{'kind':kind,'answer':answer},'context':context,'provenance':{},'messages':[{'role':'system','content':'source only'},{'role':'user','content':'[S1] literal source'},{'role':'assistant','content':answer}],'answer':answer,'task':'fact'}

class CurriculumV53(unittest.TestCase):
 def test_gender_neutral_proxy_target_and_domain_preserved(self):
  p,m=fixture();pack=data.build(p,m)
  for r in pack['records']:
   if r['task']=='reviewed_exam_proxy':
    self.assertNotIn('代表女儿',r['answer']);self.assertIn('考试求学',r['messages'][1]['content']);self.assertIn('实际考生',r['answer'])
 def test_book_only_support_exact(self):
  p,m=fixture();m[0]['support'][0]['quote']='并不存在的原文'
  with self.assertRaises(ValueError):data.build(p,m)
 def test_no_train_test_case_reuse(self):
  p,m=fixture();p['units']['book:one']['book_id']='third_book'
  with self.assertRaises(ValueError):data.build(p,m)
 def test_all_five_directions_and_reverse(self):
  reverse={'生':'受生','受生':'生','克':'受克','受克':'克','比和':'比和'}
  for a in [1,2,3,6,9]:
   for b in [1,2,3,6,9]:self.assertEqual(reverse[old.relation(a,b)],old.relation(b,a))
 def test_health_birth_definition_uses_general_source_not_exam_procedure(self):
  p,m=fixture();p['units']['book:4:L180-180:cd8e3175a112']={'book_id':'shenqi','original_quote':'年命，即出生之年的年干落宫状态。'}
  m[0]['domain']='health';m[0]['support']=[{'source_id':'book:8:L6-6:c231e5a18b12','quote':'考试规则'}]
  revised=data.reviewed_specs(p,m);self.assertEqual(revised[0]['support'][0]['source_id'],'book:4:L180-180:cd8e3175a112')
 def test_whole_course_keeps_case_catalog(self):
  p,m=fixture();pack=data.build(p,m);self.assertEqual(pack['case_catalog'],p['case_catalog']);self.assertFalse(pack['model_outputs_used_as_teachers'])
 def test_targets_not_generic_caption_or_audit(self):
  p,m=fixture();pack=data.build(p,m);pack['records'][0]['answer']='[图8 河图]';pack['records'][0]['messages'][-1]['content']=pack['records'][0]['answer']
  with self.assertRaises(ValueError):data.validate(pack)
 def test_natural_dialogue_system(self):
  self.assertIn('自然、完整',data.SYSTEM);self.assertIn('不只报一个词',data.SYSTEM)
 def test_multi_source_all_required_citations(self):
  p,m=fixture();m[0]['support'].append({'source_id':'book:two','quote':'戊/戊，甲甲比肩，名谓伏吟。'});pack=data.build(p,m)
  r=next(r for r in pack['records'] if r['task']=='reviewed_exam_proxy' and r['context']=='provided_evidence')
  self.assertIn('[S2]',r['answer']);self.assertEqual(r['provenance']['required_citations'],['S1','S2'])

class SlotChecksV53(unittest.TestCase):
 def test_star_short_name_not_false_negative(self):self.assertEqual(checks.evaluate({'answer':'天芮。'},gold())['status'],'pass')
 def test_role_sentence_accepted(self):self.assertTrue(checks.name_ok('取日干为失主。',{'answer':'日干'}))
 def test_wrong_name_not_substring_pass(self):self.assertFalse(checks.name_ok('天芮与天心都可以。',{'answer':'天芮星'}))
 def test_normal_complete_name_sentence_not_forced_to_bare_label(self):self.assertTrue(checks.name_ok('本条格名为伏吟。',{'answer':'伏吟'}))
 def test_explicit_role_label(self):self.assertTrue(checks.name_ok('本章以天心星、乙奇代表医生和医药。',{'answer':'天心星、乙奇','role_label':'医生和医药'}))
 def test_unknown_source_rejected(self):self.assertEqual(checks.evaluate({'answer':'天芮。[S8]'},gold(context='provided_evidence'))['status'],'fail')
 def test_distractor_source_cannot_replace_required_citation(self):
  g=gold(context='provided_evidence');g['messages'][-2]['content']='[S1] other source [S2] correct source';g['provenance']['required_citations']=['S2']
  self.assertEqual(checks.evaluate({'answer':'天芮。[S1]'},g)['status'],'fail');self.assertEqual(checks.evaluate({'answer':'天芮。[S2]'},g)['status'],'pass');self.assertEqual(checks.evaluate({'answer':'天芮。[S1][S2]'},g)['status'],'fail')
 def test_missing_source_rejected(self):self.assertEqual(checks.evaluate({'answer':'天芮。'},gold(context='provided_evidence'))['status'],'fail')
 def test_known_source_not_semantic_approval(self):
  result=checks.evaluate({'answer':'天心。[S1]'},gold(context='provided_evidence'));self.assertNotEqual(result['status'],'pass')
 def test_natural_chart_names_and_gua_without_number_accepted(self):
  self.assertTrue(checks.positions_ok('天心星落乾宫，五行为金；天芮星位于震三宫，属木。',{'心':6,'芮':3}))
 def test_chart_conflict_or_wrong_element_rejected(self):
  self.assertFalse(checks.positions_ok('心在乾六宫属木。',{'心':6}));self.assertFalse(checks.positions_ok('心在乾六宫，又在震三宫。心在震三宫。',{'心':6}))
 def test_joint_direction_does_not_pass_on_positions_alone(self):
  g=gold(kind='joint_relation',answer='克');g['checks']['positions']={'心':6,'芮':3}
  self.assertNotEqual(checks.evaluate({'answer':'受生。心在乾六宫属金，芮在震三宫属木。'},g)['status'],'pass')
 def test_explicit_natural_relation_and_wrong_direction(self):
  self.assertTrue(checks.relation_ok('两宫关系为克。','克'));self.assertTrue(checks.relation_ok('金克木。','克',elements=['金','木']))
  self.assertFalse(checks.relation_ok('木克金。','克',elements=['金','木']));self.assertFalse(checks.relation_ok('金克木，但关系为受生。','克',elements=['金','木']))
  self.assertFalse(checks.relation_ok('不是克。','克'));self.assertFalse(checks.relation_ok('受生。后文提到金克木。','克',elements=['金','木']))
 def test_repetition_and_output_cap_fail(self):
  self.assertEqual(checks.evaluate({'answer':'天乙'*30},gold())['status'],'fail');self.assertEqual(checks.evaluate({'answer':'天芮。','reached_token_limit':True},gold())['status'],'fail')
 def test_gate_cannot_hide_failed_directions_in_global_mean(self):
  s={'complete':True,'groups':{'provided_evidence':{'rate':1,'total':10},'closed_book':{'rate':1,'total':10},'element_direction':{'rate':.1,'total':25}}}
  self.assertFalse(checks.gate(s,'foundation')['passed']);s['groups']['element_direction']['rate']=.9;self.assertTrue(checks.gate(s,'foundation')['passed'])
 def test_missing_gate_category_blocks_long_training(self):self.assertFalse(checks.gate({'complete':True,'groups':{}},'foundation')['passed'])
 def test_prompt_or_expected_mismatch_rejected(self):
  g=gold();r={'id':'x','answer':'天芮。','question':g['messages'][:-1],'expected':'天心'}
  with self.assertRaises(ValueError):checks.score([r],[g])

class RetrievalV53(unittest.TestCase):
 def test_new_symbol_index_does_not_mutate_old_engine(self):
  from wengua_course_v53 import rag
  from wengua_corpus import book_rag as original
  self.assertIn('癸',rag.r.terms('癸是阴水还是阳水'));self.assertNotIn('癸',original.SYMBOLS);self.assertNotEqual(rag.r.INDEX_FORMAT,original.INDEX_FORMAT)
 def test_upper_lower_rewrite_and_swap_fetch_both_orders(self):
  from wengua_course_v53 import rag
  query=rag.search_query('上层丁、下层壬倒换后是什么格')
  self.assertEqual(rag.r.ordered_pairs(query),{('丁','壬'),('壬','丁')})

 def test_single_symbol_recognition_does_not_split_application_or_child(self):
  from wengua_course_v53 import rag
  self.assertEqual(rag.explicit_symbols('申请工作，看看孩子健康'),[])
  self.assertEqual(rag.explicit_symbols('癸是阴水还是阳水'),['癸'])
 def test_missing_deity_and_trade_context_recognized_only_in_new_index(self):
  from wengua_course_v53 import rag
  from wengua_corpus import book_rag as original
  self.assertIn('六合',rag.r.SYMBOLS);self.assertNotIn('六合',original.SYMBOLS)
  self.assertIn('business_profit',rag.r.domains_for('改问婚姻、官司与买卖'));self.assertNotIn('买卖',original.DOMAINS['business_profit'])
 def test_source_definition_metadata_fix_never_promotes_case(self):
  from wengua_course_v53 import rag
  from unittest.mock import patch
  source={'chapters':[],'documents':[{'id':'one','kind':'reference','case_ids':[],'cues':['stem_pair:庚:戊'],'parts':[]},{'id':'case','kind':'reference','case_ids':['historical1'],'cues':['stem_pair:庚:戊'],'parts':[]}]}
  with patch.object(rag.r,'document_text',lambda *args:'庚/戊，本段解释该干对。'):
   updated,changes=rag.classified_library(source)
  self.assertEqual(updated['documents'][0]['kind'],'rule');self.assertEqual(updated['documents'][1]['kind'],'reference');self.assertEqual(len(changes),1);self.assertEqual(source['documents'][0]['kind'],'reference')

 def test_limit_validation_precedes_internal_candidate_expansion(self):
  from wengua_course_v53 import rag
  index=object.__new__(rag.ContextIndex)
  with self.assertRaises(ValueError):index.search('癸是阴水吗',limit=0)

class SourceCaseChainsV53(unittest.TestCase):
 def parent(self):
  cid='shenqi.fixture.case01';first='初次判断中所列的是书中当时已知的情况。';feedback='后来反馈中记载事情并未按初次判断发展。'
  unit={'book_id':'shenqi','section_id':1,'case_ids':[cid],'original_quote':first+'\n'+feedback}
  case={'case_id':cid,'group':'family1','domain':'education','author_steps':[],'primary_printed_chart':[],'additional_printed_charts':{},'author_original_judgment':first,'later_feedback_separate':feedback}
  return {'records':[],'case_catalog':[case],'original_case_splits':{cid:'train'},'development_policy':{'new_phase_withdrawn_families':[]}}, {'source1':unit}
 def test_initial_and_feedback_both_visible_and_attributed(self):
  p,units=self.parent();records=[];coverage=data.case_chains(p,units,records)
  self.assertTrue(coverage[0]['initial_quote_visible']);self.assertTrue(coverage[0]['feedback_quote_visible'])
  for field in ['author_original_judgment','later_feedback_separate']:
   proof=records[0]['provenance']['phase_sources'][field]
   self.assertIn(proof['literal_quote'],records[0]['question']);self.assertIn(proof['literal_quote'],records[0]['answer'])
  self.assertLess(records[0]['answer'].index('初次判断'),records[0]['answer'].index('后来反馈'))
 def test_test_case_is_skipped_before_accessing_private_case_fields(self):
  p,units=self.parent();p['case_catalog'].append({'case_id':'shenqi.sealed.case02'});p['original_case_splits']['shenqi.sealed.case02']='test'
  records=[];data.case_chains(p,units,records);self.assertEqual(len(records),1)
 def test_missing_phase_is_not_invented(self):
  p,units=self.parent();p['case_catalog'][0]['later_feedback_separate']='此句不存在于已经提供的原始材料中，不能补进答案。'
  records=[];coverage=data.case_chains(p,units,records)
  self.assertFalse(coverage[0]['feedback_quote_visible']);self.assertNotIn('此句不存在',records[0]['answer'])
 def test_source_locator_preserves_actual_char_offsets_and_hash(self):
  p,units=self.parent();case=p['case_catalog'][0];text='前文。\n初次判断中所列的是书中 当时已知的情况。\n后文。';case['later_feedback_separate']=''
  units['source1']['original_quote']='用于确定原案例所属章节的来源材料。'
  chapter={'book_id':'shenqi','chapter_id':'1','title':'来源章节','original_text':text,'source_sha256':old.digest(text)};records=[]
  data.case_chains(p,units,records,{'chapters':[chapter],'documents':[]})
  proof=records[0]['provenance']['phase_sources']['author_original_judgment'];u=units[proof['source_id']]
  self.assertEqual(text[slice(*u['chars'])],u['original_quote']);self.assertEqual(u['source_sha256'],old.digest(text));self.assertEqual(u['lines'],[2,2])
 def test_literal_correction_boundary_does_not_make_initial_error_a_teacher(self):
  p,_=self.parent();cid=p['case_catalog'][0]['case_id'];quote='正确判断应该如此分析：';raw='早先的初断。\n'+quote+'\n后来说明。'
  chapter={'book_id':'shenqi','chapter_id':'1','original_text':raw,'source_sha256':old.digest(raw)}
  part={'chapter':'shenqi/1','lines':[2,2],'source_sha256':old.digest(raw),'quote_sha256':old.digest(quote)}
  doc={'id':'marker','kind':'case','case_ids':[cid],'parts':[part],'analysis_annotations':[{'correction_required':True}]}
  stages=data.case_stages(p,{'chapters':[chapter],'documents':[doc]})
  self.assertTrue(data.step_phase(cid,{'book_id':'shenqi','section_id':1,'lines':[1,1]},stages)[1])
  self.assertFalse(data.step_phase(cid,{'book_id':'shenqi','section_id':1,'lines':[3,3]},stages)[1])
  self.assertIn('更正',data.step_phase(cid,{'book_id':'shenqi','section_id':1,'lines':[3,3]},stages)[0])

class WorkerStartupV53(unittest.TestCase):
 def test_missing_app_dependency_still_returns_authenticated_failure_receipt(self):
  from pathlib import Path
  import tempfile,subprocess,sys,os
  (runtime.ROOT/'local').mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=runtime.ROOT/'local',prefix='v53_startup_failure_') as temp:
   request=Path(temp)/'worker_calls/request.json';old.write(request,{'nonce':'public-software-fixture'})
   result=subprocess.run([sys.executable,str(runtime.ROOT/'run_course_v53_worker.py'),str(request)],env=os.environ|{'WENGUA_APP_ROOT':str(Path(temp)/'missing-app')},capture_output=True,text=True,timeout=30)
   self.assertNotEqual(result.returncode,0);reply=old.read(request.with_suffix('.result.json'))
   self.assertFalse(reply['ok']);self.assertEqual(reply['nonce'],'public-software-fixture');self.assertEqual(reply['request_sha256'],old.digest(request.read_bytes()));self.assertEqual(reply['error']['type'],'ValueError')

if __name__=='__main__':unittest.main()
