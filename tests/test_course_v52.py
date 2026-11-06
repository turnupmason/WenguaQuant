import copy,tempfile,unittest
from pathlib import Path
from wengua_course import data,checks,runtime

def fixture():
 units={'a':{'id':'a','book_id':'shenqi','original_quote':'software valid source','kind':'fixture'},'b':{'id':'b','book_id':'kaiwu','original_quote':'software other source','kind':'fixture'}}
 rows=[];cases=[]
 for i,split in enumerate(['train','train','validation','test']):
  cid='case'+str(i);group='family:'+str(i)
  rows.append({'id':'old'+str(i),'question':'software question '+str(i),'answer':'software answer','messages':[{'role':'system','content':'system'},{'role':'user','content':'software question '+str(i)},{'role':'assistant','content':'software answer'}],'source_ids':['a'],'case_id':cid,'case_ids':[cid],'group':group,'split':split,'format':'foundation_qa','task':'reviewed_rule','source_mode':'fixture'})
  cases.append({'case_id':cid,'domain':'software_domain','group':group,'split':split,'author_steps':[],'primary_printed_chart':[]})
 return {'records':rows,'units':units,'dossiers':cases,'case_splits':{x['case_id']:x['split'] for x in cases},'source_sha256':'fixture_not_real_case'}
class Curriculum(unittest.TestCase):
 def test_titles_are_structure_not_targets(self):
  for text in ['九宫盘面：','九宫格局:','分析：']:self.assertTrue(data.title_only(text))
  self.assertFalse(data.title_only('日干。'))
 def test_paired_author_analysis_keeps_A_B_scope(self):
  p=fixture();names=list(data.COMPARISONS.values())[0];a,b=names['A'],names['B'];mapping={'case0':a,'case1':b}
  for r in p['records']:
   r['case_ids']=[mapping.get(c,c) for c in r['case_ids']];r['case_id']=mapping.get(r['case_id'],r['case_id'])
   if r['case_id'] in (a,b):r['group']='family:paired'
  for d in p['dossiers']:
   d['case_id']=mapping.get(d['case_id'],d['case_id'])
   if d['case_id'] in (a,b):d['group']='family:paired'
  p['case_splits']={mapping.get(c,c):s for c,s in p['case_splits'].items()}
  text='A卦采用软件条件甲乙丙丁；B卦使用不同的软件条件戊己庚辛壬癸，故本句只属于B卦。'
  p['units']['compare']={'id':'compare','book_id':'kaiwu','original_quote':text,'case_ids':[a,b]}
  p['dossiers'][0]['author_steps']=[{'step':1,'source_id':'compare','original':text}]
  result=data.build(p,[],revealed_groups=['family:paired']);r=next(r for r in result['records'] if r['task']=='author_evidence_to_conclusion')
  self.assertEqual(set(r['case_ids']),{a,b});self.assertTrue(r['answer'].startswith('B卦结论'));self.assertIn('两个不同病例',r['question'])
 def test_source_material_and_original_test_are_preserved_not_active(self):
  p=fixture();before=copy.deepcopy(p);result=data.build(p,[],revealed_groups=['family:0'])
  self.assertEqual(p,before);self.assertEqual(result['units'],p['units']);self.assertEqual(len(result['sealed_original_test_manifest']),1)
  self.assertFalse(any('case3' in r['case_ids'] for r in result['records']));self.assertEqual(len(result['old_task_dispositions']),3)
 def test_title_retired_not_material_deleted(self):
  p=fixture();p['records'][0]['answer']='九宫盘面：';p['records'][0]['messages'][-1]['content']='九宫盘面：'
  result=data.build(p,[]);disposition=next(d for d in result['old_task_dispositions'] if d['old_id']=='old0');self.assertEqual(disposition['reason'],'structure_only_not_content_target');self.assertIn('a',result['units'])
 def test_family_withdrawal_moves_all_related_rows(self):
  p=fixture();result=data.build(p,[],revealed_groups=['family:0']);data.validate(result)
  sets={r['split'] for r in result['records'] if r['group']=='family:0'};self.assertEqual(sets,{'train'})
 def test_bad_manual_support_is_rejected(self):
  s={'key':'bad','support':[{'source_id':'a','quote':'invented source'}]}
  with self.assertRaises(ValueError):data.build(fixture(),[s])
 def test_case_group_conflict_rejected(self):
  p=fixture();r=copy.deepcopy(p['records'][0]);r.update(id='other',group='family:unknown');p['records'].append(r)
  with self.assertRaises(ValueError):data.case_ownership(p)
 def test_third_book_teacher_rejected(self):
  p=data.build(fixture(),[]);p['units']['a']['book_id']='jinhan'
  with self.assertRaises(ValueError):data.validate(p)
 def test_test_case_cannot_become_a_train_lesson(self):
  p=data.build(fixture(),[]);r=copy.deepcopy(p['records'][0]);r.update(id='leak',case_ids=['case3'],case_id='case3');p['records'].append(r)
  with self.assertRaises(ValueError):data.validate(p)
 def test_relation_direction_symmetry_and_no_outcome_scores(self):
  self.assertEqual(data.relation(8,3),'受克');self.assertEqual(data.relation(3,8),'克');self.assertEqual(data.relation(6,2),'受生');self.assertEqual(data.relation(2,6),'生');self.assertEqual(data.relation(3,4),'比和')
 def test_multiturn_final_answer_is_the_target(self):
  p=data.build(fixture(),[]);p['records'][0]['messages'][-1]['content']='different'
  with self.assertRaises(ValueError):data.validate(p)
class CastScopes(unittest.TestCase):
 def fixture(self):
  cid='software-case';units={};boards={}
  for key,start in [('primary',10),('second',40)]:
   cells=[]
   for palace in range(1,10):
    cell={'case_id':cid,'cast_key':key,'palace':palace,'source_line':start+palace,'deity_raw':'软件','door_position_raw':'软件','star_position_raw':'软件'};cells.append(cell)
    units[f'chart:{cid}:{key}:p{palace}']={'book_id':'shenqi','section_id':1,'original_quote':data.dumps(cell)}
   boards[key]=cells
  step={'source_id':'step','original':'这是软件条件测试，不能当作真实案例。'};units['step']={'book_id':'shenqi','section_id':1,'lines':[60,60]}
  return {'case_id':cid,'primary_printed_chart':boards['primary'],'additional_printed_charts':{'second':boards['second']}},step,units
 def test_later_analysis_gets_later_board_not_primary(self):
  d,s,u=self.fixture();cells,ids,proof=data.step_board(d,s,u);self.assertEqual(proof['cast_key'],'second');self.assertTrue(all(c['cast_key']=='second' for c in cells))
 def test_before_second_chart_uses_primary(self):
  d,s,u=self.fixture();u['step']['lines']=[30,30];self.assertEqual(data.step_board(d,s,u)[2]['cast_key'],'primary')
 def test_cross_chart_argument_does_not_get_one_mixed_board(self):
  d,s,u=self.fixture();s['original']='与上次原局作比较，软件字段仍须分开。';cells,_,proof=data.step_board(d,s,u);self.assertFalse(cells);self.assertIn('cross_chart',proof['status'])
 def test_different_chapter_never_binds_by_line_number_alone(self):
  d,s,u=self.fixture();u['step']['section_id']=2;self.assertFalse(data.step_board(d,s,u)[0])
 def test_incomplete_transcription_not_called_complete_board(self):
  d,s,u=self.fixture();d['primary_printed_chart']=d['primary_printed_chart'][:2];d['additional_printed_charts']={};self.assertFalse(data.step_board(d,s,u)[0])

class FactualChecks(unittest.TestCase):
 def test_exact_name_does_not_accept_wrong_long_name(self):
  c={'kind':'exact_name','answer':'伏吟'};self.assertEqual(checks.evaluate('伏吟。',c)['status'],'pass')
  for a in ['天乙伏吟。','不是伏吟，是地户。','地户。']:self.assertNotEqual(checks.evaluate(a,c)['status'],'pass')
 def test_number_pairs_chinese_and_arabic(self):
  c={'kind':'number_pair','answer':[4,9]}
  for a in ['四、九。','4，9。','四和九。']:self.assertEqual(checks.evaluate(a,c)['status'],'pass')
  for a in ['七九。','7、9。','四九六七。']:self.assertNotEqual(checks.evaluate(a,c)['status'],'pass')
 def test_role_requires_conclusion_not_a_quoted_keyword(self):
  c={'kind':'role','answer':'日干','opposite':'时干'}
  self.assertEqual(checks.evaluate('取日干。父母代问才是时干。',c)['status'],'pass');self.assertNotEqual(checks.evaluate('取时干。不要取日干。',c)['status'],'pass')
 def test_direction_and_negation(self):
  c={'kind':'relation','answer':'受克'}
  self.assertEqual(checks.evaluate('受克。艮土受震木克。',c)['status'],'pass')
  for a in ['生。','不是受克，是前者生后者。','土木都有。']:self.assertNotEqual(checks.evaluate(a,c)['status'],'pass')
 def test_matching_rubric_terms_is_not_semantic_pass(self):
  result=checks.evaluate('并非日干，应该用时干。',{'kind':'mentions','required':[['日干']], 'forbidden':[]})
  self.assertEqual(result['status'],'needs_review');self.assertFalse(checks.automatically_checkable({'kind':'mentions'}))
 def test_free_case_answers_never_auto_certified(self):
  self.assertEqual(checks.evaluate('correct-looking essay',None)['status'],'needs_review')
  self.assertEqual(checks.evaluate('伏吟。',{'kind':'exact_name','answer':'伏吟'},capped=True)['status'],'fail')
 def test_prediction_provenance_must_match(self):
  gold={'id':'one','messages':[{'role':'user','content':'q'},{'role':'assistant','content':'a'}],'answer':'a','domain':'x','task':'y','group':'z','checks':None}
  with self.assertRaises(ValueError):checks.score([{'id':'one','question':[],'expected':'a','answer':'a'}],[gold])
 def test_complete_constraint_screen_is_not_professional_approval(self):
  g={'id':'one','messages':[{'role':'user','content':'q'},{'role':'assistant','content':'伏吟。'}],'answer':'伏吟。','domain':'x','task':'y','group':'z','checks':{'kind':'exact_name','answer':'伏吟'}}
  scored=checks.score([{'id':'one','question':g['messages'][:-1],'expected':g['answer'],'answer':'伏吟。'}],[g]);self.assertTrue(scored['complete']);self.assertFalse(scored['semantic_professional_approval'])
if __name__=='__main__':unittest.main()
