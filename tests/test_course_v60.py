import unittest
from copy import deepcopy
from wengua_course_v60 import checks,data
from wengua_course_v60.loop import retained,RATES
from wengua_course import data as d


class V60ContractTests(unittest.TestCase):
    def gold(self,kind='relation_facts',**kw):return {'checks':dict(kind=kind,**kw)}

    def test_all_25_ordered_laws_and_opposite_headlines(self):
        from wengua_course_v57.data import relation_answer
        elements={'水':1,'土':2,'木':3,'金':6,'火':9}
        for a,pa in elements.items():
            for b,pb in elements.items():
                gold=self.gold(answer=d.relation(pa,pb),elements=[a,b])
                self.assertTrue(checks.fact_check({'answer':relation_answer(a,b)},gold)['complete'])
                wrong='比和' if a!=b else '克'
                self.assertNotEqual(checks.fact_check({'answer':'结论：'+wrong+'。'+relation_answer(a,b)},gold)['status'],'pass')

    def test_bare_palace_cannot_bind_to_second_stars_palace(self):
        gold=self.gold(answer='克',elements=['火','金'],positions={'天心':9,'天芮':6},symbols=['天心','天芮'],require_positions=True)
        answer='天心在离九宫，天芮在乾六宫。天心星所在宫属火，天芮星所在宫属金；火克金，前者克后者。'
        self.assertEqual(checks.fact_check({'answer':answer},gold)['status'],'pass')
        self.assertNotEqual(checks.fact_check({'answer':answer.replace('火克金','金克火')},gold)['status'],'pass')

    def test_named_palace_alias_cannot_hide_wrong_bihe(self):
        gold=self.gold(answer='受生',elements=['金','土'],palaces=[6,2],symbols=['乾六宫','坤二宫'])
        self.assertNotEqual(checks.fact_check({'answer':'乾六金星与坤二土星比和。土生金。'},gold)['status'],'pass')

    def test_known_birth_accepts_typed_syntax_but_not_wrong_stem(self):
        gold=self.gold('person_contract',answer='年干',helper='乾宫',known_year_stem='癸')
        for phrase in ['年命已取癸','年命可取癸','年命癸']:
            answer='女儿代问父亲取年干，兼看乾宫；'+phrase+'。'
            self.assertTrue(checks.fact_check({'answer':answer},gold)['complete'])
        self.assertNotEqual(checks.fact_check({'answer':'取年干，兼看乾宫；年命甲。'},gold)['status'],'pass')

    def test_unknown_birth_cannot_use_typed_alias_to_invent_stem(self):
        gold=self.gold('person_contract',answer='年干',helper='乾宫',unknown_year=True)
        self.assertNotEqual(checks.fact_check({'answer':'取年干，兼看乾宫；年命未知但可取癸。'},gold)['status'],'pass')

    def test_source_equivalent_retirement_and_theft_phrases(self):
        for c,answer in [({'required':[['留用']],'forbidden':['准退']},'领导有意留你，不准退役。'),
                         ({'required':[['可能被盗']],'forbidden':[]},'物品可能被偷走。')]:
            self.assertTrue(checks.fact_check({'answer':answer},self.gold('condition_slots',**c))['complete'])

    def test_self_proxy_choice_contract_is_question_specific(self):
        gold=self.gold('person_contract',answer='日干');gold['question']='应先按本人求测还是代问父亲的规则取用？'
        self.assertTrue(checks.fact_check({'answer':'按本人求测，患者就是求测人。'},gold)['complete'])
        gold['question']='本人求测具体取哪个天干类别？'
        self.assertNotEqual(checks.fact_check({'answer':'按本人。'},gold)['status'],'pass')

    def test_explicit_other_proxy_branch_does_not_change_self_role(self):
        gold=self.gold('person_contract',answer='日干')
        self.assertTrue(checks.fact_check({'answer':'本人求测取日干；代问父亲取年干。'},gold)['complete'])
        self.assertNotEqual(checks.fact_check({'answer':'本人求测取年干。'},gold)['status'],'pass')

    def tiny_sources(self):
        text='病星落艮宫，艮属土；作者据此判断病在肠胃。'
        case={'case_id':'shenqi.fixture','phase_split':'train','domain':'health','group':'case:fixture',
              'author_steps':[{'step':1,'source_id':'literal','original':text}]}
        return {'case_catalog':[case], 'units':{'literal':{'book_id':'shenqi','original_quote':text}}}

    def test_author_replay_has_complete_visible_source_and_literal_target(self):
        p=self.tiny_sources();rows,ledger=data.author_tasks(p)
        self.assertEqual(len(rows),1);self.assertIn(p['case_catalog'][0]['author_steps'][0]['original'],rows[0]['question'])
        self.assertNotIn('省去',rows[0]['question']);self.assertTrue(rows[0]['provenance']['not_independent_prediction'])
        self.assertEqual(ledger[0]['action'],'explicit_source_replay')

    def test_sealed_case_never_becomes_a_replay_teacher(self):
        p=self.tiny_sources();p['case_catalog'][0]['phase_split']='sealed_test';rows,ledger=data.author_tasks(p)
        self.assertEqual(rows,[]);self.assertEqual(ledger[0]['action'],'sealed_original_test')

    def test_full_case_phase_fix_changes_generated_headings_only(self):
        p=self.tiny_sources();literal=p['units']['literal']['original_quote'];refs=[{'source_id':'literal','quote':literal}]
        r=data.row('请复盘这一个原书案例。书中初断及按语选段：'+literal,
                   '书中初断及按语选段：'+literal,refs,p['units'],skill='case',case_ids=['shenqi.fixture'],group='case:fixture',domain='health')
        p.update(records=[r],fixed_micro_ids=[],v59_actual_rag_rows=[])
        delta=data.build(p);repaired=next(x for x in delta['records'] if x['skill']=='case_replay')
        self.assertIn(literal,repaired['question']);self.assertIn(literal,repaired['answer'])
        self.assertNotIn('书中初断及按语选段',repaired['answer'])
        self.assertTrue(repaired['provenance']['phase_uncertainty_preserved'])

    def test_hidden_suffix_is_never_active(self):
        row={'split':'train','provenance':{'v60_training_allowed':False},'skill':'author_local'}
        self.assertFalse(data.active(row));row['split']='validation';row['provenance']['v60_training_allowed']=True
        self.assertFalse(data.active(row))

    def test_row_citations_are_not_appended_twice(self):
        p=self.tiny_sources();r=data.row('问题','依据。 [S1]',[{'source_id':'literal','quote':p['units']['literal']['original_quote']}],p['units'],skill='condition')
        self.assertEqual(r['answer'].count('[S1]'),1)

    def summary(self,items):
        return {'complete':True,'items':[{'id':k,'fact':{'status':'pass' if value else 'needs_review','complete':value,'reason':'typed_source_fact_contract'},'citation':{'ok':True}} for k,value in items]}

    def test_gain_does_not_pay_for_a_different_lost_fact(self):
        before=self.summary([('a',True),('b',False)]);after=self.summary([('a',False),('b',True)])
        self.assertFalse(retained(after,before,self.summary([]),self.summary([]))['passed'])

    def test_unchanged_full_stage_can_keep_accepted_pilot(self):
        cur=self.summary([('a',True)])
        self.assertFalse(retained(cur,cur,self.summary([]),self.summary([]))['passed'])
        self.assertTrue(retained(cur,cur,self.summary([]),self.summary([]),require_improvement=False)['passed'])

    def test_three_rates_are_finite_and_structure_unchanged(self):
        self.assertEqual(RATES,(1e-6,3e-6,1e-5))

if __name__=='__main__':unittest.main()
