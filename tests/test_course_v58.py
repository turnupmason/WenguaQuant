import unittest
from wengua_course_v58 import checks,data,corpus
from wengua_course_v57 import checks as frozen

class V58SourceQuestionChecks(unittest.TestCase):
    def gold(self,kind='relation_facts',**kw):return {'checks':{'kind':kind,**kw}}
    def test_all25_teachers_and_no_wrong_headline(self):
        from wengua_course_v57.data import relation_answer
        place={'水':1,'土':2,'木':3,'金':6,'火':9}
        for a in place:
            for b in place:
                rel=data.d.relation(place[a],place[b]);g=self.gold(answer=rel,elements=[a,b])
                self.assertEqual(checks.fact_check({'answer':relation_answer(a,b)},g)['status'],'pass')
                for wrong in ['生','受生','克','受克','比和']:
                    if wrong!=rel:self.assertNotEqual(checks.fact_check({'answer':'应改'+wrong+'。'+relation_answer(a,b)},g)['status'],'pass')
    def test_wrong_head_with_correct_law_fails(self):
        for a in ['应改受克。金克木，即A克B。','应改写为受克。金克木。','结论：受克。金克木。']:
            self.assertNotEqual(checks.fact_check({'answer':a},self.gold(answer='克',elements=['金','木']))['status'],'pass')
    def test_forward_subject_edge_is_a_valid_short_answer(self):
        r=checks.fact_check({'answer':'前者生后者。[S1]'},self.gold(answer='生',elements=['金','水']))
        self.assertEqual(r['status'],'pass');self.assertFalse(r['complete'])
        self.assertNotEqual(checks.fact_check({'answer':'前者生后者。'},self.gold(answer='克',elements=['金','木']))['status'],'pass')
    def test_door_name_and_relation_must_agree(self):
        g=self.gold('door_facts',door='死',palaces=[9],elements=['土','火'],relation='受生',answer='义')
        self.assertNotEqual(checks.fact_check({'answer':'门与宫比和。火生土，即宫生门。'},g)['status'],'pass')
        self.assertTrue(checks.fact_check({'answer':'本条为义。死门属土，离九宫属火；火生土，宫生门。'},g)['complete'])
    def test_missing_name_is_partial_without_inventing_wrong_relation(self):
        g=self.gold('door_facts',door='伤',palaces=[6],elements=['木','金'],relation='受克',answer='制')
        r=checks.fact_check({'answer':'金克木，伤门受克。'},g);self.assertEqual(r['status'],'pass');self.assertFalse(r['complete'])
    def test_natural_properties_all_correct(self):
        for a,symbol,element in [('水。依据：[S1]。','休门','水'),('艮八宫在本条中归于土。[S1]、[S2]','艮八宫','土'),('杜门、天辅亦是木。[S1]','杜门','木'),('休门、天蓬同是水。[S1]','休门','水'),('属火。[S1]、[S2]','离九宫','火')]:
            self.assertEqual(checks.fact_check({'answer':a},self.gold('property',symbol=symbol,element=element))['status'],'pass')
        self.assertNotEqual(checks.fact_check({'answer':'杜门属水。'},self.gold('property',symbol='杜门',element='木'))['status'],'pass')
    def test_known_birth_stem_cannot_fill_unknown_cast_year(self):
        g=self.gold('person_contract',answer='年干',helper='乾宫',known_year_stem='癸')
        self.assertNotEqual(checks.fact_check({'answer':'取年干癸，兼看乾宫。'},g)['status'],'pass')
        self.assertTrue(checks.fact_check({'answer':'代问父亲取本局年干，兼看乾宫，另兼看出生年命癸。'},g)['complete'])
    def test_explicit_equal_cast_and_birth_stems_are_permitted(self):
        g=self.gold('person_contract',answer='年干',helper='乾宫',known_year_stem='癸',cast_year_stem='癸')
        self.assertTrue(checks.fact_check({'answer':'取本局年干癸，兼看乾宫；另看年命癸。'},g)['complete'])
    def test_unknown_year_never_becomes_a_positive_year_use(self):
        g=self.gold('person_contract',answer='年干',helper='坤宫',unknown_year=True)
        self.assertNotEqual(checks.fact_check({'answer':'取年干，兼看年命。出生年未知。'},g)['status'],'pass')
    def test_asked_main_role_does_not_force_unasked_boilerplate(self):
        g=self.gold('person_contract',answer='日干',unknown_year=True,require_year_boundary=False)
        self.assertTrue(checks.fact_check({'answer':'以日干为考生。'},g)['complete'])
        self.assertNotEqual(checks.fact_check({'answer':'以日干为考生，兼看年命。'},g)['status'],'pass')
    def test_heaven_earth_numbers_are_not_ignored(self):
        g=self.gold('number_pair',answer=[2,7])
        self.assertNotEqual(checks.fact_check({'answer':'二、七。天二生火，地七成之。'},g)['status'],'pass')
        self.assertTrue(checks.fact_check({'answer':'二、七。地二生火，天七成之。'},g)['complete'])
    def test_all_requested_domains_must_be_answered(self):
        g=self.gold('domain_map')
        self.assertNotEqual(checks.fact_check({'answer':'婚姻六合为媒人。'},g)['status'],'pass')
        self.assertTrue(checks.fact_check({'answer':'婚姻为媒人；诉讼为证人证据；交易为经纪人。'},g)['complete'])
    def test_outcome_question_cannot_be_answered_as_a_date_question(self):
        self.assertNotEqual(checks.fact_check({'answer':'不能据此断婚期，还缺时间资料。'},self.gold('marriage_conditions'))['status'],'pass')
        self.assertTrue(checks.fact_check({'answer':'不能只凭相生保证结婚，还需吉门吉格。'},self.gold('marriage_conditions'))['complete'])
    def test_timing_scope_requires_actual_missing_facts(self):
        self.assertTrue(checks.fact_check({'answer':'不能列出日期，还缺实际盘面与应期时间条件。'},self.gold('timing_scope'))['complete'])
    def test_equivalent_sources_satisfy_a_curated_rule_group(self):
        g={'messages':[{'role':'user','content':'[S1] 同一规则\n[S2] 同一规则'},{'role':'assistant','content':'x'}],'provenance':{'citation_rule_groups':[['S1','S2']]}}
        self.assertTrue(checks.citation_check({'answer':'正确事实。[S1]'},g)['ok'])
        self.assertFalse(checks.citation_check({'answer':'正确事实。[S3]'},g)['ok'])
        g['provenance']['citation_rule_groups']=[['S1'],['S2']]
        self.assertFalse(checks.citation_check({'answer':'只答一个领域。[S1]'},g)['ok'])
    def test_missing_chart_guard_is_removed_only_from_knowledge_text(self):
        self.assertEqual(data.clean('知识问题。\n'+data.GUARDS[0]+'[S1] 规则'),'知识问题。\n[S1] 规则')
    def test_reading_archive_is_not_silently_deleted(self):
        self.assertFalse(data.active({'split':'train','skill':'author_reading'}))
        self.assertTrue(data.active({'split':'train','skill':'case'}))
    def test_frozen_scorer_is_unchanged(self):
        g=self.gold(answer='克',elements=['金','木'])
        self.assertEqual(frozen.fact_check({'answer':'应改受克。金克木，即A克B。'},g)['status'],'pass')

if __name__=='__main__':unittest.main()

class V58ResearchParentProvenance(unittest.TestCase):
    def test_completed_research_parent_requires_exact_ancestry_receipt(self):
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from wengua_course_v58 import initialization,runtime
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'old/cycle_01/adapters/step-65';path.mkdir(parents=True)
            data.d.write(path.parent.parent/'stage_status.json',{'state':'complete','steps':65})
            parent=root/'snapshot/parent_adapter';parent.mkdir(parents=True)
            receipt=parent.parent/'parent_provenance.json';data.d.write(receipt,{'files':{'adapter_model.safetensors':'fixture-original-sha'}})
            c={'book_parent_snapshot':str(parent),'book_parent_provenance_sha256':data.d.digest(receipt.read_bytes())}
            release=root/'release.json';data.d.write(release,{'reviewed_parent':{'path':str(path),'ancestor_v55_sha256':'fixture-original-sha'}})
            proof={'schema':initialization.previous.runtime.SCHEMA,'parent_config':c}
            with patch.object(runtime,'RELEASE',release),patch.object(initialization,'inspect',return_value=proof),patch.object(runtime,'verify_snapshot') as verify:
                self.assertEqual(initialization.default_source(),path);verify.assert_called_once_with(c)
                receipt.write_text('{}')
                with self.assertRaises(ValueError):initialization.default_source()

class V58NaturalRelationSubjects(unittest.TestCase):
    def test_reverse_edge_and_received_subject_are_not_format_errors(self):
        for text in ['后者生前者。','前者受生。','乙方生甲方。','第二方生第一方。']:
            self.assertEqual(checks.fact_check({'answer':text},{'checks':{'kind':'relation_facts','answer':'受生','elements':['水','金']}})['status'],'pass',text)
        self.assertNotEqual(checks.fact_check({'answer':'后者生前者。'},{'checks':{'kind':'relation_facts','answer':'生','elements':['金','水']}})['status'],'pass')

class V58ExplicitJointClaims(unittest.TestCase):
    def test_both_subject_bihe_is_checked_in_both_directions(self):
        self.assertEqual(checks.fact_check({'answer':'两者比和。'},{'checks':{'kind':'relation_facts','answer':'比和','elements':['金','金']}})['status'],'pass')
        self.assertNotEqual(checks.fact_check({'answer':'两者比和。金克木。'},{'checks':{'kind':'relation_facts','answer':'克','elements':['金','木']}})['status'],'pass')
