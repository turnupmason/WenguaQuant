import unittest
from wengua_course_v59 import checks,data,corpus
from wengua_course_v57 import checks as frozen

class V59SourceQuestionChecks(unittest.TestCase):
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
        self.assertEqual(data.clean('知识问题。\n'+data.base.GUARDS[0]+'[S1] 规则'),'知识问题。\n[S1] 规则')
    def test_reading_archive_is_not_silently_deleted(self):
        self.assertFalse(data.active({'split':'train','skill':'author_reading'}))
        self.assertTrue(data.active({'split':'train','skill':'case'}))
    def test_frozen_scorer_is_unchanged(self):
        g=self.gold(answer='克',elements=['金','木'])
        self.assertEqual(frozen.fact_check({'answer':'应改受克。金克木，即A克B。'},g)['status'],'pass')

if __name__=='__main__':unittest.main()


class V59RepairBoundaries(unittest.TestCase):
    def test_all_correction_forms_reject_contradictory_head(self):
        g={'checks':{'kind':'relation_facts','answer':'生','elements':['土','金']}}
        for h in ['应改记受生。','应改写为受生。','结论：受生。']:
            self.assertNotEqual(checks.fact_check({'answer':h+'土生金，即前者生后者。'},g)['status'],'pass')
    def test_element_side_subject_conflict_is_not_lost(self):
        g={'checks':{'kind':'relation_facts','answer':'受克','elements':['火','水']}}
        self.assertNotEqual(checks.fact_check({'answer':'从火方看水方，关系为受克；以火方为主，水方受克。'},g)['status'],'pass')
    def test_birth_reference_is_typed_not_contiguous_keyword_only(self):
        g={'checks':{'kind':'person_contract','answer':'日干','known_year_stem':'癸'}}
        self.assertTrue(checks.fact_check({'answer':'以日干代表本人。出生年癸为年命，应兼看癸。'},g)['complete'])
        g['checks']['cast_year_stem']='乙'
        self.assertNotEqual(checks.fact_check({'answer':'以日干代表本人。年命癸；时干乙。'},g)['status'],'pass')
    def test_stem_type_cannot_be_invented_before_a_refusal(self):
        g={'checks':{'kind':'insufficient'}}
        bad='这里只列出天盘丁加地盘壬。不能把天地盘资料看成其他信息。'
        self.assertNotEqual(checks.fact_check({'answer':bad},g)['status'],'pass')
        self.assertTrue(checks.fact_check({'answer':'不能。年干、时干不能代替天盘和地盘，还缺真实天地盘配置。'},g)['complete'])
    def test_ordered_domain_answers_and_stolen_alias(self):
        g={'checks':{'kind':'domain_map','requested_domains':['婚姻','诉讼','交易']}}
        self.assertTrue(checks.fact_check({'answer':'1、媒人；2、证人证据；3、经纪人。'},g)['complete'])
        self.assertNotEqual(checks.fact_check({'answer':'1、经纪人；2、证人证据；3、媒人。'},g)['status'],'pass')
        g={'checks':{'kind':'condition_slots','required':[['可能被盗']],'forbidden':['必定被盗']}}
        self.assertTrue(checks.fact_check({'answer':'可能被人盗走。'},g)['complete'])
    def test_sentence_overlap_removed_without_editing_rule_conditions(self):
        a='条件甲满足时，判断乙；原段要求同时核实另外一项必要条件才能使用此结论。'
        text='1、'+a+'\n2、'+a+'\n3、条件不同应另查。 [S1]'
        new,removed=data.dedupe_answer(text)
        self.assertEqual(new.count(a),1);self.assertTrue(removed);self.assertIn('条件不同应另查',new)
    def test_inference_placeholder_cannot_pass_evidence_contract(self):
        g={'checks':{'kind':'source_inference','judgment':'本项有利','evidence_terms':['证据','相生'],'minimum_evidence_terms':2},'answer':'本项有利。证据相生。'}
        self.assertFalse(checks.fact_check({'answer':'待补。本项有利。证据相生。'},g)['complete'])
        self.assertFalse(checks.fact_check({'answer':'本项有利。'},g)['complete'])
        self.assertTrue(checks.fact_check({'answer':'本项有利。依据：证据与所问方相生。'},g)['complete'])
    def test_case_rationale_restored_from_its_cited_stage(self):
        q='[S1] 同阶段分析第1步，盘次primary：事实甲已知，条件乙成立，所以判断丙。\n[S2] 后来反馈：另有情况。'
        ans='原阶段判断：丙。\n依据：所以判断丙。 [S1]\n后来反馈：另有情况。 [S2]'
        new,changed=data.enrich_case(ans,q)
        self.assertIn('事实甲已知，条件乙成立',new);self.assertEqual(changed,['S1']);self.assertTrue(new.endswith('后来反馈：另有情况。 [S2]'))

class V59NoTradeoffSelection(unittest.TestCase):
    def score(self,states):
        return {'complete':True,'items':[{'id':str(i),'fact':{'status':'pass' if good else 'needs_review','complete':good,'reason':'typed_source_fact_contract'},'citation':{'ok':True}} for i,good in enumerate(states)]}
    def test_new_gains_do_not_pay_for_lost_answer(self):
        from wengua_course_v59.runner import learning_gate
        empty=self.score([]);prior=self.score([True,False,False]);cur=self.score([False,True,True])
        self.assertFalse(learning_gate(cur,prior,empty,empty)['passed'])
        self.assertTrue(learning_gate(self.score([True,True,False]),prior,empty,empty)['passed'])
        self.assertFalse(learning_gate(prior,prior,empty,empty)['passed'])
    def test_lost_complete_condition_blocks_even_same_main_fact(self):
        from wengua_course_v59.runner import learning_gate
        prior=self.score([True,False]);cur=self.score([True,True]);cur['items'][0]['fact']['complete']=False
        self.assertFalse(learning_gate(cur,prior,self.score([]),self.score([]))['passed'])

class V59TypedSourceFacts(unittest.TestCase):
    def test_source_passage_is_not_a_new_birth_premise(self):
        c=data.normalize_person_contract({'kind':'role','answer':'年干','required':['乾']},'父亲出生年未知。原文依据：某案年命癸。')
        self.assertEqual(c['helper'],'乾宫');self.assertTrue(c['unknown_year']);self.assertNotIn('known_year_stem',c)
        c=data.normalize_person_contract({'answer':'日干','unknown_year':True},'出生年未知，不补造出生年干甲。')
        self.assertNotIn('known_year_stem',c)
    def test_input_only_relation_has_no_expected_answer_argument(self):
        from wengua_course_v59.facts import checked_prompt
        from wengua_course import data as d
        pal={'水':1,'土':2,'木':3,'金':6,'火':9}
        for a in pal:
            for b in pal:
                packet=checked_prompt('对象A属'+a+'，对象B属'+b,{'label':'对象A','element':a},{'label':'对象B','element':b})
                self.assertEqual(packet['relation'],d.relation(pal[a],pal[b]));self.assertFalse(packet['old_manual_scores_used'])
        with self.assertRaises(ValueError):checked_prompt('未给对象属性',{'label':'对象A','element':'金'},{'label':'对象B','element':'木'})

class V59ScopedClaims(unittest.TestCase):
    def test_unknown_birth_modifier_does_not_hide_positive_use(self):
        g={'checks':{'kind':'person_contract','answer':'日干','unknown_year':True}}
        self.assertNotEqual(checks.fact_check({'answer':'本人取日干，兼看出生年命。出生年未知。'},g)['status'],'pass')
    def test_added_abstract_star_identity_is_not_a_fact(self):
        g={'checks':{'kind':'relation_facts','answer':'受克','elements':['木','金']}}
        self.assertNotEqual(checks.fact_check({'answer':'关系为受克。A星为甲，B星为庚；金克木。'},g)['status'],'pass')
    def test_false_number_explanation_can_be_explicitly_corrected(self):
        g={'checks':{'kind':'number_pair','answer':[4,9]}}
        self.assertTrue(checks.fact_check({'answer':'四、九。不能说天四生金，应是地四生金、天九成之。'},g)['complete'])
    def test_explicit_reverse_named_viewpoint_is_not_a_conflict(self):
        g={'checks':{'kind':'relation_facts','answer':'受生','elements':['金','土'],'symbols':['乾宫','坤宫']}}
        self.assertEqual(checks.fact_check({'answer':'从坤宫看乾宫，关系为生。土生金，即坤宫生乾宫。'},g)['status'],'pass')

class V59EveryAssistantTurn(unittest.TestCase):
    def test_both_person_turns_keep_separate_complete_contracts(self):
        year={'kind':'person_contract','answer':'年干','helper':'乾宫','unknown_year':True}
        day={'kind':'person_contract','answer':'日干','unknown_year':True}
        r={'checks':day,'answer':'old','provenance':{'turn_contracts':[year,day]},'messages':[{'role':'system','content':'fixture'},{'role':'user','content':'女儿代问父亲，出生年未知。'},{'role':'assistant','content':'old [S1]'},{'role':'user','content':'改为父亲本人求测，出生年未知。'},{'role':'assistant','content':'old [S1]'}]}
        data.rewrite_person_turns(r)
        self.assertEqual(len(r['provenance']['turn_contracts']),2)
        for m,c in zip([r['messages'][2],r['messages'][4]],r['provenance']['turn_contracts']):
            self.assertTrue(checks.fact_check({'answer':m['content']},{'checks':c})['complete'])
        self.assertIn('乾宫',r['messages'][2]['content']);self.assertIn('日干',r['messages'][4]['content'])

class V59ExplainingRuleVersusInventingInput(unittest.TestCase):
    def test_rule_can_be_explained_without_assigning_it_to_this_chart(self):
        g={'checks':{'kind':'insufficient'}}
        text='天盘丁加地盘壬称奇仪相合格。本题只给年干、时干，不能代替天盘和地盘，仍缺真实配置。'
        self.assertTrue(checks.fact_check({'answer':text},g)['complete'])
        self.assertNotEqual(checks.fact_check({'answer':'这里只列出天盘丁加地盘壬，名为该格。不能看其他领域。'},g)['status'],'pass')

class V59ExactReviewedParent(unittest.TestCase):
    def test_step97_hash_and_ancestry_are_required(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from wengua_course_v59 import initialization,runtime
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);adapter=root/'cycle_01/adapters/step-97';adapter.mkdir(parents=True)
            data.d.write(adapter.parent.parent/'stage_status.json',{'state':'complete','steps':194,'epochs':[{'steps':97},{'steps':194}]})
            ancestor=root/'parent/parent_adapter';ancestor.mkdir(parents=True);receipt=ancestor.parent/'parent_provenance.json';data.d.write(receipt,{'files':{'adapter_model.safetensors':'v57-synthetic-sha'}})
            c={'book_parent_snapshot':str(ancestor),'book_parent_provenance_sha256':data.d.digest(receipt.read_bytes())}
            release=root/'release.json';data.d.write(release,{'reviewed_parent':{'path':str(adapter),'adapter_sha256':'epoch1-synthetic-sha','ancestor_v57_sha256':'v57-synthetic-sha'}})
            proof={'schema':initialization.previous.runtime.SCHEMA,'files':{'adapter_model.safetensors':'epoch1-synthetic-sha'},'parent_config':c}
            with patch.object(runtime,'RELEASE',release),patch.object(initialization,'inspect',return_value=proof),patch.object(runtime,'verify_snapshot') as verify:
                self.assertEqual(initialization.default_source(),adapter);verify.assert_called_once_with(c)
                proof['files']['adapter_model.safetensors']='epoch2-synthetic-sha'
                with self.assertRaises(ValueError):initialization.default_source()
                proof['files']['adapter_model.safetensors']='epoch1-synthetic-sha';receipt.write_text('{}')
                with self.assertRaises(ValueError):initialization.default_source()

class V59NestedSentenceLabels(unittest.TestCase):
    def test_nested_original_numbers_do_not_hide_the_same_sentence(self):
        clause='同一规则的这项必要条件必须同时成立，才能应用该条结论，不能略去。'
        text='1、开头说明。\n（1）'+clause+'\n2、（1）'+clause+'\n3、另一条件不同。 [S1]'
        actual,removed=data.dedupe_answer(text)
        self.assertEqual(actual.count(clause),1);self.assertTrue(removed);self.assertEqual(data.dedupe_answer(actual)[1],[])
        self.assertIn('2、另一条件不同',actual)

class V59EvidenceProseBoundaries(unittest.TestCase):
    def test_stacked_connectors_are_removed_without_cutting_real_words(self):
        self.assertEqual(data.trim_evidence('事实已经核实，据此，故'),'事实已经核实')
        self.assertEqual(data.trim_evidence('可能出交通事故'),'可能出交通事故')
        self.assertEqual(data.trim_evidence('事实已经核实，所以我断'),'事实已经核实')

class V59InferenceDoesNotHideInputFacts(unittest.TestCase):
    def test_new_palace_or_calendar_facts_are_not_inference_targets(self):
        for judgment in ['值符落艮8宫，处于旺相状态','辰月月令，土旺','太岁丑与值符同宫','上乘九天而有利']:
            self.assertIsNone(data.inference('前面原作者给出的规则和事实已经在此明确记录，所以'+judgment+'。',judgment,{'source_ids':['synthetic']},{},kind='literal_connective_author_inference'))
