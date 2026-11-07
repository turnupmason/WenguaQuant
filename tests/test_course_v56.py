import unittest
from wengua_course_v56 import data,checks,runner
from wengua_course import data as d


class CourseV56(unittest.TestCase):
    def test_remove_only_exact_repeated_paragraphs(self):
        text='1、同一条完整来源。\n2、同一条完整来源。\n3、不同条件必须保留。'
        self.assertEqual(data.dedupe_paragraphs(text),'1、同一条完整来源。\n3、不同条件必须保留。')

    def test_source_target_expands_to_the_complete_causal_sentence(self):
        text='上句结束。条件已经给出，因而可以形成判断。后句不混入。';start=text.index('可以');end=start+len('可以形成判断')
        a,b=data.complete_sentence_span(text,start,end)
        self.assertEqual(text[a:b],'条件已经给出，因而可以形成判断。')

    def test_all_25_element_chains_keep_subject_direction(self):
        palace={'水':1,'土':2,'木':3,'金':6,'火':9}
        for a,pa in palace.items():
            for b,pb in palace.items():
                g={'messages':[{'role':'user','content':'fixture'},{'role':'assistant','content':''}],'checks':{'kind':'reasoning_chain','answer':d.relation(pa,pb),'elements':[a,b]}}
                r={'answer':data.chain(a,b)}
                self.assertEqual(checks.fact_check(r,g)['status'],'pass',(a,b,r))

    def test_correct_final_word_cannot_hide_false_cycle(self):
        c={'elements':['金','土'],'answer':'受生'}
        self.assertFalse(checks.relation_chain_ok('前者属金，后者属土；土克金。结论：受生。',c))

    def test_reversed_subject_is_rejected_after_correct_cycle(self):
        c={'elements':['水','土'],'answer':'受克'}
        self.assertFalse(checks.relation_chain_ok('前者属水，后者属土；土克水，是前者克后者。结论：受克。',c))

    def test_result_must_match_explanation_and_declared_target(self):
        c={'elements':['木','火'],'answer':'生'}
        self.assertFalse(checks.relation_chain_ok('前者属木，后者属火；木生火。结论：受克。',c))

    def test_swapped_properties_cannot_pass_with_a_correct_law(self):
        c={'elements':['木','火'],'answer':'生'}
        self.assertFalse(checks.relation_chain_ok('前者属火，后者属木；木生火。结论：生。',c))
        self.assertFalse(checks.relation_chain_ok('受克。前者属木，后者属火；木生火。结论：生。',c))

    def test_unknown_year_does_not_block_known_self_identity(self):
        g={'messages':[{'role':'user','content':'本人问考试，出生年未知'},{'role':'assistant','content':''}],'checks':{'kind':'role_boundary','answer':'日干','unknown_year':True},'provenance':{'birth_year_unknown_in_question':True}}
        self.assertEqual(checks.fact_check({'answer':'取日干代表本人；暂不取年命。'},g)['status'],'pass')
        self.assertNotEqual(checks.fact_check({'answer':'不能取日干；暂不取年命。'},g)['status'],'pass')
        self.assertNotEqual(checks.fact_check({'answer':'取日干；未知出生年可兼看年命。'},g)['status'],'pass')
        self.assertEqual(checks.fact_check({'answer':'日干代表本人；出生年未知，年命暂不使用。'},g)['status'],'pass')

    def test_retirement_negative_is_not_an_affirmative_permission(self):
        g={'messages':[{'role':'user','content':'fixture'},{'role':'assistant','content':''}],'checks':{'kind':'condition_slots','required':[['留用']],'forbidden':['准退']}}
        self.assertEqual(checks.fact_check({'answer':'领导留用，不准退役。'},g)['status'],'pass')
        self.assertNotEqual(checks.fact_check({'answer':'领导留用，所以准退。'},g)['status'],'pass')

    def test_legacy_full_answers_keep_the_original_generation_budget(self):
        c={'eval_max_new_tokens':384}
        self.assertEqual(runner.evaluation_config(c,'v55')['eval_max_new_tokens'],1024)
        self.assertEqual(runner.evaluation_config(c,'legacy_rag')['eval_max_new_tokens'],1024)
        self.assertEqual(runner.evaluation_config(c,'application')['eval_max_new_tokens'],384)
        self.assertEqual(runner.evaluation_config(c,'v55',testing=True)['eval_max_new_tokens'],384)
        self.assertEqual(c,{'eval_max_new_tokens':384})

    def test_natural_conditional_marriage_answer_is_not_a_missing_condition(self):
        result=runner.rag_boundaries([{'id':'v53:actual_rag:marriage_same_palace','answer':'乙庚同宫还需要临吉门、吉格，才能据此判断婚姻有利。'}])
        self.assertTrue(result['items'][0]['passed'])
        result=runner.rag_boundaries([{'id':'v53:actual_rag:marriage_same_palace','answer':'同宫即可，不需要吉门吉格。'}])
        self.assertFalse(result['items'][0]['passed'])

    def test_empty_actual_rag_cannot_pass_acceptance(self):
        self.assertFalse(runner.rag_boundaries([])['passed'])

    def test_ambiguous_role_list_is_not_a_domain_mapping(self):
        result=runner.rag_boundaries([{'id':'v53:actual_rag:cross_domain_liuhe','answer':'媒人、中介、证人。'}])
        self.assertFalse(result['passed']);self.assertFalse(result['items'][0]['passed'])

    def test_source_reading_and_local_reasoning_have_distinct_names(self):
        self.assertEqual(len(data.CLOZE),24)
        self.assertNotEqual('author_reading','author_local')
        self.assertTrue(all(q and start<end and len(sha)==64 for q,start,end,sha in data.CLOZE.values()))

    def test_repetition_cannot_pass_rag_even_before_token_limit(self):
        result=runner.rag_boundaries([{'id':'v53:actual_rag:role_21','answer':'失主取日干。可靠'*20}])
        self.assertFalse(result['passed']);self.assertFalse(result['items'][0]['passed'])

    def test_missing_skill_category_blocks_short_acceptance(self):
        summary={'complete':True,'groups':{'direction':{'total':25,'rate':1.0}},'citation':{'rate':1.0}}
        self.assertFalse(checks.gate(summary)['passed'])

    def test_a_failing_condition_group_blocks_even_with_lower_loss(self):
        kinds=['direction','given_direction','chart_direction','positions','door_relation','identity','condition','typed_pair','property','source_fact']
        summary={'complete':True,'groups':{k:{'total':10,'rate':1.0} for k in kinds},'citation':{'rate':1.0},'validation_language_loss':0.01}
        self.assertTrue(checks.gate(summary)['passed'])
        summary['groups']['condition']['rate']=0.4
        self.assertFalse(checks.gate(summary)['passed'])

    def test_legal_role_word_without_the_correct_action_is_not_enough(self):
        g={'messages':[{'role':'user','content':'fixture'},{'role':'assistant','content':''}],'checks':{'kind':'condition_slots','required':[['原告']],'forbidden':['被告']},'provenance':{'rule_id':'rag:ff1d59b30fe9012908ac4b6b'}}
        self.assertEqual(checks.fact_check({'answer':'原告可能主动求和。'},g)['status'],'pass')
        self.assertNotEqual(checks.fact_check({'answer':'原告不会主动求和。'},g)['status'],'pass')

    def test_lost_property_possible_forgetting_does_not_allow_certain_theft(self):
        g={'messages':[{'role':'user','content':'fixture'},{'role':'assistant','content':''}],'checks':{'kind':'condition_slots','required':[['忘记']],'forbidden':['一定被偷']},'provenance':{'rule_id':'rag:d6ee057cb5c1247772a08bfc'}}
        self.assertEqual(checks.fact_check({'answer':'可能被偷，也可能自己忘记。'},g)['status'],'pass')
        self.assertNotEqual(checks.fact_check({'answer':'也可能自己忘记，但肯定被盗。'},g)['status'],'pass')


if __name__=='__main__':unittest.main()
