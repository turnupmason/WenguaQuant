import unittest
from wengua_course_v57 import data,checks

class CourseV57(unittest.TestCase):
    def test_same_material_different_fragments_both_survive(self):
        units={'u':{'original_quote':'坤二、坤土、兑七、兑金'}}
        a={'source_id':'u','quote':'坤土'};b={'source_id':'u','quote':'兑金'}
        self.assertEqual(data.supports([a,b,a],units),[a,b])

    def test_missing_quote_is_not_imputed(self):
        with self.assertRaises(ValueError):data.supports([{'source_id':'u','quote':'丢失'}],{'u':{'original_quote':'完整原文'}})

    def test_all_25_relations_accept_the_source_teacher(self):
        place={'水':1,'土':2,'木':3,'金':6,'火':9}
        for a,pa in place.items():
            for b,pb in place.items():
                c={'answer':data.d.relation(pa,pb),'elements':[a,b]}
                result=checks.relation_facts(data.relation_answer(a,b),c)
                self.assertTrue(result['correct'],(a,b,result));self.assertTrue(result['complete'])

    def test_correct_law_with_wrong_tail_fails(self):
        c={'answer':'克','elements':['木','土']}
        r=checks.relation_facts('木克土，是前者克后者。这里称为受克。',c)
        self.assertFalse(r['correct'])

    def test_correct_tail_cannot_hide_wrong_head(self):
        c={'answer':'生','elements':['土','金']}
        self.assertFalse(checks.relation_facts('施克。土生金，是前者生后者。结论：施生。',c)['correct'])

    def test_natural_aliases_and_omitted_repeated_properties(self):
        c={'answer':'生','elements':['土','金']}
        for text in ['土生金，是前者生后者。结论：施生。','土生金，是前者生后者。','生。']:
            self.assertTrue(checks.relation_facts(text,c)['correct'],text)
        self.assertFalse(checks.relation_facts('生。',c)['complete'])

    def test_copying_all_cycles_does_not_solve_the_question(self):
        c={'answer':'克','elements':['土','水']}
        self.assertFalse(checks.relation_facts('金克木，木克土，土克水，水克火，火克金。',c)['correct'])

    def test_wrong_direction_and_wrong_native_palace_properties(self):
        c={'answer':'生','elements':['木','火'],'symbols':['巽四宫','离九宫']}
        for a in ['木生火，是后者生前者。','巽四宫属火，离九宫属木。结论：生。']:
            self.assertFalse(checks.relation_facts(a,c)['correct'])

    def test_clear_short_property_is_a_fact_not_a_style_failure(self):
        g={'checks':{'kind':'property','symbol':'巽四宫','element':'木'}}
        self.assertEqual(checks.fact_check({'answer':'属木。'},g)['status'],'pass')
        self.assertNotEqual(checks.fact_check({'answer':'属土。'},g)['status'],'pass')

    def test_unknown_year_cannot_leak_from_an_old_conditional_clause(self):
        self.assertFalse(checks.year_use_conflicts('有可靠出生年再兼看年命。出生年未知，暂不取年命。',True))
        self.assertTrue(checks.year_use_conflicts('有可靠出生年再兼看年命。但当前出生年未知，仍取年命。',True))
        self.assertTrue(checks.year_use_conflicts('先兼看父亲年命。',True))

    def test_current_asker_controls_role_not_the_word_father(self):
        c={'answer':'日干','unknown_year':True,'person_key':'father_self'}
        self.assertTrue(checks.person_facts('取日干看患者本人；暂不取年命。',c)['complete'])
        self.assertFalse(checks.person_facts('改用日干。父亲身份先取年干；暂不取年命。',c)['correct'])

    def test_known_birth_stem_is_given_fact_not_computed_guess(self):
        c={'answer':'年干','helper':'乾宫','unknown_year':False,'known_year_stem':'癸'}
        self.assertTrue(checks.person_facts('以起局年干看父亲，兼看乾宫和年命癸。',c)['complete'])
        self.assertFalse(checks.person_facts('以起局年干看父亲，兼看乾宫和年命甲。',c)['correct'])

    def test_numbers_and_wrong_extra_number(self):
        g={'checks':{'kind':'number_pair','answer':[4,9]}}
        self.assertEqual(checks.fact_check({'answer':'金的两数为四、九。[S1]'},g)['status'],'pass')
        self.assertNotEqual(checks.fact_check({'answer':'金的两数为四、八。[S1]'},g)['status'],'pass')

    def test_false_attributed_element_law_never_passes(self):
        c={'answer':'克','elements':['土','水']}
        self.assertFalse(checks.relation_facts('土生水。结论：克。',c)['correct'])
        self.assertTrue(checks.relation_facts('不是土生水，而是土克水。结论：克。',c)['correct'])

if __name__=='__main__':unittest.main()

class NaturalContracts(unittest.TestCase):
    def test_conditional_alternative_is_not_the_current_patient(self):
        c={'answer':'日干','person_key':'father_self','unknown_year':True}
        r=checks.person_facts('如果父亲本人来问，取日干；如果儿女代问父亲，取年干兼看乾宫。暂不取年命。',c)
        self.assertTrue(r['correct']);self.assertTrue(r['complete'])

    def test_marriage_negative_inference_is_not_a_guarantee(self):
        g={'checks':{'kind':'marriage_conditions'}}
        yes='不能只凭宫位相生就断一定结婚；本条还要求吉门吉格。'
        no='不能忽略吉门吉格，但只凭宫位相生就一定结婚。'
        self.assertEqual(checks.fact_check({'answer':yes},g)['status'],'pass')
        self.assertNotEqual(checks.fact_check({'answer':no},g)['status'],'pass')

class NegativeRoleScope(unittest.TestCase):
    def test_causal_negative_is_not_an_extra_positive_role(self):
        c={'answer':'日干','person_key':'father_self'}
        self.assertTrue(checks.person_facts('以日干看患者本人。不能因为父亲身份自动改取年干。',c)['correct'])
        self.assertFalse(checks.person_facts('以日干看患者本人。不能忽略身份，但仍取年干。',c)['correct'])

class MissingConditions(unittest.TestCase):
    def test_asker_must_be_clarified_before_choosing_one_rule(self):
        c={'kind':'clarification','options':[['本人','日干'],['代问','年干']],'helper':'乾宫'}
        g={'checks':c}
        self.assertEqual(checks.fact_check({'answer':'先确认本人求测还是代问。本人取日干；代问父亲取年干，兼看乾宫。'},g)['status'],'pass')
        self.assertNotEqual(checks.fact_check({'answer':'先确认本人求测还是代问。本人取年干；代问取日干，兼看乾宫。'},g)['status'],'pass')

class ScopedNaturalLanguage(unittest.TestCase):
    def test_comma_does_not_swap_clarification_mapping(self):
        g={'checks':{'kind':'clarification','options':[['本人','日干'],['代问','年干']],'helper':'乾宫'}}
        self.assertNotEqual(checks.fact_check({'answer':'先确认，本人取年干，但代问取日干，兼看乾宫。'},g)['status'],'pass')

    def test_alternative_actor_must_also_use_the_right_rule(self):
        c={'answer':'日干','person_key':'father_self'}
        self.assertTrue(checks.person_facts('本人取日干，子女代问父亲才取年干。',c)['correct'])
        self.assertFalse(checks.person_facts('本人取日干，子女代问父亲也取日干。',c)['correct'])

    def test_domain_list_has_to_bind_each_role(self):
        g={'checks':{'kind':'domain_map'}}
        for a in ['婚姻为媒人，诉讼为证人，交易为经纪人。','婚姻、诉讼、交易分别看媒人、证人、经纪人。','媒人、证人、经纪人分别对应婚姻、诉讼、交易。']:
            self.assertEqual(checks.fact_check({'answer':a},g)['status'],'pass')
        self.assertNotEqual(checks.fact_check({'answer':'婚姻为证人，诉讼为媒人，交易为经纪人。'},g)['status'],'pass')

class ActorNounIsNotAsker(unittest.TestCase):
    def test_candidate_himself_is_not_necessarily_the_person_asking(self):
        c={'answer':'时干','person_key':'exam_proxy'}
        self.assertTrue(checks.person_facts('父母代问女儿本人，取时干看考生。',c)['correct'])

class LayeredTrainingContracts(unittest.TestCase):
    def test_retention_does_not_mean_professional_approval(self):
        from wengua_course_v57.runner import learning_gate
        base={'complete':True,'groups':{'direction':{'total':25,'rate':.4}},'items':[],'citation':{'rate':1}}
        rag={'complete':True,'items':[],'citation':{'rate':1}}
        result=learning_gate(base,base,rag,rag)
        self.assertTrue(result['passed']);self.assertFalse(result['professional_approval'])
        bad=base|{'groups':{'direction':{'total':25,'rate':.32}}}
        self.assertFalse(learning_gate(bad,base,rag,rag)['passed'])

    def test_cached_answers_require_adapter_and_budget_proof(self):
        from wengua_course_v57 import runner,runtime,initialization
        from unittest.mock import patch
        from pathlib import Path
        from types import SimpleNamespace
        import tempfile,json
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);adapter=root/'adapter';adapter.mkdir();(adapter/'adapter_model.safetensors').write_bytes(b'tiny fixture')
            job=root/'job';(job/'generations').mkdir(parents=True)
            gold={'id':'x','messages':[{'role':'user','content':'x'},{'role':'assistant','content':'a'}],'answer':'a'}
            generated={'id':'x','question':gold['messages'][:-1],'expected':'a','answer':'a'}
            source=root/'old.jsonl';source.write_text(json.dumps(generated)+'\n')
            target=job/'generations/cached.jsonl';target.write_bytes(source.read_bytes())
            def forbidden_load():raise AssertionError('Cache hit must not load a model')
            with patch.object(initialization,'inspect'),patch.object(runtime,'prepared_prompts',return_value=[gold]):
                args=(SimpleNamespace(testing=True),job,{'eval_max_new_tokens':768},0)
                kw={'best':{'adapter':str(adapter),'tag':'cached','eval_stage':'final'},'factory':forbidden_load}
                with self.assertRaises(ValueError):runner.evaluate_adapter(*args,**kw)
                target.unlink()
                runner.copy_answers(source,target,[gold],adapter_sha=data.d.digest((adapter/'adapter_model.safetensors').read_bytes()),scope='fixture_same_budget')
                self.assertTrue(runner.evaluate_adapter(*args,**kw)['cache_hit_without_model_load'])
                kw['best']['max_new_tokens']=1024
                with self.assertRaises(ValueError):runner.evaluate_adapter(*args,**kw)
