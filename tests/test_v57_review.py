import unittest
from research_audits.v57_review import wording_observation,validate_annotation
from wengua_course_v57 import checks
from wengua_course import data as d

class V57ReadOnlyReview(unittest.TestCase):
    def test_wrong_correction_head_remains_a_conflict(self):
        gold={'checks':{'kind':'relation_facts','elements':['金','木'],'answer':'克'}}
        row={'answer':'应改受克。金克木，即A克B。'}
        self.assertEqual(checks.fact_check(row,gold)['status'],'pass')  # Frozen bug retained, never patched mid-run.
        self.assertEqual(wording_observation(row,gold)['explicit_relation_conflicts'],['受克'])

    def test_wrong_door_label_is_not_excused_by_a_correct_law(self):
        gold={'checks':{'kind':'door_facts','elements':['土','火'],'answer':'义','relation':'受生','door':'死','palaces':[9]}}
        row={'answer':'门与宫比和。门属土，宫属火；火生土，即宫生门。'}
        self.assertEqual(checks.fact_check(row,gold)['status'],'pass')
        self.assertEqual(wording_observation(row,gold)['explicit_relation_conflicts'],['比和'])

    def test_read_answer_note_is_bound_to_exact_original_bytes(self):
        row={'id':'x','answer':'fixture'};note={'id':'x','answer_sha256':d.digest('fixture'),'verdict':'correct','reason':'bounded fixture'}
        self.assertEqual(validate_annotation(note,row),note)
        with self.assertRaises(ValueError):validate_annotation(note,row|{'answer':'changed'})

    def test_natural_property_expression_and_citations(self):
        gold={'checks':{'kind':'property','symbol':'艮八宫','element':'土'}}
        for text in ['属于土。[S1]、[S2]','艮八宫在本条中归于土。[S1]、[S2]']:
            self.assertTrue(wording_observation({'answer':text},gold)['natural_property_candidate'])
        self.assertFalse(wording_observation({'answer':'艮八宫属水。'},gold)['natural_property_candidate'])

    def test_number_pair_does_not_approve_an_inverted_explanation(self):
        gold={'checks':{'kind':'number_pair','answer':[2,7]}}
        row={'answer':'二、七。天二生火，地七成之。'}
        self.assertEqual(checks.fact_check(row,gold)['status'],'pass')
        self.assertTrue(wording_observation(row,gold)['inverted_heaven_earth_explanation'])
        self.assertFalse(wording_observation({'answer':'二、七。地二生火，天七成之。'},gold)['inverted_heaven_earth_explanation'])

    def test_long_rule_repetition_is_observed_without_a_new_score(self):
        row={'answer':('这是长度超过二十字的完整规则段落，用来模拟重复引用。'*6),'reached_token_limit':True}
        obs=wording_observation(row,{})
        self.assertTrue(obs['output_limit']);self.assertGreaterEqual(obs['maximum_repeated_long_clause'],5)
        self.assertFalse(obs['professional_approval']);self.assertTrue(obs['posthoc_not_new_score'])

if __name__=='__main__':unittest.main()
