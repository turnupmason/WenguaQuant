import unittest
from research_audits.v56_review import wording_observation, collapsed_selectors


class V56Review(unittest.TestCase):
    def gold(self, answer='生', elements=('土','金')):
        return {'checks':{'kind':'reasoning_chain','answer':answer,'elements':list(elements)}}

    def test_alias_is_observed_without_changing_frozen_score(self):
        result=wording_observation({'answer':'前者属土，后者属金；土生金，是前者生后者。结论：施生。'},self.gold())
        self.assertTrue(result['uses_explicit_alias']);self.assertTrue(result['alias_chain_passes'])
        self.assertFalse(result['semantic_professional_approval'])

    def test_wrong_headline_is_not_excused_by_correct_tail(self):
        result=wording_observation({'answer':'施克。前者属土，后者属金；土生金。结论：施生。'},self.gold())
        self.assertFalse(result['alias_chain_passes']);self.assertFalse(result.get('bounded_chain_with_question_facts',False))

    def test_correct_law_does_not_excuse_reversed_subjects(self):
        result=wording_observation({'answer':'土生金，是后者生前者。结论：生。'},self.gold())
        self.assertTrue(result['explicit_conflicts']);self.assertFalse(result.get('bounded_chain_with_question_facts',False))

    def test_correct_chain_need_not_repeat_facts_already_in_question(self):
        result=wording_observation({'answer':'木克土，是前者克后者。结论：克。'},self.gold('克',('木','土')))
        self.assertTrue(result['bounded_chain_with_question_facts'])

    def test_wrong_cycle_cannot_pass_using_given_facts(self):
        result=wording_observation({'answer':'土生水，是前者生后者。结论：生。'},self.gold('克',('土','水')))
        self.assertTrue(result['explicit_conflicts']);self.assertFalse(result.get('bounded_chain_with_question_facts',False))

    def test_unambiguous_property_short_answer_is_observed(self):
        g={'checks':{'kind':'property','symbol':'巽四宫','element':'木'}}
        self.assertTrue(wording_observation({'answer':'属木。[S1]'},g)['unambiguous_short_property'])
        self.assertFalse(wording_observation({'answer':'属土。[S1]'},g)['unambiguous_short_property'])

    def test_same_source_id_different_quote_is_not_an_exact_duplicate(self):
        a={'source_id':'fixture','quote':'fixture A'};b={'source_id':'fixture','quote':'fixture B'}
        self.assertEqual(collapsed_selectors([a,b]),[a])
        self.assertEqual(collapsed_selectors([a,a]),[])


if __name__=='__main__':unittest.main()
