import unittest
from research_audits.v54_review import reasoning_overlay,name_overlay


class V54ReviewOverlay(unittest.TestCase):
 def test_correct_headline_does_not_hide_reversed_subjects(self):
  self.assertTrue(reasoning_overlay('受克。土克水，是前者克后者。',['水','土']))
  self.assertFalse(reasoning_overlay('受克。土克水，是后者克前者。',['水','土']))
 def test_false_literal_cycle_is_still_rejected(self):
  self.assertTrue(reasoning_overlay('受生。木生金。',['金','木']))
 def test_full_natural_name_sentence_is_observed_separately_from_semantics(self):
  gold={'checks':{'answer':'fixture_name'}}
  result=name_overlay({'answer':'天盘甲加地盘乙，是fixture_name。'},gold,{'fixture_name','other_name'})
  self.assertTrue(result['expected_name_present']);self.assertFalse(result['semantic_approval'])
 def test_correct_name_mixed_with_alternative_is_not_unambiguous(self):
  result=name_overlay({'answer':'命名取fixture_name、unknown_name。'},{'checks':{'answer':'fixture_name'}},{'fixture_name'})
  self.assertTrue(result['expected_name_present']);self.assertTrue(result['explicit_name_alternatives']);self.assertFalse(result['only_expected_among_known_names'])
 def test_negated_name_is_not_counted_as_the_answer(self):
  self.assertFalse(name_overlay({'answer':'不是fixture_name。'},{'checks':{'answer':'fixture_name'}},{'fixture_name'})['expected_name_present'])

if __name__=='__main__':unittest.main()
