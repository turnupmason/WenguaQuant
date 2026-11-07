import io
import unittest
import zipfile
from research_audits.v55_review import normalized_content, answer_overlay, bridge_boundary_candidates, inspect_archive


class V55Review(unittest.TestCase):
    def test_numbering_is_not_semantic_accuracy(self):
        self.assertEqual(normalized_content('1、fixture A\n2、fixture B [S1]'),
                         normalized_content('fixture A\nfixture B [S9]'))

    def test_correct_headline_does_not_approve_false_cycle(self):
        row = {'answer': '受生。土克金。'}
        gold = {'answer': '受生。土生金。', 'checks': {'kind':'relation','answer':'受生','elements':['金','土']}}
        result = answer_overlay(row, gold)
        self.assertTrue(result['headline_matches_target'])
        self.assertTrue(result['literal_or_subject_conflicts'])
        self.assertFalse(result['semantic_professional_approval'])

    def test_natural_correction_headline_is_observed(self):
        gold = {'answer':'生。水生木。','checks':{'kind':'relation','answer':'生','elements':['水','木']}}
        result = answer_overlay({'answer':'应改生。水生木。'}, gold)
        self.assertTrue(result['headline_matches_target'])
        self.assertFalse(result['semantic_professional_approval'])

    def test_missing_premise_candidate_not_automatic_invalidity(self):
        gold = {'skill':'author_evidence','provenance':{'literal_target':'fixture结论，伤门伏吟。'},
                'messages':[{'role':'user','content':'作者证据：fixture input'} , {'role':'assistant','content':'unused'}]}
        self.assertEqual(bridge_boundary_candidates(gold), ['伤门伏吟'])
        gold['messages'][0]['content'] += '，伤门伏吟'
        self.assertEqual(bridge_boundary_candidates(gold), [])

    def test_archive_traversal_rejected_without_extracting(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream,'w') as z:
            z.writestr('../escape.json', '{}')
        with zipfile.ZipFile(stream) as z:
            with self.assertRaises(ValueError):
                inspect_archive(z)

    def test_plain_member_accepted(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream,'w') as z:
            z.writestr('trials/fixture.json', '{}')
        with zipfile.ZipFile(stream) as z:
            inspect_archive(z)


if __name__ == '__main__':
    unittest.main()
