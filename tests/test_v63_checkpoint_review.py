import tempfile, unittest
from pathlib import Path
from research_audits import v63_checkpoint_review as review
from wengua_course import data as d


class ActualCheckpointReviewBoundaries(unittest.TestCase):
    def test_only_this_actual_upload_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'other.zip'; path.write_bytes(b'another experiment')
            with self.assertRaises(ValueError): review.audit(path, Path(tmp)/'review')

    def test_zero_next_token_change_can_still_have_real_logit_effect(self):
        source = {'adapter_sha256': 'reported actual source'}
        value = {'software_fixture': False, 'parameters_frozen': True, 'weight_updates': 0,
                 'adapter_sha256': source['adapter_sha256'], 'lora_B_matrices': 252,
                 'nonzero_lora_B_matrices': 252, 'max_last_position_logit_delta': 22.0,
                 'next_token_changed': False}
        review.check_activation(value, source)
        for key, bad in [('weight_updates', 1), ('parameters_frozen', False),
                         ('adapter_sha256', 'other weights'), ('nonzero_lora_B_matrices', 0),
                         ('max_last_position_logit_delta', 0), ('software_fixture', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                review.check_activation(value | {key: bad}, source)

    def test_every_supplied_annotation_is_bound_to_the_whole_saved_answer(self):
        row = {'id': 'fixture', 'answer': 'correct start; wrong final statement'}
        notes = {'model': {'fixture': {'answer_sha256': d.digest(row['answer']), 'verdict': 'contradictory'}}}
        self.assertEqual(review.bound_note(notes, 'model', row)['verdict'], 'contradictory')
        with self.assertRaises(ValueError): review.bound_note(notes, 'model', row | {'answer': 'correct start'})
        with self.assertRaises(ValueError): review.bound_note(notes, 'other model', row)
        self.assertIsNone(review.bound_note({}, 'model', row))

    def test_curriculum_counts_do_not_count_same_self_as_an_identity_change(self):
        row = {'split': 'train', 'skill': 'identity_switch', 'answer': 'fixture', 'case_ids': [],
               'messages': [{'role': 'system', 'content': ''}, {'role': 'user', 'content': '按本人取谁？'}]}
        pack = {'records': [row, row | {'split': 'validation'}]}
        result = review.source_inventory(pack)
        self.assertEqual(result['training_cores'], 1)
        self.assertEqual(result['same_self_to_self_cores'], 1)
        self.assertEqual(result['case_linked_cores'], 0)
        self.assertTrue(result['not_a_full_two_book_case_curriculum'])


if __name__ == '__main__': unittest.main()
