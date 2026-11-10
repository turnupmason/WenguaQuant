import unittest
from copy import deepcopy
from unittest.mock import patch
from pathlib import Path
from wengua_course import data as d
from wengua_course_v63 import dataset as data
from research_audits import v62_learning_review as review


class V63SourceRepair(unittest.TestCase):
    def pack(self):
        rows = []
        for skill in data.RECALL_SKILLS:
            for i in range(2):
                rows.append({'id': f'fixture:{skill}:{i}', 'skill': skill, 'split': 'train',
                             'answer': 'source fixture answer', 'question': 'fixture question',
                             'messages': [{'role': 'system', 'content': 'old system'},
                                          {'role': 'user', 'content': 'fixture question'},
                                          {'role': 'assistant', 'content': 'source fixture answer'}]})
        return {'schema': 'fixture', 'units': {'s': {'original_quote': 'literal fixture'}},
                'case_splits': {'fixture_case': 'train'}, 'summary': {}, 'records': rows,
                'actual_rag_rows': deepcopy(rows[:1]), 'known_regression_rows': deepcopy(rows[1:2])}

    def derive(self):
        p = self.pack()
        with patch.object(data.source, 'validate_pack'):
            q = data.derive(p)
        return p, q

    def test_only_system_and_declared_metadata_change(self):
        before, after = self.derive()
        with patch.object(data.source, 'validate_pack'):
            self.assertTrue(data.verify_derivation(before, after))
        for key in ['records', 'actual_rag_rows', 'known_regression_rows']:
            for a, b in zip(before[key], after[key], strict=True):
                self.assertEqual(a['messages'][1:], b['messages'][1:])
                self.assertEqual(a['answer'], b['answer'])
                self.assertEqual(b['messages'][0]['content'], data.SYSTEM)
        self.assertEqual(before['units'], after['units'])
        self.assertEqual(before['case_splits'], after['case_splits'])

    def test_changed_teacher_user_or_source_rejected(self):
        before, good = self.derive()
        for kind in ['teacher', 'user', 'source']:
            bad = deepcopy(good)
            if kind == 'teacher': bad['records'][0]['answer'] = 'invented answer'
            elif kind == 'user': bad['records'][0]['messages'][1]['content'] = 'later feedback'
            else: bad['units']['s']['original_quote'] = 'another book'
            with patch.object(data.source, 'validate_pack'), self.assertRaises(ValueError):
                data.verify_derivation(before, bad)

    def test_recall_contains_only_existing_distinct_training_rows(self):
        _, pack = self.derive(); chosen = data.learning_rows(pack)
        self.assertEqual(len(chosen), 12)
        self.assertEqual(len({r['id'] for r in chosen}), 12)
        self.assertTrue(all(r in pack['records'] and r['split'] == 'train' for r in chosen))
        self.assertFalse(any(r is pack['actual_rag_rows'][0] for r in chosen))

    def test_all_seven_projection_structure_and_finite_loop(self):
        root = Path(__file__).resolve().parents[1]
        c = d.read(root / 'configs/course_v63.json')
        self.assertEqual(c['target_modules'], data.MODULES)
        self.assertEqual(c['num_train_epochs'], 8)
        self.assertEqual(c['gradient_accumulation_steps'], 8)
        self.assertEqual(c['generation_epochs'], [1, 2, 4, 8])
        from wengua_course_v63.runner import RATES
        self.assertEqual(RATES, (5e-5, 1e-4))
        self.assertFalse(c['production_enabled']); self.assertFalse(c['non_book_teachers_allowed'])

    def test_new_review_refuses_unrelated_archive(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'another.zip'; path.write_bytes(b'not the actual upload')
            with self.assertRaises(ValueError): review.audit(path, Path(tmp) / 'review')

    def test_export_name_is_v63_and_uses_downloads(self):
        from wengua_course_v63 import cli
        with patch('wengua_course.runner.export_review', return_value='fixture.zip') as export, patch('builtins.print'):
            self.assertEqual(cli.main(['export']), 0)
        self.assertTrue(export.call_args.args[1].name.startswith('Wengua_V63_review_'))


if __name__ == '__main__':
    unittest.main()
