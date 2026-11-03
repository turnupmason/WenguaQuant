"""Synthetic contract tests only; NOT additional historical cases."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from wengua_quant.dataset import SOURCE_SHA, load_v2_export, row_features
from wengua_quant.model import evaluate, fit, MODES, save_research_run


def fixture(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    groups = {'p1.deity': 'palace_residents_season_void_horse',
              'role.tianxin_palace': 'conditional_signifiers',
              'heuristic.star_power': 'fixed_user_heuristics'}
    entries = [{'feature_id': key, 'group': group, 'input_only': True,
                'source_paths': ['src/wengua/original_source.py']} for key, group in groups.items()]
    (root / 'feature_catalog.json').write_text(json.dumps({'schema': 'professional_features_v2_test_synthetic',
                                                            'features': entries}), encoding='utf8')
    rows = []
    for i in range(8):
        family = f'fictional-event-{i}'
        ids = [f'case-{i}']
        if i == 2:  # two items, one event family
            ids.append('case-2-cash')
        for case in ids:
            chart = [dict(palace=p, deity='值符' if i%2 else '白虎', door='开', star='心',
                          door_stems='乙', star_stems='乙') for p in range(1, 10)]
            rows.append({'case_id': case, 'event_family_id': family,
                         'target_item': 'cash' if case.endswith('cash') else 'object',
                         'task': 'lost.recovered_at_last_report',
                         'label': 'recovered' if i%2 else 'not_recovered',
                         'context': {'item_kind': 'hidden' if i%2 else 'document'},
                         'chart': chart, 'available_before_prediction': True,
                         'evidence': {'feedback_sha256': 'a'*64,
                                      'kind': 'book_report_only_not_external_verification'}})
    (root / 'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf8')
    with (root / 'flat_features.tsv').open('w', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['case_id']+list(groups), delimiter='\t')
        w.writeheader()
        for r in rows:
            i = int(r['event_family_id'].split('-')[-1])
            w.writerow({'case_id': r['case_id'], 'p1.deity': '值符' if i%2 else '白虎',
                        'role.tianxin_palace': str((i%8)+1),
                        'heuristic.star_power': '极大' if i%2 else '小'})
    (root / 'inventory.json').write_text(json.dumps({'source_sha256': SOURCE_SHA,
        'exported_rows': len(rows), 'exported_event_families': 8}), encoding='utf8')
    return rows


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rows = fixture(self.root / 'data')

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_and_same_event(self):
        cohort = load_v2_export(self.root)
        self.assertEqual(len(cohort.observations), 9)
        self.assertEqual(len({r.family for r in cohort.observations}), 8)
        self.assertEqual(len(cohort.professional_groups), 3)

    def test_ablation_isolation(self):
        c = load_v2_export(self.root)
        r = c.observations[0]
        for mode in MODES:
            x = row_features(r, c, mode)
            self.assertFalse(any('label' in k or 'case_id' in k for k in x))
        self.assertEqual(row_features(r, c, 'prior'), {})
        self.assertEqual(len(row_features(r, c, 'context')), 1)
        self.assertIn('professional.role.tianxin_palace', row_features(r, c, 'professional_facts'))
        self.assertNotIn('professional.heuristic.star_power', row_features(r, c, 'professional_facts'))
        self.assertIn('professional.heuristic.star_power', row_features(r, c, 'professional'))

    def test_grouped_loo_research_only(self):
        c = load_v2_export(self.root)
        report = evaluate(c)
        self.assertEqual(report['event_families'], 8)
        self.assertEqual(report['target_items'], 9)
        self.assertEqual(len(report['held_out']), 9 * 5)
        self.assertFalse(report['production_enabled'])
        self.assertFalse(report['probabilities_calibrated_for_users'])
        for part in report['ablation_metrics'].values():
            self.assertTrue(0 <= part['event_weighted_brier_binary'] <= 1)

    def test_holdout_label_does_not_affect_own_predictions(self):
        c = load_v2_export(self.root)
        train = [r for r in c.observations if r.family != 'fictional-event-0']
        test = [r for r in c.observations if r.family == 'fictional-event-0']
        model = fit(train, c, 'professional_facts', 'logistic')
        p = model.predict(test, c)
        from dataclasses import replace
        wrong_label = [replace(r, label='recovered') for r in test]
        self.assertEqual(p, model.predict(wrong_label, c))

    def test_negative_duplicate_and_schema(self):
        base = self.root / 'data'
        with (base / 'rows.jsonl').open('a', encoding='utf8') as f:
            f.write(json.dumps(self.rows[0])+'\n')
        with self.assertRaisesRegex(ValueError, 'duplicate samples'):
            load_v2_export(self.root)

    def test_reject_feature_leakage(self):
        p = self.root / 'data/feature_catalog.json'
        d = json.loads(p.read_text(encoding='utf8'))
        d['features'][0]['feature_id'] = 'future_feedback'
        p.write_text(json.dumps(d), encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            load_v2_export(self.root)

    def test_reject_mismatched_archive(self):
        p = self.root / 'data/inventory.json'
        d = json.loads(p.read_text(encoding='utf8'))
        d['source_sha256'] = 'bad'
        p.write_text(json.dumps(d), encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'Unreviewed source'):
            load_v2_export(self.root)

    def test_no_overwrite(self):
        c = load_v2_export(self.root)
        out = self.root / 'result'
        save_research_run(c, out)
        self.assertTrue((out / 'evaluation.json').exists())
        with self.assertRaises(FileExistsError):
            save_research_run(c, out)


if __name__ == '__main__':
    unittest.main()
