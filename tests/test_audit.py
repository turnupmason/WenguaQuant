"""Synthetic post-run diagnostic regressions, not independent field validation."""
import copy
import math
import unittest

from wengua_quant.audit import audit_saved_predictions


class AuditTests(unittest.TestCase):
    def setUp(self):
        # Two event families; one contains two different item outcomes.
        self.rows = []
        self.report = {'protocol': 'group_leave_one_event_family_out',
                       'production_enabled': False, 'probabilities_calibrated_for_users': False,
                       'event_families': 2, 'target_items': 3, 'ablation_metrics': {}}
        for mode in ('prior', 'chart'):
            # Identical diagnostic probabilities across two ablations to test completeness.
            base = [('a', 'bag', 'recovered', .8), ('a', 'cash', 'not_recovered', .8),
                    ('b', 'paper', 'recovered', .8)]
            per = []
            for family, item, label, p in base:
                row = {'mode': mode, 'family': family, 'item': item,
                       'label': label, 'research_score': p, 'case_id': family}
                self.rows.append(row)
                per.append(row)
            import math
            self.report['ablation_metrics'][mode] = {
                'event_weighted_accuracy': .75,
                'event_weighted_log_loss': (.5 * (-math.log(.8)) + .5 * (-math.log(.2)) - math.log(.8)) / 2,
                'event_weighted_brier_binary': (.5*.04 + .5*.64 + .04)/2,
            }

    def test_event_weighting_and_error_share(self):
        d = audit_saved_predictions(self.report, self.rows)
        self.assertEqual(d['event_families'], 2)
        self.assertEqual(d['observed_item_labels'], {'recovered': 2, 'not_recovered': 1})
        self.assertEqual(d['by_mode']['prior']['predicted_item_labels'], {'recovered': 3})
        self.assertAlmostEqual(d['by_mode']['prior']['event_weighted_accuracy'], .75)
        self.assertGreater(d['by_mode']['prior']['negative_item_log_loss_share'], .5)
        self.assertEqual(d['by_mode']['prior']['worst_log_loss_items'][0]['actual'], 'not_recovered')

    def test_reject_missing_prediction_and_inconsistent_label(self):
        with self.assertRaisesRegex(ValueError, 'identical'):
            audit_saved_predictions(self.report, self.rows[:-1])
        bad = copy.deepcopy(self.rows)
        bad[-1]['label'] = 'not_recovered'
        with self.assertRaisesRegex(ValueError, 'Different outcome'):
            audit_saved_predictions(self.report, bad)

    def test_reject_corrupted_and_unvalidated_report(self):
        bad = copy.deepcopy(self.rows)
        bad[0]['research_score'] = math.nan
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            audit_saved_predictions(self.report, bad)
        bad = copy.deepcopy(self.report)
        bad['probabilities_calibrated_for_users'] = True
        with self.assertRaisesRegex(ValueError, 'uncalibrated'):
            audit_saved_predictions(bad, self.rows)
        bad = copy.deepcopy(self.report)
        bad['ablation_metrics']['chart']['event_weighted_log_loss'] = .5
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            audit_saved_predictions(bad, self.rows)


if __name__ == '__main__':
    unittest.main()
