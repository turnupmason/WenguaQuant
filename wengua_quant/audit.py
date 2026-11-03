"""Post-run diagnostic of held-out research predictions; never train or tune.

This command reads a completed run's output and prints failure concentration,
including event-family weighting. It does not change models or existing files.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def audit_saved_predictions(report: dict, rows: list[dict]) -> dict:
    """Audit a saved, grouped leave-one-event-family-out research run.

    Output scores are uncalibrated book-report research scores. This function
    never claims empirical prediction validity and does not select a model.
    """
    modes = list(report.get('ablation_metrics', {}))
    if not modes or not rows:
        raise ValueError('Missing ablation metrics or held-out predictions')
    if report.get('protocol') != 'group_leave_one_event_family_out':
        raise ValueError('Expected grouped leave-one-event-family-out research protocol')
    if report.get('production_enabled') is not False or report.get('probabilities_calibrated_for_users') is not False:
        raise ValueError('Refusing to audit as an uncalibrated research-only run')
    per_mode: dict[str, dict[tuple[str, str], dict]] = {mode: {} for mode in modes}
    known_labels: dict[tuple[str, str], str] = {}
    for row in rows:
        mode, family, item, label = (row.get(k) for k in ('mode', 'family', 'item', 'label'))
        score = row.get('research_score')
        if mode not in per_mode or not all(isinstance(v, str) and v for v in (family, item)):
            raise ValueError('Unexpected mode or missing event/item identifier')
        if label not in ('recovered', 'not_recovered'):
            raise ValueError('Unsupported observed outcome')
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid held-out research score')
        key = (family, item)
        if key in per_mode[mode]:
            raise ValueError('Duplicate mode/event/item prediction')
        if key in known_labels and known_labels[key] != label:
            raise ValueError('Different outcome labels across ablation modes')
        known_labels[key] = label
        per_mode[mode][key] = row
    keys = set(known_labels)
    if any(set(mode_rows) != keys for mode_rows in per_mode.values()):
        raise ValueError('Ablation modes do not cover identical held-out observations')
    family_size = Counter(family for family, _ in keys)
    families = len(family_size)
    if families != report.get('event_families') or len(keys) != report.get('target_items'):
        raise ValueError('Held-out keys disagree with evaluation metadata')
    output = {
        'status': 'research_diagnostics_only',
        'user_probability_calibrated': False,
        'event_families': families,
        'target_items': len(keys),
        'observed_item_labels': dict(Counter(known_labels.values())),
        'by_mode': {},
    }
    for mode, dataset in per_mode.items():
        weight_correct, weighted_log_loss = 0., 0.
        weighted_brier = 0.
        predicted = Counter()
        confusion = defaultdict(float)
        losses = []
        for (family, item), row in sorted(dataset.items()):
            score = float(row['research_score'])
            label = row['label']
            prediction = 'recovered' if score >= 0.5 else 'not_recovered'
            predicted[prediction] += 1
            weight = 1.0 / family_size[family]
            hit = prediction == label
            weight_correct += weight * hit
            y = int(label == 'recovered')
            p = min(max(score, 1e-12), 1 - 1e-12)
            loss = -(y * math.log(p) + (1-y) * math.log(1-p))
            weighted_log_loss += weight * loss
            weighted_brier += weight * (score - y) ** 2
            confusion[f'{label} / predicted_{prediction}'] += weight
            losses.append({'case_id': row.get('case_id'), 'event_family': family,
                           'target_item': item, 'actual': label, 'predicted': prediction,
                           'research_score': score, 'loss_contribution': weight * loss / families,
                           'event_weight': weight})
        metrics = report['ablation_metrics'][mode]
        accuracy = weight_correct / families
        logloss = weighted_log_loss / families
        brier = weighted_brier / families
        for key, computed in (('event_weighted_accuracy', accuracy),
                              ('event_weighted_log_loss', logloss),
                              ('event_weighted_brier_binary', brier)):
            if not math.isclose(computed, metrics[key], rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError(f'Saved {mode} metric mismatch: {key}')
        negative_contribution = sum(r['loss_contribution'] for r in losses if r['actual'] == 'not_recovered')
        output['by_mode'][mode] = {
            'predicted_item_labels': dict(predicted),
            'event_weighted_accuracy': accuracy,
            'event_weighted_log_loss': logloss,
            'event_weighted_brier': brier,
            'event_weighted_confusion': dict(confusion),
            'negative_item_log_loss_share': negative_contribution / logloss if logloss else 0,
            'worst_log_loss_items': sorted(losses, key=lambda r: r['loss_contribution'], reverse=True)[:3],
        }
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Completed WenguaQuant local/runs/<timestamp> directory')
    args = parser.parse_args(argv)
    root = args.run
    report = json.loads((root / 'evaluation.json').read_text(encoding='utf-8'))
    rows = [json.loads(x) for x in (root / 'heldout_predictions.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    result = audit_saved_predictions(report, rows)
    print(f'Grouped held-out audit: {result["event_families"]} events, {result["target_items"]} items')
    print('Observed item labels:', result['observed_item_labels'])
    for mode, data in result['by_mode'].items():
        print(f'\n{mode}: predicted={data["predicted_item_labels"]}, '
              f'weighted_accuracy={data["event_weighted_accuracy"]:.4f}, '
              f'log_loss={data["event_weighted_log_loss"]:.4f}, '
              f'negative_item_share_of_log_loss={data["negative_item_log_loss_share"]:.1%}')
        for case in data['worst_log_loss_items']:
            print(f'  loss={case["loss_contribution"]:.4f} '
                  f'{case["case_id"]} item={case["target_item"]} '
                  f'actual={case["actual"]} score={case["research_score"]:.4f}')
    print('\nResearch only: these are not calibrated real-world probabilities.')


if __name__ == '__main__':
    main()
