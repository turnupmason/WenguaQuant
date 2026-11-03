"""Auditable independent ML baselines, not a V9 arithmetic score translator."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .dataset import Cohort, Observation, PILOT_CLASSES, row_features

MODES = ('prior', 'context', 'chart', 'professional_facts', 'professional')


def stable_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _weights(rows: list[Observation]) -> list[float]:
    # One event has one aggregate unit of influence, including multi-item events.
    count = Counter(r.family for r in rows)
    return [1.0 / count[r.family] for r in rows]


def _prior(rows: list[Observation]) -> float:
    weights = _weights(rows)
    # Event-weighted Laplace prior; no test outcomes used.
    total = sum(weights)
    successes = sum(w for w, r in zip(weights, rows) if r.label == 'recovered')
    return (successes + 1) / (total + 2)


@dataclass
class Fitted:
    model: object | None
    encoder: object | None
    mask: object | None
    prior: float
    mode: str
    backend: str

    def predict(self, rows: list[Observation], cohort: Cohort) -> list[float]:
        if self.model is None:
            return [self.prior for _ in rows]
        import numpy as np
        x = [row_features(r, cohort, self.mode) for r in rows]
        matrix = self.encoder.transform(x).toarray()[:, self.mask]
        probs = self.model.predict_proba(matrix)
        classes = list(self.model.classes_)
        if 'recovered' not in classes:
            raise ValueError('Model lacks the declared positive class')
        return [float(p[classes.index('recovered')]) for p in probs]


def fit(rows: list[Observation], cohort: Cohort, mode: str, backend: str,
        *, seed: int = 1080, device: str = "CPU") -> Fitted:
    import numpy as np
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    if mode not in MODES or backend not in ('logistic', 'catboost'):
        raise ValueError('Unsupported mode/backend')
    if device not in ('CPU', 'GPU') or (backend == 'logistic' and device != 'CPU'):
        raise ValueError('Logistic requires CPU; only CatBoost supports GPU')
    p = _prior(rows)
    if mode == 'prior' or len({r.label for r in rows}) < 2:
        return Fitted(None, None, None, p, mode, backend)
    inputs = [row_features(r, cohort, mode) for r in rows]
    encoder = DictVectorizer(sparse=True)
    matrix = encoder.fit_transform(inputs).toarray()
    mask = np.ptp(matrix, axis=0) > 0  # training-fold-only filtering
    if not bool(mask.any()):
        return Fitted(None, None, None, p, mode, backend)
    x = matrix[:, mask]
    y = [r.label for r in rows]
    weights = np.array(_weights(rows))
    if backend == 'logistic':
        model = LogisticRegression(C=0.05, max_iter=1000, random_state=seed)
    else:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=120, depth=3, learning_rate=0.04,
                                   l2_leaf_reg=12, random_seed=seed, task_type=device,
                                   loss_function='Logloss', verbose=False,
                                   allow_writing_files=False, thread_count=2,
                                   **({'devices': '0'} if device == 'GPU' else {}))
    model.fit(x, y, sample_weight=weights)
    return Fitted(model, encoder, mask, p, mode, backend)


def _scores(rows: list[Observation], probabilities: list[float]) -> dict:
    if len(rows) != len(probabilities):
        raise ValueError('Prediction count mismatch')
    w = _weights(rows)
    tot = sum(w)
    brier = logloss = acc = 0.0
    for row, p, weight in zip(rows, probabilities, w):
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('Invalid numerical prediction')
        y = int(row.label == 'recovered')
        brier += weight * (p - y) ** 2
        q = min(max(p, 1e-12), 1 - 1e-12)
        logloss -= weight * (y * math.log(q) + (1-y) * math.log(1-q))
        acc += weight * (int(p >= 0.5) == y)
    return {'event_weighted_brier_binary': brier/tot,
            'event_weighted_log_loss': logloss/tot,
            'event_weighted_accuracy': acc/tot}


def evaluate(cohort: Cohort, backend: str = 'logistic', device: str = 'CPU') -> dict:
    observations = list(cohort.observations)
    groups = sorted({r.family for r in observations})
    class_events = {label: len({r.family for r in observations if r.label == label}) for label in PILOT_CLASSES}
    if len(groups) < 4 or min(class_events.values()) < 2:
        raise ValueError('Insufficient independently labelled event families to evaluate both classes')
    if backend not in ('logistic', 'catboost') or device not in ('CPU', 'GPU') or (backend == 'logistic' and device != 'CPU'):
        raise ValueError('Unsupported backend/device pair')
    if device == 'GPU':
        from catboost.utils import get_gpu_device_count
        if get_gpu_device_count() < 1:
            raise ValueError('GPU requested but no CatBoost GPU device detected; use --device CPU')
    findings, predictions = {}, []
    for mode in MODES:
        probs_by_row: dict[tuple[str, str], float] = {}
        for family in groups:
            train = [r for r in observations if r.family != family]
            held = [r for r in observations if r.family == family]
            fitted = fit(train, cohort, mode, backend, device=device)
            for r, p in zip(held, fitted.predict(held, cohort)):
                probs_by_row[(r.family, r.item)] = p
        probs = [probs_by_row[(r.family, r.item)] for r in observations]
        findings[mode] = _scores(observations, probs)
        predictions.extend({'family': r.family, 'item': r.item, 'case_id': r.case_id,
                            'mode': mode, 'label': r.label, 'research_score': p}
                           for r, p in zip(observations, probs))
    return {'schema': 'wenguaquant_research_v1', 'task': cohort.task, 'backend': backend, 'device': device,
            'protocol': 'group_leave_one_event_family_out',
            'event_families': len(groups), 'target_items': len(observations),
            'class_event_counts': class_events, 'source_archive_sha256': cohort.source_sha256,
            'feature_version': cohort.feature_version, 'files_sha256': cohort.files_sha256,
            'research_only': True, 'production_enabled': False,
            'probabilities_calibrated_for_users': False, 'independently_verified_outcomes': False,
            'scores_are_not_clinical_or_real_world_probabilities': True,
            'ablation_metrics': findings, 'held_out': predictions}


def save_research_run(cohort: Cohort, output: Path, *, backend: str = 'logistic', device: str = 'CPU') -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError('Run directory already exists; never overwrite an earlier result')
    output.mkdir(parents=True)
    try:
        report = evaluate(cohort, backend, device)
        held = report.pop('held_out')
        (output / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf8')
        (output / 'heldout_predictions.jsonl').write_text(''.join(stable_json(p)+'\n' for p in held), encoding='utf8')
        # Cross-validated research evaluation is distinct from final all-event fit.
        # No production model or fake calibrated percentages are published.
        (output / 'run_status.json').write_text(stable_json({'status': 'completed', 'research_only': True})+'\n', encoding='utf8')
        return report
    except Exception as e:
        (output / 'run_status.json').write_text(stable_json({'status': 'failed', 'error': type(e).__name__,
                                                               'reason': str(e)})+'\n', encoding='utf8')
        raise
