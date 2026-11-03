"""Read-only, leak-resistant adapter for WenguaApp professional V2 export.

We intentionally do NOT use original-case prose, author judgments, model scores,
book-success labels, post-question revelations, or V9 rule totals as predictors.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SOURCE_SHA = '21362e6f0266c03dcdc4a7ef35f45954ba274ae6458de0ba341132abca7d0405'
PILOT_TASK = 'lost.recovered_at_last_report'
PILOT_CLASSES = ('not_recovered', 'recovered')
SAFE_CONTEXT = frozenset({'item_kind', 'reference_frame', 'subject_species', 'question_kind',
                          'age_band', 'sex', 'known_diagnosis_code', 'treatment_status',
                          'last_seen_environment', 'search_duration_band', 'item_movable',
                          'location_time'})
SAFE_GROUPS = frozenset({'explicit_calendar', 'pre_question_role_context', 'macro_conditions',
                         'palace_residents_season_void_horse', 'stem_lifecycle_and_hidden_jia',
                         'source_pattern_primitives', 'directed_palace_relations',
                         'conditional_signifiers', 'independent_cohost_identity',
                         'stem_branch_time_relations', 'explicit_unknown_rules',
                         'fixed_user_heuristics'})
FORBIDDEN = re.compile(r'(label|outcome|result|feedback|verdict|conclusion|prediction|author_|'
                       r'health_score|v9_score|post_question|recovered_at|future_|'
                       r'event_family_id|case_id|section_key|source_book)', re.I)


@dataclass(frozen=True)
class Observation:
    case_id: str
    family: str
    item: str
    label: str
    context: dict[str, str]
    chart: tuple[dict[str, Any], ...]
    professional: dict[str, str]


@dataclass(frozen=True)
class Cohort:
    task: str
    observations: tuple[Observation, ...]
    source_sha256: str
    feature_version: str
    files_sha256: dict[str, str]
    professional_groups: dict[str, str]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]


def _source_dir(path: Path) -> Path:
    # Accept either WenguaApp's 'export-root/data' or the 'data' directory itself.
    return path / 'data' if (path / 'data' / 'rows.jsonl').is_file() else path


def load_v2_export(path: Path, *, expected_source_sha: str = SOURCE_SHA) -> Cohort:
    root = _source_dir(Path(path))
    names = ('inventory.json', 'feature_catalog.json', 'flat_features.tsv', 'rows.jsonl')
    paths = {name: root / name for name in names}
    for name, p in paths.items():
        if not p.is_file():
            raise ValueError(f'Missing export file: {name}; run WenguaApp V2 export first')
    inventory = _load_json(paths['inventory.json'])
    catalog = _load_json(paths['feature_catalog.json'])
    sha = inventory.get('source_sha256')
    if sha != expected_source_sha:
        raise ValueError(f'Unreviewed source archive SHA; expected {expected_source_sha}, got {sha}')
    if not str(catalog.get('schema', '')).startswith('professional_features_v2_'):
        raise ValueError('Only audited WenguaApp professional V2 features are supported')
    entries = catalog.get('features')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Feature catalog missing its declared source-indexed fields')
    groups: dict[str, str] = {}
    for entry in entries:
        key, group = entry.get('feature_id'), entry.get('group')
        if (not isinstance(key, str) or not key or FORBIDDEN.search(key)
                or group not in SAFE_GROUPS or entry.get('input_only') is not True
                or not entry.get('source_paths') or key in groups):
            raise ValueError('Unsafe, missing-source, or duplicated feature catalog entry')
        groups[key] = group
    with paths['flat_features.tsv'].open(encoding='utf8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if reader.fieldnames is None or reader.fieldnames[0] != 'case_id':
            raise ValueError('Flat table must begin with case_id')
        if set(reader.fieldnames[1:]) != set(groups):
            raise ValueError('Flat table does not exactly match source feature catalog')
        flat: dict[str, dict[str, str]] = {}
        for line in reader:
            case_id = line.pop('case_id')
            if not case_id or case_id in flat:
                raise ValueError('Duplicate or missing flat-table case_id')
            if any(value is None for value in line.values()):
                raise ValueError('Incomplete flat-table row')
            flat[case_id] = line
    records = _parse_rows(paths['rows.jsonl'])
    if not records:
        raise ValueError('No independent labelled observations in export')
    task = records[0].get('task')
    if task != PILOT_TASK:
        raise ValueError('Only the defined book-reported lost-item pilot is currently label-approved')
    observations: list[Observation] = []
    seen: set[tuple[str, str, str]] = set()
    for r in records:
        if r.get('task') != task or r.get('label') not in PILOT_CLASSES:
            raise ValueError('Mixed tasks or unsupported outcome labels')
        if r.get('available_before_prediction') is not True:
            raise ValueError('Chart must have been available at prediction time')
        evidence = r.get('evidence')
        if not isinstance(evidence, dict) or not re.fullmatch(r'[0-9a-f]{64}', str(evidence.get('feedback_sha256', ''))):
            raise ValueError('An observed outcome needs an original feedback SHA receipt')
        if evidence.get('kind') != 'book_report_only_not_external_verification':
            raise ValueError('Book-report outcome evidence kind must be explicit')
        case, family, item = (r.get(field) for field in ('case_id', 'event_family_id', 'target_item'))
        if not all(isinstance(x, str) and x for x in (case, family, item)):
            raise ValueError('Missing event-family or target identifier')
        if (family, task, item) in seen:
            raise ValueError('Multiple questions/casts of one event-item cannot be duplicate samples')
        seen.add((family, task, item))
        if case not in flat:
            raise ValueError('Labelled case missing source feature record')
        ctx = r.get('context')
        if not isinstance(ctx, dict) or set(ctx) - SAFE_CONTEXT or any(not isinstance(v, str) or not v for v in ctx.values()):
            raise ValueError('Context must contain only categorical pre-question information')
        chart = r.get('chart')
        if not isinstance(chart, list) or len(chart) != 9 or {p.get('palace') for p in chart} != set(range(1, 10)):
            raise ValueError('Require actual source chart with nine numbered palace cells')
        safe_chart = []
        for p in chart:
            if set(p) != {'palace', 'deity', 'door', 'star', 'door_stems', 'star_stems'}:
                raise ValueError('Unexpected chart attribute; block potential outcome leakage')
            if any(not isinstance(p[k], str) for k in ('deity', 'door', 'star', 'door_stems', 'star_stems')):
                raise ValueError('Chart symbols must be literal source strings')
            safe_chart.append(dict(p))
        observations.append(Observation(case, family, item, r['label'], dict(ctx),
                                        tuple(safe_chart), dict(flat[case])))
    if inventory.get('exported_rows') != len(observations) or inventory.get('exported_event_families') != len({r.family for r in observations}):
        raise ValueError('Export inventory row / family counts inconsistent with labels')
    return Cohort(task, tuple(observations), sha, str(catalog['schema']),
                  {n: _fingerprint(p) for n, p in paths.items()}, groups)


def row_features(row: Observation, cohort: Cohort, mode: str) -> dict[str, str | float]:
    if mode not in ('prior', 'context', 'chart', 'professional_facts', 'professional'):
        raise ValueError('Unknown registered ablation mode')
    if mode == 'prior':
        return {}
    x: dict[str, str | float] = {'context.' + k: v for k, v in row.context.items()}
    if mode == 'context':
        return x
    for p in row.chart:
        n = p['palace']
        for k in ('deity', 'door', 'star', 'door_stems', 'star_stems'):
            x[f'chart.p{n}.{k}'] = p[k] or '__unknown__'
    if mode == 'chart':
        return x
    for key, value in row.professional.items():
        if cohort.professional_groups[key] == 'fixed_user_heuristics' and mode == 'professional_facts':
            continue
        # Include explicit 'unknown' and 'not_applicable' as distinct categories.
        x['professional.' + key] = value if value else '__unknown__'
    return x
