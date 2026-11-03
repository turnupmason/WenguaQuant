# Mac professional V2 feature-catalog validation failure · 2026-11-03

## Source and observed reproduction

User's Mac run on 2026-11-03 11:39 local time, using the WenguaQuant v0.1 runner linked to an existing WenguaApp checkout:

```text
[11:39:18] Read-only archive audit: all 194 events; labels remain separate from inputs
[11:39:21] Export complete: 194 primary records, 19 additional casts, 4606 professional fields
[11:39:21] Trainable pilot: 12 item targets / 11 independent events
ValueError: Unsafe, missing-source, or duplicated feature catalog entry
```

Export itself succeeded; the receiver rejected the generated catalog in `wengua_quant/dataset.py`, before data loading or model fitting. No real-case training result exists for this run. The local export directory and failure log are not in the public repository, and original source materials were not copied.

## Root cause and source evidence

- WenguaApp `src/wengua/probability_features_v2.py`, in `flatten_bundle()`, emits `heuristic.role_edge.<source_role>.<target_role>.control_matrix_result` as a derived traditional control matrix fact for every role-edge. It comes from the source's pre-question `user_domain_matrices_20261023.py`, **not** the later feedback label.
- WenguaApp `scripts/probability_professional.py`, `feature_group()` maps `heuristic.` fields into `fixed_user_heuristics`.
- WenguaQuant v0.1 `wengua_quant/dataset.py`, `FORBIDDEN` previously matched any substring `result`, unintentionally rejecting `control_matrix_result`.

Expected: load the source index and classify the derived matrix value as a fixed manual heuristic input **only** in the `professional` ablation.

Observed: reject entire catalog with no offending feature ID; training never started.

## Correction and limits

`_unsafe_feature_id(key, group)` allows only exact `heuristic.role_edge.<source>.<target>.control_matrix_result` names belonging to `fixed_user_heuristics`. It masks only that one suffix before the existing forbidden-token scan, so a role name such as `future_label` or `author_result` remains blocked. It never accepts a generalized `result` column or changes the supervised target. Catalog validation errors now identify the field and group.

Regression: complete V0.1 synthetic contract test suite plus exact positive/negative source-style name cases and confirmation that this column appears in `professional` but not `professional_facts`. Local synthetic tests passed, 9/9. Real-export rerun and metric report remain to be checked on the user's Mac; do not infer them from synthetic tests or existing CI. WenguaApp production V9/V11, V27/V4 and the historical archive are not edited.
