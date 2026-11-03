# WenguaQuant · independent outcome learning protocol v0.1

## Research objective

The current objective is **not** to reproduce WenguaApp's V9 scoring. Estimate associations between *pre-question professional facts* and independently defined later outcome labels, evaluating unseen **event families**, not unseen textual variations of the same story. Full-source historical outcomes are book-reported, nonprospective observations, not externally authenticated clinical outcomes. Code correctness ≠ predictive validity.

## Data contract and source pin

Input comes from a **local, read-only, source-audited WenguaApp V2 export**: `inventory.json`, `feature_catalog.json`, `flat_features.tsv`, and `rows.jsonl`, produced by WenguaApp's `run_professional_training.py export`. The export's source archive SHA must match `21362e6f0266c03dcdc4a7ef35f45954ba274ae6458de0ba341132abca7d0405`. The original 194-event SQLite database and book text stay in WenguaApp; WenguaQuant has no permission to rewrite them. No copyrighted original-book passages are copied into this public repository.

All feature columns must match the source-indexed catalog exactly. Every catalog feature must declare `input_only`, a known group and nonempty source paths. The adapter uses a strict allowlist of predictor namespaces and forbids outcome-like columns, trial IDs, source sections, author verdicts or rule-total scores as predictors. It separately reads the historically reported outcome label and its SHA receipt. Unresolved and not-applicable values remain distinct categorical inputs. Do not fill missing facts from later feedback, recast a contradictory historical chart, or duplicate additional casts as independent observations.

The current pilot target is **`lost.recovered_at_last_report`**, the book's final reported state of a specific item. `recovered` and `not_recovered` are its two labels. This is NOT eventual recovery, a probability of finding an item today, nor a fixed follow-up interval. Unequal follow-up and strong source selection bias must remain visible. Other domain labels must be separately registered after assessing definition, true negative observation, horizon, confidence and event count; a multi-label body-site registry cannot infer "healthy" from "not mentioned". Clinical severity needs a predeclared objective classification and outcome time, and recovery time needs survival/competing-risk handling (death / censoring).

## Modelling & ablation

Five **precommitted** feature groups are compared using the same held-out event-family folds:

1. `prior`: smoothed event-weighted class rate (no chart)
2. `context`: question-time background only
3. `chart`: background + literal nine-palace fields
4. `professional_facts`: chart + source-indexed professional features, **excluding user-imposed scoring/matrices**
5. `professional`: all preceding inputs + separately identified fixed traditional/manual mappings (ablation only)

Models: L2 regularized logistic regression (`C=0.05`) and optional CatBoost (120 trees, depth 3), CPU or GPU when explicitly selected; training partitions alone build the category vocabulary and remove constant columns. Event-weighted metrics give each event family one unit of influence, even if a single case has two item targets. No tuning on holdout metrics, no deterministic V9 outcome label creation, no publication of model scores as user-facing probabilities. A train fold with one observed class uses the predeclared smoothed prior; dataset with fewer than two groups per class rejects evaluation.

### Initial testing design

At current N≈11 independent lost-property events, grouped leave-one-event-family-out is **diagnostic only**, not a secret blind test. Compare Brier/log loss against the prior. No post-hoc best-model selection, probability calibration, medical claims or iOS production integration. Once a task has sufficiently many independent events and outcome classes, pre-register chronological and source-family-held-out external tests, with a separate calibration split; do not simply add 1080 unlabeled charts as 1080 observed results.

## Audit controls

Runs write exclusively into newly named `local/runs/` directories, preserve failures, hash input files, and retain per-fold group-held-out predictions and comparison metrics. `production_enabled=false` hardcoded in reports. Unit tests are **synthetic software fixtures**, never additional historical validation cases. A new dataset/source version must be reviewed and pinned explicitly; no automatic promotion to production.

## Source naming collision and permanent negative case (2026-11-03)

Source V2 uses the literal name `control_matrix_result` to describe **a traditional, pre-question rule-matrix calculation**, not the subsequently reported real-world event result. Its complete feature ID is `heuristic.role_edge.<source_role>.<target_role>.control_matrix_result`. Because a generic forbidden-word check matched the substring `result`, the first real Mac export was refused after 194 records and 4,606 features were exported. The revised validator allows only this exact role-edge shape when the source group is `fixed_user_heuristics`; the rest of the feature name is still scanned for forbidden outcome terms. This field is available only in the full `professional` ablation, never in `professional_facts`. Do **not** generalize this exception to arbitrary keys ending in `result`, and do **not** relabel it as an empirically learned observed outcome.

The diagnostic export failure is retained as a failure, not reported as a successful training experiment. A synthetic fixture exercises the source field and malicious lookalikes. Passing tests does not establish acceptance of every field in the historical 4,606-feature catalog; that requires the user's fresh real export on Mac and report inspection.
