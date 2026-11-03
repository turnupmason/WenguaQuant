# First actual Mac logistic lost-property cohort: diagnostic, not predictive validation

Date: 2026-11-03. User ran WenguaQuant on macOS CPU using previously exported WenguaApp Professional V2 facts, producing `local/runs/20261103T164538_321806Z/`. The raw local input and original report remain on the Mac, are not stored in this public repository and must never be overwritten. The values below are transcribed from the user's printed metrics and item-level scores; they are **not** a new run performed by this audit.

## Observed run receipt

- CLI: `python -m wengua_quant --export .../20261103T163918_700764Z_source_export --backend logistic --device CPU`.
- Terminal confirmed: `task=lost.recovered_at_last_report`, `events=11`, `backend=logistic`, `production_enabled=false`. The previous source export reported 194 primary case records, 19 additional casts and 4,606 professional fields, but these are **not** 194 supervised labels.
- Cohort: 12 item targets, 11 independent event families; 10 item labels `recovered` and 2 `not_recovered` (both book-reported, not independently verified). The `kaiwu.lost_property.case01` event has two item outcomes, one of each kind, and is held out as a unit.
- All five groups predicted **all 12 items as recovered** using the 0.5 threshold. Thus unweighted item accuracy is `10/12=83.33%`, while event-family-weighted accuracy is `(9 + 0.5)/11 = 86.36%` in every group. Code source: `wengua_quant/model.py:_weights`, `_scores`, `evaluate`.

| Feature group | Event-weighted Brier (lower better) | Event-weighted Log Loss (lower better) | Event-weighted accuracy |
|---|---:|---:|---:|
| prior | 0.1379 | **0.4699** | 86.4% |
| context | 0.1379 | 0.5155 | 86.4% |
| chart | **0.1358** | 0.5000 | 86.4% |
| professional_facts | 0.1364 | 0.6689 | 86.4% |
| professional | 0.1361 | 0.6735 | 86.4% |

Two observed negative targets were scored as follows (probability-like *uncalibrated research scores*, not trustworthy empirical recovery probabilities):

| Source book event | Observed item outcome | Prior | Chart | Professional facts | Professional |
|---|---|---:|---:|---:|---:|
| shenqi.lost_property.case08 | not_recovered | .8750 | .9473 | .9924 | .9928 |
| kaiwu.lost_property.case01 | not_recovered | .8333 | .8929 | .9789 | .9807 |

Since case08 carries event weight 1 and the negative target within Kaiwu case01 has weight 0.5, their approximate contribution to the overall professional log loss is 0.628 of 0.6735, or **about 93%**, calculated using the user's scores rounded to four decimals. The professional feature group therefore **magnified errors on both observed negatives**; it has not shown reliable added predictive value. The prior group gave the lowest log loss, although chart had a marginally better Brier (0.0021 absolute difference from prior). Neither is evidence of robust improvement given eleven events, only two negative-labeled event families, publication selection bias and heterogeneous last-reported follow-up duration.

## Required next research gate

1. Preserve the original run, hashes, five ablations and held-out item predictions. Do not use these revealed labels to tune parameters and later call the same events secret/blind tests.
2. Diagnose post hoc with `python -m wengua_quant.audit local/runs/20261103T164538_321806Z`; the command reads only existing evaluation/held-out artifacts, recomputes group-weighted metrics for consistency and shows dominant error cases. It trains nothing and writes nothing.
3. Record actual question-time item context, recoveries/nonrecoveries, common event grouping and follow-up window constraints. The Kaiwu event's two outcomes from the same board test whether question-time item identity truly gives different admissible inputs; source missingness must remain unknown, not filled from feedback.
4. Draft domain-target-specific feature subsets from **prior text and rule provenance** before looking at new, independently collected outcomes. Do not search over feature subsets against the eleven already revealed folds and present the best as validated.
5. Continue label audit and independent evidence collection; prevalence-only and context-only baselines remain mandatory. A model that predicts all examples as `recovered` has no observed minority-class detection at threshold 0.5, regardless of how high accuracy looks.

**Scope:** This is model-output diagnosis, not proof that divination does or does not predict actual events. It does not touch WenguaApp/V9/V11, original source, source hashes, trained feature encoders, or label definitions.
