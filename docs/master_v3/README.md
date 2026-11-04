# Wengua Master V3 · language-model training data track

Independent from `wengua_quant/` real-outcome statistics. This is a **traditional-rule instructional SFT dataset builder**, not statistical calibration or a new production App engine.

Public assets:
- `data/master_v3/public_rule_drills.jsonl` — 1,024 deterministic hypothetical palace-direction exercises; **not** real dated casts.
- `data/master_v3/review_manifest.json` — book-safe workbook source audit, 12 case IDs, 55 original author steps and 20 QA counts.
- `wengua_master_v3/` — stdlib-only generation, family holdout, private WenguaApp V2 export adapter, no silent overwrite.

Private data: original-book passages in user workbook stay in local `WenguaMaster_V3_private_review_seed.json` and `private_source_study_*.jsonl`, NOT in public GitHub. 87 candidate study tasks exist from the uploaded Excel; 74 train / 13 test. None are verified independent full-cast prognostic gold.

```bash
python -m unittest discover -s tests -v
python -m wengua_master_v3 --private-seed /your/private/WenguaMaster_V3_private_review_seed.json --output local/master_v3_run_1
```
Optionally add `--v2-export /your/WenguaApp/local/prepared_qwen_v2_data` to read additional (up to 194-case) original-author source steps from the existing **private** WenguaApp V2 export. The two repositories stay independent; V2 Qwen14B training/weights are unchanged.

See [V3 audit](V3_DATA_AUDIT_20261104.md).
