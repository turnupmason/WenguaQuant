# WenguaQuant project instructions

`WenguaQuant` is a separate, independent **data-driven** research repository. Do not silently replace or modify production `WenguaApp`, `QimenLLM`, or their historical archives. Primary branch `main`.

At task start review this file, `docs/PROJECT_STATUS.md`, `docs/RESEARCH_PROTOCOL.md`, relevant source, latest commit and Actions. Preserve old versions, original SHA receipts and failure records. Changes to data models should be append-only or versioned.

Separate three things: (1) literal chart and question-time facts; (2) authored tradition/interpretive rules and fixed V9 scores; (3) genuinely observed, provenance-tagged outcomes. Outcome or author success status must never leak into an input. 1080 possible chart configurations are **not** 1080 outcome-labelled independent historical events; event-family holdout is mandatory. No fabricated birth year, missing palace, healthy diagnosis, or outcome labels.

Research models must learn parameters from outcomes and compare with event-weighted prior and same-background/no-chart baselines. Fixed user tables may be ablated as extra predictors but never used as ground truth. Maintain exact feature/version/SHA receipts, follow-up definitions, test separation, calibration status and raw failures. Never call a model's raw score a validated real-world probability. Medical output is non-clinical research, not diagnosis or care advice.

Run `python -m unittest discover -s tests -v` before pushing; verify actual newly-triggered GitHub Actions. Do not report local synthetic CI as real-case training or verified model performance. For the first pilot, WenguaApp has the read-only source database, and `run_professional_training.py export` produces the canonical V2 export; this repository consumes that export without copying book text or database bytes into this public repo.

2026-11-03 source export compatibility: WenguaApp V2 emits `heuristic.role_edge.<source_role>.<target_role>.control_matrix_result` (a fixed manual control-matrix fact, **not** observed outcomes). The receiver may permit that exact, properly grouped source field but must still reject outcome/feedback/label fields; keep this field exclusively in the `professional` fixed-heuristic ablation, not `professional_facts`. This was the cause of the first actual Mac V2 export rejection; see `docs/audits/MAC_V2_CATALOG_REJECTION_20261103.md`. Regressions must test real source-style field names, not only simplified fixtures.
