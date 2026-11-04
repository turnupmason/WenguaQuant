# Wengua Master V3 · source-safe first data migration · 2026-11-04

## Verified sources
- New destination: [turnupmason/WenguaQuant](https://github.com/turnupmason/WenguaQuant) main. Keep old `wengua_quant/` observed-outcome studies and failures intact; V3 goes into independent `wengua_master_v3/`.
- Origin inspected: WenguaApp main `86747716d1228363c85f4891c293416d12678e8f`; latest historic source archive recorded SHA256 `21362e6f0266c03dcdc4a7ef35f45954ba274ae6458de0ba341132abca7d0405`.
- User-uploaded `Wengua_Manual_Review_Mac.xlsx` SHA256 `5697f5ebbba3904929b15f843dbe66023067f9f26a1d900d343930801881a3ea`.
- Workbook: 12 historical health case records, 55 original author paragraphs, 20 sourced knowledge QA, no new user-recorded case facts. Original workbook review statuses are **未审** on 12+55+20 rows, with no expert explanation columns populated. User authorizes use of Codex collection; this is **separate from documented per-record expert semantic audit**.

## Concrete outputs
- **Public** `data/master_v3/public_rule_drills.jsonl`: 1,024 hypothetical palace-five-element direction exercises (eight peripheral palaces × eight × eight independent role pairs × two question formats); 36 unordered physical-pair holdout groups, 848 train / 32 validation / 144 test. These are not 1,024 real nine-palace casts and do not have outcomes, clinical labels, score matrices or confirmed source strengths.
- **Public** `data/master_v3/review_manifest.json`: book-safe indexed case IDs and source/approval boundaries. No original full chapter text or private source workbook was pushed.
- **Private local only**: 87 source-study SFT candidates (55 atomic author source stages, 12 complete author-step study sequences, 20 sourced QA), 74 train / 13 held-out TEST. Both TEST case families remain wholly excluded from train. These study tasks may contain author hindsight and are NOT independently validated prospective prediction gold.
- **Builder** `wengua_master_v3`: can read the private seed and, when user has the WenguaApp V2 private export, import additional source-authored steps from `case_registry.jsonl` and `source_units.jsonl` without copying raw book text into this repository. Source family evidence components use conservative test>validation>train grouping. Rejects duplicate sample IDs, group-crossing, unsupported predictive gold, and output directory overwrite.
- V2 Qwen 14B `13,341`-dialogue course is **not changed**. User's local GPU training logs not available in this session; no assertion it finished. Existing WenguaApp severity V9, text V11, site V27 and Swift UI untouched.

## Gaps and next contract
The 194-event/1,008-author-paragraph source SQLite is not attached to this session, so full 194-case V3 re-export has **not** occurred here; the optional read-only V2 import makes this possible on user's local checkout without manual transcription. The available workbook supplies only twelve health cases. No verified canonical full nine-palace facts + pre-outcome expert conclusion corpus was generated. **Fresh complete-cast gold = 0**, even though historic author-source study is useful. Need run actual WenguaApp casting/energy evidence contracts and independently review whole-cast answers before marking those examples as full-board SFT gold. All V3 generated candidates preserve `fresh_prediction_gold=false`.

## Reproduction
```bash
python -m unittest discover -s tests -v
python -m wengua_master_v3 --private-seed /absolute/path/WenguaMaster_V3_private_review_seed.json --output local/v3_run_1
# Optional, only if actual private WenguaApp V2 data already exists:
python -m wengua_master_v3 --private-seed /absolute/path/WenguaMaster_V3_private_review_seed.json --v2-export /absolute/path/WenguaApp/local/<real_prepared_v2_data_folder> --output local/v3_run_2
```
Data preparation is stdlib-only, outputs are uniquely named and never overwritten. Never upload private seed or generated `private_source_study_*.jsonl` to this public GitHub. No actual new GPU fine-tune, App Xcode build, signed deployment, or measured prediction accuracy is claimed. GitHub CI checks software contracts rather than professional forecast validity.
