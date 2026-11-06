> **当前工作：暂停新增古籍，集中两书V5.2修订2课程与本机模型。** 入口`train_wengua_v52.cmd`，全部新代码/数据在WenguaQuant，不改App起盘或旧V4/V5.1合同。两书7614材料、194案、1008分析保留；4131核心（train3721/development400/diagnostic10）。清理标题与段落抄写，A/B比较不串案，逐步绑定多次起盘；202组来源条件设计。新阶段128题覆盖19案例领域/28家族，106明确事实检查+22人工审阅，不称祖先未见或现实预测盲测。原933test封存。V4/E1/E2先比较，再两学习率各2轮；新阶段新优化器，组内完整恢复；不无限loop、不自动专业晋升。见[课程与启动说明](docs/course_v52/README.md)。

> **2026-11-05 用户指定的数据主仓库：WenguaQuant。** 新古籍、分类、来源坐标、核心课程对照、RAG、私有数据库与审计集中在本仓库`wengua_corpus/`、`data/corpus/`及版本化`data/knowledge/`。WenguaApp保留起盘/应用/V5.1复现所需固定依赖，不删除其原代码和运行合同。原两书RAG、第三书总序、V5.1课程密文快照已逐字校验迁存，App旧副本作兼容冻结，不继续堆新资料。旧统计`wengua_quant/`和含人工规则的`wengua_master_v3/`为隔离历史轨道，不能自动混入当前纯古籍课程。第三书第二批574材料、509来源课程候选、54结构、11疑点全部有处置；对应81干对/108星时/64动应/62静应；静应缺伤生、死杜，禁止补造。原194案不变，外应口诀不算真实案例，卷一至四节选不称全书。课程候选未自动投入训练，V5.1不变。见docs/CORPUS_HOME_20261105.md。

# WenguaQuant project instructions

`WenguaQuant` is a separate, independent **data-driven** research repository. Do not silently replace or modify production `WenguaApp`, `QimenLLM`, or their historical archives. Primary branch `main`.

At task start review this file, `docs/PROJECT_STATUS.md`, `docs/RESEARCH_PROTOCOL.md`, relevant source, latest commit and Actions. Preserve old versions, original SHA receipts and failure records. Changes to data models should be append-only or versioned.

Separate three things: (1) literal chart and question-time facts; (2) authored tradition/interpretive rules and fixed V9 scores; (3) genuinely observed, provenance-tagged outcomes. Outcome or author success status must never leak into an input. 1080 possible chart configurations are **not** 1080 outcome-labelled independent historical events; event-family holdout is mandatory. No fabricated birth year, missing palace, healthy diagnosis, or outcome labels.

Research models must learn parameters from outcomes and compare with event-weighted prior and same-background/no-chart baselines. Fixed user tables may be ablated as extra predictors but never used as ground truth. Maintain exact feature/version/SHA receipts, follow-up definitions, test separation, calibration status and raw failures. Never call a model's raw score a validated real-world probability. Medical output is non-clinical research, not diagnosis or care advice.

Run `python -m unittest discover -s tests -v` before pushing; verify actual newly-triggered GitHub Actions. Do not report local synthetic CI as real-case training or verified model performance. For the first pilot, WenguaApp has the read-only source database, and `run_professional_training.py export` produces the canonical V2 export; this repository consumes that export without copying book text or database bytes into this public repo.

2026-11-03 source export compatibility: WenguaApp V2 emits `heuristic.role_edge.<source_role>.<target_role>.control_matrix_result` (a fixed manual control-matrix fact, **not** observed outcomes). The receiver may permit that exact, properly grouped source field but must still reject outcome/feedback/label fields; keep this field exclusively in the `professional` fixed-heuristic ablation, not `professional_facts`. This was the cause of the first actual Mac V2 export rejection; see `docs/audits/MAC_V2_CATALOG_REJECTION_20261103.md`. Regressions must test real source-style field names, not only simplified fixtures.


2026-11-03 first revealed real logistic pilot: all five groups classified all 12 lost-item targets as `recovered`; the 86.4% metric is **event-family weighted** (one event includes both a recovered and an unrecovered item), not a sign of minority-class discrimination. Preserve its 11-event held-out scores and original export; no retrospective feature/hyperparameter selection against these known outcomes may be advertised as new blind validation. Audit document `docs/audits/LOST_PROPERTY_MAC_PILOT_DIAGNOSTIC_20261103.md`; `wengua_quant.audit` is read-only, post-run and not another training pass.


## Wengua Master V3 independent generative-curriculum track · 2026-11-04

V3 Qwen instructional source-study is **NOT** the `wengua_quant/` real-outcome training target. Place V3 code in `wengua_master_v3/`, public synthetic rule drills in `data/master_v3/`, append-only audit in `docs/master_v3/`. Read the original WenguaApp private archive or V2 export only from user-local paths, NEVER upload whole copyrighted historical original text, personal data, or complete private SQLite to public WenguaQuant. The Codex workbook was user-authorized for reuse, but legacy row states remain `未审`, not independently certified professional gold. Keep source study, source engine validation, and real-outcome observation strictly separated. No full-board fresh prediction SFT samples may be claimed without complete verified nine-palace facts and an expert original pre-outcome target. Keep same event family / evidence component across splits and preserve V2 original train/validation/test precedence; false positives or unknown birth year never imputed. Never change the running Qwen14B V2 adapter or frozen App score tables during V3 preparation. Run all tests and check exact GitHub Actions post-push.


本批最终验收：23项软件回归通过；完整恢复后，11913核心记录与7614材料逐条存库读回相等，194案、1008分析均保留，所有分析source_id有效。本批574材料、509候选课程、54结构、11疑点全部处置；标题不作内容问答。源码1b224be的Windows/Linux×Python3.11/3.12四工作均success；WenguaApp数据归属指针b41d335e的两个Python工作success，完整日志在docs/corpus_audits/20261105_batch02/。原起盘和V5.1的49项运行文件哈希不变。候选不是已完成训练或独立预测验证。

2026-11-05 V5.1运行结果已上传，1284步完成；235条实答/66任务的语义审阅与来源校验见docs/model_reviews/20261105_v51/README.md。37正式验证同一事件家族，29诊断已见，最终test仍封存；自动选择E2只是语言loss选择，未批准专业最佳。train仍20标题题、来源跨案/切段及条件应用问题待新版修复，不能迁就错答改原书。审阅包catalog.library=null，仅评估，禁止把模型错误加入古籍RAG/SFT。V4/E1/E2与冻结V5.1均保留，未修改训练器或启动新训练。
