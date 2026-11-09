> **2026-11-09 · 当前交付V6.0：来源边界、多领域复盘与同父权重三学习率。** 两书7970材料/194案/1008分析/2049规则/46流程保留；8951登记、5947训练核心、5981监督视图。346遮结论题退出应用训练；852完整原段作者复盘、149训练案例复盘视图，阶段不再笼统称初断；20同来源条件决策对话，120回合合同核对。Excel完整194案JSON及1008作者原文逐字读回，SQLite增加可查询cases/author_analysis；22案例领域分别列覆盖与缺口。默认V5.8 E1 step97保留基线；24题官方/父权重控制，三率1e-6/3e-6/1e-5各768核心/786视图/50更新，通过112题+36RAG逐项无退步且新增正确项才完整374更新；完整阶段同时保护父与已过短训，最后192来源审阅一次。新阶段新优化器、同阶段完整恢复；80%额度/微批缓存/有界静默诊断保留。291单测、真实微型CPU继承/精确恢复与三率正负路径、真实教材短训及完整准备通过（最长3189tokens）；原V59的180代码和App49文件不变。实际8B效果待本机，不自动专业晋升。入口`train_wengua_v60.cmd`，结果`export_wengua_v60.cmd`；见[课程与PowerShell](course_v60/README.md)。

> **2026-11-09 · V5.9实跑已审阅。** 150断点含288份step150优化器状态，接续至196两轮；新进程26个train样点无free0、23次缓存释放、峰值reserved12.783GiB，恢复及后续评估正常。两轮均未过逐题留存，完整346步未启动；原80题基线问答与V5.8 E1完全相同。637存储记录含112复用，实际525生成含1短测；七组评分和两次停止决定重算一致，636正式存储记录有SHA绑定限定观察。部分关系/身份改善，但仍真错与漏条件；检查仍有同义/绑定误判，局部补句仍有历史事件/数值前提不足，不能归为缺古籍或只需更多epoch。原父权重与V5.9 E1/E2均保留；审阅仅评估、不得入RAG/SFT，未启动新训练。见[逐轮、课程与恢复审阅](model_reviews/20261109_v59/README.md)。

> **2026-11-08 · V5.9第158步运行恢复补丁。** 用户记录张量峰值8.95GiB、reserved15.12GiB、free长期0且OOM计数0；原600秒监控主动中断，不能据此确证驱动/换页根因。180项原运行/教材SHA冻结，新增独立80%分配器额度、微批次边界清理闲置缓存、静默堆栈诊断和有限宽限；新执行回执独立保存，原训练/优化器合同不改。270单测通过，原V5.9检查点由新入口恢复后CPU最终张量与不中断一致；真实4080新策略待续训核对。入口 `resume_wengua_v59.cmd` 自动选最后完整封存checkpoint，本次目录列到150；不是重训。见[原合同续训](course_v59/RECOVERY.md)。

> **2026-11-08 · 当前交付V5.9：证据推断、条件切换与逐题留存。** 8039登记/5502实际SFT核心/5526完整监督视图每轮；346局部推断课覆盖19领域，20同规则条件对话，75有向关系教学，40实际RAG教师复检不重复。7970材料/194案/1008分析/2049规则/46流程完整保留，1008分析均有教学处置；911阅读归档和原test封存。原V5.8 80题与18RAG输入精确保留；每轮112题+36RAG，逐项保护父权重正确事实/完整条件/引用，须有新增完整项；混合196更新通过才完整346更新。继承V5.8 E1 step97，LR3e-6，新阶段新优化器，同阶段完整恢复。260单测、来源/分词/SQLite及Excel读回、标准解密安装和CPU精确恢复已核对；8B实际改善待用户实跑，不自动专业晋升。入口 `train_wengua_v59.cmd`；见[课程与PowerShell启动](course_v59/README.md)。以下旧版记录冻结保留。

> **2026-11-08 · V5.8实跑已审阅。** 两轮97/194更新和6个实际worker正常完成；留存未过，完整650更新未启动。487存储原答含80第二轮复用，实际407生成含1短测；七组冻结评分和两个停止决定重算一致，486正式存储记录附限定读答注释。给定宫位/身份主项有进步，第二轮更低loss但RAG18→16；整盘方向、人物条件、局部待补及案例依据仍有真错。仍有课程段落重叠及评分漏判/假通过，均另记诊断，不改冻结V5.8或旧权重。私有Excel与原答仅评估、不入RAG/SFT；未启动V5.9。见[逐轮及192题审阅](model_reviews/20261108_v58/README.md)。

> **2026-11-08 · 当前交付V5.8：所问目标与条件应用。** 7541登记课程，5173应用训练核心/5187监督视图每轮；911纯阅读课归档，原7970材料/194案/1008分析及test家族保留。40实际检索复检与旧对话精确合并，零完全重复；补能否/日期、领域范围、本人/代问、起局年干/出生年命及宫/宫内符号对照。新检查处理纠错开头矛盾、正常属性措辞、数字附加解释、规则来源等价，不改V5.7成绩。继承本机V5.7正式step65及已核祖先回执，混合2轮194更新，留存通过后应用2轮650更新；未通过也保存一次192领域审阅。旧权重保留，不自动专业晋升。入口 `train_wengua_v58.cmd`，见[课程和PowerShell启动](course_v58/README.md)。

> **2026-11-08 · V5.7实跑已逐答审核。** 65更新混合短训正常完成，4个实际worker成功；未过留存检查，完整698更新未启动。164正式对照答+1短测答均保留；基础关系逐句7/25→16/25，但给定宫位2/6→1/6，实际RAG主要所问17/18→16/18，另1题重复至768tokens。五行属性实际两边均5/5，原检查0/5→2/5属于表达漏判；纠错开头、门宫矛盾及来源等价也有评分问题。婚姻条件原文已给且3教师已训练，仍误答婚期；不能归为缺书。冻结原代码/教材/评分及全部权重；审阅包仅评估、不入RAG/SFT，未发布新训练器。见[逐题复核](model_reviews/20261108_v57/README.md)。

> **2026-11-07 · 当前交付V5.7：完整来源合同、条件对照与混合继承训练。** 7037核心（train5568/development1469），完整监督视图5582/轮；194案、1008分析、7970材料保留，原test家族封存，无人工评分或模型错答教师。具体摘录去重并查双方前提，28多轮逐助手回合核对；40实际RAG教师、18真实检索评估。默认继承V5.5 step397，1024混合核心短训65更新，留存通过后完整课程两轮698更新；新阶段新优化器、阶段内完整恢复。每轮64事实+18RAG，最后192新题与192旧题，不重复全套生成。自动事实与开放案例语义分开，不自动专业晋升。入口 `train_wengua_v57.cmd`；见[课程、数据边界及PowerShell启动](course_v57/README.md)。以下旧版记录冻结保留。

> **2026-11-07 · V5.6实跑已审核。** 两个已有权重各完成404更新/4轮，3318原答、22 worker与冻结合同/新旧评分均核对；未通过全部短训门槛，未启动完整案例课程。新12整盘关系题各轮通过，但旧格式仍有真错和表达漏判。发现新256宫位课中224条可见来源被同ID不同摘录去重缩窄；古籍原文未改。完整原答/逐题审阅加密保存，不作RAG/SFT；冻结V5.6及所有权重保留，不热改、不自动晋升，下一轮先修教材与检查并分层评估。见[实跑审核](model_reviews/20261107_v56/README.md)。

> **2026-11-07 · 当前交付V5.6：前提修订与已有权重条件短训。** 6566核心（train5186/development1380）；258作者题补回完整原阶段，256完整句阅读+2仅有提示的证据阅读，23限定局部判断；去重原段并加方向/宫位/条件及64实际RAG教师。194案/1008分析、7970材料/2049规则保留，原test封存，无旧人工分数。V5.4 step134与V5.5 step397各短训404更新，事实/条件/旧题/RAG通过才接两学习率各650更新；继承权重、新阶段新优化器，同阶段完整checkpoint恢复。153软件单测、真实来源/完整分词/SQLite及Excel逐条读回、CPU继承和张量恢复已核对；云端未做8B CUDA或专业质量验收。入口 `train_wengua_v56.cmd`；见[课程、验收与PowerShell启动](course_v56/README.md)。以下旧版本记录保留。

> **2026-11-07 · V5.5实跑已审核。** 两组各794更新/2轮完成，3206原答与冻结合同核对，四个检查点均未过原晋升门槛；定位稳定，宫位生克与条件仍弱。发现作者证据题遗漏前提候选13条、9组规则答案重复段落；完整原答/逐轮审阅加密保存，不进入RAG/SFT。高学习率E1 step397仅作下一阶段研究候选，原V5.4父权重和所有V5.5检查点保留，未启动V5.6。见[审核报告](model_reviews/20261107_v55/README.md)。

> **当前交付：V5.5分步推导与领域规则课程。** 5984核心（train4738/development1246）；补回1118定位、952给定宫位推导，另加112条验证；148规则片段、46来源规则组、258作者证据推结论，17阶段不清的候选仅作复盘。原194案/1008分析、7970材料和2049规则逐项存库对照。默认继承已审核V5.4高学习率step134；两学习率各2轮，每组794更新，完整checkpoint恢复；新192题+旧128题及18RAG回归。未在云端做真实8B训练，不自动专业晋升。入口 `train_wengua_v55.cmd`，见[课程与启动](course_v55/README.md)。

> **V5.4结果已审核，V5.5尚未交付。** 两学习率各268步/2轮完成；1670原答与合同、旧评分核对一致。18道整盘题定位分项均18/18，但关系命名仍弱；高学习率第2轮loss更低，旧方向回归却从19/26降至12/26。仅推荐第1轮step134作下一阶段研究候选，不改自动保留V5.3父权重的结果。逐题Excel加密保存，旧训练代码、权重和成绩不改。见[审核与覆盖清单](model_reviews/20261106_v54/README.md)。

> **当前工作：V5.4修复两书课程与实际RAG，暂停新增古籍。** V5.3上传808条原答已审核：48条旧多轮记录错位（基础train16条）；真实RAG18题主要事实13正确/1部分/4错误，17缺引用；旧分数不改。新2673课程（train2135/development538）、1038实际RAG、16完整多轮，保留两书7808材料、194案、1008分析。默认继承V5.3 step256、新优化器，两学习率各两轮，逐轮新128题+旧128题+原18RAG回归；原933test继续封存，不自动专业晋升。入口 `train_wengua_v54.cmd`；见[说明](course_v54/README.md)。

> **当前工作：暂停新增古籍，先跑通两书 V5.3 来源应用与实际 RAG。** 入口 `train_wengua_v53.cmd`；5593课程（基础1236、应用4357），保留原7614材料、194案、1008分析，补194个有原文坐标的阶段材料。官方8B新LoRA与完整V5.2权重同课程短训比较，基础检查不过即停止长训；通过后两学习率各两轮，并查基础回退与实际检索。旧人工吉凶分数和模型错答不作教材，原933 test继续封存。独立子进程、完整checkpoint恢复，正常中文回答，原App起盘不变。见[课程、检索与PowerShell启动说明](course_v53/README.md)。

> **上一版归档：两书V5.2修订2课程与本机模型。** 入口`train_wengua_v52.cmd`，全部新代码/数据在WenguaQuant，不改App起盘或旧V4/V5.1合同。两书7614材料、194案、1008分析保留；4131核心（train3721/development400/diagnostic10）。清理标题与段落抄写，A/B比较不串案，逐步绑定多次起盘；202组来源条件设计。新阶段128题覆盖19案例领域/28家族，106明确事实检查+22人工审阅，不称祖先未见或现实预测盲测。原933test封存。V4/E1/E2先比较，再两学习率各2轮；新阶段新优化器，组内完整恢复；不无限loop、不自动专业晋升。见[课程与启动说明](course_v52/README.md)。

> **2026-11-05 用户指定的数据主仓库：WenguaQuant。** 新古籍、分类、来源坐标、核心课程对照、RAG、私有数据库与审计集中在本仓库`wengua_corpus/`、`data/corpus/`及版本化`data/knowledge/`。WenguaApp保留起盘/应用/V5.1复现所需固定依赖，不删除其原代码和运行合同。原两书RAG、第三书总序、V5.1课程密文快照已逐字校验迁存，App旧副本作兼容冻结，不继续堆新资料。旧统计`wengua_quant/`和含人工规则的`wengua_master_v3/`为隔离历史轨道，不能自动混入当前纯古籍课程。第三书第二批574材料、509来源课程候选、54结构、11疑点全部有处置；对应81干对/108星时/64动应/62静应；静应缺伤生、死杜，禁止补造。原194案不变，外应口诀不算真实案例，卷一至四节选不称全书。课程候选未自动投入训练，V5.1不变。见docs/CORPUS_HOME_20261105.md。

# WenguaQuant Project Status · 2026-11-03

- **Repository**: `turnupmason/WenguaQuant`, branch `main`; independent research. Initialized in this delivery, latest commit/CI to be checked after upload.
- **Linked read-only research source**: `turnupmason/WenguaApp` professional V2 export. Source archive SHA `21362e6f0266c03dcdc4a7ef35f45954ba274ae6458de0ba341132abca7d0405`. The historic source database is not included here and must not be rewritten.
- **Available research source facts**: 194 case events; 4606 source-indexed professional fields (including fixed scoring features in separate ablation), not 4606 independent samples. Source V2's pilot export has 12 targets from 11 events. The only currently approved training target is book-reported lost-item status at last feedback. Health severity/site/recovery, item direction/time need audited labels and properly defined follow-up.
- **Implemented**: pinned V2 export reader, strict feature white-list, group-held-out CV, five prior/context/chart/professional/fixed-rule ablations, logistic and optional CatBoost CPU/GPU, source-input SHA receipt, event-weighted metrics, no-overwrite output, synthetic contract/unit tests, cross-OS CI.
- **Not implemented / not claimed**: genuine medical outcome training, clinically meaningful probabilities, prospective/blind validation, time-to-event training with censoring, verified GPU execution on user's hardware, signed iOS deployment.
- **Baseline WenguaApp unchanged**: V9 severity scoring and V11 prose, V27 discomfort, V4 recovery timeline remain separate traditional rule engines. Never train by copying their scores as true outcomes.
- **Source engineering receipt**: WenguaApp main `72c6c08d201c5f6a7eac7d3f4005692250393c36`; CI #41578974964 success at time of initial inspection. This is NOT WenguaQuant's own CI.
- **Next gate**: run a real export with the local WenguaApp historical SQLite archive and train the first full independent pilot on that export; review all results against class-prior baseline. The local archive is not present in this ChatGPT container.

## 2026-11-03 Mac V2 catalog validation repair (WenguaQuant v0.1 patch)

- The first real Mac run **successfully exported** 194 primary events, 19 additional source casts and 4,606 professional feature fields from WenguaApp. The source export then failed at the WenguaQuant catalog validator: `ValueError: Unsafe, missing-source, or duplicated feature catalog entry`. **No real-case model fit or predictive metrics were produced by that attempt.**
- Cause in `wengua_quant/dataset.py`: overly broad `FORBIDDEN` token `result` rejected the legitimate source-derived `heuristic.role_edge.<source>.<target>.control_matrix_result`, which is a pre-question fixed user strength/control matrix output, not a later event outcome.
- Patched only the exact source-style, group-bound suffix for acceptance and kept it in the fixed-heuristic `professional` ablation; any outcome/feedback/label terms in role identities or other field names remain forbidden. Error reports now include the offending catalog feature ID for actionable diagnostics.
- Synthetic unit suite: **9 tests passed locally**, including matrix-feature acceptance, ablation isolation and negative leakage cases. Actual post-fix Mac export/training and new-commit CI must be reported separately once run. Failure record: `docs/audits/MAC_V2_CATALOG_REJECTION_20261103.md`. WenguaApp application, frozen rules and historical archive unchanged.


## 2026-11-03 First Mac logistic research-result audit (post hoc, no retuning)

- **Real Mac run user-reported complete**: source export `20261103T163918_700764Z_source_export` and research output `local/runs/20261103T164538_321806Z`; model Logistic CPU; 11 independent lost-property event families, 12 item targets. WenguaQuant v0.1 synthetic unit test suite passed 9/9 on user Mac. Prior commit `7f578ab3ed58e6d506e04d450558a6fa6e06513e` GitHub Actions #41677376906 completed **success**.
- **User-reported held-out scores (five precommitted groups)**: prior log loss 0.4699; context 0.5155; chart 0.5000; professional_facts 0.6689; professional 0.6735. All groups predicted all 12 item targets `recovered`, with event-weighted accuracy 86.4% and no observed minority-class detection. 10 item outcomes were reported recovered and two not recovered, including one negative in a two-item Kaiwu event family. Unweighted accuracy is 83.3%; the published score is event-family weighted, not a scoring bug.
- **Finding**: supervised pipeline runs, but this sample **does not show reliable incremental predictive value of professional features**. Both not-recovered items received greater-than-97% research scores from the professional model (uncalibrated, not user probabilities). Fixed-rule group is not shown superior. This is an **already-inspected experimental cohort** and must not be called a new blind test after tuning.
- **New read-only research diagnostic**: `python -m wengua_quant.audit local/runs/20261103T164538_321806Z` validates saved group weights and reports the largest case-level log-loss contributions. The diagnostic is post hoc, does not retrain, does not select hyperparameters and leaves prior run files intact. Permanent record: `docs/audits/LOST_PROPERTY_MAC_PILOT_DIAGNOSTIC_20261103.md`. Local synthetic unit suite **12 tests passed**; new GitHub CI and actual Mac diagnostic invocation require separate verification; no direct access to user's local original report is claimed.


### 2026-11-04 · Wengua Master V3 first data migration (separate Qwen curriculum)

New isolated `wengua_master_v3` data prep, plus 1,024 public hypothetical directional palace exercises and source-safe workbook inventory in `data/master_v3`; 12 user-supplied source health case IDs, 55 author paragraphs, 20 knowledge QAs. Local private source-study candidates **87 = 74 train + 13 test**, NOT published as original texts. Original flags all `未审`, manual expert columns blank, user authorizes using Codex compilation as source research. No fresh full nine-palace expert prediction gold; old V2 Qwen 13,341 tasks and old outcome statistics unchanged. SHA receipts and withheld private source boundaries in [V3 audit](master_v3/V3_DATA_AUDIT_20261104.md). New real GitHub CI to be checked against target commit; local software unit tests on the staged toolkit passed, no new 14B/32B GPU fit or Xcode build performed. Private 194-case SQLite absent from this runtime; V2 author-export adapter is opt-in read-only on the user's machine.


本批最终验收：23项软件回归通过；完整恢复后，11913核心记录与7614材料逐条存库读回相等，194案、1008分析均保留，所有分析source_id有效。本批574材料、509候选课程、54结构、11疑点全部处置；标题不作内容问答。源码1b224be的Windows/Linux×Python3.11/3.12四工作均success；WenguaApp数据归属指针b41d335e的两个Python工作success，完整日志在docs/corpus_audits/20261105_batch02/。原起盘和V5.1的49项运行文件哈希不变。候选不是已完成训练或独立预测验证。

## 2026-11-05 · V5.1用户上传结果复查

用户已完成V5.1的1284步（2 epochs），继承V4 step486，阶段内优化器连续。235条实答与固定课程题目/目标逐项相符；逐题比较66任务，发现规则命名、身份切换、数字/生克及反馈阶段问题。第二轮loss更低不等于专业更优。正式37题只覆盖一个事件家族，29诊断已见，最终test未上传；未批准专业晋升。仍存20道train标题题是课程清理遗漏，下一版本应修；冻结包与权重没有改。独立审阅材料加密保存在本仓库`data/delivery/v51_review_20261105/`，恢复至`local/reviews/v51_20261105/`，不加入古籍RAG。详见[逐轮审阅](model_reviews/20261105_v51/README.md)。
