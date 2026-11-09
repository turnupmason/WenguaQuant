# 当前模型课程：V6.0

**2026-11-09实跑已审阅：三组各50步正常完成，均未通过逐题留存，完整374步未启动；仍保留V5.8 step97。** 原答832条正式生成逐项附限定观察，11套原检查与三次决定重放一致。课程仍偏复述，人物/关系与条件应用有真错，亦有评分误判；不延长旧混合教材、不自动升级。见[V6.0逐组、逐题与课程审核](docs/model_reviews/20261109_v60/README.md)。下面入口保留用于原实验复现，尚未发布下一版训练。

继续使用已有 Qwen3-8B 权重，先比较基座和保留基线，再跑三档学习率短训；逐题有新增正确项且无已知退步才继续完整课程。两书 194 案例和 1008 条作者分析完整保留，旧权重与 App 起盘代码保留。

在 PowerShell 执行：

```powershell
Set-Location "$env:USERPROFILE\WenguaQuant"
git pull --ff-only origin main
.\train_wengua_v60.cmd
```

[课程、全部案例审阅表与恢复说明](docs/course_v60/README.md) · [V5.9实际结果审阅](docs/model_reviews/20261109_v59/README.md)。完成后执行 `.\export_wengua_v60.cmd` 导出结果。

2026-11-09资料更新：[《金函》第五批卷八末、卷九卷十](docs/corpus_audits/20261109_jinhan_batch05.md)，263材料、65专题流程、86课程候选、原译残文对照已入SQLite、Excel与独立RAG；[第四批](docs/corpus_audits/20261109_jinhan_batch04.md)及[第三批](docs/corpus_audits/20261109_jinhan_batch03.md)保留。新资料不改V6.0训练，重名篇目可按章节筛选。

下面为历史版本记录与资料仓库说明，保留用于复现。

---

> **当前交付：V5.5分步推导与领域规则课程。** 5984核心（train4738/development1246）；补回1118定位、952给定宫位推导，另加112条验证；148规则片段、46来源规则组、258作者证据推结论，17阶段不清的候选仅作复盘。原194案/1008分析、7970材料和2049规则逐项存库对照。默认继承已审核V5.4高学习率step134；两学习率各2轮，每组794更新，完整checkpoint恢复；新192题+旧128题及18RAG回归。未在云端做真实8B训练，不自动专业晋升。入口 `train_wengua_v55.cmd`，见[课程与启动](docs/course_v55/README.md)。

> **V5.4结果已审核，V5.5尚未交付。** 两学习率各268步/2轮完成；1670原答与合同、旧评分核对一致。18道整盘题定位分项均18/18，但关系命名仍弱；高学习率第2轮loss更低，旧方向回归却从19/26降至12/26。仅推荐第1轮step134作下一阶段研究候选，不改自动保留V5.3父权重的结果。逐题Excel加密保存，旧训练代码、权重和成绩不改。见[审核与覆盖清单](docs/model_reviews/20261106_v54/README.md)。

> **当前工作：V5.4修复两书课程与实际RAG，暂停新增古籍。** V5.3上传808条原答已审核：48条旧多轮记录错位（基础train16条）；真实RAG18题主要事实13正确/1部分/4错误，17缺引用；旧分数不改。新2673课程（train2135/development538）、1038实际RAG、16完整多轮，保留两书7808材料、194案、1008分析。默认继承V5.3 step256、新优化器，两学习率各两轮，逐轮新128题+旧128题+原18RAG回归；原933test继续封存，不自动专业晋升。入口 `train_wengua_v54.cmd`；见[说明](docs/course_v54/README.md)。

> **当前工作：暂停新增古籍，先跑通两书 V5.3 来源应用与实际 RAG。** 入口 `train_wengua_v53.cmd`；5593课程（基础1236、应用4357），保留原7614材料、194案、1008分析，补194个有原文坐标的阶段材料。官方8B新LoRA与完整V5.2权重同课程短训比较，基础检查不过即停止长训；通过后两学习率各两轮，并查基础回退与实际检索。旧人工吉凶分数和模型错答不作教材，原933 test继续封存。独立子进程、完整checkpoint恢复，正常中文回答，原App起盘不变。见[课程、检索与PowerShell启动说明](docs/course_v53/README.md)。

> **上一版归档：两书V5.2修订2课程与本机模型。** 入口`train_wengua_v52.cmd`，全部新代码/数据在WenguaQuant，不改App起盘或旧V4/V5.1合同。两书7614材料、194案、1008分析保留；4131核心（train3721/development400/diagnostic10）。清理标题与段落抄写，A/B比较不串案，逐步绑定多次起盘；202组来源条件设计。新阶段128题覆盖19案例领域/28家族，106明确事实检查+22人工审阅，不称祖先未见或现实预测盲测。原933test封存。V4/E1/E2先比较，再两学习率各2轮；新阶段新优化器，组内完整恢复；不无限loop、不自动专业晋升。见[课程与启动说明](docs/course_v52/README.md)。

# 问卦资料主仓库 · WenguaQuant

古籍原文、分类索引、核心课程对照、RAG和私有数据库统一由这里管理。起盘与应用代码继续留在WenguaApp，历史V5.1与旧运行保留，当前训练入口见上方V5.5说明。

已归档资料迁移：[金函卷一至四与资料迁移说明](docs/CORPUS_HOME_20261105.md)。原文和Excel以此前教材口令恢复，公开仅存密文、抽象索引与校验记录。新增古籍当前暂停，第三书不进入本轮两书训练。

```powershell
Set-Location "$env:USERPROFILE"
if (!(Test-Path WenguaQuant/.git)) { git clone https://github.com/turnupmason/WenguaQuant.git }
Set-Location WenguaQuant
git pull --ff-only origin main
& ..\WenguaApp\local\qwen_env\Scripts\python.exe -X utf8 -m wengua_corpus restore --include-course-v51
```

恢复后Excel在`local/corpus/jinhan_batch02_review3/`，累计SQLite在`local/databases/three_books_batch02_review3_with_v51.sqlite3`。CPU检索不加载模型。课程候选需要后续语义及答题风格验收，不因入库而自动称训练就绪。

---

以下为隔离保留的历史统计/旧V3轨道，不是当前纯古籍数据入口。

# WenguaQuant · 问卦数据驱动量化研究

独立于 [WenguaApp](https://github.com/turnupmason/WenguaApp) 的量化研究仓库。目标是**从具有来源证据的真实结果中学习统计权重**，而不是复制 WenguaApp 传统 V9 加减分算法。只在预先定义的研究任务上训练；研究输出不接入正式 iOS App。

## 目前做到了什么

- **输入**：WenguaApp V2 经过来源核对的九宫/用神/宫间关系等特征导出（4606字段，包含明确隔离的传统固定分组）。
- **监督标签**：现阶段仅允许已有观察收据的 `lost.recovered_at_last_report` 失物书内最后反馈状态。12件物品、11个独立事件；不能当作可靠真实预测能力证明。疾病轻重、部位、康复应期缺合格标签，不会拿V9分数代填。
- **研究模型**：带正则项的逻辑回归；可选 CatBoost CPU/GPU。对照无盘先验、仅背景、原盘、专业事实及加传统固定特征五组。
- **验证**：按事件家族留一，避免同一故事不同物品/起盘跨训练测试；输出 Brier、Log loss、每折预测及输入文件SHA。没有独立概率校准，不对用户输出医疗或其他现实成功率。

## Windows + PyCharm / RTX 4090

保持两个仓库**并列**：`WenguaApp/` 保存原始研究库及专业特征代码，`WenguaQuant/` 负责独立统计训练。

1. 使用 Python 3.11 或 3.12，在 PyCharm 中打开 `WenguaQuant`，创建 `.venv`，安装 `pip install -r requirements.txt`。
2. 确保本机 `WenguaApp` 是对应的专业 V2 版本且其原始 194 案 SQLite 文件仍在 `archives/sqlite/`；不需要上传到新仓库。
3. 在 WenguaQuant 的 Terminal 执行：

```powershell
python -m wengua_quant --app-root ..\WenguaApp --backend logistic --device CPU
# 可选 GPU；如果没有 GPU 将明确失败，不会悄悄使用CPU
python -m wengua_quant --app-root ..\WenguaApp --backend catboost --device GPU
```

已有 WenguaApp V2 导出时可直接读取，不必重新解析古籍：

```powershell
python -m wengua_quant --export ..\WenguaApp\local\probability_v2\<一次导出的目录>\data --backend logistic
```

每次自动创建唯一的 `local/runs/时间戳/`；`--app-root` 另外创建同级 `*_source_export`，只读源库，不覆盖已有数据。失败也留收据。研究输出 `evaluation.json` 和 `heldout_predictions.jsonl` 含逐事件结果；不应当发布含敏感信息的本地数据。对应人工审查及版本策略见 [Research Protocol](docs/RESEARCH_PROTOCOL.md)。

## 开发/验证

```bash
python -m unittest discover -s tests -v
```

测试仅验证**合成输入**的管线、分组与防泄漏，不声称已用真实医学案例训练或通过外部盲测。实际 GitHub Actions 与当前阶段见 [Project Status](docs/PROJECT_STATUS.md)。

### Mac 本地运行（两个仓库实际路径可不同）

如果 `WenguaQuant` 是放在 `WenguaApp/WenguaQuant/` 内，仍然在 WenguaQuant 仓库内运行，`--app-root` 指向上级 WenguaApp 的完整路径。出现 `Unsafe, missing-source, or duplicated feature catalog entry` 时，先 `git pull --ff-only` 获取 2026-11-03 的精确字段名修复；此错误发生在训练前的特征校验，**不能算训练成功**。已导出的旧文件不删除、不覆盖。

```bash
cd /path/to/WenguaApp/WenguaQuant
git pull --ff-only
source .venv/bin/activate
python -m unittest discover -s tests -v
python -m wengua_quant --app-root /path/to/WenguaApp --backend logistic --device CPU
```

本次 Mac 源导出校验失败的永久审计及回归用例见 `docs/audits/MAC_V2_CATALOG_REJECTION_20261103.md`。


### 已有Mac实验的结果诊断（无需重训）

2026-11-03 首次逻辑回归真实书内试验的五组模型，**全部将12件物品预测为“找回”**。独立11事件家族、12物品、10找回/2未找回；事件加权准确率86.4%，但专业特征组Log Loss高于无九宫先验。不能以该成绩证明九宫专业因子有稳定预测价值。

升级代码后，直接审计已存在的输出（只读，不训练也不覆盖文件）：

```bash
cd /Users/masonliu/WenguaApp/WenguaQuant
git pull --ff-only
source .venv/bin/activate
python -m unittest discover -s tests -v
python -m wengua_quant.audit local/runs/20261103T164538_321806Z
```

完整逐案失败模式和避免在已知11个案例上反复调参的约束：`docs/audits/LOST_PROPERTY_MAC_PILOT_DIAGNOSTIC_20261103.md`。


## Independent Wengua Master V3 model-data curriculum

WenguaQuant now also hosts an **isolated** traditional Qwen SFT data-preparation track under [wengua_master_v3/](wengua_master_v3/), completely separate from the existing observed-outcome `wengua_quant/` pipeline. Public [V3 introduction](docs/master_v3/README.md), [dataset audit](docs/master_v3/V3_DATA_AUDIT_20261104.md) and [1,024 hypothetical palace relation drills](data/master_v3/public_rule_drills.jsonl) are available. The 12-case/55-author-step/20-QA workbook-derived source-study bundle remains private and user-downloadable; it is not published as verbatim ancient book data and does not imply true foreknowledge. See audit for exact 87 candidate tasks and family holdouts. Original WenguaApp V2 13,341-dialogue course and all frozen iOS scoring remain unchanged.
