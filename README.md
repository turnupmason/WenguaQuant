> **2026-11-12 · V6.7补评完整审阅。** 150成员、原329项及9恢复worker已核；只增198实答、0更新，483答全齐，原训练与285答字节保留，414边界显存样点无OOM，回答后最低free8.822GiB。父/E1/E2清楚整答143/142/140；新9条件8/9/9、新14自身历史10/14/14，纠错10/8/8、RAG16/14/14、星属性核对整答3/1/1，均不全局晋升。开发44参考答均训练已见，仅29文本；保留/改正16:16及25:25，不能简单归因缺书或标签不平衡。已纠正旧报告原始盘/事实卡误分：新8+8题三模型均全对，不冒充新增能力。E1 step113仅下一研究起点，完整194案未进本短训；451单测，完整原包两份/483注释/Excel加密仅评估，不入RAG/SFT。未追加训练或覆盖旧权重；见[完整结果、原因与后续边界](docs/model_reviews/20261112_v67_complete/README.md)。

> **2026-11-12 · V6.7训练完成、评估OOM；先补评不重训。** 70成员/329冻结项/4worker已核，113+113正式更新完成，901视图×2实际loss调用1802，训练56样点无OOM。父161答及E1的124答共285逐答SHA绑定；E2未生成，缺198答。原星位整答11→14/16、星属性错误5→0，新9条件主结论8→9；纠错10→8/10、旧婚姻理由5→4/6、RAG16→14/18，前124整答112→111，不作整体晋升。评估异步OOM仅有加载前显存，不能断言唯一原因。独立恢复入口`resume_wengua_v67_evaluation.cmd`只追加评估，每答清缓存/最多24答换进程，原权重/检查点/答案字节不变；445单测与微型CPU中断恢复通过，实际Windows/Linux Actions按新提交核对。原包与285注释加密仅评估，不入RAG/SFT；见[完整审阅与PowerShell恢复命令](docs/model_reviews/20261112_v67/README.md)。

> **2026-11-12 · V6.7对象与独立条件续训。** 继承V6.6普通step77，普通CE固定1e-5，七投影r16连续两轮226更新，分别保留全部轮末结果，不按loss晋升。841训练核心/514原样复习，展开901唯一监督视图，合并55重复；星本体/宫属性与比较对象分列，两个条件真/假/未知交叉、单项更新及多轮，原216来源/194案划分/1804旧核心保存，无新书/模型教师/人工评分。原103父答字节复用只补58题，每阶段161答含原题、原RAG、普通任务及实际自身历史；新原始盘/事实卡成对看。真实tokenizer31069监督token/最长1135，Excel/SQLite全读回；435单测与微型CPU继承、空新优化器、两轮、精确恢复和旧答CRLF复用通过，实际8B效果待本机。入口`train_wengua_v67.cmd`，结束自动导出；见[教材、流程与PowerShell](docs/course_v67/README.md)。

> **2026-11-12 · V6.6两路训练实跑审阅完成。** 121成员/308冻结项/7实际worker，154正式+4短测更新、309实答及114逐token目标已核；615张量顺序复现一致，38 train样点无free0/OOM。普通组双宫原问2→6/6、正误判词及方向5→10/10；星位方向5→13/16（加权15），但星自身属性新增5/4条错误。加权关键NLL更低却纠错7/10、多轮条件有2条明确矛盾；两组同一日常改写退步。正确事实教师前缀下，父及两组40个生克目标都top1，不能以该NLL替代整答。婚姻24核心第一前提均满足，缺独立交叉条件。普通step77仅下一修复研究起点，旧E2与全部组保留，不作全域晋升，未启动新训练；7文件加密仅评估，不入RAG/SFT。425软件测试通过，见[全部结果、原因与下一步](docs/model_reviews/20261112_v66/README.md)。

> **2026-11-12 · V6.6可见前提与关键位置损失对照。** 继承V6.5已学E2 step458，同课程/顺序/固定1e-5，普通CE与关键位置4倍相对权重各1轮77正式更新；新阶段新优化器，同阶段完整恢复，不再扫起点/学习率。1780训练核心保留，分层586核心/615监督视图短训，婚姻领域/规则实际可见，关系参考答先核属性再给自然结论；216来源/194案划分/60来源对话及旧权重不改，无旧评分或模型错答教师。真实tokenizer19801监督token/2393关键位置、Excel/SQLite全读回；419单测与微型CPU损失/梯度、同起点两路、精确恢复通过，真实8B改善待本机。父模型+两终点各103自由答/38逐token诊断，不自动晋升或追加长训。入口`train_wengua_v66.cmd`，自动导出；见[课程、对照与PowerShell](docs/course_v66/README.md)。

> **2026-11-12 · V6.5学习定位实跑已审阅。** 26成员、289冻结项、2实际CUDA worker、162原答/76逐token目标已核，更新0次；两源252个B均非零冻结。E2原题604监督token仅7处gold-prefix非top1，集中6题，六个首错均对应实际输出；13/19逐字复现不代表专业准确率。双宫原题2/6改问6/6，婚姻原题3/3改问有错；实际可见输入另有18婚姻监督视图/7固定题未明确领域候选。星位15→16/16，但定位+方向15→5/16；本组E2四段11实际自身历史回合相符，不能外推所有多轮。保留E2作定向修复候选与全部旧源，不再按平均loss晋升；下一版需修输入范围与关键结论损失，尚未新训。162注释、原答/token/Excel加密仅评估归档，不入RAG/SFT。411软件测试通过，见[完整定位与改动依据](docs/model_reviews/20261112_v65_learning/README.md)。

> **2026-11-11 · V6.5原题、改问与实际多轮只读检查。** 原起点与继承E2 step458两模型，不重复官方分支；各19实际训练原题+19同事实改问、6宫属性+6给属性方向、16真实星位摘录、4最终条件单轮+11自身历史追问，共162自由答与76逐token条件化记录。审核判词/关系/依据分段，NLL不当自由断案准确率；不自动训练或晋升，全部权重与283旧运行项保留。70固定题/38教师前缀与offset已用真实tokenizer核对；微型Qwen无梯度、原生CE一致、自身历史、UTF8恢复/完整缓存、Excel导出核验；406软件测试通过，真实8B效果待本机。入口`check_wengua_v65_learning.cmd`，无新ZIP口令，自动导出Downloads。见[定位范围与PowerShell命令](docs/course_v65_learning/README.md)。

> **2026-11-11 · V6.5两起点逐答审阅完成。** 109成员、283冻结项、6实际worker、916正式+4短测更新已核；348正式答含52原样复用，全部绑定原答SHA注释。221 train样点无free0/OOM、峰值reserved7.936GiB；实际CUDA前两步缓存误差在预设阈值，已学adapter数值生效。已学E2人物主字段3→6/6、纠错0→2/5，但双宫整答5→3/6、放贷1→0/2；原12训练回忆与4普通问恢复。官方组不作替代。1780课762末答、724比较模板，评估没有新九宫读取或专业多轮；不称全领域升级。保留全部权重，已学E2作修复候选，不启动新训；仅评估加密归档，禁止入RAG/SFT。398软件测试通过，见[完整审阅与下一步边界](docs/model_reviews/20261111_v65/README.md)。

> **2026-11-11 · V6.5对象、属性、方向与领域结论修课。** 1780来源训练核心/1832完整监督回合，24新开发题；九宫星位只取八种真实旋转，人物关系明确、多轮修字段/条件、正误均衡及整段纠错，原194案划分/216材料保留，实际4案例课、不称全案例已覆盖。固定2e-5、同七投影r16，已学V6.3 step371与官方封存step0同课程各2轮458更新；各阶段58答，52旧答精确复用，保留所有阶段不自动晋升。394单测、真实tokenizer全遮罩前缀/完整目标、Excel/SQLite全读回、微型Qwen两起点/两轮/数值核对及精确恢复通过；实际8B改善待本机。无新ZIP口令，入口`train_wengua_v65.cmd`，自动导出。见[课程与PowerShell命令](docs/course_v65/README.md)。

> **2026-11-11 · V6.4实跑逐答审阅完成。** 两率各152正式更新/两轮、5实际worker完成；264冻结项、570正式存储答与2短测答、74父答原样复用核对，570答均有完整SHA注释。显存75个train样点无free0/OOM，峰值reserved7.893GiB。主字段旧5题2→高率E2的5，但新12星宫关系整答5→3/3/0/1，新纠错各1/5；高率RAG退成戊戊比和格、父亲本人取乾宫。6训练首回合/6开发称谓关系有歧义候选，567核心394末答、143固定身份起句。低loss不代表整体升级，父权重及四阶段全保留；原书/课程/起盘不改，审阅包仅评估、不得入RAG/SFT，未发布新训练。见[逐轮与课程分析](docs/model_reviews/20261111_v64/README.md)。

> **2026-11-11 · V6.4条件、对象与纠错续训。** 原两书216材料/194案划分保留；567训练核心展开608完整监督回合，40新开发题，405旧课逐项处置。24真实主体切换+7不变对照，6比较层次切换，25五行对各有正误核对；文字来源存在/缺失对照。只提取1080库实际星位，无旧分数/81格标签。继承V6.3高率E7 step371，同七投影r16；1e-5/3e-5各2轮152更新，每阶段114答，旧74父答复用只补40，不自动追加长训或替换模型。Excel/SQLite启动时本地生成，无新口令ZIP；入口`train_wengua_v64.cmd`，结束自动导出。当前仍是定向修复，不称两书全案例训练完成；实际8B效果待本机。见[课程、恢复与PowerShell](docs/course_v64/README.md)。

> **2026-11-11 · V6.3九模型对照已逐答审阅。** 342存储答/288新增/54原样复用、9实际worker、252冻结项及8组252个B矩阵非零核对；0更新。后期RAG并非全面更好，2答重复至上限；本人/同辈字段和星宫对象仍错。405训练核心仅4条关联案例、213种教师文本、128次固定范围尾句；15切换中5条本人到本人。高率7轮step371只作定向修课续训候选，官方和高率4轮保留，不自动晋升；未发布新训练或再跑同题评估。完整原答/Excel加密且仅评估；见[九模型分析](docs/model_reviews/20261111_v63_checkpoints/README.md)。

> **2026-11-10 · V6.3实跑审阅完成，先补权重比较。** 低率8轮424更新、高率7轮371更新正常完成，无OOM；414正式答+2短测、8 worker及247冻结项已核。两组4轮后12原题逐字复现，人物切换、纠错与自然婚姻条件仍错，非未知盘能力证明。RAG只测了早期loss候选，新增0更新全阶段对照：官方+8阶段各20来源开发题和18原RAG，复用54答、新生成288答；不重复已有12原题/24诊断，不改旧权重或原书，不发布新训练。入口`check_wengua_v63_checkpoints.cmd`，完成自动导出；见[逐轮逐答与命令](docs/model_reviews/20261110_v63/README.md)。

> **2026-11-10 · V6.2学习对照已审阅，V6.3实际训练。** 新108答/三worker/0更新与235冻结合同已核；两组LoRA均144个B非零且数值生效，但训练原题仍错，系统提示也影响基本方向。保留全部旧权重；新版本只统一SYSTEM，原405核心/424回合及来源、用户、教师逐字不变；七投影r16，5e-5/1e-4各最多8轮424更新，原题/条件/RAG分层保存，初始化含融合/缓存数值核对。不是两书完整课程已学会，不自动专业晋升。入口`train_wengua_v63.cmd`，结果`export_wengua_v63.cmd`；见[108答审阅](docs/model_reviews/20261110_v62_learning/README.md)与[参数、恢复及启动](docs/course_v63/README.md)。

> **2026-11-09 · V6.1实跑已审阅。** 两率各62步/两轮，7个worker正常完成；294正式生成、2短测，六套原检查重放一致。低率第一轮有方向改善，也把未知父亲年命等答错；高率第一轮局部更贴题，但出现方向/生成数错误及一次重复至上限。原权重保留，未启动新训练。新增四模型同题的24题正常对话诊断（96回答），只推理，不更新权重；见[逐组审阅与命令](docs/model_reviews/20261109_v61/README.md)。

> **2026-11-09 · V6.1定向对话续训。** 320训练核心、324助手视图，显式复习后486视图/轮；正常对话与人物条件约64%监督token。两率3e-6/1e-5各两轮62更新，继承V5.8 step97，新阶段新优化器，同阶段完整恢复。每轮40既有开发题、每组候选18既有RAG，原答完整保留，不自动晋升。新教材独立加密交付，原两书与194案例/起盘不变；不是两书全量课程已重编完成。入口 `train_wengua_v61.cmd`，见[训练与导出](docs/course_v61/README.md)。

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
