# V5.9 第158步停顿：保留原合同的续训入口

用户日志记录了同一worker的两类情况：张量采样约5.90 GiB、峰值8.95 GiB，但分配器reserved约15.12 GiB，可用显存长期为零；allocator OOM计数为0。第140步释放缓存后，可用显存一度回到7.17 GiB，之后reserved再次增长。末次记录仍在第158步，随后原监控因600秒没有心跳或输出而调用中断，产生`KeyboardInterrupt`。

这证明原监控发出了停止请求，并显示显存分配/缓存压力；现有片段不能独立证明最后阻塞的是哪一个CUDA操作、是否发生Windows共享显存换页或驱动停顿。不能把它报告成已经发生的CUDA OOM，也不能断言只是误报。用户目录中可见98、100、125和150检查点，恢复时仍由原程序校验完整封存，不能仅凭文件夹存在就认定有效。

## 修复范围

新增`wengua_recovery/`及`resume_wengua_v59.cmd`，固定原提交`277047d8489edf8c3db48f95eb9cf7f5cd5b376a`的180项运行代码/教材SHA。原V5.9训练文件、课程、评估、参数、父权重、优化器和检查点合同均不改；原`sweep_contract.json`、`process_execution.json`和stage合同保持原样。补丁另写`recovery_receipts/`，不靠改写旧SHA来绕过检查。

- 在新worker加载模型前设置PyTorch分配器上限为显卡总显存的80%，4080约12.79 GiB。它约束分配器，不是对其他进程预留显存，也不限制所有驱动和库的额外开销。
- 在完整微批次、优化器更新和验证批次结束时检查：reserved达到10 GiB且reserved−allocated至少2 GiB时释放闲置缓存。不会删除活跃张量、修改精度、跳过样本、重设优化器或改变课程顺序。具体释放量另行记录；碎片中不能归还的区段仍可能保留。
- 各进度文件逐个比较mtime与size，避免一个未来时间戳遮住后续真实心跳。只将主线程完成批次、原心跳与实际输出算成训练进度；诊断线程存活不冒充训练进展。
- 静默3分钟警示，5分钟记录GPU状态并请求全线程堆栈，30分钟仍无真实进度才中断并保留检查点。总预算仍受原24小时期限约束。此项是带诊断的有限宽限，不能代替显存压力处理。
- 保留旧失败文件副本和新请求/回执，继续使用nonce、请求SHA及恢复代码/策略SHA校验。旧smoke保持冻结，复用时**不称它已经验证新缓存策略**；新策略是否适合实际4080须看本机续训的显存及原答记录。

该额度根据本次约8.95 GiB张量峰值留出余量；未在云端用真实8B CUDA验证。若真实峰值或不可回收碎片超过额度，程序会报错并保留完整检查点，不自动缩短课程或跳过长样本。

## PowerShell续训

保持原V5.9输出，在PowerShell执行：

```powershell
Set-Location "$env:USERPROFILE\WenguaQuant"
git pull --ff-only origin main
.\resume_wengua_v59.cmd
```

它自动选择原阶段最后完整且通过校验的checkpoint；本次目录通常应为150，从那里继续完成196步短训，再按原112题和36个RAG包的逐题留存决定是否进入完整346步。150之后尚未落盘的更新会重做；不是从头训练。原损失和验收标准保持冻结。

若再次暂停，重复同一命令。需要显式指定时，只能指定本次阶段最后完整封存的检查点：

```powershell
.\resume_wengua_v59.cmd --resume-from-checkpoint ".\local\course_v59\run1\trials\mixed_pilot\cycle_01\trainer\checkpoint-150"
```

不删除原输出、不修改JSON合同、不重新建立实验目录。`needs_course_review`仍表示能力留存未通过，`paused`表示恢复入口暂停，二者与正常完成不同。

## 可核对记录

本次新worker请求另有`.allocator_policy.json`、`.main_progress.json`、`.cache_releases.jsonl`；停顿时新增`.stack.log`、`.stall_diagnostic.json`，监控主动停止时新增`.monitor_stop.json`。原`.heartbeat.json`、`.memory.json`、注意力/有界loss证明及逐题原答继续保留。新堆栈使用`.log`后缀，现有结果ZIP导出会自动包含。

软件回归用本次用户提供的缓存数值及时间戳反例；CPU集成先由**原V5.9进程**写一个非零权重、完整优化器检查点，再由**新恢复进程**接续，比较最终每个adapter张量与不中断训练完全相同。同时检查有限loop、失败留存及导出路径。CPU结果不能代替本机GPU的缓存策略验证。新GitHub工作流同时检查真实Windows CMD入口和Linux恢复。
