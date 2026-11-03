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
