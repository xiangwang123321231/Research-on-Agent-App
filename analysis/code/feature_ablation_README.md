# 59 维特征组消融实验

> 2026-11-06：本入口继续冻结59维，使用 `feature_vectorizer_59d.py`；不会将60维输入套入旧分组。生产提取/导出已改为[60维](../docs/feature_vector_60d.md)，本页实验结果仍按原契约解释。

当前方案：2026-10-28，完整方法与七组逐组移除，共八种配置。完整设计、列编号、解释边界和审计方法见[59 维消融方案](../docs/feature_ablation_59d_protocol_20261028.md)。排除未决样本后的 247 个 APK 正式运行及独立审计均已完成，见[本轮报告](../ablation_results/59d_20261028_153816_265669/report.md)和[审计记录](../ablation_results/59d_20261028_153816_265669/audit.json)。

## 固定配置

| 配置 | 删除内容 | 保留维数 |
|---|---|---:|
| A_full | 无 | 59 |
| B_no_service | 服务身份与访问配置 | 44 |
| C_no_protocol | 语言生成与协议语义 | 51 |
| D_no_web | 网页交互 | 53 |
| E_no_device | 端侧推理 | 40 |
| F_no_http_relations | HTTP 局部方法关系 | 56 |
| G_no_density | 通用证据数量与来源 | 55 |
| H_no_size_quality | APK 规模与分析质量 | 55 |

固定 ExtraTrees：300 棵树，`max_features=6`、`max_depth=8`、`min_samples_leaf=2`、`class_weight=balanced`、`random_state=42`。所有配置共享原始行顺序和分层五折，40 次拟合。预先比较完整组与七个消融组，不做组合搜索或逐组调参。

这是列组消融，不是关闭提取机制。删除专用列后，其他列仍可能携带同类信息；例如 f22 依赖 f08/f20，通用计数也可能包含网页或端侧事实。F 只移除 HTTP 关系，f45/f46 分别随网页/端侧组移除。

## 运行

```powershell
python D:\work\analysis\code\run_feature_ablation.py --n-jobs 4
```

默认读取项目 `output/` 下的 `features.csv`、`labels.csv`、`apks.jsonl` 和特征 manifest；参照 `result/predictions.csv` 的 ExtraTrees 折外预测。路径由脚本位置确定，Linux 下可在项目目录执行 `python code/run_feature_ablation.py --n-jobs 4`。

默认新建 `ablation_results/59d_日期_时间_微秒/`。用 `--output-dir` 可指定一个尚不存在的目录；可用 `--features`、`--labels`、`--signals`、`--reference-predictions` 成套指定其他输入。

运行前检查精确 59 列、单标签、唯一 APK 身份、有限数值、输入集合、版本和信号到特征映射。合法的 57→59 重映射需有对应 derivation 记录，不使用 `--allow-legacy` 绕过检查。当前 CLI 仅接受 `--max-features 6`，默认即为 6。

完整组与参考预测不一致时默认停止，保留失败清单；调查后才可在新目录显式使用 `--allow-reference-drift`。后者不改变本轮各组之间的比较条件，报告仍记录参考差异。

仅重新训练实验用分类器，不重新扫描 APK，不更改标签、特征、正式模型或旧结果。保存输入与代码快照后开始拟合。

## 结果与审计

`report.md` 为中文入口。另存总体/分类别指标、统一折分、逐 APK 预测与概率、配对差值和分歧案例、八配置混淆矩阵及 Macro-F1 图。案例表可以只有表头，表示没有预测分歧。

默认配对分层 bootstrap 2,000 次，种子 20261011。区间以固定 OOF 预测为条件，未经多重比较校正，不代表独立验证或完整训练不确定性。标签待复核样本不自动改标或排除。

```powershell
python D:\work\analysis\code\audit_feature_ablation.py --run-dir D:\work\analysis\ablation_results\59d_实际运行目录 --n-jobs 4
```

将占位路径替换为实际输出目录。审计读取该轮输入快照，另行独立拟合 40 次并检查指标和文件完整性；默认新增 `audit.json`。已有审计文件不覆盖，可用 `--output` 指定新文件。当前运行代码必须与实验源代码快照一致；不允许 `python -O`。

## 历史记录

- 旧五配置 31/52/47/48/43 维结果仅属于当时的代码、输入和标签；当前配置编号已重新定义，不能按 A/B 字母跨版本比较。
- 历史结果保留在原 `ablation_results/` 子目录中，例如 [2026-10-16 报告](../ablation_results/fixed6_248_20261016/report.md)。不将其指标写成当前 59 维性能。
- 本轮修改前脚本及说明已备份到 `recovery/ablation_scripts_before_59d_20261028_145140/`。该备份只保存修改前状态，旧复现仍需配套数据、代码及依赖。
- `run_max_features_sensitivity.py` 是历史 52/48 维敏感性分析，不是当前消融入口。
- f58/f59 的既有四配置对照见[实施记录](../docs/semantic_feature_optimization_20261025.md)，本轮不自动迁移或覆盖其结果。
