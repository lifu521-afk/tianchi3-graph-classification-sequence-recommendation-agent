# 数据放置说明

## 不上传数据

比赛原始数据、样例提交、训练/测试 CSV、NPZ、ZIP、模型输出和缓存均**不属于本仓库发布内容**。它们可能受比赛规则、数据许可和平台条款约束。请仅从官方比赛页面获得数据，并自行确认是否允许本地使用和二次分发。

`.gitignore` 默认忽略所有 `*.csv`、`*.npz`、`*.zip`、模型权重和本地 `data/A分类`、`data/A推荐`、`data/B分类`、`data/B推荐` 目录。提交前请运行 `git status --ignored` 再检查一次，防止误传。

## 本地目录结构

请从天池比赛页面下载并按下面结构放置：

```text
data/
├─ A分类/
│  ├─ A1.npz
│  └─ sample_submission.csv
├─ A推荐/
│  ├─ train.csv
│  ├─ test.csv
│  ├─ user.csv
│  ├─ item.csv
│  └─ sample_submission.csv
├─ B分类/
│  ├─ B1.npz
│  └─ sample_submission.csv
└─ B推荐/
   ├─ train.csv
   ├─ test.csv
   ├─ user.csv
   ├─ item.csv
   └─ sample_submission.csv
```

代码中的 `--data`、`--data-dir` 和 `--template` 参数可以指向任意本地数据目录，无需强制使用上述路径。

## 字段与提交规则摘要

### B1 节点分类

- `B1.npz` 保存 CSR 图邻接与 CSR 属性矩阵，测试节点标签为 `-1` 占位，不是真实标签。
- 提交列固定为 `test_idx,label`。
- 不得重排、删除、增加或重复模板中的 `test_idx`。
- `label` 是 0—7 的整数。

### B2 序列推荐

- 训练文件含 `target_iid`，测试文件不含目标 item。
- `item_seq_raw` 是最近交互序列；`item_seq_dedup` 与 `item_seq_counts` 是辅助序列表示。
- 提交列固定为 `uid,prediction`。
- `prediction` 为按置信度从高到低排列的 10 个 `iid`，英文逗号分隔、无重复、都存在于 `item.csv`。
- `uid` 的行数和顺序必须与 `sample_submission.csv` 保持一致。
