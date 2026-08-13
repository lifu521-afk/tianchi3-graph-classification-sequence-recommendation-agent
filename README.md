# Tianchi 3 Competition Research

天池三图机器学习挑战赛的完整代码与研究复盘。本项目围绕两类匿名化数据任务构建了**可复现基线、可靠验证、自动化实验 Agent、提交审计和技术复盘**。

> [!IMPORTANT]
> 本仓库**不上传任何比赛数据集、提交文件、模型权重、缓存 logits 或平台产物**。请从比赛平台获取数据，并遵守其数据使用规则。`.gitignore` 已默认阻止 CSV、NPZ、ZIP、checkpoint 和本地配置进入版本控制。

## 项目能做什么

| 能力 | 对应任务 | 项目提供的工作 |
|---|---|---|
| 图节点分类 | A1 / B1 | CSR 图数据读取、稀疏特征处理、入/出边聚合、线性基线、图平滑、域偏移诊断 |
| 序列推荐 | A2 / B2 | 流行度/转移召回、重复偏好、MLP 序列分类、GRU、Top10 重排与长度分桶融合 |
| 可靠评估 | 两个任务 | OOF、test-like 切分、历史长度匹配、最弱分桶、差分审计、bootstrap 稳定性分析 |
| 自动实验 | B1 / B2 | 数据画像、候选策略串行运行、自动选择、轨迹与报告输出 |
| 提交审计 | B1 / B2 | 行数、模板顺序、类别范围、Top10 长度、重复项、合法 item、ZIP 完整性检查 |

## 赛题与规则

### B1：节点分类

输入为稀疏节点特征和图邻接矩阵，预测测试节点的类别。

| 项目 | 官方数据说明 |
|---|---|
| 节点数 / 属性维度 | 7,650 / 512 |
| 类别数 | 8 类，标签为 `0`—`7` |
| 公开训练 / 测试节点 | 6,120 / 1,530 |
| 图格式 | CSR：`adj_data`、`adj_indices`、`adj_indptr`、`adj_shape` |
| 特征格式 | CSR：`attr_data`、`attr_indices`、`attr_indptr`、`attr_shape` |
| 提交格式 | `test_idx,label` |

提交时必须保留模板中的 `test_idx` 行及其顺序；标签必须为合法整数类别。邻接关系应按原数据使用，不能擅自假设图必然无向、对称或无自环。

### B2：金融场景序列推荐

给定用户近期 item 交互序列、匿名用户属性和匿名 item 属性，为每位测试用户输出按置信度排序的 Top10 item。

| 项目 | 官方数据说明 |
|---|---|
| 训练 / 测试用户 | 40,000 / 10,000 |
| 用户表规模 | 50,000 |
| item 目录 | 14,065 个 item |
| 训练字段 | `uid,target_iid,item_seq_raw,item_seq_dedup,item_seq_counts` |
| 测试字段 | `uid,item_seq_raw,item_seq_dedup,item_seq_counts` |
| 用户属性 | 8 个匿名离散字段 |
| item 属性 | 3 个匿名类别字段 + 1 个 bucket 字段 |
| 提交格式 | `uid,prediction` |

`prediction` 必须是英文逗号分隔、从高到低排序的 10 个 item id；每行不得重复，且 item 必须来自 `item.csv`。提交 `uid` 的行数和顺序必须与模板完全一致。

## 技术亮点

### 1. 不把随机交叉验证当成最终答案

OOF（折外预测）用于模型初筛和错误分析，但不会直接等价于线上表现。B1 存在训练/测试节点分布偏移，因此增加了基于 train/test propensity 的 test-like 验证。B2 通过从训练样本截断历史、匹配测试历史长度分布来评估排序策略，避免只在完整历史上乐观估计。

### 2. 从“模型堆叠”转向“保守更新”

每轮实验从已验证 anchor 出发，只接受通过重复切分、最弱分桶和差分审计的候选。推荐任务还对空历史和 Top1 结果进行保护，防止局部规则把稳定结果改坏。

### 3. 把提交文件视为可审计产物

`src/validate_submission.py` 可在不访问平台的情况下检查 B1/B2 输出及 ZIP 完整性。文件格式、行顺序、非法类别或非法 item、重复推荐项均会导致失败。

## 快速开始

### 1. 创建环境

推荐使用 CUDA 环境 `torch5070`：

```powershell
conda activate torch5070
pip install -r requirements.txt
```

### 2. 下载并放置数据

从比赛平台下载数据，按照 [`data/README.md`](data/README.md) 的目录结构放置。本仓库不会提供数据下载链接、镜像或任何原始数据副本。

### 3. 运行测试

```powershell
python -m unittest discover -s tests -v
```

### 4. 运行自动实验 Agent

```powershell
python src/autonomous_classifier_agent.py `
  --data data/B分类/B1.npz `
  --template data/B分类/sample_submission.csv `
  --output-dir outputs/b1_agent `
  --task-id B1 `
  --budget-minutes 110

python src/autonomous_recommender_agent.py `
  --data-dir data/B推荐 `
  --output-dir outputs/b2_agent `
  --task-id B2 `
  --budget-minutes 110 `
  --validation-repeats 3
```

详见 [`docs/reproducibility.md`](docs/reproducibility.md)。

## 目录结构

```text
src/
  autonomous_classifier_agent.py   # B1 自动实验与验证
  autonomous_recommender_agent.py  # B2 自动实验与验证
  run_solution.py                  # A 榜集成方案与通用工具
  b_classification_pipeline.py     # B1 稳定基线
  b_recommendation_pipeline.py     # B2 规则与评估基线
  b_neural_recommender.py          # B2 神经推荐组件
  scratch_torch_recommender.py     # MLP 序列模型
  scratch_torch_gru.py             # GRU 序列模型
  validate_submission.py           # 提交校验工具
configs/                            # 可公开配置模板
docs/                               # 技术路线、实验摘要与复现说明
data/                               # 数据放置规则，不包含数据
results/                            # 小型结果摘要，不包含预测或缓存
tests/                              # 回归测试
```

## 可以应用到哪些工作

- **图机器学习任务**：稀疏属性图分类、带方向关系的实体分类、转导学习中的训练/测试域偏移诊断。
- **序列推荐任务**：短行为序列推荐、重复消费/复购预测、冷启动与空历史用户的保守策略、Top-K 排序融合。
- **自动化实验系统**：在固定算力预算下完成“数据画像—候选策略—可靠评估—报告生成”的实验闭环。
- **模型发布审计**：将输出 schema、输入顺序、目录版本、预测差分和结果哈希纳入交付流程。

这些代码是比赛研究原型，不应直接用于金融生产决策。实际部署需要重新完成数据合规、隐私保护、偏差评估、在线 A/B 测试、监控与人工审核。

## 后续可扩展方向
更详细的能力边界、垂直项目清单和研究迭代计划见 [`docs/vertical_projects_and_research_plan.md`](docs/vertical_projects_and_research_plan.md)。

1. **B1 域泛化**：使用更严格的环境划分、域对抗/重要性加权，以及图结构与属性的稳健融合。
2. **B2 排序学习**：用 listwise/reranking 学习替代手工融合，并对不同历史长度建立校准的 gate。
3. **多任务表征**：在推荐侧联合建模目标 item、重复行为、序列长度和用户群体，以减少短历史方差。
4. **不确定性与预算分配**：将 OOF 分歧、域偏移分数和候选覆盖率用于主动选择下一轮实验，而不是全量搜索。
5. **实验 Agent**：拆分 profile、planner、runner、evaluator、auditor 模块，沉淀为可复用的表格/图数据实验框架。
6. **工程完善**：加入 CI、预提交检查、可追踪实验数据库、数据版本指纹和可选的 Docker 环境。

## 已记录结果

项目报告中记录的已确认线上 anchor：A 榜 `0.6309`（A1 `0.7590`，A2 `0.5027`）；B 榜 `0.32221`（B1 `0.42680`，B2 `0.21763`）。A/B 榜的数据、候选目录和评测协议不同，不能直接横向比较；本地 OOF 或候选结果也不能写成线上成绩。

更完整的技术与实验复盘见：

- [`docs/technical_route.md`](docs/technical_route.md)
- [`docs/experiment_summary.md`](docs/experiment_summary.md)
- [`docs/reproducibility.md`](docs/reproducibility.md)

## 数据与合规

禁止上传或公开：比赛原始数据、提交 CSV/ZIP、模型权重、大型二进制缓存、平台内部文件、个人路径、API Key、Token 或其他敏感配置。使用本项目即表示遵守天池的比赛规则和数据授权要求。
