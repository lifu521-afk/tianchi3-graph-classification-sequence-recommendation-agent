# Tianchi 3 Competition Code

天池三比赛的可运行代码，包含图节点分类、金融行为序列推荐和可选的串行实验 Agent。仓库不上传比赛数据、提交文件、模型权重、缓存或历史实验结果。

## 目录

```text
src/       比赛基线、神经模型、两个任务 Agent、提交校验
agent/     可选的实验控制器及最小配置
data/      本地数据放置说明，不含数据集
```

## 环境

```powershell
conda activate torch5070
pip install -r requirements.txt
```

## 数据

按照 [`data/README.md`](data/README.md) 将从天池下载的数据放在本地 `data/` 目录，或设置环境变量 `TIANCHI3_DATA_DIR`。数据集不会进入 Git 版本库。

## 运行 B1 图节点分类

```powershell
python src/autonomous_classifier_agent.py `
  --data data/B分类/B1.npz `
  --template data/B分类/sample_submission.csv `
  --output-dir outputs/b1 `
  --task-id B1 `
  --budget-minutes 110
```

## 运行 B2 序列推荐

```powershell
python src/autonomous_recommender_agent.py `
  --data-dir data/B推荐 `
  --output-dir outputs/b2 `
  --task-id B2 `
  --budget-minutes 110
```

也可以运行规则、MLP、GRU 和分桶融合脚本：

```text
src/b_classification_pipeline.py
src/b_recommendation_pipeline.py
src/b_neural_recommender.py
src/generate_mlp_gru_submission.py
src/generate_bucket_mlp_gru_submission.py
src/generate_fine_bucket_mlp_gru_submission.py
```

## 提交校验

```powershell
python src/validate_submission.py --b1 outputs/b1/submission.csv --b2 outputs/b2/submission.csv
```

校验包括行数、模板顺序、标签范围、Top-10 长度、item 合法性和重复推荐。

## Agent

`agent/agent.py` 是可选的实验控制器，负责在预算内串行选择已登记实验、运行脚本、读取指标和记录状态。配置位于 `agent/config.json`。默认只保留公开仓库内的 baseline 实验；本地运行生成的 `state.json`、`memory.jsonl` 和 `runs/` 不提交。

```powershell
python -m agent.agent --status
python -m agent.agent --max-rounds 1
```

## 项目效果

项目已经形成从数据读取、基线建模、结构/序列增强、可靠验证到提交审计的完整闭环。已确认的线上 anchor 为：A 榜总分 `0.6309`（A1 `0.7590`，A2 `0.5027`），B 榜总分 `0.32221`（B1 `0.42680`，B2 `0.21763`）。两套榜单协议不同，数字仅作各自项目内的复盘参考。

关键技术不是单纯堆模型，而是：

- 用内容/规则模型建立低方差 anchor；
- 用入边、出边、转移和重复偏好补充结构信号；
- 用 OOF、test-like、分桶和最差折验证泛化风险；
- 用 Agent 管理假设、预算、失败记录和候选接受；
- 用提交校验避免格式错误和结果污染。

详细内容见：

- [`docs/technical_route.md`](docs/technical_route.md)：技术路线
- [`docs/experiment_summary.md`](docs/experiment_summary.md)：结果与方法结论
- [`docs/reproducibility.md`](docs/reproducibility.md)：复现与审计说明

## 合规边界

代码仅用于比赛复现和离线研究。实际部署到金融、医疗或其他高风险场景前，需要重新完成数据授权、隐私保护、时间切分、偏差评估、人工审核、监控和回滚设计。
