# 复现说明

## 环境

建议使用 `torch5070` 环境：

```powershell
conda activate torch5070
pip install -r requirements.txt
```

## 数据

从天池下载数据后按 `data/README.md` 放置，或设置 `TIANCHI3_DATA_DIR`。数据、提交文件、权重和缓存均不进入仓库。

## 运行与审计

```powershell
python src/autonomous_classifier_agent.py --data data/B分类/B1.npz --template data/B分类/sample_submission.csv --output-dir outputs/b1 --task-id B1 --budget-minutes 110
python src/autonomous_recommender_agent.py --data-dir data/B推荐 --output-dir outputs/b2 --task-id B2 --budget-minutes 110
python src/validate_submission.py --b1 outputs/b1/submission.csv --b2 outputs/b2/submission.csv
python -m agent.agent --status
```

运行目录应保存命令、配置、指标和日志；提交前检查行数、顺序、标签范围、Top10 长度、item 合法性和重复推荐。
