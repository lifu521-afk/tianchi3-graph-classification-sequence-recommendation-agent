# 复现说明

## 环境

推荐使用 conda 环境 `torch5070`，或安装 `requirements.txt` 中的依赖。GPU 不是所有步骤的硬性要求；B1 基线和部分规则推荐可以在 CPU 运行，MLP/GRU 全量训练建议使用 CUDA GPU。

```powershell
conda activate torch5070
pip install -r requirements.txt
```

## B1 Agent

```powershell
python src/autonomous_classifier_agent.py `
  --data data/B分类/B1.npz `
  --template data/B分类/sample_submission.csv `
  --output-dir outputs/b1_agent `
  --task-id B1 `
  --budget-minutes 110
```

## B2 Agent

```powershell
python src/autonomous_recommender_agent.py `
  --data-dir data/B推荐 `
  --output-dir outputs/b2_agent `
  --task-id B2 `
  --budget-minutes 110 `
  --validation-repeats 3
```

## 测试

```powershell
python -m unittest discover -s tests -v
```

正式提交前请先在本地放置比赛数据，并运行 `src/validate_submission.py`。

