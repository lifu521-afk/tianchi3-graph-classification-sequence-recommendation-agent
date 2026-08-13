# Serial Long-Horizon Experiment Agent

The competition is not only asking for the best graph model. It asks for a
controller that can repeatedly:

1. inspect sparse feedback from previous trials;
2. form a falsifiable hypothesis for the next trial;
3. run exactly one experiment within a finite budget;
4. preserve the result, failure reason, and local validation protocol;
5. decide whether to explore, exploit, stop, or promote a candidate.

The controller is deliberately separate from prediction code. Prediction
scripts are plugins; the Agent owns the research loop.

## Guarantees

- Serial execution with an exclusive lock. No parallel trials are allowed.
- Round and wall-clock budgets, per-experiment timeouts, and early stopping.
- Persistent append-only `memory.jsonl` plus resumable `state.json`.
- Explicit lifecycle status distinguishes running, ready, budget exhaustion,
  and waiting for a genuinely new experiment.
- Official leaderboard feedback is stored separately in
  `official_feedback.json`. Local validation is never called an official score.
- Selection uses official target gap, experiment novelty, transfer hints,
  hypothesis kind, cost, and prior failures.
- Protocol-aware comparison. A five-fold OOF score is not silently compared
  with a single holdout score.
- A failed deployment gate rejects a candidate even if its mean score rises.
- Promotion validates ZIP members, backs up the current package, preserves the
  unchanged task byte-for-byte, and stages replacements atomically.

## Competition interpretation

The classification task is a sparse-supervision node classification problem.
The recommendation task is a sparse-history user-item link prediction problem.
The intended research contribution is the decision process under sparse
feedback, not a claim that one GNN family is universally best. The Agent keeps
negative evidence such as unstable GNN folds and does not repeatedly rerun a
failed global replacement without a new hypothesis.

## Commands

```powershell
python -m agent.agent --status
python -m agent.agent --dry-run --max-rounds 2
python -m agent.agent --max-rounds 1
python -m agent.agent --record-official A1_accuracy=0.87 A2_ndcg_at_10=0.512 --submitted-at "2026-07-20 12:00:00"
```

`--status` reports the global round budget, currently eligible experiment IDs,
the precise stop reason, and the action required to resume. `--max-rounds`
limits only the current invocation and never resets the global round budget.
When the registered pool is exhausted, adding a new experiment with an unused
trial budget automatically changes the computed lifecycle from
`awaiting_experiments` back to `ready`.

The experiment registry and policy are in `agent/config.json`.

## Optional API planner

The API model is a planning assistant, not a predictor. It may select exactly
one experiment from the registered catalog, explain the hypothesis, and name
the expected failure mode. It cannot create commands, modify scripts, invent
metrics, or output hidden A1/A2 labels. Artifact promotion remains controlled
by deterministic OOF and byte-preservation checks.

Do not paste an API key into chat, source files, JSON, logs, or command
history. Configure it only in the current PowerShell environment:

```powershell
$env:TIANCHI_AGENT_API_KEY = "your-key"
$env:TIANCHI_AGENT_BASE_URL = "https://your-openai-compatible-endpoint/v1"
$env:TIANCHI_AGENT_MODEL = "your-model"
python -m agent.agent --status
python -m agent.agent --max-rounds 1
```

The Agent reads the key at request time and never writes it to disk. If any
variable is missing or the endpoint fails, `planner=auto` falls back to the
deterministic serial planner. The API endpoint must support the
OpenAI-compatible `/chat/completions` JSON format.
