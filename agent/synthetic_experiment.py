"""Small, deterministic experiment used to smoke-test the public Agent.

It deliberately uses no competition data. The payload has the same fields
consumed by the evaluator, so CI can exercise planning, execution, metrics,
and multi-split stability checks in under a second.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "synthetic_agent_smoke",
        "metric": 0.75,
        "baseline": 0.70,
        "fold_lifts": [0.04, 0.05, 0.06, 0.05, 0.05],
        "note": "Synthetic only; not a competition score.",
    }
    (out / "synthetic_agent_result.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
