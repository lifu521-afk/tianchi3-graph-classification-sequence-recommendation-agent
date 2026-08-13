"""Portable paths for the public repository.

The competition data is deliberately kept outside the repository. Set
`TIANCHI3_DATA_DIR` or pass explicit CLI paths when running a script.
"""
from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("TIANCHI3_DATA_DIR", REPO_ROOT / "data"))

TASK_NAMES = {
    "A_classification": ("A分类", "A_classification"),
    "A_recommendation": ("A推荐", "A_recommendation"),
    "B_classification": ("B分类", "B_classification"),
    "B_recommendation": ("B推荐", "B_recommendation"),
}


def task_dir(name: str) -> Path:
    candidates = TASK_NAMES.get(name, (name,))
    for candidate in candidates:
        path = DATA_ROOT / candidate
        if path.exists():
            return path
    return DATA_ROOT / candidates[0]
