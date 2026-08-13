"""Validate B1/B2 CSV outputs without requiring leaderboard access."""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import pandas as pd


def validate_b1(path: Path, template: Path, n_classes: int = 8) -> list[str]:
    errors: list[str] = []
    frame = pd.read_csv(path)
    expected = pd.read_csv(template)
    if list(frame.columns) != ["test_idx", "label"]:
        errors.append("B1 columns must be test_idx,label")
    if len(frame) != len(expected):
        errors.append(f"B1 row count {len(frame)} != template {len(expected)}")
    if not frame["test_idx"].equals(expected["test_idx"]):
        errors.append("B1 test_idx order differs from template")
    if frame["label"].isna().any() or not frame["label"].between(0, n_classes - 1).all():
        errors.append("B1 labels must be integers in the legal class range")
    return errors


def validate_b2(path: Path, template: Path, items: Path, topk: int = 10) -> list[str]:
    errors: list[str] = []
    frame = pd.read_csv(path)
    expected = pd.read_csv(template)
    catalog = set(pd.read_csv(items)["iid"].astype(str))
    if list(frame.columns) != ["uid", "prediction"]:
        errors.append("B2 columns must be uid,prediction")
    if len(frame) != len(expected):
        errors.append(f"B2 row count {len(frame)} != template {len(expected)}")
    if not frame["uid"].equals(expected["uid"]):
        errors.append("B2 uid order differs from template")
    for row, value in enumerate(frame["prediction"].fillna("")):
        values = [x.strip() for x in str(value).split(",") if x.strip()]
        if len(values) != topk:
            errors.append(f"B2 row {row} does not contain exactly {topk} items")
        if len(values) != len(set(values)):
            errors.append(f"B2 row {row} contains duplicate items")
        if any(x not in catalog for x in values):
            errors.append(f"B2 row {row} contains an item outside item.csv")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b1")
    parser.add_argument("--b1-template")
    parser.add_argument("--b2")
    parser.add_argument("--b2-template")
    parser.add_argument("--items")
    parser.add_argument("--zip")
    args = parser.parse_args()
    errors: list[str] = []
    if args.b1:
        errors.extend(validate_b1(Path(args.b1), Path(args.b1_template)))
    if args.b2:
        errors.extend(validate_b2(Path(args.b2), Path(args.b2_template), Path(args.items)))
    if args.zip:
        with zipfile.ZipFile(args.zip) as archive:
            bad = archive.testzip()
            if bad:
                errors.append(f"corrupt zip entry: {bad}")
    if errors:
        print("FAILED")
        print("\n".join(f"- {e}" for e in errors))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

