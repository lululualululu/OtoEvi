#!/usr/bin/env python3
"""Recompute every released mean/sample SD from the three per-seed values."""

import json
import math
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "results"


def read(name):
    return json.loads((root / name).read_text())


count = 0
for summary in read("otoinsight_summary.json"):
    rows = [
        r
        for r in read("otoinsight_metrics_per_seed.json")
        if r["variant"] == summary["variant"] and r["control"] == summary["control"]
    ]
    assert sorted(r["seed"] for r in rows) == [7, 42, 3407]
    for key in summary:
        if not key.endswith("_mean"):
            continue
        metric = key[:-5]
        values = [r[metric] for r in rows]
        assert math.isclose(statistics.mean(values), summary[key], abs_tol=1e-12)
        assert math.isclose(
            statistics.stdev(values), summary[metric + "_sd"], abs_tol=1e-12
        )
        count += 1
for summary in read("external_metrics_summary.json"):
    rows = [
        r
        for r in read("external_metrics_per_seed.json")
        if all(
            r[k] == summary[k]
            for k in ("dataset", "variant", "reference", "control", "metric")
        )
    ]
    assert sorted(r["seed"] for r in rows) == [7, 42, 3407]
    values = [r["value"] for r in rows]
    assert math.isclose(statistics.mean(values), summary["mean"], abs_tol=1e-12)
    assert math.isclose(statistics.stdev(values), summary["sample_sd"], abs_tol=1e-12)
    count += 1
print("PASS:", count, "aggregate metrics match all three seeds")
