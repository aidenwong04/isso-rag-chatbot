"""
score - compare one run's report.json with the dev answer key (C2).

    env/bin/python agent/src/score.py agent/runs/dev-full/report.json agent/labels/dev.worksheet.json

A separate script, never imported by the agent: the agent process never opens
the answer key (A7).

Prints the counts of label 1, label 0 and null (any row that is not
classified), then precision and recall against the key, computed over the
classified rows only, so a budget cutoff lowers coverage, not precision or
recall. Then every classified row whose label disagrees with the key, for
reading the misses.

Stops with an error when the report and the key do not describe the same
alerts: a different corpus version or k, or an alert id that only one of them
has. Scoring such a pair would compare labels for different questions.
"""

import argparse
import json
import sys
from pathlib import Path


def load(path):
    with open(path) as f:
        return json.load(f)


def check_same_alerts(report, key):
    scan = report["scan"]
    if scan.get("corpus_version") != key["corpus_version"] or scan.get("k") != key["k"]:
        sys.exit(
            f"report is corpus {scan.get('corpus_version')}, k={scan.get('k')}; "
            f"key is corpus {key['corpus_version']}, k={key['k']}"
        )
    report_ids = {row["alert_id"] for row in report["alerts"]}
    key_ids = {entry["alert_id"] for entry in key["alerts"]}
    if report_ids - key_ids:
        sys.exit(f"report rows with no label in the key: {sorted(report_ids - key_ids)}")
    if key_ids - report_ids:
        sys.exit(f"key entries with no row in the report: {sorted(key_ids - report_ids)}")


def score(report, key):
    truth = {entry["alert_id"]: entry["label"] for entry in key["alerts"]}
    counts = {"1": 0, "0": 0, "null": 0}
    tp = fp = fn = tn = 0
    misses = []
    for row in report["alerts"]:
        label = row["label"] if row["status"] == "classified" else None
        counts["null" if label is None else str(label)] += 1
        if label is None:
            continue
        expected = truth[row["alert_id"]]
        if label == 1 and expected == 1:
            tp += 1
        elif label == 1:
            fp += 1
        elif expected == 1:
            fn += 1
        else:
            tn += 1
        if label != expected:
            misses.append((row, expected))

    classified = tp + fp + fn + tn
    return {
        "counts": counts,
        "classified": classified,
        "total": len(report["alerts"]),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        # None when undefined: no row labeled 1, or no key positive classified.
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "accuracy": (tp + tn) / classified if classified else None,
        "misses": misses,
    }


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("report", type=Path, help="a run's report.json")
    parser.add_argument("key", type=Path, help="the answer key, e.g. agent/labels/dev.worksheet.json")
    args = parser.parse_args()

    report, key = load(args.report), load(args.key)
    check_same_alerts(report, key)
    result = score(report, key)

    c, m = result["counts"], result["confusion"]
    print(f"label 1: {c['1']}   label 0: {c['0']}   null: {c['null']}   ({result['classified']} of {result['total']} classified)")
    print(f"tp {m['tp']}  fp {m['fp']}  fn {m['fn']}  tn {m['tn']}")
    print(f"precision {fmt(result['precision'])}   recall {fmt(result['recall'])}   accuracy {fmt(result['accuracy'])}")

    if result["misses"]:
        print("\nmisses:")
        for row, expected in result["misses"]:
            print(f"  {row['alert_id']}  agent {row['label']}, key {expected}")
            print(f"    {row['rationale']}")


if __name__ == "__main__":
    main()
