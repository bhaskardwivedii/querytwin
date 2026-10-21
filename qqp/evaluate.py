"""Error analysis for a trained model on the exact held-out test split used in training.

    python -m qqp.evaluate                      # writes outputs/error_analysis.md

Reports metrics at the default 0.5 threshold and at the validation-tuned F1 threshold,
a breakdown by question length, and the model's most confident mistakes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from .data import load_dataset
from .predict import DuplicateDetector


def split_test(df, seed=42, val_size=0.1, test_size=0.1):
    idx = np.arange(len(df))
    _, tmp = train_test_split(idx, test_size=val_size + test_size, stratify=df.is_duplicate, random_state=seed)
    _, te = train_test_split(tmp, test_size=test_size / (val_size + test_size),
                             stratify=df.is_duplicate.iloc[tmp], random_state=seed)
    return df.iloc[te].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/model.joblib")
    ap.add_argument("--metrics", default="outputs/metrics.json")
    ap.add_argument("--data")
    ap.add_argument("--out", default="outputs/error_analysis.md")
    ap.add_argument("--examples", type=int, default=12)
    a = ap.parse_args()

    det = DuplicateDetector(a.model)
    summary = json.loads(Path(a.metrics).read_text())
    test = split_test(load_dataset("quora", a.data))
    y = test.is_duplicate.values
    p = det.predict_proba(test.question1.tolist(), test.question2.tolist())
    thr_f1 = summary["results"][det.kind]["val_best_f1_threshold"]

    def row(name, thr):
        pr = p >= thr
        return (f"| {name} ({thr:.2f}) | {accuracy_score(y, pr):.4f} | {precision_score(y, pr):.4f} "
                f"| {recall_score(y, pr):.4f} | {f1_score(y, pr):.4f} |")

    L = [f"# Error analysis — {det.kind} on {len(test):,} held-out test pairs\n",
         "## Operating points\n", "| Threshold | Accuracy | Precision | Recall | F1 |", "|---|---|---|---|---|",
         row("default", 0.5), row("F1-tuned on validation", thr_f1),
         "\nLowering the threshold trades precision for recall: use it when missing a duplicate costs more "
         "than a false merge (e.g. suggesting \"similar questions\" to a user).\n",
         "## Accuracy by question length (words in the longer question)\n",
         "| Length | Pairs | Accuracy |", "|---|---|---|"]
    words = np.maximum(test.question1.str.split().str.len(), test.question2.str.split().str.len()).values
    pred = p >= 0.5
    for lo, hi in [(0, 8), (8, 12), (12, 18), (18, 30), (30, 10_000)]:
        m = (words >= lo) & (words < hi)
        if m.any():
            label = f"{lo}–{hi - 1}" if hi < 10_000 else f"{lo}+"
            L.append(f"| {label} | {m.sum():,} | {(pred[m] == y[m]).mean():.4f} |")

    def show(title, mask, order):
        L.append(f"\n## {title}\n")
        L.append("| p(dup) | Question 1 | Question 2 |")
        L.append("|---|---|---|")
        idx = np.where(mask)[0]
        for i in idx[np.argsort(order[idx])][: a.examples]:
            q1 = test.question1[i].replace("|", "/")
            q2 = test.question2[i].replace("|", "/")
            L.append(f"| {p[i]:.3f} | {q1} | {q2} |")

    show("Most confident false positives (predicted duplicate, labelled different)", (pred == 1) & (y == 0), -p)
    show("Most confident false negatives (predicted different, labelled duplicate)", (pred == 0) & (y == 1), p)
    L.append("\nSome of these are label noise: Quora's labels are human judgements and the dataset "
             "authors note they are not guaranteed to be perfect.\n")
    Path(a.out).write_text("\n".join(L))
    print("\n".join(L[:12]))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
