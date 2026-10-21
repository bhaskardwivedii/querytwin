"""Train and benchmark Logistic Regression, Random Forest and XGBoost on question pairs.

Examples
--------
    # real data (404k pairs) - download once with --dataset glue, or point at a local file
    python -m qqp.train --dataset glue
    python -m qqp.train --dataset quora  --data data/quora_duplicate_questions.tsv
    python -m qqp.train --dataset kaggle --data data/train.csv

    # offline smoke test on a synthetic stand-in dataset
    python -m qqp.train --dataset demo
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,  # noqa: E402
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from .data import augment_antonym_negatives, load_dataset  # noqa: E402
from .features import PairFeaturizer  # noqa: E402
from .pipeline import make_inputs  # noqa: E402

NAMES = {"lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost",
         "ens": "Ensemble (LR + XGBoost)"}
BLUE, ORANGE, GREY, GREEN = "#2a6fdb", "#e07a1f", "#6b7280", "#1f8a4c"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def metrics(y, proba, thr: float = 0.5) -> dict:
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
        "recall": recall_score(y, pred), "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba), "log_loss": log_loss(y, proba),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]], "threshold": thr,
    }


def best_f1_threshold(y, proba) -> float:
    grid = np.linspace(0.2, 0.8, 61)
    return float(grid[int(np.argmax([f1_score(y, proba >= t) for t in grid]))])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["demo", "glue", "quora", "kaggle"], default="quora")
    ap.add_argument("--data", help="path to quora_duplicate_questions.tsv or Kaggle train.csv")
    ap.add_argument("--n-demo", type=int, default=40_000)
    ap.add_argument("--sample", type=int, default=0, help="subsample N pairs (0 = use all)")
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--max-features", type=int, default=20_000, help="word TF-IDF vocabulary size")
    ap.add_argument("--models", default="lr,rf,xgb")
    ap.add_argument("--xgb-features", choices=["dense", "dense+tfidf"], default="dense")
    ap.add_argument("--n-svd", type=int, default=100, help="LSA dimensions (0 disables)")
    ap.add_argument("--rf-trees", type=int, default=200)
    ap.add_argument("--xgb-rounds", type=int, default=6000)
    ap.add_argument("--xgb-lr", type=float, default=0.08)
    ap.add_argument("--lr-c", type=float, nargs="+", default=[4.0, 12.0])
    ap.add_argument("--lr-iter", type=int, default=2500)
    ap.add_argument("--no-ensemble", action="store_true")
    ap.add_argument("--augment", type=int, default=10_000,
                    help="antonym-swap hard negatives added to the training split (0 disables)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------------------------ data
    df = load_dataset(args.dataset, args.data, args.n_demo)
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=args.seed).reset_index(drop=True)
    log(f"dataset={args.dataset}  pairs={len(df):,}  duplicate rate={df.is_duplicate.mean():.3f}")

    idx = np.arange(len(df))
    tr_idx, tmp_idx = train_test_split(
        idx, test_size=args.val_size + args.test_size, stratify=df.is_duplicate, random_state=args.seed)
    va_idx, te_idx = train_test_split(
        tmp_idx, test_size=args.test_size / (args.val_size + args.test_size),
        stratify=df.is_duplicate.iloc[tmp_idx], random_state=args.seed)
    parts = {k: df.iloc[v] for k, v in {"train": tr_idx, "val": va_idx, "test": te_idx}.items()}
    if args.augment and args.dataset != "demo":
        aug = augment_antonym_negatives(parts["train"], args.augment, args.seed)
        parts["train"] = (pd.concat([parts["train"], aug], ignore_index=True)
                          .sample(frac=1, random_state=args.seed).reset_index(drop=True))
        log(f"added {len(aug):,} contrastive (antonym-swap) negatives to the training split only")
    y = {k: v.is_duplicate.values for k, v in parts.items()}
    log("split sizes: " + ", ".join(f"{k}={len(v):,}" for k, v in parts.items()))

    # -------------------------------------------------------------- features
    feat = PairFeaturizer(max_word_features=args.max_features, max_char_features=args.max_features,
                          n_svd=args.n_svd, seed=args.seed)
    log("fitting TF-IDF vectorisers on training questions ...")
    feat.fit(parts["train"].question1.tolist(), parts["train"].question2.tolist())
    X = {}
    for k, v in parts.items():
        log(f"featurising {k} ...")
        Xs, Xd, _ = feat.transform(v.question1.tolist(), v.question2.tolist())
        X[k] = (Xs, Xd)
    scaler = StandardScaler().fit(X["train"][1])
    log(f"sparse width={X['train'][0].shape[1]:,}  dense features={X['train'][1].shape[1]}")

    def inputs(kind, split):
        return make_inputs(kind, X[split][0], X[split][1], scaler, args.xgb_features)

    # ---------------------------------------------------------------- models
    results, fitted, probas, val_probas = {}, {}, {}, {}
    for kind in args.models.split(","):
        name = NAMES[kind]
        t1 = time.time()
        extra = {}
        log(f"training {name} ...")
        Xtr, Xva = inputs(kind, "train"), inputs(kind, "val")

        if kind == "lr":
            best, best_ll = None, 9e9
            for C in args.lr_c:  # small grid, picked on validation log-loss
                m = LogisticRegression(C=C, max_iter=args.lr_iter, solver="lbfgs").fit(Xtr, y["train"])
                ll = log_loss(y["val"], m.predict_proba(Xva)[:, 1])
                log(f"   C={C:<4} val log-loss={ll:.4f}")
                if ll < best_ll:
                    best, best_ll = m, ll
            model = best
        elif kind == "rf":
            model = RandomForestClassifier(
                n_estimators=args.rf_trees, min_samples_leaf=3, max_features="sqrt", max_samples=0.5,
                n_jobs=-1, random_state=args.seed).fit(Xtr, y["train"])
        else:
            model = XGBClassifier(
                n_estimators=args.xgb_rounds, learning_rate=args.xgb_lr, max_depth=8,
                min_child_weight=3, subsample=0.85, colsample_bytree=0.5, reg_lambda=2.0,
                tree_method="hist", eval_metric="logloss", early_stopping_rounds=50,
                n_jobs=-1, random_state=args.seed, verbosity=0)
            model.fit(Xtr, y["train"], eval_set=[(Xva, y["val"])], verbose=False)
            log(f"   best iteration={model.best_iteration}")
            extra = {"best_iteration": int(model.best_iteration)}

        pv = model.predict_proba(Xva)[:, 1]
        pt = model.predict_proba(inputs(kind, "test"))[:, 1]
        probas[kind], val_probas[kind] = pt, pv
        results[kind] = {
            "name": name, "train_seconds": round(time.time() - t1, 1),
            "val": metrics(y["val"], pv), "test": metrics(y["test"], pt),
            "val_best_f1_threshold": best_f1_threshold(y["val"], pv), **extra,
        }
        fitted[kind] = model
        r = results[kind]["test"]
        log(f"   {name}: test acc={r['accuracy']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} "
            f"F1={r['f1']:.4f} AUC={r['roc_auc']:.4f}")

    # ------------------------------------------------------------- ensemble
    # LR sees sparse TF-IDF pair vectors, XGBoost sees dense similarity + LSA features,
    # so their errors are only partly correlated. Blend weight is tuned on validation only.
    ens_w = None
    if not args.no_ensemble and {"lr", "xgb"} <= set(fitted):
        grid = np.round(np.linspace(0, 1, 21), 2)
        ens_w = float(grid[int(np.argmin([log_loss(y["val"], w * val_probas["lr"] + (1 - w) * val_probas["xgb"])
                                           for w in grid]))])
        pv = ens_w * val_probas["lr"] + (1 - ens_w) * val_probas["xgb"]
        pt = ens_w * probas["lr"] + (1 - ens_w) * probas["xgb"]
        probas["ens"] = pt
        results["ens"] = {"name": NAMES["ens"], "lr_weight": ens_w,
                          "train_seconds": round(results["lr"]["train_seconds"] + results["xgb"]["train_seconds"], 1),
                          "val": metrics(y["val"], pv), "test": metrics(y["test"], pt),
                          "val_best_f1_threshold": best_f1_threshold(y["val"], pv)}
        fitted["ens"] = {"lr": fitted["lr"], "xgb": fitted["xgb"], "lr_weight": ens_w}
        r = results["ens"]["test"]
        log(f"   Ensemble (LR weight {ens_w}): test acc={r['accuracy']:.4f} P={r['precision']:.4f} "
            f"R={r['recall']:.4f} F1={r['f1']:.4f} AUC={r['roc_auc']:.4f}")

    best_kind = max(results, key=lambda k: results[k]["val"]["f1"])  # chosen on validation only
    log(f"best model by validation F1: {NAMES[best_kind]}")

    # ----------------------------------------------------------------- save
    bundle = {"featurizer": feat, "scaler": scaler, "model": fitted[best_kind], "kind": best_kind,
              "xgb_features": args.xgb_features, "threshold": 0.5,
              "meta": {"dataset": args.dataset, "n_pairs": len(df), "dense_names": feat.feature_names,
                       "test_metrics": results[best_kind]["test"]}}
    joblib.dump(bundle, out / "model.joblib", compress=3)

    summary = {"dataset": args.dataset, "n_pairs": len(df), "split": {k: len(v) for k, v in parts.items()},
               "duplicate_rate": float(df.is_duplicate.mean()), "best_model": best_kind,
               "results": results, "total_seconds": round(time.time() - t0, 1)}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))

    # -------------------------------------------------------- report: table
    lines = ["| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time |",
             "|---|---|---|---|---|---|---|"]
    for k, r in results.items():
        t = r["test"]
        star = " *" if k == best_kind else ""
        lines.append(f"| {r['name']}{star} | {t['accuracy']:.4f} | {t['precision']:.4f} | {t['recall']:.4f} "
                     f"| {t['f1']:.4f} | {t['roc_auc']:.4f} | {r['train_seconds']}s |")
    note = ("\n\\* best on validation F1. Metrics are on the held-out test split at threshold 0.5.\n"
            + ("\n**Dataset is the synthetic stand-in; these numbers do NOT reflect real Quora data.**\n"
               if args.dataset == "demo" else ""))
    (out / "results.md").write_text("\n".join(lines) + note)

    # ---------------------------------------------------- report: figures
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.9))
    axes = np.atleast_1d(axes)
    for ax, (k, r) in zip(axes, results.items()):
        cm = np.array(r["test"]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "#111827", fontsize=11)
        ax.set_xticks([0, 1], ["Not dup", "Dup"]); ax.set_yticks([0, 1], ["Not dup", "Dup"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{r['name']}\nacc {r['test']['accuracy']:.3f}", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "confusion_matrices.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4))
    keys = ["accuracy", "precision", "recall", "f1"]
    w = 0.8 / len(results)
    palette = [BLUE, ORANGE, GREY, GREEN]
    for n, (k, r) in enumerate(results.items()):
        vals = [r["test"][m] for m in keys]
        bars = ax.bar(np.arange(len(keys)) + n * w, vals, w, label=r["name"], color=palette[n % 4])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(keys)) + w * (len(results) - 1) / 2, [m.capitalize() if m != "f1" else "F1" for m in keys])
    lo = min(r["test"][m] for r in results.values() for m in keys)
    ax.set_ylim(max(0, lo - 0.08), 1.0); ax.set_title("Test-set comparison")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "model_comparison.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.4))
    for n, (k, r) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y["test"], probas[k])
        ax.plot(fpr, tpr, color=palette[n % 4], lw=2, label=f"{r['name']} (AUC {r['test']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#d1d5db", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC curves (test)")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "roc_curves.png", dpi=150); plt.close(fig)

    if "xgb" in fitted:
        imp = fitted["xgb"].get_booster().get_score(importance_type="gain")
        names = feat.feature_names if args.xgb_features == "dense" else None
        if names:
            top = sorted(imp.items(), key=lambda t: -t[1])[:15][::-1]
            fig, ax = plt.subplots(figsize=(6.5, 4.6))
            ax.barh([names[int(f[1:])] for f, _ in top], [v for _, v in top], color=ORANGE)
            ax.set_title("XGBoost: top features by gain"); ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout(); fig.savefig(out / "xgb_feature_importance.png", dpi=150); plt.close(fig)

    if "rf" in fitted:
        imp = fitted["rf"].feature_importances_
        order = np.argsort(imp)[-15:]
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        ax.barh([feat.feature_names[i] for i in order], imp[order], color=BLUE)
        ax.set_title("Random Forest: top features"); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(out / "feature_importance.png", dpi=150); plt.close(fig)

    log(f"done in {time.time() - t0:.0f}s. artefacts in {out}/")
    print("\n" + (out / "results.md").read_text())


if __name__ == "__main__":
    main()
