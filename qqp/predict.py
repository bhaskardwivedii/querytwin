"""Inference: score question pairs and find duplicates of a question in a corpus.

    python -m qqp.predict --q1 "How do I learn Python?" --q2 "What is the best way to learn Python?"
    python -m qqp.predict --interactive
    python -m qqp.predict --query "How can I lose weight?" --corpus questions.txt --top 5
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Sequence

import joblib
import numpy as np

from .pipeline import make_inputs


def _light(text: str) -> str:
    return re.sub(r"[\s?.!,]+", " ", str(text).lower()).strip()


class DuplicateDetector:
    def __init__(self, model_path: str = "outputs/model.joblib"):
        b = joblib.load(model_path)
        self.feat, self.scaler, self.model = b["featurizer"], b["scaler"], b["model"]
        self.kind, self.xgb_features, self.threshold = b["kind"], b["xgb_features"], b["threshold"]
        self.meta = b["meta"]

    def predict_proba(self, q1: Sequence[str], q2: Sequence[str]) -> np.ndarray:
        """Model probability, with one deterministic rule on top: pairs that are identical up to
        case, whitespace and sentence punctuation are duplicates by definition. (Deliberately
        lighter than the feature normaliser, which would merge e.g. "C++" and "C#".)"""
        q1, q2 = list(q1), list(q2)
        p = self._model_proba(q1, q2)
        same = np.array([_light(a) == _light(b) and _light(a) != "" for a, b in zip(q1, q2)])
        return np.where(same, 1.0, p)

    def _model_proba(self, q1: Sequence[str], q2: Sequence[str]) -> np.ndarray:
        Xs, Xd, _ = self.feat.transform(list(q1), list(q2))
        if self.kind == "ens":  # weighted soft vote of LR (sparse) and XGBoost (dense)
            w = self.model["lr_weight"]
            p_lr = self.model["lr"].predict_proba(make_inputs("lr", Xs, Xd, self.scaler, self.xgb_features))[:, 1]
            p_xgb = self.model["xgb"].predict_proba(make_inputs("xgb", Xs, Xd, self.scaler, self.xgb_features))[:, 1]
            return w * p_lr + (1 - w) * p_xgb
        X = make_inputs(self.kind, Xs, Xd, self.scaler, self.xgb_features)
        return self.model.predict_proba(X)[:, 1]

    def is_duplicate(self, q1: str, q2: str) -> dict:
        p = float(self.predict_proba([q1], [q2])[0])
        return {"duplicate": p >= self.threshold, "probability": round(p, 4)}

    # ------------------------------------------------------------------ #
    # Retrieval: find existing questions that duplicate a new one
    # ------------------------------------------------------------------ #
    def build_index(self, corpus: List[str]) -> None:
        """Vectorise a question bank once so later searches are fast."""
        self.corpus = list(corpus)
        _, stemmed = self.feat.prepare(self.corpus)
        self.corpus_matrix = self.feat.word_vec.transform(stemmed)

    def search(self, query: str, top: int = 5, candidates: int = 100):
        """Two-stage search over the indexed corpus: TF-IDF cosine retrieves a
        shortlist, then the trained classifier re-scores each (query, candidate) pair."""
        if not getattr(self, "corpus", None):
            raise RuntimeError("call build_index(corpus) first")
        _, s_q = self.feat.prepare([query])
        sims = (self.corpus_matrix @ self.feat.word_vec.transform(s_q).T).toarray().ravel()
        k = min(candidates, len(self.corpus))
        shortlist = np.argpartition(-sims, k - 1)[:k]
        shortlist = shortlist[sims[shortlist] > 0] if (sims[shortlist] > 0).any() else shortlist[:top]
        probs = self.predict_proba([query] * len(shortlist), [self.corpus[i] for i in shortlist])
        ranked = sorted(zip(shortlist, probs), key=lambda t: -t[1])[:top]
        return [{"question": self.corpus[i], "probability": round(float(p), 4),
                 "duplicate": bool(p >= self.threshold)} for i, p in ranked]

    def find_duplicates(self, query: str, corpus: List[str], top: int = 5, candidates: int = 100):
        self.build_index(corpus)
        return self.search(query, top, candidates)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="outputs/model.joblib")
    ap.add_argument("--q1"); ap.add_argument("--q2")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--query"); ap.add_argument("--corpus", help="text file, one question per line")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    det = DuplicateDetector(a.model)

    if a.q1 and a.q2:
        print(det.is_duplicate(a.q1, a.q2))
    elif a.query and a.corpus:
        corpus = [l.strip() for l in Path(a.corpus).read_text().splitlines() if l.strip()]
        for r in det.find_duplicates(a.query, corpus, a.top):
            print(f"{r['probability']:.3f}  {'DUP' if r['duplicate'] else '   '}  {r['question']}")
    elif a.interactive:
        print("Enter two questions (blank line to quit).")
        while True:
            q1 = input("Q1> ").strip()
            if not q1:
                break
            q2 = input("Q2> ").strip()
            print(det.is_duplicate(q1, q2))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
