"""REST API + web UI for the duplicate-question detector.

    uvicorn qqp.api:app --port 8000          # then open http://localhost:8000

Endpoints
---------
GET  /                 web demo
GET  /health           model + index status
POST /predict          {"question1": "...", "question2": "..."}
POST /predict/batch    {"pairs": [["q1", "q2"], ...]}
POST /search           {"question": "...", "top_k": 5}   duplicates from the question bank

Environment variables
---------------------
QQP_MODEL      path to model bundle            (default outputs/model.joblib)
QQP_CORPUS     question bank for /search: Quora TSV/CSV or a .txt (one question per line)
               (default: data/quora_duplicate_questions.tsv, else data/question_bank.txt)
QQP_CORPUS_N   how many unique questions to index (default 50000)
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .predict import DuplicateDetector

MODEL_PATH = os.getenv("QQP_MODEL", "outputs/model.joblib")
CORPUS_PATH = os.getenv("QQP_CORPUS") or next(
    (p for p in ("data/quora_duplicate_questions.tsv", "data/question_bank.txt") if Path(p).exists()),
    "data/question_bank.txt")
CORPUS_N = int(os.getenv("QQP_CORPUS_N", "50000"))
STATIC = Path(__file__).parent / "static" / "index.html"

state: dict = {}


@asynccontextmanager
async def lifespan(_app):
    if Path(MODEL_PATH).exists():  # warm up: load model + build search index once
        detector()
    yield


app = FastAPI(title="Duplicate Question Detector", version="1.0", lifespan=lifespan)


class Pair(BaseModel):
    question1: str = Field(..., min_length=1, max_length=1000)
    question2: str = Field(..., min_length=1, max_length=1000)


class Batch(BaseModel):
    pairs: List[List[str]] = Field(..., min_length=1, max_length=1000)


class Query(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(5, ge=1, le=50)


def _load_corpus(path: str, n: int) -> List[str]:
    if path.endswith(".txt"):  # one question per line (small file shipped for deployment)
        lines = [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        return lines[:n]

    import pandas as pd

    sep = "\t" if path.endswith(".tsv") else ","
    df = pd.read_csv(path, sep=sep, usecols=["question1", "question2"])
    qs = pd.concat([df.question1, df.question2]).dropna().astype(str).str.strip()
    qs = qs[qs.str.len() > 10].drop_duplicates()
    return qs.sample(min(n, len(qs)), random_state=0).tolist()


def detector() -> DuplicateDetector:
    if "det" not in state:
        t = time.time()
        det = DuplicateDetector(MODEL_PATH)
        if Path(CORPUS_PATH).exists() and CORPUS_N > 0:
            det.build_index(_load_corpus(CORPUS_PATH, CORPUS_N))
        state["det"], state["load_seconds"] = det, round(time.time() - t, 1)
    return state["det"]


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return STATIC.read_text()


@app.get("/health")
def health() -> dict:
    det = detector()
    m = det.meta.get("test_metrics", {})
    return {"status": "ok", "model": det.kind, "threshold": det.threshold,
            "trained_on_pairs": det.meta.get("n_pairs"),
            "test_accuracy": m.get("accuracy"), "test_f1": m.get("f1"),
            "indexed_questions": len(getattr(det, "corpus", []) or []),
            "load_seconds": state.get("load_seconds")}


@app.post("/predict")
def predict(p: Pair) -> dict:
    return {"question1": p.question1, "question2": p.question2,
            **detector().is_duplicate(p.question1, p.question2)}


@app.post("/predict/batch")
def predict_batch(b: Batch) -> dict:
    if any(len(pair) != 2 for pair in b.pairs):
        raise HTTPException(422, "each pair must be [question1, question2]")
    det = detector()
    probs = det.predict_proba([p[0] for p in b.pairs], [p[1] for p in b.pairs])
    return {"results": [{"probability": round(float(pr), 4), "duplicate": bool(pr >= det.threshold)}
                        for pr in probs]}


@app.post("/search")
def search(q: Query) -> dict:
    det = detector()
    if not getattr(det, "corpus", None):
        raise HTTPException(503, "no question bank indexed; set QQP_CORPUS")
    t = time.time()
    hits = det.search(q.question, top=q.top_k)
    return {"question": q.question, "results": hits, "ms": round(1000 * (time.time() - t), 1)}
