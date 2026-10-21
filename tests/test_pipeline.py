"""Run with:  pytest -q"""
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qqp.features import DENSE_NAMES, PairFeaturizer  # noqa: E402
from qqp.preprocess import negation_count, normalize, stem_text  # noqa: E402


# --------------------------------------------------------------- preprocessing
@pytest.mark.parametrize("raw, expected", [
    ("What's the BEST way?", "what is the best way"),
    ("I can't do it", "i can not do it"),
    ("Is $100 worth it?", "is dollar 100 worth it"),
    ("Earn 50k per month", "earn 50000 per month"),
    ("Visit https://quora.com now", "visit url now"),
    ("1,000,000 people", "1000000 people"),
    ("<b>bold</b> &amp; text", "bold and text"),
])
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_normalize_handles_non_strings():
    assert normalize(None) == "" and normalize(float("nan")) == ""


def test_negations_survive():
    n = normalize("Why don't people like it?")
    assert negation_count(n) == 1 and "not" in stem_text(n).split()


# -------------------------------------------------------------------- features
@pytest.fixture(scope="module")
def featurizer():
    qs = ["How do I learn Python?", "What is the best way to learn Python?",
          "How do I cook rice?", "What is machine learning?", "How can I learn Java?",
          "What is deep learning?", "How do I cook pasta?", "Best way to learn Java?"] * 5
    return PairFeaturizer(max_word_features=500, max_char_features=500, min_df=1, n_svd=4).fit(qs, qs[::-1])


def test_feature_shapes(featurizer):
    Xs, Xd, names = featurizer.transform(["How do I learn Python?"] * 2,
                                         ["What is the best way to learn Python?", "How do I cook rice?"])
    assert Xd.shape == (2, len(DENSE_NAMES) + 1 + 2 * 4) == (2, len(names))
    assert Xs.shape[0] == 2 and np.isfinite(Xd).all()


def test_identical_pair_is_maximally_similar(featurizer):
    _, Xd, names = featurizer.transform(["How do I learn Python?"], ["How do I learn Python?"])
    f = dict(zip(names, Xd[0]))
    assert f["jaccard"] == 1 and f["fuzz_ratio"] == 100
    assert f["cos_word"] == pytest.approx(1, abs=1e-5) and f["cos_lsa"] == pytest.approx(1, abs=1e-4)


def test_similar_beats_unrelated(featurizer):
    _, Xd, names = featurizer.transform(["How do I learn Python?"] * 2,
                                        ["What is the best way to learn Python?", "How do I cook rice?"])
    for name in ("content_jaccard", "cos_char", "fuzz_token_set"):
        i = names.index(name)
        assert Xd[0, i] > Xd[1, i], name


def test_number_mismatch_feature(featurizer):
    _, Xd, names = featurizer.transform(["Is 2020 a leap year?"], ["Is 2021 a leap year?"])
    assert dict(zip(names, Xd[0]))["num_equal"] == 0


def test_antonym_conflict_feature(featurizer):
    _, Xd, names = featurizer.transform(["How can I lose weight quickly?", "How can I lose weight quickly?"],
                                        ["How could I gain weight quickly?", "How can I shed weight quickly?"])
    i = names.index("antonym_conflict")
    assert Xd[0, i] == 1 and Xd[1, i] == 0


def test_word_order_feature(featurizer):
    _, Xd, names = featurizer.transform(["data scientist after engineering"] * 2,
                                        ["data scientist after engineering", "engineering after data scientist"])
    i = names.index("lcs_tokens")
    assert Xd[0, i] == 1 and Xd[1, i] < 1


def test_augmentation_only_makes_labelled_negatives():
    import pandas as pd
    from qqp.data import augment_antonym_negatives
    train = pd.DataFrame({"question1": ["What are the best movies of 2015?", "How do I sell a car?"],
                          "question2": ["How can I gain muscle?", "Why is the sky blue?"],
                          "is_duplicate": [0, 0]})
    aug = augment_antonym_negatives(train, 10, seed=0)
    assert len(aug) == 3 and (aug.is_duplicate == 0).all()
    pairs = {frozenset(p) for p in zip(aug.question1, aug.question2)}
    assert frozenset({"What are the best movies of 2015?", "What are the worst movies of 2015?"}) in pairs
    assert frozenset({"How can I gain muscle?", "How can I lose muscle?"}) in pairs


# ------------------------------------------------------------------ end to end
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    cmd = [sys.executable, "-m", "qqp.train", "--dataset", "demo", "--n-demo", "3000",
           "--rf-trees", "20", "--xgb-rounds", "60", "--n-svd", "16", "--max-features", "3000",
           "--outdir", str(out)]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    return out


def test_training_artifacts(trained):
    for f in ["model.joblib", "metrics.json", "results.md", "confusion_matrices.png",
              "model_comparison.png", "roc_curves.png"]:
        assert (trained / f).exists(), f


def test_detector_roundtrip(trained):
    from qqp.predict import DuplicateDetector
    det = DuplicateDetector(str(trained / "model.joblib"))
    r = det.is_duplicate("How can I learn python quickly?", "What is the fastest way to learn python?")
    assert set(r) == {"duplicate", "probability"} and 0 <= r["probability"] <= 1
    exact = det.is_duplicate("How do I learn Python?", "how do i learn python")
    assert exact == {"duplicate": True, "probability": 1.0}
    det.build_index(["What is the fastest way to learn python?", "Is chess difficult to learn?",
                     "How much does a nursing professional earn?"])
    hits = det.search("How can I learn python quickly?", top=2)
    assert len(hits) == 2 and hits[0]["question"].startswith("What is the fastest way to learn python")


def test_api(trained, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("QQP_MODEL", str(trained / "model.joblib"))
    monkeypatch.setenv("QQP_CORPUS_N", "0")
    import qqp.api as api
    api = importlib.reload(api)
    with TestClient(api.app) as c:
        assert c.get("/health").json()["status"] == "ok"
        assert "Duplicate Question Detector" in c.get("/").text
        r = c.post("/predict", json={"question1": "How to learn python?", "question2": "Best way to learn python?"})
        assert r.status_code == 200 and 0 <= r.json()["probability"] <= 1
        r = c.post("/predict/batch", json={"pairs": [["a b c", "a b c"], ["x y", "p q"]]})
        assert len(r.json()["results"]) == 2
        assert c.post("/predict", json={"question1": "", "question2": "x"}).status_code == 422
        assert c.post("/search", json={"question": "anything"}).status_code == 503
