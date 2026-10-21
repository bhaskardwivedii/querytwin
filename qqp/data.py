"""Dataset loading.

Supported sources
-----------------
* ``quora``  - Quora's original ``quora_duplicate_questions.tsv`` (404,290 pairs)
* ``kaggle`` - Kaggle "Quora Question Pairs" ``train.csv`` (404,290 pairs)
* ``glue``   - GLUE/QQP via Hugging Face ``datasets`` (train + validation, 404,290 labelled pairs)
* ``demo``   - small synthetic stand-in for smoke-testing the pipeline offline.
               Its metrics say nothing about real Quora performance.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

COLS = ["question1", "question2", "is_duplicate"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[COLS].copy()
    df["question1"] = df["question1"].fillna("").astype(str)
    df["question2"] = df["question2"].fillna("").astype(str)
    df["is_duplicate"] = df["is_duplicate"].astype(int)
    df = df[(df.question1.str.strip() != "") & (df.question2.str.strip() != "")]
    return df.reset_index(drop=True)


def load_file(path: str | Path) -> pd.DataFrame:
    """Load the Quora TSV or Kaggle CSV (same column names)."""
    path = Path(path)
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep)
    if "is_duplicate" not in df.columns:
        raise ValueError(f"{path} has no 'is_duplicate' column; is this the labelled train file?")
    return _clean(df)


def load_glue_qqp() -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise SystemExit("pip install datasets   (needed for --dataset glue)") from e
    ds = load_dataset("glue", "qqp")
    frames = [ds[s].to_pandas() for s in ("train", "validation")]
    df = pd.concat(frames).rename(columns={"label": "is_duplicate"})
    return _clean(df)


# --------------------------------------------------------------------------- #
# Synthetic stand-in dataset (offline smoke test only)
# --------------------------------------------------------------------------- #
_INTENTS = {
    "learn": [
        "How can I learn {t} quickly?", "What is the fastest way to learn {t}?",
        "How do I get started with {t}?", "What are the best resources to study {t}?",
        "Where should a beginner begin with {t}?", "Which is the best way to master {t}?",
    ],
    "career": [
        "What is the career scope of {t}?", "Is {t} a good career choice?",
        "What jobs can I get after studying {t}?", "Does {t} have a good future?",
        "Should I build my career in {t}?", "How are the job prospects for {t}?",
    ],
    "difficulty": [
        "Is {t} difficult to learn?", "How hard is it to learn {t}?",
        "Is {t} hard for beginners?", "How tough is {t} for a newcomer?",
        "Can an average person pick up {t}?",
    ],
    "salary": [
        "How much does a {t} professional earn?", "What is the average salary in {t}?",
        "How much money can I make in {t}?", "What do people in {t} get paid?",
        "What is the typical pay for a job in {t}?",
    ],
    "time": [
        "How long does it take to learn {t}?", "How many months are needed to get good at {t}?",
        "What is the time required to become proficient in {t}?",
        "How long before I become decent at {t}?",
    ],
    "compare": [
        "What is the difference between {t} and {u}?", "{t} vs {u}: which is better?",
        "How does {t} compare with {u}?", "Should I choose {t} or {u}?",
        "Which one is better, {t} or {u}?",
    ],
}
_TOPICS = [
    "python", "machine learning", "data science", "web development", "photography",
    "digital marketing", "cyber security", "graphic design", "cloud computing",
    "accounting", "spanish", "guitar", "investing", "public speaking", "robotics",
    "game development", "blockchain", "statistics", "video editing", "chess",
    "java", "sql", "UX design", "mechanical engineering", "creative writing",
    "finance", "nursing", "law", "psychology", "biotechnology", "excel", "calculus",
]
_GENERIC = [
    ("Why is the sky blue?", "What makes the sky look blue?"),
    ("How do I lose weight fast?", "What is the quickest way to lose weight?"),
    ("What is the capital of Australia?", "Which city is the capital of Australia?"),
    ("How do vaccines work?", "What is the mechanism behind vaccination?"),
    ("Why do cats purr?", "What causes cats to purr?"),
    ("How can I improve my sleep?", "What are some ways to sleep better?"),
]


def _typo(s: str, rng: random.Random, p: float = 0.12) -> str:
    if rng.random() > p or len(s) < 8:
        return s
    i = rng.randrange(1, len(s) - 2)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def make_demo_dataset(n: int = 40_000, seed: int = 7, label_noise: float = 0.06) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    intents = list(_INTENTS)
    while len(rows) < n:
        r = rng.random()
        t, u = rng.sample(_TOPICS, 2)
        if r < 0.40:  # duplicate: same intent + topic, different wording
            k = rng.choice(intents)
            a, b = rng.sample(_INTENTS[k], 2)
            rows.append((a.format(t=t, u=u), b.format(t=t, u=u), 1))
        elif r < 0.62:  # hard negative: same intent, different topic
            k = rng.choice(intents)
            a, b = rng.choice(_INTENTS[k]), rng.choice(_INTENTS[k])
            t2 = rng.choice([x for x in _TOPICS if x != t])
            rows.append((a.format(t=t, u=u), b.format(t=t2, u=u), 0))
        elif r < 0.84:  # hard negative: same topic, different intent
            k1, k2 = rng.sample(intents, 2)
            a, b = rng.choice(_INTENTS[k1]), rng.choice(_INTENTS[k2])
            rows.append((a.format(t=t, u=u), b.format(t=t, u=u), 0))
        elif r < 0.92:  # easy negative
            k1, k2 = rng.sample(intents, 2)
            t2 = rng.choice([x for x in _TOPICS if x != t])
            a, b = rng.choice(_INTENTS[k1]), rng.choice(_INTENTS[k2])
            rows.append((a.format(t=t, u=u), b.format(t=t2, u=u), 0))
        elif r < 0.96:  # generic duplicates
            a, b = rng.choice(_GENERIC)
            rows.append((a, b, 1))
        else:  # generic non-duplicates
            (a, _), (_, b) = rng.sample(_GENERIC, 2)
            rows.append((a, b, 0))
    df = pd.DataFrame(rows, columns=COLS)
    df["question1"] = [_typo(s, rng) for s in df.question1]
    df["question2"] = [_typo(s, rng) for s in df.question2]
    flip = pd.Series([rng.random() < label_noise for _ in range(len(df))])
    df.loc[flip, "is_duplicate"] = 1 - df.loc[flip, "is_duplicate"]
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def augment_antonym_negatives(train: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Contrastive augmentation for the TRAINING split only.

    Takes real questions containing a word from the contrast list and swaps it for its
    opposite ("lose weight" -> "gain weight"), labelling the pair non-duplicate. Only ~0.3%
    of Quora pairs carry this signal, so without augmentation the model barely learns that
    one flipped word changes the meaning. Validation and test splits are never augmented.
    """
    import re

    from .features import _ANTONYM_WORDS

    swap = {}
    for a, b in _ANTONYM_WORDS:
        swap.setdefault(a, b)
        swap.setdefault(b, a)
    pattern = re.compile(r"\b(" + "|".join(sorted(map(re.escape, swap), key=len, reverse=True)) + r")\b",
                         re.IGNORECASE)
    rng = random.Random(seed)
    questions = pd.concat([train.question1, train.question2]).drop_duplicates().tolist()
    rng.shuffle(questions)
    rows, per_word = [], {}
    cap = max(50, n // 12)  # keep the negatives diverse: no single word dominates
    for q in questions:
        hits = [m for m in pattern.finditer(q) if per_word.get(m.group(0).lower(), 0) < cap]
        if not hits:
            continue
        m = rng.choice(hits)
        word = m.group(0)
        per_word[word.lower()] = per_word.get(word.lower(), 0) + 1
        rep = swap[word.lower()]
        rep = rep.capitalize() if word[0].isupper() else rep
        q2 = q[: m.start()] + rep + q[m.end():]
        rows.append((q, q2, 0) if rng.random() < 0.5 else (q2, q, 0))
        if len(rows) >= n:
            break
    return pd.DataFrame(rows, columns=COLS)


QUORA_URL = ("https://raw.githubusercontent.com/MLDroid/quora_duplicate_challenge/"
             "master/data/quora_duplicate_questions.tsv")
QUORA_SHA256 = "b3350dbb1d98db1f5abca85d736ba514bf4da253fae53939303d79fe921ec7c8"
DEFAULT_PATH = Path("data/quora_duplicate_questions.tsv")


def download_quora(path: str | Path = DEFAULT_PATH) -> Path:
    """Download the official Quora release (58 MB) once and verify its checksum."""
    import hashlib
    import urllib.request

    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading Quora Question Pairs -> {path} ...", flush=True)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(QUORA_URL, tmp)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != QUORA_SHA256:
        tmp.unlink()
        raise SystemExit(f"checksum mismatch for downloaded file ({digest})")
    tmp.rename(path)
    return path


def load_dataset(source: str, path: str | None = None, n_demo: int = 40_000) -> pd.DataFrame:
    if source == "demo":
        return make_demo_dataset(n_demo)
    if source == "glue":
        return load_glue_qqp()
    if source == "quora":
        return load_file(path or download_quora())
    if source == "kaggle":
        if not path:
            raise SystemExit("--data PATH is required for --dataset kaggle")
        return load_file(path)
    raise SystemExit(f"unknown dataset source: {source}")
