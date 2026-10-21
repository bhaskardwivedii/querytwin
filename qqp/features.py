"""Pair feature engineering: TF-IDF pair vectors + handcrafted similarity features.

Everything that learns from data (the vectorisers, idf weights) is fitted on the
training split only, so validation/test metrics are not contaminated.

Deliberately NOT used: Kaggle-style "magic" graph/frequency features (how often a
question appears across pairs). They leak dataset structure and do not transfer to
new questions at inference time.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import LCSseq
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocess import STOP_WORDS, negation_count, normalize, stem_text, tokenize
from .preprocess import _stem as _stem_word

_CAP_RE = re.compile(r"\b[A-Z][a-zA-Z]+\b")
_QWORDS = {"what", "how", "why", "when", "where", "who", "which", "is", "can", "do",
           "does", "should", "are", "will", "would", "did", "could"}

DENSE_NAMES = [
    "len_q1", "len_q2", "len_diff", "len_ratio",
    "words_q1", "words_q2", "words_diff", "words_ratio",
    "common_words", "common_ratio_min", "common_ratio_max", "jaccard",
    "idf_overlap", "idf_jaccard", "content_jaccard", "content_common",
    "only_q1", "only_q2",
    "fuzz_ratio", "fuzz_partial", "fuzz_token_sort", "fuzz_token_set", "fuzz_wratio",
    "cos_word", "cos_char",
    "first_same", "last_same", "qword_same", "qword_both",
    "num_equal", "num_jaccard", "num_any",
    "neg_q1", "neg_q2", "neg_diff", "cap_jaccard", "cap_any",
    # targeted at bag-of-words blind spots (antonyms, one changed key word, word order)
    "antonym_conflict", "diff_idf_sum", "diff_idf_max", "diff_idf_ratio",
    "lcs_tokens", "lcs_content", "exact_match",
]

# Opposite-meaning pairs common in questions. Stored as stems so they match tokenised text.
_ANTONYM_WORDS = [
    ("lose", "gain"), ("increase", "decrease"), ("increase", "reduce"), ("buy", "sell"),
    ("best", "worst"), ("good", "bad"), ("better", "worse"), ("before", "after"),
    ("advantage", "disadvantage"), ("pros", "cons"), ("male", "female"), ("men", "women"),
    ("man", "woman"), ("boy", "girl"), ("positive", "negative"), ("start", "stop"),
    ("start", "end"), ("begin", "end"), ("first", "last"), ("cheap", "expensive"),
    ("high", "low"), ("fast", "slow"), ("easy", "hard"), ("easy", "difficult"),
    ("rich", "poor"), ("old", "new"), ("old", "young"), ("hot", "cold"),
    ("love", "hate"), ("win", "lose"), ("open", "close"), ("pass", "fail"), ("success", "failure"),
    ("import", "export"), ("legal", "illegal"), ("true", "false"), ("happy", "sad"),
    ("introvert", "extrovert"), ("upload", "download"), ("north", "south"), ("east", "west"),
    ("day", "night"), ("past", "future"), ("strong", "weak"), ("big", "small"), ("long", "short"),
    ("light", "dark"), ("light", "heavy"), ("fat", "thin"), ("more", "less"), ("most", "least"),
    ("maximum", "minimum"), ("teacher", "student"), ("employer", "employee"), ("buyer", "seller"),
    ("husband", "wife"), ("father", "mother"), ("son", "daughter"), ("iphone", "android"),
    ("mac", "windows"), ("android", "ios"), ("online", "offline"), ("input", "output"),
]


ANTONYMS = [(_stem_word(x), _stem_word(y)) for x, y in _ANTONYM_WORDS]


def _antonym_conflict(a: set, b: set) -> float:
    """Number of opposite-meaning word pairs split across the two questions
    (e.g. 'lose' only in q1 and 'gain' only in q2)."""
    only_a, only_b = a - b, b - a
    return float(sum((x in only_a and y in only_b) or (y in only_a and x in only_b) for x, y in ANTONYMS))


def _jacc(a: set, b: set) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


class PairFeaturizer:
    """Turns (q1, q2) pairs into model inputs.

    Outputs of ``transform``:
      * X_sparse - |W1-W2| and W1*W2 over word TF-IDF (for the linear model)
      * X_dense  - 37 handcrafted similarity features + LSA pair features:
                   cosine in LSA space, |v1-v2| and v1*v2 for each of ``n_svd``
                   latent dimensions (compact semantic signal for tree models)
    """

    def __init__(self, max_word_features: int = 50_000, max_char_features: int = 50_000,
                 min_df: int = 2, n_svd: int = 100, seed: int = 42):
        self.word_vec = TfidfVectorizer(
            ngram_range=(1, 2), min_df=min_df, max_features=max_word_features,
            sublinear_tf=True, dtype=np.float32, token_pattern=r"(?u)\b\w+\b")
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), min_df=min_df,
            max_features=max_char_features, sublinear_tf=True, dtype=np.float32)
        self.n_svd = n_svd
        self.svd = TruncatedSVD(n_components=n_svd, algorithm="randomized", n_iter=5,
                                random_state=seed) if n_svd else None
        self.idf_: dict = {}
        self.idf_max_: float = 1.0

    @property
    def feature_names(self) -> List[str]:
        if not self.n_svd:
            return list(DENSE_NAMES)
        return (list(DENSE_NAMES) + ["cos_lsa"]
                + [f"lsa_absdiff_{k}" for k in range(self.n_svd)]
                + [f"lsa_prod_{k}" for k in range(self.n_svd)])

    def _lsa(self, W, C) -> np.ndarray:
        Z = self.svd.transform(sp.hstack([W, C], format="csr")).astype(np.float32)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
        return Z

    # -- text preparation ------------------------------------------------- #
    @staticmethod
    def prepare(questions: Iterable[str]) -> Tuple[List[str], List[str]]:
        norm = [normalize(q) for q in questions]
        stemmed = [stem_text(n) for n in norm]
        return norm, stemmed

    # -- fitting ---------------------------------------------------------- #
    def fit(self, q1: Sequence[str], q2: Sequence[str]) -> "PairFeaturizer":
        n1, s1 = self.prepare(q1)
        n2, s2 = self.prepare(q2)
        corpus = s1 + s2
        W = self.word_vec.fit_transform(corpus)
        C = self.char_vec.fit_transform(corpus)
        if self.svd is not None:
            # unique questions only, so frequent questions don't dominate the latent space
            _, first = np.unique(np.array(corpus, dtype=object), return_index=True)
            self.svd.fit(sp.hstack([W[first], C[first]], format="csr"))
        vocab, idf = self.word_vec.vocabulary_, self.word_vec.idf_
        self.idf_ = {w: float(idf[i]) for w, i in vocab.items() if " " not in w}
        self.idf_max_ = float(idf.max())
        return self

    # -- transform -------------------------------------------------------- #
    def transform(self, q1: Sequence[str], q2: Sequence[str]):
        """Return (X_sparse, X_dense, dense_names)."""
        q1, q2 = list(q1), list(q2)
        n1, s1 = self.prepare(q1)
        n2, s2 = self.prepare(q2)

        W1, W2 = self.word_vec.transform(s1), self.word_vec.transform(s2)
        C1, C2 = self.char_vec.transform(s1), self.char_vec.transform(s2)
        cos_word = np.asarray(W1.multiply(W2).sum(axis=1)).ravel()
        cos_char = np.asarray(C1.multiply(C2).sum(axis=1)).ravel()

        X_sparse = sp.hstack([abs(W1 - W2), W1.multiply(W2)], format="csr", dtype=np.float32)
        dense = self._dense(q1, q2, n1, n2, s1, s2, cos_word, cos_char)
        if self.svd is not None:
            Z1, Z2 = self._lsa(W1, C1), self._lsa(W2, C2)
            cos_lsa = (Z1 * Z2).sum(axis=1, keepdims=True)
            dense = np.hstack([dense, cos_lsa, np.abs(Z1 - Z2), Z1 * Z2]).astype(np.float32)
        return X_sparse, dense, self.feature_names

    def _dense(self, q1, q2, n1, n2, s1, s2, cos_word, cos_char) -> np.ndarray:
        idf, idf_max = self.idf_, self.idf_max_
        out = np.zeros((len(q1), len(DENSE_NAMES)), dtype=np.float32)
        for i in range(len(q1)):
            t1, t2 = s1[i].split(), s2[i].split()
            a, b = set(t1), set(t2)
            common = a & b
            c1 = {w for w in tokenize(n1[i], stem=True, drop_stop=True)}
            c2 = {w for w in tokenize(n2[i], stem=True, drop_stop=True)}
            l1, l2 = len(q1[i]), len(q2[i])
            w1, w2 = len(t1), len(t2)

            wi = lambda s: sum(idf.get(w, idf_max) for w in s)  # noqa: E731
            idf_c, idf_u = wi(common), wi(a | b)
            idf_a, idf_b = wi(a), wi(b)

            f1, f2 = (n1[i].split() or [""]), (n2[i].split() or [""])
            nums1 = {w for w in f1 if w.isdigit()}
            nums2 = {w for w in f2 if w.isdigit()}
            caps1 = {m.lower() for m in _CAP_RE.findall(q1[i])[1:]}
            caps2 = {m.lower() for m in _CAP_RE.findall(q2[i])[1:]}
            ng1, ng2 = negation_count(n1[i]), negation_count(n2[i])
            qw1, qw2 = f1[0] in _QWORDS, f2[0] in _QWORDS
            diff = [idf.get(w, idf_max) for w in (c1 ^ c2)]  # content words in only one question
            d_sum, d_max = float(sum(diff)), float(max(diff, default=0.0))
            cl1 = [w for w in t1 if w in c1]
            cl2 = [w for w in t2 if w in c2]

            out[i] = [
                l1, l2, abs(l1 - l2), min(l1, l2) / max(l1, l2, 1),
                w1, w2, abs(w1 - w2), min(w1, w2) / max(w1, w2, 1),
                len(common), len(common) / max(min(len(a), len(b)), 1),
                len(common) / max(max(len(a), len(b)), 1), _jacc(a, b),
                idf_c / max(min(idf_a, idf_b), 1e-6), idf_c / max(idf_u, 1e-6),
                _jacc(c1, c2), len(c1 & c2),
                len(a - b), len(b - a),
                fuzz.ratio(n1[i], n2[i]), fuzz.partial_ratio(n1[i], n2[i]),
                fuzz.token_sort_ratio(n1[i], n2[i]), fuzz.token_set_ratio(n1[i], n2[i]),
                fuzz.WRatio(n1[i], n2[i]),
                cos_word[i], cos_char[i],
                float(f1[0] == f2[0]), float(f1[-1] == f2[-1]),
                float(qw1 and qw2 and f1[0] == f2[0]), float(qw1 and qw2),
                float(nums1 == nums2), _jacc(nums1, nums2), float(bool(nums1 or nums2)),
                ng1, ng2, abs(ng1 - ng2), _jacc(caps1, caps2), float(bool(caps1 or caps2)),
                _antonym_conflict(c1, c2),
                d_sum, d_max, d_sum / max(idf_u, 1e-6),
                LCSseq.normalized_similarity(t1, t2),
                LCSseq.normalized_similarity(cl1, cl2),
                float(n1[i] == n2[i]),
            ]
        return out
