"""Text normalisation and tokenisation for question pairs."""
from __future__ import annotations

import html
import re
from functools import lru_cache

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_stemmer = PorterStemmer()

# Words that flip meaning must survive stop-word removal.
KEEP = {"not", "no", "never", "nor", "without", "cannot", "against"}
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) - KEEP

_CONTRACTIONS = [
    (r"won't", "will not"), (r"can't", "can not"), (r"cannot", "can not"),
    (r"n't", " not"), (r"'re", " are"), (r"'ve", " have"), (r"'ll", " will"),
    (r"'d", " would"), (r"'m", " am"), (r"what's", "what is"),
    (r"where's", "where is"), (r"how's", "how is"), (r"who's", "who is"),
    (r"it's", "it is"), (r"that's", "that is"), (r"there's", "there is"),
    (r"let's", "let us"), (r"'s", " "),
]
_SYMBOLS = [
    (r"\$", " dollar "), (r"€", " euro "), (r"£", " pound "),
    (r"₹", " rupee "), (r"%", " percent "), (r"&", " and "),
    (r"@", " at "), (r"\bu\.s\.a?\b", " usa "),
]
_NUM_MAGNITUDE = [(r"(\d+)\s?k\b", r"\g<1>000"), (r"(\d+)\s?m\b", r"\g<1>000000")]
_NEGATIONS = {"not", "no", "never", "nor", "without", "cannot"}

_html_re = re.compile(r"<[^>]+>")
_url_re = re.compile(r"https?://\S+|www\.\S+")
_thousands_re = re.compile(r"(?<=\d),(?=\d{3})")
_non_alnum_re = re.compile(r"[^a-z0-9 ]+")
_space_re = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lower-case, expand contractions/symbols, strip markup and punctuation."""
    if not isinstance(text, str):
        return ""
    t = html.unescape(text).lower()
    t = _url_re.sub(" url ", t)
    t = _html_re.sub(" ", t)
    t = t.replace("’", "'").replace("`", "'")
    t = _thousands_re.sub("", t)
    for pat, rep in _SYMBOLS + _NUM_MAGNITUDE + _CONTRACTIONS:
        t = re.sub(pat, rep, t)
    t = _non_alnum_re.sub(" ", t)
    return _space_re.sub(" ", t).strip()


@lru_cache(maxsize=500_000)
def _stem(word: str) -> str:
    return _stemmer.stem(word)


def tokenize(norm_text: str, stem: bool = True, drop_stop: bool = False):
    toks = norm_text.split()
    if drop_stop:
        toks = [w for w in toks if w not in STOP_WORDS]
    if stem:
        toks = [_stem(w) for w in toks]
    return toks


def stem_text(norm_text: str) -> str:
    """Normalised, stemmed string used as TF-IDF input."""
    return " ".join(tokenize(norm_text, stem=True, drop_stop=False))


def negation_count(norm_text: str) -> int:
    return sum(w in _NEGATIONS for w in norm_text.split())
