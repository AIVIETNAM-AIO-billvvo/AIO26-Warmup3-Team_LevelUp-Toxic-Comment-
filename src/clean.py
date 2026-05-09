"""Cleaning pipeline for the Jigsaw Toxic Comment dataset.

Pipeline (in order):
    String-level (universal):
        decode_html, strip_control_ws, remove_urls, remove_paths,
        remove_emails, remove_ips, strip_wiki_markup, cap_repeats, collapse_ws
    DataFrame-level:
        language detection (fastText lid.176, batched), drop empty rows
    String-level (classical-ML):
        lowercase, expand_contractions, remove_punct, remove_stopwords, lemmatize
    DataFrame-level:
        drop empty rows again, drop duplicates

Each str->str function is independently testable. Compose via clean_text_universal /
clean_text_classical, or run end-to-end on a DataFrame via clean_dataframe.
"""
from __future__ import annotations

import html
from typing import Callable

import pandas as pd
import regex as re
from tqdm.auto import tqdm

from src.setup_nlp import (
    FASTTEXT_MODEL_PATH,
    NLTK_DATA_DIR,
    ensure_fasttext_model,
    ensure_nltk_assets,
)

_STOPWORDS_CACHE: set[str] | None = None
_LEMMATIZER_CACHE = None
_FT_MODEL = None


def _get_stopwords() -> set[str]:
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is None:
        ensure_nltk_assets(verbose=False)
        import nltk
        from nltk.corpus import stopwords as _nltk_stopwords
        nltk_data_str = str(NLTK_DATA_DIR)
        if nltk_data_str not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_str)
        _STOPWORDS_CACHE = set(_nltk_stopwords.words("english"))
    return _STOPWORDS_CACHE


def _get_lemmatizer():
    global _LEMMATIZER_CACHE
    if _LEMMATIZER_CACHE is None:
        ensure_nltk_assets(verbose=False)
        import nltk
        from nltk.stem import WordNetLemmatizer
        nltk_data_str = str(NLTK_DATA_DIR)
        if nltk_data_str not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_str)
        _LEMMATIZER_CACHE = WordNetLemmatizer()
    return _LEMMATIZER_CACHE

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_IPV6_RE = re.compile(
    r"\b(?:[A-Fa-f0-9]{0,4}:){2,7}[A-Fa-f0-9]{1,4}\b"
)
_UNIX_PATH_RE = re.compile(r"(?:/[\w.\-]+){2,}/?")
_WIN_PATH_RE = re.compile(r"[A-Za-z]:\\(?:[\w.\- ]+\\?)+")
_WIKI_LINK_RE = re.compile(r"\[\[[^\]]*\]\]")
_WIKI_TEMPLATE_RE = re.compile(r"\{\{[^\}]*\}\}")
_WIKI_HEADER_RE = re.compile(r"={2,}\s*([^=]+?)\s*={2,}")
_WIKI_BOLD_RE = re.compile(r"'{3,5}([^']+)'{3,5}")
_WIKI_ITALIC_RE = re.compile(r"'{2}([^']+)'{2}")
_REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")
_CONTROL_WS_RE = re.compile(r"[\n\t\r]+")
_MULTI_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_DIGITS_ONLY_TOKEN_RE = re.compile(r"\b\d+\b")


def decode_html(s: str) -> str:
    return html.unescape(s)


def strip_control_ws(s: str) -> str:
    return _CONTROL_WS_RE.sub(" ", s)


def remove_urls(s: str) -> str:
    return _URL_RE.sub(" ", s)


def remove_paths(s: str) -> str:
    s = _WIN_PATH_RE.sub(" ", s)
    s = _UNIX_PATH_RE.sub(" ", s)
    return s


def remove_emails(s: str) -> str:
    return _EMAIL_RE.sub(" ", s)


def remove_ips(s: str) -> str:
    s = _IPV4_RE.sub(" ", s)
    s = _IPV6_RE.sub(" ", s)
    return s


def strip_wiki_markup(s: str) -> str:
    s = _WIKI_LINK_RE.sub(" ", s)
    s = _WIKI_TEMPLATE_RE.sub(" ", s)
    s = _WIKI_HEADER_RE.sub(r"\1", s)
    s = _WIKI_BOLD_RE.sub(r"\1", s)
    s = _WIKI_ITALIC_RE.sub(r"\1", s)
    return s


def cap_repeats(s: str) -> str:
    return _REPEAT_CHAR_RE.sub(r"\1\1", s)


def collapse_ws(s: str) -> str:
    return _MULTI_WS_RE.sub(" ", s).strip()


def lowercase(s: str) -> str:
    return s.lower()


def expand_contractions(s: str) -> str:
    import contractions
    return contractions.fix(s)


def remove_punct(s: str) -> str:
    return _PUNCT_RE.sub(" ", s)


def remove_stopwords(s: str) -> str:
    sw = _get_stopwords()
    return " ".join(t for t in s.split() if t not in sw)


def lemmatize(s: str) -> str:
    lem = _get_lemmatizer()
    return " ".join(lem.lemmatize(t) for t in s.split())


_UNIVERSAL_STEPS: tuple[Callable[[str], str], ...] = (
    decode_html,
    strip_control_ws,
    remove_urls,
    remove_paths,
    remove_emails,
    remove_ips,
    strip_wiki_markup,
    cap_repeats,
    collapse_ws,
)

_CLASSICAL_STEPS: tuple[Callable[[str], str], ...] = (
    lowercase,
    expand_contractions,
    remove_punct,
    remove_stopwords,
    lemmatize,
    collapse_ws,
)


def clean_text_universal(s: str) -> str:
    if not isinstance(s, str):
        return ""
    for step in _UNIVERSAL_STEPS:
        s = step(s)
    return s


def clean_text_classical(s: str) -> str:
    if not isinstance(s, str):
        return ""
    for step in _CLASSICAL_STEPS:
        s = step(s)
    return s


def _get_fasttext_model():
    global _FT_MODEL
    if _FT_MODEL is None:
        ensure_fasttext_model(verbose=False)
        import fasttext
        _FT_MODEL = fasttext.load_model(str(FASTTEXT_MODEL_PATH))
    return _FT_MODEL


def detect_language_batch(texts: list[str]) -> list[tuple[str, float]]:
    """Predict (lang_code, confidence) for each text. Batched single fastText call.

    fastText errors on `\\n` in input — callers must run strip_control_ws first.
    """
    model = _get_fasttext_model()
    safe = [t.replace("\n", " ") if isinstance(t, str) else "" for t in texts]
    labels, probs = model.predict(safe, k=1)
    return [
        (lab[0].replace("__label__", ""), float(prob[0]))
        for lab, prob in zip(labels, probs)
    ]


def _progress_apply(series: pd.Series, fn: Callable[[str], str], desc: str) -> list[str]:
    """tqdm progress bar over a Series mapping. Avoids tqdm.pandas which breaks
    on pandas 3.x due to removed internal API."""
    return [fn(x) for x in tqdm(series.tolist(), desc=desc)]


def clean_dataframe(
    df: pd.DataFrame,
    text_col: str = "comment_text",
    classical: bool = True,
    drop_lang: bool = True,
    drop_empty: bool = True,
    drop_dups: bool = True,
    lang_threshold: float = 0.7,
) -> tuple[pd.DataFrame, dict]:
    """Apply cleaning pipeline. Returns (cleaned_df, stats_dict).

    Set drop_lang=drop_empty=drop_dups=False for test data when row count must
    be preserved for inference.
    """
    df = df.copy()
    n_in = len(df)

    df[text_col] = _progress_apply(
        df[text_col].fillna("").astype(str),
        clean_text_universal,
        desc="universal clean",
    )

    lang_results = detect_language_batch(df[text_col].tolist())
    langs = [r[0] for r in lang_results]
    confs = [r[1] for r in lang_results]

    n_dropped_lang = 0
    if drop_lang:
        keep_mask = [
            (l == "en" and c >= lang_threshold) for l, c in zip(langs, confs)
        ]
        n_dropped_lang = int(len(df) - sum(keep_mask))
        df = df[keep_mask].copy()

    n_dropped_empty = 0
    if drop_empty:
        non_empty = df[text_col].str.strip().ne("")
        n_dropped_empty = int((~non_empty).sum())
        df = df[non_empty].copy()

    if classical:
        df[text_col] = _progress_apply(
            df[text_col],
            clean_text_classical,
            desc="classical clean",
        )

    n_dropped_empty_post = 0
    if drop_empty:
        non_empty = df[text_col].str.strip().ne("")
        n_dropped_empty_post = int((~non_empty).sum())
        df = df[non_empty].copy()

    n_dropped_dup = 0
    if drop_dups:
        before = len(df)
        df = df.drop_duplicates(subset=[text_col], keep="first").copy()
        n_dropped_dup = before - len(df)

    df = df.reset_index(drop=True)

    stats = {
        "n_in": int(n_in),
        "n_out": int(len(df)),
        "n_dropped_lang": int(n_dropped_lang),
        "n_dropped_empty": int(n_dropped_empty + n_dropped_empty_post),
        "n_dropped_dup": int(n_dropped_dup),
        "lang_threshold": lang_threshold,
        "classical": classical,
    }
    return df, stats
