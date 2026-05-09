"""Idempotent download of NLTK corpora and fastText language-id model.

Call `ensure_nlp_assets()` once before using `clean.py`. Safe to call repeatedly —
each step checks for existing files before downloading.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NLTK_DATA_DIR = PROJECT_ROOT / "nltk_data"
FASTTEXT_MODEL_DIR = PROJECT_ROOT / "dataset" / "models"
FASTTEXT_MODEL_PATH = FASTTEXT_MODEL_DIR / "lid.176.ftz"
FASTTEXT_MODEL_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
)
NLTK_CORPORA = [
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]


def ensure_nltk_assets(verbose: bool = True) -> None:
    import nltk

    NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    nltk_data_str = str(NLTK_DATA_DIR)
    if nltk_data_str not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_str)

    for resource_path, package_id in NLTK_CORPORA:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            if verbose:
                print(f"[setup_nlp] Downloading NLTK '{package_id}' to {NLTK_DATA_DIR}")
            nltk.download(package_id, download_dir=nltk_data_str, quiet=not verbose)


def ensure_fasttext_model(verbose: bool = True) -> Path:
    FASTTEXT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not FASTTEXT_MODEL_PATH.exists():
        if verbose:
            print(
                f"[setup_nlp] Downloading fastText lid.176.ftz "
                f"({FASTTEXT_MODEL_URL}) to {FASTTEXT_MODEL_PATH}"
            )
        urllib.request.urlretrieve(FASTTEXT_MODEL_URL, FASTTEXT_MODEL_PATH)
    return FASTTEXT_MODEL_PATH


def ensure_nlp_assets(verbose: bool = True) -> None:
    ensure_nltk_assets(verbose=verbose)
    ensure_fasttext_model(verbose=verbose)


if __name__ == "__main__":
    ensure_nlp_assets(verbose=True)
    print("[setup_nlp] All assets present.")
