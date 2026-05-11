"""CLI entry point for the toxic-comment cleaning pipeline.

Usage:
    python scripts/run_clean.py --split train
    python scripts/run_clean.py --split test
    python scripts/run_clean.py --split both
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Union, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.clean import clean_dataframe
from src.setup_nlp import ensure_nlp_assets

RAW_TRAIN = PROJECT_ROOT / "dataset" / "raw" / "train" / "train.csv"
RAW_TEST = PROJECT_ROOT / "dataset" / "raw" / "test" / "test.csv"
OUT_DIR = PROJECT_ROOT / "dataset" / "processed"
OUT_TRAIN = OUT_DIR / "train_clean.csv"
OUT_TEST = OUT_DIR / "test_clean.csv"
REPORT_PATH = OUT_DIR / "cleaning_report.json"

LABEL_COLS = [
    "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate",
]

def clean_text(texts: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Hàm dùng để làm sạch text khi test thực tế (inference).
    Hỗ trợ đầu vào là một chuỗi (str) hoặc danh sách các chuỗi (list).
    """
    is_single_string = isinstance(texts, str)
    if is_single_string:
        texts = [texts]
        
    # Bọc vào DataFrame để tận dụng lại hàm clean_dataframe
    df = pd.DataFrame({"comment_text": texts})
    

    cleaned_df, _ = clean_dataframe(
        df,
        text_col="comment_text",
        classical=True,
        drop_lang=False,
        drop_empty=False,
        drop_dups=False,
    )
    
    # Lấy ra kết quả
    result = cleaned_df["comment_text"].tolist()
    
    # Nếu đầu vào là 1 string thì trả về 1 string, nếu là list thì trả về list
    return result[0] if is_single_string else result

def _label_dist(df: pd.DataFrame) -> dict[str, float]:
    if not all(c in df.columns for c in LABEL_COLS):
        return {}
    n = max(len(df), 1)
    return {c: round(float(df[c].sum()) / n, 6) for c in LABEL_COLS}


def _lib_versions() -> dict[str, str]:
    versions = {}
    for pkg in ("nltk", "fasttext-wheel", "contractions", "regex", "tqdm",
                "pandas", "numpy", "scikit-learn"):
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = "missing"
    return versions


def clean_train() -> dict:
    print(f"[run_clean] Loading {RAW_TRAIN}")
    df = pd.read_csv(RAW_TRAIN)
    label_dist_before = _label_dist(df)
    cleaned, stats = clean_dataframe(
        df,
        text_col="comment_text",
        classical=True,
        drop_lang=True,
        drop_empty=True,
        drop_dups=True,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[run_clean] Writing {OUT_TRAIN} ({len(cleaned):,} rows)")
    cleaned.to_csv(OUT_TRAIN, index=False, quoting=csv.QUOTE_ALL)
    stats["label_dist_before"] = label_dist_before
    stats["label_dist_after"] = _label_dist(cleaned)
    return stats


def clean_test() -> dict:
    print(f"[run_clean] Loading {RAW_TEST}")
    df = pd.read_csv(RAW_TEST)
    cleaned, stats = clean_dataframe(
        df,
        text_col="comment_text",
        classical=True,
        drop_lang=False,
        drop_empty=False,
        drop_dups=False,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[run_clean] Writing {OUT_TEST} ({len(cleaned):,} rows)")
    cleaned.to_csv(OUT_TEST, index=False, quoting=csv.QUOTE_ALL)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="Which split(s) to clean (default: both).",
    )
    args = parser.parse_args()

    ensure_nlp_assets(verbose=True)

    report = {
        "cleaned_at": datetime.now(timezone.utc).isoformat(),
        "lib_versions": _lib_versions(),
    }

    if args.split in ("train", "both"):
        report["train"] = clean_train()
    if args.split in ("test", "both"):
        report["test"] = clean_test()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"[run_clean] Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
