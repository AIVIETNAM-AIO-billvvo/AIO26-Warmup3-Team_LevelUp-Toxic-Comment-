# `dataset/processed/`

Cleaned, model-ready outputs of the cleaning pipeline. **This folder is gitignored** — re-generate locally with:

```bash
python scripts/run_clean.py --split both
```

Source: `dataset/raw/{train,test}/*.csv` → cleaned by `src/clean.py`. See `doc/components/cleaning-pipeline.md` for the full pipeline definition.

---

## Files

| File | Rows | Size | Purpose |
|---|---|---|---|
| `train_clean.csv` | ~149k | ~44 MB | Training data — text cleaned + lemmatized + English-only + deduped |
| `test_clean.csv`  | 153,164 | ~39 MB | Test data — same text cleaning, **all rows preserved** for inference |
| `cleaning_report.json` | — | ~1 KB | Stats from the last pipeline run (drops, label drift, lib versions) |

---

## Schemas

### `train_clean.csv`
| Column | Type | Notes |
|---|---|---|
| `id` | str | Original Kaggle id |
| `comment_text` | str | Cleaned text — never empty |
| `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` | int (0/1) | Multi-label targets |

### `test_clean.csv`
| Column | Type | Notes |
|---|---|---|
| `id` | str | Aligned 1:1 with `dataset/raw/test/test.csv` |
| `comment_text` | str | Cleaned text — **may be empty** (e.g., URL-only comments) |

Test labels live separately in `dataset/raw/test/test_labels.csv`. Join on `id` at evaluation time. Filter rows with label = `-1` before computing metrics — Kaggle didn't score those.

### `cleaning_report.json`
```jsonc
{
  "cleaned_at": "...",                    // ISO timestamp of last run
  "lib_versions": { "nltk": "...", ... }, // for reproducibility
  "train": { "n_in", "n_out", "n_dropped_lang", "n_dropped_empty",
             "n_dropped_dup", "label_dist_before", "label_dist_after" },
  "test":  { ... }                        // drops are 0 by design
}
```

---

## Quick load

```python
import pandas as pd
train = pd.read_csv("dataset/processed/train_clean.csv")
test  = pd.read_csv("dataset/processed/test_clean.csv")
test_labels = pd.read_csv("dataset/raw/test/test_labels.csv")

# usable test rows (Kaggle-scored)
test_eval = test.merge(test_labels, on="id")
test_eval = test_eval[(test_eval[["toxic","severe_toxic","obscene",
                                   "threat","insult","identity_hate"]] != -1).all(axis=1)]
```

---

## Latest run snapshot

From `cleaning_report.json` (2026-05-09):

- **Train**: 159,571 → 149,843 rows (dropped 9,728 = 6.1%)
  - 8,492 dropped by language filter (lang ≠ en or confidence < 0.7)
  - 37 dropped as empty after cleaning
  - 1,199 dropped as duplicates
- **Test**: 153,164 → 153,164 rows (no drops, as designed)

**Label-distribution drift** (positive rate before → after, train):

| Label | Before | After | Δ (pp) |
|---|---:|---:|---:|
| toxic         | 9.58% | 8.11% | **−1.47** |
| severe_toxic  | 1.00% | 0.60% | **−0.40** |
| obscene       | 5.29% | 4.22% | **−1.07** |
| threat        | 0.30% | 0.23% | −0.07 |
| insult        | 4.94% | 3.98% | **−0.96** |
| identity_hate | 0.88% | 0.68% | **−0.20** |

> ⚠️ **Drift is larger than the 0.5 pp target** for `toxic`, `obscene`, `insult`. The language filter is over-dropping toxic comments — heavy profanity / obfuscation often gets misclassified as non-English. Consider lowering `lang_threshold` from 0.7 to 0.5 or relaxing the rule for high-toxicity-marker rows. See `doc/components/cleaning-pipeline.md` § "Language detection".

---

## Reproducibility

The pipeline is deterministic. Re-running on the same input + same library versions produces the same output. Check:

1. `cleaning_report.json::lib_versions` — same packages?
2. `dataset/raw/` — same source CSVs (compare `md5` / row counts)?
3. `src/clean.py` — same code?


