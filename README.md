# Jigsaw Toxic Comment Classification

A team project built on the **Jigsaw Toxic Comment Classification Challenge** dataset from Kaggle. The goal is to build models that can detect different types of toxicity in online comments (toxic, severe toxic, obscene, threat, insult, identity hate).

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Getting Started](#getting-started)
3. [Working with Git & GitHub](#working-with-git--github)
4. [Dataset Description](#dataset-description)
5. [Reference](#reference)

---

## Project Structure

```
AIO26-Warmup3-Team_LevelUp-Toxic-Comment/
├── dataset/
│   ├── raw/         # Original Kaggle CSV files (gitignored — download manually)
│   └── processed/   # Cleaned / feature-engineered data (gitignored)
├── notebook/
│   └── data_expore.ipynb
├── .gitignore
└── README.md
```

> The `dataset/` folder is excluded from Git because the files are large. Each teammate downloads the data locally — see [Getting Started](#getting-started).

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<owner>/AIO26-Warmup3-Team_LevelUp-Toxic-Comment.git
cd AIO26-Warmup3-Team_LevelUp-Toxic-Comment
```

### 2. Download the dataset from Kaggle
Get the data from the competition page:
👉 https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data

Unzip and place the files inside `dataset/raw/` so the structure looks like:
```
dataset/raw/
├── train.csv
├── test.csv
├── test_labels.csv
└── sample_submission.csv
```

### 3. Set up a Python environment

**Option A — conda (recommended, matches the original setup):**
```bash
conda env create -f environment.yml
conda activate toxic-nlp
```

**Option B — pip + venv:**
```bash
python3 -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> `environment.yml` lists only the top-level packages (Python 3.12, NumPy, pandas, scikit-learn, matplotlib, Jupyter, PyArrow, ipykernel) and lets conda solve the rest.
> `requirements.txt` is a fully-pinned `pip freeze` of the same environment for exact reproducibility.

### 4. Open the notebook
```bash
jupyter notebook notebook/data_expore.ipynb
```

---

## Working with Git & GitHub

This section is a quick guide for teammates. Read it once before starting your first task.

### Golden rules
- **Never push directly to `main`.** Always work on a branch and open a Pull Request (PR).
- **Pull before you start working.** Make sure your `main` is up to date.
- **Commit often, in small logical chunks.** Easier to review and to roll back.
- **One feature / fix = one branch = one PR.** Don't mix unrelated work.

### One-time setup
```bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
```

### Daily workflow

**1. Sync your local `main`**
```bash
git checkout main
git pull origin main
```

**2. Create a branch for your task**

Use a descriptive name with a prefix:
- `feature/...` — new feature
- `fix/...` — bug fix
- `exp/...` — experiment / model
- `docs/...` — documentation

```bash
git checkout -b feature/lstm-baseline
```

**3. Do your work, then stage and commit**
```bash
git status                          # see what changed
git add path/to/file.py             # stage specific files (avoid `git add .`)
git commit -m "Add LSTM baseline model"
```

Write commit messages in the imperative mood: *"Add"*, *"Fix"*, *"Update"* — not *"Added"* or *"Adding"*.

**4. Push your branch**
```bash
git push -u origin feature/lstm-baseline
```
The `-u` flag only needs to be set the first time you push a branch.

**5. Open a Pull Request on GitHub**
- Go to the repo on GitHub → it will prompt you to open a PR for your branch.
- Set the base branch to `main`.
- Write a short description: **what** you changed and **why**.
- Request a review from at least one teammate.

**6. After the PR is merged, clean up**
```bash
git checkout main
git pull origin main
git branch -d feature/lstm-baseline           # delete local branch
git push origin --delete feature/lstm-baseline # delete remote branch (or use the GitHub UI)
```

### Keeping your branch up to date with `main`
If `main` moves forward while you're working on a long-running branch:
```bash
git checkout feature/lstm-baseline
git fetch origin
git merge origin/main
# resolve any conflicts, then commit and push
```

### Handy commands
| Command | What it does |
|---|---|
| `git status` | Show changed files |
| `git diff` | Show unstaged changes |
| `git log --oneline --graph --all` | Visual commit history |
| `git stash` | Temporarily save uncommitted work |
| `git stash pop` | Restore stashed work |
| `git restore <file>` | Discard local changes to a file |
| `git branch -a` | List all branches (local + remote) |

### Resolving merge conflicts
1. Git will mark conflicted files. Open them — look for `<<<<<<<`, `=======`, `>>>>>>>` markers.
2. Edit to keep the correct code, then remove the markers.
3. `git add <file>` and `git commit` to finish the merge.
4. If you're stuck, ask in the team chat **before** force-pushing or resetting.

### What NOT to commit
- The `dataset/` folder (already gitignored).
- Trained model files, checkpoints, logs.
- API keys, `kaggle.json`, `.env` files.
- Jupyter notebook checkpoints.

If something sensitive lands in a commit, tell the team immediately — don't just `git push --force`.

---

## Dataset Description

The dataset contains a large number of Wikipedia talk-page comments that have been labeled by human raters for toxic behavior. Each comment can have **multiple** labels (it's a multi-label classification problem).

### Files
| File | Description |
|---|---|
| `train.csv` | Training set with comment text and binary labels for each toxicity type. |
| `test.csv` | Test set — comment text only. |
| `test_labels.csv` | Labels for the test set (released after the competition). Rows with `-1` were not used for scoring. |
| `sample_submission.csv` | Example of the expected submission format. |

### Label columns (in `train.csv`)
The target is six binary columns. A single comment can be flagged with any combination of them.

| Label | Meaning |
|---|---|
| `toxic` | Generally toxic / rude. |
| `severe_toxic` | Extremely toxic. |
| `obscene` | Obscene language. |
| `threat` | Contains a threat. |
| `insult` | Insulting language. |
| `identity_hate` | Hate directed at a person's identity (race, religion, gender, etc.). |

### Why it's interesting
- **Multi-label**: a comment may be both `toxic` and `insult`, or `toxic` and `obscene` and `identity_hate`.
- **Class imbalance**: most comments are non-toxic, so naive accuracy is misleading — use ROC-AUC / F1 per label.
- **Real-world text**: messy, contains typos, slang, code-switching, and adversarial obfuscation.

⚠️ **Content warning:** the dataset contains profanity, slurs, and hateful content by design. Handle it professionally.

---

## Reference

- **Competition page:** https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge
- **Dataset download:** https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data
- **Hosted by:** Jigsaw / Conversation AI team (Google), on Kaggle.

Please follow the [Kaggle competition rules](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/rules) when using the data.
