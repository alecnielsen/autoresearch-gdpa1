# autoresearch-gdpa1

Autonomous model development for predicting antibody developability attributes. Forked from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and adapted from LLM pretraining to multi-task regression on the GDPa1 dataset.

## The idea

Give an AI agent a dataset of 246 therapeutic antibodies with experimentally measured developability properties, a baseline model, and let it experiment autonomously overnight on an A100. It modifies the code, trains, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a much better model.

The agent modifies `train.py` — everything else is fixed.

## What we're optimizing

**Mean Spearman rank correlation across 5 antibody developability targets** (higher is better).

Spearman measures how well the model *ranks* antibodies by each property, not how close the absolute predictions are. This is the right metric because in practice you use the model to rank candidates — "which antibodies are likely least sticky?" or "which will express best?" — not to predict exact assay values.

### The 5 targets

| Target | What it measures | Why it matters | N valid |
|---|---|---|---|
| **HIC** | Hydrophobicity (chromatographic retention) | Sticky antibodies aggregate, have poor PK | 242 |
| **Tm2** | Thermal stability (second melting point) | Low Tm = unstable, short shelf life | 193 |
| **PR_CHO** | Polyreactivity to CHO cell extract | High polyreactivity = off-target binding, fast clearance | 197 |
| **AC-SINS pH 7.4** | Self-interaction propensity | High self-interaction = viscosity, aggregation | 242 |
| **Titer** | Expression yield in CHO cells | Low titer = expensive/infeasible manufacturing | 239 |

These properties determine whether a therapeutic antibody candidate can actually be manufactured and dosed as a drug. A model that accurately ranks antibodies on these would let you filter candidates computationally before running expensive wet lab assays.

### Cross-validation

Folds are split by **sequence similarity clustering** (`hierarchical_cluster_fold`), not randomly. The validation set contains antibodies from sequence families the model has never seen during training. This is harder than random CV but reflects the real use case: predicting developability for *new* antibody sequences.

## How it works

### The loop

An AI coding agent runs in a terminal with access to this repo, following `program.md`:

1. **Edit `train.py`** — change model, features, hyperparameters, anything
2. **`git commit`**
3. **`modal run modal_run.py > run.log 2>&1`** — runs `train.py` on an A100 via Modal
4. **Extract result** — `grep "^mean_spearman:" run.log`
5. **If improved**: keep the commit, advance the branch
6. **If not**: `git reset --hard`, discard
7. **Log to `results.tsv`** either way
8. **Repeat forever** until manually stopped

The agent never modifies `prepare.py` (data loading + evaluation) or `modal_run.py` (GPU execution). This separation ensures the metric can't be gamed.

### What `train.py` does

Each run performs 5-fold cross-validation: train on ~196 samples, predict on ~50, collect all out-of-fold predictions, pass to `evaluate()`. The agent can change anything inside: swap the MLP for XGBoost, add physicochemical features, fine-tune ESM-2, try ensembling, etc.

### The branch structure

The agent creates a branch like `autoresearch/mar15`. Successful experiments advance the branch tip. Failed experiments get reverted. So `git log` shows the chain of improvements, and `results.tsv` has the full history including discards and crashes.

### Available packages in the Modal image

torch, numpy, pandas, scipy, scikit-learn, transformers, fair-esm, xgboost, lightgbm

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), [Modal](https://modal.com/) account (for A100 GPU execution).

```bash
# 1. Install dependencies
uv sync

# 2. Authenticate with Modal (one-time)
modal setup

# 3. Verify data and encoding (one-time sanity check)
uv run prepare.py

# 4. Run a single training experiment on A100
modal run modal_run.py
```

## Results: `autoresearch/mar15` run

The agent ran 25 experiments across two sessions, improving mean Spearman from **0.119 to 0.385** (3.2x improvement):

| # | Mean Spearman | Time (s) | Status | Description |
|---|--------------|----------|--------|-------------|
| 1 | 0.119 | 7 | keep | Baseline MLP [512, 256] |
| 2 | 0.151 | 1 | keep | RidgeCV + physicochemical features |
| 3 | 0.228 | 12 | keep | Ridge + LightGBM ensemble |
| 4 | 0.247 | 18 | keep | + ESM-2 8M embeddings |
| 5 | 0.284 | 70 | keep | + ESM-2 650M embeddings |
| 6 | 0.375 | 864 | keep | + Full HC/LC chain embeddings + XGBoost |
| 7 | -- | -- | crash | ElasticNet (Modal timeout) |
| 8 | 0.376 | 1244 | discard | Tune GBM params (marginal) |
| 9 | 0.350 | 1508 | discard | Fine-tune ESM-2 8M (overfit) |
| 10 | -- | -- | crash | Multi-layer ESM + bagging (timeout) |
| 11 | -- | -- | crash | Bagged GBM 5 seeds (timeout) |
| 12 | 0.369 | 1528 | discard | + RandomForest (worse) |
| 13 | 0.374 | 1491 | discard | + Dipeptide features |
| 14 | 0.377 | 1940 | keep | Diverse GBM configs (2 LGB + 2 XGB) |
| 15 | 0.259 | 1528 | discard | Huber loss (catastrophic for Tm2/Titer) |
| 16 | 0.342 | 2006 | discard | Rank-transform targets |
| 17 | 0.337 | 1952 | discard | PCA-Ridge (lost too much info) |
| 18 | 0.377 | 1826 | keep | IgG/LC subtype metadata + z-scored GBMs |
| 19 | 0.382 | 3632 | keep | One-hot features for GBMs |
| 20 | 0.379 | 1985 | discard | Equal blend 1/3 each |
| 21 | 0.383 | 1965 | keep | Ridge-heavy blend 0.5/0.25/0.25 |
| 22 | **0.385** | 2021 | keep | Ridge-heavy blend 0.6/0.2/0.2 |
| 23 | 0.384 | 3139 | discard | Ridge 0.7 (overshot) |
| 24 | 0.384 | 3355 | discard | Remove z-score normalization |
| 25 | 0.382 | 1562 | discard | Tighter GBM regularization |

### Per-target breakdown (best model)

| Target | Spearman |
|---|---|
| PR_CHO | 0.503 |
| HIC | 0.460 |
| AC-SINS pH 7.4 | 0.407 |
| Titer | 0.284 |
| Tm2 | 0.270 |

### Key findings

- **ESM-2 embeddings were the single biggest win.** Going from hand-crafted features to frozen ESM-2 650M embeddings roughly doubled performance (0.151 → 0.284). Including full-length heavy/light chain embeddings (not just variable regions) pushed it further to 0.375.
- **Classical ML > neural nets for this dataset size.** Ridge + gradient boosting ensembles beat the MLP baseline. Fine-tuning ESM-2 directly on 246 samples caused overfitting.
- **Ensemble diversity matters.** Blending Ridge (linear) with LightGBM and XGBoost (tree-based) using different hyperparameter configs gave consistent gains.
- **Ridge regression is the strongest single model.** Increasing Ridge's blend weight from 0.4 to 0.6 steadily improved results, with 0.6 being the sweet spot. Ridge's strong L2 regularization is well-suited to this high-dimensional, small-sample regime.
- **Feature engineering had diminishing returns once ESM was added.** Dipeptide frequencies and extra physicochemical features didn't improve over what ESM already captures. However, one-hot sequence features helped GBMs by providing exact position-AA splits.
- **Target preprocessing is tricky.** Z-score normalization helped GBMs slightly, but Huber loss and rank-transforms both hurt badly. Spearman is rank-based but MSE-trained models still optimize for it effectively.
- **Antibody metadata (IgG subtype, light chain type) provided a small boost** — these structural features are known before any assay and carry developability-relevant information.

## Running the agent

Spin up Claude Code (or similar) in this repo, then prompt:

```
Have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

The agent will establish a baseline, then autonomously iterate on `train.py` — trying different architectures, features, hyperparameters, etc. — logging results to `results.tsv`.

## Project structure

```
prepare.py      — constants, data loading, encoding, evaluation (do not modify)
train.py        — model + training loop (agent modifies this)
modal_run.py    — runs train.py on Modal A100 GPU (do not modify)
program.md      — agent instructions
data/GDPa1.csv  — dataset (246 antibodies, 5 targets)
pyproject.toml  — dependencies
```

## License

MIT
