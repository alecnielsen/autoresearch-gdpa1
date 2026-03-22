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

The agent ran 53 experiments, improving mean Spearman from **0.119 to 0.414** (3.5x improvement). Full log in `results.tsv`; key milestones:

| # | Mean Spearman | Status | Description |
|---|--------------|--------|-------------|
| 1 | 0.119 | keep | Baseline MLP [512, 256] |
| 2 | 0.151 | keep | RidgeCV + physicochemical features |
| 3 | 0.228 | keep | Ridge + LightGBM ensemble |
| 5 | 0.284 | keep | ESM-2 650M embeddings |
| 6 | 0.375 | keep | Full HC/LC chain embeddings + XGBoost |
| 14 | 0.377 | keep | Diverse GBM configs (2 LGB + 2 XGB) |
| 22 | 0.385 | keep | Ridge-heavy blend 0.6/0.2/0.2 |
| 33 | 0.386 | keep | Drop one-hot from Ridge |
| 34 | 0.400 | keep | Drop physicochemical from Ridge |
| 38 | 0.403 | keep | Drop physicochemical from GBMs too |
| 40 | 0.404 | keep | Drop composition from GBMs |
| 46 | 0.408 | keep | Per-target blend weights |
| **52** | **0.414** | **keep** | **Optimized per-target Ridge/GBM weights** |

### Per-target breakdown (best model)

| Target | Spearman | Ridge weight |
|---|---|---|
| PR_CHO | 0.541 | 0.7 |
| HIC | 0.508 | 0.7 |
| AC-SINS pH 7.4 | 0.402 | 0.5 |
| Tm2 | 0.334 | 0.0 |
| Titer | 0.287 | 0.3 |

### Key findings

- **ESM-2 embeddings were the single biggest win.** Going from hand-crafted features to frozen ESM-2 650M embeddings roughly doubled performance (0.151 → 0.284). Including full-length heavy/light chain embeddings (not just variable regions) pushed it further to 0.375. Full-length chains are critical — dropping them collapses Tm2 from 0.33 to 0.10.
- **Classical ML > neural nets for this dataset size.** Ridge + gradient boosting ensembles beat the MLP baseline. Fine-tuning ESM-2 directly on 246 samples caused overfitting.
- **Each model type needs different features.** Ridge does best with dense, low-noise features (ESM + composition + summary + metadata). GBMs do best with sparse high-dimensional features (one-hot + ESM + summary + metadata). Physicochemical features hurt both when ESM is present. This realization took us from 0.385 to 0.404.
- **Per-target blend weights were the final breakthrough.** Different targets have radically different optimal model blends. Tm2 (thermal stability) does best with pure GBM (Ridge weight = 0.0), while HIC and PR_CHO do best at 70% Ridge. This took us from 0.404 to 0.414.
- **Simplification was key.** Many of the best experiments involved *removing* features or complexity, not adding it. Less is more when your embeddings are good.

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
