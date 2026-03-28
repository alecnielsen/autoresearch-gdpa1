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

## Results

### Overall progression

The agent ran **41+ experiments across 3 sessions**, improving mean Spearman from **0.119 to 0.452** (3.8x improvement).

| Session | Experiments | Start | End | Key breakthrough |
|---|---|---|---|---|
| `mar15` | 76 | 0.119 | 0.423 | ESM-2 embeddings, per-target specialization |
| `mar23` | 41+ | 0.421 | **0.452** | Rank-transformed Ridge, KernelRidge for Titer, SVR ensemble |

### Key milestones

| Mean Spearman | Description |
|--------------|-------------|
| 0.119 | Baseline MLP [512, 256] |
| 0.151 | RidgeCV + physicochemical features |
| 0.228 | Ridge + LightGBM ensemble |
| 0.284 | ESM-2 650M embeddings |
| 0.375 | Full HC/LC chain embeddings + XGBoost |
| 0.404 | Feature selection: drop noisy features per model |
| 0.423 | Per-target blend weights + GBM specialization |
| 0.430 | Dipeptide frequency features |
| 0.434 | Codon usage, gap pattern, charge, hydrophobic patch features |
| 0.449 | **Rank-transformed Ridge targets** (biggest single-experiment gain) |
| 0.451 | KernelRidge (RBF) for Titer prediction |
| **0.452** | **SVR (RBF) as 4th ensemble model** |

### Per-target breakdown (best model)

| Target | Spearman | Strategy |
|---|---|---|
| PR_CHO | 0.562 | 85% rank Ridge + 15% SVR |
| HIC | 0.529 | 85% rank Ridge + 15% SVR |
| AC-SINS pH 7.4 | 0.488 | 60% rank Ridge + 10% SVR + 15% LGB + 15% XGB |
| Tm2 | 0.348 | Pure GBM (enriched features) |
| Titer | 0.334 | 63% rank Ridge+KernelRidge + 14% SVR + 12% GBM |

### Key findings

- **ESM-2 embeddings were the single biggest win.** Going from hand-crafted features to frozen ESM-2 650M embeddings roughly doubled performance (0.151 → 0.284). Including full-length heavy/light chain embeddings (not just variable regions) pushed it further to 0.375.
- **Classical ML > neural nets for this dataset size.** Ridge + gradient boosting ensembles beat the MLP baseline. Fine-tuning ESM-2 directly on 246 samples caused overfitting.
- **Rank-transforming Ridge targets was the second biggest win.** Since Spearman measures rank correlation, training Ridge on ranks directly aligns the objective. This single change boosted mean Spearman from 0.434 to 0.449, with AC-SINS improving by +0.055.
- **Per-target specialization is essential.** Each target has its own optimal model mix. Tm2 needs pure GBMs with enriched features. HIC and PR_CHO are pure Ridge. AC-SINS and Titer benefit from Ridge-heavy blends with GBM diversity.
- **KernelRidge captures nonlinear patterns that linear Ridge misses** — but only for Titer. Applying it to other targets hurt performance, suggesting Titer has uniquely nonlinear structure.
- **DNA-level features (codon usage) help Ridge** but not GBMs, likely because codon bias correlates with expression (Titer) and GBMs already capture this signal from ESM embeddings.
- **Model diversity matters more than individual model tuning.** Adding SVR as a 4th model type (alongside Ridge, LGB, XGB) improved scores even with a small weight (15%). Each model captures different patterns in the data.
- **Feature engineering has diminishing returns.** Each new hand-crafted feature (dipeptide, charge patches, hydrophobic patches, gap patterns) added ~0.001 to the score. The big gains came from algorithmic changes (rank transform, KernelRidge, SVR).

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
