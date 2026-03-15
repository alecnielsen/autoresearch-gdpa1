# autoresearch-gdpa1

Autonomous model development for predicting antibody developability attributes. Forked from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and adapted from LLM pretraining to multi-task regression on the GDPa1 dataset.

## The idea

Give an AI agent a dataset of 246 therapeutic antibodies with experimentally measured developability properties, a baseline model, and let it experiment autonomously overnight on an H100. It modifies the code, trains, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a much better model.

The agent modifies `train.py` — everything else is fixed. The metric is **mean Spearman correlation** across 5 targets (higher is better).

## Dataset

**GDPa1**: 246 therapeutic antibodies with 5 developability readouts:

| Target | Description | N valid |
|---|---|---|
| HIC | Hydrophobic Interaction Chromatography (hydrophobicity) | 242 |
| Tm2 | Second melting temperature (conformational stability) | 193 |
| PR_CHO | Polyreactivity against CHO cell extract | 197 |
| AC-SINS pH 7.4 | Self-interaction nanoparticle spectroscopy | 242 |
| Titer | Expression titer | 239 |

Inputs are AHo-numbered variable region sequences (heavy + light chain, 149 positions each). Cross-validation uses hierarchical sequence clustering (5 folds) to test generalization to novel sequence families.

## How it works

Three files that matter:

- **`prepare.py`** — fixed constants, data loading, sequence encoding, and evaluation metric (mean Spearman correlation). Not modified by the agent.
- **`train.py`** — the single file the agent edits. Model architecture, features, training loop, hyperparameters. Everything is fair game. **This file is edited and iterated on by the agent.**
- **`program.md`** — instructions for the agent. **This file is edited and iterated on by the human.**

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), [Modal](https://modal.com/) account (for H100 GPU execution).

```bash
# 1. Install dependencies
uv sync

# 2. Authenticate with Modal (one-time)
modal setup

# 3. Verify data and encoding (one-time sanity check)
uv run prepare.py

# 4. Run a single training experiment on H100
modal run modal_run.py
```

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
modal_run.py    — runs train.py on Modal H100 GPU (do not modify)
program.md      — agent instructions
data/GDPa1.csv  — dataset (246 antibodies, 5 targets)
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. Keeps scope manageable and diffs reviewable.
- **1-hour time budget per experiment.** Generous enough for expensive approaches (protein language model fine-tuning, large ensembles) while simple models finish in seconds.
- **Hierarchical cluster CV.** Tests generalization to new sequence families, not just held-out members of known families.
- **Mean Spearman correlation.** Rank-based, robust to scale differences across targets, handles missing values per-target.

## License

MIT
