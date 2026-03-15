# autoresearch: antibody developability prediction

This is an experiment to have the LLM do its own research on predicting antibody developability properties.

## Background

The dataset (GDPa1) contains 246 therapeutic antibodies with 5 developability readouts:
- **HIC** — Hydrophobic Interaction Chromatography (hydrophobicity)
- **Tm2** — Second melting temperature (conformational stability)
- **PR_CHO** — Polyreactivity against CHO cell extract
- **AC-SINS** pH 7.4 — Self-interaction nanoparticle spectroscopy
- **Titer** — Expression titer

Inputs are AHo-numbered variable region sequences (heavy + light, 149 positions each, fixed-length). The alphabet is 20 standard amino acids + gap character.

Cross-validation uses `hierarchical_cluster_fold` (5 folds), which clusters antibodies by sequence similarity. This is more realistic than random splits because it tests generalization to new sequence families.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar15`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `program.md` — this file, your instructions.
   - `prepare.py` — fixed constants, data loading, sequence encoding, evaluation. **Do not modify.**
   - `train.py` — the file you modify. Model, features, training loop, hyperparameters.
4. **Verify data exists**: Check that `data/GDPa1.csv` exists.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs cross-validation across 5 folds with a **fixed time budget of 5 minutes** (wall clock). You launch it as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, features (you can use other sequence columns from the CSV), training procedure, hyperparameters, regularization, ensembling, feature engineering, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation metric, data loading, sequence encoding, and constants.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml` (torch, numpy, pandas, scipy, etc.).
- Modify the evaluation function. The `evaluate()` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest mean_spearman.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game within train.py.

**Key domain context for ideas:**
- AHo numbering aligns CDR and framework positions across antibodies. CDR positions are the most variable and often most predictive.
- Position-specific amino acid preferences differ between frameworks and CDRs.
- Physicochemical properties of amino acids (hydrophobicity, charge, size) are often more predictive than raw identity.
- The dataset is small (246 samples) so regularization matters a lot. Simpler models may beat complex ones.
- Missing target values vary by assay (some targets have ~50 missing). The current approach handles this with masked loss.
- You have access to full heavy/light chain protein and DNA sequences in the CSV, not just variable regions.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
mean_spearman:    0.234567
  HIC                 : 0.3456
  Tm2                 : 0.2345
  PR_CHO              : 0.1234
  AC-SINS_pH7.4       : 0.2567
  Titer               : 0.2132
training_seconds: 42.3
model:            MultiTaskMLP [512, 256]
```

You can extract the key metric:

```
grep "^mean_spearman:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	mean_spearman	time_seconds	status	description
```

1. git commit hash (short, 7 chars)
2. mean_spearman achieved (e.g. 0.234567) — use 0.000000 for crashes
3. training time in seconds (e.g. 42.3) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	mean_spearman	time_seconds	status	description
a1b2c3d	0.234567	42.3	keep	baseline MLP [512 256]
b2c3d4e	0.278900	38.7	keep	add AA physicochemical features
c3d4e5f	0.215000	55.2	discard	deeper MLP [512 512 256 128]
d4e5f6g	0.000000	0.0	crash	transformer encoder (shape mismatch)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar15`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^mean_spearman:\|^training_seconds:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If mean_spearman improved (higher), you "advance" the branch, keeping the git commit
9. If mean_spearman is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout**: Each experiment should take ~5 minutes total. If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes, use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical approaches, try feature engineering ideas from the domain context above. The loop runs until the human interrupts you, period.

**Ideas to explore** (non-exhaustive):
- Feature engineering: physicochemical properties per position, k-mer features, CDR vs framework distinction
- Alternative encodings: learned embeddings, BLOSUM62-based features, AAindex properties
- Model architectures: Ridge/Lasso regression, random forests, gradient boosting, CNN on sequence, attention over positions
- Regularization: stronger dropout, weight decay, early stopping tuning, data augmentation
- Multi-task strategies: shared vs separate encoders, task weighting, curriculum
- Ensemble methods: bagging, blending fold models
- Using additional sequence information from CSV (full HC/LC, DNA sequences)
