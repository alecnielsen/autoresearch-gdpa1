"""
Fixed data preparation and evaluation for antibody developability prediction.

This file is READ-ONLY for the agent. It contains:
- Constants (targets, time budget, encoding)
- Data loading and CV fold management
- Sequence encoding (one-hot AHo-aligned)
- Evaluation function (mean Spearman correlation across 5 targets)

Usage:
    # As a module (imported by train.py):
    from prepare import load_data, encode_sequences, evaluate, TARGET_COLS, ...

    # Standalone sanity check:
    python prepare.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TARGET_COLS = ["HIC", "Tm2", "PR_CHO", "AC-SINS_pH7.4", "Titer"]
FOLD_COL = "hierarchical_cluster_fold"
N_FOLDS = 5
TIME_BUDGET = 3600  # training time budget in seconds (1 hour per experiment)

# Sequence encoding constants
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids
GAP_CHAR = "-"
VOCAB = GAP_CHAR + AMINO_ACIDS  # 21 characters: gap + 20 AAs
VOCAB_SIZE = len(VOCAB)  # 21
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}

# AHo-aligned sequences: 149 positions each for heavy and light
SEQ_LEN_PER_CHAIN = 149
TOTAL_SEQ_LEN = 2 * SEQ_LEN_PER_CHAIN  # 298 positions
ONEHOT_DIM = TOTAL_SEQ_LEN * VOCAB_SIZE  # 298 * 21 = 6258 features

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_FILE = os.path.join(DATA_DIR, "GDPa1.csv")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """
    Load the GDPa1 dataset.

    Returns:
        df: DataFrame with all columns
    """
    df = pd.read_csv(DATA_FILE)
    # Ensure fold column is integer
    df[FOLD_COL] = df[FOLD_COL].astype(int)
    return df


def get_fold_splits(df, val_fold):
    """
    Split data into train and validation sets based on fold column.

    Args:
        df: DataFrame with fold column
        val_fold: integer fold index (0 to N_FOLDS-1) to use as validation

    Returns:
        train_idx: numpy array of training row indices
        val_idx: numpy array of validation row indices
    """
    assert 0 <= val_fold < N_FOLDS, f"val_fold must be 0-{N_FOLDS-1}, got {val_fold}"
    val_mask = df[FOLD_COL] == val_fold
    val_idx = np.where(val_mask)[0]
    train_idx = np.where(~val_mask)[0]
    return train_idx, val_idx

# ---------------------------------------------------------------------------
# Sequence encoding
# ---------------------------------------------------------------------------

def encode_sequence_onehot(seq):
    """
    One-hot encode a single aligned sequence string.

    Args:
        seq: string of length SEQ_LEN_PER_CHAIN (149 chars, AHo-aligned)

    Returns:
        numpy array of shape (SEQ_LEN_PER_CHAIN * VOCAB_SIZE,) = (3129,)
    """
    assert len(seq) == SEQ_LEN_PER_CHAIN, f"Expected length {SEQ_LEN_PER_CHAIN}, got {len(seq)}"
    onehot = np.zeros((SEQ_LEN_PER_CHAIN, VOCAB_SIZE), dtype=np.float32)
    for i, c in enumerate(seq):
        if c in CHAR_TO_IDX:
            onehot[i, CHAR_TO_IDX[c]] = 1.0
        else:
            # Unknown character treated as gap
            onehot[i, 0] = 1.0
    return onehot.flatten()


def encode_sequences(df):
    """
    One-hot encode heavy + light AHo-aligned sequences for all antibodies.

    Args:
        df: DataFrame with 'heavy_aligned_aho' and 'light_aligned_aho' columns

    Returns:
        X: numpy array of shape (n_samples, ONEHOT_DIM) = (n, 6258)
    """
    n = len(df)
    X = np.zeros((n, ONEHOT_DIM), dtype=np.float32)
    half = SEQ_LEN_PER_CHAIN * VOCAB_SIZE  # 3129
    for i, (_, row) in enumerate(df.iterrows()):
        X[i, :half] = encode_sequence_onehot(row["heavy_aligned_aho"])
        X[i, half:] = encode_sequence_onehot(row["light_aligned_aho"])
    return X


def get_targets(df):
    """
    Extract target values as numpy array, with NaN for missing values.

    Args:
        df: DataFrame with TARGET_COLS

    Returns:
        Y: numpy array of shape (n_samples, 5), float32, NaN for missing
    """
    Y = df[TARGET_COLS].values.astype(np.float32)
    return Y

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE -- this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred):
    """
    Compute mean Spearman correlation across the 5 targets.

    For each target, computes Spearman rho on samples where the target is
    not NaN. The final metric is the mean across all targets that have at
    least 2 non-NaN samples.

    Higher is better.

    Args:
        y_true: numpy array of shape (n_samples, 5), may contain NaN
        y_pred: numpy array of shape (n_samples, 5)

    Returns:
        mean_spearman: float, mean Spearman correlation across targets
        per_target: dict mapping target name to Spearman correlation
    """
    assert y_true.shape == y_pred.shape, (
        f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
    )
    assert y_true.shape[1] == len(TARGET_COLS), (
        f"Expected {len(TARGET_COLS)} targets, got {y_true.shape[1]}"
    )

    per_target = {}
    for j, name in enumerate(TARGET_COLS):
        mask = ~np.isnan(y_true[:, j])
        if mask.sum() < 2:
            continue
        rho, _ = spearmanr(y_true[mask, j], y_pred[mask, j])
        per_target[name] = rho

    if len(per_target) == 0:
        return 0.0, per_target

    mean_spearman = np.mean(list(per_target.values()))
    return float(mean_spearman), per_target

# ---------------------------------------------------------------------------
# Main (sanity check)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running prepare.py sanity checks...")
    print()

    # Load data
    df = load_data()
    print(f"Dataset: {len(df)} antibodies, {len(df.columns)} columns")
    print(f"Targets: {TARGET_COLS}")
    print()

    # Check missing values
    Y = get_targets(df)
    for j, name in enumerate(TARGET_COLS):
        n_missing = np.isnan(Y[:, j]).sum()
        n_valid = (~np.isnan(Y[:, j])).sum()
        print(f"  {name:20s}: {n_valid:3d} valid, {n_missing:3d} missing")
    print()

    # Encode sequences
    X = encode_sequences(df)
    print(f"Feature matrix: {X.shape} (dtype={X.dtype})")
    print(f"  Non-zero fraction: {(X != 0).mean():.4f}")
    print()

    # Check folds
    for fold in range(N_FOLDS):
        train_idx, val_idx = get_fold_splits(df, fold)
        print(f"  Fold {fold}: {len(train_idx)} train, {len(val_idx)} val")
    print()

    # Dummy evaluation
    y_pred_random = np.random.randn(*Y.shape).astype(np.float32)
    mean_rho, per_target = evaluate(Y, y_pred_random)
    print(f"Random baseline Spearman: {mean_rho:.4f}")
    for name, rho in per_target.items():
        print(f"  {name:20s}: {rho:.4f}")
    print()

    print("Sanity checks passed. Ready to train.")
