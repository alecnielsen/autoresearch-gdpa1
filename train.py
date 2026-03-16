"""
Antibody developability prediction — Ridge regression with physicochemical features.

Usage: uv run train.py
"""

import time
import numpy as np
from sklearn.linear_model import RidgeCV

from prepare import (
    load_data, encode_sequences, get_targets, get_fold_splits, evaluate,
    TARGET_COLS, N_FOLDS, TIME_BUDGET, ONEHOT_DIM,
    AMINO_ACIDS, GAP_CHAR, SEQ_LEN_PER_CHAIN,
)

# ---------------------------------------------------------------------------
# Physicochemical properties per amino acid
# ---------------------------------------------------------------------------

# Kyte-Doolittle hydrophobicity
HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
    '-': 0.0,
}

# Molecular weight (Da)
MOL_WEIGHT = {
    'A': 89, 'C': 121, 'D': 133, 'E': 147, 'F': 165,
    'G': 75, 'H': 155, 'I': 131, 'K': 146, 'L': 131,
    'M': 149, 'N': 132, 'P': 115, 'Q': 146, 'R': 174,
    'S': 105, 'T': 119, 'V': 117, 'W': 204, 'Y': 181,
    '-': 0.0,
}

# Charge at pH 7
CHARGE = {
    'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
    'G': 0, 'H': 0.1, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
    'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0,
    '-': 0,
}

# Polarity (Grantham)
POLARITY = {
    'A': 8.1, 'C': 5.5, 'D': 13.0, 'E': 12.3, 'F': 5.2,
    'G': 9.0, 'H': 10.4, 'I': 5.2, 'K': 11.3, 'L': 4.9,
    'M': 5.7, 'N': 11.6, 'P': 8.0, 'Q': 10.5, 'R': 10.5,
    'S': 9.2, 'T': 8.6, 'V': 5.9, 'W': 5.4, 'Y': 6.2,
    '-': 0.0,
}

PROPERTY_DICTS = [HYDROPHOBICITY, MOL_WEIGHT, CHARGE, POLARITY]
N_PROPERTIES = len(PROPERTY_DICTS)


def encode_physicochemical(df):
    """Encode sequences as per-position physicochemical properties."""
    n = len(df)
    n_feat = 2 * SEQ_LEN_PER_CHAIN * N_PROPERTIES  # 149 * 4 * 2 = 1192
    X = np.zeros((n, n_feat), dtype=np.float32)

    for i, (_, row) in enumerate(df.iterrows()):
        for chain_idx, col in enumerate(["heavy_aligned_aho", "light_aligned_aho"]):
            seq = row[col]
            offset = chain_idx * SEQ_LEN_PER_CHAIN * N_PROPERTIES
            for pos, aa in enumerate(seq):
                for p, prop_dict in enumerate(PROPERTY_DICTS):
                    feat_idx = offset + pos * N_PROPERTIES + p
                    X[i, feat_idx] = prop_dict.get(aa, 0.0)
    return X


def encode_aa_composition(df):
    """Encode amino acid composition (fraction of each AA in each chain)."""
    alphabet = GAP_CHAR + AMINO_ACIDS  # 21 chars
    n = len(df)
    n_feat = 2 * len(alphabet)  # 42
    X = np.zeros((n, n_feat), dtype=np.float32)

    for i, (_, row) in enumerate(df.iterrows()):
        for chain_idx, col in enumerate(["heavy_aligned_aho", "light_aligned_aho"]):
            seq = row[col]
            offset = chain_idx * len(alphabet)
            for aa in seq:
                idx = alphabet.find(aa)
                if idx >= 0:
                    X[i, offset + idx] += 1.0
            # Normalize to fractions
            X[i, offset:offset + len(alphabet)] /= max(len(seq), 1)
    return X


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("Model: Ridge regression with physicochemical + composition + onehot features")
    print(f"Time budget: {TIME_BUDGET}s")
    print()

    # Load data
    df = load_data()
    X_onehot = encode_sequences(df)
    X_physchem = encode_physicochemical(df)
    X_composition = encode_aa_composition(df)
    X = np.hstack([X_onehot, X_physchem, X_composition])
    Y = get_targets(df)
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, {Y.shape[1]} targets")
    print(f"  onehot: {X_onehot.shape[1]}, physchem: {X_physchem.shape[1]}, composition: {X_composition.shape[1]}")
    print()

    # Cross-validation
    all_preds = np.full_like(Y, np.nan)
    n_targets = Y.shape[1]
    alphas = np.logspace(-2, 6, 50)

    for fold in range(N_FOLDS):
        train_idx, val_idx = get_fold_splits(df, fold)
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        print(f"Fold {fold}: {len(train_idx)} train, {len(val_idx)} val")

        # Standardize features
        mu = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std < 1e-8] = 1.0
        X_train_s = (X_train - mu) / std
        X_val_s = (X_val - mu) / std

        # Train separate Ridge per target (handles missing values)
        fold_preds = np.zeros((len(val_idx), n_targets), dtype=np.float32)
        for j in range(n_targets):
            mask = ~np.isnan(Y_train[:, j])
            if mask.sum() < 5:
                fold_preds[:, j] = 0.0
                continue
            ridge = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error")
            ridge.fit(X_train_s[mask], Y_train[mask, j])
            fold_preds[:, j] = ridge.predict(X_val_s)
            print(f"  Target {TARGET_COLS[j]:20s}: alpha={ridge.alpha_:.1f}")

        all_preds[val_idx] = fold_preds

    print()

    # Final evaluation
    mean_spearman, per_target = evaluate(Y, all_preds)

    t_end = time.time()
    total_time = t_end - t_start

    print("---")
    print(f"mean_spearman:    {mean_spearman:.6f}")
    for name, rho in per_target.items():
        print(f"  {name:20s}: {rho:.4f}")
    print(f"training_seconds: {total_time:.1f}")
    print(f"model:            RidgeCV + physicochemical + composition + onehot")


if __name__ == "__main__":
    main()
