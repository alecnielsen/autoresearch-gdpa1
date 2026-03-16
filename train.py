"""
Antibody developability prediction — ESM-2 embeddings + Ridge + LightGBM ensemble.

Uses ESM-2 (esm2_t6_8M_UR50D, smallest variant) to extract per-antibody
embeddings, combined with physicochemical features. Ridge + LightGBM ensemble.

Usage: uv run train.py
"""

import time
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
import lightgbm as lgb

from prepare import (
    load_data, encode_sequences, get_targets, get_fold_splits, evaluate,
    TARGET_COLS, N_FOLDS, TIME_BUDGET, ONEHOT_DIM,
    AMINO_ACIDS, GAP_CHAR, SEQ_LEN_PER_CHAIN,
)

# ---------------------------------------------------------------------------
# Physicochemical properties per amino acid
# ---------------------------------------------------------------------------

HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
    '-': 0.0,
}

MOL_WEIGHT = {
    'A': 89, 'C': 121, 'D': 133, 'E': 147, 'F': 165,
    'G': 75, 'H': 155, 'I': 131, 'K': 146, 'L': 131,
    'M': 149, 'N': 132, 'P': 115, 'Q': 146, 'R': 174,
    'S': 105, 'T': 119, 'V': 117, 'W': 204, 'Y': 181,
    '-': 0.0,
}

CHARGE = {
    'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
    'G': 0, 'H': 0.1, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
    'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0,
    '-': 0,
}

POLARITY = {
    'A': 8.1, 'C': 5.5, 'D': 13.0, 'E': 12.3, 'F': 5.2,
    'G': 9.0, 'H': 10.4, 'I': 5.2, 'K': 11.3, 'L': 4.9,
    'M': 5.7, 'N': 11.6, 'P': 8.0, 'Q': 10.5, 'R': 10.5,
    'S': 9.2, 'T': 8.6, 'V': 5.9, 'W': 5.4, 'Y': 6.2,
    '-': 0.0,
}

BETA_SHEET = {
    'A': 0.83, 'C': 1.19, 'D': 0.54, 'E': 0.37, 'F': 1.38,
    'G': 0.75, 'H': 0.87, 'I': 1.60, 'K': 0.74, 'L': 1.30,
    'M': 1.05, 'N': 0.89, 'P': 0.55, 'Q': 1.10, 'R': 0.93,
    'S': 0.75, 'T': 1.19, 'V': 1.70, 'W': 1.37, 'Y': 1.47,
    '-': 0.0,
}

PROPERTY_DICTS = [HYDROPHOBICITY, MOL_WEIGHT, CHARGE, POLARITY, BETA_SHEET]
N_PROPERTIES = len(PROPERTY_DICTS)


def encode_physicochemical(df):
    n = len(df)
    n_feat = 2 * SEQ_LEN_PER_CHAIN * N_PROPERTIES
    X = np.zeros((n, n_feat), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        for chain_idx, col in enumerate(["heavy_aligned_aho", "light_aligned_aho"]):
            seq = row[col]
            offset = chain_idx * SEQ_LEN_PER_CHAIN * N_PROPERTIES
            for pos, aa in enumerate(seq):
                for p, prop_dict in enumerate(PROPERTY_DICTS):
                    X[i, offset + pos * N_PROPERTIES + p] = prop_dict.get(aa, 0.0)
    return X


def encode_aa_composition(df):
    alphabet = GAP_CHAR + AMINO_ACIDS
    n = len(df)
    n_feat = 2 * len(alphabet)
    X = np.zeros((n, n_feat), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        for chain_idx, col in enumerate(["heavy_aligned_aho", "light_aligned_aho"]):
            seq = row[col]
            offset = chain_idx * len(alphabet)
            for aa in seq:
                idx = alphabet.find(aa)
                if idx >= 0:
                    X[i, offset + idx] += 1.0
            X[i, offset:offset + len(alphabet)] /= max(len(seq), 1)
    return X


def encode_summary_stats(df):
    n = len(df)
    n_stats = 2 * (N_PROPERTIES * 2 + 3)
    X = np.zeros((n, n_stats), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        for chain_idx, col in enumerate(["heavy_aligned_aho", "light_aligned_aho"]):
            seq = row[col]
            offset = chain_idx * (N_PROPERTIES * 2 + 3)
            non_gap = [aa for aa in seq if aa != '-']
            if not non_gap:
                continue
            for p, prop_dict in enumerate(PROPERTY_DICTS):
                vals = [prop_dict.get(aa, 0.0) for aa in non_gap]
                X[i, offset + p * 2] = np.mean(vals)
                X[i, offset + p * 2 + 1] = np.std(vals)
            X[i, offset + N_PROPERTIES * 2] = sum(CHARGE.get(aa, 0) for aa in non_gap)
            X[i, offset + N_PROPERTIES * 2 + 1] = sum(1 for aa in non_gap if aa in 'FWY') / len(non_gap)
            X[i, offset + N_PROPERTIES * 2 + 2] = len(non_gap)
    return X


def encode_esm2(df, device):
    """Extract ESM-2 mean-pooled embeddings for VH and VL sequences."""
    import esm

    print("Loading ESM-2 model (esm2_t6_8M_UR50D)...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    embed_dim = 320  # t6_8M embedding dimension

    n = len(df)
    X = np.zeros((n, 2 * embed_dim), dtype=np.float32)

    with torch.no_grad():
        for chain_idx, col in enumerate(["vh_protein_sequence", "vl_protein_sequence"]):
            print(f"  Encoding {col}...")
            sequences = [(f"seq_{i}", row[col]) for i, (_, row) in enumerate(df.iterrows())]

            # Process in batches to avoid OOM
            batch_size = 32
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = sequences[start:end]
                _, _, tokens = batch_converter(batch)
                tokens = tokens.to(device)
                results = model(tokens, repr_layers=[6])
                representations = results["representations"][6]

                # Mean pool over sequence length (exclude BOS/EOS tokens)
                for j in range(end - start):
                    seq_len = len(batch[j][1])
                    emb = representations[j, 1:seq_len+1].mean(dim=0).cpu().numpy()
                    X[start + j, chain_idx * embed_dim:(chain_idx + 1) * embed_dim] = emb

    print(f"  ESM-2 embeddings: {X.shape}")
    return X


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: ESM-2 + Ridge + LightGBM ensemble")
    print(f"Device: {device}")
    print(f"Time budget: {TIME_BUDGET}s")
    print()

    # Load and encode
    df = load_data()
    X_onehot = encode_sequences(df)
    X_physchem = encode_physicochemical(df)
    X_composition = encode_aa_composition(df)
    X_summary = encode_summary_stats(df)
    X_esm = encode_esm2(df, device)

    # Features for each model
    X_ridge = np.hstack([X_onehot, X_physchem, X_composition, X_summary, X_esm])
    X_gbm = np.hstack([X_physchem, X_composition, X_summary, X_esm])

    Y = get_targets(df)
    n_targets = Y.shape[1]
    print(f"Data: {X_ridge.shape[0]} samples")
    print(f"  Ridge features: {X_ridge.shape[1]}")
    print(f"  GBM features:   {X_gbm.shape[1]}")
    print()

    all_preds = np.full_like(Y, np.nan)
    alphas = np.logspace(-2, 6, 50)

    lgb_params = {
        'objective': 'regression',
        'metric': 'mse',
        'verbosity': -1,
        'n_estimators': 500,
        'learning_rate': 0.05,
        'num_leaves': 15,
        'min_child_samples': 10,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'max_depth': 4,
    }

    for fold in range(N_FOLDS):
        train_idx, val_idx = get_fold_splits(df, fold)
        print(f"Fold {fold}: {len(train_idx)} train, {len(val_idx)} val")

        # Standardize for Ridge
        mu = X_ridge[train_idx].mean(axis=0)
        std = X_ridge[train_idx].std(axis=0)
        std[std < 1e-8] = 1.0
        X_tr_s = (X_ridge[train_idx] - mu) / std
        X_va_s = (X_ridge[val_idx] - mu) / std

        ridge_preds = np.zeros((len(val_idx), n_targets), dtype=np.float32)
        gbm_preds = np.zeros((len(val_idx), n_targets), dtype=np.float32)

        for j in range(n_targets):
            mask = ~np.isnan(Y[train_idx, j])
            if mask.sum() < 5:
                continue

            ridge = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error")
            ridge.fit(X_tr_s[mask], Y[train_idx[mask], j])
            ridge_preds[:, j] = ridge.predict(X_va_s)

            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(X_gbm[train_idx[mask]], Y[train_idx[mask], j])
            gbm_preds[:, j] = model.predict(X_gbm[val_idx])

        all_preds[val_idx] = 0.6 * ridge_preds + 0.4 * gbm_preds

    print()

    mean_spearman, per_target = evaluate(Y, all_preds)

    t_end = time.time()
    total_time = t_end - t_start

    print("---")
    print(f"mean_spearman:    {mean_spearman:.6f}")
    for name, rho in per_target.items():
        print(f"  {name:20s}: {rho:.4f}")
    print(f"training_seconds: {total_time:.1f}")
    print(f"model:            ESM-2(t6_8M) + Ridge + LightGBM ensemble (0.6/0.4)")


if __name__ == "__main__":
    main()
