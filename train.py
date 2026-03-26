"""
Antibody developability prediction — ESM-2 650M + Ridge + LightGBM + XGBoost ensemble.

Uses ESM-2 embeddings from both variable and full-length chains.
Three-model ensemble with optimized blending.

Usage: uv run train.py
"""

import time
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
import lightgbm as lgb
import xgboost as xgb

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


def encode_dipeptide(df):
    """Dipeptide (2-mer) frequency features for heavy and light chains."""
    alphabet = AMINO_ACIDS  # 20 AAs (skip gaps)
    n_dipeptides = len(alphabet) ** 2  # 400
    n = len(df)
    X = np.zeros((n, 2 * n_dipeptides), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        for chain_idx, col in enumerate(["heavy_aligned_aho", "light_aligned_aho"]):
            seq = row[col].replace('-', '')  # remove gaps
            offset = chain_idx * n_dipeptides
            for k in range(len(seq) - 1):
                a1, a2 = seq[k], seq[k+1]
                idx1 = alphabet.find(a1)
                idx2 = alphabet.find(a2)
                if idx1 >= 0 and idx2 >= 0:
                    X[i, offset + idx1 * len(alphabet) + idx2] += 1.0
            # Normalize to frequencies
            total = max(len(seq) - 1, 1)
            X[i, offset:offset + n_dipeptides] /= total
    return X


def encode_chain_interactions(df):
    """Cross-chain interaction features: differences and products of H/L chain properties."""
    n = len(df)
    # Per property: H-L difference in mean, H-L difference in std, product of means
    n_feat = N_PROPERTIES * 3 + 4  # 5*3 + 4 extra
    X = np.zeros((n, n_feat), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        h_seq = [aa for aa in row["heavy_aligned_aho"] if aa != '-']
        l_seq = [aa for aa in row["light_aligned_aho"] if aa != '-']
        if not h_seq or not l_seq:
            continue
        for p, prop_dict in enumerate(PROPERTY_DICTS):
            h_vals = [prop_dict.get(aa, 0.0) for aa in h_seq]
            l_vals = [prop_dict.get(aa, 0.0) for aa in l_seq]
            h_mean, l_mean = np.mean(h_vals), np.mean(l_vals)
            h_std, l_std = np.std(h_vals), np.std(l_vals)
            X[i, p * 3] = h_mean - l_mean
            X[i, p * 3 + 1] = h_std - l_std
            X[i, p * 3 + 2] = h_mean * l_mean
        # Extra: total charge difference, length ratio, aromatic fraction diff
        h_charge = sum(CHARGE.get(aa, 0) for aa in h_seq)
        l_charge = sum(CHARGE.get(aa, 0) for aa in l_seq)
        X[i, N_PROPERTIES * 3] = h_charge - l_charge
        X[i, N_PROPERTIES * 3 + 1] = len(h_seq) / max(len(l_seq), 1)
        h_arom = sum(1 for aa in h_seq if aa in 'FWY') / len(h_seq)
        l_arom = sum(1 for aa in l_seq if aa in 'FWY') / len(l_seq)
        X[i, N_PROPERTIES * 3 + 2] = h_arom - l_arom
        X[i, N_PROPERTIES * 3 + 3] = h_arom * l_arom
    return X


def encode_hydrophobic_patches(df):
    """Hydrophobic patch features: max/mean consecutive hydrophobic runs."""
    HYDROPHOBIC = set('AILMFWVP')
    n = len(df)
    X = np.zeros((n, 6), dtype=np.float32)  # 3 features per chain
    for i, (_, row) in enumerate(df.iterrows()):
        for chain_idx, col in enumerate(["heavy_aligned_aho", "light_aligned_aho"]):
            seq = row[col]
            offset = chain_idx * 3
            non_gap = [aa for aa in seq if aa != '-']
            if not non_gap:
                continue
            # Hydrophobic runs
            runs = []
            cur = 0
            for aa in non_gap:
                if aa in HYDROPHOBIC:
                    cur += 1
                else:
                    if cur > 0:
                        runs.append(cur)
                    cur = 0
            if cur > 0:
                runs.append(cur)
            X[i, offset] = max(runs) if runs else 0  # max run
            X[i, offset + 1] = np.mean(runs) if runs else 0  # mean run
            X[i, offset + 2] = sum(1 for aa in non_gap if aa in HYDROPHOBIC) / len(non_gap)  # fraction
    return X


def encode_charge_features(df):
    """Extended charge features: net charge, charge asymmetry, +/- patches."""
    n = len(df)
    X = np.zeros((n, 10), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        for chain_idx, col in enumerate(["heavy_aligned_aho", "light_aligned_aho"]):
            seq = row[col]
            offset = chain_idx * 5
            non_gap = [aa for aa in seq if aa != '-']
            if not non_gap:
                continue
            pos_count = sum(1 for aa in non_gap if aa in 'KRH')
            neg_count = sum(1 for aa in non_gap if aa in 'DE')
            total = len(non_gap)
            X[i, offset] = (pos_count - neg_count) / total  # net charge density
            X[i, offset + 1] = pos_count / total  # positive fraction
            X[i, offset + 2] = neg_count / total  # negative fraction
            # Charge patches: max consecutive same-sign residues
            max_pos_run = max_neg_run = cur_pos = cur_neg = 0
            for aa in non_gap:
                if aa in 'KRH':
                    cur_pos += 1
                    max_pos_run = max(max_pos_run, cur_pos)
                    cur_neg = 0
                elif aa in 'DE':
                    cur_neg += 1
                    max_neg_run = max(max_neg_run, cur_neg)
                    cur_pos = 0
                else:
                    cur_pos = cur_neg = 0
            X[i, offset + 3] = max_pos_run
            X[i, offset + 4] = max_neg_run
    return X


def encode_gap_pattern(df):
    """Gap pattern features: number of gap blocks, longest gap, gap fraction per chain."""
    n = len(df)
    X = np.zeros((n, 8), dtype=np.float32)  # 4 features per chain
    for i, (_, row) in enumerate(df.iterrows()):
        for chain_idx, col in enumerate(["heavy_aligned_aho", "light_aligned_aho"]):
            seq = row[col]
            offset = chain_idx * 4
            gaps = [c == '-' for c in seq]
            n_gaps = sum(gaps)
            X[i, offset] = n_gaps / len(seq)  # gap fraction
            X[i, offset + 1] = len(seq) - n_gaps  # non-gap length
            # Count gap blocks
            n_blocks = 0
            max_block = 0
            cur_block = 0
            for g in gaps:
                if g:
                    cur_block += 1
                else:
                    if cur_block > 0:
                        n_blocks += 1
                        max_block = max(max_block, cur_block)
                        cur_block = 0
            if cur_block > 0:
                n_blocks += 1
                max_block = max(max_block, cur_block)
            X[i, offset + 2] = n_blocks
            X[i, offset + 3] = max_block
    return X


def encode_codon_usage(df):
    """Codon usage frequency features from DNA sequences (HC + LC)."""
    codons = []
    for a in 'ACGT':
        for b in 'ACGT':
            for c in 'ACGT':
                codons.append(a + b + c)
    codon_to_idx = {c: i for i, c in enumerate(codons)}
    n_codons = len(codons)  # 64
    n = len(df)
    X = np.zeros((n, 2 * n_codons), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        for chain_idx, col in enumerate(["hc_dna_sequence", "lc_dna_sequence"]):
            dna = str(row[col]).upper().replace(' ', '')
            offset = chain_idx * n_codons
            count = 0
            for k in range(0, len(dna) - 2, 3):
                codon = dna[k:k+3]
                if codon in codon_to_idx:
                    X[i, offset + codon_to_idx[codon]] += 1.0
                    count += 1
            if count > 0:
                X[i, offset:offset + n_codons] /= count
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
    """Extract ESM-2 mean-pooled embeddings for VH, VL, full HC, full LC."""
    import esm

    print("Loading ESM-2 model (esm2_t33_650M_UR50D)...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    embed_dim = 1280

    # 4 chains: VH, VL, full HC, full LC
    chain_cols = [
        "vh_protein_sequence", "vl_protein_sequence",
        "hc_protein_sequence", "lc_protein_sequence",
    ]
    n = len(df)
    X = np.zeros((n, len(chain_cols) * embed_dim), dtype=np.float32)

    with torch.no_grad():
        for chain_idx, col in enumerate(chain_cols):
            print(f"  Encoding {col}...")
            # Strip stop codons (*) and any non-standard characters
            sequences = [(f"seq_{i}", row[col].replace("*", "")) for i, (_, row) in enumerate(df.iterrows())]

            batch_size = 16 if "hc_" in col or "lc_" in col else 32
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = sequences[start:end]
                _, _, tokens = batch_converter(batch)
                tokens = tokens.to(device)
                results = model(tokens, repr_layers=[33])
                representations = results["representations"][33]

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
    print(f"Model: ESM-2(650M) + Ridge + LightGBM + XGBoost ensemble")
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

    # Encode antibody metadata (IgG subtype, light chain type)
    X_meta = np.zeros((len(df), 5), dtype=np.float32)  # IgG1, IgG2, IgG4, Kappa, Lambda
    for i, (_, row) in enumerate(df.iterrows()):
        if row['hc_subtype'] == 'IgG1': X_meta[i, 0] = 1
        elif row['hc_subtype'] == 'IgG2': X_meta[i, 1] = 1
        elif row['hc_subtype'] == 'IgG4': X_meta[i, 2] = 1
        if row['lc_subtype'] == 'Kappa': X_meta[i, 3] = 1
        elif row['lc_subtype'] == 'Lambda': X_meta[i, 4] = 1
    print(f"  Metadata features: {X_meta.shape[1]}")

    X_dipeptide = encode_dipeptide(df)
    print(f"  Dipeptide features: {X_dipeptide.shape[1]}")

    X_interactions = encode_chain_interactions(df)
    print(f"  Chain interaction features: {X_interactions.shape[1]}")

    X_hydro = encode_hydrophobic_patches(df)
    print(f"  Hydrophobic patch features: {X_hydro.shape[1]}")
    X_charge = encode_charge_features(df)
    print(f"  Charge features: {X_charge.shape[1]}")
    X_gaps = encode_gap_pattern(df)
    print(f"  Gap pattern features: {X_gaps.shape[1]}")
    X_codon = encode_codon_usage(df)
    print(f"  Codon usage features: {X_codon.shape[1]}")

    X_ridge = np.hstack([X_composition, X_summary, X_dipeptide, X_interactions, X_hydro, X_charge, X_gaps, X_codon, X_esm, X_meta])
    X_gbm = np.hstack([X_onehot, X_composition, X_summary, X_esm, X_meta])
    # Tm2-specific GBM features: add composition + physchem back (Tm2 is pure GBM)
    X_gbm_tm2 = np.hstack([X_onehot, X_physchem, X_composition, X_summary, X_esm, X_meta])

    Y = get_targets(df)
    n_targets = Y.shape[1]
    print(f"Data: {X_ridge.shape[0]} samples")
    print(f"  Ridge features: {X_ridge.shape[1]}")
    print(f"  GBM features:   {X_gbm.shape[1]}")
    print()

    all_preds = np.full_like(Y, np.nan)
    alphas = np.logspace(-2, 6, 50)

    lgb_params = {
        'objective': 'regression', 'metric': 'mse', 'verbosity': -1,
        'n_estimators': 500, 'learning_rate': 0.05,
        'num_leaves': 15, 'min_child_samples': 10,
        'reg_alpha': 1.0, 'reg_lambda': 1.0,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'max_depth': 4,
    }

    xgb_params = {
        'objective': 'reg:squarederror', 'verbosity': 0,
        'n_estimators': 500, 'learning_rate': 0.05,
        'max_depth': 4, 'min_child_weight': 10,
        'reg_alpha': 1.0, 'reg_lambda': 1.0,
        'subsample': 0.8, 'colsample_bytree': 0.8,
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
        lgb_preds = np.zeros((len(val_idx), n_targets), dtype=np.float32)
        xgb_preds = np.zeros((len(val_idx), n_targets), dtype=np.float32)

        for j in range(n_targets):
            mask = ~np.isnan(Y[train_idx, j])
            if mask.sum() < 5:
                continue

            y_tr = Y[train_idx[mask], j]

            # Z-score targets for GBMs (standardize per fold)
            y_mean = y_tr.mean()
            y_std = y_tr.std()
            if y_std < 1e-8:
                y_std = 1.0
            y_tr_z = (y_tr - y_mean) / y_std

            # Ridge on rank-transformed targets (aligned with Spearman metric)
            from scipy.stats import rankdata
            y_tr_rank = rankdata(y_tr).astype(np.float32)
            ridge = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error")
            ridge.fit(X_tr_s[mask], y_tr_rank)
            ridge_preds[:, j] = ridge.predict(X_va_s)

            # Select GBM features (Tm2 gets enriched feature set)
            X_gbm_j = X_gbm_tm2 if j == 1 else X_gbm

            # LightGBM on z-scored targets
            lgb_model1 = lgb.LGBMRegressor(**lgb_params)
            lgb_model1.fit(X_gbm_j[train_idx[mask]], y_tr_z)

            # AC-SINS gets less diverse config 2 (depth 5), others get depth 6
            if j == 3:  # AC-SINS
                lgb_p2 = {**lgb_params, 'num_leaves': 20, 'max_depth': 5,
                           'min_child_samples': 8, 'colsample_bytree': 0.7}
            else:
                lgb_p2 = {**lgb_params, 'num_leaves': 31, 'max_depth': 6,
                           'min_child_samples': 5, 'colsample_bytree': 0.6}
            lgb_model2 = lgb.LGBMRegressor(**lgb_p2)
            lgb_model2.fit(X_gbm_j[train_idx[mask]], y_tr_z)

            # Inverse z-score transform
            lgb_preds[:, j] = 0.5 * (lgb_model1.predict(X_gbm_j[val_idx]) +
                                       lgb_model2.predict(X_gbm_j[val_idx])) * y_std + y_mean

            # XGBoost on z-scored targets
            xgb_model1 = xgb.XGBRegressor(**xgb_params)
            xgb_model1.fit(X_gbm_j[train_idx[mask]], y_tr_z)

            if j == 3:  # AC-SINS
                xgb_p2 = {**xgb_params, 'max_depth': 5, 'min_child_weight': 8,
                           'colsample_bytree': 0.7}
            else:
                xgb_p2 = {**xgb_params, 'max_depth': 6, 'min_child_weight': 5,
                           'colsample_bytree': 0.6}
            xgb_model2 = xgb.XGBRegressor(**xgb_p2)
            xgb_model2.fit(X_gbm_j[train_idx[mask]], y_tr_z)

            xgb_preds[:, j] = 0.5 * (xgb_model1.predict(X_gbm_j[val_idx]) +
                                       xgb_model2.predict(X_gbm_j[val_idx])) * y_std + y_mean

        # Per-target blend weights (Ridge stronger for HIC/PR_CHO, GBMs for Tm2/Titer)
        # Target order: HIC, Tm2, PR_CHO, AC-SINS_pH7.4, Titer
        ridge_w = np.array([1.0, 0.0, 1.0, 0.7, 0.9])
        gbm_w = (1.0 - ridge_w) / 2.0
        for j in range(n_targets):
            all_preds[val_idx, j] = (ridge_w[j] * ridge_preds[:, j] +
                                      gbm_w[j] * lgb_preds[:, j] +
                                      gbm_w[j] * xgb_preds[:, j])

    print()

    mean_spearman, per_target = evaluate(Y, all_preds)

    t_end = time.time()
    total_time = t_end - t_start

    print("---")
    print(f"mean_spearman:    {mean_spearman:.6f}")
    for name, rho in per_target.items():
        print(f"  {name:20s}: {rho:.4f}")
    print(f"training_seconds: {total_time:.1f}")
    print(f"model:            ESM-2(t33_650M) + metadata + onehot-GBM + z-scored")


if __name__ == "__main__":
    main()
