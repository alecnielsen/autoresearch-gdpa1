"""
Antibody developability prediction — baseline model.
Multi-task MLP on one-hot encoded AHo-aligned sequences.

Usage: uv run train.py
"""

import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from prepare import (
    load_data, encode_sequences, get_targets, get_fold_splits, evaluate,
    TARGET_COLS, N_FOLDS, TIME_BUDGET, ONEHOT_DIM,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultiTaskMLP(nn.Module):
    """Simple multi-task MLP for antibody developability prediction."""

    def __init__(self, input_dim, hidden_dims, n_targets, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, n_targets)

    def forward(self, x):
        h = self.backbone(x)
        return self.head(h)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

HIDDEN_DIMS = [512, 256]    # MLP hidden layer sizes
DROPOUT = 0.1               # dropout rate
LEARNING_RATE = 1e-3        # Adam learning rate
WEIGHT_DECAY = 1e-4         # L2 regularization
BATCH_SIZE = 32             # training batch size
N_EPOCHS = 500              # max epochs (will stop early if time runs out)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_fold(X_train, Y_train, X_val, Y_val, device, time_remaining):
    """
    Train model on one fold and return validation predictions.

    Args:
        X_train, Y_train: training data (numpy arrays)
        X_val, Y_val: validation data (numpy arrays)
        device: torch device
        time_remaining: seconds available for this fold

    Returns:
        val_preds: numpy array of shape (n_val, n_targets)
    """
    n_targets = Y_train.shape[1]

    # Convert to tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_tr = torch.tensor(Y_train, dtype=torch.float32, device=device)
    X_v = torch.tensor(X_val, dtype=torch.float32, device=device)

    # Build mask for non-NaN targets (NaN targets are excluded from loss)
    mask_tr = ~torch.isnan(Y_tr)
    # Replace NaN with 0 for computation (masked out in loss anyway)
    Y_tr_filled = torch.where(mask_tr, Y_tr, torch.zeros_like(Y_tr))

    # Per-target normalization (mean/std from training set, non-NaN only)
    target_means = torch.zeros(n_targets, device=device)
    target_stds = torch.ones(n_targets, device=device)
    for j in range(n_targets):
        valid = mask_tr[:, j]
        if valid.sum() > 1:
            target_means[j] = Y_tr_filled[valid, j].mean()
            target_stds[j] = Y_tr_filled[valid, j].std().clamp(min=1e-6)
    Y_tr_norm = (Y_tr_filled - target_means) / target_stds

    # Model, optimizer
    model = MultiTaskMLP(
        input_dim=X_tr.shape[1],
        hidden_dims=HIDDEN_DIMS,
        n_targets=n_targets,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Training loop
    dataset = TensorDataset(X_tr, Y_tr_norm, mask_tr.float())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    t_start = time.time()
    best_val_loss = float("inf")
    patience = 50
    patience_counter = 0
    best_state = None

    for epoch in range(N_EPOCHS):
        elapsed = time.time() - t_start
        if elapsed >= time_remaining:
            break

        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb, mb in loader:
            pred = model(xb)
            # Masked MSE loss
            diff = (pred - yb) ** 2 * mb
            loss = diff.sum() / mb.sum().clamp(min=1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Validate (for early stopping)
        model.eval()
        with torch.no_grad():
            val_pred_norm = model(X_v)
            # Denormalize for validation metric
            val_pred = val_pred_norm * target_stds + target_means
            val_pred_np = val_pred.cpu().numpy()
            val_loss, _ = evaluate(Y_val, val_pred_np)
            # We want to maximize Spearman, so use negative as "loss"
            val_metric = -val_loss

        if val_metric < best_val_loss:
            best_val_loss = val_metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

        if (epoch + 1) % 50 == 0:
            elapsed = time.time() - t_start
            print(f"    Epoch {epoch+1:4d} | train_loss: {epoch_loss/max(n_batches,1):.6f} | "
                  f"val_spearman: {-val_metric:.4f} | elapsed: {elapsed:.1f}s")

    # Load best model and predict
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        val_pred_norm = model(X_v)
        val_pred = val_pred_norm * target_stds + target_means
        val_preds = val_pred.cpu().numpy()

    elapsed = time.time() - t_start
    print(f"    Fold done in {elapsed:.1f}s ({epoch+1} epochs)")
    return val_preds

# ---------------------------------------------------------------------------
# Main: cross-validated evaluation
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Time budget: {TIME_BUDGET}s")
    print()

    # Load and encode data
    df = load_data()
    X = encode_sequences(df)
    Y = get_targets(df)
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, {Y.shape[1]} targets")
    print()

    # Cross-validation
    all_preds = np.full_like(Y, np.nan)
    time_per_fold = TIME_BUDGET / N_FOLDS

    for fold in range(N_FOLDS):
        t_fold_start = time.time()
        elapsed_total = t_fold_start - t_start
        time_remaining = min(time_per_fold, TIME_BUDGET - elapsed_total)
        if time_remaining <= 0:
            print(f"Fold {fold}: skipping, out of time")
            break

        train_idx, val_idx = get_fold_splits(df, fold)
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        print(f"Fold {fold}: {len(train_idx)} train, {len(val_idx)} val "
              f"(time budget: {time_remaining:.0f}s)")

        val_preds = train_one_fold(X_train, Y_train, X_val, Y_val, device, time_remaining)
        all_preds[val_idx] = val_preds

    print()

    # Final evaluation (on all CV predictions)
    mean_spearman, per_target = evaluate(Y, all_preds)

    # Summary
    t_end = time.time()
    total_time = t_end - t_start

    print("---")
    print(f"mean_spearman:    {mean_spearman:.6f}")
    for name, rho in per_target.items():
        print(f"  {name:20s}: {rho:.4f}")
    print(f"training_seconds: {total_time:.1f}")
    print(f"model:            MultiTaskMLP {HIDDEN_DIMS}")
    print(f"dropout:          {DROPOUT}")
    print(f"learning_rate:    {LEARNING_RATE}")
    print(f"weight_decay:     {WEIGHT_DECAY}")
    print(f"batch_size:       {BATCH_SIZE}")
    print(f"device:           {device}")


if __name__ == "__main__":
    main()
