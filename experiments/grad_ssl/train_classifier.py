"""
Downstream MLP classifier trained on gradient features.

Reads the .npz file produced by extract_features.py, standardises features
using train-set statistics, trains a small MLP, and evaluates on the test set.

Usage
-----
python experiments/grad_ssl/train_classifier.py \\
    --npz_path  ./artifacts/grad_features_heartbeat.npz \\
    --hidden    256 \\
    --epochs    200 \\
    --lr        1e-3 \\
    --seed      42
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# MLP classifier
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_class: int,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train downstream MLP classifier")
    p.add_argument("--npz_path", type=str,
                   default="./artifacts/grad_features_heartbeat.npz")
    p.add_argument("--hidden",   type=int, default=256)
    p.add_argument("--epochs",   type=int, default=200)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--dropout",  type=float, default=0.3)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--device",   type=str, default="auto")
    p.add_argument("--results_dir", type=str, default="./artifacts")
    return p.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ---- Load features ----
    data = np.load(args.npz_path)
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    X_test  = data["X_test"].astype(np.float32)
    y_test  = data["y_test"].astype(np.int64)
    print(f"X_train={X_train.shape}  y_train={y_train.shape}")
    print(f"X_test ={X_test.shape}   y_test ={y_test.shape}")

    # ---- Standardise (fit on train only) ----
    mean = X_train.mean(axis=0, keepdims=True)
    std  = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train_s = (X_train - mean) / std
    X_test_s  = (X_test  - mean) / std

    input_dim = X_train_s.shape[1]
    num_class = int(max(y_train.max(), y_test.max())) + 1
    print(f"input_dim={input_dim}  num_class={num_class}")

    # ---- DataLoaders ----
    train_ds = TensorDataset(
        torch.from_numpy(X_train_s),
        torch.from_numpy(y_train),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test_s),
        torch.from_numpy(y_test),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, drop_last=False)

    # ---- Model ----
    model = MLP(input_dim, args.hidden, num_class, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    # ---- Training ----
    best_test_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 20 == 0 or epoch == args.epochs:
            model.eval()
            train_accs, test_accs = [], []
            with torch.no_grad():
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    train_accs.append(accuracy(model(xb), yb))
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    test_accs.append(accuracy(model(xb), yb))
            tr_acc  = float(np.mean(train_accs))
            te_acc  = float(np.mean(test_accs))
            best_test_acc = max(best_test_acc, te_acc)
            print(f"Epoch [{epoch:4d}/{args.epochs}]  "
                  f"train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}")

    # ---- Final evaluation ----
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(yb.numpy())
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    final_acc = (preds == labels).mean()
    print(f"\n{'='*50}")
    print(f"Final test accuracy  : {final_acc:.4f}")
    print(f"Best  test accuracy  : {best_test_acc:.4f}")

    # Per-class accuracy
    for c in range(num_class):
        mask = labels == c
        if mask.sum() > 0:
            cls_acc = (preds[mask] == labels[mask]).mean()
            print(f"  class {c}: {cls_acc:.4f}  (n={mask.sum()})")
    print(f"{'='*50}")

    # ---- Save results ----
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, "classifier_results.txt")
    with open(results_path, "w") as f:
        f.write(f"final_test_accuracy={final_acc:.4f}\n")
        f.write(f"best_test_accuracy={best_test_acc:.4f}\n")
        f.write(f"feature_dim={input_dim}\n")
        f.write(f"num_class={num_class}\n")
        f.write(f"n_train={len(y_train)}\n")
        f.write(f"n_test={len(y_test)}\n")
    print(f"Results saved → {results_path}")


if __name__ == "__main__":
    main()
