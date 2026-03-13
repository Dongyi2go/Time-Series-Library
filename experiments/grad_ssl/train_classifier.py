"""
Downstream classifier trained on gradient features.

Supports two classifier backends:
  - MLP  (default): 2-layer MLP trained on flat feature vectors.
  - DLinear: DLinear time-series model used for multi-channel feature classification.

Input formats accepted:
  1. ``.npz`` file produced by ``extract_features.py`` (legacy, flat features).
  2. Pair of UEA/UCR ``.ts`` files (TRAIN + TEST) produced by ``extract_features.py``
     when the ``--ts`` flag is used.  Pass them with ``--train_ts_path`` /
     ``--test_ts_path``.

Usage
-----
# MLP on .npz (existing workflow)
python experiments/grad_ssl/train_classifier.py \\
    --npz_path  ./artifacts/grad_features_heartbeat.npz \\
    --hidden    256 \\
    --epochs    200 \\
    --lr        1e-3 \\
    --seed      42

# DLinear on .ts feature files
python experiments/grad_ssl/train_classifier.py \\
    --train_ts_path ./artifacts/grad_features_heartbeat_TRAIN.ts \\
    --test_ts_path  ./artifacts/grad_features_heartbeat_TEST.ts \\
    --model         DLinear \\
    --epochs        200 \\
    --lr            1e-3 \\
    --seed          42

# MLP on .ts feature files
python experiments/grad_ssl/train_classifier.py \\
    --train_ts_path ./artifacts/grad_features_heartbeat_TRAIN.ts \\
    --test_ts_path  ./artifacts/grad_features_heartbeat_TEST.ts \\
    --model         MLP \\
    --hidden        256 \\
    --epochs        200 \\
    --seed          42
"""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# .ts file loader  (UEA/UCR format written by extract_features.py)
# ---------------------------------------------------------------------------

def load_ts_features(path: str):
    """Parse a UEA/UCR ``.ts`` feature file.

    Returns
    -------
    X : np.ndarray, shape ``[n_samples, series_length, n_dims]``
        Multi-channel feature array ready for DLinear (``[B, T, C]`` order).
    y : np.ndarray, shape ``[n_samples]``, integer class labels.
    """
    n_dims = None
    series_length = None
    samples: list = []
    labels: list = []
    in_data = False

    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            low = line.lower()
            if low.startswith("@dimensions"):
                try:
                    n_dims = int(line.split()[-1])
                except ValueError:
                    raise ValueError(f"{path}:{lineno}: invalid @dimensions value: {line!r}")
            elif low.startswith("@serieslength"):
                try:
                    series_length = int(line.split()[-1])
                except ValueError:
                    raise ValueError(f"{path}:{lineno}: invalid @seriesLength value: {line!r}")
            elif low.startswith("@data"):
                in_data = True
            elif in_data:
                parts = line.split(":")
                # Last token is the class label; all preceding tokens are dims.
                try:
                    labels.append(int(parts[-1]))
                except ValueError:
                    raise ValueError(
                        f"{path}:{lineno}: cannot parse class label {parts[-1]!r}"
                    )
                dim_arrays = []
                for dim_idx, part in enumerate(parts[:-1]):
                    try:
                        dim_arrays.append([float(v) for v in part.split(",")])
                    except ValueError as exc:
                        raise ValueError(
                            f"{path}:{lineno}: error in sample {len(samples)}, "
                            f"dimension {dim_idx}: {exc}"
                        ) from exc
                samples.append(dim_arrays)

    if not samples:
        raise ValueError(f"No data rows found in {path}")

    # X raw shape: [n_samples, n_dims, series_length]
    X_raw = np.array(samples, dtype=np.float32)
    # Transpose to [n_samples, series_length, n_dims]  →  DLinear's [B, T, C]
    X = X_raw.transpose(0, 2, 1)
    y = np.array(labels, dtype=np.int64)

    if n_dims is not None and X.shape[2] != n_dims:
        raise ValueError(
            f"Header says @dimensions={n_dims} but parsed {X.shape[2]} dims in {path}"
        )
    if series_length is not None and X.shape[1] != series_length:
        raise ValueError(
            f"Header says @seriesLength={series_length} but parsed {X.shape[1]} in {path}"
        )
    return X, y


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
# DLinear classifier wrapper
# ---------------------------------------------------------------------------

class DLinearClassifier(nn.Module):
    """Thin wrapper that instantiates DLinear in classification mode.

    Parameters
    ----------
    seq_len    : time-series length (T in ``[B, T, C]`` inputs).
    n_channels : number of input channels / feature dimensions (C).
    num_class  : number of output classes.
    moving_avg : kernel size for the moving-average decomposition (default 25).
    """

    def __init__(self, seq_len: int, n_channels: int, num_class: int,
                 moving_avg: int = 25):
        super().__init__()
        from models.DLinear import Model as _DLinear
        configs = SimpleNamespace(
            task_name="classification",
            seq_len=seq_len,
            pred_len=seq_len,   # unused for classification, but required
            enc_in=n_channels,
            num_class=num_class,
            moving_avg=moving_avg,
        )
        self.model = _DLinear(configs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]  →  returns [B, num_class]
        return self.model.classification(x)


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
    p = argparse.ArgumentParser(
        description="Train downstream classifier (MLP or DLinear) on gradient features"
    )
    # Input: .npz (legacy) or pair of .ts files
    p.add_argument("--npz_path", type=str,
                   default="./artifacts/grad_features_heartbeat.npz",
                   help="Path to .npz feature file (used when --train_ts_path is not set)")
    p.add_argument("--train_ts_path", type=str, default=None,
                   help="Path to TRAIN .ts feature file (UEA/UCR format)")
    p.add_argument("--test_ts_path",  type=str, default=None,
                   help="Path to TEST  .ts feature file (UEA/UCR format)")
    # Model selection
    p.add_argument("--model", type=str, default="MLP",
                   choices=["MLP", "DLinear"],
                   help="Classifier backend: MLP (default) or DLinear")
    p.add_argument("--moving_avg", type=int, default=25,
                   help="Moving-average window for DLinear series decomposition")
    # Shared hyper-parameters
    p.add_argument("--hidden",     type=int, default=256,
                   help="Hidden size for MLP layers")
    p.add_argument("--epochs",     type=int, default=200)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--dropout",    type=float, default=0.3,
                   help="Dropout rate (MLP only)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     type=str, default="auto")
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
    print(f"Model : {args.model}")

    # ---- Load features ----
    use_ts_files = args.train_ts_path is not None and args.test_ts_path is not None
    if use_ts_files:
        print(f"Loading features from .ts files")
        X_train_3d, y_train = load_ts_features(args.train_ts_path)
        X_test_3d,  y_test  = load_ts_features(args.test_ts_path)
        print(f"X_train={X_train_3d.shape}  y_train={y_train.shape}")
        print(f"X_test ={X_test_3d.shape}   y_test ={y_test.shape}")
    else:
        print(f"Loading features from {args.npz_path}")
        data          = np.load(args.npz_path)
        X_train_flat  = data["X_train"].astype(np.float32)
        y_train       = data["y_train"].astype(np.int64)
        X_test_flat   = data["X_test"].astype(np.float32)
        y_test        = data["y_test"].astype(np.int64)
        # Reshape flat [n, D] → [n, D, 1] so both branches share the same 3-D shape.
        X_train_3d = X_train_flat[:, :, np.newaxis]
        X_test_3d  = X_test_flat[:, :, np.newaxis]
        print(f"X_train={X_train_3d.shape}  y_train={y_train.shape}")
        print(f"X_test ={X_test_3d.shape}   y_test ={y_test.shape}")

    seq_len    = X_train_3d.shape[1]   # T
    n_channels = X_train_3d.shape[2]   # C
    num_class  = int(max(y_train.max(), y_test.max())) + 1

    # ---- Standardise (fit on train only) ----
    if args.model == "DLinear":
        # Standardise per (time-step, channel) position: mean/std over samples
        mean = X_train_3d.mean(axis=0, keepdims=True)   # [1, T, C]
        std  = X_train_3d.std(axis=0,  keepdims=True) + 1e-8
        X_train_s = (X_train_3d - mean) / std
        X_test_s  = (X_test_3d  - mean) / std
        input_info = f"seq_len={seq_len}  n_channels={n_channels}  num_class={num_class}"
    else:
        # MLP: flatten to [n, T*C] first, then standardise per feature
        X_train_flat = X_train_3d.reshape(len(X_train_3d), -1)
        X_test_flat  = X_test_3d.reshape(len(X_test_3d),   -1)
        mean = X_train_flat.mean(axis=0, keepdims=True)
        std  = X_train_flat.std(axis=0,  keepdims=True) + 1e-8
        X_train_s = (X_train_flat - mean) / std
        X_test_s  = (X_test_flat  - mean) / std
        input_info = f"input_dim={X_train_s.shape[1]}  num_class={num_class}"

    print(input_info)

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
    if args.model == "DLinear":
        model = DLinearClassifier(
            seq_len=seq_len,
            n_channels=n_channels,
            num_class=num_class,
            moving_avg=args.moving_avg,
        ).to(device)
    else:
        input_dim = X_train_s.shape[1]
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
        f.write(f"model={args.model}\n")
        f.write(f"final_test_accuracy={final_acc:.4f}\n")
        f.write(f"best_test_accuracy={best_test_acc:.4f}\n")
        f.write(f"seq_len={seq_len}\n")
        f.write(f"n_channels={n_channels}\n")
        f.write(f"num_class={num_class}\n")
        f.write(f"n_train={len(y_train)}\n")
        f.write(f"n_test={len(y_test)}\n")
    print(f"Results saved → {results_path}")


if __name__ == "__main__":
    main()
