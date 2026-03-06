"""
Self-supervised pretraining with masked autoencoding on Heartbeat.

Usage
-----
python experiments/grad_ssl/pretrain.py \\
    --data_path  ./dataset/Heartbeat \\
    --ckpt_dir   ./artifacts/ckpts \\
    --model      Informer \\
    --n_epochs   10 \\
    --batch_size 32 \\
    --lr         1e-3 \\
    --d_model    64 \\
    --n_heads    4 \\
    --e_layers   2 \\
    --d_ff       128 \\
    --grad_dim   128 \\
    --mask_rate  0.5 \\
    --seed       42

Run with --model Informer, LightTS, TimesNet separately (or loop in run_pipeline.sh).
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.grad_ssl.backbone_wrapper import build_wrapper  # noqa: E402
from experiments.grad_ssl.heartbeat_data import load_heartbeat   # noqa: E402


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def make_time_mask(B: int, T: int, C: int, mask_rate: float,
                   seed: int) -> torch.Tensor:
    """Time-wise mask [B, T, C]: 1 = masked (to be reconstructed)."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((B, T, C), dtype=np.float32)
    n_masked = max(1, int(T * mask_rate))
    for b in range(B):
        t_idx = rng.choice(T, size=n_masked, replace=False)
        mask[b, t_idx, :] = 1.0
    return torch.from_numpy(mask)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def masked_mse(x_hat: torch.Tensor, x_orig: torch.Tensor,
               mask: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    """MSE only on positions where mask==1 AND pad_mask==1, normalised."""
    eff = mask * pad_mask.unsqueeze(-1)          # [B, T, C]
    loss = ((x_hat - x_orig) ** 2 * eff).sum()
    n = eff.sum()
    return loss / (n + 1e-8)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_configs(seq_len: int, enc_in: int, args):
    """Build an argparse.Namespace configs object for backbone init."""
    cfg = argparse.Namespace(
        task_name="imputation",
        seq_len=seq_len,
        label_len=0,
        pred_len=0,
        enc_in=enc_in,
        dec_in=enc_in,
        c_out=enc_in,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=1,
        d_ff=args.d_ff,
        factor=5,
        dropout=0.1,
        embed="fixed",
        freq="h",
        distil=True,
        activation="gelu",
        num_class=2,        # placeholder; unused in encoder
        top_k=3,
        num_kernels=6,
    )
    return cfg


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def pretrain(model: nn.Module, train_loader, device: torch.device,
             n_epochs: int, lr: float, mask_rate: float, seed: int):
    """Run masked-autoencoding pretraining."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    global_step = 0
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        n_steps = 0

        for batch_x, _labels, padding_mask in train_loader:
            batch_x = batch_x.float().to(device)
            pad_mask = padding_mask.float().to(device)   # [B, T]
            B, T, C = batch_x.shape

            # Random mask with reproducible but varying seed
            mask = make_time_mask(B, T, C, mask_rate,
                                  seed=seed + global_step).to(device)
            x_masked = batch_x * (1.0 - mask)

            optimizer.zero_grad()
            x_hat = model(x_masked)                      # [B, T, C]
            loss = masked_mse(x_hat, batch_x, mask, pad_mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_steps += 1
            global_step += 1

        avg = epoch_loss / max(n_steps, 1)
        print(f"Epoch [{epoch:3d}/{n_epochs}]  loss={avg:.6f}")

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SSL pretraining on Heartbeat")
    p.add_argument("--data_path", type=str, required=True,
                   help="Directory containing Heartbeat_TRAIN.ts / Heartbeat_TEST.ts")
    p.add_argument("--ckpt_dir", type=str, default="./artifacts/ckpts")
    p.add_argument("--model", type=str, default="Informer",
                   choices=["Informer", "LightTS", "TimesNet"])
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--e_layers", type=int, default=2)
    p.add_argument("--d_ff", type=int, default=128)
    p.add_argument("--grad_dim", type=int, default=128)
    p.add_argument("--mask_rate", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto",
                   help="'auto', 'cpu', 'cuda', or 'cuda:N'")
    return p.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Data
    from experiments.grad_ssl.heartbeat_data import load_heartbeat
    train_ds, _test_ds, max_len = load_heartbeat(args.data_path)
    print(f"Heartbeat  seq_len={max_len}  enc_in={train_ds.enc_in}"
          f"  train={train_ds.n_samples}  test={_test_ds.n_samples}  "
          f"num_class={train_ds.num_class}")

    train_loader = train_ds.get_loader(
        batch_size=args.batch_size, max_len=max_len, shuffle=True)

    # Model
    configs = build_configs(max_len, train_ds.enc_in, args)
    model = build_wrapper(args.model, configs, grad_dim=args.grad_dim)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}  params={n_params:,}")

    # Train
    pretrain(model, train_loader, device,
             n_epochs=args.n_epochs,
             lr=args.lr,
             mask_rate=args.mask_rate,
             seed=args.seed)

    # Save checkpoint
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, f"{args.model}_ckpt_{args.n_epochs}ep.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint → {ckpt_path}")


if __name__ == "__main__":
    main()
