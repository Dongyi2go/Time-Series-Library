"""
Gradient-feature extraction for the Heartbeat dataset.

Strategy S1
-----------
For each model in {Informer, LightTS, TimesNet}:

1. Load the pretrained checkpoint (ckpt_10ep).
2. **Training-set extraction (continues training)**:
   For sample i, mask j (0 … n_masks-1):
     - zero_grad
     - forward with masked input
     - backward  →  read grad_proj.weight.grad, compute per-row L2 norm
   Take mean of n_masks norms → 128-d feature for this model.
   After recording all n_masks features, do ONE averaged-gradient step
   (re-run all n_masks masks, accumulate loss/n_masks per mask, then step).
3. **Test-set extraction (no parameter update)**:
   Same recording procedure, but optimizer.step() is NEVER called.

Each sample's 384-d feature = concat([feat_Informer, feat_LightTS, feat_TimesNet]).

Output
------
artifacts/grad_features_heartbeat_TRAIN.ts  (UEA/UCR .ts format)
artifacts/grad_features_heartbeat_TEST.ts   (UEA/UCR .ts format)

  Each file contains n_samples rows.  Each row encodes grad_dim values per
  model as one UEA dimension, so a 3-model run produces 3 dimensions of
  length grad_dim (default 128).

Usage
-----
python experiments/grad_ssl/extract_features.py \\
    --data_path   ./dataset/Heartbeat \\
    --ckpt_dir    ./artifacts/ckpts \\
    --out_path    ./artifacts/grad_features_heartbeat.npz \\
    --d_model     64  --n_heads 4  --e_layers 2  --d_ff 128 \\
    --grad_dim    128 --mask_rate 0.5 --n_masks 10 --seed 42 \\
    --device      auto
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

from experiments.grad_ssl.backbone_wrapper import build_wrapper, _SelfSupervisedWrapper  # noqa
from experiments.grad_ssl.heartbeat_data import load_heartbeat                           # noqa
from experiments.grad_ssl.pretrain import build_configs, make_time_mask, masked_mse      # noqa


# ---------------------------------------------------------------------------
# Gradient-feature extraction helpers
# ---------------------------------------------------------------------------

def _extract_one_sample(
    model: _SelfSupervisedWrapper,
    x: torch.Tensor,          # [1, T, C]
    pad_mask: torch.Tensor,   # [1, T]
    sample_idx: int,
    n_masks: int,
    mask_rate: float,
    base_seed: int,
    optimizer,                # None for test set
) -> np.ndarray:
    """Extract 128-d gradient feature for a single sample.

    For each of the n_masks independent random masks:
      zero_grad → forward → backward → record row-norms of grad_proj.weight.grad
    Take the mean of the n_masks recorded vectors.

    If optimizer is provided (train set), also perform ONE parameter update
    using the accumulated mean gradient (accumulate loss/n_masks per mask,
    then step once).
    """
    device = x.device
    _, T, C = x.shape

    grad_vecs = []                    # will hold n_masks arrays of shape [D]
    prev_accum = None                 # tracks accumulated grad for delta

    model.train()                     # need grad even for test set
    model.zero_grad()                 # single zero_grad before accumulation loop

    for j in range(n_masks):
        seed_j = base_seed + sample_idx * n_masks + j
        mask_j = make_time_mask(1, T, C, mask_rate, seed=seed_j).to(device)
        x_masked = x * (1.0 - mask_j)

        x_hat = model(x_masked)
        loss = masked_mse(x_hat, x, mask_j, pad_mask) / n_masks
        loss.backward()               # accumulates into .grad buffers

        # Delta grad since last backward = gradient of mask_j alone (scaled by 1/n_masks)
        cur_accum = model.grad_proj.weight.grad.detach().clone()  # [D, H]
        if prev_accum is None:
            delta = cur_accum
        else:
            delta = cur_accum - prev_accum
        prev_accum = cur_accum

        # Per-row L2 norm of the per-mask gradient (scaling by n_masks is
        # consistent across all masks and cancels in downstream standardisation)
        row_norms = delta.norm(dim=1).cpu().numpy()               # [D]
        grad_vecs.append(row_norms)

    feature = np.mean(grad_vecs, axis=0)   # [D=128]

    if optimizer is not None:
        # Gradients are already accumulated (sum = gradient of mean loss)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
        optimizer.step()
        model.zero_grad()

    return feature


# ---------------------------------------------------------------------------
# Per-split feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    model: _SelfSupervisedWrapper,
    dataset,
    max_len: int,
    n_masks: int,
    mask_rate: float,
    base_seed: int,
    device: torch.device,
    optimizer,                        # None → test set (no update)
    split_name: str = "",
) -> tuple:
    """Extract features for an entire split.

    Returns
    -------
    X : np.ndarray  [n_samples, grad_dim]
    y : np.ndarray  [n_samples]
    """
    X_list, y_list = [], []
    n = dataset.n_samples

    for i, (x, pad_mask, label) in enumerate(dataset.iter_samples(max_len=max_len)):
        x = x.float().to(device)
        pad_mask = pad_mask.float().to(device)

        feat = _extract_one_sample(
            model, x, pad_mask,
            sample_idx=i,
            n_masks=n_masks,
            mask_rate=mask_rate,
            base_seed=base_seed,
            optimizer=optimizer,
        )
        X_list.append(feat)
        y_list.append(label.item())

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  {split_name}  [{i+1:4d}/{n}]  feature_shape={feat.shape}")

    return np.stack(X_list), np.array(y_list)


# ---------------------------------------------------------------------------
# UEA/UCR .ts format export
# ---------------------------------------------------------------------------

def save_gradient_features_ts(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_path: str,
    problem_name: str = "GradientFeature",
    grad_dim: int = 128,
) -> None:
    """Save gradient features as UEA/UCR standard .ts format files.

    The concatenated feature matrix X (shape [n_samples, grad_dim * n_models])
    is reshaped to [n_samples, grad_dim, n_models] so that each model's
    contribution becomes one UEA dimension of length grad_dim.

    Two files are written, derived from ``out_path``:
      <base>_TRAIN.ts  – training split
      <base>_TEST.ts   – test split

    Header fields written per UEA/UCR convention:
      @problemName, @timeStamps, @missing, @univariate,
      @dimensions, @equalLength, @seriesLength, @classLabel

    Parameters
    ----------
    X_train, X_test : np.ndarray  shape [n_samples, grad_dim * n_models]
    y_train, y_test : np.ndarray  shape [n_samples]
    out_path        : str  base path used to derive output filenames;
                      a trailing ``.npz`` extension is stripped automatically.
    problem_name    : str  written into the @problemName header field.
    grad_dim        : int  feature length contributed by each model (default 128).
    """
    n_train, total_dim = X_train.shape
    if total_dim % grad_dim != 0:
        raise ValueError(
            f"Total feature dim ({total_dim}) is not divisible by "
            f"grad_dim ({grad_dim}). Check extraction parameters."
        )
    n_models = total_dim // grad_dim  # number of UEA dimensions

    # Validate X_test has the same feature width as X_train
    if X_test.shape[1] != total_dim:
        raise ValueError(
            f"X_test feature dim ({X_test.shape[1]}) does not match "
            f"X_train feature dim ({total_dim})."
        )

    # Reshape: [n_samples, grad_dim * n_models] → [n_samples, grad_dim, n_models]
    X_train_3d = X_train.reshape(n_train, grad_dim, n_models)
    X_test_3d  = X_test.reshape(X_test.shape[0], grad_dim, n_models)

    # Derive output paths from out_path (strip .npz suffix when present)
    base = out_path[:-4] if out_path.endswith(".npz") else out_path
    out_train = base + "_TRAIN.ts"
    out_test  = base + "_TEST.ts"

    # Collect all class labels across both splits for the header.
    # Labels are expected to be numeric (integers); non-numeric labels will
    # raise a ValueError at write time.
    all_labels = sorted(set(y_train.tolist()) | set(y_test.tolist()))
    label_desc = " ".join(str(int(lbl)) for lbl in all_labels)

    # @univariate is true only when a single model/dimension is present
    univariate_flag = "true" if n_models == 1 else "false"

    def _write_split(X_3d: np.ndarray, y: np.ndarray, path: str, split: str) -> None:
        """Write a single split to UEA/UCR .ts format."""
        n, series_len, dims = X_3d.shape  # series_len == grad_dim, dims == n_models
        with open(path, "w", encoding="utf-8") as fh:
            # ---- Header ----
            fh.write(f"# UEA/UCR .ts – gradient features ({split} split)\n")
            fh.write(f"@problemName {problem_name}\n")
            fh.write("@timeStamps false\n")
            fh.write("@missing false\n")
            fh.write(f"@univariate {univariate_flag}\n")
            fh.write(f"@dimensions {dims}\n")
            fh.write("@equalLength true\n")
            fh.write(f"@seriesLength {series_len}\n")
            fh.write(f"@classLabel true {label_desc}\n")
            fh.write("@data\n")
            # ---- Data rows: one sample per line ----
            for i in range(n):
                # Each model's 128-d vector is one colon-separated dimension
                dim_strs = [
                    ",".join(f"{v:.6g}" for v in X_3d[i, :, d])
                    for d in range(dims)
                ]
                row = ":".join(dim_strs) + f":{int(y[i])}"
                fh.write(row + "\n")
        print(f"{split} features saved → {path}")

    _write_split(X_train_3d, y_train, out_train, "Train")
    _write_split(X_test_3d,  y_test,  out_test,  "Test")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Extract gradient features")
    p.add_argument("--data_path",  type=str, required=True)
    p.add_argument("--ckpt_dir",   type=str, default="./artifacts/ckpts")
    p.add_argument("--out_path",   type=str,
                   default="./artifacts/grad_features_heartbeat.npz")
    p.add_argument("--d_model",    type=int, default=64)
    p.add_argument("--n_heads",    type=int, default=4)
    p.add_argument("--e_layers",   type=int, default=2)
    p.add_argument("--d_ff",       type=int, default=128)
    p.add_argument("--grad_dim",   type=int, default=128)
    p.add_argument("--mask_rate",  type=float, default=0.5)
    p.add_argument("--n_masks",    type=int, default=10)
    p.add_argument("--n_epochs",   type=int, default=10,
                   help="Epoch count used in checkpoint filename")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--lr",         type=float, default=1e-3,
                   help="Optimizer LR for continued training on train set")
    p.add_argument("--device",     type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ---- Data ----
    train_ds, test_ds, max_len = load_heartbeat(args.data_path)
    enc_in = train_ds.enc_in
    print(f"seq_len={max_len}  enc_in={enc_in}  "
          f"train={train_ds.n_samples}  test={test_ds.n_samples}")

    model_names = ["Informer", "LightTS", "TimesNet"]
    all_X_train, all_X_test = [], []
    y_train_ref, y_test_ref = None, None

    for model_name in model_names:
        print(f"\n===== {model_name} =====")
        configs = build_configs(max_len, enc_in, args)

        # Build model and load checkpoint
        model = build_wrapper(model_name, configs, grad_dim=args.grad_dim)
        ckpt_path = os.path.join(
            args.ckpt_dir,
            f"{model_name}_ckpt_{args.n_epochs}ep.pth",
        )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                "Run pretrain.py first."
            )
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.to(device)
        print(f"Loaded checkpoint: {ckpt_path}")

        # Continued-training optimizer for train-set extraction
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train-set: extract features + update
        print("  >> Train set")
        X_tr, y_tr = extract_features(
            model, train_ds, max_len,
            n_masks=args.n_masks, mask_rate=args.mask_rate,
            base_seed=args.seed,
            device=device, optimizer=optimizer, split_name="train",
        )
        all_X_train.append(X_tr)
        if y_train_ref is None:
            y_train_ref = y_tr
        else:
            assert np.array_equal(y_tr, y_train_ref), "Label mismatch on train set"

        # Test-set: extract features, NO parameter update
        print("  >> Test set")
        X_te, y_te = extract_features(
            model, test_ds, max_len,
            n_masks=args.n_masks, mask_rate=args.mask_rate,
            # Offset seed so test masks differ from train masks
            base_seed=args.seed + 100000,
            device=device, optimizer=None, split_name="test",
        )
        all_X_test.append(X_te)
        if y_test_ref is None:
            y_test_ref = y_te
        else:
            assert np.array_equal(y_te, y_test_ref), "Label mismatch on test set"

    # ---- Concatenate features from 3 models ----
    X_train = np.concatenate(all_X_train, axis=1)   # [n_train, 384]
    X_test  = np.concatenate(all_X_test,  axis=1)   # [n_test,  384]
    y_train = y_train_ref
    y_test  = y_test_ref

    print(f"\nX_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_test : {X_test.shape}   y_test : {y_test.shape}")
    assert X_train.shape[1] == args.grad_dim * len(model_names), \
        f"Expected {args.grad_dim * len(model_names)} features, got {X_train.shape[1]}"

    # ---- Save as UEA/UCR .ts format ----
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    # Derive problem name from the output filename (without extension)
    problem_name = os.path.splitext(os.path.basename(args.out_path))[0]
    save_gradient_features_ts(
        X_train, y_train, X_test, y_test,
        out_path=args.out_path,
        problem_name=problem_name,
        grad_dim=args.grad_dim,
    )


if __name__ == "__main__":
    main()
