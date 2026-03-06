# Self-Supervised Gradient Feature Validation

This experiment implements a **gradient-feature-based downstream classification**
pipeline on the [Heartbeat](https://www.timeseriesclassification.com/description.php?Dataset=Heartbeat)
UEA dataset.

---

## Overview

| Step | Script | Description |
|------|--------|-------------|
| 1 | `pretrain.py` | Self-supervised masked-autoencoding pre-training (10 epochs) for each of the three backbones |
| 2 | `extract_features.py` | Extract 128-d gradient features per model → 384-d concatenated features; save to `.npz` |
| 3 | `train_classifier.py` | Train a small MLP on the gradient features and evaluate on the test set |

---

## Method

### Backbones

Three architecturally distinct backbones share the same SSL head:

| Backbone | Architecture type | Hidden dim H |
|----------|------------------|--------------|
| Informer | Transformer (ProbAttention encoder) | `d_model` |
| LightTS  | MLP (IEBlock chunks) | `enc_in` (channel dim) |
| TimesNet | CNN (Inception + FFT period blocks) | `d_model` |

### Self-supervised head (identical for all backbones)

```
backbone hidden states  h  [B, T, H]
         ↓  grad_proj: Linear(H → D, bias=False)   D = 128
         ↓  recon_head: Linear(D → C)               C = enc_in
         x̂  [B, T, C]
```

### Masked autoencoding task

* **Time-wise masking**: randomly select 50 % of time steps per sample; zero them out.
* **Loss**: MSE only at positions where `mask == 1` AND `pad_mask == 1`, normalised by the count of effective masked positions.

### Gradient feature (per sample, per backbone)

1. Perform 10 independent forward + backward passes with different random masks (seed = `global_seed + sample_idx × n_masks + mask_idx`).
2. After each backward, read `grad_proj.weight.grad` (shape `[D, H]`), compute the **per-row L2 norm** → vector of length `D = 128`.
3. **Mean** of the 10 vectors = 128-d gradient feature for this backbone.

Concatenating the 3 backbone features → **384-d** feature per sample.

### Extraction strategy (S1)

* **Train set**: continued training while extracting.  For each sample, gradients from the 10 masks are accumulated (each loss divided by 10) and one `optimizer.step()` is performed after recording.
* **Test set**: same backward passes, **no `optimizer.step()`** (test-time adaptation is disabled).

---

## Getting the data

Place the Heartbeat `.ts` files in a local directory:

```
dataset/
└── Heartbeat/
    ├── Heartbeat_TRAIN.ts
    └── Heartbeat_TEST.ts
```

If the files are absent, the loader will attempt to download them automatically
from [HuggingFace](https://huggingface.co/datasets/thuml/Time-Series-Library)
(requires `huggingface_hub` and internet access).

---

## Quick start

```bash
# Full pipeline (GPU recommended, falls back to CPU automatically)
bash experiments/grad_ssl/run_pipeline.sh ./dataset/Heartbeat auto

# Or step by step:

# 1. Pre-train each backbone
for MODEL in Informer LightTS TimesNet; do
  python experiments/grad_ssl/pretrain.py \
    --data_path ./dataset/Heartbeat \
    --model     $MODEL \
    --n_epochs  10
done

# 2. Extract gradient features
python experiments/grad_ssl/extract_features.py \
    --data_path ./dataset/Heartbeat \
    --out_path  ./artifacts/grad_features_heartbeat.npz

# 3. Train downstream classifier
python experiments/grad_ssl/train_classifier.py \
    --npz_path ./artifacts/grad_features_heartbeat.npz
```

---

## Key hyper-parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--d_model` | 64 | Transformer / conv hidden size |
| `--n_heads` | 4 | Number of attention heads (Informer) |
| `--e_layers` | 2 | Number of encoder layers |
| `--d_ff` | 128 | Feed-forward hidden size |
| `--grad_dim` | 128 | Projection dimension D |
| `--mask_rate` | 0.5 | Fraction of time steps masked |
| `--n_masks` | 10 | Masks per sample for feature aggregation |
| `--n_epochs` | 10 | Pre-training epochs |
| `--seed` | 42 | Global random seed |

---

## Artifacts

| Path | Content |
|------|---------|
| `artifacts/ckpts/Informer_ckpt_10ep.pth` | Informer checkpoint after 10 epochs |
| `artifacts/ckpts/LightTS_ckpt_10ep.pth`  | LightTS checkpoint |
| `artifacts/ckpts/TimesNet_ckpt_10ep.pth` | TimesNet checkpoint |
| `artifacts/grad_features_heartbeat.npz`  | `X_train [n,384]`, `y_train`, `X_test [n,384]`, `y_test` |
| `artifacts/classifier_results.txt`       | Final accuracy metrics |

---

## Reproducibility

All random operations are seeded:

* PyTorch seed: `torch.manual_seed(seed)` at the start of each script.
* NumPy seed: `np.random.seed(seed)`.
* Mask generation: deterministic `numpy.random.default_rng(seed_j)` where
  `seed_j = base_seed + sample_idx × n_masks + mask_idx`.
* Train and test masks use different base seeds (`base_seed` and `base_seed + 100000`).
* DataLoader `shuffle=False` for feature extraction; train DataLoader shuffles during pre-training.
