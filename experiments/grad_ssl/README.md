# Self-Supervised Gradient Feature Validation

This experiment implements a **gradient-feature-based downstream classification**
pipeline on the [Heartbeat](https://www.timeseriesclassification.com/description.php?Dataset=Heartbeat)
UEA dataset.

---

## Overview

| Step | Script | Description |
|------|--------|-------------|
| 1 | `pretrain.py` | Self-supervised masked-autoencoding pre-training (10 epochs) for each of the three backbones |
| 2 | `extract_features.py` | Extract 128-d gradient features per model → 384-d concatenated features; save to `.npz` and UEA/UCR `.ts` files |
| 3 | `train_classifier.py` | Train a downstream classifier (**MLP** or **DLinear**) on the gradient features and evaluate on the test set |

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
# Full pipeline with default MLP classifier (GPU recommended, falls back to CPU)
bash experiments/grad_ssl/run_pipeline.sh ./dataset/Heartbeat auto

# Full pipeline with DLinear classifier
bash experiments/grad_ssl/run_pipeline.sh ./dataset/Heartbeat auto DLinear

# Or step by step:

# 1. Pre-train each backbone
for MODEL in Informer LightTS TimesNet; do
  python experiments/grad_ssl/pretrain.py \
    --data_path ./dataset/Heartbeat \
    --model     $MODEL \
    --n_epochs  10
done

# 2. Extract gradient features (.npz + .ts files)
python experiments/grad_ssl/extract_features.py \
    --data_path ./dataset/Heartbeat \
    --out_path  ./artifacts/grad_features_heartbeat.npz

# 3a. Train downstream MLP classifier (reads .npz)
python experiments/grad_ssl/train_classifier.py \
    --npz_path ./artifacts/grad_features_heartbeat.npz \
    --model    MLP

# 3b. Train downstream DLinear classifier (reads .ts feature files)
python experiments/grad_ssl/train_classifier.py \
    --train_ts_path ./artifacts/grad_features_heartbeat_TRAIN.ts \
    --test_ts_path  ./artifacts/grad_features_heartbeat_TEST.ts \
    --model         DLinear
```

---

## DLinear Classification

`train_classifier.py` supports a **DLinear** backend in addition to the default MLP.
DLinear is a lightweight linear model that decomposes the input series into seasonal
and trend components before classification.  It is particularly effective when the
gradient-feature channels carry complementary spectral structure across the three
SSL backbones.

### How it works

* The `.ts` feature files store gradient features in **UEA/UCR multi-channel format**:
  each sample has `n_models` channels (dimensions) of length `grad_dim` (default 128).
* `train_classifier.py` parses the `.ts` file into a tensor of shape
  `[n_samples, series_length, n_channels]` matching DLinear's `[B, T, C]` input.
* DLinear decomposes each channel with a moving-average filter (`--moving_avg`),
  applies per-channel linear layers, then projects the flattened output to class logits.

### .ts feature file format

```
# UEA/UCR .ts – gradient features (Train split)
@problemName grad_features_heartbeat
@timeStamps false
@missing false
@univariate false
@dimensions 3
@equalLength true
@seriesLength 128
@classLabel true 0 1
@data
0.12,-0.34,...(128 values):0.56,0.78,...:0.11,0.22,...:0
...
```

Each row encodes one sample: `dim0_v0,...,v127:dim1_v0,...:...:label`.

### DLinear-specific parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `MLP` | Classifier backend: `MLP` or `DLinear` |
| `--train_ts_path` | `None` | Path to TRAIN `.ts` feature file |
| `--test_ts_path`  | `None` | Path to TEST  `.ts` feature file |
| `--moving_avg` | `25` | Moving-average kernel size for DLinear decomposition |

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
| `artifacts/grad_features_heartbeat_TRAIN.ts` | UEA/UCR `.ts` format, `[n_train, 128, 3]` |
| `artifacts/grad_features_heartbeat_TEST.ts`  | UEA/UCR `.ts` format, `[n_test,  128, 3]` |
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

