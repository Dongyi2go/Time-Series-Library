#!/usr/bin/env bash
# run_pipeline.sh  --  End-to-end self-supervised gradient feature pipeline
#
# Usage:
#   bash experiments/grad_ssl/run_pipeline.sh [DATA_PATH] [DEVICE]
#
# Arguments (positional, optional):
#   DATA_PATH  Directory that contains Heartbeat_TRAIN.ts and Heartbeat_TEST.ts
#              Default: ./dataset/Heartbeat
#   DEVICE     pytorch device string, e.g. "cpu", "cuda", "cuda:0"
#              Default: auto (GPU if available, else CPU)
#
# Prerequisites:
#   pip install -r requirements.txt
#   # If data not present locally, huggingface_hub will download automatically.

set -euo pipefail

DATA_PATH="${1:-./dataset/Heartbeat}"
DEVICE="${2:-auto}"

CKPT_DIR="./artifacts/ckpts"
NPZ_PATH="./artifacts/grad_features_heartbeat.npz"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
N_EPOCHS=10
BATCH_SIZE=32
LR=1e-3
D_MODEL=64
N_HEADS=4
E_LAYERS=2
D_FF=128
GRAD_DIM=128
MASK_RATE=0.5
N_MASKS=10
SEED=42

echo "========================================================"
echo " Step 1/3 : Self-supervised pre-training (${N_EPOCHS} epochs each)"
echo "========================================================"

for MODEL in Informer LightTS TimesNet; do
    echo ""
    echo "--- Pretraining ${MODEL} ---"
    python experiments/grad_ssl/pretrain.py \
        --data_path  "${DATA_PATH}" \
        --ckpt_dir   "${CKPT_DIR}" \
        --model      "${MODEL}" \
        --n_epochs   "${N_EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --lr         "${LR}" \
        --d_model    "${D_MODEL}" \
        --n_heads    "${N_HEADS}" \
        --e_layers   "${E_LAYERS}" \
        --d_ff       "${D_FF}" \
        --grad_dim   "${GRAD_DIM}" \
        --mask_rate  "${MASK_RATE}" \
        --seed       "${SEED}" \
        --device     "${DEVICE}"
done

echo ""
echo "========================================================"
echo " Step 2/3 : Gradient feature extraction"
echo "========================================================"

python experiments/grad_ssl/extract_features.py \
    --data_path  "${DATA_PATH}" \
    --ckpt_dir   "${CKPT_DIR}" \
    --out_path   "${NPZ_PATH}" \
    --d_model    "${D_MODEL}" \
    --n_heads    "${N_HEADS}" \
    --e_layers   "${E_LAYERS}" \
    --d_ff       "${D_FF}" \
    --grad_dim   "${GRAD_DIM}" \
    --mask_rate  "${MASK_RATE}" \
    --n_masks    "${N_MASKS}" \
    --n_epochs   "${N_EPOCHS}" \
    --lr         "${LR}" \
    --seed       "${SEED}" \
    --device     "${DEVICE}"

echo ""
echo "========================================================"
echo " Step 3/3 : Train downstream MLP classifier"
echo "========================================================"

python experiments/grad_ssl/train_classifier.py \
    --npz_path    "${NPZ_PATH}" \
    --hidden      256 \
    --epochs      200 \
    --lr          1e-3 \
    --batch_size  64 \
    --seed        "${SEED}" \
    --device      "${DEVICE}"

echo ""
echo "========================================================"
echo " Pipeline complete!  Artifacts in ./artifacts/"
echo "========================================================"
