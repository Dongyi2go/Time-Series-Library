#!/usr/bin/env bash
# run_pipeline.sh  --  End-to-end self-supervised gradient feature pipeline
#
# Usage:
#   bash experiments/grad_ssl/run_pipeline.sh [DATA_PATH] [DEVICE] [CLASSIFIER]
#
# Arguments (positional, optional):
#   DATA_PATH   Directory that contains Heartbeat_TRAIN.ts and Heartbeat_TEST.ts
#               Default: ./dataset/Heartbeat
#   DEVICE      pytorch device string, e.g. "cpu", "cuda", "cuda:0"
#               Default: auto (GPU if available, else CPU)
#   CLASSIFIER  Downstream classifier backend: "MLP" (default) or "DLinear"
#               When "DLinear" is chosen, .ts feature files are loaded directly.
#
# Prerequisites:
#   pip install -r requirements.txt
#   # If data not present locally, huggingface_hub will download automatically.
#
# One-liner examples:
#   # Full pipeline with default MLP classifier
#   bash experiments/grad_ssl/run_pipeline.sh
#
#   # Full pipeline with DLinear classifier (reads .ts feature files)
#   bash experiments/grad_ssl/run_pipeline.sh ./dataset/Heartbeat auto DLinear

set -euo pipefail

DATA_PATH="${1:-./dataset/Heartbeat}"
DEVICE="${2:-auto}"
CLASSIFIER="${3:-MLP}"

CKPT_DIR="./artifacts/ckpts"
NPZ_PATH="./artifacts/grad_features_heartbeat.npz"
# .ts feature paths derived automatically from NPZ_PATH (strip .npz, add _TRAIN/TEST.ts)
TS_TRAIN="${NPZ_PATH%.npz}_TRAIN.ts"
TS_TEST="${NPZ_PATH%.npz}_TEST.ts"

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
echo " Step 3/3 : Train downstream ${CLASSIFIER} classifier"
echo "========================================================"

if [ "${CLASSIFIER}" = "DLinear" ]; then
    # DLinear reads the UEA/UCR .ts files produced by extract_features.py
    python experiments/grad_ssl/train_classifier.py \
        --train_ts_path "${TS_TRAIN}" \
        --test_ts_path  "${TS_TEST}" \
        --model         DLinear \
        --moving_avg    25 \
        --epochs        200 \
        --lr            1e-3 \
        --batch_size    64 \
        --seed          "${SEED}" \
        --device        "${DEVICE}"
else
    # Default MLP reads the .npz file
    python experiments/grad_ssl/train_classifier.py \
        --npz_path    "${NPZ_PATH}" \
        --model       MLP \
        --hidden      256 \
        --epochs      200 \
        --lr          1e-3 \
        --batch_size  64 \
        --seed        "${SEED}" \
        --device      "${DEVICE}"
fi

echo ""
echo "========================================================"
echo " Pipeline complete!  Artifacts in ./artifacts/"
echo "========================================================"

