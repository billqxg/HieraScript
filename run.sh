#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data}"
SAVE_BASE="${SAVE_BASE:-${ROOT_DIR}/results}"
PRETRAINED_DIR="${PRETRAINED_DIR:-${ROOT_DIR}/pretrained}"
BACKBONE="${BACKBONE:-RN50}"
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-32}"
SEED="${SEED:-42}"

mkdir -p ${SAVE_BASE}

python "${ROOT_DIR}/train_hrn.py" \
    --backbone "${BACKBONE}" \
    --clip_pretrained "${PRETRAINED_DIR}/${BACKBONE}.pt" \
    --save_dir "${SAVE_BASE}" \
    --freeze_clip \
    --epoch "${EPOCHS}" \
    --batch "${BATCH}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --data_path "${DATA_PATH}" \
    --lr_adjt Cos \
    >> "${SAVE_BASE}/run.log" 2>&1

echo "HieraScript training finished -> ${SAVE_BASE}"
