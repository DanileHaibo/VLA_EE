#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/hhb}"
ORION_DIR="${ORION_DIR:-${ROOT_DIR}/Orion}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda-envs/orion5090/bin/python}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/data/orion_ckpts/Orion.pth}"
ANN_FILE="${ANN_FILE:-data/infos/b2d_infos_val.pkl}"
OUT_DIR="${OUT_DIR:-${ORION_DIR}/work_dirs/vla_pruner_openloop}"
GPU="${GPU:-0}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"

TARGET_TOKEN_NUM="${TARGET_TOKEN_NUM:-64}"
PRUNE_LAYER="${PRUNE_LAYER:-2}"
SEMANTIC_RATIO="${SEMANTIC_RATIO:-0.5}"
TEMPORAL_ALPHA="${TEMPORAL_ALPHA:-0.7}"
USE_TEMPORAL="${USE_TEMPORAL:-1}"

mkdir -p "${OUT_DIR}"
cd "${ORION_DIR}" || exit 1

export PYTHONPATH="${ORION_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export ORION_EE_LOG_LAYERS=0
export ORION_AUTOPRUNE=0
export ORION_VLA_PRUNER=1
export ORION_VLA_PRUNER_TARGET_TOKEN_NUM="${TARGET_TOKEN_NUM}"
export ORION_VLA_PRUNER_LAYER="${PRUNE_LAYER}"
export ORION_VLA_PRUNER_SEMANTIC_RATIO="${SEMANTIC_RATIO}"
export ORION_VLA_PRUNER_TEMPORAL_ALPHA="${TEMPORAL_ALPHA}"
export ORION_VLA_PRUNER_USE_TEMPORAL="${USE_TEMPORAL}"

name="vla_pruner_t${TARGET_TOKEN_NUM}_l${PRUNE_LAYER}_sr${SEMANTIC_RATIO}_ta${TEMPORAL_ALPHA}_tmp${USE_TEMPORAL}"
name="${name//./p}"
log="${OUT_DIR}/${name}.log"
pkl="${OUT_DIR}/${name}.pkl"
time_file="${OUT_DIR}/${name}.time"

echo "[$(date)] START ${name} gpu=${GPU}" | tee "${log}"
start="$(date +%s)"
"${PYTHON}" -u adzoo/orion/test.py \
  adzoo/orion/configs/orion_stage3_fp16.py \
  "${CHECKPOINT}" \
  --launcher none \
  --eval bbox \
  --out "${pkl}" \
  --cfg-options \
    data.test.ann_file="${ANN_FILE}" \
    data.workers_per_gpu="${WORKERS_PER_GPU}" \
    data.samples_per_gpu=1 \
  2>&1 | tee -a "${log}"
status=${PIPESTATUS[0]}
end="$(date +%s)"
{
  echo "status=${status}"
  echo "gpu=${GPU}"
  echo "target_token_num=${TARGET_TOKEN_NUM}"
  echo "prune_layer=${PRUNE_LAYER}"
  echo "semantic_ratio=${SEMANTIC_RATIO}"
  echo "temporal_alpha=${TEMPORAL_ALPHA}"
  echo "use_temporal=${USE_TEMPORAL}"
  echo "start=${start}"
  echo "end=${end}"
  echo "wall_seconds=$((end - start))"
} > "${time_file}"
echo "[$(date)] END ${name} status=${status}" | tee -a "${log}"
exit "${status}"
