#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/hhb}"
ORION_DIR="${ORION_DIR:-${ROOT_DIR}/Orion}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda-envs/orion/bin/python}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/data/orion_ckpts/Orion.pth}"
ANN_FILE="${ANN_FILE:-data/infos/b2d_infos_val.pkl}"
OUT_DIR="${OUT_DIR:-${ORION_DIR}/work_dirs/autoprune_openloop}"
GPU="${GPU:-0}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"

TARGET_TOKEN_NUM="${TARGET_TOKEN_NUM:-64}"
X0="${X0:-14.9}"
K0="${K0:-0.4}"
GAMMA="${GAMMA:-0.2}"

mkdir -p "${OUT_DIR}"
cd "${ORION_DIR}" || exit 1

export PYTHONPATH="${ORION_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export ORION_EE_LOG_LAYERS=0
export ORION_AUTOPRUNE=1
export ORION_AUTOPRUNE_TARGET_TOKEN_NUM="${TARGET_TOKEN_NUM}"
export ORION_AUTOPRUNE_X0="${X0}"
export ORION_AUTOPRUNE_K0="${K0}"
export ORION_AUTOPRUNE_GAMMA="${GAMMA}"

name="autoprune_t${TARGET_TOKEN_NUM}_x0${X0}_k${K0}_g${GAMMA}"
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
  echo "x0=${X0}"
  echo "k0=${K0}"
  echo "gamma=${GAMMA}"
  echo "start=${start}"
  echo "end=${end}"
  echo "wall_seconds=$((end - start))"
} > "${time_file}"
echo "[$(date)] END ${name} status=${status}" | tee -a "${log}"
exit "${status}"
