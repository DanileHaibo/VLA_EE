#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/hhb}"
ORION_DIR="${ORION_DIR:-${ROOT_DIR}/Orion}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda-envs/orion/bin/python}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/data/orion_ckpts/Orion.pth}"
ANN_FILE="${ANN_FILE:-data/infos/b2d_infos_val.pkl}"
OUT_DIR="${OUT_DIR:-${ORION_DIR}/work_dirs/time_shift_contiguous_full_fp16_start12_fixed_fallback}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"

mkdir -p "${OUT_DIR}"
cd "${ORION_DIR}" || exit 1

export PYTHONPATH="${ORION_DIR}:${PYTHONPATH:-}"
export ORION_EE_LOG_LAYERS=0

declare -a EXP_NAMES=("time_shift_p2" "time_shift_p4" "time_shift_m2" "time_shift_m4")
declare -a EXP_VALUES=("2" "4" "-2" "-4")
read -r -a GPUS <<< "${GPU_LIST}"

run_one() {
  local name="$1"
  local value="$2"
  local gpu="$3"
  local log="${OUT_DIR}/${name}.log"
  local pkl="${OUT_DIR}/${name}.pkl"
  local time_file="${OUT_DIR}/${name}.time"

  echo "[$(date)] START ${name} contiguous FULL FP16 start_layer=12 mode=time_shift value=${value} gpu=${gpu}" | tee "${log}"
  local start
  start="$(date +%s)"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  ORION_NAV_PERTURB_MODE="time_shift" \
  ORION_NAV_PERTURB_VALUE="${value}" \
  ORION_NAV_PERTURB_SEED=2026 \
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
  local status=${PIPESTATUS[0]}
  local end
  end="$(date +%s)"
  {
    echo "status=${status}"
    echo "gpu=${gpu}"
    echo "mode=time_shift"
    echo "value=${value}"
    echo "precision=fp16"
    echo "start_layer=12"
    echo "contiguous_shift=1"
    echo "invalid_fallback_gt=1"
    echo "ann_file=${ANN_FILE}"
    echo "start=${start}"
    echo "end=${end}"
    echo "wall_seconds=$((end - start))"
  } > "${time_file}"
  echo "[$(date)] END ${name} status=${status}" | tee -a "${log}"
  return "${status}"
}

status=0
pids=()
for i in "${!EXP_NAMES[@]}"; do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  run_one "${EXP_NAMES[$i]}" "${EXP_VALUES[$i]}" "${gpu}" &
  pids+=("$!")
  sleep "${LAUNCH_STAGGER_SECONDS:-15}"
done

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
