#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/hhb}"
ORION_DIR="${ORION_DIR:-${ROOT_DIR}/Orion}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda-envs/orion/bin/python}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/data/orion_ckpts/Orion.pth}"
ANN_FILE="${ANN_FILE:-data/infos/b2d_infos_val.pkl}"
OUT_DIR="${OUT_DIR:-${ORION_DIR}/work_dirs/nav_robustness_skip_full}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
MAX_JOBS="${MAX_JOBS:-8}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"

mkdir -p "${OUT_DIR}"
cd "${ORION_DIR}" || exit 1

export PYTHONPATH="${ORION_DIR}:${PYTHONPATH:-}"
export ORION_EE_LOG_LAYERS=0

declare -a EXP_NAMES=(
  "00_no_ee_baseline"
  "01_baseline"
  "02_gaussian_0p1m"
  "03_gaussian_0p3m"
  "04_gaussian_0p5m"
  "05_gaussian_1p0m"
  "06_lateral_m0p25m"
  "07_lateral_p0p25m"
  "08_lateral_m0p5m"
  "09_lateral_p0p5m"
  "10_lateral_m1p0m"
  "11_lateral_p1p0m"
  "12_sparse_2"
  "13_sparse_4"
  "14_sparse_8"
  "15_time_m2"
  "16_time_m1"
  "17_time_p1"
  "18_time_p2"
)

declare -a EXP_MODES=(
  "disable_ee"
  "none"
  "gaussian"
  "gaussian"
  "gaussian"
  "gaussian"
  "lateral"
  "lateral"
  "lateral"
  "lateral"
  "lateral"
  "lateral"
  "sparse"
  "sparse"
  "sparse"
  "time_shift"
  "time_shift"
  "time_shift"
  "time_shift"
)

declare -a EXP_VALUES=(
  "0"
  "0"
  "0.1"
  "0.3"
  "0.5"
  "1.0"
  "-0.25"
  "0.25"
  "-0.5"
  "0.5"
  "-1.0"
  "1.0"
  "2"
  "4"
  "8"
  "-2"
  "-1"
  "1"
  "2"
)

read -r -a GPUS <<< "${GPU_LIST}"

running_jobs=0
next_gpu_idx=0
pids=()

run_one() {
  local name="$1"
  local mode="$2"
  local value="$3"
  local gpu="$4"
  local log="${OUT_DIR}/${name}.log"
  local pkl="${OUT_DIR}/${name}.pkl"
  local time_file="${OUT_DIR}/${name}.time"

  if grep -q "plan_L2_2s:" "${log}" 2>/dev/null; then
    echo "[SKIP] ${name} already has metrics" | tee -a "${OUT_DIR}/scheduler.log"
    return 0
  fi

  echo "[$(date)] START ${name} gpu=${gpu} mode=${mode} value=${value}" | tee "${log}"
  local start
  start="$(date +%s)"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  ORION_NAV_PERTURB_MODE="${mode}" \
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
    echo "mode=${mode}"
    echo "value=${value}"
    echo "start=${start}"
    echo "end=${end}"
    echo "wall_seconds=$((end - start))"
  } > "${time_file}"
  echo "[$(date)] END ${name} gpu=${gpu} status=${status}" | tee -a "${log}" | tee -a "${OUT_DIR}/scheduler.log"
  return "${status}"
}

for i in "${!EXP_NAMES[@]}"; do
  name="${EXP_NAMES[$i]}"
  mode="${EXP_MODES[$i]}"
  value="${EXP_VALUES[$i]}"
  gpu="${GPUS[$next_gpu_idx]}"
  next_gpu_idx=$(((next_gpu_idx + 1) % ${#GPUS[@]}))

  while [ "${running_jobs}" -ge "${MAX_JOBS}" ]; do
    wait -n
    running_jobs=$((running_jobs - 1))
  done

  (
    run_one "${name}" "${mode}" "${value}" "${gpu}"
  ) &
  pids+=("$!")
  running_jobs=$((running_jobs + 1))
  sleep "${LAUNCH_STAGGER_SECONDS:-20}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

"${PYTHON}" scripts/summarize_nav_robustness.py "${OUT_DIR}" | tee "${OUT_DIR}/summary.txt"
exit "${status}"
