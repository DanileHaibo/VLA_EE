#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/hhb}"
ORION_DIR="${ORION_DIR:-${ROOT_DIR}/Orion}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda-envs/orion/bin/python}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/data/orion_ckpts/Orion.pth}"
ANN_FILE="${ANN_FILE:-data/infos/b2d_infos_val_20pct.pkl}"
OUT_DIR="${OUT_DIR:-${ORION_DIR}/work_dirs/nav_robustness_20pct}"
GPU="${GPU:-0}"

mkdir -p "${OUT_DIR}"
cd "${ORION_DIR}" || exit 1

export PYTHONPATH="${ORION_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export ORION_EE_LOG_LAYERS=0

run_one() {
  local name="$1"
  local mode="$2"
  local value="$3"
  local log="${OUT_DIR}/${name}.log"
  local pkl="${OUT_DIR}/${name}.pkl"
  local time_file="${OUT_DIR}/${name}.time"

  if grep -q "plan_L2_2s:" "${log}" 2>/dev/null; then
    echo "[SKIP] ${name} already has metrics"
    return 0
  fi

  echo "[$(date)] START ${name} mode=${mode} value=${value}" | tee "${log}"
  local start
  start="$(date +%s)"
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
      data.workers_per_gpu=4 \
      data.samples_per_gpu=1 \
    2>&1 | tee -a "${log}"
  local status=${PIPESTATUS[0]}
  local end
  end="$(date +%s)"
  {
    echo "status=${status}"
    echo "start=${start}"
    echo "end=${end}"
    echo "wall_seconds=$((end - start))"
  } > "${time_file}"
  echo "[$(date)] END ${name} status=${status}" | tee -a "${log}"
  return "${status}"
}

run_one "00_no_ee_baseline" "disable_ee" "0"
run_one "01_baseline" "none" "0"
run_one "02_gaussian_0p1m" "gaussian" "0.1"
run_one "03_gaussian_0p3m" "gaussian" "0.3"
run_one "04_gaussian_0p5m" "gaussian" "0.5"
run_one "05_gaussian_1p0m" "gaussian" "1.0"
run_one "06_lateral_m0p25m" "lateral" "-0.25"
run_one "07_lateral_p0p25m" "lateral" "0.25"
run_one "08_lateral_m0p5m" "lateral" "-0.5"
run_one "09_lateral_p0p5m" "lateral" "0.5"
run_one "10_lateral_m1p0m" "lateral" "-1.0"
run_one "11_lateral_p1p0m" "lateral" "1.0"
run_one "12_sparse_2" "sparse" "2"
run_one "13_sparse_4" "sparse" "4"
run_one "14_sparse_8" "sparse" "8"
run_one "15_time_m2" "time_shift" "-2"
run_one "16_time_m1" "time_shift" "-1"
run_one "17_time_p1" "time_shift" "1"
run_one "18_time_p2" "time_shift" "2"

"${PYTHON}" scripts/summarize_nav_robustness.py "${OUT_DIR}" | tee "${OUT_DIR}/summary.txt"
