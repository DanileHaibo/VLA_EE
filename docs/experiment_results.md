# VLA-EE Reproducibility Notes

This repository contains the code changes and runner scripts used for the
Orion dynamic early-exit and efficiency baseline experiments on Bench2Drive
open-loop evaluation. Model weights, Bench2Drive data, generated `.pkl` files,
and raw `work_dirs/` logs are intentionally excluded.

## Environment

The experiments were run with Orion on the local environment:

- Python: `/root/autodl-tmp/conda-envs/orion/bin/python`
- Checkpoint: `/root/autodl-tmp/hhb/data/orion_ckpts/Orion.pth`
- Full validation ann file: `data/infos/b2d_infos_val.pkl`
- FP16 config: `adzoo/orion/configs/orion_stage3_fp16.py`

Override paths through environment variables in the scripts:

```bash
ROOT_DIR=/path/to/workspace \
ORION_DIR=/path/to/workspace/Orion \
PYTHON=/path/to/python \
CHECKPOINT=/path/to/Orion.pth \
ANN_FILE=data/infos/b2d_infos_val.pkl \
bash scripts/run_time_shift_pm2_pm4_full.sh
```

## Included Code Changes

- `mmcv/models/detectors/orion.py`
  - Dynamic early-exit probe from layer 12.
  - Navigation-reference perturbations for robustness experiments:
    Gaussian noise, lateral offset, sparse waypoints, parse waypoint, and
    contiguous-frame time shift.
  - Time-shift diagnostics:
    `early_exit_triggered`, `early_exit_probe_l2_2s`,
    `early_exit_ref_valid`, `early_exit_ref_used_original`,
    `early_exit_ref_delta_2s`.
  - AutoPrune and VLA-Pruner metric plumbing.
- `mmcv/datasets/b2d_orion_dataset.py`
  - `command_near_xy` collection.
  - Contiguous temporal-shift reference generation.
- `mmcv/utils/llava_llama.py`
  - AutoPrune and VLA-Pruner visual-token pruning logic.
- `adzoo/orion/configs/orion_stage3_fp16.py`
  - Collects the extra fields needed by the robustness experiments.

## Main Scripts

- `scripts/run_nav_robustness_20pct.sh`
  - Runs the navigation-reference robustness suite on a 20% ann file.
- `scripts/run_nav_robustness_full_parallel.sh`
  - Runs Gaussian/lateral/sparse/time-shift robustness experiments in parallel.
- `scripts/run_time_shift_pm2_pm4_full.sh`
  - Runs the final contiguous time-shift ablation:
    `+2`, `+4`, `-2`, `-4`.
- `scripts/run_autoprune_openloop.sh`
  - Runs the AutoPrune efficiency baseline.
- `scripts/run_vla_pruner_openloop.sh`
  - Runs the VLA-Pruner efficiency baseline.
- `scripts/summarize_nav_robustness.py`
  - Parses robustness logs into a LaTeX-style table.
- `scripts/summarize_time_shift.py`
  - Parses final time-shift logs and recovers trigger-only probe L2.

## Final Full-Dataset Time-Shift Results

Configuration:

- Full validation set: `data/infos/b2d_infos_val.pkl`
- FP16
- `early_exit_start_layer = 12`
- `early_exit_threshold = 0.5`
- Invalid route-boundary time-shift references fallback to original GT and are
  marked by `early_exit_ref_valid=0`, `early_exit_ref_used_original=1`.
- `Sps (%) = (32 - avg_exit_layer) / 32 * 100`.

| Shift | L2@2s | L2@3s | Avg Exit | Sps (%) | EE Trigger (%) | Shift Valid (%) | Fallback (%) | Trigger Probe L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `+2` | 0.6344 | 0.9721 | 24.35 | 23.91 | 40.45 | 95.37 | 4.63 | 0.304 |
| `+4` | 0.6424 | 1.0010 | 24.84 | 22.39 | 37.66 | 90.75 | 9.25 | 0.300 |
| `-2` | 0.6757 | 1.0867 | 23.06 | 27.93 | 45.10 | 100.00 | 0.00 | 0.325 |
| `-4` | 0.6856 | 1.0924 | 24.21 | 24.34 | 39.33 | 97.69 | 2.31 | 0.334 |

Raw logs were produced under:

```text
Orion/work_dirs/time_shift_contiguous_full_fp16_start12_fixed_fallback/
```

They are not tracked because `work_dirs/` and logs are ignored.

## Efficiency Baselines

AutoPrune and VLA-Pruner are integrated as mutually exclusive runtime modes:

```bash
bash scripts/run_autoprune_openloop.sh
bash scripts/run_vla_pruner_openloop.sh
```

Useful environment variables:

- AutoPrune:
  - `ORION_AUTOPRUNE=1`
  - `ORION_AUTOPRUNE_TARGET_TOKEN_NUM`
  - `ORION_AUTOPRUNE_X0`
  - `ORION_AUTOPRUNE_K0`
  - `ORION_AUTOPRUNE_GAMMA`
- VLA-Pruner:
  - `ORION_VLA_PRUNER=1`
  - `ORION_VLA_PRUNER_TARGET_TOKEN_NUM`
  - `ORION_VLA_PRUNER_LAYER`
  - `ORION_VLA_PRUNER_SEMANTIC_RATIO`
  - `ORION_VLA_PRUNER_TEMPORAL_ALPHA`
  - `ORION_VLA_PRUNER_USE_TEMPORAL`

## Excluded Artifacts

The following must be downloaded or generated locally and are not committed:

- Bench2Drive images and metadata `.pkl` files.
- Orion/UniDriveVLA checkpoints and HuggingFace weights.
- `work_dirs/`, raw logs, `.time`, `.pkl`, and benchmark output JSON files.
- Compiled CUDA extensions and Python caches.
