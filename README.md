# VLA_EE

Minimal reproducibility package for dynamic early exit and efficiency baselines
on Orion VLA open-loop Bench2Drive evaluation.

This repository stores code patches, configs, runner scripts, and result
summaries only. It does not include model weights, Bench2Drive data, generated
`.pkl` files, or raw `work_dirs/` outputs.

## Layout

```text
adzoo/orion/configs/orion_stage3_fp16.py
mmcv/datasets/b2d_orion_dataset.py
mmcv/models/detectors/orion.py
mmcv/utils/llava_llama.py
scripts/
docs/experiment_results.md
```

Copy or overlay these files onto an Orion checkout with the same directory
structure. The runner scripts assume the workspace layout used in the
experiments:

```text
/root/autodl-tmp/hhb/
  Orion/
  data/
    infos/
    orion_ckpts/
```

All paths can be overridden through environment variables such as `ROOT_DIR`,
`ORION_DIR`, `PYTHON`, `CHECKPOINT`, and `ANN_FILE`.

## Main Reproduction Commands

Final contiguous time-shift ablation on the full validation split:

```bash
bash scripts/run_time_shift_pm2_pm4_full.sh
python scripts/summarize_time_shift.py \
  /root/autodl-tmp/hhb/Orion/work_dirs/time_shift_contiguous_full_fp16_start12_fixed_fallback
```

Navigation robustness suite:

```bash
bash scripts/run_nav_robustness_20pct.sh
bash scripts/run_nav_robustness_full_parallel.sh
python scripts/summarize_nav_robustness.py /path/to/work_dir
```

Efficiency baselines:

```bash
bash scripts/run_autoprune_openloop.sh
bash scripts/run_vla_pruner_openloop.sh
```

See `docs/experiment_results.md` for configuration details and the final
reported results.
