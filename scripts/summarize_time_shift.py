#!/usr/bin/env python3
import re
import sys
from pathlib import Path


ORDER = [
    ("time_shift_p2", "+2"),
    ("time_shift_p4", "+4"),
    ("time_shift_m2", "-2"),
    ("time_shift_m4", "-4"),
]


def parse_float(text, key):
    match = re.search(rf"{re.escape(key)}:([0-9.eE+-]+)", text)
    return float(match.group(1)) if match else None


def fmt(value, digits=3):
    return "--" if value is None else f"{value:.{digits}f}"


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "work_dirs/time_shift_contiguous_full_fp16_start12_fixed_fallback"
    )
    print("shift,L2@2s,L2@3s,avg_exit,Sps(%),EE_trigger(%),shift_valid(%),fallback(%),trigger_probe_L2,ref_delta_2s")
    for name, label in ORDER:
        log_path = out_dir / f"{name}.log"
        if not log_path.exists():
            continue
        text = log_path.read_text(errors="ignore")
        l2_2s = parse_float(text, "plan_L2_2s")
        l2_3s = parse_float(text, "plan_L2_3s")
        exit_layer = parse_float(text, "early_exit_layer")
        triggered = parse_float(text, "early_exit_triggered")
        probe_agg = parse_float(text, "early_exit_probe_l2_2s")
        ref_valid = parse_float(text, "early_exit_ref_valid")
        ref_used_original = parse_float(text, "early_exit_ref_used_original")
        ref_delta = parse_float(text, "early_exit_ref_delta_2s")
        sps = (32.0 - exit_layer) / 32.0 * 100.0 if exit_layer is not None else None
        trigger_probe = None
        if triggered is not None and triggered > 0 and probe_agg is not None:
            # Non-triggered samples are stored as -1.0 in the aggregate metric.
            trigger_probe = (probe_agg + 1.0 - triggered) / triggered
        print(
            ",".join(
                [
                    label,
                    fmt(l2_2s, 4),
                    fmt(l2_3s, 4),
                    fmt(exit_layer, 2),
                    fmt(sps, 2),
                    fmt(triggered * 100 if triggered is not None else None, 2),
                    fmt(ref_valid * 100 if ref_valid is not None else None, 2),
                    fmt(ref_used_original * 100 if ref_used_original is not None else None, 2),
                    fmt(trigger_probe, 3),
                    fmt(ref_delta, 3),
                ]
            )
        )


if __name__ == "__main__":
    main()
