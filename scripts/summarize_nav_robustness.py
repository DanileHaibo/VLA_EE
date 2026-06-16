#!/usr/bin/env python3
import re
import sys
from pathlib import Path


LABELS = {
    "00_no_ee_baseline": "No-EE",
    "01_baseline": "Baseline",
    "02_gaussian_0p1m": "Gaussian 0.1m",
    "03_gaussian_0p3m": "Gaussian 0.3m",
    "04_gaussian_0p5m": "Gaussian 0.5m",
    "05_gaussian_1p0m": "Gaussian 1.0m",
    "06_lateral_m0p25m": "Lateral -0.25m",
    "07_lateral_p0p25m": "Lateral +0.25m",
    "08_lateral_m0p5m": "Lateral -0.5m",
    "09_lateral_p0p5m": "Lateral +0.5m",
    "10_lateral_m1p0m": "Lateral -1.0m",
    "11_lateral_p1p0m": "Lateral +1.0m",
    "12_sparse_2": "Sparse x2",
    "13_sparse_4": "Sparse x4",
    "14_sparse_8": "Sparse x8",
    "15_time_m2": "Time -2",
    "16_time_m1": "Time -1",
    "17_time_p1": "Time +1",
    "18_time_p2": "Time +2",
}


def parse_float(text, key):
    m = re.search(rf"{re.escape(key)}:([0-9.eE+-]+)", text)
    return float(m.group(1)) if m else None


def parse_elapsed(text):
    infer_text = text.split("writing results to", 1)[0]
    matches = re.findall(r"\]\s+(\d+)/(\d+),\s+[0-9.]+\s+task/s,\s+elapsed:\s+(\d+)s", infer_text)
    if not matches:
        return None, None
    completed, total, elapsed = max(((int(a), int(b), int(c)) for a, b, c in matches), key=lambda x: x[0])
    if completed == 0:
        return None, total
    return elapsed * 1000.0 / completed, total


def parse_case(log_path):
    text = log_path.read_text(errors="ignore")
    lat_ms, total = parse_elapsed(text)
    l2s = [parse_float(text, f"plan_L2_{i}s") for i in (1, 2, 3)]
    cols = [parse_float(text, f"plan_obj_col_{i}s") for i in (1, 2, 3)]
    valid_match = re.search(r"Planning metrics over (\d+)/(\d+) fut-valid targets", text)
    valid = int(valid_match.group(1)) if valid_match else None
    avg_l2 = sum(l2s) / 3 if all(v is not None for v in l2s) else None
    avg_col = sum(cols) / 3 * 100 if all(v is not None for v in cols) else None
    avg_exit_layer = parse_float(text, "early_exit_layer")
    return {
        "lat_ms": lat_ms,
        "total": total,
        "valid": valid,
        "l2s": l2s,
        "cols": cols,
        "avg_l2": avg_l2,
        "avg_col": avg_col,
        "avg_exit_layer": avg_exit_layer,
    }


def fmt(value, digits=2):
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("work_dirs/nav_robustness_20pct")
    rows = []
    for key in sorted(LABELS):
        log_path = out_dir / f"{key}.log"
        if log_path.exists() and "plan_L2_2s:" in log_path.read_text(errors="ignore"):
            data = parse_case(log_path)
            rows.append((key, LABELS[key], data))

    print(r"\begin{tabular}{l|c|c|c|c}")
    print(r"\hline")
    print(r"Method & Sps (\%) & Lat (ms) & Avg. L2 (m) & Avg. Col. (\%) \\")
    print(r"\hline")
    for key, label, data in rows:
        sps = None
        if data["avg_exit_layer"] is not None:
            sps = (32.0 - data["avg_exit_layer"]) / 32.0 * 100.0
        print(
            f"{label} & {fmt(sps, 1)} & {fmt(data['lat_ms'], 0)} & "
            f"{fmt(data['avg_l2'], 3)} & {fmt(data['avg_col'], 2)} \\\\"
        )
    print(r"\hline")
    print(r"\end{tabular}")
    print()
    print("Detailed L2/Col:")
    for _, label, data in rows:
        l2_text = "/".join(fmt(v, 3) for v in data["l2s"])
        col_text = "/".join(fmt(v * 100 if v is not None else None, 2) for v in data["cols"])
        print(
            f"{label}: valid={data['valid']}/{data['total']}, "
            f"avg_exit={fmt(data['avg_exit_layer'], 2)}, L2={l2_text}, Col%={col_text}"
        )


if __name__ == "__main__":
    main()
