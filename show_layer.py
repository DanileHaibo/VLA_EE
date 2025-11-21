#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 Early Exit 退出层分布柱状图（绿色系 + 大字体）
"""

import matplotlib.pyplot as plt
import numpy as np

# ================== 全局样式 ==================
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "axes.unicode_minus": False,
    "axes.titlesize": 24,   # 标题
    "axes.labelsize": 22,   # 坐标轴标题
    "xtick.labelsize": 18,  # 坐标轴刻度
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "axes.linewidth": 1.2,
})

# ----------------- 数据 -----------------
data1 = {
    16: (3180, 49.7),
    17: (28, 0.4),
    18: (15, 0.2),
    22: (16, 0.2),
    23: (11, 0.2),
    24: (130, 2.0),
    25: (464, 7.2),
    26: (15, 0.2),
    27: (121, 1.9),
    28: (57, 0.9),
    29: (107, 1.7),
    30: (65, 1.0),
    31: (15, 0.2),
    32: (2179, 34.0),
}

data2 = {
    16: (2936, 45.9),
    17: (28, 0.4),
    18: (9, 0.1),
    20: (3, 0.0),
    21: (5, 0.1),
    22: (9, 0.1),
    23: (8, 0.1),
    24: (24, 0.4),
    25: (27, 0.4),
    26: (34, 0.5),
    27: (53, 0.8),
    28: (486, 7.6),
    29: (49, 0.8),
    30: (18, 0.3),
    31: (20, 0.3),
    32: (2694, 42.1),
}

data3 = {
    16: (2662, 41.6),
    17: (23, 0.4),
    18: (5, 0.1),
    19: (1, 0.0),
    20: (4, 0.1),
    21: (6, 0.1),
    22: (2, 0.0),
    23: (1, 0.0),
    24: (4, 0.1),
    25: (3, 0.0),
    26: (18, 0.3),
    27: (20, 0.3),
    28: (23, 0.4),
    29: (22, 0.3),
    30: (20, 0.3),
    31: (29, 0.5),
    32: (3560, 55.6),
}

# ================== 计算函数 ==================
def calculate_sparsity(data):
    """计算相对于 32 层的稀疏性（sparsity）和平均层数"""
    total_cases = sum(data[layer][0] for layer in data)
    if total_cases == 0:
        return 0.0, 0.0

    total_layers_computed = sum(layer * data[layer][0] for layer in data)
    total_layers_full = total_cases * 32

    avg_layers = total_layers_computed / total_cases
    sparsity = 1.0 - (total_layers_computed / total_layers_full)
    return sparsity, avg_layers


# ================== 画单个分布图 ==================
def plot_distribution(data, title, xlabel, filename_suffix):
    # 把 32 层按 4 层一组合并
    group_labels = ['L1–4', 'L5–8', 'L9–12', 'L13–16',
                    'L17–20', 'L21–24', 'L25–28', 'L29–32']
    group_centers = [2.5, 6.5, 10.5, 14.5, 18.5, 22.5, 26.5, 30.5]

    group_case_counts = []
    group_percentages = []

    total_cases = sum(data[layer][0] for layer in data)

    for g in range(8):
        start_layer = g * 4 + 1
        end_layer = start_layer + 3
        group_count = 0
        for layer in range(start_layer, end_layer + 1):
            if layer in data:
                group_count += data[layer][0]

        group_case_counts.append(group_count)
        pct = (group_count / total_cases * 100) if total_cases > 0 else 0.0
        group_percentages.append(pct)

    # 颜色：绿色系，按高度做深浅
    max_pct = max(group_percentages) if max(group_percentages) > 0 else 1.0
    base_dark = np.array([46, 125, 50])    # #2E7D32 深绿
    base_light = np.array([198, 230, 202]) # 浅绿

    colors = []
    for pct in group_percentages:
        if pct == 0:
            colors.append("#E0E0E0")  # 无数据灰色
        else:
            alpha = pct / max_pct  # 0~1，越大越深
            rgb = base_light * (1 - alpha) + base_dark * alpha
            colors.append('#%02x%02x%02x' % tuple(rgb.astype(int)))

    # ========== 正式画图 ==========
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    ax.set_facecolor("white")

    bars = ax.bar(group_centers, group_percentages,
                  width=3.0,
                  color=colors,
                  edgecolor="#1B5E20",
                  linewidth=1.2,
                  zorder=3)

    # 在柱子上写 case 数（字体再大一点）
    for x, pct, cnt in zip(group_centers, group_percentages, group_case_counts):
        if cnt == 0:
            continue
        if pct > max_pct * 0.25:
            # 高柱：文字放在柱子内部，白字
            ax.text(x, pct * 0.92, f"{cnt:,}",
                    ha="center", va="top",
                    fontsize=16, fontweight="bold",
                    color="white")
        else:
            # 低柱：文字放在柱子上方
            ax.text(x, pct + max_pct * 0.02, f"{cnt:,}",
                    ha="center", va="bottom",
                    fontsize=16, fontweight="bold",
                    color="#1B5E20")

    # 坐标轴 & 标题
    ax.set_xlabel(xlabel, labelpad=10)   # 这里用传进来的 "(a)/(b)/(c) Exit layer group"
    ax.set_ylabel("Percentage (%)", labelpad=10)
    ax.set_title(title, pad=14)         # 这里用传进来的 title

    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels)
    ax.set_xlim(0, 33)

    # y 轴到整十
    ymax = (np.ceil(max_pct / 10.0) * 10.0) if max_pct > 0 else 10
    ax.set_ylim(0, ymax * 1.08)
    ax.set_yticks(np.arange(0, ymax + 1e-3, 10))
    ax.set_yticklabels([f"{int(v)}" for v in np.arange(0, ymax + 1e-3, 10)])

    # 网格线：只要水平线
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.3)
    ax.grid(b=False, axis="x")

    # 去掉上/右边框
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # 统计信息：比之前稍大
    early_exit_cases = total_cases - group_case_counts[7]  # 最后一组视为“未早退”
    early_exit_rate = (early_exit_cases / total_cases * 100) if total_cases > 0 else 0.0
    sparsity, avg_layers = calculate_sparsity(data)
    sparsity_pct = sparsity * 100

    stats_str = (
        f"Total: {total_cases:,}   "
        f"Early exit: {early_exit_rate:.1f}%   "
        f"Sparsity: {sparsity_pct:.1f}%   "
        f"Avg layers: {avg_layers:.1f}"
    )
    ax.text(0.9, 0.95, stats_str,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=14, color="dimgray")

    plt.tight_layout()

    plt.savefig(f"exit_layer_distribution{filename_suffix}.png",
                dpi=300, bbox_inches="tight")
    plt.savefig(f"exit_layer_distribution{filename_suffix}.pdf",
                dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Saved] exit_layer_distribution{filename_suffix}.png / .pdf")
    print(f"  Total cases   : {total_cases:,}")
    print(f"  Early exit    : {early_exit_rate:.1f}%")
    print(f"  Sparsity      : {sparsity_pct:.2f}%  (vs 32 layers)")
    print(f"  Avg #layers   : {avg_layers:.2f}")
    return sparsity, avg_layers, total_cases, early_exit_rate


# ================== 生成三张图 ==================
print("=" * 80)
print("数据集 1...")
s1, a1, t1, e1 = plot_distribution(
    data1,
    title="Early-exit layer distribution in δ = 2m",
    xlabel="(a) Exit layer group",
    filename_suffix="_1"
)

print("=" * 80)
print("数据集 2...")
s2, a2, t2, e2 = plot_distribution(
    data2,
    title="Early-exit layer distribution in δ = 1m",
    xlabel="(b) Exit layer group",
    filename_suffix="_2"
)

print("=" * 80)
print("数据集 3...")
s3, a3, t3, e3 = plot_distribution(
    data3,
    title="Early-exit layer distribution in δ = 0.5m",
    xlabel="(c) Exit layer group",
    filename_suffix="_3"
)

print("=" * 80)
print("Summary (sparsity vs 32 layers):")
print(f"  Data 1: sparsity = {s1*100:.2f}%, avg layers = {a1:.2f}, early exit = {e1:.1f}%")
print(f"  Data 2: sparsity = {s2*100:.2f}%, avg layers = {a2:.2f}, early exit = {e2:.1f}%")
print(f"  Data 3: sparsity = {s3*100:.2f}%, avg layers = {a3:.2f}, early exit = {e3:.1f}%")
print("=" * 80)
