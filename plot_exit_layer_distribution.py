# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# 生成Early Exit退出层分布柱状图
# """

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Rectangle
# import matplotlib.patches as mpatches

# # 设置中文字体和样式
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
# plt.rcParams['axes.unicode_minus'] = False
# # 尝试使用可用的样式，如果失败则使用默认样式
# try:
#     plt.style.use('seaborn-darkgrid')
# except:
#     try:
#         plt.style.use('seaborn-v0_8-darkgrid')
#     except:
#         try:
#             plt.style.use('dark_background')
#         except:
#             pass  # 使用默认样式

# # 数据1：Layer -> (cases, percentage)
# data1 = {
#     16: (3180, 49.7),
#     17: (28, 0.4),
#     18: (15, 0.2),
#     22: (16, 0.2),
#     23: (11, 0.2),
#     24: (130, 2.0),
#     25: (464, 7.2),
#     26: (15, 0.2),
#     27: (121, 1.9),
#     28: (57, 0.9),
#     29: (107, 1.7),
#     30: (65, 1.0),
#     31: (15, 0.2),
#     32: (2179, 34.0),
# }

# # 数据2：Layer -> (cases, percentage)
# data2 = {
#     16: (2936, 45.9),
#     17: (28, 0.4),
#     18: (9, 0.1),
#     20: (3, 0.0),
#     21: (5, 0.1),
#     22: (9, 0.1),
#     23: (8, 0.1),
#     24: (24, 0.4),
#     25: (27, 0.4),
#     26: (34, 0.5),
#     27: (53, 0.8),
#     28: (486, 7.6),
#     29: (49, 0.8),
#     30: (18, 0.3),
#     31: (20, 0.3),
#     32: (2694, 42.1),
# }

# # 数据3：Layer -> (cases, percentage)
# data3 = {
#     16: (2662, 41.6),
#     17: (23, 0.4),
#     18: (5, 0.1),
#     19: (1, 0.0),
#     20: (4, 0.1),
#     21: (6, 0.1),
#     22: (2, 0.0),
#     23: (1, 0.0),
#     24: (4, 0.1),
#     25: (3, 0.0),
#     26: (18, 0.3),
#     27: (20, 0.3),
#     28: (23, 0.4),
#     29: (22, 0.3),
#     30: (20, 0.3),
#     31: (29, 0.5),
#     32: (3560, 55.6),
# }

# def calculate_sparsity(data):
#     """计算相对于32层的稀疏性（sparsity）"""
#     total_cases = sum(data[layer][0] for layer in data)
#     if total_cases == 0:
#         return 0.0, 0.0
    
#     # 计算实际运行的总层数
#     total_layers_computed = sum(layer * data[layer][0] for layer in data)
    
#     # 如果所有case都运行32层，总层数应该是 total_cases * 32
#     total_layers_full = total_cases * 32
    
#     # 计算平均运行层数
#     avg_layers = total_layers_computed / total_cases
    
#     # Sparsity = 1 - (实际计算量 / 全量计算量) = 节省的计算比例
#     sparsity = 1.0 - (total_layers_computed / total_layers_full)
    
#     return sparsity, avg_layers

# def plot_distribution(data, title_suffix="", filename_suffix=""):
#     """生成单个分布图"""
#     # 准备数据：每4层合并为一个柱状图（共8组）
#     # 组1: L1-L4, 组2: L5-L8, 组3: L9-L12, 组4: L13-L16, 
#     # 组5: L17-L20, 组6: L21-L24, 组7: L25-L28, 组8: L29-L32
#     group_labels = ['L1-4', 'L5-8', 'L9-12', 'L13-16', 'L17-20', 'L21-24', 'L25-28', 'L29-32']
#     group_centers = [2.5, 6.5, 10.5, 14.5, 18.5, 22.5, 26.5, 30.5]  # 每组的中点位置
#     group_case_counts = []
#     group_percentages = []
    
#     total_cases = sum(data[layer][0] for layer in data)
    
#     for group_idx in range(8):
#         start_layer = group_idx * 4 + 1
#         end_layer = start_layer + 3
#         # 计算这4层的总case数
#         group_count = 0
#         for layer in range(start_layer, end_layer + 1):
#             if layer in data:
#                 group_count += data[layer][0]
        
#         group_case_counts.append(group_count)
#         # 计算百分比
#         group_percentage = (group_count / total_cases * 100) if total_cases > 0 else 0.0
#         group_percentages.append(group_percentage)

#     # 鲜艳的配色方案（使用绿色系）
#     # 根据百分比使用不同深度的绿色
#     colors = []
#     for i, count in enumerate(group_case_counts):
#         if count == 0:
#             colors.append('#E0E0E0')  # 浅灰色表示无数据
#         else:
#             # 根据百分比使用不同深度的绿色
#             if group_percentages[i] > 30:
#                 colors.append('#2E7D32')  # 深绿色（高百分比）
#             elif group_percentages[i] > 5:
#                 colors.append('#43A047')  # 中绿色（中百分比）
#             else:
#                 colors.append('#66BB6A')  # 浅绿色（低百分比）
    
#     # 创建图形，使用更大的尺寸和更好的DPI
#     fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')
#     ax.set_facecolor('white')

#     # 绘制柱状图（绿色配色，学术风格）- 每组宽度为3.5
#     bars = ax.bar(group_centers, group_percentages, color=colors, edgecolor='#1B5E20', 
#                    linewidth=1.3, alpha=1.0, width=3.5, zorder=3)
    
#     # 在柱条顶部添加case数量标签（美观字体，放大）
#     max_percentage = max(group_percentages) if max(group_percentages) > 0 else 1.0
#     for i, (center, percentage, count, label) in enumerate(zip(group_centers, group_percentages, group_case_counts, group_labels)):
#         if count > 0:  # 只显示有数据的组
#             # 根据柱条高度调整标签位置
#             if percentage > max_percentage * 0.3:  # 高柱条
#                 # 标签放在柱条内部，白色文字
#                 label_y = percentage * 0.95  # 放在柱条内部95%位置
#                 ax.text(center, label_y, f'{count:,}', 
#                         ha='center', va='center', fontsize=22, fontweight='bold',
#                         color='white', family='sans-serif')
#             else:  # 低柱条
#                 # 标签放在柱条外部顶部
#                 label_y = percentage + max_percentage * 0.02
#                 ax.text(center, label_y, f'{count:,}', 
#                         ha='center', va='bottom', fontsize=22, fontweight='bold',
#                         color='#2E7D32', family='sans-serif')
    
#     # 设置标签和标题（美观字体，放大）
#     ax.set_xlabel('Exit Layer', fontsize=30, fontweight='bold', color='black', labelpad=15, family='sans-serif')
#     ax.set_ylabel('Percentage (%)', fontsize=30, fontweight='bold', color='black', labelpad=15, family='sans-serif')
#     ax.set_title(f'Early Exit Layer Distribution{title_suffix}', fontsize=30, fontweight='bold', 
#                  pad=30, color='black', family='sans-serif')
    
#     # 设置x轴刻度（美观字体，放大）- 显示每组标签
#     ax.set_xticks(group_centers)
#     ax.set_xticklabels(group_labels, fontsize=22, color='black', family='sans-serif', fontweight='bold')
#     ax.set_xlim(-1, 33)
    
#     # 设置y轴（美观字体，放大）
#     max_percentage = max(group_percentages) if max(group_percentages) > 0 else 1.0
#     ax.set_ylim(0, max_percentage * 1.12)  # 顶部留空间给标签
#     ax.set_yticks(np.arange(0, max_percentage + 10, 10))
#     ax.set_yticklabels([f'{int(y)}%' for y in np.arange(0, max_percentage + 10, 10)], 
#                        fontsize=30, color='black', family='sans-serif', fontweight='bold')
    
#     # 添加网格线（学术风格，更简洁）
#     ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.6, color='#CCCCCC', zorder=1)
#     ax.grid(axis='x', alpha=0.1, linestyle=':', linewidth=0.4, color='#CCCCCC', zorder=1)
    
#     # 添加总case数和Early Exit率信息（放在图的中间偏上，美观字体，放大）
#     total_cases = sum(group_case_counts)
#     # 计算early exit cases（除了L29-32组的所有case）
#     early_exit_cases = total_cases - group_case_counts[7]  # 减去最后一组（L29-32）
#     early_exit_rate = (early_exit_cases / total_cases) * 100 if total_cases > 0 else 0
    
#     # 计算Sparsity
#     sparsity, avg_layers = calculate_sparsity(data)
#     sparsity_percent = sparsity * 100
    
#     # Total Cases和Early Exit Rate放在中间偏上，左右分布
#     info_text = f'Total Cases: {total_cases:,}'
#     ax.text(0.35, 0.95, info_text, transform=ax.transAxes, fontsize=30, fontweight='bold',
#             verticalalignment='top', horizontalalignment='left', color='black', family='sans-serif',
#             bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
#                      edgecolor='black', linewidth=1.2, alpha=0.95))
    
#     stats_text = f'Early Exit Rate: {early_exit_rate:.1f}%'
#     ax.text(0.65, 0.95, stats_text, transform=ax.transAxes, fontsize=30, fontweight='bold',
#             verticalalignment='top', horizontalalignment='left', color='black', family='sans-serif',
#             bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
#                      edgecolor='black', linewidth=1.2, alpha=0.95))
    
#     # Sparsity信息放在中间偏下
#     sparsity_text = f'Sparsity: {sparsity_percent:.2f}% (Avg Layers: {avg_layers:.2f})'
#     ax.text(0.5, 0.02, sparsity_text, transform=ax.transAxes, fontsize=30, fontweight='bold',
#             verticalalignment='bottom', horizontalalignment='center', color='black', family='sans-serif',
#             bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
#                      edgecolor='black', linewidth=1.2, alpha=0.95))
    
#     # 学术风格坐标轴（简洁）
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_color('black')
#     ax.spines['bottom'].set_color('black')
#     ax.spines['left'].set_linewidth(0.8)
#     ax.spines['bottom'].set_linewidth(0.8)
    
#     # 调整布局
#     plt.tight_layout()
    
#     # 保存图片（高质量）
#     plt.savefig(f'exit_layer_distribution{filename_suffix}.png', dpi=300, bbox_inches='tight', 
#                 facecolor='white', edgecolor='none')
#     plt.savefig(f'exit_layer_distribution{filename_suffix}.pdf', bbox_inches='tight', 
#                 facecolor='white', edgecolor='none')
#     print(f"图表已保存为: exit_layer_distribution{filename_suffix}.png 和 exit_layer_distribution{filename_suffix}.pdf")
#     print(f"总case数: {total_cases:,}")
#     print(f"Early Exit率: {early_exit_rate:.1f}%")
#     print(f"Sparsity: {sparsity_percent:.2f}% (相对于32层)")
#     print(f"平均运行层数: {avg_layers:.2f}")
    
#     plt.close()  # 关闭图形，释放内存
    
#     return sparsity, avg_layers, total_cases, early_exit_rate

# # 生成第一个图（数据1）
# print("="*80)
# print("生成第一个图表（数据1）...")
# sparsity1, avg_layers1, total1, eer1 = plot_distribution(data1, title_suffix="", filename_suffix="_1")

# # 生成第二个图（数据2）
# print("="*80)
# print("生成第二个图表（数据2）...")
# sparsity2, avg_layers2, total2, eer2 = plot_distribution(data2, title_suffix="", filename_suffix="_2")

# # 生成第三个图（数据3）
# print("="*80)
# print("生成第三个图表（数据3）...")
# sparsity3, avg_layers3, total3, eer3 = plot_distribution(data3, title_suffix="", filename_suffix="_3")

# # 打印汇总统计
# print("="*80)
# print("所有图表已生成完成！")
# print("="*80)
# print("Sparsity汇总统计（相对于32层）：")
# print(f"数据集1: Sparsity = {sparsity1*100:.2f}%, 平均层数 = {avg_layers1:.2f}, Early Exit率 = {eer1:.1f}%")
# print(f"数据集2: Sparsity = {sparsity2*100:.2f}%, 平均层数 = {avg_layers2:.2f}, Early Exit率 = {eer2:.1f}%")
# print(f"数据集3: Sparsity = {sparsity3*100:.2f}%, 平均层数 = {avg_layers3:.2f}, Early Exit率 = {eer3:.1f}%")
# print("="*80)


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 Early Exit 退出层分布柱状图（简洁学术风格）
"""

import matplotlib.pyplot as plt
import numpy as np

# ================== 全局样式 ==================
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "axes.unicode_minus": False,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
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
def plot_distribution(data, title_suffix="", filename_suffix=""):
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

    # 颜色：单色系绿色，按高度做深浅
    max_pct = max(group_percentages) if max(group_percentages) > 0 else 1.0
    base_dark = np.array([46, 125, 50])   # #2E7D32
    base_light = np.array([198, 230, 202])  # 淡绿

    colors = []
    for pct in group_percentages:
        if pct == 0:
            colors.append("#E0E0E0")
        else:
            alpha = pct / max_pct  # 0~1，越大越深
            rgb = base_light * (1 - alpha) + base_dark * alpha
            colors.append('#%02x%02x%02x' % tuple(rgb.astype(int)))

    # ========== 正式画图 ==========
    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor="white")
    ax.set_facecolor("white")

    bars = ax.bar(group_centers, group_percentages,
                  width=3.0,
                  color=colors,
                  edgecolor="#1B5E20",
                  linewidth=1.0,
                  zorder=3)

    # 在柱子上写 case 数（小一点）
    for x, pct, cnt in zip(group_centers, group_percentages, group_case_counts):
        if cnt == 0:
            continue
        if pct > max_pct * 0.25:
            # 高柱：文字放在柱子内部，白字
            ax.text(x, pct * 0.92, f"{cnt:,}",
                    ha="center", va="top",
                    fontsize=11, fontweight="bold",
                    color="white")
        else:
            # 低柱：文字放在柱子上方
            ax.text(x, pct + max_pct * 0.02, f"{cnt:,}",
                    ha="center", va="bottom",
                    fontsize=11, fontweight="bold",
                    color="#1B5E20")

    # 坐标轴 & 标题
    ax.set_xlabel("Exit layer group", labelpad=6)
    ax.set_ylabel("Percentage (%)", labelpad=6)
    ax.set_title(f"Early-exit layer distribution{title_suffix}", pad=10)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels)
    ax.set_xlim(0, 33)

    # y 轴上到一个整十
    ymax = (np.ceil(max_pct / 10.0) * 10.0) if max_pct > 0 else 10
    ax.set_ylim(0, ymax * 1.05)
    ax.set_yticks(np.arange(0, ymax + 1e-3, 10))
    ax.set_yticklabels([f"{int(v)}" for v in np.arange(0, ymax + 1e-3, 10)])

    # 只画水平网格线
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.3)
    ax.grid(b=False, axis="x")

    # 去掉上/右边框
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # 统计信息：小字放在图的上方右侧
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
    ax.text(0.90, 0.95, stats_str,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=10.5, color="dimgray")

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
s1, a1, t1, e1 = plot_distribution(data1, title_suffix="", filename_suffix="_1")

print("=" * 80)
print("数据集 2...")
s2, a2, t2, e2 = plot_distribution(data2, title_suffix="", filename_suffix="_2")

print("=" * 80)
print("数据集 3...")
s3, a3, t3, e3 = plot_distribution(data3, title_suffix="", filename_suffix="_3")

print("=" * 80)
print("Summary (sparsity vs 32 layers):")
print(f"  Data 1: sparsity = {s1*100:.2f}%, avg layers = {a1:.2f}, early exit = {e1:.1f}%")
print(f"  Data 2: sparsity = {s2*100:.2f}%, avg layers = {a2:.2f}, early exit = {e2:.1f}%")
print(f"  Data 3: sparsity = {s3*100:.2f}%, avg layers = {a3:.2f}, early exit = {e3:.1f}%")
print("=" * 80)
