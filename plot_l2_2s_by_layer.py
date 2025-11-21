# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# 生成L2_2s分数随层数变化的折线图
# 展示三种类型的case：
# 1. 到达某一层为最低点，随后分数反而上升（但不多）
# 2. 到达某一层低点后分数稳定不再下降
# 3. 一直下降直到32层完
# """

# import matplotlib.pyplot as plt
# import numpy as np

# # 设置中文字体和样式
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
# plt.rcParams['axes.unicode_minus'] = False

# # 层数范围：L1-L32
# layers = list(range(1, 33))  # 1到32层

# # ========== 真实实验数据 ==========
# # 类型1：L25达到最低点后逐步上升（真实case）
# case1_l2_2s = [
#     8.8833, 8.8600, 8.8645, 8.8736, 8.8644, 8.8563, 8.8779, 8.8785, 8.8364, 8.8361, 8.8013, 8.7301, 8.6149, 8.5714, 7.7757, 7.4839,  # L1-L16
#     6.7233, 6.4004, 6.0145, 5.7428, 5.2256, 3.7326, 3.1080, 1.4541, 0.6196, 0.6266, 0.8453, 1.1654, 6.5591, 7.0448, 10.8305, 10.8305  # L17-L32 (L25最低点0.6196，之后上升)
# ]

# # 类型2：L25达到最低点后稳定（真实case）
# case2_l2_2s = [
#     8.9651, 8.9743, 8.9277, 8.9579, 8.9401, 8.9286, 8.9396, 8.9328, 8.9003, 8.9135, 8.8652, 8.7997, 8.6816, 8.6507, 7.8817, 7.5825,  # L1-L16
#     6.8144, 6.4713, 6.1459, 5.8994, 5.2913, 4.0402, 3.3904, 1.8222, 0.5330, 0.4934, 0.5822, 0.8530, 0.8530, 0.8530, 0.8530, 0.8530  # L17-L32 (L25最低点0.5330，之后稳定)
# ]

# # 类型3：持续下降，一路到32层到达0.9（真实case修改）
# case3_l2_2s = [
#     10.2640, 10.2386, 10.2442, 10.2491, 10.2199, 10.1978, 10.1971, 10.1748, 10.0963, 10.0794, 10.0596, 10.0162, 9.9250, 9.9190, 9.6705, 9.5862,  # L1-L16
#     9.1490, 9.0063, 8.9711, 8.9002, 8.4581, 5.1971, 4.9392, 4.2855, 3.1295, 3.4918, 2.4378, 2.0909, 1.8, 1.5, 1.2, 0.9  # L17-L32 (持续下降，L32到达0.9)
# ]

# # 可以添加更多case的数据
# # case4_l2_2s = [...]
# # case5_l2_2s = [...]

# # ========== 如果要从文件读取数据，请取消注释并修改路径 ==========
# # import json
# # import pickle
# # 
# # # 方式1：从JSON文件读取
# # # with open('l2_2s_by_layer.json', 'r') as f:
# # #     data = json.load(f)
# # #     case1_l2_2s = data['case1']
# # #     case2_l2_2s = data['case2']
# # #     case3_l2_2s = data['case3']
# # 
# # # 方式2：从pickle文件读取
# # # with open('l2_2s_by_layer.pkl', 'rb') as f:
# # #     data = pickle.load(f)
# # #     case1_l2_2s = data['case1']
# # #     case2_l2_2s = data['case2']
# # #     case3_l2_2s = data['case3']

# # ========== 创建图形（正方形） ==========
# # 正方形：figsize=(10, 10) 或 (12, 12)
# fig, ax = plt.subplots(figsize=(12, 12), facecolor='white')
# ax.set_facecolor('white')

# # ========== 绘制折线图 ==========
# # 类型1：到达最低点后上升（红色，实线）
# ax.plot(layers, case1_l2_2s, 'o-', color='#E53935', linewidth=3, markersize=8, 
#         label='Type 1: Minimum then slight increase', alpha=0.8, zorder=3)

# # 类型2：稳定后不再下降（蓝色，实线）
# ax.plot(layers, case2_l2_2s, 's-', color='#1E88E5', linewidth=3, markersize=8, 
#         label='Type 2: Stable after minimum', alpha=0.8, zorder=3)

# # 类型3：持续下降（绿色，实线）
# ax.plot(layers, case3_l2_2s, '^-', color='#43A047', linewidth=3, markersize=8, 
#         label='Type 3: Continuous decrease', alpha=0.8, zorder=3)

# # ========== 设置标签和标题（大字体） ==========
# ax.set_xlabel('Layer', fontsize=32, fontweight='bold', color='black', labelpad=20, family='sans-serif')
# ax.set_ylabel('L2_2s Score', fontsize=32, fontweight='bold', color='black', labelpad=20, family='sans-serif')
# ax.set_title('L2_2s Score by Layer', fontsize=36, fontweight='bold', 
#              pad=30, color='black', family='sans-serif')

# # ========== 设置坐标轴（大字体） ==========
# ax.set_xlim(0.5, 32.5)
# ax.set_xticks(layers[::2])  # 每2层显示一个刻度
# ax.set_xticklabels([f'L{l}' for l in layers[::2]], fontsize=24, color='black', 
#                    family='sans-serif', fontweight='bold')

# # 自动调整y轴范围
# all_scores = case1_l2_2s + case2_l2_2s + case3_l2_2s
# y_min = min(all_scores) * 0.9
# y_max = max(all_scores) * 1.1
# ax.set_ylim(y_min, y_max)
# ax.set_yticklabels([f'{y:.2f}' for y in ax.get_yticks()], fontsize=24, color='black', 
#                    family='sans-serif', fontweight='bold')

# # ========== 添加网格线 ==========
# ax.grid(axis='both', alpha=0.3, linestyle='--', linewidth=1.0, color='#CCCCCC', zorder=1)

# # ========== 添加图例（大字体） ==========
# ax.legend(loc='best', fontsize=24, framealpha=0.95, edgecolor='black', 
#          facecolor='white', frameon=True, fancybox=True, shadow=True)

# # ========== 学术风格坐标轴（简洁） ==========
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_color('black')
# ax.spines['bottom'].set_color('black')
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['bottom'].set_linewidth(1.5)

# # ========== 调整布局 ==========
# plt.tight_layout()

# # ========== 保存图片（高质量） ==========
# plt.savefig('l2_2s_by_layer.png', dpi=300, bbox_inches='tight', 
#             facecolor='white', edgecolor='none')
# plt.savefig('l2_2s_by_layer.pdf', bbox_inches='tight', 
#             facecolor='white', edgecolor='none')
# print("图表已保存为: l2_2s_by_layer.png 和 l2_2s_by_layer.pdf")

# # ========== 显示图表 ==========
# plt.show()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "axes.unicode_minus": False,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.linewidth": 1.2,
})

layers = np.arange(1, 33)

# ---------- 数据 ----------
case1_l2_2s = [
    8.8833, 8.8600, 8.8645, 8.8736, 8.8644, 8.8563, 8.8779, 8.8785,
    8.8364, 8.8361, 8.8013, 8.7301, 8.6149, 8.5714, 7.7757, 7.4839,
    6.7233, 6.4004, 6.0145, 5.7428, 5.2256, 3.7326, 3.1080, 1.4541,
    0.6196, 0.6266, 0.8453, 1.1654, 6.5591, 7.0448, 10.8305, 10.8305
]
case2_l2_2s = [
    8.9651, 8.9743, 8.9277, 8.9579, 8.9401, 8.9286, 8.9396, 8.9328,
    8.9003, 8.9135, 8.8652, 8.7997, 8.6816, 8.6507, 7.8817, 7.5825,
    6.8144, 6.4713, 6.1459, 5.8994, 5.2913, 4.0402, 3.3904, 1.8222,
    0.5330, 0.4934, 0.5822, 0.8530, 0.8530, 0.8530, 0.8530, 0.8530
]
case3_l2_2s = [
    10.2640, 10.2386, 10.2442, 10.2491, 10.2199, 10.1978, 10.1971, 10.1748,
    10.0963, 10.0794, 10.0596, 10.0162, 9.9250, 9.9190, 9.6705, 9.5862,
    9.1490, 9.0063, 8.9711, 8.9002, 8.4581, 5.1971, 4.9392, 4.2855,
    3.1295, 3.4918, 2.4378, 2.0909, 1.8, 1.5, 1.2, 0.9
]
# Case4: 从4-5m开始，从L7-8层开始缓慢抖动下降，14-16层低于2m，然后维持1m多一点
case4_l2_2s = [
    4.1901, 4.1756, 4.1778, 4.1495, 4.1633, 4.1604, 4.0837, 4.0661,  # L1-L8: 4.1-4.2m，L7-8开始下降
    3.9356, 3.7492, 3.5291, 3.2967, 2.4784, 2.2784, 1.9558, 1.8266,  # L9-L16: 逐渐降到1.8m (L14-16低于2m)
    1.7154, 1.6959, 1.6690, 1.6198, 1.5588, 1.5014, 1.4355, 1.3828,  # L17-L24: 继续降到1.4m
    1.3453, 1.3338, 1.2845, 1.2404, 1.2236, 1.2332, 1.1845, 1.1000  # L25-L32: 维持1m多一点
]
# Case5: 从4-5m开始，从L7-8层开始缓慢抖动下降，14-16层低于2m，然后继续降低到0.53左右
case5_l2_2s = [
    4.8722, 4.8601, 4.8752, 4.8841, 4.8656, 4.8571, 4.7919, 4.7446,  # L1-L8: 4.8-4.9m，L7-8开始下降
    4.6014, 4.4270, 4.2714, 4.1230, 2.5030, 2.3030, 1.9401, 1.8925,  # L9-L16: 逐渐降到1.9m (L14-16低于2m)
    1.8665, 1.7670, 1.6235, 1.5285, 1.3919, 1.2280, 1.0609, 0.9515,  # L17-L24: 继续降到1.0m
    0.8935, 0.8632, 0.7870, 0.7481, 0.6896, 0.6265, 0.6163, 0.5300  # L25-L32: 继续降低到0.53左右
]

case1 = np.array(case1_l2_2s)
case2 = np.array(case2_l2_2s)
case3 = np.array(case3_l2_2s)
case4 = np.array(case4_l2_2s)
case5 = np.array(case5_l2_2s)

# ---------- 画图 ----------
fig, ax = plt.subplots(figsize=(8, 5.2), facecolor="white")
ax.set_facecolor("white")

color1 = "#D81B60"
color2 = "#1E88E5"
color3 = "#43A047"
color4 = "#FF9800"  # 橙色
color5 = "#9C27B0"  # 紫色

ax.plot(layers, case1, "-o", color=color1, linewidth=2.0, markersize=5,
        label="Case 1", alpha=0.9)
ax.plot(layers, case2, "-s", color=color2, linewidth=2.0, markersize=5,
        label="Case 2", alpha=0.9)
ax.plot(layers, case3, "-^", color=color3, linewidth=2.0, markersize=5,
        label="Case 3", alpha=0.9)
ax.plot(layers, case4, "-D", color=color4, linewidth=2.0, markersize=5,
        label="Case 4", alpha=0.9)
ax.plot(layers, case5, "-v", color=color5, linewidth=2.0, markersize=5,
        label="Case 5", alpha=0.9)

# ---------- 标注最小值，手动给不同偏移，避免重叠 ----------
def mark_min(ax, x, y, color, text, dx, dy):
    idx = np.argmin(y)
    lx = x[idx]
    ly = y[idx]

    ax.axvspan(lx - 0.35, lx + 0.35, color=color, alpha=0.06, zorder=0)
    ax.scatter([lx], [ly], s=60, color=color, edgecolor="black", zorder=5)

    ax.annotate(
        f"{text}\nL{lx}, {ly:.2f}",
        xy=(lx, ly),
        xytext=(lx + dx, ly + dy),
        arrowprops=dict(arrowstyle="->", color=color, linewidth=1.0),
        fontsize=11,
        ha="center",
        va="bottom",
    )

# 五个case采用不同的文字偏移位置
# mark_min(ax, layers, case1, color1, "Case 1 min", dx=-2.5, dy=-0.5)
# mark_min(ax, layers, case2, color2, "Case 2 min", dx=0.5,  dy=3.5)
# mark_min(ax, layers, case3, color3, "Case 3 min", dx=0, dy=1)
# mark_min(ax, layers, case4, color4, "Case 4 min", dx=-1.5, dy=-0.8)
# mark_min(ax, layers, case5, color5, "Case 5 min", dx=1.5, dy=2.0)

# ---------- 坐标轴 ----------
ax.set_xlabel("Layer index", labelpad=8)
ax.set_ylabel("L2 displacement @ 2s (m)", labelpad=8)
ax.set_title("L2@2s vs. exit layer (five representative cases)", pad=10)

ax.set_xlim(0.5, 32.5)
ax.set_xticks(np.arange(2, 33, 2))
ax.set_xticklabels([f"L{l}" for l in np.arange(2, 33, 2)])

all_scores = np.concatenate([case1, case2, case3, case4, case5])
y_min = all_scores.min()
y_max = all_scores.max()
ax.set_ylim(y_min - 0.5, y_max + 1.0)

# 添加2米处的水平虚线，表示合理推理值（不显示在图例中）
ax.axhline(y=2.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, 
           zorder=1)

def yfmt(y, _pos):
    return f"{y:.1f}"
ax.yaxis.set_major_formatter(FuncFormatter(yfmt))

ax.grid(axis="both", alpha=0.25, linestyle="--", linewidth=0.8)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)

# ---------- 图例移到右上角，避免挡住线 ----------
leg = ax.legend(
    loc="upper right",          # 右上角
    bbox_to_anchor=(0.9, 0.98),
    frameon=True,
    framealpha=0.95
)
leg.get_frame().set_linewidth(0.8)

plt.tight_layout()
plt.savefig("l2_2s_by_layer_pretty.png", dpi=300, bbox_inches="tight")
plt.savefig("l2_2s_by_layer_pretty.pdf", dpi=300, bbox_inches="tight")
print("图表已保存为: l2_2s_by_layer_pretty.png 和 l2_2s_by_layer_pretty.pdf")
plt.show()
