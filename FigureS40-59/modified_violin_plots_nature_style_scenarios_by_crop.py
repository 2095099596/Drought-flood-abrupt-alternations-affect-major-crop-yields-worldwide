import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 防止 PyCharm / 无界面环境后端报错

import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from pathlib import Path
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch


# ============================================================
# 1. 数据路径配置
# ============================================================
# 说明：
# 1) 本脚本读取四种作物的 DTF / FTD 总表。
# 2) 如果某个作物实际路径不是 D:\作物名，请只修改下面对应作物的
#    *_DTF_TOTAL_FILES / *_FTD_TOTAL_FILES 中的路径。
# 3) Source 中的 ALL 作为基准，其余情景分别与 ALL 计算相对变化。

# ------------------ Maize | DTF(+) ------------------
MAIZE_DTF_TOTAL_FILES = {
    'ALL': r'D:\Maize\OVERVIEW_LDFAI+0\OVERVIEW_白名单_站点单独保存\OVERVIEW_LDFAI+0_ALL_SITES_TOTAL.csv',

    # SPEI > 0
    'SPEI 0.5': r'D:\Maize\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_0p5.csv',
    'SPEI 1': r'D:\Maize\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1.csv',
    'SPEI 1.5': r'D:\Maize\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1p5.csv',

    # LDFAI
    'DFAI 1': r'D:\Maize\OVERVIEW_LDFAI+0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_1.csv',
    'DFAI 4': r'D:\Maize\OVERVIEW_LDFAI+0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_4.csv',
    'DFAI 9': r'D:\Maize\OVERVIEW_LDFAI+0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_9.csv',

    # SPEI < 0
    'SPEI -0.5': r'D:\Maize\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg0p5.csv',
    'SPEI -1': r'D:\Maize\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1.csv',
    'SPEI -1.5': r'D:\Maize\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1p5.csv'
}


# ------------------ Maize | FTD ------------------
MAIZE_FTD_TOTAL_FILES = {
    'ALL': r'D:\Maize\OVERVIEW_LDFAI-0\OVERVIEW_白名单_站点单独保存\OVERVIEW_LDFAI-0_ALL_SITES_TOTAL.csv',

    # SPEI > 0
    'SPEI 0.5': r'D:\Maize\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_0p5.csv',
    'SPEI 1': r'D:\Maize\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1.csv',
    'SPEI 1.5': r'D:\Maize\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1p5.csv',

    # LDFAI
    'DFAI 1': r'D:\Maize\OVERVIEW_LDFAI-0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_neg1.csv',
    'DFAI 4': r'D:\Maize\OVERVIEW_LDFAI-0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_neg4.csv',
    'DFAI 9': r'D:\Maize\OVERVIEW_LDFAI-0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_neg9.csv',

    # SPEI < 0
    'SPEI -0.5': r'D:\Maize\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg0p5.csv',
    'SPEI -1': r'D:\Maize\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1.csv',
    'SPEI -1.5': r'D:\Maize\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1p5.csv'
}


# ------------------ Rice | DTF(+) ------------------
RICE_DTF_TOTAL_FILES = {
    'ALL': r'D:\Rice\OVERVIEW_LDFAI+0\OVERVIEW_白名单_站点单独保存\OVERVIEW_LDFAI+0_ALL_SITES_TOTAL.csv',

    # SPEI > 0
    'SPEI 0.5': r'D:\Rice\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_0p5.csv',
    'SPEI 1': r'D:\Rice\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1.csv',
    'SPEI 1.5': r'D:\Rice\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1p5.csv',

    # LDFAI
    'DFAI 1': r'D:\Rice\OVERVIEW_LDFAI+0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_1.csv',
    'DFAI 4': r'D:\Rice\OVERVIEW_LDFAI+0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_4.csv',
    'DFAI 9': r'D:\Rice\OVERVIEW_LDFAI+0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_9.csv',

    # SPEI < 0
    'SPEI -0.5': r'D:\Rice\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg0p5.csv',
    'SPEI -1': r'D:\Rice\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1.csv',
    'SPEI -1.5': r'D:\Rice\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1p5.csv'
}


# ------------------ Rice | FTD ------------------
RICE_FTD_TOTAL_FILES = {
    'ALL': r'D:\Rice\OVERVIEW_LDFAI-0\OVERVIEW_白名单_站点单独保存\OVERVIEW_LDFAI-0_ALL_SITES_TOTAL.csv',

    # SPEI > 0
    'SPEI 0.5': r'D:\Rice\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_0p5.csv',
    'SPEI 1': r'D:\Rice\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1.csv',
    'SPEI 1.5': r'D:\Rice\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1p5.csv',

    # LDFAI
    'DFAI 1': r'D:\Rice\OVERVIEW_LDFAI-0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_neg1.csv',
    'DFAI 4': r'D:\Rice\OVERVIEW_LDFAI-0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_neg4.csv',
    'DFAI 9': r'D:\Rice\OVERVIEW_LDFAI-0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_neg9.csv',

    # SPEI < 0
    'SPEI -0.5': r'D:\Rice\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg0p5.csv',
    'SPEI -1': r'D:\Rice\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1.csv',
    'SPEI -1.5': r'D:\Rice\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1p5.csv'
}


# ------------------ Soybeans | DTF(+) ------------------
SOYBEANS_DTF_TOTAL_FILES = {
    'ALL': r'D:\Soybeans\OVERVIEW_LDFAI+0\OVERVIEW_白名单_站点单独保存\OVERVIEW_LDFAI+0_ALL_SITES_TOTAL.csv',

    # SPEI > 0
    'SPEI 0.5': r'D:\Soybeans\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_0p5.csv',
    'SPEI 1': r'D:\Soybeans\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1.csv',
    'SPEI 1.5': r'D:\Soybeans\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1p5.csv',

    # LDFAI
    'DFAI 1': r'D:\Soybeans\OVERVIEW_LDFAI+0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_1.csv',
    'DFAI 4': r'D:\Soybeans\OVERVIEW_LDFAI+0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_4.csv',
    'DFAI 9': r'D:\Soybeans\OVERVIEW_LDFAI+0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_9.csv',

    # SPEI < 0
    'SPEI -0.5': r'D:\Soybeans\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg0p5.csv',
    'SPEI -1': r'D:\Soybeans\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1.csv',
    'SPEI -1.5': r'D:\Soybeans\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1p5.csv'
}


# ------------------ Soybeans | FTD ------------------
SOYBEANS_FTD_TOTAL_FILES = {
    'ALL': r'D:\Soybeans\OVERVIEW_LDFAI-0\OVERVIEW_白名单_站点单独保存\OVERVIEW_LDFAI-0_ALL_SITES_TOTAL.csv',

    # SPEI > 0
    'SPEI 0.5': r'D:\Soybeans\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_0p5.csv',
    'SPEI 1': r'D:\Soybeans\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1.csv',
    'SPEI 1.5': r'D:\Soybeans\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1p5.csv',

    # LDFAI
    'DFAI 1': r'D:\Soybeans\OVERVIEW_LDFAI-0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_neg1.csv',
    'DFAI 4': r'D:\Soybeans\OVERVIEW_LDFAI-0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_neg4.csv',
    'DFAI 9': r'D:\Soybeans\OVERVIEW_LDFAI-0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_neg9.csv',

    # SPEI < 0
    'SPEI -0.5': r'D:\Soybeans\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg0p5.csv',
    'SPEI -1': r'D:\Soybeans\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1.csv',
    'SPEI -1.5': r'D:\Soybeans\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1p5.csv'
}


# ------------------ Wheat | DTF(+) ------------------
WHEAT_DTF_TOTAL_FILES = {
    'ALL': r'D:\Wheat\OVERVIEW_LDFAI+0\OVERVIEW_白名单_站点单独保存\OVERVIEW_LDFAI+0_ALL_SITES_TOTAL.csv',

    # SPEI > 0
    'SPEI 0.5': r'D:\Wheat\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_0p5.csv',
    'SPEI 1': r'D:\Wheat\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1.csv',
    'SPEI 1.5': r'D:\Wheat\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1p5.csv',

    # LDFAI
    'DFAI 1': r'D:\Wheat\OVERVIEW_LDFAI+0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_1.csv',
    'DFAI 4': r'D:\Wheat\OVERVIEW_LDFAI+0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_4.csv',
    'DFAI 9': r'D:\Wheat\OVERVIEW_LDFAI+0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_9.csv',

    # SPEI < 0
    'SPEI -0.5': r'D:\Wheat\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg0p5.csv',
    'SPEI -1': r'D:\Wheat\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1.csv',
    'SPEI -1.5': r'D:\Wheat\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1p5.csv'
}


# ------------------ Wheat | FTD ------------------
WHEAT_FTD_TOTAL_FILES = {
    'ALL': r'D:\Wheat\OVERVIEW_LDFAI-0\OVERVIEW_白名单_站点单独保存\OVERVIEW_LDFAI-0_ALL_SITES_TOTAL.csv',

    # SPEI > 0
    'SPEI 0.5': r'D:\Wheat\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_0p5.csv',
    'SPEI 1': r'D:\Wheat\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1.csv',
    'SPEI 1.5': r'D:\Wheat\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1p5.csv',

    # LDFAI
    'DFAI 1': r'D:\Wheat\OVERVIEW_LDFAI-0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_neg1.csv',
    'DFAI 4': r'D:\Wheat\OVERVIEW_LDFAI-0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_neg4.csv',
    'DFAI 9': r'D:\Wheat\OVERVIEW_LDFAI-0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_neg9.csv',

    # SPEI < 0
    'SPEI -0.5': r'D:\Wheat\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg0p5.csv',
    'SPEI -1': r'D:\Wheat\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1.csv',
    'SPEI -1.5': r'D:\Wheat\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1p5.csv'
}



# 作物读取配置
CROP_TOTAL_FILES = {
    'Maize': {
        'DTF': MAIZE_DTF_TOTAL_FILES,
        'FTD': MAIZE_FTD_TOTAL_FILES
    },
    'Rice': {
        'DTF': RICE_DTF_TOTAL_FILES,
        'FTD': RICE_FTD_TOTAL_FILES
    },
    'Soybeans': {
        'DTF': SOYBEANS_DTF_TOTAL_FILES,
        'FTD': SOYBEANS_FTD_TOTAL_FILES
    },
    'Wheat': {
        'DTF': WHEAT_DTF_TOTAL_FILES,
        'FTD': WHEAT_FTD_TOTAL_FILES
    }
}

# 作物绘制顺序
CROP_ORDER = ['Maize', 'Rice', 'Soybeans', 'Wheat']

# 实际参与绘图的作物顺序。默认等于 CROP_ORDER，读取数据后会根据实际读到的作物更新。
ACTIVE_CROP_ORDER = CROP_ORDER.copy()

# 是否强制要求四种作物都必须成功读取。
REQUIRE_ALL_CROPS = True


# ============================================================
# 2. 输出与分析参数
# ============================================================

OUTPUT_DIR = r'D:\Python\DssAat\NO_3\HUITUTU'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 求均值前是否忽略 0
IGNORE_ZERO_BEFORE_MEAN = True

# 接近 0 的阈值
EPS = 1e-9

# 是否转换为百分比
# False: (scenario - ALL) / ALL
# True : ((scenario - ALL) / ALL) * 100
RELATIVE_AS_PERCENT = False

# 输出图片设置
OUTPUT_DPI = 600
SAVE_PDF = True
SAVE_PNG = True

# 小提琴图：每张图最多 4 个作物，按 2 行 × 2 列排版
CROPS_PER_FIG = 4
VIOLIN_GRID_NROWS = 2
VIOLIN_GRID_NCOLS = 2

# True  = 每个作物子图中，按 trt 分组画多个 violin
# False = 每个作物子图中，把所有 trt 合并为一个 violin
VIOLIN_X_BY_TRT = True

# 是否在 violin 上叠加散点；数据量很大时建议 False
SHOW_STRIP_POINTS = False
STRIP_POINT_SIZE = 1.2
STRIP_POINT_ALPHA = 0.25

# 每张图四个作物是否共用 y 轴范围。
# 当前按你的要求：每个小图的 y 轴单独设置。
SHARE_Y_WITHIN_FIG = False

# ============================================================
# Nature 风格与 y 轴设置
# ============================================================
# True = 使用简洁黑白 Nature / Nature Communications 类期刊风格。
USE_NATURE_STYLE = True

# y 轴范围设置方式：
# 'quantile' = 按原始样本分位数设置。当前按 5%~95% 分位数设置 y 轴范围；
# 'mean'     = 按分组均值范围设置 y 轴；
# 'raw'      = 按原始样本 min/max 设置。
Y_LIMIT_METHOD = 'quantile'

# 当 Y_LIMIT_METHOD = 'mean' 时：
# True  = y 轴基于当前图中 Crop × x轴分组(trt 或 All) 的均值范围；
# False = y 轴基于当前图全部样本的总体均值。
Y_LIMIT_MEAN_BY_GROUP = True

# y 轴是否强制包含 0 线。
# 相对变化图中 0 是 ALL 基准线，建议保留，否则 0 虚线可能被裁掉。
Y_LIMIT_INCLUDE_ZERO = True

# 均值范围两端的留白比例。数值越大，y 轴越宽。
Y_MEAN_PADDING_RATIO = 0.55

# 均值范围最小留白。RELATIVE_AS_PERCENT=False 时建议 0.02；
# 如果 RELATIVE_AS_PERCENT=True，可改成 2.0。
Y_MEAN_MIN_PADDING = 0.02

# 仅当 Y_LIMIT_METHOD = 'quantile' 时生效；
# 0.05 表示使用 5%~95% 分位数作为 y 轴范围。
Y_LIMIT_QUANTILE = 0.05

# True = y 轴使用“最大分组范围”：
#        先分别计算每个 Crop × Source 小提琴的 5% 和 95% 分位数，
#        再取所有分组中最小的 5% 与最大的 95% 作为整张图 y 轴范围。
# False = 把当前图全部数据混合后，直接计算整体 5% 和 95% 分位数。
Y_LIMIT_QUANTILE_BY_GROUP = True

# 当按分位数设置 y 轴时额外留白，避免 violin、箱线和均值点贴边。
# 0.10 表示在 5%~95% 主体范围两端各增加 10% 的留白。
Y_QUANTILE_PADDING_RATIO = 0.10

# y 轴最小留白与最小跨度，避免某些参数变化很小时坐标轴过窄。
# RELATIVE_AS_PERCENT=False 时，0.01 表示 1 个百分点的相对变化。
Y_LIMIT_MIN_PADDING = 0.01
Y_LIMIT_MIN_SPAN = 0.04

# True = 相对变化图的 y 轴以 0 为中心做对称显示。
# 这样不同方向的变化不会因为非对称坐标轴被视觉夸大或压缩。
Y_AXIS_SYMMETRIC_AROUND_ZERO = True

# True = 绘图时只用当前 y 轴范围内的数据估计小提琴密度。
# 旧版只设置 ylim，但 seaborn 仍用全量数据画 KDE，超出 y 轴的长尾会被硬裁切，
# 看起来像小提琴被上下切断。开启后可以避免这种“贴边截断”的图形。
FILTER_VIOLIN_TO_AXIS_RANGE = True


# Nature 风格图形尺寸与字体
# 18 cm 接近 Nature 双栏宽度；2×2 面板适合该宽度。
CM_TO_INCH = 1 / 2.54
FIG_WIDTH_CM = 18.0
FIG_HEIGHT_CM = 12.0

FONT_FAMILY = 'Arial'
BASE_FONT_SIZE = 7.0
TITLE_FONT_SIZE = 7.0
SUPTITLE_FONT_SIZE = 7.8
AXIS_LABEL_SIZE = 6.9
TICK_LABEL_SIZE = 6.0
ANNOTATION_FONT_SIZE = 5.7
PANEL_LETTER_SIZE = 8.0

# 小提琴图外观：Nature 风格推荐低饱和、黑白灰、线条克制。
VIOLIN_FACE_COLOR = '0.82'
VIOLIN_EDGE_COLOR = 'black'
VIOLIN_ALPHA = 0.80
VIOLIN_LINEWIDTH = 0.45
VIOLIN_WIDTH = 0.68
BOX_WIDTH = 0.14

# 三类情景颜色：更接近 Nature 期刊常见的低饱和、克制且色盲友好配色。
# SPEI > 0：柔和蓝；DFAI：柔和紫灰；SPEI < 0：柔和珊瑚红。
SOURCE_GROUP_COLORS = {
    'SPEI_POS': '#4C78A8',
    'DFAI': '#7E6E85',
    'SPEI_NEG': '#D67D6E'
}

SOURCE_GROUP_LABELS = {
    'SPEI_POS': 'SPEI > 0',
    'DFAI': 'DFAI',
    'SPEI_NEG': 'SPEI < 0'
}

SOURCE_GROUP_ORDER = ['SPEI_POS', 'DFAI', 'SPEI_NEG']

MEAN_MARKER_SIZE = 10
ZERO_LINEWIDTH = 0.40
AXIS_LINEWIDTH = 0.50
TICK_WIDTH = 0.50
TICK_LENGTH = 2.0

# 是否在每个 violin 上叠加白底黑边均值点。
SHOW_MEAN_MARKERS = False

# 是否叠加窄箱线图，显示 IQR 与中位数。
# 这比 seaborn 默认 quartile 内线更符合黑白期刊图的阅读习惯。
SHOW_INNER_BOXPLOT = False

# 手工绘制 trimmed violin，避免 seaborn KDE 在长尾数据上形成细长尖峰。
USE_MANUAL_NATURE_VIOLIN = True
VIOLIN_MAX_HALF_WIDTH = 0.30
VIOLIN_DENSITY_POINTS = 220
VIOLIN_DENSITY_MIN_RATIO = 0.025
VIOLIN_BW_ADJUST = 1.15

# 分布摘要：仅保留 IQR 粗线 + 中位数短横线，不再绘制误差线。
SHOW_IQR_INTERVAL = True
IQR_LINEWIDTH = 1.35
MEDIAN_TICK_WIDTH = 0.18
MEDIAN_TICK_LINEWIDTH = 0.90
MEAN_CI_X_OFFSET = 0.085

# ============================================================
# 离散程度柱状图设置
# ============================================================
# True = 在每个作物面板中，小提琴图下方同时绘制离散程度柱状图。
SHOW_DISPERSION_BAR = True

# 离散程度指标：
# 'std' = 标准差；'iqr' = 四分位距(Q75-Q25)；'mad' = 中位数绝对偏差；
# 'cv'  = 变异系数，std / abs(mean)。相对变化均值接近 0 时不建议使用 cv。
DISPERSION_METHOD = 'iqr'

# 小提琴图和离散程度柱状图的高度比例。
VIOLIN_BAR_HEIGHT_RATIOS = [3.2, 1.15]

# ============================================================
# DTF / FTD 合并出图设置
# ============================================================
# True = 每个参数只输出一张图，图中同时包含 DTF 和 FTD。
# False = 保持旧版逻辑，DTF 和 FTD 分别输出。
COMBINE_DTF_FTD_IN_ONE_FIG = True

# 合并图尺寸。因为合并后为 “作物 × DTF/FTD” 多面板，建议比原图更大。
COMBINED_FIG_WIDTH_CM = 24.0
COMBINED_FIG_HEIGHT_CM = 24.0

# 合并图中 Category 的列顺序。
COMBINED_CATEGORY_ORDER = ['DTF', 'FTD']

# 柱状图外观。
DISPERSION_BAR_ALPHA = 0.85
DISPERSION_BAR_WIDTH = 0.62
DISPERSION_BAR_LINEWIDTH = 0.45

# 面板内样本量标注。Nature 风格建议只保留必要信息。
SHOW_PANEL_SAMPLE_TEXT = False
PANEL_SAMPLE_TEXT_MODE = 'n_only'

# 不再绘制均值点或误差线；只显示 IQR 与中位数。
SHOW_MEAN_CI_ERRORBARS = False
SHOW_MEAN_POINT = False
ERRORBAR_MARKER_SIZE = 4.2
ERRORBAR_ELINEWIDTH = 0.80
ERRORBAR_CAPSIZE = 2.0
ERRORBAR_CAPTHICK = 0.80
MEDIAN_TICK_WIDTH = 0.18
MEDIAN_TICK_LINEWIDTH = 0.90   # 'n_only' 或 'n_sites_trt'


# ============================================================
# 3. 参数分组：每个参数是一列，并按组分隔
# ============================================================

COL_GROUPS = [
    (
        'Phenology',
        [
            'Anthesis_DAP',
            'Maturity_DAP'
        ]
    ),

    (
        'Production',
        [
            'Yield_HA',
            'Tops_Maturity',
            'Tops_Anthesis',
            'Harvest_Index'
        ]
    ),

    (
        'Water stress',
        [
            'PreAnthesis_WaterPhotoStress_Mean',
            'PreAnthesis_WaterGrowthStress_Mean',
            'GrainFilling_WaterPhotoStress',
            'GrainFilling_WaterGrowthStress'
        ]
    ),

    (
        'Water balance',
        [
            'Precip_mm',
            'ET_mm',
            'Transp_mm'
        ]
    ),

    (
        'Nitrogen',
        [
            'N_Uptake_kg_ha'
        ]
    ),

    (
        'Water productivity',
        [
            'DM_Precip_Prod',
            'Yld_Precip_Prod',
            'DM_ET_Prod',
            'Yld_ET_Prod',
            'DM_Transp_Prod',
            'Yld_Transp_Prod'
        ]
    )
]

TARGET_COLS = []
for group_name, cols in COL_GROUPS:
    TARGET_COLS.extend(cols)

PARAM_LABELS = {
    # Phenology
    'Anthesis_DAP': 'Anthesis',
    'Maturity_DAP': 'Maturity',

    # Production
    'Yield_HA': 'Yield',
    'Tops_Maturity': 'Tops maturity',
    'Tops_Anthesis': 'Tops anthesis',
    'Harvest_Index': 'Harvest index',

    # Water stress
    'PreAnthesis_WaterPhotoStress_Mean': 'Pre-anthesis water photo stress',
    'PreAnthesis_WaterGrowthStress_Mean': 'Pre-anthesis water growth stress',
    'GrainFilling_WaterPhotoStress': 'Grain filling water photo stress',
    'GrainFilling_WaterGrowthStress': 'Grain filling water growth stress',

    # Water balance
    'Precip_mm': 'Precipitation',
    'ET_mm': 'ET',
    'Transp_mm': 'Transpiration',

    # Nitrogen
    'N_Uptake_kg_ha': 'N uptake',

    # Water productivity
    'DM_Precip_Prod': 'DM precipitation productivity',
    'Yld_Precip_Prod': 'Yield precipitation productivity',
    'DM_ET_Prod': 'DM ET productivity',
    'Yld_ET_Prod': 'Yield ET productivity',
    'DM_Transp_Prod': 'DM transpiration productivity',
    'Yld_Transp_Prod': 'Yield transpiration productivity'
}


# ============================================================
# 4. 情景与类别顺序
# ============================================================

# ALL 只作为基准，不直接绘图。
# 按你的要求，最终图中每个作物子图的 x 轴固定为 6 个情景：
# DTF: SPEI > 0.5, SPEI > 1.0, DFAI > 1, DFAI > 4, SPEI < -0.5, SPEI < -1
# FTD: SPEI > 0.5, SPEI > 1.0, DFAI < -1, DFAI < -4, SPEI < -0.5, SPEI < -1
# 其中 FTD 的 DFAI 源数据键仍然使用 'DFAI 1' 和 'DFAI 4'，只是显示标签为负阈值。
SOURCE_ORDER_PLOT = [
    'SPEI 0.5',
    'SPEI 1',
    'DFAI 1',
    'DFAI 4',
    'SPEI -0.5',
    'SPEI -1'
]

SOURCE_ORDER_BY_CATEGORY = {
    'DTF': ['SPEI 0.5', 'SPEI 1', 'DFAI 1', 'DFAI 4', 'SPEI -0.5', 'SPEI -1'],
    'FTD': ['SPEI 0.5', 'SPEI 1', 'DFAI 1', 'DFAI 4', 'SPEI -0.5', 'SPEI -1']
}

CATEGORY_ORDER = ['DTF', 'FTD']

# y 轴/标题显示映射：DTF 的 DFAI 显示为正阈值；FTD 的 DFAI 对应 neg 文件，显示为负阈值。
SOURCE_LABELS_BY_CATEGORY = {
    'DTF': {
        'SPEI 0.5': 'SPEI > 0.5',
        'SPEI 1': 'SPEI > 1.0',
        'SPEI 1.5': 'SPEI > 1.5',
        'DFAI 1': 'DFAI > 1',
        'DFAI 4': 'DFAI > 4',
        'DFAI 9': 'DFAI > 9',
        'SPEI -0.5': 'SPEI < -0.5',
        'SPEI -1': 'SPEI < -1',
        'SPEI -1.5': 'SPEI < -1.5'
    },
    'FTD': {
        'SPEI 0.5': 'SPEI > 0.5',
        'SPEI 1': 'SPEI > 1.0',
        'SPEI 1.5': 'SPEI > 1.5',
        'DFAI 1': 'DFAI < -1',
        'DFAI 4': 'DFAI < -4',
        'DFAI 9': 'DFAI < -9',
        'SPEI -0.5': 'SPEI < -0.5',
        'SPEI -1': 'SPEI < -1',
        'SPEI -1.5': 'SPEI < -1.5'
    }
}


def get_source_label(category, source):
    return SOURCE_LABELS_BY_CATEGORY.get(category, {}).get(source, source)


def get_source_group(source):
    """
    把每个 x 轴情景归入三类，用于 violin 填充颜色和图例。
    """
    source = str(source)

    if source.startswith('DFAI'):
        return 'DFAI'

    if source.startswith('SPEI -'):
        return 'SPEI_NEG'

    if source.startswith('SPEI'):
        return 'SPEI_POS'

    return 'DFAI'


def get_source_color(source):
    return SOURCE_GROUP_COLORS.get(get_source_group(source), VIOLIN_FACE_COLOR)


def get_source_palette(x_order):
    return {source: get_source_color(source) for source in x_order}


def add_source_group_legend(fig):
    """
    在整张 2×2 图上方添加三类情景颜色说明。
    """
    handles = [
        Patch(
            facecolor=SOURCE_GROUP_COLORS[group_key],
            edgecolor='black',
            linewidth=0.45,
            label=SOURCE_GROUP_LABELS[group_key]
        )
        for group_key in SOURCE_GROUP_ORDER
    ]

    fig.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(handles),
        frameon=False,
        fontsize=6.2,
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.9
    )


# ============================================================
# 5. 工具函数
# ============================================================

def natural_sort_key(x):
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r'(\d+)', str(x))
    ]


def sanitize_filename(text):
    text = str(text)
    text = text.replace('>', 'gt').replace('<', 'lt')
    text = re.sub(r'[^0-9A-Za-z\u4e00-\u9fff_+-]+', '_', text)
    text = re.sub(r'_+', '_', text).strip('_')
    return text or 'NA'


def chunk_list(values, chunk_size):
    for i in range(0, len(values), chunk_size):
        yield values[i:i + chunk_size], i // chunk_size + 1


def read_total_files(file_dict, category_label, crop_label):
    """
    读取一组总表，并添加 Source、Category 和 Crop。
    """
    df_list = []

    for source_label, file_path in file_dict.items():

        if not os.path.exists(file_path):
            print(f'⚠️ 文件不存在，跳过：{crop_label} | {category_label} | {source_label} | {file_path}')
            continue

        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
        except Exception as e:
            print(f'❌ 文件读取失败：{crop_label} | {category_label} | {source_label}')
            print(f'   路径：{file_path}')
            print(f'   错误：{e}')
            continue

        df['Crop'] = crop_label
        df['Source'] = source_label
        df['Category'] = category_label

        df_list.append(df)

        print(f'✅ 已读取：{crop_label} | {category_label} | {source_label} | 行数={len(df)}')

    if len(df_list) == 0:
        return pd.DataFrame()

    return pd.concat(df_list, ignore_index=True)


def prepare_relative_for_one_target(df_all, target_col):
    """
    对单个参数计算相对 ALL 的变化：
    1. 求均值前忽略 0；
    2. Crop + Category + Source + soil_id + trt 求均值；
    3. 对每个 Crop + Category + soil_id + trt 提取 ALL 作为基准；
    4. 计算 (scenario - ALL) / ALL。
    """
    required_cols = ['Crop', 'Category', 'Source', 'soil_id', 'trt', target_col]

    for col in required_cols:
        if col not in df_all.columns:
            print(f'⚠️ 缺少列 {col}，跳过参数：{target_col}')
            return pd.DataFrame()

    df = df_all[required_cols].copy()
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

    raw_n = len(df)
    df = df.dropna(subset=[target_col]).copy()

    if IGNORE_ZERO_BEFORE_MEAN:
        before_zero = len(df)
        df = df[df[target_col].abs() > EPS].copy()
        removed_zero = before_zero - len(df)
    else:
        removed_zero = 0

    if df.empty:
        print(f'⚠️ {target_col} 去掉 NaN/0 后为空，跳过')
        return pd.DataFrame()

    group_cols = ['Crop', 'Category', 'Source', 'soil_id', 'trt']

    df_mean = (
        df
        .groupby(group_cols, as_index=False)
        .agg(
            value_mean=(target_col, 'mean'),
            n_raw_records=(target_col, 'size')
        )
    )

    df_base = df_mean[df_mean['Source'] == 'ALL'].copy()
    df_base = df_base.rename(columns={'value_mean': 'ALL_mean'})
    df_base = df_base[['Crop', 'Category', 'soil_id', 'trt', 'ALL_mean']].copy()

    before_base = len(df_base)
    df_base = df_base[df_base['ALL_mean'].abs() > EPS].copy()
    removed_base_zero = before_base - len(df_base)

    df_scenario = df_mean[df_mean['Source'].isin(SOURCE_ORDER_PLOT)].copy()

    df_rel = df_scenario.merge(
        df_base,
        on=['Crop', 'Category', 'soil_id', 'trt'],
        how='inner'
    )

    if df_rel.empty:
        print(f'⚠️ {target_col} 与 ALL 基准合并后为空，跳过')
        return pd.DataFrame()

    df_rel['relative_value'] = (
        df_rel['value_mean'] - df_rel['ALL_mean']
    ) / df_rel['ALL_mean']

    if RELATIVE_AS_PERCENT:
        df_rel['relative_value'] = df_rel['relative_value'] * 100

    before_clean = len(df_rel)
    df_rel['relative_value'] = pd.to_numeric(df_rel['relative_value'], errors='coerce')
    df_rel = df_rel.replace([np.inf, -np.inf], np.nan)
    df_rel = df_rel.dropna(subset=['relative_value']).copy()
    removed_invalid = before_clean - len(df_rel)

    if df_rel.empty:
        print(f'⚠️ {target_col} 相对变化计算后为空，跳过')
        return pd.DataFrame()

    df_rel['Target'] = target_col
    df_rel['Target_Label'] = PARAM_LABELS.get(target_col, target_col)

    print(
        f'✅ {target_col}: 原始={raw_n}, 去0={removed_zero}, '
        f'ALL基准去0={removed_base_zero}, 无效相对值={removed_invalid}, '
        f'最终={len(df_rel)}'
    )

    return df_rel


def build_violin_summary(df_rel_all):
    """
    输出每个 Target × Crop × Category × Source × trt 的描述统计，便于检查 violin 图数据。
    """
    group_cols = ['Target', 'Target_Label', 'Crop', 'Category', 'Source', 'trt']

    summary = (
        df_rel_all
        .groupby(group_cols)['relative_value']
        .agg(['count', 'mean', 'median', 'std', 'min', 'max'])
        .reset_index()
    )

    summary['std'] = summary['std'].fillna(0)
    summary['sem'] = summary['std'] / np.sqrt(summary['count'])
    summary['sem'] = summary['sem'].fillna(0)
    summary['ci95'] = 1.96 * summary['sem']
    summary['ci95'] = summary['ci95'].fillna(0)

    return summary


def get_y_limits(data, value_col='relative_value', mean_group_cols=None):
    """
    设置 y 轴范围。

    当前默认 Y_LIMIT_METHOD = 'quantile' 且 Y_LIMIT_QUANTILE_BY_GROUP = True：
        先分别计算每个分组，例如 Source，也就是当前作物小图中每个小提琴的 5% 和 95% 分位数；
        再用所有分组中最小的 5% 与最大的 95% 作为该小图 y 轴主体范围。

    合理化处理：
        1) 不使用样本最大值/最小值，避免极端离群值压扁主体分布；
        2) 保留 0 基准线；
        3) 增加少量留白，避免 violin、箱线和均值点贴边；
        4) 设置最小 y 轴跨度，避免变化很小时坐标轴过窄。
    """
    if isinstance(data, pd.DataFrame):
        if value_col not in data.columns:
            return None

        df = data.copy()
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[value_col]).copy()

        if df.empty:
            return None

        raw_values = df[value_col]

        if (
            Y_LIMIT_METHOD == 'mean'
            and Y_LIMIT_MEAN_BY_GROUP
            and mean_group_cols is not None
        ):
            valid_group_cols = [
                col for col in mean_group_cols
                if col in df.columns
            ]

            if valid_group_cols:
                ref_values = (
                    df
                    .groupby(valid_group_cols, dropna=False)[value_col]
                    .mean()
                    .dropna()
                )
            else:
                ref_values = pd.Series([raw_values.mean()])
        elif Y_LIMIT_METHOD == 'mean':
            ref_values = pd.Series([raw_values.mean()])
        elif Y_LIMIT_METHOD == 'quantile':
            q = Y_LIMIT_QUANTILE
            if q is not None and 0 < q < 0.5:
                valid_group_cols = []
                if Y_LIMIT_QUANTILE_BY_GROUP and mean_group_cols is not None:
                    valid_group_cols = [
                        col for col in mean_group_cols
                        if col in df.columns
                    ]

                if valid_group_cols:
                    # “最大分组范围”：每个小提琴单独算 5% 和 95%，
                    # 再取所有分组的最小 5% 与最大 95%。
                    grouped = df.groupby(valid_group_cols, dropna=False)[value_col]
                    q_low = grouped.quantile(q).dropna()
                    q_high = grouped.quantile(1 - q).dropna()
                    ref_values = pd.concat([q_low, q_high], ignore_index=True)
                else:
                    # 备用：整体混合分位数。
                    ref_values = pd.Series([
                        raw_values.quantile(q),
                        raw_values.quantile(1 - q)
                    ])
            else:
                ref_values = raw_values
        else:
            ref_values = raw_values

    else:
        raw_values = (
            pd.to_numeric(pd.Series(data), errors='coerce')
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if raw_values.empty:
            return None

        if Y_LIMIT_METHOD == 'mean':
            ref_values = pd.Series([raw_values.mean()])
        elif Y_LIMIT_METHOD == 'quantile':
            q = Y_LIMIT_QUANTILE
            if q is not None and 0 < q < 0.5:
                ref_values = pd.Series([
                    raw_values.quantile(q),
                    raw_values.quantile(1 - q)
                ])
            else:
                ref_values = raw_values
        else:
            ref_values = raw_values

    ref_values = (
        pd.to_numeric(ref_values, errors='coerce')
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if ref_values.empty:
        return None

    low = float(ref_values.min())
    high = float(ref_values.max())

    if Y_LIMIT_INCLUDE_ZERO:
        low = min(low, 0.0)
        high = max(high, 0.0)

    if pd.isna(low) or pd.isna(high):
        return None

    # 先处理极小跨度，避免某些参数几乎不变时 y 轴过窄。
    span = high - low

    if abs(span) < EPS:
        center = (high + low) / 2.0
        half_span = max(
            Y_LIMIT_MIN_SPAN / 2.0,
            abs(center) * 0.10,
            Y_LIMIT_MIN_PADDING
        )
        low = center - half_span
        high = center + half_span
        span = high - low

    if span < Y_LIMIT_MIN_SPAN:
        center = (high + low) / 2.0
        half_span = Y_LIMIT_MIN_SPAN / 2.0
        low = center - half_span
        high = center + half_span
        span = high - low

    # 相对变化图建议围绕 0 对称设置 y 轴，避免正负变化的视觉尺度不一致。
    if Y_AXIS_SYMMETRIC_AROUND_ZERO:
        half_span = max(
            abs(low),
            abs(high),
            Y_LIMIT_MIN_SPAN / 2.0
        )
        low = -half_span
        high = half_span
        span = high - low

    if Y_LIMIT_METHOD == 'mean':
        pad = max(
            span * Y_MEAN_PADDING_RATIO,
            max(abs(low), abs(high)) * 0.08,
            Y_MEAN_MIN_PADDING
        )
    elif Y_LIMIT_METHOD == 'quantile':
        pad = max(
            span * Y_QUANTILE_PADDING_RATIO,
            Y_LIMIT_MIN_PADDING
        )
    else:
        pad = max(
            span * 0.08,
            Y_LIMIT_MIN_PADDING
        )

    return float(low - pad), float(high + pad)



def filter_data_to_y_limits(data, y_limits, value_col='relative_value'):
    """
    用于绘图的稳健裁剪。

    注意：这只影响小提琴 KDE 的可视化范围，避免长尾被 ylim 硬切；
    原始相对变化长表和汇总表仍然保留全量数据。
    """
    if not FILTER_VIOLIN_TO_AXIS_RANGE or y_limits is None:
        return data

    if value_col not in data.columns:
        return data

    low, high = y_limits
    plot_df = data.copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors='coerce')
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=[value_col]).copy()
    plot_df = plot_df[
        (plot_df[value_col] >= low) &
        (plot_df[value_col] <= high)
    ].copy()

    # 如果过滤后极端情况下为空，则回退到原始数据，避免报错。
    if plot_df.empty:
        return data

    return plot_df

def apply_nature_axis_style(ax):
    """
    Nature 风格坐标轴：白底、无网格、只保留左/下轴线、细刻度。
    """
    ax.set_facecolor('white')
    ax.grid(False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for side in ['left', 'bottom']:
        ax.spines[side].set_linewidth(AXIS_LINEWIDTH)
        ax.spines[side].set_color('black')

    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        length=TICK_LENGTH,
        width=TICK_WIDTH,
        colors='black',
        pad=2,
        labelsize=TICK_LABEL_SIZE
    )

    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
    ax.margins(x=0.08)


def style_violin_artists(ax):
    """
    将 seaborn violin 调整为更接近 Nature 的期刊图风格：保留低饱和填充色，统一黑色轮廓。
    """
    for collection in ax.collections:
        if isinstance(collection, PolyCollection):
            # 不覆盖 facecolor，保留 SPEI > 0 / DFAI / SPEI < 0 的分组颜色。
            collection.set_edgecolor(VIOLIN_EDGE_COLOR)
            collection.set_linewidth(VIOLIN_LINEWIDTH)
            collection.set_alpha(VIOLIN_ALPHA)

    # seaborn 可能生成默认内部线条；这里统一成黑色细线。
    for line in ax.lines:
        line.set_color('black')
        line.set_linewidth(0.50)

def _draw_manual_violin_for_group(ax, values, x_index, face_color, y_limits=None):
    """
    手工绘制 trimmed violin。
    目的：避免长尾数据导致 seaborn violin 出现很长、很细的针状尾部。
    """
    values = (
        pd.to_numeric(pd.Series(values), errors='coerce')
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if values.empty:
        return

    if y_limits is not None:
        y_low, y_high = y_limits
        values_for_kde = values[(values >= y_low) & (values <= y_high)].copy()
    else:
        y_low = float(values.quantile(0.02))
        y_high = float(values.quantile(0.98))
        values_for_kde = values[(values >= y_low) & (values <= y_high)].copy()

    if values_for_kde.empty:
        values_for_kde = values.copy()

    if values_for_kde.nunique() < 2 or len(values_for_kde) < 5:
        y0 = float(values_for_kde.mean())
        ax.vlines(
            x_index,
            y0 - 0.002,
            y0 + 0.002,
            color='black',
            linewidth=VIOLIN_LINEWIDTH,
            zorder=2
        )
        return

    try:
        from scipy.stats import gaussian_kde

        grid = np.linspace(y_low, y_high, VIOLIN_DENSITY_POINTS)
        kde = gaussian_kde(values_for_kde.to_numpy(dtype=float))
        kde.set_bandwidth(kde.factor * VIOLIN_BW_ADJUST)
        density = kde(grid)

        if not np.isfinite(density).any() or np.nanmax(density) <= 0:
            return

        density = density / np.nanmax(density)
        keep = density >= VIOLIN_DENSITY_MIN_RATIO

        if not keep.any():
            keep = density > 0

        idx = np.where(keep)[0]
        if len(idx) == 0:
            return

        # 分段绘制，避免阈值裁剪后把相隔很远的密度块连起来。
        breaks = np.where(np.diff(idx) > 1)[0] + 1
        chunks = np.split(idx, breaks)

        for chunk in chunks:
            if len(chunk) < 3:
                continue

            gy = grid[chunk]
            half_width = density[chunk] * VIOLIN_MAX_HALF_WIDTH

            # 两端强制收口，避免产生平顶/平底。
            gy = np.r_[gy[0], gy, gy[-1]]
            half_width = np.r_[0.0, half_width, 0.0]

            ax.fill_betweenx(
                gy,
                x_index - half_width,
                x_index + half_width,
                facecolor=face_color,
                edgecolor=VIOLIN_EDGE_COLOR,
                linewidth=VIOLIN_LINEWIDTH,
                alpha=VIOLIN_ALPHA,
                zorder=2
            )

    except Exception:
        # 如果 scipy 不可用，则退回到一个极简箱体轮廓，保证脚本不报错。
        q25 = float(values_for_kde.quantile(0.25))
        q50 = float(values_for_kde.quantile(0.50))
        q75 = float(values_for_kde.quantile(0.75))
        ax.vlines(x_index, q25, q75, color='black', linewidth=1.0, zorder=3)
        ax.hlines(q50, x_index - 0.08, x_index + 0.08, color='black', linewidth=0.9, zorder=4)


def draw_nature_violin(ax, data, x_col, x_order, y_limits=None):
    """
    绘制更接近 Nature 的小提琴图：
    - 手工 trimmed KDE，去除长尾针状伪影；
    - 低饱和配色；
    - 黑色细轮廓；
    - 不使用 seaborn 默认内部线条。
    """
    if USE_MANUAL_NATURE_VIOLIN:
        for x_index, source in enumerate(x_order):
            group_values = data.loc[data[x_col] == source, 'relative_value']
            _draw_manual_violin_for_group(
                ax=ax,
                values=group_values,
                x_index=x_index,
                face_color=get_source_color(source),
                y_limits=y_limits
            )
        return

    source_palette = get_source_palette(x_order)

    violin_kwargs = dict(
        data=data,
        x=x_col,
        y='relative_value',
        order=x_order,
        inner=None,
        cut=0,
        linewidth=VIOLIN_LINEWIDTH,
        palette=source_palette,
        hue=x_col,
        dodge=False,
        saturation=1,
        width=VIOLIN_WIDTH,
        ax=ax
    )

    try:
        sns.violinplot(**violin_kwargs, density_norm='width', bw_adjust=0.8, legend=False)
    except (TypeError, AttributeError):
        sns.violinplot(**violin_kwargs, scale='width', bw=0.25)

    style_violin_artists(ax)


def add_summary_overlays(ax, data, x_col, x_order):
    """
    叠加 Nature 风格的统计摘要：
    - IQR 粗线：25%~75%
    - 中位数短横线
    - 不绘制误差线，保持图面简洁
    """
    if data.empty:
        return

    for x_index, x_value in enumerate(x_order):
        g = data[data[x_col] == x_value]['relative_value']
        g = pd.to_numeric(g, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if g.empty:
            continue

        q25 = float(g.quantile(0.25))
        median_value = float(g.median())
        q75 = float(g.quantile(0.75))

        if SHOW_IQR_INTERVAL:
            ax.vlines(
                x_index,
                q25,
                q75,
                color='black',
                linewidth=IQR_LINEWIDTH,
                zorder=6
            )

            ax.hlines(
                median_value,
                x_index - MEDIAN_TICK_WIDTH / 2,
                x_index + MEDIAN_TICK_WIDTH / 2,
                colors='black',
                linewidth=MEDIAN_TICK_LINEWIDTH,
                zorder=7
            )



def format_panel_sample_text(crop_df):
    """
    Nature 风格面板标注：默认只显示 n，避免面板内信息过密。
    """
    n_obs = len(crop_df)
    n_soil = crop_df['soil_id'].nunique(dropna=True) if 'soil_id' in crop_df.columns else np.nan
    n_trt = crop_df['trt'].nunique(dropna=True) if 'trt' in crop_df.columns else np.nan

    if PANEL_SAMPLE_TEXT_MODE == 'n_sites_trt':
        return f'n={n_obs}\nsites={n_soil}\ntrt={n_trt}'

    return f'n={n_obs}'




def _calc_dispersion(values, method=None):
    """
    计算一个分组的离散程度。
    默认 DISPERSION_METHOD='std'，即标准差。
    """
    if method is None:
        method = DISPERSION_METHOD

    values = (
        pd.to_numeric(pd.Series(values), errors='coerce')
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if values.empty:
        return np.nan

    method = str(method).lower()

    if method == 'std':
        # ddof=1 与 pandas 默认 std 一致；只有 1 个样本时返回 0，避免柱状图缺失。
        if len(values) < 2:
            return 0.0
        return float(values.std(ddof=1))

    if method == 'iqr':
        return float(values.quantile(0.75) - values.quantile(0.25))

    if method == 'mad':
        med = float(values.median())
        return float(np.median(np.abs(values - med)))

    if method == 'cv':
        mean_value = float(values.mean())
        if abs(mean_value) <= EPS:
            return np.nan
        std_value = 0.0 if len(values) < 2 else float(values.std(ddof=1))
        return float(std_value / abs(mean_value))

    raise ValueError(
        "DISPERSION_METHOD 只能是 'std'、'iqr'、'mad' 或 'cv'。"
    )


def get_dispersion_label():
    """
    返回柱状图 y 轴标签。
    """
    method = str(DISPERSION_METHOD).lower()

    label_map = {
        'std': 'Dispersion',
        'iqr': 'Dispersion',
        'mad': 'Dispersion',
        'cv': 'Dispersion'
    }

    base_label = label_map.get(method, f'Dispersion ({method})')

    if RELATIVE_AS_PERCENT and method != 'cv':
        return base_label + ', percentage points'

    return base_label


def build_dispersion_summary(df_rel_all):
    """
    输出每个 Target × Crop × Category × Source 的离散程度统计。
    这里不再按 trt 分组，而是与当前小提琴图的 x 轴 Source 保持一致。
    """
    group_cols = ['Target', 'Target_Label', 'Crop', 'Category', 'Source']

    summary = (
        df_rel_all
        .groupby(group_cols)['relative_value']
        .agg(
            count='count',
            mean='mean',
            median='median',
            std='std',
            min='min',
            max='max',
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75)
        )
        .reset_index()
    )

    summary['std'] = summary['std'].fillna(0)
    summary['iqr'] = summary['q75'] - summary['q25']
    summary['cv'] = np.where(
        summary['mean'].abs() > EPS,
        summary['std'] / summary['mean'].abs(),
        np.nan
    )

    mad_rows = []
    for keys, group_df in df_rel_all.groupby(group_cols, dropna=False):
        mad_rows.append((*keys, _calc_dispersion(group_df['relative_value'], method='mad')))

    mad_df = pd.DataFrame(mad_rows, columns=group_cols + ['mad'])
    summary = summary.merge(mad_df, on=group_cols, how='left')

    summary['dispersion_method'] = DISPERSION_METHOD
    summary['dispersion_value'] = summary.apply(
        lambda row: row.get(str(DISPERSION_METHOD).lower(), np.nan),
        axis=1
    )

    return summary


def get_dispersion_for_plot(data, x_col, x_order):
    """
    按当前作物面板中的 x 轴分组计算离散程度，供柱状图使用。
    """
    rows = []

    for x_value in x_order:
        values = data.loc[data[x_col] == x_value, 'relative_value']
        rows.append({
            x_col: x_value,
            'dispersion_value': _calc_dispersion(values),
            'n': pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna().size
        })

    return pd.DataFrame(rows)


def draw_dispersion_bar(ax, data, x_col, x_order, category):
    """
    在小提琴图下方绘制离散程度柱状图。
    """
    dispersion_df = get_dispersion_for_plot(data, x_col, x_order)
    x_positions = np.arange(len(x_order))
    y_values = pd.to_numeric(dispersion_df['dispersion_value'], errors='coerce').to_numpy(dtype=float)

    colors = [get_source_color(source) for source in x_order]

    ax.bar(
        x_positions,
        np.nan_to_num(y_values, nan=0.0),
        width=DISPERSION_BAR_WIDTH,
        color=colors,
        edgecolor='black',
        linewidth=DISPERSION_BAR_LINEWIDTH,
        alpha=DISPERSION_BAR_ALPHA,
        zorder=2
    )

    # 如果某个分组没有有效离散程度，例如 cv 分母接近 0，则用 n.a. 标注。
    for x_pos, y_val in zip(x_positions, y_values):
        if not np.isfinite(y_val):
            ax.text(
                x_pos,
                0,
                'n.a.',
                ha='center',
                va='bottom',
                fontsize=ANNOTATION_FONT_SIZE,
                rotation=90
            )

    ymax = np.nanmax(y_values) if np.isfinite(y_values).any() else 1.0
    if ymax <= EPS:
        ymax = 1.0
    ax.set_ylim(0, ymax * 1.18)

    display_labels = [get_source_label(category, s) for s in x_order]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(display_labels)

    for label in ax.get_xticklabels():
        label.set_rotation(35)
        label.set_ha('right')
        label.set_color('black')

    ax.set_ylabel(get_dispersion_label(), fontsize=AXIS_LABEL_SIZE)

    apply_nature_axis_style(ax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))

def draw_violin_grid(df_rel_all, target_col, category, crop_batch, batch_index):
    """
    对一个参数 × 一个 Category(DTF/FTD) 绘制一张 2×2 作物图。

    当 SHOW_DISPERSION_BAR=True 时，每个作物面板由两部分组成：
        上：相对变化分布的小提琴图；
        下：相同 Source 顺序下的离散程度柱状图，默认使用标准差 std。
    """
    x_order = SOURCE_ORDER_BY_CATEGORY.get(category, SOURCE_ORDER_PLOT)

    fig_df = df_rel_all[
        (df_rel_all['Target'] == target_col) &
        (df_rel_all['Category'] == category) &
        (df_rel_all['Crop'].isin(crop_batch)) &
        (df_rel_all['Source'].isin(x_order))
    ].copy()

    if fig_df.empty:
        print(f'⚠️ 无数据，跳过：{category} | {target_col} | crops={crop_batch}')
        return None

    target_label = PARAM_LABELS.get(target_col, target_col)
    x_col = 'Source'
    x_label = ''

    if SHARE_Y_WITHIN_FIG:
        mean_group_cols = ['Crop', x_col] if Y_LIMIT_MEAN_BY_GROUP else None
        y_limits = get_y_limits(
            fig_df,
            value_col='relative_value',
            mean_group_cols=mean_group_cols
        )
    else:
        y_limits = None

    # 使用 GridSpec 是为了在每个作物面板内部再拆成 “小提琴图 + 柱状图” 两行。
    fig = plt.figure(
        figsize=(FIG_WIDTH_CM * CM_TO_INCH, FIG_HEIGHT_CM * CM_TO_INCH),
        facecolor='white'
    )

    outer_grid = fig.add_gridspec(
        VIOLIN_GRID_NROWS,
        VIOLIN_GRID_NCOLS,
        wspace=0.34,
        hspace=0.42
    )

    panel_letters = list('abcdefghijklmnopqrstuvwxyz')
    first_violin_ax = None

    for ax_index in range(VIOLIN_GRID_NROWS * VIOLIN_GRID_NCOLS):
        outer_cell = outer_grid[ax_index]

        if ax_index >= len(crop_batch):
            blank_ax = fig.add_subplot(outer_cell)
            blank_ax.axis('off')
            continue

        if SHOW_DISPERSION_BAR:
            inner_grid = outer_cell.subgridspec(
                2,
                1,
                height_ratios=VIOLIN_BAR_HEIGHT_RATIOS,
                hspace=0.05
            )

            if SHARE_Y_WITHIN_FIG and first_violin_ax is not None:
                ax = fig.add_subplot(inner_grid[0], sharey=first_violin_ax)
            else:
                ax = fig.add_subplot(inner_grid[0])

            bar_ax = fig.add_subplot(inner_grid[1], sharex=ax)
        else:
            if SHARE_Y_WITHIN_FIG and first_violin_ax is not None:
                ax = fig.add_subplot(outer_cell, sharey=first_violin_ax)
            else:
                ax = fig.add_subplot(outer_cell)
            bar_ax = None

        if first_violin_ax is None:
            first_violin_ax = ax

        crop_name = crop_batch[ax_index]
        crop_df = fig_df[fig_df['Crop'] == crop_name].copy()

        if not SHARE_Y_WITHIN_FIG:
            mean_group_cols = [x_col] if Y_LIMIT_MEAN_BY_GROUP else None
            y_limits_local = get_y_limits(
                crop_df,
                value_col='relative_value',
                mean_group_cols=mean_group_cols
            )
        else:
            y_limits_local = y_limits

        if crop_df.empty:
            ax.text(
                0.5,
                0.5,
                'No data',
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=ANNOTATION_FONT_SIZE
            )
            ax.set_title(crop_name, fontsize=TITLE_FONT_SIZE, fontweight='normal')
            ax.set_xlabel(x_label, fontsize=AXIS_LABEL_SIZE)
            ax.set_ylabel('')
            apply_nature_axis_style(ax)

            if bar_ax is not None:
                bar_ax.axis('off')
            continue

        # 用 y 轴范围内的数据估计小提琴密度，避免长尾被坐标轴硬裁切后出现贴边尖峰。
        crop_plot_df = filter_data_to_y_limits(
            crop_df,
            y_limits_local,
            value_col='relative_value'
        )

        draw_nature_violin(ax, crop_plot_df, x_col, x_order, y_limits=y_limits_local)
        add_summary_overlays(ax, crop_df, x_col, x_order)

        if SHOW_STRIP_POINTS:
            sns.stripplot(
                data=crop_plot_df,
                x=x_col,
                y='relative_value',
                order=x_order,
                size=STRIP_POINT_SIZE,
                alpha=STRIP_POINT_ALPHA,
                jitter=0.20,
                color='black',
                linewidth=0,
                ax=ax,
                zorder=3
            )

        ax.axhline(
            0,
            linewidth=ZERO_LINEWIDTH,
            linestyle=(0, (2.5, 2.5)),
            color='0.35',
            zorder=0
        )

        panel_letter = panel_letters[ax_index]
        ax.text(
            -0.13,
            1.08,
            panel_letter,
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=PANEL_LETTER_SIZE,
            fontweight='bold'
        )

        ax.set_title(
            crop_name,
            fontsize=TITLE_FONT_SIZE,
            fontweight='normal',
            pad=3
        )

        if SHOW_PANEL_SAMPLE_TEXT:
            ax.text(
                0.98,
                0.96,
                format_panel_sample_text(crop_df),
                transform=ax.transAxes,
                ha='right',
                va='top',
                fontsize=ANNOTATION_FONT_SIZE,
                linespacing=1.12
            )

        if ax_index % VIOLIN_GRID_NCOLS == 0:
            ax.set_ylabel(
                'Relative change to ALL (%)' if RELATIVE_AS_PERCENT else 'Relative change to ALL',
                fontsize=AXIS_LABEL_SIZE
            )
        else:
            ax.set_ylabel('')

        display_labels = [get_source_label(category, s) for s in x_order]
        ax.set_xticks(range(len(x_order)))
        ax.set_xticklabels(display_labels)

        if SHOW_DISPERSION_BAR:
            # 上方小提琴图不显示 x 轴标签，避免和下方柱状图重复。
            ax.tick_params(axis='x', labelbottom=False)
            ax.set_xlabel('')
        else:
            ax.set_xlabel(x_label if ax_index >= VIOLIN_GRID_NCOLS else '', fontsize=AXIS_LABEL_SIZE)
            for label in ax.get_xticklabels():
                label.set_rotation(35)
                label.set_ha('right')
                label.set_color('black')

        ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)

        if y_limits_local is not None:
            ax.set_ylim(*y_limits_local)

        apply_nature_axis_style(ax)

        if bar_ax is not None:
            draw_dispersion_bar(
                ax=bar_ax,
                data=crop_df,
                x_col=x_col,
                x_order=x_order,
                category=category
            )

            # 右列不显示柱状图 y 轴标签，减少拥挤。
            if ax_index % VIOLIN_GRID_NCOLS != 0:
                bar_ax.set_ylabel('')

    fig.suptitle(
        f'{target_label} | {category}',
        fontsize=SUPTITLE_FONT_SIZE,
        fontweight='normal',
        y=0.988
    )

    add_source_group_legend(fig)

    fig.tight_layout(rect=[0, 0, 1, 0.928])

    if SHOW_DISPERSION_BAR:
        out_dir_name = 'violin_with_dispersion_bar_by_category_parameter'
        file_prefix = 'violin_dispersion'
    else:
        out_dir_name = 'violin_nature_trimmed_no_errorbars_by_category_parameter'
        file_prefix = 'violin_nature'

    out_dir = Path(OUTPUT_DIR) / out_dir_name / category / sanitize_filename(target_col)
    out_dir.mkdir(parents=True, exist_ok=True)

    crop_tag = 'all4crops' if len(crop_batch) == 4 else f'crop_batch_{batch_index}'
    out_base = out_dir / (
        f'{file_prefix}_{sanitize_filename(category)}_{sanitize_filename(target_col)}_{crop_tag}'
    )

    if SAVE_PNG:
        png_path = out_base.with_suffix('.png')
        fig.savefig(png_path, dpi=OUTPUT_DPI, bbox_inches='tight')
        print(f'✅ 已保存 PNG：{png_path}')

    if SAVE_PDF:
        pdf_path = out_base.with_suffix('.pdf')
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f'✅ 已保存 PDF：{pdf_path}')

    plt.close(fig)
    return out_base


def load_all_crop_data():
    global ACTIVE_CROP_ORDER

    df_crop_category_list = []

    for crop_name in ACTIVE_CROP_ORDER:

        if crop_name not in CROP_TOTAL_FILES:
            print(f'⚠️ CROP_TOTAL_FILES 中没有 {crop_name}，跳过')
            continue

        for category_name in CATEGORY_ORDER:

            if category_name not in CROP_TOTAL_FILES[crop_name]:
                print(f'⚠️ {crop_name} 中没有 {category_name} 路径配置，跳过')
                continue

            df_one = read_total_files(
                file_dict=CROP_TOTAL_FILES[crop_name][category_name],
                category_label=category_name,
                crop_label=crop_name
            )

            if not df_one.empty:
                df_crop_category_list.append(df_one)

    if len(df_crop_category_list) == 0:
        raise ValueError('❌ 没有读取到任何数据，请检查 Maize / Rice / Soybeans / Wheat 文件路径。')

    df_all = pd.concat(df_crop_category_list, ignore_index=True)

    if df_all.empty:
        raise ValueError('❌ 没有读取到任何数据，请检查文件路径。')

    for required_col in ['Crop', 'Category', 'Source', 'trt', 'soil_id']:
        if required_col not in df_all.columns:
            raise ValueError(f'❌ 数据中没有 {required_col} 列。')

    print('\n==============================')
    print('✅ 数据读取完成')
    print('读取到的作物：')
    print(df_all['Crop'].value_counts())
    print('读取到的 Category：')
    print(df_all['Category'].value_counts())
    print('读取到的 Source：')
    print(df_all['Source'].value_counts())
    print('==============================\n')

    loaded_crop_values = set(df_all['Crop'].dropna().astype(str).unique())
    ACTIVE_CROP_ORDER = [crop for crop in CROP_ORDER if crop in loaded_crop_values]
    missing_crops = [crop for crop in CROP_ORDER if crop not in ACTIVE_CROP_ORDER]

    print(f'✅ 实际参与绘图的作物顺序：{ACTIVE_CROP_ORDER}')
    if missing_crops:
        print(f'⚠️ 未读取到的作物：{missing_crops}')
        if REQUIRE_ALL_CROPS:
            raise ValueError(
                '❌ 以下作物没有被成功读取：'
                + ', '.join(missing_crops)
                + '。请检查对应 *_DTF_TOTAL_FILES / *_FTD_TOTAL_FILES 的路径是否正确。'
            )

    return df_all


def compute_relative_for_all_targets(df_all):
    available_targets = [c for c in TARGET_COLS if c in df_all.columns]
    missing_targets = [c for c in TARGET_COLS if c not in df_all.columns]

    if missing_targets:
        print('⚠️ 以下参数在数据中不存在，将跳过：')
        for c in missing_targets:
            print(f'   - {c}')

    if len(available_targets) == 0:
        raise ValueError('❌ TARGET_COLS 中没有任何参数存在于数据表中。')

    df_rel_list = []

    for target_col in available_targets:
        df_rel_one = prepare_relative_for_one_target(df_all, target_col)
        if not df_rel_one.empty:
            df_rel_list.append(df_rel_one)

    if len(df_rel_list) == 0:
        raise ValueError('❌ 所有参数计算后都为空，无法绘图。')

    df_rel_all = pd.concat(df_rel_list, ignore_index=True)

    print('\n==============================')
    print('✅ 所有参数相对变化计算完成')
    print(f'最终数据行数：{len(df_rel_all)}')
    print('作物列表：')
    print(df_rel_all['Crop'].value_counts())
    print('参数列表：')
    print(df_rel_all['Target'].value_counts())
    print('情景列表：')
    print(df_rel_all['Source'].value_counts())
    print('==============================\n')

    return df_rel_all, available_targets



def _draw_one_crop_category_panel(
    fig,
    outer_cell,
    df_rel_all,
    target_col,
    category,
    crop_name,
    panel_letter,
    show_y_label=True,
    show_category_title=True,
    shared_y_ax=None
):
    """
    在合并图中的一个单元格内绘制：
        上：当前作物 × 当前 Category 的小提琴图；
        下：对应 Source 的离散程度柱状图。

    这个函数供 DTF/FTD 合并图复用，避免在主绘图函数中重复代码。
    """
    x_order = SOURCE_ORDER_BY_CATEGORY.get(category, SOURCE_ORDER_PLOT)
    x_col = 'Source'

    cell_df = df_rel_all[
        (df_rel_all['Target'] == target_col) &
        (df_rel_all['Category'] == category) &
        (df_rel_all['Crop'] == crop_name) &
        (df_rel_all['Source'].isin(x_order))
    ].copy()

    if SHOW_DISPERSION_BAR:
        inner_grid = outer_cell.subgridspec(
            2,
            1,
            height_ratios=VIOLIN_BAR_HEIGHT_RATIOS,
            hspace=0.05
        )

        if SHARE_Y_WITHIN_FIG and shared_y_ax is not None:
            ax = fig.add_subplot(inner_grid[0], sharey=shared_y_ax)
        else:
            ax = fig.add_subplot(inner_grid[0])

        bar_ax = fig.add_subplot(inner_grid[1], sharex=ax)
    else:
        if SHARE_Y_WITHIN_FIG and shared_y_ax is not None:
            ax = fig.add_subplot(outer_cell, sharey=shared_y_ax)
        else:
            ax = fig.add_subplot(outer_cell)
        bar_ax = None

    if cell_df.empty:
        ax.text(
            0.5,
            0.5,
            'No data',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=ANNOTATION_FONT_SIZE
        )
        ax.set_title(
            f'{crop_name} | {category}',
            fontsize=TITLE_FONT_SIZE,
            fontweight='normal'
        )
        ax.set_ylabel('')
        apply_nature_axis_style(ax)
        if bar_ax is not None:
            bar_ax.axis('off')
        return ax

    if SHARE_Y_WITHIN_FIG:
        # 合并图共享 y 轴时，y_limits 由外部主函数统一设置。这里仍然用 None，
        # 让 sharey 统一控制坐标轴范围。
        y_limits_local = None
    else:
        mean_group_cols = [x_col] if Y_LIMIT_MEAN_BY_GROUP else None
        y_limits_local = get_y_limits(
            cell_df,
            value_col='relative_value',
            mean_group_cols=mean_group_cols
        )

    plot_df = filter_data_to_y_limits(
        cell_df,
        y_limits_local,
        value_col='relative_value'
    )

    draw_nature_violin(ax, plot_df, x_col, x_order, y_limits=y_limits_local)
    add_summary_overlays(ax, cell_df, x_col, x_order)

    if SHOW_STRIP_POINTS:
        sns.stripplot(
            data=plot_df,
            x=x_col,
            y='relative_value',
            order=x_order,
            size=STRIP_POINT_SIZE,
            alpha=STRIP_POINT_ALPHA,
            jitter=0.20,
            color='black',
            linewidth=0,
            ax=ax,
            zorder=3
        )

    ax.axhline(
        0,
        linewidth=ZERO_LINEWIDTH,
        linestyle=(0, (2.5, 2.5)),
        color='0.35',
        zorder=0
    )

    # 面板字母。
    ax.text(
        -0.13,
        1.08,
        panel_letter,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=PANEL_LETTER_SIZE,
        fontweight='bold'
    )

    # 标题：第一行突出 DTF / FTD；其他行显示 作物 | Category，方便单独截取阅读。
    if show_category_title:
        title_text = f'{crop_name} | {category}'
    else:
        title_text = f'{crop_name} | {category}'

    ax.set_title(
        title_text,
        fontsize=TITLE_FONT_SIZE,
        fontweight='normal',
        pad=3
    )

    if SHOW_PANEL_SAMPLE_TEXT:
        ax.text(
            0.98,
            0.96,
            format_panel_sample_text(cell_df),
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=ANNOTATION_FONT_SIZE,
            linespacing=1.12
        )

    if show_y_label:
        ax.set_ylabel(
            'Relative change to ALL (%)' if RELATIVE_AS_PERCENT else 'Relative change to ALL',
            fontsize=AXIS_LABEL_SIZE
        )
    else:
        ax.set_ylabel('')

    display_labels = [get_source_label(category, s) for s in x_order]
    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(display_labels)

    if SHOW_DISPERSION_BAR:
        ax.tick_params(axis='x', labelbottom=False)
        ax.set_xlabel('')
    else:
        ax.set_xlabel('', fontsize=AXIS_LABEL_SIZE)
        for label in ax.get_xticklabels():
            label.set_rotation(35)
            label.set_ha('right')
            label.set_color('black')

    ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)

    if y_limits_local is not None:
        ax.set_ylim(*y_limits_local)

    apply_nature_axis_style(ax)

    if bar_ax is not None:
        draw_dispersion_bar(
            ax=bar_ax,
            data=cell_df,
            x_col=x_col,
            x_order=x_order,
            category=category
        )

        if not show_y_label:
            bar_ax.set_ylabel('')

    return ax


def draw_violin_grid_combined_dtfft(df_rel_all, target_col, crop_batch, batch_index):
    """
    对一个参数绘制一张 DTF + FTD 合并图。

    版式：
        行 = 作物；
        列 = DTF / FTD；
        每个单元格上方为小提琴图，下方为离散程度柱状图。

    这样每个参数只生成一张图，便于直接比较 DTF 和 FTD。
    """
    categories = [c for c in COMBINED_CATEGORY_ORDER if c in CATEGORY_ORDER]
    if not categories:
        categories = CATEGORY_ORDER.copy()

    fig_df = df_rel_all[
        (df_rel_all['Target'] == target_col) &
        (df_rel_all['Category'].isin(categories)) &
        (df_rel_all['Crop'].isin(crop_batch))
    ].copy()

    if fig_df.empty:
        print(f'⚠️ 无数据，跳过合并图：{target_col} | crops={crop_batch}')
        return None

    target_label = PARAM_LABELS.get(target_col, target_col)

    n_rows = len(crop_batch)
    n_cols = len(categories)

    # 如果只画少于 4 个作物，自动缩短高度；完整四作物时使用 COMBINED_FIG_HEIGHT_CM。
    fig_height_cm = max(
        10.0,
        COMBINED_FIG_HEIGHT_CM * (n_rows / max(1, CROPS_PER_FIG))
    )

    fig = plt.figure(
        figsize=(COMBINED_FIG_WIDTH_CM * CM_TO_INCH, fig_height_cm * CM_TO_INCH),
        facecolor='white'
    )

    outer_grid = fig.add_gridspec(
        n_rows,
        n_cols,
        wspace=0.28,
        hspace=0.36
    )

    # 共享 y 轴时，为整张合并图计算统一范围。
    shared_y_ax = None
    shared_y_limits = None
    if SHARE_Y_WITHIN_FIG:
        shared_y_limits = get_y_limits(
            fig_df,
            value_col='relative_value',
            mean_group_cols=['Crop', 'Category', 'Source'] if Y_LIMIT_MEAN_BY_GROUP else None
        )

    panel_letters = list('abcdefghijklmnopqrstuvwxyz')
    panel_index = 0

    for row_index, crop_name in enumerate(crop_batch):
        for col_index, category in enumerate(categories):
            ax = _draw_one_crop_category_panel(
                fig=fig,
                outer_cell=outer_grid[row_index, col_index],
                df_rel_all=df_rel_all,
                target_col=target_col,
                category=category,
                crop_name=crop_name,
                panel_letter=panel_letters[panel_index],
                show_y_label=(col_index == 0),
                show_category_title=(row_index == 0),
                shared_y_ax=shared_y_ax
            )

            if shared_y_ax is None:
                shared_y_ax = ax

            if SHARE_Y_WITHIN_FIG and shared_y_limits is not None:
                ax.set_ylim(*shared_y_limits)

            panel_index += 1

    fig.suptitle(
        f'{target_label} | DTF and FTD',
        fontsize=SUPTITLE_FONT_SIZE,
        fontweight='normal',
        y=0.992
    )

    add_source_group_legend(fig)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if SHOW_DISPERSION_BAR:
        out_dir_name = 'violin_with_dispersion_bar_DTF_FTD_combined_by_parameter'
        file_prefix = 'violin_dispersion_DTF_FTD_combined'
    else:
        out_dir_name = 'violin_DTF_FTD_combined_by_parameter'
        file_prefix = 'violin_DTF_FTD_combined'

    out_dir = Path(OUTPUT_DIR) / out_dir_name / sanitize_filename(target_col)
    out_dir.mkdir(parents=True, exist_ok=True)

    crop_tag = 'all4crops' if len(crop_batch) == 4 else f'crop_batch_{batch_index}'
    out_base = out_dir / (
        f'{file_prefix}_{sanitize_filename(target_col)}_{crop_tag}'
    )

    if SAVE_PNG:
        png_path = out_base.with_suffix('.png')
        fig.savefig(png_path, dpi=OUTPUT_DPI, bbox_inches='tight')
        print(f'✅ 已保存 PNG：{png_path}')

    if SAVE_PDF:
        pdf_path = out_base.with_suffix('.pdf')
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f'✅ 已保存 PDF：{pdf_path}')

    plt.close(fig)
    return out_base


def generate_all_violin_plots(df_rel_all, available_targets):
    output_count = 0

    for target_col in available_targets:

        if COMBINE_DTF_FTD_IN_ONE_FIG:
            subset = df_rel_all[
                (df_rel_all['Target'] == target_col) &
                (df_rel_all['Category'].isin(COMBINED_CATEGORY_ORDER))
            ]

            if subset.empty:
                print(f'⚠️ 无数据，跳过合并图：{target_col}')
                continue

            for crop_batch, batch_index in chunk_list(ACTIVE_CROP_ORDER, CROPS_PER_FIG):
                out_base = draw_violin_grid_combined_dtfft(
                    df_rel_all=df_rel_all,
                    target_col=target_col,
                    crop_batch=crop_batch,
                    batch_index=batch_index
                )
                if out_base is not None:
                    output_count += 1

            continue

        # 旧版逻辑：DTF 和 FTD 分别生成图片。
        for category in CATEGORY_ORDER:

            subset = df_rel_all[
                (df_rel_all['Target'] == target_col) &
                (df_rel_all['Category'] == category) &
                (df_rel_all['Source'].isin(SOURCE_ORDER_BY_CATEGORY.get(category, SOURCE_ORDER_PLOT)))
            ]

            if subset.empty:
                print(f'⚠️ 无数据，跳过：{category} | {target_col}')
                continue

            for crop_batch, batch_index in chunk_list(ACTIVE_CROP_ORDER, CROPS_PER_FIG):
                out_base = draw_violin_grid(
                    df_rel_all=df_rel_all,
                    target_col=target_col,
                    category=category,
                    crop_batch=crop_batch,
                    batch_index=batch_index
                )
                if out_base is not None:
                    output_count += 1

    return output_count

def main():
    # 全局绘图样式：Nature 风格白底、黑色细线、嵌入 TrueType 字体
    plt.style.use('default')
    sns.set_theme(style='white', context='paper')

    plt.rcParams.update({
        # 字体与输出：PDF/PS 使用 TrueType，便于 Illustrator 后期编辑。
        'font.family': FONT_FAMILY,
        'font.sans-serif': [FONT_FAMILY, 'Arial', 'DejaVu Sans'],
        'font.size': BASE_FONT_SIZE,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,

        # Nature 风格：白底、黑色细线、无网格、无图例边框。
        'axes.linewidth': AXIS_LINEWIDTH,
        'axes.edgecolor': 'black',
        'axes.facecolor': 'white',
        'axes.grid': False,
        'axes.labelsize': AXIS_LABEL_SIZE,
        'axes.titlesize': TITLE_FONT_SIZE,
        'figure.facecolor': 'white',
        'figure.dpi': OUTPUT_DPI,
        'savefig.dpi': OUTPUT_DPI,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',

        # 细刻度、外向刻度。
        'xtick.major.width': TICK_WIDTH,
        'ytick.major.width': TICK_WIDTH,
        'xtick.major.size': TICK_LENGTH,
        'ytick.major.size': TICK_LENGTH,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'xtick.labelsize': TICK_LABEL_SIZE,
        'ytick.labelsize': TICK_LABEL_SIZE,

        # 文本与图例。
        'text.color': 'black',
        'legend.frameon': False,
        'legend.fontsize': TICK_LABEL_SIZE
    })

    df_all = load_all_crop_data()
    df_rel_all, available_targets = compute_relative_for_all_targets(df_all)

    relative_csv = Path(OUTPUT_DIR) / 'violin_relative_values_long_table.csv'
    df_rel_all.to_csv(relative_csv, index=False, encoding='utf-8-sig')
    print(f'✅ violin 原始长表已保存：{relative_csv}')

    summary = build_violin_summary(df_rel_all)
    summary_csv = Path(OUTPUT_DIR) / 'violin_summary_by_target_crop_category_source_trt.csv'
    summary.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    print(f'✅ violin 汇总表已保存：{summary_csv}')

    dispersion_summary = build_dispersion_summary(df_rel_all)
    dispersion_summary_csv = Path(OUTPUT_DIR) / 'dispersion_summary_by_target_crop_category_source.csv'
    dispersion_summary.to_csv(dispersion_summary_csv, index=False, encoding='utf-8-sig')
    print(f'✅ 离散程度汇总表已保存：{dispersion_summary_csv}')

    output_count = generate_all_violin_plots(df_rel_all, available_targets)

    print('\n🎉 小提琴图全部完成！')
    print(f'📌 输出图片数量：{output_count}')
    print('📌 组织方式：每个参数输出一张 DTF + FTD 合并图；行=作物，列=DTF/FTD；每个面板上方为小提琴图，下方为离散程度柱状图。')
    print(f'📁 输出文件夹：{OUTPUT_DIR}')


if __name__ == '__main__':
    main()
