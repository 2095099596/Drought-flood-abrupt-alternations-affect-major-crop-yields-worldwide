import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 防止无图形界面环境报错

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker
import os
import re
from pathlib import Path
from PIL import Image

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Geod


# ============================================================
# 1. 数据路径配置
# ============================================================
# 说明：
# 1) 四种作物固定绘制在作物作为一级分组，每个格子只显示当前作物的一根竖条：
#    Maize / Rice / Soybeans / Wheat
# 2) 如果某个作物实际路径不是 D:\作物名，请只修改下面对应作物的路径。

# ------------------ Maize | DTF ------------------
MAIZE_DTF_TOTAL_FILES = {
    'ALL': r'D:\Maize\OVERVIEW_LDFAI+0\OVERVIEW_白名单_站点单独保存\OVERVIEW_LDFAI+0_ALL_SITES_TOTAL.csv',
    'SPEI 0.5': r'D:\Maize\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_0p5.csv',
    'SPEI 1': r'D:\Maize\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1.csv',
    'SPEI 1.5': r'D:\Maize\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1p5.csv',
    'DFAI 1': r'D:\Maize\OVERVIEW_LDFAI+0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_1.csv',
    'DFAI 4': r'D:\Maize\OVERVIEW_LDFAI+0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_4.csv',
    'DFAI 9': r'D:\Maize\OVERVIEW_LDFAI+0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_9.csv',
    'SPEI -0.5': r'D:\Maize\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg0p5.csv',
    'SPEI -1': r'D:\Maize\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1.csv',
    'SPEI -1.5': r'D:\Maize\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1p5.csv'
}

# ------------------ Maize | FTD ------------------
MAIZE_FTD_TOTAL_FILES = {
    'ALL': r'D:\Maize\OVERVIEW_LDFAI-0\OVERVIEW_白名单_站点单独保存\OVERVIEW_LDFAI-0_ALL_SITES_TOTAL.csv',
    'SPEI 0.5': r'D:\Maize\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_0p5.csv',
    'SPEI 1': r'D:\Maize\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1.csv',
    'SPEI 1.5': r'D:\Maize\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1p5.csv',
    'DFAI 1': r'D:\Maize\OVERVIEW_LDFAI-0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_neg1.csv',
    'DFAI 4': r'D:\Maize\OVERVIEW_LDFAI-0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_neg4.csv',
    'DFAI 9': r'D:\Maize\OVERVIEW_LDFAI-0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_neg9.csv',
    'SPEI -0.5': r'D:\Maize\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg0p5.csv',
    'SPEI -1': r'D:\Maize\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1.csv',
    'SPEI -1.5': r'D:\Maize\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1p5.csv'
}

# ------------------ Rice | DTF ------------------
RICE_DTF_TOTAL_FILES = {
    'ALL': r'D:\Rice\OVERVIEW_LDFAI+0\OVERVIEW_白名单_站点单独保存\OVERVIEW_LDFAI+0_ALL_SITES_TOTAL.csv',
    'SPEI 0.5': r'D:\Rice\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_0p5.csv',
    'SPEI 1': r'D:\Rice\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1.csv',
    'SPEI 1.5': r'D:\Rice\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1p5.csv',
    'DFAI 1': r'D:\Rice\OVERVIEW_LDFAI+0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_1.csv',
    'DFAI 4': r'D:\Rice\OVERVIEW_LDFAI+0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_4.csv',
    'DFAI 9': r'D:\Rice\OVERVIEW_LDFAI+0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_9.csv',
    'SPEI -0.5': r'D:\Rice\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg0p5.csv',
    'SPEI -1': r'D:\Rice\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1.csv',
    'SPEI -1.5': r'D:\Rice\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1p5.csv'
}

# ------------------ Rice | FTD ------------------
RICE_FTD_TOTAL_FILES = {
    'ALL': r'D:\Rice\OVERVIEW_LDFAI-0\OVERVIEW_白名单_站点单独保存\OVERVIEW_LDFAI-0_ALL_SITES_TOTAL.csv',
    'SPEI 0.5': r'D:\Rice\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_0p5.csv',
    'SPEI 1': r'D:\Rice\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1.csv',
    'SPEI 1.5': r'D:\Rice\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1p5.csv',
    'DFAI 1': r'D:\Rice\OVERVIEW_LDFAI-0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_neg1.csv',
    'DFAI 4': r'D:\Rice\OVERVIEW_LDFAI-0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_neg4.csv',
    'DFAI 9': r'D:\Rice\OVERVIEW_LDFAI-0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_neg9.csv',
    'SPEI -0.5': r'D:\Rice\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg0p5.csv',
    'SPEI -1': r'D:\Rice\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1.csv',
    'SPEI -1.5': r'D:\Rice\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1p5.csv'
}

# ------------------ Soybeans | DTF ------------------
SOYBEANS_DTF_TOTAL_FILES = {
    'ALL': r'D:\Soybeans\OVERVIEW_LDFAI+0\OVERVIEW_白名单_站点单独保存\OVERVIEW_LDFAI+0_ALL_SITES_TOTAL.csv',
    'SPEI 0.5': r'D:\Soybeans\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_0p5.csv',
    'SPEI 1': r'D:\Soybeans\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1.csv',
    'SPEI 1.5': r'D:\Soybeans\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1p5.csv',
    'DFAI 1': r'D:\Soybeans\OVERVIEW_LDFAI+0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_1.csv',
    'DFAI 4': r'D:\Soybeans\OVERVIEW_LDFAI+0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_4.csv',
    'DFAI 9': r'D:\Soybeans\OVERVIEW_LDFAI+0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_9.csv',
    'SPEI -0.5': r'D:\Soybeans\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg0p5.csv',
    'SPEI -1': r'D:\Soybeans\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1.csv',
    'SPEI -1.5': r'D:\Soybeans\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1p5.csv'
}

# ------------------ Soybeans | FTD ------------------
SOYBEANS_FTD_TOTAL_FILES = {
    'ALL': r'D:\Soybeans\OVERVIEW_LDFAI-0\OVERVIEW_白名单_站点单独保存\OVERVIEW_LDFAI-0_ALL_SITES_TOTAL.csv',
    'SPEI 0.5': r'D:\Soybeans\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_0p5.csv',
    'SPEI 1': r'D:\Soybeans\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1.csv',
    'SPEI 1.5': r'D:\Soybeans\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1p5.csv',
    'DFAI 1': r'D:\Soybeans\OVERVIEW_LDFAI-0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_neg1.csv',
    'DFAI 4': r'D:\Soybeans\OVERVIEW_LDFAI-0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_neg4.csv',
    'DFAI 9': r'D:\Soybeans\OVERVIEW_LDFAI-0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_neg9.csv',
    'SPEI -0.5': r'D:\Soybeans\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg0p5.csv',
    'SPEI -1': r'D:\Soybeans\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1.csv',
    'SPEI -1.5': r'D:\Soybeans\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1p5.csv'
}

# ------------------ Wheat | DTF ------------------
WHEAT_DTF_TOTAL_FILES = {
    'ALL': r'D:\Wheat\OVERVIEW_LDFAI+0\OVERVIEW_白名单_站点单独保存\OVERVIEW_LDFAI+0_ALL_SITES_TOTAL.csv',
    'SPEI 0.5': r'D:\Wheat\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_0p5.csv',
    'SPEI 1': r'D:\Wheat\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1.csv',
    'SPEI 1.5': r'D:\Wheat\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1p5.csv',
    'DFAI 1': r'D:\Wheat\OVERVIEW_LDFAI+0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_1.csv',
    'DFAI 4': r'D:\Wheat\OVERVIEW_LDFAI+0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_4.csv',
    'DFAI 9': r'D:\Wheat\OVERVIEW_LDFAI+0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_9.csv',
    'SPEI -0.5': r'D:\Wheat\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg0p5.csv',
    'SPEI -1': r'D:\Wheat\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1.csv',
    'SPEI -1.5': r'D:\Wheat\OVERVIEW_LDFAI+0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1p5.csv'
}

# ------------------ Wheat | FTD ------------------
WHEAT_FTD_TOTAL_FILES = {
    'ALL': r'D:\Wheat\OVERVIEW_LDFAI-0\OVERVIEW_白名单_站点单独保存\OVERVIEW_LDFAI-0_ALL_SITES_TOTAL.csv',
    'SPEI 0.5': r'D:\Wheat\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_0p5.csv',
    'SPEI 1': r'D:\Wheat\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1.csv',
    'SPEI 1.5': r'D:\Wheat\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_1p5.csv',
    'DFAI 1': r'D:\Wheat\OVERVIEW_LDFAI-0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_neg1.csv',
    'DFAI 4': r'D:\Wheat\OVERVIEW_LDFAI-0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_neg4.csv',
    'DFAI 9': r'D:\Wheat\OVERVIEW_LDFAI-0\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_neg9.csv',
    'SPEI -0.5': r'D:\Wheat\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg0p5.csv',
    'SPEI -1': r'D:\Wheat\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1.csv',
    'SPEI -1.5': r'D:\Wheat\OVERVIEW_LDFAI-0\OVERVIEW_SPEI\ALL_最终筛选年份_总表_neg1p5.csv'
}


CROP_TOTAL_FILES = {
    'Maize': {'DTF': MAIZE_DTF_TOTAL_FILES, 'FTD': MAIZE_FTD_TOTAL_FILES},
    'Rice': {'DTF': RICE_DTF_TOTAL_FILES, 'FTD': RICE_FTD_TOTAL_FILES},
    'Soybeans': {'DTF': SOYBEANS_DTF_TOTAL_FILES, 'FTD': SOYBEANS_FTD_TOTAL_FILES},
    'Wheat': {'DTF': WHEAT_DTF_TOTAL_FILES, 'FTD': WHEAT_FTD_TOTAL_FILES}
}

CROP_ORDER = ['Maize', 'Rice', 'Soybeans', 'Wheat']
ACTIVE_CROP_ORDER = CROP_ORDER.copy()
REQUIRE_ALL_CROPS = True


# ============================================================
# 2. 输出与分析参数
# ============================================================

OUTPUT_DIR = r'D:\Python\DssAat\NO_3\HUITU_BAR_MATRIX'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 小图 a：土壤站点地图数据
SOIL_SITE_CSV_FILE = Path(r'D:\Python\DssAat\Soybeans\all_soil_extracted_C.csv')
PANEL_A_STEM = 'panel_a_soil_sites_map_smaller'
PANEL_B_STEM_PREFIX = 'panel_b_nature_mini_violin_matrix_larger'
COMBINED_STEM_PREFIX = 'combined_panel_a_smaller_b_larger_nature_violin'

# 合并布局设置
MERGE_LAYOUT = 'vertical'        # 'vertical' = a 在上、b 在下；'horizontal' = a 在左、b 在右
MERGE_GAP_PX = 10               # 小图 a 和 b 之间的空白间距；压缩上下留白
MERGE_RESIZE_TO_SAME_WIDTH = False
MERGE_RESIZE_TO_SAME_HEIGHT = False
# 合并时让 panel a 按 panel b 的宽度缩小，避免地图过大、b 图过小。
MERGE_PANEL_A_TO_B_WIDTH = True
MERGE_PANEL_A_WIDTH_RATIO = 0.86
MERGE_OUTPUT_WIDTH_SCALE = 1.00
MERGE_MAX_WIDTH_PX = None       # 例如 4200；None 表示不额外限制

PLOT_BY_TRT = False  # 默认与小提琴图一致：合并所有 trt；如需每个 TRT 单独输出可改为 True

IGNORE_ZERO_BEFORE_MEAN = True
EPS = 1e-9
RELATIVE_AS_PERCENT = False
CI_Z = 1.96

OUTPUT_DPI = 600


# ============================================================
# 3. 参数分组：每个参数是一列，并按组分隔
# ============================================================

COL_GROUPS = [
    (
        'Phenology',
        [
            'Anthesis_DAP',
            # 'Maturity_DAP'
        ]
    ),

    (
        'Production',
        [
            'Yield_HA'
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
        'Nitrogen',
        [
            'N_Uptake_kg_ha'
        ]
    )
]

TARGET_COLS = []
for group_name, cols in COL_GROUPS:
    TARGET_COLS.extend(cols)

PARAM_LABELS = {
    'Anthesis_DAP': 'Anthesis',
    'Maturity_DAP': 'Maturity',
    'Yield_HA': 'Yield',
    'PreAnthesis_WaterPhotoStress_Mean': 'PreA\nphoto',
    'PreAnthesis_WaterGrowthStress_Mean': 'PreA\ngrowth',
    'GrainFilling_WaterPhotoStress': 'GF\nphoto',
    'GrainFilling_WaterGrowthStress': 'GF\ngrowth',
    'N_Uptake_kg_ha': 'N uptake'
}


# ============================================================
# 4. 列分组位置设置
# ============================================================

# 列更紧凑：缩小指标列之间的步长和列组间距
# 列距适度增大，避免每个 cell 左侧的 y 轴范围标签与前一列柱子重叠。
# 组间距也略增大，用于区分 Phenology / Production / Water stress / Nitrogen。
COL_STEP = 0.94
INNER_COL_GROUP_GAP = 0.34

COL_ITEMS = []
x_pos = 0.0

for group_name, col_list in COL_GROUPS:
    group_start_x = x_pos

    for target_col in col_list:
        COL_ITEMS.append({
            'Group': group_name,
            'Target': target_col,
            'x': x_pos,
            'Col_Label': PARAM_LABELS.get(target_col, target_col)
        })
        x_pos += COL_STEP

    group_end_x = x_pos - COL_STEP

    for item in COL_ITEMS:
        if item['Group'] == group_name and 'Group_Center_X' not in item:
            item['Group_Center_X'] = (group_start_x + group_end_x) / 2

    x_pos += INNER_COL_GROUP_GAP


# ============================================================
# 5. 行顺序：作物为一级分组；每个作物只保留 DTF / FTD 两行
# ============================================================

SOURCE_ORDER_PLOT = [
    'SPEI 0.5',
    'SPEI 1',
    'DFAI 1',
    'DFAI 4',
    'SPEI -0.5',
    'SPEI -1'
]

CATEGORY_ORDER = ['DTF', 'FTD']

ROW_LABELS_BY_CATEGORY = {
    'DTF': {
        'SPEI 0.5': 'SPEI > 0.5',
        'SPEI 1': 'SPEI > 1',
        'DFAI 1': 'DFAI > 1',
        'DFAI 4': 'DFAI > 4',
        'SPEI -0.5': 'SPEI < -0.5',
        'SPEI -1': 'SPEI < -1'
    },
    'FTD': {
        'SPEI 0.5': 'SPEI > 0.5',
        'SPEI 1': 'SPEI > 1',
        'DFAI 1': 'DFAI < -1',
        'DFAI 4': 'DFAI < -4',
        'SPEI -0.5': 'SPEI < -0.5',
        'SPEI -1': 'SPEI < -1'
    }
}


def get_source_label(category, source):
    """
    DTF 和 FTD 使用不同的 DFAI 标签：
        DTF: DFAI > 1 / DFAI > 4
        FTD: DFAI < -1 / DFAI < -4
    """
    return ROW_LABELS_BY_CATEGORY.get(category, {}).get(source, source)


# 每个 cell 中 6 根条的颜色。颜色用于区分三类事件，条的位置用于区分阈值。
SOURCE_BAR_COLORS = {
    # 同一类事件使用完全相同的颜色，不再用深浅区分阈值。
    'SPEI 0.5': '#4C78A8',
    'SPEI 1': '#4C78A8',
    'DFAI 1': '#7E6E85',
    'DFAI 4': '#7E6E85',
    'SPEI -0.5': '#D67D6E',
    'SPEI -1': '#D67D6E'
}

# 行距设置：现在每个作物只有 DTF / FTD 两行，
# 每一行的每个 cell 内部包含 6 根 source bar。
# 小图 b 变高：增加 DTF/FTD 行距、作物组间距
ROW_STEP = 0.98
CATEGORY_GAP = 0.10
CROP_GROUP_GAP = 0.42

ROW_ITEMS = []
y_pos = 0.0

for crop_name in CROP_ORDER:
    crop_start_y = y_pos

    for category in CATEGORY_ORDER:
        ROW_ITEMS.append({
            'Crop': crop_name,
            'Category': category,
            'y': y_pos,
            'Row_Label': category
        })
        y_pos += ROW_STEP

        # 同一作物内部 DTF 和 FTD 之间的微小间隔
        if category != CATEGORY_ORDER[-1]:
            y_pos += CATEGORY_GAP

    crop_end_y = y_pos - ROW_STEP

    for item in ROW_ITEMS:
        if item['Crop'] == crop_name and 'Crop_Center_Y' not in item:
            item['Crop_Center_Y'] = (crop_start_y + crop_end_y) / 2

    y_pos += CROP_GROUP_GAP


# ============================================================
# 6. 图形基础设置
# ============================================================

FONT_FAMILY = 'Arial'

BASE_FONT_SIZE = 6.2
ROW_LABEL_SIZE = 6.0
COL_LABEL_SIZE = 5.9
GROUP_LABEL_SIZE = 6.7
COL_GROUP_LABEL_SIZE = 6.7
LEGEND_FONT_SIZE = 5.8

# 更紧凑的画布设置：行数从 48 行压缩为 8 行，所以高度可以明显降低。
# 列距压缩后，横向画布也相应收窄；纵向画布增高
# 列距增大后，画布仅小幅增加；同时压缩左侧和上下空白，让主体更紧凑。
FIG_WIDTH_PER_COL = 0.66
FIG_HEIGHT_PER_ROW = 0.46
FIG_EXTRA_W = 1.10
FIG_EXTRA_H = 1.05

# 每个格子是一个迷你条形图：6 根 source bar
# 列距缩小后同步收窄 cell；纵向增高使条形和误差线更清楚
CELL_BOX_HALF_WIDTH = 0.375
CELL_BOX_HALF_HEIGHT = 0.430
CELL_HALF_WIDTH = 0.360
CELL_HALF_HEIGHT = 0.405

BAR_AREA_HALF_WIDTH = 0.345
BAR_MAX_HALF_HEIGHT = 0.390
BAR_WIDTH_RATIO = 0.78
# 横线全部取消，因此误差线只保留竖线，不再绘制水平端帽。
ERRORBAR_CAP_RATIO = 0.46

# 小提琴图形状设置：每个 cell 内 6 个 source 各画一个 Nature 风格迷你 violin。
# 这里与单独小提琴图脚本保持一致：trimmed KDE、低饱和配色、黑色细轮廓、IQR 竖线 + 中位数短横线。
MINI_VIOLIN_MAX_HALF_WIDTH_RATIO = 0.46
MINI_VIOLIN_DENSITY_POINTS = 220
MINI_VIOLIN_DENSITY_MIN_RATIO = 0.025
MINI_VIOLIN_BW_ADJUST = 1.15
MINI_VIOLIN_EDGE_WIDTH = 0.30
MINI_VIOLIN_ALPHA = 0.80
SHOW_MINI_VIOLIN_IQR = True
MINI_VIOLIN_IQR_LINEWIDTH = 0.62
SHOW_MINI_VIOLIN_MEDIAN_TICK = True
MINI_VIOLIN_MEDIAN_TICK_WIDTH_RATIO = 0.45
MINI_VIOLIN_MEDIAN_TICK_LINEWIDTH = 0.55
# 是否显示每个 cell 的 0 参考线。Nature 单独小提琴图有 0 虚线，
# 但矩阵图中太多横线会拥挤，所以默认关闭。
SHOW_MINI_ZERO_LINE = False
MINI_ZERO_LINEWIDTH = 0.22

# y 轴范围采用小提琴图类似的稳健分位数：每个 source 先算 5%~95%，再取 cell 内最大范围。
MINI_VIOLIN_QUANTILE = 0.05
MINI_VIOLIN_PADDING_RATIO = 0.10

# 每个小条形图左侧显示自己的 y 轴范围；最下面一行显示 6 个 source 的 x 轴标签。
SHOW_CELL_Y_RANGE_LABEL = True
CELL_Y_RANGE_LABEL_SIZE = 3.2
# y 轴范围标签向左移，避免和本 cell 的第一根柱子重叠；
# COL_STEP 同步增大，避免它和前一列柱子重叠。
CELL_Y_RANGE_LABEL_X_OFFSET = 0.088
SHOW_BOTTOM_SOURCE_X_LABELS = True
BOTTOM_SOURCE_X_LABEL_SIZE = 3.8
BOTTOM_SOURCE_X_LABEL_Y_OFFSET = 0.115
BOTTOM_SOURCE_X_LABEL_ROTATION = 90

# 左侧标签位置：手动绘制 DTF/FTD 与作物名，避免 matplotlib 自动 ytick 与作物名重叠。
LEFT_ROW_LABEL_X_OFFSET = 0.62       # DTF / FTD 距离第一列的水平偏移
LEFT_CROP_LABEL_X_OFFSET = 1.02      # 农作物名称距离第一列的水平偏移
PANEL_B_LABEL_X_OFFSET = 1.20        # 面板 b 标签距离第一列的水平偏移

# 作物颜色仍用于左侧作物标签；cell 内部条形颜色由 SOURCE_BAR_COLORS 决定。
CROP_COLORS = {
    'Maize': '#4C78A8',
    'Rice': '#F58518',
    'Soybeans': '#54A24B',
    'Wheat': '#B279A2'
}

# 紧凑版：默认不放大段说明。每张图会输出 source 顺序 CSV 和 scale CSV。
SHOW_CROP_LEGEND = False
SHOW_SCALE_NOTE = False
SHOW_TITLE = False
SHOW_SOURCE_ORDER_NOTE = False


# ============================================================
# 7. y 方向数值缩放
# ============================================================

# 现在默认使用“每个小条形图（Crop × Category × Target）单独 y 轴范围”。
# 也就是说，同一张图中每个 cell 都会根据自己的 6 根 source bar 数据计算 scale。
# 若改为 True，则会退回到每个 Target 共用范围。
USE_GLOBAL_BAR_SCALE = False

# True  = 所有 trt 图共用范围
# False = 每个 trt 图内部自己算范围
USE_GLOBAL_BAR_SCALE_ACROSS_TRT = False

# 最小对称范围
BAR_SCALE_MIN_ABS = 0.01

# 缩放方式：
# 'max_abs'         = max(abs(mean ± ci95))
# 'quantile_abs95'  = 95% 分位数 abs(mean ± ci95)
BAR_SCALE_METHOD = 'max_abs'


# ============================================================
# 8. 顶部列分组三线表
# ============================================================

SHOW_TOP_COLGROUP_TABLE = True
TOP_TABLE_YLIM_OFFSET = 1.70

TOP_TABLE_TOP_OFFSET = 1.48
TOP_TABLE_HEADER_OFFSET = 1.24
TOP_TABLE_CONTENT_OFFSET = 0.72
TOP_TABLE_BOTTOM_OFFSET = 0.28

TOP_TABLE_LINE_WIDTH = 0.52


# ============================================================
# 9. 工具函数
# ============================================================


# ============================================================
# 9A. 小图 a：土壤站点地图与图片合并函数
# ============================================================

def draw_panel_a_map(out_dir=OUTPUT_DIR):
    """
    生成小图 a：全球土壤站点分布图。
    返回 PNG/PDF/SVG 路径字典。该函数来自单独地图代码，并封装为可复用函数。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not SOIL_SITE_CSV_FILE.exists():
        raise FileNotFoundError(f'❌ 小图 a 的站点 CSV 不存在：{SOIL_SITE_CSV_FILE}')

    df = pd.read_csv(SOIL_SITE_CSV_FILE)

    required_cols = ['lat', 'lon']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'❌ 小图 a 的站点 CSV 缺少列：{col}')

    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')

    df = df.dropna(subset=['lat', 'lon']).copy()
    df = df[
        (df['lat'] >= -90) & (df['lat'] <= 90) &
        (df['lon'] >= -180) & (df['lon'] <= 180)
    ].copy()

    df_unique = df.drop_duplicates(subset=['lat', 'lon']).copy()

    print(f'小图 a 原始站点数: {len(df)}')
    print(f'小图 a 去重后站点数: {len(df_unique)}')

    plt.rcParams.update({
        'font.family': FONT_FAMILY,
        'font.sans-serif': [FONT_FAMILY, 'Arial', 'DejaVu Sans'],
        'font.size': 7,
        'axes.linewidth': 0.4,
        'axes.edgecolor': '0.25',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
    })

    # 小图 a 宽度按双栏设置；后续合并时会按 b 图宽度自动等比缩放。
    fig_width_mm = 165
    fig_height_mm = 68

    fig = plt.figure(
        figsize=(fig_width_mm / 25.4, fig_height_mm / 25.4),
        dpi=300
    )

    ax = fig.add_axes(
        [0.01, 0.04, 0.98, 0.93],
        projection=ccrs.Robinson()
    )

    ax.set_global()
    ax.set_facecolor('white')

    ax.add_feature(cfeature.OCEAN, facecolor='white', edgecolor='none', zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='#f1f1f1', edgecolor='none', zorder=1)
    ax.add_feature(cfeature.LAKES, facecolor='white', edgecolor='none', zorder=1)

    ax.coastlines(resolution='110m', linewidth=0.28, color='0.42', zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.16, edgecolor='0.70', zorder=2)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.18,
        color='0.82',
        alpha=0.42,
        linestyle='-',
        zorder=1.5
    )
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = mticker.FixedLocator(np.arange(-60, 91, 30))

    ax.scatter(
        df_unique['lon'],
        df_unique['lat'],
        s=0.25,  # 原来是 5；数值越小圆点越小
        facecolors='black',  # 空心圆
        edgecolors='black',  # 圆圈边缘为黑色
        linewidth=0.1,
        alpha=0.86,
        transform=ccrs.PlateCarree(),
        zorder=4,
        rasterized=True,
        label=f'Soil sites, n = {len(df_unique)}'
    )

    # 面板标记 a
    ax.text(
        0.012,
        0.982,
        'a',
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=8,
        fontweight='bold',
        color='0.05',
        zorder=10
    )

    add_geodesic_scalebar(
        ax,
        lon0=-165,
        lat0=-52,
        length_km=1000,
        segment_km=1000
    )

    ax.text(
        0.015,
        0.050,
        'Robinson projection\nScale varies by latitude',
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        fontsize=5.3,
        color='0.45',
        zorder=10,
        linespacing=1.15
    )

    leg = ax.legend(
        loc='lower right',
        bbox_to_anchor=(0.985, 0.075),
        frameon=True,
        fontsize=5.8,
        handlelength=0.8,
        handletextpad=0.35,
        borderpad=0.25,
        labelspacing=0.25,
        markerscale=0.9
    )

    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('0.8')
    leg.get_frame().set_linewidth(0.3)
    leg.get_frame().set_alpha(0.9)

    for text in leg.get_texts():
        text.set_color('0.25')

    if 'geo' in ax.spines:
        ax.spines['geo'].set_linewidth(0.32)
        ax.spines['geo'].set_edgecolor('0.45')

    pdf_path = out_dir / f'{PANEL_A_STEM}.pdf'
    png_path = out_dir / f'{PANEL_A_STEM}.png'
    svg_path = out_dir / f'{PANEL_A_STEM}.svg'

    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.005)
    fig.savefig(png_path, dpi=OUTPUT_DPI, bbox_inches='tight', pad_inches=0.005)
    fig.savefig(svg_path, bbox_inches='tight', pad_inches=0.005)
    plt.close(fig)

    print(f'✅ 小图 a 已保存 PNG：{png_path}')
    print(f'✅ 小图 a 已保存 PDF：{pdf_path}')

    return {
        'png': str(png_path),
        'pdf': str(pdf_path),
        'svg': str(svg_path)
    }


def add_geodesic_scalebar(
    ax,
    lon0=-165,
    lat0=-52,
    length_km=2000,
    segment_km=1000,
    height_deg=1.4,
    text_offset_deg=3.0
):
    """
    添加基于 WGS84 椭球体大地测量距离的局部比例尺。
    """
    geod = Geod(ellps='WGS84')
    n_segments = int(length_km / segment_km)

    lons = []
    lats = []

    for i in range(n_segments + 1):
        lon_i, lat_i, _ = geod.fwd(
            lon0,
            lat0,
            90,
            i * segment_km * 1000
        )
        lons.append(lon_i)
        lats.append(lat_i)

    line_color = '0.20'
    text_color = '0.20'

    lon_bg0 = lon0 - 4
    lon_bg1 = lons[-1] + 4
    lat_bg0 = lat0 - text_offset_deg - 2.5
    lat_bg1 = lat0 + height_deg + 2.8

    bg = Rectangle(
        (lon_bg0, lat_bg0),
        lon_bg1 - lon_bg0,
        lat_bg1 - lat_bg0,
        transform=ccrs.PlateCarree(),
        facecolor='white',
        edgecolor='none',
        alpha=0.72,
        zorder=8
    )
    ax.add_patch(bg)

    for i in range(n_segments):
        ax.plot(
            [lons[i], lons[i + 1]],
            [lats[i], lats[i + 1]],
            transform=ccrs.Geodetic(),
            color=line_color,
            linewidth=0.75,
            solid_capstyle='butt',
            zorder=10
        )

    for lon_i, lat_i in zip(lons, lats):
        ax.plot(
            [lon_i, lon_i],
            [lat_i - height_deg / 2, lat_i + height_deg / 2],
            transform=ccrs.PlateCarree(),
            color=line_color,
            linewidth=0.65,
            zorder=10
        )

    for i, lon_i in enumerate(lons):
        label_km = i * segment_km
        label = f'{label_km:,} km' if i == len(lons) - 1 else f'{label_km:,}'

        ax.text(
            lon_i,
            lat0 - text_offset_deg,
            label,
            transform=ccrs.PlateCarree(),
            ha='center',
            va='top',
            fontsize=5.3,
            color=text_color,
            zorder=11
        )

    ax.text(
        lon0,
        lat0 + height_deg + 1.0,
        'Local scale',
        transform=ccrs.PlateCarree(),
        ha='left',
        va='bottom',
        fontsize=5.1,
        color='0.35',
        zorder=11
    )


def _resize_by_width(img, target_width):
    if img.width == target_width:
        return img
    new_height = int(round(img.height * target_width / img.width))
    return img.resize((target_width, new_height), Image.LANCZOS)


def _resize_by_height(img, target_height):
    if img.height == target_height:
        return img
    new_width = int(round(img.width * target_height / img.height))
    return img.resize((new_width, target_height), Image.LANCZOS)


def merge_panel_images(panel_a_png, panel_b_png, out_base):
    """
    合并小图 a 和小图 b，默认上下合并。
    同时输出 PNG 和 PDF。PDF 为合并后位图版，适合直接投稿前检查排版。
    """
    img_a = Image.open(panel_a_png).convert('RGB')
    img_b = Image.open(panel_b_png).convert('RGB')

    if MERGE_LAYOUT == 'vertical':
        if MERGE_PANEL_A_TO_B_WIDTH:
            target_width_a = max(1, int(round(img_b.width * MERGE_PANEL_A_WIDTH_RATIO)))
            img_a = _resize_by_width(img_a, target_width_a)
        elif MERGE_RESIZE_TO_SAME_WIDTH:
            target_width = max(img_a.width, img_b.width)
            img_a = _resize_by_width(img_a, target_width)
            img_b = _resize_by_width(img_b, target_width)

        final_width = max(img_a.width, img_b.width)
        final_height = img_a.height + MERGE_GAP_PX + img_b.height

        merged = Image.new('RGB', (final_width, final_height), 'white')
        merged.paste(img_a, ((final_width - img_a.width) // 2, 0))
        merged.paste(img_b, ((final_width - img_b.width) // 2, img_a.height + MERGE_GAP_PX))

    elif MERGE_LAYOUT == 'horizontal':
        if MERGE_RESIZE_TO_SAME_HEIGHT:
            target_height = max(img_a.height, img_b.height)
            img_a = _resize_by_height(img_a, target_height)
            img_b = _resize_by_height(img_b, target_height)

        final_width = img_a.width + MERGE_GAP_PX + img_b.width
        final_height = max(img_a.height, img_b.height)

        merged = Image.new('RGB', (final_width, final_height), 'white')
        merged.paste(img_a, (0, (final_height - img_a.height) // 2))
        merged.paste(img_b, (img_a.width + MERGE_GAP_PX, (final_height - img_b.height) // 2))

    else:
        raise ValueError("❌ MERGE_LAYOUT 只能是 'vertical' 或 'horizontal'")

    if MERGE_OUTPUT_WIDTH_SCALE != 1.0:
        new_width = int(round(merged.width * MERGE_OUTPUT_WIDTH_SCALE))
        new_height = int(round(merged.height * MERGE_OUTPUT_WIDTH_SCALE))
        merged = merged.resize((new_width, new_height), Image.LANCZOS)

    if MERGE_MAX_WIDTH_PX is not None and merged.width > MERGE_MAX_WIDTH_PX:
        new_width = int(MERGE_MAX_WIDTH_PX)
        new_height = int(round(merged.height * new_width / merged.width))
        merged = merged.resize((new_width, new_height), Image.LANCZOS)

    out_base = str(out_base)
    png_path = out_base + '.png'
    pdf_path = out_base + '.pdf'

    merged.save(png_path, dpi=(OUTPUT_DPI, OUTPUT_DPI))
    merged.save(pdf_path, resolution=OUTPUT_DPI)

    print(f'✅ 合并总图已保存 PNG：{png_path}')
    print(f'✅ 合并总图已保存 PDF：{pdf_path}')

    return {
        'png': png_path,
        'pdf': pdf_path
    }

def natural_sort_key(x):
    return [
        int(t) if t.isdigit() else t
        for t in re.split(r'(\d+)', str(x))
    ]


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
    对单个参数计算：
    1. 求均值前忽略 0
    2. Crop + Category + Source + soil_id + trt 求均值
    3. 对每个 Crop + Category + soil_id + trt 提取 ALL 作为基准
    4. 计算 (scenario - ALL) / ALL
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


def build_summary(df_rel_all):
    """
    计算每个 Target × Crop × Category × Source × trt 的 mean 和 95% CI。
    """
    group_cols = ['Target', 'Target_Label', 'Crop', 'Category', 'Source']
    if PLOT_BY_TRT:
        group_cols = ['trt'] + group_cols

    summary = (
        df_rel_all
        .groupby(group_cols)['relative_value']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )

    summary['std'] = summary['std'].fillna(0)
    summary['sem'] = summary['std'] / np.sqrt(summary['count'])
    summary['sem'] = summary['sem'].fillna(0)
    summary['ci95'] = CI_Z * summary['sem']
    summary['ci95'] = summary['ci95'].fillna(0)

    return summary


def compute_scale_abs(series):
    """
    根据一组数值计算对称范围绝对值。
    """
    s = pd.to_numeric(series, errors='coerce')
    s = s.replace([np.inf, -np.inf], np.nan).dropna()

    if s.empty:
        return BAR_SCALE_MIN_ABS

    if BAR_SCALE_METHOD == 'max_abs':
        scale = np.nanmax(np.abs(s))
    elif BAR_SCALE_METHOD == 'quantile_abs95':
        scale = np.nanquantile(np.abs(s), 0.95)
    else:
        scale = np.nanmax(np.abs(s))

    if pd.isna(scale) or scale < BAR_SCALE_MIN_ABS:
        scale = BAR_SCALE_MIN_ABS

    return float(scale)



def get_target_scale_map(summary_all, summary_current):
    """
    备用：获取每个 Target 的统一竖条缩放范围（对称）。
    现在默认不用于绘图；仅在需要输出整体 target scale 或启用旧模式时使用。
    """
    target_scale_map = {}

    for target_col in TARGET_COLS:
        if USE_GLOBAL_BAR_SCALE_ACROSS_TRT:
            sub = summary_all[summary_all['Target'] == target_col].copy()
        else:
            sub = summary_current[summary_current['Target'] == target_col].copy()

        if sub.empty:
            target_scale_map[target_col] = BAR_SCALE_MIN_ABS
            continue

        upper = sub['mean'] + sub['ci95']
        lower = sub['mean'] - sub['ci95']
        values = pd.concat([upper, lower], ignore_index=True)

        target_scale_map[target_col] = compute_scale_abs(values)

    return target_scale_map


def get_cell_scale_map(df_rel_all, df_rel_current):
    """
    获取每个迷你小提琴图自己的 y 轴范围。

    每个 cell 对应：Crop × Category(DTF/FTD) × Target。
    这里直接使用 relative_value 的原始分布，而不是先求均值后的 summary。
    这样 panel b 与单独的小提琴图使用同一批 relative_value 数据。
    """
    scale_map = {}

    source_df = df_rel_all if USE_GLOBAL_BAR_SCALE_ACROSS_TRT else df_rel_current

    for crop_name in CROP_ORDER:
        for category in CATEGORY_ORDER:
            for target_col in TARGET_COLS:

                if USE_GLOBAL_BAR_SCALE:
                    sub = source_df[
                        (source_df['Target'] == target_col) &
                        (source_df['Source'].isin(SOURCE_ORDER_PLOT))
                    ].copy()
                else:
                    sub = source_df[
                        (source_df['Crop'] == crop_name) &
                        (source_df['Category'] == category) &
                        (source_df['Target'] == target_col) &
                        (source_df['Source'].isin(SOURCE_ORDER_PLOT))
                    ].copy()

                if sub.empty:
                    scale_map[(crop_name, category, target_col)] = BAR_SCALE_MIN_ABS
                    continue

                values_for_range = []
                q = MINI_VIOLIN_QUANTILE

                for source in SOURCE_ORDER_PLOT:
                    g = pd.to_numeric(
                        sub.loc[sub['Source'] == source, 'relative_value'],
                        errors='coerce'
                    ).replace([np.inf, -np.inf], np.nan).dropna()

                    if g.empty:
                        continue

                    if q is not None and 0 < q < 0.5 and len(g) >= 5:
                        values_for_range.append(float(g.quantile(q)))
                        values_for_range.append(float(g.quantile(1 - q)))
                    else:
                        values_for_range.append(float(g.min()))
                        values_for_range.append(float(g.max()))

                if len(values_for_range) == 0:
                    scale_abs = BAR_SCALE_MIN_ABS
                else:
                    arr = pd.to_numeric(pd.Series(values_for_range), errors='coerce')
                    arr = arr.replace([np.inf, -np.inf], np.nan).dropna()
                    if arr.empty:
                        scale_abs = BAR_SCALE_MIN_ABS
                    else:
                        low = min(float(arr.min()), 0.0)
                        high = max(float(arr.max()), 0.0)
                        half_span = max(abs(low), abs(high), BAR_SCALE_MIN_ABS)
                        # 与单独小提琴图类似，围绕 0 对称，并增加少量留白。
                        scale_abs = half_span * (1.0 + 2.0 * MINI_VIOLIN_PADDING_RATIO)
                        scale_abs = max(float(scale_abs), BAR_SCALE_MIN_ABS)

                scale_map[(crop_name, category, target_col)] = scale_abs

    return scale_map


def _value_to_cell_y(value, y_center, scale_abs):
    """
    将 relative_value 映射到当前 cell 的局部 y 坐标。
    主矩阵 y 轴用于从上到下排布行，视觉方向是反的，
    所以这里使用负号，保证正值显示在上、负值显示在下。
    """
    scale_abs = max(float(scale_abs), BAR_SCALE_MIN_ABS)
    return y_center - np.clip(value / scale_abs, -1, 1) * BAR_MAX_HALF_HEIGHT


def draw_mini_zero_line(ax, x_center, y_center):
    """
    可选：绘制每个 mini violin cell 的 0 参考线。
    默认关闭，避免矩阵图横线过多。
    """
    if not SHOW_MINI_ZERO_LINE:
        return
    ax.hlines(
        y_center,
        x_center - BAR_AREA_HALF_WIDTH,
        x_center + BAR_AREA_HALF_WIDTH,
        color='0.45',
        linewidth=MINI_ZERO_LINEWIDTH,
        linestyle=(0, (2.0, 2.0)),
        zorder=1
    )


def draw_source_violin_in_cell(
    ax,
    x_center,
    y_center,
    values,
    source_name,
    source_index,
    n_sources,
    scale_abs
):
    """
    在一个 cell 里画某个 Source 的 Nature 风格迷你小提琴图。

    风格与单独小提琴图保持一致：
        - trimmed KDE，避免长尾造成针状伪影；
        - SPEI 正 / DFAI / SPEI 负三类低饱和配色；
        - 黑色细轮廓；
        - IQR 竖线 + 中位数短横线；
        - 不画均值点和误差线。
    """
    values = (
        pd.to_numeric(pd.Series(values), errors='coerce')
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if values.empty:
        return

    scale_abs = max(float(scale_abs), BAR_SCALE_MIN_ABS)

    total_width = 2 * BAR_AREA_HALF_WIDTH
    slot_w = total_width / n_sources
    max_half_width = slot_w * MINI_VIOLIN_MAX_HALF_WIDTH_RATIO

    x_left = x_center - BAR_AREA_HALF_WIDTH
    violin_x = x_left + (source_index + 0.5) * slot_w
    face_color = SOURCE_BAR_COLORS.get(source_name, '0.65')

    # 只用当前 cell 可视范围内的数据估计 KDE，避免极端长尾压扁主体形状。
    y_low_value = -scale_abs
    y_high_value = scale_abs
    values_for_kde = values[
        (values >= y_low_value) &
        (values <= y_high_value)
    ].copy()

    if values_for_kde.empty:
        values_for_kde = values.copy()

    # 数据太少或唯一值过少时，退化为 IQR/中位数摘要。
    if values_for_kde.nunique() < 2 or len(values_for_kde) < 5:
        q25 = float(values.quantile(0.25))
        q50 = float(values.quantile(0.50))
        q75 = float(values.quantile(0.75))
        y25 = _value_to_cell_y(q25, y_center, scale_abs)
        y50 = _value_to_cell_y(q50, y_center, scale_abs)
        y75 = _value_to_cell_y(q75, y_center, scale_abs)

        ax.vlines(
            violin_x,
            min(y25, y75),
            max(y25, y75),
            color='black',
            linewidth=MINI_VIOLIN_IQR_LINEWIDTH,
            zorder=6
        )
        if SHOW_MINI_VIOLIN_MEDIAN_TICK:
            tick_half = max_half_width * MINI_VIOLIN_MEDIAN_TICK_WIDTH_RATIO
            ax.hlines(
                y50,
                violin_x - tick_half,
                violin_x + tick_half,
                color='black',
                linewidth=MINI_VIOLIN_MEDIAN_TICK_LINEWIDTH,
                zorder=7
            )
        return

    try:
        from scipy.stats import gaussian_kde

        grid = np.linspace(y_low_value, y_high_value, MINI_VIOLIN_DENSITY_POINTS)
        kde = gaussian_kde(values_for_kde.to_numpy(dtype=float))
        kde.set_bandwidth(kde.factor * MINI_VIOLIN_BW_ADJUST)
        density = kde(grid)

        if np.isfinite(density).any() and np.nanmax(density) > 0:
            density = density / np.nanmax(density)
            keep = density >= MINI_VIOLIN_DENSITY_MIN_RATIO
            if not keep.any():
                keep = density > 0

            idx = np.where(keep)[0]

            if len(idx) > 0:
                # 分段绘制，避免阈值裁剪后把相隔较远的密度块硬连起来。
                breaks = np.where(np.diff(idx) > 1)[0] + 1
                chunks = np.split(idx, breaks)

                for chunk in chunks:
                    if len(chunk) < 3:
                        continue

                    gy = grid[chunk]
                    half_width = density[chunk] * max_half_width

                    # 两端强制收口，避免平顶/平底。
                    gy = np.r_[gy[0], gy, gy[-1]]
                    half_width = np.r_[0.0, half_width, 0.0]

                    y_plot = _value_to_cell_y(gy, y_center, scale_abs)
                    poly_x = np.r_[violin_x - half_width, (violin_x + half_width)[::-1]]
                    poly_y = np.r_[y_plot, y_plot[::-1]]

                    ax.fill(
                        poly_x,
                        poly_y,
                        facecolor=face_color,
                        edgecolor='black',
                        linewidth=MINI_VIOLIN_EDGE_WIDTH,
                        alpha=MINI_VIOLIN_ALPHA,
                        zorder=4
                    )
        else:
            raise RuntimeError('invalid KDE density')

    except Exception:
        # 如果 scipy 不可用，退回为轻量直方图轮廓，保证脚本仍可运行。
        hist, edges = np.histogram(
            values_for_kde.to_numpy(dtype=float),
            bins=18,
            range=(y_low_value, y_high_value),
            density=True
        )
        if hist.size > 0 and np.nanmax(hist) > 0:
            centers = (edges[:-1] + edges[1:]) / 2
            half_width = hist / np.nanmax(hist) * max_half_width
            y_plot = _value_to_cell_y(centers, y_center, scale_abs)
            poly_x = np.r_[violin_x - half_width, (violin_x + half_width)[::-1]]
            poly_y = np.r_[y_plot, y_plot[::-1]]
            ax.fill(
                poly_x,
                poly_y,
                facecolor=face_color,
                edgecolor='black',
                linewidth=MINI_VIOLIN_EDGE_WIDTH,
                alpha=MINI_VIOLIN_ALPHA,
                zorder=4
            )

    # 叠加 Nature 风格摘要：IQR 竖线 + 中位数短横线。
    q25 = float(values.quantile(0.25))
    q50 = float(values.quantile(0.50))
    q75 = float(values.quantile(0.75))

    y25 = _value_to_cell_y(q25, y_center, scale_abs)
    y50 = _value_to_cell_y(q50, y_center, scale_abs)
    y75 = _value_to_cell_y(q75, y_center, scale_abs)

    if SHOW_MINI_VIOLIN_IQR:
        ax.vlines(
            violin_x,
            min(y25, y75),
            max(y25, y75),
            color='black',
            linewidth=MINI_VIOLIN_IQR_LINEWIDTH,
            zorder=6
        )

    if SHOW_MINI_VIOLIN_MEDIAN_TICK:
        tick_half = max_half_width * MINI_VIOLIN_MEDIAN_TICK_WIDTH_RATIO
        ax.hlines(
            y50,
            violin_x - tick_half,
            violin_x + tick_half,
            color='black',
            linewidth=MINI_VIOLIN_MEDIAN_TICK_LINEWIDTH,
            zorder=7
        )


def add_top_colgroup_three_line_table(ax, x_min, x_max, y_min):
    """
    在主图上方绘制列分组三线表。
    """
    if not SHOW_TOP_COLGROUP_TABLE:
        return

    x_left = x_min - 0.5
    x_right = x_max + 0.5

    y_top = y_min - TOP_TABLE_TOP_OFFSET
    y_group = y_min - TOP_TABLE_HEADER_OFFSET
    y_content = y_min - TOP_TABLE_CONTENT_OFFSET
    y_bottom = y_min - TOP_TABLE_BOTTOM_OFFSET

    # 不再绘制顶部表格横线，只保留组名和指标名文字。

    for group_name, col_list in COL_GROUPS:
        group_items = [item for item in COL_ITEMS if item['Group'] == group_name]
        if len(group_items) == 0:
            continue

        group_center_x = (
            min(item['x'] for item in group_items)
            + max(item['x'] for item in group_items)
        ) / 2

        ax.text(
            group_center_x,
            y_group,
            group_name,
            fontsize=COL_GROUP_LABEL_SIZE,
            fontweight='bold',
            ha='center',
            va='center',
            clip_on=False,
            zorder=11
        )

    for item in COL_ITEMS:
        ax.text(
            item['x'],
            y_content,
            item['Col_Label'],
            fontsize=COL_LABEL_SIZE,
            ha='center',
            va='center',
            clip_on=False,
            zorder=11
        )


def add_crop_legend(fig):
    """
    添加 4 个作物的颜色图例。
    """
    handles = []
    for crop_name in ACTIVE_CROP_ORDER:
        handles.append(
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=CROP_COLORS.get(crop_name, '0.6'),
                edgecolor='black',
                linewidth=0.5,
                label=crop_name
            )
        )

    fig.legend(
        handles=handles,
        loc='upper right',
        bbox_to_anchor=(0.985, 0.985),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        ncol=1
    )


def add_scale_note(fig, summary_all, summary_df, target_scale_map):
    """
    右上角增加 bar 高度缩放说明。
    """
    # 输出一个简单文字说明，而不是原来的复杂色标表
    lines = ['Bar height scale']
    for target_col in TARGET_COLS:
        if target_col not in target_scale_map:
            continue
        label = PARAM_LABELS.get(target_col, target_col).replace('\n', ' ')
        scale_abs = target_scale_map[target_col]
        if scale_abs < 0.001:
            txt = f'{label}: ±{scale_abs:.4f}'
        elif scale_abs < 0.1:
            txt = f'{label}: ±{scale_abs:.3f}'
        elif scale_abs < 1:
            txt = f'{label}: ±{scale_abs:.2f}'
        else:
            txt = f'{label}: ±{scale_abs:.1f}'
        lines.append(txt)

    note = '\n'.join(lines)

    fig.text(
        0.985,
        0.78,
        note,
        ha='right',
        va='top',
        fontsize=5.2,
        linespacing=1.15
    )




def format_cell_scale_label(scale_abs):
    """
    每个小条形图左侧的 y 轴范围标签。
    显示为 ±x，表示该 cell 的对称范围 [-x, +x]。
    """
    x = abs(float(scale_abs))
    if x < 0.001:
        return f'±{x:.4f}'
    elif x < 0.01:
        return f'±{x:.3f}'
    elif x < 0.1:
        return f'±{x:.2f}'
    elif x < 1:
        return f'±{x:.2f}'
    else:
        return f'±{x:.1f}'


def get_bar_x_position(x_center, source_index, n_sources):
    """
    返回 cell 内第 source_index 根柱子的 x 坐标，
    与 draw_source_bar_in_cell() 内部布局保持一致。
    """
    total_width = 2 * BAR_AREA_HALF_WIDTH
    slot_w = total_width / n_sources
    x_left = x_center - BAR_AREA_HALF_WIDTH
    return x_left + (source_index + 0.5) * slot_w


def add_cell_y_range_label(ax, x_center, y_center, scale_abs):
    """
    在每个小条形图左侧标注该 cell 的 y 轴范围。
    """
    if not SHOW_CELL_Y_RANGE_LABEL:
        return
    ax.text(
        x_center - BAR_AREA_HALF_WIDTH - CELL_Y_RANGE_LABEL_X_OFFSET,
        y_center,
        format_cell_scale_label(scale_abs),
        ha='right',
        va='center',
        fontsize=CELL_Y_RANGE_LABEL_SIZE,
        color='0.48',
        clip_on=False,
        zorder=20
    )


def add_bottom_source_x_labels(ax, x_center, y_center, category):
    """
    只在最下面一行小条形图下面标注 6 根柱子的 x 轴标签。
    """
    if not SHOW_BOTTOM_SOURCE_X_LABELS:
        return

    n_sources = len(SOURCE_ORDER_PLOT)
    label_y = y_center + CELL_BOX_HALF_HEIGHT + BOTTOM_SOURCE_X_LABEL_Y_OFFSET

    for source_index, source in enumerate(SOURCE_ORDER_PLOT):
        bar_x = get_bar_x_position(x_center, source_index, n_sources)
        label = get_source_label(category, source)
        ax.text(
            bar_x,
            label_y,
            label,
            ha='center',
            va='top',
            rotation=BOTTOM_SOURCE_X_LABEL_ROTATION,
            fontsize=BOTTOM_SOURCE_X_LABEL_SIZE,
            color='0.25',
            clip_on=False,
            zorder=20
        )

def draw_matrix(df_rel_all, df_rel_current, out_base, title_text=None):
    """
    绘制“作物为一级分组”的紧凑型 mini violin 矩阵图。

    布局：
        左侧一级分组 = Crop
        每个作物两行 = DTF / FTD
        每一列 = 一个 Target
        每个 cell = 当前 Crop × Category × Target 的一个迷你小提琴图组合
        每个 cell 内 6 个 violin = 6 个 Source

    与单独小提琴图保持一致：这里直接使用 relative_value 的原始分布，
    不再先汇总成 mean 后画柱子。
    """

    x_values = [item['x'] for item in COL_ITEMS]
    y_values = [item['y'] for item in ROW_ITEMS]

    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)

    n_visual_cols = len(COL_ITEMS)
    n_visual_rows = len(ROW_ITEMS)

    fig_width = max(6.3, n_visual_cols * FIG_WIDTH_PER_COL + FIG_EXTRA_W)
    fig_height = max(5.3, n_visual_rows * FIG_HEIGHT_PER_ROW + FIG_EXTRA_H)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cell_scale_map = get_cell_scale_map(
        df_rel_all=df_rel_all,
        df_rel_current=df_rel_current
    )

    # 保存当前图每个小提琴 cell 自己的 y 轴范围，方便检查。
    scale_records = []
    for (crop_name, category, target_col), scale_abs in cell_scale_map.items():
        scale_records.append({
            'Crop': crop_name,
            'Category': category,
            'Target': target_col,
            'Label': PARAM_LABELS.get(target_col, target_col).replace('\n', ' '),
            'scale_abs': scale_abs,
            'vmin': -scale_abs,
            'vcenter': 0,
            'vmax': scale_abs,
            'scale_method': 'cellwise_distribution_quantile'
        })
    pd.DataFrame(scale_records).to_csv(
        out_base + '_cell_violin_y_range.csv',
        index=False,
        encoding='utf-8-sig'
    )

    # 保存 DTF / FTD 中 6 个小提琴的顺序。
    source_order_records = []
    for category in CATEGORY_ORDER:
        for i, source in enumerate(SOURCE_ORDER_PLOT, start=1):
            source_order_records.append({
                'Category': category,
                'violin_index_in_each_cell': i,
                'Source': source,
                'Display_Label': get_source_label(category, source),
                'Color': SOURCE_BAR_COLORS.get(source, '')
            })
    pd.DataFrame(source_order_records).to_csv(
        out_base + '_source_violin_order.csv',
        index=False,
        encoding='utf-8-sig'
    )

    # ========================================================
    # 每个 cell 画 6 个 Source violin
    # ========================================================
    n_sources = len(SOURCE_ORDER_PLOT)

    for row_item in ROW_ITEMS:
        crop_name = row_item['Crop']
        category = row_item['Category']
        row_y = row_item['y']

        for col_item in COL_ITEMS:
            target_col = col_item['Target']
            col_x = col_item['x']
            scale_abs = cell_scale_map.get((crop_name, category, target_col), BAR_SCALE_MIN_ABS)

            add_cell_y_range_label(ax, col_x, row_y, scale_abs)
            draw_mini_zero_line(ax, col_x, row_y)

            for source_index, source in enumerate(SOURCE_ORDER_PLOT):
                values = df_rel_current.loc[
                    (df_rel_current['Crop'] == crop_name) &
                    (df_rel_current['Category'] == category) &
                    (df_rel_current['Source'] == source) &
                    (df_rel_current['Target'] == target_col),
                    'relative_value'
                ]

                if values.empty:
                    continue

                draw_source_violin_in_cell(
                    ax=ax,
                    x_center=col_x,
                    y_center=row_y,
                    values=values,
                    source_name=source,
                    source_index=source_index,
                    n_sources=n_sources,
                    scale_abs=scale_abs
                )

            # 只在最下面一行图标注 x 轴标签。
            if abs(row_y - y_max) < EPS:
                add_bottom_source_x_labels(ax, col_x, row_y, category)

    # ========================================================
    # 坐标轴设置
    # ========================================================
    ax.set_xlim(x_min - 1.18, x_max + 0.34)

    if SHOW_TOP_COLGROUP_TABLE:
        ax.set_ylim(y_max + 0.96, y_min - TOP_TABLE_YLIM_OFFSET)
    else:
        ax.set_ylim(y_max + 0.96, y_min - 0.55)

    ax.set_aspect('auto')

    ax.set_xticks([item['x'] for item in COL_ITEMS])
    ax.set_xticklabels([])
    ax.tick_params(axis='x', length=0, labelbottom=False, labeltop=False)

    # 不使用默认 ytick 标签，改为手动放置 DTF / FTD。
    ax.set_yticks([])
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', length=0)

    for row_item in ROW_ITEMS:
        ax.text(
            x_min - LEFT_ROW_LABEL_X_OFFSET,
            row_item['y'],
            row_item['Row_Label'],
            fontsize=ROW_LABEL_SIZE,
            fontweight='bold',
            ha='right',
            va='center',
            color='0.10',
            clip_on=False,
            zorder=25
        )

    # 面板标记 b
    panel_b_y = y_min - TOP_TABLE_YLIM_OFFSET + 0.06 if SHOW_TOP_COLGROUP_TABLE else y_min - 0.65
    ax.text(
        x_min - PANEL_B_LABEL_X_OFFSET,
        panel_b_y,
        'b',
        ha='left',
        va='top',
        fontsize=8,
        fontweight='bold',
        color='0.05',
        clip_on=False,
        zorder=30
    )

    # 作物标签
    for crop_name in CROP_ORDER:
        crop_items = [item for item in ROW_ITEMS if item['Crop'] == crop_name]
        if not crop_items:
            continue

        crop_y_min = min(item['y'] for item in crop_items)
        crop_y_max = max(item['y'] for item in crop_items)
        crop_y_center = (crop_y_min + crop_y_max) / 2

        ax.text(
            x_min - LEFT_CROP_LABEL_X_OFFSET,
            crop_y_center,
            crop_name,
            fontsize=GROUP_LABEL_SIZE,
            fontweight='bold',
            ha='center',
            va='center',
            rotation=90,
            color=CROP_COLORS.get(crop_name, 'black'),
            clip_on=False
        )

    add_top_colgroup_three_line_table(ax, x_min, x_max, y_min)

    if title_text and SHOW_TITLE:
        fig.suptitle(title_text, fontsize=7.3, y=0.985)

    if SHOW_CROP_LEGEND:
        add_crop_legend(fig)

    if SHOW_SOURCE_ORDER_NOTE:
        dtf_order = ', '.join([get_source_label('DTF', s) for s in SOURCE_ORDER_PLOT])
        ftd_order = ', '.join([get_source_label('FTD', s) for s in SOURCE_ORDER_PLOT])
        fig.text(
            0.145,
            0.012,
            f'Violin order in each cell | DTF: {dtf_order} | FTD: {ftd_order}',
            ha='left',
            va='bottom',
            fontsize=4.7
        )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    fig.subplots_adjust(
        left=0.055,
        right=0.998,
        bottom=0.030,
        top=0.970
    )

    png_path = out_base + '.png'
    pdf_path = out_base + '.pdf'

    fig.savefig(
        png_path,
        dpi=OUTPUT_DPI,
        bbox_inches='tight',
        pad_inches=0.010,
        facecolor='white'
    )
    fig.savefig(
        pdf_path,
        dpi=OUTPUT_DPI,
        bbox_inches='tight',
        pad_inches=0.010,
        facecolor='white'
    )

    plt.close(fig)

    print(f'✅ 小图 b 已保存 PNG：{png_path}')
    print(f'✅ 小图 b 已保存 PDF：{pdf_path}')

    return {
        'png': png_path,
        'pdf': pdf_path
    }


def load_all_crop_data():
    global ACTIVE_CROP_ORDER

    df_crop_category_list = []

    for crop_name in CROP_ORDER:
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
        raise ValueError('❌ 没有读取到任何数据，请检查作物文件路径。')

    df_all = pd.concat(df_crop_category_list, ignore_index=True)

    if df_all.empty:
        raise ValueError('❌ 没有读取到任何数据，请检查文件路径。')

    for required_col in ['Crop', 'Category', 'Source', 'trt', 'soil_id']:
        if required_col not in df_all.columns:
            raise ValueError(f'❌ 数据中没有 {required_col} 列。')

    loaded_crop_values = set(df_all['Crop'].dropna().astype(str).unique())
    ACTIVE_CROP_ORDER = [crop for crop in CROP_ORDER if crop in loaded_crop_values]
    missing_crops = [crop for crop in CROP_ORDER if crop not in ACTIVE_CROP_ORDER]

    print('\n==============================')
    print('✅ 数据读取完成')
    print('读取到的作物：')
    print(df_all['Crop'].value_counts())
    print('读取到的 Category：')
    print(df_all['Category'].value_counts())
    print('读取到的 Source：')
    print(df_all['Source'].value_counts())
    print('==============================\n')

    print(f'✅ 实际参与绘图的作物顺序：{ACTIVE_CROP_ORDER}')

    if missing_crops:
        print(f'⚠️ 未读取到的作物：{missing_crops}')
        if REQUIRE_ALL_CROPS:
            raise ValueError(
                '❌ 以下作物没有被成功读取：'
                + ', '.join(missing_crops)
                + '。请检查对应路径是否正确。'
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


def generate_all_matrix_plots(df_rel_all, panel_a_paths):
    """
    如果 PLOT_BY_TRT = True，则每个 trt 输出：
        1) 小图 b：mini violin 矩阵图；
        2) 合并图：小图 a + 小图 b。
    否则输出一张合并所有 trt 的总图。

    注意：默认 PLOT_BY_TRT=False，是为了让 panel b 与单独的小提琴图保持同一层级，
    即不按 trt 单独过滤。
    """
    combined_paths = []

    if PLOT_BY_TRT:
        if 'trt' not in df_rel_all.columns:
            raise ValueError('❌ df_rel_all 中没有 trt 列，但 PLOT_BY_TRT=True。')

        trt_values = sorted(df_rel_all['trt'].dropna().unique(), key=natural_sort_key)

        if len(trt_values) == 0:
            raise ValueError('❌ 没有可用的 trt。')

        for trt_value in trt_values:
            df_rel_current = df_rel_all[df_rel_all['trt'] == trt_value].copy()

            if df_rel_current.empty:
                continue

            safe_trt = re.sub(r'[^0-9A-Za-z_+\-.]+', '_', str(trt_value)).strip('_')
            out_base_b = os.path.join(
                OUTPUT_DIR,
                f'{PANEL_B_STEM_PREFIX}_{safe_trt}'
            )

            panel_b_paths = draw_matrix(
                df_rel_all=df_rel_all,
                df_rel_current=df_rel_current,
                out_base=out_base_b,
                title_text=f'Mini violin matrix | trt = {trt_value}'
            )

            out_base_combined = os.path.join(
                OUTPUT_DIR,
                f'{COMBINED_STEM_PREFIX}_{safe_trt}'
            )

            combined_paths.append(
                merge_panel_images(
                    panel_a_png=panel_a_paths['png'],
                    panel_b_png=panel_b_paths['png'],
                    out_base=out_base_combined
                )
            )
    else:
        out_base_b = os.path.join(OUTPUT_DIR, f'{PANEL_B_STEM_PREFIX}_all')
        panel_b_paths = draw_matrix(
            df_rel_all=df_rel_all,
            df_rel_current=df_rel_all.copy(),
            out_base=out_base_b,
            title_text='Mini violin matrix'
        )

        out_base_combined = os.path.join(OUTPUT_DIR, f'{COMBINED_STEM_PREFIX}_all')
        combined_paths.append(
            merge_panel_images(
                panel_a_png=panel_a_paths['png'],
                panel_b_png=panel_b_paths['png'],
                out_base=out_base_combined
            )
        )

    return combined_paths

def main():
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': FONT_FAMILY,
        'font.sans-serif': [FONT_FAMILY, 'Arial', 'DejaVu Sans'],
        'font.size': BASE_FONT_SIZE,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })

    # 先生成小图 a。
    panel_a_paths = draw_panel_a_map(OUTPUT_DIR)

    # 再计算相对变化数据并生成 mini violin 矩阵小图 b。
    df_all = load_all_crop_data()
    df_rel_all, available_targets = compute_relative_for_all_targets(df_all)

    relative_csv = os.path.join(OUTPUT_DIR, 'relative_values_long_table.csv')
    df_rel_all.to_csv(relative_csv, index=False, encoding='utf-8-sig')
    print(f'✅ 相对变化长表已保存：{relative_csv}')

    summary_all = build_summary(df_rel_all)
    summary_csv = os.path.join(OUTPUT_DIR, 'summary_mean_ci95.csv')
    summary_all.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    print(f'✅ 汇总表已保存：{summary_csv}')

    combined_paths = generate_all_matrix_plots(df_rel_all, panel_a_paths)

    print('\n🎉 全部完成！')
    print(f'✅ 合并总图数量：{len(combined_paths)}')


if __name__ == '__main__':
    main()