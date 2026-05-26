import pandas as pd
import numpy as np
import os
import glob
import re
import json
import hashlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# 1. 路径配置
# ============================================================

# 多作物配置。
# 每个作物都需要分别指定 DFAI / DFAA < -4 与 > 4 对应的 DSSAT 结果目录。
#
# 说明：
# - crop_name 会用于输出目录、输出文件名和最终合并图的列标题；
# - negative_dssat_dir 用于 DFAI / DFAA < -4；
# - positive_dssat_dir 用于 DFAI / DFAA > 4；
# - 如果你的玉米目录仍然在 D:\Soybeans 下面，请把 Maize 两个路径改成实际目录。
CROP_CONFIGS = [
    {
        "crop_name": "Maize",
        "negative_dssat_dir": r"D:\Maize\DSSAT_DAILY",
        "positive_dssat_dir": r"D:\Maize\DSSAT_DAILY_+",
        # 单位通常为 kg/ha 和 days；请按你的 DSSAT 输出单位与研究区实际情况调整。
        "crop_qc_thresholds": {
            "MIN_FINAL_YIELD": 1000,
            "MAX_FINAL_YIELD": 20000,
            "MIN_CROP_DURATION_DAYS": 90,
            "MAX_CROP_DURATION_DAYS": 240,
        },
    },
    {
        "crop_name": "Rice",
        "negative_dssat_dir": r"E:\yun\pythonProject2\Rice\DSSAT_DAILY",
        "positive_dssat_dir": r"E:\yun\pythonProject2\Rice\DSSAT_DAILY_+",
        "crop_qc_thresholds": {
            "MIN_FINAL_YIELD": 1000,
            "MAX_FINAL_YIELD": 20000,
            "MIN_CROP_DURATION_DAYS": 90,
            "MAX_CROP_DURATION_DAYS": 240,
        },
    },
    {
        "crop_name": "Soybeans",
        "negative_dssat_dir": r"D:\Soybeans\DSSAT_DAILY",
        "positive_dssat_dir": r"D:\Soybeans\DSSAT_DAILY_+",
        "crop_qc_thresholds": {
            "MIN_FINAL_YIELD": 1000,
            "MAX_FINAL_YIELD": 20000,
            "MIN_CROP_DURATION_DAYS": 70,
            "MAX_CROP_DURATION_DAYS": 240,
        },
    },
    {
        "crop_name": "Wheat",
        "negative_dssat_dir": r"E:\yun\pythonProject2\Wheat\DSSAT_DAILY",
        "positive_dssat_dir": r"E:\yun\pythonProject2\Wheat\DSSAT_DAILY_+",
        "crop_qc_thresholds": {
            "MIN_FINAL_YIELD": 1000,
            "MAX_FINAL_YIELD": 20000,
            "MIN_CROP_DURATION_DAYS": 80,
            "MAX_CROP_DURATION_DAYS": 240,
        },
    },
]

# 保留旧变量名，供已有函数内部使用；主流程会按作物和出图任务自动切换。
DSSAT_DIR_NEGATIVE_DFAI = CROP_CONFIGS[0]["negative_dssat_dir"]
DSSAT_DIR_POSITIVE_DFAI = CROP_CONFIGS[0]["positive_dssat_dir"]
dssat_dir = DSSAT_DIR_NEGATIVE_DFAI

climate_dir = r"D:\SPEI_LDFAI"

# 所有输出的总目录。
BASE_OUTPUT_DIR = r"C:\Users\lxh\Desktop\NongZuoWu_ChanLiang\Figure\test1"

# 保留旧变量名，供已有缓存、保存和绘图函数使用；后续主流程会按任务自动更新。
output_dir = BASE_OUTPUT_DIR
fig_dir = os.path.join(output_dir, "figures")
comparison_fig_dir = os.path.join(fig_dir, "event_minus_normal_full_early_middle_late_TRT1_only")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(comparison_fig_dir, exist_ok=True)

# ============================================================
# 1.1 处理结果缓存配置
# ============================================================
# True：第一次完整读取 DSSAT 和气象数据并完成事件识别后，保存处理后的结果；
#       后续如果关键参数和输入文件没有变化，就直接读取缓存并绘图。
# False：每次都重新读取原始 DSSAT 和气象文件。
ENABLE_PROCESSED_DATA_CACHE = True

# True：忽略已有缓存，强制重新读取原始文件并重建缓存。
FORCE_REBUILD_PROCESSED_DATA_CACHE = False

# 缓存目录。
cache_dir = os.path.join(output_dir, "processed_cache")
os.makedirs(cache_dir, exist_ok=True)

# 缓存版本。
# 当事件识别、QC、物候阶段或 event_daily 构建逻辑发生较大变化时，建议手动改这个版本号，
# 这样旧缓存不会被误用。
PROCESSED_CACHE_VERSION = "v2026_05_19_four_crops_2x2_crop_specific_qc_wheat_doy_repair_yield_alias_fast_v2"


# ============================================================
# 1.2 运行速度优化开关
# ============================================================
# True：读取 DSSAT daily CSV 时只读取后续分析真正需要的列。
# 对大型 daily 输出文件，这是最直接的 I/O 和内存优化。
USE_MINIMAL_DSSAT_COLUMNS = True

# True：DFAA 60 天覆盖窗口用差分数组一次性计算，替代逐窗口逐日期循环。
FAST_DFAA_WINDOW_COVERAGE = True

# True：event_daily 只保留“站点-年份-情景-事件类型-发生阶段”这一绘图独立样本单位的一份完整曲线。
# 原代码会对同一类型同一阶段的多个事件重复复制完整 early+middle+late 曲线；
# 但后续作图本来会按 full_period_event_type_unit_id 先求均值，因此这些重复行对曲线结果没有贡献。
# 如果你确实需要逐事件级 event_daily 明细，可改为 False。
DEDUPLICATE_FULL_RESPONSE_EVENT_DAILY_BY_UNIT = True

# True：缓存签名只扫描当前 DSSAT 文件清单涉及到的站点气象文件，而不是 climate_dir 下所有站点。
# 若你希望任何气象文件变化都使所有运行缓存失效，可改为 False。
CACHE_SIGNATURE_ONLY_CURRENT_SITES = True

# 同一次运行中重复生成文件签名时复用 os.stat 结果。
_FILE_SIGNATURE_CACHE = {}


# ============================================================
# 1.3 DSSAT daily 原始 CSV 预合并缓存配置
# ============================================================
# 作用：
# - 第一次处理某个 作物 × 正/负 DFAI 目录 时，把该目录下目标情景的 daily CSV
#   一次性读取、清洗、合并为一个 pickle 缓存；
# - 后续 process_one_sample() 仍按单个 fp 取样本，但不再反复 pd.read_csv 小文件；
# - 如果 ENABLE_PROCESSED_DATA_CACHE 已经命中，则不会加载这个 raw cache。
ENABLE_DSSAT_MERGED_RAW_CACHE = True

# True：忽略已有 DSSAT daily 预合并缓存，强制重新合并原始 CSV。
FORCE_REBUILD_DSSAT_MERGED_RAW_CACHE = False

# 当前实现使用 pickle，不需要额外安装 pyarrow / fastparquet。
DSSAT_MERGED_RAW_CACHE_FORMAT = "pkl"

# 当前正在使用的 DSSAT daily 预合并缓存。
_CURRENT_DSSAT_MERGED_RAW_DF = None
_CURRENT_DSSAT_MERGED_RAW_DIR = None
_CURRENT_DSSAT_MERGED_RAW_GROUP_INDICES = None

# 这些列只用于从预合并缓存中恢复每个样本的 attrs，不参与后续分析。
DSSAT_MERGED_RAW_INTERNAL_COLS = [
    "_source_fp_norm",
    "_dssat_raw_rows",
    "_dssat_clean_rows",
    "_dssat_preprocess_mode",
    "_dssat_year_doy_groups_collapsed",
    "_dssat_rows_after_year_doy_collapse",
    "_dssat_empty_crop_rows_removed",
    "_dssat_variable_aliases_applied",
]



# ============================================================
# 2. SPEI / LDFAI 阈值配置
# ============================================================

# 干旱、洪涝阈值
SPEI_DROUGHT_THRESHOLD = -1
SPEI_FLOOD_THRESHOLD = 1

# DFAA 阈值
LDFAI_POS_THRESHOLD = 4.0
LDFAI_NEG_THRESHOLD = -4.0

# LDFAI 是 60 天滑动窗口
LDFAI_WINDOW_DAYS = 60

# 连续 DFAA 窗口合并时允许的间断天数
LDFAI_MERGE_GAP_DAYS = 3

# 连续 SPEI 干旱/洪涝合并时允许的间断天数
SPEI_MERGE_GAP_DAYS = 3

# SPEI 干旱/洪涝事件最少持续天数
SPEI_MIN_EVENT_DAYS = 3

# 是否把和 DFAA 60 天窗口重叠的 SPEI 干旱/洪涝排除
# True 更适合做“纯干旱/纯洪涝”和 DFAA 的对比
EXCLUDE_DFAA_WINDOW_FOR_SPEI_EVENTS = True

# 正常事件最少连续天数
NORMAL_MIN_SEGMENT_DAYS = 10

# 每个 站点-年份-TRT-物候阶段 最多选几个正常事件
NORMAL_MAX_EVENTS_PER_STAGE_PER_SAMPLE = 1

# 事件响应图前后窗口
EVENT_PLOT_BEFORE_DAYS = 60
EVENT_PLOT_AFTER_DAYS = 60


# ============================================================
# 3. 作物模拟质量筛选阈值
# ============================================================

ENABLE_CROP_QC_FILTER = True

# 默认 QC 阈值。每个作物会优先读取 CROP_CONFIGS 中的 crop_qc_thresholds；
# 如果某个作物没有单独设置某项阈值，就回退到这里的默认值。
DEFAULT_CROP_QC_THRESHOLDS = {
    "MIN_FINAL_YIELD": 3000,
    "MAX_FINAL_YIELD": 20000,
    "MIN_CROP_DURATION_DAYS": 100,
    "MAX_CROP_DURATION_DAYS": 220,
}

# 当前正在处理的作物及其 QC 阈值。
# 注意：主流程会在进入每个 crop_cfg 时自动切换这些全局变量，
# summarize_crop_quality() 和缓存参数都会使用切换后的阈值。
CURRENT_CROP_NAME = str(CROP_CONFIGS[0].get("crop_name", "unknown_crop"))

MIN_FINAL_YIELD = DEFAULT_CROP_QC_THRESHOLDS["MIN_FINAL_YIELD"]
MAX_FINAL_YIELD = DEFAULT_CROP_QC_THRESHOLDS["MAX_FINAL_YIELD"]

MIN_CROP_DURATION_DAYS = DEFAULT_CROP_QC_THRESHOLDS["MIN_CROP_DURATION_DAYS"]
MAX_CROP_DURATION_DAYS = DEFAULT_CROP_QC_THRESHOLDS["MAX_CROP_DURATION_DAYS"]

# 下面这些 QC 条件目前仍为所有作物共用；如果后续也需要分作物，
# 可以用同样方式加入 crop_qc_thresholds。
MIN_VALID_CROP_DAYS = 30

MIN_MAX_LAI = 0.1
MAX_MAX_LAI = 15

MIN_MAX_CWAD = 500

MIN_GSTD_STAGE_COUNT = 3

MIN_CLIMATE_MATCH_RATIO = 0.80


# ============================================================
# 3.1 DSSAT Wheat 读取清洗：YEAR / DOY / DAS 拆行修复
# ============================================================
# Wheat 的 DSSAT daily CSV 中可能出现同一 YEAR-DOY 两行：
# - 一行包含 PlantGro_* 生长变量，但 DATE 为空；
# - 另一行包含 Soil / Weather / DATE，但 PlantGro_* 为空；
# 同时 DAS 可能与 PlantGro_DAP 不完全同步。
# 如果直接 dropna(DATE)，会保留大量 PlantGro 为空的行，导致后续物候和 QC 出错。
ENABLE_WHEAT_DOY_DAS_SPLIT_ROW_REPAIR = True
WHEAT_DOY_DAS_SPLIT_ROW_REPAIR_CROP_NAMES = {"wheat"}
DROP_WHEAT_ROWS_WITHOUT_PLANTGRO_STATE = True

# 用于判断某行是否真的包含作物生长状态。
# 只判断非缺失，不判断是否为 0；因为播种初期的 0 是有效模拟值。
DSSAT_REPAIR_KEY_PLANTGRO_COLS = [
    "PlantGro_DAP",
    "PlantGro_GSTD",
    "PlantGro_LAID",
    "PlantGro_CWAD",
    # Wheat 往往没有 PlantGro_GWAD，而是 PlantGro_HWAD / PlantGro_CHWAD；
    # 放进 key 列后，拆行修复时不会把有效 Wheat 生长行误判为空行。
    "PlantGro_GWAD",
    "PlantGro_HWAD",
    "PlantGro_CHWAD",
]

# ============================================================
# 3.2 作物专用产量列与变量别名
# ============================================================
# 注意：
# - Maize / Rice / Soybeans 的 grain weight 通常可以直接使用 PlantGro_GWAD；
# - Wheat 的 DSSAT daily 输出里常见的是 PlantGro_HWAD，而不是 PlantGro_GWAD；
#   如果仍用 PlantGro_GWAD 做 QC，就会得到 missing_final_yield，进而 crop_qc_failed。
# - 这里把 Wheat 的 PlantGro_HWAD 复制成统一的 PlantGro_GWAD，既用于 QC，也用于后续绘图。
DEFAULT_FINAL_YIELD_COLUMNS = ["PlantGro_GWAD"]

CROP_SPECIFIC_FINAL_YIELD_COLUMNS = {
    "wheat": ["PlantGro_GWAD", "PlantGro_HWAD", "PlantGro_CHWAD"],
}

CROP_SPECIFIC_VARIABLE_ALIASES = {
    "wheat": {
        # Grain / harvest weight: Wheat 输出常用 HWAD / CHWAD，不一定有 GWAD。
        "PlantGro_GWAD": ["PlantGro_HWAD", "PlantGro_CHWAD"],

        # Water stress: Wheat 输出常用 WFPD / WFGD，而不是 WSPD / WSGD。
        # WFPD: photosynthesis-related water stress factor -> 统一成 WSPD。
        # WFGD: growth-related water stress factor -> 统一成 WSGD。
        "PlantGro_WSPD": ["PlantGro_WFPD"],
        "PlantGro_WSGD": ["PlantGro_WFGD"],

        # 可选环境/水分胁迫变量；当前 PLOT_VARIABLES 里 EWSD 已注释，
        # 但保留别名，后续打开 EWSD 时 Wheat 也不会缺列。
        "PlantGro_EWSD": ["PlantGro_WFTD"],
    },
}


def get_crop_qc_thresholds(crop_cfg):
    """返回某个作物实际使用的 QC 阈值。"""

    thresholds = DEFAULT_CROP_QC_THRESHOLDS.copy()
    thresholds.update(crop_cfg.get("crop_qc_thresholds", {}) or {})

    required_keys = [
        "MIN_FINAL_YIELD",
        "MAX_FINAL_YIELD",
        "MIN_CROP_DURATION_DAYS",
        "MAX_CROP_DURATION_DAYS",
    ]

    for key in required_keys:
        thresholds[key] = float(thresholds[key])

    if thresholds["MIN_FINAL_YIELD"] >= thresholds["MAX_FINAL_YIELD"]:
        raise ValueError(
            f"{crop_cfg.get('crop_name', 'unknown_crop')} 的产量阈值设置错误："
            "MIN_FINAL_YIELD 必须小于 MAX_FINAL_YIELD。"
        )

    if thresholds["MIN_CROP_DURATION_DAYS"] >= thresholds["MAX_CROP_DURATION_DAYS"]:
        raise ValueError(
            f"{crop_cfg.get('crop_name', 'unknown_crop')} 的生育期阈值设置错误："
            "MIN_CROP_DURATION_DAYS 必须小于 MAX_CROP_DURATION_DAYS。"
        )

    return thresholds


def apply_crop_qc_thresholds_for_crop(crop_cfg):
    """把当前作物的 QC 阈值写入旧函数仍在使用的全局变量。"""

    global CURRENT_CROP_NAME
    global MIN_FINAL_YIELD, MAX_FINAL_YIELD
    global MIN_CROP_DURATION_DAYS, MAX_CROP_DURATION_DAYS

    CURRENT_CROP_NAME = str(crop_cfg.get("crop_name", "unknown_crop"))
    thresholds = get_crop_qc_thresholds(crop_cfg)

    MIN_FINAL_YIELD = thresholds["MIN_FINAL_YIELD"]
    MAX_FINAL_YIELD = thresholds["MAX_FINAL_YIELD"]
    MIN_CROP_DURATION_DAYS = thresholds["MIN_CROP_DURATION_DAYS"]
    MAX_CROP_DURATION_DAYS = thresholds["MAX_CROP_DURATION_DAYS"]

    print(
        "  当前作物 QC 阈值："
        f"yield=[{MIN_FINAL_YIELD:g}, {MAX_FINAL_YIELD:g}], "
        f"crop_duration_days=[{MIN_CROP_DURATION_DAYS:g}, {MAX_CROP_DURATION_DAYS:g}]"
    )

    return thresholds


# ============================================================
# 4. 绘图配置
# ============================================================

# 只生成事件类型对比图；不再生成单独事件类型响应图
ENABLE_COMPARISON_PLOTTING = True

# 只分析指定情景。当前仅分析 TRT1，其他 TRT 情景不读取、不处理、不绘图。
ANALYZE_SCENARIO = "TRT1"

# ============================================================
# 4.0 事件阶段筛选 + 完整前中后时期响应横轴配置
# ============================================================
# 先筛选极端事件发生在哪个阶段：前 / 中 / 后。
# 之后不只画事件所在阶段，而是提取同一样本的完整“前 + 中 + 后”生育时期。
# x = 0%   表示 early_stage 开始；
# x = 100% 表示 late_stage 结束。
# 每张最终图中三条线分别表示：事件发生在前期 / 中期 / 后期。
EVENT_OCCURRENCE_STAGES = [
    "early_stage",
    "middle_stage",
    "late_stage"
]

EVENT_OCCURRENCE_STAGE_LABELS = {
    "early_stage": "Occurred in early stage",
    "middle_stage": "Occurred in middle stage",
    "late_stage": "Occurred in late stage"
}

# 完整响应窗口：只画 early + middle + late，不包括 maturity_stage。
FULL_RESPONSE_STAGES = [
    "early_stage",
    "middle_stage",
    "late_stage"
]

FULL_RESPONSE_X_MODE = "progress_percent"     # 当前固定推荐使用百分比横轴
FULL_RESPONSE_BIN_WIDTH_PCT = 1.0              # 每 1% 汇总一次
FULL_RESPONSE_X_COL = "full_period_progress_bin"
FULL_RESPONSE_UNIT_COL = "full_period_event_type_unit_id"

# 保留旧变量名，避免后面还有旧函数或调试代码引用时报错。
PHENOLOGY_STAGE_X_MODE = FULL_RESPONSE_X_MODE
PHENOLOGY_STAGE_BIN_WIDTH_PCT = FULL_RESPONSE_BIN_WIDTH_PCT
PHENOLOGY_STAGE_X_COL = FULL_RESPONSE_X_COL
PHENOLOGY_STAGE_UNIT_COL = FULL_RESPONSE_UNIT_COL

# 需要绘制的极端事件类型。
# 新绘图方式：每个变量只生成一张“一行三列”的对比图。
# 三列依次为：
#   1) DFAA，方向由 DFAA_PLOT_DIRECTION 控制；
#   2) SPEI > 1 洪涝；
#   3) SPEI < -1 干旱。
#
# DFAA_PLOT_DIRECTION 可选：
#   "positive" = 绘制 positive_DFAA，即 LDFAI / DFAI > 4；
#   "negative" = 绘制 negative_DFAA，即 LDFAI / DFAI < -4。
DFAA_PLOT_DIRECTION = "negative"


def get_selected_dfaa_event_type():
    """根据 DFAA_PLOT_DIRECTION 返回当前需要绘制的 DFAA 事件类型。"""

    direction = str(DFAA_PLOT_DIRECTION).strip().lower()

    positive_keys = {
        "positive", "pos", "+", "+dfaa", "positive_dfaa", "positive dfaa"
    }
    negative_keys = {
        "negative", "neg", "-", "-dfaa", "negative_dfaa", "negative dfaa"
    }

    if direction in positive_keys:
        return "positive_DFAA"

    if direction in negative_keys:
        return "negative_DFAA"

    raise ValueError(
        "DFAA_PLOT_DIRECTION 只能设置为 'positive' 或 'negative'。"
        f" 当前值为：{DFAA_PLOT_DIRECTION}"
    )


def get_one_row_plot_event_types():
    """返回一行三列图中需要绘制的事件类型顺序。"""

    return [
        get_selected_dfaa_event_type(),
        "flood",    # SPEI > 1
        "drought"   # SPEI < -1
    ]


# 保留变量名 PLOT_EXTREME_EVENT_TYPES，避免后续代码或调试代码引用时报错。
# 当前它不再表示“四类事件分别出图”，而是表示“一行三列图”的三列事件。
PLOT_EXTREME_EVENT_TYPES = get_one_row_plot_event_types()


# ============================================================
# 4.0.1 可选：严格筛选“单一灾害阶段”
# ============================================================
# 目的：当极端天气被判定发生在某一个物候阶段时，可以进一步要求：
# 1) 该站点-年份-情景的这个完整物候阶段内，目标灾害只发生过 1 次；
# 2) 同一物候阶段内没有发生其他任何灾害。
#
# 注意：
# - “发生过 1 次”指事件识别后的连续事件段数量，而不是极端日数量；
# - DFAA 连续事件段已经由 LDFAI_MERGE_GAP_DAYS 合并；
# - SPEI 干旱/洪涝连续事件段已经由 SPEI_MERGE_GAP_DAYS 合并；
# - 开启后样本会明显减少，所以默认 False。
ENABLE_STRICT_SINGLE_DISASTER_STAGE_FILTER = True

# 只对这些目标事件类型执行严格筛选。
# 默认只对 DFAA 执行：
#   positive_DFAA = LDFAI / DFAI > 4
#   negative_DFAA = LDFAI / DFAI < -4
# 如果希望 SPEI 干旱/洪涝也执行同样规则，可改成：
# STRICT_SINGLE_DISASTER_TARGET_EVENT_TYPES = [
#     "positive_DFAA", "negative_DFAA", "flood", "drought"
# ]
STRICT_SINGLE_DISASTER_TARGET_EVENT_TYPES = [
    "positive_DFAA",
    "negative_DFAA"
]

# 哪些事件类型被视为“灾害”。
# 目标事件所在阶段内，只要这些灾害事件总数不是 1，就会被剔除。
STRICT_SINGLE_DISASTER_EVENT_TYPES = [
    "positive_DFAA",
    "negative_DFAA",
    "flood",
    "drought"
]

# 只在这些物候阶段内应用严格筛选。
STRICT_SINGLE_DISASTER_STAGES = EVENT_OCCURRENCE_STAGES



# ============================================================
# 4.1 试运行 / 调图用：限制进入后续分析的站点数量
# ============================================================
# True：只累计到 MAX_PASSED_SITES_FOR_TEST 个通过筛选的唯一站点，
#       然后停止读取新的站点，但继续保存结果并绘图。
# False：处理全部站点。
ENABLE_SITE_LIMIT_FOR_TEST = True
MAX_PASSED_SITES_FOR_TEST = 500

# 计数方式：
#   "site"      = 按唯一站点计数；同一站点的多个年份会继续处理
#   "site_year" = 按站点-年份计数；达到 10 个站点-年份就停止
SITE_LIMIT_MODE = "site"

MIN_EVENTS_PER_PLOT = 3

# ============================================================
# 4.1.1 Normal 基线相减方式
# ============================================================
# 目标：仍然控制站点差异，因为不同站点的生长曲线可能不同；
# 但不再要求 extreme 与 normal 在 time / scenario / event_id 等维度上完全一一匹配。
#
# 当前模式 same_site_stage_relaxed：
# - extreme 事件只和同一个 site_folder 下的 normal 基线相减；
# - 同时保留 event_occurrence_stage 一致，即 early 对 early normal、middle 对 middle normal、late 对 late normal；
# - 不再要求同一年份 time、同一 scenario 或同一个 x 上有完全配对的 normal 事件样本单位。
#
# 如果希望进一步放宽为“只按同站点，不按发生阶段”，可改为：
# NORMAL_BASELINE_MATCH_COLS = ["site_folder"]
NORMAL_BASELINE_MODE = "same_site_stage_relaxed"
NORMAL_BASELINE_MATCH_COLS = ["site_folder", "event_occurrence_stage"]


# ============================================================
# 4.2 总图输出控制
# ============================================================
# True：把 PLOT_VARIABLES 中所有指标画在同一张总图中；每个指标一行，
#       三列依次为 DFAA、洪涝、干旱。
ENABLE_ALL_VARIABLES_ONE_FIGURE = True

# True：额外保留原来的“每个指标一张图”的输出。
# 现在默认关闭，避免生成太多重复图。
ENABLE_SINGLE_VARIABLE_FIGURES = False

PLOT_VARIABLES = [
    # =========================
    # 1. 冠层 / 生长状态
    # =========================
    "PlantGro_LAID",   # 叶面积指数，最推荐
    "PlantGro_CWAD",   # 总地上部干物质，最推荐
    # "PlantGro_LWAD",   # 叶干重
    # "PlantGro_SWAD",   # 茎干重
    # "PlantGro_VWAD",   # 营养器官干重
    # "PlantGro_CHTD",   # 冠层高度

    # =========================
    # 2. 根系响应
    # =========================
    # "PlantGro_RWAD",   # 根干重
    # "PlantGro_RDPD",   # 根深

    # =========================
    # 3. 籽粒 / 产量形成
    # =========================
    "PlantGro_GWAD",   # 籽粒干重，最推荐
    # "PlantGro_G#AD",   # 籽粒数
    # "PlantGro_GWGD",   # 单粒重或籽粒增重相关
    # "PlantGro_HIAD",   # 收获指数

    # =========================
    # 4. 水分胁迫指标
    # =========================
    "PlantGro_WSPD",   # 光合作用水分胁迫，最推荐
    "PlantGro_WSGD",   # 生长水分胁迫，最推荐
    # "PlantGro_EWSD",   # 水分/环境胁迫相关指标，建议保留
]

# y 轴标题使用缩写，避免 PlantGro_... 过长。
Y_AXIS_ABBR_LABELS = {
    "PlantGro_LAID": "LAI",
    "PlantGro_CWAD": "CWAD",
    "PlantGro_LWAD": "LWAD",
    "PlantGro_SWAD": "SWAD",
    "PlantGro_VWAD": "VWAD",
    "PlantGro_CHTD": "CHTD",
    "PlantGro_RWAD": "RWAD",
    "PlantGro_RDPD": "RDPD",
    "PlantGro_GWAD": "GWAD",
    "PlantGro_G#AD": "G#AD",
    "PlantGro_GWGD": "GWGD",
    "PlantGro_HIAD": "HIAD",
    "PlantGro_WSPD": "WSPD",
    "PlantGro_WSGD": "WSGD",
    "PlantGro_EWSD": "EWSD",
}


def get_y_axis_label(variable):
    """返回 y 轴使用的简短缩写。"""

    return Y_AXIS_ABBR_LABELS.get(
        str(variable),
        str(variable).replace("PlantGro_", "")
    )
EVENT_TYPE_ORDER = [
    "positive_DFAA",
    "negative_DFAA",
    "drought",
    "flood",
    "normal"
]

STAGE_ORDER = [
    "early_stage",
    "middle_stage",
    "late_stage",
    "maturity_stage",
    "outside_crop_period",
    "unknown"
]


# ============================================================
# 5. 基础工具函数
# ============================================================

def safe_filename(x):
    x = str(x)
    x = re.sub(r'[<>:"/\\|?*]+', "_", x)
    x = x.replace(" ", "_")
    return x


def json_default_for_cache(obj):
    """把 NumPy / pandas 类型转换成 JSON 可序列化对象，用于参数签名。"""

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return str(obj)


def sha256_json(obj):
    """对参数字典生成稳定 hash。"""

    s = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        default=json_default_for_cache
    )
    return hashlib.sha256(s.encode("utf-8")).hexdigest()



def build_file_signature(file_paths):
    """
    生成输入文件签名。

    优化点：
    1) 只记录路径、大小和修改时间，不读取文件内容；
    2) 同一次运行中相同文件集合会复用签名结果，避免多作物/正负 DFAI 重复 os.stat。
    """

    normalized_paths = tuple(sorted([os.path.normpath(str(p)) for p in file_paths]))
    cached = _FILE_SIGNATURE_CACHE.get(normalized_paths)
    if cached is not None:
        return cached

    records = []

    for fp in normalized_paths:
        if not os.path.exists(fp):
            records.append({
                "path": fp,
                "exists": False,
                "size": None,
                "mtime_ns": None
            })
            continue

        st = os.stat(fp)
        records.append({
            "path": fp,
            "exists": True,
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns)
        })

    signature = {
        "count": len(records),
        "total_size": int(sum(r["size"] or 0 for r in records)),
        "latest_mtime_ns": int(max([r["mtime_ns"] or 0 for r in records], default=0)),
        "files_hash": sha256_json(records)
    }

    _FILE_SIGNATURE_CACHE[normalized_paths] = signature
    return signature


def build_processing_cache_params(file_df):
    """
    生成处理缓存使用的关键参数。

    注意：DFAA_PLOT_DIRECTION 不放进这里。
    因为正向 / 负向 DFAA 只是绘图选择，事件识别阶段会同时保留 positive_DFAA 和 negative_DFAA；
    所以切换 DFAA_PLOT_DIRECTION 时不需要重新读取原始文件。

    优化点：
    - 默认只把当前 DSSAT 文件清单涉及到的站点气象文件纳入缓存签名；
      避免每处理一个作物/目录都扫描 climate_dir 下所有气象文件。
    """

    dssat_input_files = []
    sites_for_climate_signature = []

    if file_df is not None and not file_df.empty:
        if "fp" in file_df.columns:
            dssat_input_files = file_df["fp"].astype(str).tolist()
        if "site_folder" in file_df.columns:
            sites_for_climate_signature = (
                file_df["site_folder"].astype(str).dropna().unique().tolist()
            )

    if CACHE_SIGNATURE_ONLY_CURRENT_SITES and sites_for_climate_signature:
        climate_input_files = [
            os.path.join(climate_dir, f"{site}_SPEI30_LDFAI_lag30.csv")
            for site in sites_for_climate_signature
        ]
    else:
        climate_input_files = glob.glob(
            os.path.join(climate_dir, "*_SPEI30_LDFAI_lag30.csv")
        )

    params = {
        "cache_version": PROCESSED_CACHE_VERSION,

        "paths": {
            "dssat_dir": os.path.normpath(dssat_dir),
            "climate_dir": os.path.normpath(climate_dir)
        },

        "performance_options": {
            "USE_MINIMAL_DSSAT_COLUMNS": USE_MINIMAL_DSSAT_COLUMNS,
            "FAST_DFAA_WINDOW_COVERAGE": FAST_DFAA_WINDOW_COVERAGE,
            "DEDUPLICATE_FULL_RESPONSE_EVENT_DAILY_BY_UNIT": DEDUPLICATE_FULL_RESPONSE_EVENT_DAILY_BY_UNIT,
            "CACHE_SIGNATURE_ONLY_CURRENT_SITES": CACHE_SIGNATURE_ONLY_CURRENT_SITES,
        },

        "scenario": {
            "ANALYZE_SCENARIO": ANALYZE_SCENARIO
        },

        "event_thresholds": {
            "SPEI_DROUGHT_THRESHOLD": SPEI_DROUGHT_THRESHOLD,
            "SPEI_FLOOD_THRESHOLD": SPEI_FLOOD_THRESHOLD,
            "LDFAI_POS_THRESHOLD": LDFAI_POS_THRESHOLD,
            "LDFAI_NEG_THRESHOLD": LDFAI_NEG_THRESHOLD,
            "LDFAI_WINDOW_DAYS": LDFAI_WINDOW_DAYS,
            "LDFAI_MERGE_GAP_DAYS": LDFAI_MERGE_GAP_DAYS,
            "SPEI_MERGE_GAP_DAYS": SPEI_MERGE_GAP_DAYS,
            "SPEI_MIN_EVENT_DAYS": SPEI_MIN_EVENT_DAYS,
            "EXCLUDE_DFAA_WINDOW_FOR_SPEI_EVENTS": EXCLUDE_DFAA_WINDOW_FOR_SPEI_EVENTS,
            "NORMAL_MIN_SEGMENT_DAYS": NORMAL_MIN_SEGMENT_DAYS,
            "NORMAL_MAX_EVENTS_PER_STAGE_PER_SAMPLE": NORMAL_MAX_EVENTS_PER_STAGE_PER_SAMPLE
        },

        "crop_qc": {
            "CURRENT_CROP_NAME": CURRENT_CROP_NAME,
            "ENABLE_CROP_QC_FILTER": ENABLE_CROP_QC_FILTER,
            "MIN_FINAL_YIELD": MIN_FINAL_YIELD,
            "MAX_FINAL_YIELD": MAX_FINAL_YIELD,
            "MIN_CROP_DURATION_DAYS": MIN_CROP_DURATION_DAYS,
            "MAX_CROP_DURATION_DAYS": MAX_CROP_DURATION_DAYS,
            "MIN_VALID_CROP_DAYS": MIN_VALID_CROP_DAYS,
            "MIN_MAX_LAI": MIN_MAX_LAI,
            "MAX_MAX_LAI": MAX_MAX_LAI,
            "MIN_MAX_CWAD": MIN_MAX_CWAD,
            "MIN_GSTD_STAGE_COUNT": MIN_GSTD_STAGE_COUNT,
            "MIN_CLIMATE_MATCH_RATIO": MIN_CLIMATE_MATCH_RATIO
        },

        "dssat_preprocessing": {
            "ENABLE_WHEAT_DOY_DAS_SPLIT_ROW_REPAIR": ENABLE_WHEAT_DOY_DAS_SPLIT_ROW_REPAIR,
            "WHEAT_DOY_DAS_SPLIT_ROW_REPAIR_CROP_NAMES": sorted(list(WHEAT_DOY_DAS_SPLIT_ROW_REPAIR_CROP_NAMES)),
            "DROP_WHEAT_ROWS_WITHOUT_PLANTGRO_STATE": DROP_WHEAT_ROWS_WITHOUT_PLANTGRO_STATE,
            "DSSAT_REPAIR_KEY_PLANTGRO_COLS": DSSAT_REPAIR_KEY_PLANTGRO_COLS,
            "DEFAULT_FINAL_YIELD_COLUMNS": DEFAULT_FINAL_YIELD_COLUMNS,
            "CROP_SPECIFIC_FINAL_YIELD_COLUMNS": CROP_SPECIFIC_FINAL_YIELD_COLUMNS,
            "CROP_SPECIFIC_VARIABLE_ALIASES": CROP_SPECIFIC_VARIABLE_ALIASES
        },

        "phenology_and_response": {
            "EVENT_OCCURRENCE_STAGES": EVENT_OCCURRENCE_STAGES,
            "FULL_RESPONSE_STAGES": FULL_RESPONSE_STAGES,
            "FULL_RESPONSE_X_MODE": FULL_RESPONSE_X_MODE,
            "FULL_RESPONSE_BIN_WIDTH_PCT": FULL_RESPONSE_BIN_WIDTH_PCT
        },

        "strict_single_disaster_filter": {
            "ENABLE_STRICT_SINGLE_DISASTER_STAGE_FILTER": ENABLE_STRICT_SINGLE_DISASTER_STAGE_FILTER,
            "STRICT_SINGLE_DISASTER_TARGET_EVENT_TYPES": STRICT_SINGLE_DISASTER_TARGET_EVENT_TYPES,
            "STRICT_SINGLE_DISASTER_EVENT_TYPES": STRICT_SINGLE_DISASTER_EVENT_TYPES,
            "STRICT_SINGLE_DISASTER_STAGES": STRICT_SINGLE_DISASTER_STAGES
        },

        "site_limit": {
            "ENABLE_SITE_LIMIT_FOR_TEST": ENABLE_SITE_LIMIT_FOR_TEST,
            "MAX_PASSED_SITES_FOR_TEST": MAX_PASSED_SITES_FOR_TEST,
            "SITE_LIMIT_MODE": SITE_LIMIT_MODE
        },

        "input_signatures": {
            "dssat_files": build_file_signature(dssat_input_files),
            "climate_files": build_file_signature(climate_input_files)
        }
    }

    return params

def build_processed_cache_paths(cache_params):
    """根据参数 hash 生成缓存文件名。"""

    digest = sha256_json(cache_params)[:16]
    site_limit_part = (
        f"limit-{SITE_LIMIT_MODE}-{MAX_PASSED_SITES_FOR_TEST}"
        if ENABLE_SITE_LIMIT_FOR_TEST
        else "limit-none"
    )

    stem = (
        f"processed_{safe_filename(ANALYZE_SCENARIO)}_"
        f"{safe_filename(site_limit_part)}_"
        f"spei{safe_filename(SPEI_DROUGHT_THRESHOLD)}to{safe_filename(SPEI_FLOOD_THRESHOLD)}_"
        f"ldfai{safe_filename(LDFAI_NEG_THRESHOLD)}to{safe_filename(LDFAI_POS_THRESHOLD)}_"
        f"bin{safe_filename(FULL_RESPONSE_BIN_WIDTH_PCT)}_"
        f"{digest}"
    )

    return {
        "digest": digest,
        "stem": stem,
        "meta": os.path.join(cache_dir, f"{stem}_metadata.json"),
        "stage_summary_total": os.path.join(cache_dir, f"{stem}_stage_summary.pkl"),
        "event_total": os.path.join(cache_dir, f"{stem}_events.pkl"),
        "event_daily_total": os.path.join(cache_dir, f"{stem}_event_daily.pkl"),
        "quality_df": os.path.join(cache_dir, f"{stem}_quality.pkl"),
        "processed_trt1_df": os.path.join(cache_dir, f"{stem}_processed_records.pkl"),
        "selected_units_df": os.path.join(cache_dir, f"{stem}_selected_units.pkl")
    }


def save_processed_data_cache(cache_paths, cache_params, results):
    """保存处理后的 DataFrame，供下次直接绘图使用。"""

    os.makedirs(cache_dir, exist_ok=True)

    dataframe_keys = [
        "stage_summary_total",
        "event_total",
        "event_daily_total",
        "quality_df",
        "processed_trt1_df",
        "selected_units_df"
    ]

    for key in dataframe_keys:
        df = results.get(key, pd.DataFrame())
        if df is None:
            df = pd.DataFrame()
        df.to_pickle(cache_paths[key])

    meta = {
        "cache_params": cache_params,
        "cache_digest": cache_paths["digest"],
        "processed_count": int(results.get("processed_count", 0)),
        "passed_count": int(results.get("passed_count", 0)),
        "non_target_scenarios_ignored_count": int(results.get("non_target_scenarios_ignored_count", 0)),
        "selected_units_mode": results.get("selected_units_mode", SITE_LIMIT_MODE),
        "accepted_sites_for_test": list(results.get("accepted_sites_for_test", [])),
        "accepted_site_years_for_test": [
            list(x) for x in results.get("accepted_site_years_for_test", [])
        ]
    }

    with open(cache_paths["meta"], "w", encoding="utf-8") as f:
        json.dump(
            meta,
            f,
            ensure_ascii=False,
            indent=2,
            default=json_default_for_cache
        )

    print(f"\n处理结果缓存已保存：{cache_paths['stem']}")
    print(f"缓存目录：{cache_dir}")


def load_processed_data_cache(cache_paths, expected_cache_params):
    """读取缓存；如果缺文件或参数不一致，则返回 None。"""

    required_paths = [
        cache_paths["meta"],
        cache_paths["stage_summary_total"],
        cache_paths["event_total"],
        cache_paths["event_daily_total"],
        cache_paths["quality_df"],
        cache_paths["processed_trt1_df"],
        cache_paths["selected_units_df"]
    ]

    if not all(os.path.exists(p) for p in required_paths):
        return None

    try:
        with open(cache_paths["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        print(f"缓存元数据读取失败，将重新处理：{e}")
        return None

    expected_digest = sha256_json(expected_cache_params)
    cached_digest = sha256_json(meta.get("cache_params", {}))

    if cached_digest != expected_digest:
        print("已有缓存的关键参数或输入文件签名与当前设置不一致，将重新读取原始文件。")
        return None

    try:
        results = {
            "stage_summary_total": pd.read_pickle(cache_paths["stage_summary_total"]),
            "event_total": pd.read_pickle(cache_paths["event_total"]),
            "event_daily_total": pd.read_pickle(cache_paths["event_daily_total"]),
            "quality_df": pd.read_pickle(cache_paths["quality_df"]),
            "processed_trt1_df": pd.read_pickle(cache_paths["processed_trt1_df"]),
            "selected_units_df": pd.read_pickle(cache_paths["selected_units_df"]),
            "processed_count": int(meta.get("processed_count", 0)),
            "passed_count": int(meta.get("passed_count", 0)),
            "non_target_scenarios_ignored_count": int(meta.get("non_target_scenarios_ignored_count", 0)),
            "selected_units_mode": meta.get("selected_units_mode", SITE_LIMIT_MODE),
            "accepted_sites_for_test": set(meta.get("accepted_sites_for_test", [])),
            "accepted_site_years_for_test": set(
                tuple(x) for x in meta.get("accepted_site_years_for_test", [])
            )
        }
    except Exception as e:
        print(f"缓存数据读取失败，将重新处理：{e}")
        return None

    print(f"\n已直接读取处理结果缓存：{cache_paths['stem']}")
    print("本次不再重复读取 DSSAT 和气象原始文件，直接进入结果表保存和绘图。")

    return results


def parse_site_time_scenario_from_filename(fp):
    name = os.path.splitext(os.path.basename(fp))[0]
    parts = name.split("-")

    if len(parts) < 3:
        raise ValueError(f"文件名不符合 站点-时间-情景.csv 格式：{name}")

    site = parts[0]
    time = parts[1]
    scenario = "-".join(parts[2:])

    return site, time, scenario


def scenario_sort_key(scenario):
    s = str(scenario).upper()
    m = re.search(r"TRT\s*([0-9]+)", s)
    if m:
        return int(m.group(1))
    return 9999


def get_numeric_series(df, col):
    if col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def get_final_value_by_date(df, col):
    if col not in df.columns:
        return np.nan

    tmp = df[["DATE", col]].copy()
    tmp["DATE"] = pd.to_datetime(tmp["DATE"], errors="coerce")
    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    tmp = tmp.dropna(subset=["DATE", col])

    if tmp.empty:
        return np.nan

    tmp = tmp.sort_values("DATE")
    return tmp[col].iloc[-1]


def split_continuous_segments(
    df,
    date_col="DATE",
    merge_gap_days=0
):
    """
    将连续日期合并成事件段。

    merge_gap_days = 0:
        只合并完全连续的日期。

    merge_gap_days = 3:
        中间断 1-3 天仍合并为同一事件段。
    """

    if df.empty:
        return df

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    if d.empty:
        return d

    d["gap"] = d[date_col].diff().dt.days
    d["new_segment"] = (
        d["gap"].isna() |
        (d["gap"] > merge_gap_days + 1)
    )

    d["segment_id"] = d["new_segment"].cumsum()

    return d


def center_date_of_interval(start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    delta_days = int((end_date - start_date).days)
    return start_date + pd.to_timedelta(delta_days // 2, unit="D")


def is_wheat_crop(crop_name=None):
    """判断当前作物是否需要 Wheat 专用的 YEAR/DOY/DAS 拆行修复。"""

    name = str(crop_name if crop_name is not None else CURRENT_CROP_NAME).strip().lower()
    return name in {str(x).strip().lower() for x in WHEAT_DOY_DAS_SPLIT_ROW_REPAIR_CROP_NAMES}


def first_non_missing_value(series):
    """返回 Series 中第一个非缺失值，用于同一 YEAR-DOY 拆行合并。"""

    s = series.dropna()
    if s.empty:
        return np.nan
    return s.iloc[0]


def fill_date_from_year_doy(df):
    """当 DATE 为空但 YEAR / DOY 存在时，用 YEAR + DOY 补出日期。"""

    out = df.copy()

    if "DATE" not in out.columns:
        out["DATE"] = pd.NaT

    out["DATE"] = pd.to_datetime(out["DATE"], errors="coerce")

    if "YEAR" not in out.columns or "DOY" not in out.columns:
        return out

    year_num = pd.to_numeric(out["YEAR"], errors="coerce")
    doy_num = pd.to_numeric(out["DOY"], errors="coerce")

    fill_mask = (
        out["DATE"].isna() &
        year_num.notna() &
        doy_num.notna()
    )

    if fill_mask.any():
        year_int = year_num.loc[fill_mask].astype(int)
        doy_int = doy_num.loc[fill_mask].astype(int)

        reconstructed_date = (
            pd.to_datetime(year_int.astype(str), format="%Y", errors="coerce") +
            pd.to_timedelta(doy_int - 1, unit="D")
        )

        out.loc[fill_mask, "DATE"] = reconstructed_date.values

    return out


def repair_wheat_split_rows_by_year_doy(dssat, crop_name=None):
    """
    修复 Wheat DSSAT daily CSV 中常见的 YEAR-DOY 拆行问题。

    典型问题：
    - PlantGro_* 行有 YEAR / DOY / DAS / PlantGro_DAP，但 DATE 为空；
    - Soil / Weather 行有 DATE，但 PlantGro_* 为空；
    - 两行 YEAR-DOY 相同，但 DAS 可能不完全一致。

    处理逻辑：
    1) 先用 YEAR + DOY 补齐缺失 DATE；
    2) 对同一 YEAR-DOY 的多行进行合并；
       PlantGro_* 与 DAS 优先取 PlantGro 信息较完整的行，
       DATE / Soil / Weather 从有值的行补齐；
    3) 可选删除完全没有 PlantGro 状态的环境-only 空行。
    """

    raw_rows = int(len(dssat))

    if dssat is None or dssat.empty:
        out = pd.DataFrame() if dssat is None else dssat.copy()
        out.attrs.update({
            "dssat_raw_rows": raw_rows,
            "dssat_clean_rows": int(len(out)),
            "dssat_preprocess_mode": "empty_input",
            "dssat_year_doy_groups_collapsed": 0,
            "dssat_empty_crop_rows_removed": 0,
        })
        return out

    df = dssat.copy()

    # 把空字符串和纯空格统一为 NaN，避免“看起来空”的单元格被当成有效值。
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        df[object_cols] = df[object_cols].replace(r"^\s*$", np.nan, regex=True)

    should_repair = (
        ENABLE_WHEAT_DOY_DAS_SPLIT_ROW_REPAIR and
        is_wheat_crop(crop_name) and
        {"YEAR", "DOY"}.issubset(df.columns)
    )

    if not should_repair:
        df = fill_date_from_year_doy(df)
        df.attrs.update({
            "dssat_raw_rows": raw_rows,
            "dssat_clean_rows": int(len(df)),
            "dssat_preprocess_mode": "date_fill_only",
            "dssat_year_doy_groups_collapsed": 0,
            "dssat_empty_crop_rows_removed": 0,
        })
        return df

    df = fill_date_from_year_doy(df)

    year_num = pd.to_numeric(df["YEAR"], errors="coerce")
    doy_num = pd.to_numeric(df["DOY"], errors="coerce")
    key_mask = year_num.notna() & doy_num.notna()

    plantgro_cols = [c for c in df.columns if str(c).startswith("PlantGro_")]
    key_plant_cols = [c for c in DSSAT_REPAIR_KEY_PLANTGRO_COLS if c in df.columns]

    if not plantgro_cols or not key_mask.any():
        df.attrs.update({
            "dssat_raw_rows": raw_rows,
            "dssat_clean_rows": int(len(df)),
            "dssat_preprocess_mode": "wheat_repair_skipped_no_keys_or_plantgro",
            "dssat_year_doy_groups_collapsed": 0,
            "dssat_empty_crop_rows_removed": 0,
        })
        return df

    keyed = df.loc[key_mask].copy()
    unkeyed = df.loc[~key_mask].copy()

    keyed["_repair_year"] = year_num.loc[key_mask].astype(int).values
    keyed["_repair_doy"] = doy_num.loc[key_mask].astype(int).values
    keyed["_plant_valid_count"] = keyed[plantgro_cols].notna().sum(axis=1)
    keyed["_date_valid"] = keyed["DATE"].notna().astype(int)
    keyed["_overall_valid_count"] = keyed.notna().sum(axis=1)

    # PlantGro 信息完整的行排在前面：这样 DAS / PlantGro_* 优先来自生长行；
    # DATE / Soil / Weather 则会从后面的环境行补齐。
    keyed = keyed.sort_values(
        [
            "_repair_year",
            "_repair_doy",
            "_plant_valid_count",
            "_date_valid",
            "_overall_valid_count",
        ],
        ascending=[True, True, False, False, False],
    )

    original_cols = [c for c in df.columns if not str(c).startswith("_repair_")]
    collapsed = (
        keyed
        .groupby(["_repair_year", "_repair_doy"], sort=True, dropna=False)[original_cols]
        .agg(first_non_missing_value)
        .reset_index(drop=True)
    )

    if not unkeyed.empty:
        unkeyed = unkeyed[original_cols].copy()
        collapsed = pd.concat([collapsed, unkeyed], ignore_index=True)

    collapsed = fill_date_from_year_doy(collapsed)

    groups_before = int(keyed.groupby(["_repair_year", "_repair_doy"]).ngroups)
    rows_after_collapse = int(len(collapsed))

    removed_empty_crop_rows = 0

    if DROP_WHEAT_ROWS_WITHOUT_PLANTGRO_STATE and key_plant_cols:
        before_drop = len(collapsed)
        crop_state_mask = collapsed[key_plant_cols].notna().any(axis=1)
        collapsed = collapsed.loc[crop_state_mask].copy()
        removed_empty_crop_rows = int(before_drop - len(collapsed))

    collapsed = collapsed.sort_values("DATE").reset_index(drop=True)

    collapsed.attrs.update({
        "dssat_raw_rows": raw_rows,
        "dssat_clean_rows": int(len(collapsed)),
        "dssat_preprocess_mode": "wheat_year_doy_split_row_repair",
        "dssat_year_doy_groups_collapsed": groups_before,
        "dssat_rows_after_year_doy_collapse": rows_after_collapse,
        "dssat_empty_crop_rows_removed": removed_empty_crop_rows,
    })

    return collapsed


def get_crop_key(crop_name=None):
    """返回小写作物名，用于作物专用配置。"""

    return str(crop_name if crop_name is not None else CURRENT_CROP_NAME).strip().lower()


def get_final_yield_column_candidates(crop_name=None):
    """返回当前作物可用于 QC 的产量列候选顺序。"""

    crop_key = get_crop_key(crop_name)
    return CROP_SPECIFIC_FINAL_YIELD_COLUMNS.get(
        crop_key,
        DEFAULT_FINAL_YIELD_COLUMNS
    )


def get_first_valid_final_value_by_date(df, candidate_cols):
    """
    从多个候选列中提取最终产量。

    返回：
    - final_value: 最后一个非缺失值；
    - source_col: 实际使用的列名。
    """

    for col in candidate_cols:
        if col not in df.columns:
            continue

        val = get_final_value_by_date(df, col)

        if not pd.isna(val):
            return val, col

        s = get_numeric_series(df, col)
        if s.notna().any():
            return s.max(), col

    return np.nan, None


def apply_crop_specific_variable_aliases(dssat, crop_name=None):
    """
    根据作物类型补齐统一变量名。

    例如 Wheat 中如果没有 PlantGro_GWAD，但有 PlantGro_HWAD，
    则生成 PlantGro_GWAD = PlantGro_HWAD；
    如果没有 PlantGro_WSPD / PlantGro_WSGD，则分别由 PlantGro_WFPD / PlantGro_WFGD 补齐。
    """

    if dssat is None or dssat.empty:
        return dssat

    out = dssat.copy()
    old_attrs = dict(getattr(dssat, "attrs", {}) or {})

    crop_key = get_crop_key(crop_name)
    alias_map = CROP_SPECIFIC_VARIABLE_ALIASES.get(crop_key, {})

    applied_aliases = {}

    for target_col, source_candidates in alias_map.items():
        target_missing_or_empty = (
            target_col not in out.columns or
            pd.to_numeric(out[target_col], errors="coerce").notna().sum() == 0
        )

        if not target_missing_or_empty:
            continue

        for source_col in source_candidates:
            if source_col not in out.columns:
                continue

            source_series = pd.to_numeric(out[source_col], errors="coerce")
            if source_series.notna().sum() == 0:
                continue

            out[target_col] = source_series
            applied_aliases[target_col] = source_col
            break

    old_attrs["dssat_variable_aliases_applied"] = json.dumps(
        applied_aliases,
        ensure_ascii=False
    )
    out.attrs.update(old_attrs)

    return out



def get_dssat_required_columns(crop_name=None):
    """
    返回读取 DSSAT daily 文件时真正需要保留的列。

    只读必要列可以显著减少大 CSV 的 I/O、解析和后续 merge/copy 成本。
    """

    crop_key = get_crop_key(crop_name)

    cols = {
        # 时间 / 文件识别
        "DATE", "YEAR", "DOY", "DAS",
        "site_folder", "time", "scenario",

        # 物候、QC 和阶段划分
        "DAP", "PlantGro_DAP", "PlantGro_GSTD",
        "PlantGro_LAID", "PlantGro_CWAD", "PlantGro_RWAD", "PlantGro_HIAD",
    }

    cols.update(PLOT_VARIABLES)
    cols.update(DEFAULT_FINAL_YIELD_COLUMNS)

    for c in get_final_yield_column_candidates(crop_name):
        cols.add(c)

    for c in DSSAT_REPAIR_KEY_PLANTGRO_COLS:
        cols.add(c)

    alias_map = CROP_SPECIFIC_VARIABLE_ALIASES.get(crop_key, {})
    for target_col, source_candidates in alias_map.items():
        cols.add(target_col)
        cols.update(source_candidates)

    return {str(c).strip() for c in cols if str(c).strip()}



def read_dssat_daily_file_from_csv(fp, crop_name=None):
    """
    从单个原始 DSSAT daily CSV 读取。

    这个函数保留原来的读取逻辑：
    1) 可选只读必要列；
    2) Wheat YEAR / DOY / DAS 拆行修复；
    3) Wheat 等作物专用变量别名补齐。
    """

    if USE_MINIMAL_DSSAT_COLUMNS:
        required_cols = get_dssat_required_columns(crop_name)

        def _usecol(col):
            return str(col).strip() in required_cols

        dssat = pd.read_csv(
            fp,
            low_memory=False,
            usecols=_usecol
        )
        dssat.columns = [str(c).strip() for c in dssat.columns]
    else:
        dssat = pd.read_csv(fp, low_memory=False)
        dssat.columns = [str(c).strip() for c in dssat.columns]

    dssat = repair_wheat_split_rows_by_year_doy(
        dssat=dssat,
        crop_name=crop_name
    )
    dssat = apply_crop_specific_variable_aliases(
        dssat=dssat,
        crop_name=crop_name
    )
    return dssat


def _build_dssat_merged_raw_group_indices(merged):
    """为预合并 DSSAT 表构建 fp -> 行号索引，避免每个样本都全表筛选。"""

    if merged is None or merged.empty or "_source_fp_norm" not in merged.columns:
        return {}

    return {
        str(fp): idx.values
        for fp, idx in merged.groupby("_source_fp_norm", sort=False).groups.items()
    }


def _restore_dssat_attrs_from_merged_sample(sample_df):
    """从预合并缓存的内部列恢复单个样本的 attrs。"""

    attrs = {}

    attr_map = {
        "_dssat_raw_rows": "dssat_raw_rows",
        "_dssat_clean_rows": "dssat_clean_rows",
        "_dssat_preprocess_mode": "dssat_preprocess_mode",
        "_dssat_year_doy_groups_collapsed": "dssat_year_doy_groups_collapsed",
        "_dssat_rows_after_year_doy_collapse": "dssat_rows_after_year_doy_collapse",
        "_dssat_empty_crop_rows_removed": "dssat_empty_crop_rows_removed",
        "_dssat_variable_aliases_applied": "dssat_variable_aliases_applied",
    }

    for col, attr_name in attr_map.items():
        if col in sample_df.columns and sample_df[col].notna().any():
            attrs[attr_name] = sample_df[col].dropna().iloc[0]

    return attrs


def read_dssat_daily_file(fp, crop_name=None):
    """
    读取 DSSAT daily 数据。

    优先从 DSSAT daily 预合并缓存中按 fp 取出单个样本；
    如果当前没有预合并缓存，或者该 fp 不在缓存中，则退回到读取单个 CSV。
    """

    global _CURRENT_DSSAT_MERGED_RAW_DF
    global _CURRENT_DSSAT_MERGED_RAW_GROUP_INDICES

    fp_norm = os.path.normpath(str(fp))

    if (
        ENABLE_DSSAT_MERGED_RAW_CACHE and
        _CURRENT_DSSAT_MERGED_RAW_DF is not None and
        _CURRENT_DSSAT_MERGED_RAW_GROUP_INDICES is not None
    ):
        row_idx = _CURRENT_DSSAT_MERGED_RAW_GROUP_INDICES.get(fp_norm)

        if row_idx is not None and len(row_idx) > 0:
            sample = _CURRENT_DSSAT_MERGED_RAW_DF.iloc[row_idx].copy()
            attrs = _restore_dssat_attrs_from_merged_sample(sample)

            drop_cols = [
                c for c in DSSAT_MERGED_RAW_INTERNAL_COLS
                if c in sample.columns
            ]
            if drop_cols:
                sample = sample.drop(columns=drop_cols)

            sample.attrs.update(attrs)
            return sample

    return read_dssat_daily_file_from_csv(fp, crop_name=crop_name)


def build_dssat_merged_raw_cache_params(file_df, crop_name):
    """生成 DSSAT daily 原始预合并缓存的参数签名。"""

    dssat_input_files = []

    if file_df is not None and not file_df.empty and "fp" in file_df.columns:
        dssat_input_files = file_df["fp"].astype(str).tolist()

    params = {
        "cache_type": "dssat_merged_raw_daily",
        "cache_version": PROCESSED_CACHE_VERSION,
        "crop_name": str(crop_name),
        "dssat_dir": os.path.normpath(str(dssat_dir)),
        "scenario": ANALYZE_SCENARIO,
        "USE_MINIMAL_DSSAT_COLUMNS": USE_MINIMAL_DSSAT_COLUMNS,
        "required_columns": sorted(list(get_dssat_required_columns(crop_name))),
        "dssat_preprocessing": {
            "ENABLE_WHEAT_DOY_DAS_SPLIT_ROW_REPAIR": ENABLE_WHEAT_DOY_DAS_SPLIT_ROW_REPAIR,
            "WHEAT_DOY_DAS_SPLIT_ROW_REPAIR_CROP_NAMES": sorted(list(WHEAT_DOY_DAS_SPLIT_ROW_REPAIR_CROP_NAMES)),
            "DROP_WHEAT_ROWS_WITHOUT_PLANTGRO_STATE": DROP_WHEAT_ROWS_WITHOUT_PLANTGRO_STATE,
            "DSSAT_REPAIR_KEY_PLANTGRO_COLS": DSSAT_REPAIR_KEY_PLANTGRO_COLS,
            "DEFAULT_FINAL_YIELD_COLUMNS": DEFAULT_FINAL_YIELD_COLUMNS,
            "CROP_SPECIFIC_FINAL_YIELD_COLUMNS": CROP_SPECIFIC_FINAL_YIELD_COLUMNS,
            "CROP_SPECIFIC_VARIABLE_ALIASES": CROP_SPECIFIC_VARIABLE_ALIASES,
        },
        "input_signatures": {
            "dssat_files": build_file_signature(dssat_input_files)
        }
    }

    return params


def build_dssat_merged_raw_cache_paths(file_df, crop_name, run_label):
    """根据当前目录、作物、情景和输入文件签名生成预合并缓存路径。"""

    cache_params = build_dssat_merged_raw_cache_params(file_df, crop_name)
    digest = sha256_json(cache_params)[:16]

    stem = (
        f"dssat_merged_raw_daily_"
        f"{safe_filename(crop_name)}_"
        f"{safe_filename(ANALYZE_SCENARIO)}_"
        f"{safe_filename(run_label)}_"
        f"{digest}"
    )

    return {
        "digest": digest,
        "stem": stem,
        "data": os.path.join(cache_dir, f"{stem}.pkl"),
        "meta": os.path.join(cache_dir, f"{stem}_metadata.json"),
    }, cache_params


def clear_current_dssat_merged_raw_cache_from_memory():
    """清空当前运行中的 DSSAT daily 预合并缓存，释放内存引用。"""

    global _CURRENT_DSSAT_MERGED_RAW_DF
    global _CURRENT_DSSAT_MERGED_RAW_DIR
    global _CURRENT_DSSAT_MERGED_RAW_GROUP_INDICES

    _CURRENT_DSSAT_MERGED_RAW_DF = None
    _CURRENT_DSSAT_MERGED_RAW_DIR = None
    _CURRENT_DSSAT_MERGED_RAW_GROUP_INDICES = None


def load_or_build_dssat_merged_raw_cache(file_df, crop_name, run_label):
    """
    读取或构建当前 DSSAT 目录的 daily 预合并缓存。

    重要：
    - 只合并 build_file_df_for_dssat_dir() 已筛选出的目标情景文件；
    - 每个 作物 × 正/负 DFAI 目录 单独生成一个缓存；
    - 缓存命中时不再读取原始 CSV；
    - 缓存不命中时，仍然调用 read_dssat_daily_file_from_csv() 完成原有清洗逻辑。
    """

    global _CURRENT_DSSAT_MERGED_RAW_DF
    global _CURRENT_DSSAT_MERGED_RAW_DIR
    global _CURRENT_DSSAT_MERGED_RAW_GROUP_INDICES

    clear_current_dssat_merged_raw_cache_from_memory()

    if not ENABLE_DSSAT_MERGED_RAW_CACHE:
        return None

    if file_df is None or file_df.empty:
        print("\n当前 DSSAT 目录没有可合并的目标情景文件，跳过 DSSAT daily 预合并缓存。")
        return None

    cache_paths, cache_params = build_dssat_merged_raw_cache_paths(
        file_df=file_df,
        crop_name=crop_name,
        run_label=run_label
    )

    if (
        os.path.exists(cache_paths["data"]) and
        os.path.exists(cache_paths["meta"]) and
        not FORCE_REBUILD_DSSAT_MERGED_RAW_CACHE
    ):
        try:
            with open(cache_paths["meta"], "r", encoding="utf-8") as f:
                meta = json.load(f)

            if sha256_json(meta.get("cache_params", {})) == sha256_json(cache_params):
                print(f"\n已读取 DSSAT daily 预合并缓存：{cache_paths['stem']}")
                print(f"缓存文件：{cache_paths['data']}")

                merged = pd.read_pickle(cache_paths["data"])
                _CURRENT_DSSAT_MERGED_RAW_DF = merged
                _CURRENT_DSSAT_MERGED_RAW_DIR = os.path.normpath(str(dssat_dir))
                _CURRENT_DSSAT_MERGED_RAW_GROUP_INDICES = _build_dssat_merged_raw_group_indices(merged)

                print(
                    f"预合并缓存行数：{len(merged)}，"
                    f"样本数：{len(_CURRENT_DSSAT_MERGED_RAW_GROUP_INDICES)}"
                )
                return merged

            print("已有 DSSAT daily 预合并缓存参数不一致，将重新合并。")
        except Exception as e:
            print(f"DSSAT daily 预合并缓存读取失败，将重新合并：{e}")

    print("\n开始构建 DSSAT daily 预合并缓存...")
    print(f"当前作物：{crop_name}")
    print(f"当前目录：{dssat_dir}")
    print(f"目标情景：{ANALYZE_SCENARIO}")
    print(f"待合并文件数：{len(file_df)}")

    parts = []
    failed_records = []

    for n, (_, row) in enumerate(file_df.iterrows(), start=1):
        fp = row["fp"]
        site = row["site_folder"]
        time = row["time"]
        scenario = row["scenario"]

        try:
            dssat = read_dssat_daily_file_from_csv(
                fp,
                crop_name=crop_name
            )

            attrs = dict(getattr(dssat, "attrs", {}) or {})

            dssat["_source_fp_norm"] = os.path.normpath(str(fp))

            # 用文件名解析得到的元信息覆盖/补齐，保证后续分组稳定。
            dssat["site_folder"] = str(site)
            dssat["time"] = str(time)
            dssat["scenario"] = str(scenario)

            dssat["_dssat_raw_rows"] = attrs.get("dssat_raw_rows", np.nan)
            dssat["_dssat_clean_rows"] = attrs.get("dssat_clean_rows", len(dssat))
            dssat["_dssat_preprocess_mode"] = attrs.get("dssat_preprocess_mode", "unknown")
            dssat["_dssat_year_doy_groups_collapsed"] = attrs.get("dssat_year_doy_groups_collapsed", np.nan)
            dssat["_dssat_rows_after_year_doy_collapse"] = attrs.get("dssat_rows_after_year_doy_collapse", np.nan)
            dssat["_dssat_empty_crop_rows_removed"] = attrs.get("dssat_empty_crop_rows_removed", np.nan)
            dssat["_dssat_variable_aliases_applied"] = attrs.get("dssat_variable_aliases_applied", "{}")

            parts.append(dssat)

            if n == 1 or n % 200 == 0 or n == len(file_df):
                print(f"  已读取并合并 {n} / {len(file_df)} 个文件")

        except Exception as e:
            failed_records.append({
                "fp": str(fp),
                "site_folder": str(site),
                "time": str(time),
                "scenario": str(scenario),
                "error": str(e),
            })
            print(f"  合并时读取失败，已跳过：{fp}")
            print(f"  错误信息：{e}")

    if not parts:
        raise RuntimeError(
            f"DSSAT daily 预合并失败：没有成功读取任何文件。目录：{dssat_dir}"
        )

    merged = pd.concat(parts, ignore_index=True)

    os.makedirs(cache_dir, exist_ok=True)
    merged.to_pickle(cache_paths["data"])

    meta = {
        "cache_params": cache_params,
        "cache_digest": cache_paths["digest"],
        "cache_path": cache_paths["data"],
        "n_files_total": int(len(file_df)),
        "n_files_success": int(len(parts)),
        "n_files_failed": int(len(failed_records)),
        "n_rows": int(len(merged)),
        "failed_records": failed_records,
    }

    with open(cache_paths["meta"], "w", encoding="utf-8") as f:
        json.dump(
            meta,
            f,
            ensure_ascii=False,
            indent=2,
            default=json_default_for_cache
        )

    print(f"\nDSSAT daily 预合并缓存已保存：{cache_paths['stem']}")
    print(f"缓存文件：{cache_paths['data']}")
    print(f"合并后行数：{len(merged)}")

    _CURRENT_DSSAT_MERGED_RAW_DF = merged
    _CURRENT_DSSAT_MERGED_RAW_DIR = os.path.normpath(str(dssat_dir))
    _CURRENT_DSSAT_MERGED_RAW_GROUP_INDICES = _build_dssat_merged_raw_group_indices(merged)

    return merged


def build_interval_covered_flag(date_series, start_series, end_series):
    """
    对一组日期区间 [start, end] 计算每个 date 是否被任一区间覆盖。

    用差分数组替代逐区间循环。复杂度约为 O(N + K)，
    N 为气象日期跨度，K 为事件窗口数。
    """

    dates = pd.to_datetime(date_series, errors="coerce").dt.normalize()
    starts = pd.to_datetime(start_series, errors="coerce").dt.normalize()
    ends = pd.to_datetime(end_series, errors="coerce").dt.normalize()

    valid_date_mask = dates.notna()
    valid_interval_mask = starts.notna() & ends.notna()

    out = np.zeros(len(dates), dtype=bool)

    if (not valid_date_mask.any()) or (not valid_interval_mask.any()):
        return out

    date_days = dates.loc[valid_date_mask].values.astype("datetime64[D]").astype(np.int64)
    start_days = starts.loc[valid_interval_mask].values.astype("datetime64[D]").astype(np.int64)
    end_days = ends.loc[valid_interval_mask].values.astype("datetime64[D]").astype(np.int64)

    min_day = int(min(date_days.min(), start_days.min()))
    max_day = int(max(date_days.max(), end_days.max()))

    diff = np.zeros(max_day - min_day + 2, dtype=np.int32)
    np.add.at(diff, start_days - min_day, 1)
    np.add.at(diff, end_days - min_day + 1, -1)

    covered_by_day = np.cumsum(diff[:-1]) > 0
    out[np.flatnonzero(valid_date_mask.values)] = covered_by_day[date_days - min_day]

    return out



def read_climate_file(site):
    """
    读取站点气象文件：
        AGAA_SPEI30_LDFAI_lag30.csv

    生成：
        drought_candidate_flag
        flood_candidate_flag
        normal_candidate_flag
        LDFAI_DFAA_pos_flag
        LDFAI_DFAA_neg_flag
        DFAA_window_covered_flag
    """

    climate_path = os.path.join(
        climate_dir,
        f"{site}_SPEI30_LDFAI_lag30.csv"
    )

    if not os.path.exists(climate_path):
        return None, climate_path

    climate = pd.read_csv(
        climate_path,
        usecols=lambda c: str(c).strip().lower() in {"date", "spei", "ldfai"}
    )
    climate.columns = [c.strip().lower() for c in climate.columns]

    required_cols = ["date", "spei", "ldfai"]

    for col in required_cols:
        if col not in climate.columns:
            raise ValueError(f"{climate_path} 缺少列：{col}")

    climate["date"] = pd.to_datetime(climate["date"], errors="coerce")
    climate["spei"] = pd.to_numeric(climate["spei"], errors="coerce")
    climate["ldfai"] = pd.to_numeric(climate["ldfai"], errors="coerce")

    climate = climate.dropna(subset=["date"]).copy()
    climate["DATE"] = climate["date"].dt.normalize()
    climate = climate.sort_values("DATE").reset_index(drop=True)

    # --------------------------------------------------------
    # 基础 SPEI 干旱 / 洪涝
    # --------------------------------------------------------
    climate["SPEI_drought_flag"] = climate["spei"] < SPEI_DROUGHT_THRESHOLD
    climate["SPEI_flood_flag"] = climate["spei"] > SPEI_FLOOD_THRESHOLD

    # --------------------------------------------------------
    # LDFAI 正负 DFAA
    # --------------------------------------------------------
    climate["LDFAI_DFAA_pos_flag"] = climate["ldfai"] > LDFAI_POS_THRESHOLD
    climate["LDFAI_DFAA_neg_flag"] = climate["ldfai"] < LDFAI_NEG_THRESHOLD

    climate["LDFAI_DFAA_flag"] = (
        climate["LDFAI_DFAA_pos_flag"] |
        climate["LDFAI_DFAA_neg_flag"]
    )

    climate["LDFAI_DFAA_direction"] = None
    climate.loc[
        climate["LDFAI_DFAA_pos_flag"],
        "LDFAI_DFAA_direction"
    ] = "positive_DFAA"

    climate.loc[
        climate["LDFAI_DFAA_neg_flag"],
        "LDFAI_DFAA_direction"
    ] = "negative_DFAA"

    # LDFAI 日期是 60 天窗口结束日
    climate["LDFAI_window_end"] = climate["DATE"]
    climate["LDFAI_window_start"] = (
        climate["DATE"] - pd.to_timedelta(LDFAI_WINDOW_DAYS - 1, unit="D")
    )

    # --------------------------------------------------------
    # 标记哪些日期落在任意 DFAA 60 天窗口内
    # --------------------------------------------------------
    if FAST_DFAA_WINDOW_COVERAGE:
        dfaa_mask = climate["LDFAI_DFAA_flag"].fillna(False).astype(bool)
        climate["DFAA_window_covered_flag"] = build_interval_covered_flag(
            date_series=climate["DATE"],
            start_series=climate.loc[dfaa_mask, "LDFAI_window_start"],
            end_series=climate.loc[dfaa_mask, "LDFAI_window_end"],
        )
    else:
        climate["DFAA_window_covered_flag"] = False
        dfaa_windows = climate[climate["LDFAI_DFAA_flag"] == True].copy()

        for _, row in dfaa_windows.iterrows():
            ws = row["LDFAI_window_start"]
            we = row["LDFAI_window_end"]

            mask = (
                (climate["DATE"] >= ws) &
                (climate["DATE"] <= we)
            )

            climate.loc[mask, "DFAA_window_covered_flag"] = True

    # --------------------------------------------------------
    # 用于事件识别的干旱 / 洪涝 / 正常候选日
    # --------------------------------------------------------
    if EXCLUDE_DFAA_WINDOW_FOR_SPEI_EVENTS:
        climate["drought_candidate_flag"] = (
            climate["SPEI_drought_flag"] &
            (~climate["DFAA_window_covered_flag"])
        )

        climate["flood_candidate_flag"] = (
            climate["SPEI_flood_flag"] &
            (~climate["DFAA_window_covered_flag"])
        )
    else:
        climate["drought_candidate_flag"] = climate["SPEI_drought_flag"]
        climate["flood_candidate_flag"] = climate["SPEI_flood_flag"]

    climate["normal_candidate_flag"] = (
        climate["spei"].notna() &
        climate["ldfai"].notna() &
        (climate["spei"] >= SPEI_DROUGHT_THRESHOLD) &
        (climate["spei"] <= SPEI_FLOOD_THRESHOLD) &
        (climate["ldfai"] >= LDFAI_NEG_THRESHOLD) &
        (climate["ldfai"] <= LDFAI_POS_THRESHOLD) &
        (~climate["DFAA_window_covered_flag"])
    )

    climate = climate[
        [
            "DATE",
            "spei",
            "ldfai",

            "SPEI_drought_flag",
            "SPEI_flood_flag",

            "drought_candidate_flag",
            "flood_candidate_flag",
            "normal_candidate_flag",

            "LDFAI_DFAA_pos_flag",
            "LDFAI_DFAA_neg_flag",
            "LDFAI_DFAA_flag",
            "LDFAI_DFAA_direction",

            "LDFAI_window_start",
            "LDFAI_window_end",
            "DFAA_window_covered_flag"
        ]
    ].copy()

    return climate, climate_path

def summarize_crop_quality(dssat, site, time, scenario):

    dssat_preprocess_attrs = dict(getattr(dssat, "attrs", {}) or {})

    df = dssat.copy()

    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.dropna(subset=["DATE"]).copy()
        df = df.sort_values("DATE")

        if not df.empty:
            start_date = df["DATE"].min()
            end_date = df["DATE"].max()
            total_date_days = (end_date - start_date).days + 1
        else:
            start_date = pd.NaT
            end_date = pd.NaT
            total_date_days = np.nan
    else:
        start_date = pd.NaT
        end_date = pd.NaT
        total_date_days = np.nan

    if "PlantGro_DAP" in df.columns:
        dap = get_numeric_series(df, "PlantGro_DAP")
    elif "DAP" in df.columns:
        dap = get_numeric_series(df, "DAP")
    else:
        dap = pd.Series([np.nan] * len(df), index=df.index)

    valid_dap = dap.dropna()

    if valid_dap.empty:
        min_dap = np.nan
        max_dap = np.nan
        crop_duration = np.nan
        valid_crop_days = 0
    else:
        min_dap = valid_dap.min()
        max_dap = valid_dap.max()
        crop_duration = max_dap - min_dap + 1
        valid_crop_days = int(valid_dap.notna().sum())

    final_yield_col_candidates = get_final_yield_column_candidates(CURRENT_CROP_NAME)
    final_yield, final_yield_source_col = get_first_valid_final_value_by_date(
        df,
        final_yield_col_candidates
    )

    laid = get_numeric_series(df, "PlantGro_LAID")
    cwad = get_numeric_series(df, "PlantGro_CWAD")
    rwad = get_numeric_series(df, "PlantGro_RWAD")

    max_lai = laid.max()
    max_cwad = cwad.max()
    max_rwad = rwad.max()
    final_hiad = get_final_value_by_date(df, "PlantGro_HIAD")

    if "PlantGro_GSTD" in df.columns:
        gstd = pd.to_numeric(df["PlantGro_GSTD"], errors="coerce")
        gstd_stage_count = int(gstd.dropna().round(0).nunique())
    else:
        gstd_stage_count = 0

    fail_reasons = []

    if pd.isna(final_yield):
        fail_reasons.append("missing_final_yield")
    elif final_yield < MIN_FINAL_YIELD:
        fail_reasons.append("final_yield_too_low")
    elif final_yield > MAX_FINAL_YIELD:
        fail_reasons.append("final_yield_too_high")

    if pd.isna(crop_duration):
        fail_reasons.append("missing_crop_duration")
    elif crop_duration < MIN_CROP_DURATION_DAYS:
        fail_reasons.append("crop_duration_too_short")
    elif crop_duration > MAX_CROP_DURATION_DAYS:
        fail_reasons.append("crop_duration_too_long")

    if valid_crop_days < MIN_VALID_CROP_DAYS:
        fail_reasons.append("too_few_valid_crop_days")

    if pd.isna(max_lai):
        fail_reasons.append("missing_lai")
    elif max_lai < MIN_MAX_LAI:
        fail_reasons.append("max_lai_too_low")
    elif max_lai > MAX_MAX_LAI:
        fail_reasons.append("max_lai_too_high")

    if pd.isna(max_cwad):
        fail_reasons.append("missing_cwad")
    elif max_cwad < MIN_MAX_CWAD:
        fail_reasons.append("max_cwad_too_low")

    if gstd_stage_count > 0 and gstd_stage_count < MIN_GSTD_STAGE_COUNT:
        fail_reasons.append("too_few_gstd_stages")

    passed_crop_qc = len(fail_reasons) == 0

    quality_record = {
        "site_folder": site,
        "time": str(time),
        "scenario": scenario,
        "crop_name": CURRENT_CROP_NAME,

        "dssat_raw_rows": dssat_preprocess_attrs.get("dssat_raw_rows", len(dssat)),
        "dssat_clean_rows": dssat_preprocess_attrs.get("dssat_clean_rows", len(dssat)),
        "dssat_preprocess_mode": dssat_preprocess_attrs.get("dssat_preprocess_mode", "unknown"),
        "dssat_year_doy_groups_collapsed": dssat_preprocess_attrs.get("dssat_year_doy_groups_collapsed", np.nan),
        "dssat_rows_after_year_doy_collapse": dssat_preprocess_attrs.get("dssat_rows_after_year_doy_collapse", np.nan),
        "dssat_empty_crop_rows_removed": dssat_preprocess_attrs.get("dssat_empty_crop_rows_removed", np.nan),
        "dssat_variable_aliases_applied": dssat_preprocess_attrs.get("dssat_variable_aliases_applied", "{}"),

        "final_yield_source_col": final_yield_source_col,
        "final_yield_candidate_cols": ";".join(final_yield_col_candidates),

        "qc_min_final_yield": MIN_FINAL_YIELD,
        "qc_max_final_yield": MAX_FINAL_YIELD,
        "qc_min_crop_duration_days": MIN_CROP_DURATION_DAYS,
        "qc_max_crop_duration_days": MAX_CROP_DURATION_DAYS,

        "start_date": start_date,
        "end_date": end_date,
        "total_date_days": total_date_days,

        "min_DAP": min_dap,
        "max_DAP": max_dap,
        "crop_duration_days": crop_duration,
        "valid_crop_days": valid_crop_days,

        "final_yield_GWAD": final_yield,
        "max_LAI": max_lai,
        "max_CWAD": max_cwad,
        "max_RWAD": max_rwad,
        "final_HIAD": final_hiad,

        "GSTD_stage_count": gstd_stage_count,

        "passed_crop_qc": passed_crop_qc,
        "crop_qc_fail_reasons": ";".join(fail_reasons)
    }

    return quality_record


def summarize_climate_match_quality(matched):

    n_total = len(matched)

    if n_total == 0:
        return {
            "climate_match_days_spei": 0,
            "climate_match_days_ldfai": 0,
            "climate_match_ratio": np.nan,
            "passed_climate_qc": False,
            "climate_qc_fail_reasons": "empty_matched_data"
        }

    n_spei_match = matched["spei"].notna().sum()
    n_ldfai_match = matched["ldfai"].notna().sum()

    match_ratio = min(n_spei_match, n_ldfai_match) / n_total

    fail_reasons = []

    if match_ratio < MIN_CLIMATE_MATCH_RATIO:
        fail_reasons.append("climate_match_ratio_too_low")

    passed = len(fail_reasons) == 0

    return {
        "climate_match_days_spei": int(n_spei_match),
        "climate_match_days_ldfai": int(n_ldfai_match),
        "climate_match_ratio": match_ratio,
        "passed_climate_qc": passed,
        "climate_qc_fail_reasons": ";".join(fail_reasons)
    }


# ============================================================
# 8. 物候阶段划分
# ============================================================

def assign_phenology_stage(df):

    df = df.copy()

    if "PlantGro_GSTD" in df.columns:
        gstd = pd.to_numeric(df["PlantGro_GSTD"], errors="coerce")

        if gstd.notna().sum() > 0:
            df["PlantGro_GSTD_num"] = gstd
            df["phenology_stage_raw"] = "unknown"
            mask = gstd.notna()
            df.loc[mask, "phenology_stage_raw"] = (
                "GSTD_" + gstd[mask].round(0).astype("Int64").astype(str)
            )
        else:
            df["phenology_stage_raw"] = "unknown"
    else:
        df["phenology_stage_raw"] = "unknown"

    if "PlantGro_DAP" in df.columns:
        dap = pd.to_numeric(df["PlantGro_DAP"], errors="coerce")
    elif "DAP" in df.columns:
        dap = pd.to_numeric(df["DAP"], errors="coerce")
    else:
        df["PlantGro_DAP_used"] = np.nan
        df["growth_progress"] = np.nan
        df["phenology_stage_group"] = "unknown"
        df["phenology_stage"] = df["phenology_stage_group"]
        return df

    df["PlantGro_DAP_used"] = dap

    max_dap = df["PlantGro_DAP_used"].max()

    if pd.isna(max_dap) or max_dap <= 0:
        df["growth_progress"] = np.nan
        df["phenology_stage_group"] = "unknown"
        df["phenology_stage"] = df["phenology_stage_group"]
        return df

    df["growth_progress"] = df["PlantGro_DAP_used"] / max_dap

    bins = [0, 0.30, 0.60, 0.85, 1.01]
    labels = [
        "early_stage",
        "middle_stage",
        "late_stage",
        "maturity_stage"
    ]

    df["phenology_stage_group"] = pd.cut(
        df["growth_progress"],
        bins=bins,
        labels=labels,
        include_lowest=True
    ).astype(str)

    df.loc[df["PlantGro_DAP_used"].isna(), "phenology_stage_group"] = "unknown"

    df["phenology_stage"] = df["phenology_stage_group"]

    return df


def get_stage_intervals(matched, stage_col="phenology_stage_group"):

    df = matched.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"]).copy()

    if stage_col not in df.columns:
        df[stage_col] = "unknown"

    stage_df = (
        df.groupby(stage_col, dropna=False)
        .agg(
            stage_start=("DATE", "min"),
            stage_end=("DATE", "max"),
            stage_days=("DATE", "count")
        )
        .reset_index()
        .rename(columns={stage_col: "window_phenology_stage"})
    )

    stage_df = stage_df.dropna(subset=["stage_start", "stage_end"]).copy()

    return stage_df


def assign_window_to_stage(window_start, window_end, stage_intervals):

    if stage_intervals.empty:
        return "unknown", 0

    tmp = stage_intervals.copy()

    tmp["overlap_start"] = tmp["stage_start"].apply(lambda x: max(x, window_start))
    tmp["overlap_end"] = tmp["stage_end"].apply(lambda x: min(x, window_end))

    tmp["overlap_days"] = (
        tmp["overlap_end"] - tmp["overlap_start"]
    ).dt.days + 1

    tmp.loc[tmp["overlap_days"] < 0, "overlap_days"] = 0

    if tmp["overlap_days"].max() <= 0:
        return "outside_crop_period", 0

    idx = tmp["overlap_days"].idxmax()

    return (
        tmp.loc[idx, "window_phenology_stage"],
        int(tmp.loc[idx, "overlap_days"])
    )


# ============================================================
# 9. 事件识别函数
# ============================================================

def build_ldfai_events_for_one_sample(
    matched,
    climate_site,
    site,
    time,
    scenario,
    stage_col="phenology_stage_group"
):

    df = matched.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"]).sort_values("DATE").copy()

    if df.empty:
        return pd.DataFrame()

    crop_start = df["DATE"].min()
    crop_end = df["DATE"].max()

    stage_intervals = get_stage_intervals(df, stage_col=stage_col)

    climate = climate_site.copy()
    climate["DATE"] = pd.to_datetime(climate["DATE"], errors="coerce")
    climate["LDFAI_window_start"] = pd.to_datetime(climate["LDFAI_window_start"], errors="coerce")
    climate["LDFAI_window_end"] = pd.to_datetime(climate["LDFAI_window_end"], errors="coerce")

    climate = climate[
        (climate["LDFAI_window_start"] <= crop_end) &
        (climate["LDFAI_window_end"] >= crop_start)
    ].copy()

    if climate.empty:
        return pd.DataFrame()

    event_records = []

    direction_specs = [
        {
            "event_type": "positive_DFAA",
            "flag_col": "LDFAI_DFAA_pos_flag",
            "peak_mode": "max"
        },
        {
            "event_type": "negative_DFAA",
            "flag_col": "LDFAI_DFAA_neg_flag",
            "peak_mode": "min"
        }
    ]

    for spec in direction_specs:

        event_type = spec["event_type"]
        flag_col = spec["flag_col"]
        peak_mode = spec["peak_mode"]

        d = climate[climate[flag_col] == True].copy()

        if d.empty:
            continue

        d = split_continuous_segments(
            d,
            date_col="DATE",
            merge_gap_days=LDFAI_MERGE_GAP_DAYS
        )

        for segment_id, e in d.groupby("segment_id"):

            e = e.copy()

            if peak_mode == "max":
                peak_idx = e["ldfai"].idxmax()
            else:
                peak_idx = e["ldfai"].idxmin()

            anchor_date = e.loc[peak_idx, "DATE"]
            anchor_value = e.loc[peak_idx, "ldfai"]

            event_window_end = anchor_date
            event_window_start = event_window_end - pd.to_timedelta(
                LDFAI_WINDOW_DAYS - 1,
                unit="D"
            )

            event_segment_start = e["DATE"].min()
            event_segment_end = e["DATE"].max()
            event_segment_days = int((event_segment_end - event_segment_start).days + 1)

            window_stage, overlap_days = assign_window_to_stage(
                window_start=event_window_start,
                window_end=event_window_end,
                stage_intervals=stage_intervals
            )

            event_records.append({
                "site_folder": site,
                "time": str(time),
                "scenario": scenario,

                "event_type": event_type,
                "event_class": "DFAA",

                "event_segment_start": event_segment_start,
                "event_segment_end": event_segment_end,
                "event_segment_days": event_segment_days,

                "anchor_date": anchor_date,
                "anchor_value": anchor_value,
                "anchor_variable": "ldfai",

                "event_window_start": event_window_start,
                "event_window_end": event_window_end,

                "window_phenology_stage": window_stage,
                "window_stage_overlap_days": overlap_days
            })

    event_df = pd.DataFrame(event_records)

    return event_df


def build_spei_events_for_one_sample(
    matched,
    climate_site,
    site,
    time,
    scenario,
    stage_col="phenology_stage_group"
):

    df = matched.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"]).sort_values("DATE").copy()

    if df.empty:
        return pd.DataFrame()

    crop_start = df["DATE"].min()
    crop_end = df["DATE"].max()

    stage_intervals = get_stage_intervals(df, stage_col=stage_col)

    climate = climate_site.copy()
    climate["DATE"] = pd.to_datetime(climate["DATE"], errors="coerce")

    climate = climate[
        (climate["DATE"] >= crop_start) &
        (climate["DATE"] <= crop_end)
    ].copy()

    if climate.empty:
        return pd.DataFrame()

    event_records = []

    specs = [
        {
            "event_type": "drought",
            "flag_col": "drought_candidate_flag",
            "peak_mode": "min"
        },
        {
            "event_type": "flood",
            "flag_col": "flood_candidate_flag",
            "peak_mode": "max"
        }
    ]

    for spec in specs:

        event_type = spec["event_type"]
        flag_col = spec["flag_col"]
        peak_mode = spec["peak_mode"]

        d = climate[climate[flag_col] == True].copy()

        if d.empty:
            continue

        d = split_continuous_segments(
            d,
            date_col="DATE",
            merge_gap_days=SPEI_MERGE_GAP_DAYS
        )

        for segment_id, e in d.groupby("segment_id"):

            e = e.copy()

            event_segment_start = e["DATE"].min()
            event_segment_end = e["DATE"].max()
            event_segment_days = int((event_segment_end - event_segment_start).days + 1)

            if event_segment_days < SPEI_MIN_EVENT_DAYS:
                continue

            if peak_mode == "min":
                peak_idx = e["spei"].idxmin()
            else:
                peak_idx = e["spei"].idxmax()

            anchor_date = e.loc[peak_idx, "DATE"]
            anchor_value = e.loc[peak_idx, "spei"]

            # 对于 SPEI 干旱/洪涝，事件窗口就是连续干旱/洪涝时段
            event_window_start = event_segment_start
            event_window_end = event_segment_end

            window_stage, overlap_days = assign_window_to_stage(
                window_start=event_window_start,
                window_end=event_window_end,
                stage_intervals=stage_intervals
            )

            event_records.append({
                "site_folder": site,
                "time": str(time),
                "scenario": scenario,

                "event_type": event_type,
                "event_class": "SPEI",

                "event_segment_start": event_segment_start,
                "event_segment_end": event_segment_end,
                "event_segment_days": event_segment_days,

                "anchor_date": anchor_date,
                "anchor_value": anchor_value,
                "anchor_variable": "spei",

                "event_window_start": event_window_start,
                "event_window_end": event_window_end,

                "window_phenology_stage": window_stage,
                "window_stage_overlap_days": overlap_days
            })

    event_df = pd.DataFrame(event_records)

    return event_df


def build_normal_events_for_one_sample(
    matched,
    climate_site,
    site,
    time,
    scenario,
    stage_col="phenology_stage_group"
):
    """
    正常事件：
        -1.5 <= SPEI <= 1.5
        -4 <= LDFAI <= 4
        不在任何 DFAA 60 天窗口内

    每个物候阶段最多选 NORMAL_MAX_EVENTS_PER_STAGE_PER_SAMPLE 个最长正常段。
    """

    df = matched.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"]).sort_values("DATE").copy()

    if df.empty:
        return pd.DataFrame()

    stage_intervals = get_stage_intervals(df, stage_col=stage_col)

    climate = climate_site.copy()
    climate["DATE"] = pd.to_datetime(climate["DATE"], errors="coerce")

    event_records = []

    for _, stage_row in stage_intervals.iterrows():

        stage = stage_row["window_phenology_stage"]
        stage_start = stage_row["stage_start"]
        stage_end = stage_row["stage_end"]

        c = climate[
            (climate["DATE"] >= stage_start) &
            (climate["DATE"] <= stage_end) &
            (climate["normal_candidate_flag"] == True)
        ].copy()

        if c.empty:
            continue

        c = split_continuous_segments(
            c,
            date_col="DATE",
            merge_gap_days=0
        )

        seg_records = []

        for segment_id, e in c.groupby("segment_id"):

            e = e.copy()

            event_segment_start = e["DATE"].min()
            event_segment_end = e["DATE"].max()
            event_segment_days = int((event_segment_end - event_segment_start).days + 1)

            if event_segment_days < NORMAL_MIN_SEGMENT_DAYS:
                continue

            anchor_date = center_date_of_interval(
                event_segment_start,
                event_segment_end
            )

            # anchor_value 使用 anchor_date 当天 SPEI
            anchor_row = e[e["DATE"] == anchor_date]

            if anchor_row.empty:
                anchor_value = e["spei"].mean()
            else:
                anchor_value = anchor_row["spei"].iloc[0]

            seg_records.append({
                "segment_id": segment_id,
                "event_segment_start": event_segment_start,
                "event_segment_end": event_segment_end,
                "event_segment_days": event_segment_days,
                "anchor_date": anchor_date,
                "anchor_value": anchor_value
            })

        if not seg_records:
            continue

        seg_df = pd.DataFrame(seg_records)

        # 每个阶段选择最长的几个正常段
        seg_df = seg_df.sort_values(
            "event_segment_days",
            ascending=False
        ).head(NORMAL_MAX_EVENTS_PER_STAGE_PER_SAMPLE)

        for _, seg in seg_df.iterrows():

            event_records.append({
                "site_folder": site,
                "time": str(time),
                "scenario": scenario,

                "event_type": "normal",
                "event_class": "normal",

                "event_segment_start": seg["event_segment_start"],
                "event_segment_end": seg["event_segment_end"],
                "event_segment_days": int(seg["event_segment_days"]),

                "anchor_date": seg["anchor_date"],
                "anchor_value": seg["anchor_value"],
                "anchor_variable": "spei",

                "event_window_start": seg["event_segment_start"],
                "event_window_end": seg["event_segment_end"],

                "window_phenology_stage": stage,
                "window_stage_overlap_days": int(seg["event_segment_days"])
            })

    event_df = pd.DataFrame(event_records)

    return event_df


def finalize_event_ids(event_df):

    if event_df.empty:
        return event_df

    event_df = event_df.sort_values(
        [
            "site_folder",
            "time",
            "scenario",
            "event_type",
            "anchor_date"
        ]
    ).reset_index(drop=True)

    event_df["event_id"] = (
        event_df["site_folder"].astype(str) + "_" +
        event_df["time"].astype(str) + "_" +
        event_df["scenario"].astype(str) + "_" +
        event_df["event_type"].astype(str) + "_E" +
        event_df.groupby(
            ["site_folder", "time", "scenario", "event_type"]
        ).cumcount().add(1).astype(str).str.zfill(3)
    )

    return event_df


def filter_events_by_strict_single_disaster_stage(event_df):
    """
    可选严格筛选：目标灾害发生在某一物候阶段时，要求该完整物候阶段内只有 1 次灾害。

    判断单位：
        site_folder × time × scenario × window_phenology_stage

    保留条件：
    - 对 STRICT_SINGLE_DISASTER_TARGET_EVENT_TYPES 中的目标事件：
        1) 同一物候阶段内该目标事件类型只出现 1 次；
        2) 同一物候阶段内所有灾害事件总数也必须是 1；
           因此不会同时存在相反方向 DFAA、SPEI 干旱或 SPEI 洪涝。
    - normal 事件不参与灾害计数，默认保留。
    - 不在 STRICT_SINGLE_DISASTER_TARGET_EVENT_TYPES 中的事件类型默认不筛选；
      如果希望 flood/drought 也按同样规则筛选，把它们加入
      STRICT_SINGLE_DISASTER_TARGET_EVENT_TYPES 即可。
    """

    if event_df.empty:
        return event_df

    if not ENABLE_STRICT_SINGLE_DISASTER_STAGE_FILTER:
        return event_df

    required_cols = [
        "site_folder",
        "time",
        "scenario",
        "event_type",
        "window_phenology_stage"
    ]

    missing_cols = [c for c in required_cols if c not in event_df.columns]
    if missing_cols:
        raise ValueError(
            "严格单一灾害阶段筛选缺少必要列："
            + ";".join(missing_cols)
        )

    df = event_df.copy()
    df["window_phenology_stage"] = df["window_phenology_stage"].astype(str)
    df["event_type"] = df["event_type"].astype(str)

    group_cols = [
        "site_folder",
        "time",
        "scenario",
        "window_phenology_stage"
    ]

    disaster_mask = df["event_type"].isin(STRICT_SINGLE_DISASTER_EVENT_TYPES)

    # 同一站点-年份-情景-阶段内，所有灾害事件数量。
    disaster_count = (
        df[disaster_mask]
        .groupby(group_cols)
        .size()
        .reset_index(name="strict_stage_total_disaster_events")
    )

    # 同一站点-年份-情景-阶段内，同一事件类型数量。
    same_type_count = (
        df[disaster_mask]
        .groupby(group_cols + ["event_type"])
        .size()
        .reset_index(name="strict_stage_same_event_type_events")
    )

    df = df.merge(disaster_count, on=group_cols, how="left")
    df = df.merge(same_type_count, on=group_cols + ["event_type"], how="left")

    df["strict_stage_total_disaster_events"] = (
        df["strict_stage_total_disaster_events"].fillna(0).astype(int)
    )
    df["strict_stage_same_event_type_events"] = (
        df["strict_stage_same_event_type_events"].fillna(0).astype(int)
    )

    target_mask = (
        df["event_type"].isin(STRICT_SINGLE_DISASTER_TARGET_EVENT_TYPES) &
        df["window_phenology_stage"].isin(STRICT_SINGLE_DISASTER_STAGES)
    )

    pass_mask = pd.Series(True, index=df.index)

    strict_pass_mask = (
        (df["strict_stage_same_event_type_events"] == 1) &
        (df["strict_stage_total_disaster_events"] == 1)
    )

    pass_mask.loc[target_mask] = strict_pass_mask.loc[target_mask]

    df["strict_single_disaster_stage_filter_enabled"] = True
    df["strict_single_disaster_stage_filter_target"] = target_mask
    df["strict_single_disaster_stage_filter_passed"] = pass_mask
    df["strict_single_disaster_stage_filter_reason"] = "not_target_or_not_checked"

    df.loc[
        target_mask & pass_mask,
        "strict_single_disaster_stage_filter_reason"
    ] = "passed_single_disaster_stage"

    df.loc[
        target_mask & (~pass_mask) & (df["strict_stage_same_event_type_events"] != 1),
        "strict_single_disaster_stage_filter_reason"
    ] = "target_event_count_not_equal_to_1"

    df.loc[
        target_mask & (~pass_mask) & (df["strict_stage_same_event_type_events"] == 1) &
        (df["strict_stage_total_disaster_events"] != 1),
        "strict_single_disaster_stage_filter_reason"
    ] = "other_disaster_event_in_same_stage"

    before_target = int(target_mask.sum())
    after_target = int((target_mask & pass_mask).sum())
    removed_target = before_target - after_target

    if before_target > 0:
        print(
            "  严格单一灾害阶段筛选："
            f"目标事件 {before_target} 个，保留 {after_target} 个，剔除 {removed_target} 个"
        )

    return df[pass_mask].copy().reset_index(drop=True)



def build_event_relative_daily(matched, event_df):
    """
    构建“事件发生阶段分组 + 完整前中后时期”的日尺度响应数据。

    优化思路：
    1) 先把完整 early + middle + late 响应窗口的横轴、阶段边界等字段一次性算好；
    2) 再与事件元数据做一次向量化 cross join；
    3) 默认按绘图独立样本单位去重，避免同一站点-年份-情景中同一事件类型/发生阶段
       重复复制完全相同的 full-period 曲线。

    注意：
    - event_total 仍然保留所有原始事件；
    - event_daily 默认保留绘图真正需要的独立曲线单位；
    - 如果需要逐事件级 event_daily 明细，把 DEDUPLICATE_FULL_RESPONSE_EVENT_DAILY_BY_UNIT 改为 False。
    """

    if event_df.empty:
        return pd.DataFrame()

    df = matched.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"]).copy()

    if df.empty:
        return pd.DataFrame()

    if "phenology_stage_group" not in df.columns:
        df = assign_phenology_stage(df)

    stage_intervals = get_stage_intervals(
        df,
        stage_col="phenology_stage_group"
    )

    if stage_intervals.empty:
        return pd.DataFrame()

    stage_intervals = stage_intervals[
        stage_intervals["window_phenology_stage"].isin(FULL_RESPONSE_STAGES)
    ].copy()

    available_stages = set(stage_intervals["window_phenology_stage"].astype(str))
    missing_stages = [s for s in FULL_RESPONSE_STAGES if s not in available_stages]
    if missing_stages:
        return pd.DataFrame()

    stage_intervals["stage_days"] = (
        stage_intervals["stage_end"] - stage_intervals["stage_start"]
    ).dt.days + 1

    full_start = stage_intervals["stage_start"].min()
    full_end = stage_intervals["stage_end"].max()
    full_days = int((full_end - full_start).days + 1)

    if full_days <= 0:
        return pd.DataFrame()

    boundary_info = {}
    for _, row in stage_intervals.iterrows():
        stage_name = str(row["window_phenology_stage"])
        st = pd.to_datetime(row["stage_start"])
        ed = pd.to_datetime(row["stage_end"])

        if full_days > 1:
            start_pct = (st - full_start).days / (full_days - 1) * 100.0
            end_pct = (ed - full_start).days / (full_days - 1) * 100.0
        else:
            start_pct = 0.0
            end_pct = 0.0

        boundary_info[stage_name] = {
            "start_date": st,
            "end_date": ed,
            "start_pct": float(np.clip(start_pct, 0, 100)),
            "end_pct": float(np.clip(end_pct, 0, 100))
        }

    full_sub_base = df[
        (df["DATE"] >= full_start) &
        (df["DATE"] <= full_end) &
        (df["phenology_stage_group"].isin(FULL_RESPONSE_STAGES))
    ].copy()

    if full_sub_base.empty:
        return pd.DataFrame()

    # 当前响应日自身属于哪个阶段。
    full_sub_base["response_phenology_stage"] = full_sub_base["phenology_stage_group"].astype(str)

    full_sub_base["full_period_start"] = full_start
    full_sub_base["full_period_end"] = full_end
    full_sub_base["full_period_days"] = full_days

    full_sub_base["full_period_relative_day"] = (
        full_sub_base["DATE"] - full_start
    ).dt.days

    if full_days > 1:
        full_sub_base["full_period_progress_pct"] = (
            full_sub_base["full_period_relative_day"] / (full_days - 1) * 100.0
        )
    else:
        full_sub_base["full_period_progress_pct"] = 0.0

    full_sub_base["full_period_progress_pct"] = full_sub_base["full_period_progress_pct"].clip(0, 100)

    if FULL_RESPONSE_X_MODE == "day_from_stage_start":
        full_sub_base["full_period_progress_bin"] = full_sub_base["full_period_relative_day"].astype(float)
    else:
        full_sub_base["full_period_progress_bin"] = (
            np.round(full_sub_base["full_period_progress_pct"] / FULL_RESPONSE_BIN_WIDTH_PCT)
            * FULL_RESPONSE_BIN_WIDTH_PCT
        )
        full_sub_base["full_period_progress_bin"] = full_sub_base["full_period_progress_bin"].clip(0, 100)

    for stage_name in FULL_RESPONSE_STAGES:
        info = boundary_info[stage_name]
        full_sub_base[f"{stage_name}_start_progress_pct"] = info["start_pct"]
        full_sub_base[f"{stage_name}_end_progress_pct"] = info["end_pct"]
        full_sub_base[f"{stage_name}_start_date"] = info["start_date"]
        full_sub_base[f"{stage_name}_end_date"] = info["end_date"]

    # 兼容旧字段名：这里表示完整前中后时期内的位置。
    full_sub_base["stage_progress_pct"] = full_sub_base["full_period_progress_pct"]
    full_sub_base["stage_progress_bin"] = full_sub_base["full_period_progress_bin"]
    full_sub_base["stage_relative_day"] = full_sub_base["full_period_relative_day"]
    full_sub_base["stage_start"] = full_start
    full_sub_base["stage_end"] = full_end
    full_sub_base["stage_days"] = full_days
    full_sub_base["event_relative_day"] = full_sub_base["full_period_relative_day"]

    event_df2 = event_df.copy()
    event_df2["window_phenology_stage"] = event_df2["window_phenology_stage"].astype(str)
    event_df2 = event_df2[
        event_df2["window_phenology_stage"].isin(EVENT_OCCURRENCE_STAGES)
    ].copy()

    if event_df2.empty:
        return pd.DataFrame()

    # 后续绘图的独立样本单位。
    event_df2["event_occurrence_stage"] = event_df2["window_phenology_stage"].astype(str)
    event_df2["event_occurrence_stage_label"] = event_df2["event_occurrence_stage"].map(
        EVENT_OCCURRENCE_STAGE_LABELS
    ).fillna(event_df2["event_occurrence_stage"])

    event_df2["full_period_event_type_unit_id"] = (
        event_df2["site_folder"].astype(str) + "_" +
        event_df2["time"].astype(str) + "_" +
        event_df2["scenario"].astype(str) + "_" +
        event_df2["event_occurrence_stage"].astype(str) + "_" +
        event_df2["event_type"].astype(str)
    )

    if DEDUPLICATE_FULL_RESPONSE_EVENT_DAILY_BY_UNIT:
        event_df2 = (
            event_df2
            .sort_values(["event_type", "event_occurrence_stage", "anchor_date", "event_id"])
            .drop_duplicates(subset=["full_period_event_type_unit_id"], keep="first")
            .reset_index(drop=True)
        )

    event_df2["anchor_day_from_full_period_start"] = (
        pd.to_datetime(event_df2["anchor_date"]) - full_start
    ).dt.days
    event_df2["event_window_start_day_from_full_period_start"] = (
        pd.to_datetime(event_df2["event_window_start"]) - full_start
    ).dt.days
    event_df2["event_window_end_day_from_full_period_start"] = (
        pd.to_datetime(event_df2["event_window_end"]) - full_start
    ).dt.days

    if full_days > 1:
        event_df2["anchor_full_period_progress_pct"] = (
            event_df2["anchor_day_from_full_period_start"] / (full_days - 1) * 100.0
        )
        event_df2["event_window_start_full_period_progress_pct"] = (
            event_df2["event_window_start_day_from_full_period_start"] / (full_days - 1) * 100.0
        )
        event_df2["event_window_end_full_period_progress_pct"] = (
            event_df2["event_window_end_day_from_full_period_start"] / (full_days - 1) * 100.0
        )
    else:
        event_df2["anchor_full_period_progress_pct"] = 0.0
        event_df2["event_window_start_full_period_progress_pct"] = 0.0
        event_df2["event_window_end_full_period_progress_pct"] = 0.0

    event_cols = [
        "event_id",
        "event_type",
        "event_class",
        "event_segment_start",
        "event_segment_end",
        "event_segment_days",
        "anchor_date",
        "anchor_value",
        "anchor_variable",
        "event_window_start",
        "event_window_end",
        "window_phenology_stage",
        "window_stage_overlap_days",
        "event_occurrence_stage",
        "event_occurrence_stage_label",
        "full_period_event_type_unit_id",
        "anchor_day_from_full_period_start",
        "event_window_start_day_from_full_period_start",
        "event_window_end_day_from_full_period_start",
        "anchor_full_period_progress_pct",
        "event_window_start_full_period_progress_pct",
        "event_window_end_full_period_progress_pct",
    ]

    # 避免与 full_sub_base 中的样本列重复，事件元数据中只保留真正的事件字段。
    event_meta = event_df2[[c for c in event_cols if c in event_df2.columns]].copy()
    event_meta["_join_key"] = 1

    base = full_sub_base.copy()
    base["_join_key"] = 1

    out = base.merge(event_meta, on="_join_key", how="inner")
    out = out.drop(columns=["_join_key"])

    out["stage_event_type_unit_id"] = out["full_period_event_type_unit_id"]

    return out

def summarize_by_phenology_stage(df, climate_site):

    group_cols = [
        "site_folder",
        "time",
        "scenario",
        "phenology_stage_group"
    ]

    summary_list = []

    for keys, g in df.groupby(group_cols, dropna=False):

        if not isinstance(keys, tuple):
            keys = (keys,)

        g = g.sort_values("DATE").copy()

        if g.empty:
            continue

        stage_start = g["DATE"].min()
        stage_end = g["DATE"].max()

        drought_days = int(g["drought_candidate_flag"].fillna(False).sum())
        flood_days = int(g["flood_candidate_flag"].fillna(False).sum())
        normal_days = int(g["normal_candidate_flag"].fillna(False).sum())

        pos_anchor_days = int(g["LDFAI_DFAA_pos_flag"].fillna(False).sum())
        neg_anchor_days = int(g["LDFAI_DFAA_neg_flag"].fillna(False).sum())

        rec = {}

        for col, val in zip(group_cols, keys):
            rec[col] = val

        rec.update({
            "stage_start": stage_start,
            "stage_end": stage_end,
            "stage_days": int((stage_end - stage_start).days + 1),

            "SPEI_mean": g["spei"].mean(),
            "SPEI_min": g["spei"].min(),
            "SPEI_max": g["spei"].max(),

            "LDFAI_mean": g["ldfai"].mean(),
            "LDFAI_max": g["ldfai"].max(),
            "LDFAI_min": g["ldfai"].min(),

            "drought_days": drought_days,
            "flood_days": flood_days,
            "normal_days": normal_days,

            "positive_DFAA_anchor_days": pos_anchor_days,
            "negative_DFAA_anchor_days": neg_anchor_days,

            "has_drought": drought_days > 0,
            "has_flood": flood_days > 0,
            "has_normal": normal_days > 0,
            "has_positive_DFAA_anchor_day": pos_anchor_days > 0,
            "has_negative_DFAA_anchor_day": neg_anchor_days > 0
        })

        summary_list.append(rec)

    if not summary_list:
        return pd.DataFrame()

    return pd.DataFrame(summary_list)


# ============================================================
# 11. 绘图函数
# ============================================================

# ------------------------------------------------------------
# 11.1 Nature 风格绘图参数
# ------------------------------------------------------------
# 说明：
# 1) 只生成事件类型对比图，不再生成单独事件类型响应图；
# 2) x 轴为物候阶段窗口，而不是事件窗口；
# 3) 对比图每条均值线均绘制 95% 置信区间；
# 4) 置信区间按 stage_event_type_unit_id 作为独立样本单位计算；
# 5) PNG 用于快速查看，PDF 用于论文排版中的矢量图。

NATURE_CONFIDENCE_LEVEL = 0.95
NATURE_CI_Z_VALUE = 1.96       # 近似 95% CI；若样本很小，可改成 t 分布临界值
NATURE_CI_ALPHA = 0.18
NATURE_MEAN_LINEWIDTH = 1.0
NATURE_RAW_LINEWIDTH = 0.45
NATURE_AXIS_LINEWIDTH = 0.8
NATURE_REFERENCE_LINEWIDTH = 0.8

# Nature 常用单双栏宽度：89 mm / 183 mm
NATURE_SINGLE_COL_WIDTH_MM = 89
NATURE_DOUBLE_COL_WIDTH_MM = 183

# 事件类型颜色：高对比、色盲友好，适合论文图
EVENT_TYPE_COLORS = {
    "positive_DFAA": "#D55E00",   # vermillion
    "negative_DFAA": "#0072B2",   # blue
    "drought": "#CC79A7",         # reddish purple
    "flood": "#009E73",           # bluish green
    "normal": "#4D4D4D"           # dark grey
}

EVENT_TYPE_LABELS = {
    "positive_DFAA": "Positive DFAA (LDFAI > 4)",
    "negative_DFAA": "Negative DFAA (LDFAI < -4)",
    "drought": "SPEI < -1 (Drought)",
    "flood": "SPEI > 1 (Flood)",
    "normal": "Normal"
}

# 三条线颜色：表示极端事件发生在前 / 中 / 后三个阶段。
OCCURRENCE_STAGE_COLORS = {
    "early_stage": "#D55E00",
    "middle_stage": "#0072B2",
    "late_stage": "#009E73"
}

SAVE_FIG_FORMATS = ["png", "pdf"]


def mm_to_inch(mm):
    return mm / 25.4


NATURE_COMPARISON_FIGSIZE = (
    mm_to_inch(NATURE_DOUBLE_COL_WIDTH_MM),
    mm_to_inch(105)
)

# 一行三列图的宽高。
# 三个子图分别为：DFAA、SPEI > 1 洪涝、SPEI < -1 干旱。
NATURE_ONE_ROW_COMPARISON_FIGSIZE = (
    mm_to_inch(NATURE_DOUBLE_COL_WIDTH_MM),
    mm_to_inch(62)
)

# 所有指标总图尺寸控制。
# 总图宽度固定为 Nature 双栏宽度；高度按指标行数自动增加。
NATURE_ALL_VARIABLES_FIG_WIDTH_MM = NATURE_DOUBLE_COL_WIDTH_MM
NATURE_ALL_VARIABLES_ROW_HEIGHT_MM = 21
NATURE_ALL_VARIABLES_MIN_HEIGHT_MM = 120


def set_nature_rcparams():
    """设置接近 Nature 期刊风格的 Matplotlib 全局参数。"""

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": NATURE_AXIS_LINEWIDTH,
        "xtick.major.width": NATURE_AXIS_LINEWIDTH,
        "ytick.major.width": NATURE_AXIS_LINEWIDTH,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": False,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none"
    })


set_nature_rcparams()


def format_nature_axes(ax):
    """统一轴样式：去掉上、右边框，保留简洁论文图风格。"""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(NATURE_AXIS_LINEWIDTH)
    ax.spines["bottom"].set_linewidth(NATURE_AXIS_LINEWIDTH)

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        width=NATURE_AXIS_LINEWIDTH,
        length=3,
        pad=2
    )


def get_full_response_x_axis_label():
    """返回总图共用的 x 轴标题，避免每个小图重复显示相同 x 轴名称。"""

    if FULL_RESPONSE_X_MODE == "day_from_stage_start":
        return "Days from early-stage start"
    return "Growth progress (%)"


def add_shared_x_axis_label(fig, fontsize=6.5, y=0.010):
    """只在整张图底部显示一次 x 轴名称。"""

    label = get_full_response_x_axis_label()

    try:
        return fig.supxlabel(label, y=y, fontsize=fontsize)
    except AttributeError:
        return fig.text(
            0.5,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def record_axis_y_bounds(ax, lower_values, upper_values=None):
    """
    记录当前小图真实绘制数据的 y 轴范围。

    这里记录的是均值线及其 95% CI 带的上下界；后续每个小图会只根据
    自己的数据单独设置 y 轴范围，不再与同一行其他小图强制共用 y 轴。
    """

    if upper_values is None:
        values = np.asarray(lower_values, dtype=float)
    else:
        values = np.concatenate([
            np.asarray(lower_values, dtype=float).ravel(),
            np.asarray(upper_values, dtype=float).ravel(),
        ])

    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    bounds = getattr(ax, "_subplot_y_data_bounds", [])
    bounds.append((float(np.nanmin(values)), float(np.nanmax(values))))
    setattr(ax, "_subplot_y_data_bounds", bounds)


def set_subplot_y_range_from_data(
    ax,
    padding_ratio=0.08,
    include_zero=True,
):
    """按当前小图的数据单独设置 y 轴范围，并保留适当留白。"""

    bounds = getattr(ax, "_subplot_y_data_bounds", [])
    if not bounds:
        return

    finite_bounds = [
        (lo, hi)
        for lo, hi in bounds
        if np.isfinite(lo) and np.isfinite(hi)
    ]
    if not finite_bounds:
        return

    y_min = min(lo for lo, _ in finite_bounds)
    y_max = max(hi for _, hi in finite_bounds)

    if include_zero:
        y_min = min(y_min, 0.0)
        y_max = max(y_max, 0.0)

    if np.isclose(y_min, y_max):
        pad = max(abs(y_min), abs(y_max), 1.0) * padding_ratio
    else:
        pad = (y_max - y_min) * padding_ratio

    ax.set_ylim(y_min - pad, y_max + pad)


def summarize_event_response_curve(
    sub,
    variable,
    x_col=PHENOLOGY_STAGE_X_COL,
    unit_col=PHENOLOGY_STAGE_UNIT_COL
):
    """
    按“物候阶段窗口 x 轴”计算响应曲线均值和 95% 置信区间。

    重要：
    - x_col 默认使用 stage_progress_bin，即物候阶段进程 0–100%；
    - unit_col 默认使用 stage_event_type_unit_id，避免同一样本同一阶段同一事件类型
      因多个事件被重复加权；
    - 先对 unit_col × x_col 求均值，再计算不同独立样本单位之间的 mean / SE / CI。
    """

    if unit_col not in sub.columns:
        unit_col = "event_id"

    d = sub[[unit_col, x_col, variable]].copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[variable] = pd.to_numeric(d[variable], errors="coerce")
    d = d.dropna(subset=[unit_col, x_col, variable]).copy()

    if d.empty:
        return pd.DataFrame()

    unit_x = (
        d.groupby([unit_col, x_col], as_index=False)[variable]
        .mean()
    )

    mean_curve = (
        unit_x.groupby(x_col)[variable]
        .agg(["mean", "count", "std"])
        .reset_index()
        .sort_values(x_col)
    )

    mean_curve = mean_curve.rename(columns={x_col: "x_value"})
    mean_curve["std"] = mean_curve["std"].fillna(0)
    mean_curve["se"] = mean_curve["std"] / np.sqrt(mean_curve["count"])
    mean_curve["ci95"] = NATURE_CI_Z_VALUE * mean_curve["se"]

    # count <= 1 时无法估计样本间不确定性；这里设为 0，避免 fill_between 报错。
    mean_curve.loc[mean_curve["count"] <= 1, "ci95"] = 0

    return mean_curve


def plot_mean_with_ci(
    ax,
    mean_curve,
    color,
    label,
    linewidth=NATURE_MEAN_LINEWIDTH,
    alpha_ci=NATURE_CI_ALPHA,
    zorder=3
):
    """绘制均值线和 95% 置信区间带。"""

    if mean_curve.empty:
        return False

    x = mean_curve["x_value"].astype(float).to_numpy()
    y = mean_curve["mean"].astype(float).to_numpy()
    ci = mean_curve["ci95"].astype(float).fillna(0).to_numpy()

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(ci)
    x = x[valid]
    y = y[valid]
    ci = ci[valid]

    if len(x) == 0:
        return False

    record_axis_y_bounds(ax, y - ci, y + ci)

    ax.fill_between(
        x,
        y - ci,
        y + ci,
        color=color,
        alpha=alpha_ci,
        linewidth=0,
        zorder=zorder - 1
    )

    ax.plot(
        x,
        y,
        color=color,
        linewidth=linewidth,
        label=label,
        zorder=zorder
    )

    return True


def save_nature_figure(fig, save_stem, formats=None):
    """同时保存 PNG 和 PDF。save_stem 不需要带扩展名。"""

    if formats is None:
        formats = SAVE_FIG_FORMATS

    saved_paths = []

    for ext in formats:
        save_path = f"{save_stem}.{ext}"
        if ext.lower() == "png":
            fig.savefig(save_path, dpi=600)
        else:
            fig.savefig(save_path)
        saved_paths.append(save_path)

    return saved_paths



def summarize_event_minus_normal_curve(
    df,
    variable,
    event_type,
    occurrence_stage,
    x_col=FULL_RESPONSE_X_COL,
    unit_col=FULL_RESPONSE_UNIT_COL
):
    """
    计算某个极端事件类型在某个发生阶段下的：极端事件 - normal 基线。

    当前版本使用“同站点宽松 normal 基线”，不再做完全一一配对：
    1) extreme 事件按独立样本单位 unit_col × x_col 先求均值；
    2) normal 事件先在同一个 site_folder、同一个 event_occurrence_stage、同一个 x_col 下
       汇总成 normal 基线；
    3) 每个 extreme 样本单位只减去同站点、同发生阶段的 normal 基线；
    4) 不再要求 extreme 和 normal 的 time / scenario / event_id 完全一致。

    这样既避免不同站点生长曲线混在一起相减，也避免“完全匹配”导致样本大量丢失。
    """

    if variable not in df.columns:
        return pd.DataFrame()

    if x_col not in df.columns:
        return pd.DataFrame()

    if unit_col not in df.columns:
        unit_col = "event_id"

    event_sub = df[
        (df["event_type"] == event_type) &
        (df["event_occurrence_stage"] == occurrence_stage)
    ].copy()

    normal_sub = df[
        (df["event_type"] == "normal") &
        (df["event_occurrence_stage"] == occurrence_stage)
    ].copy()

    if event_sub.empty or normal_sub.empty:
        return pd.DataFrame()

    # 只使用当前数据中实际存在的匹配列。默认是 site_folder + event_occurrence_stage。
    # 不把 time / scenario 放进匹配键，避免 normal 基线必须与 extreme 完全一一对应。
    match_cols = [c for c in NORMAL_BASELINE_MATCH_COLS if c in df.columns]

    if not match_cols:
        # 极端情况下没有 site_folder 等列，则退回到原来的 occurrence_stage 群体 normal 基线。
        match_cols = ["event_occurrence_stage"]

    required_event_cols = [unit_col, x_col, variable] + match_cols
    required_normal_cols = [unit_col, x_col, variable] + match_cols

    event_missing = [c for c in required_event_cols if c not in event_sub.columns]
    normal_missing = [c for c in required_normal_cols if c not in normal_sub.columns]

    if event_missing or normal_missing:
        return pd.DataFrame()

    e = event_sub[required_event_cols].copy()
    n = normal_sub[required_normal_cols].copy()

    e[x_col] = pd.to_numeric(e[x_col], errors="coerce")
    n[x_col] = pd.to_numeric(n[x_col], errors="coerce")
    e[variable] = pd.to_numeric(e[variable], errors="coerce")
    n[variable] = pd.to_numeric(n[variable], errors="coerce")

    e = e.dropna(subset=[unit_col, x_col, variable] + match_cols).copy()
    n = n.dropna(subset=[unit_col, x_col, variable] + match_cols).copy()

    if e.empty or n.empty:
        return pd.DataFrame()

    # extreme：先在独立事件样本单位 × x 上求均值，避免同一事件重复加权。
    event_unit_cols = match_cols + [unit_col, x_col]
    event_unit_x = (
        e.groupby(event_unit_cols, as_index=False)[variable]
        .mean()
        .rename(columns={variable: "event_value"})
    )

    # normal：先在 normal 独立样本单位 × x 上求均值。
    normal_unit_cols = match_cols + [unit_col, x_col]
    normal_unit_x = (
        n.groupby(normal_unit_cols, as_index=False)[variable]
        .mean()
        .rename(columns={variable: "normal_unit_value"})
    )

    # normal 基线：在同站点 / 同发生阶段 / 同 x 下，对 normal 单位取均值。
    # 这里就是“宽松匹配”的核心：不要求 time、scenario、event_id 与 extreme 完全一致。
    normal_baseline = (
        normal_unit_x
        .groupby(match_cols + [x_col])
        .agg(
            normal_value=("normal_unit_value", "mean"),
            normal_units_available=(unit_col, "nunique")
        )
        .reset_index()
    )

    paired = event_unit_x.merge(
        normal_baseline,
        on=match_cols + [x_col],
        how="inner"
    )

    if paired.empty:
        return pd.DataFrame()

    paired["diff_value"] = paired["event_value"] - paired["normal_value"]

    # 对配对后的 extreme-normal 差值按 x 汇总。
    # count 表示在该 x 上成功找到同站点 normal 基线的 extreme 独立样本数。
    mean_curve = (
        paired.groupby(x_col)
        .agg(
            mean=("diff_value", "mean"),
            count=(unit_col, "nunique"),
            std=("diff_value", "std"),
            mean_event=("event_value", "mean"),
            mean_normal=("normal_value", "mean"),
            normal_units=("normal_units_available", "sum")
        )
        .reset_index()
        .rename(columns={x_col: "x_value"})
        .sort_values("x_value")
    )

    if mean_curve.empty:
        return pd.DataFrame()

    mean_curve["std"] = mean_curve["std"].fillna(0)
    mean_curve["se"] = mean_curve["std"] / np.sqrt(mean_curve["count"])
    mean_curve["ci95"] = NATURE_CI_Z_VALUE * mean_curve["se"]
    mean_curve.loc[mean_curve["count"] <= 1, "ci95"] = 0

    mean_curve["event_type"] = event_type
    mean_curve["event_occurrence_stage"] = occurrence_stage
    mean_curve["event_units"] = mean_curve["count"]
    mean_curve["count_event"] = mean_curve["count"]
    mean_curve["count_normal"] = mean_curve["normal_units"]
    mean_curve["normal_baseline_mode"] = NORMAL_BASELINE_MODE
    mean_curve["normal_baseline_match_cols"] = "+".join(match_cols)

    # 记录配对覆盖信息，便于检查宽松同站点基线是否真正生效。
    if "site_folder" in match_cols:
        paired_sites_by_x = (
            paired.groupby(x_col)["site_folder"]
            .nunique()
            .reset_index()
            .rename(columns={x_col: "x_value", "site_folder": "paired_site_count"})
        )
        mean_curve = mean_curve.merge(paired_sites_by_x, on="x_value", how="left")
    else:
        mean_curve["paired_site_count"] = np.nan

    return mean_curve[
        [
            "x_value",
            "mean",
            "ci95",
            "event_type",
            "event_occurrence_stage",
            "event_units",
            "normal_units",
            "mean_event",
            "mean_normal",
            "count_event",
            "count_normal",
            "paired_site_count",
            "normal_baseline_mode",
            "normal_baseline_match_cols"
        ]
    ].copy()

def get_full_period_boundary_lines(df):
    """返回完整前中后时期中的阶段边界位置，用于画 early|middle、middle|late 分界线。"""

    boundaries = []

    if "early_stage_end_progress_pct" in df.columns:
        val = pd.to_numeric(df["early_stage_end_progress_pct"], errors="coerce").median()
        if pd.notna(val):
            boundaries.append((float(val), "Early | Middle"))

    if "middle_stage_end_progress_pct" in df.columns:
        val = pd.to_numeric(df["middle_stage_end_progress_pct"], errors="coerce").median()
        if pd.notna(val):
            boundaries.append((float(val), "Middle | Late"))

    return boundaries


def plot_event_type_comparison_by_stage(
    event_daily_total,
    variable,
    output_dir,
    min_events_per_group=3
):
    """
    绘制“一行三列”的极端事件 - 正常组差值图。

    每个作物变量只生成一张图，三列分别为：
    1) DFAA：positive_DFAA 或 negative_DFAA，由 DFAA_PLOT_DIRECTION 控制；
    2) SPEI > 1 洪涝；
    3) SPEI < -1 干旱。

    每个子图内仍然保留三条主线：
    - 事件发生在 early_stage 的差值曲线；
    - 事件发生在 middle_stage 的差值曲线；
    - 事件发生在 late_stage 的差值曲线。

    y 轴为 极端事件组均值 - 正常组均值，所以曲线围绕 0 波动。
    """

    df = event_daily_total.copy()

    if variable not in df.columns:
        print(f"跳过一行三列差值图，缺少变量：{variable}")
        return

    required_cols = [
        variable,
        FULL_RESPONSE_X_COL,
        "event_type",
        "event_occurrence_stage"
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"跳过一行三列差值图，缺少必要列：{missing_cols}")
        return

    df[variable] = pd.to_numeric(df[variable], errors="coerce")
    df[FULL_RESPONSE_X_COL] = pd.to_numeric(df[FULL_RESPONSE_X_COL], errors="coerce")

    df = df.dropna(subset=[variable, FULL_RESPONSE_X_COL]).copy()
    df = df[df["event_occurrence_stage"].isin(EVENT_OCCURRENCE_STAGES)].copy()

    if df.empty:
        print(f"跳过一行三列差值图，变量无有效数据：{variable}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 每次绘图时重新读取选项，方便用户只修改 DFAA_PLOT_DIRECTION 后直接运行。
    plot_event_types = get_one_row_plot_event_types()
    dfaa_event_type = plot_event_types[0]

    fig, axes = plt.subplots(
        1,
        len(plot_event_types),
        figsize=NATURE_ONE_ROW_COMPARISON_FIGSIZE,
        sharex=True,
        sharey=False
    )

    if len(plot_event_types) == 1:
        axes = [axes]
    else:
        axes = np.ravel(axes)

    diff_curve_records = []
    plotted_any = False

    for panel_idx, (ax, event_type) in enumerate(zip(axes, plot_event_types)):

        plotted_this_panel = False
        plotted_stage_count = 0

        for occurrence_stage in EVENT_OCCURRENCE_STAGES:

            event_sub = df[
                (df["event_type"] == event_type) &
                (df["event_occurrence_stage"] == occurrence_stage)
            ].copy()

            normal_sub = df[
                (df["event_type"] == "normal") &
                (df["event_occurrence_stage"] == occurrence_stage)
            ].copy()

            if event_sub.empty or normal_sub.empty:
                continue

            unit_col = FULL_RESPONSE_UNIT_COL if FULL_RESPONSE_UNIT_COL in df.columns else "event_id"
            n_event_units = event_sub[unit_col].nunique()
            n_normal_units = normal_sub[unit_col].nunique()
            n_event_events = event_sub["event_id"].nunique()
            n_normal_events = normal_sub["event_id"].nunique()
            n_sites = event_sub["site_folder"].nunique()

            if (n_event_units < min_events_per_group) or (n_normal_units < min_events_per_group):
                continue

            diff_curve = summarize_event_minus_normal_curve(
                df=df,
                variable=variable,
                event_type=event_type,
                occurrence_stage=occurrence_stage,
                x_col=FULL_RESPONSE_X_COL,
                unit_col=unit_col
            )

            if diff_curve.empty:
                continue

            diff_curve["variable"] = variable
            diff_curve["panel_event_type"] = event_type
            diff_curve["dfaa_plot_direction"] = DFAA_PLOT_DIRECTION
            diff_curve_records.append(diff_curve)

            color = OCCURRENCE_STAGE_COLORS.get(occurrence_stage, "#333333")
            stage_label = EVENT_OCCURRENCE_STAGE_LABELS.get(occurrence_stage, occurrence_stage)
            stage_label = (
                stage_label
                .replace("Occurred in ", "")
                .replace(" stage", "")
                .title()
            )

            label = (
                f"{stage_label} "
                f"(sites={n_sites}, E={n_event_units}, N={n_normal_units})"
            )

            did_plot = plot_mean_with_ci(
                ax=ax,
                mean_curve=diff_curve,
                color=color,
                label=label,
                linewidth=NATURE_MEAN_LINEWIDTH,
                alpha_ci=NATURE_CI_ALPHA,
                zorder=3 + plotted_stage_count
            )

            if did_plot:
                plotted_any = True
                plotted_this_panel = True
                plotted_stage_count += 1

        # 完整前中后时期边界。
        if FULL_RESPONSE_X_MODE == "progress_percent":
            ax.axvline(
                0,
                color="0.35",
                linestyle=":",
                linewidth=0.7,
                zorder=1
            )
            ax.axvline(
                100,
                color="0.35",
                linestyle=":",
                linewidth=0.7,
                zorder=1
            )
            ax.set_xlim(0, 100)

            for boundary_x, boundary_label in get_full_period_boundary_lines(df):
                ax.axvline(
                    boundary_x,
                    color="0.55",
                    linestyle="--",
                    linewidth=0.65,
                    zorder=1
                )
                ax.text(
                    boundary_x,
                    0.98,
                    boundary_label,
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=5.5,
                    color="0.35",
                    transform=ax.get_xaxis_transform()
                )

        ax.axhline(
            0,
            color="0.20",
            linestyle=":",
            linewidth=0.8,
            zorder=1
        )

        event_label = EVENT_TYPE_LABELS.get(event_type, event_type)
        ax.set_title(event_label)

        if panel_idx == 0:
            ax.set_ylabel(f"Δ {get_y_axis_label(variable)}")
        else:
            ax.set_ylabel("")

        # x 轴名称在整张图底部统一显示，避免重复。
        ax.set_xlabel("")

        format_nature_axes(ax)
        set_subplot_y_range_from_data(ax)

        if plotted_this_panel:
            ax.legend(
                loc="best",
                handlelength=1.2,
                borderaxespad=0.35,
                labelspacing=0.28,
                fontsize=5.5
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No valid group\nunder current filter",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=7,
                color="0.35"
            )

    if not plotted_any:
        plt.close(fig)
        print(
            f"跳过一行三列差值图：{variable}；"
            "DFAA / SPEI 洪涝 / SPEI 干旱均没有达到最小样本数要求。"
        )
        return

    fig.suptitle(
        f"Extreme events minus Normal | {variable}\n"
        f"DFAA option = {DFAA_PLOT_DIRECTION}; lines = occurrence stage; "
        "window = early + middle + late",
        y=1.05,
        fontsize=8
    )

    add_shared_x_axis_label(fig, fontsize=6.5, y=0.015)
    fig.tight_layout(rect=[0, 0.070, 1, 1])

    save_stem = os.path.join(
        output_dir,
        f"full_period_delta_vs_normal_one_row_"
        f"{safe_filename(dfaa_event_type)}_flood_drought_"
        f"{safe_filename(variable)}"
    )

    saved_paths = save_nature_figure(fig, save_stem)
    plt.close(fig)

    for p in saved_paths:
        print(f"已保存一行三列差值图：{p}")

    # 每个变量保存一份差值曲线表，便于检查三列图中的三条线是否围绕 0 波动。
    if diff_curve_records:
        diff_total = pd.concat(diff_curve_records, ignore_index=True)
        diff_path = os.path.join(
            output_dir,
            f"full_period_delta_curves_one_row_"
            f"{safe_filename(dfaa_event_type)}_flood_drought_"
            f"{safe_filename(variable)}.csv"
        )
        diff_total.to_csv(diff_path, index=False, encoding="utf-8-sig")
        print(f"已保存一行三列差值曲线表：{diff_path}")



def plot_all_variables_one_figure_by_stage(
    event_daily_total,
    variables,
    output_dir,
    min_events_per_group=3
):
    """
    绘制“所有指标总图”：
    - 每个指标一行；
    - 三列依次为 DFAA、SPEI > 1 洪涝、SPEI < -1 干旱；
    - 每个子图内三条线表示事件发生在 early / middle / late；
    - y 轴为 极端事件组均值 - 正常组均值。
    """

    df = event_daily_total.copy()

    if df.empty:
        print("没有事件响应数据，跳过所有指标总图。")
        return

    required_cols = [
        FULL_RESPONSE_X_COL,
        "event_type",
        "event_occurrence_stage",
        "event_id",
        "site_folder"
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"跳过所有指标总图，缺少必要列：{missing_cols}")
        return

    plot_variables = [v for v in variables if v in df.columns]
    missing_variables = [v for v in variables if v not in df.columns]

    if missing_variables:
        print("以下指标不在 event_daily_total 中，绘图时跳过：")
        print(missing_variables)

    if not plot_variables:
        print("PLOT_VARIABLES 中没有任何指标存在于 event_daily_total，跳过所有指标总图。")
        return

    df[FULL_RESPONSE_X_COL] = pd.to_numeric(df[FULL_RESPONSE_X_COL], errors="coerce")
    df = df.dropna(subset=[FULL_RESPONSE_X_COL]).copy()
    df = df[df["event_occurrence_stage"].isin(EVENT_OCCURRENCE_STAGES)].copy()

    if df.empty:
        print("事件响应数据在筛选 occurrence stage 后为空，跳过所有指标总图。")
        return

    os.makedirs(output_dir, exist_ok=True)

    plot_event_types = get_one_row_plot_event_types()
    dfaa_event_type = plot_event_types[0]

    n_rows = len(plot_variables)
    n_cols = len(plot_event_types)

    fig_height_mm = max(
        NATURE_ALL_VARIABLES_MIN_HEIGHT_MM,
        NATURE_ALL_VARIABLES_ROW_HEIGHT_MM * n_rows
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            mm_to_inch(NATURE_ALL_VARIABLES_FIG_WIDTH_MM),
            mm_to_inch(fig_height_mm)
        ),
        sharex=True,
        squeeze=False
    )

    diff_curve_records = []
    plotted_any = False

    boundary_lines = get_full_period_boundary_lines(df)

    for row_idx, variable in enumerate(plot_variables):

        df_var = df.copy()
        df_var[variable] = pd.to_numeric(df_var[variable], errors="coerce")
        df_var = df_var.dropna(subset=[variable]).copy()

        row_plotted_any = False

        for col_idx, event_type in enumerate(plot_event_types):

            ax = axes[row_idx, col_idx]

            if row_idx == 0:
                event_label = EVENT_TYPE_LABELS.get(event_type, event_type)
                ax.set_title(event_label, fontsize=7.5)

            if df_var.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No valid\nvariable data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="0.35"
                )
                format_nature_axes(ax)
                set_subplot_y_range_from_data(ax)
                continue

            plotted_this_panel = False
            plotted_stage_count = 0

            for occurrence_stage in EVENT_OCCURRENCE_STAGES:

                event_sub = df_var[
                    (df_var["event_type"] == event_type) &
                    (df_var["event_occurrence_stage"] == occurrence_stage)
                ].copy()

                normal_sub = df_var[
                    (df_var["event_type"] == "normal") &
                    (df_var["event_occurrence_stage"] == occurrence_stage)
                ].copy()

                if event_sub.empty or normal_sub.empty:
                    continue

                unit_col = FULL_RESPONSE_UNIT_COL if FULL_RESPONSE_UNIT_COL in df_var.columns else "event_id"
                n_event_units = event_sub[unit_col].nunique()
                n_normal_units = normal_sub[unit_col].nunique()

                if (n_event_units < min_events_per_group) or (n_normal_units < min_events_per_group):
                    continue

                diff_curve = summarize_event_minus_normal_curve(
                    df=df_var,
                    variable=variable,
                    event_type=event_type,
                    occurrence_stage=occurrence_stage,
                    x_col=FULL_RESPONSE_X_COL,
                    unit_col=unit_col
                )

                if diff_curve.empty:
                    continue

                diff_curve["variable"] = variable
                diff_curve["panel_event_type"] = event_type
                diff_curve["dfaa_plot_direction"] = DFAA_PLOT_DIRECTION
                diff_curve_records.append(diff_curve)

                color = OCCURRENCE_STAGE_COLORS.get(occurrence_stage, "#333333")

                did_plot = plot_mean_with_ci(
                    ax=ax,
                    mean_curve=diff_curve,
                    color=color,
                    label=EVENT_OCCURRENCE_STAGE_LABELS.get(occurrence_stage, occurrence_stage),
                    linewidth=NATURE_MEAN_LINEWIDTH,
                    alpha_ci=NATURE_CI_ALPHA,
                    zorder=3 + plotted_stage_count
                )

                if did_plot:
                    plotted_any = True
                    row_plotted_any = True
                    plotted_this_panel = True
                    plotted_stage_count += 1

            # 完整 early-middle-late 响应窗口辅助线。
            if FULL_RESPONSE_X_MODE == "progress_percent":
                ax.axvline(0, color="0.35", linestyle=":", linewidth=0.55, zorder=1)
                ax.axvline(100, color="0.35", linestyle=":", linewidth=0.55, zorder=1)
                ax.set_xlim(0, 100)

                for boundary_x, boundary_label in boundary_lines:
                    ax.axvline(
                        boundary_x,
                        color="0.60",
                        linestyle="--",
                        linewidth=0.45,
                        zorder=1
                    )
                    # 只在第一行标注阶段边界，避免 15 行图中重复文字太多。
                    if row_idx == 0:
                        ax.text(
                            boundary_x,
                            0.98,
                            boundary_label,
                            rotation=90,
                            va="top",
                            ha="right",
                            fontsize=4.8,
                            color="0.35",
                            transform=ax.get_xaxis_transform()
                        )

            ax.axhline(
                0,
                color="0.20",
                linestyle=":",
                linewidth=0.55,
                zorder=1
            )

            if not plotted_this_panel:
                ax.text(
                    0.5,
                    0.5,
                    "No valid group",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=5.2,
                    color="0.35"
                )

            if col_idx == 0:
                ax.set_ylabel(f"Δ {get_y_axis_label(variable)}", fontsize=6.3)
            else:
                ax.set_ylabel("")

            # x 轴名称在整张图底部统一显示，避免重复。
            ax.set_xlabel("")

            format_nature_axes(ax)
            set_subplot_y_range_from_data(ax)

        # 每个小图已在 set_subplot_y_range_from_data() 中按自身数据单独设置 y 轴范围。

    if not plotted_any:
        plt.close(fig)
        print(
            "跳过所有指标总图：DFAA / SPEI 洪涝 / SPEI 干旱均没有达到最小样本数要求。"
        )
        return

    legend_handles = []
    for occurrence_stage in EVENT_OCCURRENCE_STAGES:
        label = EVENT_OCCURRENCE_STAGE_LABELS.get(occurrence_stage, occurrence_stage)
        label = label.replace("Occurred in ", "").replace(" stage", "").title()
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=OCCURRENCE_STAGE_COLORS.get(occurrence_stage, "#333333"),
                linewidth=NATURE_MEAN_LINEWIDTH,
                label=label
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=1.4,
        columnspacing=1.3,
        fontsize=6.3
    )

    fig.suptitle(
        f"Extreme events minus Normal | all variables\n"
        f"DFAA option = {DFAA_PLOT_DIRECTION}; rows = variables; columns = DFAA / flood / drought; "
        "lines = occurrence stage",
        y=1.018,
        fontsize=8
    )

    add_shared_x_axis_label(fig, fontsize=6.5, y=0.008)
    fig.tight_layout(rect=[0, 0.040, 1, 0.975], h_pad=0.35, w_pad=0.35)

    save_stem = os.path.join(
        output_dir,
        f"full_period_delta_vs_normal_ALL_VARIABLES_grid_"
        f"{safe_filename(dfaa_event_type)}_flood_drought"
    )

    saved_paths = save_nature_figure(fig, save_stem)
    plt.close(fig)

    for p in saved_paths:
        print(f"已保存所有指标总图：{p}")

    if diff_curve_records:
        diff_total = pd.concat(diff_curve_records, ignore_index=True)
        diff_path = os.path.join(
            output_dir,
            f"full_period_delta_curves_ALL_VARIABLES_grid_"
            f"{safe_filename(dfaa_event_type)}_flood_drought.csv"
        )
        diff_total.to_csv(diff_path, index=False, encoding="utf-8-sig")
        print(f"已保存所有指标总图对应差值曲线表：{diff_path}")




def plot_dfai_positive_negative_one_figure_by_stage(
    positive_event_daily_total,
    negative_event_daily_total,
    variables,
    output_dir,
    min_events_per_group=3
):
    """
    绘制 DFAI > 4 与 DFAI < -4 的总图。

    数据来源：
    - DFAI > 4 / positive_DFAA：读取 DSSAT_DIR_POSITIVE_DFAI；
    - DFAI < -4 / negative_DFAA：读取 DSSAT_DIR_NEGATIVE_DFAI。

    图形结构：
    - 每个指标一行；
    - 两列依次为 DFAI > 4 和 DFAI < -4；
    - 每个子图内三条线表示事件发生在 early / middle / late；
    - y 轴为 DFAA 事件组均值 - 正常组均值。
    """

    pos_df = positive_event_daily_total.copy()
    neg_df = negative_event_daily_total.copy()

    if pos_df.empty or neg_df.empty:
        print("DFAI > 4 或 DFAI < -4 的事件响应数据为空，跳过正负 DFAI 总图。")
        return

    source_specs = [
        {
            "event_type": "positive_DFAA",
            "df": pos_df,
            "title": "DFAI > 4",
            "source_dir": DSSAT_DIR_POSITIVE_DFAI
        },
        {
            "event_type": "negative_DFAA",
            "df": neg_df,
            "title": "DFAI < -4",
            "source_dir": DSSAT_DIR_NEGATIVE_DFAI
        }
    ]

    required_cols = [
        FULL_RESPONSE_X_COL,
        "event_type",
        "event_occurrence_stage",
        "event_id",
        "site_folder"
    ]

    for spec in source_specs:
        missing_cols = [c for c in required_cols if c not in spec["df"].columns]
        if missing_cols:
            print(f"跳过正负 DFAI 总图，{spec['title']} 数据缺少必要列：{missing_cols}")
            return

    plot_variables = [
        v for v in variables
        if v in pos_df.columns and v in neg_df.columns
    ]

    if not plot_variables:
        print("正负 DFAI 两套数据没有共同的 PLOT_VARIABLES 指标，跳过正负 DFAI 总图。")
        return

    os.makedirs(output_dir, exist_ok=True)

    n_rows = len(plot_variables)
    n_cols = len(source_specs)

    fig_height_mm = max(
        NATURE_ALL_VARIABLES_MIN_HEIGHT_MM,
        NATURE_ALL_VARIABLES_ROW_HEIGHT_MM * n_rows
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            mm_to_inch(NATURE_ALL_VARIABLES_FIG_WIDTH_MM),
            mm_to_inch(fig_height_mm)
        ),
        sharex=True,
        squeeze=False
    )

    diff_curve_records = []
    plotted_any = False

    # 两套 DSSAT 目录可能导致物候阶段边界略有差异；这里优先用正向 DFAI 数据的边界作为标注。
    boundary_df = pos_df.copy()
    boundary_df[FULL_RESPONSE_X_COL] = pd.to_numeric(
        boundary_df[FULL_RESPONSE_X_COL],
        errors="coerce"
    )
    boundary_df = boundary_df.dropna(subset=[FULL_RESPONSE_X_COL]).copy()
    boundary_lines = get_full_period_boundary_lines(boundary_df)

    for row_idx, variable in enumerate(plot_variables):

        row_plotted_any = False

        for col_idx, spec in enumerate(source_specs):

            ax = axes[row_idx, col_idx]
            event_type = spec["event_type"]
            df_var = spec["df"].copy()

            if row_idx == 0:
                ax.set_title(
                    f"{spec['title']}\n{os.path.basename(spec['source_dir'])}",
                    fontsize=7.5
                )

            df_var[FULL_RESPONSE_X_COL] = pd.to_numeric(
                df_var[FULL_RESPONSE_X_COL],
                errors="coerce"
            )
            df_var[variable] = pd.to_numeric(df_var[variable], errors="coerce")
            df_var = df_var.dropna(subset=[FULL_RESPONSE_X_COL, variable]).copy()
            df_var = df_var[
                df_var["event_occurrence_stage"].isin(EVENT_OCCURRENCE_STAGES)
            ].copy()

            if df_var.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No valid\nvariable data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="0.35"
                )
                format_nature_axes(ax)
                set_subplot_y_range_from_data(ax)
                continue

            plotted_this_panel = False
            plotted_stage_count = 0

            for occurrence_stage in EVENT_OCCURRENCE_STAGES:

                event_sub = df_var[
                    (df_var["event_type"] == event_type) &
                    (df_var["event_occurrence_stage"] == occurrence_stage)
                ].copy()

                normal_sub = df_var[
                    (df_var["event_type"] == "normal") &
                    (df_var["event_occurrence_stage"] == occurrence_stage)
                ].copy()

                if event_sub.empty or normal_sub.empty:
                    continue

                unit_col = FULL_RESPONSE_UNIT_COL if FULL_RESPONSE_UNIT_COL in df_var.columns else "event_id"
                n_event_units = event_sub[unit_col].nunique()
                n_normal_units = normal_sub[unit_col].nunique()

                if (n_event_units < min_events_per_group) or (n_normal_units < min_events_per_group):
                    continue

                diff_curve = summarize_event_minus_normal_curve(
                    df=df_var,
                    variable=variable,
                    event_type=event_type,
                    occurrence_stage=occurrence_stage,
                    x_col=FULL_RESPONSE_X_COL,
                    unit_col=unit_col
                )

                if diff_curve.empty:
                    continue

                diff_curve["variable"] = variable
                diff_curve["panel_event_type"] = event_type
                diff_curve["dfai_source_dir"] = spec["source_dir"]
                diff_curve_records.append(diff_curve)

                color = OCCURRENCE_STAGE_COLORS.get(occurrence_stage, "#333333")

                did_plot = plot_mean_with_ci(
                    ax=ax,
                    mean_curve=diff_curve,
                    color=color,
                    label=EVENT_OCCURRENCE_STAGE_LABELS.get(occurrence_stage, occurrence_stage),
                    linewidth=NATURE_MEAN_LINEWIDTH,
                    alpha_ci=NATURE_CI_ALPHA,
                    zorder=3 + plotted_stage_count
                )

                if did_plot:
                    plotted_any = True
                    row_plotted_any = True
                    plotted_this_panel = True
                    plotted_stage_count += 1

            if FULL_RESPONSE_X_MODE == "progress_percent":
                ax.axvline(0, color="0.35", linestyle=":", linewidth=0.55, zorder=1)
                ax.axvline(100, color="0.35", linestyle=":", linewidth=0.55, zorder=1)
                ax.set_xlim(0, 100)

                for boundary_x, boundary_label in boundary_lines:
                    ax.axvline(
                        boundary_x,
                        color="0.60",
                        linestyle="--",
                        linewidth=0.45,
                        zorder=1
                    )
                    if row_idx == 0:
                        ax.text(
                            boundary_x,
                            0.98,
                            boundary_label,
                            rotation=90,
                            va="top",
                            ha="right",
                            fontsize=4.8,
                            color="0.35",
                            transform=ax.get_xaxis_transform()
                        )

            ax.axhline(
                0,
                color="0.20",
                linestyle=":",
                linewidth=0.55,
                zorder=1
            )

            if not plotted_this_panel:
                ax.text(
                    0.5,
                    0.5,
                    "No valid group",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=5.2,
                    color="0.35"
                )

            if col_idx == 0:
                ax.set_ylabel(f"Δ {get_y_axis_label(variable)}", fontsize=6.3)
            else:
                ax.set_ylabel("")

            # x 轴名称在整张图底部统一显示，避免重复。
            ax.set_xlabel("")

            format_nature_axes(ax)
            set_subplot_y_range_from_data(ax)

        # 每个小图已在 set_subplot_y_range_from_data() 中按自身数据单独设置 y 轴范围。

    if not plotted_any:
        plt.close(fig)
        print("跳过正负 DFAI 总图：DFAI > 4 和 DFAI < -4 均没有达到最小样本数要求。")
        return

    legend_handles = []
    for occurrence_stage in EVENT_OCCURRENCE_STAGES:
        label = EVENT_OCCURRENCE_STAGE_LABELS.get(occurrence_stage, occurrence_stage)
        label = label.replace("Occurred in ", "").replace(" stage", "").title()
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=OCCURRENCE_STAGE_COLORS.get(occurrence_stage, "#333333"),
                linewidth=NATURE_MEAN_LINEWIDTH,
                label=label
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=1.4,
        columnspacing=1.3,
        fontsize=6.3
    )

    fig.suptitle(
        "DFAI positive and negative events minus Normal | all variables\n"
        "DFAI > 4 uses DSSAT_DAILY_+; DFAI < -4 uses DSSAT_DAILY; lines = occurrence stage",
        y=1.018,
        fontsize=8
    )

    add_shared_x_axis_label(fig, fontsize=6.5, y=0.008)
    fig.tight_layout(rect=[0, 0.040, 1, 0.975], h_pad=0.35, w_pad=0.35)

    save_stem = os.path.join(
        output_dir,
        "full_period_delta_vs_normal_ALL_VARIABLES_grid_DFAI_gt4_and_lt_minus4_two_dssat_dirs"
    )

    saved_paths = save_nature_figure(fig, save_stem)
    plt.close(fig)

    for p in saved_paths:
        print(f"已保存正负 DFAI 总图：{p}")

    if diff_curve_records:
        diff_total = pd.concat(diff_curve_records, ignore_index=True)
        diff_path = os.path.join(
            output_dir,
            "full_period_delta_curves_ALL_VARIABLES_grid_DFAI_gt4_and_lt_minus4_two_dssat_dirs.csv"
        )
        diff_total.to_csv(diff_path, index=False, encoding="utf-8-sig")
        print(f"已保存正负 DFAI 总图对应差值曲线表：{diff_path}")



# ============================================================
# 11.4 多作物合并总图函数
# ============================================================
# 目标：在完成每个作物各自的 positive / negative DFAI 数据处理后，
#      把四个作物放在同一张最终图里，并按 2 行 × 2 列的作物块排列。
#      每个作物块内部继续保留 DFAI / SPEI 事件列和变量行。

MULTI_CROP_PANEL_WIDTH_MM = 43
MULTI_CROP_MIN_FIG_WIDTH_MM = NATURE_DOUBLE_COL_WIDTH_MM

# 四个作物的版式：
# 第一行：Maize, Rice
# 第二行：Soybeans, Wheat
MULTI_CROP_GRID_NROWS = 2
MULTI_CROP_GRID_NCOLS = 2


def get_multi_crop_figure_size(n_cols, n_rows):
    """根据列数和指标行数自动放大合并图尺寸。"""

    fig_width_mm = max(
        MULTI_CROP_MIN_FIG_WIDTH_MM,
        MULTI_CROP_PANEL_WIDTH_MM * max(1, n_cols)
    )
    fig_height_mm = max(
        NATURE_ALL_VARIABLES_MIN_HEIGHT_MM,
        NATURE_ALL_VARIABLES_ROW_HEIGHT_MM * max(1, n_rows)
    )

    return mm_to_inch(fig_width_mm), mm_to_inch(fig_height_mm)


# ============================================================
# 多作物合并 CSV 内存优化配置
# ============================================================
# False：不再保存四作物合并后的 event_daily_total 超大明细表。
#       强烈建议保持 False，因为该表可能有 4000 万行以上，极易内存溢出。
#       每个作物、每个正负 DFAI 的 daily 表前面已经单独保存过。
# True：仍然保存四作物合并后的 event_daily_total，但采用逐块追加写入，
#      不再 pd.concat 全量合并。
SAVE_COMBINED_EVENT_DAILY_TABLE = False

# 逐块写 CSV 的块大小。内存小可以调低，比如 50_000。
COMBINED_CSV_CHUNKSIZE = 200_000


def get_crop_result_value(crop_result, key):
    """安全读取 crop_result 中的对象。"""

    if crop_result is None:
        return pd.DataFrame()

    value = crop_result.get(key, pd.DataFrame())
    if value is None:
        return pd.DataFrame()
    return value


def add_crop_column(df, crop_name, dfai_direction=None, run_label=None, copy_df=False):
    """
    给输出表添加 crop_name / dfai_direction / run_label，便于多作物合并保存。

    内存优化：
    - 默认 copy_df=False，不再对几十万 / 几千万行的大表做 df.copy()；
    - 如果你确实想保留原表完全不变，可手动传 copy_df=True。
    """

    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy() if copy_df else df

    if "crop_name" not in out.columns:
        out["crop_name"] = crop_name

    if dfai_direction is not None and "dfai_direction" not in out.columns:
        out["dfai_direction"] = dfai_direction

    if run_label is not None and "run_label" not in out.columns:
        out["run_label"] = run_label

    return out


def write_dataframe_to_csv_in_chunks(df, path, header=True, mode="w", columns=None):
    """
    把单个 DataFrame 写入 CSV。

    说明：
    - pandas 的 to_csv(chunksize=...) 会分块写出，降低峰值内存；
    - columns 不为 None 时，会按统一列顺序输出，避免不同作物列不一致。
    """

    if df is None or df.empty:
        return 0

    if columns is not None and list(df.columns) != list(columns):
        out_df = df.reindex(columns=columns)
    else:
        out_df = df

    out_df.to_csv(
        path,
        mode=mode,
        header=header,
        index=False,
        encoding="utf-8-sig" if mode == "w" else "utf-8",
        chunksize=COMBINED_CSV_CHUNKSIZE,
    )

    n_rows = int(len(out_df))

    if out_df is not df:
        del out_df

    return n_rows


def save_combined_crop_tables(crop_results, output_dir):
    """
    保存四个作物合并后的主要中间结果表。

    重要修改：
    1) 不再使用 pd.concat(pieces, ignore_index=True) 合并超大表；
    2) 默认跳过四作物合并 event_daily_total 超大 CSV；
    3) event_total / quality_df 等较小表也采用逐块追加写入；
    4) 即使开启 SAVE_COMBINED_EVENT_DAILY_TABLE=True，也采用分块写入，避免一次性占用巨大内存。
    """

    import gc

    os.makedirs(output_dir, exist_ok=True)

    table_specs = [
        {
            "positive_key": "positive_event_daily_total",
            "negative_key": "negative_event_daily_total",
            "filename": "combined_four_crops_event_daily_positive_negative.csv",
            "description": "完整前中后时期日尺度响应表",
            "is_huge_daily_table": True,
        },
        {
            "positive_key": "positive_event_total",
            "negative_key": "negative_event_total",
            "filename": "combined_four_crops_event_table_positive_negative.csv",
            "description": "事件表",
            "is_huge_daily_table": False,
        },
        {
            "positive_key": "positive_quality_df",
            "negative_key": "negative_quality_df",
            "filename": "combined_four_crops_quality_positive_negative.csv",
            "description": "质量筛选表",
            "is_huge_daily_table": False,
        },
    ]

    for table_spec in table_specs:

        # 默认跳过四作物 daily 超大总表。
        # 单作物 / 单方向 daily 表已经由 save_result_tables_for_run() 保存。
        if table_spec.get("is_huge_daily_table", False) and not SAVE_COMBINED_EVENT_DAILY_TABLE:
            print(
                f"跳过多作物合并{table_spec['description']}："
                "该表行数极大，容易导致内存溢出；"
                "已保留各作物、正负 DFAI 的单独 daily CSV。"
            )
            continue

        piece_infos = []

        for crop_result in crop_results:
            crop_name = crop_result.get("crop_name", "unknown_crop")

            pos_run_label = crop_result.get("positive_run_label", "positive")
            neg_run_label = crop_result.get("negative_run_label", "negative")

            for direction, run_label, key in [
                ("positive", pos_run_label, table_spec["positive_key"]),
                ("negative", neg_run_label, table_spec["negative_key"]),
            ]:
                df = get_crop_result_value(crop_result, key)

                if df is None or df.empty:
                    continue

                # 不复制大表，只在缺少列时补充标识列。
                df = add_crop_column(
                    df,
                    crop_name=crop_name,
                    dfai_direction=direction,
                    run_label=run_label,
                    copy_df=False,
                )

                piece_infos.append({
                    "df": df,
                    "crop_name": crop_name,
                    "direction": direction,
                    "run_label": run_label,
                })

        if not piece_infos:
            print(f"没有可合并的多作物{table_spec['description']}，跳过保存。")
            continue

        path = os.path.join(output_dir, table_spec["filename"])

        # 如果之前运行失败留下半个文件，先删除，避免 append 到旧文件后面。
        if os.path.exists(path):
            os.remove(path)

        # 收集所有列名，避免不同作物列不完全一致导致 CSV 列错位。
        all_columns = []
        seen_columns = set()

        for info in piece_infos:
            df = info["df"]
            for col in df.columns:
                if col not in seen_columns:
                    seen_columns.add(col)
                    all_columns.append(col)

        wrote_any = False
        total_rows = 0

        for info in piece_infos:
            df = info["df"]
            crop_name = info["crop_name"]
            direction = info["direction"]

            n_rows = int(len(df))
            print(
                f"  写入多作物合并{table_spec['description']}："
                f"{crop_name} / {direction}，{n_rows:,} 行"
            )

            written_rows = write_dataframe_to_csv_in_chunks(
                df=df,
                path=path,
                header=not wrote_any,
                mode="w" if not wrote_any else "a",
                columns=all_columns,
            )
            total_rows += written_rows
            wrote_any = True

            gc.collect()

        print(
            f"已保存多作物合并{table_spec['description']}：{path}，"
            f"总行数 {total_rows:,}"
        )

        gc.collect()

def plot_multi_crop_dfai_positive_negative_one_figure_by_stage(
    crop_results,
    variables,
    output_dir,
    min_events_per_group=3
):
    """
    绘制四作物 DFAI > 4 与 DFAI < -4 合并总图。

    图形结构：
    - 作物按 2 行 × 2 列排列；
    - 每个作物块内部：每个指标一行、两列分别为 DFAI > 4 和 DFAI < -4；
    - 每个子图内三条线表示事件发生在 early / middle / late；
    - y 轴为 DFAA 事件组均值 - 正常组均值。
    """

    crop_specs = []

    for crop_result in crop_results:
        crop_name = crop_result.get("crop_name", "unknown_crop")

        pos_df = get_crop_result_value(crop_result, "positive_event_daily_total")
        neg_df = get_crop_result_value(crop_result, "negative_event_daily_total")

        crop_specs.append({
            "crop_name": crop_name,
            "panels": [
                {
                    "crop_name": crop_name,
                    "event_type": "positive_DFAA",
                    "df": pos_df,
                    "title": "DFAI > 4",
                    "source_dir": crop_result.get("positive_dssat_dir", ""),
                },
                {
                    "crop_name": crop_name,
                    "event_type": "negative_DFAA",
                    "df": neg_df,
                    "title": "DFAI < -4",
                    "source_dir": crop_result.get("negative_dssat_dir", ""),
                },
            ],
        })

    if not crop_specs:
        print("没有可用于四作物正负 DFAI 合并总图的数据。")
        return

    source_specs = [panel for crop_spec in crop_specs for panel in crop_spec["panels"]]

    required_cols = [
        FULL_RESPONSE_X_COL,
        "event_type",
        "event_occurrence_stage",
        "event_id",
        "site_folder",
    ]

    for spec in source_specs:
        if spec["df"].empty:
            continue
        missing_cols = [c for c in required_cols if c not in spec["df"].columns]
        if missing_cols:
            print(
                f"跳过四作物正负 DFAI 合并总图，"
                f"{spec['crop_name']} {spec['title']} 数据缺少必要列：{missing_cols}"
            )
            return

    plot_variables = [
        v for v in variables
        if any((not spec["df"].empty) and (v in spec["df"].columns) for spec in source_specs)
    ]

    if not plot_variables:
        print("四作物正负 DFAI 合并总图没有可绘制指标，跳过。")
        return

    os.makedirs(output_dir, exist_ok=True)

    n_variable_rows = len(plot_variables)
    panels_per_crop = 2
    n_crops = len(crop_specs)
    n_crop_grid_cols = min(MULTI_CROP_GRID_NCOLS, max(1, n_crops))
    n_crop_grid_rows = int(np.ceil(n_crops / n_crop_grid_cols))
    n_crop_grid_rows = max(MULTI_CROP_GRID_NROWS, n_crop_grid_rows) if n_crops == 4 else n_crop_grid_rows

    n_rows = n_variable_rows * n_crop_grid_rows
    n_cols = panels_per_crop * n_crop_grid_cols
    figsize = get_multi_crop_figure_size(n_cols=n_cols, n_rows=n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharex=True,
        squeeze=False,
    )

    # 如果作物数不足 2×2，先把所有轴隐藏，后续有数据的轴再打开。
    for ax in axes.ravel():
        ax.set_visible(False)

    diff_curve_records = []
    plotted_any = False

    boundary_df = pd.DataFrame()
    for spec in source_specs:
        if not spec["df"].empty:
            boundary_df = spec["df"].copy()
            break

    if not boundary_df.empty and FULL_RESPONSE_X_COL in boundary_df.columns:
        boundary_df[FULL_RESPONSE_X_COL] = pd.to_numeric(
            boundary_df[FULL_RESPONSE_X_COL],
            errors="coerce",
        )
        boundary_df = boundary_df.dropna(subset=[FULL_RESPONSE_X_COL]).copy()
    boundary_lines = get_full_period_boundary_lines(boundary_df)

    for crop_idx, crop_spec in enumerate(crop_specs):

        crop_grid_row = crop_idx // n_crop_grid_cols
        crop_grid_col = crop_idx % n_crop_grid_cols
        crop_name = crop_spec["crop_name"]

        for row_idx, variable in enumerate(plot_variables):

            actual_row = crop_grid_row * n_variable_rows + row_idx
            row_plotted_any = False

            for panel_idx, spec in enumerate(crop_spec["panels"]):

                actual_col = crop_grid_col * panels_per_crop + panel_idx
                ax = axes[actual_row, actual_col]
                ax.set_visible(True)

                event_type = spec["event_type"]
                df_var = spec["df"].copy()

                if row_idx == 0:
                    ax.set_title(
                        f"{crop_name}\n{spec['title']}",
                        fontsize=7.0,
                    )

                if df_var.empty or variable not in df_var.columns:
                    ax.text(
                        0.5,
                        0.5,
                        "No data",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color="0.35",
                    )
                    format_nature_axes(ax)
                    set_subplot_y_range_from_data(ax)
                    continue

                df_var[FULL_RESPONSE_X_COL] = pd.to_numeric(
                    df_var[FULL_RESPONSE_X_COL],
                    errors="coerce",
                )
                df_var[variable] = pd.to_numeric(df_var[variable], errors="coerce")
                df_var = df_var.dropna(subset=[FULL_RESPONSE_X_COL, variable]).copy()
                df_var = df_var[
                    df_var["event_occurrence_stage"].isin(EVENT_OCCURRENCE_STAGES)
                ].copy()

                if df_var.empty:
                    ax.text(
                        0.5,
                        0.5,
                        "No valid\nvariable data",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color="0.35",
                    )
                    format_nature_axes(ax)
                    set_subplot_y_range_from_data(ax)
                    continue

                plotted_this_panel = False
                plotted_stage_count = 0

                for occurrence_stage in EVENT_OCCURRENCE_STAGES:

                    event_sub = df_var[
                        (df_var["event_type"] == event_type) &
                        (df_var["event_occurrence_stage"] == occurrence_stage)
                    ].copy()

                    normal_sub = df_var[
                        (df_var["event_type"] == "normal") &
                        (df_var["event_occurrence_stage"] == occurrence_stage)
                    ].copy()

                    if event_sub.empty or normal_sub.empty:
                        continue

                    unit_col = FULL_RESPONSE_UNIT_COL if FULL_RESPONSE_UNIT_COL in df_var.columns else "event_id"
                    n_event_units = event_sub[unit_col].nunique()
                    n_normal_units = normal_sub[unit_col].nunique()

                    if (n_event_units < min_events_per_group) or (n_normal_units < min_events_per_group):
                        continue

                    diff_curve = summarize_event_minus_normal_curve(
                        df=df_var,
                        variable=variable,
                        event_type=event_type,
                        occurrence_stage=occurrence_stage,
                        x_col=FULL_RESPONSE_X_COL,
                        unit_col=unit_col,
                    )

                    if diff_curve.empty:
                        continue

                    diff_curve["crop_name"] = crop_name
                    diff_curve["variable"] = variable
                    diff_curve["panel_event_type"] = event_type
                    diff_curve["dfai_source_dir"] = spec["source_dir"]
                    diff_curve_records.append(diff_curve)

                    color = OCCURRENCE_STAGE_COLORS.get(occurrence_stage, "#333333")

                    did_plot = plot_mean_with_ci(
                        ax=ax,
                        mean_curve=diff_curve,
                        color=color,
                        label=EVENT_OCCURRENCE_STAGE_LABELS.get(occurrence_stage, occurrence_stage),
                        linewidth=NATURE_MEAN_LINEWIDTH,
                        alpha_ci=NATURE_CI_ALPHA,
                        zorder=3 + plotted_stage_count,
                    )

                    if did_plot:
                        plotted_any = True
                        row_plotted_any = True
                        plotted_this_panel = True
                        plotted_stage_count += 1

                if FULL_RESPONSE_X_MODE == "progress_percent":
                    ax.axvline(0, color="0.35", linestyle=":", linewidth=0.55, zorder=1)
                    ax.axvline(100, color="0.35", linestyle=":", linewidth=0.55, zorder=1)
                    ax.set_xlim(0, 100)

                    for boundary_x, boundary_label in boundary_lines:
                        ax.axvline(
                            boundary_x,
                            color="0.60",
                            linestyle="--",
                            linewidth=0.45,
                            zorder=1,
                        )
                        if row_idx == 0:
                            ax.text(
                                boundary_x,
                                0.98,
                                boundary_label,
                                rotation=90,
                                va="top",
                                ha="right",
                                fontsize=4.5,
                                color="0.35",
                                transform=ax.get_xaxis_transform(),
                            )

                ax.axhline(0, color="0.20", linestyle=":", linewidth=0.55, zorder=1)

                if not plotted_this_panel:
                    ax.text(
                        0.5,
                        0.5,
                        "No valid group",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=5.2,
                        color="0.35",
                    )

                if actual_col == 0:
                    ax.set_ylabel(f"Δ {get_y_axis_label(variable)}", fontsize=6.3)
                else:
                    ax.set_ylabel("")

                # x 轴名称在整张图底部统一显示，避免重复。
                ax.set_xlabel("")

                format_nature_axes(ax)
                set_subplot_y_range_from_data(ax)

            # 每个小图已在 set_subplot_y_range_from_data() 中按自身数据单独设置 y 轴范围。

    if not plotted_any:
        plt.close(fig)
        print("跳过四作物正负 DFAI 合并总图：没有达到最小样本数要求。")
        return

    legend_handles = []
    for occurrence_stage in EVENT_OCCURRENCE_STAGES:
        label = EVENT_OCCURRENCE_STAGE_LABELS.get(occurrence_stage, occurrence_stage)
        label = label.replace("Occurred in ", "").replace(" stage", "").title()
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=OCCURRENCE_STAGE_COLORS.get(occurrence_stage, "#333333"),
                linewidth=NATURE_MEAN_LINEWIDTH,
                label=label,
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.996),
        handlelength=1.4,
        columnspacing=1.3,
        fontsize=6.3,
    )

    fig.suptitle(
        "Four crops 2×2 | DFAI positive and negative events minus Normal | all variables\n"
        "Crop blocks are arranged as Maize/Rice/Soybeans/Wheat; lines = occurrence stage",
        y=1.018,
        fontsize=8,
    )

    add_shared_x_axis_label(fig, fontsize=6.2, y=0.008)
    fig.tight_layout(rect=[0, 0.040, 1, 0.975], h_pad=0.35, w_pad=0.30)

    save_stem = os.path.join(
        output_dir,
        "four_crops_2x2_full_period_delta_vs_normal_ALL_VARIABLES_grid_DFAI_gt4_and_lt_minus4",
    )

    saved_paths = save_nature_figure(fig, save_stem)
    plt.close(fig)

    for p in saved_paths:
        print(f"已保存四作物 2×2 正负 DFAI 合并总图：{p}")

    if diff_curve_records:
        diff_total = pd.concat(diff_curve_records, ignore_index=True)
        diff_path = os.path.join(
            output_dir,
            "four_crops_2x2_full_period_delta_curves_ALL_VARIABLES_grid_DFAI_gt4_and_lt_minus4.csv",
        )
        diff_total.to_csv(diff_path, index=False, encoding="utf-8-sig")
        print(f"已保存四作物 2×2 正负 DFAI 合并总图对应差值曲线表：{diff_path}")


def plot_multi_crop_dfai_spei_one_figure_by_stage(
    crop_results,
    dfai_direction,
    variables,
    output_dir,
    min_events_per_group=3
):
    """
    绘制四作物 DFAA + SPEI 洪涝 / 干旱合并总图。

    dfai_direction:
    - "positive"：每个作物使用 positive_event_daily_total，列为 DFAI > 4 / Flood / Drought；
    - "negative"：每个作物使用 negative_event_daily_total，列为 DFAI < -4 / Flood / Drought。

    图形结构：
    - 作物按 2 行 × 2 列排列；
    - 每个作物块内部：每个指标一行、三列分别为 DFAA、SPEI > 1 Flood、SPEI < -1 Drought。
    """

    direction = str(dfai_direction).strip().lower()

    if direction in {"positive", "pos", "+", "+dfai", "positive_dfai", "positive_dfaa"}:
        data_key = "positive_event_daily_total"
        dfai_event_type = "positive_DFAA"
        dfai_title = "DFAI > 4"
        direction_label = "positive"
    elif direction in {"negative", "neg", "-", "-dfai", "negative_dfai", "negative_dfaa"}:
        data_key = "negative_event_daily_total"
        dfai_event_type = "negative_DFAA"
        dfai_title = "DFAI < -4"
        direction_label = "negative"
    else:
        raise ValueError("dfai_direction 只能是 'positive' 或 'negative'。")

    panel_specs = [
        {"event_type": dfai_event_type, "title": dfai_title},
        {"event_type": "flood", "title": "SPEI > 1"},
        {"event_type": "drought", "title": "SPEI < -1"},
    ]

    crop_specs = []

    for crop_result in crop_results:
        crop_name = crop_result.get("crop_name", "unknown_crop")
        df = get_crop_result_value(crop_result, data_key)
        source_dir = crop_result.get(
            "positive_dssat_dir" if direction_label == "positive" else "negative_dssat_dir",
            "",
        )

        crop_specs.append({
            "crop_name": crop_name,
            "panels": [
                {
                    "crop_name": crop_name,
                    "event_type": panel_spec["event_type"],
                    "df": df,
                    "title": panel_spec["title"],
                    "source_dir": source_dir,
                }
                for panel_spec in panel_specs
            ],
        })

    if not crop_specs:
        print(f"没有可用于四作物 {direction_label} DFAI + SPEI 合并总图的数据。")
        return

    source_specs = [panel for crop_spec in crop_specs for panel in crop_spec["panels"]]

    required_cols = [
        FULL_RESPONSE_X_COL,
        "event_type",
        "event_occurrence_stage",
        "event_id",
        "site_folder",
    ]

    for spec in source_specs:
        if spec["df"].empty:
            continue
        missing_cols = [c for c in required_cols if c not in spec["df"].columns]
        if missing_cols:
            print(
                f"跳过四作物 {direction_label} DFAI + SPEI 合并总图，"
                f"{spec['crop_name']} 数据缺少必要列：{missing_cols}"
            )
            return

    plot_variables = [
        v for v in variables
        if any((not spec["df"].empty) and (v in spec["df"].columns) for spec in source_specs)
    ]

    if not plot_variables:
        print(f"四作物 {direction_label} DFAI + SPEI 合并总图没有可绘制指标，跳过。")
        return

    os.makedirs(output_dir, exist_ok=True)

    n_variable_rows = len(plot_variables)
    panels_per_crop = len(panel_specs)
    n_crops = len(crop_specs)
    n_crop_grid_cols = min(MULTI_CROP_GRID_NCOLS, max(1, n_crops))
    n_crop_grid_rows = int(np.ceil(n_crops / n_crop_grid_cols))
    n_crop_grid_rows = max(MULTI_CROP_GRID_NROWS, n_crop_grid_rows) if n_crops == 4 else n_crop_grid_rows

    n_rows = n_variable_rows * n_crop_grid_rows
    n_cols = panels_per_crop * n_crop_grid_cols
    figsize = get_multi_crop_figure_size(n_cols=n_cols, n_rows=n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharex=True,
        squeeze=False,
    )

    # 如果作物数不足 2×2，先把所有轴隐藏，后续有数据的轴再打开。
    for ax in axes.ravel():
        ax.set_visible(False)

    diff_curve_records = []
    plotted_any = False

    boundary_df = pd.DataFrame()
    for spec in source_specs:
        if not spec["df"].empty:
            boundary_df = spec["df"].copy()
            break

    if not boundary_df.empty and FULL_RESPONSE_X_COL in boundary_df.columns:
        boundary_df[FULL_RESPONSE_X_COL] = pd.to_numeric(
            boundary_df[FULL_RESPONSE_X_COL],
            errors="coerce",
        )
        boundary_df = boundary_df.dropna(subset=[FULL_RESPONSE_X_COL]).copy()
    boundary_lines = get_full_period_boundary_lines(boundary_df)

    for crop_idx, crop_spec in enumerate(crop_specs):

        crop_grid_row = crop_idx // n_crop_grid_cols
        crop_grid_col = crop_idx % n_crop_grid_cols
        crop_name = crop_spec["crop_name"]

        for row_idx, variable in enumerate(plot_variables):

            actual_row = crop_grid_row * n_variable_rows + row_idx
            row_plotted_any = False

            for panel_idx, spec in enumerate(crop_spec["panels"]):

                actual_col = crop_grid_col * panels_per_crop + panel_idx
                ax = axes[actual_row, actual_col]
                ax.set_visible(True)

                event_type = spec["event_type"]
                df_var = spec["df"].copy()

                if row_idx == 0:
                    ax.set_title(
                        f"{crop_name}\n{spec['title']}",
                        fontsize=7.0,
                    )

                if df_var.empty or variable not in df_var.columns:
                    ax.text(
                        0.5,
                        0.5,
                        "No data",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color="0.35",
                    )
                    format_nature_axes(ax)
                    set_subplot_y_range_from_data(ax)
                    continue

                df_var[FULL_RESPONSE_X_COL] = pd.to_numeric(
                    df_var[FULL_RESPONSE_X_COL],
                    errors="coerce",
                )
                df_var[variable] = pd.to_numeric(df_var[variable], errors="coerce")
                df_var = df_var.dropna(subset=[FULL_RESPONSE_X_COL, variable]).copy()
                df_var = df_var[
                    df_var["event_occurrence_stage"].isin(EVENT_OCCURRENCE_STAGES)
                ].copy()

                if df_var.empty:
                    ax.text(
                        0.5,
                        0.5,
                        "No valid\nvariable data",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color="0.35",
                    )
                    format_nature_axes(ax)
                    set_subplot_y_range_from_data(ax)
                    continue

                plotted_this_panel = False
                plotted_stage_count = 0

                for occurrence_stage in EVENT_OCCURRENCE_STAGES:

                    event_sub = df_var[
                        (df_var["event_type"] == event_type) &
                        (df_var["event_occurrence_stage"] == occurrence_stage)
                    ].copy()

                    normal_sub = df_var[
                        (df_var["event_type"] == "normal") &
                        (df_var["event_occurrence_stage"] == occurrence_stage)
                    ].copy()

                    if event_sub.empty or normal_sub.empty:
                        continue

                    unit_col = FULL_RESPONSE_UNIT_COL if FULL_RESPONSE_UNIT_COL in df_var.columns else "event_id"
                    n_event_units = event_sub[unit_col].nunique()
                    n_normal_units = normal_sub[unit_col].nunique()

                    if (n_event_units < min_events_per_group) or (n_normal_units < min_events_per_group):
                        continue

                    diff_curve = summarize_event_minus_normal_curve(
                        df=df_var,
                        variable=variable,
                        event_type=event_type,
                        occurrence_stage=occurrence_stage,
                        x_col=FULL_RESPONSE_X_COL,
                        unit_col=unit_col,
                    )

                    if diff_curve.empty:
                        continue

                    diff_curve["crop_name"] = crop_name
                    diff_curve["variable"] = variable
                    diff_curve["panel_event_type"] = event_type
                    diff_curve["dfai_direction"] = direction_label
                    diff_curve["dfai_source_dir"] = spec["source_dir"]
                    diff_curve_records.append(diff_curve)

                    color = OCCURRENCE_STAGE_COLORS.get(occurrence_stage, "#333333")

                    did_plot = plot_mean_with_ci(
                        ax=ax,
                        mean_curve=diff_curve,
                        color=color,
                        label=EVENT_OCCURRENCE_STAGE_LABELS.get(occurrence_stage, occurrence_stage),
                        linewidth=NATURE_MEAN_LINEWIDTH,
                        alpha_ci=NATURE_CI_ALPHA,
                        zorder=3 + plotted_stage_count,
                    )

                    if did_plot:
                        plotted_any = True
                        row_plotted_any = True
                        plotted_this_panel = True
                        plotted_stage_count += 1

                if FULL_RESPONSE_X_MODE == "progress_percent":
                    ax.axvline(0, color="0.35", linestyle=":", linewidth=0.55, zorder=1)
                    ax.axvline(100, color="0.35", linestyle=":", linewidth=0.55, zorder=1)
                    ax.set_xlim(0, 100)

                    for boundary_x, boundary_label in boundary_lines:
                        ax.axvline(
                            boundary_x,
                            color="0.60",
                            linestyle="--",
                            linewidth=0.45,
                            zorder=1,
                        )
                        if row_idx == 0:
                            ax.text(
                                boundary_x,
                                0.98,
                                boundary_label,
                                rotation=90,
                                va="top",
                                ha="right",
                                fontsize=4.5,
                                color="0.35",
                                transform=ax.get_xaxis_transform(),
                            )

                ax.axhline(0, color="0.20", linestyle=":", linewidth=0.55, zorder=1)

                if not plotted_this_panel:
                    ax.text(
                        0.5,
                        0.5,
                        "No valid group",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=5.2,
                        color="0.35",
                    )

                if actual_col == 0:
                    ax.set_ylabel(f"Δ {get_y_axis_label(variable)}", fontsize=6.3)
                else:
                    ax.set_ylabel("")

                # x 轴名称在整张图底部统一显示，避免重复。
                ax.set_xlabel("")

                format_nature_axes(ax)
                set_subplot_y_range_from_data(ax)

            # 每个小图已在 set_subplot_y_range_from_data() 中按自身数据单独设置 y 轴范围。

    if not plotted_any:
        plt.close(fig)
        print(f"跳过四作物 {direction_label} DFAI + SPEI 合并总图：没有达到最小样本数要求。")
        return

    legend_handles = []
    for occurrence_stage in EVENT_OCCURRENCE_STAGES:
        label = EVENT_OCCURRENCE_STAGE_LABELS.get(occurrence_stage, occurrence_stage)
        label = label.replace("Occurred in ", "").replace(" stage", "").title()
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=OCCURRENCE_STAGE_COLORS.get(occurrence_stage, "#333333"),
                linewidth=NATURE_MEAN_LINEWIDTH,
                label=label,
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.996),
        handlelength=1.4,
        columnspacing=1.3,
        fontsize=6.3,
    )

    fig.suptitle(
        f"Four crops 2×2 | {dfai_title} and SPEI events minus Normal | all variables\n"
        "Crop blocks are arranged as Maize/Rice/Soybeans/Wheat; lines = occurrence stage",
        y=1.018,
        fontsize=8,
    )

    add_shared_x_axis_label(fig, fontsize=6.2, y=0.008)
    fig.tight_layout(rect=[0, 0.040, 1, 0.975], h_pad=0.35, w_pad=0.30)

    save_stem = os.path.join(
        output_dir,
        f"four_crops_2x2_full_period_delta_vs_normal_ALL_VARIABLES_grid_"
        f"{safe_filename(dfai_event_type)}_flood_drought",
    )

    saved_paths = save_nature_figure(fig, save_stem)
    plt.close(fig)

    for p in saved_paths:
        print(f"已保存四作物 2×2 {direction_label} DFAI + SPEI 合并总图：{p}")

    if diff_curve_records:
        diff_total = pd.concat(diff_curve_records, ignore_index=True)
        diff_path = os.path.join(
            output_dir,
            f"four_crops_2x2_full_period_delta_curves_ALL_VARIABLES_grid_"
            f"{safe_filename(dfai_event_type)}_flood_drought.csv",
        )
        diff_total.to_csv(diff_path, index=False, encoding="utf-8-sig")
        print(f"已保存四作物 2×2 {direction_label} DFAI + SPEI 合并总图对应差值曲线表：{diff_path}")

# ============================================================
# 12. 单个样本处理函数
# ============================================================


def make_empty_process_result(status="unknown"):
    """生成单个样本处理结果的空结构，便于旧流程和站点批处理流程共用。"""

    return {
        "status": status,
        "quality_record": None,
        "stage_summary": pd.DataFrame(),
        "event_df": pd.DataFrame(),
        "event_daily": pd.DataFrame()
    }


def prepare_dssat_sample(dssat, site, time, scenario):
    """
    准备单个 站点-年份-情景 DSSAT daily 样本。

    只做轻量清洗和元信息补齐：
    - 保留 read_dssat_daily_file() 产生的 attrs；
    - DATE 转换、去空、归一化到日尺度；
    - 补齐 site_folder / time / scenario。

    返回：
    - dssat_clean: 可继续处理的样本 DataFrame，失败时为 None；
    - attrs: 原 DSSAT 预处理属性；
    - status: None 表示成功，否则为失败原因字符串。
    """

    attrs = dict(getattr(dssat, "attrs", {}) or {})

    if dssat is None or dssat.empty:
        return None, attrs, "empty_input"

    if "DATE" not in dssat.columns:
        return None, attrs, "missing_DATE"

    out = dssat.copy()
    out["DATE"] = pd.to_datetime(out["DATE"], errors="coerce")
    out = out.dropna(subset=["DATE"]).copy()
    out.attrs.update(attrs)
    out["DATE"] = out["DATE"].dt.normalize()

    if out.empty:
        return None, attrs, "empty_DATE"

    meta_values = {
        "site_folder": str(site),
        "time": str(time),
        "scenario": str(scenario),
    }

    for col, val in meta_values.items():
        if col not in out.columns:
            out[col] = val
        else:
            out[col] = out[col].fillna(val)
            out[col] = out[col].replace(r"^\s*$", val, regex=True)
        out[col] = out[col].astype(str)

    return out, attrs, None


def process_prepared_sample(dssat, site, time, scenario, climate=None, matched=None):
    """
    处理已经准备好的单个 站点-年份-情景 样本。

    与原 process_one_sample() 的分析逻辑保持一致；区别是：
    - 不在函数内部读取 DSSAT；
    - climate 可由站点批处理流程提前读取；
    - matched 可由站点批处理流程提前一次性 merge 后传入，避免逐年份重复 merge。
    """

    result = make_empty_process_result()

    if dssat is None or dssat.empty:
        result["status"] = "empty_DATE"
        return result

    crop_quality = summarize_crop_quality(
        dssat=dssat,
        site=site,
        time=time,
        scenario=scenario
    )

    result["quality_record"] = crop_quality

    if ENABLE_CROP_QC_FILTER and not crop_quality["passed_crop_qc"]:
        result["status"] = "crop_qc_failed"
        return result

    if climate is None:
        crop_quality["passed_climate_qc"] = False
        crop_quality["climate_qc_fail_reasons"] = "missing_climate_file"
        crop_quality["climate_match_days_spei"] = 0
        crop_quality["climate_match_days_ldfai"] = 0
        crop_quality["climate_match_ratio"] = np.nan

        result["quality_record"] = crop_quality
        result["status"] = "missing_climate_file"
        return result

    if matched is None:
        matched = dssat.merge(
            climate,
            on="DATE",
            how="left"
        )
    else:
        matched = matched.copy()

    climate_quality = summarize_climate_match_quality(matched)

    for k, v in climate_quality.items():
        crop_quality[k] = v

    result["quality_record"] = crop_quality

    if ENABLE_CROP_QC_FILTER and not climate_quality["passed_climate_qc"]:
        result["status"] = "climate_qc_failed"
        return result

    matched = assign_phenology_stage(matched)

    stage_summary = summarize_by_phenology_stage(
        df=matched,
        climate_site=climate
    )

    dfaa_events = build_ldfai_events_for_one_sample(
        matched=matched,
        climate_site=climate,
        site=site,
        time=time,
        scenario=scenario,
        stage_col="phenology_stage_group"
    )

    spei_events = build_spei_events_for_one_sample(
        matched=matched,
        climate_site=climate,
        site=site,
        time=time,
        scenario=scenario,
        stage_col="phenology_stage_group"
    )

    normal_events = build_normal_events_for_one_sample(
        matched=matched,
        climate_site=climate,
        site=site,
        time=time,
        scenario=scenario,
        stage_col="phenology_stage_group"
    )

    events = pd.concat(
        [
            dfaa_events,
            spei_events,
            normal_events
        ],
        ignore_index=True
    )

    # 可选：严格筛选“单一灾害阶段”。
    # 开启后，目标事件发生阶段内必须只有 1 次目标灾害，且不能有其他灾害。
    events = filter_events_by_strict_single_disaster_stage(events)

    events = finalize_event_ids(events)

    if not events.empty:
        event_daily = build_event_relative_daily(
            matched=matched,
            event_df=events
        )
    else:
        event_daily = pd.DataFrame()

    result["status"] = "passed"
    result["stage_summary"] = stage_summary
    result["event_df"] = events
    result["event_daily"] = event_daily

    return result


def process_one_sample(fp, site, time, scenario, climate_cache):
    """
    兼容旧接口的单样本处理函数。

    主流程已改为 run_processing_pipeline() 中的“按站点批处理”；
    保留该函数是为了方便你后续单独调试某个站点-年份样本。
    """

    dssat = read_dssat_daily_file(fp, crop_name=CURRENT_CROP_NAME)
    dssat, attrs, status = prepare_dssat_sample(
        dssat=dssat,
        site=site,
        time=time,
        scenario=scenario
    )

    if status is not None:
        return make_empty_process_result(status=status)

    if site in climate_cache:
        climate, climate_path = climate_cache[site]
    else:
        climate, climate_path = read_climate_file(site)
        climate_cache[site] = (climate, climate_path)

    return process_prepared_sample(
        dssat=dssat,
        site=site,
        time=time,
        scenario=scenario,
        climate=climate,
        matched=None
    )



# ============================================================
# 13. 批量处理：分别读取两套 DSSAT 目录，并生成指定三类总图
# ============================================================

def build_file_df_for_dssat_dir(current_dssat_dir):
    """读取指定 DSSAT 目录下的目标情景文件清单。"""

    dssat_files = glob.glob(os.path.join(current_dssat_dir, "*.csv"))

    dssat_files = [
        fp for fp in dssat_files
        if "ALL_SITES_TOTAL" not in os.path.basename(fp)
        and "_matched" not in os.path.basename(fp)
    ]

    print(f"\n当前 DSSAT 目录：{current_dssat_dir}")
    print(f"找到 DSSAT 文件总数：{len(dssat_files)}")
    print(f"当前仅分析情景：{ANALYZE_SCENARIO}")

    file_records = []

    for fp in dssat_files:
        try:
            site, time, scenario = parse_site_time_scenario_from_filename(fp)
            file_records.append({
                "fp": fp,
                "site_folder": site,
                "time": str(time),
                "scenario": scenario,
                "scenario_sort": scenario_sort_key(scenario),
                "scenario_clean": str(scenario).upper().strip()
            })
        except Exception as e:
            print(f"文件名解析失败，跳过：{fp}")
            print(f"错误信息：{e}")

    raw_file_df = pd.DataFrame(file_records)

    if raw_file_df.empty:
        raise RuntimeError(f"没有可处理的 DSSAT 文件：{current_dssat_dir}")

    target_scenario_clean = ANALYZE_SCENARIO.upper().strip()

    non_target_scenarios_ignored_count = int(
        (raw_file_df["scenario_clean"] != target_scenario_clean).sum()
    )

    file_df = raw_file_df[
        raw_file_df["scenario_clean"] == target_scenario_clean
    ].copy()

    if file_df.empty:
        raise RuntimeError(
            f"在 {current_dssat_dir} 中没有找到情景为 {ANALYZE_SCENARIO} 的 DSSAT 文件。"
            "请检查文件名是否为 站点-时间-TRT1.csv 格式。"
        )

    file_df = file_df.sort_values(
        ["site_folder", "time", "scenario_sort"]
    ).reset_index(drop=True)

    print(f"用于分析的 {ANALYZE_SCENARIO} 文件数：{len(file_df)}")
    print(f"已忽略非 {ANALYZE_SCENARIO} 情景文件数：{non_target_scenarios_ignored_count}")

    return file_df, non_target_scenarios_ignored_count



def run_processing_pipeline(file_df, non_target_scenarios_ignored_count):
    """
    完整读取原始文件并构建所有结果表。

    加速版：按站点批处理。
    - 原流程：逐个 站点-年份-情景 读取/切片 DSSAT，然后逐样本 merge 气象；
    - 新流程：同一站点的所有目标情景文件先合并为一个站点级 DSSAT 表，
      再和该站点气象数据 merge 一次，最后按文件来源拆回站点-年份样本做 QC 和事件识别。

    注意：QC、事件识别、normal 基线和 event_daily 构建仍然按单个站点-年份样本执行，
    因此不会把不同年份混成一个统计样本。
    """

    all_stage_summary = []
    all_quality_records = []
    all_events = []
    all_event_relative_daily = []

    processed_count = 0
    passed_count = 0

    # 记录试运行中已经纳入后续分析的站点 / 站点-年份。
    accepted_sites_for_test = set()
    accepted_site_years_for_test = set()

    # 记录目标情景样本处理状态。
    processed_trt1_records = []

    def site_limit_reached_for_new_group(site, time):
        """
        判断是否已经达到试运行站点上限。
        达到上限后停止读取新的站点或新的站点-年份，
        但不会影响后面保存 CSV 和绘图。
        """

        if not ENABLE_SITE_LIMIT_FOR_TEST:
            return False

        if SITE_LIMIT_MODE == "site":
            return (
                len(accepted_sites_for_test) >= MAX_PASSED_SITES_FOR_TEST
                and site not in accepted_sites_for_test
            )

        if SITE_LIMIT_MODE == "site_year":
            return (
                len(accepted_site_years_for_test) >= MAX_PASSED_SITES_FOR_TEST
                and (site, str(time)) not in accepted_site_years_for_test
            )

        raise ValueError("SITE_LIMIT_MODE 只能是 'site' 或 'site_year'")

    def register_passed_group_for_test(site, time):
        """记录已经通过 QC、进入后续分析的站点 / 站点-年份。"""

        if not ENABLE_SITE_LIMIT_FOR_TEST:
            return

        if SITE_LIMIT_MODE == "site":
            before = len(accepted_sites_for_test)
            accepted_sites_for_test.add(site)
            after = len(accepted_sites_for_test)

            if after > before:
                print(
                    f"  当前已累计通过站点数："
                    f"{after}/{MAX_PASSED_SITES_FOR_TEST}"
                )

        elif SITE_LIMIT_MODE == "site_year":
            before = len(accepted_site_years_for_test)
            accepted_site_years_for_test.add((site, str(time)))
            after = len(accepted_site_years_for_test)

            if after > before:
                print(
                    f"  当前已累计通过站点-年份数："
                    f"{after}/{MAX_PASSED_SITES_FOR_TEST}"
                )

        else:
            raise ValueError("SITE_LIMIT_MODE 只能是 'site' 或 'site_year'")

    def print_skip_message(site, time, scenario, res):
        """打印未通过样本的状态和 QC 详情。"""

        qc_reason = ""
        if res.get("quality_record") is not None:
            qrec = res["quality_record"]
            if res["status"] == "crop_qc_failed":
                qc_reason = str(qrec.get("crop_qc_fail_reasons", ""))
            elif res["status"] == "climate_qc_failed":
                qc_reason = str(qrec.get("climate_qc_fail_reasons", ""))
            if qc_reason:
                qc_reason = f"，QC详情：{qc_reason}"

        print(
            f"  {ANALYZE_SCENARIO} 未通过，跳过该样本后续分析："
            f"{site}-{time}-{scenario}，原因：{res['status']}{qc_reason}"
        )

    def append_result_tables(res):
        """把通过 QC 的样本结果追加到总表列表。"""

        if not res["stage_summary"].empty:
            all_stage_summary.append(res["stage_summary"])

        if not res["event_df"].empty:
            all_events.append(res["event_df"])

        if not res["event_daily"].empty:
            all_event_relative_daily.append(res["event_daily"])

    # file_df 已按 site_folder / time 排序；这里保持 sort=False，避免改变原处理顺序。
    for site, site_file_df in file_df.groupby("site_folder", sort=False):

        first_time = str(site_file_df["time"].iloc[0])

        if site_limit_reached_for_new_group(site, first_time):
            if SITE_LIMIT_MODE == "site":
                print(
                    f"\n已累计 {MAX_PASSED_SITES_FOR_TEST} 个通过筛选的唯一站点，"
                    f"停止读取新的站点；下一个未处理站点为：{site}"
                )
            else:
                print(
                    f"\n已累计 {MAX_PASSED_SITES_FOR_TEST} 个通过筛选的站点-年份，"
                    f"停止读取新的站点-年份；下一个未处理组合为：{site}-{first_time}"
                )
            break

        print(f"\n==============================")
        print(f"按站点批处理 {ANALYZE_SCENARIO} 样本：{site}，样本数：{len(site_file_df)}")
        print(f"==============================")

        climate, climate_path = read_climate_file(site)

        dssat_parts = []
        sample_meta = []

        for _, row in site_file_df.iterrows():

            fp = row["fp"]
            time = row["time"]
            scenario = row["scenario"]
            fp_norm = os.path.normpath(str(fp))

            if site_limit_reached_for_new_group(site, time):
                if SITE_LIMIT_MODE == "site_year":
                    print(
                        f"\n已累计 {MAX_PASSED_SITES_FOR_TEST} 个通过筛选的站点-年份，"
                        f"停止读取新的站点-年份；下一个未处理组合为：{site}-{time}"
                    )
                break

            try:
                dssat = read_dssat_daily_file(fp, crop_name=CURRENT_CROP_NAME)
                dssat, attrs, status = prepare_dssat_sample(
                    dssat=dssat,
                    site=site,
                    time=time,
                    scenario=scenario
                )
            except Exception as e:
                print(f"  DSSAT 读取失败，已跳过：{fp}")
                print(f"  错误信息：{e}")

                processed_count += 1
                processed_trt1_records.append({
                    "site_folder": site,
                    "time": time,
                    "scenario": scenario,
                    "fp": fp,
                    "status": f"read_error: {e}"
                })
                continue

            if status is not None:
                processed_count += 1
                processed_trt1_records.append({
                    "site_folder": site,
                    "time": time,
                    "scenario": scenario,
                    "fp": fp,
                    "status": status
                })
                print(
                    f"  {ANALYZE_SCENARIO} 未通过，跳过该样本后续分析："
                    f"{site}-{time}-{scenario}，原因：{status}"
                )
                continue

            # 标记来源文件，站点级 merge 后再按这个字段拆回单个站点-年份样本。
            dssat["_source_fp_norm"] = fp_norm

            # 把 attrs 临时写成列，concat / groupby 之后可以恢复到单样本 attrs。
            dssat["_dssat_raw_rows"] = attrs.get("dssat_raw_rows", len(dssat))
            dssat["_dssat_clean_rows"] = attrs.get("dssat_clean_rows", len(dssat))
            dssat["_dssat_preprocess_mode"] = attrs.get("dssat_preprocess_mode", "unknown")
            dssat["_dssat_year_doy_groups_collapsed"] = attrs.get("dssat_year_doy_groups_collapsed", np.nan)
            dssat["_dssat_rows_after_year_doy_collapse"] = attrs.get("dssat_rows_after_year_doy_collapse", np.nan)
            dssat["_dssat_empty_crop_rows_removed"] = attrs.get("dssat_empty_crop_rows_removed", np.nan)
            dssat["_dssat_variable_aliases_applied"] = attrs.get("dssat_variable_aliases_applied", "{}")

            dssat_parts.append(dssat)
            sample_meta.append({
                "fp": fp,
                "fp_norm": fp_norm,
                "site_folder": site,
                "time": str(time),
                "scenario": scenario,
            })

        if not dssat_parts:
            continue

        site_dssat = pd.concat(dssat_parts, ignore_index=True)

        if climate is not None:
            # 关键加速点：同一个站点的所有年份只和气象表 merge 一次。
            site_matched = site_dssat.merge(
                climate,
                on="DATE",
                how="left"
            )
        else:
            site_matched = None

        dssat_groups = {
            str(fp_key): idx.values
            for fp_key, idx in site_dssat.groupby("_source_fp_norm", sort=False).groups.items()
        }

        if site_matched is not None and not site_matched.empty:
            matched_groups = {
                str(fp_key): idx.values
                for fp_key, idx in site_matched.groupby("_source_fp_norm", sort=False).groups.items()
            }
        else:
            matched_groups = {}

        for meta in sample_meta:

            site = meta["site_folder"]
            time = meta["time"]
            scenario = meta["scenario"]
            fp = meta["fp"]
            fp_norm = meta["fp_norm"]

            if site_limit_reached_for_new_group(site, time):
                if SITE_LIMIT_MODE == "site_year":
                    print(
                        f"\n已累计 {MAX_PASSED_SITES_FOR_TEST} 个通过筛选的站点-年份，"
                        f"停止读取新的站点-年份；下一个未处理组合为：{site}-{time}"
                    )
                break

            row_idx = dssat_groups.get(fp_norm)
            if row_idx is None or len(row_idx) == 0:
                processed_count += 1
                processed_trt1_records.append({
                    "site_folder": site,
                    "time": time,
                    "scenario": scenario,
                    "fp": fp,
                    "status": "missing_group_after_site_batch"
                })
                continue

            dssat_sample = site_dssat.iloc[row_idx].copy()
            attrs = _restore_dssat_attrs_from_merged_sample(dssat_sample)

            drop_cols = [
                c for c in DSSAT_MERGED_RAW_INTERNAL_COLS
                if c in dssat_sample.columns
            ]
            if drop_cols:
                dssat_sample = dssat_sample.drop(columns=drop_cols)

            dssat_sample.attrs.update(attrs)

            matched_sample = None
            matched_idx = matched_groups.get(fp_norm)
            if matched_idx is not None and len(matched_idx) > 0:
                matched_sample = site_matched.iloc[matched_idx].copy()
                matched_drop_cols = [
                    c for c in DSSAT_MERGED_RAW_INTERNAL_COLS
                    if c in matched_sample.columns
                ]
                if matched_drop_cols:
                    matched_sample = matched_sample.drop(columns=matched_drop_cols)

            print(f"  处理 {ANALYZE_SCENARIO} 样本：{site}-{time}-{scenario}")

            res = process_prepared_sample(
                dssat=dssat_sample,
                site=site,
                time=time,
                scenario=scenario,
                climate=climate,
                matched=matched_sample
            )

            processed_count += 1

            processed_trt1_records.append({
                "site_folder": site,
                "time": time,
                "scenario": scenario,
                "fp": fp,
                "status": res["status"]
            })

            if res["quality_record"] is not None:
                all_quality_records.append(res["quality_record"])

            if res["status"] != "passed":
                print_skip_message(site, time, scenario, res)
                continue

            passed_count += 1

            # 只有通过 QC 并进入后续分析后，才计入试运行站点上限。
            register_passed_group_for_test(site, time)
            append_result_tables(res)

    stage_summary_total = (
        pd.concat(all_stage_summary, ignore_index=True)
        if all_stage_summary else pd.DataFrame()
    )

    event_total = (
        pd.concat(all_events, ignore_index=True)
        if all_events else pd.DataFrame()
    )

    event_daily_total = (
        pd.concat(all_event_relative_daily, ignore_index=True)
        if all_event_relative_daily else pd.DataFrame()
    )

    quality_df = (
        pd.DataFrame(all_quality_records)
        if all_quality_records else pd.DataFrame()
    )

    processed_trt1_df = (
        pd.DataFrame(processed_trt1_records)
        if processed_trt1_records else pd.DataFrame()
    )

    if ENABLE_SITE_LIMIT_FOR_TEST:
        if SITE_LIMIT_MODE == "site":
            selected_units_df = pd.DataFrame({
                "site_folder": sorted(accepted_sites_for_test)
            })
            selected_units_mode = "site"
        else:
            selected_units_df = pd.DataFrame(
                [
                    {"site_folder": s, "time": t}
                    for s, t in sorted(accepted_site_years_for_test)
                ]
            )
            selected_units_mode = "site_year"
    else:
        selected_units_df = pd.DataFrame()
        selected_units_mode = "none"

    return {
        "stage_summary_total": stage_summary_total,
        "event_total": event_total,
        "event_daily_total": event_daily_total,
        "quality_df": quality_df,
        "processed_trt1_df": processed_trt1_df,
        "selected_units_df": selected_units_df,
        "selected_units_mode": selected_units_mode,
        "processed_count": processed_count,
        "passed_count": passed_count,
        "non_target_scenarios_ignored_count": non_target_scenarios_ignored_count,
        "accepted_sites_for_test": accepted_sites_for_test,
        "accepted_site_years_for_test": accepted_site_years_for_test
    }



def save_result_tables_for_run(results, run_output_dir, run_label):
    """按 DSSAT 目录分别保存中间结果，避免两套目录互相覆盖。"""

    os.makedirs(run_output_dir, exist_ok=True)

    stage_summary_total = results.get("stage_summary_total", pd.DataFrame())
    event_total = results.get("event_total", pd.DataFrame())
    event_daily_total = results.get("event_daily_total", pd.DataFrame())
    quality_df = results.get("quality_df", pd.DataFrame())
    processed_trt1_df = results.get("processed_trt1_df", pd.DataFrame())
    selected_units_df = results.get("selected_units_df", pd.DataFrame())
    selected_units_mode = results.get("selected_units_mode", SITE_LIMIT_MODE)

    if not stage_summary_total.empty:
        path = os.path.join(
            run_output_dir,
            f"{run_label}_DSSAT_SPEI_LDFAI_phenology_stage_summary_TRT1_only_stage_window.csv"
        )
        stage_summary_total.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n物候阶段汇总表已保存：{path}")
    else:
        print(f"\n{run_label} 没有生成物候阶段汇总表")

    if not event_total.empty:
        path = os.path.join(
            run_output_dir,
            f"{run_label}_ALL_event_table_DFAA_drought_flood_normal_TRT1_only_stage_window.csv"
        )
        event_total.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n所有事件表已保存：{path}")
        print("\n事件数量统计：")
        print(
            event_total
            .groupby(["event_type", "window_phenology_stage"])
            .size()
            .reset_index(name="event_count")
        )
    else:
        print(f"\n{run_label} 没有识别到任何事件")

    if not event_daily_total.empty:
        path = os.path.join(
            run_output_dir,
            f"{run_label}_ALL_event_full_early_middle_late_daily_DFAA_drought_flood_normal_TRT1_only.csv"
        )
        event_daily_total.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n所有事件对应完整前中后时期日尺度响应表已保存：{path}")
    else:
        print(f"\n{run_label} 没有生成事件对应完整前中后时期响应数据")

    if not quality_df.empty:
        quality_path = os.path.join(
            run_output_dir,
            f"{run_label}_DSSAT_crop_quality_filter_summary_TRT1_only_stage_window.csv"
        )
        quality_df.to_csv(quality_path, index=False, encoding="utf-8-sig")
        print(f"\n作物质量筛选表已保存：{quality_path}")

        if "passed_crop_qc" in quality_df.columns:
            failed_crop_path = os.path.join(
                run_output_dir,
                f"{run_label}_DSSAT_crop_quality_filter_failed_TRT1_only_stage_window.csv"
            )
            quality_df[quality_df["passed_crop_qc"] == False].to_csv(
                failed_crop_path,
                index=False,
                encoding="utf-8-sig"
            )
            print(f"未通过作物质量筛选样本已保存：{failed_crop_path}")

            passed_crop_path = os.path.join(
                run_output_dir,
                f"{run_label}_DSSAT_crop_quality_filter_passed_TRT1_only_stage_window.csv"
            )
            quality_df[quality_df["passed_crop_qc"] == True].to_csv(
                passed_crop_path,
                index=False,
                encoding="utf-8-sig"
            )
            print(f"通过作物质量筛选样本已保存：{passed_crop_path}")

        if "passed_climate_qc" in quality_df.columns and "passed_crop_qc" in quality_df.columns:
            quality_df = quality_df.copy()
            quality_df["passed_all_qc"] = (
                quality_df["passed_crop_qc"].fillna(False) &
                quality_df["passed_climate_qc"].fillna(False)
            )
            passed_all_path = os.path.join(
                run_output_dir,
                f"{run_label}_DSSAT_quality_filter_passed_all_TRT1_only_stage_window.csv"
            )
            quality_df[quality_df["passed_all_qc"] == True].to_csv(
                passed_all_path,
                index=False,
                encoding="utf-8-sig"
            )
            print(f"作物和气象均通过筛选样本已保存：{passed_all_path}")

        print("\n==============================")
        print(f"{run_label} 筛选结果统计")
        print("==============================")
        print(f"仅分析目标情景：{ANALYZE_SCENARIO}")
        print(f"实际处理 {ANALYZE_SCENARIO} 样本数：{int(results.get('processed_count', 0))}")
        print(f"通过筛选并进入后续分析的 {ANALYZE_SCENARIO} 样本数：{int(results.get('passed_count', 0))}")
        print(
            f"已忽略非 {ANALYZE_SCENARIO} 情景文件数："
            f"{int(results.get('non_target_scenarios_ignored_count', 0))}"
        )

    if not processed_trt1_df.empty:
        path = os.path.join(run_output_dir, f"{run_label}_DSSAT_processed_TRT1_records.csv")
        processed_trt1_df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\nTRT1 样本处理记录已保存：{path}")

    if ENABLE_SITE_LIMIT_FOR_TEST and not selected_units_df.empty:
        if selected_units_mode == "site":
            filename = f"{run_label}_DSSAT_test_selected_TRT1_sites.csv"
        elif selected_units_mode == "site_year":
            filename = f"{run_label}_DSSAT_test_selected_TRT1_site_years.csv"
        else:
            filename = f"{run_label}_DSSAT_test_selected_TRT1_units.csv"

        path = os.path.join(run_output_dir, filename)
        selected_units_df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n试运行纳入的 {ANALYZE_SCENARIO} 样本清单已保存：{path}")


def load_or_process_one_dssat_dir(current_dssat_dir, run_label):
    """切换全局路径后，对一个 DSSAT 目录读取缓存或重新处理。"""

    global dssat_dir, output_dir, fig_dir, comparison_fig_dir, cache_dir

    # 切换目录前先清空上一轮预合并 raw cache，避免不同作物/方向互相串用。
    clear_current_dssat_merged_raw_cache_from_memory()

    dssat_dir = current_dssat_dir
    output_dir = os.path.join(BASE_OUTPUT_DIR, run_label)
    fig_dir = os.path.join(output_dir, "figures")
    comparison_fig_dir = os.path.join(
        fig_dir,
        "event_minus_normal_full_early_middle_late_TRT1_only"
    )
    cache_dir = os.path.join(output_dir, "processed_cache")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(comparison_fig_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    file_df, non_target_scenarios_ignored_count = build_file_df_for_dssat_dir(dssat_dir)

    cache_params = build_processing_cache_params(file_df)
    cache_paths = build_processed_cache_paths(cache_params)

    results = None

    if ENABLE_PROCESSED_DATA_CACHE and not FORCE_REBUILD_PROCESSED_DATA_CACHE:
        results = load_processed_data_cache(cache_paths, cache_params)

    if results is None:
        if FORCE_REBUILD_PROCESSED_DATA_CACHE:
            print("\nFORCE_REBUILD_PROCESSED_DATA_CACHE = True，强制重新读取原始文件。")
        else:
            print("\n没有可用处理结果缓存，开始读取原始 DSSAT 和气象文件并重新处理。")

        # 只有当处理后缓存没有命中时，才加载/构建 DSSAT daily 预合并缓存。
        # 这样可以避免明明已经可以直接绘图，却还把大型 raw daily 表读进内存。
        load_or_build_dssat_merged_raw_cache(
            file_df=file_df,
            crop_name=CURRENT_CROP_NAME,
            run_label=run_label
        )

        results = run_processing_pipeline(
            file_df=file_df,
            non_target_scenarios_ignored_count=non_target_scenarios_ignored_count
        )

        if ENABLE_PROCESSED_DATA_CACHE:
            save_processed_data_cache(cache_paths, cache_params, results)

        # 处理完成后清空 raw cache 引用，降低多作物连续运行时的内存压力。
        clear_current_dssat_merged_raw_cache_from_memory()

    save_result_tables_for_run(
        results=results,
        run_output_dir=output_dir,
        run_label=run_label
    )

    return results


# ============================================================
# 14. 分作物处理正负 DFAI 对应的 DSSAT 目录
# ============================================================

crop_results = []

for crop_cfg in CROP_CONFIGS:

    crop_name = str(crop_cfg.get("crop_name", "unknown_crop"))
    crop_safe_name = safe_filename(crop_name)

    # 每个作物进入处理前，先切换该作物自己的产量阈值和生育期/种植时长阈值。
    crop_qc_thresholds = apply_crop_qc_thresholds_for_crop(crop_cfg)

    positive_dssat_dir = crop_cfg.get("positive_dssat_dir")
    negative_dssat_dir = crop_cfg.get("negative_dssat_dir")

    positive_run_label = f"{crop_safe_name}_DFAI_gt4_from_DSSAT_DAILY_plus"
    negative_run_label = f"{crop_safe_name}_DFAI_lt_minus4_from_DSSAT_DAILY"

    print("\n" + "=" * 70)
    print(f"开始处理作物：{crop_name}")
    print(f"  DFAI > 4 DSSAT 目录：{positive_dssat_dir}")
    print(f"  DFAI < -4 DSSAT 目录：{negative_dssat_dir}")
    print("=" * 70)

    positive_results = None
    negative_results = None

    try:
        positive_results = load_or_process_one_dssat_dir(
            current_dssat_dir=positive_dssat_dir,
            run_label=positive_run_label,
        )
    except Exception as e:
        print(f"\n{crop_name} 的 DFAI > 4 数据处理失败，后续合并图中该部分会显示为空。")
        print(f"错误信息：{e}")
        positive_results = {}

    try:
        negative_results = load_or_process_one_dssat_dir(
            current_dssat_dir=negative_dssat_dir,
            run_label=negative_run_label,
        )
    except Exception as e:
        print(f"\n{crop_name} 的 DFAI < -4 数据处理失败，后续合并图中该部分会显示为空。")
        print(f"错误信息：{e}")
        negative_results = {}

    positive_event_daily_total = positive_results.get("event_daily_total", pd.DataFrame())
    negative_event_daily_total = negative_results.get("event_daily_total", pd.DataFrame())

    positive_event_total = positive_results.get("event_total", pd.DataFrame())
    negative_event_total = negative_results.get("event_total", pd.DataFrame())

    positive_quality_df = positive_results.get("quality_df", pd.DataFrame())
    negative_quality_df = negative_results.get("quality_df", pd.DataFrame())

    # 给后续合并表和合并图保留作物信息。
    if not positive_event_daily_total.empty:
        positive_event_daily_total = add_crop_column(
            positive_event_daily_total,
            crop_name=crop_name,
            dfai_direction="positive",
            run_label=positive_run_label,
        )

    if not negative_event_daily_total.empty:
        negative_event_daily_total = add_crop_column(
            negative_event_daily_total,
            crop_name=crop_name,
            dfai_direction="negative",
            run_label=negative_run_label,
        )

    if not positive_event_total.empty:
        positive_event_total = add_crop_column(
            positive_event_total,
            crop_name=crop_name,
            dfai_direction="positive",
            run_label=positive_run_label,
        )

    if not negative_event_total.empty:
        negative_event_total = add_crop_column(
            negative_event_total,
            crop_name=crop_name,
            dfai_direction="negative",
            run_label=negative_run_label,
        )

    if not positive_quality_df.empty:
        positive_quality_df = add_crop_column(
            positive_quality_df,
            crop_name=crop_name,
            dfai_direction="positive",
            run_label=positive_run_label,
        )

    if not negative_quality_df.empty:
        negative_quality_df = add_crop_column(
            negative_quality_df,
            crop_name=crop_name,
            dfai_direction="negative",
            run_label=negative_run_label,
        )

    crop_results.append({
        "crop_name": crop_name,
        "positive_dssat_dir": positive_dssat_dir,
        "negative_dssat_dir": negative_dssat_dir,
        "crop_qc_thresholds": crop_qc_thresholds,
        "positive_run_label": positive_run_label,
        "negative_run_label": negative_run_label,
        "positive_results": positive_results,
        "negative_results": negative_results,
        "positive_event_daily_total": positive_event_daily_total,
        "negative_event_daily_total": negative_event_daily_total,
        "positive_event_total": positive_event_total,
        "negative_event_total": negative_event_total,
        "positive_quality_df": positive_quality_df,
        "negative_quality_df": negative_quality_df,
    })


# ============================================================
# 15. 生成四个作物 2×2 排列在同一张图中的合并文件
# ============================================================

requested_fig_dir = os.path.join(BASE_OUTPUT_DIR, "figures", "requested_four_crop_2x2_figures")
os.makedirs(requested_fig_dir, exist_ok=True)

if not crop_results:
    print("没有任何作物成功进入结果列表，无法生成多作物合并文件。")
else:
    # 合并 CSV 文件：方便后续复核或二次绘图。
    save_combined_crop_tables(
        crop_results=crop_results,
        output_dir=requested_fig_dir,
    )

    # 图 1：四个作物 2×2 排列的 DFAI > 4 与 DFAI < -4 对比总图。
    plot_multi_crop_dfai_positive_negative_one_figure_by_stage(
        crop_results=crop_results,
        variables=PLOT_VARIABLES,
        output_dir=requested_fig_dir,
        min_events_per_group=MIN_EVENTS_PER_PLOT,
    )

    # 图 2：四个作物 2×2 排列的 DFAI > 4 与 SPEI 洪涝 / 干旱对比总图。
    plot_multi_crop_dfai_spei_one_figure_by_stage(
        crop_results=crop_results,
        dfai_direction="positive",
        variables=PLOT_VARIABLES,
        output_dir=requested_fig_dir,
        min_events_per_group=MIN_EVENTS_PER_PLOT,
    )

    # 图 3：四个作物 2×2 排列的 DFAI < -4 与 SPEI 洪涝 / 干旱对比总图。
    plot_multi_crop_dfai_spei_one_figure_by_stage(
        crop_results=crop_results,
        dfai_direction="negative",
        variables=PLOT_VARIABLES,
        output_dir=requested_fig_dir,
        min_events_per_group=MIN_EVENTS_PER_PLOT,
    )

print(f"\n四个作物 2×2 排列的合并图和合并表已保存到：{requested_fig_dir}")
print("\n全部处理完成！")
