import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch


# ============================================================
# 1. 基础配置
# ============================================================

OUTPUT_DIR = r"D:\Python\DssAat\NO_3\HUITUTU\DFAA1_DFAA4_TRT_7Comparison_Nature_Violin_STRICT_YEAR"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CROP_ORDER = ["Maize", "Rice", "Soybeans", "Wheat"]
CATEGORY_ORDER = ["DTF", "FTD"]

REQUIRE_ALL_CROPS = True
EPS = 1e-9

# False: (compare_trt - base_trt) / base_trt
# True : ((compare_trt - base_trt) / base_trt) * 100
RELATIVE_AS_PERCENT = False

IGNORE_ZERO_BEFORE_MEAN = True

OUTPUT_DPI = 600
SAVE_PNG = True
SAVE_PDF = True

SAVE_LONG_CSV = True
SAVE_SUMMARY_CSV = False

# 调试用：只画前几个变量。None 表示全部变量。
# 例如先测试第一个变量：MAX_TARGETS_TO_DRAW = 1
MAX_TARGETS_TO_DRAW = None


# ============================================================
# 2. 严格配对设置
# ============================================================
# 不同 TRT 对比时，除了 trt 不同，下面这些键必须完全一致。
# year 强制要求存在，因为你要求“同一个时间”。
# site_folder 如果存在，也会强制加入配对键。
# soil_id 强制加入配对键。

REQUIRE_YEAR_FOR_PAIRING = True
USE_SITE_FOLDER_FOR_PAIRING = True
USE_SOIL_ID_FOR_PAIRING = True


# ============================================================
# 3. 只保留 DFAA 1 和 DFAA 4
# ============================================================

SOURCE_ORDER = ["DFAA 1", "DFAA 4"]

SOURCE_LABELS_BY_CATEGORY = {
    "DTF": {
        "DFAA 1": "DFAA > 1",
        "DFAA 4": "DFAA > 4"
    },
    "FTD": {
        "DFAA 1": "DFAA < -1",
        "DFAA 4": "DFAA < -4"
    }
}


# ============================================================
# 4. TRT 对比设置：7 个对比全部放在一张图
# ============================================================

BASE_TRT_LOW = "TRT1"
BASE_TRT_HIGH = "TRT6"


def make_compare_label(compare_trt, base_trt):
    return f"{compare_trt}_vs_{base_trt}"


COMPARE_PAIRS = [
    ("TRT2", BASE_TRT_LOW),
    ("TRT3", BASE_TRT_LOW),
    ("TRT4", BASE_TRT_LOW),

    (BASE_TRT_HIGH, BASE_TRT_LOW),

    ("TRT7", BASE_TRT_HIGH),
    ("TRT8", BASE_TRT_HIGH),
    ("TRT9", BASE_TRT_HIGH)
]

ALL_COMPARE_ORDER = [
    make_compare_label("TRT2", BASE_TRT_LOW),
    make_compare_label("TRT3", BASE_TRT_LOW),
    make_compare_label("TRT4", BASE_TRT_LOW),

    make_compare_label(BASE_TRT_HIGH, BASE_TRT_LOW),

    make_compare_label("TRT7", BASE_TRT_HIGH),
    make_compare_label("TRT8", BASE_TRT_HIGH),
    make_compare_label("TRT9", BASE_TRT_HIGH)
]

TRT_GROUPS = {
    "TRT_all_7_comparisons": ALL_COMPARE_ORDER
}

TRT_GROUP_TITLES = {
    "TRT_all_7_comparisons": (
        f"TRT2-TRT4 relative to {BASE_TRT_LOW}; "
        f"{BASE_TRT_HIGH} relative to {BASE_TRT_LOW}; "
        f"TRT7-TRT9 relative to {BASE_TRT_HIGH}"
    )
}


COMPARE_SECTION_MAP = {
    make_compare_label("TRT2", BASE_TRT_LOW): "LOW_TO_TRT1",
    make_compare_label("TRT3", BASE_TRT_LOW): "LOW_TO_TRT1",
    make_compare_label("TRT4", BASE_TRT_LOW): "LOW_TO_TRT1",

    make_compare_label(BASE_TRT_HIGH, BASE_TRT_LOW): "TRT6_TO_TRT1",

    make_compare_label("TRT7", BASE_TRT_HIGH): "HIGH_TO_TRT6",
    make_compare_label("TRT8", BASE_TRT_HIGH): "HIGH_TO_TRT6",
    make_compare_label("TRT9", BASE_TRT_HIGH): "HIGH_TO_TRT6"
}

COMPARE_SECTION_ORDER = [
    "LOW_TO_TRT1",
    "TRT6_TO_TRT1",
    "HIGH_TO_TRT6"
]

COMPARE_SECTION_COLORS = {
    "LOW_TO_TRT1": "#4C78A8",
    "TRT6_TO_TRT1": "#7E6E85",
    "HIGH_TO_TRT6": "#D67D6E"
}

COMPARE_SECTION_LABELS = {
    "LOW_TO_TRT1": f"TRT2-TRT4 / {BASE_TRT_LOW}",
    "TRT6_TO_TRT1": f"{BASE_TRT_HIGH} / {BASE_TRT_LOW}",
    "HIGH_TO_TRT6": f"TRT7-TRT9 / {BASE_TRT_HIGH}"
}

SOURCE_HATCHES = {
    "DFAA 1": ".....",
    "DFAA 4": "/////"
}


def get_compare_section(compare_label):
    return COMPARE_SECTION_MAP.get(compare_label, "LOW_TO_TRT1")


def get_compare_color(compare_label):
    return COMPARE_SECTION_COLORS.get(
        get_compare_section(compare_label),
        "0.80"
    )


def get_compare_display_label(compare_label):
    return str(compare_label).replace("_vs_", " / ")


# ============================================================
# 5. 目标变量
# ============================================================
# 只想画一个变量时，例如：
# TARGET_COLS = ["Yield_HA"]

COL_GROUPS = [
    (
        "Phenology",
        [
            "Anthesis_DAP",
            "Maturity_DAP"
        ]
    ),
    (
        "Production",
        [
            "Yield_HA",
            "Tops_Maturity",
            "Tops_Anthesis",
            "Harvest_Index"
        ]
    ),
    (
        "Water stress",
        [
            "PreAnthesis_WaterPhotoStress_Mean",
            "PreAnthesis_WaterGrowthStress_Mean",
            "GrainFilling_WaterPhotoStress",
            "GrainFilling_WaterGrowthStress"
        ]
    ),
    (
        "Water balance",
        [
            "Precip_mm",
            "ET_mm",
            "Transp_mm"
        ]
    ),
    (
        "Nitrogen",
        [
            "N_Uptake_kg_ha"
        ]
    ),
    (
        "Water productivity",
        [
            "DM_Precip_Prod",
            "Yld_Precip_Prod",
            "DM_ET_Prod",
            "Yld_ET_Prod",
            "DM_Transp_Prod",
            "Yld_Transp_Prod"
        ]
    )
]

TARGET_COLS = []
for _, cols in COL_GROUPS:
    TARGET_COLS.extend(cols)

PARAM_LABELS = {
    "Anthesis_DAP": "Anthesis",
    "Maturity_DAP": "Maturity",
    "Yield_HA": "Yield",
    "Tops_Maturity": "Tops maturity",
    "Tops_Anthesis": "Tops anthesis",
    "Harvest_Index": "Harvest index",
    "PreAnthesis_WaterPhotoStress_Mean": "Pre-anthesis water photo stress",
    "PreAnthesis_WaterGrowthStress_Mean": "Pre-anthesis water growth stress",
    "GrainFilling_WaterPhotoStress": "Grain filling water photo stress",
    "GrainFilling_WaterGrowthStress": "Grain filling water growth stress",
    "Precip_mm": "Precipitation",
    "ET_mm": "ET",
    "Transp_mm": "Transpiration",
    "N_Uptake_kg_ha": "N uptake",
    "DM_Precip_Prod": "DM precipitation productivity",
    "Yld_Precip_Prod": "Yield precipitation productivity",
    "DM_ET_Prod": "DM ET productivity",
    "Yld_ET_Prod": "Yield ET productivity",
    "DM_Transp_Prod": "DM transpiration productivity",
    "Yld_Transp_Prod": "Yield transpiration productivity"
}


# ============================================================
# 6. Nature 风格图形参数
# ============================================================

CM_TO_INCH = 1 / 2.54

COMBINED_FIG_WIDTH_CM = 24.0
COMBINED_FIG_HEIGHT_CM = 24.0

FONT_FAMILY = "Arial"
BASE_FONT_SIZE = 7.0
TITLE_FONT_SIZE = 7.0
SUPTITLE_FONT_SIZE = 8.0
AXIS_LABEL_SIZE = 6.9
TICK_LABEL_SIZE = 6.0
ANNOTATION_FONT_SIZE = 5.7
PANEL_LETTER_SIZE = 8.0
LEGEND_FONT_SIZE = 6.3

AXIS_LINEWIDTH = 0.50
TICK_WIDTH = 0.50
TICK_LENGTH = 2.0
ZERO_LINEWIDTH = 0.40

VIOLIN_EDGE_COLOR = "black"
VIOLIN_ALPHA = 0.80
VIOLIN_LINEWIDTH = 0.45

VIOLIN_MAX_HALF_WIDTH = 0.135
VIOLIN_DENSITY_POINTS = 220
VIOLIN_DENSITY_MIN_RATIO = 0.025
VIOLIN_BW_ADJUST = 1.15

SOURCE_X_OFFSET = {
    "DFAA 1": -0.17,
    "DFAA 4": 0.17
}

SHOW_IQR_INTERVAL = True
IQR_LINEWIDTH = 1.25
MEDIAN_TICK_WIDTH = 0.13
MEDIAN_TICK_LINEWIDTH = 0.85

SHOW_STRIP_POINTS = False
STRIP_POINT_SIZE = 1.0
STRIP_POINT_ALPHA = 0.22

SHARE_Y_WITHIN_FIG = False

Y_LIMIT_METHOD = "quantile"
Y_LIMIT_QUANTILE = 0.05
Y_LIMIT_QUANTILE_BY_GROUP = True
Y_LIMIT_INCLUDE_ZERO = True
Y_QUANTILE_PADDING_RATIO = 0.10
Y_LIMIT_MIN_PADDING = 0.01
Y_LIMIT_MIN_SPAN = 0.04
Y_AXIS_SYMMETRIC_AROUND_ZERO = True
FILTER_VIOLIN_TO_AXIS_RANGE = True


# ============================================================
# 7. 文件路径
# ============================================================

def make_total_files(crop, category):
    """
    只读取 DFAA 1、DFAA 4。
    本脚本计算的是 TRT 之间的相对变化，因此不再读取 ALL。
    """
    if category == "DTF":
        sign = "+0"
        dfaa_tags = {
            "DFAA 1": "1",
            "DFAA 4": "4"
        }
    elif category == "FTD":
        sign = "-0"
        dfaa_tags = {
            "DFAA 1": "neg1",
            "DFAA 4": "neg4"
        }
    else:
        raise ValueError("category 只能是 DTF 或 FTD")

    base_dir = fr"D:\{crop}\OVERVIEW_LDFAI{sign}"

    files = {
        "DFAA 1": fr"{base_dir}\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_{dfaa_tags['DFAA 1']}.csv",
        "DFAA 4": fr"{base_dir}\OVERVIEW_LDFAI\ALL_最终筛选年份_总表_{dfaa_tags['DFAA 4']}.csv"
    }

    return files


CROP_TOTAL_FILES = {
    crop: {
        category: make_total_files(crop, category)
        for category in CATEGORY_ORDER
    }
    for crop in CROP_ORDER
}


# ============================================================
# 8. 工具函数
# ============================================================

def natural_sort_key(x):
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", str(x))
    ]


def sanitize_filename(text):
    text = str(text)
    text = text.replace(">", "gt").replace("<", "lt")
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_+\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "NA"


def normalize_trt(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip().upper()
    match = re.search(r"(\d+)", text)

    if match:
        return f"TRT{int(match.group(1))}"

    return text


def get_source_label(category, source):
    return SOURCE_LABELS_BY_CATEGORY.get(category, {}).get(source, source)


def read_one_csv(file_path):
    try:
        return pd.read_csv(file_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="gbk")


def read_total_files(file_dict, crop_label, category_label):
    df_list = []

    for source_label, file_path in file_dict.items():
        if not os.path.exists(file_path):
            print(f"⚠️ 文件不存在，跳过：{crop_label} | {category_label} | {source_label} | {file_path}")
            continue

        try:
            df = read_one_csv(file_path)
        except Exception as e:
            print(f"❌ 文件读取失败：{crop_label} | {category_label} | {source_label}")
            print(f"   路径：{file_path}")
            print(f"   错误：{e}")
            continue

        df["Crop"] = crop_label
        df["Category"] = category_label
        df["Source"] = source_label

        df_list.append(df)

        print(f"✅ 已读取：{crop_label} | {category_label} | {source_label} | 行数={len(df)}")

    if not df_list:
        return pd.DataFrame()

    return pd.concat(df_list, ignore_index=True)


def build_pair_id_cols(df):
    """
    构建严格配对键。
    不同 TRT 对比时，这些列必须完全相同。
    """
    pair_cols = ["Crop", "Category", "Source"]

    if USE_SITE_FOLDER_FOR_PAIRING:
        if "site_folder" not in df.columns:
            raise ValueError("❌ 要求同一站点配对，但数据中没有 site_folder 列。")
        pair_cols.append("site_folder")

    if USE_SOIL_ID_FOR_PAIRING:
        if "soil_id" not in df.columns:
            raise ValueError("❌ 要求同一 soil_id 配对，但数据中没有 soil_id 列。")
        pair_cols.append("soil_id")

    if REQUIRE_YEAR_FOR_PAIRING:
        if "year" not in df.columns:
            raise ValueError("❌ 要求同一时间配对，但数据中没有 year 列。")
        pair_cols.append("year")

    return pair_cols


def load_all_crop_data():
    all_list = []

    for crop in CROP_ORDER:
        for category in CATEGORY_ORDER:
            df_one = read_total_files(
                CROP_TOTAL_FILES[crop][category],
                crop_label=crop,
                category_label=category
            )
            if not df_one.empty:
                all_list.append(df_one)

    if not all_list:
        raise ValueError("❌ 没有读取到任何数据，请检查路径。")

    df_all = pd.concat(all_list, ignore_index=True)

    required_cols = ["Crop", "Category", "Source", "soil_id", "trt", "year"]
    for col in required_cols:
        if col not in df_all.columns:
            raise ValueError(f"❌ 数据中缺少必要列：{col}")

    if USE_SITE_FOLDER_FOR_PAIRING and "site_folder" not in df_all.columns:
        raise ValueError("❌ USE_SITE_FOLDER_FOR_PAIRING=True，但数据中没有 site_folder 列。")

    df_all["trt"] = df_all["trt"].apply(normalize_trt)

    loaded_crops = set(df_all["Crop"].dropna().astype(str).unique())
    missing_crops = [crop for crop in CROP_ORDER if crop not in loaded_crops]

    print("\n==============================")
    print("✅ 数据读取完成")
    print("作物：")
    print(df_all["Crop"].value_counts())
    print("Category：")
    print(df_all["Category"].value_counts())
    print("Source：")
    print(df_all["Source"].value_counts())
    print("trt：")
    print(df_all["trt"].value_counts().sort_index())
    print("year：")
    print(df_all["year"].value_counts().sort_index())
    print("==============================\n")

    if missing_crops:
        print(f"⚠️ 未读取到的作物：{missing_crops}")
        if REQUIRE_ALL_CROPS:
            raise ValueError(
                "❌ 以下作物没有被成功读取："
                + ", ".join(missing_crops)
                + "。请检查对应路径。"
            )

    available_trts = set(df_all["trt"].dropna().astype(str).unique())
    needed_trts = sorted(
        set([x for pair in COMPARE_PAIRS for x in pair]),
        key=natural_sort_key
    )
    missing_trts = [t for t in needed_trts if t not in available_trts]

    if missing_trts:
        print(f"⚠️ 以下 TRT 在数据中不存在，相关对比会自动跳过：{missing_trts}")

    pair_cols = build_pair_id_cols(df_all)
    print("✅ 当前严格配对键：")
    print(pair_cols)
    print("📌 不同 TRT 对比时，上述列必须完全相同，只有 trt 不同。\n")

    return df_all


# ============================================================
# 9. 数据整理：计算 TRT 对比变化
# ============================================================

def prepare_relative_for_one_target(df_all, target_col):
    """
    严格同站点、同 soil_id、同 year 内部做 TRT 对比。

    不同情景对比时，配对键为：
        Crop + Category + Source + site_folder + soil_id + year

    只有 trt 不同。
    """
    pair_cols = build_pair_id_cols(df_all)

    required_cols = pair_cols + ["trt", target_col]

    for col in required_cols:
        if col not in df_all.columns:
            print(f"⚠️ 缺少列 {col}，跳过参数：{target_col}")
            return pd.DataFrame()

    df = df_all[required_cols].copy()
    df = df[df["Source"].isin(SOURCE_ORDER)].copy()

    if df.empty:
        print(f"⚠️ {target_col}: 没有 DFAA 1 / DFAA 4 数据，跳过")
        return pd.DataFrame()

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    raw_n = len(df)
    df = df.dropna(subset=[target_col]).copy()

    if IGNORE_ZERO_BEFORE_MEAN:
        before_zero = len(df)
        df = df[df[target_col].abs() > EPS].copy()
        removed_zero = before_zero - len(df)
    else:
        removed_zero = 0

    if df.empty:
        print(f"⚠️ {target_col} 去掉 NaN/0 后为空，跳过")
        return pd.DataFrame()

    group_cols = pair_cols + ["trt"]

    df_mean = (
        df
        .groupby(group_cols, as_index=False)
        .agg(
            value_mean=(target_col, "mean"),
            n_raw_records=(target_col, "size")
        )
    )

    available_trts = set(df_mean["trt"].dropna().astype(str).unique())

    df_rel_list = []

    for compare_trt, base_trt in COMPARE_PAIRS:
        compare_label = make_compare_label(compare_trt, base_trt)

        if compare_trt not in available_trts:
            print(f"⚠️ {target_col}: 缺少 {compare_trt}，跳过 {compare_label}")
            continue

        if base_trt not in available_trts:
            print(f"⚠️ {target_col}: 缺少基准 {base_trt}，跳过 {compare_label}")
            continue

        compare_df = df_mean[df_mean["trt"] == compare_trt].copy()
        compare_df = compare_df.rename(
            columns={
                "value_mean": "compare_mean",
                "n_raw_records": "n_compare_records"
            }
        )
        compare_df = compare_df[
            pair_cols + ["compare_mean", "n_compare_records"]
        ].copy()

        base_df = df_mean[df_mean["trt"] == base_trt].copy()
        base_df = base_df.rename(
            columns={
                "value_mean": "base_mean",
                "n_raw_records": "n_base_records"
            }
        )
        base_df = base_df[
            pair_cols + ["base_mean", "n_base_records"]
        ].copy()

        # 关键：严格按 pair_cols 合并。
        # pair_cols 已包含 site_folder + soil_id + year。
        df_cmp = compare_df.merge(
            base_df,
            on=pair_cols,
            how="inner"
        )

        if df_cmp.empty:
            print(f"⚠️ {target_col}: {compare_label} 严格配对后为空，跳过")
            continue

        before_base_zero = len(df_cmp)
        df_cmp = df_cmp[df_cmp["base_mean"].abs() > EPS].copy()
        removed_base_zero = before_base_zero - len(df_cmp)

        if df_cmp.empty:
            print(f"⚠️ {target_col}: {compare_label} 基准值全为 0，跳过")
            continue

        df_cmp["relative_value"] = (
            df_cmp["compare_mean"] - df_cmp["base_mean"]
        ) / df_cmp["base_mean"]

        if RELATIVE_AS_PERCENT:
            df_cmp["relative_value"] = df_cmp["relative_value"] * 100

        df_cmp["relative_value"] = pd.to_numeric(
            df_cmp["relative_value"],
            errors="coerce"
        )
        df_cmp = df_cmp.replace([np.inf, -np.inf], np.nan)
        df_cmp = df_cmp.dropna(subset=["relative_value"]).copy()

        if df_cmp.empty:
            print(f"⚠️ {target_col}: {compare_label} 计算后为空，跳过")
            continue

        df_cmp["Target"] = target_col
        df_cmp["Target_Label"] = PARAM_LABELS.get(target_col, target_col)

        df_cmp["trt"] = compare_label
        df_cmp["Comparison"] = compare_label
        df_cmp["Compare_TRT"] = compare_trt
        df_cmp["Base_TRT"] = base_trt
        df_cmp["Compare_Section"] = get_compare_section(compare_label)

        df_cmp["Source_Label"] = df_cmp.apply(
            lambda row: get_source_label(row["Category"], row["Source"]),
            axis=1
        )

        df_rel_list.append(df_cmp)

        print(
            f"✅ {target_col}: {compare_label}, "
            f"严格配对数={len(df_cmp)}, "
            f"去基准0={removed_base_zero}"
        )

    if not df_rel_list:
        print(f"⚠️ {target_col}: 所有 TRT 对比均为空，跳过")
        return pd.DataFrame()

    df_rel = pd.concat(df_rel_list, ignore_index=True)

    print(
        f"✅ {target_col}: 原始={raw_n}, 去0={removed_zero}, "
        f"最终严格 TRT 对比数据={len(df_rel)}"
    )

    return df_rel


def build_all_plot_data(df_all):
    df_list = []

    for target_col in TARGET_COLS:
        one = prepare_relative_for_one_target(df_all, target_col)
        if not one.empty:
            df_list.append(one)

    if not df_list:
        raise ValueError("❌ 没有任何变量生成有效绘图数据。")

    return pd.concat(df_list, ignore_index=True)


def build_violin_summary(df_rel_all):
    group_cols = [
        "Target",
        "Target_Label",
        "Crop",
        "Category",
        "Source",
        "Source_Label",
        "trt",
        "Comparison",
        "Compare_TRT",
        "Base_TRT",
        "Compare_Section"
    ]

    for col in ["site_folder", "soil_id", "year"]:
        if col in df_rel_all.columns and col not in group_cols:
            group_cols.insert(4, col)

    summary = (
        df_rel_all
        .groupby(group_cols, dropna=False)["relative_value"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max"
        )
        .reset_index()
    )

    summary["std"] = summary["std"].fillna(0)

    return summary


# ============================================================
# 10. y 轴与裁剪
# ============================================================

def get_y_limits(data, value_col="relative_value", mean_group_cols=None):
    if data is None or data.empty:
        return None

    df = data.copy()

    if value_col not in df.columns:
        return None

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[value_col]).copy()

    if df.empty:
        return None

    raw_values = df[value_col]

    if Y_LIMIT_METHOD == "quantile":
        q = Y_LIMIT_QUANTILE

        valid_group_cols = []
        if Y_LIMIT_QUANTILE_BY_GROUP and mean_group_cols is not None:
            valid_group_cols = [
                col for col in mean_group_cols
                if col in df.columns
            ]

        if valid_group_cols:
            grouped = df.groupby(valid_group_cols, dropna=False)[value_col]
            q_low = grouped.quantile(q).dropna()
            q_high = grouped.quantile(1 - q).dropna()
            ref_values = pd.concat([q_low, q_high], ignore_index=True)
        else:
            ref_values = pd.Series([
                raw_values.quantile(q),
                raw_values.quantile(1 - q)
            ])
    else:
        ref_values = raw_values

    ref_values = (
        pd.to_numeric(ref_values, errors="coerce")
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

    if Y_AXIS_SYMMETRIC_AROUND_ZERO:
        half_span = max(
            abs(low),
            abs(high),
            Y_LIMIT_MIN_SPAN / 2.0
        )
        low = -half_span
        high = half_span
        span = high - low

    pad = max(
        span * Y_QUANTILE_PADDING_RATIO,
        Y_LIMIT_MIN_PADDING
    )

    return float(low - pad), float(high + pad)


def filter_data_to_y_limits(data, y_limits, value_col="relative_value"):
    if not FILTER_VIOLIN_TO_AXIS_RANGE or y_limits is None:
        return data

    if value_col not in data.columns:
        return data

    low, high = y_limits

    plot_df = data.copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=[value_col]).copy()

    plot_df = plot_df[
        (plot_df[value_col] >= low) &
        (plot_df[value_col] <= high)
    ].copy()

    if plot_df.empty:
        return data

    return plot_df


# ============================================================
# 11. Nature 风格绘图函数
# ============================================================

def apply_nature_axis_style(ax):
    ax.set_facecolor("white")
    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(AXIS_LINEWIDTH)
        ax.spines[side].set_color("black")

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=TICK_LENGTH,
        width=TICK_WIDTH,
        colors="black",
        pad=2,
        labelsize=TICK_LABEL_SIZE
    )

    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
    ax.margins(x=0.08)


def _draw_manual_violin_for_group(
    ax,
    values,
    x_pos,
    face_color,
    y_limits=None,
    max_half_width=None,
    hatch=None
):
    if max_half_width is None:
        max_half_width = VIOLIN_MAX_HALF_WIDTH

    values = (
        pd.to_numeric(pd.Series(values), errors="coerce")
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
            x_pos,
            y0 - 0.002,
            y0 + 0.002,
            color="black",
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

        breaks = np.where(np.diff(idx) > 1)[0] + 1
        chunks = np.split(idx, breaks)

        for chunk in chunks:
            if len(chunk) < 3:
                continue

            gy = grid[chunk]
            half_width = density[chunk] * max_half_width

            gy = np.r_[gy[0], gy, gy[-1]]
            half_width = np.r_[0.0, half_width, 0.0]

            ax.fill_betweenx(
                gy,
                x_pos - half_width,
                x_pos + half_width,
                facecolor=face_color,
                edgecolor=VIOLIN_EDGE_COLOR,
                linewidth=VIOLIN_LINEWIDTH,
                alpha=VIOLIN_ALPHA,
                hatch=hatch,
                zorder=2
            )

    except Exception:
        q25 = float(values_for_kde.quantile(0.25))
        q50 = float(values_for_kde.quantile(0.50))
        q75 = float(values_for_kde.quantile(0.75))

        ax.vlines(
            x_pos,
            q25,
            q75,
            color="black",
            linewidth=1.0,
            zorder=3
        )

        ax.hlines(
            q50,
            x_pos - 0.06,
            x_pos + 0.06,
            color="black",
            linewidth=0.9,
            zorder=4
        )


def draw_grouped_nature_violin(ax, data, trt_order, y_limits=None):
    x_positions = np.arange(len(trt_order))

    for i, trt_value in enumerate(trt_order):
        section_color = get_compare_color(trt_value)

        for source in SOURCE_ORDER:
            x_pos = x_positions[i] + SOURCE_X_OFFSET[source]

            group_values = data.loc[
                (data["trt"] == trt_value) &
                (data["Source"] == source),
                "relative_value"
            ]

            _draw_manual_violin_for_group(
                ax=ax,
                values=group_values,
                x_pos=x_pos,
                face_color=section_color,
                y_limits=y_limits,
                max_half_width=VIOLIN_MAX_HALF_WIDTH,
                hatch=SOURCE_HATCHES.get(source)
            )


def add_grouped_summary_overlays(ax, data, trt_order):
    if data.empty:
        return

    x_positions = np.arange(len(trt_order))

    for i, trt_value in enumerate(trt_order):
        for source in SOURCE_ORDER:
            x_pos = x_positions[i] + SOURCE_X_OFFSET[source]

            g = data.loc[
                (data["trt"] == trt_value) &
                (data["Source"] == source),
                "relative_value"
            ]

            g = (
                pd.to_numeric(g, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            if g.empty:
                continue

            q25 = float(g.quantile(0.25))
            median_value = float(g.median())
            q75 = float(g.quantile(0.75))

            if SHOW_IQR_INTERVAL:
                ax.vlines(
                    x_pos,
                    q25,
                    q75,
                    color="black",
                    linewidth=IQR_LINEWIDTH,
                    zorder=6
                )

                ax.hlines(
                    median_value,
                    x_pos - MEDIAN_TICK_WIDTH / 2,
                    x_pos + MEDIAN_TICK_WIDTH / 2,
                    colors="black",
                    linewidth=MEDIAN_TICK_LINEWIDTH,
                    zorder=7
                )


def add_comparison_section_guides(ax, trt_order):
    present = list(trt_order)

    if len(present) <= 1:
        return

    for i in range(len(present) - 1):
        current_section = get_compare_section(present[i])
        next_section = get_compare_section(present[i + 1])

        if current_section != next_section:
            ax.axvline(
                i + 0.5,
                color="0.45",
                linewidth=0.55,
                linestyle=(0, (2.0, 2.0)),
                zorder=1
            )


def setup_plot_style():
    plt.style.use("default")
    sns.set_theme(style="white", context="paper")

    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.sans-serif": [FONT_FAMILY, "Arial", "DejaVu Sans"],
        "font.size": BASE_FONT_SIZE,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "figure.facecolor": "white",
        "figure.dpi": OUTPUT_DPI,
        "savefig.dpi": OUTPUT_DPI,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",

        "axes.linewidth": AXIS_LINEWIDTH,
        "axes.edgecolor": "black",
        "axes.facecolor": "white",
        "axes.grid": False,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.titlesize": TITLE_FONT_SIZE,

        "xtick.major.width": TICK_WIDTH,
        "ytick.major.width": TICK_WIDTH,
        "xtick.major.size": TICK_LENGTH,
        "ytick.major.size": TICK_LENGTH,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,

        "text.color": "black",
        "legend.frameon": False,
        "legend.fontsize": LEGEND_FONT_SIZE
    })


# ============================================================
# 12. 绘图
# ============================================================

def draw_panel(
    ax,
    cell_df,
    crop,
    category,
    trt_order,
    target_label,
    panel_letter,
    show_y_label=True,
    show_x_label=True,
    shared_y_limits=None
):
    present_trts = [t for t in trt_order if t in set(cell_df["trt"])]

    if cell_df.empty or not present_trts:
        ax.text(
            0.5,
            0.5,
            "No data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=ANNOTATION_FONT_SIZE
        )

        ax.set_title(
            f"{crop} | {category}",
            fontsize=TITLE_FONT_SIZE,
            fontweight="normal",
            pad=3
        )

        apply_nature_axis_style(ax)
        return

    cell_df = cell_df[cell_df["trt"].isin(present_trts)].copy()

    if shared_y_limits is None:
        y_limits = get_y_limits(
            cell_df,
            value_col="relative_value",
            mean_group_cols=["trt", "Source"]
        )
    else:
        y_limits = shared_y_limits

    plot_df = filter_data_to_y_limits(
        cell_df,
        y_limits,
        value_col="relative_value"
    )

    draw_grouped_nature_violin(
        ax=ax,
        data=plot_df,
        trt_order=present_trts,
        y_limits=y_limits
    )

    add_grouped_summary_overlays(
        ax=ax,
        data=cell_df,
        trt_order=present_trts
    )

    if SHOW_STRIP_POINTS:
        x_positions = np.arange(len(present_trts))

        for i, trt_value in enumerate(present_trts):
            for source in SOURCE_ORDER:
                g = plot_df[
                    (plot_df["trt"] == trt_value) &
                    (plot_df["Source"] == source)
                ].copy()

                if g.empty:
                    continue

                x_center = x_positions[i] + SOURCE_X_OFFSET[source]
                jitter = np.random.normal(0, 0.025, size=len(g))

                ax.scatter(
                    x_center + jitter,
                    g["relative_value"],
                    s=STRIP_POINT_SIZE,
                    alpha=STRIP_POINT_ALPHA,
                    color="black",
                    linewidths=0,
                    zorder=4
                )

    ax.axhline(
        0,
        linewidth=ZERO_LINEWIDTH,
        linestyle=(0, (2.5, 2.5)),
        color="0.35",
        zorder=0
    )

    if y_limits is not None:
        ax.set_ylim(*y_limits)

    ax.text(
        -0.12,
        1.06,
        panel_letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LETTER_SIZE,
        fontweight="bold"
    )

    ax.set_title(
        f"{crop} | {category}",
        fontsize=TITLE_FONT_SIZE,
        fontweight="normal",
        pad=3
    )

    ax.set_xticks(np.arange(len(present_trts)))
    ax.set_xticklabels([get_compare_display_label(x) for x in present_trts])

    add_comparison_section_guides(
        ax=ax,
        trt_order=present_trts
    )

    if show_y_label:
        ylabel = "Relative change (%)" if RELATIVE_AS_PERCENT else "Relative change"
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    else:
        ax.set_ylabel("")

    if show_x_label:
        ax.set_xlabel("TRT comparison", fontsize=AXIS_LABEL_SIZE)
        for label in ax.get_xticklabels():
            label.set_rotation(35)
            label.set_ha("right")
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)

    apply_nature_axis_style(ax)


def add_dfaa_legend(fig):
    section_handles = [
        Patch(
            facecolor=COMPARE_SECTION_COLORS["LOW_TO_TRT1"],
            edgecolor="black",
            linewidth=VIOLIN_LINEWIDTH,
            alpha=VIOLIN_ALPHA,
            label=COMPARE_SECTION_LABELS["LOW_TO_TRT1"]
        ),
        Patch(
            facecolor=COMPARE_SECTION_COLORS["TRT6_TO_TRT1"],
            edgecolor="black",
            linewidth=VIOLIN_LINEWIDTH,
            alpha=VIOLIN_ALPHA,
            label=COMPARE_SECTION_LABELS["TRT6_TO_TRT1"]
        ),
        Patch(
            facecolor=COMPARE_SECTION_COLORS["HIGH_TO_TRT6"],
            edgecolor="black",
            linewidth=VIOLIN_LINEWIDTH,
            alpha=VIOLIN_ALPHA,
            label=COMPARE_SECTION_LABELS["HIGH_TO_TRT6"]
        )
    ]

    source_handles = [
        Patch(
            facecolor="white",
            edgecolor="black",
            linewidth=VIOLIN_LINEWIDTH,
            hatch=SOURCE_HATCHES["DFAA 1"],
            label="DFAA 1"
        ),
        Patch(
            facecolor="white",
            edgecolor="black",
            linewidth=VIOLIN_LINEWIDTH,
            hatch=SOURCE_HATCHES["DFAA 4"],
            label="DFAA 4"
        )
    ]

    handles = section_handles + source_handles

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(handles),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=0.9
    )


def draw_big_figure(df_rel_all, target_col, trt_group_name, trt_order):
    target_label = PARAM_LABELS.get(target_col, target_col)

    fig_df = df_rel_all[
        (df_rel_all["Target"] == target_col) &
        (df_rel_all["trt"].isin(trt_order))
    ].copy()

    if fig_df.empty:
        print(f"⚠️ 无数据，跳过：{target_col} | {trt_group_name}")
        return None

    shared_y_limits = None
    if SHARE_Y_WITHIN_FIG:
        shared_y_limits = get_y_limits(
            fig_df,
            value_col="relative_value",
            mean_group_cols=["Crop", "Category", "trt", "Source"]
        )

    fig, axes = plt.subplots(
        nrows=len(CROP_ORDER),
        ncols=len(CATEGORY_ORDER),
        figsize=(
            COMBINED_FIG_WIDTH_CM * CM_TO_INCH,
            COMBINED_FIG_HEIGHT_CM * CM_TO_INCH
        ),
        facecolor="white",
        sharey=SHARE_Y_WITHIN_FIG
    )

    panel_letters = list("abcdefghijklmnopqrstuvwxyz")
    panel_index = 0

    for row_idx, crop in enumerate(CROP_ORDER):
        for col_idx, category in enumerate(CATEGORY_ORDER):
            ax = axes[row_idx, col_idx]

            cell_df = fig_df[
                (fig_df["Crop"] == crop) &
                (fig_df["Category"] == category)
            ].copy()

            draw_panel(
                ax=ax,
                cell_df=cell_df,
                crop=crop,
                category=category,
                trt_order=trt_order,
                target_label=target_label,
                panel_letter=panel_letters[panel_index],
                show_y_label=(col_idx == 0),
                show_x_label=(row_idx == len(CROP_ORDER) - 1),
                shared_y_limits=shared_y_limits
            )

            panel_index += 1

    fig.suptitle(
        f"{target_label} | TRT comparisons under DFAA 1 and DFAA 4",
        fontsize=SUPTITLE_FONT_SIZE,
        fontweight="normal",
        y=0.988
    )

    add_dfaa_legend(fig)

    fig.tight_layout(rect=[0.03, 0.02, 0.99, 0.915])

    out_dir = Path(OUTPUT_DIR) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_target = sanitize_filename(target_col)
    out_base = out_dir / f"{safe_target}_DFAA1_DFAA4_7TRTcomparison_STRICT_YEAR_Nature_8panels"

    if SAVE_PNG:
        png_path = out_base.with_suffix(".png")
        fig.savefig(png_path, dpi=OUTPUT_DPI, bbox_inches="tight")
        print(f"✅ 已保存 PNG：{png_path}")

    if SAVE_PDF:
        pdf_path = out_base.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"✅ 已保存 PDF：{pdf_path}")

    plt.close(fig)

    return out_base


# ============================================================
# 13. 主程序
# ============================================================

def main():
    setup_plot_style()

    print("🚀 开始读取数据 ...")
    df_all = load_all_crop_data()

    print("🚀 开始计算 7 个 TRT 严格配对对比变化 ...")
    df_rel_all = build_all_plot_data(df_all)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_LONG_CSV:
        print("🚀 开始保存长表 ...")
        long_csv = out_dir / "DFAA1_DFAA4_7TRT_comparison_relative_long_table_STRICT_YEAR.csv"
        df_rel_all.to_csv(long_csv, index=False, encoding="utf-8-sig")
        print(f"✅ 长表已保存：{long_csv}")

    if SAVE_SUMMARY_CSV:
        print("🚀 开始生成 summary，这一步数据量大时可能较慢 ...")
        summary = build_violin_summary(df_rel_all)

        print("🚀 开始保存 summary ...")
        summary_csv = out_dir / "DFAA1_DFAA4_7TRT_comparison_relative_summary_STRICT_YEAR.csv"
        summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        print(f"✅ 汇总表已保存：{summary_csv}")
    else:
        print("⏭️ 已跳过 summary csv 保存，直接开始绘图。")

    output_count = 0

    target_list = sorted(df_rel_all["Target"].unique(), key=natural_sort_key)

    if MAX_TARGETS_TO_DRAW is not None:
        target_list = target_list[:MAX_TARGETS_TO_DRAW]
        print(f"⚠️ 当前为测试模式，只绘制前 {MAX_TARGETS_TO_DRAW} 个变量：{target_list}")

    print("🚀 开始绘图 ...")

    for target_col in target_list:
        print(f"🎨 正在绘制变量：{target_col}")

        for trt_group_name, trt_order in TRT_GROUPS.items():

            subset = df_rel_all[
                (df_rel_all["Target"] == target_col) &
                (df_rel_all["trt"].isin(trt_order))
            ].copy()

            if subset.empty:
                print(f"⚠️ {target_col} | {trt_group_name} 无数据，跳过")
                continue

            draw_big_figure(
                df_rel_all=df_rel_all,
                target_col=target_col,
                trt_group_name=trt_group_name,
                trt_order=trt_order
            )

            output_count += 1

    print("\n🎉 全部完成！")
    print(f"📌 输出图片数量：{output_count}")
    print("📌 每个变量输出一张 8 小图大图")
    print("📌 每张图结构：4 作物 × DTF/FTD = 8 个小图")
    print("📌 横轴：7 个 TRT 对比")
    print("📌 严格配对键：Crop + Category + Source + site_folder + soil_id + year")
    print("📌 含义：同站点、同 soil_id、同年份、同 DFAA 情景内部，只比较不同 TRT")
    print("📌 横轴顺序：TRT2/TRT1, TRT3/TRT1, TRT4/TRT1, TRT6/TRT1, TRT7/TRT6, TRT8/TRT6, TRT9/TRT6")
    print("📌 竖向虚线：分隔 TRT2-4/TRT1、TRT6/TRT1、TRT7-9/TRT6 三部分")
    print("📌 填充颜色：表示三类 TRT 对比部分")
    print("📌 hatch：区分 DFAA 1 和 DFAA 4")
    print(f"📁 输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()