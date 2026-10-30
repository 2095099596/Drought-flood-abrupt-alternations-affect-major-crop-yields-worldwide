import os
import re
import warnings

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

warnings.filterwarnings("ignore")

# ===================== 全局样式 =====================
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["figure.max_open_warning"] = 50
plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Sans"]
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2
plt.rcParams["axes.facecolor"] = "white"

# ===================== 基础配置 =====================
METHODS = [
    "linear_bic",
    "quadratic_bic",
]

CROPS = [
    "maize",
    "rice",
    "soybeans",
    "wheat",
]

BASE_ROOT = r"C:\Users\lxh\Desktop\NongZuoWu_ChanLiang\New_result\Data\LDFAI_SPEI_SPEI_3_2"
OUTPUT_DIR = r"D:\Python\DssAat\NO_1\Figure1b"

GLOBAL_CONFIG = {
    "font_size": 12,
    "tick_size": 13,
    "legend_size": 15,
    "title_size": 15,
}

BAR_SPACING = 1.0

COLOR_MAP = {
    "planting": "#ff7f0e",      # 橙色
    "growing": "#2ca02c",       # 绿色
    "harvesting": "#1f77b4",    # 深蓝色
}

# ===================== 原始数据 95% 分布区间设置 =====================
# 使用原始数据的 2.5% 和 97.5% 分位数表示 95% 分布区间
SHOW_RAWDATA_95_INTERVAL = True
ERROR_BAR_CAPSIZE = 3
ERROR_BAR_LINEWIDTH = 1.2
AUTO_EXPAND_Y_FOR_INTERVAL = True

# ===================== 每个作物、每个 method 单独设置 y 轴范围 =====================
# 如需不同范围，直接修改这里即可
Y_AXIS_CONFIG = {
    "maize": {
        "linear_bic": {
            "bar_y_min": -0.10, "bar_y_max": 0.10,
            "scatter_y_min": -0.10, "scatter_y_max": 0.10,
        },
        "quadratic_bic": {
            "bar_y_min": -0.10, "bar_y_max": 0.10,
            "scatter_y_min": -0.10, "scatter_y_max": 0.10,
        },
    },
    "rice": {
        "linear_bic": {
            "bar_y_min": -0.10, "bar_y_max": 0.10,
            "scatter_y_min": -0.10, "scatter_y_max": 0.10,
        },
        "quadratic_bic": {
            "bar_y_min": -0.10, "bar_y_max": 0.10,
            "scatter_y_min": -0.10, "scatter_y_max": 0.10,
        },
    },
    "soybeans": {
        "linear_bic": {
            "bar_y_min": -0.05, "bar_y_max": 0.05,
            "scatter_y_min": -0.05, "scatter_y_max": 0.05,
        },
        "quadratic_bic": {
            "bar_y_min": -0.05, "bar_y_max": 0.05,
            "scatter_y_min": -0.05, "scatter_y_max": 0.05,
        },
    },
    "wheat": {
        "linear_bic": {
            "bar_y_min": -0.15, "bar_y_max": 0.15,
            "scatter_y_min": -0.15, "scatter_y_max": 0.15,
        },
        "quadratic_bic": {
            "bar_y_min": -0.15, "bar_y_max": 0.15,
            "scatter_y_min": -0.15, "scatter_y_max": 0.15,
        },
    },
}


def apply_y_axis_config(ax_bar, ax_scatter, crop, method):
    """给不同作物、不同 method 的子图单独设置 y 轴范围。"""
    cfg = Y_AXIS_CONFIG[crop][method]

    ax_bar.set_ylim(cfg["bar_y_min"], cfg["bar_y_max"])
    ax_scatter.set_ylim(cfg["scatter_y_min"], cfg["scatter_y_max"])

    print(
        f"✅ y 轴范围已设置 | 作物: {crop} | 方法: {method} | "
        f"左轴 Mean: [{cfg['bar_y_min']}, {cfg['bar_y_max']}] | "
        f"右轴 Residual: [{cfg['scatter_y_min']}, {cfg['scatter_y_max']}]"
    )


# ===================== 生成 LDFAI 标签：左正右负 + 显示平方值 =====================
def generate_ldfai_labels():
    positive = np.arange(3.0, 0.9, -0.2)   # 左：正数 3.0 → 1.0
    negative = np.arange(-1.0, -3.1, -0.2) # 右：负数 -1.0 → -3.0

    all_values = np.concatenate([positive, negative])

    tick_labels = []
    for v in all_values:
        sq = v ** 2
        if v > 0:
            tick_labels.append(f"LDFAI>{sq:.2f}")
        else:
            tick_labels.append(f"LDFAI<-{sq:.2f}")

    return tick_labels


LDFAI_TICK_LABELS = generate_ldfai_labels()


# ===================== 读取 nc 文件并计算统计量 =====================
def calculate_nc_stats(file_path):
    try:
        ds = xr.open_dataset(file_path, decode_cf=True)
        stats = {}

        for var in ds.data_vars:
            data = ds[var].values
            valid_mask = ~np.isnan(data)
            valid_data = data[valid_mask]

            valid_count = np.sum(valid_mask)
            mean_value = np.nanmean(data)

            if valid_count > 0:
                p2_5_value = np.nanpercentile(valid_data, 2.5)
                p97_5_value = np.nanpercentile(valid_data, 97.5)
            else:
                p2_5_value = np.nan
                p97_5_value = np.nan

            stats[var] = {
                "mean": mean_value,
                "valid_count": valid_count,
                "valid_data": valid_data,
                "p2_5": p2_5_value,
                "p97_5": p97_5_value,
            }

        ds.close()
        return stats

    except Exception as e:
        print(f"读取文件失败: {file_path}")
        print(f"错误信息: {e}")
        return None


# ===================== 从文件名中提取数值部分 =====================
def extract_numeric_part(filename):
    match = re.search(r"ls_(.*?)_residual", filename)
    return float(match.group(1)) if match else 0.0


# ===================== 批量处理 nc 文件 =====================
def process_all_nc_files(directory):
    directory = os.path.abspath(directory)
    nc_files = [f for f in os.listdir(directory) if f.endswith(".nc")]

    stage_priority = {
        "planting": 0,
        "growing": 1,
        "harvesting": 2,
    }

    def sort_key(fn):
        v = extract_numeric_part(fn)
        s = next((k for k in stage_priority if k in fn), None)
        return (-v, stage_priority.get(s, 99))

    nc_files.sort(key=sort_key)

    results = {}
    for fn in nc_files:
        fp = os.path.join(directory, fn)
        res = calculate_nc_stats(fp)
        if res:
            results[fn] = res

    return results


# ===================== 解析 LDFAI 标签 =====================
def parse_ldfai_label(label):
    """
    例如：
    LDFAI<-1.00 -> ("lt", 1.00)
    LDFAI>1.00  -> ("gt", 1.00)
    """
    s = str(label).replace(" ", "")

    try:
        if s.startswith("LDFAI<-"):
            value = float(s.replace("LDFAI<-", ""))
            return "lt", round(value, 2)

        if s.startswith("LDFAI>"):
            value = float(s.replace("LDFAI>", ""))
            return "gt", round(value, 2)

    except ValueError:
        return None

    return None


# ===================== 整理 planting / growing / harvesting 数据 =====================
def build_all_data(results):
    all_data = {}

    for stage in ["planting", "growing", "harvesting"]:
        mean_list = []
        data_list = []
        p2_5_list = []
        p97_5_list = []

        for fname, vdict in results.items():
            if stage in fname:
                for _, d in vdict.items():
                    mean_list.append(d["mean"])
                    data_list.append(d["valid_data"])
                    p2_5_list.append(d["p2_5"])
                    p97_5_list.append(d["p97_5"])

        all_data[stage] = {
            "mean": mean_list,
            "data": data_list,
            "p2_5": p2_5_list,
            "p97_5": p97_5_list,
            "n": len(mean_list),
        }

    return all_data


# ===================== 获取筛选 LDFAI 数据 =====================
def get_selected_data(results):
    all_data = build_all_data(results)
    N_common = min(d["n"] for d in all_data.values())

    if N_common == 0:
        return None

    x_tick_labels = LDFAI_TICK_LABELS[:N_common]

    # 只保留这两个阈值
    target_items = [
        (("lt", 4.00), "DFAI<-4"),
        (("lt", 1.00), "DFAI<-1"),
        (("gt", 1.00), "DFAI>1"),
        (("gt", 4.00), "DFAI>4"),
    ]

    available_label_index = {}
    for idx, label in enumerate(x_tick_labels[:N_common]):
        key = parse_ldfai_label(label)
        if key is not None and key not in available_label_index:
            available_label_index[key] = idx

    selected_indices = []
    selected_display_labels = []

    for key, display_label in target_items:
        if key in available_label_index:
            selected_indices.append(available_label_index[key])
            selected_display_labels.append(display_label)
        else:
            print(f"⚠️ 未找到目标 LDFAI 标签: {display_label}")

    if len(selected_indices) == 0:
        return None

    return {
        "all_data": all_data,
        "selected_indices": selected_indices,
        "selected_display_labels": selected_display_labels,
    }


# ===================== 在指定子图上绘制一个作物 =====================
def draw_one_crop_subplot(ax_bar, crop, method, selected_data, show_left_ylabel=False, show_bottom_xlabel=False):
    ax_scatter = ax_bar.twinx()

    all_data = selected_data["all_data"]
    selected_indices = selected_data["selected_indices"]
    selected_display_labels = selected_data["selected_display_labels"]

    hatch_patterns = ["///", "...", "+++"]
    rng = np.random.default_rng(2024)

    max_scatter_points = 1000
    jitter_width = 0.025
    scatter_alpha = 0.07
    scatter_size = 6

    x_base_selected = np.arange(len(selected_indices)) * BAR_SPACING
    bar_offsets = np.array([-0.24, 0.00, 0.24])

    interval_y_min_values = []
    interval_y_max_values = []

    for i, stage in enumerate(["planting", "growing", "harvesting"]):
        d = all_data[stage]
        color = COLOR_MAP[stage]

        selected_means = [d["mean"][idx] for idx in selected_indices]
        selected_p2_5 = [d["p2_5"][idx] for idx in selected_indices]
        selected_p97_5 = [d["p97_5"][idx] for idx in selected_indices]

        lower_err = []
        upper_err = []

        for mean_v, p2_v, p97_v in zip(selected_means, selected_p2_5, selected_p97_5):
            if np.isfinite(mean_v) and np.isfinite(p2_v) and np.isfinite(p97_v):
                interval_y_min_values.append(p2_v)
                interval_y_max_values.append(p97_v)
                lower_err.append(max(0.0, mean_v - p2_v))
                upper_err.append(max(0.0, p97_v - mean_v))
            else:
                lower_err.append(0.0)
                upper_err.append(0.0)

        bar_x = x_base_selected + bar_offsets[i]
        scatter_center_x = x_base_selected + bar_offsets[i]

        # 左轴：柱体均值
        ax_bar.bar(
            bar_x,
            selected_means,
            width=0.18,
            color=color,
            edgecolor="black",
            linewidth=1,
            hatch=hatch_patterns[i],
            alpha=0.85,
            label=stage,
            zorder=3,
        )

        # 原始数据 95% 分布区间误差线：2.5% ~ 97.5% 分位数
        if SHOW_RAWDATA_95_INTERVAL:
            ax_bar.errorbar(
                bar_x,
                selected_means,
                yerr=[lower_err, upper_err],
                fmt="none",
                ecolor="black",
                elinewidth=ERROR_BAR_LINEWIDTH,
                capsize=ERROR_BAR_CAPSIZE,
                capthick=ERROR_BAR_LINEWIDTH,
                zorder=6,
            )

        # 右轴：原始残差散点
        for j, original_idx in enumerate(selected_indices):
            raw_data = d["data"][original_idx]

            if raw_data is None or len(raw_data) == 0:
                continue

            raw_data = np.asarray(raw_data)
            raw_data = raw_data[~np.isnan(raw_data)]

            if len(raw_data) == 0:
                continue

            if len(raw_data) > max_scatter_points:
                raw_data = rng.choice(raw_data, size=max_scatter_points, replace=False)

            jitter = rng.uniform(-jitter_width, jitter_width, size=len(raw_data))
            scatter_x = np.full(len(raw_data), scatter_center_x[j]) + jitter

            ax_scatter.scatter(
                scatter_x,
                raw_data,
                s=scatter_size,
                color=color,
                alpha=scatter_alpha,
                edgecolors="none",
                zorder=1,
            )

    # -------------------- 坐标轴设置 --------------------
    ax_bar.axhline(0, color="black", lw=1.5, zorder=4)

    ax_bar.set_xticks(x_base_selected)

    if show_bottom_xlabel:
        ax_bar.set_xticklabels(
            selected_display_labels,
            fontsize=GLOBAL_CONFIG["font_size"],
            rotation=35,
            ha="right",
        )
        ax_bar.set_xlabel(
            "LDFAI Threshold Interval",
            fontsize=GLOBAL_CONFIG["font_size"] + 1,
            labelpad=10,
        )
    else:
        ax_bar.set_xticklabels([])
        ax_bar.set_xlabel("")

    if show_left_ylabel:
        ax_bar.set_ylabel(
            "Mean Value",
            fontsize=GLOBAL_CONFIG["font_size"] + 1,
            labelpad=10,
        )
    else:
        # 每个小图都保留 y 轴刻度数字，只取消重复的 y 轴标题
        ax_bar.set_ylabel("")
        ax_bar.tick_params(axis="y", labelleft=True)

    ax_bar.set_title(crop.capitalize(), fontsize=GLOBAL_CONFIG["title_size"], pad=10)

    ax_bar.set_xlim(x_base_selected[0] - 0.6, x_base_selected[-1] + 0.6)

    apply_y_axis_config(ax_bar, ax_scatter, crop, method)

    # 如果原始数据 95% 分布区间超出预设 y 轴范围，自动扩展左侧 y 轴，避免误差线被裁剪
    if SHOW_RAWDATA_95_INTERVAL and AUTO_EXPAND_Y_FOR_INTERVAL and interval_y_min_values and interval_y_max_values:
        current_y_min, current_y_max = ax_bar.get_ylim()
        interval_y_min = min(interval_y_min_values)
        interval_y_max = max(interval_y_max_values)

        new_y_min = min(current_y_min, interval_y_min)
        new_y_max = max(current_y_max, interval_y_max)

        if new_y_min < current_y_min or new_y_max > current_y_max:
            padding = (new_y_max - new_y_min) * 0.08
            if padding == 0:
                padding = 0.01
            ax_bar.set_ylim(new_y_min - padding, new_y_max + padding)

    # 左侧 y 轴刻度值保留两位小数
    ax_bar.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_bar.tick_params(axis="y", labelsize=GLOBAL_CONFIG["tick_size"])
    ax_bar.tick_params(axis="x", labelsize=GLOBAL_CONFIG["font_size"])

    # 取消右侧 y 轴刻度、刻度标签、标题和右侧轴线
    ax_scatter.set_ylabel("")
    ax_scatter.tick_params(axis="y", which="both", right=False, labelright=False)
    ax_scatter.set_yticks([])
    ax_scatter.spines["right"].set_visible(False)

    ax_bar.grid(alpha=0.3, linestyle="--", zorder=0)

    ax_bar.set_zorder(2)
    ax_scatter.set_zorder(1)
    ax_bar.patch.set_visible(False)


# ===================== 合并四个作物为一张图 =====================
def plot_combined_4_crops(method, crop_results_dict, output_dir):
    # 2 × 2 合并图
    fig, axes = plt.subplots(2, 2, figsize=(6, 10), dpi=300)
    axes = axes.flatten()

    subplot_order = ["maize", "rice", "soybeans", "wheat"]

    legend_handles = None
    legend_labels = None

    for idx, crop in enumerate(subplot_order):
        ax = axes[idx]
        results = crop_results_dict.get(crop)

        if not results:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{crop}\nNo data", ha="center", va="center", fontsize=14)
            continue

        selected_data = get_selected_data(results)

        if selected_data is None:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{crop}\nNo valid selected DFAI data", ha="center", va="center", fontsize=14)
            continue

        show_left_ylabel = idx in [0, 2]   # 左列显示 y 轴
        show_bottom_xlabel = idx in [2, 3] # 下排显示 x 轴

        draw_one_crop_subplot(
            ax_bar=ax,
            crop=crop,
            method=method,
            selected_data=selected_data,
            show_left_ylabel=show_left_ylabel,
            show_bottom_xlabel=show_bottom_xlabel,
        )

        # 取一次 legend
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    # 全局图例
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=3,
            fontsize=GLOBAL_CONFIG["legend_size"],
            frameon=True,
            bbox_to_anchor=(0.5, 0.01),
        )

    plt.tight_layout(rect=[0.03, 0.06, 1, 0.98])
    plt.subplots_adjust(wspace=0.35, hspace=0.12)

    os.makedirs(output_dir, exist_ok=True)

    fig_path = os.path.join(
        output_dir,
        f"COMBINED_4crops_bar_scatter_dual_axis_selected_LDFAI_raw95interval_{method}.png",
    )

    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ 已保存四作物合并图: {fig_path}")


# ===================== 主程序 =====================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for method in METHODS:
        print("=" * 90)
        print(f"开始处理方法：{method}")

        crop_results_dict = {}

        for crop in CROPS:
            nc_dir = os.path.join(BASE_ROOT, crop, method)

            if not os.path.isdir(nc_dir):
                print(f"⚠️ 目录不存在，跳过 {crop}:")
                print(nc_dir)
                continue

            data = process_all_nc_files(nc_dir)

            if not data:
                print(f"⚠️ 未读取到有效 nc 数据，跳过：{crop} - {method}")
                continue

            crop_results_dict[crop] = data
            print(f"✅ 已读取: {crop} - {method}")

        if not crop_results_dict:
            print(f"⚠️ 方法 {method} 没有任何可用作物数据，跳过。")
            continue

        plot_combined_4_crops(
            method=method,
            crop_results_dict=crop_results_dict,
            output_dir=OUTPUT_DIR,
        )

        print(f"完成处理方法：{method}")

    print("=" * 90)
    print("✅ 最终只生成两张四作物合并图：")
    print(r"D:\Python\DssAat\NO_1\Figure1b\COMBINED_4crops_bar_scatter_dual_axis_selected_LDFAI_raw95interval_linear_bic.png")
    print(r"D:\Python\DssAat\NO_1\Figure1b\COMBINED_4crops_bar_scatter_dual_axis_selected_LDFAI_raw95interval_quadratic_bic.png")
