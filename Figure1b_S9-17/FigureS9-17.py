import os
import re
import warnings

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ===================== 全局样式 =====================
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Sans"]
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.max_open_warning"] = 20


# ===================== 路径设置 =====================
BASE_DIR = r"C:\Users\lxh\Desktop\NongZuoWu_ChanLiang\New_result\Data\LDFAI_SPEI_ERA5_3_2\maize"
OUTPUT_DIR = r"D:\Python\DssAat\NO_1\Figure1"

Data_From = 'ERA5'
# ===================== 只保留这三个方法，对应输出三张图 =====================
METHODS = [
    "linear_bic",
    "multimodel_aic",
    "quadratic_bic",
]


# ===================== 图形参数 =====================
GLOBAL_CONFIG = {
    "fig_size": (18, 8),
    "font_size": 13,
    "tick_size": 16,
    "legend_size": 18,
}

BAR_CONFIG = {
    "bar_width": 0.18,
    "bar_spacing": 1.0,
}

COLOR_MAP = {
    "planting": "#ff7f0e",
    "growing": "#2ca02c",
    "harvesting": "#1f77b4",
}

HATCH_PATTERNS = ['//', '..', '++']


# ===================== y 轴范围：只保留完整图 =====================
a = -0.2
b = 0.2

Y_AXIS_CONFIG = {
    "multimodel_aic": {
        "bar_y_min": a,
        "bar_y_max": b,
        "scatter_y_min": a,
        "scatter_y_max": b,
    },
    "linear_bic": {
        "bar_y_min": a,
        "bar_y_max": b,
        "scatter_y_min": a,
        "scatter_y_max": b,
    },
    "quadratic_bic": {
        "bar_y_min": a,
        "bar_y_max": b,
        "scatter_y_min": a,
        "scatter_y_max": b,
    },
}


def apply_y_axis_config(ax_bar, ax_scatter, method):
    """设置左右 y 轴范围。"""
    if method not in Y_AXIS_CONFIG:
        raise ValueError(f"Y_AXIS_CONFIG 中没有 method: {method}")

    cfg = Y_AXIS_CONFIG[method]

    ax_bar.set_ylim(cfg["bar_y_min"], cfg["bar_y_max"])
    ax_scatter.set_ylim(cfg["scatter_y_min"], cfg["scatter_y_max"])

    print(
        f"✅ y 轴范围已设置 | 方法: {method} | "
        f"左轴 Mean: [{cfg['bar_y_min']}, {cfg['bar_y_max']}] | "
        f"右轴 Residual: [{cfg['scatter_y_min']}, {cfg['scatter_y_max']}]"
    )


# ===================== 生成 LDFAI 标签 =====================
def generate_ldfai_labels():
    positive = np.arange(3.0, 0.9, -0.2)
    negative = np.arange(-1.0, -3.1, -0.2)

    all_values = np.concatenate([positive, negative])

    tick_labels = []

    for v in all_values:
        sq = v ** 2

        if v > 0:
            tick_labels.append(f"DFAI>{sq:.2f}")
        else:
            tick_labels.append(f"DFAI<-{sq:.2f}")

    return tick_labels


LDFAI_TICK_LABELS = generate_ldfai_labels()


# ===================== 读取 nc 文件并计算均值、有效样本数、95% CI =====================
def calculate_nc_stats(file_path):
    """
    对每个变量计算：
    1. mean：均值
    2. valid_count：有效样本数
    3. valid_data：原始有效值，用于散点
    4. ci95：均值的 95% 置信区间半宽

    95% CI 半宽：
        ci95 = 1.96 * std / sqrt(n)

    其中 std 使用样本标准差 ddof=1。
    """
    try:
        ds = xr.open_dataset(file_path, decode_cf=True)

        stats = {}

        for var in ds.data_vars:
            data = ds[var].values

            valid_mask = ~np.isnan(data)
            valid_data = np.asarray(data[valid_mask], dtype=float)

            valid_count = len(valid_data)

            if valid_count == 0:
                mean_value = np.nan
                ci95 = np.nan
            elif valid_count == 1:
                mean_value = float(valid_data[0])
                ci95 = 0.0
            else:
                mean_value = float(np.mean(valid_data))
                std_value = float(np.std(valid_data, ddof=1))
                ci95 = 1.96 * std_value / np.sqrt(valid_count)

            stats[var] = {
                "mean": mean_value,
                "valid_count": valid_count,
                "valid_data": valid_data,
                "ci95": ci95,
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


# ===================== 整理绘图数据 =====================
def collect_plot_data(results):
    all_data = {}

    for stage in ["planting", "growing", "harvesting"]:
        mean_list = []
        ci95_list = []
        count_list = []
        data_list = []

        for fname, vdict in results.items():
            if stage in fname:
                for _, d in vdict.items():
                    mean_list.append(d["mean"])
                    ci95_list.append(d["ci95"])
                    count_list.append(d["valid_count"])
                    data_list.append(d["valid_data"])

        all_data[stage] = {
            "mean": mean_list,
            "ci95": ci95_list,
            "count": count_list,
            "data": data_list,
            "n": len(mean_list),
        }

    return all_data


# ===================== 绘制组合图：柱体 + 95% CI + 散点 =====================
def plot_combined_bar_scatter_with_ci(results, method, output_dir,Data_From):
    all_data = collect_plot_data(results)

    N_common = min([d["n"] for d in all_data.values()])

    if N_common == 0:
        print("没有可用于绘图的数据，请检查 nc 文件命名或变量内容。")
        return

    x_tick_labels = LDFAI_TICK_LABELS[:N_common]

    spacing = BAR_CONFIG["bar_spacing"]
    x_base = np.arange(N_common) * spacing

    fig, ax_bar = plt.subplots(figsize=GLOBAL_CONFIG["fig_size"], dpi=300)
    ax_scatter = ax_bar.twinx()

    # 固定随机种子，保证每次散点位置一致
    rng = np.random.default_rng(2024)

    max_scatter_points = 1000
    jitter_width = 0.025
    scatter_alpha = 0.07
    scatter_size = 6

    bar_offsets = np.array([-0.24, 0.00, 0.24])

    for i, stage in enumerate(["planting", "growing", "harvesting"]):
        d = all_data[stage]
        color = COLOR_MAP[stage]

        means = np.asarray(d["mean"][:N_common], dtype=float)
        ci95 = np.asarray(d["ci95"][:N_common], dtype=float)

        bar_x = x_base + bar_offsets[i]
        scatter_center_x = x_base + bar_offsets[i]

        # -------------------------
        # 左轴：柱体均值 + 95% CI
        # -------------------------
        ax_bar.bar(
            bar_x,
            means,
            yerr=ci95,
            width=BAR_CONFIG["bar_width"],
            color=color,
            edgecolor="black",
            linewidth=1,
            hatch=HATCH_PATTERNS[i],
            alpha=0.85,
            label=stage,
            zorder=3,
            capsize=3,
            error_kw={
                "elinewidth": 1.2,
                "ecolor": "black",
                "capthick": 1.2,
                "zorder": 5,
            },
        )

        # -------------------------
        # 右轴：原始数据散点
        # -------------------------
        for j in range(N_common):
            raw_data = d["data"][j]

            if raw_data is None or len(raw_data) == 0:
                continue

            raw_data = np.asarray(raw_data, dtype=float)
            raw_data = raw_data[~np.isnan(raw_data)]

            if len(raw_data) == 0:
                continue

            if len(raw_data) > max_scatter_points:
                raw_data = rng.choice(
                    raw_data,
                    size=max_scatter_points,
                    replace=False,
                )

            jitter = rng.uniform(
                -jitter_width,
                jitter_width,
                size=len(raw_data),
            )

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

    # ===================== 坐标轴设置 =====================
    ax_bar.axhline(0, color="black", lw=1.5, zorder=4)

    ax_bar.set_xticks(x_base)
    ax_bar.set_xticklabels(
        x_tick_labels,
        fontsize=GLOBAL_CONFIG["font_size"],
        rotation=45,
        ha="right",
    )

    ax_bar.set_xlabel(
        "  ",
        fontsize=GLOBAL_CONFIG["font_size"] + 2,
        labelpad=12,
    )

    ax_bar.set_ylabel(
        " ",
        fontsize=GLOBAL_CONFIG["font_size"] + 2,
        labelpad=12,
    )

    # 右侧 y 轴只用于绘制散点，不显示刻度、刻度标签和轴标签
    ax_scatter.set_ylabel("")

    ax_bar.set_xlim(x_base[0] - 0.5, x_base[-1] + 0.55)

    apply_y_axis_config(
        ax_bar=ax_bar,
        ax_scatter=ax_scatter,
        method=method,
    )

    # 只保留左侧 y 轴刻度
    ax_bar.tick_params(axis="y", labelsize=GLOBAL_CONFIG["tick_size"])
    ax_scatter.tick_params(
        axis="y",
        which="both",
        right=False,
        labelright=False,
    )
    ax_scatter.spines["right"].set_visible(False)

    ax_bar.grid(alpha=0.3, linestyle="--", zorder=0)

    # 让柱体轴显示在上层，避免被散点轴遮挡
    ax_bar.set_zorder(2)
    ax_scatter.set_zorder(1)
    ax_bar.patch.set_visible(False)

    ax_bar.legend(
        loc="lower center",
        ncol=3,
        fontsize=GLOBAL_CONFIG["legend_size"],
        frameon=True,
    )

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)

    fig_path = os.path.join(
        output_dir,
        f"COMBINED_bar_scatter_dual_axis_{Data_From}_{method}.png",
    )

    plt.savefig(
        fig_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"✅ 已保存带 95% CI 的双 y 轴柱状 + 散点图: {fig_path}")


# ===================== 主程序 =====================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for method in METHODS:
        print("=" * 80)
        print(f"开始处理方法：{method}")

        nc_dir = os.path.join(BASE_DIR, method)

        if not os.path.isdir(nc_dir):
            print("目录错误，请检查 nc_dir 路径：")
            print(nc_dir)
            continue

        data = process_all_nc_files(nc_dir)

        if not data:
            print(f"未读取到有效 nc 数据，跳过：{method}")
            continue

        plot_combined_bar_scatter_with_ci(
            results=data,
            method=method,
            output_dir=OUTPUT_DIR,
            Data_From=Data_From,
        )

        print(f"完成处理方法：{method}")

    print("=" * 80)
    print("✅ 三张图全部处理完成！")
    print("输出文件：")
    print(os.path.join(OUTPUT_DIR, "COMBINED_bar_scatter_dual_axis_ERA5_linear_bic.png"))
    print(os.path.join(OUTPUT_DIR, "COMBINED_bar_scatter_dual_axis_ERA5_multimodel_aic.png"))
    print(os.path.join(OUTPUT_DIR, "COMBINED_bar_scatter_dual_axis_ERA5_quadratic_bic.png"))
