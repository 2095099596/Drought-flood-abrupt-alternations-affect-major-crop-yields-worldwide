import os
import re
import warnings

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ===================== Nature 风格全局样式 =====================
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["font.family"] = ["Arial", "Times New Roman", "DejaVu Sans"]
plt.rcParams["figure.max_open_warning"] = 50
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8
plt.rcParams["xtick.major.size"] = 3.5
plt.rcParams["ytick.major.size"] = 3.5
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

# ===================== 配置区 =====================
# 四个作物文件夹都位于这个根目录下：
# C:\Users\lxh\Desktop\NongZuoWu_ChanLiang\New_result\Data\LDFAI_SPEI_SPEI_3_2\maize
# C:\Users\lxh\Desktop\NongZuoWu_ChanLiang\New_result\Data\LDFAI_SPEI_SPEI_3_2\rice
# C:\Users\lxh\Desktop\NongZuoWu_ChanLiang\New_result\Data\LDFAI_SPEI_SPEI_3_2\soybeans
# C:\Users\lxh\Desktop\NongZuoWu_ChanLiang\New_result\Data\LDFAI_SPEI_SPEI_3_2\wheat
BASE_ROOT = r"C:\Users\lxh\Desktop\NongZuoWu_ChanLiang\New_result\Data\LDFAI_SPEI_CSIC_3_2"
OUTPUT_DIR = r"D:\Python\DssAat\NO_1\Figure1CC\CSIC"

CROPS = [
    "maize",
    "rice",
    "soybeans",
    "wheat",
]

CROP_TITLES = {
    "maize": "Maize",
    "rice": "Rice",
    "soybeans": "Soybeans",
    "wheat": "Wheat",
}

# 只生成这两个方法对应的四作物合并图
METHODS = [
    "linear_bic",
    "quadratic_bic",
]

GLOBAL_CONFIG = {
    # 1×4 横向排布，更紧凑
    "fig_size": (13.2, 3.2),
    "font_size": 9,
    "tick_size": 8,
    "legend_size": 8.5,
    "title_size": 9.5,

    # 折线变细
    "line_width": 1.15,
}

# Nature 风格：低饱和、偏浅的三阶段颜色
COLOR_MAP = {
    "planting": "#E8B06A",
    "growing": "#7FBC8C",
    "harvesting": "#7AA6C2",
}

STAGE_LABELS = {
    "planting": "Planting",
    "growing": "Growing",
    "harvesting": "Harvesting",
}

# 每个方法、每个作物可单独设置 y 轴范围。
# 如果设为 None，则该子图自动根据数据范围确定 y 轴。
Y_AXIS_CONFIG = {
    "linear_bic": {
        "maize": {"y_min": None, "y_max": None},
        "rice": {"y_min": None, "y_max": None},
        "soybeans": {"y_min": None, "y_max": None},
        "wheat": {"y_min": None, "y_max": None},
    },
    "quadratic_bic": {
        "maize": {"y_min": None, "y_max": None},
        "rice": {"y_min": None, "y_max": None},
        "soybeans": {"y_min": None, "y_max": None},
        "wheat": {"y_min": None, "y_max": None},
    },
}

# x 轴只显示这些关键阈值标签，但折线保留全部数据点
TARGET_LDFAI_LABELS = [
    "DFAI>9.00",
    "DFAI>4.00",
    "DFAI>1.00",
    "DFAI<-1.00",
    "DFAI<-4.00",
    "DFAI<-9.00",
]

TARGET_DISPLAY_LABELS = [
    "DFAI>9",
    "DFAI>4",
    "DFAI>1",
    "DFAI<-1",
    "DFAI<-4",
    "DFAI<-9",
]


# ===================== 生成 LDFAI 标签 =====================
def generate_ldfai_labels():
    """
    生成与原脚本一致的 LDFAI 标签：
    左侧：LDFAI>9.00 到 LDFAI>1.00
    右侧：LDFAI<-1.00 到 LDFAI<-9.00
    """
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


# ===================== 读取 nc 文件并计算均值 =====================
def calculate_nc_stats(file_path):
    try:
        stats = {}

        with xr.open_dataset(file_path, decode_cf=True) as ds:
            for var in ds.data_vars:
                data = ds[var].values
                mean_value = np.nanmean(data)
                valid_count = np.sum(~np.isnan(data))

                stats[var] = {
                    "mean": mean_value,
                    "valid_count": valid_count,
                }

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


# ===================== 整理折线图数据 =====================
def collect_stage_mean_data(results):
    all_data = {}

    for stage in ["planting", "growing", "harvesting"]:
        mean_list = []
        count_list = []
        file_list = []
        var_list = []

        for fname, vdict in results.items():
            if stage in fname:
                for var_name, d in vdict.items():
                    mean_list.append(d["mean"])
                    count_list.append(d["valid_count"])
                    file_list.append(fname)
                    var_list.append(var_name)

        all_data[stage] = {
            "mean": mean_list,
            "count": count_list,
            "file": file_list,
            "var": var_list,
            "n": len(mean_list),
        }

    return all_data


# ===================== 获取关键阈值在完整序列中的索引 =====================
def get_target_tick_positions(x_tick_labels, n_common):
    label_to_index = {label: idx for idx, label in enumerate(x_tick_labels[:n_common])}

    tick_positions = []
    tick_labels = []

    for raw_label, display_label in zip(TARGET_LDFAI_LABELS, TARGET_DISPLAY_LABELS):
        if raw_label in label_to_index:
            tick_positions.append(label_to_index[raw_label])
            tick_labels.append(display_label)
        else:
            print(f"⚠️ 未找到目标 LDFAI 标签：{raw_label}")

    return tick_positions, tick_labels


# ===================== y 轴自动留白 =====================
def apply_auto_ylim(ax, y_values):
    y_values = np.asarray(y_values, dtype=float)
    y_values = y_values[np.isfinite(y_values)]

    if len(y_values) == 0:
        return

    y_min = np.nanmin(y_values)
    y_max = np.nanmax(y_values)

    if np.isclose(y_min, y_max):
        pad = max(abs(y_min) * 0.2, 0.01)
    else:
        pad = (y_max - y_min) * 0.15

    # 尽量把 0 线纳入视野
    lower = min(y_min - pad, 0)
    upper = max(y_max + pad, 0)
    ax.set_ylim(lower, upper)


# ===================== 单个子图绘制 =====================
def plot_crop_on_axis(ax, results, method, crop, col):
    all_data = collect_stage_mean_data(results)
    n_common = min(d["n"] for d in all_data.values())

    if n_common == 0:
        ax.text(
            0.5,
            0.5,
            "No data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=GLOBAL_CONFIG["font_size"],
        )
        return None, None

    x_tick_labels = LDFAI_TICK_LABELS[:n_common]
    x = np.arange(n_common)
    tick_positions, tick_labels = get_target_tick_positions(x_tick_labels, n_common)

    all_y_values = []

    for stage in ["planting", "growing", "harvesting"]:
        y = np.asarray(all_data[stage]["mean"][:n_common], dtype=float)
        all_y_values.extend(y[np.isfinite(y)])

        # 保留全部 x 数据点，只是不显示 marker
        ax.plot(
            x,
            y,
            linewidth=GLOBAL_CONFIG["line_width"],
            color=COLOR_MAP[stage],
            label=STAGE_LABELS[stage],
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
        )

    # 只在 y=0 添加虚线
    ax.axhline(
        0,
        color="0.45",
        lw=0.9,
        linestyle=(0, (4, 3)),
        zorder=1,
    )

    # x 轴：完整范围，但只显示关键刻度标签
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        tick_labels,
        fontsize=GLOBAL_CONFIG["tick_size"],
        rotation=35,
        ha="right",
    )

    ax.set_xlim(x[0], x[-1])

    # 每个子图独立 y 轴范围。
    # 如果配置了固定范围，使用固定范围；否则自动范围。
    y_cfg = Y_AXIS_CONFIG.get(method, {}).get(crop, {})
    y_min = y_cfg.get("y_min")
    y_max = y_cfg.get("y_max")

    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
        print(f"✅ y 轴范围已设置 | 方法: {method} | 作物: {crop} | [{y_min}, {y_max}]")
    else:
        apply_auto_ylim(ax, all_y_values)

    # Nature 风格：去掉上、右边框，保留左、下边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=GLOBAL_CONFIG["tick_size"],
        width=0.8,
        length=3.5,
        direction="out",
    )

    ax.grid(False)
    # 1×4 布局下不在每个子图重复显示坐标轴名称；
    # y 轴刻度值因各子图范围可能不同，默认保留。
    ax.set_xlabel("")
    ax.set_ylabel("")

    # 每个子图都显示自己的 y 轴真实刻度数值。
    ax.tick_params(axis="y", labelleft=True)

    return ax.get_legend_handles_labels()


# ===================== 绘制四作物合并图 =====================
def plot_four_crops_combined_figure(method):
    fig, axes = plt.subplots(
        1,
        4,
        figsize=GLOBAL_CONFIG["fig_size"],
        dpi=300,
        sharex=False,
        sharey=False,
    )

    axes = np.atleast_1d(axes).flatten()
    legend_handles = None
    legend_labels = None

    for idx, crop in enumerate(CROPS):
        ax = axes[idx]
        col = idx

        nc_dir = os.path.join(BASE_ROOT, crop, method)

        if not os.path.isdir(nc_dir):
            print("目录错误，请检查 nc_dir 路径：")
            print(nc_dir)
            ax.text(
                0.5,
                0.5,
                "Missing directory",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=GLOBAL_CONFIG["font_size"],
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            continue

        data = process_all_nc_files(nc_dir)

        if not data:
            print(f"未读取到有效 nc 数据，跳过：{crop} | {method}")
            ax.text(
                0.5,
                0.5,
                "No valid data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=GLOBAL_CONFIG["font_size"],
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            continue

        result = plot_crop_on_axis(
            ax=ax,
            results=data,
            method=method,
            crop=crop,
            col=col,
        )

        if result is not None:
            handles, labels = result
            if handles and legend_handles is None:
                legend_handles = handles
                legend_labels = labels

    # 全图只保留一个图例，避免四个子图重复显示。
    # 图例放在子图上方，略微靠近图，但通过 top 留白避免重叠。
    if legend_handles is not None and legend_labels is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=3,
            fontsize=GLOBAL_CONFIG["legend_size"],
            frameon=False,
            handlelength=2.3,
            borderaxespad=0.0,
        )

    # 取消图标题、x 轴标题、y 轴标题，只保留坐标刻度。
    # 1×4 子图间距收紧，同时保留图例和 x 轴刻度标签空间。
    plt.tight_layout(rect=[0.03, 0.06, 1, 0.86])
    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        bottom=0.20,
        top=0.80,
        wspace=0.12,
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig_path = os.path.join(
        OUTPUT_DIR,
        f"COMBINED_four_crops_nature_line_{method}.png",
    )

    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ 四作物合并 Nature 风格折线图已保存: {fig_path}")


# ===================== 主程序 =====================
if __name__ == "__main__":
    for method in METHODS:
        print("=" * 80)
        print(f"开始处理方法：{method}")
        plot_four_crops_combined_figure(method)
        print(f"完成处理方法：{method}")

    print("=" * 80)
    print("✅ linear_bic 和 quadratic_bic 的四作物合并图全部处理完成！")
