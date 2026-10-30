# -*- coding: utf-8 -*-
"""
Create figures for four crops:
1. For each crop, output its own full one-figure summary.
2. Only merge the focused linear_bic Raw Max / Raw Min figure into one compact 2x2 crop summary.

Crops:
    maize, rice, soybeans, wheat
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ============================================================
# 0. User settings
# ============================================================
base_dir = "/mnt/hdb/LXH/data/NZW/LDFAI_step3"
csv_name = "event_ldfai_abs_gt4_spi_spei_compare.csv"

crop_order = ["maize", "rice", "soybeans", "wheat"]

# 如果 soybeans 文件夹实际不是 soybeans_GPCC，而是 soybean_GPCC，在这里改
crop_input_csv = {
    crop: os.path.join(base_dir, f"{crop}_spei", csv_name)
    for crop in crop_order
}

# Full figure: 不合并，四个作物分别输出
full_output_paths = {
    crop: {
        "png": os.path.join(
            base_dir,
            f"{crop}_spei",
            f"{crop}_event_ldfai_abs_gt4_spi_spei_compare_one_figure_en.png"
        ),
        "pdf": os.path.join(
            base_dir,
            f"{crop}_spei",
            f"{crop}_event_ldfai_abs_gt4_spi_spei_compare_one_figure_en.pdf"
        ),
    }
    for crop in crop_order
}

# Focused figure: 只合并 linear_bic + Raw Max/Raw Min
merged_output_dir = os.path.join(base_dir, "all_crops_CSIC")
merged_linear_bic_png = os.path.join(
    merged_output_dir,
    "maize_rice_soybeans_wheat_event_ldfai_abs_gt4_spi_spei_compare_linear_bic_raw_max_min_en.png"
)
merged_linear_bic_pdf = os.path.join(
    merged_output_dir,
    "maize_rice_soybeans_wheat_event_ldfai_abs_gt4_spi_spei_compare_linear_bic_raw_max_min_en.pdf"
)

phenology_order = ["Planting", "Growing", "Harvesting"]

feature_order = [
    "ldfai_raw_max",
    "ldfai_raw_min",
    "ldfai_abs_sum",
    "ldfai_pos_event_max",
    "ldfai_neg_event_max",
]

feature_label_map = {
    "ldfai_raw_max": "Raw Max",
    "ldfai_raw_min": "Raw Min",
    "ldfai_abs_sum": "Abs Sum",
    "ldfai_pos_event_max": "Positive Event Max",
    "ldfai_neg_event_max": "Negative Event Max",
}

phenology_map = {
    "种植期": "Planting",
    "生长期": "Growing",
    "收获期": "Harvesting",
    "Planting": "Planting",
    "Growing": "Growing",
    "Harvesting": "Harvesting",
}

required_cols = [
    "run_name",
    "phenology",
    "feature_name",
    "delta_abs_pearson_SPEI_minus_SPI",
    "delta_abs_spearman_SPEI_minus_SPI",
]

metric_order = ["Pearson", "Spearman"]


# ============================================================
# 1. Helper functions
# ============================================================
def load_crop_data(crop, input_csv):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found for {crop}: {input_csv}")

    df = pd.read_csv(input_csv)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{crop} CSV missing required columns: {missing}")

    df = df[required_cols].copy()
    df["crop"] = crop

    df["phenology_en"] = (
        df["phenology"]
        .map(phenology_map)
        .fillna(df["phenology"].astype(str))
    )

    df["delta_abs_pearson_SPEI_minus_SPI"] = pd.to_numeric(
        df["delta_abs_pearson_SPEI_minus_SPI"],
        errors="coerce"
    )
    df["delta_abs_spearman_SPEI_minus_SPI"] = pd.to_numeric(
        df["delta_abs_spearman_SPEI_minus_SPI"],
        errors="coerce"
    )

    return df


def to_long_format(df):
    pearson_df = df[
        [
            "crop",
            "run_name",
            "phenology_en",
            "feature_name",
            "delta_abs_pearson_SPEI_minus_SPI",
        ]
    ].copy()

    pearson_df = pearson_df.rename(
        columns={"delta_abs_pearson_SPEI_minus_SPI": "value"}
    )
    pearson_df["metric"] = "Pearson"

    spearman_df = df[
        [
            "crop",
            "run_name",
            "phenology_en",
            "feature_name",
            "delta_abs_spearman_SPEI_minus_SPI",
        ]
    ].copy()

    spearman_df = spearman_df.rename(
        columns={"delta_abs_spearman_SPEI_minus_SPI": "value"}
    )
    spearman_df["metric"] = "Spearman"

    plot_df = pd.concat([pearson_df, spearman_df], ignore_index=True)

    run_order = list(dict.fromkeys(plot_df["run_name"].dropna().astype(str).tolist()))

    plot_df["run_order"] = plot_df["run_name"].map(
        {v: i for i, v in enumerate(run_order)}
    )
    plot_df["phenology_order"] = (
        plot_df["phenology_en"]
        .map({v: i for i, v in enumerate(phenology_order)})
        .fillna(999)
    )
    plot_df["metric_order"] = (
        plot_df["metric"]
        .map({v: i for i, v in enumerate(metric_order)})
        .fillna(999)
    )
    plot_df["feature_order"] = (
        plot_df["feature_name"]
        .map({v: i for i, v in enumerate(feature_order)})
        .fillna(999)
    )

    plot_df = plot_df.sort_values(
        [
            "run_order",
            "phenology_order",
            "metric_order",
            "feature_order",
            "feature_name",
        ]
    )

    return plot_df


# ============================================================
# 2. Plot full one-figure summary for each crop separately
# ============================================================
def plot_full_single_crop(crop, plot_df, output_png, output_pdf):
    row_keys = (
        plot_df[
            [
                "run_name",
                "phenology_en",
                "metric",
                "run_order",
                "phenology_order",
                "metric_order",
            ]
        ]
        .drop_duplicates()
        .sort_values(["run_order", "phenology_order", "metric_order"])
        .reset_index(drop=True)
    )

    row_keys["row_label"] = (
        row_keys["run_name"].astype(str)
        + " | "
        + row_keys["phenology_en"].astype(str)
        + " | "
        + row_keys["metric"].astype(str)
    )

    col_keys = [
        f for f in feature_order
        if f in plot_df["feature_name"].dropna().unique().tolist()
    ]

    pivot = plot_df.pivot_table(
        index=["run_name", "phenology_en", "metric"],
        columns="feature_name",
        values="value",
        aggfunc="first"
    )

    pivot = pivot.reindex(
        index=[
            tuple(x)
            for x in row_keys[["run_name", "phenology_en", "metric"]].to_numpy()
        ],
        columns=col_keys
    )

    Z = pivot.to_numpy(dtype=float)
    row_labels = row_keys["row_label"].tolist()
    col_labels = [feature_label_map.get(c, c) for c in col_keys]

    valid_vals = Z[np.isfinite(Z)]
    if valid_vals.size == 0:
        raise ValueError(f"No valid numeric values found for plotting: {crop}")

    max_abs = max(np.nanmax(np.abs(valid_vals)), 1e-6)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    fig_w = max(10, len(col_labels) * 2.0 + 2)
    fig_h = max(8, len(row_labels) * 0.55 + 2.8)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(Z, aspect="auto", norm=norm, cmap="RdBu_r")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right", fontsize=10)

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)

    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.4, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            val = Z[i, j]
            if np.isfinite(val):
                text_color = "white" if abs(val) > max_abs * 0.55 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                    fontweight="bold" if val > 0 else "normal"
                )
            else:
                ax.text(
                    j,
                    i,
                    "NA",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="gray"
                )

    # ax.set_title(
    #     f"{crop}: SPI-LDFAI vs SPEI-LDFAI Correlation Comparison\n"
    #     "Positive values indicate stronger absolute correlation for SPEI-LDFAI",
    #     fontsize=14,
    #     pad=16
    # )

    # ax.set_xlabel("LDFAI Feature", fontsize=12)
    # ax.set_ylabel("Run Name | Phenological Stage | Correlation Metric", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Delta absolute correlation (SPEI - SPI)", fontsize=11)

    pearson_positive = int((plot_df.loc[plot_df["metric"] == "Pearson", "value"] > 0).sum())
    pearson_total = int(plot_df.loc[plot_df["metric"] == "Pearson", "value"].notna().sum())
    spearman_positive = int((plot_df.loc[plot_df["metric"] == "Spearman", "value"] > 0).sum())
    spearman_total = int(plot_df.loc[plot_df["metric"] == "Spearman", "value"].notna().sum())

    note = (
        f"Summary: Pearson SPEI-better cells = {pearson_positive}/{pearson_total}; "
        f"Spearman SPEI-better cells = {spearman_positive}/{spearman_total}.\n"
        "Cell value = |corr(SPEI)| - |corr(SPI)|. Bold numbers indicate SPEI-LDFAI is better."
    )

    # fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=10)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Full figure saved: {output_png}")
    print(f"Full figure saved: {output_pdf}")


# ============================================================
# 3. Plot merged focused figure only:
#    linear_bic + Raw Max / Raw Min across four crops
# ============================================================
def plot_merged_linear_bic(all_plot_df, output_png, output_pdf):
    focused_run_name = "linear_bic"
    focused_features = ["ldfai_raw_max", "ldfai_raw_min"]

    focused_df = all_plot_df[
        (all_plot_df["run_name"].astype(str) == focused_run_name)
        & (all_plot_df["feature_name"].isin(focused_features))
    ].copy()

    if focused_df.empty:
        raise ValueError(
            f"No valid data found for merged focused figure: "
            f"run_name={focused_run_name}, features={focused_features}"
        )

    focused_df["crop_order"] = focused_df["crop"].map(
        {v: i for i, v in enumerate(crop_order)}
    )

    focused_df["phenology_order"] = (
        focused_df["phenology_en"]
        .map({v: i for i, v in enumerate(phenology_order)})
        .fillna(999)
    )

    focused_df["metric_order"] = (
        focused_df["metric"]
        .map({v: i for i, v in enumerate(metric_order)})
        .fillna(999)
    )

    focused_df["feature_order"] = (
        focused_df["feature_name"]
        .map({v: i for i, v in enumerate(focused_features)})
        .fillna(999)
    )

    focused_df = focused_df.sort_values(
        [
            "crop_order",
            "phenology_order",
            "metric_order",
            "feature_order",
        ]
    )

    row_keys = (
        focused_df[
            [
                "phenology_en",
                "metric",
                "phenology_order",
                "metric_order",
            ]
        ]
        .drop_duplicates()
        .sort_values(["phenology_order", "metric_order"])
        .reset_index(drop=True)
    )

    row_keys["row_label"] = (
        row_keys["phenology_en"].astype(str)
        + " | "
        + row_keys["metric"].astype(str)
    )

    col_keys = [
        f for f in focused_features
        if f in focused_df["feature_name"].dropna().unique().tolist()
    ]

    row_index = [
        tuple(x)
        for x in row_keys[["phenology_en", "metric"]].to_numpy()
    ]

    row_labels = row_keys["row_label"].tolist()
    col_labels = [feature_label_map.get(c, c) for c in col_keys]

    Z_by_crop = {}

    for crop in crop_order:
        crop_df = focused_df[focused_df["crop"] == crop].copy()

        pivot = crop_df.pivot_table(
            index=["phenology_en", "metric"],
            columns="feature_name",
            values="value",
            aggfunc="first"
        )

        pivot = pivot.reindex(index=row_index, columns=col_keys)
        Z_by_crop[crop] = pivot.to_numpy(dtype=float)

    all_values = np.concatenate([
        Z[np.isfinite(Z)]
        for Z in Z_by_crop.values()
        if np.isfinite(Z).any()
    ])

    if all_values.size == 0:
        raise ValueError("No valid numeric values found for merged focused plotting.")

    max_abs = max(np.nanmax(np.abs(all_values)), 1e-6)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    n_rows = len(row_labels)
    n_cols = len(col_labels)

    # 更紧凑的画布
    fig = plt.figure(figsize=(5,10))

    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1, 1, 0.08],  # 最后一列专门给 colorbar
        wspace=0.08,
        hspace=0.13
    )

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1], sharex=ax00, sharey=ax00)
    ax10 = fig.add_subplot(gs[1, 0], sharex=ax00, sharey=ax00)
    ax11 = fig.add_subplot(gs[1, 1], sharex=ax00, sharey=ax00)

    axes = [ax00, ax01, ax10, ax11]

    # 单独的 colorbar 轴
    cax = fig.add_subplot(gs[:, 2])

    for idx, crop in enumerate(crop_order):
        ax = axes[idx]
        Z = Z_by_crop[crop]

        im = ax.imshow(
            Z,
            aspect="auto",
            norm=norm,
            cmap="RdBu_r"
        )

        # 作物名保留，但更紧凑
        ax.set_title(
            crop,
            fontsize=12,
            fontweight="bold",
            pad=6
        )

        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(col_labels, rotation=25, ha="right", fontsize=9)

        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(row_labels, fontsize=9)

        # 顶部两幅图不显示 x 轴标签
        if idx < 2:
            ax.tick_params(labelbottom=False)

        # 右侧两幅图不显示 y 轴标签
        if idx % 2 != 0:
            ax.tick_params(labelleft=False)

        # 让刻度标签更靠近图一些
        ax.tick_params(axis="x", pad=2)
        ax.tick_params(axis="y", pad=2)

        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(
            which="minor",
            color="black",
            linestyle="-",
            linewidth=0.30,
            alpha=0.22
        )
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                val = Z[i, j]
                if np.isfinite(val):
                    text_color = "white" if abs(val) > max_abs * 0.55 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.3f}",
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color=text_color,
                        fontweight="bold" if val > 0 else "normal"
                    )
                else:
                    ax.text(
                        j,
                        i,
                        "NA",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="gray"
                    )

    # 去掉总标题、总x标题、总y标题、底部说明文字
    # 不再使用：
    # fig.suptitle(...)
    # fig.supxlabel(...)
    # fig.supylabel(...)
    # fig.text(...)

    # 右侧保留 colorbar，但去掉标题
    cbar = fig.colorbar(im, cax=cax)
    # 不设置 cbar.set_label(...)

    # 更紧凑的边距

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Merged focused figure saved: {output_png}")
    print(f"Merged focused figure saved: {output_pdf}")

# ============================================================
# 4. Main
# ============================================================
all_plot_dfs = []

for crop in crop_order:
    df_crop = load_crop_data(crop, crop_input_csv[crop])
    plot_df_crop = to_long_format(df_crop)

    all_plot_dfs.append(plot_df_crop)

    # 这两个不合并，分别输出，只改命名
    plot_full_single_crop(
        crop=crop,
        plot_df=plot_df_crop,
        output_png=full_output_paths[crop]["png"],
        output_pdf=full_output_paths[crop]["pdf"]
    )

all_plot_df = pd.concat(all_plot_dfs, ignore_index=True)

# 只合并这个 focused linear_bic 图
plot_merged_linear_bic(
    all_plot_df=all_plot_df,
    output_png=merged_linear_bic_png,
    output_pdf=merged_linear_bic_pdf
)