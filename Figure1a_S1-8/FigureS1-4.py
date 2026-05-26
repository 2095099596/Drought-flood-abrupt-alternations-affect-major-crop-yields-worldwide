# -*- coding: utf-8 -*-
"""
Create ONE English summary figure for SPI-LDFAI vs SPEI-LDFAI correlation comparison.

Input CSV example:
    event_ldfai_abs_gt4_spi_spei_compare.csv

Required columns:
    run_name
    phenology
    feature_name
    delta_abs_pearson_SPEI_minus_SPI
    delta_abs_spearman_SPEI_minus_SPI

Output:
    one English summary heatmap figure (PNG and PDF)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ============================================================
# 0. User settings
# ============================================================
input_csv = "/mnt/hdb/LXH/data/NZW/LDFAI_step3/wheat_GPCC/event_ldfai_abs_gt4_spi_spei_compare.csv"
output_png = "/mnt/hdb/LXH/data/NZW/LDFAI_step3/wheat_GPCC/event_ldfai_abs_gt4_spi_spei_compare_one_figure_en.png"
output_pdf = "/mnt/hdb/LXH/data/NZW/LDFAI_step3/wheat_GPCC/event_ldfai_abs_gt4_spi_spei_compare_one_figure_en.pdf"
output_linear_bic_png = "/mnt/hdb/LXH/data/NZW/LDFAI_step3/wheat_GPCC/event_ldfai_abs_gt4_spi_spei_compare_linear_bic_raw_max_min_en.png"
output_linear_bic_pdf = "/mnt/hdb/LXH/data/NZW/LDFAI_step3/wheat_GPCC/event_ldfai_abs_gt4_spi_spei_compare_linear_bic_raw_max_min_en.pdf"
# Preferred orders
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

# ============================================================
# 1. Load and prepare data
# ============================================================
required_cols = [
    "run_name",
    "phenology",
    "feature_name",
    "delta_abs_pearson_SPEI_minus_SPI",
    "delta_abs_spearman_SPEI_minus_SPI",
]

if not os.path.exists(input_csv):
    raise FileNotFoundError(f"Input CSV not found: {input_csv}")

df = pd.read_csv(input_csv)
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df[required_cols].copy()
df["phenology_en"] = df["phenology"].map(phenology_map).fillna(df["phenology"].astype(str))
df["delta_abs_pearson_SPEI_minus_SPI"] = pd.to_numeric(df["delta_abs_pearson_SPEI_minus_SPI"], errors="coerce")
df["delta_abs_spearman_SPEI_minus_SPI"] = pd.to_numeric(df["delta_abs_spearman_SPEI_minus_SPI"], errors="coerce")

# Long format for one combined figure
pearson_df = df[["run_name", "phenology_en", "feature_name", "delta_abs_pearson_SPEI_minus_SPI"]].copy()
pearson_df = pearson_df.rename(columns={"delta_abs_pearson_SPEI_minus_SPI": "value"})
pearson_df["metric"] = "Pearson"

spearman_df = df[["run_name", "phenology_en", "feature_name", "delta_abs_spearman_SPEI_minus_SPI"]].copy()
spearman_df = spearman_df.rename(columns={"delta_abs_spearman_SPEI_minus_SPI": "value"})
spearman_df["metric"] = "Spearman"

plot_df = pd.concat([pearson_df, spearman_df], ignore_index=True)

# Orders
run_order = list(dict.fromkeys(plot_df["run_name"].dropna().astype(str).tolist()))
metric_order = ["Pearson", "Spearman"]

plot_df["run_order"] = plot_df["run_name"].map({v: i for i, v in enumerate(run_order)})
plot_df["phenology_order"] = plot_df["phenology_en"].map({v: i for i, v in enumerate(phenology_order)}).fillna(999)
plot_df["metric_order"] = plot_df["metric"].map({v: i for i, v in enumerate(metric_order)}).fillna(999)
plot_df["feature_order"] = plot_df["feature_name"].map({v: i for i, v in enumerate(feature_order)}).fillna(999)

plot_df = plot_df.sort_values(["run_order", "phenology_order", "metric_order", "feature_order", "feature_name"])

row_keys = (
    plot_df[["run_name", "phenology_en", "metric", "run_order", "phenology_order", "metric_order"]]
    .drop_duplicates()
    .sort_values(["run_order", "phenology_order", "metric_order"])
    .reset_index(drop=True)
)
row_keys["row_label"] = row_keys["run_name"].astype(str) + " | " + row_keys["phenology_en"].astype(str) + " | " + row_keys["metric"].astype(str)

col_keys = [f for f in feature_order if f in plot_df["feature_name"].unique().tolist()]

pivot = plot_df.pivot_table(
    index=["run_name", "phenology_en", "metric"],
    columns="feature_name",
    values="value",
    aggfunc="first"
)

pivot = pivot.reindex(index=[tuple(x) for x in row_keys[["run_name", "phenology_en", "metric"]].to_numpy()], columns=col_keys)

Z = pivot.to_numpy(dtype=float)
row_labels = row_keys["row_label"].tolist()
col_labels = [feature_label_map.get(c, c) for c in col_keys]

# ============================================================
# 2. Plot one summary figure
# ============================================================
valid_vals = Z[np.isfinite(Z)]
if valid_vals.size == 0:
    raise ValueError("No valid numeric values found for plotting.")

max_abs = max(np.nanmax(np.abs(valid_vals)), 1e-6)
norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

# Figure size adapts to table size
fig_w = max(10, len(col_labels) * 2.0 + 2)
fig_h = max(8, len(row_labels) * 0.55 + 2.8)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

im = ax.imshow(Z, aspect="auto", norm=norm, cmap="RdBu_r")

ax.set_xticks(np.arange(len(col_labels)))
ax.set_xticklabels(col_labels, rotation=25, ha="right", fontsize=10)
ax.set_yticks(np.arange(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=10)

# Grid lines
ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
ax.grid(which="minor", color="black", linestyle="-", linewidth=0.4, alpha=0.35)
ax.tick_params(which="minor", bottom=False, left=False)

# Annotate each cell
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        val = Z[i, j]
        if np.isfinite(val):
            text_color = "white" if abs(val) > max_abs * 0.55 else "black"
            ax.text(
                j, i, f"{val:.3f}",
                ha="center", va="center",
                fontsize=9,
                color=text_color,
                fontweight="bold" if val > 0 else "normal"
            )
        else:
            ax.text(j, i, "NA", ha="center", va="center", fontsize=8, color="gray")

# Titles and labels
ax.set_title(
    "One-Figure Summary of SPI-LDFAI vs SPEI-LDFAI Correlation Comparison\n"
    "Positive values indicate stronger absolute correlation for SPEI-LDFAI",
    fontsize=14,
    pad=16
)
ax.set_xlabel("LDFAI Feature", fontsize=12)
ax.set_ylabel("Run Name | Phenological Stage | Correlation Metric", fontsize=12)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Delta absolute correlation (SPEI - SPI)", fontsize=11)

# Note box
pearson_positive = int((plot_df.loc[plot_df["metric"] == "Pearson", "value"] > 0).sum())
pearson_total = int(plot_df.loc[plot_df["metric"] == "Pearson", "value"].notna().sum())
spearman_positive = int((plot_df.loc[plot_df["metric"] == "Spearman", "value"] > 0).sum())
spearman_total = int(plot_df.loc[plot_df["metric"] == "Spearman", "value"].notna().sum())

note = (
    f"Summary: Pearson SPEI-better cells = {pearson_positive}/{pearson_total}; "
    f"Spearman SPEI-better cells = {spearman_positive}/{spearman_total}.\n"
    "Cell value = |corr(SPEI)| - |corr(SPI)|. Bold numbers indicate SPEI-LDFAI is better."
)
fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=10)

plt.tight_layout(rect=[0, 0.04, 1, 1])

os.makedirs(os.path.dirname(output_png), exist_ok=True)
fig.savefig(output_png, dpi=300, bbox_inches="tight")
fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
plt.close(fig)

# ============================================================
# 3. Plot extra focused figure: linear_bic only, Raw Max/Raw Min only
# ============================================================
focused_run_name = "linear_bic"
focused_features = ["ldfai_raw_max", "ldfai_raw_min"]

focused_df = plot_df[
    (plot_df["run_name"].astype(str) == focused_run_name)
    & (plot_df["feature_name"].isin(focused_features))
].copy()

if focused_df.empty:
    raise ValueError(
        f"No valid data found for focused figure: "
        f"run_name={focused_run_name}, features={focused_features}"
    )

focused_col_keys = [f for f in focused_features if f in focused_df["feature_name"].unique().tolist()]

focused_row_keys = (
    focused_df[["run_name", "phenology_en", "metric", "run_order", "phenology_order", "metric_order"]]
    .drop_duplicates()
    .sort_values(["run_order", "phenology_order", "metric_order"])
    .reset_index(drop=True)
)

# Since this figure only keeps linear_bic, row labels can be shorter
focused_row_keys["row_label"] = (
    focused_row_keys["phenology_en"].astype(str)
    + " | "
    + focused_row_keys["metric"].astype(str)
)

focused_pivot = focused_df.pivot_table(
    index=["run_name", "phenology_en", "metric"],
    columns="feature_name",
    values="value",
    aggfunc="first"
)

focused_pivot = focused_pivot.reindex(
    index=[tuple(x) for x in focused_row_keys[["run_name", "phenology_en", "metric"]].to_numpy()],
    columns=focused_col_keys
)

Z2 = focused_pivot.to_numpy(dtype=float)
row_labels2 = focused_row_keys["row_label"].tolist()
col_labels2 = [feature_label_map.get(c, c) for c in focused_col_keys]

valid_vals2 = Z2[np.isfinite(Z2)]
if valid_vals2.size == 0:
    raise ValueError("No valid numeric values found for focused plotting.")

max_abs2 = max(np.nanmax(np.abs(valid_vals2)), 1e-6)
norm2 = TwoSlopeNorm(vmin=-max_abs2, vcenter=0.0, vmax=max_abs2)

fig_w2 = max(7, len(col_labels2) * 2.4 + 2)
fig_h2 = max(5, len(row_labels2) * 0.6 + 2.8)

fig2, ax2 = plt.subplots(figsize=(fig_w2, fig_h2))

im2 = ax2.imshow(Z2, aspect="auto", norm=norm2, cmap="RdBu_r")

ax2.set_xticks(np.arange(len(col_labels2)))
ax2.set_xticklabels(col_labels2, rotation=25, ha="right", fontsize=10)

ax2.set_yticks(np.arange(len(row_labels2)))
ax2.set_yticklabels(row_labels2, fontsize=10)

# Grid lines
ax2.set_xticks(np.arange(-0.5, len(col_labels2), 1), minor=True)
ax2.set_yticks(np.arange(-0.5, len(row_labels2), 1), minor=True)
ax2.grid(which="minor", color="black", linestyle="-", linewidth=0.4, alpha=0.35)
ax2.tick_params(which="minor", bottom=False, left=False)

# Annotate each cell
for i in range(Z2.shape[0]):
    for j in range(Z2.shape[1]):
        val = Z2[i, j]
        if np.isfinite(val):
            text_color = "white" if abs(val) > max_abs2 * 0.55 else "black"
            ax2.text(
                j, i, f"{val:.3f}",
                ha="center", va="center",
                fontsize=10,
                color=text_color,
                fontweight="bold" if val > 0 else "normal"
            )
        else:
            ax2.text(
                j, i, "NA",
                ha="center", va="center",
                fontsize=9,
                color="gray"
            )

ax2.set_title(
    "Focused Summary: linear_bic Only\n"
    "Raw Max and Raw Min Features",
    fontsize=14,
    pad=16
)

ax2.set_xlabel("LDFAI Feature", fontsize=12)
ax2.set_ylabel("Phenological Stage | Correlation Metric", fontsize=12)

cbar2 = fig2.colorbar(im2, ax=ax2, fraction=0.05, pad=0.03)
cbar2.set_label("Delta absolute correlation (SPEI - SPI)", fontsize=11)

focused_pearson_positive = int(
    (focused_df.loc[focused_df["metric"] == "Pearson", "value"] > 0).sum()
)
focused_pearson_total = int(
    focused_df.loc[focused_df["metric"] == "Pearson", "value"].notna().sum()
)
focused_spearman_positive = int(
    (focused_df.loc[focused_df["metric"] == "Spearman", "value"] > 0).sum()
)
focused_spearman_total = int(
    focused_df.loc[focused_df["metric"] == "Spearman", "value"].notna().sum()
)

note2 = (
    f"Run name: {focused_run_name}. "
    f"Pearson SPEI-better cells = {focused_pearson_positive}/{focused_pearson_total}; "
    f"Spearman SPEI-better cells = {focused_spearman_positive}/{focused_spearman_total}.\n"
    "Cell value = |corr(SPEI)| - |corr(SPI)|. Bold numbers indicate SPEI-LDFAI is better."
)

fig2.text(0.01, 0.01, note2, ha="left", va="bottom", fontsize=10)

plt.tight_layout(rect=[0, 0.06, 1, 1])

fig2.savefig(output_linear_bic_png, dpi=300, bbox_inches="tight")
fig2.savefig(output_linear_bic_pdf, dpi=300, bbox_inches="tight")
plt.close(fig2)

print(f"Focused figure saved: {output_linear_bic_png}")
print(f"Focused figure saved: {output_linear_bic_pdf}")

print(f"Figure saved: {output_png}")
print(f"Figure saved: {output_pdf}")
