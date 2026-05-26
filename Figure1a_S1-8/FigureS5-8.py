# -*- coding: utf-8 -*-
"""
Use ONE figure to summarize selected Step-4 univariate model results for:
    scenario = event_ldfai_abs_gt4
    run_name = linear_bic
    features = ldfai_raw_max, ldfai_raw_min

Displayed metrics:
    delta_r2_SPEI_minus_SPI
    delta_adj_r2_SPEI_minus_SPI
    delta_rmse_SPEI_minus_SPI
    delta_mae_SPEI_minus_SPI

Input:
    step4_event_thresholds_spi_spei_model_compare.csv

Output:
    A single heatmap figure showing raw delta values.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ============================================================
# 0. Parameters
# ============================================================
input_csv = "/mnt/hdb/LXH/data/NZW/LDFAI_step3/wheat_spei/step4_regression_by_threshold/step4_event_thresholds_spi_spei_model_compare.csv"
output_dir = "/mnt/hdb/LXH/data/NZW/LDFAI_step3/wheat_spei/step4_regression_by_threshold/figures_univariate_event_gt4"
os.makedirs(output_dir, exist_ok=True)

output_png = os.path.join(
    output_dir,
    "univariate_event_gt4_linear_bic_rawmax_rawmin_four_delta_metrics.png"
)
output_pdf = os.path.join(
    output_dir,
    "univariate_event_gt4_linear_bic_rawmax_rawmin_four_delta_metrics.pdf"
)
output_csv = os.path.join(
    output_dir,
    "univariate_event_gt4_linear_bic_rawmax_rawmin_four_delta_metrics_table.csv"
)

selected_model_type = "univariate_ols"
selected_scenario = "event_ldfai_abs_gt4"
selected_event_threshold = 4
selected_run_name = "linear_bic"
selected_features = ["ldfai_raw_max", "ldfai_raw_min"]

metric_cols = [
    "delta_r2_SPEI_minus_SPI",
    "delta_adj_r2_SPEI_minus_SPI",
    "delta_rmse_SPEI_minus_SPI",
    "delta_mae_SPEI_minus_SPI",
]

metric_label_map = {
    "delta_r2_SPEI_minus_SPI": "Delta R²",
    "delta_adj_r2_SPEI_minus_SPI": "Delta Adj. R²",
    "delta_rmse_SPEI_minus_SPI": "Delta RMSE",
    "delta_mae_SPEI_minus_SPI": "Delta MAE",
}

phenology_map = {
    "种植期": "Planting",
    "生长期": "Growing",
    "收获期": "Harvesting",
    "Planting": "Planting",
    "Growing": "Growing",
    "Harvesting": "Harvesting",
}
phenology_order = ["Planting", "Growing", "Harvesting"]

feature_order = [
    "ldfai_raw_max",
    "ldfai_raw_min",
]

feature_label_map = {
    "ldfai_raw_max": "Raw Max",
    "ldfai_raw_min": "Raw Min",
}

# ============================================================
# 1. Load and filter data
# ============================================================
if not os.path.exists(input_csv):
    raise FileNotFoundError(f"Input CSV not found: {input_csv}")

df = pd.read_csv(input_csv)

required_base_cols = ["model_type", "run_name", "phenology"]
missing_base = [c for c in required_base_cols if c not in df.columns]
if missing_base:
    raise ValueError(f"Missing required columns: {missing_base}")

missing_metrics = [c for c in metric_cols if c not in df.columns]
if missing_metrics:
    raise ValueError(f"Missing required metric columns: {missing_metrics}")

# Filter model_type
sub = df[df["model_type"] == selected_model_type].copy()

# Filter scenario or fallback to event_threshold
if "scenario" in sub.columns:
    sub = sub[sub["scenario"] == selected_scenario].copy()
elif "event_threshold" in sub.columns:
    sub = sub[pd.to_numeric(sub["event_threshold"], errors="coerce") == selected_event_threshold].copy()

if len(sub) == 0:
    raise ValueError("No rows remain after filtering by model_type and scenario/event_threshold.")

# Identify feature column
if "x_var" in sub.columns:
    sub["feature_name_plot"] = sub["x_var"].astype(str)
elif "feature_set" in sub.columns:
    sub["feature_name_plot"] = sub["feature_set"].astype(str)
else:
    raise ValueError("Missing feature identifier column: x_var or feature_set")

# Keep only selected run and selected features
sub = sub[sub["run_name"].astype(str) == selected_run_name].copy()
sub = sub[sub["feature_name_plot"].isin(selected_features)].copy()

if len(sub) == 0:
    raise ValueError(
        f"No rows found for run_name == {selected_run_name} "
        f"and features in {selected_features}"
    )

sub["phenology_en"] = sub["phenology"].map(phenology_map).fillna(sub["phenology"].astype(str))

for c in metric_cols:
    sub[c] = pd.to_numeric(sub[c], errors="coerce")

# Keep one row per run_name × phenology × feature
sub = (
    sub.sort_values([col for col in ["run_name", "phenology_en", "feature_name_plot"] if col in sub.columns])
       .drop_duplicates(subset=["run_name", "phenology_en", "feature_name_plot"], keep="first")
       .copy()
)

# Orders for plotting/sorting
sub["phenology_order"] = sub["phenology_en"].map(
    {v: i for i, v in enumerate(phenology_order)}
).fillna(999)

sub["feature_order"] = sub["feature_name_plot"].map(
    {v: i for i, v in enumerate(feature_order)}
).fillna(999)

# ============================================================
# 2. Save filtered table
# ============================================================
table_cols = [
    "run_name",
    "phenology_en",
    "feature_name_plot",
] + metric_cols

out_table = (
    sub
    .sort_values(["phenology_order", "feature_order", "phenology_en", "feature_name_plot"])
    [table_cols]
    .copy()
)

out_table.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"Filtered table saved: {output_csv}")

# ============================================================
# 3. Build heatmap
#    rows = phenology × (Raw Max / Raw Min)
#    cols = four delta metrics
# ============================================================
row_info = (
    sub[["phenology_en", "feature_name_plot", "phenology_order", "feature_order"]]
    .drop_duplicates()
    .sort_values(["phenology_order", "feature_order", "phenology_en", "feature_name_plot"])
    .reset_index(drop=True)
)

row_info["row_label"] = (
    row_info["phenology_en"].astype(str)
    + " | "
    + row_info["feature_name_plot"].map(feature_label_map).fillna(row_info["feature_name_plot"].astype(str))
)

pivot_value = sub.pivot_table(
    index=["phenology_en", "feature_name_plot"],
    values=metric_cols,
    aggfunc="first"
)

index_tuples = [tuple(x) for x in row_info[["phenology_en", "feature_name_plot"]].to_numpy()]
pivot_value = pivot_value.reindex(index=index_tuples, columns=metric_cols)

Z = pivot_value.to_numpy(dtype=float)

valid_vals = Z[np.isfinite(Z)]
if valid_vals.size == 0:
    raise ValueError("No valid numeric values found for plotting.")

max_abs = max(np.nanmax(np.abs(valid_vals)), 1e-6)
norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

fig_w = max(9, len(metric_cols) * 2.0 + 2.5)
fig_h = max(5.5, len(row_info) * 0.62 + 2.8)

fig, ax = plt.subplots(figsize=(fig_w, fig_h))

im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", norm=norm)

ax.set_xticks(np.arange(len(metric_cols)))
ax.set_xticklabels(
    [metric_label_map.get(c, c) for c in metric_cols],
    rotation=25,
    ha="right",
    fontsize=10
)

ax.set_yticks(np.arange(len(row_info)))
ax.set_yticklabels(row_info["row_label"].tolist(), fontsize=10)

# Grid lines
ax.set_xticks(np.arange(-0.5, len(metric_cols), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(row_info), 1), minor=True)
ax.grid(which="minor", color="black", linestyle="-", linewidth=0.45, alpha=0.30)
ax.tick_params(which="minor", bottom=False, left=False)

def format_value(v):
    """Adaptive number formatting for annotations."""
    if pd.isna(v):
        return "NA"
    v = float(v)
    if abs(v) >= 100:
        return f"{v:.1f}"
    elif abs(v) >= 10:
        return f"{v:.2f}"
    elif abs(v) >= 1:
        return f"{v:.3f}"
    else:
        return f"{v:.4f}"

# Annotate cells
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        val = Z[i, j]
        if np.isfinite(val):
            text_color = "white" if abs(val) > max_abs * 0.55 else "black"
            ax.text(
                j, i, format_value(val),
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="bold"
            )
        else:
            ax.text(
                j, i, "NA",
                ha="center",
                va="center",
                fontsize=8,
                color="gray"
            )

# ax.set_title(
#     "Selected Delta Metrics for Univariate Model Comparison\n"
#     f"Scenario: {selected_scenario}; Run name: {selected_run_name}",
#     fontsize=15,
#     pad=16
# )

# ax.set_xlabel("Model-comparison metric", fontsize=12)
# ax.set_ylabel("Phenological Stage | LDFAI Feature", fontsize=12)

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Raw delta value: SPEI - SPI", fontsize=10)

# note = (
#     "Cell value = metric(SPEI-LDFAI) - metric(SPI-LDFAI).\n"
#     "Displayed metrics: Delta R², Delta Adjusted R², Delta RMSE, Delta MAE.\n"
#     "Rows only include Raw Max and Raw Min under run_name = linear_bic.\n"
#     "For R² and adjusted R², positive values generally favor SPEI-LDFAI.\n"
#     "For RMSE and MAE, negative values generally favor SPEI-LDFAI."
# )
# fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=9)

plt.tight_layout(rect=[0, 0.10, 1, 1])

fig.savefig(output_png, dpi=300, bbox_inches="tight")
fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Figure saved: {output_png}")
print(f"Figure saved: {output_pdf}")
print("Done.")