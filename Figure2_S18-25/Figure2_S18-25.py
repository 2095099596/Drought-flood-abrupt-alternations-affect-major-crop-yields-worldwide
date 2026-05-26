# -*- coding: utf-8 -*-

import os
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.util import add_cyclic_point

Data_from = 'GPCC'
# ==========================================================
# 1. 基本设置
# ==========================================================
DATA_ROOT = r"C:\Users\lxh\Desktop\NongZuoWu_ChanLiang\New_result\Data\LDFAI_SPEI_GPCC_3_2"

CROPS = ["maize", "rice", "soybeans", "wheat"]

# 三个阶段：种植、生长、收获
STAGES = [
    ("planting", "Planting"),
    ("growing", "Growing"),
    ("harvesting", "Harvesting")
]

# 1.0 代表 DTF，-1.0 代表 FTD
LAG_SETTINGS = [
    ("1.0", "DTF"),
    ("-1.0", "FTD")
]

OUT_DIR = r"D:\Python\DssAat\NO_2\Figure2a\Figure1c\four_crops2"
os.makedirs(OUT_DIR, exist_ok=True)

LAT_MIN = -60
LAT_MAX = 60

CBAR_LABEL = " "

PANEL_TITLES = {
    "maize": "Maize",
    "rice": "Rice",
    "soybeans": "Soybeans",
    "wheat": "Wheat"
}

# 每个作物需要统计的国家，柱状图按这里的顺序绘制
CROP_COUNTRY_ORDER = {
    "maize": [
        "United States of America",
        "China",
        "Brazil",
        "Argentina",
        "Mexico"
    ],

    "rice": [
        "China",
        "India",
        "Bangladesh",
        "Indonesia",
        "Vietnam"
    ],

    "soybeans": [
        "Brazil",
        "United States of America",
        "Argentina",
        "China",
        "India"
    ],

    "wheat": [
        "China",
        "United States of America",
        "Ukraine",
        "Australia",
        "Canada",
    ]
}

# 只影响柱状图显示标签；国家匹配仍使用 CROP_COUNTRY_ORDER 里的完整名称
COUNTRY_DISPLAY_NAMES = {
    "United States of America": "USA",
}


# ==========================================================
# 2. 每个地图单独设置色带范围
# ==========================================================
V_LIMS = {
    "maize": 0.45,
    "rice": 0.2,
    "soybeans": 0.1,
    "wheat": 0.4
}

CBAR_TICKS = {
    "maize": [-0.4, -0.2, 0, 0.2, 0.4],
    "rice": [-0.2, 0, 0.2],
    "soybeans": [-0.1, 0, 0.1],
    "wheat": [-0.4, 0, 0.4]
}


# ==========================================================
# 3. 绘图参数
# ==========================================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7.5,

    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,

    "axes.linewidth": 0.55,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
    "figure.dpi": 300
})


# ==========================================================
# 4. 色带
# ==========================================================
nature_cmap = LinearSegmentedColormap.from_list(
    "nature_residual",
    [
        "#B2182B",
        "#D6604D",
        "#F4A582",
        "#FDDBC7",
        "#F7F7F7",
        "#D1E5F0",
        "#92C5DE",
        "#4393C3",
        "#2166AC"
    ],
    N=256
)

nature_cmap.set_bad(color=(1, 1, 1, 0))


# ==========================================================
# 5. 根据 stage 和 lag 构建 NC 文件名
# ==========================================================
def get_nc_file(stage, lag):
    return f"ls_{lag}_residual_mean_{stage}_-60_60_0_360.0s_multimodel_aic.nc"


def get_nc_path(crop, stage, lag):
    nc_file = get_nc_file(stage, lag)
    return os.path.join(DATA_ROOT, crop, "multimodel_aic", nc_file)


# ==========================================================
# 6. 读取单个作物数据
# ==========================================================
def read_crop_data(nc_path):
    ds = xr.open_dataset(nc_path)

    lat_candidates = ["lat", "latitude", "LAT", "Latitude", "y"]
    lon_candidates = ["lon", "longitude", "LON", "Longitude", "x"]

    lat_name = next(
        name for name in lat_candidates
        if name in ds.coords or name in ds.dims
    )

    lon_name = next(
        name for name in lon_candidates
        if name in ds.coords or name in ds.dims
    )

    var_name = None
    for name, da_tmp in ds.data_vars.items():
        if (
            lat_name in da_tmp.dims
            and lon_name in da_tmp.dims
            and np.issubdtype(da_tmp.dtype, np.number)
        ):
            var_name = name
            break

    if var_name is None:
        ds.close()
        raise ValueError(
            f"{nc_path} 中没有找到同时包含经纬度维度的数值变量，请手动指定 var_name。"
        )

    da = ds[var_name]

    # 若存在 time、level 等额外维度，自动平均
    extra_dims = [d for d in da.dims if d not in [lat_name, lon_name]]
    if extra_dims:
        da = da.mean(dim=extra_dims, skipna=True)

    # 纬度排序
    da = da.sortby(lat_name)

    # 经度 0–360 转为 -180–180
    lon_new = ((da[lon_name] + 180) % 360) - 180
    da = da.assign_coords({lon_name: lon_new}).sortby(lon_name)

    # 只取 -60 到 60
    da = da.sel({lat_name: slice(LAT_MIN, LAT_MAX)})

    # 保证维度顺序
    da = da.transpose(lat_name, lon_name)

    lat = da[lat_name].values
    lon = da[lon_name].values
    data = da.values

    # 纬向均值
    lat_mean = da.mean(dim=lon_name, skipna=True).values

    # 避免地图接缝
    data_cyclic, lon_cyclic = add_cyclic_point(data, coord=lon)

    ds.close()

    return lat, lon, lon_cyclic, data_cyclic, lat_mean, data


# ==========================================================
# 7. 合并图手动布局参数
# ==========================================================
# 原来单张四作物图是 11.2 x 4.8
# 现在上下合并 DTF 和 FTD，并在每个折线图右侧增加国家均值柱状图，
# 因此采用适度紧凑但不重叠的横向布局：
# 适当增加折线图与柱状图、左右两列之间的留白。
FIGSIZE = (14.6, 9.6)

X_LEFT = 0.038
X_RIGHT = 0.500

# DTF 两行
Y_DTF_TOP = 0.785
Y_DTF_BOTTOM = 0.575

# FTD 两行
Y_FTD_TOP = 0.325
Y_FTD_BOTTOM = 0.115

MAP_W = 0.280
MAP_H = 0.150

LINE_GAP = 0.012
LINE_W = 0.048

# 柱状图放在折线图右侧；BAR_GAP 预留给折线图右侧纬度刻度和柱状图国家名
BAR_GAP = 0.055
BAR_W = 0.060

CBAR_GAP = 0.010
CBAR_H = 0.006
CBAR_W_RATIO = 0.84


def get_panel_positions(lag_index):
    """
    lag_index = 0: DTF
    lag_index = 1: FTD
    """

    if lag_index == 0:
        y_top = Y_DTF_TOP
        y_bottom = Y_DTF_BOTTOM
    else:
        y_top = Y_FTD_TOP
        y_bottom = Y_FTD_BOTTOM

    return {
        "maize": {
            "map_rect": [X_LEFT, y_top, MAP_W, MAP_H]
        },
        "rice": {
            "map_rect": [X_RIGHT, y_top, MAP_W, MAP_H]
        },
        "soybeans": {
            "map_rect": [X_LEFT, y_bottom, MAP_W, MAP_H]
        },
        "wheat": {
            "map_rect": [X_RIGHT, y_bottom, MAP_W, MAP_H]
        }
    }



# ==========================================================
# 8. 国家 / 全球均值计算工具
# ==========================================================
@lru_cache(maxsize=1)
def load_country_geometries():
    """读取 Natural Earth 国家边界，并建立多个名称字段的索引。"""
    shp_path = shpreader.natural_earth(
        resolution="110m",
        category="cultural",
        name="admin_0_countries"
    )

    records = list(shpreader.Reader(shp_path).records())
    country_geoms = {}

    # 先用 ADMIN / NAME 等国家自身名称建索引，再用 SOVEREIGNT 兜底。
    # 这样可以避免 USA、China 等主权名被海外属地记录提前占用。
    primary_fields = ["ADMIN", "NAME_LONG", "NAME", "BRK_NAME", "FORMAL_EN", "GEOUNIT"]
    secondary_fields = ["SOVEREIGNT"]

    for field in primary_fields + secondary_fields:
        for rec in records:
            name = rec.attributes.get(field)
            if name and name not in country_geoms:
                country_geoms[name] = rec.geometry

    return country_geoms


def get_country_geometry(country_name):
    country_geoms = load_country_geometries()
    geom = country_geoms.get(country_name)

    if geom is None:
        available = sorted(country_geoms.keys())
        close_names = [name for name in available if country_name.lower() in name.lower()]
        hint = f" 可检查相近名称：{close_names[:8]}" if close_names else ""
        raise KeyError(f"Natural Earth 中未找到国家：{country_name}.{hint}")

    return geom


_COUNTRY_MASK_CACHE = {}


def get_country_mask(country_name, lat, lon):
    """返回指定国家在当前经纬度网格上的布尔掩膜。"""
    key = (
        country_name,
        len(lat), len(lon),
        round(float(lat[0]), 6), round(float(lat[-1]), 6),
        round(float(lon[0]), 6), round(float(lon[-1]), 6),
    )

    if key in _COUNTRY_MASK_CACHE:
        return _COUNTRY_MASK_CACHE[key]

    geom = get_country_geometry(country_name)
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Shapely 2.x
    try:
        from shapely import contains_xy
        mask = contains_xy(geom, lon2d, lat2d)
    except Exception:
        # Shapely 1.x
        try:
            from shapely.vectorized import contains
            mask = contains(geom, lon2d, lat2d)
        except Exception:
            # 最稳妥但较慢的回退方案
            from shapely.geometry import Point
            from shapely.prepared import prep
            prepared_geom = prep(geom)
            mask = np.zeros(lon2d.shape, dtype=bool)
            for i in range(lon2d.shape[0]):
                for j in range(lon2d.shape[1]):
                    mask[i, j] = prepared_geom.contains(
                        Point(float(lon2d[i, j]), float(lat2d[i, j]))
                    )

    _COUNTRY_MASK_CACHE[key] = mask
    return mask


def area_weighted_mean(data, lat, mask=None):
    """对规则经纬度网格做 cos(lat) 面积加权平均。"""
    valid = np.isfinite(data)
    if mask is not None:
        valid = valid & mask

    if not np.any(valid):
        return np.nan

    weights = np.cos(np.deg2rad(lat))[:, None]
    weights = np.where(valid, weights, np.nan)

    weighted_sum = np.nansum(data * weights)
    weight_sum = np.nansum(weights)

    if not np.isfinite(weight_sum) or weight_sum == 0:
        return np.nan

    return weighted_sum / weight_sum


def compute_bar_means(crop, lat, lon, data_raw):
    """返回柱状图标签、完整国家名和均值。第一个始终是 Global。"""
    country_names = CROP_COUNTRY_ORDER[crop]

    names_for_calc = ["Global"] + country_names
    labels_for_plot = ["★ Global"] + [
        COUNTRY_DISPLAY_NAMES.get(country, country)
        for country in country_names
    ]

    values = [area_weighted_mean(data_raw, lat)]

    for country in country_names:
        mask = get_country_mask(country, lat, lon)
        values.append(area_weighted_mean(data_raw, lat, mask=mask))

    return labels_for_plot, names_for_calc, np.array(values, dtype=float)


def plot_country_bar(ax_bar, labels, values):
    """绘制右侧国家 / 全球均值横向柱状图。"""
    y = np.arange(len(labels))
    plot_values = np.where(np.isfinite(values), values, 0.0)

    colors = ["0.18"] + ["0.68"] * (len(labels) - 1)
    bars = ax_bar.barh(
        y,
        plot_values,
        height=0.62,
        color=colors,
        edgecolor="0.25",
        linewidth=0.35,
        zorder=2
    )

    # Global 特殊标记：深色 + hatch + 星形点
    bars[0].set_hatch("////")
    bars[0].set_edgecolor("0.05")
    bars[0].set_linewidth(0.70)

    if np.isfinite(values[0]):
        ax_bar.scatter(
            values[0],
            0,
            marker="*",
            s=18,
            color="0.02",
            zorder=4
        )

    # NA 标注，避免国家没有有效网格时误解为 0
    for yi, value in enumerate(values):
        if not np.isfinite(value):
            ax_bar.text(
                0,
                yi,
                "NA",
                ha="center",
                va="center",
                fontsize=5.8,
                color="0.25",
                zorder=5
            )

    ax_bar.axvline(
        0,
        color="0.45",
        linewidth=0.50,
        linestyle="--",
        zorder=1
    )

    finite_values = values[np.isfinite(values)]
    if finite_values.size > 0:
        x_abs = np.nanmax(np.abs(finite_values)) * 1.30
    else:
        x_abs = 0.05
    if not np.isfinite(x_abs) or x_abs == 0:
        x_abs = 0.05
    x_abs = max(x_abs, 0.05)

    ax_bar.set_xlim(-x_abs, x_abs)
    ax_bar.set_ylim(-0.6, len(labels) - 0.4)
    ax_bar.invert_yaxis()

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels, fontsize=5.9)
    ax_bar.tick_params(
        axis="y",
        direction="out",
        length=0,
        width=0,
        pad=1.2
    )

    ax_bar.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax_bar.tick_params(
        axis="x",
        direction="out",
        length=2.0,
        width=0.45,
        pad=1.2,
        labelsize=6.2
    )

    ax_bar.grid(
        True,
        axis="x",
        color="0.88",
        linewidth=0.30,
        linestyle="-",
        alpha=0.8,
        zorder=0
    )

    ax_bar.set_xlabel(
        "Mean",
        fontsize=6.5,
        labelpad=1.0
    )

    for spine in ax_bar.spines.values():
        spine.set_linewidth(0.50)
        spine.set_color("0.25")

# ==========================================================
# 8. 绘制单个作物
# ==========================================================
def plot_one_crop(fig, crop, stage, lag, lag_title, map_rect, panel_label):
    nc_path = get_nc_path(crop, stage, lag)

    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"找不到文件：{nc_path}")

    lat, lon, lon_cyclic, data_cyclic, lat_mean, data_raw = read_crop_data(nc_path)

    # 每个图单独读取自己的色带范围
    vlim = V_LIMS.get(crop, None)
    if vlim is None:
        vlim = np.nanpercentile(np.abs(data_raw), 98)
        if not np.isfinite(vlim) or vlim == 0:
            vlim = 0.05

    bar_labels, _, bar_values = compute_bar_means(
        crop,
        lat,
        lon,
        data_raw
    )

    norm = mcolors.TwoSlopeNorm(
        vmin=-vlim,
        vcenter=0,
        vmax=vlim
    )

    map_proj = ccrs.Robinson(central_longitude=0)
    data_proj = ccrs.PlateCarree()

    # --------------------------
    # 地图
    # --------------------------
    ax_map = fig.add_axes(map_rect, projection=map_proj)
    ax_map.set_facecolor("white")

    ax_map.add_feature(
        cfeature.LAND,
        facecolor="#FAFAFA",
        edgecolor="none",
        zorder=0
    )

    im = ax_map.pcolormesh(
        lon_cyclic,
        lat,
        data_cyclic,
        transform=data_proj,
        cmap=nature_cmap,
        norm=norm,
        shading="auto",
        rasterized=True,
        zorder=2
    )

    ax_map.set_extent([-180, 180, LAT_MIN, LAT_MAX], crs=data_proj)

    ax_map.coastlines(
        resolution="110m",
        linewidth=0.35,
        color="0.25",
        zorder=4
    )

    ax_map.add_feature(
        cfeature.BORDERS,
        linewidth=0.16,
        edgecolor="0.45",
        zorder=4
    )

    ax_map.gridlines(
        crs=data_proj,
        linewidth=0.25,
        color="0.82",
        linestyle="-",
        alpha=0.55,
        draw_labels=False,
        xlocs=np.arange(-180, 181, 60),
        ylocs=np.arange(-60, 61, 30),
        zorder=1
    )

    try:
        ax_map.spines["geo"].set_linewidth(0.50)
        ax_map.spines["geo"].set_edgecolor("0.25")
    except Exception:
        pass

    # 标题格式：小字母 + 作物名
    ax_map.set_title(
        f"({panel_label}) {PANEL_TITLES[crop]}",
        fontsize=9.2,
        fontweight="bold",
        pad=3
    )

    fig.canvas.draw()
    map_pos = ax_map.get_position()

    # --------------------------
    # 折线图
    # --------------------------
    line_rect = [
        map_pos.x1 + LINE_GAP,
        map_pos.y0,
        LINE_W,
        map_pos.height
    ]

    ax_line = fig.add_axes(line_rect)
    ax_line.set_facecolor("white")

    ax_line.plot(
        lat_mean,
        lat,
        color="0.05",
        linewidth=0.80,
        solid_capstyle="round"
    )

    ax_line.axvline(
        0,
        color="0.65",
        linewidth=0.50,
        linestyle="--",
        zorder=0
    )

    ax_line.set_ylim(LAT_MIN, LAT_MAX)
    ax_line.set_yticks([-60, -30, 0, 30, 60])

    ax_line.yaxis.tick_right()
    ax_line.yaxis.set_label_position("right")
    ax_line.tick_params(
        axis="y",
        labelleft=False,
        labelright=True
    )

    x_abs = np.nanpercentile(np.abs(lat_mean), 98)
    if not np.isfinite(x_abs) or x_abs == 0:
        x_abs = 0.05
    x_abs = max(x_abs * 1.25, 0.05)

    ax_line.set_xlim(-x_abs, x_abs)
    ax_line.xaxis.set_major_locator(MaxNLocator(nbins=3))

    ax_line.tick_params(
        axis="x",
        which="both",
        labelbottom=True,
        bottom=True
    )

    ax_line.tick_params(
        axis="both",
        direction="out",
        length=2.3,
        width=0.50,
        pad=1.5
    )

    for spine in ax_line.spines.values():
        spine.set_linewidth(0.50)
        spine.set_color("0.25")

    ax_line.grid(
        True,
        color="0.88",
        linewidth=0.30,
        linestyle="-",
        alpha=0.8
    )

    # --------------------------
    # 国家 / 全球均值柱状图
    # --------------------------
    line_pos = ax_line.get_position()
    bar_rect = [
        line_pos.x1 + BAR_GAP,
        line_pos.y0,
        BAR_W,
        line_pos.height
    ]

    ax_bar = fig.add_axes(bar_rect)
    ax_bar.set_facecolor("white")
    plot_country_bar(ax_bar, bar_labels, bar_values)

    # --------------------------
    # 色条
    # --------------------------
    cbar_w = map_pos.width * CBAR_W_RATIO
    cbar_x = map_pos.x0 + (map_pos.width - cbar_w) / 2
    cbar_y = map_pos.y0 - CBAR_GAP - CBAR_H

    cbar_rect = [
        cbar_x,
        cbar_y,
        cbar_w,
        CBAR_H
    ]

    cbar_ax = fig.add_axes(cbar_rect)

    cbar = fig.colorbar(
        im,
        cax=cbar_ax,
        orientation="horizontal"
    )

    cbar.outline.set_linewidth(0.35)
    cbar.ax.tick_params(
        direction="out",
        length=1.5,
        width=0.35,
        pad=1.0,
        labelsize=6.5
    )

    if crop in CBAR_TICKS:
        cbar.set_ticks(CBAR_TICKS[crop])
    else:
        cbar.locator = MaxNLocator(nbins=5)
        cbar.update_ticks()

    cbar.set_label(
        CBAR_LABEL,
        fontsize=6.8,
        labelpad=1.2
    )

    return ax_map, ax_line, ax_bar, cbar_ax


# ==========================================================
# 9. 绘制某一个阶段的 DTF + FTD 合并图
# ==========================================================
def plot_merged_stage(stage, stage_title):
    fig = plt.figure(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")

    # 总标题：阶段名
    # fig.text(
    #     0.5,
    #     0.975,
    #     stage_title,
    #     ha="center",
    #     va="top",
    #     fontsize=12,
    #     fontweight="bold"
    # )

    # DTF / FTD 标签更靠近左侧图片
    GROUP_LABEL_X = X_LEFT + 0.01 # 数值越大越靠右，越贴近图片

    fig.text(
        GROUP_LABEL_X,
        0.735,
        "DTF",
        ha="right",
        va="center",
        rotation=90,
        fontsize=11,
        fontweight="bold"
    )

    fig.text(
        GROUP_LABEL_X,
        0.275,
        "FTD",
        ha="right",
        va="center",
        rotation=90,
        fontsize=11,
        fontweight="bold"
    )

    panel_index = 0

    for lag_index, (lag, lag_title) in enumerate(LAG_SETTINGS):
        positions = get_panel_positions(lag_index)

        for crop in CROPS:
            panel_label = chr(ord("a") + panel_index)

            plot_one_crop(
                fig=fig,
                crop=crop,
                stage=stage,
                lag=lag,
                lag_title=lag_title,
                map_rect=positions[crop]["map_rect"],
                panel_label=panel_label
            )

            panel_index += 1

    out_png = os.path.join(
        OUT_DIR,
        f"Figure2aa_four_crops_merged_{stage}_{Data_from}.png"
    )

    out_pdf = os.path.join(
        OUT_DIR,
        f"Figure2aa_four_crops_merged_{stage}_{Data_from}.pdf"
    )

    plt.savefig(
        out_png,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white"
    )

    plt.savefig(
        out_pdf,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white"
    )

    plt.close(fig)

    print(f"{stage_title} PNG saved to:", out_png)
    print(f"{stage_title} PDF saved to:", out_pdf)


# ==========================================================
# 10. 主程序：生成种植、生长、收获三张合并图
# ==========================================================
for stage, stage_title in STAGES:
    plot_merged_stage(stage, stage_title)

print("全部合并图生成完成。")