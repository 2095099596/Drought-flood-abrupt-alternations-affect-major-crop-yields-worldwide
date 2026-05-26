# ============================================================
# 全部样本 + 正负 LDFAI + 分物候期相关分析脚本
# 加入：随机输出 LDFAI-残差配对样本，用于检查配对是否成功
#
# 功能：
# 1. 不按阈值筛选，使用全部样本；
# 2. 分别计算种植期、生长期、收获期；
# 3. 同时考虑 LDFAI 原始值、绝对强度、正向急转、负向急转；
# 4. 同时比较 SPI-LDFAI 和 SPEI-LDFAI；
# 5. 只输出 CSV，不输出 NC；
# 6. 运行过程中随机打印若干 LDFAI-产量残差配对样本。
#
# 输出：
# 1. event_ldfai_abs_gt1~gt9_correlation_summary.csv
# 2. event_ldfai_abs_gt1~gt9_spi_spei_compare.csv
# 3. 可选：event_ldfai_abs_gt1~gt9_pairs.csv
# ============================================================

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit
from scipy.stats import t as student_t


# ============================================================
# 0. 参数配置
# ============================================================

# ----------------------------
# 产量 NC 文件
# 如果这里填的是目录，代码会自动尝试拼接 merged_yield_data.nc
# ----------------------------
nc_file = "/mnt/hdb/LXH/data/NZW/gdhy_v1.2_v1.3_20190128/soybean/merged_yield_data.nc"
data_var = "var"
lat_var = "lat"
lon_var = "lon"
time_var = "time"

# ----------------------------
# 作物物候文件
# ----------------------------
crop_nc = "/mnt/hdb/LXH/data/NZW/plant & harvest(0.5 degree)/Soybeans.crop.calendar.fill.nc/Soybeans.crop.calendar.fill.nc"

crop_vars = [
    "plant.start", "plant.range", "plant.end", "plant",
    "tot.days", "harvest", "harvest.start", "harvest.range", "harvest.end"
]

# ----------------------------
# SPI-LDFAI 和 SPEI-LDFAI 文件
# ----------------------------
index_files = {
    "SPI": "/mnt/hdb/LXH/data/NZW/ldfai_from_spi_1month_region_-60_60_0_360.nc",
    "SPEI": "/mnt/hdb/LXH/data/NZW/ldfai_from_spei_1month_region_-60_60_0_360.nc"
}

target_var = "ldfai"

# ----------------------------
# 研究区域
# ----------------------------
lat_min = -60
lat_max = 60
lon_min = 0
lon_max = 360.0

# ----------------------------
# 物候转换基准年
# ----------------------------
base_year = 1980

# ----------------------------
# 是否跳过物候期起始月份
# True 表示沿用你原脚本 ldfai_year_month[1:] 的逻辑
# ----------------------------
skip_first_ldfai_month = True

# ----------------------------
# 趋势拟合设置
# 为节省时间，默认只用 linear
# ----------------------------
model_runs = [
    {
        "run_name": "linear_bic",
        "criterion": "bic",
        "candidate_models": ["linear"]
    },

    # 稳健性分析时可以打开：
    {
        "run_name": "quadratic_bic",
        "criterion": "bic",
        "candidate_models": ["quadratic"]
    },

    {
        "run_name": "multimodel_aic",
        "criterion": "aic",
        "candidate_models": ["linear", "quadratic", "cubic", "log", "gaussian"]
    }
]

min_valid_points = 20
max_outlier_iter = 3
z_threshold = 2.0

# ----------------------------
# 提速模式
# True：关闭 Spearman 样本值缓存、关闭全部 pairs 表、关闭 debug 打印。
# 这会显著减少内存占用和 CSV 写出时间；Pearson summary/compare 仍正常输出。
# 如果需要 Spearman 或 pairs 明细，把 FAST_MODE 改为 False，或手动改下面三个开关。
# ----------------------------
FAST_MODE = True

# ----------------------------
# 是否计算 Spearman
# True 会保存样本值，内存占用更高
# 全球格点很多时建议先 False
# ----------------------------
calc_spearman =  FAST_MODE

# ----------------------------
# 是否保存全部配对样本表
# 如果 True，输出文件可能很大，且阈值 1~9 会重复保存嵌套样本。
# ----------------------------
save_pair_table =  FAST_MODE

# ----------------------------
# 随机输出配对样本，用于检查是否匹配成功
# ----------------------------
debug_pair_sample =  FAST_MODE
debug_pair_max_print = 1
debug_pair_probability = 0.0005
debug_pair_seed = 20260507
debug_rng = np.random.default_rng(debug_pair_seed)

# 如果你只想输出第一个成功样本，可改成：
# debug_pair_max_print = 1
# debug_pair_probability = 1.0

# ----------------------------
# 仅保留的 LDFAI 特征
# ----------------------------
kept_ldfai_feature_names = [
    "ldfai_raw_max",
    "ldfai_raw_min",
    "ldfai_abs_sum",
    "ldfai_pos_event_max",
    "ldfai_neg_event_max",
]

# ----------------------------
# 输出目录
# ----------------------------
output_dir = "/mnt/hdb/LXH/data/NZW/LDFAI_signed_correlation/soybeans_/"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 样本筛选与批量阈值输出设置
# ============================================================
# 目标：分别对 |LDFAI| > 1, 2, ..., 9 的强旱涝急转事件样本进行 common sample 分析。

# 要求 SPI 和 SPEI 使用完全相同的格点-年份-物候期样本。
require_common_spi_spei_sample = True

# 建议保持 True：确保 SPI 和 SPEI 在同一物候期内使用完全相同的月份窗口。
require_same_ldfai_months = True

# 每个物候期至少需要多少个有效 LDFAI 月份。
min_ldfai_months_required = 1

# 强旱涝急转事件阈值批量运行范围：1, 2, ..., 9。
# 输出文件会分别命名为 event_ldfai_abs_gt1_*.csv ... event_ldfai_abs_gt9_*.csv。
event_ldfai_abs_thresholds = list(range(4, 10))

# "either"：SPI 或 SPEI 任一指数在某月 |LDFAI| > 阈值，即把该月作为事件月；
# "both"：SPI 和 SPEI 同一月都 |LDFAI| > 阈值，才把该月作为事件月。
# 推荐 either，样本不会过少，且 SPI/SPEI 仍然使用同一批事件月份。
event_pair_rule = "both"


def format_threshold_label(threshold):
    """将阈值转成适合文件名的标签。"""
    threshold = float(threshold)
    if threshold.is_integer():
        return str(int(threshold))
    return str(threshold).replace(".", "p")


def make_event_scenario_name(threshold):
    """根据阈值生成情景名。"""
    return f"event_ldfai_abs_gt{format_threshold_label(threshold)}"


def make_event_scenario_config(threshold):
    """根据阈值生成输出文件配置。"""
    threshold_label = format_threshold_label(threshold)
    scenario_name = make_event_scenario_name(threshold)

    return {
        "event_ldfai_abs_threshold": float(threshold),
        "description": (
            "common event sample: same SPI/SPEI samples; "
            f"selected months satisfy |LDFAI| > {threshold_label} by either/both rule"
        ),
        "summary_csv": os.path.join(output_dir, f"{scenario_name}_correlation_summary.csv"),
        "compare_csv": os.path.join(output_dir, f"{scenario_name}_spi_spei_compare.csv"),
        "pair_csv": os.path.join(output_dir, f"{scenario_name}_pairs.csv"),
        "count_check_csv": os.path.join(output_dir, f"{scenario_name}_count_check.csv"),
    }


# 批量事件分析情景。
analysis_scenarios = {
    make_event_scenario_name(threshold): make_event_scenario_config(threshold)
    for threshold in event_ldfai_abs_thresholds
}


# ============================================================
# 1. 时间、空间、物候基础函数
# ============================================================

def resolve_nc_file(path):
    """
    如果传入的是目录，则自动尝试查找 merged_yield_data.nc。
    """
    if os.path.isdir(path):
        candidate = os.path.join(path, "merged_yield_data.nc")
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(
            f"nc_file 是目录，但未找到 merged_yield_data.nc：{candidate}"
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"产量 NC 文件不存在：{path}")

    return path


def parse_years_from_time(time_values):
    """
    从 NC 文件 time 坐标中解析真实年份。
    """
    time_values = np.asarray(time_values)

    if np.issubdtype(time_values.dtype, np.number):
        years = time_values.astype(int)
        if np.nanmin(years) > 1000 and np.nanmax(years) < 3000:
            return years

    try:
        return pd.to_datetime(time_values).year.astype(int)
    except Exception:
        pass

    try:
        return np.array([int(t.year) for t in time_values], dtype=int)
    except Exception as e:
        raise ValueError(f"无法从 time 坐标解析年份：{e}")


def prepare_annual_yield_data(ds, data_var, time_var):
    """
    将产量数据整理为年尺度 DataArray。
    输出维度：[year, lat, lon]
    """
    if data_var not in ds:
        raise ValueError(f"数据变量 {data_var} 不在产量 NC 文件中")

    if time_var not in ds:
        raise ValueError(f"时间变量 {time_var} 不在产量 NC 文件中")

    yield_da = ds[data_var]

    if time_var not in yield_da.dims:
        raise ValueError(
            f"变量 {data_var} 的维度中不包含时间维 {time_var}，实际维度为 {yield_da.dims}"
        )

    yield_years = parse_years_from_time(ds[time_var].values)

    yield_da = yield_da.assign_coords(year=(time_var, yield_years))
    yield_da_annual = yield_da.groupby("year").mean(dim=time_var, skipna=True)
    yield_da_annual = yield_da_annual.sortby("year")

    years = yield_da_annual["year"].values.astype(int)

    print(f"产量年份范围：{years.min()} ~ {years.max()}，共 {len(years)} 年")
    print(f"产量年尺度数组维度：{yield_da_annual.dims}, shape={yield_da_annual.shape}")

    return yield_da_annual, years


def nearest_index(arr, value):
    """返回 arr 中距离 value 最近的索引。"""
    return int(np.nanargmin(np.abs(np.asarray(arr, dtype=float) - float(value))))


def convert_lon_for_dataset(lon_value, lon_array):
    """
    根据目标数据经度范围转换经度。
    """
    lon_value = float(lon_value)
    lon_array = np.asarray(lon_array, dtype=float)

    if np.nanmax(lon_array) > 180:
        return lon_value % 360

    if lon_value > 180:
        return lon_value - 360

    return lon_value


def days_to_month_cross(days, plant_start, day_range, base_year, start=True):
    """
    将物候天数转换为：
        month, is_cross_year

    month: 1-12
    is_cross_year: 0 表示本年，1 表示次年
    """
    if pd.isna(days):
        return np.nan, np.nan

    base_date = pd.to_datetime(f"{base_year}-01-01")

    if days < plant_start:
        base_date += pd.Timedelta(days=np.round(365))

    if day_range >= 365:
        base_date += pd.Timedelta(days=np.round(365))

    target_date = base_date + pd.Timedelta(days=np.round(days))

    is_cross_year = 1 if target_date.year > base_year else 0

    if start:
        month = target_date.month + 1 if target_date.day > 15 else target_date.month
    else:
        month = target_date.month - 1 if target_date.day < 15 else target_date.month

    if month == 0:
        month = 12
        is_cross_year = 0

    if month == 13:
        month = 1
        is_cross_year = 1

    return month, is_cross_year


def generate_monthly_dates(start_year, start_month, end_year, end_month):
    """
    生成从 start_year/start_month 到 end_year/end_month 的逐月列表。
    """
    current_date = datetime(int(start_year), int(start_month), 1)
    end_date = datetime(int(end_year), int(end_month), 1)

    monthly_dates = []

    while current_date <= end_date:
        monthly_dates.append((current_date.year, current_date.month))

        next_month = current_date.month % 12 + 1
        next_year = current_date.year + (1 if current_date.month == 12 else 0)

        current_date = current_date.replace(
            year=next_year,
            month=next_month,
            day=1
        )

    return monthly_dates


# ============================================================
# 2. 读取物候数据
# ============================================================

def load_crop_cache(crop_nc, crop_vars):
    """
    一次性读取物候数据。
    """
    print("正在读取物候数据...")

    crop_ds = xr.open_dataset(crop_nc)

    try:
        if "latitude" not in crop_ds.coords or "longitude" not in crop_ds.coords:
            raise ValueError("物候文件中未找到 latitude / longitude 坐标")

        crop_lat = crop_ds["latitude"].values
        crop_lon = crop_ds["longitude"].values

        crop_arrays = {}

        for var in crop_vars:
            if var not in crop_ds:
                raise ValueError(f"物候变量 {var} 不在文件中")

            da = crop_ds[var]

            if "time" in da.dims:
                da = da.isel(time=0)

            raw = da.values
            units = da.attrs.get("units", "").lower()

            if "nanosecond" in units or units == "":
                arr = raw / (1e9 * 86400)
            elif "second" in units:
                arr = raw / 86400
            elif "day of year" in units or "day" in units:
                arr = raw
            else:
                raise ValueError(f"{var} 未知单位：{units}")

            crop_arrays[var] = arr.astype(np.float32)

        print(f"物候数据读取完成：{len(crop_arrays)} 个变量")

        return {
            "lat": crop_lat,
            "lon": crop_lon,
            "arrays": crop_arrays
        }

    finally:
        crop_ds.close()


def extract_crop_days_fast(crop_cache, lat_value, lon_value, crop_vars):
    """
    从物候缓存中提取单个格点的所有物候变量。
    """
    crop_lat = crop_cache["lat"]
    crop_lon = crop_cache["lon"]

    target_lon = convert_lon_for_dataset(lon_value, crop_lon)

    lat_i = nearest_index(crop_lat, lat_value)
    lon_i = nearest_index(crop_lon, target_lon)

    result = {}

    for var in crop_vars:
        value = crop_cache["arrays"][var][lat_i, lon_i]
        result[var] = float(value)

    return result


def get_phenology_periods(point_result):
    """
    根据物候天数得到三个物候阶段：
    1. 种植期
    2. 生长期
    3. 收获期
    """
    plant_start_month = days_to_month_cross(
        point_result["plant.start"],
        0,
        0,
        base_year,
        start=True
    )

    plant_end_month = days_to_month_cross(
        point_result["plant.end"],
        point_result["plant.start"],
        point_result["plant.start"],
        base_year,
        start=False
    )

    growth_start_month = days_to_month_cross(
        point_result["plant"],
        point_result["plant.start"],
        0,
        base_year,
        start=True
    )

    growth_end_month = days_to_month_cross(
        point_result["harvest"],
        point_result["plant"],
        point_result["tot.days"],
        base_year,
        start=False
    )

    harvest_start_month = days_to_month_cross(
        point_result["harvest.start"],
        point_result["plant.start"],
        0,
        base_year,
        start=True
    )

    harvest_end_month = days_to_month_cross(
        point_result["harvest.end"],
        point_result["plant.start"],
        point_result["plant.start"],
        base_year,
        start=False
    )

    return {
        "种植期": (plant_start_month, plant_end_month),
        "生长期": (growth_start_month, growth_end_month),
        "收获期": (harvest_start_month, harvest_end_month)
    }


# ============================================================
# 3. 读取 LDFAI 数据
# ============================================================

def infer_coord_names_for_index_file(ds, target_var):
    """
    自动识别 LDFAI 文件中的时间、纬度、经度维度名。
    """
    if target_var not in ds:
        raise ValueError(
            f"变量 {target_var} 不在文件中，当前变量有：{list(ds.data_vars)}"
        )

    da = ds[target_var]
    dims = list(da.dims)
    all_names = set(list(ds.coords) + list(ds.dims))

    lat_candidates = ["lat", "latitude", "Lat", "LAT", "y"]
    lon_candidates = ["lon", "longitude", "Lon", "LON", "x"]
    time_candidates = ["valid_time", "time", "Time", "date", "month"]

    lat_name = next((name for name in lat_candidates if name in all_names), None)
    lon_name = next((name for name in lon_candidates if name in all_names), None)
    time_name = next((name for name in time_candidates if name in all_names), None)

    if lat_name is None:
        lat_name = next((d for d in dims if "lat" in d.lower()), None)

    if lon_name is None:
        lon_name = next((d for d in dims if "lon" in d.lower()), None)

    if time_name is None:
        non_space_dims = [d for d in dims if d not in {lat_name, lon_name}]
        if len(non_space_dims) >= 1:
            time_name = non_space_dims[0]

    if lat_name is None or lon_name is None or time_name is None:
        raise ValueError(
            f"无法识别时间/纬度/经度维度。dims={dims}, coords={list(ds.coords)}"
        )

    return time_name, lat_name, lon_name


def parse_year_month_from_time_values(time_values):
    """
    从 LDFAI 时间坐标中解析 year/month。
    支持：
    1. datetime64；
    2. cftime；
    3. 198101、198102 这类整数年月。
    """
    time_values = np.asarray(time_values)

    if np.issubdtype(time_values.dtype, np.number):
        vals = time_values.astype(int)

        if np.nanmin(vals) > 100000 and np.nanmax(vals) < 300000:
            years = vals // 100
            months = vals % 100
            return years.astype(int), months.astype(int)

        if np.nanmin(vals) > 1000 and np.nanmax(vals) < 3000:
            years = vals.astype(int)
            months = np.ones_like(years, dtype=int)
            return years, months

    try:
        times = pd.to_datetime(time_values)
        years = times.year.astype(int)
        months = times.month.astype(int)
        return years, months
    except Exception:
        pass

    try:
        years = np.array([int(t.year) for t in time_values], dtype=int)
        months = np.array([int(t.month) for t in time_values], dtype=int)
        return years, months
    except Exception:
        pass

    raise ValueError(f"无法从时间坐标解析 year/month，示例：{time_values[:5]}")


def load_ldfai_cache(target_nc_file, target_var):
    """
    一次性读取 LDFAI 数据。
    统一为 [time, lat, lon]。
    """
    print(f"正在读取 LDFAI 数据：{target_nc_file}")

    ds = xr.open_dataset(target_nc_file)

    try:
        time_name, lat_name, lon_name = infer_coord_names_for_index_file(
            ds,
            target_var
        )

        rename_dict = {}

        if time_name != "time_index":
            rename_dict[time_name] = "time_index"

        if lat_name != "lat":
            rename_dict[lat_name] = "lat"

        if lon_name != "lon":
            rename_dict[lon_name] = "lon"

        ds2 = ds.rename(rename_dict)

        da = ds2[target_var]

        required_dims = ["time_index", "lat", "lon"]
        missing_dims = [d for d in required_dims if d not in da.dims]

        if missing_dims:
            raise ValueError(
                f"变量 {target_var} 缺少维度 {missing_dims}，实际 dims={da.dims}"
            )

        da = da.transpose("time_index", "lat", "lon")

        years, months = parse_year_month_from_time_values(
            ds2["time_index"].values
        )

        time_lookup = {}

        for i, (yy, mm) in enumerate(zip(years, months)):
            time_lookup[(int(yy), int(mm))] = int(i)

        data = da.values.astype(np.float32)

        print(
            f"LDFAI 读取完成：shape={data.shape}, "
            f"年份范围={years.min()}~{years.max()}"
        )

        return {
            "lat": ds2["lat"].values,
            "lon": ds2["lon"].values,
            "years": years.astype(int),
            "months": months.astype(int),
            "time_lookup": time_lookup,
            "data": data
        }

    finally:
        ds.close()


def get_region_indices_1d(coord, lower, upper):
    """
    获取一维坐标中位于 [lower, upper] 的索引。
    """
    coord = np.asarray(coord, dtype=float)
    low, high = sorted([float(lower), float(upper)])
    return np.where((coord >= low) & (coord <= high))[0]


def get_lon_indices(lon_array, lon_min, lon_max):
    """
    获取经度索引，兼容 0~360 和 -180~180。
    """
    lon = np.asarray(lon_array, dtype=float)

    lon_min = float(lon_min)
    lon_max = float(lon_max)

    if np.nanmax(lon) > 180:
        lon_min_c = lon_min % 360
        lon_max_c = lon_max % 360

        if lon_min_c <= lon_max_c:
            return np.where((lon >= lon_min_c) & (lon <= lon_max_c))[0]
        else:
            return np.where((lon >= lon_min_c) | (lon <= lon_max_c))[0]

    def to_minus180_180(x):
        return ((x + 180) % 360) - 180

    lon_min_c = to_minus180_180(lon_min)
    lon_max_c = to_minus180_180(lon_max)

    if lon_min_c <= lon_max_c:
        return np.where((lon >= lon_min_c) & (lon <= lon_max_c))[0]
    else:
        return np.where((lon >= lon_min_c) | (lon <= lon_max_c))[0]


def mean_ldfai_for_month_with_indices(
    ldfai_cache,
    year,
    month,
    lat_indices,
    lon_indices
):
    """
    已知空间索引时，提取某年某月 LDFAI 区域平均值。
    """
    t_i = ldfai_cache["time_lookup"].get((int(year), int(month)))

    if t_i is None:
        return np.nan

    if len(lat_indices) == 0 or len(lon_indices) == 0:
        return np.nan

    region = ldfai_cache["data"][t_i, :, :][np.ix_(lat_indices, lon_indices)]

    if np.all(np.isnan(region)):
        return np.nan

    return float(np.nanmean(region))


def extract_ldfai_records_for_period(
    ldfai_cache,
    base_year_value,
    period_start,
    period_end,
    lat_indices,
    lon_indices,
    skip_first_month=True
):
    """
    提取某格点区域、某一年、某物候期内的所有月尺度 LDFAI 记录。

    返回：
        list of dict:
        [
            {"ldfai_year": 1998, "ldfai_month": 6, "ldfai_value": 0.123},
            ...
        ]
    """
    start_month, start_cross = period_start
    end_month, end_cross = period_end

    if (
        pd.isna(start_month) or pd.isna(start_cross)
        or pd.isna(end_month) or pd.isna(end_cross)
    ):
        return []

    start_year = int(base_year_value) + int(start_cross)
    end_year = int(base_year_value) + int(end_cross)

    ym_list = generate_monthly_dates(
        start_year,
        int(start_month),
        end_year,
        int(end_month)
    )

    if skip_first_month:
        ym_list = ym_list[1:]

    records = []

    for yy, mm in ym_list:
        value = mean_ldfai_for_month_with_indices(
            ldfai_cache=ldfai_cache,
            year=yy,
            month=mm,
            lat_indices=lat_indices,
            lon_indices=lon_indices
        )

        if np.isfinite(value):
            records.append({
                "ldfai_year": int(yy),
                "ldfai_month": int(mm),
                "ldfai_value": float(value)
            })

    return records


# ============================================================
# 4. 趋势拟合与残差计算
# ============================================================

def gaussian_trend(t, A, mu, sigma, B):
    return A * np.exp(-(t - mu) ** 2 / (2 * sigma ** 2)) + B


def get_gaussian_initial_params_and_bounds(t_data, y_data):
    t_data = np.asarray(t_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    t_min, t_max = np.min(t_data), np.max(t_data)
    y_min, y_max = np.min(y_data), np.max(y_data)

    eps = 1e-8
    t_range = max(t_max - t_min, 1)
    y_range = max(y_max - y_min, eps)

    A_guess = y_range
    mu_guess = np.mean(t_data)
    sigma_guess = max(t_range / 4, eps)
    B_guess = y_min

    p0 = [A_guess, mu_guess, sigma_guess, B_guess]

    lower_bounds = [eps, t_min, eps, y_min - y_range]
    upper_bounds = [y_range * 3 + eps, t_max, t_range * 2 + eps, y_max + y_range]

    return p0, (lower_bounds, upper_bounds)


def calc_fit_metrics(y_valid, y_fit, k, model_name, params):
    residuals = y_valid - y_fit
    n = len(y_valid)

    rss = float(np.sum(residuals ** 2))
    rss_safe = max(rss, 1e-12)

    rmse = float(np.sqrt(rss / n))
    aic = float(n * np.log(rss_safe / n) + 2 * k)
    bic = float(n * np.log(rss_safe / n) + k * np.log(n))

    return {
        "model": model_name,
        "params": np.asarray(params, dtype=float),
        "n": n,
        "k": k,
        "rss": rss,
        "rmse": rmse,
        "aic": aic,
        "bic": bic
    }


def fit_one_model_fast(model_name, t_data, y_data):
    t_data = np.asarray(t_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    valid_mask = np.isfinite(t_data) & np.isfinite(y_data)
    t_valid = t_data[valid_mask]
    y_valid = y_data[valid_mask]

    if len(y_valid) < 6:
        return None

    try:
        if model_name == "linear":
            coeff = np.polyfit(t_valid, y_valid, deg=1)
            y_fit = np.polyval(coeff, t_valid)
            return calc_fit_metrics(y_valid, y_fit, 2, model_name, coeff)

        if model_name == "quadratic":
            coeff = np.polyfit(t_valid, y_valid, deg=2)
            y_fit = np.polyval(coeff, t_valid)
            return calc_fit_metrics(y_valid, y_fit, 3, model_name, coeff)

        if model_name == "cubic":
            coeff = np.polyfit(t_valid, y_valid, deg=3)
            y_fit = np.polyval(coeff, t_valid)
            return calc_fit_metrics(y_valid, y_fit, 4, model_name, coeff)

        if model_name == "log":
            X = np.column_stack([np.log1p(t_valid), np.ones_like(t_valid)])
            params, _, _, _ = np.linalg.lstsq(X, y_valid, rcond=None)
            y_fit = X @ params
            return calc_fit_metrics(y_valid, y_fit, 2, model_name, params)

        if model_name == "gaussian":
            p0, bounds = get_gaussian_initial_params_and_bounds(
                t_valid,
                y_valid
            )

            popt, _ = curve_fit(
                f=gaussian_trend,
                xdata=t_valid,
                ydata=y_valid,
                p0=p0,
                bounds=bounds,
                maxfev=10000
            )

            y_fit = gaussian_trend(t_valid, *popt)
            return calc_fit_metrics(y_valid, y_fit, 4, model_name, popt)

        raise ValueError(f"未知趋势模型：{model_name}")

    except Exception:
        return None


def predict_trend(t, trend_result):
    model_name = trend_result["model"]
    params = trend_result["params"]

    if model_name in {"linear", "quadratic", "cubic"}:
        return np.polyval(params, t)

    if model_name == "log":
        a, b = params
        return a * np.log1p(t) + b

    if model_name == "gaussian":
        return gaussian_trend(t, *params)

    raise ValueError(f"未知趋势模型：{model_name}")


def fit_multiple_models_with_outlier_removal_fast(
    t_data,
    y_data,
    max_iter=3,
    z_threshold=2.0,
    criterion="bic",
    candidate_model_names=None
):
    if criterion not in {"aic", "bic", "rmse"}:
        raise ValueError("criterion 必须是 aic、bic 或 rmse")

    if candidate_model_names is None:
        candidate_model_names = ["linear"]

    t_data = np.asarray(t_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    mask_global = np.isfinite(t_data) & np.isfinite(y_data)

    best_result = None

    for _ in range(max_iter):
        t_current = t_data[mask_global]
        y_current = y_data[mask_global]

        if len(y_current) < 10:
            break

        all_results = {}

        for model_name in candidate_model_names:
            result = fit_one_model_fast(
                model_name,
                t_current,
                y_current
            )

            if result is not None:
                all_results[model_name] = result

        if len(all_results) == 0:
            return None

        best_model_name = min(
            all_results,
            key=lambda name: all_results[name][criterion]
        )

        best_result = all_results[best_model_name]

        y_fit = predict_trend(t_current, best_result)
        residuals = y_current - y_fit

        res_std = np.std(residuals)

        if not np.isfinite(res_std) or res_std < 1e-8:
            break

        z_scores = np.abs((residuals - np.mean(residuals)) / res_std)
        outlier_mask = z_scores > z_threshold

        if not np.any(outlier_mask):
            break

        mask_indices = np.where(mask_global)[0]
        mask_global[mask_indices[outlier_mask]] = False

    return best_result


# ============================================================
# 5. 构建考虑正负号的 LDFAI 特征
# ============================================================

def build_signed_ldfai_features(values):
    """
    将某一年、某格点、某物候期内的所有月尺度 LDFAI 值
    聚合成年尺度特征。

    本版本只保留 5 个特征：
    1. ldfai_raw_max
    2. ldfai_raw_min
    3. ldfai_abs_sum
    4. ldfai_pos_event_max
    5. ldfai_neg_event_max
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {}

    abs_values = np.abs(values)
    pos_values = values[values > 0]
    neg_values = values[values < 0]
    neg_intensity = np.abs(neg_values)

    features = {
        # 原始 LDFAI 最大值：保留正负号，通常代表最强正向急转。
        "ldfai_raw_max": float(np.nanmax(values)),

        # 原始 LDFAI 最小值：保留正负号，通常代表最强负向急转。
        "ldfai_raw_min": float(np.nanmin(values)),

        # 物候期内 abs(LDFAI) 累计强度。
        "ldfai_abs_sum": float(np.nansum(abs_values)),

        # 正向事件最大值；没有正向事件则为 NaN。
        "ldfai_pos_event_max": (
            float(np.nanmax(pos_values)) if len(pos_values) > 0 else np.nan
        ),

        # 负向事件最大强度；负值取绝对值后计算，没有负向事件则为 NaN。
        "ldfai_neg_event_max": (
            float(np.nanmax(neg_intensity)) if len(neg_values) > 0 else np.nan
        ),
    }

    return features

# ============================================================
# 6. 相关性累计器
# ============================================================

class CorrAccumulator:
    """
    在线累计 Pearson 相关。
    如果 calc_spearman=True，则额外保存样本值用于 Spearman。
    """

    def __init__(self, keep_values=False):
        self.n = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0

        self.keep_values = keep_values
        self.x_values = [] if keep_values else None
        self.y_values = [] if keep_values else None

    def update(self, x, y):
        if not np.isfinite(x) or not np.isfinite(y):
            return

        x = float(x)
        y = float(y)

        self.n += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_x2 += x * x
        self.sum_y2 += y * y
        self.sum_xy += x * y

        if self.keep_values:
            self.x_values.append(x)
            self.y_values.append(y)

    def pearson(self):
        if self.n < 3:
            return np.nan, np.nan

        n = self.n

        numerator = n * self.sum_xy - self.sum_x * self.sum_y
        denominator_x = n * self.sum_x2 - self.sum_x ** 2
        denominator_y = n * self.sum_y2 - self.sum_y ** 2

        denominator = np.sqrt(denominator_x * denominator_y)

        if denominator <= 0 or not np.isfinite(denominator):
            return np.nan, np.nan

        r = numerator / denominator
        r = float(np.clip(r, -1.0, 1.0))

        if abs(r) >= 1:
            p = 0.0
        else:
            t_stat = r * np.sqrt((n - 2) / max(1e-12, 1 - r ** 2))
            p = float(2 * student_t.sf(abs(t_stat), df=n - 2))

        return r, p

    def spearman(self):
        if not self.keep_values or self.n < 3:
            return np.nan, np.nan

        x_rank = pd.Series(self.x_values).rank(method="average").values
        y_rank = pd.Series(self.y_values).rank(method="average").values

        temp = CorrAccumulator(keep_values=False)

        for x, y in zip(x_rank, y_rank):
            temp.update(x, y)

        return temp.pearson()

    def mean_x(self):
        if self.n == 0:
            return np.nan
        return self.sum_x / self.n

    def mean_y(self):
        if self.n == 0:
            return np.nan
        return self.sum_y / self.n


def make_group_key(index_type, run_name, phenology, feature_name):
    return (
        str(index_type),
        str(run_name),
        str(phenology),
        str(feature_name)
    )


def update_corr_dict(
    corr_dict,
    index_type,
    run_name,
    phenology,
    feature_name,
    feature_value,
    residual_value
):
    key = make_group_key(
        index_type=index_type,
        run_name=run_name,
        phenology=phenology,
        feature_name=feature_name
    )

    if key not in corr_dict:
        corr_dict[key] = CorrAccumulator(keep_values=calc_spearman)

    corr_dict[key].update(feature_value, residual_value)



# ============================================================
# 6.1 双情景 common sample 辅助函数
# ============================================================

def get_month_keys_from_records(ldfai_records):
    """提取 LDFAI 记录对应的年月键。"""
    return tuple(
        (int(rec["ldfai_year"]), int(rec["ldfai_month"]))
        for rec in ldfai_records
    )


def get_values_from_records(ldfai_records):
    """提取 LDFAI 记录中的数值数组。"""
    return np.array(
        [rec["ldfai_value"] for rec in ldfai_records],
        dtype=float
    )


def summarize_corr_dict(corr_dict):
    """将某个情景下的相关性累计器汇总为 DataFrame。"""
    rows = []

    for key, acc in corr_dict.items():
        index_type, run_name, phenology, feature_name = key

        pearson_r, pearson_p = acc.pearson()
        spearman_rho, spearman_p = acc.spearman()

        rows.append({
            "index_type": index_type,
            "run_name": run_name,
            "phenology": phenology,
            "feature_name": feature_name,
            "n": acc.n,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_rho": spearman_rho,
            "spearman_p": spearman_p,
            "mean_feature": acc.mean_x(),
            "mean_residual": acc.mean_y()
        })

    summary_df = pd.DataFrame(rows)

    if len(summary_df) == 0:
        return summary_df

    return summary_df.sort_values(
        ["run_name", "phenology", "feature_name", "index_type"]
    )


def make_spi_spei_compare_df(summary_df):
    """根据 summary_df 生成 SPI vs SPEI 对比表。"""
    if len(summary_df) == 0:
        return pd.DataFrame()

    compare_base_cols = [
        "run_name",
        "phenology",
        "feature_name"
    ]

    pivot = summary_df.pivot_table(
        index=compare_base_cols,
        columns="index_type",
        values=[
            "n",
            "pearson_r",
            "pearson_p",
            "spearman_rho",
            "spearman_p"
        ],
        aggfunc="first"
    )

    pivot.columns = [
        f"{metric}_{index_type}"
        for metric, index_type in pivot.columns
    ]

    compare_df = pivot.reset_index()

    if "pearson_r_SPI" in compare_df.columns and "pearson_r_SPEI" in compare_df.columns:
        compare_df["delta_abs_pearson_SPEI_minus_SPI"] = (
            compare_df["pearson_r_SPEI"].abs()
            - compare_df["pearson_r_SPI"].abs()
        )

    if "spearman_rho_SPI" in compare_df.columns and "spearman_rho_SPEI" in compare_df.columns:
        compare_df["delta_abs_spearman_SPEI_minus_SPI"] = (
            compare_df["spearman_rho_SPEI"].abs()
            - compare_df["spearman_rho_SPI"].abs()
        )

    return compare_df


def save_scenario_outputs(
    scenario_name,
    scenario_cfg,
    corr_dict,
    pair_records,
    save_pair_table=True
):
    """保存某个情景的 summary、compare、pairs、count_check。"""
    summary_df = summarize_corr_dict(corr_dict)

    if len(summary_df) == 0:
        print(f"[{scenario_name}] 没有有效相关性结果，跳过输出。")
        return

    summary_df.to_csv(
        scenario_cfg["summary_csv"],
        index=False,
        encoding="utf-8-sig"
    )
    print(f"[{scenario_name}] 相关性结果已保存：{scenario_cfg['summary_csv']}")

    compare_df = make_spi_spei_compare_df(summary_df)
    compare_df.to_csv(
        scenario_cfg["compare_csv"],
        index=False,
        encoding="utf-8-sig"
    )
    print(f"[{scenario_name}] SPI-SPEI 对比结果已保存：{scenario_cfg['compare_csv']}")

    if save_pair_table:
        pair_df = pd.DataFrame(pair_records)
        pair_df.to_csv(
            scenario_cfg["pair_csv"],
            index=False,
            encoding="utf-8-sig"
        )
        print(f"[{scenario_name}] 配对样本已保存：{scenario_cfg['pair_csv']}")

        if len(pair_df) > 0:
            check_counts = (
                pair_df
                .groupby(["run_name", "phenology", "feature_name", "index_type"])
                .size()
                .unstack("index_type")
                .reset_index()
            )

            if "SPI" not in check_counts.columns:
                check_counts["SPI"] = 0
            if "SPEI" not in check_counts.columns:
                check_counts["SPEI"] = 0

            check_counts["n_equal_SPI_SPEI"] = (
                check_counts["SPI"] == check_counts["SPEI"]
            )

            check_counts.to_csv(
                scenario_cfg["count_check_csv"],
                index=False,
                encoding="utf-8-sig"
            )
            print(f"[{scenario_name}] SPI/SPEI 数量检查已保存：{scenario_cfg['count_check_csv']}")

            if not check_counts["n_equal_SPI_SPEI"].all():
                print(f"[{scenario_name}] 警告：仍有 feature 的 SPI/SPEI 样本数不一致。")
            else:
                print(f"[{scenario_name}] 检查通过：所有 feature 的 SPI/SPEI 样本数一致。")


def add_common_sample_to_scenario(
    scenario_name,
    corr_dict,
    pair_records,
    run_name,
    center_lat,
    center_lon,
    base_year_value,
    phenology,
    selected_month_keys,
    values_by_index,
    residual_value,
    residual_z,
    extra_info=None
):
    """
    将同一个格点-年份-物候期样本同时写入 SPI 与 SPEI。

    只有某个 feature 在 SPI 和 SPEI 两边都为有限值时，才会同时更新，
    因而可以保证每个 feature 的 n_SPI == n_SPEI。
    """
    extra_info = extra_info or {}

    if "SPI" not in values_by_index or "SPEI" not in values_by_index:
        return 0

    features_by_index = {}

    for index_type in ["SPI", "SPEI"]:
        values = np.asarray(values_by_index[index_type], dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return 0

        features = build_signed_ldfai_features(values)

        if len(features) == 0:
            return 0

        features_by_index[index_type] = features

    common_feature_names = sorted(
        set(features_by_index["SPI"].keys())
        & set(features_by_index["SPEI"].keys())
    )

    n_added_features = 0
    selected_month_text = ";".join(
        [f"{yy}-{mm:02d}" for yy, mm in selected_month_keys]
    )

    for feature_name in common_feature_names:
        feature_value_spi = features_by_index["SPI"].get(feature_name, np.nan)
        feature_value_spei = features_by_index["SPEI"].get(feature_name, np.nan)

        if not np.isfinite(feature_value_spi) or not np.isfinite(feature_value_spei):
            continue

        # 同一 feature 同时写入 SPI 和 SPEI，保证两边 n 完全一致。
        update_corr_dict(
            corr_dict=corr_dict,
            index_type="SPI",
            run_name=run_name,
            phenology=phenology,
            feature_name=feature_name,
            feature_value=feature_value_spi,
            residual_value=residual_value
        )

        update_corr_dict(
            corr_dict=corr_dict,
            index_type="SPEI",
            run_name=run_name,
            phenology=phenology,
            feature_name=feature_name,
            feature_value=feature_value_spei,
            residual_value=residual_value
        )

        n_added_features += 1

        if pair_records is not None:
            base_record = {
                "scenario": scenario_name,
                "run_name": run_name,
                "lat": round(center_lat, 4),
                "lon": round(center_lon, 4),
                "year": int(base_year_value),
                "phenology": phenology,
                "feature_name": feature_name,
                "residual": float(residual_value),
                "yield_loss": float(-residual_value),
                "residual_z": float(residual_z) if np.isfinite(residual_z) else np.nan,
                "selected_ldfai_months": selected_month_text,
                "n_selected_months": int(len(selected_month_keys)),
                "matched_spi_spei": 1,
            }
            base_record.update(extra_info)

            rec_spi = dict(base_record)
            rec_spi.update({
                "index_type": "SPI",
                "feature_value": float(feature_value_spi)
            })
            pair_records.append(rec_spi)

            rec_spei = dict(base_record)
            rec_spei.update({
                "index_type": "SPEI",
                "feature_value": float(feature_value_spei)
            })
            pair_records.append(rec_spei)

    return n_added_features


# ============================================================
# 6.2 提速辅助函数：预提取当前格点所有年份/物候期 LDFAI
# ============================================================

def precompute_common_ldfai_for_point(
    years,
    phenology_periods,
    ldfai_caches,
    ldfai_spatial_indices,
):
    """
    对同一个产量格点，提前提取所有年份 × 物候期的 SPI/SPEI LDFAI。

    这些 LDFAI 记录只依赖格点、年份和物候期，与趋势模型 run_name 无关。
    原脚本在每个 model_run 里重复提取一次；这里改为每个格点只提取一次，
    后续 linear/quadratic 等趋势方案共享同一份 LDFAI 缓存。

    返回：
        cache: dict，key=(year_pos, phenology)，value 包含月份、SPI/SPEI 值、事件得分等；
        skipped_common: 因 SPI/SPEI 缺失而跳过的 year-phenology 次数；
        skipped_mismatch: 因 SPI/SPEI 月份不一致而跳过的 year-phenology 次数。
    """
    cache = {}
    skipped_common = 0
    skipped_mismatch = 0

    for year_pos, base_year_value in enumerate(years):
        base_year_value = int(base_year_value)

        for phenology, (period_start, period_end) in phenology_periods.items():
            values_by_index = {}
            month_keys_by_index = {}

            for index_type in ["SPI", "SPEI"]:
                ldfai_cache = ldfai_caches[index_type]
                ldfai_lat_indices, ldfai_lon_indices = ldfai_spatial_indices[index_type]

                ldfai_records = extract_ldfai_records_for_period(
                    ldfai_cache=ldfai_cache,
                    base_year_value=base_year_value,
                    period_start=period_start,
                    period_end=period_end,
                    lat_indices=ldfai_lat_indices,
                    lon_indices=ldfai_lon_indices,
                    skip_first_month=skip_first_ldfai_month
                )

                if len(ldfai_records) < min_ldfai_months_required:
                    continue

                values_by_index[index_type] = get_values_from_records(ldfai_records)
                month_keys_by_index[index_type] = get_month_keys_from_records(ldfai_records)

            if "SPI" not in values_by_index or "SPEI" not in values_by_index:
                skipped_common += 1
                continue

            if require_same_ldfai_months and (
                month_keys_by_index["SPI"] != month_keys_by_index["SPEI"]
            ):
                skipped_mismatch += 1
                continue

            spi_values_all = np.asarray(values_by_index["SPI"], dtype=float)
            spei_values_all = np.asarray(values_by_index["SPEI"], dtype=float)

            valid_pair_mask = np.isfinite(spi_values_all) & np.isfinite(spei_values_all)
            if not np.any(valid_pair_mask):
                skipped_common += 1
                continue

            abs_spi = np.abs(spi_values_all)
            abs_spei = np.abs(spei_values_all)

            if event_pair_rule == "either":
                # 任一指数超过阈值即可，因此事件得分取两者绝对值最大值。
                event_score = np.maximum(abs_spi, abs_spei)
            elif event_pair_rule == "both":
                # 两个指数同月都超过阈值，因此事件得分取两者绝对值最小值。
                event_score = np.minimum(abs_spi, abs_spei)
            else:
                raise ValueError("event_pair_rule 必须是 'either' 或 'both'")

            cache[(int(year_pos), str(phenology))] = {
                "base_year_value": base_year_value,
                "month_keys_common": month_keys_by_index["SPI"],
                "spi_values_all": spi_values_all,
                "spei_values_all": spei_values_all,
                "valid_pair_mask": valid_pair_mask,
                "event_score": event_score,
            }

    return cache, skipped_common, skipped_mismatch

# ============================================================
# 7. 调试输出函数：随机打印配对样本
# ============================================================

def print_debug_pair_sample(
    debug_pair_printed,
    lat_idx,
    lon_idx,
    center_lat,
    center_lon,
    min_lat_cell,
    max_lat_cell,
    min_lon_cell,
    max_lon_cell,
    ldfai_lat_indices,
    ldfai_lon_indices,
    base_year_value,
    year_pos,
    years,
    phenology,
    period_start,
    period_end,
    ldfai_records,
    features,
    ts_values,
    fitted_values,
    residual_value,
    best_trend,
    index_type,
    run_name
):
    """
    打印一个完整 LDFAI-残差配对样本。
    """
    ldfai_month_text = ", ".join([
        f"{rec['ldfai_year']}-{rec['ldfai_month']:02d}:{rec['ldfai_value']:.4f}"
        for rec in ldfai_records
    ])

    print("\n" + "=" * 90)
    print(f"随机配对样本 #{debug_pair_printed}")
    print("=" * 90)

    print("[1] 空间位置")
    print(f"  lat_idx = {lat_idx}, lon_idx = {lon_idx}")
    print(f"  center_lat = {center_lat:.6f}, center_lon = {center_lon:.6f}")
    print(f"  LDFAI lat索引数 = {len(ldfai_lat_indices)}, lon索引数 = {len(ldfai_lon_indices)}")
    print(f"  LDFAI 匹配区域 lat = {min_lat_cell:.6f} ~ {max_lat_cell:.6f}")
    print(f"  LDFAI 匹配区域 lon = {min_lon_cell:.6f} ~ {max_lon_cell:.6f}")

    print("[2] 年份配对")
    print(f"  产量年份 base_year = {base_year_value}")
    print(f"  year_pos = {year_pos}")
    print(f"  years[year_pos] = {years[year_pos]}")
    print("  说明：该年的产量残差与该年物候期内的 LDFAI 聚合值配对。")

    print("[3] 物候阶段")
    print(f"  phenology = {phenology}")
    print(f"  period_start = {period_start}  # (月份, 是否跨年)")
    print(f"  period_end   = {period_end}    # (月份, 是否跨年)")
    print(f"  skip_first_ldfai_month = {skip_first_ldfai_month}")

    print("[4] 该物候期内参与聚合的 LDFAI 月值")
    print(f"  {ldfai_month_text}")

    print("[5] 聚合后的 LDFAI 特征")
    print(f"  ldfai_raw_mean = {features.get('ldfai_raw_mean', np.nan):.6f}")
    print(f"  ldfai_abs_mean = {features.get('ldfai_abs_mean', np.nan):.6f}")
    print(f"  ldfai_abs_max  = {features.get('ldfai_abs_max', np.nan):.6f}")
    print(f"  ldfai_abs_sum  = {features.get('ldfai_abs_sum', np.nan):.6f}")
    print(f"  ldfai_pos_sum  = {features.get('ldfai_pos_sum', np.nan):.6f}")
    print(f"  ldfai_neg_sum  = {features.get('ldfai_neg_sum', np.nan):.6f}")
    print(f"  ldfai_valid_months = {features.get('ldfai_valid_months', np.nan):.0f}")

    print("[6] 产量与残差")
    print(f"  actual_yield = {ts_values[year_pos]:.6f}")
    print(f"  fitted_yield = {fitted_values[year_pos]:.6f}")
    print(f"  residual = actual_yield - fitted_yield = {residual_value:.6f}")
    print(f"  trend_model = {best_trend['model']}")
    print(f"  AIC = {best_trend['aic']:.6f}, BIC = {best_trend['bic']:.6f}, RMSE = {best_trend['rmse']:.6f}")

    print("[7] 指数类型")
    print(f"  index_type = {index_type}")
    print(f"  run_name = {run_name}")

    print("[8] 配对检查结论")
    print("  检查 base_year 是否等于 years[year_pos]；")
    print("  检查 LDFAI 月份是否落在该物候期内；")
    print("  检查 residual 是否等于 actual_yield - fitted_yield。")
    print("=" * 90 + "\n")


# ============================================================
# 8. 主程序
# ============================================================


def run_signed_phenology_correlation():
    print("=" * 90)
    print("开始：批量生成强旱涝急转事件 common sample 分析（提速版）")
    print(f"FAST_MODE={FAST_MODE}, calc_spearman={calc_spearman}, save_pair_table={save_pair_table}, debug_pair_sample={debug_pair_sample}")
    print(f"事件阈值范围：{event_ldfai_abs_thresholds}")
    print(f"事件筛选规则：event_pair_rule={event_pair_rule}")
    print("=" * 90)

    # --------------------------------------------------------
    # 读取产量数据
    # --------------------------------------------------------
    actual_nc_file = resolve_nc_file(nc_file)
    print(f"使用产量文件：{actual_nc_file}")

    ds_yield = xr.open_dataset(actual_nc_file)

    try:
        lat = ds_yield[lat_var].values
        lon = ds_yield[lon_var].values

        yield_da_annual, years = prepare_annual_yield_data(
            ds_yield,
            data_var=data_var,
            time_var=time_var
        )

        data = yield_da_annual.values.astype(np.float32)

    finally:
        ds_yield.close()

    n_time, n_lat, n_lon = data.shape
    t_all = np.arange(len(years), dtype=float)

    lat_res = np.abs(lat[1] - lat[0]) if n_lat > 1 else 0.0
    lon_res = np.abs(lon[1] - lon[0]) if n_lon > 1 else 0.0

    # --------------------------------------------------------
    # 读取物候数据
    # --------------------------------------------------------
    crop_cache = load_crop_cache(crop_nc, crop_vars)

    # --------------------------------------------------------
    # 读取 SPI-LDFAI 和 SPEI-LDFAI
    # --------------------------------------------------------
    ldfai_caches = {}

    for index_type, file_path in index_files.items():
        ldfai_caches[index_type] = load_ldfai_cache(
            target_nc_file=file_path,
            target_var=target_var
        )

    if "SPI" not in ldfai_caches or "SPEI" not in ldfai_caches:
        raise ValueError("index_files 中必须同时包含 SPI 和 SPEI。")

    # --------------------------------------------------------
    # 研究区有效格点
    # --------------------------------------------------------
    first_time_data = np.nanmean(data, axis=0)
    non_nan_mask = ~np.isnan(first_time_data)

    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    lon_mask = (lon >= lon_min) & (lon <= lon_max)

    region_mask = non_nan_mask & np.outer(lat_mask, lon_mask)

    non_nan_indices = np.where(region_mask)
    n_points = len(non_nan_indices[0])

    print(f"研究区有效格点数：{n_points}")

    scenario_items = list(analysis_scenarios.items())
    min_event_threshold = min(
        float(cfg["event_ldfai_abs_threshold"])
        for _, cfg in scenario_items
    )

    # 每个情景单独维护累计器与 pair 表。
    corr_dict_by_scenario = {
        scenario_name: {}
        for scenario_name in analysis_scenarios
    }

    pair_records_by_scenario = {
        scenario_name: []
        for scenario_name in analysis_scenarios
    }

    scenario_sample_counter = {
        scenario_name: 0
        for scenario_name in analysis_scenarios
    }

    processed = 0
    skipped_no_yield = 0
    skipped_no_pheno = 0
    skipped_no_trend = 0
    skipped_no_common_ldfai = 0
    skipped_month_mismatch = 0
    debug_pair_printed = 0

    # ========================================================
    # 逐格点处理
    # ========================================================
    for point_idx in range(n_points):
        lat_idx = int(non_nan_indices[0][point_idx])
        lon_idx = int(non_nan_indices[1][point_idx])

        center_lat = float(lat[lat_idx])
        center_lon = float(lon[lon_idx])

        ts_values = data[:, lat_idx, lon_idx].astype(float)

        if np.sum(np.isfinite(ts_values)) < min_valid_points:
            skipped_no_yield += 1
            continue

        # ----------------------------------------------------
        # 提取当前格点物候期
        # ----------------------------------------------------
        try:
            point_result = extract_crop_days_fast(
                crop_cache=crop_cache,
                lat_value=center_lat,
                lon_value=center_lon,
                crop_vars=crop_vars
            )

            if any(np.isnan(point_result[var]) for var in crop_vars):
                skipped_no_pheno += 1
                continue

            phenology_periods = get_phenology_periods(point_result)

        except Exception:
            skipped_no_pheno += 1
            continue

        # ----------------------------------------------------
        # 当前产量格点对应 LDFAI 空间范围
        # ----------------------------------------------------
        min_lon_cell = center_lon - lon_res / 2.0
        max_lon_cell = center_lon + lon_res / 2.0
        min_lat_cell = center_lat - lat_res / 2.0
        max_lat_cell = center_lat + lat_res / 2.0

        # 同一格点上先计算 SPI 与 SPEI 的空间索引。
        ldfai_spatial_indices = {}

        for index_type, ldfai_cache in ldfai_caches.items():
            ldfai_lat_indices = get_region_indices_1d(
                ldfai_cache["lat"],
                min_lat_cell,
                max_lat_cell
            )

            ldfai_lon_indices = get_lon_indices(
                ldfai_cache["lon"],
                min_lon_cell,
                max_lon_cell
            )

            if len(ldfai_lat_indices) > 0 and len(ldfai_lon_indices) > 0:
                ldfai_spatial_indices[index_type] = (
                    ldfai_lat_indices,
                    ldfai_lon_indices
                )

        if "SPI" not in ldfai_spatial_indices or "SPEI" not in ldfai_spatial_indices:
            skipped_no_common_ldfai += 1
            continue

        # ----------------------------------------------------
        # 提速点 1：LDFAI 与趋势模型无关，每个格点只预提取一次。
        # ----------------------------------------------------
        ldfai_common_cache, skipped_common_point, skipped_mismatch_point = (
            precompute_common_ldfai_for_point(
                years=years,
                phenology_periods=phenology_periods,
                ldfai_caches=ldfai_caches,
                ldfai_spatial_indices=ldfai_spatial_indices,
            )
        )
        skipped_no_common_ldfai += skipped_common_point
        skipped_month_mismatch += skipped_mismatch_point

        if len(ldfai_common_cache) == 0:
            continue

        # ====================================================
        # 趋势方案循环
        # ====================================================
        for run_cfg in model_runs:
            run_name = run_cfg["run_name"]
            criterion = run_cfg["criterion"]
            candidate_models = run_cfg["candidate_models"]

            best_trend = fit_multiple_models_with_outlier_removal_fast(
                t_data=t_all,
                y_data=ts_values,
                max_iter=max_outlier_iter,
                z_threshold=z_threshold,
                criterion=criterion,
                candidate_model_names=candidate_models
            )

            if best_trend is None:
                skipped_no_trend += 1
                continue

            # 提速点 2：趋势预测向量化，避免逐年 Python 循环。
            fitted_values = np.asarray(predict_trend(t_all, best_trend), dtype=float)

            # 残差：实际产量 - 拟合产量；负值表示减产。
            residual_values = ts_values - fitted_values

            residual_mean = np.nanmean(residual_values)
            residual_std = np.nanstd(residual_values)

            if np.isfinite(residual_std) and residual_std > 1e-8:
                residual_z_values = (residual_values - residual_mean) / residual_std
            else:
                residual_z_values = np.full_like(residual_values, np.nan, dtype=float)

            # =============================================
            # 使用预提取的年份 × 物候期 LDFAI 缓存
            # =============================================
            for (year_pos, phenology), ldfai_pack in ldfai_common_cache.items():
                residual_value = residual_values[year_pos]
                residual_z = residual_z_values[year_pos]

                if not np.isfinite(residual_value):
                    continue

                event_score = ldfai_pack["event_score"]
                valid_pair_mask = ldfai_pack["valid_pair_mask"]

                # 如果最低阈值都没有超过，后续 1~9 全部不用检查。
                if not np.any(valid_pair_mask & (event_score > min_event_threshold)):
                    continue

                base_year_value = ldfai_pack["base_year_value"]
                month_keys_common = ldfai_pack["month_keys_common"]
                spi_values_all = ldfai_pack["spi_values_all"]
                spei_values_all = ldfai_pack["spei_values_all"]

                # =====================================================
                # 批量事件分析：|LDFAI| > 1, 2, ..., 9 的 common event sample
                # =====================================================
                for scenario_name, scenario_cfg in scenario_items:
                    event_ldfai_abs_threshold = float(
                        scenario_cfg["event_ldfai_abs_threshold"]
                    )

                    event_mask = valid_pair_mask & (event_score > event_ldfai_abs_threshold)

                    if not np.any(event_mask):
                        continue

                    event_indices = np.flatnonzero(event_mask)
                    selected_month_keys_event = tuple(
                        month_keys_common[i]
                        for i in event_indices
                    )

                    selected_values_event = {
                        "SPI": spi_values_all[event_mask],
                        "SPEI": spei_values_all[event_mask]
                    }

                    n_added = add_common_sample_to_scenario(
                        scenario_name=scenario_name,
                        corr_dict=corr_dict_by_scenario[scenario_name],
                        pair_records=(
                            pair_records_by_scenario[scenario_name]
                            if save_pair_table else None
                        ),
                        run_name=run_name,
                        center_lat=center_lat,
                        center_lon=center_lon,
                        base_year_value=base_year_value,
                        phenology=phenology,
                        selected_month_keys=selected_month_keys_event,
                        values_by_index=selected_values_event,
                        residual_value=residual_value,
                        residual_z=residual_z,
                        extra_info={
                            "event_abs_threshold": float(event_ldfai_abs_threshold),
                            "event_pair_rule": event_pair_rule,
                            "sample_filter": "ldfai_abs_gt_threshold"
                        }
                    )

                    if n_added > 0:
                        scenario_sample_counter[scenario_name] += 1

                # 简化版 debug：随机打印一个 common sample 状态。
                if (
                    debug_pair_sample
                    and debug_pair_printed < debug_pair_max_print
                    and debug_rng.random() < debug_pair_probability
                ):
                    debug_pair_printed += 1
                    print("\n" + "=" * 90)
                    print(f"随机 common 配对样本 #{debug_pair_printed}")
                    print(f"lat={center_lat:.4f}, lon={center_lon:.4f}, year={base_year_value}, phenology={phenology}")
                    print(f"residual={residual_value:.6f}, residual_z={residual_z:.6f}")
                    print(f"months={';'.join([f'{yy}-{mm:02d}' for yy, mm in month_keys_common])}")
                    print(f"SPI values={np.array2string(spi_values_all, precision=4)}")
                    print(f"SPEI values={np.array2string(spei_values_all, precision=4)}")
                    print("=" * 90 + "\n")

        processed += 1

        if processed % 500 == 0:
            event_sample_text = ", ".join(
                f"gt{format_threshold_label(scenario_cfg['event_ldfai_abs_threshold'])}={scenario_sample_counter[scenario_name]}"
                for scenario_name, scenario_cfg in scenario_items
            )
            print(
                f"已处理 {processed}/{n_points} 个有效格点；"
                f"跳过 yield={skipped_no_yield}, "
                f"pheno={skipped_no_pheno}, "
                f"trend={skipped_no_trend}, "
                f"common_ldfai={skipped_no_common_ldfai}, "
                f"month_mismatch={skipped_month_mismatch}, "
                f"event_samples={{ {event_sample_text} }}, "
                f"debug_samples={debug_pair_printed}"
            )

    # ========================================================
    # 分情景保存结果
    # ========================================================
    for scenario_name, scenario_cfg in scenario_items:
        save_scenario_outputs(
            scenario_name=scenario_name,
            scenario_cfg=scenario_cfg,
            corr_dict=corr_dict_by_scenario[scenario_name],
            pair_records=pair_records_by_scenario[scenario_name],
            save_pair_table=save_pair_table
        )

    print("=" * 90)
    print("分析完成")
    print("=" * 90)

    print("结果解释：")
    print("1. event_ldfai_abs_gt1 ~ event_ldfai_abs_gt9：分别表示 |LDFAI| > 1 到 > 9 的强旱涝急转事件样本。")
    print(f"   当前 event_pair_rule='{event_pair_rule}'：either 表示 SPI 或 SPEI 任一指数超过阈值；both 表示两者同月都超过阈值。")
    print("2. SPI 与 SPEI 的特征均基于同一批 selected_ldfai_months 构建，因此样本公平可比。")
    print("3. residual = actual_yield - fitted_yield，负残差表示减产；yield_loss = -residual。")
    print("4. 提速版每个格点只提取一次 LDFAI，并共享给所有趋势模型和阈值。")
    print("5. FAST_MODE=True 时不计算 Spearman、不保存 pairs 明细；如需完整输出，可改为 FAST_MODE=False。")
    print("6. delta_abs_pearson_SPEI_minus_SPI > 0 表示 SPEI-LDFAI 的相关性绝对值强于 SPI-LDFAI。")
    print(f"7. 本次运行共随机输出 debug 配对样本数：{debug_pair_printed}")


# ============================================================
# 9. 执行
# ============================================================

if __name__ == "__main__":
    run_signed_phenology_correlation()