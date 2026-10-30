"""
加速版：LDFAI 物候期产量残差计算脚本

"""


import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False


# ============================================================
# 1. 基础工具函数
# ============================================================

def parse_years_from_time(time_values):
    """
    从 NC 文件 time 坐标中解析真实年份。
    支持 datetime64、cftime、pandas 可解析时间，以及本身就是年份的数值坐标。
    """
    time_values = np.asarray(time_values)

    # 情况 1：time 本身就是类似 1981, 1982 的年份数值
    if np.issubdtype(time_values.dtype, np.number):
        years = time_values.astype(int)
        if np.nanmin(years) > 1000 and np.nanmax(years) < 3000:
            return years

    # 情况 2：pandas 可以直接解析
    try:
        return pd.to_datetime(time_values).year.astype(int)
    except Exception:
        pass

    # 情况 3：cftime 对象通常有 year 属性
    try:
        return np.array([int(t.year) for t in time_values], dtype=int)
    except Exception as e:
        raise ValueError(f"无法从 time 坐标解析年份，请检查 time 数据格式：{e}")


def prepare_annual_yield_data(ds, data_var, time_var):
    """
    将产量数据整理为年尺度 DataArray，维度为 [year, lat, lon]。

    关键点：
    1. 不再硬编码年份。
    2. 从 ds[time_var] 读取真实年份。
    3. 如果同一年有多个时间步，则按年平均。
    """
    if data_var not in ds:
        raise ValueError(f"数据变量 {data_var} 不在 NC 文件中")
    if time_var not in ds:
        raise ValueError(f"时间变量 {time_var} 不在 NC 文件中")

    yield_da = ds[data_var]
    if time_var not in yield_da.dims:
        raise ValueError(f"变量 {data_var} 的维度中不包含时间维 {time_var}，实际维度为 {yield_da.dims}")

    yield_years = parse_years_from_time(ds[time_var].values)
    yield_da = yield_da.assign_coords(year=(time_var, yield_years))
    yield_da_annual = yield_da.groupby("year").mean(dim=time_var, skipna=True)
    yield_da_annual = yield_da_annual.sortby("year")

    years = yield_da_annual["year"].values.astype(int)
    print(f"产量数据真实年份范围：{years.min()} ~ {years.max()}，共 {len(years)} 年")
    print(f"产量年尺度数组维度：{yield_da_annual.dims}，shape={yield_da_annual.shape}")

    return yield_da_annual, years


def days_to_month_cross(days, plant_start, day_range, base_year, start=True):
    """
    将物候期天数转换为 (month, is_cross_year)。

    month: 1-12
    is_cross_year: 0 表示本年，1 表示次年
    """
    if pd.isna(days):
        return (np.nan, np.nan)

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

    return (month, is_cross_year)


def generate_monthly_dates(start_year, start_month, end_year, end_month):
    """生成从 start_year/start_month 到 end_year/end_month 的逐月列表。"""
    current_date = datetime(int(start_year), int(start_month), 1)
    end_date = datetime(int(end_year), int(end_month), 1) + timedelta(days=32)
    end_date = end_date.replace(day=1) - timedelta(days=1)

    monthly_dates = []
    while current_date <= end_date:
        monthly_dates.append((current_date.year, current_date.month))
        next_month = current_date.month % 12 + 1
        next_year = current_date.year + (1 if current_date.month == 12 else 0)
        current_date = current_date.replace(year=next_year, month=next_month, day=1)

    return monthly_dates


def nearest_index(arr, value):
    """返回 arr 中离 value 最近的索引。"""
    return int(np.nanargmin(np.abs(np.asarray(arr, dtype=float) - float(value))))


def convert_lon_for_dataset(lon_value, lon_array):
    """
    根据目标数据集经度范围转换经度。
    - 若数据集是 0~360，经度转为 0~360。
    - 若数据集是 -180~180，经度转为 -180~180。
    """
    lon_value = float(lon_value)
    lon_array = np.asarray(lon_array, dtype=float)

    if np.nanmax(lon_array) > 180:
        return lon_value % 360

    if lon_value > 180:
        return lon_value - 360

    return lon_value


# ============================================================
# 2. 物候数据缓存：只打开一次物候文件
# ============================================================

def load_crop_cache(crop_nc, crop_vars):
    """
    一次性读取所有物候变量，避免每个格点、每个变量都打开 NC 文件。
    返回字典：lat、lon、arrays。
    """
    print("正在读取物候数据缓存...")
    crop_ds = xr.open_dataset(crop_nc)

    try:
        if "latitude" not in crop_ds.coords or "longitude" not in crop_ds.coords:
            raise ValueError("物候文件中未找到 latitude/longitude 坐标")

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

        print(f"物候数据缓存完成：{len(crop_vars)} 个变量，shape={next(iter(crop_arrays.values())).shape}")
        return {
            "lat": crop_lat,
            "lon": crop_lon,
            "arrays": crop_arrays
        }

    finally:
        crop_ds.close()


def extract_crop_days_fast(crop_cache, lat_value, lon_value, crop_vars):
    """
    从缓存中快速提取一个格点的所有物候变量。
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


# ============================================================
# 3. LDFAI 数据缓存：只打开一次 LDFAI 文件
# ============================================================

# ============================================================
# 3. 气候指数数据缓存：兼容 LDFAI / SPEI / 其他月尺度指数
# ============================================================

def infer_coord_names_for_index_file(ds, target_var):
    """
    自动识别气候指数文件中的时间、纬度、经度维度名。

    支持常见命名：
      时间: valid_time / time / Time / date / month
      纬度: lat / latitude / Lat / LAT / y
      经度: lon / longitude / Lon / LON / x
    """
    if target_var not in ds:
        raise ValueError(
            f"变量 {target_var} 不在文件中。当前文件变量有：{list(ds.data_vars)}"
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

    # 如果坐标名没找到，则从变量维度名中推断
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
            "无法自动识别时间/经纬度维度。"
            f" target_var={target_var}, dims={dims}, coords={list(ds.coords)}"
        )

    return time_name, lat_name, lon_name


def parse_year_month_from_time_values(time_values):
    """
    从时间坐标中解析 year/month。

    支持：
    1. datetime64 或 pandas 可解析时间；
    2. cftime 对象；
    3. 形如 198001、198002 的整数年月。
    """
    time_values = np.asarray(time_values)

    # datetime64 或 pandas 可解析时间
    try:
        times = pd.to_datetime(time_values)
        years = times.year.astype(int)
        months = times.month.astype(int)
        return years, months
    except Exception:
        pass

    # cftime 对象
    try:
        years = np.array([int(t.year) for t in time_values], dtype=int)
        months = np.array([int(t.month) for t in time_values], dtype=int)
        return years, months
    except Exception:
        pass

    # 198001、198002 这类整数年月
    if np.issubdtype(time_values.dtype, np.number):
        vals = time_values.astype(int)
        if np.nanmin(vals) > 100000 and np.nanmax(vals) < 300000:
            years = vals // 100
            months = vals % 100
            return years.astype(int), months.astype(int)

    raise ValueError(f"无法从时间坐标解析 year/month，time 示例：{time_values[:5]}")


def load_ldfai_cache(target_nc_file, target_var):
    """
    一次性读取 LDFAI/SPEI 等月尺度气候指数数据。

    该函数不再强制时间维为 valid_time，可自动识别 time / valid_time 等命名，
    并将变量统一转为 [time, lat, lon] 顺序。
    """
    print("正在读取气候指数数据缓存...")
    ds = xr.open_dataset(target_nc_file)

    try:
        time_name, lat_name, lon_name = infer_coord_names_for_index_file(ds, target_var)

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
                f"变量 {target_var} 维度不符合 [time, lat, lon] 结构。"
                f"缺少 {missing_dims}，实际 dims={da.dims}"
            )

        # 统一为 [time, lat, lon]
        da = da.transpose("time_index", "lat", "lon")

        time_years, time_months = parse_year_month_from_time_values(
            ds2["time_index"].values
        )

        # 建立 (year, month) -> time_index 的快速查找表
        time_lookup = {}
        for i, (yy, mm) in enumerate(zip(time_years, time_months)):
            time_lookup.setdefault((int(yy), int(mm)), int(i))

        data = da.values.astype(np.float32)

        print(
            f"气候指数缓存完成：变量={target_var}, "
            f"time={data.shape[0]}, lat={data.shape[1]}, lon={data.shape[2]}, "
            f"年份范围={time_years.min()}~{time_years.max()}, "
            f"原始时间维={time_name}, 原始纬度维={lat_name}, 原始经度维={lon_name}"
        )

        return {
            "lat": ds2["lat"].values,
            "lon": ds2["lon"].values,
            "years": time_years.astype(int),
            "months": time_months.astype(int),
            "unique_years": np.unique(time_years.astype(int)),
            "year_set": set(np.unique(time_years.astype(int)).tolist()),
            "time_lookup": time_lookup,
            "data": data,
            "var_name": target_var,
            "time_name_original": time_name,
            "lat_name_original": lat_name,
            "lon_name_original": lon_name
        }

    finally:
        ds.close()


def get_region_indices_1d(coord, lower, upper):
    """获取一维坐标中位于 [lower, upper] 范围内的索引。"""
    coord = np.asarray(coord, dtype=float)
    low, high = sorted([float(lower), float(upper)])
    return np.where((coord >= low) & (coord <= high))[0]


def get_lon_indices(lon_array, lon_min, lon_max):
    """
    获取经度索引，兼容：
    1. 目标数据经度为 0~360；
    2. 目标数据经度为 -180~180；
    3. 输入经度为 0~360 或 -180~180；
    4. 跨 0 度或跨 180 度情况。
    """
    lon = np.asarray(lon_array, dtype=float)

    lon_min = float(lon_min)
    lon_max = float(lon_max)

    # 情况 A：目标数据是 0~360
    if np.nanmax(lon) > 180:
        lon_min_c = lon_min % 360
        lon_max_c = lon_max % 360

        if lon_min_c <= lon_max_c:
            return np.where((lon >= lon_min_c) & (lon <= lon_max_c))[0]
        else:
            return np.where((lon >= lon_min_c) | (lon <= lon_max_c))[0]

    # 情况 B：目标数据是 -180~180
    else:
        def to_minus180_180(x):
            x = ((x + 180) % 360) - 180
            return x

        lon_min_c = to_minus180_180(lon_min)
        lon_max_c = to_minus180_180(lon_max)

        # 普通情况，例如 287~287.5 -> -73~-72.5
        if lon_min_c <= lon_max_c:
            return np.where((lon >= lon_min_c) & (lon <= lon_max_c))[0]

        # 跨 -180/180 边界情况
        else:
            return np.where((lon >= lon_min_c) | (lon <= lon_max_c))[0]


def mean_ldfai_for_month_fast(ldfai_cache, year, month, lat_min, lat_max, lon_min, lon_max):
    """
    从缓存数组中快速提取某年某月某区域的 LDFAI 平均值。
    """
    t_i = ldfai_cache["time_lookup"].get((int(year), int(month)))
    if t_i is None:
        return np.nan

    lat_indices = get_region_indices_1d(ldfai_cache["lat"], lat_min, lat_max)
    lon_indices = get_lon_indices(ldfai_cache["lon"], lon_min, lon_max)

    if len(lat_indices) == 0 or len(lon_indices) == 0:
        return np.nan

    region = ldfai_cache["data"][t_i, :, :][np.ix_(lat_indices, lon_indices)]
    if np.all(np.isnan(region)):
        return np.nan

    return float(np.nanmean(region))


def collect_ldfai_features_fast(
    ldfai_cache,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    start,
    end,
    threshold,
    skip_first_month=True
):
    """
    快速提取某个格点区域、某个物候期内 LDFAI < threshold 的时间键和值。

    返回格式：
        {"1984_1984_10": value, ...}
    """
    start_month, start_cross = start
    end_month, end_cross = end

    if pd.isna(start_month) or pd.isna(start_cross) or pd.isna(end_month) or pd.isna(end_cross):
        return {}

    result = {}
    unique_years = ldfai_cache["unique_years"]
    year_set = ldfai_cache["year_set"]

    for Y in unique_years:
        Y = int(Y)
        if (Y + 1) not in year_set:
            continue

        ldfai_year_month = generate_monthly_dates(
            Y + int(start_cross), int(start_month),
            Y + int(end_cross), int(end_month)
        )

        month_list = ldfai_year_month[1:] if skip_first_month else ldfai_year_month

        for ly, lm in month_list:
            value = mean_ldfai_for_month_fast(
                ldfai_cache=ldfai_cache,
                year=ly,
                month=lm,
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max
            )

            if np.isfinite(value):
                if threshold > 0:
                    condition = value > threshold
                elif threshold < 0:
                    condition = value < threshold
                else:
                    condition = False

                if condition:
                    key = f"{Y}_{ly}_{lm}"
                    result[key] = value

    return result


# ============================================================
# 4. 多模型趋势拟合：快速版
# ============================================================

def gaussian_trend(t, A, mu, sigma, B):
    return A * np.exp(-(t - mu) ** 2 / (2 * sigma ** 2)) + B


def get_gaussian_initial_params_and_bounds(t_data, y_data):
    """仅为 gaussian 模型设置初值和边界。"""
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
    """计算 RSS、RMSE、AIC、BIC。"""
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
        "bic": bic,
        "success": True
    }


def fit_one_model_fast(model_name, t_data, y_data):
    """
    加速版单模型拟合：
    - linear / quadratic / cubic 用 np.polyfit
    - log 用 np.linalg.lstsq
    - gaussian 用 curve_fit
    """
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
            p0, bounds = get_gaussian_initial_params_and_bounds(t_valid, y_valid)
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

        raise ValueError(f"未知模型：{model_name}")

    except Exception:
        # 大范围批处理时不要频繁 print 单个模型失败信息，否则会拖慢速度
        return None


def predict_trend(t, trend_result):
    """根据 trend_result 计算拟合值。"""
    model_name = trend_result["model"]
    params = trend_result["params"]

    if model_name in {"linear", "quadratic", "cubic"}:
        return np.polyval(params, t)

    if model_name == "log":
        a, b = params
        return a * np.log1p(t) + b

    if model_name == "gaussian":
        return gaussian_trend(t, *params)

    raise ValueError(f"未知模型：{model_name}")


def fit_multiple_models_with_outlier_removal_fast(
    t_data,
    y_data,
    max_iter=3,
    z_threshold=2.0,
    criterion="bic",
    candidate_model_names=None
):
    """
    同时拟合多个趋势模型，并按 AIC/BIC/RMSE 选择最优模型。
    """
    if criterion not in {"aic", "bic", "rmse"}:
        raise ValueError("criterion 必须是 'aic'、'bic' 或 'rmse'")

    if candidate_model_names is None:
        candidate_model_names = ["linear", "quadratic", "cubic", "log", "gaussian"]

    t_data = np.asarray(t_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    mask_global = np.isfinite(t_data) & np.isfinite(y_data)

    best_result = None
    all_results = {}

    for _ in range(max_iter):
        t_current = t_data[mask_global]
        y_current = y_data[mask_global]

        if len(y_current) < 10:
            break

        all_results = {}
        for model_name in candidate_model_names:
            result = fit_one_model_fast(model_name, t_current, y_current)
            if result is not None:
                all_results[model_name] = result

        if len(all_results) == 0:
            return None, {}, mask_global

        best_model_name = min(all_results, key=lambda name: all_results[name][criterion])
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

    return best_result, all_results, mask_global


# ============================================================
# 5. 残差计算与输出函数
# ============================================================

def calculate_residuals(ldfai_dict, actual_yield, years, trend_result):
    """
    计算 LDFAI 时间序列对应年份的实际产量与趋势拟合值的残差。

    残差定义：
        residual = actual_yield - fitted_yield
    """
    residuals = {}
    years = np.asarray(years).astype(int)
    actual_yield = np.asarray(actual_yield, dtype=float)

    for series_name, time_dict in ldfai_dict.items():
        residuals[series_name] = {}

        for time_key, ldfai_value in time_dict.items():
            base_year = int(time_key.split("_")[0])

            year_pos = np.where(years == base_year)[0]
            if len(year_pos) == 0:
                continue

            year_idx = int(year_pos[0])
            actual_value = actual_yield[year_idx]
            if not np.isfinite(actual_value):
                continue

            fitted_value = float(predict_trend(year_idx, trend_result))
            residual = float(actual_value - fitted_value)

            residuals[series_name][time_key] = {
                "年份": base_year,
                "实际产量": float(actual_value),
                "拟合值": fitted_value,
                "残差": residual,
                "LDFAI值": float(ldfai_value),
                "趋势模型": trend_result["model"],
                "AIC": float(trend_result["aic"]),
                "BIC": float(trend_result["bic"]),
                "RMSE": float(trend_result["rmse"])
            }

    return residuals


def extract_phenology_residual_mean(residuals):
    """从 residuals 字典中提取分物候期残差均值；同一物候期内同一年只保留一次。"""
    pheno_year_residuals = {
        "种植时间序列": {},
        "生长时间序列": {},
        "收获时间序列": {}
    }

    for pheno_name, time_dict in residuals.items():
        if pheno_name not in pheno_year_residuals:
            continue

        for _, metrics in time_dict.items():
            year = metrics.get("年份")
            residual = metrics.get("残差")

            if year is None or residual is None or not np.isfinite(residual):
                continue

            if year not in pheno_year_residuals[pheno_name]:
                pheno_year_residuals[pheno_name][year] = residual

    pheno_means = {}
    for pheno_name, year_res in pheno_year_residuals.items():
        if not year_res:
            pheno_means[pheno_name] = np.nan
        else:
            pheno_means[pheno_name] = float(np.mean(list(year_res.values())))

    return pheno_means


def create_phenology_nc(lat, lon, residual_array, pheno_name, output_path):
    """生成单个物候期的区域残差均值 NC 文件。"""
    ds_out = xr.Dataset(
        data_vars={
            "residual_mean": (["lat", "lon"], residual_array),
        },
        coords={
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "title": f"{pheno_name} Residual Mean",
            "phenology_period": pheno_name,
            "residual_definition": "实际产量 - 最优趋势模型拟合产量",
            "trend_model_selection": "linear/quadratic/cubic/log/gaussian; selected by configured criterion",
            "creation_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "valid_pixels": f"{np.sum(~np.isnan(residual_array))} / {residual_array.size}"
        }
    )

    ds_out["residual_mean"].attrs = {
        "long_name": f"{pheno_name} 残差均值",
        "units": "与产量单位一致",
        "missing_value": np.nan
    }
    ds_out["lat"].attrs = {"units": "degrees_north", "long_name": "纬度"}
    ds_out["lon"].attrs = {"units": "degrees_east", "long_name": "经度"}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds_out.to_netcdf(output_path, encoding={"residual_mean": {"zlib": True, "complevel": 4}})
    print(f"已保存：{os.path.abspath(output_path)}")
    return ds_out


def create_model_code_nc(lat, lon, model_code_array, output_path):
    """保存每个格点选择的最优趋势模型编号。"""
    ds_out = xr.Dataset(
        data_vars={
            "best_model_code": (["lat", "lon"], model_code_array),
        },
        coords={"lat": lat, "lon": lon},
        attrs={
            "title": "Best trend model code",
            "model_code": "1=linear, 2=quadratic, 3=cubic, 4=log, 5=gaussian, -1=failed/no data",
            "creation_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )
    ds_out["best_model_code"].attrs = {
        "long_name": "每个格点选择的最优趋势模型编号",
        "missing_value": -1
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds_out.to_netcdf(output_path, encoding={"best_model_code": {"zlib": True, "complevel": 4}})
    print(f"已保存：{os.path.abspath(output_path)}")
    return ds_out


# ============================================================
# 6. 主程序配置
# ============================================================

# 产量 NC 文件
nc_file = "/mnt/hdb/LXH/data/NZW/gdhy_v1.2_v1.3_20190128/rice_major/merged_yield_data.nc"
data_var = "var"
lat_var = "lat"
lon_var = "lon"
time_var = "time"

# 物候数据
crop_nc = "/mnt/hdb/LXH/data/NZW/plant & harvest(0.5 degree)/Rice.crop.calendar.fill.nc/Rice.crop.calendar.fill.nc"
crop_vars = [
    "plant.start", "plant.range", "plant.end", "plant",
    "tot.days", "harvest", "harvest.start", "harvest.range", "harvest.end"
]

# LDFAI 数据
target_nc_file = "/mnt/hdb/LXH/data/NZW/ERA5/data/ldfai_from_existing_spi_1month_1980_2020_region_-60_60_0_360_3.2_GPCC.nc"
target_var = "ldfai"

# 研究区域
lat_min = -60
lat_max = 60
lon_min = 0
lon_max = 360.0

# LDFAI 阈值：从 0.5 到 1.5，每次加 1
# np.arange(0.5, 1.5 + 1e-9, 1.0) 会得到：0.5、1.5
h_values = np.array([-9, -7.84, -6.76, -5.76, -4.84,-4,-3.24,-2.56,-1.96,-1.44,-1,1,1.44,1.96,2.56,3.24,4,4.84,5.76,6.76,7.84,9], dtype=float)

# 物候转换基准年
base_year = 1980

# 邻域设置
window_size = 1
half_window = (window_size - 1) // 2

# 多模型拟合设置
min_valid_points = 20
max_outlier_iter = 3
z_threshold = 2.0

# 一次运行三套趋势方案，并分别输出文件
# 1) 统一线性趋势：适合作为全球均值主结果
# 2) 多模型 AIC：适合作为局地最优/敏感性分析
# 3) 统一二次趋势：适合作为线性结果的敏感性分析
model_runs = [
    {
        "run_name": "linear_bic",
        "criterion": "bic",
        "candidate_models": ["linear"]
    },
    {
        "run_name": "multimodel_aic",
        "criterion": "aic",
        "candidate_models": ["linear", "quadratic", "cubic", "log", "gaussian"]
    },
    {
        "run_name": "quadratic_bic",
        "criterion": "bic",
        "candidate_models": ["quadratic"]
    }
]

# 是否跳过物候期起始月份：True 保留你原代码 ldfai_year_month[1:] 的逻辑
skip_first_ldfai_month = True

# 并行设置
# 建议先设为 1 跑通；确认无误后再设为 4、8 等。
# 使用 threading 可以共享已经缓存的 LDFAI 大数组，避免多进程复制大内存。
n_jobs = 1
parallel_backend = "threading"

# 输出目录
output_dir = "/mnt/hdb/LXH/data/NZW/LDFAI_SPI_ERA5_3_2_Y/rice/"

# 日志设置
verbose = False
progress_every = 500


# ============================================================
# 7. 一次性诊断输出函数：检查经纬度、时间、物候、LDFAI 匹配
# ============================================================

def print_one_time_diagnostics(
    lat,
    lon,
    data,
    years,
    non_nan_indices,
    crop_cache,
    ldfai_cache,
    n_lat,
    n_lon,
    lat_res,
    lon_res,
    h
):
    """
    在批量计算前只输出一次诊断信息，用于检查：
    1. 产量数据年份和时间维是否匹配；
    2. 当前样本格点经纬度是否能正确匹配到物候数据；
    3. 当前样本格点经纬度是否能正确匹配到 LDFAI 数据；
    4. 物候期月份转换是否合理；
    5. LDFAI 时间键是否能和产量年份对应。

    该函数只打印一次，不参与结果计算。
    """
    print("" + "#" * 90)
    print("一次性诊断输出：检查时间、经纬度、物候和 LDFAI 匹配")
    print("#" * 90)

    print("[1] 产量数据基本信息")
    print(f"  data shape = {data.shape}，预期为 [year, lat, lon]")
    print(f"  years 范围 = {years.min()} ~ {years.max()}，年份数 = {len(years)}")
    print(f"  lat 范围 = {float(np.nanmin(lat)):.6f} ~ {float(np.nanmax(lat)):.6f}，数量 = {len(lat)}，分辨率约 = {lat_res}")
    print(f"  lon 范围 = {float(np.nanmin(lon)):.6f} ~ {float(np.nanmax(lon)):.6f}，数量 = {len(lon)}，分辨率约 = {lon_res}")

    print("[2] 物候数据基本信息")
    crop_lat = crop_cache["lat"]
    crop_lon = crop_cache["lon"]
    print(f"  crop lat 范围 = {float(np.nanmin(crop_lat)):.6f} ~ {float(np.nanmax(crop_lat)):.6f}，数量 = {len(crop_lat)}")
    print(f"  crop lon 范围 = {float(np.nanmin(crop_lon)):.6f} ~ {float(np.nanmax(crop_lon)):.6f}，数量 = {len(crop_lon)}")
    print(f"  crop 变量 = {list(crop_cache['arrays'].keys())}")

    print("[3] LDFAI 数据基本信息")
    ldfai_lat = ldfai_cache["lat"]
    ldfai_lon = ldfai_cache["lon"]
    print(f"  LDFAI data shape = {ldfai_cache['data'].shape}，预期为 [time, lat, lon]")
    print(f"  LDFAI 年份范围 = {int(np.nanmin(ldfai_cache['years']))} ~ {int(np.nanmax(ldfai_cache['years']))}")
    print(f"  LDFAI 月份范围 = {int(np.nanmin(ldfai_cache['months']))} ~ {int(np.nanmax(ldfai_cache['months']))}")
    print(f"  LDFAI lat 范围 = {float(np.nanmin(ldfai_lat)):.6f} ~ {float(np.nanmax(ldfai_lat)):.6f}，数量 = {len(ldfai_lat)}")
    print(f"  LDFAI lon 范围 = {float(np.nanmin(ldfai_lon)):.6f} ~ {float(np.nanmax(ldfai_lon)):.6f}，数量 = {len(ldfai_lon)}")

    yield_year_set = set(np.asarray(years, dtype=int).tolist())
    ldfai_year_set = set(np.asarray(ldfai_cache["unique_years"], dtype=int).tolist())
    common_years = sorted(yield_year_set & ldfai_year_set)
    print("4] 产量年份与 LDFAI 年份交集")
    print(f"  交集年份范围 = {common_years[0]} ~ {common_years[-1]}，交集数量 = {len(common_years)}")
    missing_in_ldfai = sorted(yield_year_set - ldfai_year_set)
    missing_in_yield = sorted(ldfai_year_set - yield_year_set)
    print(f"  产量有但 LDFAI 没有的年份 = {missing_in_ldfai[:10]}{' ...' if len(missing_in_ldfai) > 10 else ''}")
    print(f"  LDFAI 有但产量没有的年份示例 = {missing_in_yield[:10]}{' ...' if len(missing_in_yield) > 10 else ''}")

    if len(non_nan_indices[0]) == 0:
        print("没有可用于诊断的有效格点。")
        print("#" * 90 + "")
        return

    # 选择第一个有效格点作为样本
    sample_lat_idx = int(non_nan_indices[0][0])
    sample_lon_idx = int(non_nan_indices[1][0])
    center_lat = float(lat[sample_lat_idx])
    center_lon = float(lon[sample_lon_idx])
    target_lon_for_crop = convert_lon_for_dataset(center_lon, crop_lon)
    target_lon_for_ldfai = convert_lon_for_dataset(center_lon, ldfai_lon)

    print("[5] 样本格点匹配检查")
    print(f"  样本索引 = lat_idx={sample_lat_idx}, lon_idx={sample_lon_idx}")
    print(f"  产量格点中心 = lat={center_lat:.6f}, lon={center_lon:.6f}")
    print(f"  转换到物候经度体系后的 lon = {target_lon_for_crop:.6f}")
    print(f"  转换到 LDFAI 经度体系后的 lon = {target_lon_for_ldfai:.6f}")

    crop_lat_i = nearest_index(crop_lat, center_lat)
    crop_lon_i = nearest_index(crop_lon, target_lon_for_crop)
    print("[6] 样本格点 -> 物候最近邻匹配")
    print(f"  crop 最近 lat = {float(crop_lat[crop_lat_i]):.6f}，差值 = {abs(float(crop_lat[crop_lat_i]) - center_lat):.6f}")
    print(f"  crop 最近 lon = {float(crop_lon[crop_lon_i]):.6f}，差值 = {abs(float(crop_lon[crop_lon_i]) - target_lon_for_crop):.6f}")

    try:
        point_result = extract_crop_days_fast(
            crop_cache=crop_cache,
            lat_value=center_lat,
            lon_value=center_lon,
            crop_vars=crop_vars
        )
        print("  样本物候天数：")
        for var in crop_vars:
            print(f"    {var}: {point_result[var]}")

        plant_start_month = days_to_month_cross(
            point_result["plant.start"], 0, 0, base_year, start=True
        )
        plant_end_month = days_to_month_cross(
            point_result["plant.end"], point_result["plant.start"], point_result["plant.start"], base_year, start=False
        )
        plant_month = days_to_month_cross(
            point_result["plant"], point_result["plant.start"], 0, base_year, start=True
        )
        harvest_month = days_to_month_cross(
            point_result["harvest"], point_result["plant"], point_result["tot.days"], base_year, start=False
        )
        harvest_start_month = days_to_month_cross(
            point_result["harvest.start"], point_result["plant.start"], 0, base_year, start=True
        )
        harvest_end_month = days_to_month_cross(
            point_result["harvest.end"], point_result["plant.start"], point_result["plant.start"], base_year, start=False
        )

        print("[7] 样本物候期月份转换")
        print(f"  种植期: {plant_start_month} -> {plant_end_month}")
        print(f"  生长期: {plant_month} -> {harvest_month}")
        print(f"  收获期: {harvest_start_month} -> {harvest_end_month}")

    except Exception as e:
        print(f"  物候诊断失败：{e}")
        print("#" * 90 + "")
        return

    min_lon = center_lon - lon_res / 2.0
    max_lon = center_lon + lon_res / 2.0
    min_lat = center_lat - lat_res / 2.0
    max_lat = center_lat + lat_res / 2.0

    ldfai_lat_indices = get_region_indices_1d(ldfai_lat, min_lat, max_lat)
    ldfai_lon_indices = get_lon_indices(ldfai_lon, min_lon, max_lon)

    print("8] 样本格点 -> LDFAI 区域匹配")
    print(f"  产量格点边界 lat = {min_lat:.6f} ~ {max_lat:.6f}")
    print(f"  产量格点边界 lon = {min_lon:.6f} ~ {max_lon:.6f}")
    print(f"  匹配到 LDFAI lat 索引数 = {len(ldfai_lat_indices)}")
    if len(ldfai_lat_indices) > 0:
        print(f"    LDFAI lat 覆盖 = {float(ldfai_lat[ldfai_lat_indices[0]]):.6f} ~ {float(ldfai_lat[ldfai_lat_indices[-1]]):.6f}")
    print(f"  匹配到 LDFAI lon 索引数 = {len(ldfai_lon_indices)}")
    if len(ldfai_lon_indices) > 0:
        print(f"    LDFAI lon 覆盖示例 = {float(ldfai_lon[ldfai_lon_indices[0]]):.6f} ~ {float(ldfai_lon[ldfai_lon_indices[-1]]):.6f}")

    print("[9] 样本 LDFAI 特征提取检查")
    all_features_pl = collect_ldfai_features_fast(
        ldfai_cache=ldfai_cache,
        lat_min=min_lat,
        lat_max=max_lat,
        lon_min=min_lon,
        lon_max=max_lon,
        start=plant_start_month,
        end=plant_end_month,
        threshold=h,
        skip_first_month=skip_first_ldfai_month
    )
    all_features_gro = collect_ldfai_features_fast(
        ldfai_cache=ldfai_cache,
        lat_min=min_lat,
        lat_max=max_lat,
        lon_min=min_lon,
        lon_max=max_lon,
        start=plant_month,
        end=harvest_month,
        threshold=h,
        skip_first_month=skip_first_ldfai_month
    )
    all_features_hv = collect_ldfai_features_fast(
        ldfai_cache=ldfai_cache,
        lat_min=min_lat,
        lat_max=max_lat,
        lon_min=min_lon,
        lon_max=max_lon,
        start=harvest_start_month,
        end=harvest_end_month,
        threshold=h,
        skip_first_month=skip_first_ldfai_month
    )

    print(f"  种植期 LDFAI<{h} 数量 = {len(all_features_pl)}，前 5 个 = {list(all_features_pl.items())[:5]}")
    print(f"  生长期 LDFAI<{h} 数量 = {len(all_features_gro)}，前 5 个 = {list(all_features_gro.items())[:5]}")
    print(f"  收获期 LDFAI<{h} 数量 = {len(all_features_hv)}，前 5 个 = {list(all_features_hv.items())[:5]}")

    print("[10] 样本产量时间序列检查")
    sample_yield = data[:, sample_lat_idx, sample_lon_idx]
    finite_mask = np.isfinite(sample_yield)
    print(f"  有效产量年份数 = {int(np.sum(finite_mask))} / {len(years)}")
    print(f"  前 10 年产量样本 = {list(zip(years[:10].tolist(), sample_yield[:10].tolist()))}")
    if np.any(finite_mask):
        print(f"  样本产量范围 = {float(np.nanmin(sample_yield)):.6f} ~ {float(np.nanmax(sample_yield)):.6f}")

    print("诊断说明：如果上述经纬度差值小于或接近数据分辨率的一半，年份交集覆盖 1981~2016，且 LDFAI 区域索引数不为 0，则匹配逻辑通常是正常的。")
    print("#" * 90 + "")


# ============================================================
# 8. 单个格点处理函数
# ============================================================

def process_one_point(
    idx,
    lat_idx,
    lon_idx,
    lat,
    lon,
    data,
    years,
    crop_cache,
    ldfai_cache,
    model_code,
    n_lat,
    n_lon,
    lat_res,
    lon_res,
    h,
    run_criterion,
    run_candidate_models
):
    """
    处理单个格点。

    返回：
        lat_idx, lon_idx, plant_mean, growth_mean, harvest_mean, best_model_code
    """
    center_lat = float(lat[lat_idx])
    center_lon = float(lon[lon_idx])
    target_lon = center_lon - 360 if center_lon > 180 else center_lon

    min_lon = center_lon - lon_res / 2.0
    max_lon = center_lon + lon_res / 2.0
    min_lat = center_lat - lat_res / 2.0
    max_lat = center_lat + lat_res / 2.0

    lat_start = max(0, lat_idx - half_window)
    lat_end = min(n_lat, lat_idx + half_window + 1)
    lon_start = max(0, lon_idx - half_window)
    lon_end = min(n_lon, lon_idx + half_window + 1)

    # 直接用 numpy 切片，避免每个格点使用 xarray.stack 带来的开销
    neighborhood = data[:, lat_start:lat_end, lon_start:lon_end]
    time_series_array = neighborhood.reshape(data.shape[0], -1)

    t = np.arange(len(years), dtype=float)
    n_local_points = time_series_array.shape[1]
    all_t = np.repeat(t, n_local_points)
    all_values = time_series_array.reshape(-1)

    mask_initial = np.isfinite(all_values)
    t_clean = all_t[mask_initial]
    values_clean = all_values[mask_initial]

    if len(values_clean) < min_valid_points:
        return lat_idx, lon_idx, np.nan, np.nan, np.nan, -1

    # 多模型趋势拟合
    best_trend, all_trend_results, final_mask = fit_multiple_models_with_outlier_removal_fast(
        t_clean,
        values_clean,
        max_iter=max_outlier_iter,
        z_threshold=z_threshold,
        criterion=run_criterion,
        candidate_model_names=run_candidate_models
    )

    if best_trend is None:
        return lat_idx, lon_idx, np.nan, np.nan, np.nan, -1

    best_code = model_code.get(best_trend["model"], -1)

    if verbose:
        print(
            f"idx={idx}, lat={center_lat:.3f}, lon={center_lon:.3f}, "
            f"run={run_criterion}/{run_candidate_models}, best={best_trend['model']}, BIC={best_trend['bic']:.2f}, RMSE={best_trend['rmse']:.4f}"
        )

    # 物候数据快速提取
    try:
        point_result = extract_crop_days_fast(
            crop_cache=crop_cache,
            lat_value=center_lat,
            lon_value=center_lon,
            crop_vars=crop_vars
        )

        if any(np.isnan(point_result[var]) for var in crop_vars):
            return lat_idx, lon_idx, np.nan, np.nan, np.nan, best_code

    except Exception:
        return lat_idx, lon_idx, np.nan, np.nan, np.nan, best_code

    # 物候期转换
    try:
        plant_start_month = days_to_month_cross(
            point_result["plant.start"], 0, 0, base_year, start=True
        )
        plant_end_month = days_to_month_cross(
            point_result["plant.end"], point_result["plant.start"], point_result["plant.start"], base_year, start=False
        )
        plant_month = days_to_month_cross(
            point_result["plant"], point_result["plant.start"], 0, base_year, start=True
        )
        harvest_month = days_to_month_cross(
            point_result["harvest"], point_result["plant"], point_result["tot.days"], base_year, start=False
        )
        harvest_start_month = days_to_month_cross(
            point_result["harvest.start"], point_result["plant.start"], 0, base_year, start=True
        )
        harvest_end_month = days_to_month_cross(
            point_result["harvest.end"], point_result["plant.start"], point_result["plant.start"], base_year, start=False
        )
    except Exception:
        return lat_idx, lon_idx, np.nan, np.nan, np.nan, best_code

    # 快速提取三个物候期内 LDFAI < h 的年份/月
    try:
        all_features_pl = collect_ldfai_features_fast(
            ldfai_cache=ldfai_cache,
            lat_min=min_lat,
            lat_max=max_lat,
            lon_min=min_lon,
            lon_max=max_lon,
            start=plant_start_month,
            end=plant_end_month,
            threshold=h,
            skip_first_month=skip_first_ldfai_month
        )

        all_features_gro = collect_ldfai_features_fast(
            ldfai_cache=ldfai_cache,
            lat_min=min_lat,
            lat_max=max_lat,
            lon_min=min_lon,
            lon_max=max_lon,
            start=plant_month,
            end=harvest_month,
            threshold=h,
            skip_first_month=skip_first_ldfai_month
        )

        all_features_hv = collect_ldfai_features_fast(
            ldfai_cache=ldfai_cache,
            lat_min=min_lat,
            lat_max=max_lat,
            lon_min=min_lon,
            lon_max=max_lon,
            start=harvest_start_month,
            end=harvest_end_month,
            threshold=h,
            skip_first_month=skip_first_ldfai_month
        )
    except Exception:
        return lat_idx, lon_idx, np.nan, np.nan, np.nan, best_code

    ldfai_series = {
        "种植时间序列": all_features_pl,
        "生长时间序列": all_features_gro,
        "收获时间序列": all_features_hv
    }

    ts_values = data[:, lat_idx, lon_idx]

    residuals = calculate_residuals(
        ldfai_series,
        actual_yield=ts_values,
        years=years,
        trend_result=best_trend
    )

    pheno_means = extract_phenology_residual_mean(residuals)

    plant_mean = pheno_means["种植时间序列"]
    growth_mean = pheno_means["生长时间序列"]
    harvest_mean = pheno_means["收获时间序列"]

    return lat_idx, lon_idx, plant_mean, growth_mean, harvest_mean, best_code


# ============================================================
# 8. 主程序
# ============================================================

if __name__ == "__main__":
    ds = xr.open_dataset(nc_file)

    try:
        lat = ds[lat_var].values
        lon = ds[lon_var].values

        # 产量数据：按真实年份整理
        yield_da_annual, years = prepare_annual_yield_data(ds, data_var, time_var)
        data = yield_da_annual.values.astype(np.float32)

        n_time, n_lat, n_lon = data.shape
        lat_res = np.abs(lat[1] - lat[0]) if n_lat > 1 else 0.0
        lon_res = np.abs(lon[1] - lon[0]) if n_lon > 1 else 0.0

        # 一次性读取物候和 LDFAI 缓存
        crop_cache = load_crop_cache(crop_nc, crop_vars)
        ldfai_cache = load_ldfai_cache(target_nc_file, target_var)

        # 目标区域和有效产量格点
        first_time_data = np.nanmean(data, axis=0)
        non_nan_mask = ~np.isnan(first_time_data)

        lat_mask = (lat >= lat_min) & (lat <= lat_max)
        lon_mask = (lon >= lon_min) & (lon <= lon_max)
        region_mask = non_nan_mask & np.outer(lat_mask, lon_mask)
        non_nan_indices = np.where(region_mask)
        n_points = len(non_nan_indices[0])

        print(f"目标区域 [{lat_min}~{lat_max}°N, {lon_min}~{lon_max}°E] 内共发现 {n_points} 个非 NaN 栅格点")

        # 构造任务列表，只需要构造一次
        tasks = [
            (idx, int(non_nan_indices[0][idx]), int(non_nan_indices[1][idx]))
            for idx in range(n_points)
        ]

        model_code = {
            "linear": 1,
            "quadratic": 2,
            "cubic": 3,
            "log": 4,
            "gaussian": 5
        }

        # ====================================================
        # 外层循环：依次运行不同 h 阈值
        # h_values = [0.5, 1.5]
        # ====================================================
        for h in h_values:
            h = float(h)

            print("" + "#" * 90)
            print(f"开始运行 LDFAI/SPEI 阈值：h = {h}")
            print("#" * 90)

            # 每个 h 输出一次诊断信息，确认时间、经纬度、物候和 LDFAI/SPEI 匹配是否正常
            print_one_time_diagnostics(
                lat=lat,
                lon=lon,
                data=data,
                years=years,
                non_nan_indices=non_nan_indices,
                crop_cache=crop_cache,
                ldfai_cache=ldfai_cache,
                n_lat=n_lat,
                n_lon=n_lon,
                lat_res=lat_res,
                lon_res=lon_res,
                h=h
            )

            # ====================================================
            # 内层循环：依次运行三套趋势方案，并分别保存结果
            # ====================================================
            for run_cfg in model_runs:
                run_name = run_cfg["run_name"]
                run_criterion = run_cfg["criterion"]
                run_candidate_models = run_cfg["candidate_models"]

                print("" + "=" * 90)
                print(f"开始运行趋势方案：{run_name}, h = {h}")
                print(f"  criterion = {run_criterion}")
                print(f"  candidate_models = {run_candidate_models}")
                print(f"  n_jobs = {n_jobs}, backend = {parallel_backend}")
                print("=" * 90)

                residual_mean_plant = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
                residual_mean_growth = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
                residual_mean_harvest = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
                best_model_map = np.full((n_lat, n_lon), -1, dtype=np.int16)

                if n_jobs == 1 or not JOBLIB_AVAILABLE:
                    results = []
                    for idx, lat_idx, lon_idx in tasks:
                        result = process_one_point(
                            idx=idx,
                            lat_idx=lat_idx,
                            lon_idx=lon_idx,
                            lat=lat,
                            lon=lon,
                            data=data,
                            years=years,
                            crop_cache=crop_cache,
                            ldfai_cache=ldfai_cache,
                            model_code=model_code,
                            n_lat=n_lat,
                            n_lon=n_lon,
                            lat_res=lat_res,
                            lon_res=lon_res,
                            h=h,
                            run_criterion=run_criterion,
                            run_candidate_models=run_candidate_models
                        )
                        results.append(result)

                        if (idx + 1) % progress_every == 0:
                            print(f"[h={h}, {run_name}] 已处理 {idx + 1}/{n_points} 个点")
                else:
                    results = Parallel(n_jobs=n_jobs, backend=parallel_backend, verbose=5)(
                        delayed(process_one_point)(
                            idx=idx,
                            lat_idx=lat_idx,
                            lon_idx=lon_idx,
                            lat=lat,
                            lon=lon,
                            data=data,
                            years=years,
                            crop_cache=crop_cache,
                            ldfai_cache=ldfai_cache,
                            model_code=model_code,
                            n_lat=n_lat,
                            n_lon=n_lon,
                            lat_res=lat_res,
                            lon_res=lon_res,
                            h=h,
                            run_criterion=run_criterion,
                            run_candidate_models=run_candidate_models
                        )
                        for idx, lat_idx, lon_idx in tasks
                    )

                # 汇总当前 h + 当前趋势方案的结果
                for lat_idx, lon_idx, plant_mean, growth_mean, harvest_mean, best_code in results:
                    residual_mean_plant[lat_idx, lon_idx] = plant_mean
                    residual_mean_growth[lat_idx, lon_idx] = growth_mean
                    residual_mean_harvest[lat_idx, lon_idx] = harvest_mean
                    best_model_map[lat_idx, lon_idx] = best_code

                valid_plant = int(np.sum(~np.isnan(residual_mean_plant)))
                valid_growth = int(np.sum(~np.isnan(residual_mean_growth)))
                valid_harvest = int(np.sum(~np.isnan(residual_mean_harvest)))
                print(
                    f"[h={h}, {run_name}] 有效格点数："
                    f"planting={valid_plant}, growing={valid_growth}, harvesting={valid_harvest}"
                )

                # 保存当前 h + 当前趋势方案结果；文件名中包含 h 和 run_name
                create_phenology_nc(
                    lat=lat,
                    lon=lon,
                    residual_array=residual_mean_plant,
                    pheno_name=f"种植时间序列_h{h}_{run_name}",
                    output_path=f"{output_dir}/ls_{h}_residual_mean_planting_{lat_min}_{lat_max}_{lon_min}_{lon_max}s_{run_name}.nc"
                )

                create_phenology_nc(
                    lat=lat,
                    lon=lon,
                    residual_array=residual_mean_growth,
                    pheno_name=f"生长时间序列_h{h}_{run_name}",
                    output_path=f"{output_dir}/ls_{h}_residual_mean_growing_{lat_min}_{lat_max}_{lon_min}_{lon_max}s_{run_name}.nc"
                )

                create_phenology_nc(
                    lat=lat,
                    lon=lon,
                    residual_array=residual_mean_harvest,
                    pheno_name=f"收获时间序列_h{h}_{run_name}",
                    output_path=f"{output_dir}/ls_{h}_residual_mean_harvesting_{lat_min}_{lat_max}_{lon_min}_{lon_max}s_{run_name}.nc"
                )

                create_model_code_nc(
                    lat=lat,
                    lon=lon,
                    model_code_array=best_model_map,
                    output_path=f"{output_dir}/ls_{h}_best_trend_model_code_{lat_min}_{lat_max}_{lon_min}_{lon_max}s_{run_name}.nc"
                )

                print(f"趋势方案 {run_name}, h={h} 处理完成。")

            print(f"LDFAI/SPEI 阈值 h={h} 的全部趋势方案处理完成。")

        print("全部 h 阈值和全部趋势方案处理完成！")

    finally:
        ds.close()
