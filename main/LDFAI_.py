import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import stats

try:
    import spei as si
except ImportError as exc:
    raise ImportError(
        "缺少 spei 库，请先安装：pip install spei"
    ) from exc


# ======================
# 1. 读取DSSAT WTH文件
# ======================
def read_dssat_wth(file_path, start_year=2015, end_year=2100):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    station_info = {}
    data_start = None

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        if line.startswith('*'):
            parts = line[1:].split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip().upper()
                value = parts[1].strip()
                station_info[key] = value

                if key == 'LATITUDE':
                    try:
                        lat_str = value.split()[0]
                        station_info['LAT'] = float(lat_str)
                        if 'S' in value.upper():
                            station_info['LAT'] = -station_info['LAT']
                    except:
                        pass
                elif key == 'LONGITUDE':
                    try:
                        lon_str = value.split()[0]
                        station_info['LONG'] = float(lon_str)
                        if 'W' in value.upper():
                            station_info['LONG'] = -station_info['LONG']
                    except:
                        pass
                elif key == 'ELEVATION':
                    try:
                        elev_str = value.split()[0]
                        station_info['ELEV'] = float(elev_str)
                    except:
                        pass

        elif line.startswith('@ INSI'):
            if i + 1 < len(lines):
                data_line = lines[i + 1].strip()
                parts = data_line.split()
                if len(parts) >= 4:
                    station_info['INSI'] = parts[0]
                    station_info['LAT'] = float(parts[1])
                    station_info['LONG'] = float(parts[2])
                    station_info['ELEV'] = float(parts[3])

        elif line.startswith('@DATE'):
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError("文件中未找到@DATE数据头")
    if 'LAT' not in station_info:
        raise ValueError("无法从WTH文件中读取纬度信息，请检查文件格式")

    # DSSAT 的日期通常是 YYDDD（两位年份 + 年积日）。
    # 本数据真实时间范围是 2015-2100，因此不能再使用“80以上归为1900年代”的规则。
    # 下面根据目标年份范围建立唯一映射：
    # 15 -> 2015, ..., 99 -> 2099, 00 -> 2100。
    if start_year > end_year:
        raise ValueError("start_year 不能大于 end_year")

    year_map = {}
    for full_year in range(start_year, end_year + 1):
        yy = full_year % 100
        if yy in year_map:
            raise ValueError(
                f"年份范围 {start_year}-{end_year} 超过两位年份可唯一解析的范围，请缩短时间范围"
            )
        year_map[yy] = full_year

    data = []
    skipped_rows = 0

    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith('@'):
            continue
        parts = line.split()
        if len(parts) >= 5:
            try:
                date_str = parts[0]

                # YYDDD，例如：15001=2015年第1天，99365=2099年第365天，00001=2100年第1天
                year_two_digit = int(date_str[:2])
                if year_two_digit not in year_map:
                    raise ValueError(
                        f"两位年份 {year_two_digit:02d} 不在目标范围 {start_year}-{end_year} 内"
                    )
                year = year_map[year_two_digit]

                doy = int(date_str[2:])
                if doy < 1 or doy > 366:
                    raise ValueError(f"非法年积日 DOY={doy}")

                date = datetime(year, 1, 1) + timedelta(days=doy - 1)

                # 防止非闰年 DOY=366 被滚动到下一年
                if date.year != year:
                    raise ValueError(f"年份 {year} 中不存在 DOY={doy}")

                data.append([
                    date,
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4])
                ])
            except (ValueError, IndexError):
                skipped_rows += 1
                continue

    if not data:
        raise ValueError(
            f"未读取到 {start_year}-{end_year} 范围内的有效气象数据，请检查 WTH 日期格式"
        )

    df = pd.DataFrame(data, columns=['date', 'srad', 'tmax', 'tmin', 'rain'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    print("=" * 70)
    print("✅ 成功读取站点信息:")
    print(f"   站点代码: {station_info.get('INSI', '未知')}")
    print(f"   纬度: {station_info['LAT']:.3f}°")
    print(f"   经度: {station_info.get('LONG', '未知'):.3f}°")
    print(f"   海拔: {station_info.get('ELEV', '未知')} m")
    print(f"   目标年份: {start_year} 到 {end_year}")
    print(f"   数据范围: {df.index.min().date()} 到 {df.index.max().date()}")
    print(f"   总天数: {len(df)} 天")
    if skipped_rows > 0:
        print(f"   ⚠️ 跳过无法解析/超出年份范围的数据行: {skipped_rows} 行")
    print("=" * 70)

    return df, station_info


# ======================
# 1.1 快速读取站点代码（用于主进程判断是否已处理）
# ======================
def read_station_code_fast(file_path):
    """
    只读取WTH文件头部以获取站点代码，不解析整段日气象数据。
    用于批处理前判断输出文件是否已存在，避免每个站点被完整读取两次。
    """
    default_code = os.path.splitext(os.path.basename(file_path))[0]

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                if line.startswith('@ INSI'):
                    data_line = next(f, '').strip()
                    parts = data_line.split()
                    if parts:
                        return parts[0]

                # 已经到数据区仍未找到站点代码，就不再继续扫描
                if line.startswith('@DATE'):
                    break
    except Exception:
        pass

    return default_code


# ======================
# 2. 日尺度Thornthwaite PET计算
# ======================
def daily_thornthwaite_pet(daily_temp, latitude):
    temp = daily_temp.copy()
    temp[temp < 0] = 0
    yearly_data = temp.groupby(temp.index.year)
    yearly_I = yearly_data.apply(lambda x: np.sum((x.resample('ME').mean() / 5) ** 1.514))
    yearly_m = 0.492 + 1.792e-2 * yearly_I - 7.71e-5 * yearly_I ** 2 + 6.75e-7 * yearly_I ** 3
    daily_I = temp.index.year.map(yearly_I)
    daily_m = temp.index.year.map(yearly_m)
    pet_uncorrected = 16 * (10 * temp / daily_I) ** daily_m / 30

    lat_rad = np.radians(latitude)
    j = temp.index.to_series().dt.dayofyear
    delta = 0.4093 * np.sin(2 * np.pi * j / 365 - 1.405)
    cos_omega_s = -np.tan(lat_rad) * np.tan(delta)
    cos_omega_s = np.clip(cos_omega_s, -1, 1)
    omega_s = np.arccos(cos_omega_s)
    N = 24 / np.pi * omega_s
    K = N / 12
    return pet_uncorrected * K


# ======================
# 3. 滑动窗口日尺度SPEI计算
# ======================
def calculate_daily_spei_with_window(df_daily, latitude, window_days=60):
    print(f"\n🔄 正在计算日尺度PET...")
    df_daily = df_daily.copy()
    df_daily['tavg'] = (df_daily['tmax'] + df_daily['tmin']) / 2
    df_daily['pet_daily'] = daily_thornthwaite_pet(df_daily['tavg'], latitude)

    print(f"📊 正在计算{window_days}天滑动窗口累积...")
    df_daily['rain_window'] = df_daily['rain'].rolling(window=window_days, min_periods=window_days).sum()
    df_daily['pet_window'] = df_daily['pet_daily'].rolling(window=window_days, min_periods=window_days).sum()
    df_daily['D'] = df_daily['rain_window'] - df_daily['pet_window']

    print("🌡️  正在调用 spei 库计算日尺度SPEI...")
    water_balance = (df_daily['rain'] - df_daily['pet_daily']).dropna()
    if len(water_balance) < window_days:
        raise ValueError(f"有效日数据不足，至少需要 {window_days} 天才能计算SPEI")

    spei_values = si.spei(
        water_balance,
        dist=stats.fisk,
        timescale=window_days,
        fit_freq="D",
        fit_window=31
    )
    df_daily['spei'] = spei_values.reindex(df_daily.index)
    spei_valid = df_daily['spei'].dropna()
    print(f"   - 有效SPEI天数: {len(spei_valid)}")
    if len(spei_valid) > 0:
        print(f"   - SPEI范围: [{spei_valid.min():.3f}, {spei_valid.max():.3f}]")
    return df_daily


# ======================
# 4. 滑动窗口LDFAI计算
# ======================
def compute_ldfai_sliding_window(spei_series, window_days=60):
    print(f"\n⚡ 正在计算滑动窗口LDFAI (窗口: {window_days}天)...")
    spei_valid = spei_series.dropna().sort_index()
    if len(spei_valid) < window_days + 1:
        raise ValueError(f"有效SPEI数据点不足，需要至少{window_days + 1}天数据")
    spei_prev = spei_valid.shift(window_days)
    a = spei_valid - spei_prev
    b = spei_prev.abs() + spei_valid.abs()
    c = 3.2 ** (-(spei_prev + spei_valid).abs())
    ldfai = a * b * c
    ldfai_valid = ldfai.dropna()
    print(f"   - 原始有效SPEI天数: {len(spei_valid)}")
    print(f"   - 有效LDFAI天数: {len(ldfai_valid)}")
    print(f"   - LDFAI范围: [{ldfai_valid.min():.3f}, {ldfai_valid.max():.3f}]")
    return ldfai


# ======================
# 5. 单站点处理函数
# ======================
def process_single_station(
    wth_file, output_dir, SPEI_WINDOW_DAYS=60, LDFAI_WINDOW_DAYS=60,
    START_YEAR=2015, END_YEAR=2100
):
    try:
        print(f"\n{'=' * 80}")
        print(f"正在处理文件: {os.path.basename(wth_file)}")
        print(f"{'=' * 80}")

        df_daily, station_info = read_dssat_wth(
            wth_file, start_year=START_YEAR, end_year=END_YEAR
        )
        LATITUDE = station_info['LAT']

        df_result = calculate_daily_spei_with_window(df_daily, LATITUDE, window_days=SPEI_WINDOW_DAYS)
        df_result['ldfai'] = compute_ldfai_sliding_window(df_result['spei'], window_days=LDFAI_WINDOW_DAYS)

        station_code = station_info.get('INSI', os.path.splitext(os.path.basename(wth_file))[0])
        output_file = os.path.join(
            output_dir,
            f"{station_code}_SPEI{SPEI_WINDOW_DAYS}_LDFAI_lag{LDFAI_WINDOW_DAYS}.csv"
        )

        df_result[['spei', 'ldfai']].to_csv(output_file, encoding='utf-8-sig', index_label='date')
        print(f"\n✅ 处理完成！结果已保存到: {output_file}")

        # 统计验证
        print(f"\n{'=' * 70}")
        print(f"📊 最终统计验证")
        print(f"{'=' * 70}")
        spei_vals = df_result['spei'].dropna()
        print(f"\n【SPEI统计】 (滑动窗口: {SPEI_WINDOW_DAYS}天)")
        print(f"   有效天数: {len(spei_vals)}")
        print(f"   均值: {spei_vals.mean():.3f} (理论值≈0)")
        print(f"   标准差: {spei_vals.std():.3f} (理论值≈1)")

        ldfai_vals = df_result['ldfai'].dropna()
        print(f"\n【LDFAI统计】 (滑动窗口: {LDFAI_WINDOW_DAYS}天)")
        print(f"   有效天数: {len(ldfai_vals)}")
        if len(ldfai_vals) > 0:
            print(f"   均值: {ldfai_vals.mean():.3f}")
            print(f"   标准差: {ldfai_vals.std():.3f}")
            print(f"   最小值: {ldfai_vals.min():.3f}")
            print(f"   最大值: {ldfai_vals.max():.3f}")

        return True

    except Exception as e:
        print(f"\n❌ 处理文件 {os.path.basename(wth_file)} 出错: {str(e)}")
        return False


# ======================
# 6. 多进程工作函数
# ======================
def process_station_worker(task):
    """
    ProcessPoolExecutor 的顶层工作函数。
    每个子进程独立处理一个WTH站点文件，避免站点之间相互影响。
    """
    (
        file_path,
        output_dir,
        spei_window_days,
        ldfai_window_days,
        start_year,
        end_year,
    ) = task

    success = process_single_station(
        file_path,
        output_dir,
        spei_window_days,
        ldfai_window_days,
        start_year,
        end_year,
    )

    return os.path.basename(file_path), success


# ======================
# 主程序：批量处理所有WTH文件（自动跳过已处理）
# ======================
if __name__ == "__main__":
    WTH_FOLDER = r"/mnt/hdb/LXH/NZW_DG/MME/WTH/ssp585/"
    OUTPUT_DIR = r"/mnt/hdb/LXH/NZW_DG/MME/WTH/SPEI_LDFAI_ssp585"

    # 当前 WTH 数据的真实时间范围
    START_YEAR = 2015
    END_YEAR = 2100

    SPEI_WINDOW_DAYS = 30
    LDFAI_WINDOW_DAYS = 30

    # ======================
    # CPU并行设置
    # ======================
    # 建议先用 4~8 个进程。每个进程都会独立运行一次 SPEI 拟合并占用内存。
    # 如果服务器 CPU 核数和内存较多，可适当增大。
    CPU_COUNT = os.cpu_count() or 20
    MAX_WORKERS = min(20, CPU_COUNT)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wth_files = sorted(
        f for f in os.listdir(WTH_FOLDER)
        if f.lower().endswith('.wth')
    )

    if not wth_files:
        print(f"❌ 在 {WTH_FOLDER} 中未找到任何 .WTH 文件！")
        raise SystemExit(1)

    print(f"✅ 找到 {len(wth_files)} 个 WTH 文件")
    print(f"🖥️  检测到 CPU 逻辑核心数: {CPU_COUNT}")
    print(f"🚀 并行进程数: {MAX_WORKERS}")

    # --------------------------------------------------
    # 第一步：主进程快速扫描文件头，跳过已经处理的站点
    # --------------------------------------------------
    tasks = []
    skipped_count = 0

    for wth_file in wth_files:
        file_path = os.path.join(WTH_FOLDER, wth_file)
        station_code = read_station_code_fast(file_path)

        output_file = os.path.join(
            OUTPUT_DIR,
            f"{station_code}_SPEI{SPEI_WINDOW_DAYS}_LDFAI_lag{LDFAI_WINDOW_DAYS}.csv"
        )

        if os.path.exists(output_file):
            print(f"⏭️ 已处理，跳过: {wth_file}")
            skipped_count += 1
            continue

        tasks.append((
            file_path,
            OUTPUT_DIR,
            SPEI_WINDOW_DAYS,
            LDFAI_WINDOW_DAYS,
            START_YEAR,
            END_YEAR,
        ))

    print(f"\n📋 待处理文件: {len(tasks)} 个")
    print(f"⏭️ 已跳过文件: {skipped_count} 个")

    if not tasks:
        print("✅ 所有WTH文件均已处理，无需重复计算。")
        raise SystemExit(0)

    # 实际并行数不能超过待处理任务数
    actual_workers = min(MAX_WORKERS, len(tasks))
    print(f"🚀 开始多站点 CPU 并行处理，实际进程数: {actual_workers}\n")

    success_count = 0
    fail_count = 0
    completed_count = 0

    # --------------------------------------------------
    # 第二步：每个WTH站点由一个独立CPU进程处理
    # --------------------------------------------------
    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        future_to_file = {
            executor.submit(process_station_worker, task): os.path.basename(task[0])
            for task in tasks
        }

        for future in as_completed(future_to_file):
            wth_file = future_to_file[future]
            completed_count += 1

            try:
                result_file, success = future.result()
                if success:
                    success_count += 1
                    status = "✅ 成功"
                else:
                    fail_count += 1
                    status = "❌ 失败"

                print(
                    f"\n[{completed_count}/{len(tasks)}] {status}: {result_file} | "
                    f"成功 {success_count} | 失败 {fail_count}"
                )

            except Exception as e:
                fail_count += 1
                print(
                    f"\n[{completed_count}/{len(tasks)}] ❌ 子进程异常: {wth_file} | "
                    f"{type(e).__name__}: {e}"
                )

    print(f"\n{'=' * 80}")
    print("📋 多站点 CPU 并行批处理完成！")
    print(f"总文件数: {len(wth_files)}")
    print(f"本次待处理: {len(tasks)} 个")
    print(f"成功处理: {success_count} 个")
    print(f"跳过文件: {skipped_count} 个")
    print(f"失败文件: {fail_count} 个")
    print(f"并行进程数: {actual_workers}")
    print(f"结果保存路径: {OUTPUT_DIR}")
    print(f"{'=' * 80}")
