# =============================================================
# 福建省 2025 年逐日 SPI2（SPI_20d）计算脚本
# =============================================================
# 方法说明：
#   本脚本使用 climate_indices 官方包计算 daily SPI。
#   daily SPI1 在这里定义为 30 天滑动累计降雨对应的 SPI，即 SPI_20d。
#
# 数据要求：
#   1. 输入为 CHIRPS 福建省逐日降雨 NC，逐年一个文件，年份覆盖 1981-2025。
#   2. 降水单位应为 mm/day。
#   3. 维度可以是 time/lat/lon，也可以是 time/latitude/longitude。
#
# 关键逻辑：
#   1. 读取并合并 1981-2025 年逐日降雨。
#   2. 按 climate_indices daily 模式要求，重组为“每年 366 个位置”的序列。
#      非闰年会保留 2 月 29 日位置为 NaN，3 月 1 日及之后日期后移一位。
#      这是你原稿里最容易写错的地方：不能简单把非闰年 365 天塞到前 365 位。
#   3. 对每个网格调用 climate_indices.indices.spi()。
#   4. 仅输出 2025 年真实 365 天的 SPI_20d。
#   5. 输出图表使用中文文件名，NC 文件使用英文文件名。
#   6. 脚本结束后删除本次运行产生的临时缓存目录。
# =============================================================

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import xarray as xr
from climate_indices import compute, indices
from tqdm import tqdm


# =============================================================
# 1. 参数配置区
# =============================================================
# 输入目录：这里应放置 fujian_1981_pre_CHIRPS_daily.nc 至 fujian_2025_pre_CHIRPS_daily.nc。
INPUT_DIR = Path(r"E:\forest_microclimate\ForestMicroclimate\Precipitation_CHIRPS\daily\fujian_pre_daily")

# 输出目录：NC、统计表、验证图都会保存到这里。
OUTPUT_DIR = Path(r"E:\forest_microclimate\ForestMicroclimate\results\daily_SPI20_result")

# 临时缓存目录：如运行过程中产生缓存，脚本结束后会自动删除。
TEMP_DIR = OUTPUT_DIR / "本次运行临时缓存"

# NC 文件中的降水变量名。当前福建 CHIRPS 文件中变量名为 precip。
PRECIP_VAR = "precip"

# 年份设置：
#   DATA_START_YEAR：输入日序列起始年份，必须为完整年份。
#   CALIB_START_YEAR/CALIB_END_YEAR：Gamma 分布拟合校准期，不包含 2025。
#   TARGET_YEAR：只导出这一年的逐日 SPI。
DATA_START_YEAR = 1981
CALIB_START_YEAR = 1981
CALIB_END_YEAR = 2024
TARGET_YEAR = 2025

# SPI1（日尺度）采用 20 天滑动累计降雨；按你的要求不计算 90 天或更长尺度。
SPI_SCALE_DAYS = 20

# climate_indices 的 scale 参数：
#   本脚本会先在真实连续日期上自行完成 30 天 rolling sum。
#   因此传给 climate_indices.indices.spi() 的序列已经是 30 天累计降雨，
#   这里必须使用 scale=1，只做 Gamma 分布标准化，不能再次 rolling。
CLIMATE_INDICES_SCALE = 1

# 输出文件。按要求：NC 文件名保留英文，其它图表文件使用中文命名。
OUT_NC = OUTPUT_DIR / f"Fujian_daily_SPI20d_{TARGET_YEAR}.nc"
OUT_CSV = OUTPUT_DIR / f"逐日SPI20d全省统计_{TARGET_YEAR}.csv"
OUT_FIG = OUTPUT_DIR / f"福建省逐日平均SPI20d时间序列图_{TARGET_YEAR}.png"
OUT_QUANTILE_FIG = OUTPUT_DIR / f"SPI20d全省极值与分位数图_{TARGET_YEAR}.png"
OUT_DROUGHT_RATIO_FIG = OUTPUT_DIR / f"SPI20d干旱面积比例图_{TARGET_YEAR}.png"
OUT_METHOD_XLSX = OUTPUT_DIR / "Daily_SPI20d计算方法说明.xlsx"

# 真实站点坐标表：必须包含 Site_ID、Longitude、Latitude 三列。
SITE_CSV = Path(r"E:\forest_microclimate\ForestMicroclimate\Tensor_LatLong.csv")
OUT_SITE_SPI_XLSX = OUTPUT_DIR / f"各站点SPI20d逐日宽表_{TARGET_YEAR}.xlsx"
SITE_FIG_DIR = OUTPUT_DIR / f"各站点SPI20d时间序列图_{TARGET_YEAR}"

# 进度条显示参数：
#   dynamic_ncols=True：根据终端宽度自动调整长度。
#   leave=True：关键步骤完成后保留一行摘要，方便回看耗时和速度。
#   ncols=None：不固定宽度，让 tqdm 自行计算。
#   ascii=False：允许显示彩色 Unicode 进度条。
PROGRESS_DYNAMIC_NCOLS = True
PROGRESS_LEAVE = True

# 验证图整体参数：
#   FIG_SIZE 控制图片宽高，DPI 控制清晰度。
#   FONT_FAMILY 是中文字体候选列表，前面的字体优先使用。
#   DATE_FORMAT / MONTH_INTERVAL 控制横轴日期显示。
FIG_SIZE = (16, 6.2)
STAT_FIG_SIZE = (16, 6.7)
SITE_FIG_SIZE = (14, 5.5)
FIG_DPI = 220
FONT_FAMILY = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
DATE_FORMAT = "%Y-%m"
MONTH_INTERVAL = 1

# SPI 柱状图参数：
#   BAR_WIDTH 是柱宽，BAR_ALPHA 是透明度。
#   BAR_COLOR_DRY 用于 SPI<0，BAR_COLOR_WET 用于 SPI>=0。
BAR_WIDTH = 1.0
BAR_ALPHA = 0.68
BAR_COLOR_DRY = "tomato"
BAR_COLOR_WET = "steelblue"

# 阈值线参数：
#   每个元组格式为：(阈值, 颜色, 线型, 线宽, 图例文字)。
#   后续若要改干旱等级颜色、线型或阈值，优先改这里。
THRESHOLD_LINES = [
    (-0.5, "orange", "--", 0.9, "Light drought (-0.5)"),
    (-1.0, "red", "--", 0.9, "Moderate drought (-1.0)"),
    (-1.5, "darkred", "-.", 0.9, "Severe drought (-1.5)"),
    (-2.0, "black", ":", 0.9, "Extreme drought (-2.0)"),
]

# 坐标轴、网格和图例参数：
#   MEAN_SPI_Y_LIMITS 控制主图和站点图 SPI 纵轴范围；None 表示根据各自数据自动设置。
#   STAT_SPI_Y_LIMITS 控制分位数/极值图纵轴范围；None 表示根据数据自动设置。
#   自动范围会同时考虑实际 SPI 值和干旱阈值线，并额外留出少量边距。
#   ZERO_LINE_* 控制 SPI=0 基准线。
#   LEGEND_* 控制图例位置和字号；列数由图例项数量自动决定，保证单行显示。
MEAN_SPI_Y_LIMITS = None
STAT_SPI_Y_LIMITS = None
ZERO_LINE_COLOR = "gray"
ZERO_LINE_WIDTH = 0.7
GRID_ALPHA = 0.3
LEGEND_LOCATION = "lower center"
LEGEND_FONT_SIZE = 10
LEGEND_BOTTOM_ANCHOR = 0.075
FIG_BOTTOM_MARGIN = 0.20
TIGHT_LAYOUT_RECT = (0.0, 0.10, 1.0, 1.0)
XTICK_ROTATION = 30
POSITIVE_SPI_LABEL = "SPI20d >= 0"
NEGATIVE_SPI_LABEL = "SPI20d < 0"

# 字体大小参数：
#   TITLE_FONT_SIZE 控制图标题；AXIS_LABEL_FONT_SIZE 控制横纵轴标题；
#   TICK_LABEL_FONT_SIZE 控制刻度文字。
TITLE_FONT_SIZE = 14
AXIS_LABEL_FONT_SIZE = 12
TICK_LABEL_FONT_SIZE = 10

# 分位数图参数：
#   PERCENTILE_LEVELS 控制输出哪些全省空间分位数。
#   PERCENTILE_BAND_ALPHA 控制 P5-P95 阴影带透明度。
PERCENTILE_LEVELS = [5, 50, 95]
PERCENTILE_BAND_ALPHA = 0.18
PERCENTILE_LINE_WIDTH = 1.25

# 干旱面积比例图参数：
#   面积比例取值为 0-1；若想显示百分比，可在绘图时乘以 100。
DROUGHT_RATIO_LINE_WIDTH = 1.35
DROUGHT_STACK_ALPHA = 0.86


# =============================================================
# 2. 工具函数
# =============================================================
def progress_bar(iterable=None, *, total=None, desc: str, unit: str, colour: str):
    """
    创建统一风格的 tqdm 单行动态进度条。

    参数说明：
      iterable：需要迭代的对象；如果只想手动 update，可传 None 并指定 total。
      total：总量，用于手动 update 的进度条。
      desc：进度条左侧说明文字，例如“读取并合并逐年NC”。
      unit：当前量/总量的单位，例如 file、day、grid、fig。
      colour：进度条颜色，不同步骤使用不同颜色便于区分。
    """
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        colour=colour,
        dynamic_ncols=PROGRESS_DYNAMIC_NCOLS,
        leave=PROGRESS_LEAVE,
        ascii=False,
    )


def silence_third_party_logs() -> None:
    """
    关闭第三方库的终端 info 日志，避免干扰 tqdm 单行进度条。

    背景：
      climate_indices 在逐网格调用 indices.spi() 时会输出 calculation_started、
      distribution_fitting_started 等 info 日志。网格很多时这些日志会刷屏，
      使 tqdm 的单行动态进度条失效。

    调整方式：
      1. 将 climate_indices 相关 logger 提高到 ERROR 级别。
      2. 禁止它们向根 logger 继续传播。
      3. 保留 ERROR 级别，真正异常仍可通过脚本的 failed_count 汇总体现。
    """
    logging.getLogger().setLevel(logging.WARNING)
    for logger_name in (
        "climate_indices",
        "climate_indices.indices",
        "climate_indices.compute",
    ):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        logger.addHandler(logging.NullHandler())


@contextlib.contextmanager
def suppress_library_console_output():
    """
    临时抑制第三方库直接写入 stdout/stderr 的内容。

    tqdm 默认写入 stderr。为避免同时屏蔽进度条，这个上下文只包住单次
    indices.spi() 调用，不包住外层 tqdm 循环本身。
    """
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def print_step(title: str) -> None:
    """打印清晰的步骤标题，方便查看终端日志。"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def extract_year_from_name(path: Path) -> int | None:
    """从文件名中提取 4 位年份，例如 fujian_1981_pre_CHIRPS_daily.nc -> 1981。"""
    match = re.search(r"(\d{4})", path.name)
    return int(match.group(1)) if match else None


def normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    """统一经纬度维度名，兼容 latitude/longitude 和 lat/lon 两种命名。"""
    rename_map = {}
    if "latitude" in ds.coords or "latitude" in ds.dims:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.coords or "longitude" in ds.dims:
        rename_map["longitude"] = "lon"
    return ds.rename(rename_map) if rename_map else ds


def day_slot_366(date: pd.Timestamp) -> int:
    """
    返回 climate_indices daily 模式中的 366 天年内位置，范围为 0-365。

    映射规则：
      - 闰年：直接使用真实 dayofyear，2 月 29 日占第 60 个位置。
      - 非闰年：2 月 29 日位置留空，3 月 1 日及之后整体后移 1 位。

    这样可以保证所有年份的 3 月 1 日都落在同一个 climatological day 位置。
    """
    doy = int(date.dayofyear)
    if (not date.is_leap_year) and (date.month > 2):
        doy += 1
    return doy - 1


def collect_nc_files() -> list[Path]:
    """检查并收集 1981-2025 年所有逐日 NC 文件。"""
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"输入目录不存在：{INPUT_DIR}")

    all_files = sorted(INPUT_DIR.glob("*.nc"))
    if not all_files:
        raise FileNotFoundError(f"输入目录中没有 NC 文件：{INPUT_DIR}")

    year_to_file: dict[int, Path] = {}
    for path in progress_bar(all_files, desc="检查NC文件年份", unit="file", colour="green"):
        year = extract_year_from_name(path)
        if year is not None:
            year_to_file[year] = path

    required_years = list(range(DATA_START_YEAR, TARGET_YEAR + 1))
    missing_years = [year for year in required_years if year not in year_to_file]
    if missing_years:
        missing_text = ", ".join(str(year) for year in missing_years)
        raise FileNotFoundError(f"缺少以下年份 NC 文件：{missing_text}")

    return [year_to_file[year] for year in required_years]


def open_and_combine(files: list[Path]) -> xr.Dataset:
    """
    逐年读取 NC 文件并合并。

    这里不用单个 INPUT_NC，因为你的数据是逐年文件夹组织。
    每个文件较小，直接 load 后合并可以避免后续文件句柄长期占用。
    """
    datasets = []
    for path in progress_bar(files, desc="读取并合并逐年NC", unit="file", colour="cyan"):
        ds_one = xr.open_dataset(path)
        ds_one = normalize_dataset(ds_one)

        if PRECIP_VAR not in ds_one.data_vars:
            raise KeyError(f"{path.name} 中未找到降水变量：{PRECIP_VAR}")
        for dim in ("time", "lat", "lon"):
            if dim not in ds_one.dims:
                raise KeyError(f"{path.name} 缺少必要维度：{dim}")

        datasets.append(ds_one[[PRECIP_VAR]].load())
        ds_one.close()

    ds = xr.concat(
        datasets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        combine_attrs="override",
    ).sortby("time")

    return ds


def check_time_axis(times: pd.DatetimeIndex) -> None:
    """检查合并后的日尺度时间轴是否完整、连续、无重复。"""
    expected_start = pd.Timestamp(f"{DATA_START_YEAR}-01-01")
    expected_end = pd.Timestamp(f"{TARGET_YEAR}-12-31")

    if times[0] != expected_start or times[-1] != expected_end:
        raise ValueError(f"时间范围应为 {expected_start.date()} 至 {expected_end.date()}，实际为 {times[0].date()} 至 {times[-1].date()}")
    if times.has_duplicates:
        raise ValueError("时间轴存在重复日期，请检查输入 NC。")

    full_times = pd.date_range(expected_start, expected_end, freq="D")
    if len(times) != len(full_times) or not np.array_equal(times.values, full_times.values):
        missing = full_times.difference(times)
        example = ", ".join(str(v.date()) for v in missing[:10])
        raise ValueError(f"时间轴不连续，缺失日期示例：{example}")


def build_prerolled_366_array(P_raw: xr.DataArray, times: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """
    先在真实日历上计算 30 天累计降雨，再重组为 climate_indices daily 需要的 366 天/年序列。

    为什么必须先 rolling 再插入 2 月 29 日占位：
      climate_indices daily 输入需要每年 366 个位置，非闰年的 2 月 29 日位置为 NaN。
      如果先把原始日降雨放进 366 日历，再让 indices.spi(scale=30) 内部 rolling，
      那么 3 月 1 日到 3 月 29 日的 30 天窗口都会碰到这个 NaN 占位，
      结果会整段传播为 NaN。

    本函数的处理原则：
      1. 在真实连续日期 time 轴上做 rolling(time=30)，真实时间轴没有非闰年 2 月 29 日。
      2. rolling 天数仍然严格是 30 个真实日，跨年窗口也正常保留。
      3. rolling 完成后，再把累计结果映射到 366 日历。
      4. 非闰年 2 月 29 日位置仅作为占位保持 NaN，不参与 rolling。

    返回：
      P_366:
        形状为 (n_years, 366, n_lat, n_lon)。
        值已经是 30 天累计降雨，而不是原始日降雨。
        非闰年的 2 月 29 日位置为 NaN，省界外网格也保持 NaN。
      target_slots:
        2025 年真实日期对应到展平数组中的位置，用于从 SPI 结果中抽取真实 365 天。
    """
    years = np.arange(DATA_START_YEAR, TARGET_YEAR + 1)
    year_to_index = {year: idx for idx, year in enumerate(years)}

    n_years = len(years)
    n_lat = P_raw.sizes["lat"]
    n_lon = P_raw.sizes["lon"]
    P_366 = np.full((n_years, 366, n_lat, n_lon), np.nan, dtype=np.float32)

    with progress_bar(total=1, desc=f"真实日期{SPI_SCALE_DAYS}天累计", unit="step", colour="yellow") as pbar:
        P_roll = (
            P_raw
            .rolling(time=SPI_SCALE_DAYS, min_periods=SPI_SCALE_DAYS)
            .sum()
            .astype("float32")
        )
        P_values = P_roll.values.astype(np.float32, copy=False)
        pbar.update(1)

    for time_index, date in enumerate(progress_bar(times, desc="累计值映射到366天日历", unit="day", colour="yellow")):
        yi = year_to_index[date.year]
        slot = day_slot_366(date)
        P_366[yi, slot, :, :] = P_values[time_index, :, :]

    target_dates = pd.date_range(f"{TARGET_YEAR}-01-01", f"{TARGET_YEAR}-12-31", freq="D")
    target_year_index = year_to_index[TARGET_YEAR]
    target_slots = np.array(
        [target_year_index * 366 + day_slot_366(date) for date in target_dates],
        dtype=np.int64,
    )

    return P_366, target_slots


def calculate_spi20_from_prerolled(P_366: np.ndarray) -> np.ndarray:
    """
    对预先累计好的 20 天降雨逐网格调用 climate_indices.indices.spi() 计算 SPI_30d。

    注意：
      - 不把 NaN 替换为 0。省界外 NaN 不是“无降雨”，而是无效网格。
      - 2 月 29 日占位 NaN 也不是 0 降雨，不能参与 Gamma 拟合。
      - 全时段均为 NaN 的网格直接跳过。
      - P_366 中的值已经是 30 天累计降雨，所以 indices.spi() 必须使用 scale=1。
    """
    n_years, n_days_per_year, n_lat, n_lon = P_366.shape
    n_total = n_years * n_days_per_year
    n_grids = n_lat * n_lon

    P_flat = P_366.reshape(n_total, n_grids)
    spi_flat = np.full((n_total, n_grids), np.nan, dtype=np.float32)

    failed_count = 0
    for grid_idx in progress_bar(range(n_grids), desc="计算逐网格SPI20d", unit="grid", colour="red"):
        precip_1d = P_flat[:, grid_idx].astype(float, copy=False)

        if np.all(np.isnan(precip_1d)):
            continue

        try:
            with suppress_library_console_output():
                spi_1d = indices.spi(
                    values=precip_1d,
                    scale=CLIMATE_INDICES_SCALE,
                    distribution=indices.Distribution.gamma,
                    data_start_year=DATA_START_YEAR,
                    calibration_year_initial=CALIB_START_YEAR,
                    calibration_year_final=CALIB_END_YEAR,
                    periodicity=compute.Periodicity.daily,
                )
            spi_flat[:, grid_idx] = spi_1d.astype(np.float32)
        except Exception:
            failed_count += 1
            continue

    if failed_count:
        print(f"提示：共有 {failed_count} 个网格计算失败，已保持为 NaN。")

    return spi_flat


def save_outputs(ds_out: xr.Dataset) -> None:
    """保存 NC、中文统计表和中文验证图。"""
    print_step("步骤6：保存结果文件")

    encoding = {
        "SPI_20d": {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        }
    }

    for _ in progress_bar(range(1), desc="保存NC文件", unit="file", colour="blue"):
        ds_out.to_netcdf(OUT_NC, encoding=encoding)
    print(f"NC 文件已保存：{OUT_NC}")

    spi = ds_out["SPI_20d"].values
    valid_mask = np.isfinite(spi)
    valid_count = valid_mask.sum(axis=(1, 2))
    dry_light_count = ((spi < -0.5) & valid_mask).sum(axis=(1, 2))
    dry_moderate_count = ((spi < -1.0) & valid_mask).sum(axis=(1, 2))
    dry_severe_count = ((spi < -1.5) & valid_mask).sum(axis=(1, 2))
    dry_extreme_count = ((spi < -2.0) & valid_mask).sum(axis=(1, 2))
    dry_light_ratio = np.divide(
        dry_light_count,
        valid_count,
        out=np.full(valid_count.shape, np.nan, dtype=np.float64),
        where=valid_count > 0,
    )
    dry_moderate_ratio = np.divide(
        dry_moderate_count,
        valid_count,
        out=np.full(valid_count.shape, np.nan, dtype=np.float64),
        where=valid_count > 0,
    )
    dry_severe_ratio = np.divide(
        dry_severe_count,
        valid_count,
        out=np.full(valid_count.shape, np.nan, dtype=np.float64),
        where=valid_count > 0,
    )
    dry_extreme_ratio = np.divide(
        dry_extreme_count,
        valid_count,
        out=np.full(valid_count.shape, np.nan, dtype=np.float64),
        where=valid_count > 0,
    )
    spi_p5 = np.nanpercentile(spi, 5, axis=(1, 2))
    spi_p50 = np.nanpercentile(spi, 50, axis=(1, 2))
    spi_p95 = np.nanpercentile(spi, 95, axis=(1, 2))

    stats_df = pd.DataFrame(
        {
            "日期": pd.DatetimeIndex(ds_out["time"].values).strftime("%Y-%m-%d"),
            "全省平均SPI20d": np.nanmean(spi, axis=(1, 2)),
            "全省最小SPI20d": np.nanmin(spi, axis=(1, 2)),
            "全省最大SPI20d": np.nanmax(spi, axis=(1, 2)),
            "全省P5_SPI20d": spi_p5,
            "全省P50_SPI20d": spi_p50,
            "全省P95_SPI20d": spi_p95,
            "轻旱及以上面积比例_SPI小于-0.5": dry_light_ratio,
            "中旱及以上面积比例_SPI小于-1.0": dry_moderate_ratio,
            "重旱及以上面积比例_SPI小于-1.5": dry_severe_ratio,
            "特旱面积比例_SPI小于-2.0": dry_extreme_ratio,
            "有效网格数": valid_count,
        }
    )
    for _ in progress_bar(range(1), desc="保存中文统计表", unit="file", colour="magenta"):
        stats_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"统计表已保存：{OUT_CSV}")

    make_mean_spi_plot(stats_df)
    make_quantile_plot(stats_df)
    make_drought_ratio_plot(stats_df)
    save_method_workbook()
    save_site_spi_outputs(ds_out)


def save_method_workbook() -> None:
    """
    生成脚本方法说明 Excel。

    这个文件用于组会汇报和脚本交接，重点解释“为什么这么算”，而不是只列代码。
    使用 Excel 而不是 Word 的原因是当前解释器已安装 openpyxl，但未安装 python-docx。
    """
    workflow_rows = [
        ("1. 数据输入", "读取 1981-2025 年福建省 CHIRPS 逐日降雨 NC。", "保证有足够长的历史样本拟合 SPI 分布。"),
        ("2. 负值处理", "仅将真实负降雨裁剪为 0，保留 NaN 掩膜。", "CHIRPS 极小负值是噪声；省界外 NaN 不能当成 0 降雨。"),
        ("3. 30天累计", "在真实连续日期上 rolling(time=30) 求和。", "daily SPI1 通常表达近 1 个月累计降雨异常，因此用 30 个真实日。"),
        ("4. 闰日处理", "rolling 完成后再映射到每年 366 个位置，非闰年 2月29日为 NaN。", "避免 2月29日占位 NaN 污染 3月1日-3月29日的 30 天窗口。"),
        ("5. 标准化", "把预先累计好的 30天降雨传入 climate_indices，scale=1。", "输入已是 30天累计值，scale=1 表示只做 Gamma 分布标准化，不能重复 rolling。"),
        ("6. 校准期", "使用 1981-2024 拟合 Gamma 分布，2025 只用于评估。", "避免目标年参与分布拟合导致评价偏差。"),
        ("7. 输出", "输出 2025 年逐日 SPI20d NC、统计表、图和站点宽表。", "同时满足栅格分析、省域概览和站点干旱事件识别。"),
    ]
    concept_rows = [
        ("为什么不是单日 SPI？", "单日降雨零值多、随机性强，直接标准化不稳定；30天累计更接近短期农业/生态干旱过程。"),
        ("为什么叫 SPI20d？", "因为每一天的 SPI 是基于该日及此前 29 个真实日累计降雨计算得到。"),
        ("为什么用全省平均 SPI20d 作主图？", "主图用于表达福建省整体干湿过程；平均值比单点更代表省域总体状态。"),
        ("为什么还输出分位数图？", "分位数图展示空间异质性：平均偏干时，是全省都干，还是局地特别干。"),
        ("为什么还输出干旱面积比例？", "面积比例回答干旱影响范围，即多少比例网格达到轻旱、中旱、重旱或特旱。"),
        ("为什么站点值叫邻近网格值？", "CHIRPS 是栅格数据，站点 SPI 是按站点坐标提取最近栅格的 SPI20d，不是实测站降雨直接计算。"),
    ]
    output_rows = [
        ("Fujian_daily_SPI20d_2025.nc", "2025 年福建省逐日 SPI20d 栅格结果。"),
        ("逐日SPI20d全省统计_2025.csv", "每日省平均、极值、分位数、干旱面积比例和有效网格数。"),
        ("福建省逐日平均SPI20d时间序列图_2025.png", "省域整体干湿变化主图。"),
        ("SPI20d全省极值与分位数图_2025.png", "空间分布范围和异质性辅助图。"),
        ("SPI20d干旱面积比例图_2025.png", "不同干旱等级影响范围图。"),
        ("各站点SPI20d逐日宽表_2025.xlsx", "站点编号为列、日期为行的逐日 SPI20d 宽表。"),
        ("各站点SPI20d时间序列图_2025", "逐站点邻近网格 SPI20d 时间序列图文件夹。"),
    ]
    qa_rows = [
        ("问：为什么 2025 年不参与校准？", "答：SPI 是相对历史分布的异常度，目标年参与校准会削弱异常信号。"),
        ("问：为什么非闰年 2月29日不能填 0？", "答：这一天不存在，填 0 会人为制造一天无雨，影响 30天累计。"),
        ("问：为什么先 rolling 再转 366 日历？", "答：rolling 应在真实日期上完成，否则 2月29日占位 NaN 会造成 3月1日后约 29 天空值。"),
        ("问：站点 SPI 是否等于站点实测 SPI？", "答：不是。这里是站点坐标对应的 CHIRPS 最近网格 SPI20d，用于与站点位置关联分析。"),
        ("问：全省平均 SPI 低代表什么？", "答：代表省域整体短期降雨偏少；是否大范围干旱需结合干旱面积比例和分位数图判断。"),
    ]

    with progress_bar(range(1), desc="保存方法说明Excel", unit="file", colour="magenta"):
        with pd.ExcelWriter(OUT_METHOD_XLSX, engine="openpyxl") as writer:
            pd.DataFrame(workflow_rows, columns=["步骤", "具体操作", "为什么这样做"]).to_excel(writer, sheet_name="计算流程", index=False)
            pd.DataFrame(concept_rows, columns=["问题", "解释"]).to_excel(writer, sheet_name="关键概念", index=False)
            pd.DataFrame(output_rows, columns=["输出文件", "用途"]).to_excel(writer, sheet_name="输出说明", index=False)
            pd.DataFrame(qa_rows, columns=["导师可能会问", "建议回答"]).to_excel(writer, sheet_name="组会问答", index=False)
    print(f"方法说明Excel已保存：{OUT_METHOD_XLSX}")


def load_site_table() -> pd.DataFrame:
    """读取真实站点坐标表，并检查必要字段。"""
    if not SITE_CSV.exists():
        raise FileNotFoundError(f"未找到站点坐标表：{SITE_CSV}")
    sites = pd.read_csv(SITE_CSV, dtype={"Site_ID": str})
    required_columns = {"Site_ID", "Longitude", "Latitude"}
    missing_columns = required_columns - set(sites.columns)
    if missing_columns:
        raise KeyError(f"站点坐标表缺少字段：{', '.join(sorted(missing_columns))}")
    sites = sites[["Site_ID", "Longitude", "Latitude"]].copy()
    sites["Longitude"] = pd.to_numeric(sites["Longitude"], errors="coerce")
    sites["Latitude"] = pd.to_numeric(sites["Latitude"], errors="coerce")
    sites = sites.dropna(subset=["Site_ID", "Longitude", "Latitude"])
    if sites.empty:
        raise ValueError("站点坐标表没有有效站点。")
    return sites


def save_site_spi_outputs(ds_out: xr.Dataset) -> None:
    """
    按真实站点坐标提取最近 CHIRPS 网格 SPI20d，并输出站点宽表和逐站点图。

    注意：
      CHIRPS 是栅格数据，站点坐标通常不完全落在网格中心。
      这里使用 method='nearest' 提取最近网格值，所以图和表代表“站点邻近网格 SPI20d”。
    """
    sites = load_site_table()
    times = pd.DatetimeIndex(ds_out["time"].values)
    site_spi = {"Date": times.strftime("%Y-%m-%d")}
    nearest_info = []

    SITE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    spi_da = ds_out["SPI_20d"]
    for _, row in progress_bar(sites.iterrows(), total=len(sites), desc="提取并绘制站点SPI20d", unit="site", colour="cyan"):
        site_id = str(row["Site_ID"])
        lon = float(row["Longitude"])
        lat = float(row["Latitude"])
        site_series_da = spi_da.sel(lon=lon, lat=lat, method="nearest")
        site_series = site_series_da.values.astype(float)
        nearest_lon = float(site_series_da["lon"].values)
        nearest_lat = float(site_series_da["lat"].values)
        site_spi[site_id] = site_series
        nearest_info.append(
            {
                "Site_ID": site_id,
                "Longitude": lon,
                "Latitude": lat,
                "Nearest_lon": nearest_lon,
                "Nearest_lat": nearest_lat,
            }
        )
        make_single_site_plot(site_id, times, site_series)

    with progress_bar(range(1), desc="保存站点SPI20d宽表", unit="file", colour="blue"):
        with pd.ExcelWriter(OUT_SITE_SPI_XLSX, engine="openpyxl") as writer:
            pd.DataFrame(site_spi).to_excel(writer, sheet_name="站点SPI20d宽表", index=False)
            pd.DataFrame(nearest_info).to_excel(writer, sheet_name="站点最近网格信息", index=False)
    print(f"站点SPI20d宽表已保存：{OUT_SITE_SPI_XLSX}")
    print(f"站点SPI20d时间序列图目录：{SITE_FIG_DIR}")


def make_single_site_plot(site_id: str, times: pd.DatetimeIndex, site_series: np.ndarray) -> None:
    """绘制单个站点邻近网格 SPI20d 时间序列图。"""
    y_limits = get_spi_y_limits(site_series, fixed_limits=MEAN_SPI_Y_LIMITS)
    fig, ax = plt.subplots(1, 1, figsize=SITE_FIG_SIZE)
    colors = np.where(site_series < 0, BAR_COLOR_DRY, BAR_COLOR_WET)
    ax.bar(times, site_series, color=colors, width=BAR_WIDTH, alpha=BAR_ALPHA)
    add_drought_threshold_lines(ax)
    ax.set_title(f"{site_id} daily_SPI Time Series", fontsize=TITLE_FONT_SIZE)
    ax.set_ylabel("SPI20d", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xlabel("Date", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylim(*y_limits)
    ax.grid(alpha=GRID_ALPHA)
    set_month_axis(ax)
    plt.xticks(rotation=XTICK_ROTATION)
    fig.legend(
        handles=build_spi_legend_handles(),
        loc=LEGEND_LOCATION,
        ncol=len(build_spi_legend_handles()),
        fontsize=LEGEND_FONT_SIZE,
        bbox_to_anchor=(0.5, LEGEND_BOTTOM_ANCHOR),
        frameon=False,
    )
    fig.subplots_adjust(bottom=FIG_BOTTOM_MARGIN)
    fig.tight_layout(rect=TIGHT_LAYOUT_RECT)
    fig.savefig(SITE_FIG_DIR / f"{site_id}_SPI20d时间序列.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def build_spi_legend_handles() -> list:
    """构造 SPI 主图图例句柄，避免柱状图正负两种颜色只显示一个标签。"""
    legend_handles = [
        Patch(facecolor=BAR_COLOR_WET, alpha=BAR_ALPHA, label=POSITIVE_SPI_LABEL),
        Patch(facecolor=BAR_COLOR_DRY, alpha=BAR_ALPHA, label=NEGATIVE_SPI_LABEL),
    ]
    legend_handles.extend(
        Line2D([0], [0], color=color, linestyle=linestyle, linewidth=linewidth, label=label)
        for threshold, color, linestyle, linewidth, label in THRESHOLD_LINES
    )
    return legend_handles


def set_month_axis(ax) -> None:
    """统一设置横轴月份刻度，便于多个输出图保持一致。"""
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=MONTH_INTERVAL))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(DATE_FORMAT))
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)


def add_drought_threshold_lines(ax) -> None:
    """添加干旱等级阈值线。"""
    ax.axhline(0, color=ZERO_LINE_COLOR, linewidth=ZERO_LINE_WIDTH)
    for threshold, color, linestyle, linewidth, label in THRESHOLD_LINES:
        ax.axhline(
            threshold,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )


def get_spi_y_limits(*series_list: np.ndarray, fixed_limits: tuple[float, float] | None = None) -> tuple[float, float]:
    """根据实际 SPI 序列和干旱阈值自动生成纵轴范围。"""
    finite_values = [
        np.asarray(series)[np.isfinite(series)]
        for series in series_list
        if np.asarray(series)[np.isfinite(series)].size > 0
    ]
    threshold_values = np.array([item[0] for item in THRESHOLD_LINES], dtype=float)
    if fixed_limits is not None:
        return fixed_limits
    if finite_values:
        all_values = np.concatenate(finite_values)
        y_min = min(float(np.nanmin(all_values)), float(np.nanmin(threshold_values))) - 0.25
        y_max = max(float(np.nanmax(all_values)), 0.0) + 0.25
        return (np.floor(y_min * 10) / 10, np.ceil(y_max * 10) / 10)
    return (-2.5, 1.5)


def make_mean_spi_plot(stats_df: pd.DataFrame) -> None:
    """绘制正式主图：福建省平均 SPI20d 时间序列。"""
    plt.rcParams["font.sans-serif"] = FONT_FAMILY
    plt.rcParams["axes.unicode_minus"] = False

    times = pd.to_datetime(stats_df["日期"])
    province_mean = stats_df["全省平均SPI20d"].to_numpy()
    y_limits = get_spi_y_limits(province_mean, fixed_limits=MEAN_SPI_Y_LIMITS)

    with progress_bar(total=1, desc="绘制省平均SPI20d主图", unit="fig", colour="yellow") as pbar:
        fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)
        colors = np.where(province_mean < 0, BAR_COLOR_DRY, BAR_COLOR_WET)
        ax.bar(times, province_mean, color=colors, width=BAR_WIDTH, alpha=BAR_ALPHA)
        add_drought_threshold_lines(ax)
        ax.set_title("福建省平均 SPI20d", fontsize=TITLE_FONT_SIZE)
        ax.set_ylabel("SPI20d", fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_xlabel("Date", fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_ylim(*y_limits)
        ax.grid(alpha=GRID_ALPHA)
        set_month_axis(ax)
        plt.xticks(rotation=XTICK_ROTATION)
        fig.legend(
            handles=build_spi_legend_handles(),
            loc=LEGEND_LOCATION,
            ncol=len(build_spi_legend_handles()),
            fontsize=LEGEND_FONT_SIZE,
            bbox_to_anchor=(0.5, LEGEND_BOTTOM_ANCHOR),
            frameon=False,
        )
        fig.subplots_adjust(bottom=FIG_BOTTOM_MARGIN)
        fig.tight_layout(rect=TIGHT_LAYOUT_RECT)
        fig.savefig(OUT_FIG, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        pbar.update(1)

    print(f"省平均SPI20d主图已保存：{OUT_FIG}")


def make_quantile_plot(stats_df: pd.DataFrame) -> None:
    """绘制全省 SPI20d 最小值、最大值和 P5/P50/P95 分位数图。"""
    times = pd.to_datetime(stats_df["日期"])
    spi_min = stats_df["全省最小SPI20d"].to_numpy()
    spi_max = stats_df["全省最大SPI20d"].to_numpy()
    spi_p5 = stats_df["全省P5_SPI20d"].to_numpy()
    spi_p50 = stats_df["全省P50_SPI20d"].to_numpy()
    spi_p95 = stats_df["全省P95_SPI20d"].to_numpy()
    y_limits = get_spi_y_limits(spi_min, spi_max, spi_p5, spi_p50, spi_p95, fixed_limits=STAT_SPI_Y_LIMITS)

    with progress_bar(total=1, desc="绘制全省极值与分位数图", unit="fig", colour="yellow") as pbar:
        fig, ax = plt.subplots(1, 1, figsize=STAT_FIG_SIZE)
        ax.fill_between(times, spi_p5, spi_p95, color="steelblue", alpha=PERCENTILE_BAND_ALPHA, label="P5-P95")
        ax.plot(times, spi_p50, color="navy", linewidth=PERCENTILE_LINE_WIDTH, label="P50")
        ax.plot(times, spi_min, color="darkred", linewidth=PERCENTILE_LINE_WIDTH, linestyle="--", label="Min")
        ax.plot(times, spi_max, color="darkgreen", linewidth=PERCENTILE_LINE_WIDTH, linestyle="--", label="Max")
        add_drought_threshold_lines(ax)
        ax.set_title("福建省 SPI20d 极值与分位数", fontsize=TITLE_FONT_SIZE)
        ax.set_ylabel("SPI20d", fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_xlabel("Date", fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_ylim(*y_limits)
        ax.grid(alpha=GRID_ALPHA)
        set_month_axis(ax)
        plt.xticks(rotation=XTICK_ROTATION)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles=handles,
            labels=labels,
            loc=LEGEND_LOCATION,
            ncol=len(handles),
            fontsize=LEGEND_FONT_SIZE,
            bbox_to_anchor=(0.5, LEGEND_BOTTOM_ANCHOR),
            frameon=False,
        )
        fig.subplots_adjust(bottom=FIG_BOTTOM_MARGIN)
        fig.tight_layout(rect=TIGHT_LAYOUT_RECT)
        fig.savefig(OUT_QUANTILE_FIG, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        pbar.update(1)

    print(f"全省极值与分位数图已保存：{OUT_QUANTILE_FIG}")


def make_drought_ratio_plot(stats_df: pd.DataFrame) -> None:
    """绘制互斥干旱等级面积比例堆叠图。"""
    times = pd.to_datetime(stats_df["日期"])
    light_or_worse = stats_df["轻旱及以上面积比例_SPI小于-0.5"].to_numpy(dtype=float)
    moderate_or_worse = stats_df["中旱及以上面积比例_SPI小于-1.0"].to_numpy(dtype=float)
    severe_or_worse = stats_df["重旱及以上面积比例_SPI小于-1.5"].to_numpy(dtype=float)
    extreme = stats_df["特旱面积比例_SPI小于-2.0"].to_numpy(dtype=float)

    # 将“及以上”累计比例拆成互斥等级，便于堆叠面积图表达不同等级的构成。
    # 例如轻旱等级 = SPI<-0.5 的比例 - SPI<-1.0 的比例。
    light_only = np.clip(light_or_worse - moderate_or_worse, 0, 1)
    moderate_only = np.clip(moderate_or_worse - severe_or_worse, 0, 1)
    severe_only = np.clip(severe_or_worse - extreme, 0, 1)
    extreme_only = np.clip(extreme, 0, 1)

    with progress_bar(total=1, desc="绘制干旱面积比例图", unit="fig", colour="yellow") as pbar:
        fig, ax = plt.subplots(1, 1, figsize=STAT_FIG_SIZE)
        ax.stackplot(
            times,
            light_only,
            moderate_only,
            severe_only,
            extreme_only,
            labels=[
                "Light only (-1.0 <= SPI < -0.5)",
                "Moderate only (-1.5 <= SPI < -1.0)",
                "Severe only (-2.0 <= SPI < -1.5)",
                "Extreme (SPI < -2.0)",
            ],
            colors=["#f6c85f", "#f08a4b", "#c0392b", "#3b3b3b"],
            alpha=DROUGHT_STACK_ALPHA,
        )
        total_drought = light_or_worse
        ax.plot(times, total_drought, color="black", linewidth=DROUGHT_RATIO_LINE_WIDTH, label="Total drought (SPI < -0.5)")
        ax.set_title("福建省 SPI20d 干旱面积比例", fontsize=TITLE_FONT_SIZE)
        ax.set_ylabel("Area ratio", fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_xlabel("Date", fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=GRID_ALPHA)
        set_month_axis(ax)
        plt.xticks(rotation=XTICK_ROTATION)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles=handles,
            labels=labels,
            loc=LEGEND_LOCATION,
            ncol=len(handles),
            fontsize=LEGEND_FONT_SIZE,
            bbox_to_anchor=(0.5, LEGEND_BOTTOM_ANCHOR),
            frameon=False,
        )
        fig.subplots_adjust(bottom=FIG_BOTTOM_MARGIN)
        fig.tight_layout(rect=TIGHT_LAYOUT_RECT)
        fig.savefig(OUT_DROUGHT_RATIO_FIG, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        pbar.update(1)

    print(f"干旱面积比例图已保存：{OUT_DROUGHT_RATIO_FIG}")


def clean_temp_dir() -> None:
    """删除本次脚本运行产生的临时缓存目录。"""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        print(f"已清理临时缓存目录：{TEMP_DIR}")


# =============================================================
# 3. 主流程
# =============================================================
def main() -> None:
    silence_third_party_logs()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(TEMP_DIR)
    os.environ["TEMP"] = str(TEMP_DIR)
    os.environ["TMP"] = str(TEMP_DIR)

    try:
        print_step("步骤1：收集并检查输入NC文件")
        files = collect_nc_files()
        print(f"已找到 {len(files)} 个文件：{files[0].name} 至 {files[-1].name}")

        print_step("步骤2：读取并合并1981-2025逐日降雨")
        ds = open_and_combine(files)
        print(ds)

        # 负降雨清零，但必须保留原始 NaN 掩膜。
        # 注意：xarray.where(cond, 0.0) 会把 cond 为 False 的 NaN 位置也填成 0，
        # 这会把省界外无效网格误当作真实 0 降雨。这里用 clip(min=0) 只修正负值，
        # NaN 会继续保持 NaN。
        P_raw = ds[PRECIP_VAR].clip(min=0)
        times_all = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
        check_time_axis(times_all)

        lats = ds["lat"].values
        lons = ds["lon"].values
        print(f"时间范围：{times_all[0].date()} 至 {times_all[-1].date()}，共 {len(times_all)} 天")
        print(f"空间范围：lat {float(np.nanmin(lats)):.3f} 至 {float(np.nanmax(lats)):.3f}，lon {float(np.nanmin(lons)):.3f} 至 {float(np.nanmax(lons)):.3f}")
        print(f"网格数量：{len(lats)} x {len(lons)} = {len(lats) * len(lons):,}")

        print_step("步骤3：真实日期30天累计并重组为climate_indices daily 366天日历")
        P_366, target_slots = build_prerolled_366_array(P_raw, times_all)
        print(f"重组后形状：{P_366.shape} = (年份数, 366, lat, lon)")
        print("说明：P_366 中的值已经是 20 天累计降雨，后续 climate_indices 使用 scale=1。")

        print_step("步骤4：调用climate_indices对30天累计降雨做Gamma标准化")
        spi_flat = calculate_spi20_from_prerolled(P_366)

        print_step("步骤5：提取2025真实日期并组织输出数据集")
        target_times = pd.date_range(f"{TARGET_YEAR}-01-01", f"{TARGET_YEAR}-12-31", freq="D")
        spi_2025 = spi_flat[target_slots, :].reshape(len(target_times), len(lats), len(lons))

        ds_out = xr.Dataset(
            data_vars={
                "SPI_20d": (
                    ("time", "lat", "lon"),
                    spi_2025,
                    {
                        "long_name": "Daily SPI1 based on 20-day accumulated precipitation",
                        "units": "dimensionless",
                        "scale_days": str(SPI_SCALE_DAYS),
                        "calibration": f"{CALIB_START_YEAR}-{CALIB_END_YEAR}",
                        "method": "20-day precipitation pre-rolled on real calendar; climate_indices.indices.spi scale=1; Gamma distribution; daily periodicity",
                    },
                )
            },
            coords={
                "time": target_times,
                "lat": lats,
                "lon": lons,
            },
            attrs={
                "title": f"Fujian daily SPI_20d for {TARGET_YEAR}",
                "source": "CHIRPS daily precipitation",
                "package": "climate_indices",
                "target_year": str(TARGET_YEAR),
                "calibration_period": f"{CALIB_START_YEAR}-{CALIB_END_YEAR}",
                "daily_calendar": "366 slots per year; non-leap Feb-29 is NaN after real-calendar rolling",
                "rolling_note": "20-day rolling sum is computed before inserting non-leap Feb-29 placeholders",
            },
        )

        save_outputs(ds_out)

        print_step("全部计算完成")
        print(f"结果目录：{OUTPUT_DIR}")

    finally:
        clean_temp_dir()


if __name__ == "__main__":
    main()
