"""
福建省 2025 年 daily SPI 干旱事件特征提取脚本。

整体目标
1. 从 daily SPI20d NetCDF 中读取 2025 年逐日 SPI 栅格数据。
2. 在每个经纬度格点上，按照逐日 SPI 时间序列识别全年所有有效干旱事件。
3. 将格点事件写入 NetCDF，保留空间栅格结构，便于后续 GIS 或栅格分析。
4. 将观测站匹配到最近 SPI 格点，输出站点尺度的事件长表、年度统计表和逐日 SPI 表。

核心事件定义
1. 干旱日：daily SPI < DROUGHT_THRESHOLD，当前阈值为 -0.5。
2. 有效干旱事件：连续干旱日数 >= MIN_DURATION_DAYS，当前为 6 天。
3. 事件等级：按事件期间最小 daily SPI 判定，即该事件达到的最严重等级。
4. 事件严重度：事件期间所有 daily SPI 绝对值之和，即 sum(abs(SPI))。
5. 空间范围：
   - Drought_Extent_Peak(%)：事件最小 SPI 出现当天，全省处于干旱状态的有效格点比例。
   - Drought_Extent_Union(%)：事件起止日期内，至少有一天处于干旱状态的有效格点比例。

输出文件
1. fujian_daily_spi_drought_events_2025.nc
   全省格点事件 NC，结构为 event x lat x lon。
2. 福建省观测站2025年daily_SPI干旱事件长表.csv
   每一行代表一个站点的一次有效干旱事件。
3. 福建省观测站2025年daily_SPI干旱年度统计表.csv
   每一行代表一个站点的全年汇总统计。
4. 福建省观测站2025年daily_SPI逐日序列表.csv
   每一行代表一个站点的一天，保留 daily SPI 和是否为干旱日。
5. 福建省观测站2025年daily_SPI严重度-持续时间散点图.png
   一个点代表一个站点的一次有效干旱事件，用于查看持续时间和严重度关系。
6. 福建省观测站2025年daily_SPI严重度-持续时间散点图说明.png
   单独保存图注级别说明，便于论文制图或报告排版时引用。
7. 敏感性分析输出文件
   比较 Nearest、Bilinear、Mean3x3 三种站点取值方法对 daily SPI 和干旱事件指标的影响。

边界说明
1. 当前只使用 2025-01-01 到 2025-12-31 的数据。
2. 如果事件从 2024 年延续到 2025 年，或从 2025 年延续到 2026 年，
   脚本无法看到窗口外的数据，因此会用 Edge_Truncated 标记可能被年度边界截断的事件。
3. 本脚本只绘制站点严重度-持续时间散点图，不绘制空间分布图。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


# ============================================================
# 0. 路径参数
# 本区集中管理输入、输出和运行缓存路径。
# 若未来更换年份、数据目录或输出目录，优先修改本区变量。
# ============================================================
# SPI_FILE：
# 输入 daily SPI NetCDF 文件。
# 当前脚本要求该文件至少包含：
# 1. 变量 SPI_VAR，当前为 "SPI_20d"；
# 2. 时间维度 TIME_DIM，当前为 "time"；
# 3. 纬度和经度坐标 LAT_NAME / LON_NAME，当前为 "lat" / "lon"；
# 4. 2025-01-01 到 2025-12-31 的逐日数据。
SPI_FILE = Path(
    r"E:\forest_microclimate\ForestMicroclimate\results\daily_SPI20_result\Fujian_daily_SPI20d_2025.nc"
)

# OUTPUT_DIR：
# 所有正式输出文件写入该目录。
# 脚本会自动创建该目录；不会主动删除该目录下已有正式结果。
OUTPUT_DIR = Path(r"E:\forest_microclimate\ForestMicroclimate\results\daily_SPI20_features")

# SITE_CSV：
# 观测站点经纬度表。
# 必须包含 SITE_ID_FIELD、SITE_LON_FIELD、SITE_LAT_FIELD 三列。
# 每个站点会通过最近邻方式匹配到一个 SPI 栅格点。
SITE_CSV = Path(r"E:\forest_microclimate\ForestMicroclimate\Tensor_LatLong.csv")

# OUTPUT_GRID_EVENT_NC：
# 全省格点事件 NetCDF。
# 因每个格点事件数量可能不同，脚本使用 event x lat x lon 结构保存；
# 某个格点不足的事件序号位置使用 NaN 填充。
OUTPUT_GRID_EVENT_NC = OUTPUT_DIR / "fujian_daily_spi20_drought_events_2025.nc"

# OUTPUT_SITE_EVENT_CSV：
# 站点事件长表。
# 一行代表一个站点的一次有效干旱事件，同一站点全年多次事件会占多行。
OUTPUT_SITE_EVENT_CSV = OUTPUT_DIR / "福建省观测站2025年daily_SPI20干旱事件长表.csv"

# OUTPUT_SITE_ANNUAL_CSV：
# 站点年度统计表。
# 一行代表一个站点，汇总该站点全年 SPI、干旱日数、事件次数和最严重事件等指标。
OUTPUT_SITE_ANNUAL_CSV = OUTPUT_DIR / "福建省观测站2025年daily_SPI20干旱年度统计表.csv"

# OUTPUT_SITE_DAILY_CSV：
# 站点逐日 SPI 序列表。
# 一行代表一个站点的一天，便于后续按任意时段重新筛选或核查事件。
OUTPUT_SITE_DAILY_CSV = OUTPUT_DIR / "福建省观测站2025年daily_SPI20逐日序列表.csv"

# OUTPUT_SCATTER_PNG：
# 站点事件严重度-持续时间散点图。
# 一点代表一个站点的一次有效干旱事件，横轴为 Duration_Days，纵轴为 Severity。
OUTPUT_SCATTER_PNG = OUTPUT_DIR / "福建省观测站2025年daily_SPI20严重度-持续时间散点图.png"

# OUTPUT_SCATTER_NOTE_PNG：
# 站点严重度-持续时间散点图说明。
# 单独输出为 PNG，记录点、颜色、形状、参考线和事件定义的解释。
OUTPUT_SCATTER_NOTE_PNG = OUTPUT_DIR / "福建省观测站2025年daily_SPI20严重度-持续时间散点图说明.png"

# 敏感性分析输出：
# 目的不是替代主分析，而是评估站点取值方法选择对结果的影响。
OUTPUT_SENSITIVITY_DAILY_CSV = OUTPUT_DIR / "福建省观测站2025年daily_SPI20取值方法逐日对比表.csv"
OUTPUT_SENSITIVITY_EVENT_CSV = OUTPUT_DIR / "福建省观测站2025年daily_SPI20敏感性分析事件长表.csv"
OUTPUT_SENSITIVITY_ANNUAL_CSV = OUTPUT_DIR / "福建省观测站2025年daily_SPI20敏感性分析年度统计表.csv"
OUTPUT_SENSITIVITY_DELTA_CSV = OUTPUT_DIR / "福建省观测站2025年daily_SPI20敏感性分析差异表.csv"
OUTPUT_SENSITIVITY_SUMMARY_CSV = OUTPUT_DIR / "福建省观测站2025年daily_SPI20敏感性分析总体统计表.csv"
OUTPUT_SENSITIVITY_FIG_PNG = OUTPUT_DIR / "福建省观测站2025年daily_SPI20敏感性分析图.png"

# RUN_CACHE_DIR / TEMP_DIR：
# 本次脚本运行期间使用的缓存目录。
# configure_runtime_cache() 会把 Python 临时目录指向 TEMP_DIR；
# cleanup_runtime_cache() 会在 finally 中删除 RUN_CACHE_DIR。
RUN_CACHE_DIR = OUTPUT_DIR / "_本次运行缓存_提取daily_SPI20干旱事件"
TEMP_DIR = RUN_CACHE_DIR / "temp"


# ============================================================
# 1. 数据维度与变量参数
# 本区用于适配 NetCDF 和站点 CSV 的字段名。
# 如果未来输入文件的变量名或字段名变化，通常只需要改本区。
# ============================================================
# SPI_VAR：
# NetCDF 中 daily SPI 数据变量名。
# 当前 daily SPI 计算脚本输出的是 SPI_20d，表示 200 日累计降水对应的逐日 SPI。
SPI_VAR = "SPI_20d"

# TIME_DIM / LAT_NAME / LON_NAME：
# NetCDF 中时间、纬度、经度维度或坐标的名称。
# 后续读取、切片、组装输出 NC 都依赖这些名称。
TIME_DIM = "time"
LAT_NAME = "lat"
LON_NAME = "lon"

# SITE_ID_FIELD / SITE_LON_FIELD / SITE_LAT_FIELD：
# 站点 CSV 中站点编号、经度、纬度字段名。
# 站点匹配采用经纬度最近邻，不做插值。
SITE_ID_FIELD = "Site_ID"
SITE_LON_FIELD = "Longitude"
SITE_LAT_FIELD = "Latitude"


# ============================================================
# 2. 干旱识别参数
# 本区参数直接决定“哪些天是干旱日”和“哪些连续片段算有效干旱事件”。
# ============================================================
# YEAR：
# 分析年份，仅用于语义说明和输出命名时保持一致。
# 实际时间切片由 WINDOW_START 和 WINDOW_END 控制。
YEAR = 2025

# WINDOW_START / WINDOW_END：
# 从输入 daily SPI 文件中截取的分析时间范围。
# 当前为完整 2025 年；如果输入数据包含多年，可以通过这两个参数切换年份。
WINDOW_START = "2025-01-01"
WINDOW_END = "2025-12-31"

# DROUGHT_THRESHOLD：
# 干旱日判定阈值。
# daily SPI < -0.5 视为干旱日；等于 -0.5 不计入干旱日。
DROUGHT_THRESHOLD = -0.5

# MIN_DURATION_DAYS：
# 有效干旱事件的最短持续天数。
# 用户要求“大于 5 天”才算事件，因此这里设置为 6。
# extract_drought_events() 中实际判断为 duration_days >= MIN_DURATION_DAYS。
MIN_DURATION_DAYS = 6

# DROUGHT_LEVELS：
# 干旱等级编码到英文名称的映射。
# 编码用于 NC 或机器读取，英文名称用于 CSV、图例和论文图件展示。
# 等级判定口径见 classify_drought_level()。
DROUGHT_LEVELS = {
    1: "Light",
    2: "Moderate",
    3: "Severe",
    4: "Extreme",
}


# ============================================================
# 3. 进度条参数
# 本区只影响终端显示，不影响计算结果。
# ============================================================
# TQDM_BAR_FORMAT：
# 统一进度条格式，显示描述、进度条、当前/总量、耗时、预计剩余时间和速度。
TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

# TQDM_CONFIG：
# 子任务进度条的默认配置。
# dynamic_ncols=True 可以根据终端宽度自动调整进度条长度。
# leave=False 表示子进度条完成后不常驻终端，减少输出刷屏。
TQDM_CONFIG = {
    "bar_format": TQDM_BAR_FORMAT,
    "dynamic_ncols": True,
    "leave": False,
}

# OVERALL_TQDM_CONFIG：
# 总进度条配置。
# leave=True 表示主流程完成后保留最终状态，便于回看整体进度。
OVERALL_TQDM_CONFIG = {
    **TQDM_CONFIG,
    "leave": True,
}

# PROGRESS_COLORS：
# 不同阶段使用不同颜色，方便在终端中快速区分当前步骤。
# 如果终端不支持 ANSI 颜色，tqdm 会退化为普通显示。
PROGRESS_COLORS = {
    "overall": "green",
    "prepare": "white",
    "read": "cyan",
    "compute": "yellow",
    "station": "red",
    "save": "blue",
    "plot": "magenta",
    "cleanup": "magenta",
}


# ============================================================
# 4. 绘图参数
# 本区只控制图件样式，不影响事件识别和表格计算结果。
# ============================================================
# PLOT_CONFIG：
# scatter_figsize 控制散点图尺寸，单位为英寸；
# dpi 控制输出分辨率，300 dpi 通常可满足论文或报告插图需求。
PLOT_CONFIG = {
    "dpi": 300,
    "scatter_figsize": (9.6, 7.4),
    "scatter_note_figsize": (9.6, 3.2),
    "sensitivity_figsize": (12.8, 9.2),
    "scatter_alpha": 0.74,
    "scatter_size": 46,
    "reference_intensities": [1.0, 2.0],
    "level_markers": {
        "Light": "o",
        "Moderate": "s",
        "Severe": "^",
        "Extreme": "D",
    },
}


def configure_runtime_cache() -> None:
    """
    配置本次运行的缓存目录。

    作用
    1. 创建 TEMP_DIR。
    2. 将环境变量 TMP、TEMP 指向 TEMP_DIR。
    3. 将 tempfile.tempdir 指向 TEMP_DIR。

    这样做的目的是把 xarray、pandas、NetCDF 后端或 Python 标准库可能产生的临时文件
    集中到 RUN_CACHE_DIR 下，便于脚本结束后统一清理。
    """
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(TEMP_DIR)
    os.environ["TEMP"] = str(TEMP_DIR)
    tempfile.tempdir = str(TEMP_DIR)


def cleanup_runtime_cache() -> None:
    """
    清理本次脚本显式创建的缓存目录。

    注意
    1. 只删除 RUN_CACHE_DIR，不删除 OUTPUT_DIR 下的正式输出文件。
    2. ignore_errors=True 可避免缓存目录不存在时抛出异常。
    """
    shutil.rmtree(RUN_CACHE_DIR, ignore_errors=True)


def date_to_yyyymmdd(value: pd.Timestamp) -> float:
    """
    将日期编码为 YYYYMMDD 数值。

    参数
    value：
        pandas 可识别的日期对象，通常是 pd.Timestamp。

    返回
    float：
        形如 20250131 的数值。这里使用 float 是为了和 NetCDF 中的 NaN 填充值兼容；
        如果使用整数数组，就无法自然表示缺失日期。
    """
    timestamp = pd.Timestamp(value)
    return float(timestamp.year * 10000 + timestamp.month * 100 + timestamp.day)


def date_to_text(value: pd.Timestamp | float | int | None) -> str:
    """
    将日期值转换为 CSV 中易读的 YYYY-MM-DD 文本。

    参数
    value：
        pandas 可识别的日期对象；如果为 None 或 NaN，返回空字符串。

    返回
    str：
        日期文本，例如 "2025-01-31"；缺失日期返回 ""。
    """
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def classify_drought_level(min_spi: float) -> tuple[int, str]:
    """
    按事件期间最小 daily SPI 判定该事件达到的最高干旱等级。

    参数
    min_spi：
        一个有效干旱事件内的最小 daily SPI。

    判定规则
    1. 轻旱：-1.0 < min_spi <= -0.5
    2. 中旱：-1.5 < min_spi <= -1.0
    3. 重旱：-2.0 < min_spi <= -1.5
    4. 特旱：min_spi <= -2.0

    返回
    tuple[int, str]：
        第一个值是等级编码，第二个值是英文等级名称。
        编码规则为 1=Light、2=Moderate、3=Severe、4=Extreme。
    """
    if min_spi <= -2.0:
        return 4, DROUGHT_LEVELS[4]
    if min_spi <= -1.5:
        return 3, DROUGHT_LEVELS[3]
    if min_spi <= -1.0:
        return 2, DROUGHT_LEVELS[2]
    return 1, DROUGHT_LEVELS[1]


def compute_event_extent_peak(
    spi_values: np.ndarray,
    times: pd.DatetimeIndex,
    peak_date: pd.Timestamp,
    threshold: float,
) -> float:
    """
    按事件峰值日计算空间范围。

    参数
    spi_values：
        全省 daily SPI 三维数组，形状应为 (time, lat, lon)。
    times：
        与 spi_values 第一维对应的逐日时间索引。
    peak_date：
        当前事件峰值日，通常使用事件的 min_spi_date，即事件内最小 SPI 出现日期。
    threshold：
        干旱日阈值，当前主流程使用 DROUGHT_THRESHOLD=-0.5。

    计算口径
    1. 找到 peak_date 对应的全省 SPI 栅格。
    2. 有效格点为该日 SPI 非 NaN 的格点。
    3. 干旱格点为有效格点中 SPI < threshold 的格点。
    4. 返回 干旱格点数 / 有效格点数 * 100。

    返回
    float：
        百分比，保留两位小数；如果日期不存在或该日无有效格点，返回 np.nan。
    """
    try:
        day_idx = int(times.get_loc(peak_date))
    except KeyError:
        return np.nan

    day_spi = spi_values[day_idx]  # (n_lat, n_lon)
    finite_mask = np.isfinite(day_spi)
    if not np.any(finite_mask):
        return np.nan

    drought_mask = finite_mask & (day_spi < threshold)
    drought_cells = int(np.count_nonzero(drought_mask))
    valid_cells = int(np.count_nonzero(finite_mask))
    return round(drought_cells / valid_cells * 100.0, 2)


def compute_event_extent_union(
    spi_values: np.ndarray,
    times: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    threshold: float,
) -> float:
    """
    按整个事件期间的 union 计算空间范围。

    参数
    spi_values：
        全省 daily SPI 三维数组，形状应为 (time, lat, lon)。
    times：
        与 spi_values 第一维对应的逐日时间索引。
    start_date / end_date：
        当前事件的开始日期和结束日期，闭区间包含两端日期。
    threshold：
        干旱日阈值，当前主流程使用 DROUGHT_THRESHOLD=-0.5。

    计算口径
    1. 截取 start_date 到 end_date 期间的全省 SPI 栅格。
    2. 对每个格点，只要事件期间任意一天 SPI < threshold，就记为 union 干旱格点。
    3. 有效格点为事件期间至少有一天非 NaN 的格点。
    4. 返回 union 干旱格点数 / union 有效格点数 * 100。

    返回
    float：
        百分比，保留两位小数；如果时间范围无数据或无有效格点，返回 np.nan。
    """
    time_mask = (times >= start_date) & (times <= end_date)
    if not np.any(time_mask):
        return np.nan

    sub_spi = spi_values[time_mask]  # (n_days_event, n_lat, n_lon)
    finite_mask = np.isfinite(sub_spi)
    if not np.any(finite_mask):
        return np.nan

    drought_mask = finite_mask & (sub_spi < threshold)
    # 事件期间 union：只要某一天干旱就记入
    union_drought = np.any(drought_mask, axis=0)  # (n_lat, n_lon)
    union_valid = np.any(finite_mask, axis=0)  # (n_lat, n_lon)

    drought_cells = int(np.count_nonzero(union_drought & union_valid))
    valid_cells = int(np.count_nonzero(union_valid))
    if valid_cells == 0:
        return np.nan

    return round(drought_cells / valid_cells * 100.0, 2)


def extract_drought_events(
    spi_1d: np.ndarray,
    times: pd.DatetimeIndex,
    threshold: float,
    min_duration_days: int,
) -> list[dict[str, object]]:
    """
    从单个格点或站点匹配格点的 daily SPI 序列中识别全年所有有效干旱事件。

    参数
    spi_1d：
        单个格点的一维 daily SPI 序列，长度必须与 times 一致。
    times：
        与 spi_1d 一一对应的逐日时间索引。
    threshold：
        干旱日阈值。只有 SPI < threshold 的日期才被视为干旱日。
    min_duration_days：
        有效事件的最短持续天数。当前主流程为 6，即连续至少 6 天干旱才保留。

    事件识别规则
    1. 从头到尾扫描 spi_1d。
    2. SPI < threshold 时进入或延续干旱片段。
    3. SPI >= threshold 或 NaN 会结束当前干旱片段。
    4. NaN 被视为中断，避免缺测把前后两个片段错误拼接成一个事件。
    5. 片段持续天数小于 min_duration_days 时丢弃。

    每个返回事件包含
    start_date：
        事件开始日期。
    end_date：
        事件结束日期。
    duration_days：
        连续干旱天数。
    min_daily_spi / max_daily_spi：
        事件期间最小和最大 daily SPI。
    min_spi_date / max_spi_date：
        最小和最大 daily SPI 出现日期。
    severity：
        事件期间 sum(abs(SPI))。
    drought_level_code / drought_level：
        按 min_daily_spi 判定的等级编码和英文名称。
    edge_truncated：
        如果事件贴着输入序列第一天或最后一天，则为 True，表示可能被分析窗口边界截断。
    event_id：
        同一格点内按时间顺序编号，从 1 开始。
    """
    events: list[dict[str, object]] = []
    in_drought = False
    segment_start = -1

    def close_segment(start_idx: int, end_idx: int) -> None:
        duration_days = end_idx - start_idx
        if duration_days < min_duration_days:
            return

        event_spi = np.asarray(spi_1d[start_idx:end_idx], dtype=float)
        event_times = pd.DatetimeIndex(times[start_idx:end_idx])
        min_idx = int(np.nanargmin(event_spi))
        max_idx = int(np.nanargmax(event_spi))
        min_spi = float(event_spi[min_idx])
        max_spi = float(event_spi[max_idx])
        level_code, level_name = classify_drought_level(min_spi)

        events.append(
            {
                "start_date": pd.Timestamp(event_times[0]),
                "end_date": pd.Timestamp(event_times[-1]),
                "duration_days": int(duration_days),
                "min_daily_spi": min_spi,
                "max_daily_spi": max_spi,
                "min_spi_date": pd.Timestamp(event_times[min_idx]),
                "max_spi_date": pd.Timestamp(event_times[max_idx]),
                "severity": float(np.nansum(np.abs(event_spi))),
                "drought_level_code": int(level_code),
                "drought_level": level_name,
                "edge_truncated": bool(start_idx == 0 or end_idx == len(spi_1d)),
            }
        )

    for idx, value in enumerate(spi_1d):
        if np.isnan(value):
            if in_drought:
                close_segment(segment_start, idx)
                in_drought = False
                segment_start = -1
            continue

        if value < threshold and not in_drought:
            in_drought = True
            segment_start = idx
            continue

        if value >= threshold and in_drought:
            close_segment(segment_start, idx)
            in_drought = False
            segment_start = -1

    if in_drought:
        close_segment(segment_start, len(spi_1d))

    for event_idx, event in enumerate(events, start=1):
        event["event_id"] = int(event_idx)

    return events


def summarize_annual_features(
    spi_1d: np.ndarray,
    events: list[dict[str, object]],
    threshold: float,
) -> dict[str, object]:
    """
    汇总单个格点或站点匹配格点的全年 SPI 和有效干旱事件统计。

    参数
    spi_1d：
        单个格点全年 daily SPI 序列。
    events：
        extract_drought_events() 返回的有效干旱事件列表。
        注意：只包含持续天数 >= MIN_DURATION_DAYS 的事件。
    threshold：
        干旱日阈值，用于计算全年干旱日数和全年累计严重度。

    统计口径
    1. Annual_Min_SPI / Annual_Max_SPI：
       基于全年所有非 NaN daily SPI。
    2. Drought_Days：
       全年所有 SPI < threshold 的天数，包括不足 6 天的短片段。
    3. Total_Severity：
       全年所有干旱日 abs(SPI) 的累加，也包括不足 6 天的短片段。
    4. Event_Count_*：
       只统计有效干旱事件，即持续天数 >= MIN_DURATION_DAYS 的事件。
    5. First_Onset、Max_Event_Duration、Max_Event_Severity、Worst_Event_*：
       只基于有效干旱事件计算。

    返回
    dict[str, object]：
        可直接展开写入站点年度统计 CSV 的字段字典。
    """
    valid_spi = np.asarray(spi_1d, dtype=float)
    finite_mask = np.isfinite(valid_spi)
    drought_mask = finite_mask & (valid_spi < threshold)

    stats: dict[str, object] = {
        "Annual_Min_SPI": float(np.nanmin(valid_spi)) if np.any(finite_mask) else np.nan,
        "Annual_Max_SPI": float(np.nanmax(valid_spi)) if np.any(finite_mask) else np.nan,
        "Drought_Days": int(np.count_nonzero(drought_mask)),
        "Total_Severity": float(np.nansum(np.abs(valid_spi[drought_mask]))),
        "First_Onset": "",
        "Event_Count_Total": int(len(events)),
        "Event_Count_Light": 0,
        "Event_Count_Moderate": 0,
        "Event_Count_Severe": 0,
        "Event_Count_Extreme": 0,
        "Max_Event_Duration": pd.NA,
        "Max_Event_Severity": np.nan,
        "Worst_Event_Min_SPI": np.nan,
        "Worst_Event_Level": "",
    }

    count_columns = {
        1: "Event_Count_Light",
        2: "Event_Count_Moderate",
        3: "Event_Count_Severe",
        4: "Event_Count_Extreme",
    }
    for event in events:
        stats[count_columns[int(event["drought_level_code"])]] += 1

    if not events:
        return stats

    first_event = min(events, key=lambda item: pd.Timestamp(item["start_date"]))
    duration_event = max(events, key=lambda item: int(item["duration_days"]))
    severity_event = max(events, key=lambda item: float(item["severity"]))
    worst_event = min(events, key=lambda item: float(item["min_daily_spi"]))

    stats.update(
        {
            "First_Onset": date_to_text(pd.Timestamp(first_event["start_date"])),
            "Max_Event_Duration": int(duration_event["duration_days"]),
            "Max_Event_Severity": float(severity_event["severity"]),
            "Worst_Event_Min_SPI": float(worst_event["min_daily_spi"]),
            "Worst_Event_Level": str(worst_event["drought_level"]),
        }
    )
    return stats


def build_grid_event_dataset(
    lats: np.ndarray,
    lons: np.ndarray,
    grid_events: list[list[list[dict[str, object]]]],
) -> xr.Dataset:
    """
    将全省格点的变长事件列表写成 event x lat x lon 的 NetCDF 数据集。

    参数
    lats / lons：
        输入 SPI 栅格的纬度和经度坐标。
    grid_events：
        二维嵌套列表，结构为 grid_events[lat_idx][lon_idx] = 该格点事件列表。

    NetCDF 组织方式
    1. 不同格点的事件数量可能不同，NetCDF 不适合直接保存不规则列表。
    2. 脚本先找出全省单个格点最大事件数 max_events。
    3. 输出维度为 event x lat x lon。
    4. 事件少于 max_events 的格点，剩余位置填 NaN。

    主要变量
    event_id：
        某格点内的事件序号。
    start_date / end_date：
        事件起止日期，数值格式 YYYYMMDD。
    duration_days：
        事件持续天数。
    min_daily_spi / max_daily_spi：
        事件期间最小和最大 daily SPI。
    severity：
        事件期间 sum(abs(SPI))。
    drought_level_code：
        1=Light、2=Moderate、3=Severe、4=Extreme。
    edge_truncated：
        1 表示事件贴着年度窗口边界，可能被截断；0 表示未贴边。
    """
    n_lat = len(lats)
    n_lon = len(lons)
    max_events = max(
        (len(grid_events[lat_idx][lon_idx]) for lat_idx in range(n_lat) for lon_idx in range(n_lon)),
        default=0,
    )

    if max_events == 0:
        max_events = 1

    shape = (max_events, n_lat, n_lon)
    event_id = np.full(shape, np.nan, dtype=np.float32)
    start_date = np.full(shape, np.nan, dtype=np.float64)
    end_date = np.full(shape, np.nan, dtype=np.float64)
    duration_days = np.full(shape, np.nan, dtype=np.float32)
    min_daily_spi = np.full(shape, np.nan, dtype=np.float32)
    max_daily_spi = np.full(shape, np.nan, dtype=np.float32)
    min_spi_date = np.full(shape, np.nan, dtype=np.float64)
    max_spi_date = np.full(shape, np.nan, dtype=np.float64)
    severity = np.full(shape, np.nan, dtype=np.float32)
    drought_level_code = np.full(shape, np.nan, dtype=np.float32)
    edge_truncated = np.full(shape, np.nan, dtype=np.float32)

    for lat_idx in range(n_lat):
        for lon_idx in range(n_lon):
            for event_idx, event in enumerate(grid_events[lat_idx][lon_idx]):
                event_id[event_idx, lat_idx, lon_idx] = float(event["event_id"])
                start_date[event_idx, lat_idx, lon_idx] = date_to_yyyymmdd(
                    pd.Timestamp(event["start_date"])
                )
                end_date[event_idx, lat_idx, lon_idx] = date_to_yyyymmdd(
                    pd.Timestamp(event["end_date"])
                )
                duration_days[event_idx, lat_idx, lon_idx] = float(event["duration_days"])
                min_daily_spi[event_idx, lat_idx, lon_idx] = float(event["min_daily_spi"])
                max_daily_spi[event_idx, lat_idx, lon_idx] = float(event["max_daily_spi"])
                min_spi_date[event_idx, lat_idx, lon_idx] = date_to_yyyymmdd(
                    pd.Timestamp(event["min_spi_date"])
                )
                max_spi_date[event_idx, lat_idx, lon_idx] = date_to_yyyymmdd(
                    pd.Timestamp(event["max_spi_date"])
                )
                severity[event_idx, lat_idx, lon_idx] = float(event["severity"])
                drought_level_code[event_idx, lat_idx, lon_idx] = float(
                    event["drought_level_code"]
                )
                edge_truncated[event_idx, lat_idx, lon_idx] = float(event["edge_truncated"])

    coords = {
        "event": np.arange(1, max_events + 1, dtype=np.int32),
        LAT_NAME: lats,
        LON_NAME: lons,
    }
    dims = ["event", LAT_NAME, LON_NAME]

    return xr.Dataset(
        data_vars={
            "event_id": (dims, event_id, {"description": "Event order within each grid cell"}),
            "start_date": (
                dims,
                start_date,
                {"units": "YYYYMMDD", "description": "Drought event start date"},
            ),
            "end_date": (
                dims,
                end_date,
                {"units": "YYYYMMDD", "description": "Drought event end date"},
            ),
            "duration_days": (
                dims,
                duration_days,
                {"units": "days", "description": "Consecutive drought days in the event"},
            ),
            "min_daily_spi": (
                dims,
                min_daily_spi,
                {"description": "Minimum daily SPI during the event"},
            ),
            "max_daily_spi": (
                dims,
                max_daily_spi,
                {"description": "Maximum daily SPI during the event"},
            ),
            "min_spi_date": (
                dims,
                min_spi_date,
                {"units": "YYYYMMDD", "description": "Date of minimum daily SPI"},
            ),
            "max_spi_date": (
                dims,
                max_spi_date,
                {"units": "YYYYMMDD", "description": "Date of maximum daily SPI"},
            ),
            "severity": (
                dims,
                severity,
                {"description": "Sum of absolute daily SPI values during the event"},
            ),
            "drought_level_code": (
                dims,
                drought_level_code,
                {"description": "1=Light, 2=Moderate, 3=Severe, 4=Extreme"},
            ),
            "edge_truncated": (
                dims,
                edge_truncated,
                {"description": "1 means the event touches the annual data boundary"},
            ),
        },
        coords=coords,
        attrs={
            "title": "Fujian daily SPI drought events for 2025",
            "source_file": str(SPI_FILE),
            "spi_variable": SPI_VAR,
            "analysis_window": f"{WINDOW_START} to {WINDOW_END}",
            "event_definition": (
                f"Consecutive daily SPI < {DROUGHT_THRESHOLD} for at least "
                f"{MIN_DURATION_DAYS} days"
            ),
            "drought_level_definition": (
                "Level is assigned by event minimum SPI: "
                "1 light (-1.0 < SPI <= -0.5), "
                "2 moderate (-1.5 < SPI <= -1.0), "
                "3 severe (-2.0 < SPI <= -1.5), "
                "4 extreme (SPI <= -2.0)."
            ),
        },
    )


def build_station_tables(
    site_file: Path,
    lats: np.ndarray,
    lons: np.ndarray,
    spi_values: np.ndarray,
    times: pd.DatetimeIndex,
    grid_events: list[list[list[dict[str, object]]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    生成站点事件长表、站点年度统计表和站点逐日 SPI 序列表。

    参数
    site_file：
        观测站点 CSV 路径，必须包含站点编号、经度、纬度三列。
    lats / lons：
        SPI 栅格纬度和经度坐标。
    spi_values：
        全省 daily SPI 三维数组，形状为 (time, lat, lon)。
    times：
        与 spi_values 第一维对应的逐日时间索引。
    grid_events：
        已经逐格点识别好的事件列表，结构为 grid_events[lat_idx][lon_idx]。

    站点匹配方法
    1. 对每个站点，分别计算站点纬度与所有栅格纬度的绝对差，取最小者。
    2. 同理匹配最近经度。
    3. 使用该最近邻格点的 daily SPI 和事件列表作为站点结果。
    4. 本脚本不做空间插值，避免引入额外假设。

    返回
    station_event_table：
        站点事件长表，一行代表一个站点的一次有效干旱事件。
        除事件自身特征外，还附带该站点全年不同等级事件次数。
    station_annual_table：
        站点年度统计表，一行代表一个站点。
    station_daily_table：
        站点逐日 SPI 序列表，一行代表一个站点的一天。
    """
    if not site_file.exists():
        raise FileNotFoundError(f"找不到观测站点表：{site_file}")

    site_table = pd.read_csv(site_file)
    required_columns = {SITE_ID_FIELD, SITE_LON_FIELD, SITE_LAT_FIELD}
    missing_columns = required_columns.difference(site_table.columns)
    if missing_columns:
        raise KeyError(f"观测站点表缺少必要字段：{sorted(missing_columns)}")

    event_rows: list[dict[str, object]] = []
    annual_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []

    with tqdm(
        total=len(site_table),
        desc="提取站点事件和年度统计",
        colour=PROGRESS_COLORS["station"],
        **TQDM_CONFIG,
    ) as pbar_station:
        for _, row in site_table.iterrows():
            site_id = row[SITE_ID_FIELD]
            station_lon = float(row[SITE_LON_FIELD])
            station_lat = float(row[SITE_LAT_FIELD])
            lat_idx = int(np.abs(lats - station_lat).argmin())
            lon_idx = int(np.abs(lons - station_lon).argmin())
            matched_lon = float(lons[lon_idx])
            matched_lat = float(lats[lat_idx])

            site_events = grid_events[lat_idx][lon_idx]
            spi_1d = spi_values[:, lat_idx, lon_idx]

            annual_stats = summarize_annual_features(
                spi_1d=spi_1d,
                events=site_events,
                threshold=DROUGHT_THRESHOLD,
            )

            common_fields = {
                "Site_ID": site_id,
                "Station_Lon": station_lon,
                "Station_Lat": station_lat,
                "Matched_Grid_Lon": matched_lon,
                "Matched_Grid_Lat": matched_lat,
            }

            # 年度统计表是一站一行，用于快速查看该站点全年总体干旱情况。
            annual_rows.append({**common_fields, **annual_stats})

            # 这些年度事件次数字段也附加到每条事件记录中。
            # 好处是筛选某一次事件时，仍能直接看到该站点全年事件背景。
            station_count_fields = {
                "Station_Event_Count_Total": annual_stats["Event_Count_Total"],
                "Station_Event_Count_Light": annual_stats["Event_Count_Light"],
                "Station_Event_Count_Moderate": annual_stats["Event_Count_Moderate"],
                "Station_Event_Count_Severe": annual_stats["Event_Count_Severe"],
                "Station_Event_Count_Extreme": annual_stats["Event_Count_Extreme"],
            }

            # 事件长表是一事件一行。
            # 两种 Extent 都是全省空间百分比，不是站点局地指标。
            for event in site_events:
                start_date = pd.Timestamp(event["start_date"])
                end_date = pd.Timestamp(event["end_date"])
                peak_date = pd.Timestamp(event["min_spi_date"])

                extent_peak = compute_event_extent_peak(
                    spi_values=spi_values,
                    times=times,
                    peak_date=peak_date,
                    threshold=DROUGHT_THRESHOLD,
                )
                extent_union = compute_event_extent_union(
                    spi_values=spi_values,
                    times=times,
                    start_date=start_date,
                    end_date=end_date,
                    threshold=DROUGHT_THRESHOLD,
                )

                event_rows.append(
                    {
                        **common_fields,
                        "Event_ID": int(event["event_id"]),
                        "Start_Date": date_to_text(start_date),
                        "End_Date": date_to_text(end_date),
                        "Duration_Days": int(event["duration_days"]),
                        "Min_Daily_SPI": float(event["min_daily_spi"]),
                        "Max_Daily_SPI": float(event["max_daily_spi"]),
                        "Min_SPI_Date": date_to_text(peak_date),
                        "Max_SPI_Date": date_to_text(pd.Timestamp(event["max_spi_date"])),
                        "Severity": float(event["severity"]),
                        "Drought_Level": str(event["drought_level"]),
                        "Drought_Level_Code": int(event["drought_level_code"]),
                        "Edge_Truncated": bool(event["edge_truncated"]),
                        "Drought_Extent_Peak(%)": extent_peak,
                        "Drought_Extent_Union(%)": extent_union,
                        **station_count_fields,
                    }
                )

            # 逐日 SPI 表用于后续按任意时间窗口重新筛选、画时序图或核查事件。
            # Is_Drought_Day 只判断当天 SPI 是否低于阈值，不要求连续天数达到 6 天。
            for day_time, spi_val in zip(times, spi_1d):
                daily_rows.append(
                    {
                        **common_fields,
                        "Date": day_time.strftime("%Y-%m-%d"),
                        "Daily_SPI_20d": round(float(spi_val), 4)
                        if np.isfinite(spi_val)
                        else np.nan,
                        "Is_Drought_Day": bool(
                            np.isfinite(spi_val) and spi_val < DROUGHT_THRESHOLD
                        ),
                    }
                )

            pbar_station.update(1)

    station_event_table = pd.DataFrame(event_rows)
    station_annual_table = pd.DataFrame(annual_rows)
    station_daily_table = pd.DataFrame(daily_rows)
    return station_event_table, station_annual_table, station_daily_table


def calculate_haversine_distance_km(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
) -> float:
    """
    计算两个经纬度点之间的大圆距离，单位为 km。

    该距离用于评估站点和最近 SPI 格点之间的空间匹配误差。
    """
    radius_km = 6371.0088
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = np.radians([lon1, lat1, lon2, lat2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    hav = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * radius_km * np.arcsin(np.sqrt(hav)))


def extract_nearest_series(
    spi_values: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    station_lat: float,
    station_lon: float,
) -> tuple[np.ndarray, int, int]:
    """
    提取最近邻格点 daily SPI 序列。

    返回
    spi_1d：
        最近格点的逐日 SPI。
    lat_idx / lon_idx：
        最近格点在 lats / lons 中的索引。
    """
    lat_idx = int(np.abs(lats - station_lat).argmin())
    lon_idx = int(np.abs(lons - station_lon).argmin())
    return np.asarray(spi_values[:, lat_idx, lon_idx], dtype=float), lat_idx, lon_idx


def extract_bilinear_series(
    spi_values: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    station_lat: float,
    station_lon: float,
) -> np.ndarray:
    """
    使用双线性插值提取站点 daily SPI 序列。

    处理规则
    1. 找到包围站点的 2 x 2 格点。
    2. 按站点在经纬度方向上的相对位置计算权重。
    3. 每一天单独插值；若某一天 4 个格点中部分为 NaN，则对有效格点权重重新归一化。
    4. 若站点位于网格边缘外侧，则裁剪到最近可用的 2 x 2 网格。
    """
    lat_upper = int(np.searchsorted(lats, station_lat, side="right"))
    lon_upper = int(np.searchsorted(lons, station_lon, side="right"))
    lat0 = int(np.clip(lat_upper - 1, 0, len(lats) - 2))
    lat1 = lat0 + 1
    lon0 = int(np.clip(lon_upper - 1, 0, len(lons) - 2))
    lon1 = lon0 + 1

    lat_span = float(lats[lat1] - lats[lat0])
    lon_span = float(lons[lon1] - lons[lon0])
    lat_weight = 0.0 if lat_span == 0 else float((station_lat - lats[lat0]) / lat_span)
    lon_weight = 0.0 if lon_span == 0 else float((station_lon - lons[lon0]) / lon_span)
    lat_weight = float(np.clip(lat_weight, 0.0, 1.0))
    lon_weight = float(np.clip(lon_weight, 0.0, 1.0))

    values = np.stack(
        [
            spi_values[:, lat0, lon0],
            spi_values[:, lat0, lon1],
            spi_values[:, lat1, lon0],
            spi_values[:, lat1, lon1],
        ],
        axis=1,
    ).astype(float)
    weights = np.asarray(
        [
            (1.0 - lat_weight) * (1.0 - lon_weight),
            (1.0 - lat_weight) * lon_weight,
            lat_weight * (1.0 - lon_weight),
            lat_weight * lon_weight,
        ],
        dtype=float,
    )

    finite_mask = np.isfinite(values)
    weighted_values = np.where(finite_mask, values * weights, 0.0)
    valid_weights = np.where(finite_mask, weights, 0.0).sum(axis=1)
    output = np.full(values.shape[0], np.nan, dtype=float)
    valid_days = valid_weights > 0
    output[valid_days] = weighted_values[valid_days].sum(axis=1) / valid_weights[valid_days]
    return output


def extract_mean3x3_series(
    spi_values: np.ndarray,
    nearest_lat_idx: int,
    nearest_lon_idx: int,
) -> np.ndarray:
    """
    提取最近格点周围 3 x 3 邻域的逐日平均 SPI。

    边缘站点不足 3 x 3 时，使用可用窗口。
    每一天使用 np.nanmean 忽略缺测值；若当天邻域全为 NaN，则返回 NaN。
    """
    lat_start = max(0, nearest_lat_idx - 1)
    lat_end = min(spi_values.shape[1], nearest_lat_idx + 2)
    lon_start = max(0, nearest_lon_idx - 1)
    lon_end = min(spi_values.shape[2], nearest_lon_idx + 2)
    neighborhood = spi_values[:, lat_start:lat_end, lon_start:lon_end]
    with np.errstate(all="ignore"):
        return np.nanmean(neighborhood, axis=(1, 2)).astype(float)


def events_to_station_rows(
    events: list[dict[str, object]],
    common_fields: dict[str, object],
    method_name: str,
) -> list[dict[str, object]]:
    """把某站点某方法下的事件列表转换成事件长表行。"""
    rows: list[dict[str, object]] = []
    for event in events:
        rows.append(
            {
                **common_fields,
                "Extraction_Method": method_name,
                "Event_ID": int(event["event_id"]),
                "Start_Date": date_to_text(pd.Timestamp(event["start_date"])),
                "End_Date": date_to_text(pd.Timestamp(event["end_date"])),
                "Duration_Days": int(event["duration_days"]),
                "Min_Daily_SPI": float(event["min_daily_spi"]),
                "Max_Daily_SPI": float(event["max_daily_spi"]),
                "Min_SPI_Date": date_to_text(pd.Timestamp(event["min_spi_date"])),
                "Max_SPI_Date": date_to_text(pd.Timestamp(event["max_spi_date"])),
                "Severity": float(event["severity"]),
                "Drought_Level": str(event["drought_level"]),
                "Drought_Level_Code": int(event["drought_level_code"]),
                "Edge_Truncated": bool(event["edge_truncated"]),
            }
        )
    return rows


def build_sensitivity_analysis_tables(
    site_file: Path,
    lats: np.ndarray,
    lons: np.ndarray,
    spi_values: np.ndarray,
    times: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    构建敏感性分析的逐日对比表、事件长表、年度统计表、差异表和总体统计表。

    比较方法
    1. Nearest：最近邻格点。
    2. Bilinear：双线性插值。
    3. Mean3x3：最近格点周围 3 x 3 邻域平均。
    """
    if not site_file.exists():
        raise FileNotFoundError(f"找不到观测站点表：{site_file}")

    site_table = pd.read_csv(site_file)
    required_columns = {SITE_ID_FIELD, SITE_LON_FIELD, SITE_LAT_FIELD}
    missing_columns = required_columns.difference(site_table.columns)
    if missing_columns:
        raise KeyError(f"观测站点表缺少必要字段：{sorted(missing_columns)}")

    daily_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    annual_rows: list[dict[str, object]] = []

    with tqdm(
        total=len(site_table),
        desc="敏感性分析",
        colour=PROGRESS_COLORS["station"],
        **TQDM_CONFIG,
    ) as pbar_sensitivity:
        for _, row in site_table.iterrows():
            site_id = row[SITE_ID_FIELD]
            station_lon = float(row[SITE_LON_FIELD])
            station_lat = float(row[SITE_LAT_FIELD])

            nearest_spi, lat_idx, lon_idx = extract_nearest_series(
                spi_values=spi_values,
                lats=lats,
                lons=lons,
                station_lat=station_lat,
                station_lon=station_lon,
            )
            bilinear_spi = extract_bilinear_series(
                spi_values=spi_values,
                lats=lats,
                lons=lons,
                station_lat=station_lat,
                station_lon=station_lon,
            )
            mean3x3_spi = extract_mean3x3_series(
                spi_values=spi_values,
                nearest_lat_idx=lat_idx,
                nearest_lon_idx=lon_idx,
            )

            matched_lon = float(lons[lon_idx])
            matched_lat = float(lats[lat_idx])
            distance_km = calculate_haversine_distance_km(
                lon1=station_lon,
                lat1=station_lat,
                lon2=matched_lon,
                lat2=matched_lat,
            )
            common_fields = {
                "Site_ID": site_id,
                "Station_Lon": station_lon,
                "Station_Lat": station_lat,
                "Matched_Grid_Lon": matched_lon,
                "Matched_Grid_Lat": matched_lat,
                "Distance_to_Grid_km": round(distance_km, 4),
            }

            for day_time, nearest_value, bilinear_value, mean3x3_value in zip(
                times,
                nearest_spi,
                bilinear_spi,
                mean3x3_spi,
            ):
                daily_rows.append(
                    {
                        **common_fields,
                        "Date": day_time.strftime("%Y-%m-%d"),
                        "Nearest_SPI": nearest_value,
                        "Bilinear_SPI": bilinear_value,
                        "Mean3x3_SPI": mean3x3_value,
                        "Diff_Bilinear_vs_Nearest": bilinear_value - nearest_value
                        if np.isfinite(bilinear_value) and np.isfinite(nearest_value)
                        else np.nan,
                        "Diff_Mean3x3_vs_Nearest": mean3x3_value - nearest_value
                        if np.isfinite(mean3x3_value) and np.isfinite(nearest_value)
                        else np.nan,
                        "AbsDiff_Bilinear_vs_Nearest": abs(bilinear_value - nearest_value)
                        if np.isfinite(bilinear_value) and np.isfinite(nearest_value)
                        else np.nan,
                        "AbsDiff_Mean3x3_vs_Nearest": abs(mean3x3_value - nearest_value)
                        if np.isfinite(mean3x3_value) and np.isfinite(nearest_value)
                        else np.nan,
                    }
                )

            method_series = {
                "Nearest": nearest_spi,
                "Bilinear": bilinear_spi,
                "Mean3x3": mean3x3_spi,
            }
            for method_name, spi_1d in method_series.items():
                events = extract_drought_events(
                    spi_1d=spi_1d,
                    times=times,
                    threshold=DROUGHT_THRESHOLD,
                    min_duration_days=MIN_DURATION_DAYS,
                )
                annual_stats = summarize_annual_features(
                    spi_1d=spi_1d,
                    events=events,
                    threshold=DROUGHT_THRESHOLD,
                )
                annual_rows.append(
                    {
                        **common_fields,
                        "Extraction_Method": method_name,
                        **annual_stats,
                    }
                )
                event_rows.extend(
                    events_to_station_rows(
                        events=events,
                        common_fields=common_fields,
                        method_name=method_name,
                    )
                )

            pbar_sensitivity.update(1)

    daily_table = pd.DataFrame(daily_rows)
    event_table = pd.DataFrame(event_rows)
    annual_table = pd.DataFrame(annual_rows)
    delta_table = build_sensitivity_delta_table(annual_table)
    summary_table = build_sensitivity_summary_table(daily_table, annual_table, delta_table)
    return daily_table, event_table, annual_table, delta_table, summary_table


def build_sensitivity_delta_table(annual_table: pd.DataFrame) -> pd.DataFrame:
    """以 Nearest 为基准，计算 Bilinear 和 Mean3x3 的站点年度指标差异。"""
    metrics = [
        "Drought_Days",
        "Total_Severity",
        "Event_Count_Total",
        "Max_Event_Duration",
        "Max_Event_Severity",
        "Worst_Event_Min_SPI",
    ]
    base_columns = [
        "Site_ID",
        "Station_Lon",
        "Station_Lat",
        "Matched_Grid_Lon",
        "Matched_Grid_Lat",
        "Distance_to_Grid_km",
    ]
    rows: list[dict[str, object]] = []

    for site_id, site_group in annual_table.groupby("Site_ID", sort=False):
        nearest_rows = site_group[site_group["Extraction_Method"] == "Nearest"]
        if nearest_rows.empty:
            continue
        nearest_row = nearest_rows.iloc[0]
        output_row = {col: nearest_row[col] for col in base_columns if col in nearest_row}

        for method_name in ["Bilinear", "Mean3x3"]:
            method_rows = site_group[site_group["Extraction_Method"] == method_name]
            if method_rows.empty:
                for metric in metrics:
                    output_row[f"Delta_{metric}_{method_name}"] = np.nan
                    output_row[f"AbsDelta_{metric}_{method_name}"] = np.nan
                output_row[f"Same_Worst_Event_Level_{method_name}"] = pd.NA
                continue

            method_row = method_rows.iloc[0]
            for metric in metrics:
                nearest_value = nearest_row[metric]
                method_value = method_row[metric]
                if pd.isna(nearest_value) or pd.isna(method_value):
                    delta = np.nan
                else:
                    delta = float(method_value) - float(nearest_value)
                output_row[f"Delta_{metric}_{method_name}"] = delta
                output_row[f"AbsDelta_{metric}_{method_name}"] = abs(delta) if np.isfinite(delta) else np.nan

            output_row[f"Same_Worst_Event_Level_{method_name}"] = (
                method_row["Worst_Event_Level"] == nearest_row["Worst_Event_Level"]
            )

        rows.append(output_row)

    return pd.DataFrame(rows)


def calculate_pairwise_stats(
    reference: pd.Series,
    comparison: pd.Series,
) -> dict[str, float]:
    """计算两列数值的差异统计、相关系数和误差指标。"""
    paired = pd.DataFrame({"reference": reference, "comparison": comparison}).dropna()
    if paired.empty:
        return {
            "Mean_Difference": np.nan,
            "Median_Difference": np.nan,
            "Mean_Absolute_Difference": np.nan,
            "RMSE": np.nan,
            "Max_Absolute_Difference": np.nan,
            "Pearson_R": np.nan,
            "Spearman_R": np.nan,
        }

    difference = paired["comparison"] - paired["reference"]
    if paired["reference"].nunique() >= 2 and paired["comparison"].nunique() >= 2:
        pearson_r = float(paired["reference"].corr(paired["comparison"], method="pearson"))
        spearman_r = float(paired["reference"].corr(paired["comparison"], method="spearman"))
    else:
        pearson_r = np.nan
        spearman_r = np.nan

    return {
        "Mean_Difference": float(difference.mean()),
        "Median_Difference": float(difference.median()),
        "Mean_Absolute_Difference": float(difference.abs().mean()),
        "RMSE": float(np.sqrt(np.mean(np.square(difference)))),
        "Max_Absolute_Difference": float(difference.abs().max()),
        "Pearson_R": pearson_r,
        "Spearman_R": spearman_r,
    }


def build_sensitivity_summary_table(
    daily_table: pd.DataFrame,
    annual_table: pd.DataFrame,
    delta_table: pd.DataFrame,
) -> pd.DataFrame:
    """汇总敏感性分析总体统计，供论文方法和补充材料引用。"""
    rows: list[dict[str, object]] = []

    daily_comparisons = {
        "Bilinear_vs_Nearest": "Bilinear_SPI",
        "Mean3x3_vs_Nearest": "Mean3x3_SPI",
    }
    for comparison_name, comparison_column in daily_comparisons.items():
        stats = calculate_pairwise_stats(daily_table["Nearest_SPI"], daily_table[comparison_column])
        rows.append(
            {
                "Comparison": comparison_name,
                "Metric": "Daily_SPI",
                **stats,
                "Same_Count_Percent": np.nan,
            }
        )

    annual_metrics = [
        "Drought_Days",
        "Total_Severity",
        "Event_Count_Total",
        "Max_Event_Duration",
        "Max_Event_Severity",
        "Worst_Event_Min_SPI",
    ]
    for metric in annual_metrics:
        pivot = annual_table.pivot(index="Site_ID", columns="Extraction_Method", values=metric)
        for method_name in ["Bilinear", "Mean3x3"]:
            if "Nearest" not in pivot or method_name not in pivot:
                continue
            stats = calculate_pairwise_stats(pivot["Nearest"], pivot[method_name])
            same_percent = np.nan
            if metric == "Event_Count_Total":
                paired = pivot[["Nearest", method_name]].dropna()
                same_percent = (
                    float((paired["Nearest"] == paired[method_name]).mean() * 100.0)
                    if not paired.empty
                    else np.nan
                )
            rows.append(
                {
                    "Comparison": f"{method_name}_vs_Nearest",
                    "Metric": metric,
                    **stats,
                    "Same_Count_Percent": same_percent,
                }
            )

    for method_name in ["Bilinear", "Mean3x3"]:
        column = f"Same_Worst_Event_Level_{method_name}"
        if column in delta_table:
            valid = delta_table[column].dropna()
            same_percent = float(valid.mean() * 100.0) if len(valid) else np.nan
            rows.append(
                {
                    "Comparison": f"{method_name}_vs_Nearest",
                    "Metric": "Worst_Event_Level",
                    "Mean_Difference": np.nan,
                    "Median_Difference": np.nan,
                    "Mean_Absolute_Difference": np.nan,
                    "RMSE": np.nan,
                    "Max_Absolute_Difference": np.nan,
                    "Pearson_R": np.nan,
                    "Spearman_R": np.nan,
                    "Same_Count_Percent": same_percent,
                }
            )

    return pd.DataFrame(rows)


def plot_sensitivity_analysis_figure(
    daily_table: pd.DataFrame,
    delta_table: pd.DataFrame,
    output_file: Path,
) -> None:
    """
    绘制敏感性分析综合图。

    图件结构
    1. Nearest vs Bilinear daily SPI 散点图。
    2. Nearest vs Mean3x3 daily SPI 散点图。
    3. 年度 Drought_Days 差异箱线图。
    4. 年度 Total_Severity 差异箱线图。
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2,
        2,
        figsize=PLOT_CONFIG["sensitivity_figsize"],
        dpi=PLOT_CONFIG["dpi"],
    )
    axes = axes.ravel()

    scatter_specs = [
        ("Bilinear_SPI", "Nearest vs Bilinear daily SPI", axes[0]),
        ("Mean3x3_SPI", "Nearest vs Mean3x3 daily SPI", axes[1]),
    ]
    for column, title, ax in scatter_specs:
        plot_data = daily_table[["Nearest_SPI", column]].dropna()
        if plot_data.empty:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.scatter(
                plot_data["Nearest_SPI"],
                plot_data[column],
                s=7,
                alpha=0.18,
                color="#2f6f9f",
                edgecolors="none",
            )
            min_value = float(np.nanmin(plot_data.to_numpy()))
            max_value = float(np.nanmax(plot_data.to_numpy()))
            ax.plot([min_value, max_value], [min_value, max_value], color="black", linewidth=1.0)
            if plot_data["Nearest_SPI"].nunique() >= 2 and plot_data[column].nunique() >= 2:
                pearson = plot_data["Nearest_SPI"].corr(plot_data[column], method="pearson")
                spearman = plot_data["Nearest_SPI"].corr(plot_data[column], method="spearman")
            else:
                pearson = np.nan
                spearman = np.nan
            ax.text(
                0.04,
                0.96,
                f"n = {len(plot_data)}\nPearson r = {pearson:.3f}\nSpearman r = {spearman:.3f}",
                ha="left",
                va="top",
                transform=ax.transAxes,
                fontsize=8.5,
                bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#888888", "alpha": 0.84},
            )
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Nearest SPI")
        ax.set_ylabel(column.replace("_", " "))
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.28)

    box_specs = [
        (
            ["Delta_Drought_Days_Bilinear", "Delta_Drought_Days_Mean3x3"],
            "Difference in annual drought days",
            "Delta drought days",
            axes[2],
        ),
        (
            ["Delta_Total_Severity_Bilinear", "Delta_Total_Severity_Mean3x3"],
            "Difference in annual total severity",
            "Delta total severity",
            axes[3],
        ),
    ]
    for columns, title, ylabel, ax in box_specs:
        data = [delta_table[column].dropna().to_numpy() for column in columns if column in delta_table]
        labels = [column.replace("Delta_", "").replace("_Bilinear", "\nBilinear").replace("_Mean3x3", "\nMean3x3") for column in columns if column in delta_table]
        if not data or all(len(values) == 0 for values in data):
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.axhline(0, color="black", linewidth=0.9, linestyle="-")
            box = ax.boxplot(
                data,
                labels=labels,
                patch_artist=True,
                showfliers=True,
                medianprops={"color": "black", "linewidth": 1.2},
            )
            for patch in box["boxes"]:
                patch.set_facecolor("#d9a441")
                patch.set_alpha(0.68)
                patch.set_edgecolor("#4d4d4d")
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.28)

    fig.suptitle("Sensitivity of Site-Level Daily SPI Extraction Methods", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def plot_station_severity_duration_scatter(
    station_event_table: pd.DataFrame,
    output_file: Path,
) -> None:
    """
    绘制福建省观测站严重度-持续时间散点图。

    参数
    station_event_table：
        站点事件长表，由 build_station_tables() 生成。
        一行代表一个站点的一次有效干旱事件。
    output_file：
        PNG 输出路径。

    绘图口径
    1. 一个点 = 一个站点的一次有效干旱事件。
    2. 横轴 = Duration_Days，单位为天。
    3. 纵轴 = Severity，即事件期间 sum(abs(SPI))。
    4. 点颜色 = Mean_Intensity，即 Severity / Duration_Days，用于表示事件平均强度。
    5. 点形状 = Drought_Level，用于区分 Light、Moderate、Severe、Extreme。
    6. 上方和右侧边缘直方图分别展示 Duration_Days 和 Severity 的分布。

    缺失值处理
    1. 绘图前会删除 Duration_Days、Severity、Min_Daily_SPI、Drought_Level 任一字段缺失的记录。
    2. 如果没有可绘制事件，仍会输出一张空图并写入提示文字，保证主流程不中断。
    """
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    required_columns = ["Duration_Days", "Severity", "Min_Daily_SPI", "Drought_Level"]
    missing_columns = [col for col in required_columns if col not in station_event_table.columns]
    if missing_columns:
        raise KeyError(f"站点事件表缺少绘图字段：{missing_columns}")

    plot_table = station_event_table.dropna(subset=required_columns).copy()

    fig = plt.figure(
        figsize=PLOT_CONFIG["scatter_figsize"],
        dpi=PLOT_CONFIG["dpi"],
    )
    grid = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=(4.6, 1.15),
        height_ratios=(1.05, 4.4),
        wspace=0.05,
        hspace=0.05,
    )
    ax_histx = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0], sharex=ax_histx)
    ax_histy = fig.add_subplot(grid[1, 1], sharey=ax)

    if plot_table.empty:
        ax.text(
            0.5,
            0.5,
            "No valid drought events",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_xlabel("Duration (days)")
        ax.set_ylabel("Severity")
        ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.3)
        ax_histx.axis("off")
        ax_histy.axis("off")
        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)
        return

    duration = plot_table["Duration_Days"].astype(float)
    severity = plot_table["Severity"].astype(float)
    min_spi = plot_table["Min_Daily_SPI"].astype(float)
    mean_intensity = severity / duration
    plot_table["Mean_Intensity"] = mean_intensity

    # Spearman 更适合这里的单调关系判断；若 scipy 不可用，则保留 rho，p 值记为 NA。
    rho_text = "Spearman rho = NA\np = NA"
    try:
        from scipy.stats import spearmanr

        rho, p_value = spearmanr(duration, severity, nan_policy="omit")
        if np.isfinite(rho):
            p_text = "< 0.001" if p_value < 0.001 else f"= {p_value:.3f}"
            rho_text = f"Spearman rho = {rho:.2f}\np {p_text}"
    except Exception:
        rank_duration = duration.rank()
        rank_severity = severity.rank()
        rho = rank_duration.corr(rank_severity)
        if pd.notna(rho):
            rho_text = f"Spearman rho = {rho:.2f}\np = NA"

    # 用线性拟合线概括 Duration 与 Severity 的总体关系。
    if len(plot_table) >= 2 and duration.nunique() >= 2:
        slope, intercept = np.polyfit(duration.to_numpy(), severity.to_numpy(), deg=1)
        x_line = np.linspace(float(duration.min()), float(duration.max()), 100)
        y_line = slope * x_line + intercept
        ax.plot(
            x_line,
            y_line,
            color="#1f1f1f",
            linewidth=1.6,
            linestyle="-",
            label="Linear fit",
            zorder=4,
        )

    # 平均强度参考线：Severity = Duration x mean |SPI|。
    x_ref = np.linspace(0, float(duration.max()) * 1.04, 100)
    for mean_intensity in PLOT_CONFIG["reference_intensities"]:
        y_ref = x_ref * mean_intensity
        ax.plot(
            x_ref,
            y_ref,
            color="#7a7a7a",
            linewidth=0.85,
            linestyle="--",
            alpha=0.62,
            zorder=1,
        )
        visible_mask = y_ref <= float(severity.max()) * 0.96
        if not np.any(visible_mask):
            continue

        label_x = float(x_ref[visible_mask][-1] * 0.92)
        label_y = float(label_x * mean_intensity)

        # 按当前坐标变换计算屏幕角度，让标签真正平行于参考线。
        p0 = ax.transData.transform((label_x, label_y))
        p1 = ax.transData.transform((label_x + 1.0, label_y + mean_intensity))
        angle = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
        ax.annotate(
            f"mean |SPI|={mean_intensity:g}",
            xy=(label_x, label_y),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=7.5,
            color="#666666",
            ha="right",
            va="bottom",
            rotation=angle,
            rotation_mode="anchor",
        )

    scatter_for_colorbar = None
    marker_handles = []
    for level_name, marker in PLOT_CONFIG["level_markers"].items():
        level_table = plot_table[plot_table["Drought_Level"] == level_name]
        marker_handles.append(
            mlines.Line2D(
                [],
                [],
                color="black",
                marker=marker,
                linestyle="None",
                markersize=6,
                markerfacecolor="white",
                markeredgewidth=0.8,
                label=level_name,
            )
        )
        if level_table.empty:
            continue

        scatter = ax.scatter(
            level_table["Duration_Days"],
            level_table["Severity"],
            c=level_table["Mean_Intensity"],
            cmap="YlOrRd",
            vmin=float(plot_table["Mean_Intensity"].min()),
            vmax=float(plot_table["Mean_Intensity"].max()),
            s=PLOT_CONFIG["scatter_size"],
            marker=marker,
            edgecolors="black",
            linewidths=0.35,
            alpha=PLOT_CONFIG["scatter_alpha"],
            zorder=3,
        )
        scatter_for_colorbar = scatter

    ax_histx.hist(
        duration,
        bins="auto",
        color="#d9a441",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.82,
    )
    ax_histy.hist(
        severity,
        bins="auto",
        orientation="horizontal",
        color="#d9a441",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.82,
    )

    ax.set_xlabel("Duration (days)")
    ax.set_ylabel("Severity")
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.3)
    ax.text(
        0.03,
        0.97,
        f"n = {len(plot_table)} events\n{rho_text}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": "#888888", "alpha": 0.86},
    )
    ax.legend(
        handles=marker_handles,
        title="Drought level",
        loc="lower right",
        frameon=True,
        fontsize=8,
        title_fontsize=8.5,
    )

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histx.tick_params(axis="y", labelsize=8)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.tick_params(axis="x", labelsize=8)
    ax_histx.grid(True, axis="y", linestyle="--", linewidth=0.35, alpha=0.25)
    ax_histy.grid(True, axis="x", linestyle="--", linewidth=0.35, alpha=0.25)
    ax_histx.set_ylabel("Count", fontsize=8)
    ax_histx.set_title("Duration distribution", fontsize=8.5, pad=4)
    ax_histy.set_xlabel("Count", fontsize=8)
    ax_histy.set_title("Severity\ndistribution", fontsize=8.5, pad=4)

    if scatter_for_colorbar is not None:
        cbar = fig.colorbar(scatter_for_colorbar, ax=[ax, ax_histy], pad=0.025, shrink=0.88)
        cbar.set_label("Mean event intensity (Severity / Duration)")

    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def create_severity_duration_caption_png(output_file: Path) -> None:
    """
    生成严重度-持续时间散点图的图注说明 PNG。

    这张说明图不参与数据计算，只把图件定义、符号含义和事件判定口径单独写成图片，
    方便后续论文排版或汇报时与主图配套使用。
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=PLOT_CONFIG["scatter_note_figsize"],
        dpi=PLOT_CONFIG["dpi"],
    )
    ax.axis("off")

    note_text = (
        "Figure note: Severity-Duration relationship of daily SPI drought events at observation stations.\n"
        "Each point represents one drought event at one observation station. A drought event is defined as "
        f"at least {MIN_DURATION_DAYS} consecutive days with daily SPI20d < {DROUGHT_THRESHOLD}.\n"
        "Duration is the number of consecutive drought days. Severity is the cumulative absolute daily SPI "
        "during the event, i.e., sum(|SPI|).\n"
        "Point color indicates mean event intensity, calculated as Severity / Duration. "
        "Point shape indicates drought level classified by the event minimum daily SPI.\n"
        "Dashed reference lines represent constant mean event intensity, where Severity = Duration x mean |SPI|. "
        "Only mean |SPI| = 1.0 and 2.0 reference lines are shown to avoid overloading the figure. "
        "Marginal histograms show the distributions of duration and severity."
    )

    ax.text(
        0.02,
        0.92,
        note_text,
        ha="left",
        va="top",
        fontsize=10.5,
        linespacing=1.55,
        wrap=True,
        transform=ax.transAxes,
    )
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """
    主流程。

    步骤
    1. 创建输出目录并配置运行缓存。
    2. 读取 daily SPI NetCDF，并截取 WINDOW_START 到 WINDOW_END。
    3. 逐格点识别全年有效干旱事件。
    4. 将观测站点匹配到最近 SPI 格点，生成三张站点表。
    5. 保存全省格点事件 NC 和三张站点 CSV。
    6. 绘制站点严重度-持续时间 joint plot，并生成配套说明 PNG。
    7. 执行站点取值方法敏感性分析，并导出表格和图件。
    8. 无论成功或失败，最后关闭打开的 Dataset 并清理缓存目录。

    重要说明
    1. 主流程不重新计算 SPI，只读取前序脚本已经生成的 daily SPI。
    2. 所有事件识别都基于 daily SPI 原始时间顺序，不做重采样或逐月聚合。
    3. 站点表使用最近邻格点值，因此站点结果与其匹配格点完全一致。
    """
    overall_steps = 8

    with tqdm(
        total=overall_steps,
        desc="总体流程",
        colour=PROGRESS_COLORS["overall"],
        **OVERALL_TQDM_CONFIG,
    ) as overall_pbar:
        ds_spi: xr.Dataset | None = None
        ds_grid_events: xr.Dataset | None = None

        try:
            # 步骤 1：准备输出目录和临时缓存目录。
            with tqdm(
                total=1,
                desc="准备目录与缓存",
                colour=PROGRESS_COLORS["prepare"],
                **TQDM_CONFIG,
            ) as pbar_prepare:
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                configure_runtime_cache()
                pbar_prepare.update(1)
            overall_pbar.update(1)

            # 步骤 2：读取 daily SPI 数据，并裁剪到指定年度窗口。
            with tqdm(
                total=3,
                desc="读取daily SPI数据",
                colour=PROGRESS_COLORS["read"],
                **TQDM_CONFIG,
            ) as pbar_read:
                if not SPI_FILE.exists():
                    raise FileNotFoundError(f"找不到 SPI 文件：{SPI_FILE}")
                pbar_read.update(1)

                ds_spi = xr.open_dataset(SPI_FILE)
                if SPI_VAR not in ds_spi:
                    raise KeyError(f"变量 {SPI_VAR} 不存在，当前变量为：{list(ds_spi.data_vars)}")
                pbar_read.update(1)

                spi_window = ds_spi[SPI_VAR].sel({TIME_DIM: slice(WINDOW_START, WINDOW_END)})
                if spi_window.sizes.get(TIME_DIM, 0) == 0:
                    raise ValueError(f"分析窗口 {WINDOW_START} 到 {WINDOW_END} 内没有数据。")
                pbar_read.update(1)
            overall_pbar.update(1)

            # 将 xarray DataArray 拆成 numpy 数组和坐标，便于高频逐格点循环。
            spi_values = np.asarray(spi_window.values, dtype=float)
            times = pd.DatetimeIndex(pd.to_datetime(spi_window[TIME_DIM].values))
            latitudes = np.asarray(spi_window[LAT_NAME].values)
            longitudes = np.asarray(spi_window[LON_NAME].values)
            _, n_lat, n_lon = spi_values.shape

            print(f"分析窗口：{times[0].strftime('%Y-%m-%d')} 到 {times[-1].strftime('%Y-%m-%d')}")
            print(f"数据形状：{spi_values.shape}")
            print(
                f"干旱事件定义：连续 daily SPI < {DROUGHT_THRESHOLD} 且持续天数 >= {MIN_DURATION_DAYS}"
            )

            # grid_events[lat_idx][lon_idx] 保存该格点全年所有有效事件。
            # 每个事件是一个字段字典，字段由 extract_drought_events() 统一生成。
            grid_events: list[list[list[dict[str, object]]]] = [
                [[] for _ in range(n_lon)] for _ in range(n_lat)
            ]

            # 步骤 3：逐格点扫描 daily SPI 序列，识别全年有效干旱事件。
            total_grids = n_lat * n_lon
            total_event_count = 0
            with tqdm(
                total=total_grids,
                desc="逐格识别全年干旱事件",
                colour=PROGRESS_COLORS["compute"],
                **TQDM_CONFIG,
            ) as pbar_compute:
                for lat_idx in range(n_lat):
                    for lon_idx in range(n_lon):
                        events = extract_drought_events(
                            spi_1d=spi_values[:, lat_idx, lon_idx],
                            times=times,
                            threshold=DROUGHT_THRESHOLD,
                            min_duration_days=MIN_DURATION_DAYS,
                        )
                        grid_events[lat_idx][lon_idx] = events
                        total_event_count += len(events)
                        pbar_compute.update(1)
            overall_pbar.update(1)

            print(f"全省格点有效干旱事件总数：{total_event_count}")

            # 步骤 4：提取站点尺度结果。
            # 这里返回三张表：事件长表、年度统计表、逐日 SPI 表。
            station_event_table, station_annual_table, station_daily_table = build_station_tables(
                site_file=SITE_CSV,
                lats=latitudes,
                lons=longitudes,
                spi_values=spi_values,
                times=times,
                grid_events=grid_events,
            )
            overall_pbar.update(1)

            # 步骤 5：保存正式输出文件。
            # 保存顺序为：格点 NC、站点事件表、站点年度表、站点逐日表。
            with tqdm(
                total=4,
                desc="保存结果文件",
                colour=PROGRESS_COLORS["save"],
                **TQDM_CONFIG,
            ) as pbar_save:
                ds_grid_events = build_grid_event_dataset(
                    lats=latitudes,
                    lons=longitudes,
                    grid_events=grid_events,
                )
                ds_grid_events.to_netcdf(OUTPUT_GRID_EVENT_NC)
                pbar_save.update(1)

                station_event_table.to_csv(
                    OUTPUT_SITE_EVENT_CSV,
                    index=False,
                    encoding="utf-8-sig",
                )
                pbar_save.update(1)

                station_annual_table.to_csv(
                    OUTPUT_SITE_ANNUAL_CSV,
                    index=False,
                    encoding="utf-8-sig",
                )
                pbar_save.update(1)

                # 逐日 SPI 表通常行数为 站点数 x 天数。
                # 当前 2025 年非闰年为每站 365 行。
                station_daily_table.to_csv(
                    OUTPUT_SITE_DAILY_CSV,
                    index=False,
                    encoding="utf-8-sig",
                )
                pbar_save.update(1)
            overall_pbar.update(1)

            # 步骤 6：基于站点事件长表绘制严重度-持续时间 joint plot 和说明图。
            with tqdm(
                total=2,
                desc="生成站点散点图",
                colour=PROGRESS_COLORS["plot"],
                **TQDM_CONFIG,
            ) as pbar_plot:
                plot_station_severity_duration_scatter(
                    station_event_table=station_event_table,
                    output_file=OUTPUT_SCATTER_PNG,
                )
                pbar_plot.update(1)

                create_severity_duration_caption_png(
                    output_file=OUTPUT_SCATTER_NOTE_PNG,
                )
                pbar_plot.update(1)
            overall_pbar.update(1)

            # 步骤 7：敏感性分析，比较 Nearest、Bilinear 和 Mean3x3 三种站点取值方法。
            sensitivity_daily_table: pd.DataFrame
            sensitivity_event_table: pd.DataFrame
            sensitivity_annual_table: pd.DataFrame
            sensitivity_delta_table: pd.DataFrame
            sensitivity_summary_table: pd.DataFrame
            (
                sensitivity_daily_table,
                sensitivity_event_table,
                sensitivity_annual_table,
                sensitivity_delta_table,
                sensitivity_summary_table,
            ) = build_sensitivity_analysis_tables(
                site_file=SITE_CSV,
                lats=latitudes,
                lons=longitudes,
                spi_values=spi_values,
                times=times,
            )

            with tqdm(
                total=6,
                desc="保存敏感性分析",
                colour=PROGRESS_COLORS["save"],
                **TQDM_CONFIG,
            ) as pbar_sensitivity_save:
                sensitivity_daily_table.to_csv(
                    OUTPUT_SENSITIVITY_DAILY_CSV,
                    index=False,
                    encoding="utf-8-sig",
                )
                pbar_sensitivity_save.update(1)

                sensitivity_event_table.to_csv(
                    OUTPUT_SENSITIVITY_EVENT_CSV,
                    index=False,
                    encoding="utf-8-sig",
                )
                pbar_sensitivity_save.update(1)

                sensitivity_annual_table.to_csv(
                    OUTPUT_SENSITIVITY_ANNUAL_CSV,
                    index=False,
                    encoding="utf-8-sig",
                )
                pbar_sensitivity_save.update(1)

                sensitivity_delta_table.to_csv(
                    OUTPUT_SENSITIVITY_DELTA_CSV,
                    index=False,
                    encoding="utf-8-sig",
                )
                pbar_sensitivity_save.update(1)

                sensitivity_summary_table.to_csv(
                    OUTPUT_SENSITIVITY_SUMMARY_CSV,
                    index=False,
                    encoding="utf-8-sig",
                )
                pbar_sensitivity_save.update(1)

                plot_sensitivity_analysis_figure(
                    daily_table=sensitivity_daily_table,
                    delta_table=sensitivity_delta_table,
                    output_file=OUTPUT_SENSITIVITY_FIG_PNG,
                )
                pbar_sensitivity_save.update(1)
            overall_pbar.update(1)

            print(f"已保存全省格点事件 NC：{OUTPUT_GRID_EVENT_NC}")
            print(f"已保存站点事件长表 CSV：{OUTPUT_SITE_EVENT_CSV}")
            print(f"已保存站点年度统计 CSV：{OUTPUT_SITE_ANNUAL_CSV}")
            print(f"已保存站点逐日 SPI CSV：{OUTPUT_SITE_DAILY_CSV}")
            print(f"已保存站点严重度-持续时间散点图：{OUTPUT_SCATTER_PNG}")
            print(f"已保存站点严重度-持续时间散点图说明：{OUTPUT_SCATTER_NOTE_PNG}")
            print(f"已保存敏感性逐日对比表：{OUTPUT_SENSITIVITY_DAILY_CSV}")
            print(f"已保存敏感性事件长表：{OUTPUT_SENSITIVITY_EVENT_CSV}")
            print(f"已保存敏感性年度统计表：{OUTPUT_SENSITIVITY_ANNUAL_CSV}")
            print(f"已保存敏感性差异表：{OUTPUT_SENSITIVITY_DELTA_CSV}")
            print(f"已保存敏感性总体统计表：{OUTPUT_SENSITIVITY_SUMMARY_CSV}")
            print(f"已保存敏感性分析图：{OUTPUT_SENSITIVITY_FIG_PNG}")
            print(f"站点事件记录数：{len(station_event_table)}")
            print(f"站点年度统计记录数：{len(station_annual_table)}")
            print(f"站点逐日 SPI 记录数：{len(station_daily_table)}")
            print(f"敏感性逐日对比记录数：{len(sensitivity_daily_table)}")
            print(f"敏感性事件记录数：{len(sensitivity_event_table)}")
            print(f"敏感性年度统计记录数：{len(sensitivity_annual_table)}")

            if not station_event_table.empty:
                print("\n站点事件长表前 5 行预览：")
                print(station_event_table.head())

            if not station_annual_table.empty:
                print("\n站点年度统计表前 5 行预览：")
                print(station_annual_table.head())

            if not station_daily_table.empty:
                print("\n站点逐日 SPI 表前 5 行预览：")
                print(station_daily_table.head())

        finally:
            if ds_spi is not None:
                ds_spi.close()
            if ds_grid_events is not None:
                ds_grid_events.close()

            with tqdm(
                total=1,
                desc="清理运行缓存",
                colour=PROGRESS_COLORS["cleanup"],
                **TQDM_CONFIG,
            ) as pbar_cleanup:
                cleanup_runtime_cache()
                pbar_cleanup.update(1)
            overall_pbar.update(1)

            print(f"已清理本次运行缓存目录：{RUN_CACHE_DIR}")


if __name__ == "__main__":
    main()
