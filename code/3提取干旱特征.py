"""
福建省 2025 年 1-2 月 SPI 干旱特征提取脚本

脚本用途
1. 读取 2025 年逐月 SPI1 NetCDF 数据。
2. 针对“2025 年 1-2 月”这个目标时段，逐格点识别与其重叠的干旱事件。
3. 计算四个干旱特征：
   - Magnitude：事件期间最小 SPI 值
   - Duration：事件持续月数
   - Severity：事件期间 |SPI| 的累加和
   - Onset：事件开始月份，保存为 YYYYMM
4. 输出两个结果文件：
   - NC 文件：英文命名，便于后续程序读取
   - CSV 文件：中文命名，便于人工查看
5. 关键步骤使用 tqdm 彩色进度条，显示百分比、当前量/总量、耗时、剩余时间和速度。
6. 脚本结束时删除本次运行产生的缓存目录。

说明
1. 当前只使用 2025 年数据，不向前补 2024 年 12 月。
2. 因此若真实干旱事件在 2025 年之前已开始，Onset 和 Duration 会是“截断结果”。
3. 本脚本不安装任何依赖，默认你已经准备好运行环境。
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
# 集中管理输入、输出和本次运行缓存目录，后续如需更换年份或目录，只改这里。
# ============================================================
# SPI 输入文件：
# 由前一步 SPI 计算脚本生成，当前文件中包含 2025-01 到 2025-12 的逐月 SPI1 栅格。
SPI_FILE = Path(r"E:\forest_microclimate\ForestMicroclimate\results\SPI_result\Fujian_SPI1_2025.nc")

# 结果输出目录：
# 所有最终结果写入这里。
OUTPUT_DIR = Path(r"E:\forest_microclimate\ForestMicroclimate\results\SPI_features")

# 观测站点表：
# 用于提取“观测站所在位置”对应的目标干旱事件特征。
SITE_CSV = Path(r"E:\forest_microclimate\ForestMicroclimate\Tensor_LatLong.csv")

# 福建省行政边界：
# 绘制经纬度点位图时使用地理坐标系边界，确保边界、格点和站点都在同一坐标系下叠加。
FUJIAN_SHP_GEOGRAPHIC = Path(r"E:\forest_microclimate\ForestMicroclimate\Fujian_Shp\福建省行政边界.shp")

# 输出文件命名规则：
# 1. NC 文件按你的要求保留英文命名；
# 2. 其他文件全部使用中文命名。
OUTPUT_NC = OUTPUT_DIR / "fujian_drought_features_2025_jan_feb.nc"
OUTPUT_CSV = OUTPUT_DIR / "福建省2025年1-2月SPI干旱特征表.csv"
OUTPUT_SITE_CSV = OUTPUT_DIR / "福建省观测站2025年1-2月干旱特征表.csv"
OUTPUT_PANEL_PNG = OUTPUT_DIR / "福建省观测站干旱特征四图面板.png"
OUTPUT_SCATTER_PNG = OUTPUT_DIR / "福建省观测站严重度-持续时间散点图.png"
OUTPUT_RANKING_PNG = OUTPUT_DIR / "福建省观测站干旱严重度站点排序图.png"

# 本次脚本运行缓存目录：
# 用于接管 Python 临时目录，尽量把运行期产生的缓存集中到一个可删除的位置。
# 脚本结束后会在 finally 中自动清理。
RUN_CACHE_DIR = OUTPUT_DIR / "_本次运行缓存_提取干旱特征"
TEMP_DIR = RUN_CACHE_DIR / "temp"


# ============================================================
# 1. 数据维度与变量参数
# 这些参数用于适配 NetCDF 文件中的变量名和坐标维度名。
# 若未来更换数据源，只要文件结构变化不大，通常只需调整这里。
# ============================================================
# SPI_VAR：
# NetCDF 中用于提取干旱特征的变量名。
SPI_VAR = "SPI1"

# TIME_DIM / LAT_NAME / LON_NAME：
# NetCDF 中时间、纬度、经度维度的名称。
TIME_DIM = "time"
LAT_NAME = "lat"
LON_NAME = "lon"

# 观测站点表字段名：
# 如果未来站点表换了列名，只需修改这里，不必改后面的提取逻辑。
SITE_ID_FIELD = "Site_ID"
SITE_LON_FIELD = "Longitude"
SITE_LAT_FIELD = "Latitude"


# ============================================================
# 2. 干旱识别参数
# 这部分参数直接决定“什么算一次干旱事件”以及“提取哪一段目标事件”。
# ============================================================
# THRESHOLD：
# 干旱判定阈值。这里采用 WMO 常用轻旱起点 -0.5。
# 只有 SPI < -0.5 的月份才视为处于干旱过程。
THRESHOLD = -0.5

# WINDOW_START / WINDOW_END：
# 分析窗口。脚本会先从年度 SPI 数据中截取这个时间范围，再在范围内识别连续干旱事件。
# 当前你只提供了 2025 年数据，因此分析窗口设置为 2025 全年。
WINDOW_START = "2025-01"
WINDOW_END = "2025-12"

# EVENT_START / EVENT_END：
# 目标事件窗口。脚本会从所有连续干旱事件中，筛选出“与这段时间有重叠”的事件。
# 你已明确要提取 2025 年 1-2 月的冬季事件，因此固定为 2025-01 到 2025-02。
EVENT_START = "2025-01"
EVENT_END = "2025-02"


# ============================================================
# 3. 进度条参数
# 使用不同颜色区分不同类型的步骤，方便从终端快速判断脚本当前运行到哪一步。
# ============================================================
# bar_format：
# - l_bar：左侧描述和百分比
# - bar：彩色进度条主体
# - n_fmt/total_fmt：当前量/总量
# - elapsed：已耗时
# - remaining：预计剩余时间
# - rate_fmt：处理速度
TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

# TQDM_CONFIG：
# 大多数子步骤共用的进度条配置。
TQDM_CONFIG = {
    "bar_format": TQDM_BAR_FORMAT,
    "dynamic_ncols": True,
    "leave": False,
}

# OVERALL_TQDM_CONFIG：
# 总进度条保留在终端末尾，便于脚本结束后回看总体完成状态。
OVERALL_TQDM_CONFIG = {
    **TQDM_CONFIG,
    "leave": True,
}

# PROGRESS_COLORS：
# 不同步骤使用不同颜色，前提是当前终端支持 ANSI 颜色显示。
PROGRESS_COLORS = {
    "overall": "green",
    "prepare": "white",
    "read": "cyan",
    "compute": "yellow",
    "station": "red",
    "plot": "magenta",
    "save": "blue",
    "cleanup": "magenta",
}

# CSV 备注说明：
# 将四个特征的计算口径直接写入表格，避免结果文件脱离脚本后难以理解。
FEATURE_NOTE_TEXT = (
    "强度Magnitude=目标干旱事件期间最小SPI；"
    "持续时间Duration=连续SPI<-0.5的月数；"
    "严重度Severity=目标干旱事件期间各月|SPI|之和；"
    "开始时间Onset=事件首次进入SPI<-0.5的月份，格式为YYYYMM。"
)


# ============================================================
# 4. 绘图参数
# 所有图件样式集中放在这里，后续如果想改站点颜色、大小、透明度、DPI 等，不用进入函数内部。
# ============================================================
PLOT_CONFIG = {
    # dpi：图片分辨率。论文或报告图通常可设为 300。
    "dpi": 300,
    # panel_figsize：四图面板尺寸，单位是英寸。
    "panel_figsize": (13.5, 10.5),
    # scatter_figsize：严重度-持续时间散点图尺寸。
    "scatter_figsize": (8.8, 6.4),
    # ranking_figsize_base_height：排序图基础高度，脚本会根据站点数量自动加高。
    "ranking_figsize_width": 10.5,
    "ranking_figsize_base_height": 4.8,
    # boundary_color / boundary_linewidth：福建省边界线颜色和宽度。
    "boundary_color": "#202020",
    "boundary_linewidth": 0.9,
    # grid_alpha：全省格点背景透明度；越小越淡，站点越突出。
    "grid_alpha": 0.72,
    # station_marker：站点符号形状，常用 "o" 圆点、"^" 三角、"s" 方形。
    "station_marker": "o",
    # station_size：站点符号大小。
    "station_size": 26,
    # station_edgecolor / station_linewidth：站点外边框颜色和线宽。
    "station_edgecolor": "black",
    "station_linewidth": 0.35,
    # station_alpha：站点透明度。
    "station_alpha": 0.95,
    # extent_padding：边界外扩距离，单位是经纬度度数，避免地图贴边。
    "extent_padding": 0.18,
    # gridline_alpha：经纬网透明度。
    "gridline_alpha": 0.25,
}

# 字体候选：
# matplotlib 会按顺序尝试这些字体。不同电脑中文字体名称可能不同，因此保留多个候选。
FONT_CANDIDATES = [
    "SimHei",
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]


def configure_runtime_cache() -> None:
    """
    配置本次运行缓存目录。

    目的
    1. 尽量把 Python 运行期间的临时文件统一写入 RUN_CACHE_DIR。
    2. 便于脚本结束后一次性删除本次运行产生的缓存。

    注意
    1. 这里不会删除正式输出目录中的结果文件。
    2. 只会删除本脚本显式指定的缓存目录。
    """
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(TEMP_DIR)
    os.environ["TEMP"] = str(TEMP_DIR)
    tempfile.tempdir = str(TEMP_DIR)


def extract_target_drought_event(
    spi_1d: np.ndarray,
    times: pd.DatetimeIndex,
    threshold: float,
    event_start: str,
    event_end: str,
) -> dict[str, float]:
    """
    对单个格点的 SPI 时间序列提取目标干旱事件的四个特征。

    参数说明
    spi_1d：
        单个格点的月尺度 SPI 序列，长度应与 times 一致。
    times：
        SPI 序列对应的时间索引。
    threshold：
        干旱判定阈值。SPI 小于该值时认为该月处于干旱状态。
    event_start / event_end：
        目标事件窗口。只保留与该时间窗有重叠的连续干旱事件。

    返回值说明
    magnitude：
        目标干旱事件期间的最小 SPI，数值越小表示强度越大。
    duration：
        连续 SPI < threshold 的月数。
    severity：
        干旱事件期间所有月份 |SPI| 的累加和。
    onset_yyyymm：
        事件开始时间，编码为 YYYYMM，例如 202501。

    处理规则
    1. 连续月份 SPI < threshold 视为一次干旱事件。
    2. 遇到 NaN 时，当前连续事件立即中断。
    3. 若多个事件都与目标时段重叠，则优先选最强事件：
       先比较最小 SPI，再比较 severity，再比较 duration。
    """
    empty_result = {
        "magnitude": np.nan,
        "duration": np.nan,
        "severity": np.nan,
        "onset_yyyymm": np.nan,
    }

    if np.all(np.isnan(spi_1d)):
        return empty_result

    target_start = pd.Timestamp(event_start)
    target_end = pd.Timestamp(event_end)

    events: list[dict[str, np.ndarray | pd.DatetimeIndex]] = []
    in_drought = False
    segment_start = -1

    for idx, value in enumerate(spi_1d):
        # NaN 视为当前事件中断，避免缺测值把两个不连续片段错误拼成一个事件。
        if np.isnan(value):
            if in_drought:
                events.append(
                    {
                        "spi": spi_1d[segment_start:idx].copy(),
                        "times": times[segment_start:idx],
                    }
                )
                in_drought = False
                segment_start = -1
            continue

        # 进入一次新的连续干旱事件。
        if value < threshold and not in_drought:
            in_drought = True
            segment_start = idx
            continue

        # 连续干旱在本月结束，收集该事件。
        if value >= threshold and in_drought:
            events.append(
                {
                    "spi": spi_1d[segment_start:idx].copy(),
                    "times": times[segment_start:idx],
                }
            )
            in_drought = False
            segment_start = -1

    # 如果序列结束时仍在干旱状态，需要把尾段补进事件列表。
    if in_drought:
        events.append(
            {
                "spi": spi_1d[segment_start:].copy(),
                "times": times[segment_start:],
            }
        )

    if not events:
        return empty_result

    target_events = []
    for event in events:
        event_times = event["times"]
        event_start_time = event_times[0]
        event_end_time = event_times[-1]

        # 只保留与目标事件窗口有时间重叠的干旱事件。
        if event_start_time <= target_end and event_end_time >= target_start:
            target_events.append(event)

    if not target_events:
        return empty_result

    # 选最强事件：
    # 1. magnitude 更低者优先；
    # 2. 若 magnitude 相同，则 severity 更大者优先；
    # 3. 若还相同，则 duration 更长者优先。
    worst_event = min(
        target_events,
        key=lambda event: (
            float(np.min(event["spi"])),
            -float(np.sum(np.abs(event["spi"]))),
            -len(event["spi"]),
        ),
    )

    worst_spi = np.asarray(worst_event["spi"], dtype=float)
    worst_times = worst_event["times"]
    onset_time = pd.Timestamp(worst_times[0])

    return {
        "magnitude": float(np.min(worst_spi)),
        "duration": float(len(worst_spi)),
        "severity": float(np.sum(np.abs(worst_spi))),
        "onset_yyyymm": float(onset_time.year * 100 + onset_time.month),
    }


def build_output_dataset(
    lats: np.ndarray,
    lons: np.ndarray,
    magnitude_array: np.ndarray,
    duration_array: np.ndarray,
    severity_array: np.ndarray,
    onset_array: np.ndarray,
) -> xr.Dataset:
    """
    组装 NC 输出数据集。

    说明
    1. NC 文件更适合保留原始栅格结构，便于后续 GIS、栅格计算和空间分析。
    2. 变量属性里保留了阈值、分析窗口和目标事件窗口，便于后期回溯参数。
    """
    return xr.Dataset(
        data_vars={
            "magnitude": (
                [LAT_NAME, LON_NAME],
                magnitude_array,
                {
                    "long_name": "Drought magnitude",
                    "units": "dimensionless",
                    "description": "Minimum SPI during the target drought event",
                },
            ),
            "duration": (
                [LAT_NAME, LON_NAME],
                duration_array,
                {
                    "long_name": "Drought duration",
                    "units": "months",
                    "description": "Number of consecutive months with SPI below the threshold",
                },
            ),
            "severity": (
                [LAT_NAME, LON_NAME],
                severity_array,
                {
                    "long_name": "Drought severity",
                    "units": "dimensionless",
                    "description": "Sum of absolute SPI values during the target drought event",
                },
            ),
            "onset": (
                [LAT_NAME, LON_NAME],
                onset_array,
                {
                    "long_name": "Drought onset",
                    "units": "YYYYMM",
                    "description": "Event start month stored as year*100 + month",
                },
            ),
        },
        coords={
            LAT_NAME: lats,
            LON_NAME: lons,
        },
        attrs={
            "title": "Fujian drought features extracted from SPI1 for the Jan-Feb 2025 target event",
            "source_file": str(SPI_FILE),
            "spi_variable": SPI_VAR,
            "threshold": THRESHOLD,
            "analysis_window": f"{WINDOW_START} to {WINDOW_END}",
            "target_event_window": f"{EVENT_START} to {EVENT_END}",
            "note": "Only 2025 data are used; onset and duration may be truncated if the drought began before 2025-01.",
        },
    )


def build_output_table(
    lats: np.ndarray,
    lons: np.ndarray,
    magnitude_array: np.ndarray,
    duration_array: np.ndarray,
    severity_array: np.ndarray,
    onset_array: np.ndarray,
) -> pd.DataFrame:
    """
    组装 CSV 输出表。

    说明
    1. CSV 用于人工查看和表格统计，因此将二维栅格展开为逐格点记录。
    2. 只保留识别出目标干旱事件的格点，去掉无事件或全缺测格点。
    3. 增加“备注”字段，把四个特征的计算口径直接写进表格。
    """
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    table = pd.DataFrame(
        {
            "纬度": lat_grid.ravel(),
            "经度": lon_grid.ravel(),
            "强度Magnitude": magnitude_array.ravel(),
            "持续时间Duration(月)": duration_array.ravel(),
            "严重度Severity": severity_array.ravel(),
            "开始时间Onset(YYYYMM)": onset_array.ravel(),
        }
    )

    table = table.dropna(subset=["强度Magnitude"]).copy()
    table["持续时间Duration(月)"] = table["持续时间Duration(月)"].astype("Int64")
    table["开始时间Onset(YYYYMM)"] = table["开始时间Onset(YYYYMM)"].round().astype("Int64")
    table["备注"] = FEATURE_NOTE_TEXT
    return table


def extract_station_feature_table(
    site_file: Path,
    lats: np.ndarray,
    lons: np.ndarray,
    magnitude_array: np.ndarray,
    duration_array: np.ndarray,
    severity_array: np.ndarray,
    onset_array: np.ndarray,
) -> pd.DataFrame:
    """
    提取观测站点对应的目标干旱事件特征表。

    参数说明
    site_file：
        观测站点经纬度表，必须至少包含站点编号、经度、纬度三列。
    lats / lons：
        SPI 栅格的纬度和经度坐标。
    magnitude_array / duration_array / severity_array / onset_array：
        已经按格点计算好的四个干旱特征数组。

    提取方法
    1. 采用“最近邻格点匹配”。
    2. 对每个站点，分别找到最接近的纬度索引和经度索引。
    3. 用该格点的四个干旱特征作为站点在本次事件中的特征值。

    输出字段说明
    1. 保留原始站点编号、经纬度。
    2. 额外记录匹配到的最近格点经纬度，便于后续检查空间对应关系。
    3. 输出四个干旱特征，字段命名与栅格表保持一致，方便对照。
    4. 增加“备注”字段，明确四个特征的计算方式。
    """
    if not site_file.exists():
        raise FileNotFoundError(f"找不到观测站点表：{site_file}")

    site_table = pd.read_csv(site_file)
    required_columns = {SITE_ID_FIELD, SITE_LON_FIELD, SITE_LAT_FIELD}
    missing_columns = required_columns.difference(site_table.columns)
    if missing_columns:
        raise KeyError(f"观测站点表缺少必要字段：{sorted(missing_columns)}")

    output_rows: list[dict[str, float | int]] = []

    with tqdm(
        total=len(site_table),
        desc="提取站点特征",
        colour=PROGRESS_COLORS["station"],
        **TQDM_CONFIG,
    ) as pbar_station:
        for _, row in site_table.iterrows():
            site_id = row[SITE_ID_FIELD]
            site_lon = float(row[SITE_LON_FIELD])
            site_lat = float(row[SITE_LAT_FIELD])

            # 最近邻索引匹配：
            # 对规则经纬网格，这种方式直观、稳定，且不依赖额外插值库。
            lat_idx = int(np.abs(lats - site_lat).argmin())
            lon_idx = int(np.abs(lons - site_lon).argmin())

            output_rows.append(
                {
                    "站点编号": site_id,
                    "站点经度": site_lon,
                    "站点纬度": site_lat,
                    "匹配格点经度": float(lons[lon_idx]),
                    "匹配格点纬度": float(lats[lat_idx]),
                    "强度Magnitude": float(magnitude_array[lat_idx, lon_idx]),
                    "持续时间Duration(月)": pd.array([duration_array[lat_idx, lon_idx]], dtype="Float64")[0],
                    "严重度Severity": float(severity_array[lat_idx, lon_idx]),
                    "开始时间Onset(YYYYMM)": pd.array([onset_array[lat_idx, lon_idx]], dtype="Float64")[0],
                }
            )
            pbar_station.update(1)

    station_feature_table = pd.DataFrame(output_rows)
    station_feature_table["持续时间Duration(月)"] = station_feature_table["持续时间Duration(月)"].round().astype("Int64")
    station_feature_table["开始时间Onset(YYYYMM)"] = station_feature_table["开始时间Onset(YYYYMM)"].round().astype("Int64")
    station_feature_table["备注"] = FEATURE_NOTE_TEXT
    return station_feature_table


def load_fujian_boundary(shp_file: Path):
    """
    读取福建省地理坐标系边界。

    参数说明
    shp_file：
        福建省行政边界 shp 文件，要求为经纬度坐标系，便于和 SPI 格点、观测站经纬度直接叠加。

    返回值
    boundary：
        geopandas.GeoDataFrame，用于后续地图绘制。

    设计说明
    1. geopandas 在这里延迟导入，避免脚本启动阶段就加载绘图库。
    2. 若边界文件坐标系不是 EPSG:4326，会尽量转换到 EPSG:4326。
    """
    if not shp_file.exists():
        raise FileNotFoundError(f"找不到福建省边界文件：{shp_file}")

    import geopandas as gpd

    boundary = gpd.read_file(shp_file)
    if boundary.crs is not None and boundary.crs.to_epsg() != 4326:
        boundary = boundary.to_crs(epsg=4326)
    return boundary


def configure_matplotlib_for_chinese() -> None:
    """
    配置 matplotlib 中文显示。

    说明
    1. 设置多个中文字体候选，matplotlib 会按顺序寻找系统中可用字体。
    2. axes.unicode_minus=False 用于避免负号显示成方块。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = FONT_CANDIDATES
    plt.rcParams["axes.unicode_minus"] = False


def get_map_extent(boundary, lats: np.ndarray, lons: np.ndarray) -> tuple[float, float, float, float]:
    """
    计算地图显示范围。

    优先使用福建省边界范围；若边界读取异常或为空，则回退到 SPI 格点经纬度范围。
    返回顺序为 xmin, xmax, ymin, ymax。
    """
    padding = PLOT_CONFIG["extent_padding"]
    if boundary is not None and not boundary.empty:
        xmin, ymin, xmax, ymax = boundary.total_bounds
    else:
        xmin, xmax = float(np.nanmin(lons)), float(np.nanmax(lons))
        ymin, ymax = float(np.nanmin(lats)), float(np.nanmax(lats))
    return xmin - padding, xmax + padding, ymin - padding, ymax + padding


def draw_boundary_and_grid(ax, boundary, extent: tuple[float, float, float, float]) -> None:
    """
    绘制福建省边界、设置地图范围和经纬网。

    参数说明
    ax：
        matplotlib 坐标轴。
    boundary：
        福建省边界 GeoDataFrame。
    extent：
        地图显示范围，顺序为 xmin, xmax, ymin, ymax。
    """
    if boundary is not None and not boundary.empty:
        boundary.boundary.plot(
            ax=ax,
            color=PLOT_CONFIG["boundary_color"],
            linewidth=PLOT_CONFIG["boundary_linewidth"],
            zorder=5,
        )

    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.1f}°")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.1f}°")
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=PLOT_CONFIG["gridline_alpha"])


def format_site_id(site_id) -> str:
    """
    格式化站点编号。

    目的
    1. CSV 读取后，站点编号有时会被 pandas 识别为浮点数，绘图时显示成 95332217.0。
    2. 若编号是整数型浮点数，则去掉末尾 .0，保证排序图纵轴更清爽。
    """
    if pd.isna(site_id):
        return ""
    try:
        site_id_float = float(site_id)
        if site_id_float.is_integer():
            return str(int(site_id_float))
    except (TypeError, ValueError):
        pass
    return str(site_id)


def add_station_points(ax, station_feature_table: pd.DataFrame, color_values, cmap, norm=None) -> None:
    """
    在地图上叠加观测站点。

    参数说明
    color_values：
        用于给站点着色的数组，通常与当前子图指标一致。
    cmap / norm：
        matplotlib 颜色映射和归一化规则。
    """
    ax.scatter(
        station_feature_table["站点经度"],
        station_feature_table["站点纬度"],
        c=color_values,
        cmap=cmap,
        norm=norm,
        s=PLOT_CONFIG["station_size"],
        marker=PLOT_CONFIG["station_marker"],
        edgecolors=PLOT_CONFIG["station_edgecolor"],
        linewidths=PLOT_CONFIG["station_linewidth"],
        alpha=PLOT_CONFIG["station_alpha"],
        zorder=10,
        label="观测站点",
    )


def plot_station_feature_panel(
    boundary,
    lats: np.ndarray,
    lons: np.ndarray,
    magnitude_array: np.ndarray,
    duration_array: np.ndarray,
    severity_array: np.ndarray,
    onset_array: np.ndarray,
    station_feature_table: pd.DataFrame,
    output_file: Path,
) -> None:
    """
    绘制“福建省观测站干旱特征四图面板”。

    图件设计
    1. 四个子图分别展示 Magnitude、Duration、Severity、Onset。
    2. 每个子图先绘制全省格点背景，展示全省空间格局。
    3. 再叠加所有观测站点，不标站点编号，避免图面拥挤。
    4. 站点颜色与当前子图指标一致，便于比较站点与周边格点背景。
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    extent = get_map_extent(boundary, lats, lons)

    fig, axes = plt.subplots(2, 2, figsize=PLOT_CONFIG["panel_figsize"], dpi=PLOT_CONFIG["dpi"])
    axes = axes.ravel()

    panel_items = [
        {
            "title": "Magnitude",
            "array": magnitude_array,
            "station_values": station_feature_table["强度Magnitude"],
            "cmap": "YlOrRd_r",
            "label": "",
            "norm": None,
        },
        {
            "title": "Duration",
            "array": duration_array,
            "station_values": station_feature_table["持续时间Duration(月)"],
            "cmap": "YlGnBu",
            "label": "months",
            "norm": mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], ncolors=256),
        },
        {
            "title": "Severity",
            "array": severity_array,
            "station_values": station_feature_table["严重度Severity"],
            "cmap": "OrRd",
            "label": "",
            "norm": None,
        },
        {
            "title": "Onset",
            "array": onset_array,
            "station_values": station_feature_table["开始时间Onset(YYYYMM)"],
            "cmap": "Set2",
            "label": "YYYYMM",
            "norm": mcolors.BoundaryNorm([202500.5, 202501.5, 202502.5, 202503.5], ncolors=256),
        },
    ]

    for ax, item in zip(axes, panel_items):
        mesh = ax.pcolormesh(
            lon_grid,
            lat_grid,
            item["array"],
            shading="auto",
            cmap=item["cmap"],
            norm=item["norm"],
            alpha=PLOT_CONFIG["grid_alpha"],
            zorder=1,
        )
        draw_boundary_and_grid(ax, boundary, extent)
        add_station_points(
            ax=ax,
            station_feature_table=station_feature_table,
            color_values=item["station_values"],
            cmap=item["cmap"],
            norm=item["norm"],
        )
        ax.set_title(item["title"], fontsize=12)
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.84, pad=0.02)
        cbar.set_label(item["label"])

    fig.suptitle("Drought Features at Observation Stations in Fujian", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def plot_station_severity_duration_scatter(
    station_feature_table: pd.DataFrame,
    output_file: Path,
) -> None:
    """
    绘制严重度-持续时间散点图。

    图件含义
    1. 横轴为 Duration，纵轴为 Severity。
    2. 点颜色表示 Magnitude，颜色越深通常代表强度越强。
    3. 所有观测站点都绘制出来，不标编号。
    """
    import matplotlib.pyplot as plt

    plot_table = station_feature_table.dropna(
        subset=["强度Magnitude", "持续时间Duration(月)", "严重度Severity"]
    ).copy()

    fig, ax = plt.subplots(figsize=PLOT_CONFIG["scatter_figsize"], dpi=PLOT_CONFIG["dpi"])
    scatter = ax.scatter(
        plot_table["持续时间Duration(月)"],
        plot_table["严重度Severity"],
        c=plot_table["强度Magnitude"],
        cmap="YlOrRd_r",
        s=42,
        edgecolors="black",
        linewidths=0.35,
        alpha=0.9,
    )
    ax.set_title("Severity-Duration Relationship at Observation Stations", fontsize=13, fontweight="bold")
    ax.set_xlabel("Duration (months)")
    ax.set_ylabel("Severity")
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Magnitude")
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def plot_station_severity_ranking(
    station_feature_table: pd.DataFrame,
    output_file: Path,
) -> None:
    """
    绘制所有观测站点严重度排序图。

    图件含义
    1. 按 Severity 从大到小排列全部站点。
    2. 横轴为严重度，纵轴为站点编号。
    3. 颜色用 Magnitude 表示，便于同时观察强度和累计严重度。
    """
    import matplotlib.pyplot as plt

    plot_table = station_feature_table.dropna(subset=["严重度Severity"]).copy()
    plot_table = plot_table.sort_values("严重度Severity", ascending=True)
    station_count = len(plot_table)
    fig_height = max(
        PLOT_CONFIG["ranking_figsize_base_height"],
        min(18.0, 0.22 * station_count + 2.2),
    )

    fig, ax = plt.subplots(
        figsize=(PLOT_CONFIG["ranking_figsize_width"], fig_height),
        dpi=PLOT_CONFIG["dpi"],
    )
    bars = ax.barh(
        plot_table["站点编号"].map(format_site_id),
        plot_table["严重度Severity"],
        color="#D55E00",
        alpha=0.82,
        edgecolor="black",
        linewidth=0.25,
    )

    ax.set_title("Severity Ranking of Observation Stations", fontsize=13, fontweight="bold")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Site ID")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.45, alpha=0.3)

    for bar in bars:
        width = bar.get_width()
        if np.isfinite(width):
            ax.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f" {width:.2f}",
                va="center",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def create_station_feature_figures(
    lats: np.ndarray,
    lons: np.ndarray,
    magnitude_array: np.ndarray,
    duration_array: np.ndarray,
    severity_array: np.ndarray,
    onset_array: np.ndarray,
    station_feature_table: pd.DataFrame,
) -> None:
    """
    统一生成本次需求的三张图。

    输出内容
    1. 福建省观测站干旱特征四图面板.png
    2. 福建省观测站严重度-持续时间散点图.png
    3. 福建省观测站干旱严重度站点排序图.png
    """
    configure_matplotlib_for_chinese()
    boundary = load_fujian_boundary(FUJIAN_SHP_GEOGRAPHIC)

    with tqdm(
        total=3,
        desc="生成站点图件",
        colour=PROGRESS_COLORS["plot"],
        **TQDM_CONFIG,
    ) as pbar_plot:
        plot_station_feature_panel(
            boundary=boundary,
            lats=lats,
            lons=lons,
            magnitude_array=magnitude_array,
            duration_array=duration_array,
            severity_array=severity_array,
            onset_array=onset_array,
            station_feature_table=station_feature_table,
            output_file=OUTPUT_PANEL_PNG,
        )
        pbar_plot.update(1)

        plot_station_severity_duration_scatter(
            station_feature_table=station_feature_table,
            output_file=OUTPUT_SCATTER_PNG,
        )
        pbar_plot.update(1)

        plot_station_severity_ranking(
            station_feature_table=station_feature_table,
            output_file=OUTPUT_RANKING_PNG,
        )
        pbar_plot.update(1)


def cleanup_runtime_cache() -> None:
    """
    删除本次运行缓存目录。

    说明
    1. 只删除 RUN_CACHE_DIR，不删除输出结果。
    2. ignore_errors=True 可以避免缓存已经不存在时抛出异常。
    """
    shutil.rmtree(RUN_CACHE_DIR, ignore_errors=True)


def main() -> None:
    """
    主流程。

    步骤说明
    1. 准备输出目录和缓存目录。
    2. 读取 SPI 数据并截取分析窗口。
    3. 逐格点提取目标干旱事件的四个特征。
    4. 提取观测站点对应的干旱特征。
    5. 绘制观测站点干旱特征图。
    6. 分别保存 NC 和 CSV。
    7. 无论成功或失败，最后都清理本次缓存目录。
    """
    overall_steps = 7

    with tqdm(
        total=overall_steps,
        desc="总体流程",
        colour=PROGRESS_COLORS["overall"],
        **OVERALL_TQDM_CONFIG,
    ) as overall_pbar:
        ds_spi: xr.Dataset | None = None
        ds_feature: xr.Dataset | None = None

        try:
            # ========================================================
            # 步骤 1：准备目录与缓存
            # ========================================================
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

            # ========================================================
            # 步骤 2：读取 SPI 数据
            # ========================================================
            with tqdm(
                total=3,
                desc="读取SPI数据",
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

            spi_values = spi_window.values
            times = pd.to_datetime(spi_window[TIME_DIM].values)
            latitudes = spi_window[LAT_NAME].values
            longitudes = spi_window[LON_NAME].values
            _, n_lat, n_lon = spi_values.shape

            print(f"分析窗口：{times[0].strftime('%Y-%m')} 到 {times[-1].strftime('%Y-%m')}")
            print(f"目标事件窗口：{EVENT_START} 到 {EVENT_END}")
            print(f"数据形状：{spi_values.shape}")

            # ========================================================
            # 步骤 3：逐格点计算四个干旱特征
            # ========================================================
            magnitude_array = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
            duration_array = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
            severity_array = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
            onset_array = np.full((n_lat, n_lon), np.nan, dtype=np.float32)

            total_grids = n_lat * n_lon
            with tqdm(
                total=total_grids,
                desc="逐格提取特征",
                colour=PROGRESS_COLORS["compute"],
                **TQDM_CONFIG,
            ) as pbar_compute:
                for lat_idx in range(n_lat):
                    for lon_idx in range(n_lon):
                        result = extract_target_drought_event(
                            spi_1d=spi_values[:, lat_idx, lon_idx],
                            times=times,
                            threshold=THRESHOLD,
                            event_start=EVENT_START,
                            event_end=EVENT_END,
                        )

                        magnitude_array[lat_idx, lon_idx] = result["magnitude"]
                        duration_array[lat_idx, lon_idx] = result["duration"]
                        severity_array[lat_idx, lon_idx] = result["severity"]
                        onset_array[lat_idx, lon_idx] = result["onset_yyyymm"]

                        pbar_compute.update(1)
            overall_pbar.update(1)

            valid_mask = ~np.isnan(magnitude_array)
            valid_count = int(np.count_nonzero(valid_mask))

            print("干旱特征提取完成。")
            print(f"发生目标干旱事件的格点数：{valid_count}")
            if valid_count > 0:
                print(f"Magnitude 范围：{np.nanmin(magnitude_array):.3f} 到 {np.nanmax(magnitude_array):.3f}")
                print(f"Duration 范围：{int(np.nanmin(duration_array))} 到 {int(np.nanmax(duration_array))} 个月")
                print(f"Severity 范围：{np.nanmin(severity_array):.3f} 到 {np.nanmax(severity_array):.3f}")
                print(f"Onset 范围：{int(np.nanmin(onset_array))} 到 {int(np.nanmax(onset_array))}")

            # ========================================================
            # 步骤 4：提取观测站点特征
            # 将站点经纬度匹配到最近 SPI 格点，输出一份站点尺度的事件特征表。
            # ========================================================
            station_feature_table = extract_station_feature_table(
                site_file=SITE_CSV,
                lats=latitudes,
                lons=longitudes,
                magnitude_array=magnitude_array,
                duration_array=duration_array,
                severity_array=severity_array,
                onset_array=onset_array,
            )
            overall_pbar.update(1)

            # ========================================================
            # 步骤 5：生成观测站点干旱特征图
            # 图件包括四图面板、严重度-持续时间散点图、严重度站点排序图。
            # ========================================================
            create_station_feature_figures(
                lats=latitudes,
                lons=longitudes,
                magnitude_array=magnitude_array,
                duration_array=duration_array,
                severity_array=severity_array,
                onset_array=onset_array,
                station_feature_table=station_feature_table,
            )
            overall_pbar.update(1)

            # ========================================================
            # 步骤 6：保存表格和 NC 结果
            # ========================================================
            with tqdm(
                total=3,
                desc="保存结果文件",
                colour=PROGRESS_COLORS["save"],
                **TQDM_CONFIG,
            ) as pbar_save:
                ds_feature = build_output_dataset(
                    lats=latitudes,
                    lons=longitudes,
                    magnitude_array=magnitude_array,
                    duration_array=duration_array,
                    severity_array=severity_array,
                    onset_array=onset_array,
                )
                ds_feature.to_netcdf(OUTPUT_NC)
                pbar_save.update(1)

                feature_table = build_output_table(
                    lats=latitudes,
                    lons=longitudes,
                    magnitude_array=magnitude_array,
                    duration_array=duration_array,
                    severity_array=severity_array,
                    onset_array=onset_array,
                )
                feature_table.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
                pbar_save.update(1)
                
                station_feature_table.to_csv(OUTPUT_SITE_CSV, index=False, encoding="utf-8-sig")
                pbar_save.update(1)
            overall_pbar.update(1)

            print(f"已保存 NC：{OUTPUT_NC}")
            print(f"已保存 CSV：{OUTPUT_CSV}")
            print(f"已保存站点CSV：{OUTPUT_SITE_CSV}")
            print(f"已保存四图面板：{OUTPUT_PANEL_PNG}")
            print(f"已保存散点图：{OUTPUT_SCATTER_PNG}")
            print(f"已保存排序图：{OUTPUT_RANKING_PNG}")

            if not feature_table.empty:
                print("\nCSV 前 5 行预览：")
                print(feature_table.head())

            if not station_feature_table.empty:
                print("\n站点特征表前 5 行预览：")
                print(station_feature_table.head())

        finally:
            # ========================================================
            # 步骤 7：清理缓存
            # 无论脚本是否中途报错，都尝试清理本次运行缓存目录。
            # ========================================================
            if ds_spi is not None:
                ds_spi.close()
            if ds_feature is not None:
                ds_feature.close()

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
