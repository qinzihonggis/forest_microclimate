"""
福建省 2025 年 SPI_20d 干旱等级标签脚本

脚本功能：
1. 读取 2025 年 SPI_20d NetCDF 结果。
2. 按截图红框中的 SPI 范围生成干旱等级标签。
3. 输出标签 NetCDF、逐日统计 CSV、站点提取 CSV、逐日空间分布图 PNG、逐日占比堆叠面积图 PNG。
4. 每个关键步骤使用 tqdm 彩色进度条显示百分比、当前量/总量、耗时、剩余时间和速度。
5. 运行结束后删除本次脚本创建的临时缓存目录。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


# ============================================================
# 0. 路径参数：集中管理输入、输出和本次运行缓存
# ============================================================
# 输入文件：第一步 SPI 计算得到的 2025 年逐日 SPI_20d 结果。
SPI_FILE = Path(r"E:\forest_microclimate\ForestMicroclimate\results\daily_SPI20_result\Fujian_daily_SPI20d_2025.nc")

# 输出目录：所有图、表、标签 NC 都写入该目录。
OUTPUT_DIR = Path(r"E:\forest_microclimate\ForestMicroclimate\results\daily_SPI20_mark")

# 输出 NC 文件名按要求使用英文小写，其他图表文件使用中文命名。
OUTPUT_NC = OUTPUT_DIR / "fujian_drought_label_2025_daily.nc"
OUTPUT_CSV = OUTPUT_DIR / "福建省2025年SPI_20d干旱等级逐日统计表.csv"
OUTPUT_SITE_CSV = OUTPUT_DIR / "福建省2025年SPI_20d站点干旱等级与SPI值表（daily）.csv"
OUTPUT_LINE_CHART = OUTPUT_DIR / "福建省2025年SPI_20d干旱等级逐日占比堆叠面积图.png"
SITE_CSV = Path(r"E:\forest_microclimate\ForestMicroclimate\Tensor_LatLong.csv")

# 福建省行政边界 shp：
# 1. 地理坐标系边界用于当前空间分布图，因为 SPI 网格坐标是经纬度 lat/lon。
# 2. 投影坐标系边界保留为参数，后续如果要按米制坐标计算真实面积或绘制投影图，可切换使用。
FUJIAN_SHP_GEOGRAPHIC = Path(r"E:\forest_microclimate\ForestMicroclimate\Fujian_Shp\福建省行政边界.shp")
FUJIAN_SHP_PROJECTED = Path(r"E:\forest_microclimate\ForestMicroclimate\Fujian_Shp\福建省行政边界_WGS1984UTM50N.shp")

# 本次脚本运行缓存目录：用于 matplotlib 字体缓存等临时文件，finally 中会删除。
CACHE_DIR = OUTPUT_DIR / "_本次运行缓存"
MATPLOTLIB_CACHE_DIR = CACHE_DIR / "matplotlib"

# 先创建输出目录和缓存目录，再导入 matplotlib，保证 matplotlib 缓存写到本次运行缓存中。
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MATPLOTLIB_CACHE_DIR)


import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


# ============================================================
# 1. 数据参数：变量名、年份、坐标名
# ============================================================
# SPI_VAR：NetCDF 中待标记的 SPI 变量名。当前文件中变量名为 SPI_20d。
SPI_VAR = "SPI_20d"

# YEAR：目标年份。当前输入文件已经是 2025 年，但仍保留该参数，便于后续替换多年文件。
YEAR = "2025"

# 坐标名：当前 NetCDF 的维度是 time, lat, lon。
TIME_DIM = "time"
LAT_NAME = "lat"
LON_NAME = "lon"


# ============================================================
# 2. 干旱等级参数：只标记截图红框范围，湿润部分单独标记为“非干旱范围”
# ============================================================
# 标签编码说明：
# -1 = 缺失值或无效格点
#  0 = 正常范围：      -0.5 <= SPI <= 0.5
#  1 = 轻度干旱：      -1.0 <= SPI < -0.5
#  2 = 中度干旱：      -1.5 <= SPI < -1.0
#  3 = 重度干旱：      -2.0 <= SPI < -1.5
#  4 = 极端干旱：       SPI < -2.0
#  5 = 非干旱范围：     SPI > 0.5，表示湿润侧结果，不纳入干旱等级
#
# 边界处理说明：
# 截图中的区间端点存在相邻等级共用的问题，因此这里采用“左闭右开”的方式处理干旱等级。
# 例如 SPI = -1.0 归为中度干旱的上一级边界？为了避免重复，本脚本明确设为轻度干旱：
# 轻度干旱包含 -1.0，且不包含 -0.5；正常范围包含 -0.5 和 0.5。
LEVELS = {
    -1: {
        "name": "缺失值",
        "meaning": "missing",
        "color": "#D0D0D0",
        "legend_label": "Missing / invalid grid",
    },
    0: {
        "name": "正常",
        "meaning": "normal",
        "color": "#F7F7F7",
        "legend_label": "SPI in [-0.5, 0.5]: Normal",
    },
    1: {
        "name": "轻度干旱",
        "meaning": "mild_drought",
        "color": "#FFD166",
        "legend_label": "SPI in [-1.0, -0.5): Mild drought",
    },
    2: {
        "name": "中度干旱",
        "meaning": "moderate_drought",
        "color": "#F8961E",
        "legend_label": "SPI in [-1.5, -1.0): Moderate drought",
    },
    3: {
        "name": "重度干旱",
        "meaning": "severe_drought",
        "color": "#D62828",
        "legend_label": "SPI in [-2.0, -1.5): Severe drought",
    },
    4: {
        "name": "极端干旱",
        "meaning": "extreme_drought",
        "color": "#7F0000",
        "legend_label": "SPI < -2.0: Extreme drought",
    },
    5: {
        "name": "非干旱范围",
        "meaning": "non_drought_wet_side",
        "color": "#B8D8F8",
        "legend_label": "SPI > 0.5: Non-drought range",
    },
}


# ============================================================
# 3. 进度条参数：不同类型任务使用不同颜色
# ============================================================
# colour：tqdm 进度条颜色，终端支持 ANSI 颜色时会显示彩色条。
# bar_format：显示百分比、当前量/总量、耗时、剩余时间、速度。
TQDM_CONFIG = {
    "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    "dynamic_ncols": True,
    # leave=False：子进度条结束后不保留历史行，避免终端日志刷屏。
    "leave": False,
}

# 总体进度条保留最后一行，方便运行结束时确认整体完成情况。
OVERALL_TQDM_CONFIG = {
    **TQDM_CONFIG,
    "leave": True,
}

PROGRESS_COLORS = {
    "overall": "green",
    "read": "cyan",
    "label": "yellow",
    "save": "blue",
    "map": "magenta",
    "table": "white",
    "line": "red",
}


# ============================================================
# 4. 绘图参数：地图、图例、堆叠面积图参数集中放置，便于后续调整
# ============================================================
MAP_CONFIG = {
    # figsize：单张空间分布图尺寸，单位为英寸。
    "figsize": (8.5, 7.2),
    # dpi：图片分辨率，论文或报告通常可用 300。
    "dpi": 300,
    # use_shp_extent：True 表示使用福建省行政边界范围自动确定地图显示范围。
    "use_shp_extent": True,
    # extent_padding：边界外扩距离，单位为经纬度度数，避免省界贴住图框。
    "extent_padding": 0.15,
    # fallback_extent：当 shp 不可用或不使用 shp 范围时，使用该经纬度范围。
    "fallback_extent": [115.5, 121.0, 23.3, 28.6],
    # boundary_linewidth / boundary_color：福建省行政边界线宽和颜色。
    "boundary_linewidth": 1.1,
    "boundary_color": "#111111",
    # background_color：地图坐标轴背景色，用于覆盖完整 shp 范围内的非网格区域。
    "background_color": "#F4F4F4",
    # gridline_width：经纬网线宽。
    "gridline_width": 0.3,
    # grid_alpha：经纬网透明度。
    "grid_alpha": 0.30,
    # title_fontsize：标题字号。
    "title_fontsize": 13,
    # tick_label_fontsize：经纬度刻度标签字号。
    "tick_label_fontsize": 10,
    # longitude_tick_rotation / latitude_tick_rotation：经纬度刻度文字旋转角度。
    # 纬度刻度设为 90 表示逆时针旋转 90°，如果想水平显示可改为 0。
    "longitude_tick_rotation": 0,
    "latitude_tick_rotation": 90,
}

LEGEND_CONFIG = {
    # loc：图例位置，可改为 upper right、lower right 等。
    "loc": "lower right",
    # fontsize：图例字号。
    "fontsize": 8,
    # frameon：是否显示图例边框和背景。False 表示透明背景且无边框。
    "frameon": False,
    # include_absent_levels：是否显示当月图上没有出现的等级。False 表示只显示实际出现的等级。
    "include_absent_levels": False,
    # include_missing：是否在图例中显示缺失值。False 可去掉灰色背景/无效格点图例。
    "include_missing": False,
}

LINE_CHART_CONFIG = {
    # figsize / dpi：堆叠面积图尺寸和分辨率。
    "figsize": (10.5, 6.2),
    "dpi": 300,
    # alpha：面积填充透明度。
    "alpha": 0.88,
    # grid_alpha：背景网格透明度。
    "grid_alpha": 0.28,
    # legend_loc / legend_bbox_to_anchor：图例位置。
    # 当前设置表示图例放在横轴下方居中位置。
    "legend_loc": "upper center",
    "legend_bbox_to_anchor": (0.5, -0.18),
    # legend_ncol：图例列数。4 表示四个干旱等级一行四列显示。
    "legend_ncol": 4,
    # legend_frameon：False 表示图例透明背景且无边框。
    "legend_frameon": False,
    # tick_rotation：横轴月份标签旋转角度。月份英文简写较短，设为 0 可水平显示。
    "tick_rotation": 0,
    # title_fontsize：标题字号。
    "title_fontsize": 14,
}

SITE_PLOT_CONFIG = {
    # marker：观测点形状，可改为 "o"、"^"、"s"、"D" 等。
    "marker": "^",
    # size：观测点大小。
    "size": 42,
    # facecolor / edgecolor：观测点填充色和边框色。
    "facecolor": "#1F78B4",
    "edgecolor": "#FFFFFF",
    # linewidth：观测点边框线宽。
    "linewidth": 0.8,
    # alpha：观测点透明度。
    "alpha": 0.95,
    # zorder：图层顺序，值越大越靠上。
    "zorder": 6,
    # legend_label：观测点图例文字。
    "legend_label": "Observation sites",
}


def make_progress(iterable, desc: str, colour: str, unit: str, **kwargs):
    """创建统一样式的 tqdm 进度条。"""
    return tqdm(iterable, desc=desc, colour=colour, unit=unit, **TQDM_CONFIG, **kwargs)


def classify_one_month(spi_values: np.ndarray) -> np.ndarray:
    """
    对单个月份的 SPI 二维数组进行等级标记。

    参数：
    spi_values：形状为 (lat, lon) 的 SPI 数组。

    返回：
    labels：形状同 spi_values 的 int16 标签数组。
    """
    labels = np.full(spi_values.shape, -1, dtype=np.int16)
    valid = np.isfinite(spi_values)

    labels[valid & (spi_values > 0.5)] = 5
    labels[valid & (spi_values >= -0.5) & (spi_values <= 0.5)] = 0
    labels[valid & (spi_values >= -1.0) & (spi_values < -0.5)] = 1
    labels[valid & (spi_values >= -1.5) & (spi_values < -1.0)] = 2
    labels[valid & (spi_values >= -2.0) & (spi_values < -1.5)] = 3
    labels[valid & (spi_values < -2.0)] = 4

    return labels


def build_label_dataset(spi_2025: xr.DataArray, labels: np.ndarray) -> xr.Dataset:
    """将标签数组封装为带坐标和属性说明的 xarray Dataset。"""
    return xr.Dataset(
        data_vars={
            "drought_level": (
                [TIME_DIM, LAT_NAME, LON_NAME],
                labels,
                {
                    "long_name": "2025年SPI_20d干旱等级标签",
                    "units": "category",
                    "description": "按SPI阈值标记正常、轻度干旱、中度干旱、重度干旱、极端干旱和湿润侧非干旱范围",
                    "flag_values": np.array(list(LEVELS.keys()), dtype=np.int16),
                    "flag_meanings": " ".join([LEVELS[k]["meaning"] for k in LEVELS.keys()]),
                    "classification_rule": (
                        "-1=缺失值; 0=正常(-0.5<=SPI<=0.5); "
                        "1=轻度干旱(-1.0<=SPI<-0.5); "
                        "2=中度干旱(-1.5<=SPI<-1.0); "
                        "3=重度干旱(-2.0<=SPI<-1.5); "
                        "4=极端干旱(SPI<-2.0); 5=非干旱范围(SPI>0.5)"
                    ),
                },
            )
        },
        coords={
            TIME_DIM: spi_2025[TIME_DIM],
            LAT_NAME: spi_2025[LAT_NAME],
            LON_NAME: spi_2025[LON_NAME],
        },
        attrs={
            "title": "福建省2025年SPI_20d干旱等级标签",
            "source_spi_file": str(SPI_FILE),
            "source_spi_variable": SPI_VAR,
            "year": YEAR,
            "created_by": Path(__file__).name,
        },
    )


def save_label_nc(ds_label: xr.Dataset) -> None:
    """保存干旱等级标签 NetCDF 文件。"""
    encoding = {
        "drought_level": {
            "dtype": "int16",
            "zlib": True,
            "complevel": 4,
        }
    }
    ds_label.to_netcdf(OUTPUT_NC, encoding=encoding)


def build_statistics(labels: np.ndarray, times: xr.DataArray) -> pd.DataFrame:
    """
    生成逐月统计表。

    统计口径：
    1. 有效网格 = 标签不等于 -1 的网格。
    2. 各等级占比 = 该等级网格数 / 当月有效网格数 * 100。
    3. 受旱网格占比 = 轻度 + 中度 + 重度 + 极端干旱四类占比之和。
    """
    rows = []

    for month_index in make_progress(
        range(labels.shape[0]),
        desc="统计逐日等级占比",
        colour=PROGRESS_COLORS["table"],
        unit="天",
    ):
        month_labels = labels[month_index]
        valid_count = int(np.sum(month_labels != -1))
        time_value = pd.to_datetime(times.values[month_index])

        row = {
            "月份": time_value.strftime("%Y年%m月"),
            "有效网格数": valid_count,
        }

        drought_pct_sum = 0.0
        for code in [0, 1, 2, 3, 4, 5, -1]:
            level_name = LEVELS[code]["name"]
            count = int(np.sum(month_labels == code))
            pct = round(count / valid_count * 100, 2) if valid_count > 0 and code != -1 else 0.0
            row[f"{level_name}网格数"] = count
            row[f"{level_name}占比(%)"] = pct

            if code in [1, 2, 3, 4]:
                drought_pct_sum += pct

        row["受旱网格占比(%)"] = round(drought_pct_sum, 2)
        rows.append(row)

    return pd.DataFrame(rows)


def save_statistics_csv(df_stat: pd.DataFrame) -> None:
    """保存逐月统计表，使用 utf-8-sig 编码，便于 Excel 直接打开中文不乱码。"""
    df_stat.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


def read_site_table() -> pd.DataFrame:
    """
    读取站点经纬度表，并检查必需字段是否存在。

    必需字段：
    1. Site_ID：站点编号
    2. Longitude：站点经度
    3. Latitude：站点纬度
    """
    if not SITE_CSV.exists():
        raise FileNotFoundError(f"未找到站点经纬度表：{SITE_CSV}")

    site_df = pd.read_csv(SITE_CSV)
    required_columns = ["Site_ID", "Longitude", "Latitude"]
    missing_columns = [column for column in required_columns if column not in site_df.columns]
    if missing_columns:
        raise KeyError(f"站点表缺少字段：{missing_columns}，实际字段为：{site_df.columns.tolist()}")

    return site_df[required_columns].copy()


def build_site_monthly_table(spi_2025: xr.DataArray, ds_label: xr.Dataset, site_df: pd.DataFrame) -> pd.DataFrame:
    """
    按最近邻网格提取每个站点每个月的 SPI 值和干旱等级。

    提取策略：
    1. 站点经纬度与 SPI 网格存在空间离散化差异，因此采用 method='nearest' 的最近邻提取。
    2. 输出表保留原始站点编号和经纬度，并附加匹配到的网格经纬度。
    3. 每个月输出两列：SPI 值、干旱等级名称；另加一列等级编码，便于后续程序再次读取。
    """
    result_rows = []
    month_times = pd.to_datetime(spi_2025[TIME_DIM].values)

    for _, site in make_progress(
        site_df.iterrows(),
        desc="提取站点SPI与等级",
        colour=PROGRESS_COLORS["table"],
        unit="站",
        total=len(site_df),
    ):
        site_lon = float(site["Longitude"])
        site_lat = float(site["Latitude"])

        spi_at_site = spi_2025.sel({LON_NAME: site_lon, LAT_NAME: site_lat}, method="nearest")
        label_at_site = ds_label["drought_level"].sel({LON_NAME: site_lon, LAT_NAME: site_lat}, method="nearest")

        row = {
            "Site_ID": site["Site_ID"],
            "Longitude": site_lon,
            "Latitude": site_lat,
            "匹配网格经度": round(float(spi_at_site[LON_NAME].values), 5),
            "匹配网格纬度": round(float(spi_at_site[LAT_NAME].values), 5),
        }

        for month_index, month_time in enumerate(month_times):
            month_prefix = month_time.strftime("%Y年%m月")
            spi_value = float(spi_at_site.isel({TIME_DIM: month_index}).values)
            level_code = int(label_at_site.isel({TIME_DIM: month_index}).values)
            level_name = LEVELS.get(level_code, {"name": "未知等级"})["name"]

            row[f"{month_prefix}_SPI_20d"] = round(spi_value, 4) if np.isfinite(spi_value) else np.nan
            row[f"{month_prefix}_干旱等级编码"] = level_code
            row[f"{month_prefix}_干旱等级"] = level_name

        result_rows.append(row)

    return pd.DataFrame(result_rows)


def save_site_monthly_table(site_result_df: pd.DataFrame) -> None:
    """保存站点逐月 SPI 与干旱等级表。"""
    site_result_df.to_csv(OUTPUT_SITE_CSV, index=False, encoding="utf-8-sig")


def build_colormap_and_norm():
    """根据标签编码生成离散色带和分级边界。"""
    ordered_codes = [-1, 0, 1, 2, 3, 4, 5]
    colors = [LEVELS[code]["color"] for code in ordered_codes]
    cmap = mcolors.ListedColormap(colors)
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def format_longitude_tick(value: float, _: object) -> str:
    """将经度刻度格式化为 119°E / 119°W 形式。"""
    hemisphere = "E" if value >= 0 else "W"
    return f"{abs(value):.1f}°{hemisphere}"


def format_latitude_tick(value: float, _: object) -> str:
    """将纬度刻度格式化为 27°N / 27°S 形式。"""
    hemisphere = "N" if value >= 0 else "S"
    return f"{abs(value):.1f}°{hemisphere}"


def build_legend_patches(month_label_values: np.ndarray | None = None):
    """
    生成地图图例。

    month_label_values 不为空时，只显示当月实际出现在图上的等级；
    这样可以避免图例列出 Normal 但图上没有 Normal 像元的误导。
    """
    if LEGEND_CONFIG["include_absent_levels"] or month_label_values is None:
        legend_order = [0, 1, 2, 3, 4, 5]
    else:
        present_codes = set(np.unique(month_label_values[np.isfinite(month_label_values)]).astype(int).tolist())
        legend_order = [code for code in [0, 1, 2, 3, 4, 5] if code in present_codes]

    if LEGEND_CONFIG["include_missing"]:
        legend_order.append(-1)

    legend_handles = [
        mpatches.Patch(color=LEVELS[code]["color"], label=LEVELS[code]["legend_label"])
        for code in legend_order
    ]

    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker=SITE_PLOT_CONFIG["marker"],
            color="none",
            markerfacecolor=SITE_PLOT_CONFIG["facecolor"],
            markeredgecolor=SITE_PLOT_CONFIG["edgecolor"],
            markeredgewidth=SITE_PLOT_CONFIG["linewidth"],
            markersize=np.sqrt(SITE_PLOT_CONFIG["size"]),
            label=SITE_PLOT_CONFIG["legend_label"],
        )
    )
    return legend_handles


def read_fujian_boundary() -> gpd.GeoDataFrame:
    """
    读取福建省行政边界 shp，并统一到 EPSG:4326 经纬度坐标系。

    当前 SPI 网格的坐标是 lat/lon，因此地图叠加优先使用地理坐标系 shp。
    如果后续误传投影坐标系 shp，本函数也会根据 CRS 自动转为 EPSG:4326。
    """
    if not FUJIAN_SHP_GEOGRAPHIC.exists():
        raise FileNotFoundError(f"未找到福建省行政边界文件：{FUJIAN_SHP_GEOGRAPHIC}")

    boundary = gpd.read_file(FUJIAN_SHP_GEOGRAPHIC)
    if boundary.empty:
        raise ValueError(f"福建省行政边界 shp 为空：{FUJIAN_SHP_GEOGRAPHIC}")

    if boundary.crs is None:
        boundary = boundary.set_crs(epsg=4326)
    elif boundary.crs.to_epsg() != 4326:
        boundary = boundary.to_crs(epsg=4326)

    return boundary


def get_map_extent(boundary: gpd.GeoDataFrame) -> list[float]:
    """
    根据福建省行政边界计算地图显示范围。

    返回顺序为 [最小经度, 最大经度, 最小纬度, 最大纬度]，
    可直接传给 matplotlib 的 set_xlim / set_ylim。
    """
    if not MAP_CONFIG["use_shp_extent"]:
        return MAP_CONFIG["fallback_extent"]

    min_lon, min_lat, max_lon, max_lat = boundary.total_bounds
    padding = MAP_CONFIG["extent_padding"]
    return [
        float(min_lon - padding),
        float(max_lon + padding),
        float(min_lat - padding),
        float(max_lat + padding),
    ]


def draw_monthly_maps(ds_label: xr.Dataset, site_df: pd.DataFrame) -> None:
    """逐月绘制 SPI_20d 干旱等级空间分布图，并叠加福建省行政边界和站点位置。"""
    cmap, norm = build_colormap_and_norm()
    fujian_boundary = read_fujian_boundary()
    map_extent = get_map_extent(fujian_boundary)
    longitude_formatter = FuncFormatter(format_longitude_tick)
    latitude_formatter = FuncFormatter(format_latitude_tick)

    for month_index in make_progress(
        range(ds_label.sizes[TIME_DIM]),
        desc="绘制逐日空间分布图",
        colour=PROGRESS_COLORS["map"],
        unit="张",
    ):
        month_label = ds_label["drought_level"].isel({TIME_DIM: month_index})
        month_time = pd.to_datetime(ds_label[TIME_DIM].values[month_index])
        month_text = month_time.strftime("%Y-%m-%d")
        legend_patches = build_legend_patches(month_label.values)

        fig, ax = plt.subplots(figsize=MAP_CONFIG["figsize"])
        ax.set_facecolor(MAP_CONFIG["background_color"])

        ax.pcolormesh(
            ds_label[LON_NAME],
            ds_label[LAT_NAME],
            month_label.values,
            cmap=cmap,
            norm=norm,
            shading="auto",
        )

        fujian_boundary.boundary.plot(
            ax=ax,
            linewidth=MAP_CONFIG["boundary_linewidth"],
            edgecolor=MAP_CONFIG["boundary_color"],
        )

        ax.scatter(
            site_df["Longitude"],
            site_df["Latitude"],
            marker=SITE_PLOT_CONFIG["marker"],
            s=SITE_PLOT_CONFIG["size"],
            c=SITE_PLOT_CONFIG["facecolor"],
            edgecolors=SITE_PLOT_CONFIG["edgecolor"],
            linewidths=SITE_PLOT_CONFIG["linewidth"],
            alpha=SITE_PLOT_CONFIG["alpha"],
            zorder=SITE_PLOT_CONFIG["zorder"],
        )

        ax.set_xlim(map_extent[0], map_extent[1])
        ax.set_ylim(map_extent[2], map_extent[3])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(longitude_formatter)
        ax.yaxis.set_major_formatter(latitude_formatter)
        ax.tick_params(axis="x", labelsize=MAP_CONFIG["tick_label_fontsize"], labelrotation=MAP_CONFIG["longitude_tick_rotation"])
        ax.tick_params(axis="y", labelsize=MAP_CONFIG["tick_label_fontsize"], labelrotation=MAP_CONFIG["latitude_tick_rotation"])
        ax.grid(
            True,
            linewidth=MAP_CONFIG["gridline_width"],
            color="gray",
            alpha=MAP_CONFIG["grid_alpha"],
            linestyle="--",
        )

        ax.legend(
            handles=legend_patches,
            loc=LEGEND_CONFIG["loc"],
            fontsize=LEGEND_CONFIG["fontsize"],
            frameon=LEGEND_CONFIG["frameon"],
        )

        ax.set_title(
            f"Fujian Drought Level ({month_text})",
            fontsize=MAP_CONFIG["title_fontsize"],
            fontweight="bold",
        )

        fig.tight_layout()
        fig_path = OUTPUT_DIR / f"SPI_20d_{month_time.strftime('%Y_%m_%d')}.png"
        fig.savefig(fig_path, dpi=MAP_CONFIG["dpi"], bbox_inches="tight")
        plt.close(fig)


def draw_monthly_line_chart(df_stat: pd.DataFrame) -> None:
    """绘制逐日干旱等级占比堆叠面积图，面积总高度即受旱网格占比。"""
    x_labels = df_stat["月份"].tolist()          # 365个日期字符串
    x_positions = np.arange(len(x_labels), dtype=float)

    drought_series = [
        ("轻度干旱占比(%)", "Mild drought", LEVELS[1]["color"]),
        ("中度干旱占比(%)", "Moderate drought", LEVELS[2]["color"]),
        ("重度干旱占比(%)", "Severe drought", LEVELS[3]["color"]),
        ("极端干旱占比(%)", "Extreme drought", LEVELS[4]["color"]),
    ]

    fig, ax = plt.subplots(figsize=LINE_CHART_CONFIG["figsize"])
    y_values_list = []
    labels = []
    colors = []

    for column_name, label, color in make_progress(
        drought_series,
        desc="准备逐日占比堆叠面积",
        colour=PROGRESS_COLORS["line"],
        unit="层",
    ):
        y_values_list.append(df_stat[column_name].to_numpy(dtype=float))
        labels.append(label)
        colors.append(color)

    ax.stackplot(
        x_positions,
        *y_values_list,
        labels=labels,
        colors=colors,
        alpha=LINE_CHART_CONFIG["alpha"],
        linewidth=0,
    )

    ax.set_title(
        "Daily SPI_20d Drought Level Percentage in Fujian, 2025",
        fontsize=LINE_CHART_CONFIG["title_fontsize"],
        fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    bottom_values = np.sum(np.vstack(y_values_list), axis=0)
    ax.set_ylim(0, max(100, float(np.nanmax(bottom_values)) * 1.08))
    tick_positions = [i for i, label in enumerate(x_labels) if label.endswith("01日")]
    tick_labels_show = [x_labels[i][:7] for i in tick_positions]  # 只取"YYYY年MM月"部分
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels_show, rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=LINE_CHART_CONFIG["grid_alpha"])
    ax.legend(
        loc=LINE_CHART_CONFIG["legend_loc"],
        bbox_to_anchor=LINE_CHART_CONFIG["legend_bbox_to_anchor"],
        ncol=LINE_CHART_CONFIG["legend_ncol"],
        frameon=LINE_CHART_CONFIG["legend_frameon"],
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(OUTPUT_LINE_CHART, dpi=LINE_CHART_CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)


def cleanup_runtime_cache() -> None:
    """删除本次脚本创建的缓存目录，避免输出目录残留临时文件。"""
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR, ignore_errors=True)


def main() -> None:
    """主流程：读取数据、生成标签、保存结果、绘图、统计、清理缓存。"""
    ds_spi = None

    try:
        with tqdm(
            total=7,
            desc="总体步骤进度",
            colour=PROGRESS_COLORS["overall"],
            unit="步",
            **OVERALL_TQDM_CONFIG,
        ) as overall_bar:
            # 步骤 1：读取 SPI 数据，并裁剪目标年份。
            with tqdm(
                total=1,
                desc="读取SPI数据",
                colour=PROGRESS_COLORS["read"],
                unit="项",
                **TQDM_CONFIG,
            ) as read_bar:
                if not SPI_FILE.exists():
                    raise FileNotFoundError(f"未找到输入文件：{SPI_FILE}")

                ds_spi = xr.open_dataset(SPI_FILE)
                if SPI_VAR not in ds_spi.data_vars:
                    raise KeyError(f"输入文件中不存在变量 {SPI_VAR}，实际变量为：{list(ds_spi.data_vars)}")

                spi_2025 = ds_spi[SPI_VAR].sel({TIME_DIM: slice(f"{YEAR}-01-01", f"{YEAR}-12-31")})
                if spi_2025.sizes.get(TIME_DIM, 0) == 0:
                    raise ValueError(f"未从输入文件中筛选到 {YEAR} 年数据")

                read_bar.update(1)
            overall_bar.update(1)

            # 步骤 2：逐月标记 SPI 干旱等级。
            labels = np.empty(spi_2025.shape, dtype=np.int16)
            for month_index in make_progress(
                range(spi_2025.sizes[TIME_DIM]),
                desc="标记逐月SPI等级",
                colour=PROGRESS_COLORS["label"],
                unit="月",
            ):
                labels[month_index] = classify_one_month(spi_2025.isel({TIME_DIM: month_index}).values)

            ds_label = build_label_dataset(spi_2025, labels)
            overall_bar.update(1)

            # 步骤 3：保存 NetCDF 标签文件。
            with tqdm(
                total=1,
                desc="保存标签NC文件",
                colour=PROGRESS_COLORS["save"],
                unit="个",
                **TQDM_CONFIG,
            ) as save_bar:
                save_label_nc(ds_label)
                save_bar.update(1)
            overall_bar.update(1)

            # 步骤 4：读取站点表，并提取每个站点逐月的 SPI 与干旱等级。
            site_df = read_site_table()
            site_result_df = build_site_monthly_table(spi_2025, ds_label, site_df)
            save_site_monthly_table(site_result_df)
            overall_bar.update(1)

            # 步骤 5：生成逐月统计表 CSV。
            df_stat = build_statistics(labels, ds_label[TIME_DIM])
            save_statistics_csv(df_stat)
            overall_bar.update(1)

            # 步骤 6：生成 12 张逐月空间分布图，并叠加站点位置。
            draw_monthly_maps(ds_label, site_df)
            overall_bar.update(1)

            # 步骤 7：生成逐日干旱等级占比堆叠面积图。
            draw_monthly_line_chart(df_stat)
            overall_bar.update(1)

        print("\n运行完成，输出文件如下：")
        print(f"1. 标签NC文件：{OUTPUT_NC}")
        print(f"2. 站点逐月SPI与干旱等级表：{OUTPUT_SITE_CSV}")
        print(f"3. 逐日统计表：{OUTPUT_CSV}")
        print(f"4. 逐日空间分布图：{OUTPUT_DIR}\\SPI_20d_YYYY_MM_DD.png")
        print(f"5. 逐日占比堆叠面积图：{OUTPUT_LINE_CHART}")

    finally:
        if ds_spi is not None:
            ds_spi.close()
        cleanup_runtime_cache()
        print(f"\n已删除本次运行缓存目录：{CACHE_DIR}")


if __name__ == "__main__":
    main()
