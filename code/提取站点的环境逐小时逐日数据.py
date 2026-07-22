# -*- coding: utf-8 -*-
"""
提取站点环境因子的逐小时和逐日时间序列数据。

当前脚本提取三类环境因子：
    1. Srad：从 NetCDF 提取站点逐小时数据，并按日累计生成逐日表。
    2. T2m：从逐小时 TIF 提取站点 2 米气温，并按日平均生成逐日表。
    3. Precipitation：从 CHIRPS daily NetCDF 提取站点逐日降雨表。

所有输出表均为宽表：
    第一列：datetime。
    第二列及以后：站点编号，每一列为该站点对应时间点的环境因子数值。
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.warp import transform as transform_coords
from tqdm import tqdm


# =============================================================================
# 一、用户可调参数区
# =============================================================================
@dataclass(frozen=True)
class Config:
    # -------------------------------------------------------------------------
    # 1. 项目路径参数
    # -------------------------------------------------------------------------
    # base_dir:
    #   项目数据主目录。站点经纬度表、Srad、T2m 和 CHIRPS 降雨目录均默认位于该目录下。
    #   如果整个项目数据目录迁移，只需要优先修改这一项。
    base_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate")

    # site_csv:
    #   站点经纬度表路径。表格必须至少包含 Site_ID、Longitude、Latitude 三列。
    #   Site_ID 会作为输出表中第二列及以后各站点列的列名。
    site_csv: Path = base_dir / "Tensor_LatLong.csv"

    # srad_dir:
    #   Srad NetCDF 文件所在目录，也是 Srad 结果表保存目录。
    #   当前预计包含 ssrd_202501_hourly.nc 到 ssrd_202601_hourly.nc。
    srad_dir: Path = base_dir / "Srad"

    # t2m_tif_dir:
    #   T2m 逐小时 TIF 文件所在目录，也是 T2m 结果表保存目录。
    #   当前预计文件名类似：福建省2米气温_2025年01月01日00时.tif。
    t2m_tif_dir: Path = base_dir / "T2m" / "fujian_T2"

    # precip_nc_dir:
    #   CHIRPS 逐日降雨 NetCDF 文件所在目录。
    #   当前预计包含 fujian_1981_pre_CHIRPS_daily.nc 到 fujian_2025_pre_CHIRPS_daily.nc。
    precip_nc_dir: Path = base_dir / "Precipitation_CHIRPS" / "daily" / "fujian_pre_daily"

    # precip_output_dir:
    #   CHIRPS 站点逐日降雨结果表保存目录。按你的要求保存到 Precipitation_CHIRPS 根目录。
    precip_output_dir: Path = base_dir / "Precipitation_CHIRPS"

    # fapar_dir:
    #   FAPAR 8 日尺度 TIF 文件所在目录，也是 FAPAR 站点提取结果表保存目录。
    #   当前规则文件名为 FPAR_YYYYMMDD.tif，例如 FPAR_20250101.tif。
    fapar_dir: Path = base_dir / "FAPAR"

    # lai_dir:
    #   LAI 8 日尺度 TIF 文件所在目录，也是 LAI 站点提取结果表保存目录。
    #   当前规则文件名为 LAI_YYYYMMDD.tif，例如 LAI_20250101.tif。
    lai_dir: Path = base_dir / "LAI"

    # -------------------------------------------------------------------------
    # 2. 输入文件匹配规则
    # -------------------------------------------------------------------------
    # srad_nc_pattern:
    #   Srad NetCDF 文件通配符。当前提取 hourly 文件；如果以后改为 daily 文件，可修改此处。
    srad_nc_pattern: str = "ssrd_*_hourly.nc"

    # t2m_tif_patterns:
    #   T2m TIF 文件通配符。同时兼容 .tif 和 .tiff 后缀。
    t2m_tif_patterns: tuple[str, ...] = ("*.tif", "*.tiff")

    # precip_nc_pattern:
    #   CHIRPS 逐日降雨 NetCDF 文件通配符。
    precip_nc_pattern: str = "fujian_*_pre_CHIRPS_daily.nc"

    # fapar_tif_pattern / lai_tif_pattern:
    #   FAPAR 和 LAI TIF 文件的初筛通配符。后续还会用严格正则只保留 8 位日期文件，
    #   因此 FAPAR_Daily_Jan26.tif、FPAR_202601.tif、LAI_202601.tif 等不规则文件会被排除。
    fapar_tif_pattern: str = "FPAR_*.tif"
    lai_tif_pattern: str = "LAI_*.tif"

    # t2m_filename_time_regex:
    #   从 T2m 中文 TIF 文件名中解析时间的正则表达式。
    #   要求能捕获 year、month、day、hour 四个命名分组。
    t2m_filename_time_regex: str = (
        r"(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日(?P<hour>\d{1,2})时"
    )

    # fapar_filename_date_regex / lai_filename_date_regex:
    #   从 FAPAR 和 LAI 文件名中解析 8 日尺度日期的严格正则。
    #   使用 ^ 和 $ 限定完整文件名主体，确保只处理 FPAR_YYYYMMDD / LAI_YYYYMMDD。
    fapar_filename_date_regex: str = r"^FPAR_(?P<date>\d{8})$"
    lai_filename_date_regex: str = r"^LAI_(?P<date>\d{8})$"

    # -------------------------------------------------------------------------
    # 3. 输出文件名参数
    # -------------------------------------------------------------------------
    # 所有输出文件均使用中文命名，便于直接识别数据内容和时间尺度。
    srad_hourly_csv_name: str = "站点Srad逐小时提取结果.csv"
    srad_daily_csv_name: str = "站点Srad逐日累计提取结果.csv"
    t2m_hourly_csv_name: str = "站点2米气温逐小时提取结果.csv"
    t2m_daily_csv_name: str = "站点2米气温逐日平均提取结果.csv"
    precip_daily_csv_name: str = "站点CHIRPS逐日降雨提取结果.csv"
    fapar_8day_csv_name: str = "站点FAPAR_8日尺度提取结果.csv"
    lai_8day_csv_name: str = "站点LAI_8日尺度提取结果.csv"

    # -------------------------------------------------------------------------
    # 4. 站点表字段和 CSV 编码参数
    # -------------------------------------------------------------------------
    # site_id_col / lon_col / lat_col:
    #   站点编号、经度、纬度列名。如果 Tensor_LatLong.csv 字段名变化，只需修改这里。
    site_id_col: str = "Site_ID"
    lon_col: str = "Longitude"
    lat_col: str = "Latitude"

    # csv_encodings:
    #   读取 CSV 时依次尝试的编码，兼容 UTF-8 BOM、UTF-8 和常见中文 Windows 编码。
    csv_encodings: tuple[str, ...] = ("utf-8-sig", "utf-8", "gbk", "gb18030")

    # output_csv_encoding:
    #   输出 CSV 编码。utf-8-sig 方便 Excel/WPS 直接打开中文表名和表头。
    output_csv_encoding: str = "utf-8-sig"

    # -------------------------------------------------------------------------
    # 5. NetCDF 变量和坐标参数
    # -------------------------------------------------------------------------
    # srad_variable_candidates:
    #   Srad 变量名候选。脚本按顺序查找，优先使用 ssrd。
    srad_variable_candidates: tuple[str, ...] = (
        "ssrd",
        "Srad",
        "srad",
        "surface_solar_radiation_downwards",
    )

    # precip_variable_candidates:
    #   降雨变量名候选。CHIRPS 常用 precip，也可能被预处理成 precipitation/pre。
    precip_variable_candidates: tuple[str, ...] = (
        "precip",
        "precipitation",
        "pre",
        "rain",
        "tp",
    )

    # time_coord_candidates:
    #   NetCDF 时间坐标名候选。ERA5 常见 time/valid_time，CHIRPS 常见 time。
    time_coord_candidates: tuple[str, ...] = ("time", "valid_time", "datetime", "date")

    # lon_coord_candidates / lat_coord_candidates:
    #   NetCDF 经纬度坐标名候选。常见为 longitude/latitude 或 lon/lat。
    lon_coord_candidates: tuple[str, ...] = ("longitude", "lon", "x")
    lat_coord_candidates: tuple[str, ...] = ("latitude", "lat", "y")

    # nc_spatial_method:
    #   NetCDF 站点空间提取方式。nearest 表示提取离站点最近的网格像元。
    #   该方式稳定快速，不依赖额外插值库；如果未来要双线性插值，可在提取函数中扩展。
    nc_spatial_method: str = "nearest"

    # extra_dimension_strategy:
    #   如果 NetCDF 变量除 time/lat/lon 外还有额外维度，默认取第一个索引。
    #   适合 expver、number 等长度为 1 或可取首项的维度。
    extra_dimension_strategy: str = "first"

    # -------------------------------------------------------------------------
    # 6. 时间和数值处理参数
    # -------------------------------------------------------------------------
    # datetime_format:
    #   逐小时表 datetime 列格式。T2m 和 Srad 小时结果会显示到秒。
    datetime_format: str = "%Y-%m-%d %H:%M:%S"

    # daily_datetime_format:
    #   逐日表 datetime 列格式。逐日输出只保留日期，便于后续按日期合并分析。
    daily_datetime_format: str = "%Y-%m-%d"

    # nc_time_offset_hours:
    #   NetCDF 输出时间偏移小时数。默认 0 表示保留源文件时间。
    #   如果确认某类 NetCDF 时间为 UTC 且需要北京时间，可改为 8。
    nc_time_offset_hours: int = 0

    # t2m_time_offset_hours:
    #   T2m TIF 文件名时间偏移小时数。默认 0 表示文件名中的时间就是输出时间。
    t2m_time_offset_hours: int = 0

    # srad_value_scale / srad_value_offset:
    #   Srad 数值线性换算参数。输出值 = 原始值 * scale + offset。
    #   当前按原始值输出，因此保持 1 和 0；如需单位换算可在这里调整。
    srad_value_scale: float = 1.0
    srad_value_offset: float = 0.0

    # precip_value_scale / precip_value_offset:
    #   降雨数值线性换算参数。当前按原始 CHIRPS daily 数值输出。
    precip_value_scale: float = 1.0
    precip_value_offset: float = 0.0

    # t2m_value_scale / t2m_value_offset:
    #   T2m 数值线性换算参数。当前假定 TIF 已是摄氏度，按原始值输出。
    #   如果源 TIF 是 K，可设置 offset=-273.15。
    t2m_value_scale: float = 1.0
    t2m_value_offset: float = 0.0

    # fapar_value_scale / fapar_value_offset:
    #   FAPAR 数值线性换算参数。当前按原始 TIF 像元值输出。
    fapar_value_scale: float = 1.0
    fapar_value_offset: float = 0.0

    # lai_value_scale / lai_value_offset:
    #   LAI 数值线性换算参数。当前按原始 TIF 像元值输出。
    lai_value_scale: float = 1.0
    lai_value_offset: float = 0.0

    # srad_daily_aggregation:
    #   Srad 小时转逐日方法。ssrd 通常是累计辐射量，逐日应求和，因此默认 sum。
    #   如果你的 Srad 是 W m-2 等平均通量，可改为 mean。
    srad_daily_aggregation: str = "sum"

    # t2m_daily_aggregation:
    #   T2m 小时转逐日方法。逐日气温通常使用日平均，因此默认 mean。
    t2m_daily_aggregation: str = "mean"

    # raster_band_index:
    #   TIF 读取波段编号。rasterio 波段从 1 开始计数；单波段气温 TIF 通常为 1。
    raster_band_index: int = 1

    # default_tif_crs:
    #   当 TIF 缺少 CRS 信息时采用的默认坐标系。福建区域裁剪结果通常为 WGS84 经纬度。
    default_tif_crs: str = "EPSG:4326"

    # -------------------------------------------------------------------------
    # 7. 临时缓存和进度条参数
    # -------------------------------------------------------------------------
    # temp_dir_prefix:
    #   本次运行专用临时缓存目录前缀。脚本结束后会删除该目录。
    temp_dir_prefix: str = "提取站点环境数据临时缓存_"

    # tqdm_bar_format:
    #   统一进度条格式，显示百分比、彩色进度条、当前量/总量、耗时、剩余时间和速度。
    tqdm_bar_format: str = (
        "{l_bar}{bar:32}| {percentage:3.0f}% {n_fmt}/{total_fmt} "
        "[已用 {elapsed} < 剩余 {remaining}, {rate_fmt}]"
    )

    # tqdm_dynamic_ncols:
    #   根据终端宽度自动调整进度条长度，尽量保持单行动态刷新。
    tqdm_dynamic_ncols: bool = True

    # tqdm_leave:
    #   False 表示步骤结束后清除该步骤进度条，避免日志刷屏。
    tqdm_leave: bool = False

    # progress_colours:
    #   不同步骤使用不同颜色，便于观察当前处理阶段。
    progress_colours: dict[str, str] | None = None


CONFIG = Config(
    progress_colours={
        "准备": "cyan",
        "站点": "green",
        "文件": "blue",
        "提取": "magenta",
        "合并": "yellow",
        "聚合": "white",
        "写出": "cyan",
        "清理": "red",
    }
)


# =============================================================================
# 二、通用工具函数
# =============================================================================
def make_bar(total: int, desc: str, unit: str, colour_key: str) -> tqdm:
    """创建统一样式的 tqdm 单行动态进度条。"""
    colour = (CONFIG.progress_colours or {}).get(colour_key, "white")
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        colour=colour,
        dynamic_ncols=CONFIG.tqdm_dynamic_ncols,
        leave=CONFIG.tqdm_leave,
        bar_format=CONFIG.tqdm_bar_format,
    )


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    """按多个常见编码读取 CSV，避免中文 Windows CSV 读取失败。"""
    last_error: Exception | None = None
    for encoding in CONFIG.csv_encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"无法读取 CSV：{path}；最后一次编码错误：{last_error}")


def normalise_site_id(value: object) -> str:
    """把站点编号统一成字符串，避免 95332217 被 pandas 读成 95332217.0。"""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def validate_site_table(site_df: pd.DataFrame) -> None:
    """检查站点表必要列，提前暴露列名或表格格式问题。"""
    required_columns = [CONFIG.site_id_col, CONFIG.lon_col, CONFIG.lat_col]
    missing_columns = [column for column in required_columns if column not in site_df.columns]
    if missing_columns:
        raise ValueError(f"站点表缺少必要列：{', '.join(missing_columns)}")


def load_sites() -> pd.DataFrame:
    """读取并清洗站点经纬度表，只保留后续点提取所需字段。"""
    with make_bar(total=1, desc="读取站点经纬度表", unit="file", colour_key="站点") as progress:
        site_df = read_csv_with_fallback(CONFIG.site_csv)
        validate_site_table(site_df)
        site_df = site_df[[CONFIG.site_id_col, CONFIG.lon_col, CONFIG.lat_col]].copy()
        site_df[CONFIG.site_id_col] = site_df[CONFIG.site_id_col].map(normalise_site_id)
        site_df[CONFIG.lon_col] = pd.to_numeric(site_df[CONFIG.lon_col], errors="coerce")
        site_df[CONFIG.lat_col] = pd.to_numeric(site_df[CONFIG.lat_col], errors="coerce")
        site_df = site_df.dropna(subset=[CONFIG.site_id_col, CONFIG.lon_col, CONFIG.lat_col])
        site_df = site_df[site_df[CONFIG.site_id_col] != ""]
        site_df = site_df.drop_duplicates(subset=[CONFIG.site_id_col], keep="first")
        if site_df.empty:
            raise ValueError("站点表中没有可用站点，请检查 Site_ID、Longitude、Latitude。")
        progress.update(1)
    return site_df


def discover_files(directory: Path, patterns: tuple[str, ...] | str, desc: str) -> list[Path]:
    """按通配符查找输入文件，并按文件名排序，保证输出时间顺序稳定。"""
    pattern_list = (patterns,) if isinstance(patterns, str) else patterns
    with make_bar(total=len(pattern_list), desc=desc, unit="pattern", colour_key="文件") as progress:
        files: list[Path] = []
        for pattern in pattern_list:
            files.extend(sorted(directory.glob(pattern)))
            progress.update(1)
    files = sorted({path for path in files if path.is_file()})
    if not files:
        raise FileNotFoundError(f"未找到输入文件：{directory}；匹配规则：{pattern_list}")
    return files


def write_csv(result_df: pd.DataFrame, output_path: Path, desc: str) -> Path:
    """写出宽表 CSV，所有输出均不保存 pandas 行号索引。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with make_bar(total=1, desc=desc, unit="file", colour_key="写出") as progress:
        result_df.to_csv(output_path, index=False, encoding=CONFIG.output_csv_encoding)
        progress.update(1)
    return output_path


def output_exists(output_path: Path) -> bool:
    """判断目标输出表是否已经存在且不是空文件，用于避免重复计算。"""
    return output_path.exists() and output_path.is_file() and output_path.stat().st_size > 0


def skip_existing_output(output_path: Path) -> Path:
    """打印已有表格的跳过信息，并把路径返回给主流程汇总。"""
    print(f"已存在，跳过：{output_path}")
    return output_path


def read_existing_output_csv(output_path: Path) -> pd.DataFrame:
    """读取已经生成的宽表，用于只补算缺失的逐日表。"""
    table = read_csv_with_fallback(output_path)
    if "datetime" not in table.columns:
        raise ValueError(f"已有表缺少 datetime 列，无法复用：{output_path}")
    return table


def prepare_temp_dir() -> Path:
    """创建本次运行专用临时缓存目录，并让常见缓存环境变量指向该目录。"""
    with make_bar(total=1, desc="准备临时缓存目录", unit="dir", colour_key="准备") as progress:
        temp_dir = Path(tempfile.mkdtemp(prefix=CONFIG.temp_dir_prefix))
        os.environ["TMP"] = str(temp_dir)
        os.environ["TEMP"] = str(temp_dir)
        os.environ["TMPDIR"] = str(temp_dir)
        os.environ["CPL_TMPDIR"] = str(temp_dir)
        progress.update(1)
    return temp_dir


def cleanup_temp_dir(temp_dir: Path | None) -> None:
    """删除本次脚本运行产生的临时缓存目录，不触碰原始数据和输出结果。"""
    with make_bar(total=1, desc="清理临时缓存文件", unit="dir", colour_key="清理") as progress:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        progress.update(1)


# =============================================================================
# 三、NetCDF 站点提取函数：用于 Srad 和 CHIRPS 降雨
# =============================================================================
def find_first_existing_name(dataset: xr.Dataset, candidates: tuple[str, ...], label: str) -> str:
    """从候选名称中找到第一个存在于 Dataset 变量或坐标中的名称。"""
    for name in candidates:
        if name in dataset.variables or name in dataset.coords:
            return name
    raise KeyError(f"NetCDF 中找不到{label}，候选名称：{candidates}")


def choose_data_variable(dataset: xr.Dataset, candidates: tuple[str, ...], label: str) -> str:
    """确定 NetCDF 中要提取的数据变量名；如果只有一个数据变量则自动使用它。"""
    for name in candidates:
        if name in dataset.data_vars:
            return name
    data_vars = list(dataset.data_vars)
    if len(data_vars) == 1:
        return data_vars[0]
    raise KeyError(f"无法自动确定{label}变量名。候选：{candidates}；文件内变量：{data_vars}")


def drop_or_select_extra_dimensions(
    data_array: xr.DataArray,
    time_name: str,
    lon_name: str,
    lat_name: str,
) -> xr.DataArray:
    """处理除 time/lat/lon 外的额外维度，保证站点提取面对的是时空三维数据。"""
    core_dims = {time_name, lon_name, lat_name}
    extra_dims = [dim for dim in data_array.dims if dim not in core_dims]
    for dim in extra_dims:
        if data_array.sizes[dim] == 1 or CONFIG.extra_dimension_strategy == "first":
            data_array = data_array.isel({dim: 0}, drop=True)
        else:
            raise ValueError(f"变量存在额外维度 {dim}，且未配置可用处理策略。")
    return data_array


def adjust_station_longitude(lon: float, lon_values: np.ndarray) -> float:
    """根据 NetCDF 经度范围自动兼容 -180~180 和 0~360 两种经度系统。"""
    finite_lon_values = lon_values[np.isfinite(lon_values)]
    if finite_lon_values.size == 0:
        return lon
    lon_min = float(np.nanmin(finite_lon_values))
    lon_max = float(np.nanmax(finite_lon_values))
    if lon_min >= 0 and lon_max > 180 and lon < 0:
        return lon % 360
    if lon_max <= 180 and lon > 180:
        return ((lon + 180) % 360) - 180
    return lon


def format_nc_datetime_index(time_values: np.ndarray) -> pd.Index:
    """把 NetCDF 时间坐标转换为输出表使用的 datetime 字符串索引。"""
    datetime_index = pd.to_datetime(time_values)
    if CONFIG.nc_time_offset_hours:
        datetime_index = datetime_index + pd.to_timedelta(CONFIG.nc_time_offset_hours, unit="h")
    return pd.Index(datetime_index.strftime(CONFIG.datetime_format), name="datetime")


def extract_one_nc_site_series(
    data_array: xr.DataArray,
    site_lon: float,
    site_lat: float,
    time_name: str,
    lon_name: str,
    lat_name: str,
    value_scale: float,
    value_offset: float,
) -> np.ndarray:
    """从一个 NetCDF DataArray 中提取单个站点完整时间序列。"""
    adjusted_lon = adjust_station_longitude(site_lon, np.asarray(data_array[lon_name].values))
    point_data = data_array.sel(
        {lon_name: adjusted_lon, lat_name: site_lat},
        method=CONFIG.nc_spatial_method,
    )
    if time_name not in point_data.dims:
        raise ValueError(f"站点提取结果中缺少时间维度：{time_name}")
    values = point_data.transpose(time_name).values.astype(float)
    return values * value_scale + value_offset


def extract_nc_from_file(
    nc_path: Path,
    site_df: pd.DataFrame,
    variable_candidates: tuple[str, ...],
    variable_label: str,
    value_scale: float,
    value_offset: float,
    progress: tqdm,
) -> pd.DataFrame:
    """从单个 NetCDF 文件中提取所有站点的时间序列，并返回宽表 DataFrame。"""
    with xr.open_dataset(nc_path) as dataset:
        variable_name = choose_data_variable(dataset, variable_candidates, variable_label)
        time_name = find_first_existing_name(dataset, CONFIG.time_coord_candidates, "时间坐标")
        lon_name = find_first_existing_name(dataset, CONFIG.lon_coord_candidates, "经度坐标")
        lat_name = find_first_existing_name(dataset, CONFIG.lat_coord_candidates, "纬度坐标")
        data_array = drop_or_select_extra_dimensions(dataset[variable_name], time_name, lon_name, lat_name)

        required_dims = {time_name, lon_name, lat_name}
        if not required_dims.issubset(set(data_array.dims)):
            raise ValueError(
                f"{nc_path.name} 的变量 {variable_name} 缺少必要维度："
                f"{required_dims - set(data_array.dims)}"
            )

        output = pd.DataFrame(index=format_nc_datetime_index(np.asarray(data_array[time_name].values)))
        for _, site in site_df.iterrows():
            site_id = site[CONFIG.site_id_col]
            output[site_id] = extract_one_nc_site_series(
                data_array=data_array,
                site_lon=float(site[CONFIG.lon_col]),
                site_lat=float(site[CONFIG.lat_col]),
                time_name=time_name,
                lon_name=lon_name,
                lat_name=lat_name,
                value_scale=value_scale,
                value_offset=value_offset,
            )
            progress.update(1)
    return output


def extract_nc_files(
    nc_files: list[Path],
    site_df: pd.DataFrame,
    variable_candidates: tuple[str, ...],
    variable_label: str,
    value_scale: float,
    value_offset: float,
    desc: str,
) -> pd.DataFrame:
    """批量提取 NetCDF 文件；整个提取阶段只显示一个总进度条。"""
    total_tasks = len(nc_files) * len(site_df)
    tables: list[pd.DataFrame] = []
    with make_bar(total=total_tasks, desc=desc, unit="site", colour_key="提取") as progress:
        for nc_path in nc_files:
            tables.append(
                extract_nc_from_file(
                    nc_path=nc_path,
                    site_df=site_df,
                    variable_candidates=variable_candidates,
                    variable_label=variable_label,
                    value_scale=value_scale,
                    value_offset=value_offset,
                    progress=progress,
                )
            )

    with make_bar(total=1, desc=f"合并{variable_label}时间序列表", unit="step", colour_key="合并") as progress:
        merged = pd.concat(tables, axis=0)
        merged = merged[~merged.index.duplicated(keep="first")]
        merged = merged.sort_index()
        merged.index.name = "datetime"
        merged = merged.reset_index()
        progress.update(1)
    return merged


# =============================================================================
# 四、TIF 站点提取函数：用于 T2m 逐小时气温
# =============================================================================
def parse_t2m_time_from_name(path: Path) -> pd.Timestamp:
    """从 T2m TIF 中文文件名中解析逐小时时间。"""
    match = re.search(CONFIG.t2m_filename_time_regex, path.stem)
    if not match:
        raise ValueError(f"无法从 T2m 文件名解析时间：{path.name}")
    timestamp = pd.Timestamp(
        year=int(match.group("year")),
        month=int(match.group("month")),
        day=int(match.group("day")),
        hour=int(match.group("hour")),
    )
    if CONFIG.t2m_time_offset_hours:
        timestamp = timestamp + pd.to_timedelta(CONFIG.t2m_time_offset_hours, unit="h")
    return timestamp


def list_t2m_tif_files(tif_files: list[Path]) -> pd.DataFrame:
    """解析所有 T2m TIF 文件时间，得到按时间排序的文件清单。"""
    records: list[dict[str, object]] = []
    failed_files: list[str] = []
    with make_bar(total=len(tif_files), desc="解析T2m文件时间", unit="file", colour_key="文件") as progress:
        for path in tif_files:
            try:
                records.append({"datetime": parse_t2m_time_from_name(path), "path": path})
            except ValueError:
                failed_files.append(path.name)
            progress.update(1)

    if failed_files:
        examples = "\n".join(failed_files[:10])
        raise ValueError(f"以下 T2m TIF 文件名无法解析时间，请检查正则参数：\n{examples}")
    if not records:
        raise ValueError("没有可用的 T2m TIF 文件。")
    return pd.DataFrame(records).sort_values("datetime").reset_index(drop=True)


def prepare_tif_sample_coordinates(src: rasterio.io.DatasetReader, site_df: pd.DataFrame) -> list[tuple[float, float]]:
    """根据 TIF 坐标系准备采样坐标；必要时把 WGS84 经纬度转换到 TIF 坐标系。"""
    lons = site_df[CONFIG.lon_col].astype(float).to_numpy()
    lats = site_df[CONFIG.lat_col].astype(float).to_numpy()
    src_crs = src.crs if src.crs is not None else CONFIG.default_tif_crs

    if str(src_crs).upper() in {"EPSG:4326", "OGC:CRS84"}:
        xs, ys = lons, lats
    else:
        xs, ys = transform_coords("EPSG:4326", src_crs, lons.tolist(), lats.tolist())
    return list(zip(xs, ys))


def extract_values_from_one_tif(
    path: Path,
    site_df: pd.DataFrame,
    value_scale: float,
    value_offset: float,
) -> list[float]:
    """从单个 TIF 中按站点坐标提取像元值，返回顺序与 site_df 一致的数值列表。"""
    with rasterio.open(path) as src:
        sample_coords = prepare_tif_sample_coordinates(src, site_df)
        nodata = src.nodata
        values: list[float] = []
        for sample in src.sample(sample_coords, indexes=CONFIG.raster_band_index, masked=True):
            value = sample[0]
            if np.ma.is_masked(value):
                values.append(np.nan)
                continue
            value = float(value)
            if nodata is not None and np.isclose(value, nodata):
                values.append(np.nan)
            else:
                values.append(value * value_scale + value_offset)
    return values


def extract_tif_time_series(
    file_table: pd.DataFrame,
    site_df: pd.DataFrame,
    value_scale: float,
    value_offset: float,
    datetime_format: str,
    desc: str,
) -> pd.DataFrame:
    """批量从 TIF 文件提取站点宽表；每个 TIF 对应输出表一行。"""
    site_ids = site_df[CONFIG.site_id_col].tolist()
    records: list[dict[str, object]] = []
    with make_bar(total=len(file_table), desc=desc, unit="file", colour_key="提取") as progress:
        for row in file_table.itertuples(index=False):
            values = extract_values_from_one_tif(
                path=row.path,
                site_df=site_df,
                value_scale=value_scale,
                value_offset=value_offset,
            )
            record = {"datetime": row.datetime.strftime(datetime_format)}
            record.update(dict(zip(site_ids, values)))
            records.append(record)
            progress.update(1)

    if not records:
        raise ValueError(f"{desc}结果为空。")
    return pd.DataFrame(records)


def extract_t2m_hourly(t2m_files: pd.DataFrame, site_df: pd.DataFrame) -> pd.DataFrame:
    """批量从逐小时 TIF 提取 T2m 站点宽表；每个 TIF 对应输出表一行。"""
    return extract_tif_time_series(
        file_table=t2m_files,
        site_df=site_df,
        value_scale=CONFIG.t2m_value_scale,
        value_offset=CONFIG.t2m_value_offset,
        datetime_format=CONFIG.datetime_format,
        desc="提取站点T2m逐小时数据",
    )


# =============================================================================
# 五、8 日尺度 TIF 站点提取函数：用于 FAPAR 和 LAI
# =============================================================================
def parse_8day_date_from_name(path: Path, filename_regex: str, variable_label: str) -> pd.Timestamp:
    """从 FAPAR/LAI 的规则文件名中解析 8 位日期。"""
    match = re.search(filename_regex, path.stem)
    if not match:
        raise ValueError(f"{variable_label} 文件名不符合规则，已排除：{path.name}")
    return pd.to_datetime(match.group("date"), format="%Y%m%d")


def list_8day_tif_files(
    tif_files: list[Path],
    filename_regex: str,
    variable_label: str,
) -> pd.DataFrame:
    """筛选 FAPAR/LAI 规则 TIF 文件，解析日期，并确认主体时间间隔为 8 天。"""
    records: list[dict[str, object]] = []
    skipped_files: list[str] = []
    with make_bar(total=len(tif_files), desc=f"解析{variable_label} 8日文件日期", unit="file", colour_key="文件") as progress:
        for path in tif_files:
            try:
                records.append(
                    {
                        "datetime": parse_8day_date_from_name(path, filename_regex, variable_label),
                        "path": path,
                    }
                )
            except ValueError:
                skipped_files.append(path.name)
            progress.update(1)

    if not records:
        raise ValueError(f"没有可用的 {variable_label} 8日尺度 TIF 文件。")

    file_table = pd.DataFrame(records).sort_values("datetime").reset_index(drop=True)
    date_diffs = sorted(set(file_table["datetime"].diff().dropna().dt.days.astype(int).tolist()))
    if date_diffs != [8]:
        raise ValueError(
            f"{variable_label} 规则文件的日期间隔不是稳定 8 天，实际间隔为：{date_diffs}。"
            "请先检查是否混入非 8 日尺度文件。"
        )

    if skipped_files:
        print(f"{variable_label} 已排除不符合 YYYYMMDD 规则的文件：{', '.join(skipped_files[:10])}")
        if len(skipped_files) > 10:
            print(f"{variable_label} 还有 {len(skipped_files) - 10} 个不规则文件未显示。")

    return file_table


def extract_8day_tif_series(
    file_table: pd.DataFrame,
    site_df: pd.DataFrame,
    value_scale: float,
    value_offset: float,
    variable_label: str,
) -> pd.DataFrame:
    """批量从 8 日尺度 TIF 提取站点宽表；datetime 列为每个合成周期的文件日期。"""
    return extract_tif_time_series(
        file_table=file_table,
        site_df=site_df,
        value_scale=value_scale,
        value_offset=value_offset,
        datetime_format=CONFIG.daily_datetime_format,
        desc=f"提取站点{variable_label} 8日尺度数据",
    )


# =============================================================================
# 六、逐日聚合函数
# =============================================================================
def aggregate_to_daily(table: pd.DataFrame, method: str, desc: str) -> pd.DataFrame:
    """把宽表按自然日聚合为逐日表，method 支持 sum 和 mean。"""
    if method not in {"sum", "mean"}:
        raise ValueError(f"不支持的逐日聚合方法：{method}，只能是 sum 或 mean。")

    with make_bar(total=1, desc=desc, unit="step", colour_key="聚合") as progress:
        work = table.copy()
        work["datetime"] = pd.to_datetime(work["datetime"])
        value_columns = [column for column in work.columns if column != "datetime"]
        work["date"] = work["datetime"].dt.strftime(CONFIG.daily_datetime_format)

        if method == "sum":
            daily = work.groupby("date", as_index=False)[value_columns].sum(min_count=1)
        else:
            daily = work.groupby("date", as_index=False)[value_columns].mean()

        daily = daily.rename(columns={"date": "datetime"})
        progress.update(1)
    return daily


# =============================================================================
# 七、各环境因子主流程
# =============================================================================
def process_srad(site_df: pd.DataFrame) -> list[Path]:
    """提取 Srad 逐小时表，并按日累计生成逐日 Srad 表。"""
    hourly_path = CONFIG.srad_dir / CONFIG.srad_hourly_csv_name
    daily_path = CONFIG.srad_dir / CONFIG.srad_daily_csv_name
    output_paths: list[Path] = []

    if output_exists(hourly_path) and output_exists(daily_path):
        return [skip_existing_output(hourly_path), skip_existing_output(daily_path)]

    if output_exists(hourly_path):
        output_paths.append(skip_existing_output(hourly_path))
        srad_hourly = read_existing_output_csv(hourly_path)
    else:
        srad_files = discover_files(CONFIG.srad_dir, CONFIG.srad_nc_pattern, "查找Srad NetCDF文件")
        srad_hourly = extract_nc_files(
            nc_files=srad_files,
            site_df=site_df,
            variable_candidates=CONFIG.srad_variable_candidates,
            variable_label="Srad",
            value_scale=CONFIG.srad_value_scale,
            value_offset=CONFIG.srad_value_offset,
            desc="提取站点Srad数据",
        )
        output_paths.append(write_csv(srad_hourly, hourly_path, "写出Srad逐小时CSV"))

    if output_exists(daily_path):
        output_paths.append(skip_existing_output(daily_path))
    else:
        srad_daily = aggregate_to_daily(srad_hourly, CONFIG.srad_daily_aggregation, "聚合Srad逐日累计数据")
        output_paths.append(write_csv(srad_daily, daily_path, "写出Srad逐日CSV"))

    return output_paths


def process_t2m(site_df: pd.DataFrame) -> list[Path]:
    """提取 T2m 逐小时表，并按日平均生成逐日 T2m 表。"""
    hourly_path = CONFIG.t2m_tif_dir / CONFIG.t2m_hourly_csv_name
    daily_path = CONFIG.t2m_tif_dir / CONFIG.t2m_daily_csv_name
    output_paths: list[Path] = []

    if output_exists(hourly_path) and output_exists(daily_path):
        return [skip_existing_output(hourly_path), skip_existing_output(daily_path)]

    if output_exists(hourly_path):
        output_paths.append(skip_existing_output(hourly_path))
        t2m_hourly = read_existing_output_csv(hourly_path)
    else:
        tif_files = discover_files(CONFIG.t2m_tif_dir, CONFIG.t2m_tif_patterns, "查找T2m TIF文件")
        t2m_files = list_t2m_tif_files(tif_files)
        t2m_hourly = extract_t2m_hourly(t2m_files, site_df)
        output_paths.append(write_csv(t2m_hourly, hourly_path, "写出T2m逐小时CSV"))

    if output_exists(daily_path):
        output_paths.append(skip_existing_output(daily_path))
    else:
        t2m_daily = aggregate_to_daily(t2m_hourly, CONFIG.t2m_daily_aggregation, "聚合T2m逐日平均数据")
        output_paths.append(write_csv(t2m_daily, daily_path, "写出T2m逐日CSV"))

    return output_paths


def process_precip(site_df: pd.DataFrame) -> list[Path]:
    """提取 CHIRPS 逐日降雨表；源数据已为 daily，因此不再做时间聚合。"""
    daily_path = CONFIG.precip_output_dir / CONFIG.precip_daily_csv_name
    if output_exists(daily_path):
        return [skip_existing_output(daily_path)]

    precip_files = discover_files(CONFIG.precip_nc_dir, CONFIG.precip_nc_pattern, "查找CHIRPS逐日降雨NetCDF文件")
    precip_daily = extract_nc_files(
        nc_files=precip_files,
        site_df=site_df,
        variable_candidates=CONFIG.precip_variable_candidates,
        variable_label="CHIRPS降雨",
        value_scale=CONFIG.precip_value_scale,
        value_offset=CONFIG.precip_value_offset,
        desc="提取站点CHIRPS逐日降雨数据",
    )

    # CHIRPS 源数据是逐日尺度，但这里仍格式化为日期，避免不同 NetCDF 时间编码带来 00:00:00 差异。
    with make_bar(total=1, desc="整理CHIRPS逐日日期格式", unit="step", colour_key="聚合") as progress:
        precip_daily["datetime"] = pd.to_datetime(precip_daily["datetime"]).dt.strftime(CONFIG.daily_datetime_format)
        precip_daily = precip_daily.drop_duplicates(subset=["datetime"], keep="first")
        progress.update(1)

    daily_path = write_csv(
        precip_daily,
        daily_path,
        "写出CHIRPS逐日降雨CSV",
    )
    return [daily_path]


def process_fapar(site_df: pd.DataFrame) -> list[Path]:
    """提取 FAPAR 8 日尺度站点宽表；只处理 FPAR_YYYYMMDD.tif 规则文件。"""
    output_path = CONFIG.fapar_dir / CONFIG.fapar_8day_csv_name
    if output_exists(output_path):
        return [skip_existing_output(output_path)]

    tif_files = discover_files(CONFIG.fapar_dir, CONFIG.fapar_tif_pattern, "查找FAPAR TIF文件")
    fapar_files = list_8day_tif_files(tif_files, CONFIG.fapar_filename_date_regex, "FAPAR")
    fapar_8day = extract_8day_tif_series(
        file_table=fapar_files,
        site_df=site_df,
        value_scale=CONFIG.fapar_value_scale,
        value_offset=CONFIG.fapar_value_offset,
        variable_label="FAPAR",
    )
    output_path = write_csv(
        fapar_8day,
        output_path,
        "写出FAPAR 8日尺度CSV",
    )
    return [output_path]


def process_lai(site_df: pd.DataFrame) -> list[Path]:
    """提取 LAI 8 日尺度站点宽表；只处理 LAI_YYYYMMDD.tif 规则文件。"""
    output_path = CONFIG.lai_dir / CONFIG.lai_8day_csv_name
    if output_exists(output_path):
        return [skip_existing_output(output_path)]

    tif_files = discover_files(CONFIG.lai_dir, CONFIG.lai_tif_pattern, "查找LAI TIF文件")
    lai_files = list_8day_tif_files(tif_files, CONFIG.lai_filename_date_regex, "LAI")
    lai_8day = extract_8day_tif_series(
        file_table=lai_files,
        site_df=site_df,
        value_scale=CONFIG.lai_value_scale,
        value_offset=CONFIG.lai_value_offset,
        variable_label="LAI",
    )
    output_path = write_csv(
        lai_8day,
        output_path,
        "写出LAI 8日尺度CSV",
    )
    return [output_path]


# =============================================================================
# 八、主流程
# =============================================================================
def main() -> None:
    """主流程：准备缓存 -> 读取站点 -> 提取多类因子 -> 写出表格 -> 清理缓存。"""
    temp_dir: Path | None = None
    output_paths: list[Path] = []
    try:
        temp_dir = prepare_temp_dir()
        site_df = load_sites()
        output_paths.extend(process_srad(site_df))
        output_paths.extend(process_t2m(site_df))
        output_paths.extend(process_precip(site_df))
        output_paths.extend(process_fapar(site_df))
        output_paths.extend(process_lai(site_df))
        print("已生成以下文件：")
        for path in output_paths:
            print(path)
    finally:
        cleanup_temp_dir(temp_dir)


if __name__ == "__main__":
    main()
