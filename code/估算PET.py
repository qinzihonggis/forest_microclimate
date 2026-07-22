# -*- coding: utf-8 -*-
"""
日尺度 FAO-56 Penman-Monteith 近似 PET 计算脚本。

本脚本用于在缺少湿度和风速观测的情况下，利用已有 ERA5 小时尺度气温、
小时尺度太阳辐射和 DEM 海拔数据，构建一套适合 SPEI/干旱指数时间序列
分析的日尺度 PET 数据。

核心方法：
1. 小时 2 m 气温聚合为日均温、日最高温、日最低温。
2. 小时太阳辐射 J/m² 聚合为日累计 MJ/m²/day。
3. 使用 DEM 海拔估算气压和晴空辐射。
4. 使用 FAO-56 日尺度 Penman-Monteith 公式计算 PET。

必要近似：
1. 缺少风速数据时，固定使用 u2 = 2.0 m/s。
2. 缺少湿度/露点数据时，使用 ea = esat(Tmin) 近似实际水汽压。
3. 日尺度土壤热通量 G 设为 0。

输出说明：
所有输出文件均使用中文命名，并保存在：
E:\\forest_microclimate\\ForestMicroclimate\\PET_Estimate_era5
"""

from __future__ import annotations

import csv
import math
import re
import shutil
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date
from tqdm import tqdm


# =============================================================================
# 一、输入路径参数
# =============================================================================
# DEM_PATH：
# - 用途：读取海拔，FAO-56 PM 需要由海拔估算大气压和干湿表常数。
# - 当前值：你提供的福建 DEM 路径。
# - 修改建议：如果后续换研究区，只需要改成新 DEM tif 路径。
DEM_PATH = Path(r"E:\forest_microclimate\ForestMicroclimate\DEM_fujian\fujian_dem.tif")

# FUJIAN_BOUNDARY_SHP：
# - 用途：提供福建省行政边界，用于从 CHIRPS 全球 0.05°坐标中裁出福建范围，
#   并生成省界掩膜。最终 PET 在省界外会被设为 NaN。
# - 修改建议：如果后续换研究区，需要同时更换 DEM、温度、辐射和这个边界文件。
FUJIAN_BOUNDARY_SHP = Path(r"E:\forest_microclimate\ForestMicroclimate\Fujian_Shp\福建省行政边界.shp")

# CHIRPS_TEMPLATE_NC：
# - 用途：只读取 CHIRPS 的 latitude/longitude 坐标，作为 PET 输出目标网格。
# - 注意：脚本不会读取该文件中的降雨三维数据，因此不会额外占用大量内存。
# - 当前值：使用 2025 年 CHIRPS daily 文件作为 0.05°网格模板。
CHIRPS_TEMPLATE_NC = Path(
    r"E:\forest_microclimate\ForestMicroclimate\Precipitation_CHIRPS\daily\chirps-v2.0.2025.days_p05.nc"
)

# SITE_COORDINATE_CSV：
# - 用途：提供样地/站点经纬度，用于从 0.05°网格 PET 中提取各站点逐日 PET。
# - 必需列：
#   Site_ID：站点编号，作为输出 CSV 的列名。
#   Longitude：站点经度，单位十进制度。
#   Latitude：站点纬度，单位十进制度。
SITE_COORDINATE_CSV = Path(r"E:\forest_microclimate\ForestMicroclimate\Tensor_LatLong.csv")

# TARGET_GRID_BUFFER_DEGREES：
# - 用途：在福建边界外接矩形四周扩展少量范围，避免边界附近像元因中心点略在外侧而缺失。
# - 0.05°约等于 CHIRPS 一个像元大小；最终仍会用福建边界掩膜裁掉省界外像元。
TARGET_GRID_BUFFER_DEGREES = 0.05

# T2M_INPUT：
# - 用途：读取小时尺度 2 m 气温 tif。
# - 支持形式：
#   1. 一个包含很多小时 tif 的文件夹；
#   2. 一个多波段 tif，每个波段代表一个小时；
#   3. 一个单波段 tif，但单波段无法可靠聚合 daily PET，通常不建议。
# - 单位处理：脚本会按数值范围自动判断是 °C 还是 K。如果识别为 K，会自动减 273.15。
# - 当前值：使用你已有的福建小时气温 tif 目录。
T2M_INPUT = Path(r"E:\forest_microclimate\ForestMicroclimate\T2m\fujian_T2")

# SRAD_INPUT：
# - 用途：读取太阳辐射 NetCDF。
# - 支持形式：
#   1. 一个 nc 文件；
#   2. 一个包含多个 nc 文件的文件夹，脚本会按文件名排序后依次读取并拼接时间维。
# - 单位要求：变量值应为 J/m²。
# - 重要提醒：如果你的 ssrd 是从当天 00:00 起累计的辐射量，而不是逐小时量，
#   需要先做相邻时次差分，否则日累计会被严重高估。
# - 当前值：使用你提供的 ssrd_202501_hourly.nc 到 ssrd_202601_hourly.nc 所在文件夹。
SRAD_INPUT = Path(r"E:\forest_microclimate\ForestMicroclimate\Srad")

# SRAD_FILE_PATTERN：
# - 用途：当 SRAD_INPUT 是文件夹时，用这个通配符筛选要读取的 nc 文件。
# - 当前命名形如 ssrd_202501_hourly.nc，因此默认使用 ssrd_*_hourly.nc。
# - 如果以后换成逐日文件，例如 ssrd_202501_daily.nc，可改成 ssrd_*_daily.nc。
SRAD_FILE_PATTERN = "ssrd_*_hourly.nc"

# SRAD_VAR_NAME：
# - 用途：指定 NetCDF 中太阳辐射变量名。
# - ERA5/ERA5-Land 常见变量名是 ssrd，即 surface solar radiation downwards。
# - 如果你的 nc 中变量名不同，把这里改成实际变量名。
SRAD_VAR_NAME = "ssrd"

# SRAD_IS_ACCUMULATED：
# - 用途：说明辐射变量是否是随时间累计的 J/m²。
# - ERA5/ERA5-Land 的 ssrd 通常是累计辐射量，不是每小时独立辐射量。
# - "auto" 表示脚本根据相邻时次是否频繁下降做粗略判断；判断为累计时会做逐时差分。
# - 如果你已经确认 nc 中每个时次就是独立小时辐射量，可改为 False。
# - 如果你确认 nc 中是累计量，可改为 True。
# - 你的 ERA5 ssrd 文件已确认是日内累计量，因此固定为 True，避免 auto 对个别月份误判。
SRAD_IS_ACCUMULATED = True


# =============================================================================
# 二、时间、公式和输出参数
# =============================================================================
# DAY_AGGREGATION_UTC_OFFSET_HOURS：
# - 用途：小时数据聚合为“日”时使用的时区偏移。
# - ERA5 小时数据通常是 UTC 时间；福建本地日建议使用 UTC+8，因此默认值为 8。
# - 如果你的 tif 和 nc 已经转换为北京时间或本地日，请改为 0。
DAY_AGGREGATION_UTC_OFFSET_HOURS = 8

# T2M_START_DATETIME：
# - 用途：当多波段 tif 或文件名无法解析时间时，用这里的起始时间生成逐小时时间轴。
# - 示例：datetime(1990, 1, 1, 0)
# - 如果保持 None 且无法解析时间，脚本会退化为每 24 个时间步一组聚合，
#   日期会从 1900-01-01 开始占位；这种情况不适合跨年 SPEI，建议尽量提供真实时间。
T2M_START_DATETIME = None

# ASSUMED_WIND_SPEED_M_S：
# - 用途：FAO-56 PM 公式中的 2 m 风速 u2。
# - 原因：你当前没有风速数据，因此使用 FAO-56 常用默认近似值 2.0 m/s。
# - 影响：真实风速更大时 PET 可能被低估，真实风速更小时 PET 可能被高估。
ASSUMED_WIND_SPEED_M_S = 2.0

# DEFAULT_LATITUDE_DEG：
# - 用途：当 DEM 不是经纬度坐标，无法从栅格仿射变换提取逐像元纬度时使用。
# - 福建大致在 24-28°N，默认 26°N。
# - 如果 DEM 是 EPSG:4326 地理坐标，脚本会自动使用逐像元纬度，不用这个固定值。
DEFAULT_LATITUDE_DEG = 26.0

# OUTPUT_DIR：
# - 用途：保存本次计算生成的 PET 数据、日期文件、统计表和图片。
# - 你要求固定保存在 PET_Estimate_era5 文件夹，因此这里不建议改动。
OUTPUT_DIR = Path(r"E:\forest_microclimate\ForestMicroclimate\PET_Estimate_era5")

# TEMP_CACHE_DIR：
# - 用途：保存脚本运行过程中可能产生的临时缓存。
# - 运行结束后，无论成功还是失败，都会尝试删除这个目录。
# - 只删除该脚本专用缓存目录，不删除输入数据或最终输出结果。
TEMP_CACHE_DIR = OUTPUT_DIR / "临时缓存"

# 中文命名输出文件：
# - NetCDF 文件使用英文命名，避免 netCDF4 底层库在 Windows 中文路径下出现编码问题。
# - TXT 文件保存每个 day 索引对应的日期。
# - CSV 文件保存空间平均时间序列和全时段统计。
# - PNG 文件用于快速检查 PET 的时间变化和多年平均空间格局。
OUTPUT_PET_NC = OUTPUT_DIR / "PET_daily_FAO56_approx.nc"
OUTPUT_DATES_TXT = OUTPUT_DIR / "日尺度PET_日期.txt"
OUTPUT_DAILY_MEAN_CSV = OUTPUT_DIR / "日尺度PET_空间平均时间序列.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "日尺度PET_全时段统计表.csv"
OUTPUT_SITE_DAILY_PET_CSV = OUTPUT_DIR / "站点PET逐日时间序列.csv"
OUTPUT_MEAN_SPATIAL_NC = OUTPUT_DIR / "PET_mean_spatial.nc"
OUTPUT_TIME_SERIES_PNG = OUTPUT_DIR / "日尺度PET_空间平均时间序列图.png"
OUTPUT_MEAN_SPATIAL_PNG = OUTPUT_DIR / "日尺度PET_平均空间分布图.png"

# CHINESE_FONT_PATH：
# - 用途：matplotlib 默认字体通常不支持中文，会导致图片标题出现方框或乱码。
# - 当前优先使用 Windows 自带微软雅黑；如果你的系统没有该字体，可以改成 simhei.ttf 或 simsun.ttc。
CHINESE_FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")


# =============================================================================
# 三、进度条参数
# =============================================================================
# PROGRESS_BAR_FORMAT：
# - 用途：统一控制 tqdm 进度条显示内容。
# - 显示内容包括：百分比、当前量/总量、已用时间、预计剩余时间和速度。
# - dynamic_ncols=True 会自动适配终端宽度，减少换行刷屏。
PROGRESS_BAR_FORMAT = (
    "{l_bar}{bar:32}| {percentage:3.0f}% {n_fmt}/{total_fmt} "
    "[已用 {elapsed} < 剩余 {remaining}, {rate_fmt}]"
)

# 不同步骤使用不同颜色，便于从终端上区分当前阶段。
BAR_COLOR_IO = "cyan"
BAR_COLOR_PREPARE = "yellow"
BAR_COLOR_CALCULATE = "green"
BAR_COLOR_SAVE = "magenta"
BAR_COLOR_CLEAN = "white"


def make_bar(total: int, desc: str, unit: str, colour: str) -> tqdm:
    """创建统一风格的单行彩色 tqdm 进度条。"""
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        colour=colour,
        dynamic_ncols=True,
        leave=False,
        bar_format=PROGRESS_BAR_FORMAT,
    )


# =============================================================================
# 四、目标网格、掩膜和空间重采样函数
# =============================================================================
def read_chirps_target_grid(
    chirps_template_nc: Path,
    boundary_shp: Path,
    buffer_degrees: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 CHIRPS 0.05°坐标，裁出福建范围，并生成福建边界掩膜。"""
    import geopandas as gpd
    from shapely.geometry import Point

    with make_bar(3, "步骤0 构建目标网格", "项", BAR_COLOR_PREPARE) as bar:
        boundary = gpd.read_file(boundary_shp).to_crs("EPSG:4326")
        minx, miny, maxx, maxy = boundary.total_bounds
        minx -= buffer_degrees
        miny -= buffer_degrees
        maxx += buffer_degrees
        maxy += buffer_degrees
        geometry = boundary.unary_union
        bar.update(1)

        with Dataset(chirps_template_nc) as nc:
            lon_name = "longitude" if "longitude" in nc.variables else "lon"
            lat_name = "latitude" if "latitude" in nc.variables else "lat"
            lon_all = np.array(nc.variables[lon_name][:], dtype=float)
            lat_all = np.array(nc.variables[lat_name][:], dtype=float)

        lon_mask = (lon_all >= minx) & (lon_all <= maxx)
        lat_mask = (lat_all >= miny) & (lat_all <= maxy)
        target_lon = lon_all[lon_mask]
        target_lat = lat_all[lat_mask]
        if target_lon.size == 0 or target_lat.size == 0:
            raise ValueError("CHIRPS 模板中没有覆盖福建边界的经纬度坐标")
        bar.update(1)

        target_mask = np.zeros((target_lat.size, target_lon.size), dtype=bool)
        for row_index, lat_value in enumerate(target_lat):
            for col_index, lon_value in enumerate(target_lon):
                point = Point(float(lon_value), float(lat_value))
                target_mask[row_index, col_index] = geometry.contains(point) or geometry.touches(point)
        bar.update(1)

    return target_lon, target_lat, target_mask


def latitude_grid_from_target(target_lat: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    """根据目标网格纬度一维坐标生成逐像元纬度数组。"""
    return np.repeat(target_lat[:, np.newaxis], target_lon.size, axis=1)


def get_tif_grid(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """从 GeoTIFF 仿射变换中提取像元中心经纬度坐标。"""
    from osgeo import gdal

    ds = gdal.Open(str(filepath))
    if ds is None:
        raise FileNotFoundError(f"无法打开 tif: {filepath}")

    gt = ds.GetGeoTransform()
    cols = np.arange(ds.RasterXSize, dtype=float)
    rows = np.arange(ds.RasterYSize, dtype=float)
    lon = gt[0] + (cols + 0.5) * gt[1] + 0.5 * gt[2]
    lat = gt[3] + 0.5 * gt[4] + (rows + 0.5) * gt[5]
    return lon, lat


def find_first_tif(input_path: Path) -> Path:
    """从 tif 文件或 tif 文件夹中找到一个样例文件，用于读取源网格坐标。"""
    if input_path.is_file():
        return input_path

    tif_files = sorted(list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff")))
    if not tif_files:
        raise FileNotFoundError(f"没有找到 tif 文件: {input_path}")
    return tif_files[0]


def ensure_ascending_axis(values: np.ndarray, data: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """确保插值坐标轴递增；如果递减，同步翻转数据。"""
    if values[0] <= values[-1]:
        return values, data
    return values[::-1], np.flip(data, axis=axis)


def interpolate_2d_regular_grid(
    data_2d: np.ndarray,
    source_lon: np.ndarray,
    source_lat: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
) -> np.ndarray:
    """把二维规则经纬度网格双线性插值到目标规则经纬度网格。"""
    source_lon, data_2d = ensure_ascending_axis(source_lon, data_2d, axis=1)
    source_lat, data_2d = ensure_ascending_axis(source_lat, data_2d, axis=0)

    temp = np.empty((data_2d.shape[0], target_lon.size), dtype=float)
    for row_index in range(data_2d.shape[0]):
        row = data_2d[row_index]
        valid = np.isfinite(row)
        if valid.sum() < 2:
            temp[row_index, :] = np.nan
        else:
            temp[row_index, :] = np.interp(target_lon, source_lon[valid], row[valid], left=np.nan, right=np.nan)

    out = np.empty((target_lat.size, target_lon.size), dtype=float)
    for col_index in range(target_lon.size):
        col = temp[:, col_index]
        valid = np.isfinite(col)
        if valid.sum() < 2:
            out[:, col_index] = np.nan
        else:
            out[:, col_index] = np.interp(target_lat, source_lat[valid], col[valid], left=np.nan, right=np.nan)

    return out


def interpolate_3d_regular_grid(
    data_3d: np.ndarray,
    source_lon: np.ndarray,
    source_lat: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    desc: str,
) -> np.ndarray:
    """把三维 time/y/x 数组逐时间片插值到目标规则经纬度网格。"""
    out = np.empty((data_3d.shape[0], target_lat.size, target_lon.size), dtype=float)
    with make_bar(data_3d.shape[0], desc, "日", BAR_COLOR_PREPARE) as bar:
        for time_index in range(data_3d.shape[0]):
            out[time_index] = interpolate_2d_regular_grid(
                data_3d[time_index],
                source_lon,
                source_lat,
                target_lon,
                target_lat,
            )
            bar.update(1)
    return out


def resample_dem_to_target(
    dem_path: Path,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
) -> np.ndarray:
    """用 GDAL 直接把 DEM 重采样到 CHIRPS 目标网格，避免整幅 30 m DEM 进入内存。"""
    from osgeo import gdal

    x_res = float(np.nanmedian(np.diff(target_lon)))
    y_res = abs(float(np.nanmedian(np.diff(target_lat))))
    output_bounds = (
        float(target_lon.min() - x_res / 2.0),
        float(target_lat.min() - y_res / 2.0),
        float(target_lon.max() + x_res / 2.0),
        float(target_lat.max() + y_res / 2.0),
    )

    with make_bar(1, "步骤1 重采样DEM", "项", BAR_COLOR_IO) as bar:
        warped = gdal.Warp(
            "",
            str(dem_path),
            format="MEM",
            dstSRS="EPSG:4326",
            outputBounds=output_bounds,
            width=int(target_lon.size),
            height=int(target_lat.size),
            resampleAlg=gdal.GRA_Bilinear,
            outputType=gdal.GDT_Float32,
            srcNodata=-32768,
            dstNodata=np.nan,
        )
        if warped is None:
            raise RuntimeError("DEM 重采样失败")

        dem = warped.ReadAsArray().astype(float)
        dem[dem < -1000] = np.nan

        # GDAL 输出通常按北到南排列；目标 CHIRPS 纬度是南到北时需要翻转。
        gt = warped.GetGeoTransform()
        output_lat_descending = gt[5] < 0
        target_lat_ascending = target_lat[0] < target_lat[-1]
        if output_lat_descending and target_lat_ascending:
            dem = dem[::-1, :]

        bar.update(1)

    return dem


# =============================================================================
# 五、FAO-56 基础公式函数
# =============================================================================
def esat(temp_c: np.ndarray | float) -> np.ndarray | float:
    """饱和水汽压，单位 kPa。"""
    return 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))


def delta_esat(temp_c: np.ndarray) -> np.ndarray:
    """饱和水汽压曲线斜率，单位 kPa/°C。"""
    return 4098.0 * esat(temp_c) / (temp_c + 237.3) ** 2


def atm_pressure(elev_m: np.ndarray) -> np.ndarray:
    """由海拔估算大气压，单位 kPa。"""
    return 101.3 * ((293.0 - 0.0065 * elev_m) / 293.0) ** 5.26


def psychrometric_constant(pressure_kpa: np.ndarray) -> np.ndarray:
    """干湿表常数，单位 kPa/°C。"""
    return 0.000665 * pressure_kpa


def extraterrestrial_radiation_daily(
    day_of_year: int,
    latitude_deg: np.ndarray | float,
) -> np.ndarray | float:
    """FAO-56 日尺度地外辐射 Ra，单位 MJ/m²/day。"""
    phi = np.deg2rad(latitude_deg)
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * day_of_year / 365.0)
    solar_declination = 0.409 * math.sin(2.0 * math.pi * day_of_year / 365.0 - 1.39)

    sunset_arg = -np.tan(phi) * math.tan(solar_declination)
    sunset_arg = np.clip(sunset_arg, -1.0, 1.0)
    sunset_hour_angle = np.arccos(sunset_arg)

    ra = (
        (24.0 * 60.0 / math.pi)
        * 0.0820
        * dr
        * (
            sunset_hour_angle * np.sin(phi) * math.sin(solar_declination)
            + np.cos(phi) * math.cos(solar_declination) * np.sin(sunset_hour_angle)
        )
    )
    return np.maximum(ra, 0.0)


def clear_sky_radiation(ra_mj_m2_day: np.ndarray, elev_m: np.ndarray) -> np.ndarray:
    """FAO-56 晴空辐射 Rso，单位 MJ/m²/day。"""
    return (0.75 + 2.0e-5 * elev_m) * ra_mj_m2_day


def net_radiation_daily(
    rs_mj_m2_day: np.ndarray,
    tmax_c: np.ndarray,
    tmin_c: np.ndarray,
    ea_kpa: np.ndarray,
    rso_mj_m2_day: np.ndarray,
    albedo: float = 0.23,
) -> np.ndarray:
    """FAO-56 日尺度净辐射 Rn，单位 MJ/m²/day。"""
    rns = (1.0 - albedo) * rs_mj_m2_day

    tmax_k = tmax_c + 273.16
    tmin_k = tmin_c + 273.16
    rs_rso = np.divide(
        rs_mj_m2_day,
        rso_mj_m2_day,
        out=np.zeros_like(rs_mj_m2_day, dtype=float),
        where=rso_mj_m2_day > 0,
    )
    rs_rso = np.clip(rs_rso, 0.0, 1.0)

    sigma = 4.903e-9
    rnl = (
        sigma
        * ((tmax_k**4 + tmin_k**4) / 2.0)
        * (0.34 - 0.14 * np.sqrt(np.maximum(ea_kpa, 0.0)))
        * (1.35 * rs_rso - 0.35)
    )
    return rns - rnl


def compute_daily_pet_fao56_approx(
    tmean_c: np.ndarray,
    tmax_c: np.ndarray,
    tmin_c: np.ndarray,
    rs_mj_m2_day: np.ndarray,
    elev_m: np.ndarray,
    dates: list[datetime],
    latitude_deg: np.ndarray | float,
    wind_speed_m_s: float,
) -> np.ndarray:
    """逐日计算 FAO-56 PM 近似 PET，返回 mm/day，形状为 (day, y, x)。"""
    pressure = atm_pressure(elev_m)
    gamma = psychrometric_constant(pressure)
    pet_days = []

    with make_bar(len(dates), "步骤7 逐日计算PET", "日", BAR_COLOR_CALCULATE) as bar:
        for day_index, date_value in enumerate(dates):
            tmean = tmean_c[day_index]
            tmax = tmax_c[day_index]
            tmin = tmin_c[day_index]
            rs = rs_mj_m2_day[day_index]

            delta = delta_esat(tmean)
            es = (esat(tmax) + esat(tmin)) / 2.0
            ea = esat(tmin)

            day_of_year = int(date_value.strftime("%j"))
            ra = extraterrestrial_radiation_daily(day_of_year, latitude_deg)
            rso = clear_sky_radiation(ra, elev_m)
            rn = net_radiation_daily(rs, tmax, tmin, ea, rso)

            numerator = (
                0.408 * delta * rn
                + gamma
                * (900.0 / (tmean + 273.0))
                * wind_speed_m_s
                * (es - ea)
            )
            denominator = delta + gamma * (1.0 + 0.34 * wind_speed_m_s)

            pet = np.divide(
                numerator,
                denominator,
                out=np.full_like(tmean, np.nan, dtype=float),
                where=denominator != 0,
            )
            pet_days.append(np.maximum(pet, 0.0))
            bar.update(1)

    return np.stack(pet_days, axis=0)


# =============================================================================
# 六、数据读取函数
# =============================================================================
def read_tif_array(filepath: Path) -> np.ndarray:
    """读取单个 tif；单波段返回 (y, x)，多波段返回 (band, y, x)。"""
    from osgeo import gdal

    ds = gdal.Open(str(filepath))
    if ds is None:
        raise FileNotFoundError(f"无法打开 tif: {filepath}")

    arr = ds.ReadAsArray().astype(float)
    nodata_values = []
    for band_index in range(1, ds.RasterCount + 1):
        nodata = ds.GetRasterBand(band_index).GetNoDataValue()
        if nodata is not None:
            nodata_values.append(float(nodata))

    for nodata in set(nodata_values):
        arr[arr == nodata] = np.nan

    return arr


def parse_datetime_from_name(filepath: Path) -> datetime | None:
    """从文件名中识别 YYYYMMDDHH 或 YYYY-MM-DD-HH 等小时信息。"""
    name = filepath.stem

    match = re.search(r"((?:19|20)\d{2})年(\d{1,2})月(\d{1,2})日(\d{1,2})时", name)
    if match:
        year, month, day, hour = map(int, match.groups())
        return datetime(year, month, day, hour)

    match = re.search(r"((?:19|20)\d{2})(\d{2})(\d{2})(\d{2})", name)
    if match:
        year, month, day, hour = map(int, match.groups())
        return datetime(year, month, day, hour)

    match = re.search(
        r"((?:19|20)\d{2})[-_]?(\d{1,2})[-_]?(\d{1,2})[T_\-\s]?(\d{1,2})",
        name,
    )
    if match:
        year, month, day, hour = map(int, match.groups())
        return datetime(year, month, day, hour)

    return None


def detect_temperature_unit(temp_array: np.ndarray) -> tuple[np.ndarray, str]:
    """按数值范围识别气温单位，并统一转换为 °C。"""
    sample_mean = float(np.nanmean(temp_array))
    sample_min = float(np.nanmin(temp_array))
    sample_max = float(np.nanmax(temp_array))

    if sample_mean > 150.0 or sample_max > 100.0:
        return temp_array - 273.15, "K -> °C"

    if sample_min < -90.0 or sample_max > 70.0:
        tqdm.write(
            "警告：温度数值范围不像常规 °C，也不像 K，"
            f"min={sample_min:.2f}, mean={sample_mean:.2f}, max={sample_max:.2f}"
        )

    return temp_array, "°C"


def read_t2m_hourly_series(input_path: Path) -> tuple[np.ndarray, list[datetime] | None, str]:
    """读取小时气温 tif 序列，并返回气温数组、小时日期和单位识别说明。"""
    if input_path.is_dir():
        tif_files = sorted(list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff")))
        if not tif_files:
            raise FileNotFoundError(f"温度文件夹内没有 tif: {input_path}")

        arrays = []
        dates = []
        all_dates_found = True
        with make_bar(len(tif_files), "步骤2 读取小时温度", "文件", BAR_COLOR_IO) as bar:
            for tif_file in tif_files:
                arrays.append(read_tif_array(tif_file))
                parsed_date = parse_datetime_from_name(tif_file)
                if parsed_date is None:
                    all_dates_found = False
                dates.append(parsed_date)
                bar.update(1)

        temp = np.stack(arrays, axis=0)
        temp, unit_note = detect_temperature_unit(temp)
        if all_dates_found:
            return temp, [date for date in dates if date is not None], unit_note
        return temp, None, unit_note

    with make_bar(1, "步骤2 读取小时温度", "文件", BAR_COLOR_IO) as bar:
        temp = read_tif_array(input_path)
        bar.update(1)

    if temp.ndim == 2:
        temp = temp[np.newaxis, :, :]
    elif temp.ndim != 3:
        raise ValueError(f"温度 tif 维度异常: {temp.shape}")

    temp, unit_note = detect_temperature_unit(temp)
    if T2M_START_DATETIME is not None:
        dates = [T2M_START_DATETIME + timedelta(hours=i) for i in range(temp.shape[0])]
        return temp, dates, unit_note

    return temp, None, unit_note


def list_srad_nc_files(input_path: Path, file_pattern: str) -> list[Path]:
    """根据输入路径获取待读取的太阳辐射 nc 文件列表。"""
    if input_path.is_dir():
        nc_files = sorted(input_path.glob(file_pattern))
        if not nc_files:
            raise FileNotFoundError(f"辐射文件夹内没有匹配 {file_pattern} 的 nc 文件: {input_path}")
        return nc_files

    if input_path.is_file():
        return [input_path]

    raise FileNotFoundError(f"太阳辐射输入路径不存在: {input_path}")


def decode_nc_time(nc: Dataset) -> list[datetime] | None:
    """从 NetCDF 的 time 或 valid_time 变量中解析时间轴。"""
    for time_name in ("time", "valid_time"):
        if time_name not in nc.variables:
            continue

        time_var = nc.variables[time_name]
        if not hasattr(time_var, "units"):
            return None

        decoded = num2date(
            time_var[:],
            units=time_var.units,
            calendar=getattr(time_var, "calendar", "standard"),
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
        return [
            datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
            for d in decoded
        ]

    return None


def infer_srad_time_resolution(dates: list[datetime] | None, n_steps: int) -> str:
    """判断辐射数据时间分辨率：hourly、daily 或 unknown。"""
    if dates is None or len(dates) < 2:
        if n_steps >= 24 and n_steps % 24 == 0:
            return "hourly_without_time"
        return "unknown"

    deltas_hours = [
        (dates[index + 1] - dates[index]).total_seconds() / 3600.0
        for index in range(len(dates) - 1)
    ]
    median_delta = float(np.nanmedian(deltas_hours))

    if 0.5 <= median_delta <= 1.5:
        return "hourly"
    if 20.0 <= median_delta <= 28.0:
        return "daily"

    return f"unknown_delta_{median_delta:.2f}h"


def should_treat_srad_as_accumulated(srad: np.ndarray, setting: bool | str) -> bool:
    """判断太阳辐射是否应按累计量做相邻时次差分。"""
    if isinstance(setting, bool):
        return setting

    if str(setting).lower() != "auto":
        raise ValueError("SRAD_IS_ACCUMULATED 只能是 True、False 或 'auto'")

    if srad.shape[0] < 3:
        return False

    sample = np.nanmean(srad.reshape((srad.shape[0], -1)), axis=1)
    diffs = np.diff(sample)
    finite_diffs = diffs[np.isfinite(diffs)]
    if finite_diffs.size == 0:
        return False

    negative_ratio = float(np.mean(finite_diffs < 0))
    first_mean = float(sample[0]) if np.isfinite(sample[0]) else 0.0
    overall_mean = float(np.nanmean(sample))

    # 累计量通常绝大多数相邻差分为非负，且首时次可能已是较大的累计值。
    return negative_ratio < 0.10 and overall_mean > 1.0e5 and first_mean > 1.0e4


def convert_accumulated_srad_to_step_values(
    srad: np.ndarray,
    dates: list[datetime] | None,
) -> np.ndarray:
    """把单个文件内的日内累计辐射量转换为逐时间步辐射量。

    ERA5/ERA5-Land 的 ssrd 常表现为日内累计：01 时为 00-01 的累计，
    02 时为 00-02 的累计，依次增加；00 时常是上一日累计值。若直接把 00 时
    当作本小时辐射，会把每日辐射重复计入，导致 PET 异常偏大。
    """
    step = np.empty_like(srad, dtype=float)

    for time_index in range(srad.shape[0]):
        current_hour = dates[time_index].hour if dates is not None else None
        previous_hour = dates[time_index - 1].hour if dates is not None and time_index > 0 else None

        if time_index == 0:
            step[time_index] = 0.0 if current_hour == 0 else srad[time_index]
            continue

        if current_hour == 0:
            step[time_index] = 0.0
        elif previous_hour == 0:
            step[time_index] = srad[time_index]
        else:
            step[time_index] = np.maximum(srad[time_index] - srad[time_index - 1], 0.0)

    step[step < 0] = 0.0
    return step


def read_srad_nc_series(
    input_path: Path,
    file_pattern: str,
    var_name: str,
) -> tuple[np.ndarray, list[datetime] | None, str, np.ndarray, np.ndarray]:
    """读取单个或多个太阳辐射 NetCDF，并判断其时间分辨率和空间坐标。"""
    nc_files = list_srad_nc_files(input_path, file_pattern)
    arrays = []
    all_dates: list[datetime] = []
    all_dates_found = True
    source_lon = None
    source_lat = None

    with make_bar(len(nc_files), "步骤4 读取太阳辐射", "文件", BAR_COLOR_IO) as bar:
        for nc_file in nc_files:
            with Dataset(nc_file) as nc:
                if var_name not in nc.variables:
                    raise KeyError(
                        f"{nc_file.name} 中未找到变量 {var_name}，"
                        f"可用变量: {list(nc.variables)}"
                    )

                srad_raw = nc.variables[var_name][:]
                srad = np.ma.filled(srad_raw, np.nan).astype(float).squeeze()
                if srad.ndim == 2:
                    srad = srad[np.newaxis, :, :]
                elif srad.ndim != 3:
                    raise ValueError(f"{nc_file.name} 的 Srad 维度异常: {srad.shape}")

                dates = decode_nc_time(nc)
                if dates is None:
                    all_dates_found = False
                elif len(dates) != srad.shape[0]:
                    raise ValueError(
                        f"{nc_file.name} 时间长度 {len(dates)} 与数据步数 {srad.shape[0]} 不一致"
                    )
                else:
                    all_dates.extend(dates)

                if should_treat_srad_as_accumulated(srad, SRAD_IS_ACCUMULATED):
                    srad = convert_accumulated_srad_to_step_values(srad, dates)

                lon_name = "longitude" if "longitude" in nc.variables else "lon"
                lat_name = "latitude" if "latitude" in nc.variables else "lat"
                current_lon = np.array(nc.variables[lon_name][:], dtype=float)
                current_lat = np.array(nc.variables[lat_name][:], dtype=float)
                if source_lon is None or source_lat is None:
                    source_lon = current_lon
                    source_lat = current_lat
                elif not (np.array_equal(source_lon, current_lon) and np.array_equal(source_lat, current_lat)):
                    raise ValueError(f"{nc_file.name} 的经纬度网格与前序辐射文件不一致")

                arrays.append(srad)
            bar.update(1)

    srad_all = np.concatenate(arrays, axis=0)
    dates_out = all_dates if all_dates_found else None
    resolution = infer_srad_time_resolution(dates_out, srad_all.shape[0])
    if source_lon is None or source_lat is None:
        raise ValueError("未能从太阳辐射 NetCDF 中读取经纬度坐标")
    return srad_all, dates_out, resolution, source_lon, source_lat


def make_hourly_dates(
    n_steps: int,
    parsed_dates: list[datetime] | None,
    start_datetime: datetime | None,
) -> list[datetime] | None:
    """返回已有小时日期，或用起始时间生成逐小时日期。"""
    if parsed_dates is not None:
        if len(parsed_dates) != n_steps:
            raise ValueError(f"时间轴长度 {len(parsed_dates)} 与数据步数 {n_steps} 不一致")
        return parsed_dates

    if start_datetime is not None:
        return [start_datetime + timedelta(hours=i) for i in range(n_steps)]

    return None


# =============================================================================
# 六、日尺度聚合和输入检查
# =============================================================================
def aggregate_temperature_daily(
    temp_hourly_c: np.ndarray,
    dates: list[datetime] | None,
    utc_offset_hours: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[datetime]]:
    """把小时气温聚合为日均温、日最高温和日最低温。"""
    if dates is None:
        usable_steps = (temp_hourly_c.shape[0] // 24) * 24
        if usable_steps == 0:
            raise ValueError("没有足够的 24 个小时温度数据用于日尺度聚合")

        n_days = usable_steps // 24
        tmean_days, tmax_days, tmin_days = [], [], []
        with make_bar(n_days, "步骤3 聚合日温度", "日", BAR_COLOR_PREPARE) as bar:
            for day_index in range(n_days):
                start = day_index * 24
                end = start + 24
                day_values = temp_hourly_c[start:end]
                tmean_days.append(np.nanmean(day_values, axis=0))
                tmax_days.append(np.nanmax(day_values, axis=0))
                tmin_days.append(np.nanmin(day_values, axis=0))
                bar.update(1)

        out_dates = [datetime(1900, 1, 1) + timedelta(days=i) for i in range(n_days)]
        return np.stack(tmean_days), np.stack(tmax_days), np.stack(tmin_days), out_dates

    groups: dict[datetime, list[int]] = {}
    for index, date_value in enumerate(dates):
        local_date = (date_value + timedelta(hours=utc_offset_hours)).date()
        group_key = datetime(local_date.year, local_date.month, local_date.day)
        groups.setdefault(group_key, []).append(index)

    out_dates = sorted(groups)
    tmean_days, tmax_days, tmin_days = [], [], []
    with make_bar(len(out_dates), "步骤3 聚合日温度", "日", BAR_COLOR_PREPARE) as bar:
        for day in out_dates:
            day_values = temp_hourly_c[groups[day]]
            tmean_days.append(np.nanmean(day_values, axis=0))
            tmax_days.append(np.nanmax(day_values, axis=0))
            tmin_days.append(np.nanmin(day_values, axis=0))
            bar.update(1)

    return np.stack(tmean_days), np.stack(tmax_days), np.stack(tmin_days), out_dates


def aggregate_srad_daily(
    srad_j_m2: np.ndarray,
    dates: list[datetime] | None,
    utc_offset_hours: int,
    time_resolution: str,
) -> tuple[np.ndarray, list[datetime]]:
    """把太阳辐射 J/m² 整理为日尺度 MJ/m²/day。"""
    if time_resolution == "daily":
        if dates is None:
            out_dates = [datetime(1900, 1, 1) + timedelta(days=i) for i in range(srad_j_m2.shape[0])]
            day_groups = {out_dates[i]: [i] for i in range(srad_j_m2.shape[0])}
        else:
            day_groups: dict[datetime, list[int]] = {}
            for index, date_value in enumerate(dates):
                local_date = (date_value + timedelta(hours=utc_offset_hours)).date()
                group_key = datetime(local_date.year, local_date.month, local_date.day)
                day_groups.setdefault(group_key, []).append(index)
            out_dates = sorted(day_groups)

        rs_days = []
        with make_bar(len(out_dates), "步骤5 整理日辐射", "日", BAR_COLOR_PREPARE) as bar:
            for day in out_dates:
                day_values = srad_j_m2[day_groups[day]]
                rs_days.append(np.nanmean(day_values, axis=0) / 1.0e6)
                bar.update(1)

        return np.stack(rs_days), out_dates

    if not time_resolution.startswith("hourly"):
        raise ValueError(
            "无法判断太阳辐射 nc 是逐小时还是逐日，"
            f"识别结果为 {time_resolution}。请检查 NetCDF 时间轴。"
        )

    if dates is None:
        usable_steps = (srad_j_m2.shape[0] // 24) * 24
        if usable_steps == 0:
            raise ValueError("没有足够的 24 个小时辐射数据用于日尺度聚合")

        n_days = usable_steps // 24
        rs_days = []
        with make_bar(n_days, "步骤5 聚合日辐射", "日", BAR_COLOR_PREPARE) as bar:
            for day_index in range(n_days):
                start = day_index * 24
                end = start + 24
                rs_days.append(np.nansum(srad_j_m2[start:end], axis=0) / 1.0e6)
                bar.update(1)

        out_dates = [datetime(1900, 1, 1) + timedelta(days=i) for i in range(n_days)]
        return np.stack(rs_days), out_dates

    groups: dict[datetime, list[int]] = {}
    for index, date_value in enumerate(dates):
        local_date = (date_value + timedelta(hours=utc_offset_hours)).date()
        group_key = datetime(local_date.year, local_date.month, local_date.day)
        groups.setdefault(group_key, []).append(index)

    out_dates = sorted(groups)
    rs_days = []
    with make_bar(len(out_dates), "步骤5 聚合日辐射", "日", BAR_COLOR_PREPARE) as bar:
        for day in out_dates:
            rs_days.append(np.nansum(srad_j_m2[groups[day]], axis=0) / 1.0e6)
            bar.update(1)

    return np.stack(rs_days), out_dates


def align_daily_inputs(
    tmean: np.ndarray,
    tmax: np.ndarray,
    tmin: np.ndarray,
    temp_dates: list[datetime],
    rs: np.ndarray,
    srad_dates: list[datetime],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[datetime]]:
    """按共同日期对齐日温度和日辐射，避免不同数据源日期错位。"""
    with make_bar(1, "步骤6 对齐日期", "项", BAR_COLOR_PREPARE) as bar:
        temp_map = {date.date(): index for index, date in enumerate(temp_dates)}
        srad_map = {date.date(): index for index, date in enumerate(srad_dates)}
        common_dates = sorted(set(temp_map).intersection(srad_map))

        if not common_dates:
            raise ValueError("温度和辐射没有共同日期，无法计算 daily PET")

        temp_indices = [temp_map[date] for date in common_dates]
        srad_indices = [srad_map[date] for date in common_dates]
        out_dates = [datetime(date.year, date.month, date.day) for date in common_dates]
        bar.update(1)

    return tmean[temp_indices], tmax[temp_indices], tmin[temp_indices], rs[srad_indices], out_dates


def validate_shapes(elev: np.ndarray, tmean: np.ndarray, rs: np.ndarray) -> None:
    """检查 DEM、温度和辐射的空间网格是否一致。"""
    if tmean.shape[1:] != elev.shape:
        raise ValueError(f"温度空间形状 {tmean.shape[1:]} 与 DEM {elev.shape} 不一致")
    if rs.shape[1:] != elev.shape:
        raise ValueError(f"辐射空间形状 {rs.shape[1:]} 与 DEM {elev.shape} 不一致")


def apply_boundary_mask(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """把福建省界外像元设置为 NaN，保留省界内 PET 结果。"""
    if data.ndim == 2:
        return np.where(mask, data, np.nan)
    if data.ndim == 3:
        return np.where(mask[np.newaxis, :, :], data, np.nan)
    raise ValueError(f"不支持的掩膜数据维度: {data.shape}")


def write_daily_pet_netcdf(
    output_path: Path,
    pet: np.ndarray,
    dates: list[datetime],
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    srad_resolution: str,
) -> None:
    """把三维日尺度 PET 写为带完整坐标和属性的 NetCDF 文件。"""
    with Dataset(output_path, "w", format="NETCDF4") as nc:
        nc.createDimension("time", len(dates))
        nc.createDimension("latitude", target_lat.size)
        nc.createDimension("longitude", target_lon.size)

        time_var = nc.createVariable("time", "f8", ("time",))
        lat_var = nc.createVariable("latitude", "f4", ("latitude",))
        lon_var = nc.createVariable("longitude", "f4", ("longitude",))
        pet_var = nc.createVariable(
            "PET",
            "f4",
            ("time", "latitude", "longitude"),
            zlib=True,
            complevel=4,
            fill_value=np.float32(np.nan),
        )

        base_date = datetime(1970, 1, 1)
        time_var[:] = np.array([(date - base_date).days for date in dates], dtype=float)
        lat_var[:] = target_lat.astype(np.float32)
        lon_var[:] = target_lon.astype(np.float32)
        pet_var[:] = pet.astype(np.float32)

        time_var.units = "days since 1970-01-01 00:00:00"
        time_var.calendar = "standard"
        time_var.long_name = "date"
        lat_var.units = "degrees_north"
        lat_var.long_name = "latitude"
        lon_var.units = "degrees_east"
        lon_var.long_name = "longitude"
        pet_var.units = "mm/day"
        pet_var.long_name = "Daily potential evapotranspiration"
        pet_var.method = "FAO-56 Penman-Monteith approximation"

        nc.title = "福建省日尺度PET_FAO56近似"
        nc.Conventions = "CF-1.8"
        nc.spatial_reference = "EPSG:4326"
        nc.grid_source = "CHIRPS daily 0.05 degree grid clipped by Fujian boundary"
        nc.wind_speed_assumption = f"u2 = {ASSUMED_WIND_SPEED_M_S} m/s"
        nc.actual_vapor_pressure_assumption = "ea = esat(Tmin)"
        nc.soil_heat_flux_assumption = "G = 0 for daily timestep"
        nc.srad_time_resolution = srad_resolution
        nc.srad_accumulated_handling = str(SRAD_IS_ACCUMULATED)
        nc.created_by = Path(__file__).name


def write_mean_pet_netcdf(
    output_path: Path,
    mean_spatial: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
) -> None:
    """把二维平均 PET 空间分布写为带坐标的 NetCDF 文件。"""
    with Dataset(output_path, "w", format="NETCDF4") as nc:
        nc.createDimension("latitude", target_lat.size)
        nc.createDimension("longitude", target_lon.size)

        lat_var = nc.createVariable("latitude", "f4", ("latitude",))
        lon_var = nc.createVariable("longitude", "f4", ("longitude",))
        mean_var = nc.createVariable(
            "PET_mean",
            "f4",
            ("latitude", "longitude"),
            zlib=True,
            complevel=4,
            fill_value=np.float32(np.nan),
        )

        lat_var[:] = target_lat.astype(np.float32)
        lon_var[:] = target_lon.astype(np.float32)
        mean_var[:] = mean_spatial.astype(np.float32)

        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"
        mean_var.units = "mm/day"
        mean_var.long_name = "Mean daily potential evapotranspiration"

        nc.title = "福建省日尺度PET平均空间分布"
        nc.Conventions = "CF-1.8"
        nc.spatial_reference = "EPSG:4326"
        nc.grid_source = "CHIRPS daily 0.05 degree grid clipped by Fujian boundary"


def read_site_coordinates(site_csv: Path) -> list[dict[str, float | str]]:
    """读取站点坐标表，返回站点编号、经度和纬度。"""
    sites = []
    with site_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {"Site_ID", "Longitude", "Latitude"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"站点坐标表缺少必要列: {sorted(missing_columns)}")

        for row in reader:
            site_id = str(row["Site_ID"]).strip()
            if not site_id:
                continue
            sites.append(
                {
                    "site_id": site_id,
                    "longitude": float(row["Longitude"]),
                    "latitude": float(row["Latitude"]),
                }
            )

    if not sites:
        raise ValueError(f"站点坐标表没有有效站点: {site_csv}")
    return sites


def nearest_index(values: np.ndarray, target_value: float) -> int:
    """返回一维坐标中距离目标值最近的索引。"""
    return int(np.nanargmin(np.abs(values - target_value)))


def save_site_daily_pet_csv(
    pet: np.ndarray,
    dates: list[datetime],
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    site_csv: Path,
    output_csv: Path,
) -> None:
    """按站点最近邻网格提取逐日 PET，并保存为宽表 CSV。"""
    sites = read_site_coordinates(site_csv)
    site_columns = []
    site_values = []

    for site in sites:
        site_id = str(site["site_id"])
        lon = float(site["longitude"])
        lat = float(site["latitude"])

        lon_index = nearest_index(target_lon, lon)
        lat_index = nearest_index(target_lat, lat)
        site_columns.append(site_id)
        site_values.append(pet[:, lat_index, lon_index])

    site_matrix = np.stack(site_values, axis=1)

    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["datetime"] + site_columns)
        for day_index, date_value in enumerate(dates):
            row = [date_value.strftime("%Y-%m-%d")]
            for value in site_matrix[day_index]:
                row.append("" if not np.isfinite(value) else round(float(value), 6))
            writer.writerow(row)


# =============================================================================
# 七、结果保存、绘图和缓存清理
# =============================================================================
def save_pet_outputs(
    pet: np.ndarray,
    dates: list[datetime],
    srad_resolution: str,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
) -> None:
    """保存 PET 数组、日期文件、统计表和快速检查图。"""
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    chinese_font = FontProperties(fname=str(CHINESE_FONT_PATH)) if CHINESE_FONT_PATH.exists() else None
    if chinese_font is not None:
        plt.rcParams["font.sans-serif"] = [chinese_font.get_name()]
    plt.rcParams["axes.unicode_minus"] = False

    with make_bar(8, "步骤8 保存结果", "项", BAR_COLOR_SAVE) as bar:
        write_daily_pet_netcdf(
            OUTPUT_PET_NC,
            pet,
            dates,
            target_lon,
            target_lat,
            srad_resolution,
        )
        bar.update(1)

        OUTPUT_DATES_TXT.write_text(
            "\n".join(date.strftime("%Y-%m-%d") for date in dates) + "\n",
            encoding="utf-8",
        )
        bar.update(1)

        daily_mean = np.nanmean(pet, axis=(1, 2))
        daily_min = np.nanmin(pet, axis=(1, 2))
        daily_max = np.nanmax(pet, axis=(1, 2))
        with OUTPUT_DAILY_MEAN_CSV.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["日期", "PET空间平均_mm_day", "PET空间最小_mm_day", "PET空间最大_mm_day"])
            for date_value, mean_value, min_value, max_value in zip(dates, daily_mean, daily_min, daily_max):
                writer.writerow([
                    date_value.strftime("%Y-%m-%d"),
                    round(float(mean_value), 6),
                    round(float(min_value), 6),
                    round(float(max_value), 6),
                ])
        bar.update(1)

        summary_rows = [
            ("日期数量", len(dates)),
            ("PET全时段平均_mm_day", float(np.nanmean(pet))),
            ("PET全时段最小_mm_day", float(np.nanmin(pet))),
            ("PET全时段最大_mm_day", float(np.nanmax(pet))),
            ("PET全时段标准差_mm_day", float(np.nanstd(pet))),
            ("风速假设_m_s", ASSUMED_WIND_SPEED_M_S),
            ("实际水汽压假设", "ea = esat(Tmin)"),
            ("日尺度土壤热通量假设", "G = 0"),
            ("日聚合时区偏移_小时", DAY_AGGREGATION_UTC_OFFSET_HOURS),
            ("太阳辐射时间分辨率识别", srad_resolution),
            ("目标网格来源", "CHIRPS daily 0.05°"),
            ("目标网格经度数量", target_lon.size),
            ("目标网格纬度数量", target_lat.size),
            ("目标网格经度范围", f"{float(target_lon.min()):.4f} 到 {float(target_lon.max()):.4f}"),
            ("目标网格纬度范围", f"{float(target_lat.min()):.4f} 到 {float(target_lat.max()):.4f}"),
            ("站点PET提取方法", "最近邻网格"),
            ("站点坐标表", str(SITE_COORDINATE_CSV)),
        ]
        with OUTPUT_SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["指标", "值"])
            for key, value in summary_rows:
                writer.writerow([key, value])
        bar.update(1)

        save_site_daily_pet_csv(
            pet,
            dates,
            target_lon,
            target_lat,
            SITE_COORDINATE_CSV,
            OUTPUT_SITE_DAILY_PET_CSV,
        )
        bar.update(1)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            mean_spatial = np.nanmean(pet, axis=0)
        write_mean_pet_netcdf(
            OUTPUT_MEAN_SPATIAL_NC,
            mean_spatial,
            target_lon,
            target_lat,
        )
        bar.update(1)

        plt.figure(figsize=(12, 4))
        plt.plot(dates, daily_mean, linewidth=1.0, color="#1f77b4")
        plt.title("日尺度PET空间平均时间序列", fontproperties=chinese_font)
        plt.xlabel("日期", fontproperties=chinese_font)
        plt.ylabel("PET (mm/day)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_TIME_SERIES_PNG, dpi=300)
        plt.close()
        bar.update(1)

        plt.figure(figsize=(8, 6))
        lon_res = abs(float(np.nanmedian(np.diff(target_lon))))
        lat_res = abs(float(np.nanmedian(np.diff(target_lat))))
        extent = [
            float(target_lon.min() - lon_res / 2.0),
            float(target_lon.max() + lon_res / 2.0),
            float(target_lat.min() - lat_res / 2.0),
            float(target_lat.max() + lat_res / 2.0),
        ]
        image = plt.imshow(
            mean_spatial,
            cmap="YlOrRd",
            origin="lower",
            extent=extent,
            aspect="equal",
        )
        plt.title("日尺度PET多年平均空间分布", fontproperties=chinese_font)
        plt.xlabel("经度", fontproperties=chinese_font)
        plt.ylabel("纬度", fontproperties=chinese_font)
        plt.colorbar(image, label="PET (mm/day)")
        plt.tight_layout()
        plt.savefig(OUTPUT_MEAN_SPATIAL_PNG, dpi=300)
        plt.close()
        bar.update(1)


def cleanup_temp_cache() -> None:
    """删除本次脚本运行产生的专用临时缓存目录。"""
    with make_bar(1, "步骤9 清理缓存", "项", BAR_COLOR_CLEAN) as bar:
        if TEMP_CACHE_DIR.exists():
            shutil.rmtree(TEMP_CACHE_DIR)
        bar.update(1)


# =============================================================================
# 八、主流程
# =============================================================================
def main() -> None:
    """按固定流程执行 daily PET 估算。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        target_lon, target_lat, target_mask = read_chirps_target_grid(
            CHIRPS_TEMPLATE_NC,
            FUJIAN_BOUNDARY_SHP,
            TARGET_GRID_BUFFER_DEGREES,
        )
        latitude = latitude_grid_from_target(target_lat, target_lon)
        elev = resample_dem_to_target(DEM_PATH, target_lon, target_lat)
        elev = apply_boundary_mask(elev, target_mask)

        t2m_source_lon, t2m_source_lat = get_tif_grid(find_first_tif(T2M_INPUT))
        temp_hourly, temp_dates_raw, unit_note = read_t2m_hourly_series(T2M_INPUT)
        temp_dates = make_hourly_dates(temp_hourly.shape[0], temp_dates_raw, T2M_START_DATETIME)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            tmean, tmax, tmin, temp_daily_dates = aggregate_temperature_daily(
                temp_hourly,
                temp_dates,
                DAY_AGGREGATION_UTC_OFFSET_HOURS,
            )
        tmean = interpolate_3d_regular_grid(
            tmean,
            t2m_source_lon,
            t2m_source_lat,
            target_lon,
            target_lat,
            "步骤3b 温度重采样",
        )
        tmax = interpolate_3d_regular_grid(
            tmax,
            t2m_source_lon,
            t2m_source_lat,
            target_lon,
            target_lat,
            "步骤3c 最高温重采样",
        )
        tmin = interpolate_3d_regular_grid(
            tmin,
            t2m_source_lon,
            t2m_source_lat,
            target_lon,
            target_lat,
            "步骤3d 最低温重采样",
        )
        tmean = apply_boundary_mask(tmean, target_mask)
        tmax = apply_boundary_mask(tmax, target_mask)
        tmin = apply_boundary_mask(tmin, target_mask)

        srad_data, srad_dates_raw, srad_resolution, srad_source_lon, srad_source_lat = read_srad_nc_series(
            SRAD_INPUT,
            SRAD_FILE_PATTERN,
            SRAD_VAR_NAME,
        )
        srad_dates = make_hourly_dates(srad_data.shape[0], srad_dates_raw, None)
        rs, srad_daily_dates = aggregate_srad_daily(
            srad_data,
            srad_dates,
            DAY_AGGREGATION_UTC_OFFSET_HOURS,
            srad_resolution,
        )
        rs = interpolate_3d_regular_grid(
            rs,
            srad_source_lon,
            srad_source_lat,
            target_lon,
            target_lat,
            "步骤5b 辐射重采样",
        )
        rs = apply_boundary_mask(rs, target_mask)

        tmean, tmax, tmin, rs, dates = align_daily_inputs(
            tmean,
            tmax,
            tmin,
            temp_daily_dates,
            rs,
            srad_daily_dates,
        )
        validate_shapes(elev, tmean, rs)

        tqdm.write(f"温度单位识别结果: {unit_note}")
        tqdm.write(f"太阳辐射时间分辨率识别结果: {srad_resolution}")
        tqdm.write(f"共同日期数量: {len(dates)}")
        tqdm.write(
            f"PET 计算假设: u2={ASSUMED_WIND_SPEED_M_S} m/s, "
            "ea=esat(Tmin), G=0"
        )

        pet = compute_daily_pet_fao56_approx(
            tmean,
            tmax,
            tmin,
            rs,
            elev,
            dates,
            latitude_deg=latitude,
            wind_speed_m_s=ASSUMED_WIND_SPEED_M_S,
        )

        save_pet_outputs(pet, dates, srad_resolution, target_lon, target_lat)

        tqdm.write(f"计算完成，结果目录: {OUTPUT_DIR}")
        tqdm.write(f"PET 均值: {float(np.nanmean(pet)):.4f} mm/day")
        tqdm.write(f"PET 最大值: {float(np.nanmax(pet)):.4f} mm/day")

    finally:
        cleanup_temp_cache()


if __name__ == "__main__":
    main()
