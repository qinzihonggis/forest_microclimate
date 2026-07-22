import os
import shutil
import tempfile
from contextlib import ExitStack

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm


# ====================== 1. 路径参数 ======================
# base_path:
#   项目主目录。脚本会从该目录读取站点 CSV，并把提取后的结果写回同一个 CSV。
#   如果后续项目目录变化，只需要修改这个路径。
# dem_path / slope_path / aspect_path:
#   福建区域地形栅格路径，分别对应海拔、坡度、坡向。
# canopy_nasia_path / canopy_sasia_path:
#   冠层高度栅格路径。站点可能落在 NASIA 或 SASIA 的覆盖范围内，脚本会按坐标自动匹配。
# csv_path:
#   站点经纬度表格路径，要求至少包含 Site_ID、Longitude、Latitude 三列。
base_path = r"E:\forest_microclimate\ForestMicroclimate"
dem_path = os.path.join(base_path, "DEM_fujian", "fujian_dem.tif")
slope_path = os.path.join(base_path, "DEM_fujian", "fujian_slope.tif")
aspect_path = os.path.join(base_path, "DEM_fujian", "fujian_aspect.tif")
canopy_nasia_path = os.path.join(base_path, "Canopy_height", "Forest_height_2019_NASIA.tif")
canopy_sasia_path = os.path.join(base_path, "Canopy_height", "Forest_height_2019_SASIA.tif")
csv_path = os.path.join(base_path, "Tensor_LatLong.csv")


# ====================== 2. 进度条参数 ======================
# PROGRESS_BAR_FORMAT:
#   统一控制 tqdm 进度条显示内容。
#   percentage 显示百分比，bar 显示彩色进度条，n_fmt/total_fmt 显示当前量和总量，
#   elapsed/remaining 显示已耗时和预计剩余时间，rate_fmt 显示处理速度。
# dynamic_ncols=True:
#   根据终端宽度自动调整进度条长度，避免换行刷屏。
# leave=False:
#   每个步骤完成后清除该步骤进度条，保持终端输出简洁。
PROGRESS_BAR_FORMAT = (
    "{desc} | {percentage:3.0f}% |{bar}| "
    "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
)


def make_progress(desc, total, colour, unit="it"):
    """创建单行动态 tqdm 进度条，所有关键步骤共用同一套显示格式。"""
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        colour=colour,
        dynamic_ncols=True,
        leave=False,
        bar_format=PROGRESS_BAR_FORMAT,
    )


def sample_point(src, lon, lat, invalid_ge_101=False):
    """提取单个经纬度位置的栅格值。

    参数说明：
    src:
        已打开的 rasterio 栅格对象。这里不在函数内反复打开文件，可以明显减少 I/O 开销。
    lon / lat:
        站点经度和纬度，来自 Tensor_LatLong.csv 的 Longitude 和 Latitude 列。
    invalid_ge_101:
        是否把大于等于 101 的像元设置为空值。
        该参数只用于冠层高度数据，因为冠层数据中 >=101 表示需要剔除的异常/水体编码。

    返回值：
        有效像元返回 float；超出栅格范围、NoData、掩膜值或冠层 >=101 时返回 np.nan。
    """
    if not (src.bounds.left <= lon <= src.bounds.right and src.bounds.bottom <= lat <= src.bounds.top):
        return np.nan

    try:
        value = next(src.sample([(lon, lat)], masked=True))[0]
    except Exception:
        return np.nan

    if np.ma.is_masked(value):
        return np.nan

    value = float(value)
    if invalid_ge_101 and value >= 101:
        return np.nan
    return value


def sample_canopy(canopy_sources, lon, lat):
    """从 NASIA 和 SASIA 两张冠层高度栅格中提取站点冠层高度。

    参数说明：
    canopy_sources:
        已打开的冠层高度栅格列表。当前顺序为 NASIA 优先，SASIA 其次。
        如果某个站点落在两张图的重叠区域，脚本会先尝试 NASIA；
        如果 NASIA 对应像元为空值或 >=101，则继续尝试 SASIA。
    lon / lat:
        站点经纬度。

    关键规则：
        冠层高度像元值 >=101 会在 sample_point 中被转换为 np.nan，
        避免把湖泊等异常编码当成 101 米森林高度。
    """
    for src in canopy_sources:
        value = sample_point(src, lon, lat, invalid_ge_101=True)
        if not np.isnan(value):
            return value

    return np.nan


def validate_required_columns(df):
    """检查站点表是否包含脚本运行必需字段，避免列名错误导致结果错位。"""
    required_columns = ["Site_ID", "Longitude", "Latitude"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV 缺少必要列: {', '.join(missing_columns)}")


def read_station_csv():
    """读取站点 CSV，并用进度条标记读取步骤完成情况。"""
    with make_progress("读取站点CSV", total=1, colour="cyan", unit="file") as progress:
        df = pd.read_csv(csv_path)
        validate_required_columns(df)
        progress.update(1)
    return df


def open_rasters(stack):
    """打开所有需要采样的栅格文件。

    参数说明：
    stack:
        ExitStack 上下文管理器。所有栅格对象交给 stack 管理，
        脚本结束或报错时会自动关闭文件句柄，避免文件被占用。

    返回值：
        raster_sources 字典，键名用于后续明确区分海拔、坡度、坡向、冠层高度数据。
    """
    raster_paths = {
        "dem": dem_path,
        "slope": slope_path,
        "aspect": aspect_path,
        "canopy_nasia": canopy_nasia_path,
        "canopy_sasia": canopy_sasia_path,
    }

    raster_sources = {}
    with make_progress("打开栅格文件", total=len(raster_paths), colour="yellow", unit="file") as progress:
        for name, path in raster_paths.items():
            raster_sources[name] = stack.enter_context(rasterio.open(path))
            progress.update(1)

    return raster_sources


def extract_site_values(df, raster_sources):
    """逐站点提取地形和冠层高度数据。

    参数说明：
    df:
        站点表，必须包含 Site_ID、Longitude、Latitude。
    raster_sources:
        open_rasters 返回的栅格对象字典。

    输出字段：
        Elevation:
            站点海拔，来自 fujian_dem.tif。
        Slope:
            站点坡度，来自 fujian_slope.tif。
        Aspect:
            站点坡向，来自 fujian_aspect.tif。
        Canopy_Height:
            站点冠层高度，单位为米，来自两张 Forest_height_2019 栅格；
            原始像元值 >=101 会被剔除并写为 NaN。
    """
    elevations = []
    slopes = []
    aspects = []
    canopy_heights = []

    canopy_sources = [raster_sources["canopy_nasia"], raster_sources["canopy_sasia"]]

    with make_progress("提取站点数据", total=len(df), colour="green", unit="site") as progress:
        for _, row in df.iterrows():
            lon = row["Longitude"]
            lat = row["Latitude"]

            elevations.append(sample_point(raster_sources["dem"], lon, lat))
            slopes.append(sample_point(raster_sources["slope"], lon, lat))
            aspects.append(sample_point(raster_sources["aspect"], lon, lat))
            canopy_heights.append(sample_canopy(canopy_sources, lon, lat))

            progress.update(1)

    df["Elevation"] = elevations
    df["Slope"] = slopes
    df["Aspect"] = aspects
    df["Canopy_Height"] = canopy_heights
    return df


def write_station_csv(df):
    """把提取结果写回原 CSV。

    参数说明：
    index=False:
        不额外写入 pandas 行号，避免 CSV 多出无意义索引列。
    encoding="utf-8-sig":
        使用带 BOM 的 UTF-8，Excel 直接打开时中文列名和内容更不容易乱码。
    """
    with make_progress("写回结果CSV", total=1, colour="blue", unit="file") as progress:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        progress.update(1)


def cleanup_temp_files(temp_dir):
    """删除本次脚本运行产生的临时目录。

    参数说明：
    temp_dir:
        脚本启动时创建的专用临时目录。脚本会把 TMP/TEMP/TMPDIR/CPL_TMPDIR 指向这里，
        让 pandas、rasterio、GDAL 等库在需要临时文件时优先写到该目录。
        运行结束后删除整个目录，避免残留本次运行产生的缓存或临时文件。
    """
    with make_progress("清理临时文件", total=1, colour="magenta", unit="dir") as progress:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        progress.update(1)


def main():
    """脚本主流程：读取 CSV -> 打开栅格 -> 提取数据 -> 写回 CSV -> 清理临时文件。"""
    temp_dir = tempfile.mkdtemp(prefix="extract_dem_canopy_")

    # 将常见临时目录环境变量指向本次运行的专用目录，便于最后统一清理。
    os.environ["TMP"] = temp_dir
    os.environ["TEMP"] = temp_dir
    os.environ["TMPDIR"] = temp_dir
    os.environ["CPL_TMPDIR"] = temp_dir

    try:
        df = read_station_csv()
        with ExitStack() as stack:
            raster_sources = open_rasters(stack)
            df = extract_site_values(df, raster_sources)
        write_station_csv(df)
    finally:
        cleanup_temp_files(temp_dir)


if __name__ == "__main__":
    main()
