# -*- coding: utf-8 -*-
"""
ERA5-LAND 福建省 2 米气温 GeoTIFF 数据插值为严格 1 km 投影栅格。
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import reproject
from tqdm import tqdm


# =============================================================================
# 一、可调参数区
# =============================================================================
# 原始 ERA5-LAND 福建省 2 米气温 tif 文件所在文件夹。
# 修改意义：
# 1. 脚本会在该文件夹第一层查找 .tif 和 .tiff 文件，不递归读取子文件夹。
# 2. 你的 8760 张逐小时温度 tif 应全部放在这个文件夹中。
INPUT_DIR = Path(r"E:\forest_microclimate\ForestMicroclimate\T2m\fujian_T2")

# 插值结果输出文件夹。
# 修改意义：
# 1. 默认在原始温度数据路径下新建 fujian_T2m_1km 文件夹。
# 2. 每小时输出一张 tif，文件名在原始文件名前加 “1km”。
OUTPUT_DIR = INPUT_DIR / "fujian_T2m_1km"

# 福建省行政边界，地理坐标系版本。
# 修改意义：
# 1. 当前脚本主要使用投影坐标系边界完成 1 km 建网和掩膜。
# 2. 保留该参数用于路径检查和后续如需按经纬度范围预筛选时扩展。
BOUNDARY_GEO_PATH = Path(
    r"E:\forest_microclimate\ForestMicroclimate\Fujian_Shp\福建省行政边界.shp"
)

# 福建省行政边界，投影坐标系版本。
# 修改意义：
# 1. 该文件用于构建严格 1000 m x 1000 m 目标网格。
# 2. 当前按 WGS 84 / UTM zone 50N 处理，即 EPSG:32650，坐标单位为米。
# 3. 如果后续换成其它投影边界，需要同时确认 TARGET_CRS 是否一致。
BOUNDARY_UTM_PATH = Path(
    r"E:\forest_microclimate\ForestMicroclimate\Fujian_Shp\福建省行政边界_WGS1984UTM50N.shp"
)

# 原始 tif 缺失坐标系时采用的默认坐标系。
# 修改意义：
# 1. ERA5-LAND 裁剪后的 tif 通常为 WGS84 经纬度坐标，即 EPSG:4326。
# 2. 如果原始 tif 自带 CRS，脚本优先使用 tif 内部 CRS，不会强行覆盖。
# 3. 只有当 tif 没有 CRS 时，才使用这里的 SOURCE_CRS_IF_MISSING。
SOURCE_CRS_IF_MISSING = "EPSG:4326"

# 输出目标坐标系。
# 修改意义：
# 1. EPSG:32650 是 WGS 84 / UTM zone 50N，单位为米，适合构建严格 1 km 网格。
# 2. 不建议把 1 km 输出为 EPSG:4326，因为经纬度单位是度，不是米。
TARGET_CRS = "EPSG:32650"

# 输出目标空间分辨率，单位为米。
# 修改意义：
# 1. 1000.0 表示输出像元大小严格为 1000 m x 1000 m。
# 2. 如需改成 500 m 或 2000 m，只改这里即可。
TARGET_RESOLUTION_M = 1000.0

# 重采样方法。
# 修改意义：
# 1. bilinear 表示双线性插值，适合温度这类连续变量。
# 2. nearest 表示最近邻，通常适合分类数据，不建议用于温度。
# 3. cubic 表示三次卷积，结果更平滑但计算更慢。
RESAMPLING_METHOD = "bilinear"

# 输出 NoData 值。
# 修改意义：
# 1. 福建省边界外像元会写为该值。
# 2. GeoTIFF 中使用明确的 -9999 通常比 NaN 更利于 GIS 软件识别 NoData。
NODATA_VALUE = -9999.0

# 是否保留所有接触边界的像元。
# 修改意义：
# 1. False 表示仅保留像元中心落在福建省边界内的像元，边界更保守。
# 2. True 表示只要像元接触边界就保留，边界覆盖更充分，但可能多保留少量边界外像元。
ALL_TOUCHED = False

# 输入 tif 的波段序号。
# 修改意义：
# 1. rasterio 波段序号从 1 开始。
# 2. 当前温度 tif 通常只有 1 个波段，因此默认读取第 1 波段。
SOURCE_BAND_INDEX = 1

# 输出文件名前缀。
# 修改意义：
# 1. 原始文件名为 “福建省2米气温_2025年01月01日00时.tif” 时，
#    输出文件名为 “1km福建省2米气温_2025年01月01日00时.tif”。
OUTPUT_NAME_PREFIX = "1km"

# 输出文件扩展名。
# 修改意义：
# 1. 即使原始文件为 .tiff，输出也统一使用 .tif，便于后续批量识别。
OUTPUT_SUFFIX = ".tif"

# 输出数据类型。
# 修改意义：
# 1. float32 对温度数据通常足够，且文件体积明显小于 float64。
# 2. 如后续对数值精度要求极高，可以改为 float64，但文件会更大。
OUTPUT_DTYPE = "float32"

# GeoTIFF 压缩方式。
# 修改意义：
# 1. LZW 是常用无损压缩，可减小文件体积。
# 2. 如需最快写出速度，可以改为 None，但输出文件会更大。
GTIFF_COMPRESS = "LZW"

# 是否创建分块 GeoTIFF。
# 修改意义：
# 1. True 通常利于后续 GIS 软件分块读取。
# 2. 对很小的栅格差异不明显，但保持开启通常更稳妥。
GTIFF_TILED = True

# tqdm 进度条格式。
# 修改意义：
# 1. percentage 显示百分比。
# 2. n_fmt/total_fmt 显示当前数量和总数量。
# 3. elapsed/remaining/rate_fmt 显示已用时间、剩余时间和处理速度。
PROGRESS_BAR_FORMAT = (
    "{l_bar}{bar:34}| {percentage:3.0f}% {n_fmt}/{total_fmt} "
    "[已用 {elapsed} < 剩余 {remaining}, {rate_fmt}]"
)

# 进度条颜色。
# 修改意义：
# 1. 不同步骤使用不同颜色，便于在终端中快速区分当前处理阶段。
# 2. tqdm 支持常见颜色名，例如 blue、green、cyan、magenta、yellow、red。
OVERALL_BAR_COLOR = "cyan"
CHECK_BAR_COLOR = "yellow"
IO_BAR_COLOR = "blue"
PROCESS_BAR_COLOR = "green"
SAVE_BAR_COLOR = "magenta"


# =============================================================================
# 二、工具函数区
# =============================================================================
def make_bar(total: int, desc: str, unit: str, colour: str) -> tqdm:
    """创建统一样式的彩色 tqdm 进度条。"""
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        colour=colour,
        dynamic_ncols=True,
        bar_format=PROGRESS_BAR_FORMAT,
    )


def print_parameters() -> None:
    """打印当前可调参数，方便运行前确认和后续修改。"""
    print("\n当前可调参数如下：")
    print(f"输入温度文件夹: {INPUT_DIR}")
    print(f"输出 1 km 文件夹: {OUTPUT_DIR}")
    print(f"地理坐标系边界: {BOUNDARY_GEO_PATH}")
    print(f"投影坐标系边界: {BOUNDARY_UTM_PATH}")
    print(f"原始 tif 缺失 CRS 时默认使用: {SOURCE_CRS_IF_MISSING}")
    print(f"输出目标 CRS: {TARGET_CRS}")
    print(f"输出目标分辨率: {TARGET_RESOLUTION_M:.0f} m")
    print(f"重采样方法: {RESAMPLING_METHOD}")
    print(f"输出 NoData: {NODATA_VALUE}")
    print(f"边界 all_touched: {ALL_TOUCHED}")
    print(f"输入波段序号: {SOURCE_BAND_INDEX}")
    print(f"输出文件名前缀: {OUTPUT_NAME_PREFIX}")
    print(f"输出文件扩展名: {OUTPUT_SUFFIX}")
    print(f"输出数据类型: {OUTPUT_DTYPE}")
    print(f"GeoTIFF 压缩方式: {GTIFF_COMPRESS}")
    print(f"GeoTIFF 分块写出: {GTIFF_TILED}\n")


def get_resampling_method() -> Resampling:
    """把字符串形式的重采样方法转换为 rasterio 可识别的枚举值。"""
    methods = {
        "bilinear": Resampling.bilinear,
        "nearest": Resampling.nearest,
        "cubic": Resampling.cubic,
    }
    if RESAMPLING_METHOD not in methods:
        supported = ", ".join(methods)
        raise ValueError(f"不支持的重采样方法：{RESAMPLING_METHOD}；可选值为：{supported}")
    return methods[RESAMPLING_METHOD]


def list_input_tifs(input_dir: Path) -> list[Path]:
    """列出输入文件夹第一层的 tif 文件，并按文件名排序。"""
    tif_files = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))
    tif_files = [path for path in tif_files if path.is_file()]
    if not tif_files:
        raise FileNotFoundError(f"输入文件夹中没有找到 tif 文件：{input_dir}")
    return tif_files


def check_input_paths() -> list[Path]:
    """检查输入路径是否存在，并创建输出文件夹。"""
    missing_paths = [
        path
        for path in [INPUT_DIR, BOUNDARY_GEO_PATH, BOUNDARY_UTM_PATH]
        if not path.exists()
    ]
    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"以下输入路径不存在：\n{missing_text}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return list_input_tifs(INPUT_DIR)


def read_projected_boundary() -> gpd.GeoDataFrame:
    """读取福建省投影边界，清理空几何，并统一到目标投影坐标系。"""
    boundary = gpd.read_file(BOUNDARY_UTM_PATH)
    if boundary.empty:
        raise ValueError(f"投影边界文件为空：{BOUNDARY_UTM_PATH}")

    boundary = boundary[boundary.geometry.notna() & ~boundary.geometry.is_empty].copy()
    if boundary.empty:
        raise ValueError(f"投影边界文件没有有效几何：{BOUNDARY_UTM_PATH}")

    if boundary.crs is None:
        boundary = boundary.set_crs(TARGET_CRS, allow_override=True)
    else:
        boundary = boundary.to_crs(TARGET_CRS)

    return boundary


def build_target_grid(boundary_utm: gpd.GeoDataFrame) -> tuple:
    """根据福建省投影边界构建严格 1000 m 目标网格。"""
    min_x, min_y, max_x, max_y = boundary_utm.total_bounds

    left = np.floor(min_x / TARGET_RESOLUTION_M) * TARGET_RESOLUTION_M
    right = np.ceil(max_x / TARGET_RESOLUTION_M) * TARGET_RESOLUTION_M
    bottom = np.floor(min_y / TARGET_RESOLUTION_M) * TARGET_RESOLUTION_M
    top = np.ceil(max_y / TARGET_RESOLUTION_M) * TARGET_RESOLUTION_M

    width = int(round((right - left) / TARGET_RESOLUTION_M))
    height = int(round((top - bottom) / TARGET_RESOLUTION_M))
    if width <= 0 or height <= 0:
        raise ValueError("目标 1 km 网格为空，请检查投影边界范围。")

    transform = from_origin(left, top, TARGET_RESOLUTION_M, TARGET_RESOLUTION_M)
    return transform, width, height


def build_boundary_mask(
    boundary_utm: gpd.GeoDataFrame,
    transform,
    width: int,
    height: int,
) -> np.ndarray:
    """根据福建省边界生成目标网格掩膜，True 表示保留福建省范围内像元。"""
    return geometry_mask(
        geometries=boundary_utm.geometry,
        out_shape=(height, width),
        transform=transform,
        invert=True,
        all_touched=ALL_TOUCHED,
    )


def get_source_crs(dataset: rasterio.DatasetReader) -> CRS:
    """获取输入 tif 的 CRS；如果缺失，则使用参数区指定的默认源 CRS。"""
    if dataset.crs is not None:
        return dataset.crs
    return CRS.from_string(SOURCE_CRS_IF_MISSING)


def build_output_path(source_path: Path) -> Path:
    """根据原始 tif 文件名生成 1 km 输出 tif 路径。"""
    output_name = f"{OUTPUT_NAME_PREFIX}{source_path.stem}{OUTPUT_SUFFIX}"
    return OUTPUT_DIR / output_name


def build_output_profile(transform, width: int, height: int) -> dict:
    """生成输出 GeoTIFF 的空间参考、数据类型、压缩和 NoData 参数。"""
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": OUTPUT_DTYPE,
        "crs": CRS.from_string(TARGET_CRS),
        "transform": transform,
        "nodata": NODATA_VALUE,
        "compress": GTIFF_COMPRESS,
        "tiled": GTIFF_TILED,
        "BIGTIFF": "IF_SAFER",
    }
    if OUTPUT_DTYPE == "float32":
        profile["predictor"] = 3
    return profile


def read_source_band(dataset: rasterio.DatasetReader) -> tuple[np.ndarray, float | int | None]:
    """读取源 tif 波段，并判断是否需要把 NaN 当作源 NoData。"""
    source_data = dataset.read(SOURCE_BAND_INDEX)
    source_nodata = dataset.nodata

    if source_nodata is None and np.issubdtype(source_data.dtype, np.floating):
        if np.isnan(source_data).any():
            source_nodata = np.nan

    return source_data, source_nodata


def interpolate_one_tif(
    source_path: Path,
    target_transform,
    target_width: int,
    target_height: int,
    boundary_mask: np.ndarray,
    output_profile: dict,
    resampling_method: Resampling,
) -> Path:
    """对单张温度 tif 执行双线性重投影插值、边界掩膜和 GeoTIFF 写出。"""
    output_path = build_output_path(source_path)

    with rasterio.open(source_path) as src:
        source_data, source_nodata = read_source_band(src)
        destination = np.full(
            (target_height, target_width),
            NODATA_VALUE,
            dtype=OUTPUT_DTYPE,
        )

        reproject(
            source=source_data,
            destination=destination,
            src_transform=src.transform,
            src_crs=get_source_crs(src),
            src_nodata=source_nodata,
            dst_transform=target_transform,
            dst_crs=CRS.from_string(TARGET_CRS),
            dst_nodata=NODATA_VALUE,
            resampling=resampling_method,
        )

        destination[~boundary_mask] = NODATA_VALUE
        if np.issubdtype(destination.dtype, np.floating):
            destination[np.isnan(destination)] = NODATA_VALUE

        with rasterio.open(output_path, "w", **output_profile) as dst:
            dst.write(destination, 1)
            dst.set_band_description(1, "2m_temperature_1km")

    return output_path


# =============================================================================
# 三、主流程
# =============================================================================
def main() -> None:
    """执行 ERA5-LAND 2 米气温逐小时 tif 插值到福建省 1 km 网格的完整流程。"""
    print_parameters()

    with make_bar(6, "总进度", "步", OVERALL_BAR_COLOR) as overall:
        # ---------------------------------------------------------------------
        # 步骤 1：检查输入路径、创建输出文件夹，并统计需要处理的逐小时 tif 数量。
        # ---------------------------------------------------------------------
        with make_bar(1, "步骤1 检查路径", "项", CHECK_BAR_COLOR) as bar:
            tif_files = check_input_paths()
            print(f"发现输入 tif 数量: {len(tif_files)}")
            bar.update(1)
        overall.update(1)

        # ---------------------------------------------------------------------
        # 步骤 2：读取福建省投影边界，并统一到 EPSG:32650。
        # ---------------------------------------------------------------------
        with make_bar(1, "步骤2 读取边界", "项", IO_BAR_COLOR) as bar:
            boundary_utm = read_projected_boundary()
            bar.update(1)
        overall.update(1)

        # ---------------------------------------------------------------------
        # 步骤 3：根据投影边界范围构建严格 1000 m x 1000 m 目标网格。
        # ---------------------------------------------------------------------
        with make_bar(1, "步骤3 构建网格", "项", PROCESS_BAR_COLOR) as bar:
            target_transform, target_width, target_height = build_target_grid(boundary_utm)
            print(f"目标网格大小: width={target_width}, height={target_height}")
            bar.update(1)
        overall.update(1)

        # ---------------------------------------------------------------------
        # 步骤 4：在目标网格上生成福建省边界掩膜，后续每张 tif 复用该掩膜。
        # ---------------------------------------------------------------------
        with make_bar(1, "步骤4 生成掩膜", "项", PROCESS_BAR_COLOR) as bar:
            boundary_mask = build_boundary_mask(
                boundary_utm,
                target_transform,
                target_width,
                target_height,
            )
            bar.update(1)
        overall.update(1)

        # ---------------------------------------------------------------------
        # 步骤 5：逐小时处理 8760 张温度 tif，执行双线性插值、掩膜和写出。
        # ---------------------------------------------------------------------
        output_profile = build_output_profile(target_transform, target_width, target_height)
        resampling_method = get_resampling_method()
        output_files = []
        with tqdm(
            tif_files,
            total=len(tif_files),
            desc="步骤5 插值写出",
            unit="张",
            colour=SAVE_BAR_COLOR,
            dynamic_ncols=True,
            bar_format=PROGRESS_BAR_FORMAT,
        ) as file_bar:
            for tif_path in file_bar:
                file_bar.set_postfix_str(tif_path.name[:24])
                output_path = interpolate_one_tif(
                    tif_path,
                    target_transform,
                    target_width,
                    target_height,
                    boundary_mask,
                    output_profile,
                    resampling_method,
                )
                output_files.append(output_path)
        overall.update(1)

        # ---------------------------------------------------------------------
        # 步骤 6：输出处理摘要，确认输出路径、文件数量、分辨率和坐标系。
        # ---------------------------------------------------------------------
        with make_bar(1, "步骤6 输出摘要", "项", IO_BAR_COLOR) as bar:
            print("\n处理完成")
            print(f"输入文件夹: {INPUT_DIR}")
            print(f"输出文件夹: {OUTPUT_DIR}")
            print(f"输入 tif 数量: {len(tif_files)}")
            print(f"输出 tif 数量: {len(output_files)}")
            print(f"输出坐标系: {TARGET_CRS}")
            print(f"输出分辨率: {TARGET_RESOLUTION_M:.0f} m x {TARGET_RESOLUTION_M:.0f} m")
            print(f"输出 NoData: {NODATA_VALUE}")
            bar.update(1)
        overall.update(1)


if __name__ == "__main__":
    main()
