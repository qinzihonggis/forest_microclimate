# -*- coding: utf-8 -*-
"""
批量将 CHIRPS 5 天降雨 NetCDF 数据重采样为严格 1 km 网格。

核心逻辑：
1. 批量处理 fujian_pre 目录当前层级下的 .nc 文件，不递归子目录。
2. 保持 CHIRPS 原始 5 天时间步，不做月汇总。
3. 使用福建省 UTM 50N 投影边界，构建单位为米的 1000 m 目标网格。
4. 对每个文件、每个时间步执行双线性重投影，得到严格 1 km 降雨格网。
5. 使用福建省行政边界掩膜，边界外设为 NaN。
6. 输出到 fujian_pre_1km 目录，文件名格式为 fujian_年份_pre_1km.nc。
7. 脚本结束时只清理本次运行创建的临时目录，不删除原始数据和输出结果。
"""

from pathlib import Path
import os
import re
import shutil

import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401  # 启用 xarray 对象的 .rio 空间方法
import xarray as xr
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from tqdm import tqdm


# =============================================================================
# 一、可调参数区
# =============================================================================
# 输入 CHIRPS 降雨 NetCDF 文件所在目录。
# 当前只处理该目录当前层级下的 .nc 文件，不处理子目录。
INPUT_DIR = Path(
    r"E:\forest_microclimate\ForestMicroclimate\Precipitation_CHIRPS\fujian_pre"
)

# 输出目录。
# 所有 1 km 结果都会写入该目录，避免和原始 NC 混在一起。
OUTPUT_DIR = INPUT_DIR / "fujian_pre_1km"

# 输出文件命名模板。
# {year} 会被输入文件名中的 4 位年份替换，例如 2025 -> fujian_2025_pre_1km.nc。
OUTPUT_NAME_TEMPLATE = "fujian_{year}_pre_1km.nc"

# 是否递归处理 INPUT_DIR 下的子目录。
# 你的需求是只处理 fujian_pre 当前目录，因此默认 False。
RECURSIVE_INPUT = False

# 如果输出文件已经存在，是否覆盖。
# False：默认跳过已有 _1km.nc，适合断点续跑。
# True：重新生成并覆盖已有输出。
OVERWRITE = False

# 用于排除已经处理过的文件。
# 输入扫描时，文件名以该后缀结尾的 .nc 会被跳过，避免重复把 1 km 结果再次插值。
PROCESSED_STEM_SUFFIX = "_1km"

# 福建省行政边界投影坐标 shp 所在路径。
# 当前使用 WGS 1984 UTM Zone 50N，也就是 EPSG:32650，坐标单位是米。
# 如果该路径是目录，脚本会自动在目录下寻找 .shp 文件。
BOUNDARY_PATH = Path(
    r"E:\forest_microclimate\ForestMicroclimate\Fujian_Shp"
) / "\u798f\u5efa\u7701\u884c\u653f\u8fb9\u754c_WGS1984UTM50N.shp"

# 目标投影坐标系。
# EPSG:32650 = WGS 84 / UTM zone 50N，适合福建区域，单位为米。
# 严格 1 km 网格必须在米制投影坐标系下构建，不能用 0.01 度近似。
TARGET_CRS = "EPSG:32650"

# 目标空间分辨率，单位为米。
# 1000.0 表示输出网格像元大小严格为 1000 m x 1000 m。
TARGET_RES = 1000.0

# CHIRPS 原始数据的坐标系。
# CHIRPS 通常是 WGS84 经纬度坐标，即 EPSG:4326。
SOURCE_CRS = "EPSG:4326"

# 原始数据裁剪缓冲区，单位为经纬度度。
# 作用：重投影前在福建边界外多保留一圈原始 CHIRPS 数据，避免边缘插值缺少邻近格点。
# 0.25 度约等于 5 个 CHIRPS 0.05 度格点，通常足够；如果边缘出现 NaN，可适当增大。
BOUNDARY_PAD_DEG = 0.25

# 降雨变量名。
# 你说明多个 .nc 的变量和坐标结构一致；如果没有 precip 但只有一个数据变量，脚本会自动改名。
PRECIP_VAR = "precip"

# 是否将插值后出现的极小负值截断为 0。
# 降雨量不应为负；双线性重采样一般不会产生明显负值，但边缘或数值误差可能出现极小负值。
CLIP_NEGATIVE_TO_ZERO = True

# 是否仅保留与福建省边界中心点相交的 1 km 像元。
# False：只保留像元中心落在边界内的像元，边界更保守。
# True：只要像元接触边界就保留，边界覆盖更充分，但边界外可能多保留少量像元。
ALL_TOUCHED = False

# 本次运行临时目录。
# 脚本只会清理这个目录，不会删除原始 NC、输出 NC、shp 配套文件或系统全局缓存。
TEMP_DIR = OUTPUT_DIR / "_tmp_chirps_1km"

# 是否在脚本结束后删除 TEMP_DIR。
# True：推荐，保持输出目录干净。
# False：调试时可保留临时目录，便于检查中间文件。
CLEAN_TEMP_FILES = True

# tqdm 进度条显示参数。
# colour 控制彩色条；dynamic_ncols=True 可让进度条自动适应终端宽度。
PROGRESS_BAR_FORMAT = (
    "{l_bar}{bar:32}| {n_fmt}/{total_fmt} "
    "[已用 {elapsed} < 剩余 {remaining}, {rate_fmt}]"
)
BATCH_BAR_COLOR = "cyan"
FILE_BAR_COLOR = "yellow"
IO_BAR_COLOR = "blue"
PROCESS_BAR_COLOR = "green"
SAVE_BAR_COLOR = "magenta"
SKIP_BAR_COLOR = "white"


# =============================================================================
# 二、工具函数
# =============================================================================
def make_bar(total: int, desc: str, unit: str, colour: str) -> tqdm:
    """创建统一样式的 tqdm 彩色进度条。"""
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        colour=colour,
        dynamic_ncols=True,
        bar_format=PROGRESS_BAR_FORMAT,
    )


def configure_temp_dir() -> None:
    """
    创建本次运行专用临时目录，并让常见临时文件优先写到该目录。

    说明：
    这些环境变量只影响当前 Python 进程及其调用的底层库。
    脚本结束后，cleanup_temp_dir() 只删除 TEMP_DIR 本身，避免误删系统缓存。
    """
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = str(TEMP_DIR)
    os.environ["TMP"] = temp_path
    os.environ["TEMP"] = temp_path
    os.environ["TMPDIR"] = temp_path
    os.environ["CPL_TMPDIR"] = temp_path


def cleanup_temp_dir() -> None:
    """
    清理本次运行产生的临时目录。

    安全边界：
    只删除 TEMP_DIR，且 TEMP_DIR 必须位于 OUTPUT_DIR 内。
    这样可以避免错误路径导致原始数据目录、输出目录或其它文件被删除。
    """
    if not CLEAN_TEMP_FILES or not TEMP_DIR.exists():
        return

    temp_resolved = TEMP_DIR.resolve()
    output_resolved = OUTPUT_DIR.resolve()
    if output_resolved not in temp_resolved.parents:
        raise RuntimeError(f"临时目录不在输出目录内，拒绝删除: {TEMP_DIR}")

    shutil.rmtree(TEMP_DIR)


def find_shapefile(path: Path) -> Path:
    """
    查找福建省边界 .shp 文件。

    参数说明：
    path 可以是具体 .shp 文件，也可以是存放 shp 相关文件的目录。
    如果是目录，脚本会先找目录第一层的 .shp；找不到时再递归搜索子目录。
    """
    if path.is_file() and path.suffix.lower() == ".shp":
        return path

    if not path.exists():
        raise FileNotFoundError(f"边界路径不存在: {path}")

    shp_files = sorted(path.glob("*.shp"))
    if not shp_files:
        shp_files = sorted(path.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"边界路径下没有找到 .shp 文件: {path}")
    if len(shp_files) > 1:
        print(f"发现多个 shp 文件，默认使用第一个: {shp_files[0]}")
    return shp_files[0]


def discover_input_files() -> list[Path]:
    """
    扫描待处理的 NC 文件。

    规则：
    1. 默认只扫描 INPUT_DIR 当前层级的 .nc 文件。
    2. 跳过文件名已经以 _1km 结尾的结果文件。
    3. 跳过 OUTPUT_DIR 内的文件，避免把输出目录中的结果再次作为输入。
    """
    pattern = "**/*.nc" if RECURSIVE_INPUT else "*.nc"
    nc_files = sorted(INPUT_DIR.glob(pattern))

    input_files = []
    output_dir_resolved = OUTPUT_DIR.resolve()
    for path in nc_files:
        if path.stem.endswith(PROCESSED_STEM_SUFFIX):
            continue
        if output_dir_resolved in path.resolve().parents:
            continue
        input_files.append(path)

    if not input_files:
        raise FileNotFoundError(f"没有在输入目录找到可处理的 .nc 文件: {INPUT_DIR}")
    return input_files


def extract_year(input_file: Path) -> str:
    """
    从输入文件名中提取 4 位年份。

    例如：
    fujian_2025_pre.nc -> 2025
    chirps_fujian_2024.nc -> 2024

    如果文件名中没有年份，脚本会报错，避免生成无法追踪来源的输出文件名。
    """
    match = re.search(r"(19|20)\d{2}", input_file.stem)
    if not match:
        raise ValueError(f"文件名中没有找到 4 位年份，无法命名输出文件: {input_file.name}")
    return match.group(0)


def make_output_file(input_file: Path) -> Path:
    """根据输入文件名中的年份生成输出文件路径。"""
    year = extract_year(input_file)
    return OUTPUT_DIR / OUTPUT_NAME_TEMPLATE.format(year=year)


def validate_unique_outputs(input_files: list[Path]) -> list[tuple[Path, Path]]:
    """
    检查批量输出文件名是否重复。

    如果多个输入文件包含同一年份，会生成同一个输出名。
    这种情况会导致结果互相覆盖或被错误跳过，因此直接报错提醒用户检查文件名。
    """
    pairs = [(input_file, make_output_file(input_file)) for input_file in input_files]
    seen = {}
    duplicates = []
    for input_file, output_file in pairs:
        if output_file in seen:
            duplicates.append((seen[output_file], input_file, output_file))
        else:
            seen[output_file] = input_file

    if duplicates:
        message_lines = ["发现多个输入文件会生成同一个输出文件，请检查文件年份:"]
        for first_file, second_file, output_file in duplicates:
            message_lines.append(f"  {first_file.name} 和 {second_file.name} -> {output_file.name}")
        raise ValueError("\n".join(message_lines))
    return pairs


def read_boundary() -> tuple[Path, gpd.GeoDataFrame]:
    """
    读取福建省边界并统一到目标投影。

    边界只需要读取一次，后续所有 NC 文件共用同一个边界和同一个 1 km 目标网格。
    """
    boundary_shp = find_shapefile(BOUNDARY_PATH)
    boundary = gpd.read_file(boundary_shp)
    if boundary.empty:
        raise ValueError(f"边界 shp 为空: {boundary_shp}")

    if boundary.crs is None:
        print(f"边界 CRS 缺失，按 {TARGET_CRS} 处理。")
        boundary = boundary.set_crs(TARGET_CRS, allow_override=True)
    else:
        boundary = boundary.to_crs(TARGET_CRS)

    boundary = boundary[~boundary.geometry.is_empty & boundary.geometry.notna()]
    if boundary.empty:
        raise ValueError("边界 shp 中没有有效几何。")
    return boundary_shp, boundary


def normalize_chirps_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    标准化 CHIRPS 数据的坐标名和变量名。

    目的：
    不同来源的 NetCDF 可能使用 longitude/latitude 或 lon/lat。
    后续空间处理统一使用 lon、lat、time 和 precip，避免坐标名不一致导致报错。
    """
    rename_map = {}
    if "longitude" in ds.coords:
        rename_map["longitude"] = "lon"
    if "latitude" in ds.coords:
        rename_map["latitude"] = "lat"
    if "Longitude" in ds.coords:
        rename_map["Longitude"] = "lon"
    if "Latitude" in ds.coords:
        rename_map["Latitude"] = "lat"
    if rename_map:
        ds = ds.rename(rename_map)

    missing = {"lon", "lat", "time"} - set(ds.coords)
    if missing:
        raise ValueError(f"缺少必要坐标: {sorted(missing)}")

    if PRECIP_VAR not in ds.data_vars:
        if len(ds.data_vars) == 1:
            only_var = next(iter(ds.data_vars))
            print(f"没有找到变量 '{PRECIP_VAR}'，自动使用唯一变量 '{only_var}'。")
            ds = ds.rename({only_var: PRECIP_VAR})
        else:
            raise ValueError(
                f"没有找到变量 '{PRECIP_VAR}'。当前变量包括: {list(ds.data_vars)}"
            )

    return ds


def subset_to_boundary(ds: xr.Dataset, boundary_utm: gpd.GeoDataFrame) -> xr.Dataset:
    """
    将原始 CHIRPS 经纬度数据裁剪到福建边界附近。

    目的：
    CHIRPS 原始数据是经纬度格网，重投影前没必要处理远离福建的格点。
    这里先把 UTM 边界转回 WGS84，经纬度外扩 BOUNDARY_PAD_DEG 后裁剪，
    既减少计算量，也保留边界附近插值所需的邻近格点。
    """
    boundary_wgs84 = boundary_utm.to_crs(SOURCE_CRS)
    min_lon, min_lat, max_lon, max_lat = boundary_wgs84.total_bounds
    min_lon -= BOUNDARY_PAD_DEG
    max_lon += BOUNDARY_PAD_DEG
    min_lat -= BOUNDARY_PAD_DEG
    max_lat += BOUNDARY_PAD_DEG

    lat_values = ds["lat"].values
    lat_slice = (
        slice(min_lat, max_lat)
        if lat_values[0] < lat_values[-1]
        else slice(max_lat, min_lat)
    )

    return ds.sel(lon=slice(min_lon, max_lon), lat=lat_slice)


def build_target_grid(boundary: gpd.GeoDataFrame) -> xr.DataArray:
    """
    根据福建省 UTM 边界构建严格 1000 m 目标网格。

    关键点：
    1. 网格单位是米，因此 TARGET_RES=1000.0 就是严格 1 km。
    2. 网格边界对齐到整千米，便于后续和其它 1 km 数据对齐。
    3. x、y 坐标是像元中心点坐标；transform 描述的是像元左上角和像元大小。
    """
    min_x, min_y, max_x, max_y = boundary.total_bounds

    left = np.floor(min_x / TARGET_RES) * TARGET_RES
    right = np.ceil(max_x / TARGET_RES) * TARGET_RES
    bottom = np.floor(min_y / TARGET_RES) * TARGET_RES
    top = np.ceil(max_y / TARGET_RES) * TARGET_RES

    x = np.arange(left + TARGET_RES / 2, right, TARGET_RES)
    y = np.arange(top - TARGET_RES / 2, bottom, -TARGET_RES)

    target = xr.DataArray(
        np.zeros((len(y), len(x)), dtype=np.float32),
        coords={"y": y, "x": x},
        dims=("y", "x"),
        name="target_grid",
    )
    target = target.rio.write_crs(TARGET_CRS)
    target = target.rio.write_transform(from_origin(left, top, TARGET_RES, TARGET_RES))
    return target


def prepare_source_array(ds: xr.Dataset) -> xr.DataArray:
    """
    将 CHIRPS 降雨变量整理为 rioxarray 可重投影的栅格数组。

    处理内容：
    1. 将 lon/lat 改名为 x/y，让 rioxarray 明确空间维度。
    2. 将纬度按从北到南排序，保持常规 north-up 栅格方向。
    3. 写入原始 CRS=EPSG:4326 和 nodata=NaN。
    """
    da = ds[PRECIP_VAR]
    da = da.sortby("lat", ascending=False).rename({"lon": "x", "lat": "y"})
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    da = da.rio.write_crs(SOURCE_CRS)
    da = da.rio.write_nodata(np.nan)
    return da


def reproject_and_clip_one_step(
    src_2d: xr.DataArray,
    target: xr.DataArray,
    boundary: gpd.GeoDataFrame,
) -> xr.DataArray:
    """
    对单个 5 天时间步执行双线性重投影和福建边界掩膜。

    为什么按时间步处理：
    CHIRPS 单年通常约有 73 个 5 天时间步。逐时间步处理可以显示清晰进度，
    也能避免一次性重投影整个三维数组时内存占用过高。
    """
    out = src_2d.rio.reproject_match(
        target,
        resampling=Resampling.bilinear,
        nodata=np.nan,
    )
    out = out.rio.clip(
        boundary.geometry,
        boundary.crs,
        drop=False,
        all_touched=ALL_TOUCHED,
    )
    if CLIP_NEGATIVE_TO_ZERO:
        out = out.clip(min=0)
    return out


def process_one_file(
    input_file: Path,
    output_file: Path,
    boundary_shp: Path,
    boundary: gpd.GeoDataFrame,
    target: xr.DataArray,
) -> str:
    """
    处理单个 NC 文件。

    返回值：
    processed：成功处理并写出。
    skipped：输出已存在且 OVERWRITE=False，跳过。
    """
    if output_file.exists() and not OVERWRITE:
        print(f"跳过已存在文件: {output_file}")
        return "skipped"

    ds = None
    out_ds = None
    with make_bar(total=5, desc=f"当前文件 {input_file.name}", unit="步", colour=FILE_BAR_COLOR) as file_bar:
        # 步骤 1：读取并标准化当前 NC 文件。
        with make_bar(total=1, desc="  读NC", unit="项", colour=IO_BAR_COLOR) as bar:
            ds = xr.open_dataset(input_file)
            ds = normalize_chirps_dataset(ds)
            source_attrs = ds[PRECIP_VAR].attrs.copy()
            bar.update(1)
        file_bar.update(1)

        # 步骤 2：按福建边界附近范围预裁剪，减少后续重投影计算量。
        with make_bar(total=1, desc="  预裁剪", unit="项", colour=PROCESS_BAR_COLOR) as bar:
            ds = subset_to_boundary(ds, boundary)
            bar.update(1)
        file_bar.update(1)

        # 步骤 3：准备 rioxarray 空间属性。
        with make_bar(total=1, desc="  空间属性", unit="项", colour=PROCESS_BAR_COLOR) as bar:
            source = prepare_source_array(ds)
            bar.update(1)
        file_bar.update(1)

        # 步骤 4：逐 5 天时间步执行双线性重投影和边界掩膜。
        if "time" in source.dims:
            outputs = []
            time_values = source["time"].values
            with tqdm(
                time_values,
                total=len(time_values),
                desc="  双线性重采样",
                unit="期",
                colour=PROCESS_BAR_COLOR,
                dynamic_ncols=True,
                bar_format=PROGRESS_BAR_FORMAT,
            ) as time_bar:
                for time_value in time_bar:
                    src_2d = source.sel(time=time_value)
                    out_2d = reproject_and_clip_one_step(src_2d, target, boundary)
                    outputs.append(out_2d.expand_dims(time=[time_value]))
            precip_1km = xr.concat(outputs, dim="time")
            precip_1km["time"].attrs.update(source["time"].attrs)
        else:
            with make_bar(total=1, desc="  双线性重采样", unit="项", colour=PROCESS_BAR_COLOR) as bar:
                precip_1km = reproject_and_clip_one_step(source, target, boundary)
                bar.update(1)
        file_bar.update(1)

        # 步骤 5：写入元数据并保存当前文件结果。
        with make_bar(total=1, desc="  保存", unit="项", colour=SAVE_BAR_COLOR) as bar:
            precip_1km.name = PRECIP_VAR
            precip_1km.attrs.update(source_attrs)
            precip_1km.attrs.update(
                {
                    "long_name": "CHIRPS 5-day precipitation resampled to 1 km",
                    "units": source_attrs.get("units", "mm"),
                    "description": (
                        "CHIRPS 5-day precipitation bilinearly reprojected to "
                        "a strict 1000 m grid in EPSG:32650 and masked by the "
                        "Fujian administrative boundary."
                    ),
                    "original_spatial_resolution": "0.05 degree",
                    "target_spatial_resolution": "1000 m",
                    "original_temporal_resolution": "pentad / 5-day",
                    "resampling_method": "bilinear reprojection",
                    "mask_boundary": str(boundary_shp),
                }
            )

            out_ds = precip_1km.to_dataset()
            out_ds.attrs.update(
                {
                    "title": "Fujian CHIRPS 1 km 5-day precipitation",
                    "source_file": str(input_file),
                    "boundary_file": str(boundary_shp),
                    "crs": TARGET_CRS,
                    "grid_resolution_m": TARGET_RES,
                    "all_touched": str(ALL_TOUCHED),
                }
            )

            encoding = {
                PRECIP_VAR: {
                    "zlib": True,
                    "complevel": 4,
                    "dtype": "float32",
                    "_FillValue": np.float32(np.nan),
                }
            }
            output_file.parent.mkdir(parents=True, exist_ok=True)
            out_ds.to_netcdf(output_file, encoding=encoding)
            bar.update(1)
        file_bar.update(1)

    if out_ds is not None:
        out_ds.close()
    if ds is not None:
        ds.close()
    print(f"完成: {input_file.name} -> {output_file.name}")
    return "processed"


# =============================================================================
# 三、主流程
# =============================================================================
def main() -> None:
    """
    主程序入口。

    进度条层级：
    1. 初始化进度：创建目录、扫描文件、读取边界、构建目标网格。
    2. 批量文件进度：第几个 NC / 总 NC。
    3. 当前文件步骤进度：读取、裁剪、空间属性、重采样、保存。
    4. 当前文件时间步进度：第几个 5 天时间步 / 总时间步。
    """
    processed_count = 0
    skipped_count = 0
    failed_files = []

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        configure_temp_dir()

        with make_bar(total=4, desc="初始化", unit="步", colour=BATCH_BAR_COLOR) as init_bar:
            input_files = discover_input_files()
            file_pairs = validate_unique_outputs(input_files)
            init_bar.update(1)

            boundary_shp, boundary = read_boundary()
            init_bar.update(1)

            target = build_target_grid(boundary)
            init_bar.update(1)

            print(f"\n输入目录: {INPUT_DIR}")
            print(f"输出目录: {OUTPUT_DIR}")
            print(f"边界文件: {boundary_shp}")
            print(f"待处理文件数: {len(file_pairs)}")
            print(f"目标网格: {target.sizes['x']} 列 x {target.sizes['y']} 行")
            print(f"空间分辨率: {TARGET_RES:.0f} m x {TARGET_RES:.0f} m\n")
            init_bar.update(1)

        with tqdm(
            file_pairs,
            total=len(file_pairs),
            desc="批量文件进度",
            unit="个文件",
            colour=BATCH_BAR_COLOR,
            dynamic_ncols=True,
            bar_format=PROGRESS_BAR_FORMAT,
        ) as file_pairs_bar:
            for input_file, output_file in file_pairs_bar:
                file_pairs_bar.set_postfix_str(input_file.name)
                try:
                    status = process_one_file(
                        input_file=input_file,
                        output_file=output_file,
                        boundary_shp=boundary_shp,
                        boundary=boundary,
                        target=target,
                    )
                    if status == "skipped":
                        skipped_count += 1
                    else:
                        processed_count += 1
                except Exception as exc:
                    failed_files.append((input_file, exc))
                    print(f"\n处理失败: {input_file}")
                    print(f"错误信息: {exc}")

    finally:
        cleanup_temp_dir()

    print("\n批处理完成")
    print(f"成功处理: {processed_count} 个")
    print(f"跳过已有: {skipped_count} 个")
    print(f"处理失败: {len(failed_files)} 个")
    print(f"输出目录: {OUTPUT_DIR}")
    if CLEAN_TEMP_FILES:
        print(f"已清理临时目录: {TEMP_DIR}")
    else:
        print(f"保留临时目录: {TEMP_DIR}")

    if failed_files:
        print("\n失败文件列表:")
        for input_file, exc in failed_files:
            print(f"  {input_file.name}: {exc}")


if __name__ == "__main__":
    main()
