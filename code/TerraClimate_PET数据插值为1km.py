# -*- coding: utf-8 -*-
"""
批量将福建省 TerraClimate PET NetCDF 数据插值为 1 km 网格。
"""

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401  # 只需导入即可为 xarray 对象启用 .rio 空间处理方法
import xarray as xr
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from tqdm import tqdm


# =============================================================================
# 一、可调参数区
# =============================================================================
# 输入 PET NetCDF 文件所在文件夹。
# 修改意义：
# 1. 脚本会批量读取该文件夹第一层目录下的所有 .nc 文件。
# 2. 不会递归读取子文件夹，因此输出文件夹里的结果不会被再次当作输入。
INPUT_DIR = Path(
    r"E:\forest_microclimate\ForestMicroclimate\PET_TerraClimate"
    r"\fujian_PET_TerraClimate"
)

# 输出 1 km PET NetCDF 文件夹。
# 修改意义：
# 1. 按你的要求，输出文件夹建立在原始 PET 数据目录下。
# 2. 所有插值后的 1 km NetCDF 都保存到该文件夹中。
OUTPUT_DIR = INPUT_DIR / "fujian_PET_1km"

# 输入文件匹配规则。
# 修改意义：
# 1. "*.nc" 表示处理输入文件夹内所有 NetCDF 文件。
# 2. 如果后续只想处理某一类文件，可改为例如 "fujian_TerraClimate_pet_*.nc"。
INPUT_FILE_PATTERN = "*.nc"

# 是否跳过已经包含 1km 标记的 NetCDF 文件。
# 修改意义：
# 1. True 可以避免把已经插值后的文件再次插值。
# 2. 当前脚本只读取 INPUT_DIR 第一层文件，正常不会读到 OUTPUT_DIR 内文件；该参数作为额外保险。
SKIP_1KM_FILES = True

# 输出文件命名模板。
# 修改意义：
# 1. {year} 会被替换成从输入文件名中识别到的 4 位年份。
# 2. 例如输入文件名包含 1990，则输出为 fujian_PET_1990_1km.nc。
OUTPUT_FILENAME_TEMPLATE = "fujian_PET_{year}_1km.nc"

# 福建省行政边界，地理坐标系版本。
# 修改意义：
# 1. 该文件用于按经纬度范围对原始 PET 做预裁剪。
# 2. 预裁剪可以减少重投影计算量，并保留边界外少量缓冲区用于插值。
BOUNDARY_GEO_FILE = Path(
    r"E:\forest_microclimate\ForestMicroclimate\Fujian_Shp\福建省行政边界.shp"
)

# 福建省行政边界，投影坐标系版本。
# 修改意义：
# 1. 该文件用于构建严格 1000 m 目标网格和最终掩膜。
# 2. 当前投影应为 WGS 1984 UTM Zone 50N，即 EPSG:32650，坐标单位为米。
BOUNDARY_UTM_FILE = Path(
    r"E:\forest_microclimate\ForestMicroclimate\Fujian_Shp"
    r"\福建省行政边界_WGS1984UTM50N.shp"
)

# PET 变量候选名，按优先级从高到低自动识别。
# 修改意义：
# 1. 前一步裁剪脚本输出的变量名通常是 PET_mm。
# 2. TerraClimate 原始文件变量名通常是 pet。
# 3. 脚本会跳过 crs 这类空间参考辅助变量，只对真正的 PET 变量插值。
PET_VARIABLE_CANDIDATES = ("PET_mm", "pet", "PET", "pev")

# 输出变量名。
# 修改意义：
# 1. "same_as_input" 表示输出变量名与每个输入文件中识别到的 PET 变量名一致。
# 2. 如果希望所有输出统一为 PET_mm，可改为 "PET_mm"。
OUTPUT_VARIABLE_NAME = "same_as_input"

# 原始 TerraClimate PET 坐标系。
# 修改意义：
# 1. TerraClimate 通常为 WGS84 经纬度坐标，即 EPSG:4326。
# 2. 如果输入文件没有明确 CRS，脚本会按该坐标系写入空间参考。
SOURCE_CRS = "EPSG:4326"

# 目标投影坐标系。
# 修改意义：
# 1. EPSG:32650 是 WGS 84 / UTM zone 50N，适合福建区域。
# 2. 严格 1 km 网格必须在米制投影坐标系下构建，不能直接用经纬度近似。
TARGET_CRS = "EPSG:32650"

# 目标空间分辨率，单位为米。
# 修改意义：
# 1. 1000.0 表示输出像元大小为 1000 m x 1000 m。
# 2. 如果后续要改成 500 m 或 2 km，可以调整该参数。
TARGET_RESOLUTION_M = 1000.0

# 原始数据预裁剪缓冲区，单位为经纬度度。
# 修改意义：
# 1. 重投影前在福建边界外保留一圈原始网格，避免边界附近插值缺少邻近格点。
# 2. TerraClimate 原始分辨率约 1/24 度，0.25 度通常足够。
# 3. 如果边缘出现异常空值，可以适当增大该值。
BOUNDARY_PADDING_DEG = 0.25

# 重采样方法。
# 修改意义：
# 1. bilinear 表示双线性插值，适合 PET 这类连续变量。
# 2. nearest 表示最近邻，通常更适合分类数据，不建议用于 PET。
RESAMPLING_METHOD = "bilinear"

# 是否将 PET 负值修正为 0。
# 修改意义：
# 1. PET 理论上不应为负值。
# 2. True 表示插值后若出现极小负值，会修正为 0。
# 3. False 表示完全保留插值结果。
CLIP_NEGATIVE_TO_ZERO = True

# 掩膜方式。
# 修改意义：
# 1. False 表示只保留像元中心落在福建边界内的 1 km 像元，边界更保守。
# 2. True 表示只要像元接触福建边界就保留，边界覆盖更充分但可能多保留少量外部像元。
ALL_TOUCHED = False

# 如果输出文件已经存在，是否覆盖。
# 修改意义：
# 1. True 表示重新运行时会覆盖同名 1 km 结果。
# 2. False 表示遇到已有结果时跳过该年份，适合断点续跑。
OVERWRITE_EXISTING = True

# 是否启用 NetCDF 压缩。
# 修改意义：
# 1. True 可以减小输出文件体积，但写出速度会慢一些。
# 2. False 写出更快，但文件可能更大。
ENABLE_NETCDF_COMPRESSION = True

# NetCDF 压缩等级。
# 修改意义：
# 1. 取值范围通常为 1 到 9。
# 2. 数值越大压缩率越高，但保存越慢。
# 3. 仅在 ENABLE_NETCDF_COMPRESSION = True 时生效。
NETCDF_COMPRESSION_LEVEL = 4

# 输出数据类型。
# 修改意义：
# 1. float32 可以显著减小文件体积，通常足够保存 PET 数据。
# 2. 如果需要更高数值精度，可改为 float64。
OUTPUT_DTYPE = "float32"

# tqdm 进度条显示格式。
# 修改意义：
# 1. percentage 显示百分比。
# 2. n_fmt/total_fmt 显示当前数量和总数量。
# 3. elapsed/remaining/rate_fmt 显示耗时、剩余时间和速度。
PROGRESS_BAR_FORMAT = (
    "{l_bar}{bar:34}| {percentage:3.0f}% {n_fmt}/{total_fmt} "
    "[已用 {elapsed} < 剩余 {remaining}, {rate_fmt}]"
)

# 进度条颜色。
# 修改意义：
# 1. 不同类型任务使用不同颜色，便于在终端快速区分当前阶段。
# 2. tqdm 支持常见颜色名，例如 blue、green、cyan、magenta、yellow、red。
OVERALL_BAR_COLOR = "cyan"
FILE_BAR_COLOR = "yellow"
IO_BAR_COLOR = "blue"
PROCESS_BAR_COLOR = "green"
SAVE_BAR_COLOR = "magenta"
CHECK_BAR_COLOR = "yellow"


# =============================================================================
# 二、工具函数区
# =============================================================================
def make_bar(total: int, desc: str, unit: str, colour: str) -> tqdm:
    """创建统一格式的彩色 tqdm 进度条。"""
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        colour=colour,
        dynamic_ncols=True,
        bar_format=PROGRESS_BAR_FORMAT,
    )


def print_parameters() -> None:
    """打印当前关键参数，方便运行前确认和后续调整。"""
    print("\n当前可调参数如下：")
    print(f"输入 PET 文件夹: {INPUT_DIR}")
    print(f"输入文件匹配规则: {INPUT_FILE_PATTERN}")
    print(f"输出 1 km 文件夹: {OUTPUT_DIR}")
    print(f"输出命名模板: {OUTPUT_FILENAME_TEMPLATE}")
    print(f"地理坐标系边界: {BOUNDARY_GEO_FILE}")
    print(f"投影坐标系边界: {BOUNDARY_UTM_FILE}")
    print(f"PET 变量候选名: {PET_VARIABLE_CANDIDATES}")
    print(f"输出变量名规则: {OUTPUT_VARIABLE_NAME}")
    print(f"源坐标系: {SOURCE_CRS}")
    print(f"目标坐标系: {TARGET_CRS}")
    print(f"目标分辨率: {TARGET_RESOLUTION_M:.0f} m")
    print(f"预裁剪缓冲区: {BOUNDARY_PADDING_DEG} 度")
    print(f"重采样方法: {RESAMPLING_METHOD}")
    print(f"负值修正为 0: {CLIP_NEGATIVE_TO_ZERO}")
    print(f"边界 all_touched: {ALL_TOUCHED}")
    print(f"覆盖已有输出: {OVERWRITE_EXISTING}")
    print(f"NetCDF 压缩: {ENABLE_NETCDF_COMPRESSION}")
    print(f"NetCDF 压缩等级: {NETCDF_COMPRESSION_LEVEL}")
    print(f"输出数据类型: {OUTPUT_DTYPE}\n")


def check_base_paths() -> None:
    """检查输入目录和边界文件是否存在，尽早发现路径错误。"""
    missing_paths = [
        path
        for path in [INPUT_DIR, BOUNDARY_GEO_FILE, BOUNDARY_UTM_FILE]
        if not path.exists()
    ]
    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"以下输入路径不存在：\n{missing_text}")

    if not INPUT_DIR.is_dir():
        raise NotADirectoryError(f"输入路径不是文件夹：{INPUT_DIR}")


def list_input_files() -> list[Path]:
    """列出需要批量插值的 NetCDF 文件，并排除明显已经处理过的 1 km 文件。"""
    input_files = sorted(path for path in INPUT_DIR.glob(INPUT_FILE_PATTERN) if path.is_file())

    if SKIP_1KM_FILES:
        input_files = [
            path
            for path in input_files
            if "1km" not in path.stem.lower() and "1_km" not in path.stem.lower()
        ]

    if not input_files:
        raise FileNotFoundError(f"未在输入文件夹中找到待处理 NetCDF：{INPUT_DIR}")

    return input_files


def extract_year_from_filename(input_file: Path) -> str:
    """从文件名中提取 4 位年份，用于生成 fujian_PET_1990_1km.nc 这类输出名。"""
    match = re.search(r"(19\d{2}|20\d{2})", input_file.stem)
    if not match:
        raise ValueError(f"无法从文件名识别年份，请检查文件名是否包含 4 位年份：{input_file.name}")
    return match.group(1)


def build_output_file(input_file: Path) -> Path:
    """根据输入文件年份构建批量输出路径。"""
    year = extract_year_from_filename(input_file)
    return OUTPUT_DIR / OUTPUT_FILENAME_TEMPLATE.format(year=year)


def get_resampling_method() -> Resampling:
    """将字符串形式的重采样参数转换为 rasterio 可识别的枚举。"""
    methods = {
        "bilinear": Resampling.bilinear,
        "nearest": Resampling.nearest,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }
    if RESAMPLING_METHOD not in methods:
        supported = ", ".join(methods)
        raise ValueError(f"不支持的重采样方法：{RESAMPLING_METHOD}；可选值为：{supported}")
    return methods[RESAMPLING_METHOD]


def read_boundary(path: Path, expected_crs: str) -> gpd.GeoDataFrame:
    """读取边界矢量，清理空几何并统一到指定坐标系。"""
    boundary = gpd.read_file(path)
    if boundary.empty:
        raise ValueError(f"边界文件为空：{path}")

    boundary = boundary[boundary.geometry.notna() & ~boundary.geometry.is_empty].copy()
    if boundary.empty:
        raise ValueError(f"边界文件没有有效几何：{path}")

    if boundary.crs is None:
        boundary = boundary.set_crs(expected_crs, allow_override=True)
    else:
        boundary = boundary.to_crs(expected_crs)

    return boundary


def normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    """统一 NetCDF 中常见的经纬度坐标名称，方便后续空间处理。"""
    rename_map = {}
    coord_names = set(ds.coords) | set(ds.dims)

    if "longitude" in coord_names:
        rename_map["longitude"] = "lon"
    if "latitude" in coord_names:
        rename_map["latitude"] = "lat"
    if "Longitude" in coord_names:
        rename_map["Longitude"] = "lon"
    if "Latitude" in coord_names:
        rename_map["Latitude"] = "lat"

    if rename_map:
        ds = ds.rename(rename_map)

    required_coords = {"lon", "lat"}
    missing_coords = required_coords - (set(ds.coords) | set(ds.dims))
    if missing_coords:
        raise ValueError(f"缺少必要空间坐标：{sorted(missing_coords)}")

    return ds


def detect_pet_variable(ds: xr.Dataset) -> str:
    """自动识别 PET 数据变量，兼容 PET_mm 和 pet 等常见命名。"""
    for candidate in PET_VARIABLE_CANDIDATES:
        if candidate in ds.data_vars:
            return candidate

    valid_vars = [
        name
        for name, data_array in ds.data_vars.items()
        if name.lower() != "crs" and {"lon", "lat"}.issubset(set(data_array.dims))
    ]
    if len(valid_vars) == 1:
        return valid_vars[0]

    raise ValueError(
        "无法自动识别 PET 变量；"
        f"候选名为 {PET_VARIABLE_CANDIDATES}，当前可用变量为 {list(ds.data_vars)}"
    )


def get_output_variable_name(input_variable_name: str) -> str:
    """确定输出变量名，支持保持输入变量名或统一指定名称。"""
    if OUTPUT_VARIABLE_NAME == "same_as_input":
        return input_variable_name
    return OUTPUT_VARIABLE_NAME


def subset_to_boundary(
    ds: xr.Dataset,
    pet_var: str,
    boundary_geo: gpd.GeoDataFrame,
) -> xr.Dataset:
    """按福建边界外扩范围预裁剪原始经纬度数据，减少重投影计算量。"""
    min_lon, min_lat, max_lon, max_lat = boundary_geo.total_bounds
    min_lon -= BOUNDARY_PADDING_DEG
    max_lon += BOUNDARY_PADDING_DEG
    min_lat -= BOUNDARY_PADDING_DEG
    max_lat += BOUNDARY_PADDING_DEG

    lat_values = ds["lat"].values
    lat_slice = (
        slice(min_lat, max_lat)
        if lat_values[0] < lat_values[-1]
        else slice(max_lat, min_lat)
    )

    lon_values = ds["lon"].values
    lon_slice = (
        slice(min_lon, max_lon)
        if lon_values[0] < lon_values[-1]
        else slice(max_lon, min_lon)
    )

    subset = ds.sel(lon=lon_slice, lat=lat_slice)
    if subset[pet_var].sizes.get("lon", 0) == 0 or subset[pet_var].sizes.get("lat", 0) == 0:
        raise ValueError("预裁剪后没有剩余格点，请检查输入 PET 范围或福建边界坐标系。")
    return subset


def build_target_grid(boundary_utm: gpd.GeoDataFrame) -> xr.DataArray:
    """根据福建省投影边界构建严格 1000 m 目标网格。"""
    min_x, min_y, max_x, max_y = boundary_utm.total_bounds

    left = np.floor(min_x / TARGET_RESOLUTION_M) * TARGET_RESOLUTION_M
    right = np.ceil(max_x / TARGET_RESOLUTION_M) * TARGET_RESOLUTION_M
    bottom = np.floor(min_y / TARGET_RESOLUTION_M) * TARGET_RESOLUTION_M
    top = np.ceil(max_y / TARGET_RESOLUTION_M) * TARGET_RESOLUTION_M

    x = np.arange(left + TARGET_RESOLUTION_M / 2, right, TARGET_RESOLUTION_M)
    y = np.arange(top - TARGET_RESOLUTION_M / 2, bottom, -TARGET_RESOLUTION_M)

    if len(x) == 0 or len(y) == 0:
        raise ValueError("目标 1 km 网格为空，请检查投影边界范围。")

    target = xr.DataArray(
        np.zeros((len(y), len(x)), dtype=np.float32),
        coords={"y": y, "x": x},
        dims=("y", "x"),
        name="target_grid",
    )
    target = target.rio.write_crs(TARGET_CRS)
    target = target.rio.write_transform(
        from_origin(left, top, TARGET_RESOLUTION_M, TARGET_RESOLUTION_M)
    )
    return target


def prepare_source_array(ds: xr.Dataset, pet_var: str) -> xr.DataArray:
    """把 PET 变量整理成 rioxarray 可重投影的空间 DataArray。"""
    pet = ds[pet_var]
    pet_attrs = pet.attrs.copy()

    pet = pet.sortby("lat", ascending=False).rename({"lon": "x", "lat": "y"})
    pet = pet.rio.set_spatial_dims(x_dim="x", y_dim="y")
    pet = pet.rio.write_crs(SOURCE_CRS)
    pet = pet.rio.write_nodata(np.nan)
    pet.attrs.update(pet_attrs)
    return pet


def reproject_and_mask_one_slice(
    source_2d: xr.DataArray,
    target_grid: xr.DataArray,
    boundary_utm: gpd.GeoDataFrame,
) -> xr.DataArray:
    """对单个时间片执行双线性重投影、福建边界掩膜和负值修正。"""
    output = source_2d.rio.reproject_match(
        target_grid,
        resampling=get_resampling_method(),
        nodata=np.nan,
    )
    output = output.rio.clip(
        boundary_utm.geometry,
        boundary_utm.crs,
        drop=False,
        all_touched=ALL_TOUCHED,
    )

    if CLIP_NEGATIVE_TO_ZERO:
        output = output.clip(min=0)

    return output


def reproject_all_slices(
    source: xr.DataArray,
    target_grid: xr.DataArray,
    boundary_utm: gpd.GeoDataFrame,
    progress_desc: str,
) -> xr.DataArray:
    """逐时间片插值 PET，避免一次性重投影三维数组造成内存压力。"""
    if "time" not in source.dims:
        with make_bar(1, f"{progress_desc} 插值掩膜", "层", PROCESS_BAR_COLOR) as bar:
            output = reproject_and_mask_one_slice(source, target_grid, boundary_utm)
            bar.update(1)
        return output

    outputs = []
    time_values = source["time"].values

    with tqdm(
        time_values,
        total=len(time_values),
        desc=f"{progress_desc} 逐时间片插值",
        unit="片",
        colour=PROCESS_BAR_COLOR,
        dynamic_ncols=True,
        bar_format=PROGRESS_BAR_FORMAT,
    ) as time_bar:
        for time_value in time_bar:
            source_2d = source.sel(time=time_value)
            output_2d = reproject_and_mask_one_slice(source_2d, target_grid, boundary_utm)
            outputs.append(output_2d.expand_dims(time=[time_value]))

    output = xr.concat(outputs, dim="time")
    output["time"].attrs.update(source["time"].attrs)
    return output


def remove_conflicting_encoding_attrs(data_array: xr.DataArray) -> xr.DataArray:
    """移除可能与 to_netcdf encoding 冲突的属性。"""
    cleaned = data_array.copy()
    for attr_name in ["_FillValue", "missing_value"]:
        cleaned.attrs.pop(attr_name, None)
    return cleaned


def build_output_dataset(
    pet_1km: xr.DataArray,
    source_attrs: dict,
    input_file: Path,
    output_var_name: str,
) -> xr.Dataset:
    """整理输出 Dataset 的变量名、属性和全局元数据。"""
    pet_1km = remove_conflicting_encoding_attrs(pet_1km)
    pet_1km.name = output_var_name
    pet_1km.attrs.update(source_attrs)
    pet_1km.attrs.update(
        {
            "long_name": "TerraClimate PET resampled to 1 km over Fujian",
            "units": source_attrs.get("units", "mm"),
            "description": (
                "TerraClimate PET bilinearly reprojected to a strict 1000 m "
                "grid in EPSG:32650 and masked by the Fujian administrative boundary."
            ),
            "source_crs": SOURCE_CRS,
            "target_crs": TARGET_CRS,
            "target_spatial_resolution": f"{TARGET_RESOLUTION_M:.0f} m",
            "resampling_method": RESAMPLING_METHOD,
            "mask_boundary": str(BOUNDARY_UTM_FILE),
        }
    )
    pet_1km = remove_conflicting_encoding_attrs(pet_1km)

    output_ds = pet_1km.to_dataset()
    output_ds.attrs.update(
        {
            "title": "Fujian TerraClimate PET 1 km",
            "source_file": str(input_file),
            "boundary_geo_file": str(BOUNDARY_GEO_FILE),
            "boundary_utm_file": str(BOUNDARY_UTM_FILE),
            "crs": TARGET_CRS,
            "grid_resolution_m": TARGET_RESOLUTION_M,
            "all_touched": str(ALL_TOUCHED),
        }
    )
    return output_ds


def build_netcdf_encoding(output_ds: xr.Dataset, output_var_name: str) -> dict:
    """生成 NetCDF 写出编码参数，控制压缩、数据类型和空值。"""
    encoding = {
        output_var_name: {
            "dtype": OUTPUT_DTYPE,
            "_FillValue": np.float32(np.nan) if OUTPUT_DTYPE == "float32" else np.nan,
        }
    }

    if ENABLE_NETCDF_COMPRESSION:
        encoding[output_var_name].update(
            {
                "zlib": True,
                "complevel": NETCDF_COMPRESSION_LEVEL,
            }
        )

    for coord_name in output_ds.coords:
        output_ds[coord_name].attrs.pop("_FillValue", None)

    return encoding


def process_one_file(
    input_file: Path,
    output_file: Path,
    boundary_geo: gpd.GeoDataFrame,
    boundary_utm: gpd.GeoDataFrame,
    target_grid: xr.DataArray,
) -> None:
    """处理单个年份 PET 文件：读取、识别变量、插值、掩膜并保存 NetCDF。"""
    if output_file.exists() and not OVERWRITE_EXISTING:
        print(f"已存在，跳过: {output_file}")
        return

    print(f"\n开始处理: {input_file.name}")
    print(f"输出文件: {output_file}")

    ds = None
    try:
        with make_bar(5, "单文件进度", "步", OVERALL_BAR_COLOR) as file_bar:
            # -----------------------------------------------------------------
            # 步骤 1：打开单个 NetCDF，统一坐标名并自动识别 PET 变量。
            # -----------------------------------------------------------------
            with make_bar(1, "步骤1 读取PET", "项", IO_BAR_COLOR) as bar:
                ds = xr.open_dataset(input_file)
                ds = normalize_dataset(ds)
                pet_var = detect_pet_variable(ds)
                output_var_name = get_output_variable_name(pet_var)
                source_attrs = ds[pet_var].attrs.copy()
                print(f"识别 PET 变量: {pet_var}")
                print(f"输出变量名: {output_var_name}")
                print(f"原始数据维度: {dict(ds[pet_var].sizes)}")
                bar.update(1)
            file_bar.update(1)

            # -----------------------------------------------------------------
            # 步骤 2：按福建省经纬度范围外扩后预裁剪，降低插值计算量。
            # -----------------------------------------------------------------
            with make_bar(1, "步骤2 预裁剪", "项", PROCESS_BAR_COLOR) as bar:
                ds = subset_to_boundary(ds, pet_var, boundary_geo)
                print(f"预裁剪后维度: {dict(ds[pet_var].sizes)}")
                bar.update(1)
            file_bar.update(1)

            # -----------------------------------------------------------------
            # 步骤 3：把 PET 整理成带 CRS、nodata 和空间维度的 DataArray。
            # -----------------------------------------------------------------
            with make_bar(1, "步骤3 准备空间数组", "项", PROCESS_BAR_COLOR) as bar:
                source = prepare_source_array(ds, pet_var)
                bar.update(1)
            file_bar.update(1)

            # -----------------------------------------------------------------
            # 步骤 4：逐时间片执行双线性重投影和福建省边界掩膜。
            # -----------------------------------------------------------------
            pet_1km = reproject_all_slices(source, target_grid, boundary_utm, "步骤4")
            file_bar.update(1)

            # -----------------------------------------------------------------
            # 步骤 5：整理属性并保存为该年份的 1 km NetCDF。
            # -----------------------------------------------------------------
            with make_bar(1, "步骤5 保存NetCDF", "项", SAVE_BAR_COLOR) as bar:
                output_ds = build_output_dataset(
                    pet_1km,
                    source_attrs,
                    input_file,
                    output_var_name,
                )
                encoding = build_netcdf_encoding(output_ds, output_var_name)
                output_ds.to_netcdf(output_file, encoding=encoding)
                print(f"输出维度: {dict(output_ds.sizes)}")
                bar.update(1)
            file_bar.update(1)
    finally:
        if ds is not None:
            ds.close()


# =============================================================================
# 三、主流程
# =============================================================================
def main() -> None:
    """批量执行福建省 TerraClimate PET 插值到 1 km 网格的完整流程。"""
    print_parameters()

    # -------------------------------------------------------------------------
    # 批量步骤 1：检查基础路径、创建输出目录、扫描待处理 NetCDF 文件。
    # -------------------------------------------------------------------------
    with make_bar(3, "初始化", "步", CHECK_BAR_COLOR) as init_bar:
        check_base_paths()
        init_bar.update(1)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        init_bar.update(1)

        input_files = list_input_files()
        init_bar.update(1)

    print(f"发现待处理 NetCDF 数量: {len(input_files)}")
    for input_file in input_files:
        print(f"待处理文件: {input_file.name}")

    # -------------------------------------------------------------------------
    # 批量步骤 2：读取边界并构建目标 1 km 网格。该步骤只做一次，所有年份复用。
    # -------------------------------------------------------------------------
    with make_bar(3, "准备公共空间数据", "项", IO_BAR_COLOR) as setup_bar:
        boundary_geo = read_boundary(BOUNDARY_GEO_FILE, SOURCE_CRS)
        setup_bar.update(1)

        boundary_utm = read_boundary(BOUNDARY_UTM_FILE, TARGET_CRS)
        setup_bar.update(1)

        target_grid = build_target_grid(boundary_utm)
        print(f"目标网格大小: x={target_grid.sizes['x']}, y={target_grid.sizes['y']}")
        setup_bar.update(1)

    # -------------------------------------------------------------------------
    # 批量步骤 3：逐文件插值。外层进度条显示文件级进度，内层进度条显示单文件关键步骤。
    # -------------------------------------------------------------------------
    processed_count = 0
    skipped_count = 0

    with tqdm(
        input_files,
        total=len(input_files),
        desc="批量文件进度",
        unit="个文件",
        colour=FILE_BAR_COLOR,
        dynamic_ncols=True,
        bar_format=PROGRESS_BAR_FORMAT,
    ) as file_progress:
        for input_file in file_progress:
            output_file = build_output_file(input_file)
            if output_file.exists() and not OVERWRITE_EXISTING:
                skipped_count += 1
                print(f"已存在，跳过: {output_file}")
                continue

            process_one_file(
                input_file=input_file,
                output_file=output_file,
                boundary_geo=boundary_geo,
                boundary_utm=boundary_utm,
                target_grid=target_grid,
            )
            processed_count += 1

    print("\n全部处理完成")
    print(f"输入文件夹: {INPUT_DIR}")
    print(f"输出文件夹: {OUTPUT_DIR}")
    print(f"成功处理文件数: {processed_count}")
    print(f"跳过文件数: {skipped_count}")
    print(f"空间分辨率: {TARGET_RESOLUTION_M:.0f} m x {TARGET_RESOLUTION_M:.0f} m")


if __name__ == "__main__":
    main()
