# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from rasterio import features
from rasterio.transform import from_bounds
from tqdm import tqdm


@dataclass(frozen=True)
class Config:
    # =========================
    # 1. 输入、输出路径参数
    # =========================
    # source_dir：ERA5-Land 逐月 NC 数据所在目录。
    # 每个 NC 文件内部保存该月逐小时 2 米气温数据；脚本只读取原始 NC，不修改原始文件。
    source_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\T2m")

    # shapefile_path：福建省行政边界 shp 文件路径。
    # 裁剪时使用该边界生成栅格掩膜，边界外像元会被写成缺失值。
    shapefile_path: Path = Path(
        r"E:\forest_microclimate\ForestMicroclimate\Fujian_Shp\福建省行政边界.shp"
    )

    # output_folder_name：输出文件夹名称。
    # 按要求在 source_dir 下创建 fujian_T2，所有裁剪后的 NC 和统计表都保存到这里。
    output_folder_name: str = "fujian_T2"

    # input_pattern：待处理 NC 文件匹配规则。
    # 当前文件示例为 T2m_199001_hourly.nc；如果以后命名改变，可在这里调整。
    input_pattern: str = "*.nc"

    # output_name_template：裁剪后 NC 文件命名模板。
    # {yyyymm} 会从原始文件名或时间坐标中解析，例如输出 fujian_T2m_202001_hourly.nc。
    output_name_template: str = "fujian_T2m_{yyyymm}_hourly.nc"

    # =========================
    # 2. 变量、维度自动识别参数
    # =========================
    # preferred_temperature_names：优先识别为 2 米气温的变量名。
    # 不确定 NC 变量名时，脚本会先按这些名称查找，再根据维度和单位自动判断。
    preferred_temperature_names: tuple[str, ...] = (
        "t2m",
        "T2m",
        "T2M",
        "2t",
        "temperature_2m",
        "air_temperature",
    )

    # preferred_time_names：优先识别为时间坐标或时间维度的名称。
    # ERA5/ERA5-Land 常见为 time 或 valid_time。
    preferred_time_names: tuple[str, ...] = ("time", "valid_time")

    # preferred_lat_names / preferred_lon_names：优先识别为纬度、经度坐标的名称。
    # ERA5-Land 常见为 latitude、longitude，也可能简写为 lat、lon。
    preferred_lat_names: tuple[str, ...] = ("latitude", "lat", "y")
    preferred_lon_names: tuple[str, ...] = ("longitude", "lon", "x")

    # expected_spatial_crs：ERA5-Land 经纬度网格常用 EPSG:4326。
    # 如果 NC 中没有明确 CRS，脚本默认把经纬度坐标按 EPSG:4326 处理，并把福建省 shp 转到该坐标系。
    expected_spatial_crs: str = "EPSG:4326"

    # =========================
    # 3. 裁剪和单位转换参数
    # =========================
    # kelvin_to_celsius_offset：ERA5-Land t2m 原始单位为 Kelvin。
    # 输出时统一转换为摄氏度：℃ = K - 273.15。
    kelvin_to_celsius_offset: float = 273.15

    # output_temperature_units：输出 NC 中气温变量的单位属性。
    # 数据值已经转换为摄氏度，因此单位写为 degree_Celsius。
    output_temperature_units: str = "degree_Celsius"

    # output_missing_value：输出 NC 中边界外像元和原始缺失值统一使用的缺失值。
    # NetCDF 里会写入 _FillValue 和 missing_value 属性，便于后续软件识别。
    output_missing_value: float = -9999.0

    # output_dtype：输出气温变量的数据类型。
    # float32 足够保存摄氏度小数，文件体积也比 float64 小。
    output_dtype: str = "float32"

    # all_touched：边界像元裁剪规则。
    # False 表示只保留像元中心落在福建边界内的像元；True 表示所有接触边界的像元都保留。
    all_touched: bool = False

    # =========================
    # 4. NetCDF 写出参数
    # =========================
    # netcdf_engine：写出 NC 时使用的 xarray 后端。
    # h5netcdf/netcdf4 支持压缩；如果指定引擎不可用，脚本会自动尝试备用引擎。
    netcdf_engine: str = "h5netcdf"

    # fallback_netcdf_engines：当 netcdf_engine 不可用时依次尝试的备用引擎。
    # None 表示使用 xarray 默认引擎；这样可以提高不同 Python 环境下的兼容性。
    fallback_netcdf_engines: tuple[str | None, ...] = ("netcdf4", None)

    # compression_level：NC 压缩等级。
    # 0 表示不压缩，1-9 表示压缩等级逐渐增高；4 通常兼顾速度和体积。
    compression_level: int = 4

    # =========================
    # 5. 进度条和统计表参数
    # =========================
    # tqdm_ncols：进度条宽度。
    # 终端显示换行时可调小；窗口较宽时可调大。
    tqdm_ncols: int = 110

    # tqdm_colour_step / tqdm_colour_file：进度条颜色。
    # 颜色只影响终端显示，不影响裁剪计算。
    tqdm_colour_step: str = "cyan"
    tqdm_colour_file: str = "green"

    # csv_encoding：中文统计表编码。
    # utf-8-sig 便于 Excel 或 WPS 直接打开中文表头。
    csv_encoding: str = "utf-8-sig"

    # csv_newline：Windows 下写 CSV 时保持 newline=""，避免出现空行。
    csv_newline: str = ""


CONFIG = Config()


def build_output_dir(config: Config) -> Path:
    """生成输出目录路径，固定为原始 NC 数据目录下的 fujian_T2。"""
    return config.source_dir / config.output_folder_name


def print_runtime_parameters(config: Config, output_dir: Path) -> None:
    """打印本次运行的关键参数，便于运行前核对输入、输出、单位转换和裁剪设置。"""
    print("运行参数")
    print(f"原始 NC 目录: {config.source_dir}")
    print(f"福建省边界 shp: {config.shapefile_path}")
    print(f"输出目录: {output_dir}")
    print(f"输入文件匹配规则: {config.input_pattern}")
    print(f"输出命名模板: {config.output_name_template}")
    print(f"温度单位转换: 摄氏度 = Kelvin - {config.kelvin_to_celsius_offset}")
    print(f"输出缺失值: {config.output_missing_value}")
    print(f"边界像元规则 all_touched: {config.all_touched}")
    print(f"NC 写出引擎: {config.netcdf_engine}")
    print("")


def list_source_nc_files(config: Config) -> list[Path]:
    """
    查找待处理 NC 文件。
    会排除输出目录 fujian_T2 内的 NC，避免脚本重复处理自己已经生成的结果。
    """
    output_dir = build_output_dir(config).resolve()
    nc_paths = []
    for nc_path in sorted(config.source_dir.glob(config.input_pattern)):
        if not nc_path.is_file():
            continue
        if output_dir in nc_path.resolve().parents:
            continue
        nc_paths.append(nc_path)

    if not nc_paths:
        raise FileNotFoundError(f"没有在目录中找到 NC 文件: {config.source_dir}")

    return nc_paths


def validate_inputs(config: Config, output_dir: Path) -> list[Path]:
    """检查输入目录、shp 和 NC 文件是否存在，并创建输出目录。"""
    if not config.source_dir.exists():
        raise FileNotFoundError(f"原始 NC 目录不存在: {config.source_dir}")
    if not config.shapefile_path.exists():
        raise FileNotFoundError(f"福建省边界 shp 不存在: {config.shapefile_path}")

    nc_paths = list_source_nc_files(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    return nc_paths


def open_dataset(nc_path: Path) -> xr.Dataset:
    """
    打开单个 NC 文件。
    chunks=None 表示不启用并行或 dask 分块计算，后续按一个文件一个文件顺序处理。
    """
    return xr.open_dataset(nc_path, chunks=None)


def find_first_existing_name(candidates: tuple[str, ...], names: set[str]) -> str | None:
    """按候选名称顺序查找第一个存在的变量、坐标或维度名。"""
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def detect_coordinate_name(
    ds: xr.Dataset,
    candidates: tuple[str, ...],
    axis_keywords: tuple[str, ...],
    target_dims: tuple[str, ...] | None = None,
) -> str:
    """
    自动识别经度或纬度坐标名。
    优先按常见名称查找，再根据 standard_name、axis、units 等属性辅助判断。
    """
    all_names = set(ds.coords) | set(ds.variables) | set(ds.dims)
    preferred = find_first_existing_name(candidates, all_names)
    if preferred is not None:
        return preferred

    for name in all_names:
        variable = ds[name] if name in ds.variables else None
        attrs = {key.lower(): str(value).lower() for key, value in (variable.attrs if variable is not None else {}).items()}
        text = " ".join([name.lower(), *attrs.values()])
        if all(keyword in text for keyword in axis_keywords):
            if target_dims is None or name in target_dims or any(dim in target_dims for dim in getattr(variable, "dims", ())):
                return name

    raise ValueError(
        f"无法自动识别坐标名，候选名称为 {candidates}。"
        "请检查 NC 中经纬度变量是否为 latitude/longitude 或 lat/lon。"
    )


def detect_time_name(ds: xr.Dataset, config: Config) -> str:
    """自动识别时间坐标或时间维度名称，优先使用 time、valid_time。"""
    all_names = set(ds.coords) | set(ds.variables) | set(ds.dims)
    preferred = find_first_existing_name(config.preferred_time_names, all_names)
    if preferred is not None:
        return preferred

    for name in all_names:
        variable = ds[name] if name in ds.variables else None
        attrs = {key.lower(): str(value).lower() for key, value in (variable.attrs if variable is not None else {}).items()}
        text = " ".join([name.lower(), *attrs.values()])
        if "time" in text:
            return name

    raise ValueError("无法自动识别时间维度或时间坐标，请检查 NC 中是否包含 time 或 valid_time。")


def data_var_score(data_array: xr.DataArray, var_name: str, config: Config) -> int:
    """
    给候选气温变量打分。
    名称像 t2m、单位像 Kelvin、维度包含时间和空间的变量会获得更高分。
    """
    score = 0
    lower_name = var_name.lower()
    attrs_text = " ".join(str(value).lower() for value in data_array.attrs.values())
    dims_text = " ".join(dim.lower() for dim in data_array.dims)

    if var_name in config.preferred_temperature_names or lower_name in [name.lower() for name in config.preferred_temperature_names]:
        score += 100
    if "t2m" in lower_name or "2m" in lower_name:
        score += 40
    if "kelvin" in attrs_text or attrs_text.strip() == "k" or " k" in attrs_text:
        score += 30
    if "temperature" in lower_name or "temperature" in attrs_text:
        score += 20
    if any(word in dims_text for word in ("time", "valid_time")):
        score += 10
    if any(word in dims_text for word in ("lat", "latitude", "y")) and any(word in dims_text for word in ("lon", "longitude", "x")):
        score += 10

    return score


def detect_temperature_var(ds: xr.Dataset, config: Config) -> str:
    """
    自动识别 2 米气温变量名。
    如果 NC 中变量名不是 t2m，也会根据单位、属性和维度推断最可能的温度变量。
    """
    if not ds.data_vars:
        raise ValueError("NC 文件中没有数据变量，无法裁剪。")

    for preferred_name in config.preferred_temperature_names:
        if preferred_name in ds.data_vars:
            return preferred_name

    scored_vars = sorted(
        ((data_var_score(ds[var_name], var_name, config), var_name) for var_name in ds.data_vars),
        reverse=True,
    )
    best_score, best_name = scored_vars[0]
    if best_score <= 0:
        raise ValueError(
            f"无法自动识别 2 米气温变量。当前数据变量为: {', '.join(ds.data_vars)}。"
            "请在 Config.preferred_temperature_names 中添加正确变量名。"
        )

    return best_name


def ensure_1d_spatial_coordinates(ds: xr.Dataset, lat_name: str, lon_name: str) -> None:
    """
    确认经纬度坐标是一维规则网格。
    ERA5-Land 常规 NC 是一维 latitude 和 longitude；如果是二维曲线网格，需要另写裁剪逻辑。
    """
    if ds[lat_name].ndim != 1 or ds[lon_name].ndim != 1:
        raise ValueError(
            "当前脚本支持一维 latitude/longitude 规则网格。"
            f"检测到 {lat_name}.ndim={ds[lat_name].ndim}, {lon_name}.ndim={ds[lon_name].ndim}。"
        )


def load_boundary(config: Config) -> gpd.GeoDataFrame:
    """
    读取福建省边界，并转换到 ERA5-Land 经纬度坐标系。
    如果 NC 缺少 CRS，脚本按经纬度网格 EPSG:4326 处理。
    """
    boundary = gpd.read_file(config.shapefile_path)
    boundary = boundary[boundary.geometry.notna() & ~boundary.geometry.is_empty].copy()
    if boundary.empty:
        raise ValueError("福建省边界 shp 没有可用几何，请检查 shp 文件。")
    if boundary.crs is None:
        raise ValueError("福建省边界 shp 缺少坐标系信息，请先为 shp 定义正确 CRS。")

    return boundary.to_crs(config.expected_spatial_crs)


def crop_dataset_to_boundary_bounds(
    ds: xr.Dataset,
    boundary: gpd.GeoDataFrame,
    lat_name: str,
    lon_name: str,
) -> xr.Dataset:
    """
    先按福建省边界外接矩形裁出较小 NC 范围。
    这一步减少后续掩膜计算和写出文件体积；真正边界外像元会在下一步精确置为缺失值。
    """
    min_lon, min_lat, max_lon, max_lat = boundary.total_bounds
    lat_values = ds[lat_name].values
    lon_values = ds[lon_name].values

    lat_index = np.where((lat_values >= min_lat) & (lat_values <= max_lat))[0]
    lon_index = np.where((lon_values >= min_lon) & (lon_values <= max_lon))[0]
    if lat_index.size == 0 or lon_index.size == 0:
        raise ValueError("福建省边界范围与 NC 经纬度范围没有交集，请检查坐标系或数据范围。")

    return ds.isel(
        {
            ds[lat_name].dims[0]: lat_index,
            ds[lon_name].dims[0]: lon_index,
        }
    )


def normalize_spatial_order(ds: xr.Dataset, lat_name: str, lon_name: str) -> xr.Dataset:
    """
    统一空间坐标顺序，确保掩膜行列方向与数据数组一致。
    纬度统一为从北到南，经度统一为从西到东；ERA5-Land 通常本来就是这种纬度方向。
    """
    output_ds = ds
    lat_values = output_ds[lat_name].values
    lon_values = output_ds[lon_name].values

    if lat_values[0] < lat_values[-1]:
        output_ds = output_ds.sortby(lat_name, ascending=False)
    if lon_values[0] > lon_values[-1]:
        output_ds = output_ds.sortby(lon_name, ascending=True)

    return output_ds


def build_spatial_mask(
    ds: xr.Dataset,
    boundary: gpd.GeoDataFrame,
    lat_name: str,
    lon_name: str,
    config: Config,
) -> xr.DataArray:
    """
    根据福建省边界生成二维布尔掩膜。
    True 表示福建省边界内像元，False 表示边界外像元。
    """
    lat_values = ds[lat_name].values
    lon_values = ds[lon_name].values
    height = len(lat_values)
    width = len(lon_values)

    # 这里使用像元边界而不是像元中心生成仿射变换，避免掩膜整体偏移半个像元。
    lon_resolution = float(np.median(np.abs(np.diff(lon_values)))) if width > 1 else 0.1
    lat_resolution = float(np.median(np.abs(np.diff(lat_values)))) if height > 1 else 0.1
    west = float(np.min(lon_values) - lon_resolution / 2)
    east = float(np.max(lon_values) + lon_resolution / 2)
    south = float(np.min(lat_values) - lat_resolution / 2)
    north = float(np.max(lat_values) + lat_resolution / 2)
    transform = from_bounds(west, south, east, north, width, height)

    geometry_mask = features.geometry_mask(
        [geometry.__geo_interface__ for geometry in boundary.geometry],
        out_shape=(height, width),
        transform=transform,
        invert=True,
        all_touched=config.all_touched,
    )

    lat_dim = ds[lat_name].dims[0]
    lon_dim = ds[lon_name].dims[0]
    return xr.DataArray(
        geometry_mask,
        dims=(lat_dim, lon_dim),
        coords={lat_dim: ds[lat_name], lon_dim: ds[lon_name]},
    )


def parse_yyyymm_from_filename(nc_path: Path) -> str | None:
    """优先从文件名中解析 YYYYMM，用于输出 fujian_T2m_YYYYMM_hourly.nc。"""
    match = re.search(r"(19|20)\d{2}(0[1-9]|1[0-2])", nc_path.stem)
    if match is None:
        return None
    return match.group(0)


def parse_yyyymm_from_time(ds: xr.Dataset, time_name: str) -> str:
    """当文件名没有 YYYYMM 时，从时间坐标第一个时间值解析年月。"""
    time_values = ds[time_name].values
    if time_values.size == 0:
        raise ValueError("时间坐标为空，无法生成输出文件名。")
    first_time = np.asarray(time_values).reshape(-1)[0]
    timestamp = np.datetime_as_string(first_time, unit="M")
    return timestamp.replace("-", "")


def build_output_name(nc_path: Path, ds: xr.Dataset, time_name: str, config: Config) -> str:
    """按 fujian_T2m_YYYYMM_hourly.nc 规则生成裁剪后 NC 文件名。"""
    yyyymm = parse_yyyymm_from_filename(nc_path)
    if yyyymm is None:
        yyyymm = parse_yyyymm_from_time(ds, time_name)
    return config.output_name_template.format(yyyymm=yyyymm)


def convert_temperature_to_celsius(
    ds: xr.Dataset,
    temperature_var: str,
    spatial_mask: xr.DataArray,
    config: Config,
) -> xr.Dataset:
    """
    将 2 米气温变量从 Kelvin 转换为摄氏度，并把福建省边界外像元设置为缺失值。
    其他变量保持原样裁剪到同一外接矩形范围，避免破坏原始 NC 的辅助坐标信息。
    """
    output_ds = ds.copy()
    celsius = (output_ds[temperature_var].astype(config.output_dtype) - config.kelvin_to_celsius_offset).where(
        spatial_mask
    )
    celsius = celsius.fillna(config.output_missing_value).astype(config.output_dtype)

    attrs = dict(output_ds[temperature_var].attrs)
    attrs["units"] = config.output_temperature_units
    attrs["long_name"] = "福建省2米气温"
    attrs["处理说明"] = "由ERA5-Land t2m裁剪福建省边界后，从Kelvin转换为摄氏度"
    attrs["missing_value"] = config.output_missing_value
    output_ds[temperature_var] = celsius
    output_ds[temperature_var].attrs = attrs
    output_ds.attrs = dict(output_ds.attrs)
    output_ds.attrs["处理说明"] = "按福建省行政边界裁剪，边界外像元设为缺失值，2米气温单位转换为摄氏度"
    output_ds.attrs["空间范围"] = "福建省"
    return output_ds


def build_encoding(ds: xr.Dataset, temperature_var: str, config: Config) -> dict[str, dict]:
    """
    为写出 NC 构造编码参数。
    只对气温变量设置缺失值、数据类型和压缩；坐标变量保持 xarray 默认写出方式。
    """
    encoding: dict[str, dict] = {}
    for var_name in ds.data_vars:
        if var_name == temperature_var:
            encoding[var_name] = {
                "dtype": config.output_dtype,
                "_FillValue": config.output_missing_value,
                "zlib": True,
                "complevel": config.compression_level,
            }
        else:
            encoding[var_name] = {
                "zlib": True,
                "complevel": config.compression_level,
            }
    return encoding


def write_netcdf_with_fallback(
    ds: xr.Dataset,
    output_path: Path,
    encoding: dict[str, dict],
    config: Config,
) -> str:
    """
    写出裁剪后的 NC，并在指定引擎不可用时自动尝试备用引擎。
    返回实际使用的引擎名称，便于写入统计表核对运行环境。
    """
    engines = (config.netcdf_engine, *config.fallback_netcdf_engines)
    errors: list[str] = []

    for engine in engines:
        try:
            if engine is None:
                ds.to_netcdf(output_path, encoding=encoding)
                return "xarray默认引擎"
            ds.to_netcdf(output_path, engine=engine, encoding=encoding)
            return engine
        except Exception as error:
            errors.append(f"{engine or 'xarray默认引擎'}: {error}")

    raise RuntimeError("所有 NetCDF 写出引擎均失败：\n" + "\n".join(errors))


def clip_one_nc(
    nc_path: Path,
    output_dir: Path,
    boundary: gpd.GeoDataFrame,
    config: Config,
) -> dict[str, str]:
    """
    裁剪单个逐月 NC 文件。
    该函数只处理一张 NC，主循环会按文件顺序逐张调用，因此不会进行并行运算。
    """
    with open_dataset(nc_path) as ds:
        temperature_var = detect_temperature_var(ds, config)
        time_name = detect_time_name(ds, config)
        lat_name = detect_coordinate_name(ds, config.preferred_lat_names, ("lat",), ds[temperature_var].dims)
        lon_name = detect_coordinate_name(ds, config.preferred_lon_names, ("lon",), ds[temperature_var].dims)
        ensure_1d_spatial_coordinates(ds, lat_name, lon_name)

        cropped_ds = crop_dataset_to_boundary_bounds(ds, boundary, lat_name, lon_name)
        cropped_ds = normalize_spatial_order(cropped_ds, lat_name, lon_name)
        spatial_mask = build_spatial_mask(cropped_ds, boundary, lat_name, lon_name, config)
        output_ds = convert_temperature_to_celsius(cropped_ds, temperature_var, spatial_mask, config)
        output_name = build_output_name(nc_path, output_ds, time_name, config)
        output_path = output_dir / output_name
        encoding = build_encoding(output_ds, temperature_var, config)
        used_engine = write_netcdf_with_fallback(output_ds, output_path, encoding, config)

        valid_data = output_ds[temperature_var].where(output_ds[temperature_var] != config.output_missing_value)
        valid_count = int(valid_data.count().item())
        mean_value = float(valid_data.mean(skipna=True).item()) if valid_count else np.nan

    return {
        "原始文件": nc_path.name,
        "输出文件": output_name,
        "气温变量名": temperature_var,
        "时间坐标名": time_name,
        "纬度坐标名": lat_name,
        "经度坐标名": lon_name,
        "NC写出引擎": used_engine,
        "有效气温值数量": str(valid_count),
        "平均气温（℃）": format_float(mean_value),
    }


def format_float(value: float) -> str:
    """统一格式化统计表中的浮点数，空值或无效值写为空字符串。"""
    if np.isnan(value):
        return ""
    return f"{value:.3f}"


def write_csv_table(path: Path, rows: list[dict[str, str]], fieldnames: list[str], config: Config) -> None:
    """写出中文命名 CSV 表格，保留中文表头，便于直接用 Excel 或 WPS 查看。"""
    with path.open("w", encoding=config.csv_encoding, newline=config.csv_newline) as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_parameter_note(output_dir: Path, nc_count: int, config: Config) -> None:
    """写出本次裁剪参数说明，方便以后不用打开代码也能核对处理设置。"""
    note_path = output_dir / "处理参数说明.txt"
    lines = [
        "处理参数说明",
        "",
        "一、输入输出",
        f"原始 NC 目录：{config.source_dir}",
        f"福建省边界 shp：{config.shapefile_path}",
        f"输出目录：{output_dir}",
        f"处理 NC 文件数量：{nc_count}",
        f"输出命名规则：{config.output_name_template}",
        "",
        "二、变量识别",
        f"优先气温变量名：{', '.join(config.preferred_temperature_names)}",
        f"优先时间坐标名：{', '.join(config.preferred_time_names)}",
        f"优先纬度坐标名：{', '.join(config.preferred_lat_names)}",
        f"优先经度坐标名：{', '.join(config.preferred_lon_names)}",
        "",
        "三、裁剪设置",
        "裁剪方式：先按福建省外接矩形缩小范围，再按福建省行政边界生成掩膜，边界外写为缺失值。",
        f"空间坐标系：{config.expected_spatial_crs}",
        f"边界像元规则 all_touched：{config.all_touched}",
        f"输出缺失值：{config.output_missing_value}",
        f"输出数据类型：{config.output_dtype}",
        "",
        "四、温度单位",
        "原始 ERA5-Land t2m 为 Kelvin。",
        f"输出 NC 中 2 米气温统一转换为摄氏度：℃ = K - {config.kelvin_to_celsius_offset}",
        f"输出气温单位属性：{config.output_temperature_units}",
        "",
        "五、NetCDF 写出",
        f"写出引擎：{config.netcdf_engine}",
        f"压缩等级：{config.compression_level}",
        "",
        "六、说明",
        "本脚本按 NC 文件逐张顺序处理，不使用并行运算。",
        "本版本不绘制空间分布图，以节省处理时间和磁盘空间。",
    ]
    note_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """主流程：检查输入、读取福建边界、逐张裁剪月 NC、写出统计表和参数说明。"""
    config = CONFIG
    output_dir = build_output_dir(config)
    print_runtime_parameters(config, output_dir)

    step_names = [
        "检查输入文件并创建输出目录",
        "读取福建省边界",
        "逐张裁剪月NC并转换为摄氏度",
        "写出中文统计表和参数说明",
    ]

    summary_rows: list[dict[str, str]] = []

    with tqdm(
        total=len(step_names),
        desc="关键步骤",
        ncols=config.tqdm_ncols,
        colour=config.tqdm_colour_step,
    ) as step_progress:
        # 步骤 1：检查输入目录、福建省 shp 和 NC 文件列表，并创建输出目录。
        # 这一步不会读取完整数据，只确认路径和文件是否具备后续处理条件。
        nc_paths = validate_inputs(config, output_dir)
        step_progress.update(1)

        # 步骤 2：读取福建省行政边界，并转换到 ERA5-Land 常用经纬度坐标系 EPSG:4326。
        # 后续每张 NC 都复用这个边界，避免重复读取 shp。
        boundary = load_boundary(config)
        step_progress.update(1)

        # 步骤 3：逐张处理月 NC 文件。
        # 主循环是顺序 for 循环，没有并行计算；每处理完一个 NC 就立即写出一个裁剪后的 NC。
        for nc_path in tqdm(
            nc_paths,
            desc="裁剪月NC文件",
            unit="个",
            ncols=config.tqdm_ncols,
            colour=config.tqdm_colour_file,
        ):
            summary_rows.append(clip_one_nc(nc_path, output_dir, boundary, config))
        step_progress.update(1)

        # 步骤 4：写出中文统计表和参数说明。
        # 统计表记录每个 NC 自动识别到的变量名、坐标名和裁剪后有效气温值数量。
        write_csv_table(
            output_dir / "NC裁剪统计表.csv",
            summary_rows,
            [
                "原始文件",
                "输出文件",
                "气温变量名",
                "时间坐标名",
                "纬度坐标名",
                "经度坐标名",
                "NC写出引擎",
                "有效气温值数量",
                "平均气温（℃）",
            ],
            config,
        )
        write_parameter_note(output_dir, len(nc_paths), config)
        step_progress.update(1)

    print(f"处理完成。所有结果已保存到：{output_dir}")


if __name__ == "__main__":
    main()
