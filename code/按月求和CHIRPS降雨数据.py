# -*- coding: utf-8 -*-
"""
批量将福建 CHIRPS 1 km、5 天时间分辨率降雨 NetCDF 数据按自然月求和。

脚本目标：
1. 输入目录：
   E:\\forest_microclimate\\ForestMicroclimate\\Precipitation_CHIRPS\\fujian_pre\\fujian_pre_1km
2. 输出目录：
   E:\\forest_microclimate\\ForestMicroclimate\\Precipitation_CHIRPS\\fujian_pre\\fujian_pre_1km_monthly
3. 输入文件格式：
   fujian_年份_pre_1km.nc
4. 输出文件格式：
   fujian_年份_pre_1km_monthly.nc
5. 只沿 time 维度做自然月求和，空间网格保持原样，不做重投影、不做插值。
6. 如果输出目录中已有同名文件，直接覆盖。
7. 脚本结束时清理本次运行创建的临时文件，不删除输入文件和正式输出文件。
"""

from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm


# =============================================================================
# 一、可调参数区
# =============================================================================
# 输入目录。
# 含义：
# 这里放的是已经插值到 1 km 空间分辨率、但时间分辨率仍为 5 天的福建 CHIRPS 年度文件。
# 脚本会在该目录下批量查找 fujian_*_pre_1km.nc。
INPUT_DIR = Path(
    r"E:\forest_microclimate\ForestMicroclimate\Precipitation_CHIRPS"
    r"\fujian_pre"
)

# 输出目录。
# 含义：
# 月尺度结果统一保存到该目录。若目录不存在，脚本运行时会自动创建。
OUTPUT_DIR = Path(
    r"E:\forest_microclimate\ForestMicroclimate\Precipitation_CHIRPS"
    r"\fujian_pre\fujian_pre_WGS_monthly"
)

# 输入文件匹配规则。
# 含义：
# 只处理符合 fujian_年份_pre_1km.nc 格式的文件，避免误处理其它 NetCDF 数据。
INPUT_GLOB = "fujian_*_pre_CHIRPS.nc"

# 期望使用的降雨变量名。
# 含义：
# 参考 1 km 插值脚本，输出降雨变量通常为 precip。
# 如果文件中没有 precip，但只有一个数据变量，则自动使用唯一变量并改名为 precip。
PRECIP_VAR = "precip"

# 月汇总频率。
# 含义：
# "1MS" 表示 Month Start，即自然月汇总后用每月 1 日作为新的 time 标签。
MONTHLY_RESAMPLE_RULE = "1MS"

# 输出数据类型。
# 含义：
# 月累计降雨保留 float32，可减小文件体积，并保持降雨量分析所需精度。
OUTPUT_DTYPE = "float32"

# NetCDF 压缩级别。
# 含义：
# 数值越大压缩越强、写出越慢。4 是速度和体积之间的常用折中值。
COMPRESSION_LEVEL = 4

# 是否覆盖已有输出。
# 含义：
# True 表示如果输出目录已有同名月尺度文件，则本次运行结果会覆盖它。
OVERWRITE_EXISTING = True

# 临时文件后缀。
# 含义：
# 为避免写出中断时留下损坏的正式输出，脚本先写入临时文件，成功后再替换正式文件。
TEMP_SUFFIX = ".tmp"

# tqdm 进度条统一样式。
# 含义：
# 显示百分比、彩色条、当前量/总量、已耗时、剩余时间和速度。
PROGRESS_BAR_FORMAT = (
    "{l_bar}{bar:28}| {n_fmt}/{total_fmt} "
    "[{percentage:3.0f}% | 已用 {elapsed} | 剩余 {remaining} | {rate_fmt}]"
)

# 不同类型进度条颜色。
# 含义：
# 用不同颜色区分总进度、文件处理、检查、月汇总、保存和清理步骤。
OVERALL_BAR_COLOR = "cyan"
FILE_BAR_COLOR = "white"
CHECK_BAR_COLOR = "yellow"
PROCESS_BAR_COLOR = "green"
SAVE_BAR_COLOR = "magenta"
CLEAN_BAR_COLOR = "blue"


# =============================================================================
# 二、工具函数
# =============================================================================
def make_bar(total: int, desc: str, unit: str, colour: str) -> tqdm:
    """
    创建统一样式的 tqdm 彩色进度条。

    参数说明：
    total:
        当前进度条的总任务量，用于显示百分比和剩余时间。
    desc:
        进度条左侧显示的步骤名称。
    unit:
        当前任务的计量单位，例如“个”“项”“月”。
    colour:
        进度条颜色，用于区分不同类型的处理阶段。
    """
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        colour=colour,
        dynamic_ncols=True,
        bar_format=PROGRESS_BAR_FORMAT,
    )


def find_input_files(input_dir: Path) -> list[Path]:
    """
    查找需要批量处理的年度 1 km CHIRPS 文件。

    用意：
    输入目录中可能包含其它文件，因此只匹配 INPUT_GLOB 指定的文件名格式。
    返回结果按文件名排序，保证处理顺序稳定，例如从 1990 到 2025。
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是目录: {input_dir}")

    input_files = sorted(input_dir.glob(INPUT_GLOB))
    if not input_files:
        raise FileNotFoundError(
            f"输入目录下未找到匹配文件: {input_dir}\\{INPUT_GLOB}"
        )
    return input_files


def make_output_path(input_file: Path) -> Path:
    """
    根据输入文件名生成对应的月尺度输出路径。

    命名规则：
    fujian_1990_pre_1km.nc -> fujian_1990_pre_1km_monthly.nc

    用意：
    保留年份和数据来源信息，同时用 monthly 明确标记时间分辨率已经变为月尺度。
    """
    return OUTPUT_DIR / f"{input_file.stem}_monthly.nc"


def make_temp_path(output_file: Path) -> Path:
    """
    为正式输出文件生成本次写出使用的临时文件路径。

    用意：
    先写临时文件，写出成功后再替换正式输出，避免程序中断时生成不完整的正式文件。
    """
    return output_file.with_name(f".{output_file.stem}{TEMP_SUFFIX}{output_file.suffix}")


def normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    标准化数据集的坐标名称和降雨变量名称。

    处理规则：
    1. 若坐标名为 longitude/latitude，则改名为 lon/lat。
    2. 若坐标名为 Longitude/Latitude，也改名为 lon/lat。
    3. 必须存在 time 坐标，因为月求和需要沿 time 维度聚合。
    4. 优先使用 precip 变量。
    5. 如果没有 precip，但文件中只有一个数据变量，则自动将该变量改名为 precip。
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

    if "time" not in ds.coords:
        raise ValueError("输入文件缺少必要坐标: time")

    if PRECIP_VAR not in ds.data_vars:
        if len(ds.data_vars) == 1:
            only_var = next(iter(ds.data_vars))
            ds = ds.rename({only_var: PRECIP_VAR})
        else:
            raise ValueError(
                f"未找到变量 '{PRECIP_VAR}'，当前数据变量为: {list(ds.data_vars)}"
            )

    return ds


def detect_spatial_dims(da: xr.DataArray) -> list[str]:
    """
    识别降雨变量中除 time 外的空间维度。

    用意：
    1 km 插值结果可能使用 x/y 作为投影坐标，也可能使用 lon/lat。
    本脚本不改变空间网格，因此只需要确认除 time 外仍有空间维度并原样保留。
    """
    spatial_dims = [dim for dim in da.dims if dim != "time"]
    if len(spatial_dims) < 2:
        raise ValueError(f"空间维度不足，当前变量维度为: {list(da.dims)}")
    return spatial_dims


def monthly_sum_with_progress(da: xr.DataArray, file_label: str) -> xr.DataArray:
    """
    对单个年度文件的 5 天降雨数据逐月求和。

    处理逻辑：
    1. 使用 xarray.resample 按自然月分组。
    2. 每个月沿 time 维度求和，得到月累计降雨。
    3. 使用 min_count=1 保留全为空值的像元为空值，避免把全 NaN 错误写成 0。
    4. 将所有月份结果重新拼接成月尺度 DataArray。
    """
    resampled = da.resample(time=MONTHLY_RESAMPLE_RULE)
    total_months = len(resampled.groups)
    monthly_outputs = []

    with tqdm(
        resampled,
        total=total_months,
        desc=f"月汇总 {file_label}",
        unit="月",
        colour=PROCESS_BAR_COLOR,
        dynamic_ncols=True,
        bar_format=PROGRESS_BAR_FORMAT,
    ) as month_bar:
        for month_start, month_slice in month_bar:
            month_sum = month_slice.sum(
                dim="time",
                skipna=True,
                keep_attrs=True,
                min_count=1,
            )
            month_sum = month_sum.expand_dims(time=[month_start])
            monthly_outputs.append(month_sum)

    monthly_da = xr.concat(monthly_outputs, dim="time")
    monthly_da["time"].attrs.update(da["time"].attrs)
    return monthly_da


def update_attrs(
    monthly_da: xr.DataArray,
    source_attrs: dict,
    input_file: Path,
    output_file: Path,
) -> xr.DataArray:
    """
    更新月尺度降雨变量属性。

    用意：
    输出文件应明确记录数据来源、聚合方法、时间分辨率变化和空间网格状态，
    便于后续检查、复现和用于 SPI/SPEI 等指数计算。
    """
    monthly_da.name = PRECIP_VAR
    monthly_da.attrs.update(source_attrs)
    monthly_da.attrs.update(
        {
            "long_name": "CHIRPS monthly precipitation sum",
            "units": "mm/month",
            "description": (
                "Monthly precipitation sums aggregated from 5-day Fujian CHIRPS "
                "1 km data using natural calendar months. Spatial grid is unchanged."
            ),
            "source_file": str(input_file),
            "output_file": str(output_file),
            "original_temporal_resolution": "pentad / 5-day",
            "target_temporal_resolution": "monthly",
            "spatial_operation": "none; original 1 km grid preserved",
            "aggregation_method": "sum over time within each natural month",
        }
    )
    return monthly_da


def save_monthly_dataset(
    monthly_da: xr.DataArray,
    input_file: Path,
    output_file: Path,
    temp_file: Path,
) -> None:
    """
    将单个文件的月尺度结果保存为 NetCDF。

    保存策略：
    1. 先写入 temp_file。
    2. 写入成功后用 temp_file 覆盖 output_file。
    3. 如果 output_file 已存在且 OVERWRITE_EXISTING=True，则直接覆盖。

    这样可以降低写出失败时正式输出文件损坏的风险。
    """
    if output_file.exists() and not OVERWRITE_EXISTING:
        raise FileExistsError(f"输出文件已存在: {output_file}")

    out_ds = monthly_da.to_dataset()
    out_ds.attrs.update(
        {
            "title": "Fujian CHIRPS 1 km monthly precipitation",
            "source_file": str(input_file),
            "output_file": str(output_file),
            "temporal_aggregation": "natural month sum",
            "spatial_grid_status": "unchanged from input 1 km grid",
        }
    )

    encoding = {
        PRECIP_VAR: {
            "zlib": True,
            "complevel": COMPRESSION_LEVEL,
            "dtype": OUTPUT_DTYPE,
            "_FillValue": np.float32(np.nan),
        }
    }

    if temp_file.exists():
        temp_file.unlink()
    out_ds.to_netcdf(temp_file, encoding=encoding)
    out_ds.close()
    temp_file.replace(output_file)


def cleanup_temp_files(temp_files: list[Path]) -> None:
    """
    清理本次脚本运行创建的临时文件。

    安全边界：
    1. 只删除 temp_files 列表中记录的临时文件。
    2. 不删除输入文件。
    3. 不删除正式输出文件。
    4. 即使前面某个文件处理失败，也会尝试清理已经登记的临时文件。
    """
    with make_bar(total=len(temp_files), desc="清理临时文件", unit="个", colour=CLEAN_BAR_COLOR) as bar:
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()
            bar.update(1)


def process_one_file(input_file: Path, temp_files: list[Path]) -> tuple[Path, list[str], int]:
    """
    处理单个年度 1 km CHIRPS 文件。

    返回内容：
    1. 输出文件路径；
    2. 空间维度名称列表；
    3. 输出月时间步数量。
    """
    output_file = make_output_path(input_file)
    temp_file = make_temp_path(output_file)
    temp_files.append(temp_file)

    ds = None
    try:
        with make_bar(total=5, desc=f"处理 {input_file.stem}", unit="步", colour=FILE_BAR_COLOR) as file_bar:
            # 步骤 1：读取数据。
            # 用意：打开当前年度 1 km 降雨 NetCDF 文件，为后续结构检查和月汇总做准备。
            ds = xr.open_dataset(input_file)
            file_bar.update(1)

            # 步骤 2：标准化结构。
            # 用意：统一常见坐标名，确认 time 坐标和 precip 变量可用。
            ds = normalize_dataset(ds)
            precip_5day = ds[PRECIP_VAR]
            source_attrs = precip_5day.attrs.copy()
            if "time" not in precip_5day.dims or precip_5day.sizes["time"] == 0:
                raise ValueError(f"降雨变量缺少有效 time 维度: {input_file}")
            spatial_dims = detect_spatial_dims(precip_5day)
            file_bar.update(1)

            # 步骤 3：按自然月求和。
            # 用意：只沿 time 维度聚合，空间维度和像元网格完全保持输入文件状态。
            precip_monthly = monthly_sum_with_progress(precip_5day, input_file.stem)
            file_bar.update(1)

            # 步骤 4：写入属性。
            # 用意：记录来源文件、输出文件、聚合方法和空间网格未改变等元数据。
            precip_monthly = update_attrs(
                precip_monthly,
                source_attrs,
                input_file,
                output_file,
            )
            file_bar.update(1)

            # 步骤 5：保存结果。
            # 用意：先写临时文件，再覆盖正式输出，降低中断时生成坏文件的风险。
            save_monthly_dataset(precip_monthly, input_file, output_file, temp_file)
            file_bar.update(1)
    finally:
        if ds is not None:
            ds.close()

    return output_file, spatial_dims, precip_monthly.sizes.get("time", 0)


# =============================================================================
# 三、主流程
# =============================================================================
def main() -> None:
    """
    主程序入口。

    整体流程：
    1. 检查输入目录，收集所有年度 1 km 文件。
    2. 创建输出目录。
    3. 逐个文件执行自然月求和。
    4. 每个文件保存为独立月尺度 NetCDF。
    5. 运行结束后清理本次创建的临时文件。
    """
    temp_files: list[Path] = []
    processed_outputs: list[Path] = []

    try:
        with make_bar(total=4, desc="总进度", unit="步", colour=OVERALL_BAR_COLOR) as overall:
            # 步骤 1：查找输入文件。
            # 用意：只处理最终确认的 fujian_pre_1km 目录中的年度 1 km NetCDF 文件。
            input_files = find_input_files(INPUT_DIR)
            overall.update(1)

            # 步骤 2：创建输出目录。
            # 用意：确保 fujian_pre_1km_monthly 存在，便于集中保存所有月尺度结果。
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            overall.update(1)

            # 步骤 3：批量处理文件。
            # 用意：对每个年度文件单独读取、月求和、保存，降低内存占用并便于定位问题。
            with tqdm(
                input_files,
                total=len(input_files),
                desc="批量文件",
                unit="个",
                colour=FILE_BAR_COLOR,
                dynamic_ncols=True,
                bar_format=PROGRESS_BAR_FORMAT,
            ) as file_list_bar:
                last_spatial_dims = []
                last_time_count = 0
                for input_file in file_list_bar:
                    output_file, spatial_dims, time_count = process_one_file(
                        input_file,
                        temp_files,
                    )
                    processed_outputs.append(output_file)
                    last_spatial_dims = spatial_dims
                    last_time_count = time_count
            overall.update(1)

            # 步骤 4：清理临时文件。
            # 用意：删除本次运行登记的 .tmp 中间文件，不影响输入和正式输出。
            cleanup_temp_files(temp_files)
            overall.update(1)

        print("\n处理完成")
        print(f"输入目录: {INPUT_DIR}")
        print(f"输出目录: {OUTPUT_DIR}")
        print(f"处理文件数: {len(processed_outputs)}")
        print(f"输出文件名格式: fujian_年份_pre_1km_monthly.nc")
        print(f"最后一个文件空间维度: {last_spatial_dims}")
        print(f"最后一个文件月时间步数量: {last_time_count}")

    finally:
        # 兜底清理：
        # 如果中途发生异常，仍尝试删除本次运行创建的临时文件。
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()


if __name__ == "__main__":
    main()
