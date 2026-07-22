"""
ERA5-Land 2m temperature hourly GeoTIFF -> monthly mean GeoTIFF.

功能：
1. 读取 2025 年福建省 2 米气温逐小时 GeoTIFF，共 8760 张。
2. 按文件名中的年月日时解析时间，并按月份分组。
3. 对每个月内所有逐小时影像逐像元求平均，忽略 NoData。
4. 输出 12 张月均温 GeoTIFF，保留原始影像的投影、范围和 1 km 分辨率。

注意：
- 输入温度单位已经是摄氏度，因此本脚本不做 Kelvin 到摄氏度转换。
- 如果输出 tif 已存在，会直接覆盖。
- 运行前请确保当前 Python 环境已安装 rasterio、numpy、tqdm。
"""

from __future__ import annotations

import calendar
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm


# =========================
# 1. 用户可修改参数
# =========================

# 输入文件夹：存放 8760 张逐小时 ERA5-Land 2m 温度 tif。
# 文件名需要包含类似“_2025年01月01日00时.tif”的时间信息。
INPUT_DIR = Path(
    r"E:\forest_microclimate\ForestMicroclimate\T2m\fujian_T2\fujian_T2m_1km"
)

# 输出文件夹：脚本会自动创建该文件夹。
# 如果里面已有同名月均值 tif，后续写入时会直接覆盖。
OUTPUT_DIR = Path(
    r"E:\forest_microclimate\ForestMicroclimate\T2m\fujian_T2\fujian_T2m_1km_month"
)

# 目标年份：只处理文件名中年份等于 TARGET_YEAR 的 tif。
# 如果以后处理其他年份，只需要修改这里。
TARGET_YEAR = 2025

# 输入和输出影像的 NoData 值。
# 计算月均值时，像元值等于该值的位置会被忽略；如果某像元整个月都无有效值，输出仍为该值。
NODATA_VALUE = -9999.0

# 输出文件命名模板。
# {year} 会替换为年份，{month:02d} 会替换为两位月份，例如 01、02、...、12。
OUTPUT_NAME_TEMPLATE = "fujian_T2m_monthly_mean_{year}_{month:02d}.tif"

# 输出数据类型。
# 原始数据是 float32，月均值继续使用 float32 可以减少文件体积，并满足温度数据精度需求。
OUTPUT_DTYPE = "float32"

# 是否覆盖已有结果。
# True：如果输出 tif 已存在，直接替换；False：遇到已有文件会报错停止。
OVERWRITE_EXISTING = True


# =========================
# 2. 进度条和文件名解析工具
# =========================

# 从文件名中提取时间信息的正则表达式。
# 示例文件名：1km福建省2米气温_2025年01月01日00时.tif
TIME_PATTERN = re.compile(r"_(\d{4})年(\d{2})月(\d{2})日(\d{2})时\.tif$", re.IGNORECASE)


def progress(iterable, *, desc: str, total: int | None = None, colour: str = "green"):
    """
    创建 tqdm 终端进度条。

    参数说明：
    - iterable：要迭代处理的对象，例如文件列表或月份列表。
    - desc：进度条左侧显示的步骤名称，用于区分当前正在执行的任务。
    - total：总任务量；如果 iterable 本身有长度，可以不传。
    - colour：进度条颜色。不同步骤使用不同颜色，便于在终端中快速区分。

    说明：
    - tqdm 会自动显示百分比、当前量/总量、耗时、剩余时间和处理速度。
    - 部分较旧 tqdm 版本可能不支持 colour 参数；此时自动退回普通进度条，保证脚本可运行。
    """
    try:
        return tqdm(iterable, desc=desc, total=total, colour=colour, unit="it")
    except TypeError:
        return tqdm(iterable, desc=desc, total=total, unit="it")


def parse_time_from_name(tif_path: Path) -> tuple[int, int, int, int]:
    """
    从 tif 文件名中解析 year、month、day、hour。

    参数说明：
    - tif_path：单个 tif 文件路径。

    返回：
    - (year, month, day, hour)：整数形式的年、月、日、小时。

    如果文件名不符合约定格式，直接报错，避免错误文件混入月均值计算。
    """
    match = TIME_PATTERN.search(tif_path.name)
    if not match:
        raise ValueError(f"文件名无法解析时间：{tif_path.name}")

    year, month, day, hour = map(int, match.groups())
    return year, month, day, hour


def collect_monthly_files(input_dir: Path, target_year: int) -> dict[int, list[Path]]:
    """
    扫描输入目录，将逐小时 tif 按月份分组。

    参数说明：
    - input_dir：逐小时 tif 所在文件夹。
    - target_year：需要处理的年份；其他年份文件会被跳过。

    返回：
    - monthly_files：字典，键为月份 1-12，值为该月所有逐小时 tif 路径列表。

    该步骤会检查每个月文件数量是否等于“当月天数 x 24小时”。
    2025 年不是闰年，因此全年应为 8760 张。
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"输入文件夹不存在：{input_dir}")

    tif_files = sorted(input_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"输入文件夹中没有 tif 文件：{input_dir}")

    monthly_files: dict[int, list[Path]] = defaultdict(list)

    for tif_path in progress(tif_files, desc="扫描并按月份分组", colour="cyan"):
        year, month, day, hour = parse_time_from_name(tif_path)

        # 校验日期和小时范围，避免异常文件名参与计算。
        if not 1 <= month <= 12:
            raise ValueError(f"月份超出范围：{tif_path.name}")
        if not 1 <= day <= calendar.monthrange(year, month)[1]:
            raise ValueError(f"日期超出范围：{tif_path.name}")
        if not 0 <= hour <= 23:
            raise ValueError(f"小时超出范围：{tif_path.name}")

        if year == target_year:
            monthly_files[month].append(tif_path)

    validate_monthly_file_counts(monthly_files, target_year)
    return {month: sorted(paths) for month, paths in monthly_files.items()}


def validate_monthly_file_counts(monthly_files: dict[int, list[Path]], target_year: int) -> None:
    """
    检查每个月逐小时 tif 数量是否完整。

    参数说明：
    - monthly_files：按月份分组后的文件路径字典。
    - target_year：目标年份，用于判断每个月应有多少小时。

    如果某个月数量不足或过多，说明输入数据不完整或有重复文件，脚本会停止并提示具体月份。
    """
    problems = []
    for month in range(1, 13):
        expected_count = calendar.monthrange(target_year, month)[1] * 24
        actual_count = len(monthly_files.get(month, []))
        if actual_count != expected_count:
            problems.append(f"{target_year}-{month:02d}: 应有 {expected_count} 张，实际 {actual_count} 张")

    if problems:
        message = "逐小时 tif 数量检查未通过：\n" + "\n".join(problems)
        raise ValueError(message)


# =========================
# 3. 月均值计算核心逻辑
# =========================

def calculate_monthly_mean(month: int, tif_paths: list[Path], reference_profile: dict) -> np.ndarray:
    """
    计算单个月份的逐像元平均温度。

    参数说明：
    - month：月份编号，用于进度条显示。
    - tif_paths：该月所有逐小时 tif 路径，正常情况下为“当月天数 x 24”张。
    - reference_profile：第一张影像的元数据，用于获取影像高度、宽度等信息。

    计算方法：
    - sum_array 累加每个像元的有效温度值。
    - count_array 记录每个像元参与平均的有效小时数量。
    - 对 count_array > 0 的像元计算 sum / count。
    - 对整个月都没有有效值的像元写入 NODATA_VALUE。
    """
    height = reference_profile["height"]
    width = reference_profile["width"]

    # float64 用于累加，减少 700 多张逐小时影像求和时的浮点误差。
    sum_array = np.zeros((height, width), dtype=np.float64)

    # uint16 足够记录单月小时数；最大 31 x 24 = 744。
    count_array = np.zeros((height, width), dtype=np.uint16)

    for tif_path in progress(tif_paths, desc=f"读取并累加 {TARGET_YEAR}-{month:02d}", colour="green"):
        with rasterio.open(tif_path) as src:
            check_raster_consistency(src, reference_profile, tif_path)
            data = src.read(1).astype(np.float64)

        # 有效值条件：
        # 1. 不是输入 NoData；
        # 2. 不是 NaN 或无穷值。
        valid_mask = (data != NODATA_VALUE) & np.isfinite(data)
        sum_array[valid_mask] += data[valid_mask]
        count_array[valid_mask] += 1

    monthly_mean = np.full((height, width), NODATA_VALUE, dtype=np.float32)
    valid_count_mask = count_array > 0
    monthly_mean[valid_count_mask] = (
        sum_array[valid_count_mask] / count_array[valid_count_mask]
    ).astype(np.float32)

    return monthly_mean


def check_raster_consistency(src: rasterio.io.DatasetReader, reference_profile: dict, tif_path: Path) -> None:
    """
    检查当前 tif 是否与第一张参考影像空间信息一致。

    参数说明：
    - src：当前打开的 rasterio 数据集。
    - reference_profile：第一张影像的元数据。
    - tif_path：当前 tif 路径，用于报错时定位问题文件。

    检查内容：
    - 影像宽度、高度一致；
    - 坐标系 CRS 一致；
    - 仿射变换 transform 一致，即范围和分辨率一致；
    - 仅处理单波段影像。

    这样可以避免不同网格的 tif 被错误地逐像元相加。
    """
    if src.count != 1:
        raise ValueError(f"仅支持单波段 tif，当前文件波段数为 {src.count}：{tif_path}")
    if src.width != reference_profile["width"] or src.height != reference_profile["height"]:
        raise ValueError(f"影像尺寸不一致：{tif_path}")
    if src.crs != reference_profile["crs"]:
        raise ValueError(f"坐标系不一致：{tif_path}")
    if src.transform != reference_profile["transform"]:
        raise ValueError(f"仿射变换不一致：{tif_path}")


def write_monthly_tif(output_path: Path, monthly_mean: np.ndarray, reference_profile: dict) -> None:
    """
    写出单个月份的月均温 GeoTIFF。

    参数说明：
    - output_path：输出 tif 路径。
    - monthly_mean：已经计算完成的二维月均温数组。
    - reference_profile：第一张输入 tif 的元数据。

    输出设置：
    - driver="GTiff"：输出 GeoTIFF。
    - dtype=float32：输出浮点型温度。
    - count=1：单波段。
    - nodata=-9999：无效像元值。
    - compress="lzw"：无损压缩，减小文件体积。
    """
    if output_path.exists() and not OVERWRITE_EXISTING:
        raise FileExistsError(f"输出文件已存在，且 OVERWRITE_EXISTING=False：{output_path}")

    output_profile = reference_profile.copy()
    output_profile.update(
        driver="GTiff",
        dtype=OUTPUT_DTYPE,
        count=1,
        nodata=NODATA_VALUE,
        compress="lzw",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(monthly_mean.astype(OUTPUT_DTYPE), 1)


# =========================
# 4. 主流程
# =========================

def main() -> None:
    """
    主函数：按顺序执行目录创建、文件分组、月均值计算和结果写出。

    进度条说明：
    - 青色：扫描输入文件并按月份分组。
    - 洋红色：12 个月整体处理进度。
    - 绿色：某个月内逐小时 tif 读取和累加进度。
    - 黄色：月均值 tif 写出进度。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    monthly_files = collect_monthly_files(INPUT_DIR, TARGET_YEAR)
    first_tif = monthly_files[1][0]

    with rasterio.open(first_tif) as first_src:
        reference_profile = first_src.profile.copy()

    for month in progress(range(1, 13), desc="计算 12 个月月均温", colour="magenta"):
        tif_paths = monthly_files[month]
        monthly_mean = calculate_monthly_mean(month, tif_paths, reference_profile)

        output_name = OUTPUT_NAME_TEMPLATE.format(year=TARGET_YEAR, month=month)
        output_path = OUTPUT_DIR / output_name

        for _ in progress([output_path], desc=f"写出 {TARGET_YEAR}-{month:02d}", colour="yellow"):
            write_monthly_tif(output_path, monthly_mean, reference_profile)

    print("月均温计算完成。")
    print(f"输入文件夹：{INPUT_DIR}")
    print(f"输出文件夹：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
