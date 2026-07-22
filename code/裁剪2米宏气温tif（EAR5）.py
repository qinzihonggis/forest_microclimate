# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
from rasterio.mask import mask
from tqdm import tqdm


matplotlib.use("Agg")
from matplotlib import pyplot as plt


@dataclass(frozen=True)
class Config:
    # =========================
    # 1. 输入、输出路径参数
    # =========================
    # source_dir：2 米气温 tif 原始数据目录。
    # 脚本只读取该目录下的 tif，不修改原始 tif 文件。
    source_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\T2m")

    # shapefile_path：福建省行政边界 shp 文件路径。
    # 当前 tif 和该 shp 均为 EPSG:4326，经纬度坐标系一致，裁剪时不需要使用投影坐标系 shp。
    shapefile_path: Path = Path(
        r"E:\forest_microclimate\ForestMicroclimate\Fujian_Shp\福建省行政边界.shp"
    )

    # output_folder_name：输出文件夹名称。
    # 按要求在 source_dir 下创建 fujian_T2，裁剪 tif、月平均图和统计表都保存到这里。
    output_folder_name: str = "fujian_T2"

    # input_pattern：输入 tif 文件名匹配规则。
    # 当前数据命名示例为 t2m_20250101_00.tif，所以使用该通配符匹配逐小时 tif。
    input_pattern: str = "t2m_????????_??.tif"

    # source_time_format：从文件名解析时间的格式。
    # t2m_20250101_00.tif 的 stem 是 t2m_20250101_00，对应格式为 t2m_%Y%m%d_%H。
    source_time_format: str = "t2m_%Y%m%d_%H"

    # =========================
    # 2. 单位判断和温度转换参数
    # =========================
    # kelvin_mean_threshold：自动判断单位的平均值阈值。
    # 如果样例 tif 有效像元平均值 > 100，基本可判定为 Kelvin；否则判定为摄氏度。
    kelvin_mean_threshold: float = 100.0

    # kelvin_to_celsius_offset：Kelvin 转摄氏度的偏移量。
    # 当自动判断为 Kelvin 时，输出值 = 原始值 - 273.15；当前这批 tif 会被判断为已经是 ℃。
    kelvin_to_celsius_offset: float = 273.15

    # output_temperature_unit：输出 tif 和图件使用的温度单位。
    output_temperature_unit: str = "℃"

    # =========================
    # 3. 裁剪输出参数
    # =========================
    # output_nodata：输出 tif 的无效值。
    # 福建省边界外、原始 NaN 或非有限值都会统一写为该值。
    output_nodata: float = -9999.0

    # output_dtype：输出 tif 数据类型。
    # float32 足够保存摄氏度小数，同时比 float64 更节省磁盘空间。
    output_dtype: str = "float32"

    # all_touched：边界像元裁剪规则。
    # False 表示只保留像元中心落在福建省边界内的像元；True 会保留所有接触边界的像元。
    all_touched: bool = False

    # compress、predictor、zlevel：GeoTIFF 压缩参数。
    # deflate + predictor=3 适合浮点栅格，可减小输出 tif 体积，不改变数据值。
    compress: str = "deflate"
    predictor: int = 3
    zlevel: int = 6

    # =========================
    # 4. 月平均空间分布图参数
    # =========================
    # figure_size：月平均空间分布图尺寸，单位为英寸。
    figure_size: tuple[float, float] = (8.0, 7.0)

    # figure_dpi：输出图片分辨率。
    # 数值越大图片越清晰，文件体积也越大。
    figure_dpi: int = 220

    # cmap：空间分布图色带。
    # RdYlBu_r 表示低温偏蓝、高温偏红，适合温度空间分布展示。
    cmap: str = "RdYlBu_r"

    # boundary_line_color、boundary_line_width：福建省边界线样式。
    # 仅影响月平均图上的边界叠加显示，不影响裁剪结果。
    boundary_line_color: str = "black"
    boundary_line_width: float = 0.8

    # grid_alpha：图中网格线透明度。
    # 0 表示不显示，1 表示完全不透明；较小值可以辅助读坐标但不抢画面。
    grid_alpha: float = 0.25

    # =========================
    # 5. 进度条和统计表参数
    # =========================
    # tqdm_ncols：进度条显示宽度。
    # 这里故意设为 80，避免终端窗口较窄时进度条自动换行，导致看起来像日志不断刷屏。
    tqdm_ncols: int = 80

    # tqdm_dynamic_ncols：允许 tqdm 根据终端宽度自动微调显示宽度。
    # 和较短的 tqdm_bar_format 配合使用，可以让进度条尽量停留在同一行内覆盖刷新。
    tqdm_dynamic_ncols: bool = True

    # tqdm_bar_format：进度条显示格式。
    # 保留百分比、彩色条、当前数量/总量、耗时、剩余时间和速度，但去掉冗长字段，降低换行概率。
    tqdm_bar_format: str = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

    # tqdm_colour_*：进度条颜色，只影响终端显示，不影响计算结果。
    tqdm_colour_step: str = "cyan"
    tqdm_colour_file: str = "green"
    tqdm_colour_plot: str = "yellow"

    # csv_encoding：中文统计表编码。
    # utf-8-sig 便于 Excel 或 WPS 直接打开中文表头。
    csv_encoding: str = "utf-8-sig"

    # csv_newline：Windows 下写 CSV 时保持 newline=""，避免出现空行。
    csv_newline: str = ""


@dataclass
class UnitDecision:
    # should_convert：True 表示原始 tif 为 Kelvin，需要转成 ℃；False 表示原始 tif 已经是 ℃。
    should_convert: bool
    sample_file: str
    finite_count: int
    nan_count: int
    min_value: float
    max_value: float
    mean_value: float
    decision_text: str


@dataclass
class MonthlyAccumulator:
    # sum_array：逐像元累加该月所有小时的气温值。
    # count_array：逐像元记录该月有效小时数，用于最后计算月平均。
    sum_array: np.ndarray
    count_array: np.ndarray
    transform: rasterio.Affine
    crs: rasterio.crs.CRS


CONFIG = Config()


def tqdm_options(config: Config, colour: str, leave: bool = True) -> dict:
    """
    统一 tqdm 显示参数。
    使用短宽度和紧凑格式，确保进度条在同一行用回车覆盖刷新，避免终端刷出多行日志。
    """
    return {
        "ncols": config.tqdm_ncols,
        "dynamic_ncols": config.tqdm_dynamic_ncols,
        "bar_format": config.tqdm_bar_format,
        "colour": colour,
        "leave": leave,
    }


def configure_matplotlib() -> None:
    """设置中文字体，避免月平均图标题、色标、坐标轴中文乱码。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def build_output_dir(config: Config) -> Path:
    """根据配置生成输出目录路径，目录固定为原始 T2m 目录下的 fujian_T2。"""
    return config.source_dir / config.output_folder_name


def print_runtime_parameters(config: Config, output_dir: Path) -> None:
    """打印关键运行参数，便于运行前核对路径、单位判断、裁剪和绘图设置。"""
    print("运行参数")
    print(f"原始 TIF 目录: {config.source_dir}")
    print(f"福建省边界 shp: {config.shapefile_path}")
    print(f"输出目录: {output_dir}")
    print(f"输入文件匹配规则: {config.input_pattern}")
    print(f"自动判断 Kelvin 阈值: 有效像元平均值 > {config.kelvin_mean_threshold}")
    print(f"输出 NoData: {config.output_nodata}")
    print(f"边界像元规则 all_touched: {config.all_touched}")
    print(f"月平均图色带: {config.cmap}")
    print("")


def parse_time_from_name(file_path: Path, config: Config) -> datetime:
    """从 t2m_YYYYMMDD_HH.tif 文件名中解析时间，用于中文命名和按月份分组。"""
    return datetime.strptime(file_path.stem, config.source_time_format)


def list_source_tifs(config: Config) -> list[Path]:
    """
    查找并校验原始逐小时 tif 文件。
    脚本处理 source_dir 下所有匹配 input_pattern 的 tif，并按文件名时间顺序排序。
    """
    tif_paths = sorted(config.source_dir.glob(config.input_pattern))
    if not tif_paths:
        raise FileNotFoundError(f"没有在目录中找到匹配的 tif 文件: {config.source_dir}")

    for tif_path in tif_paths:
        parse_time_from_name(tif_path, config)

    return tif_paths


def validate_inputs(config: Config, output_dir: Path) -> list[Path]:
    """检查输入目录、shp 和逐小时 tif 是否存在，并创建输出目录。"""
    if not config.source_dir.exists():
        raise FileNotFoundError(f"原始 T2m 目录不存在: {config.source_dir}")
    if not config.shapefile_path.exists():
        raise FileNotFoundError(f"福建省边界 shp 不存在: {config.shapefile_path}")

    tif_paths = list_source_tifs(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    return tif_paths


def detect_temperature_unit(sample_tif: Path, config: Config) -> UnitDecision:
    """
    读取一张 tif 的有限值统计，自动判断温度单位。
    平均值大于 100 通常说明是 Kelvin；平均值在常见气温范围内则说明已经是摄氏度。
    """
    with rasterio.open(sample_tif) as src:
        data = src.read(1).astype("float64")

    finite_values = data[np.isfinite(data)]
    if finite_values.size == 0:
        raise ValueError(f"样例 tif 没有有限值，无法判断单位: {sample_tif}")

    mean_value = float(finite_values.mean())
    should_convert = mean_value > config.kelvin_mean_threshold
    decision_text = "判断为 Kelvin，裁剪输出时转换为摄氏度" if should_convert else "判断为摄氏度，裁剪输出时不转换"
    return UnitDecision(
        should_convert=should_convert,
        sample_file=sample_tif.name,
        finite_count=int(finite_values.size),
        nan_count=int(np.isnan(data).sum()),
        min_value=float(finite_values.min()),
        max_value=float(finite_values.max()),
        mean_value=mean_value,
        decision_text=decision_text,
    )


def load_boundary(config: Config, raster_crs: rasterio.crs.CRS) -> gpd.GeoDataFrame:
    """
    读取福建省边界，并确保边界坐标系与 tif 坐标系一致。
    当前数据和 shp 都是 EPSG:4326；如果以后数据 CRS 改变，这里只临时重投影边界用于裁剪，不改变输出 tif CRS。
    """
    boundary = gpd.read_file(config.shapefile_path)
    boundary = boundary[boundary.geometry.notna() & ~boundary.geometry.is_empty].copy()
    if boundary.empty:
        raise ValueError("福建省边界 shp 没有可用几何，请检查 shp 文件。")
    if boundary.crs is None:
        raise ValueError("福建省边界 shp 缺少坐标系信息，请先为 shp 定义正确 CRS。")
    if raster_crs is None:
        raise ValueError("原始 tif 缺少坐标系信息，无法与福建省边界对齐裁剪。")

    if boundary.crs != raster_crs:
        boundary = boundary.to_crs(raster_crs)

    return boundary


def build_chinese_tif_name(timestamp: datetime) -> str:
    """生成中文裁剪 tif 文件名，示例：福建省2米气温_2025年01月01日00时.tif。"""
    return f"福建省2米气温_{timestamp:%Y年%m月%d日%H时}.tif"


def build_output_profile(
    src: rasterio.DatasetReader,
    data: np.ndarray,
    transform: rasterio.Affine,
    config: Config,
) -> dict:
    """
    根据原始 tif 元数据生成输出 tif 元数据。
    高度、宽度和 transform 来自裁剪后的结果；dtype、NoData 和压缩参数来自配置区。
    """
    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        transform=transform,
        count=1,
        dtype=config.output_dtype,
        nodata=config.output_nodata,
        compress=config.compress,
        predictor=config.predictor,
        zlevel=config.zlevel,
    )
    return profile


def convert_temperature(data: np.ndarray, unit_decision: UnitDecision, config: Config) -> np.ndarray:
    """根据自动单位判断结果决定是否把 Kelvin 转换为摄氏度。"""
    converted = data.astype(np.float32)
    if unit_decision.should_convert:
        converted = converted - config.kelvin_to_celsius_offset
    return converted


def update_monthly_accumulator(
    monthly_data: dict[str, MonthlyAccumulator],
    month_key: str,
    data: np.ndarray,
    valid_mask: np.ndarray,
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
) -> None:
    """
    把当前小时裁剪后的气温栅格累加到对应月份。
    后续用 sum_array / count_array 得到该月每个像元的逐小时平均气温。
    """
    if month_key not in monthly_data:
        monthly_data[month_key] = MonthlyAccumulator(
            sum_array=np.zeros(data.shape, dtype=np.float64),
            count_array=np.zeros(data.shape, dtype=np.uint16),
            transform=transform,
            crs=crs,
        )

    accumulator = monthly_data[month_key]
    if accumulator.sum_array.shape != data.shape or accumulator.transform != transform:
        raise ValueError(f"{month_key} 内裁剪结果的网格范围或分辨率不一致，无法直接计算月平均。")

    accumulator.sum_array[valid_mask] += data[valid_mask]
    accumulator.count_array[valid_mask] += 1


def clip_one_tif(
    tif_path: Path,
    output_dir: Path,
    boundary: gpd.GeoDataFrame,
    monthly_data: dict[str, MonthlyAccumulator],
    unit_decision: UnitDecision,
    config: Config,
) -> dict[str, str]:
    """
    裁剪单张逐小时 tif。
    边界外像元和原始 NaN 会写为 NoData；如果自动判断为 Kelvin，则输出前转换为摄氏度。
    """
    timestamp = parse_time_from_name(tif_path, config)
    output_path = output_dir / build_chinese_tif_name(timestamp)
    geometries = [geometry.__geo_interface__ for geometry in boundary.geometry]

    with rasterio.open(tif_path) as src:
        masked_data, out_transform = mask(
            src,
            geometries,
            crop=True,
            filled=False,
            all_touched=config.all_touched,
        )
        clipped = masked_data[0]
        clipped_mask = np.ma.getmaskarray(clipped)
        converted = convert_temperature(np.asarray(clipped), unit_decision, config)
        valid_mask = (~clipped_mask) & np.isfinite(converted)
        output_data = np.full(converted.shape, config.output_nodata, dtype=np.float32)
        output_data[valid_mask] = converted[valid_mask]

        profile = build_output_profile(src, output_data, out_transform, config)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(output_data, 1)
            dst.set_band_description(1, "福建省2米气温（℃）")
            dst.update_tags(
                数据来源="ERA5-Land 2米气温TIF",
                处理方式="按福建省行政边界精确掩膜裁剪",
                单位判断=unit_decision.decision_text,
                温度单位=config.output_temperature_unit,
            )

        update_monthly_accumulator(
            monthly_data=monthly_data,
            month_key=timestamp.strftime("%Y年%m月"),
            data=output_data,
            valid_mask=valid_mask,
            transform=out_transform,
            crs=src.crs,
        )

    valid_values = output_data[valid_mask]
    return {
        "原始文件": tif_path.name,
        "输出文件": output_path.name,
        "时间": timestamp.strftime("%Y-%m-%d %H:00"),
        "有效像元数": str(int(valid_mask.sum())),
        "最低气温（℃）": format_float(float(valid_values.min()) if valid_values.size else np.nan),
        "最高气温（℃）": format_float(float(valid_values.max()) if valid_values.size else np.nan),
        "平均气温（℃）": format_float(float(valid_values.mean()) if valid_values.size else np.nan),
    }


def calculate_monthly_mean(accumulator: MonthlyAccumulator, nodata: float) -> np.ndarray:
    """根据月累加值和有效小时数计算逐像元月平均气温。"""
    monthly_mean = np.full(accumulator.sum_array.shape, nodata, dtype=np.float32)
    valid_mask = accumulator.count_array > 0
    monthly_mean[valid_mask] = (
        accumulator.sum_array[valid_mask] / accumulator.count_array[valid_mask]
    ).astype(np.float32)
    return monthly_mean


def get_raster_extent(transform: rasterio.Affine, width: int, height: int) -> tuple[float, float, float, float]:
    """根据仿射变换计算 imshow 需要的显示范围，确保图片坐标与栅格空间位置一致。"""
    left = transform.c
    top = transform.f
    right = left + transform.a * width
    bottom = top + transform.e * height
    return min(left, right), max(left, right), min(bottom, top), max(bottom, top)


def axis_labels(crs: rasterio.crs.CRS) -> tuple[str, str]:
    """根据坐标系类型设置坐标轴名称，经纬度坐标使用经度/纬度，投影坐标使用 X/Y 坐标。"""
    if crs and crs.is_geographic:
        return "经度", "纬度"
    return "X坐标", "Y坐标"


def plot_monthly_map(
    month_key: str,
    accumulator: MonthlyAccumulator,
    boundary: gpd.GeoDataFrame,
    output_dir: Path,
    config: Config,
) -> dict[str, str]:
    """
    绘制单个月份的福建省 2 米气温月平均空间分布图。
    图中颜色表示该月所有逐小时裁剪结果的像元平均值，边界线用于标示福建省轮廓。
    """
    monthly_mean = calculate_monthly_mean(accumulator, config.output_nodata)
    masked_mean = np.ma.masked_equal(monthly_mean, config.output_nodata)
    height, width = monthly_mean.shape
    extent = get_raster_extent(accumulator.transform, width, height)
    xlabel, ylabel = axis_labels(accumulator.crs)

    fig, ax = plt.subplots(figsize=config.figure_size, dpi=config.figure_dpi)
    if masked_mean.count() > 0:
        image = ax.imshow(masked_mean, extent=extent, origin="upper", cmap=config.cmap)
        colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
        colorbar.set_label("2米气温（℃）")
    else:
        ax.text(0.5, 0.5, "没有可绘制的有效数据", transform=ax.transAxes, ha="center", va="center")

    boundary.boundary.plot(
        ax=ax,
        color=config.boundary_line_color,
        linewidth=config.boundary_line_width,
    )
    ax.set_title(f"{month_key}福建省2米气温月平均空间分布")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=config.grid_alpha)
    fig.tight_layout()

    output_path = output_dir / f"{month_key}福建省2米气温月平均空间分布图.png"
    fig.savefig(output_path)
    plt.close(fig)

    valid_values = monthly_mean[monthly_mean != config.output_nodata]
    return {
        "月份": month_key,
        "空间分布图": output_path.name,
        "有效像元数": str(int(valid_values.size)),
        "月平均最低值（℃）": format_float(float(valid_values.min()) if valid_values.size else np.nan),
        "月平均最高值（℃）": format_float(float(valid_values.max()) if valid_values.size else np.nan),
        "区域平均值（℃）": format_float(float(valid_values.mean()) if valid_values.size else np.nan),
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


def write_summary_tables(
    output_dir: Path,
    hourly_rows: list[dict[str, str]],
    monthly_rows: list[dict[str, str]],
    config: Config,
) -> None:
    """写出逐小时裁剪统计表和月平均图统计表，文件名和表头均使用中文。"""
    write_csv_table(
        output_dir / "逐小时裁剪统计表.csv",
        hourly_rows,
        ["原始文件", "输出文件", "时间", "有效像元数", "最低气温（℃）", "最高气温（℃）", "平均气温（℃）"],
        config,
    )
    write_csv_table(
        output_dir / "月平均空间分布图统计表.csv",
        monthly_rows,
        ["月份", "空间分布图", "有效像元数", "月平均最低值（℃）", "月平均最高值（℃）", "区域平均值（℃）"],
        config,
    )


def write_parameter_note(
    output_dir: Path,
    tif_count: int,
    map_count: int,
    unit_decision: UnitDecision,
    config: Config,
) -> None:
    """写出本次处理参数说明，方便以后不用打开代码也能核对处理设置。"""
    note_path = output_dir / "处理参数说明.txt"
    lines = [
        "处理参数说明",
        "",
        "一、输入输出",
        f"原始 TIF 目录：{config.source_dir}",
        f"福建省边界 shp：{config.shapefile_path}",
        f"输出目录：{output_dir}",
        f"输入 tif 数量：{tif_count}",
        f"月平均空间分布图数量：{map_count}",
        "",
        "二、单位判断",
        f"样例文件：{unit_decision.sample_file}",
        f"有限值数量：{unit_decision.finite_count}",
        f"NaN 数量：{unit_decision.nan_count}",
        f"样例最小值：{format_float(unit_decision.min_value)}",
        f"样例最大值：{format_float(unit_decision.max_value)}",
        f"样例平均值：{format_float(unit_decision.mean_value)}",
        f"判断阈值：平均值 > {config.kelvin_mean_threshold} 判定为 Kelvin",
        f"判断结果：{unit_decision.decision_text}",
        "",
        "三、裁剪设置",
        "裁剪方式：按福建省行政边界做精确掩膜裁剪，边界外写为 NoData。",
        "坐标处理：当前 tif 和福建省行政边界 shp 均为 EPSG:4326，不进行输出投影转换。",
        f"边界像元规则 all_touched：{config.all_touched}",
        f"输出 NoData：{config.output_nodata}",
        f"输出数据类型：{config.output_dtype}",
        f"GeoTIFF 压缩：{config.compress}, predictor={config.predictor}, zlevel={config.zlevel}",
        "",
        "四、绘图设置",
        "空间分布图含义：每月所有逐小时裁剪结果的像元平均值。",
        f"图片尺寸：{config.figure_size}",
        f"图片 DPI：{config.figure_dpi}",
        f"色带：{config.cmap}",
        f"边界线颜色：{config.boundary_line_color}",
        f"边界线宽度：{config.boundary_line_width}",
    ]
    note_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """主流程：检查输入、判断单位、读取边界、裁剪逐小时 tif、绘制月平均空间分布图、写出统计说明。"""
    config = CONFIG
    output_dir = build_output_dir(config)
    configure_matplotlib()
    print_runtime_parameters(config, output_dir)

    step_names = [
        "检查输入文件并创建输出目录",
        "读取样例TIF并自动判断温度单位",
        "读取福建省边界并匹配栅格坐标系",
        "裁剪逐小时TIF并累计月平均",
        "绘制每月平均空间分布图",
        "写出中文统计表和参数说明",
    ]

    monthly_data: dict[str, MonthlyAccumulator] = {}
    hourly_rows: list[dict[str, str]] = []
    monthly_rows: list[dict[str, str]] = []

    with tqdm(
        total=len(step_names),
        desc="关键步骤",
        **tqdm_options(config, config.tqdm_colour_step),
    ) as step_progress:
        # 步骤 1：检查输入数据和输出目录。
        # 这里会查找 source_dir 下所有逐小时 tif，并创建 fujian_T2 输出目录。
        tif_paths = validate_inputs(config, output_dir)
        step_progress.update(1)

        # 步骤 2：读取第一张 tif 的有限值统计，自动判断当前数据是 Kelvin 还是 ℃。
        # 当前这批数据有效值均在常见气温范围内，会被判断为已经是摄氏度。
        unit_decision = detect_temperature_unit(tif_paths[0], config)
        print(f"单位判断: {unit_decision.decision_text}")
        print(
            f"样例 {unit_decision.sample_file}: "
            f"min={unit_decision.min_value:.3f}, "
            f"max={unit_decision.max_value:.3f}, "
            f"mean={unit_decision.mean_value:.3f}"
        )
        step_progress.update(1)

        # 步骤 3：读取福建省行政边界，并确保边界 CRS 与 tif CRS 一致。
        # 当前 tif 与 shp 均为 EPSG:4326，因此不会进行输出投影转换。
        with rasterio.open(tif_paths[0]) as first_src:
            boundary = load_boundary(config, first_src.crs)
        step_progress.update(1)

        # 步骤 4：逐小时裁剪 tif，并同时按月份累计逐像元温度和有效小时数。
        # 主循环是顺序 for 循环，没有并行计算；每处理一张 tif 就写出一张中文命名 tif。
        for tif_path in tqdm(
            tif_paths,
            desc="裁剪逐小时TIF",
            unit="张",
            **tqdm_options(config, config.tqdm_colour_file, leave=False),
        ):
            hourly_rows.append(
                clip_one_tif(
                    tif_path=tif_path,
                    output_dir=output_dir,
                    boundary=boundary,
                    monthly_data=monthly_data,
                    unit_decision=unit_decision,
                    config=config,
                )
            )
        step_progress.update(1)

        # 步骤 5：根据逐小时裁剪结果的月累计值，绘制每月一张月平均空间分布图。
        # 每个像元的月平均值 = 该月有效小时气温累加值 / 该月有效小时数。
        for month_key in tqdm(
            sorted(monthly_data),
            desc="绘制月平均空间分布图",
            unit="张",
            **tqdm_options(config, config.tqdm_colour_plot, leave=False),
        ):
            monthly_rows.append(
                plot_monthly_map(
                    month_key=month_key,
                    accumulator=monthly_data[month_key],
                    boundary=boundary,
                    output_dir=output_dir,
                    config=config,
                )
            )
        step_progress.update(1)

        # 步骤 6：写出中文统计表和处理参数说明。
        # 表格用于核对每张裁剪 tif 的有效像元数、温度范围，以及每月图的区域统计值。
        write_summary_tables(output_dir, hourly_rows, monthly_rows, config)
        write_parameter_note(output_dir, len(tif_paths), len(monthly_rows), unit_decision, config)
        step_progress.update(1)

    print(f"处理完成。所有结果已保存到：{output_dir}")


if __name__ == "__main__":
    main()
