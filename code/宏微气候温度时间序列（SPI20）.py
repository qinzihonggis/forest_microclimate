# -*- coding: utf-8 -*-
"""
宏微气候温度小时尺度时间序列对比图

功能：
1. 读取 27 个 TOMST 站点 CSV，提取 T3_15 列作为林下 15 cm 空气温度。
2. 读取样地经纬度表，根据每个站点坐标从 ERA5 2 m 气温逐小时 tif 中提取像元值。
3. 将 TOMST 15 分钟数据和 ERA5 逐小时数据统一转换到北京时间 UTC+8 后，按小时尺度对齐。
4. 每个站点输出 1 张小时尺度折线图，并输出小时尺度对比表、汇总表和运行参数说明。
5. 脚本结束后清理本次运行产生的临时缓存目录。

运行方式示例：
D:/ProgramData/anaconda3/envs/gee/python.exe E:/forest_microclimate/code/宏微气候温度时间序列.py
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, fields
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as transform_coords
from tqdm import tqdm


# =============================================================================
# 一、用户可调参数区
# =============================================================================
@dataclass(frozen=True)
class Config:
    # -------------------------------------------------------------------------
    # 1. 输入输出路径参数
    # -------------------------------------------------------------------------
    # tomst_dir：TOMST 传感器 CSV 所在文件夹。
    # 脚本会在该目录下查找 95332217 到 95332244 中除 95332241 外的站点文件。
    tomst_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\Tensor_Data")

    # site_csv：样地坐标表路径。
    # 必须包含 Site_ID、Longitude、Latitude 三列，分别表示站点编号、经度、纬度。
    site_csv: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\Tensor_LatLong.csv")

    # era5_tif_dir：ERA5 2 m 气温逐小时 tif 文件夹。
    # 文件名示例：福建省2米气温_2025年01月01日00时.tif
    era5_tif_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\T2m\fujian_T2")

    # drought_event_csv：daily SPI 干旱事件长表路径。
    # 逐日折线图会读取其中 Drought_Level 为 Extreme 的事件，并把对应日期范围标成浅红色背景。
    drought_event_csv: Path = Path(
        r"E:\forest_microclimate\ForestMicroclimate\results\daily_SPI_features\福建省观测站2025年daily_SPI干旱事件长表.csv"
    )

    # output_dir：所有结果输出目录。
    # 图、小时尺度表、汇总表、参数说明都会输出到该目录或其子目录，文件名均使用中文。
    output_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\results\时间序列图")

    # temp_dir_name：本次运行临时缓存目录名称。
    # 当前脚本主要在内存中处理数据，此目录作为预留缓存位置，结束时会自动删除。
    temp_dir_name: str = "_本次脚本运行临时缓存"

    # -------------------------------------------------------------------------
    # 2. 站点和表格列名参数
    # -------------------------------------------------------------------------
    # site_id_start、site_id_end：站点编号范围，包含两端。
    # 95332217 到 95332244 理论共 28 个编号，下面 excluded_site_ids 会排除缺失站点。
    site_id_start: int = 95332217
    site_id_end: int = 95332244

    # excluded_site_ids：明确缺失或不参与处理的站点。
    # 你说明缺少 95332241，因此默认排除该站点，最终处理 27 个站点。
    excluded_site_ids: tuple[int, ...] = (95332241,)

    # tomst_utc_time_col：TOMST CSV 中 UTC 时间列。
    # 格式示例：2024.10.31 10:15。脚本用它作为内部时间基准，避免时区错位。
    tomst_utc_time_col: str = "data_time"

    # tomst_local_time_col：TOMST CSV 中已有的 UTC+8 时间列。
    # 当前脚本默认不用该列计算，而是由 data_time 加 8 小时得到北京时间，保证内部对齐一致。
    tomst_local_time_col: str = "data_time8"

    # tomst_temp_col：TOMST CSV 中林下 15 cm 空气温度列。
    # 如果以后要改成 0 cm 气温或 5 cm 土温，只需要修改该列名。
    tomst_temp_col: str = "T3_15"

    # site_id_col、site_lon_col、site_lat_col：样地坐标表列名。
    site_id_col: str = "Site_ID"
    site_lon_col: str = "Longitude"
    site_lat_col: str = "Latitude"

    # drought_site_id_col、drought_start_col、drought_end_col、drought_level_col：干旱事件长表列名。
    # Site_ID 用于让每个站点只绘制自己的 Extreme 干旱背景。
    drought_site_id_col: str = "Site_ID"
    drought_start_col: str = "Start_Date"
    drought_end_col: str = "End_Date"
    drought_level_col: str = "Drought_Level"
    extreme_drought_level_name: str = "Extreme"

    # csv_encodings：读取 CSV 时依次尝试的编码。
    # utf-8-sig 适合 Excel 导出的 UTF-8 文件；gbk/gb18030 适合常见中文 Windows CSV。
    csv_encodings: tuple[str, ...] = ("utf-8-sig", "utf-8", "gbk", "gb18030")

    # output_csv_encoding：输出 CSV 编码。
    # utf-8-sig 便于 Excel/WPS 直接打开中文表头不乱码。
    output_csv_encoding: str = "utf-8-sig"

    # -------------------------------------------------------------------------
    # 3. 时间和 ERA5 参数
    # -------------------------------------------------------------------------
    # tomst_time_format：TOMST UTC 时间字符串格式。
    # 对应示例 2024.10.31 10:15。如果源表格式变化，需要同步修改。
    tomst_time_format: str = "%Y.%m.%d %H:%M"

    # timezone_offset_hours：北京时间相对 UTC 的小时偏移。
    # 中国样地展示用 UTC+8，因此这里设为 8。
    timezone_offset_hours: int = 8

    # output_start_date、output_end_date：最终图表输出的北京时间日期范围。
    # 当前只绘制和输出 2025 年 1 月 1 日到 2025 年 12 月 31 日的数据。
    # 如果以后需要绘制其他年份或更短时间段，只改这两个日期即可。
    output_start_date: str = "2025-01-01"
    output_end_date: str = "2025-12-31"

    # era5_filename_regex：从 ERA5 tif 文件名解析 UTC 时间的正则表达式。
    # 支持 2025年01月01日00时，也支持 2025年1月1日0时。
    era5_filename_regex: str = r"(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日(?P<hour>\d{1,2})时"

    # era5_time_is_utc：ERA5 文件名时间是否按 UTC 理解。
    # 你要求内部对齐 UTC，因此默认 True；脚本会将其加 8 小时后按北京时间小时尺度对齐。
    era5_time_is_utc: bool = True

    # era5_temperature_unit：ERA5 tif 温度单位。
    # 当前数据已是摄氏度，因此不做 K 到 ℃ 转换。
    era5_temperature_unit: str = "摄氏度"

    # raster_band_index：读取 tif 的波段序号。
    # rasterio 波段从 1 开始计数；单波段温度 tif 通常保持 1。
    raster_band_index: int = 1

    # default_tif_crs：当 tif 缺失坐标系时采用的默认坐标系。
    # 福建 ERA5 裁剪结果通常为 WGS84 经纬度坐标，即 EPSG:4326。
    default_tif_crs: str = "EPSG:4326"

    # -------------------------------------------------------------------------
    # 4. 输出表格和图形目录参数
    # -------------------------------------------------------------------------
    # plot_subdir、table_subdir：图件和表格子目录名称。
    # 均为中文命名，便于直接识别输出内容。
    hourly_plot_subdir: str = "小时尺度折线图"
    hourly_table_subdir: str = "小时尺度温度表"
    daily_plot_subdir: str = "逐日折线图"
    daily_table_subdir: str = "逐日温度表"
    daily_combined_plot_subdir: str = "逐日折线合并图"

    # per_site_table_suffix：每个站点小时尺度对比表后缀。
    hourly_per_site_table_suffix: str = "_小时尺度温度对比表.csv"
    daily_per_site_table_suffix: str = "_逐日温度对比表.csv"

    # all_sites_table_name：所有站点合并后的小时尺度温度汇总表文件名。
    hourly_all_sites_table_name: str = "所有站点_小时尺度温度对比汇总表.csv"
    daily_all_sites_table_name: str = "所有站点_逐日温度对比汇总表.csv"

    # era5_hourly_extract_table_name、hourly_aligned_table_name：新增的完整逐小时数据表文件名。
    # ERA5逐小时提取表只保存 ERA5 按站点坐标提取出的逐小时温度。
    # 逐小时温度对齐表保存同一站点同一北京时间小时下的 ERA5 2 m 温度和 TOMST 15 cm 温度。
    era5_hourly_extract_table_name: str = "ERA5逐小时提取表.csv"
    hourly_aligned_table_name: str = "逐小时温度对齐表.csv"

    # parameter_report_name：运行参数说明文件名。
    parameter_report_name: str = "运行参数说明.txt"

    # -------------------------------------------------------------------------
    # 5. tqdm 进度条参数
    # -------------------------------------------------------------------------
    # tqdm_bar_format：进度条显示格式。
    # 包含百分比、彩色进度条、当前量/总量、耗时、剩余时间和处理速度。
    tqdm_bar_format: str = (
        "{l_bar}{bar:32}| {percentage:3.0f}% {n_fmt}/{total_fmt} "
        "[已用 {elapsed} < 剩余 {remaining}, {rate_fmt}]"
    )

    # tqdm_dynamic_ncols：根据终端宽度自动调整，尽量保持单行原地刷新。
    tqdm_dynamic_ncols: bool = True

    # tqdm_leave：每个步骤结束后是否保留该步骤的最终进度条。
    # True 会让每个关键步骤保留一行，不会为每个文件刷屏。
    tqdm_leave: bool = True

    # progress_colours：不同处理阶段使用不同颜色，便于在终端区分当前步骤。
    progress_colours: dict[str, str] = None

    # -------------------------------------------------------------------------
    # 6. 折线图参数
    # -------------------------------------------------------------------------
    # figure_size：单张图尺寸，单位为英寸。
    figure_size: tuple[float, float] = (12.0, 5.8)

    # combined_subplot_size：合并图中每个子图的基础尺寸，单位为英寸。
    # 例如 5 行 x 2 列时，整张图约为 11 x 16 英寸，便于容纳 10 个站点子图。
    combined_subplot_size: tuple[float, float] = (5.5, 3.2)

    # figure_dpi：输出图片分辨率。
    # 数值越大越清晰，文件体积也会增加。
    figure_dpi: int = 300

    # tomst_line_color、era5_line_color：两条温度曲线的颜色。
    # TOMST 实测微气候使用蓝色，ERA5 宏气候使用红色。
    tomst_line_color: str = "#1F77B4"
    era5_line_color: str = "#D62728"

    # tomst_line_width、era5_line_width：折线宽度。
    tomst_line_width: float = 1.8
    era5_line_width: float = 1.8

    # tomst_line_style、era5_line_style：折线类型。
    # "-" 为实线，"--" 为虚线，"-." 为点划线。
    tomst_line_style: str = "-"
    era5_line_style: str = "-"

    # marker、marker_size：小时值点标记样式和大小。
    # 如果不想显示点，可将 marker 改为 None。
    marker: str | None = "o"
    marker_size: float = 2.4

    # line_alpha：折线透明度，1 为不透明，0 为完全透明。
    line_alpha: float = 0.92

    # tomst_label、era5_label：图例名称。
    tomst_label: str = "Microclimate"
    era5_label: str = "Macroclimate"

    # legend_loc、legend_fontsize：图例位置和字号。
    legend_loc: str = "best"
    legend_fontsize: int = 10

    # title_fontsize、axis_label_fontsize、tick_label_fontsize：标题、坐标轴和刻度字号。
    title_fontsize: int = 14
    axis_label_fontsize: int = 11
    tick_label_fontsize: int = 9

    # grid_visible、grid_alpha、grid_line_style：网格显示参数。
    grid_visible: bool = True
    grid_alpha: float = 0.25
    grid_line_style: str = "--"

    # month_tick_interval：横轴每隔几个月显示一个月份刻度。
    # 你要求每隔两个月显示一个月刻度，因此默认值为 2。
    month_tick_interval: int = 2

    # date_tick_format：横轴日期刻度格式。
    # %Y-%m 表示年-月，例如 2025-01。
    date_tick_format: str = "%Y-%m"

    # y_axis_label：纵轴标签。
    y_axis_label: str = "小时气温（℃）"
    daily_y_axis_label: str = "日均气温（℃）"

    # extreme_background_color、extreme_background_alpha：逐日图中 Extreme 干旱事件背景样式。
    # 颜色越浅、透明度越低，对折线遮挡越少；只影响逐日折线图，不影响小时尺度折线图。
    extreme_background_color: str = "#F4A3A3"
    extreme_background_alpha: float = 0.28
    extreme_background_label: str = "Extreme drought"


CONFIG = Config(
    progress_colours={
        "检查": "yellow",
        "坐标": "cyan",
        "TOMST": "green",
        "ERA5文件": "blue",
        "ERA5提取": "magenta",
        "干旱事件": "yellow",
        "绘图输出": "cyan",
    }
)


# =============================================================================
# 二、通用工具函数
# =============================================================================
def make_bar(total: int, desc: str, unit: str, colour: str) -> tqdm:
    """创建统一样式的 tqdm 彩色进度条，确保每个关键步骤只有一个动态进度条。"""
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        colour=colour,
        dynamic_ncols=CONFIG.tqdm_dynamic_ncols,
        leave=CONFIG.tqdm_leave,
        bar_format=CONFIG.tqdm_bar_format,
    )


def configure_matplotlib() -> None:
    """设置中文字体和负号显示，避免中文标题、图例和坐标轴乱码。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def read_csv_with_fallback(path: Path, **kwargs) -> pd.DataFrame:
    """按多个常见编码尝试读取 CSV，提高对 Excel/WPS 导出中文文件的兼容性。"""
    last_error: Exception | None = None
    for encoding in CONFIG.csv_encodings:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(
        f"无法用这些编码读取文件：{CONFIG.csv_encodings}；文件：{path}；最后错误：{last_error}"
    )


def normalise_site_id(value: object) -> str:
    """将站点编号统一成不带小数点的字符串，避免 CSV 把编号读成 95332217.0。"""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def expected_site_ids() -> list[str]:
    """生成需要处理的 27 个站点编号列表，并排除明确缺失的 95332241。"""
    excluded = {str(site_id) for site_id in CONFIG.excluded_site_ids}
    return [
        str(site_id)
        for site_id in range(CONFIG.site_id_start, CONFIG.site_id_end + 1)
        if str(site_id) not in excluded
    ]


def find_tomst_csv(site_id: str) -> Path:
    """查找指定站点的 TOMST CSV，兼容 95332217.csv 和 95332217 两种命名。"""
    candidates = [
        CONFIG.tomst_dir / f"{site_id}.csv",
        CONFIG.tomst_dir / site_id,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"未找到站点 {site_id} 的 TOMST CSV：{candidates}")


def parse_era5_time_from_name(path: Path) -> pd.Timestamp:
    """从 ERA5 中文 tif 文件名中解析逐小时时间，并按配置作为 UTC 时间处理。"""
    match = re.search(CONFIG.era5_filename_regex, path.stem)
    if not match:
        raise ValueError(f"无法从文件名解析 ERA5 时间：{path.name}")

    year = int(match.group("year"))
    month = int(match.group("month"))
    day = int(match.group("day"))
    hour = int(match.group("hour"))
    return pd.Timestamp(year=year, month=month, day=day, hour=hour)


def output_paths() -> tuple[Path, Path, Path, Path, Path, Path]:
    """创建输出根目录、小时尺度目录和逐日目录，并返回这些路径。"""
    hourly_plot_dir = CONFIG.output_dir / CONFIG.hourly_plot_subdir
    hourly_table_dir = CONFIG.output_dir / CONFIG.hourly_table_subdir
    daily_plot_dir = CONFIG.output_dir / CONFIG.daily_plot_subdir
    daily_table_dir = CONFIG.output_dir / CONFIG.daily_table_subdir
    daily_combined_plot_dir = CONFIG.output_dir / CONFIG.daily_combined_plot_subdir
    CONFIG.output_dir.mkdir(parents=True, exist_ok=True)
    hourly_plot_dir.mkdir(parents=True, exist_ok=True)
    hourly_table_dir.mkdir(parents=True, exist_ok=True)
    daily_plot_dir.mkdir(parents=True, exist_ok=True)
    daily_table_dir.mkdir(parents=True, exist_ok=True)
    daily_combined_plot_dir.mkdir(parents=True, exist_ok=True)
    return CONFIG.output_dir, hourly_plot_dir, hourly_table_dir, daily_plot_dir, daily_table_dir, daily_combined_plot_dir


def cleanup_temp_dir() -> None:
    """删除本次脚本运行产生的临时缓存目录，避免残留中间文件。"""
    temp_dir = CONFIG.output_dir / CONFIG.temp_dir_name
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


# =============================================================================
# 三、输入检查和数据读取
# =============================================================================
def check_inputs(site_ids: list[str]) -> list[Path]:
    """检查输入路径、站点文件和 ERA5 tif 是否存在，提前暴露缺文件问题。"""
    checks = [
        ("TOMST表格文件夹", CONFIG.tomst_dir.exists()),
        ("样地坐标表", CONFIG.site_csv.exists()),
        ("ERA5 tif文件夹", CONFIG.era5_tif_dir.exists()),
        ("干旱事件长表", CONFIG.drought_event_csv.exists()),
    ]

    with make_bar(
        total=len(checks) + len(site_ids) + 1,
        desc="步骤1/7 输入检查",
        unit="项",
        colour=CONFIG.progress_colours["检查"],
    ) as bar:
        missing_messages: list[str] = []
        for label, exists in checks:
            if not exists:
                missing_messages.append(f"{label}不存在")
            bar.update(1)

        for site_id in site_ids:
            try:
                find_tomst_csv(site_id)
            except FileNotFoundError as exc:
                missing_messages.append(str(exc))
            bar.update(1)

        tif_files = sorted(CONFIG.era5_tif_dir.glob("*.tif")) + sorted(CONFIG.era5_tif_dir.glob("*.tiff"))
        tif_files = [path for path in tif_files if path.is_file()]
        if not tif_files:
            missing_messages.append(f"ERA5 tif文件夹中未找到 tif/tiff 文件：{CONFIG.era5_tif_dir}")
        bar.update(1)

    if missing_messages:
        raise FileNotFoundError("\n".join(missing_messages))
    return tif_files


def read_site_coordinates(site_ids: list[str]) -> pd.DataFrame:
    """读取站点经纬度，并只保留本次需要处理的 27 个站点。"""
    with make_bar(
        total=3,
        desc="步骤2/7 读取坐标",
        unit="项",
        colour=CONFIG.progress_colours["坐标"],
    ) as bar:
        sites = read_csv_with_fallback(CONFIG.site_csv)
        bar.update(1)

        required_cols = [CONFIG.site_id_col, CONFIG.site_lon_col, CONFIG.site_lat_col]
        missing_cols = [col for col in required_cols if col not in sites.columns]
        if missing_cols:
            raise ValueError(f"样地坐标表缺少列：{missing_cols}")

        sites = sites[required_cols].copy()
        sites[CONFIG.site_id_col] = sites[CONFIG.site_id_col].map(normalise_site_id)
        sites[CONFIG.site_lon_col] = pd.to_numeric(sites[CONFIG.site_lon_col], errors="coerce")
        sites[CONFIG.site_lat_col] = pd.to_numeric(sites[CONFIG.site_lat_col], errors="coerce")
        bar.update(1)

        sites = sites[sites[CONFIG.site_id_col].isin(site_ids)].copy()
        sites = sites.dropna(subset=[CONFIG.site_lon_col, CONFIG.site_lat_col])
        missing_site_coords = sorted(set(site_ids) - set(sites[CONFIG.site_id_col]))
        if missing_site_coords:
            raise ValueError(f"坐标表缺少这些站点的有效经纬度：{missing_site_coords}")
        sites = sites.sort_values(CONFIG.site_id_col).reset_index(drop=True)
        bar.update(1)

    return sites


def read_extreme_drought_events(site_ids: list[str]) -> pd.DataFrame:
    """读取每个站点 Extreme 干旱事件，用于逐日折线图浅红色背景标注。"""
    with make_bar(
        total=4,
        desc="步骤3/7 读取干旱事件",
        unit="项",
        colour=CONFIG.progress_colours["干旱事件"],
    ) as bar:
        events = read_csv_with_fallback(CONFIG.drought_event_csv)
        bar.update(1)

        required_cols = [
            CONFIG.drought_site_id_col,
            CONFIG.drought_start_col,
            CONFIG.drought_end_col,
            CONFIG.drought_level_col,
        ]
        missing_cols = [col for col in required_cols if col not in events.columns]
        if missing_cols:
            raise ValueError(f"干旱事件长表缺少列：{missing_cols}")
        bar.update(1)

        events = events[required_cols].copy()
        events[CONFIG.drought_site_id_col] = events[CONFIG.drought_site_id_col].map(normalise_site_id)
        events[CONFIG.drought_start_col] = pd.to_datetime(events[CONFIG.drought_start_col], errors="coerce")
        events[CONFIG.drought_end_col] = pd.to_datetime(events[CONFIG.drought_end_col], errors="coerce")
        events[CONFIG.drought_level_col] = events[CONFIG.drought_level_col].astype(str).str.strip()
        bar.update(1)

        events = events[
            events[CONFIG.drought_site_id_col].isin(site_ids)
            & events[CONFIG.drought_start_col].notna()
            & events[CONFIG.drought_end_col].notna()
            & (events[CONFIG.drought_level_col].str.lower() == CONFIG.extreme_drought_level_name.lower())
        ].copy()
        events = events.sort_values([CONFIG.drought_site_id_col, CONFIG.drought_start_col]).reset_index(drop=True)
        bar.update(1)

    return events


def read_one_tomst_site(site_id: str, csv_path: Path) -> pd.DataFrame:
    """读取单个 TOMST CSV，转换时间和温度，并按北京时间小时求小时平均。"""
    df = read_csv_with_fallback(csv_path)
    required_cols = [CONFIG.tomst_utc_time_col, CONFIG.tomst_temp_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"站点 {site_id} 的 CSV 缺少列：{missing_cols}")

    utc_time = pd.to_datetime(
        df[CONFIG.tomst_utc_time_col],
        format=CONFIG.tomst_time_format,
        errors="coerce",
    )
    temp = pd.to_numeric(df[CONFIG.tomst_temp_col], errors="coerce")
    local_time = utc_time + timedelta(hours=CONFIG.timezone_offset_hours)

    clean = pd.DataFrame(
        {
            "站点编号": site_id,
            "时间_UTC": utc_time,
            "时间_北京时间": local_time,
            "本地日期": local_time.dt.floor("D"),
            "北京时间小时": local_time.dt.floor("h"),
            "TOMST_15cm气温_摄氏度": temp,
        }
    )
    clean = clean.dropna(subset=["时间_UTC", "TOMST_15cm气温_摄氏度"])

    hourly = (
        clean.groupby(["站点编号", "北京时间小时"], as_index=False)
        .agg(
            TOMST_15cm小时平均气温_摄氏度=("TOMST_15cm气温_摄氏度", "mean"),
            TOMST_小时内有效记录数=("TOMST_15cm气温_摄氏度", "count"),
        )
        .sort_values(["站点编号", "北京时间小时"])
    )
    hourly["本地日期"] = hourly["北京时间小时"].dt.floor("D")
    return hourly


def read_tomst_hourly(site_ids: list[str]) -> pd.DataFrame:
    """批量读取 27 个 TOMST CSV，并合并为站点小时尺度温度表。"""
    hourly_frames: list[pd.DataFrame] = []
    with make_bar(
        total=len(site_ids),
        desc="步骤4/7 读取TOMST",
        unit="站",
        colour=CONFIG.progress_colours["TOMST"],
    ) as bar:
        for site_id in site_ids:
            csv_path = find_tomst_csv(site_id)
            hourly_frames.append(read_one_tomst_site(site_id, csv_path))
            bar.update(1)

    if not hourly_frames:
        raise ValueError("没有读取到任何 TOMST 小时尺度数据")
    return pd.concat(hourly_frames, ignore_index=True)


def list_era5_files(tif_files: list[Path]) -> pd.DataFrame:
    """解析 ERA5 tif 文件时间，得到按 UTC 时间排序的文件清单。"""
    records: list[dict[str, object]] = []
    failed_files: list[str] = []

    with make_bar(
        total=len(tif_files),
        desc="步骤5/7 解析ERA5文件",
        unit="个",
        colour=CONFIG.progress_colours["ERA5文件"],
    ) as bar:
        for path in tif_files:
            try:
                utc_time = parse_era5_time_from_name(path)
                if CONFIG.era5_time_is_utc:
                    local_time = utc_time + timedelta(hours=CONFIG.timezone_offset_hours)
                else:
                    local_time = utc_time
                records.append(
                    {
                        "文件路径": path,
                        "时间_UTC": utc_time,
                        "时间_北京时间": local_time,
                        "本地日期": local_time.floor("D"),
                        "北京时间小时": local_time.floor("h"),
                    }
                )
            except ValueError:
                failed_files.append(path.name)
            bar.update(1)

    if failed_files:
        examples = "\n".join(failed_files[:10])
        raise ValueError(
            "以下 ERA5 tif 文件名无法解析时间，请检查 era5_filename_regex 参数。"
            f"\n前10个失败文件：\n{examples}"
        )
    if not records:
        raise ValueError("没有可用的 ERA5 tif 文件")

    era5_files = pd.DataFrame(records).sort_values("时间_UTC").reset_index(drop=True)
    return era5_files


# =============================================================================
# 四、ERA5 站点提取和小时尺度计算
# =============================================================================
def prepare_sample_coordinates(src: rasterio.io.DatasetReader, sites: pd.DataFrame) -> list[tuple[float, float]]:
    """根据 tif 坐标系准备采样坐标，必要时从 WGS84 经纬度转换到 tif 坐标系。"""
    lons = sites[CONFIG.site_lon_col].astype(float).to_numpy()
    lats = sites[CONFIG.site_lat_col].astype(float).to_numpy()
    src_crs = src.crs if src.crs is not None else CONFIG.default_tif_crs

    if str(src_crs).upper() in {"EPSG:4326", "OGC:CRS84"}:
        xs, ys = lons, lats
    else:
        xs, ys = transform_coords("EPSG:4326", src_crs, lons.tolist(), lats.tolist())

    return list(zip(xs, ys))


def extract_values_from_one_tif(path: Path, sites: pd.DataFrame) -> list[float]:
    """从单个 ERA5 tif 中按站点坐标提取温度值，返回与 sites 顺序一致的数值列表。"""
    with rasterio.open(path) as src:
        sample_coords = prepare_sample_coordinates(src, sites)
        nodata = src.nodata
        values: list[float] = []

        for sample in src.sample(sample_coords, indexes=CONFIG.raster_band_index, masked=True):
            value = sample[0]
            if np.ma.is_masked(value):
                values.append(np.nan)
                continue

            value_float = float(value)
            if nodata is not None and np.isclose(value_float, float(nodata)):
                values.append(np.nan)
            elif not np.isfinite(value_float):
                values.append(np.nan)
            else:
                values.append(value_float)

    return values


def extract_era5_hourly(era5_files: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """批量从 ERA5 tif 提取所有站点逐小时温度，保留北京时间小时用于绘图。"""
    site_ids = sites[CONFIG.site_id_col].tolist()
    records: list[dict[str, object]] = []

    with make_bar(
        total=len(era5_files),
        desc="步骤6/7 提取ERA5",
        unit="时",
        colour=CONFIG.progress_colours["ERA5提取"],
    ) as bar:
        for row in era5_files.itertuples(index=False):
            values = extract_values_from_one_tif(row.文件路径, sites)
            for site_id, value in zip(site_ids, values):
                records.append(
                    {
                        "站点编号": site_id,
                        "时间_UTC": row.时间_UTC,
                        "时间_北京时间": row.时间_北京时间,
                        "本地日期": row.本地日期,
                        "北京时间小时": row.北京时间小时,
                        "ERA5_2m小时气温_摄氏度": value,
                    }
                )
            bar.update(1)

    hourly = pd.DataFrame(records)
    hourly = hourly.dropna(subset=["ERA5_2m小时气温_摄氏度"])
    if hourly.empty:
        raise ValueError("ERA5 提取结果为空，请检查站点坐标是否落在 tif 范围内")
    hourly = hourly.sort_values(["站点编号", "北京时间小时"]).reset_index(drop=True)
    return hourly


# =============================================================================
# 五、结果合并、绘图和保存
# =============================================================================
def merge_hourly_tables(tomst_hourly: pd.DataFrame, era5_hourly: pd.DataFrame) -> pd.DataFrame:
    """按站点和北京时间小时合并 TOMST 小时平均值与 ERA5 逐小时值。"""
    merged = pd.merge(
        tomst_hourly,
        era5_hourly,
        on=["站点编号", "北京时间小时"],
        how="outer",
        validate="one_to_one",
        suffixes=("_TOMST", "_ERA5"),
    )
    if "本地日期_TOMST" in merged.columns or "本地日期_ERA5" in merged.columns:
        merged["本地日期"] = merged.get("本地日期_TOMST").combine_first(merged.get("本地日期_ERA5"))
        merged = merged.drop(columns=[col for col in ["本地日期_TOMST", "本地日期_ERA5"] if col in merged.columns])

    if "时间_UTC_TOMST" in merged.columns or "时间_UTC_ERA5" in merged.columns:
        merged["时间_UTC"] = merged.get("时间_UTC_TOMST").combine_first(merged.get("时间_UTC_ERA5"))
    elif "时间_UTC" not in merged.columns:
        merged["时间_UTC"] = pd.NaT

    if "时间_北京时间_TOMST" in merged.columns or "时间_北京时间_ERA5" in merged.columns:
        merged["时间_北京时间"] = merged.get("时间_北京时间_TOMST").combine_first(merged.get("时间_北京时间_ERA5"))
    elif "时间_北京时间" not in merged.columns:
        merged["时间_北京时间"] = merged["北京时间小时"]

    redundant_time_cols = [
        col
        for col in [
            "时间_UTC_TOMST",
            "时间_UTC_ERA5",
            "时间_北京时间_TOMST",
            "时间_北京时间_ERA5",
        ]
        if col in merged.columns
    ]
    if redundant_time_cols:
        merged = merged.drop(columns=redundant_time_cols)

    merged["是否两者都有数据"] = (
        merged["TOMST_15cm小时平均气温_摄氏度"].notna()
        & merged["ERA5_2m小时气温_摄氏度"].notna()
    )
    merged = merged.sort_values(["站点编号", "北京时间小时"]).reset_index(drop=True)
    return merged


def filter_to_output_date_range(merged_hourly: pd.DataFrame) -> pd.DataFrame:
    """只保留指定北京时间日期范围内的小时数据，避免图中出现 2024 或 2026 等其他年份。"""
    start_date = pd.Timestamp(CONFIG.output_start_date)
    end_time = pd.Timestamp(CONFIG.output_end_date) + pd.Timedelta(days=1) - pd.Timedelta(hours=1)

    filtered = merged_hourly.copy()
    filtered["北京时间小时"] = pd.to_datetime(filtered["北京时间小时"])
    filtered = filtered[
        (filtered["北京时间小时"] >= start_date)
        & (filtered["北京时间小时"] <= end_time)
    ].copy()

    if filtered.empty:
        raise ValueError(
            f"筛选 {CONFIG.output_start_date} 到 {CONFIG.output_end_date} 后没有可绘制数据，"
            "请检查 TOMST 与 ERA5 的时间范围。"
        )
    return filtered.reset_index(drop=True)


def add_site_coordinates(data: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """给逐小时表补充站点经纬度信息，使输出表具备完整空间定位字段。"""
    site_coords = sites[
        [CONFIG.site_id_col, CONFIG.site_lon_col, CONFIG.site_lat_col]
    ].copy()
    site_coords = site_coords.rename(
        columns={
            CONFIG.site_id_col: "站点编号",
            CONFIG.site_lon_col: "经度",
            CONFIG.site_lat_col: "纬度",
        }
    )
    site_coords["站点编号"] = site_coords["站点编号"].map(normalise_site_id)
    output = pd.merge(data, site_coords, on="站点编号", how="left", validate="many_to_one")
    return output


def filter_hourly_table_to_output_range(hourly_data: pd.DataFrame) -> pd.DataFrame:
    """把任意逐小时表限制到 2025-01-01 00:00 到 2025-12-31 23:00 的北京时间范围。"""
    start_time = pd.Timestamp(CONFIG.output_start_date)
    end_time = pd.Timestamp(CONFIG.output_end_date) + pd.Timedelta(days=1) - pd.Timedelta(hours=1)

    output = hourly_data.copy()
    output["北京时间小时"] = pd.to_datetime(output["北京时间小时"])
    output = output[
        (output["北京时间小时"] >= start_time)
        & (output["北京时间小时"] <= end_time)
    ].copy()
    output["日期"] = output["北京时间小时"].dt.floor("D")
    return output.sort_values(["站点编号", "北京时间小时"]).reset_index(drop=True)


def prepare_era5_hourly_extract_table(era5_hourly: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """整理 ERA5 逐小时提取表：站点编号、经纬度、UTC 时间、北京时间小时、日期和 2 m 温度。"""
    output = filter_hourly_table_to_output_range(era5_hourly)
    output = add_site_coordinates(output, sites)
    output = output.rename(columns={"ERA5_2m小时气温_摄氏度": "ERA5_2米温度_摄氏度"})
    columns = [
        "站点编号",
        "经度",
        "纬度",
        "时间_UTC",
        "北京时间小时",
        "日期",
        "ERA5_2米温度_摄氏度",
    ]
    return output[columns]


def prepare_hourly_aligned_table(merged_hourly: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """整理逐小时温度对齐表：同一站点同一北京时间小时下的 ERA5 和 TOMST 温度。"""
    output = filter_hourly_table_to_output_range(merged_hourly)
    output = add_site_coordinates(output, sites)
    output = output.rename(
        columns={
            "ERA5_2m小时气温_摄氏度": "ERA5_2米温度_摄氏度",
            "TOMST_15cm小时平均气温_摄氏度": "观测_15厘米温度_摄氏度",
            "TOMST_小时内有效记录数": "观测_小时内有效15分钟记录数",
        }
    )
    columns = [
        "站点编号",
        "经度",
        "纬度",
        "时间_UTC",
        "北京时间小时",
        "日期",
        "ERA5_2米温度_摄氏度",
        "观测_15厘米温度_摄氏度",
        "观测_小时内有效15分钟记录数",
        "是否两者都有数据",
    ]
    return output[columns]


def build_daily_from_hourly(merged_hourly: pd.DataFrame) -> pd.DataFrame:
    """由小时尺度合并表计算逐日均值，供逐日折线图和逐日温度表使用。"""
    daily_source = merged_hourly.copy()
    daily_source["本地日期"] = pd.to_datetime(daily_source["北京时间小时"]).dt.floor("D")

    daily = (
        daily_source.groupby(["站点编号", "本地日期"], as_index=False)
        .agg(
            TOMST_15cm日均气温_摄氏度=("TOMST_15cm小时平均气温_摄氏度", "mean"),
            TOMST_有效小时数=("TOMST_15cm小时平均气温_摄氏度", "count"),
            ERA5_2m日均气温_摄氏度=("ERA5_2m小时气温_摄氏度", "mean"),
            ERA5_有效小时数=("ERA5_2m小时气温_摄氏度", "count"),
        )
        .sort_values(["站点编号", "本地日期"])
        .reset_index(drop=True)
    )
    daily["是否两者都有数据"] = (
        daily["TOMST_15cm日均气温_摄氏度"].notna()
        & daily["ERA5_2m日均气温_摄氏度"].notna()
    )
    return daily


def draw_extreme_background(ax: plt.Axes, site_id: str, drought_events: pd.DataFrame) -> None:
    """在逐日折线图上为本站点 Extreme 干旱事件绘制浅红色背景。"""
    site_events = drought_events[drought_events[CONFIG.drought_site_id_col] == site_id]
    label_added = False
    for _, event in site_events.iterrows():
        start = event[CONFIG.drought_start_col]
        end = event[CONFIG.drought_end_col] + pd.Timedelta(days=1)
        ax.axvspan(
            start,
            end,
            color=CONFIG.extreme_background_color,
            alpha=CONFIG.extreme_background_alpha,
            linewidth=0,
            label=CONFIG.extreme_background_label if not label_added else None,
            zorder=0,
        )
        label_added = True


def plot_one_hourly_site(site_id: str, site_hourly: pd.DataFrame, plot_dir: Path) -> Path:
    """为单个站点绘制 TOMST 微气候与 ERA5 宏气候小时尺度温度折线图。"""
    fig, ax = plt.subplots(figsize=CONFIG.figure_size)

    ax.plot(
        site_hourly["北京时间小时"],
        site_hourly["TOMST_15cm小时平均气温_摄氏度"],
        color=CONFIG.tomst_line_color,
        linewidth=CONFIG.tomst_line_width,
        linestyle=CONFIG.tomst_line_style,
        marker=CONFIG.marker,
        markersize=CONFIG.marker_size,
        alpha=CONFIG.line_alpha,
        label=CONFIG.tomst_label,
    )
    ax.plot(
        site_hourly["北京时间小时"],
        site_hourly["ERA5_2m小时气温_摄氏度"],
        color=CONFIG.era5_line_color,
        linewidth=CONFIG.era5_line_width,
        linestyle=CONFIG.era5_line_style,
        marker=CONFIG.marker,
        markersize=CONFIG.marker_size,
        alpha=CONFIG.line_alpha,
        label=CONFIG.era5_label,
    )

    ax.set_title(f"站点{site_id} 微气候与宏气候小时尺度气温对比", fontsize=CONFIG.title_fontsize)
    ax.set_xlabel("北京时间", fontsize=CONFIG.axis_label_fontsize)
    ax.set_ylabel(CONFIG.y_axis_label, fontsize=CONFIG.axis_label_fontsize)
    ax.legend(loc=CONFIG.legend_loc, fontsize=CONFIG.legend_fontsize)
    ax.tick_params(axis="both", labelsize=CONFIG.tick_label_fontsize)

    if CONFIG.grid_visible:
        ax.grid(True, linestyle=CONFIG.grid_line_style, alpha=CONFIG.grid_alpha)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=CONFIG.month_tick_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(CONFIG.date_tick_format))
    ax.set_xlim(
        pd.Timestamp(CONFIG.output_start_date),
        pd.Timestamp(CONFIG.output_end_date) + pd.Timedelta(days=1) - pd.Timedelta(hours=1),
    )
    fig.autofmt_xdate(rotation=35, ha="right")
    fig.tight_layout()

    output_path = plot_dir / f"站点{site_id}_微气候与宏气候小时尺度气温对比图.png"
    fig.savefig(output_path, dpi=CONFIG.figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_one_daily_site(
    site_id: str,
    site_daily: pd.DataFrame,
    drought_events: pd.DataFrame,
    plot_dir: Path,
) -> Path:
    """为单个站点绘制逐日折线图，并为 Extreme 干旱事件增加浅红色背景。"""
    fig, ax = plt.subplots(figsize=CONFIG.figure_size)

    draw_extreme_background(ax, site_id, drought_events)
    ax.plot(
        site_daily["本地日期"],
        site_daily["TOMST_15cm日均气温_摄氏度"],
        color=CONFIG.tomst_line_color,
        linewidth=CONFIG.tomst_line_width,
        linestyle=CONFIG.tomst_line_style,
        marker=CONFIG.marker,
        markersize=CONFIG.marker_size,
        alpha=CONFIG.line_alpha,
        label=CONFIG.tomst_label,
        zorder=2,
    )
    ax.plot(
        site_daily["本地日期"],
        site_daily["ERA5_2m日均气温_摄氏度"],
        color=CONFIG.era5_line_color,
        linewidth=CONFIG.era5_line_width,
        linestyle=CONFIG.era5_line_style,
        marker=CONFIG.marker,
        markersize=CONFIG.marker_size,
        alpha=CONFIG.line_alpha,
        label=CONFIG.era5_label,
        zorder=2,
    )

    ax.set_title(f"站点{site_id} 微气候与宏气候逐日气温对比", fontsize=CONFIG.title_fontsize)
    ax.set_xlabel("日期", fontsize=CONFIG.axis_label_fontsize)
    ax.set_ylabel(CONFIG.daily_y_axis_label, fontsize=CONFIG.axis_label_fontsize)
    ax.legend(loc=CONFIG.legend_loc, fontsize=CONFIG.legend_fontsize)
    ax.tick_params(axis="both", labelsize=CONFIG.tick_label_fontsize)

    if CONFIG.grid_visible:
        ax.grid(True, linestyle=CONFIG.grid_line_style, alpha=CONFIG.grid_alpha)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=CONFIG.month_tick_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(CONFIG.date_tick_format))
    ax.set_xlim(pd.Timestamp(CONFIG.output_start_date), pd.Timestamp(CONFIG.output_end_date))
    fig.autofmt_xdate(rotation=35, ha="right")
    fig.tight_layout()

    output_path = plot_dir / f"站点{site_id}_微气候与宏气候逐日气温对比图_含Extreme干旱背景.png"
    fig.savefig(output_path, dpi=CONFIG.figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def chunk_list(values: list[str], chunk_size: int) -> list[list[str]]:
    """把站点列表按固定数量分组，用于把 20 个 Extreme 站点拆成两张 10 子图合并图。"""
    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


def subplot_grid(site_count: int) -> tuple[int, int]:
    """根据子图数量选择紧凑布局：7 个站点默认 4x2，10 个站点默认 5x2。"""
    if site_count <= 1:
        return 1, 1
    if site_count <= 4:
        return 2, 2
    if site_count <= 8:
        return 4, 2
    return 5, 2


def plot_daily_combined_group(
    group_title: str,
    output_name: str,
    group_site_ids: list[str],
    merged_daily: pd.DataFrame,
    drought_events: pd.DataFrame,
    output_dir: Path,
    draw_drought_background: bool,
) -> Path:
    """把多个站点的逐日折线图合并到一张多子图图片中。"""
    if not group_site_ids:
        raise ValueError(f"{group_title} 没有可绘制站点")

    nrows, ncols = subplot_grid(len(group_site_ids))
    fig_width = CONFIG.combined_subplot_size[0] * ncols
    fig_height = CONFIG.combined_subplot_size[1] * nrows
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=False,
    )
    axes_array = np.atleast_1d(axes).ravel()

    for ax, site_id in zip(axes_array, group_site_ids):
        site_daily = merged_daily[merged_daily["站点编号"] == site_id].copy()
        site_daily["本地日期"] = pd.to_datetime(site_daily["本地日期"])

        if draw_drought_background:
            draw_extreme_background(ax, site_id, drought_events)

        ax.plot(
            site_daily["本地日期"],
            site_daily["TOMST_15cm日均气温_摄氏度"],
            color=CONFIG.tomst_line_color,
            linewidth=CONFIG.tomst_line_width,
            linestyle=CONFIG.tomst_line_style,
            marker=CONFIG.marker,
            markersize=CONFIG.marker_size,
            alpha=CONFIG.line_alpha,
            label=CONFIG.tomst_label,
            zorder=2,
        )
        ax.plot(
            site_daily["本地日期"],
            site_daily["ERA5_2m日均气温_摄氏度"],
            color=CONFIG.era5_line_color,
            linewidth=CONFIG.era5_line_width,
            linestyle=CONFIG.era5_line_style,
            marker=CONFIG.marker,
            markersize=CONFIG.marker_size,
            alpha=CONFIG.line_alpha,
            label=CONFIG.era5_label,
            zorder=2,
        )

        ax.set_title(f"Site {site_id}", fontsize=CONFIG.axis_label_fontsize)
        ax.set_xlim(pd.Timestamp(CONFIG.output_start_date), pd.Timestamp(CONFIG.output_end_date))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=CONFIG.month_tick_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter(CONFIG.date_tick_format))
        ax.tick_params(axis="both", labelsize=CONFIG.tick_label_fontsize)
        if CONFIG.grid_visible:
            ax.grid(True, linestyle=CONFIG.grid_line_style, alpha=CONFIG.grid_alpha)

    for ax in axes_array[len(group_site_ids) :]:
        ax.set_visible(False)

    handles, labels = axes_array[0].get_legend_handles_labels()
    fig.suptitle(group_title, fontsize=CONFIG.title_fontsize, y=0.992)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=min(3, len(handles)),
        fontsize=CONFIG.legend_fontsize,
        frameon=True,
    )
    fig.supxlabel("Date", fontsize=CONFIG.axis_label_fontsize)
    fig.supylabel("Daily mean temperature (°C)", fontsize=CONFIG.axis_label_fontsize)
    fig.autofmt_xdate(rotation=35, ha="right")
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.92))

    output_path = output_dir / output_name
    fig.savefig(output_path, dpi=CONFIG.figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_daily_combined_plots(
    merged_daily: pd.DataFrame,
    drought_events: pd.DataFrame,
    site_ids: list[str],
    output_dir: Path,
) -> int:
    """按是否出现 Extreme 干旱事件输出逐日折线合并图，返回输出图片数量。"""
    extreme_site_ids = sorted(set(drought_events[CONFIG.drought_site_id_col].astype(str)))
    extreme_site_ids = [site_id for site_id in site_ids if site_id in extreme_site_ids]
    non_extreme_site_ids = [site_id for site_id in site_ids if site_id not in set(extreme_site_ids)]

    output_count = 0
    if non_extreme_site_ids:
        plot_daily_combined_group(
            group_title="Daily Temperature Comparison at Sites without Extreme Drought",
            output_name="未出现Extreme干旱站点_逐日折线合并图.png",
            group_site_ids=non_extreme_site_ids,
            merged_daily=merged_daily,
            drought_events=drought_events,
            output_dir=output_dir,
            draw_drought_background=False,
        )
        output_count += 1

    for group_index, group_site_ids in enumerate(chunk_list(extreme_site_ids, 10), start=1):
        plot_daily_combined_group(
            group_title=f"Daily Temperature Comparison at Sites with Extreme Drought - Group {group_index}",
            output_name=f"出现Extreme干旱站点_逐日折线合并图_第{group_index}组.png",
            group_site_ids=group_site_ids,
            merged_daily=merged_daily,
            drought_events=drought_events,
            output_dir=output_dir,
            draw_drought_background=True,
        )
        output_count += 1

    return output_count


def write_parameter_report(output_dir: Path, site_count: int, era5_count: int) -> Path:
    """保存运行参数说明，方便后续复现实验和调整图例、折线等绘图参数。"""
    report_path = output_dir / CONFIG.parameter_report_name
    lines = [
        "宏微气候温度时间序列脚本运行参数说明",
        "",
        f"处理站点数量：{site_count}",
        f"ERA5 tif 文件数量：{era5_count}",
        "",
        "参数列表：",
    ]
    for field in fields(CONFIG):
        value = getattr(CONFIG, field.name)
        lines.append(f"{field.name}: {value}")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def save_outputs(
    era5_hourly: pd.DataFrame,
    merged_hourly: pd.DataFrame,
    merged_daily: pd.DataFrame,
    drought_events: pd.DataFrame,
    sites: pd.DataFrame,
    site_ids: list[str],
    era5_count: int,
) -> None:
    """输出小时尺度和逐日两套表格、折线图，以及参数说明文件。"""
    (
        output_dir,
        hourly_plot_dir,
        hourly_table_dir,
        daily_plot_dir,
        daily_table_dir,
        daily_combined_plot_dir,
    ) = output_paths()

    with make_bar(
        total=len(site_ids) * 2 + 6,
        desc="步骤7/7 绘图输出",
        unit="项",
        colour=CONFIG.progress_colours["绘图输出"],
    ) as bar:
        for site_id in site_ids:
            site_hourly = merged_hourly[merged_hourly["站点编号"] == site_id].copy()
            site_hourly["北京时间小时"] = pd.to_datetime(site_hourly["北京时间小时"])
            site_hourly.to_csv(
                hourly_table_dir / f"站点{site_id}{CONFIG.hourly_per_site_table_suffix}",
                index=False,
                encoding=CONFIG.output_csv_encoding,
            )
            plot_one_hourly_site(site_id, site_hourly, hourly_plot_dir)
            bar.update(1)

        for site_id in site_ids:
            site_daily = merged_daily[merged_daily["站点编号"] == site_id].copy()
            site_daily["本地日期"] = pd.to_datetime(site_daily["本地日期"])
            site_daily.to_csv(
                daily_table_dir / f"站点{site_id}{CONFIG.daily_per_site_table_suffix}",
                index=False,
                encoding=CONFIG.output_csv_encoding,
            )
            plot_one_daily_site(site_id, site_daily, drought_events, daily_plot_dir)
            bar.update(1)

        merged_hourly.to_csv(
            hourly_table_dir / CONFIG.hourly_all_sites_table_name,
            index=False,
            encoding=CONFIG.output_csv_encoding,
        )
        bar.update(1)

        era5_hourly_extract_table = prepare_era5_hourly_extract_table(era5_hourly, sites)
        era5_hourly_extract_table.to_csv(
            hourly_table_dir / CONFIG.era5_hourly_extract_table_name,
            index=False,
            encoding=CONFIG.output_csv_encoding,
        )
        bar.update(1)

        hourly_aligned_table = prepare_hourly_aligned_table(merged_hourly, sites)
        hourly_aligned_table.to_csv(
            hourly_table_dir / CONFIG.hourly_aligned_table_name,
            index=False,
            encoding=CONFIG.output_csv_encoding,
        )
        bar.update(1)

        merged_daily.to_csv(
            daily_table_dir / CONFIG.daily_all_sites_table_name,
            index=False,
            encoding=CONFIG.output_csv_encoding,
        )
        bar.update(1)

        save_daily_combined_plots(merged_daily, drought_events, site_ids, daily_combined_plot_dir)
        bar.update(1)

        write_parameter_report(output_dir, site_count=len(site_ids), era5_count=era5_count)
        bar.update(1)


def print_runtime_parameters(site_ids: list[str]) -> None:
    """打印关键参数，尤其是图例、折线、横轴刻度等后续常调参数。"""
    print("\n当前运行参数")
    print(f"TOMST表格文件夹：{CONFIG.tomst_dir}")
    print(f"样地坐标表：{CONFIG.site_csv}")
    print(f"ERA5 tif文件夹：{CONFIG.era5_tif_dir}")
    print(f"干旱事件长表：{CONFIG.drought_event_csv}")
    print(f"输出目录：{CONFIG.output_dir}")
    print(f"处理站点数：{len(site_ids)}")
    print(f"排除站点：{CONFIG.excluded_site_ids}")
    print(f"时间处理：TOMST data_time 和 ERA5 文件名时间均按 UTC 处理，再转换为 UTC+{CONFIG.timezone_offset_hours} 按小时尺度对齐")
    print(f"输出日期范围：{CONFIG.output_start_date} 到 {CONFIG.output_end_date}（北京时间日期）")
    print(f"ERA5温度单位：{CONFIG.era5_temperature_unit}")
    print(f"横轴刻度：每隔 {CONFIG.month_tick_interval} 个月显示一次，格式 {CONFIG.date_tick_format}")
    print(f"TOMST折线：颜色 {CONFIG.tomst_line_color}，线宽 {CONFIG.tomst_line_width}，图例 {CONFIG.tomst_label}")
    print(f"ERA5折线：颜色 {CONFIG.era5_line_color}，线宽 {CONFIG.era5_line_width}，图例 {CONFIG.era5_label}")
    print(f"逐日图Extreme干旱背景：颜色 {CONFIG.extreme_background_color}，透明度 {CONFIG.extreme_background_alpha}")
    print(f"图片尺寸：{CONFIG.figure_size}，DPI：{CONFIG.figure_dpi}\n")


# =============================================================================
# 六、主流程
# =============================================================================
def main() -> None:
    """脚本主入口，按检查、读取、提取、合并、绘图、清理的顺序执行。"""
    configure_matplotlib()
    site_ids = expected_site_ids()
    print_runtime_parameters(site_ids)

    try:
        CONFIG.output_dir.mkdir(parents=True, exist_ok=True)
        (CONFIG.output_dir / CONFIG.temp_dir_name).mkdir(parents=True, exist_ok=True)

        tif_files = check_inputs(site_ids)
        sites = read_site_coordinates(site_ids)
        drought_events = read_extreme_drought_events(site_ids)
        tomst_hourly = read_tomst_hourly(site_ids)
        era5_files = list_era5_files(tif_files)
        era5_hourly = extract_era5_hourly(era5_files, sites)
        merged_hourly = merge_hourly_tables(tomst_hourly, era5_hourly)
        merged_hourly = filter_to_output_date_range(merged_hourly)
        merged_daily = build_daily_from_hourly(merged_hourly)
        save_outputs(
            era5_hourly,
            merged_hourly,
            merged_daily,
            drought_events,
            sites,
            site_ids,
            era5_count=len(era5_files),
        )

        print("\n处理完成")
        print(f"输出目录：{CONFIG.output_dir}")
        print(f"小时尺度折线图：{CONFIG.output_dir / CONFIG.hourly_plot_subdir}")
        print(f"小时尺度温度表：{CONFIG.output_dir / CONFIG.hourly_table_subdir}")
        print(f"逐日折线图：{CONFIG.output_dir / CONFIG.daily_plot_subdir}")
        print(f"逐日温度表：{CONFIG.output_dir / CONFIG.daily_table_subdir}")
        print(f"逐日折线合并图：{CONFIG.output_dir / CONFIG.daily_combined_plot_subdir}")
    finally:
        cleanup_temp_dir()


if __name__ == "__main__":
    main()
