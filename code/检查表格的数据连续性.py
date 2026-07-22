from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from tqdm import tqdm


@dataclass(frozen=True)
class Config:
    # =========================
    # 1. 输入、输出路径参数
    # =========================
    # data_dir：原始 CSV 表格所在文件夹。脚本不会修改原始 CSV，只会读取这些文件。
    data_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\Tensor_Data")

    # file_ids：需要处理的文件编号范围。95332241.csv 不存在，因此在这里主动跳过。
    file_ids: tuple[int, ...] = tuple(
        file_id for file_id in range(95332217, 95332245) if file_id != 95332241
    )

    # output_folder_name：输出文件夹名称。按你的要求，默认与脚本文件名相同，不包含 .py 后缀。
    output_folder_name: str = Path(__file__).stem

    # =========================
    # 2. 数据列与连续性检查参数
    # =========================
    # time_column：用于检查连续性的时间列。该列应是北京时间/UTC+8 后的时间序列。
    time_column: str = "data_time8"

    # temperature_column：用于绘制折线图的 15 cm 林下空气温度列。
    temperature_column: str = "T3_15"

    # time_format：data_time8 的时间格式。例如：2024.10.31 18:45。
    # 如果源数据时间格式变化，只需要同步修改这里。
    time_format: str = "%Y.%m.%d %H:%M"

    # interval_minutes：数据应保持的固定时间间隔。这里按你的要求设置为 15 分钟。
    interval_minutes: int = 15

    # csv_encoding：CSV 读写编码。utf-8-sig 能兼容带 BOM 的 Excel CSV，也方便中文表头正常打开。
    csv_encoding: str = "utf-8-sig"

    # csv_newline：保持 csv 模块推荐的 newline=""，避免 Windows 下写出多余空行。
    csv_newline: str = ""

    # =========================
    # 3. 进度条参数
    # =========================
    # chunk_size：读取文件时每次读取的字节数。越大速度可能越快，越小字节级进度条刷新越细。
    chunk_size: int = 1024 * 1024

    # tqdm_ncols：进度条显示宽度。如果终端显示换行，可以适当调小。
    tqdm_ncols: int = 110

    # 颜色只影响支持彩色的终端，不影响计算结果。
    tqdm_colour_step: str = "cyan"
    tqdm_colour_file: str = "green"
    tqdm_colour_byte: str = "yellow"

    # =========================
    # 4. 折线图参数：后续主要改这里即可调整图例、折线、坐标轴、清晰度
    # =========================
    # figure_size：图片宽高，单位是英寸。时间跨度长时可增大宽度。
    figure_size: tuple[float, float] = (16.0, 6.0)

    # figure_dpi：图片分辨率。数值越大越清晰，文件也会更大。
    figure_dpi: int = 180

    # line_color：折线颜色。
    line_color: str = "#1F6F50"

    # line_width：折线粗细。数值越大线越粗。
    line_width: float = 1.0

    # line_alpha：折线透明度，1.0 为完全不透明。
    line_alpha: float = 0.9

    # marker_style：数据点标记样式。"." 表示小点；如果不想显示点，可改为 ""。
    marker_style: str = "."

    # marker_size：数据点大小。数据很多时建议保持较小。
    marker_size: float = 2.0

    # legend_label：图例文字。图例位置、字号等参数见下面几项。
    legend_label: str = "15cm林下空气温度"
    legend_location: str = "best"
    legend_fontsize: int = 10

    # title_fontsize、label_fontsize、tick_fontsize：标题、坐标轴标题、刻度文字字号。
    title_fontsize: int = 15
    label_fontsize: int = 12
    tick_fontsize: int = 9

    # x_date_format：横坐标时间显示格式。\n 会让日期和时刻分两行显示，避免标签互相重叠。
    x_date_format: str = "%Y-%m-%d\n%H:%M"

    # x_major_minticks / x_major_maxticks：横坐标主刻度数量范围。
    # 图太挤时可降低 maxticks，想显示更密时可增大 maxticks。
    x_major_minticks: int = 5
    x_major_maxticks: int = 12

    # grid_alpha：网格线透明度。0 表示不明显，1 表示完全不透明。
    grid_alpha: float = 0.25

    # output_image_format：图像格式。按你的要求使用 png。
    output_image_format: str = "png"


CONFIG = Config()


def build_file_paths(config: Config) -> list[Path]:
    """根据编号列表生成 27 个 CSV 文件路径。"""
    return [config.data_dir / f"{file_id}.csv" for file_id in config.file_ids]


def build_output_dir(config: Config) -> Path:
    """输出文件夹固定建在原始 CSV 数据目录下，并与脚本名称保持一致。"""
    return config.data_dir / config.output_folder_name


def print_runtime_parameters(config: Config, output_dir: Path, file_paths: list[Path]) -> None:
    """打印关键参数，方便运行前后核对，也方便后续按参数区注释调整脚本。"""
    print("运行参数")
    print(f"数据目录: {config.data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"CSV 文件数量: {len(file_paths)}")
    print(f"文件编号范围: {config.file_ids[0]} - {config.file_ids[-1]}，跳过 95332241")
    print(f"连续性检查时间列: {config.time_column}")
    print(f"绘图温度列: {config.temperature_column}")
    print(f"时间格式: {config.time_format}")
    print(f"连续时间间隔: {config.interval_minutes} 分钟")
    print(f"CSV 编码: {config.csv_encoding}")
    print(f"字节级读取块大小: {config.chunk_size} bytes")
    print("绘图参数")
    print(f"图片尺寸: {config.figure_size}")
    print(f"图片 DPI: {config.figure_dpi}")
    print(f"折线颜色: {config.line_color}")
    print(f"折线粗细: {config.line_width}")
    print(f"折线透明度: {config.line_alpha}")
    print(f"数据点样式: {config.marker_style}")
    print(f"数据点大小: {config.marker_size}")
    print(f"图例文字: {config.legend_label}")
    print(f"图例位置: {config.legend_location}")
    print(f"图例字号: {config.legend_fontsize}")
    print(f"横坐标时间格式: {config.x_date_format}")
    print("")


def write_parameter_note(config: Config, output_dir: Path) -> None:
    """把主要可调参数另存为中文说明文件，方便不打开代码也能查看绘图和检查设置。"""
    note_path = output_dir / "参数说明.txt"
    lines = [
        "参数说明",
        "",
        "一、连续性检查参数",
        f"时间列：{config.time_column}",
        f"温度列：{config.temperature_column}",
        f"时间格式：{config.time_format}",
        f"连续间隔：{config.interval_minutes} 分钟",
        "",
        "二、进度条参数",
        f"字节级读取块大小：{config.chunk_size} bytes",
        f"进度条宽度：{config.tqdm_ncols}",
        "",
        "三、折线图参数",
        f"图片尺寸：{config.figure_size}",
        f"图片 DPI：{config.figure_dpi}",
        f"折线颜色：{config.line_color}",
        f"折线粗细：{config.line_width}",
        f"折线透明度：{config.line_alpha}",
        f"数据点样式：{config.marker_style}",
        f"数据点大小：{config.marker_size}",
        f"图例文字：{config.legend_label}",
        f"图例位置：{config.legend_location}",
        f"图例字号：{config.legend_fontsize}",
        f"横坐标时间格式：{config.x_date_format}",
        "",
        "如需调整图像样式，优先修改脚本顶部 Config 中的折线图参数。",
    ]
    note_path.write_text("\n".join(lines), encoding=config.csv_encoding)


def configure_matplotlib() -> None:
    """设置中文字体候选和负号显示，避免中文标题、图例或 ℃ 单位乱码。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def read_csv_with_byte_progress(file_path: Path, config: Config) -> list[dict[str, str]]:
    """
    按字节读取 CSV，并显示每个文件的字节级进度条。

    这样做的目的：即使单个 CSV 文件很大，也能看到当前文件读取到多少字节。
    读取完成后再交给 csv.DictReader 解析为字典行，便于按列名访问 data_time8 和 T3_15。
    """
    total_bytes = file_path.stat().st_size
    buffer = bytearray()

    with (
        file_path.open("rb") as file_handle,
        tqdm(
            total=total_bytes,
            desc=f"读取字节 {file_path.name}",
            unit="B",
            unit_scale=True,
            ncols=config.tqdm_ncols,
            colour=config.tqdm_colour_byte,
            leave=False,
        ) as progress,
    ):
        while True:
            chunk = file_handle.read(config.chunk_size)
            if not chunk:
                break
            buffer.extend(chunk)
            progress.update(len(chunk))

    text_stream = io.StringIO(buffer.decode(config.csv_encoding))
    return list(csv.DictReader(text_stream))


def parse_time(time_text: str, config: Config) -> datetime | None:
    """把 data_time8 文本解析为 datetime；解析失败时返回 None，后续写入异常明细表。"""
    cleaned_text = (time_text or "").strip()
    if not cleaned_text:
        return None

    try:
        return datetime.strptime(cleaned_text, config.time_format)
    except ValueError:
        return None


def parse_temperature(temperature_text: str) -> float | None:
    """把 T3_15 文本解析为浮点温度；空值或非数字返回 None。"""
    cleaned_text = (temperature_text or "").strip()
    if not cleaned_text:
        return None

    try:
        return float(cleaned_text)
    except ValueError:
        return None


def format_time(value: datetime | None, config: Config) -> str:
    """统一输出时间文本，保证所有结果表时间格式一致。"""
    if value is None:
        return ""
    return value.strftime(config.time_format)


def build_missing_times(sorted_unique_times: list[datetime], config: Config) -> list[datetime]:
    """
    根据最早和最晚时间生成完整的 15 分钟时间序列，再找出源数据缺少的时间点。

    注意：这里检查的是单个文件内部从最早时间到最晚时间之间是否连续，
    不跨文件补齐，也不假定所有文件起止时间完全相同。
    """
    if not sorted_unique_times:
        return []

    expected_delta = timedelta(minutes=config.interval_minutes)
    existing_times = set(sorted_unique_times)
    current_time = sorted_unique_times[0]
    end_time = sorted_unique_times[-1]
    missing_times: list[datetime] = []

    while current_time <= end_time:
        if current_time not in existing_times:
            missing_times.append(current_time)
        current_time += expected_delta

    return missing_times


def analyze_file(
    file_path: Path,
    rows: list[dict[str, str]],
    config: Config,
) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[tuple[datetime, float]]]:
    """
    对单个 CSV 完成连续性检查，并返回汇总记录、缺失明细、重复明细、异常明细和绘图数据。

    关键检查项：
    1. data_time8 是否能按指定格式解析。
    2. T3_15 是否能解析为数值。
    3. data_time8 是否按 15 分钟连续。
    4. data_time8 是否有重复时间点。
    5. 原始行顺序中的 data_time8 是否出现倒序。
    """
    abnormal_rows: list[dict[str, str]] = []
    valid_times: list[datetime] = []
    plot_points: list[tuple[datetime, float]] = []
    row_numbers_by_time: dict[datetime, list[int]] = defaultdict(list)
    previous_time: datetime | None = None
    reverse_order_count = 0
    valid_temperature_count = 0

    for data_index, row in enumerate(rows, start=2):
        raw_time = row.get(config.time_column, "")
        raw_temperature = row.get(config.temperature_column, "")
        parsed_time = parse_time(raw_time, config)
        parsed_temperature = parse_temperature(raw_temperature)

        if parsed_time is None:
            abnormal_rows.append(
                {
                    "文件名": file_path.name,
                    "行号": str(data_index),
                    "异常类型": "时间无法解析或为空",
                    "原始时间": raw_time,
                    "原始温度": raw_temperature,
                    "说明": f"{config.time_column} 需要符合格式 {config.time_format}",
                }
            )
        else:
            valid_times.append(parsed_time)
            row_numbers_by_time[parsed_time].append(data_index)

            if previous_time is not None and parsed_time < previous_time:
                reverse_order_count += 1
                abnormal_rows.append(
                    {
                        "文件名": file_path.name,
                        "行号": str(data_index),
                        "异常类型": "时间倒序",
                        "原始时间": raw_time,
                        "原始温度": raw_temperature,
                        "说明": "当前行时间早于上一条可解析时间",
                    }
                )
            previous_time = parsed_time

        if parsed_temperature is None:
            abnormal_rows.append(
                {
                    "文件名": file_path.name,
                    "行号": str(data_index),
                    "异常类型": "温度无法解析或为空",
                    "原始时间": raw_time,
                    "原始温度": raw_temperature,
                    "说明": f"{config.temperature_column} 需要是数值",
                }
            )
        else:
            valid_temperature_count += 1

        if parsed_time is not None and parsed_temperature is not None:
            plot_points.append((parsed_time, parsed_temperature))

    sorted_unique_times = sorted(set(valid_times))
    missing_times = build_missing_times(sorted_unique_times, config)
    time_counter = Counter(valid_times)

    missing_rows = [
        {
            "文件名": file_path.name,
            "缺失时间": missing_time.strftime(config.time_format),
            "要求间隔分钟": str(config.interval_minutes),
        }
        for missing_time in missing_times
    ]

    duplicate_rows = [
        {
            "文件名": file_path.name,
            "重复时间": repeated_time.strftime(config.time_format),
            "重复次数": str(repeated_count),
            "所在行号": "、".join(str(row_number) for row_number in row_numbers_by_time[repeated_time]),
        }
        for repeated_time, repeated_count in sorted(time_counter.items())
        if repeated_count > 1
    ]

    expected_count = 0
    if sorted_unique_times:
        total_minutes = int((sorted_unique_times[-1] - sorted_unique_times[0]).total_seconds() // 60)
        expected_count = total_minutes // config.interval_minutes + 1

    continuity_ok = (
        bool(sorted_unique_times)
        and not missing_times
        and not duplicate_rows
        and reverse_order_count == 0
        and len(valid_times) == len(rows)
    )

    summary_row = {
        "文件名": file_path.name,
        "总行数": str(len(rows)),
        "可解析时间数量": str(len(valid_times)),
        "可解析温度数量": str(valid_temperature_count),
        "开始时间": format_time(sorted_unique_times[0] if sorted_unique_times else None, config),
        "结束时间": format_time(sorted_unique_times[-1] if sorted_unique_times else None, config),
        "理论时间点数量": str(expected_count),
        "实际唯一时间点数量": str(len(sorted_unique_times)),
        "缺失时间点数量": str(len(missing_times)),
        "重复时间点数量": str(len(duplicate_rows)),
        "时间倒序次数": str(reverse_order_count),
        "异常记录数量": str(len(abnormal_rows)),
        "连续性结论": "连续" if continuity_ok else "不连续或存在异常",
    }

    return summary_row, missing_rows, duplicate_rows, abnormal_rows, sorted(plot_points)


def write_csv_table(path: Path, rows: list[dict[str, str]], fieldnames: list[str], config: Config) -> None:
    """写出中文命名 CSV 表格；即使没有明细，也保留表头，方便确认该项无异常。"""
    with path.open("w", encoding=config.csv_encoding, newline=config.csv_newline) as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_temperature_line(
    file_path: Path,
    plot_points: list[tuple[datetime, float]],
    output_dir: Path,
    config: Config,
) -> None:
    """
    绘制单个 CSV 的 15 cm 林下空气温度折线图。

    横坐标：data_time8 时间序列。
    纵坐标：T3_15，单位 ℃。
    图例、折线、点、字体、坐标轴刻度等参数都集中在 Config 的“折线图参数”区域。
    """
    fig, ax = plt.subplots(figsize=config.figure_size, dpi=config.figure_dpi)

    if plot_points:
        x_values = [point_time for point_time, _ in plot_points]
        y_values = [temperature for _, temperature in plot_points]
        ax.plot(
            x_values,
            y_values,
            color=config.line_color,
            linewidth=config.line_width,
            alpha=config.line_alpha,
            marker=config.marker_style,
            markersize=config.marker_size,
            label=config.legend_label,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "没有可绘制的有效数据",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=config.label_fontsize,
        )

    ax.set_title(f" microclimate temperature", fontsize=config.title_fontsize)
    ax.set_xlabel("Time", fontsize=config.label_fontsize)
    ax.set_ylabel("Microclimate temperature（℃）", fontsize=config.label_fontsize)
    ax.grid(True, alpha=config.grid_alpha)
    ax.legend(loc=config.legend_location, fontsize=config.legend_fontsize)
    ax.tick_params(axis="both", labelsize=config.tick_fontsize)

    date_locator = mdates.AutoDateLocator(
        minticks=config.x_major_minticks,
        maxticks=config.x_major_maxticks,
    )
    ax.xaxis.set_major_locator(date_locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(config.x_date_format))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()

    image_path = output_dir / f"{file_path.stem}_林下15cm空气温度折线图.{config.output_image_format}"
    fig.savefig(image_path)
    plt.close(fig)


def validate_source_files(file_paths: list[Path]) -> None:
    """运行前检查目标 CSV 是否存在，避免处理到一半才发现文件缺失。"""
    missing_files = [file_path.name for file_path in file_paths if not file_path.exists()]
    if missing_files:
        raise FileNotFoundError("以下 CSV 文件不存在：" + "、".join(missing_files))


def main() -> None:
    """脚本主流程：准备输出目录、读取数据、检查连续性、写表格、绘图。"""
    config = CONFIG
    file_paths = build_file_paths(config)
    output_dir = build_output_dir(config)

    print_runtime_parameters(config, output_dir, file_paths)
    configure_matplotlib()

    step_names = [
        "检查输入文件并创建输出目录",
        "读取 CSV 并检查数据连续性",
        "写出中文结果表格",
        "绘制 27 张单文件折线图",
        "写出参数说明文件",
    ]

    with tqdm(
        total=len(step_names),
        desc="关键步骤",
        ncols=config.tqdm_ncols,
        colour=config.tqdm_colour_step,
    ) as step_progress:
        # 步骤 1：检查输入文件完整性，并创建输出目录。
        # 输出目录名称与脚本名称相同，所有结果文件集中保存在这里。
        validate_source_files(file_paths)
        output_dir.mkdir(parents=True, exist_ok=True)
        step_progress.update(1)

        # 步骤 2：逐个读取 CSV，读取时显示字节级进度条，随后按 15 分钟规则检查连续性。
        summary_rows: list[dict[str, str]] = []
        missing_rows: list[dict[str, str]] = []
        duplicate_rows: list[dict[str, str]] = []
        abnormal_rows: list[dict[str, str]] = []
        plot_data_by_file: dict[Path, list[tuple[datetime, float]]] = {}

        for file_path in tqdm(
            file_paths,
            desc="读取并检查文件",
            ncols=config.tqdm_ncols,
            colour=config.tqdm_colour_file,
        ):
            rows = read_csv_with_byte_progress(file_path, config)
            summary_row, file_missing_rows, file_duplicate_rows, file_abnormal_rows, plot_points = analyze_file(
                file_path=file_path,
                rows=rows,
                config=config,
            )
            summary_rows.append(summary_row)
            missing_rows.extend(file_missing_rows)
            duplicate_rows.extend(file_duplicate_rows)
            abnormal_rows.extend(file_abnormal_rows)
            plot_data_by_file[file_path] = plot_points
        step_progress.update(1)

        # 步骤 3：写出检查结果表格。
        # 表格均使用中文文件名和中文表头，方便直接用 Excel 或 WPS 查看。
        write_csv_table(
            output_dir / "数据连续性检查汇总表.csv",
            summary_rows,
            [
                "文件名",
                "总行数",
                "可解析时间数量",
                "可解析温度数量",
                "开始时间",
                "结束时间",
                "理论时间点数量",
                "实际唯一时间点数量",
                "缺失时间点数量",
                "重复时间点数量",
                "时间倒序次数",
                "异常记录数量",
                "连续性结论",
            ],
            config,
        )
        write_csv_table(
            output_dir / "缺失时间点明细表.csv",
            missing_rows,
            ["文件名", "缺失时间", "要求间隔分钟"],
            config,
        )
        write_csv_table(
            output_dir / "重复时间点明细表.csv",
            duplicate_rows,
            ["文件名", "重复时间", "重复次数", "所在行号"],
            config,
        )
        write_csv_table(
            output_dir / "异常数据明细表.csv",
            abnormal_rows,
            ["文件名", "行号", "异常类型", "原始时间", "原始温度", "说明"],
            config,
        )
        step_progress.update(1)

        # 步骤 4：为每个 CSV 单独绘制一张折线图，不生成合并图。
        # 绘图时只使用 data_time8 与 T3_15 都能解析成功的记录。
        for file_path in tqdm(
            file_paths,
            desc="绘制折线图",
            ncols=config.tqdm_ncols,
            colour=config.tqdm_colour_file,
        ):
            plot_temperature_line(file_path, plot_data_by_file[file_path], output_dir, config)
        step_progress.update(1)

        # 步骤 5：写出参数说明文件，记录本次脚本使用的连续性检查和绘图参数。
        write_parameter_note(config, output_dir)
        step_progress.update(1)

    print(f"处理完成。结果已保存到：{output_dir}")


if __name__ == "__main__":
    main()
