from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from tqdm import tqdm


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\Tensor_Data")
    template_file: str = "95332217.csv"
    file_ids: tuple[int, ...] = tuple(
        file_id for file_id in range(95332217, 95332245) if file_id != 95332241
    )
    source_time_column: str = "data_time"
    target_time_column: str = "data_time8"
    summary_file_name: str = "起始时间.csv"
    filter_start_time: str = "2025.01.20 00:00"
    utc_offset_hours: int = 8
    input_time_format: str = "%Y.%m.%d %H:%M"
    output_time_format: str = "%Y.%m.%d %H:%M"
    input_date_format: str = "%Y.%m.%d"
    input_clock_format: str = "%H:%M"
    csv_encoding: str = "utf-8-sig"
    csv_newline: str = ""
    chunk_size: int = 1024 * 1024
    tqdm_ncols: int = 110
    tqdm_unit: str = "B"
    tqdm_unit_scale: bool = True
    tqdm_colour_step: str = "cyan"
    tqdm_colour_file: str = "green"
    tqdm_colour_byte: str = "yellow"
    step_desc: str = "Steps"
    file_desc: str = "Files"
    read_desc_prefix: str = "Read"
    write_desc_prefix: str = "Write"


CONFIG = Config()


def build_file_paths(config: Config) -> list[Path]:
    return [config.data_dir / f"{file_id}.csv" for file_id in config.file_ids]


def get_file_size(file_path: Path) -> int:
    return file_path.stat().st_size


def parse_time_to_utc8(time_text: str, config: Config) -> str:
    utc_time = parse_input_time(time_text, config)
    utc8_time = utc_time + timedelta(hours=config.utc_offset_hours)
    return utc8_time.strftime(config.output_time_format)


def parse_input_time(time_text: str, config: Config) -> datetime:
    return datetime.strptime(time_text.strip(), config.input_time_format)


def parse_output_time(time_text: str, config: Config) -> datetime:
    return datetime.strptime(time_text.strip(), config.output_time_format)


def matches_time_format(text: str, time_format: str) -> bool:
    try:
        datetime.strptime(text.strip(), time_format)
        return True
    except ValueError:
        return False


def normalize_data_row(row: list[str], expected_columns: int, config: Config) -> list[str]:
    if len(row) == expected_columns:
        return row

    # Some source files split data_time into separate date and clock columns.
    if len(row) == expected_columns + 1 and len(row) >= 3:
        date_part = row[1].strip()
        clock_part = row[2].strip()
        if (
            matches_time_format(date_part, config.input_date_format)
            and matches_time_format(clock_part, config.input_clock_format)
        ):
            merged_row = [row[0], f"{date_part} {clock_part}", *row[3:]]
            if len(merged_row) == expected_columns:
                return merged_row

    return row


def print_runtime_parameters(config: Config, template_header: list[str], file_paths: list[Path]) -> None:
    print("Runtime parameters")
    print(f"data_dir: {config.data_dir}")
    print(f"template_file: {config.template_file}")
    print(f"file_count: {len(file_paths)}")
    print(f"file_id_range: {config.file_ids[0]} - {config.file_ids[-1]} (skip 95332241)")
    print(f"source_time_column: {config.source_time_column}")
    print(f"target_time_column: {config.target_time_column}")
    print(f"summary_file_name: {config.summary_file_name}")
    print(f"filter_start_time: {config.filter_start_time}")
    print(f"utc_offset_hours: +{config.utc_offset_hours}")
    print(f"input_time_format: {config.input_time_format}")
    print(f"output_time_format: {config.output_time_format}")
    print(f"input_date_format: {config.input_date_format}")
    print(f"input_clock_format: {config.input_clock_format}")
    print(f"csv_encoding: {config.csv_encoding}")
    print(f"chunk_size_bytes: {config.chunk_size}")
    print(f"step_progress_label: {config.step_desc}")
    print(f"file_progress_label: {config.file_desc}")
    print(f"read_progress_prefix: {config.read_desc_prefix}")
    print(f"write_progress_prefix: {config.write_desc_prefix}")
    print(f"template_header: {template_header}")
    print("")


def load_template_header(template_path: Path, config: Config) -> list[str]:
    with template_path.open("r", encoding=config.csv_encoding, newline=config.csv_newline) as fh:
        reader = csv.reader(fh)
        header = next(reader)

    if config.target_time_column not in header:
        return [*header, config.target_time_column]
    return header


def is_existing_header(row: list[str], config: Config) -> bool:
    return config.source_time_column in row and "ID" in row


def read_csv_with_progress(file_path: Path, config: Config) -> list[list[str]]:
    total_bytes = get_file_size(file_path)
    buffer = bytearray()

    with (
        file_path.open("rb") as fh,
        tqdm(
            total=total_bytes,
            desc=f"{config.read_desc_prefix} {file_path.name}",
            unit=config.tqdm_unit,
            unit_scale=config.tqdm_unit_scale,
            ncols=config.tqdm_ncols,
            colour=config.tqdm_colour_byte,
            leave=False,
        ) as progress,
    ):
        while True:
            chunk = fh.read(config.chunk_size)
            if not chunk:
                break
            buffer.extend(chunk)
            progress.update(len(chunk))

    text_stream = io.StringIO(buffer.decode(config.csv_encoding))
    return list(csv.reader(text_stream))


def build_output_rows(
    input_rows: list[list[str]],
    template_header: list[str],
    file_path: Path,
    config: Config,
) -> tuple[list[list[str]], str, str]:
    output_rows: list[list[str]] = []
    data_time8_values: list[datetime] = []
    filter_start_time = parse_input_time(config.filter_start_time, config)

    if not input_rows:
        output_rows.append(template_header)
        return output_rows, "", ""

    first_row = input_rows[0]
    has_header = is_existing_header(first_row, config)
    data_rows = input_rows[1:] if has_header else input_rows

    if has_header and config.target_time_column in first_row:
        output_rows.append(first_row)
        effective_header = first_row
    else:
        output_rows.append(template_header)
        effective_header = template_header

    source_header = effective_header[:-1] if effective_header[-1] == config.target_time_column else effective_header

    for row in data_rows:
        normalized_row = normalize_data_row(row, len(source_header), config)
        row_map = dict(zip(source_header, normalized_row))
        source_time_text = row_map.get(config.source_time_column, "").strip()
        if source_time_text and parse_input_time(source_time_text, config) < filter_start_time:
            continue
        row_map[config.target_time_column] = (
            parse_time_to_utc8(source_time_text, config) if source_time_text else ""
        )
        if row_map[config.target_time_column]:
            data_time8_values.append(parse_output_time(row_map[config.target_time_column], config))
        output_rows.append([row_map.get(column, "") for column in effective_header])

    start_time = min(data_time8_values).strftime(config.output_time_format) if data_time8_values else ""
    end_time = max(data_time8_values).strftime(config.output_time_format) if data_time8_values else ""

    if file_path.name != config.template_file and not has_header:
        return output_rows, start_time, end_time

    if effective_header == template_header:
        return output_rows, start_time, end_time

    normalized_rows = [template_header]
    for row in output_rows[1:]:
        row_map = dict(zip(effective_header, row))
        normalized_rows.append([row_map.get(column, "") for column in template_header])
    return normalized_rows, start_time, end_time


def write_csv_temp_file(temp_path: Path, rows: list[list[str]], config: Config) -> None:
    with temp_path.open("w", encoding=config.csv_encoding, newline=config.csv_newline) as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)


def copy_file_with_progress(source: Path, target: Path, config: Config) -> None:
    total_bytes = get_file_size(source)
    with (
        source.open("rb") as src,
        target.open("wb") as dst,
        tqdm(
            total=total_bytes,
            desc=f"{config.write_desc_prefix} {target.name}",
            unit=config.tqdm_unit,
            unit_scale=config.tqdm_unit_scale,
            ncols=config.tqdm_ncols,
            colour=config.tqdm_colour_byte,
            leave=False,
        ) as progress,
    ):
        while True:
            chunk = src.read(config.chunk_size)
            if not chunk:
                break
            dst.write(chunk)
            progress.update(len(chunk))


def process_single_file(file_path: Path, template_header: list[str], config: Config) -> dict[str, str]:
    input_rows = read_csv_with_progress(file_path, config)
    output_rows, start_time, end_time = build_output_rows(input_rows, template_header, file_path, config)

    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    write_csv_temp_file(temp_path, output_rows, config)
    copy_file_with_progress(temp_path, file_path, config)
    temp_path.unlink(missing_ok=True)

    return {
        "name": file_path.name,
        "start": start_time,
        "end": end_time,
    }


def write_summary_file(summary_rows: list[dict[str, str]], config: Config) -> None:
    summary_path = config.data_dir / config.summary_file_name
    with summary_path.open("w", encoding=config.csv_encoding, newline=config.csv_newline) as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "start", "end"])
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    config = CONFIG
    file_paths = build_file_paths(config)

    step_progress = tqdm(
        total=4,
        desc=config.step_desc,
        ncols=config.tqdm_ncols,
        colour=config.tqdm_colour_step,
    )

    template_path = config.data_dir / config.template_file
    template_header = load_template_header(template_path, config)
    step_progress.update(1)

    print_runtime_parameters(config, template_header, file_paths)
    step_progress.update(1)

    summary_rows: list[dict[str, str]] = []
    with tqdm(
        total=len(file_paths),
        desc=config.file_desc,
        ncols=config.tqdm_ncols,
        colour=config.tqdm_colour_file,
    ) as file_progress:
        for file_path in file_paths:
            file_progress.set_postfix_str(file_path.name)
            summary_rows.append(process_single_file(file_path, template_header, config))
            file_progress.update(1)

    step_progress.update(1)
    write_summary_file(summary_rows, config)
    step_progress.update(1)
    step_progress.close()
    print("Done.")


if __name__ == "__main__":
    main()
