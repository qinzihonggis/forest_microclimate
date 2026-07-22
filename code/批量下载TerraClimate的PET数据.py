# -*- coding: utf-8 -*-
"""
批量下载 TerraClimate PET 年度 NetCDF 数据。

脚本功能：
1. 从 THREDDS Data Server 的 HTTPServer 接口下载 TerraClimate PET 数据。
2. 年份范围默认为 1990-2023，每个年份对应一个 NetCDF 文件。
3. 下载结果保存到 E:\forest_microclimate\ForestMicroclimate\PET_TerraClimate。
4. 已经存在的目标文件会自动跳过，避免重复下载。
5. 下载过程中先写入 .part 临时文件，下载完成后再改名为正式 .nc 文件。
6. 如果下载中断，.part 文件会保留下来；下次运行时会删除旧 .part 并重新下载该年份文件，
   不做复杂断点续传，避免残缺文件被误认为完整数据。

说明：
1. 本脚本不安装任何依赖。
2. 下载功能使用 Python 标准库 urllib，进度条使用你已安装的 tqdm。
3. TerraClimate 文件名当前使用官方命名格式：TerraClimate_pet_年份.nc。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tqdm import tqdm


# =============================================================================
# 一、可调整参数区
# =============================================================================
# THREDDS 目录页面。
# 用意：
# 1. 这个地址是你截图中展示 PET 年度文件列表的 catalog 页面。
# 2. 脚本实际下载时不需要逐页点击网页，而是根据该目录对应的 HTTPServer 规则直接拼接下载地址。
# 3. 如果以后服务器目录发生变化，可以先打开该页面核对文件名和 HTTPServer 路径是否仍然一致。
CATALOG_URL = (
    "http://thredds.northwestknowledge.net:8080/thredds/catalog/"
    "TERRACLIMATE_ALL/data/catalog.html"
)

# HTTPServer 直接下载地址的基础部分。
# 用意：
# 1. 你截图第二张中 "2. HTTPServer:" 后面的链接就是这种 fileServer 地址。
# 2. 后面只需要追加文件名，例如 TerraClimate_pet_2014.nc，就能得到完整下载链接。
# 3. 如果将来要下载其他变量，例如 tmax、tmin、ppt，只需要在文件名模板中修改变量名。
HTTP_FILESERVER_BASE_URL = (
    "http://thredds.northwestknowledge.net:8080/thredds/fileServer/"
    "TERRACLIMATE_ALL/data"
)

# 下载年份范围。
# 用意：
# 1. START_YEAR 是开始年份，END_YEAR 是结束年份，两端都包含。
# 2. 当前设置会下载 1990、1991、...、2023，共 34 个年度 NetCDF 文件。
# 3. 如果以后只想下载某一段年份，可以直接修改这两个参数。
START_YEAR = 1990
END_YEAR = 2024

# 下载文件名模板。
# 用意：
# 1. TerraClimate PET 官方文件名格式为 TerraClimate_pet_年份.nc。
# 2. {year} 会在程序运行时替换成具体年份。
# 3. 不建议随意修改 pet 和下划线格式，除非官方文件名发生变化。
FILE_NAME_TEMPLATE = "TerraClimate_pet_{year}.nc"

# 输出目录。
# 用意：
# 1. 所有下载后的 .nc 文件都会保存到这个目录。
# 2. 如果目录不存在，脚本会自动创建。
# 3. 路径前面的 r 表示原始字符串，Windows 反斜杠不需要额外转义。
OUTPUT_DIR = Path(r"E:\forest_microclimate\ForestMicroclimate\PET_TerraClimate")

# 单次网络读取的数据块大小，单位为字节。
# 用意：
# 1. 1024 * 1024 表示每次读取 1 MB，适合 100 MB 左右的 NetCDF 文件。
# 2. 数值较大通常下载效率更高，但进度条刷新频率会降低。
# 3. 数值较小进度条更细腻，但会增加一些循环和刷新开销。
CHUNK_SIZE = 1024 * 1024

# 网络请求超时时间，单位为秒。
# 用意：
# 1. 如果服务器长时间没有响应，超过该时间后会报错并进入失败列表。
# 2. 网络较慢或服务器繁忙时，可以适当增大该值。
# 3. 该参数不是整个文件下载的总时长限制，而是单次网络连接或读取的等待限制。
REQUEST_TIMEOUT = 120

# User-Agent 请求头。
# 用意：
# 1. 明确告诉服务器这是 Python 脚本下载请求。
# 2. 某些服务器会对没有 User-Agent 的请求返回异常响应。
# 3. 一般不需要修改。
USER_AGENT = "Python TerraClimate PET Downloader"

# tqdm 进度条统一显示格式。
# 用意：
# 1. percentage 显示百分比。
# 2. n_fmt/total_fmt 显示当前量和总量。
# 3. elapsed/remaining/rate_fmt 显示已用时间、预计剩余时间和速度。
# 4. bar:32 控制进度条主体宽度，dynamic_ncols=True 会让它适应终端宽度。
PROGRESS_BAR_FORMAT = (
    "{l_bar}{bar:32}| {percentage:3.0f}% {n_fmt}/{total_fmt} "
    "[已用 {elapsed} < 剩余 {remaining}, {rate_fmt}]"
)

# 当服务器没有返回 Content-Length 时使用的进度条格式。
# 用意：
# 1. 没有总字节数时，tqdm 无法计算百分比和剩余时间。
# 2. 这种情况下仍然显示已下载量、耗时和速度，保证下载过程可观察。
# 3. 正常情况下 TerraClimate 服务器会返回 Content-Length，因此大多数下载仍会显示百分比。
UNKNOWN_TOTAL_BAR_FORMAT = (
    "{l_bar}{bar:32}| {n_fmt} "
    "[已用 {elapsed}, {rate_fmt}]"
)

# 不同类型进度条的颜色。
# 用意：
# 1. 不同步骤使用不同颜色，便于在终端中快速区分当前正在做什么。
# 2. tqdm 常见可用颜色包括 blue、green、cyan、magenta、yellow、red、white。
# 3. 如果你的终端不支持颜色，tqdm 会自动退化为普通进度条显示。
PREPARE_BAR_COLOR = "yellow"
CHECK_BAR_COLOR = "blue"
OVERALL_BAR_COLOR = "cyan"
DOWNLOAD_BAR_COLOR = "green"
SKIP_BAR_COLOR = "magenta"


# =============================================================================
# 二、工具函数区
# =============================================================================
def make_bar(
    total: Optional[int],
    desc: str,
    unit: str,
    colour: str,
    unit_scale: bool = False,
    unit_divisor: int = 1000,
) -> tqdm:
    """
    创建统一样式的 tqdm 彩色进度条。

    参数说明：
    total:
        当前步骤的总任务量。对于文件数量，传入整数；对于下载字节数，传入 Content-Length。
        如果服务器没有返回文件大小，可以传入 None，此时 tqdm 会显示已下载量和速度，但无法准确显示百分比。
    desc:
        进度条左侧显示的步骤名称，例如“准备清单”“检查文件”“下载 2014”。
    unit:
        计量单位，例如 file、year、B。下载字节时使用 B，并配合 unit_scale=True。
    colour:
        进度条颜色。不同步骤传入不同颜色，便于区分进度条类型。
    unit_scale:
        是否自动缩放单位。下载字节时设为 True，可以显示 KiB、MiB 等可读单位。
    unit_divisor:
        单位换算基数。下载文件时使用 1024，符合文件大小显示习惯。
    """
    bar_format = UNKNOWN_TOTAL_BAR_FORMAT if total is None else PROGRESS_BAR_FORMAT

    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        colour=colour,
        unit_scale=unit_scale,
        unit_divisor=unit_divisor,
        dynamic_ncols=True,
        bar_format=bar_format,
    )


def ensure_output_dir(output_dir: Path) -> None:
    """
    创建输出目录。

    参数说明：
    output_dir:
        保存 TerraClimate PET 文件的目录。

    处理逻辑：
    1. 如果目录已经存在，不会删除或覆盖其中任何文件。
    2. 如果目录不存在，则自动创建完整目录层级。
    3. 使用 exist_ok=True 是为了让重复运行脚本更加安全。
    """
    with make_bar(1, "创建输出目录", "step", PREPARE_BAR_COLOR) as bar:
        output_dir.mkdir(parents=True, exist_ok=True)
        bar.update(1)


def build_download_tasks() -> List[Dict[str, Any]]:
    """
    根据年份范围生成下载任务清单。

    返回值说明：
    每个任务都是一个字典，包含：
    1. year: 年份，例如 2014。
    2. file_name: 官方文件名，例如 TerraClimate_pet_2014.nc。
    3. url: HTTPServer 直接下载链接。
    4. target_path: 下载完成后的正式 .nc 文件路径。
    5. part_path: 下载过程中的 .part 临时文件路径。

    这样设计的用意：
    1. 下载前先统一生成任务，后续检查、下载、汇总都基于同一份任务清单。
    2. 文件名和 URL 都由参数区统一控制，后续修改年份或变量名更方便。
    """
    years = list(range(START_YEAR, END_YEAR + 1))
    tasks: List[Dict[str, Any]] = []

    with make_bar(len(years), "准备下载清单", "year", PREPARE_BAR_COLOR) as bar:
        for year in years:
            file_name = FILE_NAME_TEMPLATE.format(year=year)
            target_path = OUTPUT_DIR / file_name
            part_path = target_path.with_suffix(target_path.suffix + ".part")

            tasks.append(
                {
                    "year": year,
                    "file_name": file_name,
                    "url": f"{HTTP_FILESERVER_BASE_URL}/{file_name}",
                    "target_path": target_path,
                    "part_path": part_path,
                }
            )
            bar.update(1)

    return tasks


def split_existing_and_pending(
    tasks: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    检查哪些文件已经存在，哪些文件仍需下载。

    参数说明：
    tasks:
        build_download_tasks() 生成的完整年度任务清单。

    返回值说明：
    1. existing_tasks: 目标 .nc 文件已经存在的任务，后续直接跳过。
    2. pending_tasks: 目标 .nc 文件不存在的任务，后续需要下载。

    处理逻辑：
    1. 只要正式 .nc 文件存在，就认为该年份已经下载完成，不重复下载。
    2. .part 文件不作为完成依据，因为它可能是中断后留下的残缺临时文件。
    3. 该检查步骤单独显示进度条，方便确认脚本正在逐年判断文件状态。
    """
    existing_tasks: List[Dict[str, Any]] = []
    pending_tasks: List[Dict[str, Any]] = []

    with make_bar(len(tasks), "检查已有文件", "file", CHECK_BAR_COLOR) as bar:
        for task in tasks:
            target_path = task["target_path"]
            if not isinstance(target_path, Path):
                raise TypeError("target_path 必须是 pathlib.Path 类型。")

            if target_path.exists():
                existing_tasks.append(task)
            else:
                pending_tasks.append(task)

            bar.update(1)

    return existing_tasks, pending_tasks


def show_skipped_files(existing_tasks: List[Dict[str, Any]]) -> None:
    """
    显示已存在文件的跳过进度。

    参数说明：
    existing_tasks:
        split_existing_and_pending() 返回的已存在任务列表。

    用意：
    1. 已存在文件不会下载，但它们仍然是总任务的一部分。
    2. 单独显示“跳过已有文件”进度条，能清楚看到跳过步骤也已经完成。
    3. 如果没有已存在文件，则不显示该进度条，避免输出无意义信息。
    """
    if not existing_tasks:
        return

    with make_bar(len(existing_tasks), "跳过已有文件", "file", SKIP_BAR_COLOR) as bar:
        for task in existing_tasks:
            year = task["year"]
            bar.set_postfix_str(f"{year}")
            bar.update(1)


def get_content_length(response) -> Optional[int]:
    """
    从 HTTP 响应头读取文件大小。

    参数说明：
    response:
        urllib.request.urlopen() 返回的 HTTP 响应对象。

    返回值说明：
    1. 如果服务器返回 Content-Length，则返回整数形式的字节数。
    2. 如果服务器没有返回或返回值异常，则返回 None。

    用意：
    1. 有文件大小时，tqdm 可以显示百分比和预计剩余时间。
    2. 没有文件大小时，脚本仍能下载，只是进度条无法准确计算百分比。
    """
    content_length = response.headers.get("Content-Length")
    if content_length is None:
        return None

    try:
        return int(content_length)
    except ValueError:
        return None


def download_one_file(task: Dict[str, Any]) -> None:
    """
    下载单个年份的 TerraClimate PET 文件。

    参数说明：
    task:
        单个下载任务，必须包含 year、file_name、url、target_path、part_path。

    下载策略：
    1. 正式文件不存在时才调用本函数。
    2. 如果旧的 .part 文件存在，说明上次下载可能中断，本次先删除旧 .part 再重新下载。
    3. 新数据先写入 .part 文件，全部下载完成并通过大小检查后，再改名为正式 .nc 文件。
    4. 如果下载途中报错或用户中断，.part 文件会保留在目录中，便于识别该年份曾经下载失败。
    """
    year = task["year"]
    url = task["url"]
    target_path = task["target_path"]
    part_path = task["part_path"]

    if not isinstance(year, int):
        raise TypeError("year 必须是 int 类型。")
    if not isinstance(url, str):
        raise TypeError("url 必须是 str 类型。")
    if not isinstance(target_path, Path):
        raise TypeError("target_path 必须是 pathlib.Path 类型。")
    if not isinstance(part_path, Path):
        raise TypeError("part_path 必须是 pathlib.Path 类型。")

    if part_path.exists():
        part_path.unlink()

    request = Request(url, headers={"User-Agent": USER_AGENT})

    with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        total_size = get_content_length(response)
        downloaded_size = 0

        with part_path.open("wb") as file_obj:
            with make_bar(
                total=total_size,
                desc=f"下载 {year}",
                unit="B",
                colour=DOWNLOAD_BAR_COLOR,
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break

                    file_obj.write(chunk)
                    downloaded_size += len(chunk)
                    bar.update(len(chunk))

        if total_size is not None and downloaded_size != total_size:
            raise IOError(
                f"{year} 年文件下载不完整：已下载 {downloaded_size} 字节，"
                f"服务器声明大小为 {total_size} 字节。"
            )

    part_path.replace(target_path)


def download_pending_files(tasks: List[Dict[str, Any]]) -> List[str]:
    """
    按年份顺序下载所有待下载文件。

    参数说明：
    tasks:
        目标 .nc 文件尚不存在、需要下载的任务列表。

    返回值说明：
    failures:
        下载失败信息列表。为空表示全部待下载文件都成功完成。

    处理逻辑：
    1. 外层“总体下载进度”显示年度文件层面的完成情况。
    2. 内层“下载 年份”显示单个文件的字节级下载进度。
    3. 某一年失败后不会立刻停止，而是记录错误并继续尝试后续年份。
    4. 全部尝试结束后，如果存在失败项，再统一抛出 SystemExit(1)。
    """
    failures: List[str] = []

    if not tasks:
        return failures

    with make_bar(len(tasks), "总体下载进度", "file", OVERALL_BAR_COLOR) as overall_bar:
        for task in tasks:
            year = task["year"]
            overall_bar.set_postfix_str(f"{year}")

            try:
                download_one_file(task)
            except HTTPError as exc:
                failures.append(f"{year} 年：HTTP {exc.code}，{exc.reason}")
            except URLError as exc:
                failures.append(f"{year} 年：网络连接失败，{exc.reason}")
            except Exception as exc:
                failures.append(f"{year} 年：{exc}")
            finally:
                overall_bar.update(1)

    return failures


def print_summary(
    total_count: int,
    skipped_count: int,
    downloaded_count: int,
    failures: List[str],
) -> None:
    """
    打印最终汇总信息。

    参数说明：
    total_count:
        年度任务总数。
    skipped_count:
        因正式 .nc 文件已存在而跳过的数量。
    downloaded_count:
        本次成功下载的数量，不包含已跳过文件。
    failures:
        下载失败信息列表。

    用意：
    1. 进度条适合观察过程，汇总信息适合运行结束后核对结果。
    2. 如果失败列表不为空，会逐条列出具体年份和错误原因。
    3. 失败时脚本会以非 0 状态退出，便于后续批处理或日志系统识别异常。
    """
    print("\n下载任务汇总：")
    print(f"  总任务数：{total_count}")
    print(f"  已存在并跳过：{skipped_count}")
    print(f"  本次成功下载：{downloaded_count}")
    print(f"  下载失败：{len(failures)}")
    print(f"  保存目录：{OUTPUT_DIR}")
    print(f"  目录页面：{CATALOG_URL}")

    if failures:
        print("\n失败详情：")
        for failure in failures:
            print(f"  - {failure}")


# =============================================================================
# 三、主程序入口
# =============================================================================
def main() -> None:
    """
    主流程。

    步骤说明：
    1. 创建输出目录。
    2. 根据年份范围生成下载任务清单。
    3. 检查哪些年份文件已经存在。
    4. 对已存在文件显示跳过进度。
    5. 对不存在文件逐年下载，并显示总体进度和单文件字节进度。
    6. 打印最终汇总；如果有失败任务，以非 0 状态退出。
    """
    ensure_output_dir(OUTPUT_DIR)
    tasks = build_download_tasks()
    existing_tasks, pending_tasks = split_existing_and_pending(tasks)
    show_skipped_files(existing_tasks)

    failures = download_pending_files(pending_tasks)
    downloaded_count = len(pending_tasks) - len(failures)

    print_summary(
        total_count=len(tasks),
        skipped_count=len(existing_tasks),
        downloaded_count=downloaded_count,
        failures=failures,
    )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
