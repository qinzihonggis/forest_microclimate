# -*- coding: utf-8 -*-
"""
批量下载 CHIRPS pentad 降雨 NetCDF 数据。

脚本功能：
1. 从 CHIRPS 官方目录下载逐年 pentad 降雨数据。
2. 默认下载 1990-2024 年，每一年对应一个 .nc 文件。
3. 下载结果保存到 E:\forest_microclimate\ForestMicroclimate\Precipitation_CHIRPS。
4. 如果目标 .nc 文件已经存在，自动跳过，不重复下载。
5. 下载过程中先写入 .part 临时文件，完整下载后再改名为正式 .nc 文件。
6. 下载失败会自动重试；重试仍失败时继续下载下一年，最后统一汇总成功、跳过、失败年份。

说明：
1. 本脚本不安装任何依赖。
2. 下载功能使用 Python 标准库 urllib，终端进度条使用你已经安装好的 tqdm。
3. CHIRPS 文件名格式为 chirps-v2.0.年份.pentads.nc，例如 chirps-v2.0.1990.pentads.nc。
"""

import ssl
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tqdm import tqdm


# =============================================================================
# 一、可调整参数区
# =============================================================================
# CHIRPS pentad NetCDF 目录地址。
# 参数说明：
# 1. 这是网页目录地址，浏览器打开后可以看到所有年度 .nc 文件列表。
# 2. 本脚本不需要解析网页目录，而是根据 CHIRPS 固定文件名规则直接拼接下载链接。
# 3. 如果官方目录将来改变，只需要优先检查这个基础目录是否仍然正确。
BASE_URL = "http://data.chc.ucsb.edu/products/CHIRPS-2.0/global_pentad/netcdf"

# 下载年份范围。
# 参数说明：
# 1. START_YEAR 是起始年份，END_YEAR 是结束年份，两端都包含。
# 2. 当前设置会下载 1990、1991、...、2024，共 35 个年度 NetCDF 文件。
# 3. 如果以后只想下载某一段年份，直接修改这两个参数即可。
START_YEAR = 1990
END_YEAR = 2024

# CHIRPS 官方文件名模板。
# 参数说明：
# 1. {year} 会在程序运行时替换为具体年份。
# 2. 例如 year=1990 时，文件名为 chirps-v2.0.1990.pentads.nc。
# 3. 除非 CHIRPS 官方文件名规则改变，否则不要修改 v2.0、pentads、nc 等字段。
FILE_NAME_TEMPLATE = "chirps-v2.0.{year}.pentads.nc"

# 输出目录。
# 参数说明：
# 1. 所有下载后的 .nc 文件都会保存到该目录。
# 2. 如果目录不存在，脚本运行时会自动创建。
# 3. 路径前面的 r 表示原始字符串，Windows 反斜杠不需要额外转义。
OUTPUT_DIR = Path(r"E:\forest_microclimate\ForestMicroclimate\Precipitation_CHIRPS")

# 单次读取的数据块大小，单位为字节。
# 参数说明：
# 1. 1024 * 1024 表示每次从网络读取 1 MiB 数据。
# 2. 数值越大，通常下载效率越高，但进度条刷新频率会降低。
# 3. 如果网络不稳定或希望进度条更细，可以适当调小，例如 512 * 1024。
CHUNK_SIZE = 1024 * 1024

# 网络请求超时时间，单位为秒。
# 参数说明：
# 1. 服务器长时间无响应时，超过该时间会触发异常并进入重试逻辑。
# 2. 该参数不是整个文件下载的总时长限制，而是单次连接或读取等待时间。
# 3. 如果你的网络较慢，可以适当增大，例如 180 或 300。
REQUEST_TIMEOUT = 120

# 最大下载尝试次数。
# 参数说明：
# 1. 1 表示每个年份只尝试下载 1 次，不重复重试，避免失败时反复刷屏和浪费时间。
# 2. 某一年失败后，脚本会记录失败原因并继续处理下一年。
# 3. 如果以后希望网络失败时自动重试，可以改成 2 或 3。
MAX_RETRIES = 1

# User-Agent 请求头。
# 参数说明：
# 1. 明确告诉服务器这是 Python 脚本下载请求。
# 2. 某些服务器会对没有 User-Agent 的请求返回异常响应。
# 3. 一般不需要修改。
USER_AGENT = "Python CHIRPS Pentad Downloader"

# 是否关闭 HTTPS 证书校验。
# 参数说明：
# 1. True 表示不校验 HTTPS 证书，用于解决 CHIRPS 服务器证书过期导致的下载失败。
# 2. 该设置只影响本脚本通过 urllib 访问下载链接时的 SSL 校验，不会修改系统或 Python 环境。
# 3. 如果将来 CHIRPS 服务器证书恢复正常，并且你希望严格校验证书，可以改为 False。
DISABLE_SSL_VERIFY = True

# HTTPS 连接使用的 SSL 上下文。
# 参数说明：
# 1. 当 DISABLE_SSL_VERIFY=True 时，创建不校验证书的上下文，绕过 certificate has expired 报错。
# 2. 当 DISABLE_SSL_VERIFY=False 时，使用 None，让 urllib 使用默认的安全证书校验。
# 3. 这里集中定义，后续所有 urlopen 下载请求都统一使用该设置。
SSL_CONTEXT = ssl._create_unverified_context() if DISABLE_SSL_VERIFY else None

# tqdm 进度条统一显示格式。
# 参数说明：
# 1. percentage 显示百分比。
# 2. n_fmt/total_fmt 显示当前量/总量。
# 3. elapsed/remaining/rate_fmt 显示已用时间、剩余时间和速度。
# 4. bar:32 控制进度条主体宽度，dynamic_ncols=True 会让进度条适配终端宽度。
PROGRESS_BAR_FORMAT = (
    "{l_bar}{bar:32}| {percentage:3.0f}% {n_fmt}/{total_fmt} "
    "[已用 {elapsed} < 剩余 {remaining}, {rate_fmt}]"
)

# 服务器没有返回 Content-Length 时使用的进度条格式。
# 参数说明：
# 1. 没有总字节数时，tqdm 无法计算准确百分比和剩余时间。
# 2. 这种情况下仍然显示已下载量、耗时和速度，保证下载过程可观察。
# 3. CHIRPS 服务器通常会返回 Content-Length，因此大多数下载会显示百分比。
UNKNOWN_TOTAL_BAR_FORMAT = "{l_bar}{bar:32}| {n_fmt} [已用 {elapsed}, {rate_fmt}]"

# 不同类型进度条的颜色。
# 参数说明：
# 1. 只保留“总体下载进度”和“当前文件下载进度”两类进度条，减少终端刷新开销。
# 2. tqdm 常见颜色包括 blue、green、cyan、magenta、yellow、red、white。
# 3. 如果终端或 tqdm 版本不支持颜色，脚本会自动退回普通进度条显示。
OVERALL_BAR_COLOR = "cyan"
DOWNLOAD_BAR_COLOR = "green"

# tqdm 最小刷新间隔，单位为秒。
# 参数说明：
# 1. 2.0 表示同一个进度条最多约每 2 秒刷新一次，避免频繁刷屏影响下载速度。
# 2. 如果你希望显示更实时，可以调小到 0.5；如果更重视速度和终端整洁，可以调大到 2.0。
# 3. 该参数只影响显示刷新频率，不影响实际下载逻辑和保存结果。
PROGRESS_MIN_INTERVAL = 2.0


# =============================================================================
# 二、进度条工具函数
# =============================================================================
def make_bar(
    total: Optional[int],
    desc: str,
    unit: str,
    colour: str,
    unit_scale: bool = False,
    unit_divisor: int = 1000,
    leave: bool = True,
) -> tqdm:
    """
    创建统一样式的 tqdm 彩色进度条。

    参数说明：
    total:
        当前步骤的总任务量。
        对于年份数、文件数、重试次数，传入整数。
        对于文件下载字节数，传入服务器返回的 Content-Length。
        如果服务器没有返回文件大小，传入 None，此时不能准确显示百分比。
    desc:
        进度条左侧显示的步骤名称，例如“准备清单”“检查文件”“下载 1990”。
    unit:
        计量单位，例如 step、year、file、try、B。
        下载字节时使用 B，并配合 unit_scale=True。
    colour:
        进度条颜色。不同步骤传入不同颜色，便于区分进度条类型。
    unit_scale:
        是否自动缩放单位。下载字节时设为 True，可显示 KiB、MiB 等可读单位。
    unit_divisor:
        单位换算基数。下载文件时使用 1024，更符合文件大小显示习惯。
    leave:
        进度条结束后是否保留在终端。
        总体进度条保留，方便看最终完成状态；单文件下载进度条不保留，避免每年都留下很多行。
    """
    bar_format = UNKNOWN_TOTAL_BAR_FORMAT if total is None else PROGRESS_BAR_FORMAT
    kwargs = {
        "total": total,
        "desc": desc,
        "unit": unit,
        "unit_scale": unit_scale,
        "unit_divisor": unit_divisor,
        "dynamic_ncols": True,
        "bar_format": bar_format,
        "mininterval": PROGRESS_MIN_INTERVAL,
        "leave": leave,
    }

    # 兼容较旧 tqdm：旧版本可能不支持 colour 参数，报错时退回普通进度条。
    try:
        return tqdm(colour=colour, **kwargs)
    except TypeError:
        return tqdm(**kwargs)


# =============================================================================
# 三、下载任务构建与文件检查
# =============================================================================
def ensure_output_dir(output_dir: Path) -> None:
    """
    创建输出目录。

    参数说明：
    output_dir:
        保存 CHIRPS 降雨 .nc 文件的目录。

    处理逻辑：
    1. 如果目录已经存在，不删除、不覆盖其中任何文件。
    2. 如果目录不存在，自动创建完整目录层级。
    3. exist_ok=True 可以保证脚本重复运行时不会因为目录已存在而报错。
    """
    output_dir.mkdir(parents=True, exist_ok=True)


def build_download_tasks() -> List[Dict[str, Any]]:
    """
    根据年份范围生成下载任务清单。

    返回值说明：
    每个任务都是一个字典，包含：
    1. year: 年份，例如 1990。
    2. file_name: 官方文件名，例如 chirps-v2.0.1990.pentads.nc。
    3. url: 直接下载链接。
    4. target_path: 下载完成后的正式 .nc 文件路径。
    5. part_path: 下载过程中的 .part 临时文件路径。

    设计用意：
    1. 下载前先统一生成任务，后续检查、下载、汇总都基于同一份任务清单。
    2. 文件名和 URL 都由参数区统一控制，后续修改年份或文件名规则更方便。
    """
    years = list(range(START_YEAR, END_YEAR + 1))
    tasks: List[Dict[str, Any]] = []

    for year in years:
        file_name = FILE_NAME_TEMPLATE.format(year=year)
        target_path = OUTPUT_DIR / file_name
        part_path = target_path.with_suffix(target_path.suffix + ".part")

        tasks.append(
            {
                "year": year,
                "file_name": file_name,
                "url": f"{BASE_URL}/{file_name}",
                "target_path": target_path,
                "part_path": part_path,
            }
        )

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
    3. 该检查步骤单独显示进度条，便于确认脚本正在逐年判断文件状态。
    """
    existing_tasks: List[Dict[str, Any]] = []
    pending_tasks: List[Dict[str, Any]] = []

    for task in tasks:
        target_path = task["target_path"]
        if not isinstance(target_path, Path):
            raise TypeError("target_path 必须是 pathlib.Path 类型。")

        if target_path.exists():
            existing_tasks.append(task)
        else:
            pending_tasks.append(task)

    return existing_tasks, pending_tasks


def show_skipped_files(existing_tasks: List[Dict[str, Any]]) -> None:
    """
    显示已存在文件的跳过进度。

    参数说明：
    existing_tasks:
        split_existing_and_pending() 返回的已存在任务列表。

    用意：
    1. 已存在文件不会下载，但它们仍然是总任务的一部分。
    2. 单独显示“跳过已有文件”进度条，可以清楚看到跳过步骤已经完成。
    3. 如果没有已存在文件，则不显示该进度条，避免无意义输出。
    """
    if not existing_tasks:
        return

    skipped_years = [str(task["year"]) for task in existing_tasks]
    print(f"跳过已存在文件：{len(existing_tasks)} 个年份（{', '.join(skipped_years)}）")


# =============================================================================
# 四、单文件下载与重试逻辑
# =============================================================================
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
    1. 有文件大小时，tqdm 可以显示百分比、当前量/总量和预计剩余时间。
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
    下载单个年份的 CHIRPS pentad 降雨文件。

    参数说明：
    task:
        单个下载任务，必须包含 year、url、target_path、part_path。

    下载策略：
    1. 正式 .nc 文件不存在时才调用本函数。
    2. 如果旧的 .part 文件存在，说明上次下载可能中断，本次先删除旧 .part 再重新下载。
    3. 新数据先写入 .part 文件，全部下载完成并通过大小检查后，再改名为正式 .nc 文件。
    4. 如果下载途中报错，外层重试逻辑会重新调用本函数。
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

    with urlopen(request, timeout=REQUEST_TIMEOUT, context=SSL_CONTEXT) as response:
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
                leave=False,
            ) as bar:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break

                    file_obj.write(chunk)
                    downloaded_size += len(chunk)
                    bar.update(len(chunk))

    if total_size is not None and downloaded_size != total_size:
        raise RuntimeError(
            f"{year} 年文件下载不完整：已下载 {downloaded_size} 字节，"
            f"服务器标记大小为 {total_size} 字节。"
        )

    part_path.replace(target_path)


def download_with_retries(task: Dict[str, Any]) -> Optional[str]:
    """
    对单个年份执行带重试的下载。

    参数说明：
    task:
        单个下载任务。

    返回值说明：
    1. 返回 None 表示该年份下载成功。
    2. 返回字符串表示该年份最终失败，字符串中包含年份和最后一次错误原因。

    失败处理逻辑：
    1. 每个年份最多尝试 MAX_RETRIES 次，当前默认只尝试 1 次。
    2. HTTPError、URLError、TimeoutError、OSError、RuntimeError 会记录为失败。
    3. 某一年失败后不终止程序，交给外层继续下载下一年。
    4. KeyboardInterrupt 不在这里捕获，确保用户按 Ctrl+C 时可以直接停止整个脚本。
    """
    year = task["year"]
    last_error: Optional[BaseException] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            download_one_file(task)
            return None
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_error = exc

            if attempt < MAX_RETRIES:
                print(f"{year} 下载失败，准备第 {attempt + 1}/{MAX_RETRIES} 次尝试：{exc}")

    return f"{year}: {last_error}"


def download_pending_files(tasks: List[Dict[str, Any]]) -> Tuple[List[int], List[str]]:
    """
    按年份顺序下载所有待下载文件。

    参数说明：
    tasks:
        目标 .nc 文件尚不存在、需要下载的任务列表。

    返回值说明：
    1. success_years: 本次成功下载的年份列表。
    2. failures: 下载失败信息列表，包含失败年份和错误原因。

    进度条说明：
    1. 外层“总体下载进度”显示年度文件层面的完成情况。
    2. 内层“下载 年份”显示单个文件的字节级下载进度。
    3. 默认不重复重试，失败年份会记录到最终汇总，避免反复刷屏和浪费时间。
    """
    success_years: List[int] = []
    failures: List[str] = []

    if not tasks:
        return success_years, failures

    with make_bar(len(tasks), "总体下载进度", "file", OVERALL_BAR_COLOR) as overall_bar:
        for task in tasks:
            year = task["year"]
            overall_bar.set_postfix_str(str(year))

            failure = download_with_retries(task)
            if failure is None:
                success_years.append(year)
            else:
                failures.append(failure)

            overall_bar.update(1)

    return success_years, failures


# =============================================================================
# 五、汇总输出与主流程
# =============================================================================
def print_summary(
    total_count: int,
    skipped_years: List[int],
    success_years: List[int],
    failures: List[str],
    output_dir: Path,
) -> None:
    """
    打印下载任务汇总。

    参数说明：
    total_count:
        本次任务总年份数量。
    skipped_years:
        因目标文件已存在而跳过的年份。
    success_years:
        本次成功下载的年份，不包含已跳过年份。
    failures:
        最终下载失败的信息列表。
    output_dir:
        下载文件保存目录。

    用意：
    1. 运行结束后集中展示结果，方便确认哪些年份成功、哪些年份跳过、哪些年份失败。
    2. 失败信息保留错误原因，便于判断是网络问题、服务器问题还是文件名问题。
    """
    print("\n下载任务汇总：")
    print(f"  保存目录：{output_dir}")
    print(f"  计划年份数：{total_count}")
    print(f"  已存在跳过：{len(skipped_years)} 个年份")
    print(f"  本次成功下载：{len(success_years)} 个年份")
    print(f"  下载失败：{len(failures)} 个年份")

    if skipped_years:
        print(f"  跳过年份：{', '.join(map(str, skipped_years))}")

    if success_years:
        print(f"  成功年份：{', '.join(map(str, success_years))}")

    if failures:
        print("  失败年份及原因：")
        for failure in failures:
            print(f"    - {failure}")


def main() -> None:
    """
    主流程。

    执行步骤：
    1. 创建输出目录。
    2. 根据 START_YEAR 和 END_YEAR 生成年度下载任务清单。
    3. 检查目标目录中哪些年份文件已经存在。
    4. 对已存在文件显示跳过进度。
    5. 对不存在文件逐年下载，并显示总体进度和单文件字节进度。
    6. 打印最终汇总，包括成功、跳过、失败年份。
    """
    ensure_output_dir(OUTPUT_DIR)
    tasks = build_download_tasks()
    existing_tasks, pending_tasks = split_existing_and_pending(tasks)
    show_skipped_files(existing_tasks)

    success_years, failures = download_pending_files(pending_tasks)
    skipped_years = [task["year"] for task in existing_tasks]

    print_summary(
        total_count=len(tasks),
        skipped_years=skipped_years,
        success_years=success_years,
        failures=failures,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户按 Ctrl+C 中断下载，脚本已停止。当前未完成文件会保留为 .part 临时文件。")
