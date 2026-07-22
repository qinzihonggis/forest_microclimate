# -*- coding: utf-8 -*-
"""
批量下载 ERA5-Land 2 米气温小时尺度 NetCDF 数据。

脚本功能：
1. 使用 Copernicus Climate Data Store（CDS）API 下载 ERA5-Land 数据集。
2. 变量固定为 2m temperature，对应 CDS API 变量名为 2m_temperature。
3. 下载年份为 1990-2025，每个月单独保存为一个 .nc 文件。
4. 下载范围为你指定的子区域：North=28.386763, South=23.446029,
   West=115.792840, East=120.733574。
5. 输出文件保存到 E:\forest_microclimate\ForestMicroclimate\T2m。
6. 如果某一个年月的目标 .nc 文件已经存在，则自动跳过该文件，不重复下载。
7. 关键步骤使用 tqdm 彩色进度条显示。下载阶段使用 2 并发，并只保留一个总进度条，
   减少终端刷新和多进度条互相覆盖。

重要说明：
1. 本脚本不安装任何依赖。运行前请确认你已经安装 cdsapi 和 tqdm。
2. CDS API 下载不使用网页登录密码。网页账号密码只用于登录 CDS 网站、同意条款、
   查看个人 API token。脚本认证需要使用 CDS API token。
3. 请先在网页数据集页面点击 Accept terms，同意数据集条款；否则 API 请求会失败。
4. ERA5-Land 的 2m_temperature 原始单位通常是 K（开尔文），如果后续需要摄氏度，
   需要在数据处理阶段执行：摄氏度 = K - 273.15。
"""

from pathlib import Path
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import cdsapi
from tqdm import tqdm


# =============================================================================
# 一、CDS API 账号与凭据设置
# =============================================================================
# 本脚本不再读取 C:\Users\用户名\.cdsapirc 配置文件。
# 你只需要在下面两个变量中直接填写 CDS API 地址和 key/token。
#
# 填写位置：
# 1. CDS_API_URL 通常保持为 "https://cds.climate.copernicus.eu/api"。
# 2. CDS_API_KEY 填写你在 CDS 个人账户页面看到的 API key/token。
# 3. 注意这里填的是 API key/token，不是网页登录邮箱，也不是网页登录密码。
# 4. 字符串两边必须保留英文双引号，否则 Python 会报语法错误。
CDS_API_URL: str = "https://cds.climate.copernicus.eu/api"
CDS_API_KEY: str = "74e0c3b0-6a83-4964-a7a0-77b7bbbeb842"


# =============================================================================
# 二、可调整参数区
# =============================================================================
# CDS 数据集名称。
# 参数说明：
# 1. reanalysis-era5-land 是 ERA5-Land 小时尺度再分析数据集。
# 2. 本脚本只针对这个数据集构造请求，不会下载 ERA5 pressure levels 或其他数据集。
DATASET = "reanalysis-era5-land"

# 下载变量。
# 参数说明：
# 1. 2m_temperature 对应网页 Temperature 分类下的 “2m temperature”。
# 2. 这是近地表 2 米高度气温，不是露点温度、土壤温度或皮肤温度。
# 3. 如果以后要下载降水，应改为 total_precipitation，同时还要重新检查单位和处理逻辑。
VARIABLE = "2m_temperature"

# 下载年份范围。
# 参数说明：
# 1. START_YEAR 是起始年份，END_YEAR 是结束年份，两端都包含。
# 2. 当前设置会覆盖 1990、1991、...、2025，共 36 个年份。
# 3. 下载顺序会按年份倒序执行，即先处理 END_YEAR，再处理 START_YEAR。
# 4. 脚本会进一步按月拆分请求，即每一年提交 12 个较小的月度请求。
START_YEAR = 1990
END_YEAR = 2025

# 月份选择。
# 参数说明：
# 1. 这里定义每一年需要下载哪些月份。
# 2. CDS API 使用两位字符串表示月份，例如 "01" 表示 1 月。
# 3. 当前设置会为每一年生成 12 个独立月度文件，降低单次 CDS 请求大小。
# 4. 下载顺序会按列表顺序执行；当前列表保持 01-12，但任务构建时会倒序使用，
#    因此实际下载会先从 12 月开始，再到 01 月。
# 5. 如只下载生长季，可把列表改为 ["04", "05", "06", "07", "08", "09"] 等。
MONTHS = [f"{month:02d}" for month in range(1, 13)]

# 日期选择。
# 参数说明：
# 1. 这里等价于网页 Day 中全选 01-31。
# 2. 对于没有 29/30/31 日的月份，CDS 会按该月份实际可用日期处理。
# 3. 保持 01-31 可以避免手动区分平年、闰年和不同月份天数。
DAYS = [f"{day:02d}" for day in range(1, 32)]

# 小时选择。
# 参数说明：
# 1. 这里等价于网页 Time 中全选 00:00-23:00。
# 2. ERA5-Land 小时数据使用 UTC 时间，不是北京时间。
# 3. 如果后续要换算为北京时间，需要在数据处理阶段执行 UTC+8。
TIMES = [f"{hour:02d}:00" for hour in range(24)]

# 地理范围，顺序必须是 [North, West, South, East]。
# 参数说明：
# 1. 这对应网页 Geographical area 中的 Sub-region extraction。
# 2. North 是北边界纬度，South 是南边界纬度。
# 3. West 是西边界经度，East 是东边界经度。
# 4. 该范围会减小下载数据体积，避免下载全球数据。
AREA = [28.386763, 115.792840, 23.446029, 120.733574]

# 数据格式。
# 参数说明：
# 1. netcdf 对应网页 Data format 中的 NetCDF4 (Experimental)。
# 2. NetCDF 适合后续用 Python、R、ArcGIS、QGIS、NCO、CDO 等工具处理。
# 3. 如果改为 grib，输出文件扩展名和后续读取方式都需要相应调整。
DATA_FORMAT = "netcdf"

# 下载格式。
# 参数说明：
# 1. unarchived 对应网页 Download format 中的 Unarchived。
# 2. 单文件请求时 CDS 会尽量直接返回 .nc 文件，而不是 zip 压缩包。
# 3. 本脚本按月请求，每个目标就是一个 .nc 文件。
DOWNLOAD_FORMAT = "unarchived"

# 输出目录。
# 参数说明：
# 1. 所有下载后的月度 .nc 文件都会保存到这个目录。
# 2. 如果目录不存在，脚本运行时会自动创建。
# 3. 路径前面的 r 表示原始字符串，Windows 反斜杠不需要额外转义。
OUTPUT_DIR = Path(r"E:\forest_microclimate\ForestMicroclimate\T2m")

# 输出文件名模板。
# 参数说明：
# 1. {year} 会在程序运行时替换为具体年份，{month} 会替换为具体月份。
# 2. 例如 year=1990、month=01 时，文件名为 T2m_199001_hourly.nc。
# 3. 月份使用两位数字可以保证文件在资源管理器和代码中按时间顺序排序。
FILE_NAME_TEMPLATE = "T2m_{year}{month}_hourly.nc"

# 最大重试次数。
# 参数说明：
# 1. CDS 请求可能因为队列繁忙、网络中断或服务端临时错误失败。
# 2. MAX_RETRIES=3 表示每个月度文件最多尝试 3 次。
# 3. 某个月全部失败后，脚本不会停止，会继续处理后续年月，最后汇总失败清单。
MAX_RETRIES = 3

# 重试等待时间，单位为秒。
# 参数说明：
# 1. 每次失败后等待一段时间再重试，可以减少连续撞上临时网络问题的概率。
# 2. 如果 CDS 队列非常繁忙，可以适当增大为 30、60 或更高。
RETRY_WAIT_SECONDS = 20

# 并发下载数量。
# 参数说明：
# 1. 2 表示同时处理 2 个月度文件，通常能比单线程更快。
# 2. 不建议设置太大。CDS 服务器会排队和限流，过高并发可能反而更慢或更容易失败。
# 3. 如果网络或 CDS 队列不稳定，可以改回 1；如果运行稳定，可谨慎尝试 3。
MAX_WORKERS = 2


# =============================================================================
# 三、tqdm 进度条显示设置
# =============================================================================
# tqdm 统一进度条格式。
# 参数说明：
# 1. percentage 显示百分比。
# 2. n_fmt/total_fmt 显示当前量和总量。
# 3. elapsed/remaining/rate_fmt 显示已用时间、预计剩余时间和速度。
# 4. dynamic_ncols=True 会让进度条自动适配终端宽度。
PROGRESS_BAR_FORMAT = (
    "{l_bar}{bar:32}| {percentage:3.0f}% {n_fmt}/{total_fmt} "
    "[已用 {elapsed} < 剩余 {remaining}, {rate_fmt}]"
)

# 不同步骤使用不同颜色。
# 参数说明：
# 1. 不同颜色便于在终端中快速区分当前阶段。
# 2. tqdm 常见可用颜色包括 blue、green、cyan、magenta、yellow、red、white。
# 3. 如果终端或 tqdm 版本不支持 colour 参数，脚本会自动退回普通进度条。
PREPARE_BAR_COLOR = "yellow"
CHECK_BAR_COLOR = "blue"
SKIP_BAR_COLOR = "magenta"
OVERALL_BAR_COLOR = "cyan"
SUMMARY_BAR_COLOR = "white"


# =============================================================================
# 四、进度条工具函数
# =============================================================================
def make_bar(total: int, desc: str, unit: str, colour: str) -> tqdm:
    """
    创建统一格式的 tqdm 彩色进度条。

    参数说明：
    total:
        当前进度条的总任务量。例如总年月数、总文件数、当前年月步骤数。
    desc:
        进度条左侧显示的说明文字。例如“准备任务”“检查已有文件”“下载 1990-01”。
    unit:
        计量单位。例如 year、file、step、try。
    colour:
        进度条颜色。不同步骤传入不同颜色，便于观察当前任务类型。

    返回值：
    一个 tqdm 进度条对象，配合 with 语句使用时可以自动关闭。
    """
    kwargs = {
        "total": total,
        "desc": desc,
        "unit": unit,
        "dynamic_ncols": True,
        "bar_format": PROGRESS_BAR_FORMAT,
        "mininterval": 1.0,
    }

    try:
        return tqdm(colour=colour, **kwargs)
    except TypeError:
        return tqdm(**kwargs)


# =============================================================================
# 五、任务构建与请求参数函数
# =============================================================================
def create_cds_client() -> cdsapi.Client:
    """
    创建 CDS API 客户端。

    处理逻辑：
    1. 只使用脚本顶部 CDS_API_URL 和 CDS_API_KEY 两个变量。
    2. 不读取 .cdsapirc，因此不会依赖 Windows 当前用户目录。
    3. 这里不使用网页登录邮箱和网页登录密码，因为 CDS API 认证不靠网页密码。
    4. 如果你忘记替换 CDS_API_KEY 占位文字，会主动报出清晰提示。

    返回值：
    已初始化的 cdsapi.Client 对象。
    """
    if not CDS_API_URL.strip():
        raise ValueError("请先在脚本顶部填写 CDS_API_URL。")
    if not CDS_API_KEY.strip() or CDS_API_KEY == "请在这里粘贴你的CDS_API_KEY":
        raise ValueError("请先在脚本顶部把 CDS_API_KEY 替换为你的 CDS API key/token。")

    return cdsapi.Client(url=CDS_API_URL, key=CDS_API_KEY)


def ensure_output_dir(output_dir: Path) -> None:
    """
    创建输出目录。

    参数说明：
    output_dir:
        保存 ERA5-Land 2 米气温月度 .nc 文件的目录。

    处理逻辑：
    1. 如果目录已经存在，不删除、不覆盖其中任何文件。
    2. 如果目录不存在，自动创建完整目录层级。
    3. exist_ok=True 可以保证脚本重复运行时不会因为目录已存在而报错。
    """
    with make_bar(1, "创建输出目录", "step", PREPARE_BAR_COLOR) as bar:
        output_dir.mkdir(parents=True, exist_ok=True)
        bar.update(1)


def build_download_tasks() -> List[Dict[str, Any]]:
    """
    根据年份范围和月份列表生成下载任务清单。

    返回值说明：
    每个任务都是一个字典，包含：
    1. year: 年份，例如 1990。
    2. month: 月份，例如 "01"。
    3. label: 年月标签，例如 "1990-01"，用于进度条和汇总显示。
    4. file_name: 输出文件名，例如 ERA5_Land_T2m_1990_01.nc。
    5. target_path: 下载完成后的正式 .nc 文件路径。

    设计用意：
    1. 先统一生成任务清单，后续检查、跳过、下载、汇总都基于同一份清单。
    2. 任务顺序按年份倒序、月份倒序生成，因此会先下载最新年月，再下载更早年月。
    3. 按月拆分可以降低单次 CDS 请求大小，避免年度请求触发 cost limits exceeded。
    4. 文件命名规则集中在 FILE_NAME_TEMPLATE，后续修改更方便。
    """
    years = list(range(END_YEAR, START_YEAR - 1, -1))
    total_tasks = len(years) * len(MONTHS)
    tasks: List[Dict[str, Any]] = []

    with make_bar(total_tasks, "准备下载清单", "month", PREPARE_BAR_COLOR) as bar:
        for year in years:
            for month in reversed(MONTHS):
                file_name = FILE_NAME_TEMPLATE.format(year=year, month=month)
                label = f"{year}-{month}"
                tasks.append(
                    {
                        "year": year,
                        "month": month,
                        "label": label,
                        "file_name": file_name,
                        "target_path": OUTPUT_DIR / file_name,
                    }
                )
                bar.set_postfix_str(label)
                bar.update(1)

    return tasks


def build_request(year: int, month: str) -> Dict[str, Any]:
    """
    构建单个年月的 CDS API 请求参数。

    参数说明：
    year:
        当前要下载的年份。网页端 Year 只能单选，因此脚本每次只提交一个年份。
    month:
        当前要下载的月份。脚本每次只提交一个月份，用于降低单次请求数据量。

    返回值说明：
    返回一个字典，内容等价于你在 CDS 下载页面中选择的表单项。

    请求字段说明：
    variable:
        选择 2m_temperature，即网页 Temperature 下的 2m temperature。
    year:
        当前年份，用四位数字字符串表示。
    month:
        当前月份，用两位数字字符串表示，例如 "01"。
    day:
        01-31，等价于 Day 全选；CDS 会处理不同月份实际天数。
    time:
        00:00-23:00，等价于 Time 全选。
    area:
        [North, West, South, East]，对应网页 Sub-region extraction。
    data_format:
        netcdf，对应 NetCDF4 (Experimental)。
    download_format:
        unarchived，对应 Unarchived。
    """
    return {
        "variable": [VARIABLE],
        "year": str(year),
        "month": [month],
        "day": DAYS,
        "time": TIMES,
        "area": AREA,
        "data_format": DATA_FORMAT,
        "download_format": DOWNLOAD_FORMAT,
    }


def split_existing_and_pending(
    tasks: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    检查哪些月度文件已经存在，哪些仍需要下载。

    参数说明：
    tasks:
        build_download_tasks() 生成的完整月度任务清单。

    返回值说明：
    1. existing_tasks: 目标 .nc 文件已经存在的月度任务，后续直接跳过。
    2. pending_tasks: 目标 .nc 文件不存在的任务，后续需要提交 CDS 下载。

    处理逻辑：
    1. 只要正式 .nc 文件存在，就认为该年月已经下载完成。
    2. 不主动覆盖已有文件，避免重复排队和重复下载。
    3. 检查过程单独显示进度条，便于确认脚本正在逐月判断文件状态。
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
    显示已经存在文件的跳过进度。

    参数说明：
    existing_tasks:
        split_existing_and_pending() 返回的已存在任务列表。

    设计用意：
    1. 已有文件不会下载，但它们仍然属于总任务的一部分。
    2. 单独显示“跳过已有文件”进度条，能清楚看到跳过步骤也已完成。
    3. 如果没有已有文件，则不显示该进度条，避免输出无意义信息。
    """
    if not existing_tasks:
        return

    with make_bar(len(existing_tasks), "跳过已有文件", "file", SKIP_BAR_COLOR) as bar:
        for task in existing_tasks:
            bar.set_postfix_str(str(task["label"]))
            bar.update(1)


# =============================================================================
# 六、单月下载函数
# =============================================================================
def download_one_month(task: Dict[str, Any]) -> None:
    """
    下载单个年月的 ERA5-Land 2 米气温数据。

    参数说明：
    task:
        单个月度下载任务，必须包含 year、month、label、file_name、target_path。

    下载步骤：
    1. 构造该年月的 CDS API 请求参数。
    2. 提交请求到 CDS 队列。
    3. 等待 CDS 后台处理并下载到目标 .nc 文件。
    4. 检查目标文件是否已生成且大小大于 0。

    并发说明：
    1. 该函数会在线程池的工作线程中执行。
    2. 每个工作线程为当前任务创建一个独立的 CDS API 客户端，避免多个线程共享同一个客户端。
    3. 函数内部不创建 tqdm 进度条，避免并发输出互相覆盖；总进度只在主线程更新。
    """
    year = task["year"]
    month = task["month"]
    label = task["label"]
    target_path = task["target_path"]

    if not isinstance(year, int):
        raise TypeError("year 必须是 int 类型。")
    if not isinstance(month, str):
        raise TypeError("month 必须是 str 类型。")
    if not isinstance(label, str):
        raise TypeError("label 必须是 str 类型。")
    if not isinstance(target_path, Path):
        raise TypeError("target_path 必须是 pathlib.Path 类型。")

    client = create_cds_client()
    request = build_request(year, month)
    result = client.retrieve(DATASET, request)
    result.download(str(target_path))

    if not target_path.exists():
        raise FileNotFoundError(f"下载结束后未找到目标文件：{target_path}")
    if target_path.stat().st_size <= 0:
        raise OSError(f"目标文件大小为 0，可能下载失败：{target_path}")


def download_with_retries(task: Dict[str, Any]) -> Tuple[str, bool, Optional[str]]:
    """
    带重试机制地下载单个月度文件。

    参数说明：
    task:
        单个月度下载任务。

    返回值说明：
    1. 第一个返回值是年月标签，例如 "1990-01"。
    2. 第二个返回值是是否成功，True 表示下载成功，False 表示失败。
    3. 第三个返回值是失败原因；成功时返回 None。

    处理逻辑：
    1. 每个月度文件最多尝试 MAX_RETRIES 次。
    2. 如果某次尝试成功，立即返回成功。
    3. 如果失败且未达到最大次数，等待 RETRY_WAIT_SECONDS 秒后重试。
    4. 如果全部失败，返回失败原因，主流程继续处理下一个年月。
    """
    label = str(task["label"])
    last_error: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            download_one_month(task)
            return label, True, None
        except Exception as exc:
            last_error = repr(exc)

            if attempt < MAX_RETRIES:
                tqdm.write(
                    f"{label} 第 {attempt}/{MAX_RETRIES} 次下载失败，"
                    f"{RETRY_WAIT_SECONDS} 秒后重试。错误：{last_error}"
                )
                sleep(RETRY_WAIT_SECONDS)

    return label, False, last_error


# =============================================================================
# 七、主流程
# =============================================================================
def main() -> None:
    """
    主执行流程。

    执行顺序：
    1. 创建输出目录。
    2. 生成 1990-2025 的月度任务清单。
    3. 检查已有文件并跳过。
    4. 对缺失年月使用 2 并发提交 CDS API 下载请求。
    5. 汇总成功、跳过、失败年月，便于后续检查。
    """
    ensure_output_dir(OUTPUT_DIR)
    tasks = build_download_tasks()
    existing_tasks, pending_tasks = split_existing_and_pending(tasks)

    show_skipped_files(existing_tasks)

    success_labels: List[str] = []
    failed_items: List[Tuple[str, Optional[str]]] = []

    if pending_tasks:
        with make_bar(len(pending_tasks), f"月度下载总进度({MAX_WORKERS}并发)", "month", OVERALL_BAR_COLOR) as bar:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_label = {
                    executor.submit(download_with_retries, task): str(task["label"])
                    for task in pending_tasks
                }

                for future in as_completed(future_to_label):
                    fallback_label = future_to_label[future]

                    try:
                        label, success, error_message = future.result()
                    except Exception as exc:
                        label = fallback_label
                        success = False
                        error_message = repr(exc)

                    if success:
                        success_labels.append(label)
                        bar.set_postfix_str(f"完成 {label}")
                    else:
                        failed_items.append((label, error_message))
                        bar.set_postfix_str(f"失败 {label}")
                        tqdm.write(f"{label} 下载失败：{error_message}")

                    bar.update(1)

    with make_bar(1, "汇总下载结果", "step", SUMMARY_BAR_COLOR) as bar:
        skipped_labels = [task["label"] for task in existing_tasks]

        tqdm.write("")
        tqdm.write("下载任务汇总：")
        tqdm.write(f"输出目录：{OUTPUT_DIR}")
        tqdm.write(f"年份范围：{START_YEAR}-{END_YEAR}")
        tqdm.write(f"月份范围：{', '.join(MONTHS)}")
        tqdm.write(f"下载并发数：{MAX_WORKERS}")
        tqdm.write(f"总月度任务数：{len(tasks)} 个")
        tqdm.write(f"已存在并跳过：{len(skipped_labels)} 个")
        tqdm.write(f"本次成功下载：{len(success_labels)} 个")
        tqdm.write(f"本次下载失败：{len(failed_items)} 个")

        if skipped_labels:
            tqdm.write("跳过年月：" + ", ".join(map(str, skipped_labels)))
        if success_labels:
            tqdm.write("成功年月：" + ", ".join(map(str, success_labels)))
        if failed_items:
            tqdm.write("失败年月及原因：")
            for label, error_message in failed_items:
                tqdm.write(f"  {label}: {error_message}")

        bar.update(1)


if __name__ == "__main__":
    main()
