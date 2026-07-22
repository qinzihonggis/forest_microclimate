from pathlib import Path
import contextlib
import logging
import os
import re
import shutil
import warnings


# ============================== 0. 运行前全局参数 ==============================
# 本脚本用于计算福建省 2025 年逐月 SPI-1 干旱指数。
# 计算逻辑：
# 1）读取 1990-2025 年 CHIRPS 月累计降雨量；
# 2）用 1990-2024 年作为校准期拟合历史降雨分布；
# 3）只输出 2025 年 1-12 月 SPI-1；
# 4）保存 NetCDF、逐月统计表和 12 张月度空间分布图。

# 输入降雨数据目录：
# 1）该目录存放已经聚合为“月累计降雨量”的 CHIRPS NetCDF 文件；
# 2）脚本会自动读取目录下所有 .nc 文件，并按文件名中的年份排序；
# 3）本次 SPI 计算要求文件必须覆盖 1990-2025 年，共 36 个年度文件；
# 4）如果后续换成其他区域或其他数据源，只需要把这里改成新的月尺度降雨 NC 目录。
INPUT_DIR = Path(
    r"E:\forest_microclimate\ForestMicroclimate\Precipitation_CHIRPS\fujian_pre_pentad\fujian_pre_WGS_monthly"
)

# 福建省行政边界：
# 1）用于绘制 2025 年 12 张 SPI 空间分布图时叠加省界线；
# 2）当前降雨栅格坐标是经纬度坐标 lat/lon，因此这里必须使用地理坐标系 shp；
# 3）不要在这里填 UTM 投影坐标系 shp，否则边界线会和经纬度栅格错位；
# 4）如果只想导出栅格图、不叠加边界，可以保留路径不变，也可以在 read_boundary 函数中跳过读取。
BOUNDARY_SHP = Path(
    r"E:\forest_microclimate\ForestMicroclimate\Fujian_Shp\福建省行政边界.shp"
)

# 输出目录：
# 1）OUTPUT_ROOT 是结果总目录；
# 2）OUTPUT_DIR 是本次 SPI 结果专用文件夹；
# 3）脚本运行时会自动创建 SPI_result 文件夹；
# 4）NetCDF、统计表、格点日志和 12 张月度图都会写入 OUTPUT_DIR。
OUTPUT_ROOT = Path(r"E:\forest_microclimate\ForestMicroclimate\results")
OUTPUT_DIR = OUTPUT_ROOT / "SPI_result"

# 本次运行临时缓存目录：
# 1）Matplotlib 第一次绘图时可能生成字体缓存等临时文件；
# 2）这里把缓存限制在结果目录下，避免污染系统默认缓存目录；
# 3）脚本无论成功还是报错，finally 中都会尝试删除该目录；
# 4）这里只删除本脚本创建的临时缓存，不会删除其他历史结果文件。
TEMP_CACHE_DIR = OUTPUT_DIR / "本次运行临时缓存"

# NetCDF 变量名和坐标名：
# 1）PRECIP_VAR 是月累计降雨变量名，当前 NC 中为 precip；
# 2）TIME_NAME 是时间维度名称，当前为 time；
# 3）LAT_NAME 和 LON_NAME 是经纬度坐标名称，当前为 lat/lon；
# 4）如果后续输入 NC 的变量名变化，例如 precipitation、pr、latitude、longitude，需要同步修改这里。
PRECIP_VAR = "precip"
TIME_NAME = "time"
LAT_NAME = "lat"
LON_NAME = "lon"

# SPI 关键参数：
# DATA_START_YEAR：
# 表示传入 climate_indices.indices.spi 的完整时间序列从哪一年开始。
# 本脚本传入的是 1990-01 到 2025-12 的完整月序列，所以这里必须是 1990。
# 如果这里填错，SPI 函数会把月份和年份对应错，校准期也会错位。
DATA_START_YEAR = 1990

# CALIBRATION_YEAR_INITIAL：
# SPI 历史基准期的起始年份，也就是用哪一年开始的降雨数据拟合 Gamma 分布。
# 本次要求用 1990-2024 建立历史基准，因此起始年份为 1990。
# 该年份必须大于或等于 DATA_START_YEAR。
CALIBRATION_YEAR_INITIAL = 1990

# CALIBRATION_YEAR_FINAL：
# SPI 历史基准期的结束年份，本脚本设置为 2024。
# climate_indices 会用 1990-2024 的同月降雨样本进行分布拟合；
# 2025 年不会参与拟合，只作为待评估年份输出 SPI。
# 如果想改成更短或更长的基准期，应优先调整这个参数。
CALIBRATION_YEAR_FINAL = 2024

# EVALUATION_YEAR：
# 待评估年份，也就是最终要输出 SPI 结果的年份。
# 本脚本只保存 2025 年 1-12 月，不保存 1990-2024 年历史 SPI。
# 如果将来要计算其他年份，例如 2024 年，把这里改为 2024，并保证输入数据覆盖该年份。
EVALUATION_YEAR = 2025

# SPI_SCALE：
# SPI 的时间尺度，单位是“月”。
# SPI_SCALE=1 表示 SPI-1，即每个月只用当月降雨量评价该月干湿状况；
# SPI_SCALE=3 表示 SPI-3，会使用连续 3 个月累计降雨，更接近季节尺度；
# 本次需求只要 2025 年每个月 SPI，因此固定为 1，不计算季度或年度 SPI。
SPI_SCALE = 1

# NaN 处理策略：
# 1）True 表示只要某个格点在 1990-2025 任意月份存在 NaN，就跳过该格点；
# 2）跳过后该格点在 2025 年 12 个月输出中继续保持 NaN；
# 3）这样能保留海域、省外区域或无效区域的原始掩膜；
# 4）不要把 NaN 填成 0，因为 0 表示真实无雨，NaN 表示无数据，两者含义完全不同；
# 5）不做插值，是因为 SPI 拟合依赖历史降雨分布，插值可能改变极端降雨统计特征。
SKIP_GRID_IF_ANY_NAN = True

# 输出文件名：
# 1）NetCDF 使用英文文件名，避免 Windows 下 netCDF4 后端写入中文 NC 文件名时报 PermissionError；
# 2）PNG 图片、逐月统计表和格点日志继续使用中文命名；
# 3）NetCDF 保存完整 2025 年 12 个月 SPI 栅格；
# 4）逐月统计表保存每个月有效格点数、均值、最小值、最大值和干旱格点数量；
# 5）格点计算日志记录每个格点是成功、跳过还是失败，便于排查 NaN 或异常格点。
NETCDF_NAME = "Fujian_SPI1_2025.nc"
STATS_CSV_NAME = "福建省2025年SPI1逐月统计表.csv"
LOG_CSV_NAME = "福建省2025年SPI1格点计算日志.csv"
FIGURE_NAME_TEMPLATE = "福建省2025年{month:02d}月SPI1空间分布图.png"
THRESHOLD_LEGEND_FIGURE_NAME = "SPI1阈值线图例.png"

# 图像基础参数：
# FIGURE_DPI 控制输出 PNG 清晰度，值越大图片越清晰、文件也越大；
# FIGURE_SIZE 控制图片宽高，单位是英寸；
# FONT_FAMILY_CANDIDATES 是中文字体候选列表，系统会按顺序寻找可用字体；
# TITLE_FONT_SIZE、AXIS_LABEL_FONT_SIZE、TICK_FONT_SIZE 控制标题、坐标轴和刻度字号；
# COLORBAR_LABEL_FONT_SIZE、COLORBAR_TICK_FONT_SIZE 控制色带标题和色带刻度字号。
FIGURE_DPI = 300
FIGURE_SIZE = (8.0, 7.0)
FONT_FAMILY_CANDIDATES = ["SimHei", "Microsoft YaHei", "SimSun", "Arial Unicode MS"]
TITLE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 12
TICK_FONT_SIZE = 10
COLORBAR_LABEL_FONT_SIZE = 11
COLORBAR_TICK_FONT_SIZE = 9
THRESHOLD_LEGEND_FONT_SIZE = 8
THRESHOLD_LEGEND_FIGURE_SIZE = (8.0, 0.7)
THRESHOLD_LEGEND_DPI = 300

# 地图边界线和经纬网参数：
# BOUNDARY_LINE_COLOR 和 BOUNDARY_LINE_WIDTH 控制福建省边界线颜色和粗细；
# GRID_LINE_COLOR、GRID_LINE_WIDTH、GRID_LINE_ALPHA 控制底图经纬网颜色、线宽和透明度；
# GRID_LINE_ALPHA 是经纬线透明度参数，取值范围 0-1：
# 0 表示完全透明、看不见经纬线；1 表示完全不透明；数值越小，经纬线越淡。
# LON_TICK_INTERVAL 和 LAT_TICK_INTERVAL 控制横轴经度、纵轴纬度刻度间隔，单位是度。
# 如果省界线过粗、网格线太显眼或经纬度刻度太密，优先调整这些参数。
BOUNDARY_LINE_COLOR = "black"
BOUNDARY_LINE_WIDTH = 0.9
GRID_LINE_COLOR = "0.82"
GRID_LINE_WIDTH = 0.4
GRID_LINE_ALPHA = 0.6
LON_TICK_INTERVAL = 1.0
LAT_TICK_INTERVAL = 1.0

# NaN 区域显示颜色：
# SPI=0 附近在 BrBG 色带中接近白色，容易和“无数据”混淆；
# 因此这里把真正的 NaN 区域单独显示为浅灰色，便于区分“接近正常”和“没有数据”。
NAN_DISPLAY_COLOR = "lightgray"

# SPI 色带参数：
# SPI_COLOR_MAP 控制空间图使用的色带，BrBG 通常适合表达“干旱-湿润”两端差异；
# SPI_VMIN 和 SPI_VMAX 控制色带数值范围，本脚本固定为 -3 到 3；
# 负值表示偏旱，正值表示偏湿，0 附近表示接近历史同期正常水平；
# SPI_COLORBAR_TICKS 控制色带上显示哪些数字刻度；
# 这里使用等间距整数刻度 -3、-2、-1、0、1、2、3，不再把刻度对齐到干旱等级阈值；
# 图中使用 extend="both"，超出 -3 或 3 的值会使用色带两端颜色表达。
SPI_COLOR_MAP = "BrBG"
SPI_VMIN = -3.0
SPI_VMAX = 3.0
SPI_COLORBAR_TICKS = [-3, -2, -1, 0, 1, 2, 3]

# SPI 阈值标注：
# 字典的 key 是 SPI 阈值，value 是画在色带上的横线颜色；
# 常用解释包括：SPI<=-1 为中度及以上干旱，SPI<=-1.5 为重度及以上干旱，SPI<=-2 为极端干旱；
# 这里不显示“极端干旱阈值”等文字，只用不同颜色横线标记阈值，避免图例拥挤。
SPI_THRESHOLD_LINES = {
    -2.0: "darkred",
    -1.5: "red",
    -1.0: "gold",
    1.0: "blue",
}

# 阈值线图例：
# 这些文字不再显示在每张空间图下方，而是单独导出为一张 PNG 图例；
# 这样每张月度空间图更干净，同时仍能通过单独图例解释色带上横线颜色。
SPI_THRESHOLD_LEGEND_LABELS = {
    -2.0: "Extreme drought (-2.0)",
    -1.5: "Severe drought (-1.5)",
    -1.0: "Moderate drought (-1.0)",
    1.0: "Wet threshold (1.0)",
}

# SPI 阈值标注线样式：
# 控制色带内部阈值辅助线的线宽和透明度；
# 阈值线颜色在 SPI_THRESHOLD_LINES 中单独设置。
SPI_THRESHOLD_LINE_WIDTH = 1.2
SPI_THRESHOLD_LINE_ALPHA = 0.95

# 色带两端文字：
# 在色带上端显示 wet，在下端显示 drought；
# X/Y 参数用于微调文字位置，fontsize 用于调整字号。
COLORBAR_TOP_TEXT = "wet"
COLORBAR_BOTTOM_TEXT = "drought"
COLORBAR_END_TEXT_X = 0.5
COLORBAR_TOP_TEXT_Y = 1.10
COLORBAR_BOTTOM_TEXT_Y = -0.10
COLORBAR_END_TEXT_FONT_SIZE = 10

# 阈值线图例排版：
# THRESHOLD_LEGEND_COLUMNS 控制单独图例中横向排列的列数；
# 如果标签显示太挤，可以减小列数或增大 THRESHOLD_LEGEND_FIGURE_SIZE。
THRESHOLD_LEGEND_COLUMNS = 4

# 进度条格式：
# tqdm 默认会显示百分比，这里额外统一显示当前量/总量、已用时间、预计剩余时间和速度；
# 不同步骤会设置不同 colour，便于在终端中区分读取、计算、保存、绘图和清理阶段。
TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
    "[已用 {elapsed}，剩余 {remaining}，速度 {rate_fmt}]"
)

# climate_indices 日志控制：
# 1）部分版本的 climate_indices 会在每个格点计算时输出大量 info 日志；
# 2）这些日志会把 tqdm 的单行进度条冲散，造成终端刷屏；
# 3）True 表示静默这些库内部日志，只保留 tqdm 进度条一点一点前进；
# 4）如果需要调试 climate_indices 内部拟合过程，可临时改为 False。
SUPPRESS_CLIMATE_INDICES_LOGS = True

# 需要静默的日志名称：
# 截图中的刷屏日志来自 climate_indices.indices 和 climate_indices.compute。
CLIMATE_INDICES_LOGGER_NAMES = [
    "climate_indices",
    "climate_indices.indices",
    "climate_indices.compute",
]


# Matplotlib 会在导入 pyplot 时确定缓存位置，因此必须在导入 pyplot 前设置缓存目录。
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(TEMP_CACHE_DIR)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from climate_indices import compute, indices
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from tqdm import tqdm


try:
    import geopandas as gpd
except ImportError:
    gpd = None


warnings.filterwarnings("ignore", category=RuntimeWarning)


class _NullWriter:
    # 用于吞掉第三方库直接写到 stdout/stderr 的内容。
    # tqdm 进度条不经过这个对象，因此外层进度条仍会正常显示。
    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        return None


def configure_quiet_logging() -> None:
    # 步骤 0.2：关闭 climate_indices 的 info 级日志输出。
    # 这样逐格点计算时终端不会被 calculation_started、distribution_fitting_started 等日志刷屏。
    # 这里只调整 climate_indices 相关 logger，不影响 tqdm 进度条，也不影响脚本自己的 print 提示。
    if not SUPPRESS_CLIMATE_INDICES_LOGS:
        return

    for logger_name in CLIMATE_INDICES_LOGGER_NAMES:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = False
        logger.disabled = True


def run_spi_quietly(precip_1d: np.ndarray) -> np.ndarray:
    # 步骤 3.1：静默执行单个格点的 SPI 计算。
    # 有些 climate_indices 版本不是通过标准 logging，而是直接向 stdout/stderr 写日志；
    # 因此在单格点 SPI 计算期间额外重定向 stdout/stderr，防止日志打断 tqdm 单行进度条。
    # 只静默 indices.spi 这一小段，不影响脚本其他阶段的正常提示和进度条显示。
    if not SUPPRESS_CLIMATE_INDICES_LOGS:
        return indices.spi(
            values=precip_1d,
            scale=SPI_SCALE,
            distribution=indices.Distribution.gamma,
            data_start_year=DATA_START_YEAR,
            calibration_year_initial=CALIBRATION_YEAR_INITIAL,
            calibration_year_final=CALIBRATION_YEAR_FINAL,
            periodicity=compute.Periodicity.monthly,
        )

    null_writer = _NullWriter()
    with contextlib.redirect_stdout(null_writer), contextlib.redirect_stderr(null_writer):
        return indices.spi(
            values=precip_1d,
            scale=SPI_SCALE,
            distribution=indices.Distribution.gamma,
            data_start_year=DATA_START_YEAR,
            calibration_year_initial=CALIBRATION_YEAR_INITIAL,
            calibration_year_final=CALIBRATION_YEAR_FINAL,
            periodicity=compute.Periodicity.monthly,
        )


def list_precip_files(input_dir: Path) -> list[Path]:
    # 步骤 1.1：查找输入目录中的所有 NetCDF 文件。
    # 这里不手动写死 36 个文件名，而是自动搜索 .nc 文件，便于后续换数据目录。
    # 排序时使用文件名中的年份，而不是简单按字符串排序，避免时间拼接顺序错误。
    files = sorted(input_dir.glob("*.nc"), key=lambda p: extract_year_from_name(p.name))
    if not files:
        raise FileNotFoundError(f"未在目录中找到 NC 文件：{input_dir}")
    return files


def extract_year_from_name(file_name: str) -> int:
    # 步骤 1.2：从文件名中提取 4 位年份。
    # 当前文件名类似 fujian_1990_pre_CHIRPS_monthly.nc，可以从中识别 1990。
    # 如果后续文件名不含年份，脚本会主动报错，避免数据顺序无法确认。
    match = re.search(r"(19|20)\d{2}", file_name)
    if match is None:
        raise ValueError(f"文件名中未识别到年份：{file_name}")
    return int(match.group(0))


def setup_matplotlib_fonts() -> None:
    # 步骤 0.1：设置 Matplotlib 中文字体。
    # 因为输出图标题、坐标轴、色带说明和阈值标注均为中文，所以必须设置中文字体候选列表。
    # axes.unicode_minus=False 用于保证负号正常显示，否则 SPI 负值刻度可能显示成方块。
    plt.rcParams["font.sans-serif"] = FONT_FAMILY_CANDIDATES
    plt.rcParams["axes.unicode_minus"] = False


def read_precip_dataset(files: list[Path]) -> xr.Dataset:
    # 步骤 1.3：逐年读取降雨 NetCDF，并沿 time 维度拼接成完整时间序列。
    # 不使用 open_mfdataset，是为了避免 dask 等额外依赖，也让读取进度条能按年度文件显示进度。
    # 每个年度文件读取后只保留 precip 变量，减少内存中无关变量。
    datasets = []

    for file_path in tqdm(
        files,
        desc="读取逐年月降雨NC",
        colour="cyan",
        bar_format=TQDM_BAR_FORMAT,
    ):
        with xr.open_dataset(file_path) as ds_year:
            # 检查每个年度文件是否包含必要的降雨变量和 time/lat/lon 坐标。
            # 如果某一年文件结构异常，立即报错，避免后续 SPI 计算得到错位结果。
            required_names = {PRECIP_VAR, TIME_NAME, LAT_NAME, LON_NAME}
            available_names = set(ds_year.data_vars) | set(ds_year.coords)
            missing_names = required_names - available_names
            if missing_names:
                raise KeyError(
                    f"{file_path.name} 缺少必要变量或坐标：{sorted(missing_names)}"
                )

            datasets.append(ds_year[[PRECIP_VAR]].load())

    ds = xr.concat(datasets, dim=TIME_NAME)
    ds = ds.sortby(TIME_NAME)
    return ds


def validate_dataset(ds: xr.Dataset) -> pd.DatetimeIndex:
    # 步骤 2：校验拼接后的数据是否满足 SPI 计算要求。
    # SPI 函数要求 values 是从 data_start_year 开始的连续月序列；
    # 本脚本要求实际时间轴必须严格等于 1990-01 到 2025-12，不能缺月、不能多月、不能乱序。
    times = pd.DatetimeIndex(ds[TIME_NAME].values)
    expected_times = pd.date_range(
        f"{DATA_START_YEAR}-01-01",
        f"{EVALUATION_YEAR}-12-01",
        freq="MS",
    )

    if not times.equals(expected_times):
        raise ValueError(
            "时间轴不符合预期：应为 "
            f"{expected_times[0].date()} 到 {expected_times[-1].date()} "
            f"共 {len(expected_times)} 个月；实际为 "
            f"{times[0].date()} 到 {times[-1].date()} 共 {len(times)} 个月。"
        )

    if ds[PRECIP_VAR].dims != (TIME_NAME, LAT_NAME, LON_NAME):
        # climate_indices.spi 每次只接收一个格点的一维时间序列。
        # 因此这里要求三维数组顺序固定为 time, lat, lon，便于后续按格点抽取 precip[:, i, j]。
        raise ValueError(
            f"{PRECIP_VAR} 维度应为 {(TIME_NAME, LAT_NAME, LON_NAME)}，"
            f"实际为 {ds[PRECIP_VAR].dims}"
        )

    return times


def calculate_spi_2025(precip: np.ndarray, times: pd.DatetimeIndex) -> tuple[np.ndarray, pd.DataFrame]:
    # 步骤 3：逐格点计算 SPI-1。
    # climate_indices.spi 的输入是一维时间序列，所以需要对每个 lat/lon 格点分别计算。
    # 每个格点传入 1990-01 到 2025-12 的完整月降雨序列；
    # 函数内部只用 1990-2024 拟合 Gamma 分布，然后返回全时段 SPI；
    # 本脚本再从返回结果中截取 2025 年 12 个月保存。
    n_months, n_lat, n_lon = precip.shape
    evaluation_mask = times.year == EVALUATION_YEAR
    evaluation_count = int(evaluation_mask.sum())
    spi_2025 = np.full((evaluation_count, n_lat, n_lon), np.nan, dtype=np.float32)

    # 为了让进度条按“格点数”推进，把三维数组展平成 time x grid。
    # flat_precip[:, grid_index] 就是某个格点 1990-2025 的完整月降雨序列。
    log_rows = []
    total_grids = n_lat * n_lon
    flat_precip = precip.reshape(n_months, total_grids)
    flat_spi = spi_2025.reshape(evaluation_count, total_grids)

    for grid_index in tqdm(
        range(total_grids),
        desc="逐格点计算SPI1",
        colour="green",
        bar_format=TQDM_BAR_FORMAT,
    ):
        precip_1d = flat_precip[:, grid_index].astype(np.float64)

        # 全时段均为 NaN 的格点通常是海域、省外区域或掩膜外区域。
        # 这类位置没有有效降雨历史，不能计算 SPI，输出保持 NaN。
        if np.all(np.isnan(precip_1d)):
            log_rows.append((grid_index, "跳过", "全时段均为NaN"))
            continue

        # 只要历史序列中存在任意 NaN，就跳过该格点。
        # 原因是 SPI 需要按月份拟合历史分布，缺测会影响同月样本统计；
        # 默认不插值、不填 0，保证结果不引入人为假设。
        if SKIP_GRID_IF_ANY_NAN and np.any(np.isnan(precip_1d)):
            log_rows.append((grid_index, "跳过", "时间序列包含NaN"))
            continue

        # 全时段都为 0 的格点无法拟合 Gamma 降雨分布。
        # 福建省 CHIRPS 正常陆地区域通常不会出现这种情况，但保留该检查可增强稳健性。
        if np.all(precip_1d == 0):
            log_rows.append((grid_index, "跳过", "全时段降雨量均为0"))
            continue

        try:
            # SPI 核心计算：
            # values：该格点 1990-2025 的完整月降雨量；
            # scale=1：计算 SPI-1，只评价当月降雨距平；
            # distribution=gamma：降雨 SPI 的常用拟合分布；
            # data_start_year=1990：告诉函数 values 的第一个值对应 1990 年 1 月；
            # calibration_year_initial/final：只用 1990-2024 建立历史基准；
            # periodicity=monthly：按月数据处理，每个日历月份分别拟合同月历史分布。
            spi_full = run_spi_quietly(precip_1d)
            flat_spi[:, grid_index] = spi_full[evaluation_mask].astype(np.float32)
            log_rows.append((grid_index, "成功", "已计算2025年SPI1"))
        except Exception as exc:
            log_rows.append((grid_index, "失败", str(exc)))

    log_df = pd.DataFrame(log_rows, columns=["格点序号", "状态", "说明"])
    return spi_2025, log_df


def build_output_dataset(
    spi_2025: np.ndarray,
    ds_source: xr.Dataset,
    times: pd.DatetimeIndex,
) -> xr.Dataset:
    # 步骤 4.1：构建输出 NetCDF 数据集。
    # 输出只包含 2025 年 12 个月，而不是保存 1990-2025 全时段 SPI；
    # 这样文件更小，也完全符合“只想知道 2025 年每个月 SPI”的需求。
    # 坐标 lat/lon 直接继承原始降雨数据，确保和输入栅格空间位置一致。
    evaluation_times = times[times.year == EVALUATION_YEAR]
    ds_spi = xr.Dataset(
        data_vars={
            "SPI1": (
                [TIME_NAME, LAT_NAME, LON_NAME],
                spi_2025,
                {
                    "long_name": "SPI-1 drought index",
                    "description": "2025 monthly SPI-1 calibrated by 1990-2024 monthly precipitation",
                    "units": "1",
                    "scale": SPI_SCALE,
                    "distribution": "gamma",
                    "calibration_period": f"{CALIBRATION_YEAR_INITIAL}-{CALIBRATION_YEAR_FINAL}",
                    "nan_policy": "Grids with any NaN in 1990-2025 are skipped and kept as NaN.",
                },
            )
        },
        coords={
            TIME_NAME: evaluation_times,
            LAT_NAME: ds_source[LAT_NAME].values,
            LON_NAME: ds_source[LON_NAME].values,
        },
        attrs={
            "title": "福建省2025年逐月SPI1干旱指数",
            "input_precipitation": str(INPUT_DIR),
            "precipitation_variable": PRECIP_VAR,
            "data_period": f"{DATA_START_YEAR}-{EVALUATION_YEAR}",
            "calibration_period": f"{CALIBRATION_YEAR_INITIAL}-{CALIBRATION_YEAR_FINAL}",
            "evaluation_period": str(EVALUATION_YEAR),
            "created_by": "SPI计算.py",
        },
    )
    return ds_spi


def save_netcdf(ds_spi: xr.Dataset) -> Path:
    # 步骤 4.2：保存 NetCDF 主结果。
    # NetCDF 是后续 GIS、Python、R 或其他空间分析软件最适合读取的主结果格式；
    # 文件中 NaN 区域会继续保留为 NaN，不会被填补为 0。
    output_path = OUTPUT_DIR / NETCDF_NAME

    with tqdm(
        total=1,
        desc="保存SPI结果NC",
        colour="blue",
        bar_format=TQDM_BAR_FORMAT,
    ) as progress:
        ds_spi.to_netcdf(output_path)
        progress.update(1)

    return output_path


def make_monthly_statistics(ds_spi: xr.Dataset) -> pd.DataFrame:
    # 步骤 5.1：生成 2025 年逐月统计表。
    # 统计表用于快速查看每个月全省有效格点的 SPI 分布；
    # 其中干旱格点数量使用常用 SPI 阈值统计：
    # SPI<=-1 为中度及以上干旱，SPI<=-1.5 为重度及以上干旱，SPI<=-2 为极端干旱。
    rows = []
    spi = ds_spi["SPI1"].values
    times = pd.DatetimeIndex(ds_spi[TIME_NAME].values)

    for month_index in tqdm(
        range(spi.shape[0]),
        desc="生成逐月统计表",
        colour="magenta",
        bar_format=TQDM_BAR_FORMAT,
    ):
        values = spi[month_index, :, :]
        valid_values = values[np.isfinite(values)]
        rows.append(
            {
                "月份": times[month_index].strftime("%Y-%m"),
                "有效格点数": int(valid_values.size),
                "无效格点数": int(values.size - valid_values.size),
                "SPI最小值": float(np.nanmin(values)) if valid_values.size else np.nan,
                "SPI最大值": float(np.nanmax(values)) if valid_values.size else np.nan,
                "SPI平均值": float(np.nanmean(values)) if valid_values.size else np.nan,
                "SPI中位数": float(np.nanmedian(values)) if valid_values.size else np.nan,
                "SPI标准差": float(np.nanstd(values)) if valid_values.size else np.nan,
                "中度及以上干旱格点数(SPI<=-1)": int(np.sum(valid_values <= -1.0)),
                "重度及以上干旱格点数(SPI<=-1.5)": int(np.sum(valid_values <= -1.5)),
                "极端干旱格点数(SPI<=-2)": int(np.sum(valid_values <= -2.0)),
            }
        )

    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, file_name: str, description: str) -> Path:
    # 步骤 5.2：保存 CSV 表格。
    # 使用 utf-8-sig 编码，是为了 Windows Excel 直接打开时能正确识别中文列名。
    # 这里同时用于保存逐月统计表和格点计算日志。
    output_path = OUTPUT_DIR / file_name

    with tqdm(
        total=1,
        desc=description,
        colour="yellow",
        bar_format=TQDM_BAR_FORMAT,
    ) as progress:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        progress.update(1)

    return output_path


def read_boundary() -> object | None:
    # 步骤 6.1：读取福建省边界。
    # 边界只用于绘图叠加，不参与 SPI 数值计算；
    # 如果 geopandas 未安装或 shp 文件不存在，脚本会继续运行，只是不绘制省界线；
    # 如果边界文件不是 EPSG:4326，会自动转为 WGS84 经纬度坐标，以匹配降雨栅格。
    if gpd is None:
        print("提示：未检测到 geopandas，将不叠加福建省边界线。")
        return None

    if not BOUNDARY_SHP.exists():
        print(f"提示：未找到边界文件，将不叠加福建省边界线：{BOUNDARY_SHP}")
        return None

    with tqdm(
        total=1,
        desc="读取福建省边界",
        colour="white",
        bar_format=TQDM_BAR_FORMAT,
    ) as progress:
        boundary = gpd.read_file(BOUNDARY_SHP)
        if boundary.crs is not None and boundary.crs.to_epsg() != 4326:
            boundary = boundary.to_crs(epsg=4326)
        progress.update(1)

    return boundary


def plot_monthly_maps(ds_spi: xr.Dataset, boundary: object | None) -> list[Path]:
    # 步骤 6.2：按月份导出 SPI-1 空间分布图。
    # 每个月输出 1 张 PNG，所以 2025 年共输出 12 张图；
    # pcolormesh 直接使用经纬度栅格绘制，不做投影转换；
    # NaN 区域通过 masked_invalid 和 cmap.set_bad 设置为空白/透明。
    # 图内标题按 SPI1-月份 命名，例如 SPI1-1、SPI1-2、SPI1-12；
    # 输出 PNG 文件名仍使用中文，便于按月份识别文件。
    output_paths = []
    spi = ds_spi["SPI1"]
    lats = ds_spi[LAT_NAME].values
    lons = ds_spi[LON_NAME].values
    times = pd.DatetimeIndex(ds_spi[TIME_NAME].values)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    norm = Normalize(vmin=SPI_VMIN, vmax=SPI_VMAX)
    cmap = plt.get_cmap(SPI_COLOR_MAP).copy()
    cmap.set_bad(color=NAN_DISPLAY_COLOR)
    lon_min = float(np.nanmin(lons))
    lon_max = float(np.nanmax(lons))
    lat_min = float(np.nanmin(lats))
    lat_max = float(np.nanmax(lats))
    lon_ticks = np.arange(np.ceil(lon_min), np.floor(lon_max) + 1, LON_TICK_INTERVAL)
    lat_ticks = np.arange(np.ceil(lat_min), np.floor(lat_max) + 1, LAT_TICK_INTERVAL)

    for month_index in tqdm(
        range(spi.sizes[TIME_NAME]),
        desc="导出逐月空间图",
        colour="red",
        bar_format=TQDM_BAR_FORMAT,
    ):
        current_time = times[month_index]
        current_values = np.ma.masked_invalid(spi.isel({TIME_NAME: month_index}).values)

        # 创建单月空间图。
        # FIGURE_SIZE、FIGURE_DPI、SPI_COLOR_MAP、SPI_VMIN、SPI_VMAX 等绘图参数均在脚本顶部集中设置。
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        mesh = ax.pcolormesh(
            lon_grid,
            lat_grid,
            current_values,
            cmap=cmap,
            norm=norm,
            shading="auto",
        )

        if boundary is not None:
            # 叠加福建省行政边界，只用于增强地图可读性，不改变 SPI 栅格值。
            boundary.boundary.plot(
                ax=ax,
                color=BOUNDARY_LINE_COLOR,
                linewidth=BOUNDARY_LINE_WIDTH,
            )

        ax.set_title(
            f"SPI1-{current_time.month}",
            fontsize=TITLE_FONT_SIZE,
        )
        ax.set_xlabel("Longitude", fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_ylabel("Latitude", fontsize=AXIS_LABEL_FONT_SIZE)
        ax.tick_params(labelsize=TICK_FONT_SIZE)
        ax.grid(
            color=GRID_LINE_COLOR,
            linewidth=GRID_LINE_WIDTH,
            alpha=GRID_LINE_ALPHA,
        )
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xticks(lon_ticks)
        ax.set_yticks(lat_ticks)
        ax.set_aspect("equal", adjustable="box")

        colorbar = fig.colorbar(
            mesh,
            ax=ax,
            ticks=SPI_COLORBAR_TICKS,
            extend="both",
            shrink=0.86,
            pad=0.03,
        )
        colorbar.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE)
        colorbar.ax.text(
            COLORBAR_END_TEXT_X,
            COLORBAR_TOP_TEXT_Y,
            COLORBAR_TOP_TEXT,
            ha="center",
            va="bottom",
            fontsize=COLORBAR_END_TEXT_FONT_SIZE,
            transform=colorbar.ax.transAxes,
        )
        colorbar.ax.text(
            COLORBAR_END_TEXT_X,
            COLORBAR_BOTTOM_TEXT_Y,
            COLORBAR_BOTTOM_TEXT,
            ha="center",
            va="top",
            fontsize=COLORBAR_END_TEXT_FONT_SIZE,
            transform=colorbar.ax.transAxes,
        )

        for threshold, line_color in SPI_THRESHOLD_LINES.items():
            # 在色带上绘制 SPI 等级阈值辅助线，方便识别干旱等级。
            # 这些线仅用于读图，不参与任何统计或计算；线条颜色在参数区集中设置。
            colorbar.ax.axhline(
                threshold,
                color=line_color,
                linewidth=SPI_THRESHOLD_LINE_WIDTH,
                alpha=SPI_THRESHOLD_LINE_ALPHA,
            )

        output_path = OUTPUT_DIR / FIGURE_NAME_TEMPLATE.format(month=current_time.month)
        fig.tight_layout()
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def export_threshold_legend() -> Path:
    # 步骤 6.3：单独导出 SPI 阈值线图例。
    # 每张空间图的色带上仍保留彩色阈值横线；
    # 但文字说明不再放在月度空间图内，而是集中保存为一张独立 PNG，便于论文排版或报告组合使用。
    output_path = OUTPUT_DIR / THRESHOLD_LEGEND_FIGURE_NAME
    handles = [
        Line2D(
            [0],
            [0],
            color=line_color,
            lw=SPI_THRESHOLD_LINE_WIDTH,
            alpha=SPI_THRESHOLD_LINE_ALPHA,
            label=SPI_THRESHOLD_LEGEND_LABELS[threshold],
        )
        for threshold, line_color in SPI_THRESHOLD_LINES.items()
    ]

    with tqdm(
        total=1,
        desc="导出阈值线图例",
        colour="red",
        bar_format=TQDM_BAR_FORMAT,
    ) as progress:
        fig, ax = plt.subplots(figsize=THRESHOLD_LEGEND_FIGURE_SIZE)
        ax.axis("off")
        ax.legend(
            handles=handles,
            loc="center",
            ncol=THRESHOLD_LEGEND_COLUMNS,
            frameon=False,
            fontsize=THRESHOLD_LEGEND_FONT_SIZE,
        )
        fig.savefig(output_path, dpi=THRESHOLD_LEGEND_DPI, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        progress.update(1)

    return output_path


def cleanup_temp_cache() -> None:
    # 步骤 7：清理本次脚本运行产生的临时缓存。
    # 只删除 TEMP_CACHE_DIR，不删除 NetCDF、CSV、PNG 等正式结果；
    # 该函数放在 finally 中执行，因此即使中途报错也会尽量清理缓存。
    with tqdm(
        total=1,
        desc="清理临时缓存",
        colour="white",
        bar_format=TQDM_BAR_FORMAT,
    ) as progress:
        if TEMP_CACHE_DIR.exists():
            shutil.rmtree(TEMP_CACHE_DIR)
        progress.update(1)


def main() -> None:
    # 主流程入口：
    # 按“设置字体 -> 读取数据 -> 校验时间轴 -> 计算 SPI -> 保存 NC -> 保存表格 -> 绘图 -> 清理缓存”的顺序执行。
    # 如果运行中出现错误，错误会直接抛出，方便定位问题；finally 仍会清理临时缓存。
    setup_matplotlib_fonts()
    configure_quiet_logging()

    print("开始计算福建省2025年SPI-1指数")
    print(f"输入降雨目录：{INPUT_DIR}")
    print(f"输出结果目录：{OUTPUT_DIR}")
    print(f"校准期：{CALIBRATION_YEAR_INITIAL}-{CALIBRATION_YEAR_FINAL}")
    print(f"评估期：{EVALUATION_YEAR}年1-12月")
    print(f"NaN处理：含任意NaN的格点跳过，输出继续保留NaN")
    print(f"NaN显示颜色：{NAN_DISPLAY_COLOR}（用于区分无数据区域和SPI接近0的浅色区域）")
    print(f"绘图色带：{SPI_COLOR_MAP}，范围 {SPI_VMIN} 到 {SPI_VMAX}")
    print(f"省界线参数：颜色 {BOUNDARY_LINE_COLOR}，线宽 {BOUNDARY_LINE_WIDTH}")
    print(
        "经纬网参数："
        f"颜色 {GRID_LINE_COLOR}，"
        f"线宽 {GRID_LINE_WIDTH}，"
        f"透明度 {GRID_LINE_ALPHA}，"
        f"经度刻度间隔 {LON_TICK_INTERVAL}°，"
        f"纬度刻度间隔 {LAT_TICK_INTERVAL}°"
    )
    print(
        "SPI阈值标注线参数："
        f"颜色映射 {SPI_THRESHOLD_LINES}，"
        f"线宽 {SPI_THRESHOLD_LINE_WIDTH}，"
        f"透明度 {SPI_THRESHOLD_LINE_ALPHA}"
    )
    print(
        "SPI阈值图例参数："
        f"单独图例文件 {THRESHOLD_LEGEND_FIGURE_NAME}，"
        f"列数 {THRESHOLD_LEGEND_COLUMNS}，"
        f"字号 {THRESHOLD_LEGEND_FONT_SIZE}，"
        f"尺寸 {THRESHOLD_LEGEND_FIGURE_SIZE}"
    )
    print(
        "色带端点文字参数："
        f"顶部 {COLORBAR_TOP_TEXT} y={COLORBAR_TOP_TEXT_Y}，"
        f"底部 {COLORBAR_BOTTOM_TEXT} y={COLORBAR_BOTTOM_TEXT_Y}"
    )
    print(
        "日志输出控制："
        f"{'已关闭 climate_indices 逐格点info日志，仅显示进度条' if SUPPRESS_CLIMATE_INDICES_LOGS else '未关闭，调试时会显示库内部日志'}"
    )

    try:
        files = list_precip_files(INPUT_DIR)
        ds_precip = read_precip_dataset(files)
        times = validate_dataset(ds_precip)

        precip_values = ds_precip[PRECIP_VAR].values
        spi_2025, log_df = calculate_spi_2025(precip_values, times)

        ds_spi = build_output_dataset(spi_2025, ds_precip, times)
        netcdf_path = save_netcdf(ds_spi)

        stats_df = make_monthly_statistics(ds_spi)
        stats_path = save_table(stats_df, STATS_CSV_NAME, "保存逐月统计表")
        log_path = save_table(log_df, LOG_CSV_NAME, "保存格点计算日志")

        boundary = read_boundary()
        figure_paths = plot_monthly_maps(ds_spi, boundary)
        threshold_legend_path = export_threshold_legend()

        print("SPI-1计算与导出完成")
        print(f"NetCDF结果：{netcdf_path}")
        print(f"逐月统计表：{stats_path}")
        print(f"格点计算日志：{log_path}")
        print(f"空间分布图数量：{len(figure_paths)} 张")
        print(f"阈值线图例：{threshold_legend_path}")
    finally:
        cleanup_temp_cache()


if __name__ == "__main__":
    main()
