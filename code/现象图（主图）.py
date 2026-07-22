from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import warnings

import geopandas as gpd
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# 0. 论文主线与后续图件路线图
# =============================================================================
# 这段注释用于固定整篇论文的绘图主线，防止后续绘图或建模时偏离主题。
# 当前脚本只实现 Fig. 1“现象图/主图”，但后续 Fig. 2-Fig. 5 都应服务于
# Fig. 1 提出的核心现象，而不是变成相互独立的数据展示。
#
# 全文核心问题：
#   极端干旱是否会改变森林微气候缓冲能力，而且这种改变是否在站点间呈现
#   明显空间异质性？如果存在空间异质性，它是由干旱暴露强度、局地环境过程、
#   空间背景共同塑造，还是由少数站点、某个事件定义或某个区域干旱过程造成？
#
# 核心响应变量：
#   Delta_CBI = Event_CBI - Reference_CBI
#   其中 CBI 是 15 cm 微气候温度对 ERA5 2 m 温度的响应斜率；
#   Delta_CBI > 0 表示极端干旱事件期 CBI 高于参考期，即缓冲减弱；
#   Delta_CBI < 0 表示事件期 CBI 低于参考期，即表观缓冲增强或维持。
#
# 事件定义：
#   主分析把极端干旱定义为 SPI30d <= -2.0；
#   正常参考期定义为 -0.5 < SPI30d < 0.5；
#   有效样本以“事件 x 站点”为基本单元，并要求事件期 CBI 和参考期 CBI
#   均通过质量控制，即 Pair_flag == "ok"。
#
# 推荐正文图件闭环：
#   Fig. 1  发现问题：
#       极端干旱下 CBI 响应在站点间显著分化，并呈现空间结构。
#       本脚本实现该图。它回答“发生了什么、在哪里发生、差异有多大”。
#       需要数据：
#           - Tensor_LatLong.csv：Site_ID, Longitude, Latitude；
#           - daily_SPI 干旱事件长表：Event_ID, Start_Date, End_Date,
#             Duration_Days, Min_Daily_SPI, Drought_Level, Severity；
#           - 事件-参考期 CBI 配对表：Event_CBI, Reference_CBI,
#             Delta_CBI, Pair_flag；
#           - 福建省行政边界 shp；
#           - SPI30d 逐日宽表，用于插图中的每日极端干旱站点比例。
#
#   Fig. 2  排除一个最直接解释：
#       站点差异是否只是因为有些站点遭遇的干旱更强、更久？
#       建议做 3 个面板：
#           Fig. 2A：干旱等级相对 Normal 的 Delta_CBI 剂量响应图。
#               横轴为 Normal/Mild/Moderate/Severe/Extreme；
#               纵轴为相对 Normal 的 Delta_CBI；
#               使用等级混合效应模型的估计值、95% CI 和原始样本点。
#           Fig. 2B：事件最小 SPI30d 或 Severity 与 Delta_CBI 的散点图。
#               每个点为一个有效“事件 x 站点”样本；
#               拟合线应考虑站点、区域或干旱过程的非独立性。
#           Fig. 2C：Duration_Days 与 Delta_CBI 的散点图。
#               用来检查持续时间是否可解释响应分化。
#       需要数据：
#           - 有效事件-站点 CBI 配对表；
#           - 干旱事件长表中的 Minimum_SPI/Min_Daily_SPI、Severity、
#             Duration_Days、Drought_Level；
#           - 等级混合效应模型输出，如等级 Delta_CBI、CI、FDR p 值；
#           - 小时温度/SPI 配对数据，用于模型控制月份、站点重复观测等。
#       预期逻辑：
#           如果控制干旱强度和持续时间后，Delta_CBI 仍存在明显站点差异，
#           则说明“干旱暴露强度”不足以解释 Fig. 1 的空间异质性。
#
#   Fig. 3  识别过程差异：
#       不同缓冲响应组是否经历了不同的水分、植被、能量或大气干燥过程？
#       这不是正式路径模型，而是“机制观察图”，用于筛选进入 Fig. 4 的候选过程。
#       建议做 2 个面板：
#           Fig. 3A：环境变量变化热图。
#               行为有效“事件 x 站点”样本；
#               列为候选环境变量；
#               单元格为事件期相对参考期的标准化变化 Delta_X；
#               行按 Delta_CBI 或响应组排序。
#           Fig. 3B：关键变量在响应组之间的比较图。
#               响应组可按 Delta_CBI < 0、接近 0、Delta_CBI > 0 分组；
#               每个点为事件-站点样本；
#               展示组间均值差异和 95% CI，而不是只报显著性。
#       候选变量模块：
#           - 干旱暴露：Min_Daily_SPI, Severity, Duration_Days；
#           - 水分过程：土壤水分及其事件-参考期变化 Delta_SM；
#           - 植被状态：植被指数、蒸散、冠层调节相关变量；
#           - 能量/辐射过程：辐射、能量输入、地表热状况；
#           - 大气背景：气温、湿度、VPD 或空气干燥度；
#           - 地形/空间背景：海拔、坡度、坡向、经纬度、空间位置。
#       需要数据：
#           - Pair_flag == "ok" 的事件-站点表；
#           - Tensor_Data 或其他动态环境变量；
#           - 每个变量在事件期和参考期的均值、差值 Delta_X、缺失标记；
#           - 站点静态背景表，如地形、植被、区位变量。
#       重要限制：
#           动态变量必须按统一事件期和对应参考期计算 Delta_X；
#           静态变量不能计算事件-参考期差值，只能作为背景变量或分层变量；
#           缺失率高、理论含义弱、与其他变量高度重复的变量不应进入 Fig. 4。
#
#   Fig. 4  解释机制：
#       用路径框架组织 Fig. 3 筛选出的关键过程，回答“为什么空间背景会对应
#       不同 Delta_CBI”。补充框架固定为：
#           Spatial background
#             -> Drought exposure
#             -> Hydrological / vegetation / energy changes
#             -> Delta_CBI
#       建议做 2 个面板：
#           Fig. 4A：预设并筛选后的路径图。
#               节点包括空间背景、干旱暴露、关键环境过程、Delta_CBI；
#               箭头颜色表示正/负效应；
#               箭头宽度表示标准化路径系数绝对值；
#               节点框内可标注解释度 R2 或模型稳定性信息。
#           Fig. 4B：总效应、直接效应和间接效应分解图。
#               用森林图或堆叠条形图展示每个上游变量对 Delta_CBI 的总效应、
#               直接效应、经水分过程的间接效应、经植被/能量过程的间接效应。
#       需要数据：
#           - 每行一个有效“事件 x 站点”样本；
#           - Delta_CBI；
#           - 干旱暴露变量：Minimum_SPI, Severity, Duration_Days；
#           - 动态中介变量：事件-参考期差值 Delta_X；
#           - 静态背景变量：地形、植被、空间位置等；
#           - 分层/控制变量：Site_ID, Event_ID, 区域干旱过程 ID、月份/季节、
#             经纬度、有效小时数、缺失率标记等。
#       方法边界：
#           如果样本量、变量缺失率、共线性或随机效应结构不支持路径模型，
#           不应强行做 SEM；可以退回为“标准化效应森林图/分层总效应模型”，
#           仍然回答哪些过程与 Delta_CBI 最相关，但不声称因果路径。
#       表述原则：
#           Fig. 4 应写成 observational pathway framework 或 pathway analysis；
#           不应写成已经证明严格因果机制。
#
#   Fig. 5  证明可信：
#       回答审稿人对 Fig. 1-Fig. 4 的稳定性质疑：
#           换极端干旱阈值是否还成立？
#           换正常期定义是否还成立？
#           改变最低有效小时数或 CBI 宏观温度变异门槛是否还成立？
#           删除某个站点是否还成立？
#           删除某个区域干旱过程是否还成立？
#       建议做 3 个面板：
#           Fig. 5A：不同事件定义/质量阈值下总体 Delta_CBI 的森林图；
#           Fig. 5B：Leave-one-site-out 逐站点删除森林图；
#           Fig. 5C：Leave-one-regional-process-out 逐区域干旱过程删除森林图。
#       需要数据：
#           - sensitivity_results 中的情景汇总、LMM 输出、事件 CBI 输出；
#           - Leave_one_site_out 目录中的逐站点删除结果；
#           - Leave_one_regional_drought_process_out 目录中的逐区域过程删除结果；
#           - 每个情景下的 Delta_CBI、CI、方向一致性、样本量和收敛状态。
#       预期逻辑：
#           Fig. 5 不要求每个情景都显著，而是要求核心方向、空间异质性模式
#           和主要结论不依赖单一阈值、单一站点或单一区域干旱过程。
#
# 可选 Fig. 6：
#   如果论文问题扩展到“干旱结束后的恢复/韧性”，可增加 Fig. 6；
#   否则建议放补充材料，避免正文主线从“干旱期响应机制”扩散到“恢复过程”。
#   可能内容包括：
#       - 干旱过程 Early/Middle/Late 阶段 CBI；
#       - 干旱结束后 1-7、8-14、15-30 天恢复窗口 CBI；
#       - 7 天或 14 天滑动窗口恢复轨迹。
#
# 当前和下一步的执行顺序：
#   1. 先固定并审定 Fig. 1：现象是否表达清楚，英文标题、色带、图例是否适合 SCI。
#   2. 整理 Fig. 2 所需表：把有效事件-站点配对表与干旱事件暴露变量合并。
#   3. 做 Fig. 2 的暴露解释检验：等级响应、最小 SPI/Severity、持续时间。
#   4. 若 Fig. 2 说明暴露不足以解释空间异质性，再整理 Tensor_Data 动态变量，
#      计算事件期相对参考期的 Delta_X，进入 Fig. 3。
#   5. 从 Fig. 3 选择少量理论明确、缺失可控、非高度重复的变量进入 Fig. 4。
#   6. 最后用 Fig. 5 汇总敏感性、逐站点删除和逐区域过程删除结果，支撑可信度。
#


# =============================================================================
# 1. 路径与科学阈值配置
# =============================================================================
# 本脚本只负责“现象图/主图”的可视化，不重新计算 CBI，也不改动前面分析脚本。
# 需要调整输入或输出位置时，优先修改 Config 中的路径；需要调整图形观感时，
# 优先修改 FigureParameters。所有图、表、参数说明均写入 results\现象图。


@dataclass(frozen=True)
class Config:
    """输入输出路径和干旱判定阈值。

    project_dir:
        项目数据根目录。脚本从这里读取站点坐标、干旱事件表、SPI30d 宽表、
        事件-参考期 CBI 配对表和福建省边界。
    output_dir:
        本脚本所有输出文件的目录。按用户要求固定为 results\现象图。
    extreme_spi_threshold:
        极端干旱事件筛选阈值。SPI30d <= -2.0 对应 extreme drought；
        Fig. 1A 的事件计数和时间足迹均使用该阈值。
    figure_dpi:
        PNG 输出分辨率。SCI 初稿通常 300 dpi 即可，投稿/排版图建议 600 dpi。
    """

    project_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate")
    output_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\results\论文主图\Fig1_现象图")
    extreme_spi_threshold: float = -2.0
    figure_dpi: int = 600

    @property
    def sites_csv(self) -> Path:
        return self.project_dir / "Tensor_LatLong.csv"

    @property
    def drought_events_csv(self) -> Path:
        return (
            self.project_dir
            / "results"
            / "daily_SPI_features"
            / "福建省观测站2025年daily_SPI干旱事件长表.csv"
        )

    @property
    def spi_wide_xlsx(self) -> Path:
        return (
            self.project_dir
            / "results"
            / "daily_SPI_result"
            / "各站点SPI30d逐日宽表_2025.xlsx"
        )

    @property
    def event_pairs_csv(self) -> Path:
        return (
            self.project_dir
            / "results"
            / "compare_differences_results"
            / "04_极端事件CBI与事件后正常参考期对比表.csv"
        )

    @property
    def fujian_shp(self) -> Path:
        return self.project_dir / "Fujian_Shp" / "福建省行政边界.shp"


@dataclass(frozen=True)
class FigureParameters:
    """集中管理 Fig. 1 的图例、线条、点大小、色带和版式参数。

    修改原则：
        - 点大小参数控制视觉权重，不改变统计结果；
        - 颜色参数只改变显示方式，不改变 Delta_CBI 或 SPI30d 数值；
        - 版式参数控制面板间距、色标位置、插图位置，适合后续按期刊版面微调。
    """

    # 全图尺寸和排版。fig_width/fig_height 单位为英寸；wspace/hspace 控制面板间距。
    fig_width: float = 12.2
    fig_height: float = 9.8
    grid_left: float = 0.06
    grid_right: float = 0.94
    grid_bottom: float = 0.10
    grid_top: float = 0.94
    grid_wspace: float = 0.22
    grid_hspace: float = 0.38
    suptitle_y: float = 0.982

    # 字体和线宽。SCI 图建议字体清晰、线宽克制，避免过度装饰。
    axes_linewidth: float = 0.8
    axis_label_size: int = 9
    title_size: int = 10
    tick_label_size: int = 8
    legend_font_size: int = 8
    suptitle_size: int = 11

    # 地图边界与底色。边界线用于明确研究区范围，底色保持低饱和避免压过站点。
    map_facecolor_a: str = "#f7f4ec"
    map_facecolor_c: str = "#f7f7f7"
    map_edgecolor: str = "#404040"
    map_edgewidth: float = 0.8
    map_padding_fraction: float = 0.06

    # Fig. 1A：极端干旱事件频次点图参数。
    panel_a_zero_site_size: float = 18
    panel_a_zero_site_color: str = "#b8b8b8"
    panel_a_event_size_min: float = 35
    panel_a_event_size_max: float = 115
    panel_a_event_cmap: str = "OrRd"
    panel_a_event_edgecolor: str = "#3a1f12"
    panel_a_event_linewidth: float = 0.45
    panel_a_event_alpha: float = 0.92
    panel_a_legend_loc: str = "upper left"
    panel_a_legend_anchor_x: float = 0.01
    panel_a_legend_anchor_y: float = 0.98

    # Fig. 1A 插入时间轴：显示每日达到极端干旱阈值的站点比例。
    panel_a_inset_x: float = 0.52
    panel_a_inset_y: float = 0.06
    panel_a_inset_width: float = 0.44
    panel_a_inset_height: float = 0.25
    panel_a_inset_title_size: int = 6
    panel_a_inset_label_size: int = 6
    panel_a_inset_tick_size: int = 5
    panel_a_timeline_fill_color: str = "#d95f02"
    panel_a_timeline_fill_alpha: float = 0.26
    panel_a_timeline_line_color: str = "#9d2f14"
    panel_a_timeline_linewidth: float = 1.1
    panel_a_timeline_month_interval: int = 2

    # Fig. 1B/C：Delta_CBI 发散色带。以 0 为中心，蓝色表示事件期 CBI 低于参考期，
    # 红色表示事件期 CBI 高于参考期。
    delta_cmap: str = "RdBu_r"
    delta_limit_padding: float = 1.15
    panel_b_point_size_min: float = 35
    panel_b_point_size_max: float = 110
    panel_b_point_alpha: float = 0.86
    panel_b_point_edgecolor: str = "#252525"
    panel_b_point_linewidth: float = 0.45
    panel_b_identity_line_color: str = "#606060"
    panel_b_identity_linewidth: float = 0.9
    panel_b_identity_linestyle: str = "--"
    panel_b_axis_padding_fraction: float = 0.12
    panel_b_hist_bins: int = 8
    panel_b_hist_color: str = "#6f6f6f"
    panel_b_hist_alpha: float = 0.85
    panel_b_hist_zero_line_color: str = "#222222"
    panel_b_hist_zero_linewidth: float = 0.8

    # Fig. 1C：站点中位数 Delta_CBI 地图参数。
    panel_c_background_site_size: float = 15
    panel_c_background_site_color: str = "#c7c7c7"
    panel_c_response_size_min: float = 45
    panel_c_response_size_max: float = 145
    panel_c_response_edgecolor: str = "#1f1f1f"
    panel_c_response_linewidth: float = 0.45
    panel_c_response_alpha: float = 0.93
    panel_c_legend_loc: str = "lower right"
    panel_c_legend_anchor_x: float = 0.99
    panel_c_legend_anchor_y: float = 0.01

    # Fig. 1D：站点排序散点图。点颜色按 Min_Daily_SPI，不重复编码横轴 Delta_CBI。
    panel_d_spi_cmap: str = "YlOrBr_r"
    panel_d_event_point_size: float = 30
    panel_d_event_point_alpha: float = 0.78
    panel_d_event_edgecolor: str = "white"
    panel_d_event_linewidth: float = 0.35
    panel_d_site_median_marker: str = "D"
    panel_d_site_median_size: float = 45
    panel_d_site_median_facecolor: str = "white"
    panel_d_site_median_edgecolor: str = "#111111"
    panel_d_site_median_linewidth: float = 0.9
    panel_d_zero_line_color: str = "#333333"
    panel_d_zero_linewidth: float = 0.85
    panel_d_zero_linestyle: str = "--"
    panel_d_row_line_color: str = "#e5e5e5"
    panel_d_row_linewidth: float = 0.45
    panel_d_jitter_half_width: float = 0.16
    panel_d_jitter_seed: int = 20250720
    panel_d_n_label_fontsize: int = 7
    panel_d_n_label_offset_fraction: float = 0.018
    panel_d_site_label_size: int = 6

    # 色标位置。坐标为相对所在面板的 [x, y, width, height]。
    delta_colorbar_x: float = 0.04
    delta_colorbar_y: float = 0.88
    delta_colorbar_width: float = 0.30
    delta_colorbar_height: float = 0.035
    spi_colorbar_x: float = 0.04
    spi_colorbar_y: float = 0.90
    spi_colorbar_width: float = 0.30
    spi_colorbar_height: float = 0.035
    colorbar_tick_size: int = 7
    colorbar_label_size: int = 7


CFG = Config()
FP = FigureParameters()


# =============================================================================
# 2. 进度条工具与基础函数
# =============================================================================
# tqdm 使用 leave=False 和 dynamic_ncols=True，实现终端单行动态刷新，避免日志刷屏。
# 不同步骤使用不同 colour，便于快速区分“读取、处理、绘图、保存、清理”等阶段。


PROGRESS_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
    "[elapsed: {elapsed}, remaining: {remaining}, {rate_fmt}]"
)


def progress_bar(desc: str, total: int, colour: str) -> tqdm:
    """创建一个单步骤、单行刷新的彩色 tqdm 进度条。"""

    return tqdm(
        total=total,
        desc=desc,
        colour=colour,
        dynamic_ncols=True,
        leave=False,
        ncols=100,
        bar_format=PROGRESS_BAR_FORMAT,
    )


def normalize_site_id(value: object) -> str:
    """统一站点编号格式，避免 CSV/Excel 把站点 ID 读成 95332217.0。"""

    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def setup_style() -> None:
    """设置全局 Matplotlib 风格。

    这里使用英文字体和可嵌入 PDF 字体，适合 SCI 主图后续进入 AI/Inkscape
    或期刊排版系统继续编辑。
    """

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "axes.unicode_minus": False,
            "axes.linewidth": FP.axes_linewidth,
            "axes.labelsize": FP.axis_label_size,
            "axes.titlesize": FP.title_size,
            "xtick.labelsize": FP.tick_label_size,
            "ytick.labelsize": FP.tick_label_size,
            "legend.fontsize": FP.legend_font_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def require_files(paths: list[Path]) -> None:
    """检查必要输入是否存在；缺失时提前报错，避免绘图中途失败。"""

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))


def finite_symmetric_limit(values: pd.Series, fallback: float = 0.2) -> float:
    """计算以 0 为中心的对称色带/坐标范围，保证正负 Delta_CBI 可比。"""

    arr = pd.to_numeric(values, errors="coerce").to_numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return fallback
    lim = float(np.nanmax(np.abs(arr)))
    if not np.isfinite(lim) or lim == 0:
        return fallback
    return lim * FP.delta_limit_padding


def size_from_count(
    count: pd.Series | np.ndarray,
    min_size: float,
    max_size: float,
    vmax: float | None = None,
) -> np.ndarray:
    """把事件数/配对数转换为散点面积。

    使用平方根缩放，避免最大值站点视觉上过度支配图面。
    """

    values = np.asarray(count, dtype=float)
    if values.size == 0:
        return values
    if vmax is None:
        vmax = np.nanmax(values)
    if not np.isfinite(vmax) or vmax <= 0:
        return np.full_like(values, min_size, dtype=float)
    return min_size + (np.sqrt(values) / np.sqrt(vmax)) * (max_size - min_size)


def format_longitude(value: float, _pos: object) -> str:
    """把经度刻度显示为 116°E，而不是在坐标轴标题中写 deg E。"""

    return f"{value:g}°E"


def format_latitude(value: float, _pos: object) -> str:
    """把纬度刻度显示为 24°N，而不是在坐标轴标题中写 deg N。"""

    return f"{value:g}°N"


def apply_lon_lat_tick_format(ax: plt.Axes) -> None:
    """统一地图面板经纬度刻度格式。"""

    ax.xaxis.set_major_formatter(FuncFormatter(format_longitude))
    ax.yaxis.set_major_formatter(FuncFormatter(format_latitude))


# =============================================================================
# 3. 数据读取与汇总
# =============================================================================


def read_inputs() -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取边界、站点、事件、CBI 配对和 SPI30d 宽表。

    关键处理：
        - 福建边界统一到 EPSG:4326，保证和站点经纬度一致；
        - Site_ID 全部转为字符串，避免跨表合并失败；
        - 日期列和数值列显式转换，减少 Excel/CSV 自动类型推断造成的隐患。
    """

    steps = [
        "检查输入文件",
        "读取福建边界",
        "读取站点坐标",
        "读取干旱事件表",
        "读取事件-参考期CBI配对表",
        "读取SPI30d逐日宽表",
        "统一字段类型",
    ]
    with progress_bar("步骤1/5 读取输入数据", len(steps), "cyan") as bar:
        require_files(
            [
                CFG.sites_csv,
                CFG.drought_events_csv,
                CFG.spi_wide_xlsx,
                CFG.event_pairs_csv,
                CFG.fujian_shp,
            ]
        )
        bar.update()

        boundary = gpd.read_file(CFG.fujian_shp)
        if boundary.crs is None:
            raise ValueError(f"Boundary shapefile has no CRS: {CFG.fujian_shp}")
        boundary = boundary.to_crs(epsg=4326)
        bar.update()

        sites = pd.read_csv(CFG.sites_csv)
        bar.update()

        events = pd.read_csv(CFG.drought_events_csv)
        bar.update()

        pairs = pd.read_csv(CFG.event_pairs_csv)
        bar.update()

        spi_wide = pd.read_excel(CFG.spi_wide_xlsx)
        bar.update()

        for df in (sites, events, pairs):
            df["Site_ID"] = df["Site_ID"].map(normalize_site_id)

        for col in ["Longitude", "Latitude"]:
            sites[col] = pd.to_numeric(sites[col], errors="coerce")

        events["Start_Date"] = pd.to_datetime(events["Start_Date"], errors="coerce")
        events["End_Date"] = pd.to_datetime(events["End_Date"], errors="coerce")
        events["Min_Daily_SPI"] = pd.to_numeric(events["Min_Daily_SPI"], errors="coerce")

        pairs["Start_Date"] = pd.to_datetime(pairs["Start_Date"], errors="coerce")
        pairs["End_Date"] = pd.to_datetime(pairs["End_Date"], errors="coerce")
        numeric_pair_cols = [
            "Duration_Days",
            "Min_Daily_SPI",
            "Event_CBI",
            "Reference_CBI",
            "Delta_CBI_Event_minus_Reference",
        ]
        for col in numeric_pair_cols:
            pairs[col] = pd.to_numeric(pairs[col], errors="coerce")

        first_col = spi_wide.columns[0]
        spi_wide = spi_wide.rename(columns={first_col: "Date"})
        spi_wide["Date"] = pd.to_datetime(spi_wide["Date"], errors="coerce")
        for col in spi_wide.columns:
            if col != "Date":
                spi_wide[col] = pd.to_numeric(spi_wide[col], errors="coerce")
        bar.update()

    return boundary, sites, events, pairs, spi_wide


def prepare_data(
    sites: pd.DataFrame,
    events: pd.DataFrame,
    pairs: pd.DataFrame,
    spi_wide: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """生成 Fig. 1A-D 所需的统计表。

    输出含义：
        - sites_summary：每个站点的极端干旱事件数，用于 Fig. 1A；
        - extreme_events：所有极端干旱事件，用于审计；
        - valid_pairs：Pair_flag == ok 的事件-参考期 CBI 配对，用于 Fig. 1B/D；
        - site_response：按站点汇总的 Delta_CBI 中位数和有效配对数，用于 Fig. 1C/D；
        - timeline：每日极端干旱站点比例，用于 Fig. 1A 插图。
    """

    with progress_bar("步骤2/5 清洗并汇总数据", 5, "green") as bar:
        extreme_events = events.loc[
            events["Min_Daily_SPI"].le(CFG.extreme_spi_threshold)
            & events["Start_Date"].notna()
            & events["End_Date"].notna()
        ].copy()
        bar.update()

        extreme_counts = (
            extreme_events.groupby("Site_ID", as_index=False)
            .agg(
                Extreme_events=("Event_ID", "count"),
                Earliest_event=("Start_Date", "min"),
                Latest_event=("End_Date", "max"),
                Min_SPI=("Min_Daily_SPI", "min"),
            )
        )
        sites_summary = sites.merge(extreme_counts, on="Site_ID", how="left")
        sites_summary["Extreme_events"] = sites_summary["Extreme_events"].fillna(0).astype(int)
        bar.update()

        valid_pairs = pairs.loc[
            pairs["Pair_flag"].eq("ok")
            & pairs["Event_CBI"].notna()
            & pairs["Reference_CBI"].notna()
            & pairs["Delta_CBI_Event_minus_Reference"].notna()
        ].copy()
        valid_pairs = valid_pairs.rename(columns={"Delta_CBI_Event_minus_Reference": "Delta_CBI"})
        bar.update()

        site_response = (
            valid_pairs.groupby("Site_ID", as_index=False)
            .agg(
                Median_Delta_CBI=("Delta_CBI", "median"),
                Mean_Delta_CBI=("Delta_CBI", "mean"),
                N_pairs=("Delta_CBI", "size"),
                Min_SPI=("Min_Daily_SPI", "min"),
                Median_duration_days=("Duration_Days", "median"),
            )
            .merge(sites[["Site_ID", "Longitude", "Latitude"]], on="Site_ID", how="left")
        )
        bar.update()

        spi_cols = [col for col in spi_wide.columns if col != "Date"]
        timeline = pd.DataFrame({"Date": spi_wide["Date"]})
        spi_values = spi_wide[spi_cols]
        timeline["Mean_SPI30d"] = spi_values.mean(axis=1, skipna=True)
        timeline["Extreme_site_fraction"] = spi_values.le(CFG.extreme_spi_threshold).sum(
            axis=1
        ) / spi_values.notna().sum(axis=1)
        timeline = timeline.loc[timeline["Date"].notna()].copy()
        bar.update()

    return sites_summary, extreme_events, valid_pairs, site_response, timeline


# =============================================================================
# 4. 四个面板绘图函数
# =============================================================================


def draw_panel_a(
    ax: plt.Axes,
    boundary: gpd.GeoDataFrame,
    sites_summary: pd.DataFrame,
    timeline: pd.DataFrame,
) -> None:
    """Fig. 1A：福建边界、全部站点和极端干旱事件频次。

    点大小表示每个站点 2025 年满足 SPI30d <= extreme_spi_threshold 的事件数；
    右下角插图显示每日处于极端干旱的站点比例，用来交代事件发生时间背景。
    """

    boundary.plot(
        ax=ax,
        facecolor=FP.map_facecolor_a,
        edgecolor=FP.map_edgecolor,
        linewidth=FP.map_edgewidth,
        zorder=1,
    )
    boundary.boundary.plot(ax=ax, color=FP.map_edgecolor, linewidth=FP.map_edgewidth, zorder=2)

    zero = sites_summary["Extreme_events"].eq(0)
    ax.scatter(
        sites_summary.loc[zero, "Longitude"],
        sites_summary.loc[zero, "Latitude"],
        s=FP.panel_a_zero_site_size,
        c=FP.panel_a_zero_site_color,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )

    affected = sites_summary.loc[~zero].copy()
    max_events = float(affected["Extreme_events"].max()) if not affected.empty else 1.0
    point_sizes = size_from_count(
        affected["Extreme_events"],
        FP.panel_a_event_size_min,
        FP.panel_a_event_size_max,
        vmax=max_events,
    )
    ax.scatter(
        affected["Longitude"],
        affected["Latitude"],
        s=point_sizes,
        c=affected["Extreme_events"],
        cmap=FP.panel_a_event_cmap,
        vmin=1,
        vmax=max(1, int(affected["Extreme_events"].max())),
        edgecolors=FP.panel_a_event_edgecolor,
        linewidths=FP.panel_a_event_linewidth,
        alpha=FP.panel_a_event_alpha,
        zorder=4,
    )

    bounds = boundary.total_bounds
    dx = bounds[2] - bounds[0]
    dy = bounds[3] - bounds[1]
    ax.set_xlim(bounds[0] - dx * FP.map_padding_fraction, bounds[2] + dx * FP.map_padding_fraction)
    ax.set_ylim(bounds[1] - dy * FP.map_padding_fraction, bounds[3] + dy * FP.map_padding_fraction)
    apply_lon_lat_tick_format(ax)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("A  Extreme drought events and monitoring sites", loc="left", fontweight="bold")

    legend_counts = sorted(set(affected["Extreme_events"].astype(int)))
    if len(legend_counts) > 3:
        legend_counts = [legend_counts[0], int(np.median(legend_counts)), legend_counts[-1]]
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#f26d3d",
            markeredgecolor=FP.panel_a_event_edgecolor,
            markersize=np.sqrt(
                size_from_count(
                    np.array([count]),
                    FP.panel_a_event_size_min,
                    FP.panel_a_event_size_max,
                    vmax=max_events,
                )[0]
            )
            / 1.35,
            label=str(count),
        )
        for count in legend_counts
    ]
    ax.legend(
        handles=handles,
        title="Extreme events",
        frameon=False,
        loc=FP.panel_a_legend_loc,
        bbox_to_anchor=(FP.panel_a_legend_anchor_x, FP.panel_a_legend_anchor_y),
        borderaxespad=0.0,
    )

    inset = ax.inset_axes(
        [
            FP.panel_a_inset_x,
            FP.panel_a_inset_y,
            FP.panel_a_inset_width,
            FP.panel_a_inset_height,
        ]
    )
    inset.fill_between(
        timeline["Date"],
        timeline["Extreme_site_fraction"] * 100,
        color=FP.panel_a_timeline_fill_color,
        alpha=FP.panel_a_timeline_fill_alpha,
        linewidth=0,
    )
    inset.plot(
        timeline["Date"],
        timeline["Extreme_site_fraction"] * 100,
        color=FP.panel_a_timeline_line_color,
        linewidth=FP.panel_a_timeline_linewidth,
    )
    inset.set_ylabel("% sites", fontsize=FP.panel_a_inset_label_size)
    inset.set_xlabel("2025", fontsize=FP.panel_a_inset_label_size)
    inset.set_title("Extreme-drought footprint", fontsize=FP.panel_a_inset_title_size, pad=2)
    inset.tick_params(axis="both", labelsize=FP.panel_a_inset_tick_size, length=2)
    inset.xaxis.set_major_locator(mdates.MonthLocator(interval=FP.panel_a_timeline_month_interval))
    inset.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    inset.set_ylim(bottom=0)
    for spine in inset.spines.values():
        spine.set_linewidth(0.5)


def draw_panel_b(ax: plt.Axes, valid_pairs: pd.DataFrame, delta_norm: TwoSlopeNorm) -> None:
    """Fig. 1B：事件期 CBI 与参考期 CBI 的配对散点图。

    y = x 虚线表示事件期和参考期 CBI 相同；点在虚线上方表示事件期 CBI 更高。
    点颜色为 Delta_CBI，点大小随干旱持续时间变化，插图显示 Delta_CBI 分布。
    """

    durations = valid_pairs["Duration_Days"].fillna(valid_pairs["Duration_Days"].median())
    size = FP.panel_b_point_size_min + (
        (FP.panel_b_point_size_max - FP.panel_b_point_size_min)
        * (durations - durations.min())
        / max(1, durations.max() - durations.min())
    )
    ax.scatter(
        valid_pairs["Reference_CBI"],
        valid_pairs["Event_CBI"],
        c=valid_pairs["Delta_CBI"],
        s=size,
        cmap=FP.delta_cmap,
        norm=delta_norm,
        edgecolors=FP.panel_b_point_edgecolor,
        linewidths=FP.panel_b_point_linewidth,
        alpha=FP.panel_b_point_alpha,
    )

    low = float(np.nanmin(valid_pairs[["Reference_CBI", "Event_CBI"]].to_numpy()))
    high = float(np.nanmax(valid_pairs[["Reference_CBI", "Event_CBI"]].to_numpy()))
    pad = (high - low) * FP.panel_b_axis_padding_fraction if high > low else 0.1
    ax.plot(
        [low - pad, high + pad],
        [low - pad, high + pad],
        FP.panel_b_identity_linestyle,
        color=FP.panel_b_identity_line_color,
        lw=FP.panel_b_identity_linewidth,
    )
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel("Reference-period CBI")
    ax.set_ylabel("Extreme-event CBI")
    ax.set_title("B  Event-period CBI shifted from reference conditions", loc="left", fontweight="bold")
    ax.text(
        0.04,
        0.76,
        f"n = {len(valid_pairs)} event-site pairs\n"
        f"median ΔCBI = {valid_pairs['Delta_CBI'].median():.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.28",
            "fc": "white",
            "ec": "#d0d0d0",
            "lw": 0.6,
            "alpha": 0.88,
        },
    )

    inset = ax.inset_axes([0.63, 0.08, 0.32, 0.28])
    inset.hist(
        valid_pairs["Delta_CBI"],
        bins=FP.panel_b_hist_bins,
        color=FP.panel_b_hist_color,
        alpha=FP.panel_b_hist_alpha,
        edgecolor="white",
    )
    inset.axvline(0, color=FP.panel_b_hist_zero_line_color, lw=FP.panel_b_hist_zero_linewidth)
    inset.set_title("ΔCBI", fontsize=7, pad=1)
    inset.tick_params(axis="both", labelsize=6, length=2)
    for spine in inset.spines.values():
        spine.set_linewidth(0.5)


def draw_panel_c(
    ax: plt.Axes,
    boundary: gpd.GeoDataFrame,
    sites_summary: pd.DataFrame,
    site_response: pd.DataFrame,
    delta_norm: TwoSlopeNorm,
) -> None:
    """Fig. 1C：站点尺度中位数 Delta_CBI 的空间分布图。

    底图灰点显示全部站点；有有效事件-参考期配对的站点按中位数 Delta_CBI 着色，
    点大小表示有效配对数量 n。
    """

    boundary.plot(
        ax=ax,
        facecolor=FP.map_facecolor_c,
        edgecolor=FP.map_edgecolor,
        linewidth=FP.map_edgewidth,
        zorder=1,
    )
    ax.scatter(
        sites_summary["Longitude"],
        sites_summary["Latitude"],
        s=FP.panel_c_background_site_size,
        c=FP.panel_c_background_site_color,
        edgecolors="white",
        linewidths=0.35,
        zorder=2,
    )
    max_pairs = float(site_response["N_pairs"].max()) if not site_response.empty else 1.0
    sizes = size_from_count(
        site_response["N_pairs"],
        FP.panel_c_response_size_min,
        FP.panel_c_response_size_max,
        vmax=max_pairs,
    )
    ax.scatter(
        site_response["Longitude"],
        site_response["Latitude"],
        s=sizes,
        c=site_response["Median_Delta_CBI"],
        cmap=FP.delta_cmap,
        norm=delta_norm,
        edgecolors=FP.panel_c_response_edgecolor,
        linewidths=FP.panel_c_response_linewidth,
        alpha=FP.panel_c_response_alpha,
        zorder=3,
    )

    bounds = boundary.total_bounds
    dx = bounds[2] - bounds[0]
    dy = bounds[3] - bounds[1]
    ax.set_xlim(bounds[0] - dx * FP.map_padding_fraction, bounds[2] + dx * FP.map_padding_fraction)
    ax.set_ylim(bounds[1] - dy * FP.map_padding_fraction, bounds[3] + dy * FP.map_padding_fraction)
    apply_lon_lat_tick_format(ax)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("C  Spatially structured median responses", loc="left", fontweight="bold")

    legend_counts = sorted(set(site_response["N_pairs"].astype(int)))
    if len(legend_counts) > 3:
        legend_counts = [legend_counts[0], int(np.median(legend_counts)), legend_counts[-1]]
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#d9d9d9",
            markeredgecolor=FP.panel_c_response_edgecolor,
            markersize=np.sqrt(
                size_from_count(
                    np.array([count]),
                    FP.panel_c_response_size_min,
                    FP.panel_c_response_size_max,
                    vmax=max_pairs,
                )[0]
            )
            / 1.3,
            label=str(count),
        )
        for count in legend_counts
    ]
    ax.legend(
        handles=handles,
        title="Valid pairs",
        frameon=False,
        loc=FP.panel_c_legend_loc,
        bbox_to_anchor=(FP.panel_c_legend_anchor_x, FP.panel_c_legend_anchor_y),
        borderaxespad=0,
    )


def draw_panel_d(
    ax: plt.Axes,
    valid_pairs: pd.DataFrame,
    site_response: pd.DataFrame,
) -> mpl.collections.PathCollection:
    """Fig. 1D：按站点中位数排序的事件级 Delta_CBI 点图。

    横轴为单个事件-站点配对的 Delta_CBI；白色菱形表示站点中位数；
    点颜色表示该事件的最低 SPI30d，用来保留事件强度信息。
    """

    ordered = site_response.sort_values("Median_Delta_CBI", ascending=True).reset_index(drop=True)
    site_to_y = {site: i for i, site in enumerate(ordered["Site_ID"])}
    plot_data = valid_pairs.loc[valid_pairs["Site_ID"].isin(site_to_y)].copy()
    plot_data["y"] = plot_data["Site_ID"].map(site_to_y).astype(float)

    rng = np.random.default_rng(FP.panel_d_jitter_seed)
    plot_data["jitter"] = rng.uniform(-FP.panel_d_jitter_half_width, FP.panel_d_jitter_half_width, len(plot_data))
    spi_norm = Normalize(
        vmin=float(plot_data["Min_Daily_SPI"].min()),
        vmax=float(plot_data["Min_Daily_SPI"].max()),
    )
    cmap = mpl.colormaps[FP.panel_d_spi_cmap]

    ax.axvline(
        0,
        color=FP.panel_d_zero_line_color,
        lw=FP.panel_d_zero_linewidth,
        linestyle=FP.panel_d_zero_linestyle,
        zorder=1,
    )
    sc = ax.scatter(
        plot_data["Delta_CBI"],
        plot_data["y"] + plot_data["jitter"],
        c=plot_data["Min_Daily_SPI"],
        cmap=cmap,
        norm=spi_norm,
        s=FP.panel_d_event_point_size,
        alpha=FP.panel_d_event_point_alpha,
        edgecolors=FP.panel_d_event_edgecolor,
        linewidths=FP.panel_d_event_linewidth,
        zorder=2,
    )

    ax.scatter(
        ordered["Median_Delta_CBI"],
        np.arange(len(ordered)),
        marker=FP.panel_d_site_median_marker,
        s=FP.panel_d_site_median_size,
        c=FP.panel_d_site_median_facecolor,
        edgecolors=FP.panel_d_site_median_edgecolor,
        linewidths=FP.panel_d_site_median_linewidth,
        zorder=4,
        label="Site median",
    )
    for y in range(len(ordered)):
        ax.hlines(y, -1, 1, color=FP.panel_d_row_line_color, lw=FP.panel_d_row_linewidth, zorder=0)

    xlim = finite_symmetric_limit(valid_pairs["Delta_CBI"], fallback=0.25)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-0.75, len(ordered) - 0.25)
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels(ordered["Site_ID"], fontsize=FP.panel_d_site_label_size)
    ax.set_xlabel("ΔCBI (event - reference)")
    ax.set_ylabel("Site ordered by median ΔCBI")
    ax.set_title("D  Site heterogeneity persisted within events", loc="left", fontweight="bold")

    right_x = ax.get_xlim()[1]
    offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * FP.panel_d_n_label_offset_fraction
    for _, row in ordered.iterrows():
        ax.text(
            right_x + offset,
            site_to_y[row["Site_ID"]],
            f"n={int(row['N_pairs'])}",
            va="center",
            ha="left",
            fontsize=FP.panel_d_n_label_fontsize,
            clip_on=False,
        )
    ax.legend(frameon=False, loc="upper left")
    return sc


# =============================================================================
# 5. 输出表格、参数说明和缓存清理
# =============================================================================


def parameters_to_table() -> pd.DataFrame:
    """把 Config 和 FigureParameters 写成可读表格，便于后续按参数名修改。"""

    rows: list[dict[str, object]] = []
    descriptions = {
        "extreme_spi_threshold": "极端干旱阈值；SPI30d <= 该值被视为 extreme drought。",
        "figure_dpi": "PNG 输出分辨率；投稿图建议 600 dpi。",
        "fig_width": "组合图宽度，单位英寸。",
        "fig_height": "组合图高度，单位英寸。",
        "grid_wspace": "左右面板水平间距；增大可减少面板拥挤。",
        "grid_hspace": "上下两行面板间距；增大可避免色标和标题重叠。",
        "panel_a_event_size_min": "Fig.1A 极端事件站点最小点面积。",
        "panel_a_event_size_max": "Fig.1A 极端事件站点最大点面积。",
        "panel_a_timeline_linewidth": "Fig.1A 插图中每日极端干旱站点比例折线宽度。",
        "delta_cmap": "Fig.1B/C ΔCBI 发散色带；当前红色为增加、蓝色为降低。",
        "panel_b_identity_linewidth": "Fig.1B y=x 参考线宽度。",
        "panel_b_identity_linestyle": "Fig.1B y=x 参考线线型。",
        "panel_b_hist_bins": "Fig.1B 插图 Delta_CBI 直方图分箱数量。",
        "panel_c_response_size_min": "Fig.1C 有效配对站点最小点面积。",
        "panel_c_response_size_max": "Fig.1C 有效配对站点最大点面积。",
        "panel_d_spi_cmap": "Fig.1D 事件点按最低 SPI30d 着色的色带。",
        "panel_d_event_point_size": "Fig.1D 单个事件点大小。",
        "panel_d_jitter_half_width": "Fig.1D 同站点事件点的上下抖动半宽，避免点完全重叠。",
        "delta_colorbar_y": "Fig.1B/C ΔCBI 色标相对位置 y；负值表示放在面板下方。",
        "spi_colorbar_y": "Fig.1D SPI 色标相对位置 y；负值表示放在面板下方。",
    }

    for name, value in asdict(CFG).items():
        rows.append(
            {
                "参数组": "路径与阈值",
                "参数名": name,
                "当前值": str(value),
                "用途说明": descriptions.get(name, "输入输出路径或全局阈值。"),
            }
        )
    for name, value in asdict(FP).items():
        rows.append(
            {
                "参数组": "图形参数",
                "参数名": name,
                "当前值": value,
                "用途说明": descriptions.get(name, "控制图例、线条、点大小、颜色、字体或版式。"),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    fig: plt.Figure,
    sites_summary: pd.DataFrame,
    extreme_events: pd.DataFrame,
    valid_pairs: pd.DataFrame,
    site_response: pd.DataFrame,
) -> pd.DataFrame:
    """保存所有图、表和参数说明，文件名全部使用中文。"""

    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    with progress_bar("步骤4/5 保存图表与参数", 8, "magenta") as bar:
        fig.savefig(CFG.output_dir / "图1_现象图_主图.png", dpi=CFG.figure_dpi)
        bar.update()

        fig.savefig(CFG.output_dir / "图1_现象图_主图.pdf")
        bar.update()

        sites_summary.to_csv(CFG.output_dir / "图1A_站点极端干旱事件次数表.csv", index=False, encoding="utf-8-sig")
        bar.update()

        extreme_events.to_csv(CFG.output_dir / "图1A_极端干旱事件明细表.csv", index=False, encoding="utf-8-sig")
        bar.update()

        valid_pairs.to_csv(CFG.output_dir / "图1B和图1D_有效事件站点配对表.csv", index=False, encoding="utf-8-sig")
        bar.update()

        site_response.to_csv(CFG.output_dir / "图1C和图1D_站点DeltaCBI汇总表.csv", index=False, encoding="utf-8-sig")
        bar.update()

        parameters_to_table().to_csv(CFG.output_dir / "00_现象图绘图参数说明表.csv", index=False, encoding="utf-8-sig")
        bar.update()

        summary = pd.DataFrame(
            [
                {
                    "总站点数": int(sites_summary["Site_ID"].nunique()),
                    "出现极端干旱事件的站点数": int((sites_summary["Extreme_events"] > 0).sum()),
                    "有效事件站点配对数": int(len(valid_pairs)),
                    "有有效配对的站点数": int(valid_pairs["Site_ID"].nunique()),
                    "Delta_CBI均值": float(valid_pairs["Delta_CBI"].mean()),
                    "Delta_CBI中位数": float(valid_pairs["Delta_CBI"].median()),
                    "Delta_CBI为正的配对数": int((valid_pairs["Delta_CBI"] > 0).sum()),
                    "Delta_CBI为负的配对数": int((valid_pairs["Delta_CBI"] < 0).sum()),
                    "PNG图件": str(CFG.output_dir / "图1_现象图_主图.png"),
                    "PDF图件": str(CFG.output_dir / "图1_现象图_主图.pdf"),
                }
            ]
        )
        summary.to_csv(CFG.output_dir / "图1_现象图_运行摘要表.csv", index=False, encoding="utf-8-sig")
        bar.update()

    return summary


def cleanup_runtime_cache() -> None:
    """清理本脚本运行可能产生的缓存和临时文件。

    清理范围刻意保守：
        - 仅删除当前代码目录 __pycache__ 中与本脚本同名的 .pyc；
        - 仅删除输出目录下显式临时目录或临时文件后缀；
        - 不删除全局 Matplotlib 字体缓存和其他脚本缓存，避免影响已有环境。
    """

    with progress_bar("步骤5/5 清理运行缓存", 3, "blue") as bar:
        script_stem = Path(__file__).stem
        pycache_dir = Path(__file__).resolve().parent / "__pycache__"
        if pycache_dir.exists():
            for pyc in pycache_dir.glob(f"{script_stem}*.pyc"):
                pyc.unlink(missing_ok=True)
        bar.update()

        temp_dir = CFG.output_dir / "_临时缓存"
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        bar.update()

        if CFG.output_dir.exists():
            for pattern in ("*.tmp", "*.temp", "~$*"):
                for temp_file in CFG.output_dir.glob(pattern):
                    if temp_file.is_file():
                        temp_file.unlink(missing_ok=True)
        bar.update()


# =============================================================================
# 6. 主流程
# =============================================================================


def make_figure() -> None:
    """执行完整绘图流程。"""

    setup_style()
    CFG.output_dir.mkdir(parents=True, exist_ok=True)

    boundary, sites, events, pairs, spi_wide = read_inputs()
    sites_summary, extreme_events, valid_pairs, site_response, timeline = prepare_data(
        sites, events, pairs, spi_wide
    )

    if valid_pairs.empty:
        raise ValueError("No valid event-reference CBI pairs were found.")
    if site_response.empty:
        raise ValueError("No site-level CBI response summaries were found.")

    with progress_bar("步骤3/5 绘制四个面板", 8, "yellow") as bar:
        delta_limit = finite_symmetric_limit(
            pd.concat([valid_pairs["Delta_CBI"], site_response["Median_Delta_CBI"]]),
            fallback=0.25,
        )
        delta_norm = TwoSlopeNorm(vmin=-delta_limit, vcenter=0, vmax=delta_limit)
        bar.update()

        fig = plt.figure(figsize=(FP.fig_width, FP.fig_height), constrained_layout=False)
        grid = fig.add_gridspec(
            2,
            2,
            left=FP.grid_left,
            right=FP.grid_right,
            bottom=FP.grid_bottom,
            top=FP.grid_top,
            wspace=FP.grid_wspace,
            hspace=FP.grid_hspace,
        )
        ax_a = fig.add_subplot(grid[0, 0])
        ax_b = fig.add_subplot(grid[0, 1])
        ax_c = fig.add_subplot(grid[1, 0])
        ax_d = fig.add_subplot(grid[1, 1])
        bar.update()

        draw_panel_a(ax_a, boundary, sites_summary, timeline)
        bar.update()

        draw_panel_b(ax_b, valid_pairs, delta_norm)
        bar.update()

        draw_panel_c(ax_c, boundary, sites_summary, site_response, delta_norm)
        bar.update()

        spi_scatter = draw_panel_d(ax_d, valid_pairs, site_response)
        bar.update()

        cax_delta = ax_b.inset_axes(
            [
                FP.delta_colorbar_x,
                FP.delta_colorbar_y,
                FP.delta_colorbar_width,
                FP.delta_colorbar_height,
            ]
        )
        cbar_delta = fig.colorbar(
            mpl.cm.ScalarMappable(norm=delta_norm, cmap=FP.delta_cmap),
            cax=cax_delta,
            orientation="horizontal",
        )
        cbar_delta.set_label("ΔCBI (event - reference)", labelpad=1, fontsize=FP.colorbar_label_size)
        cbar_delta.ax.xaxis.set_label_position("top")
        cbar_delta.ax.tick_params(labelsize=FP.colorbar_tick_size, length=2)

        cax_spi = ax_d.inset_axes(
            [
                FP.spi_colorbar_x,
                FP.spi_colorbar_y,
                FP.spi_colorbar_width,
                FP.spi_colorbar_height,
            ]
        )
        cbar_spi = fig.colorbar(spi_scatter, cax=cax_spi, orientation="horizontal")
        cbar_spi.set_label("Minimum SPI30d during event", labelpad=1, fontsize=FP.colorbar_label_size)
        cbar_spi.ax.xaxis.set_label_position("top")
        cbar_spi.ax.tick_params(labelsize=FP.colorbar_tick_size, length=2)
        bar.update()

        fig.suptitle(
            "Spatial heterogeneity in drought-induced microclimate buffering responses",
            fontsize=FP.suptitle_size,
            fontweight="bold",
            y=FP.suptitle_y,
        )
        bar.update()

    summary = write_outputs(fig, sites_summary, extreme_events, valid_pairs, site_response)
    plt.close(fig)

    cleanup_runtime_cache()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        make_figure()
