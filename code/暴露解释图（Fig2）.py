from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm


# =============================================================================
# 0. 论文主线与后续图件路线图
# =============================================================================
# 这段注释用于固定整篇论文的绘图主线，防止后续绘图或建模时偏离主题。
# 当前脚本实现 Fig. 2“暴露解释图”，但它必须服务于 Fig. 1 提出的核心现象，
# 而不是变成独立的数据展示。
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
#       它回答“发生了什么、在哪里发生、差异有多大”。
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
#       本脚本实现该图。它不解释机制，而是检验干旱暴露本身是否足以解释
#       Fig. 1 中观察到的 Delta_CBI 空间异质性。
#       建议做 3 个面板：
#           Fig. 2A：干旱等级相对 Normal 的 Delta_CBI 剂量响应图。
#               横轴为 Mild/Moderate/Severe/Extreme；
#               纵轴为相对 Normal 的 Delta_CBI；
#               使用多等级 LMM 的估计值、95% CI 和 FDR 结果。
#           Fig. 2B：事件最小 SPI30d 与 Delta_CBI 的散点图。
#               每个点为一个有效“事件 x 站点”样本；
#               越负的 Min_Daily_SPI 表示事件越极端。
#           Fig. 2C：Duration_Days 与 Delta_CBI 的散点图。
#               用来检查干旱持续时间是否足以解释响应分化。
#       需要数据：
#           - 有效事件-站点 CBI 配对表；
#           - 干旱事件暴露变量：Min_Daily_SPI、Severity、Duration_Days；
#           - 多等级 LMM 结果：Delta_CBI、CI、FDR p 值、站点数、站点月份数；
#           - 多等级有序趋势检验结果。
#       预期逻辑：
#           如果控制干旱等级、强度和持续时间后，Delta_CBI 仍存在明显离散，
#           则说明“干旱暴露强度”不足以解释 Fig. 1 的空间异质性。
#
#   Fig. 3  识别过程差异：
#       不同缓冲响应组是否经历了不同的水分、植被、能量或大气干燥过程？
#       这不是正式路径模型，而是“机制观察图”，用于筛选进入 Fig. 4 的候选过程。
#       建议做 2 个面板：
#           Fig. 3A：环境变量变化热图；
#           Fig. 3B：关键变量在响应组之间的比较图。
#       候选变量模块：
#           - 干旱暴露：Min_Daily_SPI, Severity, Duration_Days；
#           - 水分过程：土壤水分及其事件-参考期变化 Delta_SM；
#           - 植被状态：植被指数、蒸散、冠层调节相关变量；
#           - 能量/辐射过程：辐射、能量输入、地表热状况；
#           - 大气背景：气温、湿度、VPD 或空气干燥度；
#           - 地形/空间背景：海拔、坡度、坡向、经纬度、空间位置。
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
#           Fig. 4A：预设并筛选后的路径图；
#           Fig. 4B：总效应、直接效应和间接效应分解图。
#       方法边界：
#           如果样本量、变量缺失率、共线性或随机效应结构不支持路径模型，
#           不应强行做 SEM；可以退回为“标准化效应森林图/分层总效应模型”，
#           仍然回答哪些过程与 Delta_CBI 最相关，但不声称因果路径。
#
#   Fig. 5  证明可信：
#       回答审稿人对 Fig. 1-Fig. 4 的稳定性质疑：
#           换极端干旱阈值是否还成立？
#           换正常期定义是否还成立？
#           删除某个站点或某个区域干旱过程是否还成立？
#       建议做 3 个面板：
#           Fig. 5A：不同事件定义/质量阈值下总体 Delta_CBI 的森林图；
#           Fig. 5B：Leave-one-site-out 逐站点删除森林图；
#           Fig. 5C：Leave-one-regional-process-out 逐区域干旱过程删除森林图。
#
# 可选 Fig. 6：
#   如果论文问题扩展到“干旱结束后的恢复/韧性”，可增加 Fig. 6；
#   否则建议放补充材料，避免正文主线从“干旱期响应机制”扩散到“恢复过程”。
#
# 当前和下一步的执行顺序：
#   1. 先固定并审定 Fig. 1：现象是否表达清楚，英文标题、色带、图例是否适合 SCI。
#   2. 本脚本整理 Fig. 2 所需表：把有效事件-站点配对表与干旱暴露变量合并。
#   3. 做 Fig. 2 的暴露解释检验：等级响应、最小 SPI、持续时间。
#   4. 若 Fig. 2 说明暴露不足以解释空间异质性，再整理 Tensor_Data 动态变量，
#      计算事件期相对参考期的 Delta_X，进入 Fig. 3。
#   5. 从 Fig. 3 选择少量理论明确、缺失可控、非高度重复的变量进入 Fig. 4。
#   6. 最后用 Fig. 5 汇总敏感性、逐站点删除和逐区域过程删除结果，支撑可信度。
#


# =============================================================================
# 1. 路径、阈值和图形参数配置
# =============================================================================


@dataclass(frozen=True)
class Config:
    """Fig. 2 输入输出路径和分组阈值。

    response_near_zero_threshold:
        用于 Fig. 2B/C 将事件-站点样本分为 strengthened/stable/weakened。
        abs(Delta_CBI) <= 0.05 视为近似稳定，避免把极小差异过度解释。
    """

    project_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate")
    output_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\results\论文主图\Fig2_暴露解释图")
    response_near_zero_threshold: float = 0.05
    figure_dpi: int = 600

    @property
    def event_pairs_csv(self) -> Path:
        return (
            self.project_dir
            / "results"
            / "compare_differences_results"
            / "04_极端事件CBI与事件后正常参考期对比表.csv"
        )

    @property
    def drought_events_csv(self) -> Path:
        return (
            self.project_dir
            / "results"
            / "daily_SPI_features"
            / "福建省观测站2025年daily_SPI干旱事件长表.csv"
        )

    @property
    def multilevel_dir(self) -> Path:
        return (
            self.project_dir
            / "results"
            / "compare_differences_results"
            / "多等级干旱扩展分析"
        )

    @property
    def multilevel_lmm_csv(self) -> Path:
        return self.multilevel_dir / "05_多等级_LMM结果汇总.csv"

    @property
    def multilevel_trend_csv(self) -> Path:
        return self.multilevel_dir / "09_多等级_DeltaCBI有序趋势检验.csv"

    @property
    def multilevel_run_audit_csv(self) -> Path:
        return self.multilevel_dir / "04_多等级模型可运行性审计.csv"


@dataclass(frozen=True)
class FigureParameters:
    """Fig. 2 的可调图形参数。

    后续主要会调整点大小、透明度、色彩、趋势线和版式，不建议在绘图阶段改变
    Delta_CBI、Min_Daily_SPI 或 Duration_Days 的统计定义。
    """

    fig_width: float = 12.0
    fig_height: float = 5.3
    grid_left: float = 0.07
    grid_right: float = 0.96
    grid_bottom: float = 0.18
    grid_top: float = 0.72
    grid_wspace: float = 0.30

    axes_linewidth: float = 0.8
    axis_label_size: int = 9
    title_size: int = 10
    tick_label_size: int = 8
    legend_font_size: int = 7
    suptitle_size: int = 12

    panel_a_point_size: float = 52
    panel_a_ci_linewidth: float = 1.4
    panel_a_zero_line_color: str = "#404040"
    panel_a_zero_linewidth: float = 0.8
    panel_a_zero_linestyle: str = "--"
    panel_a_point_color: str = "#b45a3c"
    panel_a_ci_color: str = "#5c2f25"
    panel_a_raw_point_alpha: float = 0.0
    panel_a_label_fontsize: int = 7
    panel_a_label_offset: float = 0.004
    panel_a_ylim_padding: float = 0.018
    panel_a_xlim_padding: float = 0.18
    panel_a_edge_label_inset: float = 0.10
    panel_a_trend_box_x: float = 0.03
    panel_a_trend_box_y: float = 0.94

    scatter_point_size: float = 48
    scatter_point_alpha: float = 0.82
    scatter_edgecolor: str = "white"
    scatter_linewidth: float = 0.45
    trend_line_color: str = "#2f2f2f"
    trend_linewidth: float = 1.2
    trend_band_alpha: float = 0.16
    zero_line_color: str = "#4d4d4d"
    zero_linewidth: float = 0.8
    zero_linestyle: str = "--"

    strengthened_color: str = "#2b83ba"
    stable_color: str = "#bdbdbd"
    weakened_color: str = "#d7191c"
    strengthened_label: str = "Strengthened or maintained"
    stable_label: str = "Near-zero change"
    weakened_label: str = "Weakened buffering"
    legend_anchor_x: float = 0.50
    legend_anchor_y: float = 0.905
    legend_marker_size: float = 7


CFG = Config()
FP = FigureParameters()


# =============================================================================
# 2. 进度条、样式和通用工具
# =============================================================================


PROGRESS_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
    "[elapsed: {elapsed}, remaining: {remaining}, {rate_fmt}]"
)


def progress_bar(desc: str, total: int, colour: str) -> tqdm:
    """创建单行动态彩色进度条，避免终端刷屏。"""

    return tqdm(
        total=total,
        desc=desc,
        colour=colour,
        dynamic_ncols=True,
        leave=False,
        ncols=100,
        bar_format=PROGRESS_BAR_FORMAT,
    )


def setup_style() -> None:
    """设置英文学术图件风格和 PDF 可编辑字体。"""

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


def normalize_site_id(value: object) -> str:
    """统一站点编号格式，避免 95332217 与 95332217.0 合并失败。"""

    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))


def classify_response(delta_cbi: float) -> str:
    """按 Delta_CBI 方向给 Fig. 2B/C 点分类。"""

    if pd.isna(delta_cbi):
        return "Missing"
    if delta_cbi < -CFG.response_near_zero_threshold:
        return "Strengthened"
    if delta_cbi > CFG.response_near_zero_threshold:
        return "Weakened"
    return "Stable"


def response_palette() -> dict[str, str]:
    return {
        "Strengthened": FP.strengthened_color,
        "Stable": FP.stable_color,
        "Weakened": FP.weakened_color,
    }


def response_label_map() -> dict[str, str]:
    return {
        "Strengthened": FP.strengthened_label,
        "Stable": FP.stable_label,
        "Weakened": FP.weakened_label,
    }


# =============================================================================
# 3. 数据读取与整理
# =============================================================================


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取 Fig. 2 所需输入。

    Fig. 2A 使用已有多等级 LMM 结果，不重新建模；
    Fig. 2B/C 使用有效事件-站点 CBI 配对表，并与干旱事件长表核对暴露变量。
    """

    with progress_bar("步骤1/5 读取Fig2输入", 5, "cyan") as bar:
        require_files(
            [
                CFG.event_pairs_csv,
                CFG.drought_events_csv,
                CFG.multilevel_lmm_csv,
                CFG.multilevel_trend_csv,
                CFG.multilevel_run_audit_csv,
            ]
        )
        bar.update()

        pairs = pd.read_csv(CFG.event_pairs_csv)
        bar.update()

        events = pd.read_csv(CFG.drought_events_csv)
        bar.update()

        lmm = pd.read_csv(CFG.multilevel_lmm_csv)
        trend = pd.read_csv(CFG.multilevel_trend_csv)
        bar.update()

        audit = pd.read_csv(CFG.multilevel_run_audit_csv)
        bar.update()

    return pairs, events, lmm, trend, audit


def prepare_event_exposure_table(pairs: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """整理 Fig. 2B/C 的事件暴露基础表。

    主表采用 Pair_flag == "ok" 的事件-站点 CBI 配对。事件长表用于补充
    Drought_Level、Drought_Level_Code 和站点经纬度等暴露背景字段。
    """

    with progress_bar("步骤2/5 整理暴露基础表", 4, "green") as bar:
        for df in (pairs, events):
            df["Site_ID"] = df["Site_ID"].map(normalize_site_id)
            df["Event_ID"] = pd.to_numeric(df["Event_ID"], errors="coerce").astype("Int64")
        bar.update()

        pairs = pairs.rename(columns={"Delta_CBI_Event_minus_Reference": "Delta_CBI"}).copy()
        numeric_cols = [
            "Duration_Days",
            "Severity",
            "Min_Daily_SPI",
            "Event_CBI",
            "Reference_CBI",
            "Delta_CBI",
        ]
        for col in numeric_cols:
            if col in pairs.columns:
                pairs[col] = pd.to_numeric(pairs[col], errors="coerce")
        bar.update()

        event_cols = [
            "Site_ID",
            "Event_ID",
            "Drought_Level",
            "Drought_Level_Code",
            "Station_Lon",
            "Station_Lat",
            "Edge_Truncated",
        ]
        event_cols = [col for col in event_cols if col in events.columns]
        base = pairs.loc[pairs["Pair_flag"].eq("ok")].copy()
        base = base.merge(events[event_cols], on=["Site_ID", "Event_ID"], how="left")
        bar.update()

        base["Response_Group"] = base["Delta_CBI"].map(classify_response)
        base = base.loc[
            base["Delta_CBI"].notna()
            & base["Min_Daily_SPI"].notna()
            & base["Duration_Days"].notna()
        ].copy()
        bar.update()

    return base


def prepare_lmm_table(lmm: pd.DataFrame) -> pd.DataFrame:
    """整理 Fig. 2A 的多等级 LMM 结果表。"""

    class_order = ["Mild", "Moderate", "Severe", "Extreme"]
    d = lmm.loc[lmm["AnalysisVersion"].eq("Main_hours_only")].copy()
    d["Target_Class"] = pd.Categorical(d["Target_Class"], categories=class_order, ordered=True)
    d = d.sort_values("Target_Class").reset_index(drop=True)
    for col in [
        "DeltaCBI_Target_minus_Normal",
        "DeltaCBI_CI_low_95",
        "DeltaCBI_CI_high_95",
        "DeltaCBI_p_FDR",
        "n_sites",
        "n_site_months",
    ]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    return d


# =============================================================================
# 4. 统计摘要
# =============================================================================


def linear_trend_summary(df: pd.DataFrame, x_col: str, y_col: str, label: str) -> dict[str, object]:
    """给 Fig. 2B/C 输出描述性线性趋势摘要。

    这里不把事件-站点样本当作完全独立的正式显著性检验；斜率和 p 值仅作为
    图件审计与描述性辅助。正式解释以“暴露变量不能充分解释离散响应”为主。
    """

    d = df[[x_col, y_col]].dropna().copy()
    if len(d) < 3:
        return {
            "Panel": label,
            "x": x_col,
            "y": y_col,
            "n": len(d),
            "slope": np.nan,
            "intercept": np.nan,
            "r": np.nan,
            "p": np.nan,
            "stderr": np.nan,
            "note": "样本数不足，未计算线性趋势。",
        }
    res = stats.linregress(d[x_col], d[y_col])
    return {
        "Panel": label,
        "x": x_col,
        "y": y_col,
        "n": len(d),
        "slope": res.slope,
        "intercept": res.intercept,
        "r": res.rvalue,
        "p": res.pvalue,
        "stderr": res.stderr,
        "note": "描述性线性趋势；事件-站点样本存在站点/事件非独立性，不作为唯一推断依据。",
    }


def build_trend_table(event_exposure: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            linear_trend_summary(event_exposure, "Min_Daily_SPI", "Delta_CBI", "Fig.2B"),
            linear_trend_summary(event_exposure, "Duration_Days", "Delta_CBI", "Fig.2C"),
        ]
    )


# =============================================================================
# 5. 绘图函数
# =============================================================================


def draw_panel_a(ax: plt.Axes, lmm: pd.DataFrame, trend: pd.DataFrame) -> None:
    """Fig. 2A：多等级干旱相对 Normal 的 CBI 响应。"""

    x = np.arange(len(lmm))
    y = lmm["DeltaCBI_Target_minus_Normal"].to_numpy(dtype=float)
    yerr = np.vstack(
        [
            y - lmm["DeltaCBI_CI_low_95"].to_numpy(dtype=float),
            lmm["DeltaCBI_CI_high_95"].to_numpy(dtype=float) - y,
        ]
    )
    ax.axhline(
        0,
        color=FP.panel_a_zero_line_color,
        lw=FP.panel_a_zero_linewidth,
        linestyle=FP.panel_a_zero_linestyle,
        zorder=1,
    )
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        color=FP.panel_a_ci_color,
        ecolor=FP.panel_a_ci_color,
        elinewidth=FP.panel_a_ci_linewidth,
        capsize=3,
        markersize=np.sqrt(FP.panel_a_point_size),
        markerfacecolor=FP.panel_a_point_color,
        markeredgecolor="#222222",
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(lmm["Target_Class"].astype(str), rotation=0)
    ax.set_ylabel("Estimated Delta CBI relative to Normal")
    ax.set_xlabel("Drought class")
    ax.set_title("A  Drought class response was not monotonic", loc="left", fontweight="bold")
    ax.set_xlim(-0.5 - FP.panel_a_xlim_padding, len(lmm) - 0.5 + FP.panel_a_xlim_padding)
    ax.set_ylim(
        min(-0.003, float(lmm["DeltaCBI_CI_low_95"].min()) - 0.003),
        float(lmm["DeltaCBI_CI_high_95"].max()) + FP.panel_a_ylim_padding,
    )

    for i, (xi, row) in enumerate(zip(x, lmm.itertuples(index=False))):
        label_x = xi
        ha = "center"
        if i == 0:
            label_x = xi + FP.panel_a_edge_label_inset
            ha = "left"
        elif i == len(lmm) - 1:
            label_x = xi - FP.panel_a_edge_label_inset
            ha = "right"
        ax.text(
            label_x,
            row.DeltaCBI_CI_high_95 + FP.panel_a_label_offset,
            f"sites={int(row.n_sites)}\nSM={int(row.n_site_months)}",
            ha=ha,
            va="bottom",
            fontsize=FP.panel_a_label_fontsize,
        )

    trend_main = trend.loc[
        trend["AnalysisVersion"].eq("Main_hours_only")
        & trend["Trend_Method"].eq("Weighted_linear_trend")
    ]
    if not trend_main.empty:
        row = trend_main.iloc[0]
        ax.text(
            FP.panel_a_trend_box_x,
            FP.panel_a_trend_box_y,
            f"Ordinal trend p = {row['Slope_p']:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d0d0d0", "lw": 0.6},
        )


def add_linear_fit(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str) -> None:
    """添加描述性线性趋势线和简单置信带。"""

    d = df[[x_col, y_col]].dropna().sort_values(x_col)
    if len(d) < 3:
        return
    res = stats.linregress(d[x_col], d[y_col])
    x = np.linspace(d[x_col].min(), d[x_col].max(), 100)
    y = res.intercept + res.slope * x
    ax.plot(x, y, color=FP.trend_line_color, lw=FP.trend_linewidth, zorder=2)

    residual = d[y_col] - (res.intercept + res.slope * d[x_col])
    band = 1.96 * residual.std(ddof=2) / np.sqrt(len(d))
    ax.fill_between(
        x,
        y - band,
        y + band,
        color=FP.trend_line_color,
        alpha=FP.trend_band_alpha,
        linewidth=0,
        zorder=1,
    )


def draw_exposure_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    xlabel: str,
    title: str,
) -> None:
    """Fig. 2B/C 通用散点图。"""

    palette = response_palette()
    labels = response_label_map()
    for group in ["Strengthened", "Stable", "Weakened"]:
        d = df.loc[df["Response_Group"].eq(group)]
        ax.scatter(
            d[x_col],
            d["Delta_CBI"],
            s=FP.scatter_point_size,
            c=palette[group],
            label=labels[group],
            alpha=FP.scatter_point_alpha,
            edgecolors=FP.scatter_edgecolor,
            linewidths=FP.scatter_linewidth,
            zorder=3,
        )
    ax.axhline(0, color=FP.zero_line_color, lw=FP.zero_linewidth, linestyle=FP.zero_linestyle, zorder=1)
    add_linear_fit(ax, df, x_col, "Delta_CBI")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Delta CBI (event - reference)")
    ax.set_title(title, loc="left", fontweight="bold")


# =============================================================================
# 6. 输出与缓存清理
# =============================================================================


def parameters_to_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    descriptions = {
        "response_near_zero_threshold": "响应方向分组阈值；abs(Delta_CBI) <= 该值视为近似稳定。",
        "panel_a_point_size": "Fig.2A 模型估计点大小。",
        "panel_a_ci_linewidth": "Fig.2A 95% CI 误差线宽度。",
        "panel_a_label_offset": "Fig.2A n_sites 和 SiteMonth 标注距离 CI 上界的偏移量。",
        "panel_a_ylim_padding": "Fig.2A y 轴上限额外留白，避免标注贴近标题。",
        "panel_a_xlim_padding": "Fig.2A x 轴左右额外留白，避免首末标签压住图框。",
        "panel_a_edge_label_inset": "Fig.2A 首末样本量标签向图内移动的距离。",
        "panel_a_trend_box_x": "Fig.2A 有序趋势检验 p 值标签的横向相对位置。",
        "panel_a_trend_box_y": "Fig.2A 有序趋势检验 p 值标签的纵向相对位置。",
        "scatter_point_size": "Fig.2B/C 事件-站点散点大小。",
        "scatter_point_alpha": "Fig.2B/C 散点透明度。",
        "trend_linewidth": "Fig.2B/C 描述性趋势线宽度。",
        "trend_band_alpha": "Fig.2B/C 描述性趋势带透明度。",
    }
    for name, value in asdict(CFG).items():
        rows.append(
            {
                "参数组": "路径与阈值",
                "参数名": name,
                "当前值": str(value),
                "用途说明": descriptions.get(name, "输入输出路径或分析阈值。"),
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
    lmm: pd.DataFrame,
    trend: pd.DataFrame,
    run_audit: pd.DataFrame,
    event_exposure: pd.DataFrame,
    trend_summary: pd.DataFrame,
) -> pd.DataFrame:
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    with progress_bar("步骤4/5 保存Fig2输出", 8, "magenta") as bar:
        fig.savefig(CFG.output_dir / "图2_暴露解释图.png", dpi=CFG.figure_dpi)
        bar.update()

        fig.savefig(CFG.output_dir / "图2_暴露解释图.pdf")
        bar.update()

        event_exposure.to_csv(CFG.output_dir / "图2_干旱暴露与DeltaCBI分析基础表.csv", index=False, encoding="utf-8-sig")
        bar.update()

        lmm.to_csv(CFG.output_dir / "图2A_多等级LMM结果整理表.csv", index=False, encoding="utf-8-sig")
        bar.update()

        trend.to_csv(CFG.output_dir / "图2A_多等级有序趋势检验表.csv", index=False, encoding="utf-8-sig")
        bar.update()

        run_audit.to_csv(CFG.output_dir / "图2A_多等级模型可运行性审计表.csv", index=False, encoding="utf-8-sig")
        bar.update()

        trend_summary.to_csv(CFG.output_dir / "图2B和图2C_暴露趋势描述性统计表.csv", index=False, encoding="utf-8-sig")
        bar.update()

        parameters_to_table().to_csv(CFG.output_dir / "00_Fig2绘图参数说明表.csv", index=False, encoding="utf-8-sig")
        bar.update()

    summary = pd.DataFrame(
        [
            {
                "有效事件站点配对数": int(len(event_exposure)),
                "有效站点数": int(event_exposure["Site_ID"].nunique()),
                "Delta_CBI均值": float(event_exposure["Delta_CBI"].mean()),
                "Delta_CBI中位数": float(event_exposure["Delta_CBI"].median()),
                "缓冲增强或维持样本数": int(event_exposure["Response_Group"].eq("Strengthened").sum()),
                "近似稳定样本数": int(event_exposure["Response_Group"].eq("Stable").sum()),
                "缓冲减弱样本数": int(event_exposure["Response_Group"].eq("Weakened").sum()),
                "PNG图件": str(CFG.output_dir / "图2_暴露解释图.png"),
                "PDF图件": str(CFG.output_dir / "图2_暴露解释图.pdf"),
            }
        ]
    )
    summary.to_csv(CFG.output_dir / "图2_暴露解释图运行摘要表.csv", index=False, encoding="utf-8-sig")
    return summary


def cleanup_runtime_cache() -> None:
    """清理本脚本运行可能产生的缓存。"""

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
# 7. 主流程
# =============================================================================


def make_figure() -> None:
    setup_style()
    CFG.output_dir.mkdir(parents=True, exist_ok=True)

    pairs, events, lmm_raw, trend, run_audit = read_inputs()
    event_exposure = prepare_event_exposure_table(pairs, events)
    lmm = prepare_lmm_table(lmm_raw)
    trend_summary = build_trend_table(event_exposure)

    if event_exposure.empty:
        raise ValueError("Fig. 2B/C 没有可用的有效事件-站点配对样本。")
    if lmm.empty:
        raise ValueError("Fig. 2A 没有可用的 Main_hours_only 多等级 LMM 结果。")

    with progress_bar("步骤3/5 绘制Fig2三面板", 5, "yellow") as bar:
        fig = plt.figure(figsize=(FP.fig_width, FP.fig_height), constrained_layout=False)
        grid = fig.add_gridspec(
            1,
            3,
            left=FP.grid_left,
            right=FP.grid_right,
            bottom=FP.grid_bottom,
            top=FP.grid_top,
            wspace=FP.grid_wspace,
        )
        ax_a = fig.add_subplot(grid[0, 0])
        ax_b = fig.add_subplot(grid[0, 1])
        ax_c = fig.add_subplot(grid[0, 2])
        bar.update()

        draw_panel_a(ax_a, lmm, trend)
        bar.update()

        draw_exposure_scatter(
            ax_b,
            event_exposure,
            "Min_Daily_SPI",
            "Minimum SPI30d during event",
            "B  Drought intensity did not fully explain Delta CBI",
        )
        bar.update()

        draw_exposure_scatter(
            ax_c,
            event_exposure,
            "Duration_Days",
            "Drought duration (days)",
            "C  Duration did not fully explain Delta CBI",
        )
        bar.update()

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=response_palette()[group],
                markeredgecolor="white",
                markersize=FP.legend_marker_size,
                label=response_label_map()[group],
            )
            for group in ["Strengthened", "Stable", "Weakened"]
        ]
        fig.legend(
            handles=handles,
            frameon=False,
            loc="center",
            bbox_to_anchor=(FP.legend_anchor_x, FP.legend_anchor_y),
            borderaxespad=0,
            ncol=3,
        )
        fig.suptitle(
            "Drought exposure alone did not explain heterogeneous buffering responses",
            fontsize=FP.suptitle_size,
            fontweight="bold",
            y=0.98,
        )
        bar.update()

    summary = write_outputs(fig, lmm, trend, run_audit, event_exposure, trend_summary)
    plt.close(fig)
    cleanup_runtime_cache()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        make_figure()
