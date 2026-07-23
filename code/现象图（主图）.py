from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import warnings

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import linregress
from tqdm import tqdm


# =============================================================================
# 0. 论文主线与后续图件路线图
# =============================================================================
# 这段注释用于固定整篇论文的绘图主线，防止后续绘图或建模时偏离主题。
# 当前脚本只实现 Fig. 1“现象图/主图”，但后续 Fig. 2-Fig. 5 都应服务于
# Fig. 1 提出的核心现象，而不是变成相互独立的数据展示。
#
# 全文核心问题：
#   不同等级干旱是否改变森林微气候缓冲能力（有没有），这种改变的幅度有多大
#   （整体等级差异和站点差异），这种幅度是否呈单调剂量效应，还是存在更复杂的
#   非单调模式？如果存在空间异质性，它是由干旱暴露强度、局地环境过程、空间背景
#   共同塑造，还是由少数站点、样本支持域或某个区域干旱过程造成？
#
# 核心响应变量：
#   CBI = 15 cm 微气候温度对 ERA5 2 m 温度的响应斜率。
#   Delta_CBI = Target_Class_CBI - Normal_CBI，其中 Target_Class 包括
#   Mild / Moderate / Severe / Extreme 四级干旱。
#   Delta_CBI > 0 表示干旱等级下 CBI 高于 Normal，即缓冲减弱；
#   Delta_CBI < 0 表示干旱等级下 CBI 低于 Normal，即表观缓冲增强或维持。
#
# Fig.1 在全文中的定位：
#   这是第一张结果图，只回答“有没有、差异多大、差异模式是什么、结果是否可信、
#   物理意义是否重要”。它不解释机制，不做归因；机制解释留给 Fig.2-Fig.4，
#   稳健性证据集中放入 Fig.5。
#
# Fig.1 四面板新版设计：
#   A. Multi-level Delta CBI forest plot
#       回答问题：有没有区域总体差异；差异是否呈单调等级梯度。
#       图中信息：横轴为 Delta_CBI，纵轴为 Mild/Moderate/Severe/Extreme；
#       点为多等级 LMM 估计，误差线为 95% CI；x=0 表示与 Normal 无差异；
#       主模型点上方标注 n_sites 和 FDR q 值；同时叠加较弱的 restricted common-support
#       方块点作为样本支持域稳健性对照，但不显示其误差线，避免主图过于拥挤。
#       体现维度：有没有差异、差异是什么模式、结果是否可信。
#       数据来源：05_多等级_LMM结果汇总.csv、
#       16_多等级_限制样本LMM结果汇总.csv、09_多等级_DeltaCBI有序趋势检验.csv。
#
#   B. Absolute CBI distributions across classes
#       回答问题：区域总体差异量级有多大；CBI 绝对值在物理上意味着什么。
#       图中信息：Normal/Mild/Moderate/Severe/Extreme 五组箱线图和散点；
#       横轴标签下直接标注每组 n；水平参考线 CBI=1 表示“林内温度完全跟随
#       宏观温度，即无缓冲”的理论参照。
#       体现维度：差异多大、物理意义是否重要。
#       数据来源：00_逐小时温度_SPI合并审计表.csv 重算 SiteMonth CBI，
#       再用 03_各等级_SiteMonth_配对审计.csv 的 Pass_Hours_Plus_Macro_SD 过滤。
#
#   C. Spatial structure of site-level Delta CBI
#       回答问题：差异是否具有空间结构，还是随机分散在站点之间。
#       图中信息：福建边界和站点位置；点颜色为站点中位 Delta_CBI，红色表示
#       缓冲减弱、蓝色表示缓冲增强或维持；点大小统一，避免样本量视觉编码干扰
#       空间结构判断。n_pairs 不在主图中编码，写入输出表和图注。
#       体现维度：差异多大（站点间）、是否存在空间结构。
#       数据来源：Tensor_LatLong.csv、福建省行政边界 shp、五级有效 SiteMonth 配对表。
#
#   D. Site ranking across all drought classes
#       回答问题：哪些站点响应最强/最弱；站点排序是否只由 Extreme 小样本驱动。
#       图中信息：横轴为站点中位 Delta_CBI；纵轴为按中位 Delta_CBI 排序后的站点排名；
#       棒棒糖线段从 0 延伸到站点中位数，端点颜色仍按 Delta_CBI 发散色带编码；
#       只轻标注响应最强和最弱的少数站点，完整站点 ID 与 n_pairs 写入输出表。
#       体现维度：差异多大（站点排序）、哪些站点是主要异常值。
#       数据来源：03_各等级_SiteMonth_配对审计.csv 与重算 SiteMonth CBI 合并后的长表。
#
# 五个信息维度在 Fig.1 中的落点：
#   1. 有没有差异：面板 A 的 Delta_CBI 点估计、95% CI 和 FDR q 值。
#   2. 差异多大：面板 A 的模型效应量，面板 B 的绝对 CBI 分布，面板 C/D 的站点差异。
#   3. 差异是什么模式：面板 A 的等级排序和有序趋势检验。
#   4. 结果是否可信：面板 A 的 n_sites/q 值和限制公共支持集模型；
#      面板 C/D 对应的 n_pairs 保存在输出表，图注中透明报告。
#   5. 物理意义是否重要：面板 B 保留绝对 CBI，并加入 CBI=1 无缓冲参考线。
#
# Fig.1 绝对 CBI 对比版（候选对照输出）：
#   输出文件名：图1_现象图_主图_绝对CBI对比版.png/pdf。
#   设计目的：
#       这套图不把 Delta_CBI 作为主视觉，而直接展示相减前的 Normal_CBI 与
#       drought-class CBI。它回答的是“干旱期和 Normal 期的 CBI 本身分别处在
#       什么绝对位置，肉眼是否能看到成对偏移”，更直观，但不如 Delta_CBI
#       森林图那样直接表达模型效应量。因此该版本用于和 Delta_CBI 优化版比较，
#       不自动取代主推断图。
#   A. Paired absolute CBI contrast by drought class
#       每个干旱等级一组，每组包含对应配对的 Normal reference CBI 和 Target
#       drought-class CBI。箱线图/散点/淡配对线共同显示“相减前两组值”是否分开；
#       上方标注 n_site 和 FDR q，帮助保留主模型推断信息。
#   B. Absolute CBI distributions across all states
#       汇总 Normal/Mild/Moderate/Severe/Extreme 五类状态的绝对 CBI 分布，并保留
#       CBI=1 无缓冲参考线。它回答“不同状态的 CBI 物理量级在哪里”。
#   C. Spatial map of drought-period absolute CBI
#       地图点颜色表示各站点干旱期 Target_CBI 的中位数，而不是 Delta_CBI。
#       它回答“干旱期间哪些站点的绝对缓冲状态更弱/更接近无缓冲”。
#   D. Site-level paired absolute CBI ranking
#       每个站点一行，显示站点 Normal median CBI 与 drought-period median CBI 两个点，
#       并用水平线连接。排序按 drought-period median CBI，从而突出干旱期绝对缓冲状态
#       的站点差异；该面板不直接显示 Delta_CBI 数值，但读者可通过两个点的间距判断迁移。
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
    severe_spi_threshold: float = -1.5
    moderate_spi_threshold: float = -1.0
    normal_spi_low: float = -0.5
    normal_spi_high: float = 0.5
    min_site_month_status_cbi_hours: int = 72
    min_site_month_status_macro_sd: float = 1.0
    figure_dpi: int = 600

    @property
    def sites_csv(self) -> Path:
        return self.project_dir / "Tensor_LatLong.csv"

    @property
    def hourly_spi_merged_csv(self) -> Path:
        return (
            self.project_dir
            / "results"
            / "compare_differences_results"
            / "00_逐小时温度_SPI合并审计表.csv"
        )

    @property
    def multi_pair_audit_csv(self) -> Path:
        return (
            self.project_dir
            / "results"
            / "compare_differences_results"
            / "多等级干旱扩展分析"
            / "03_各等级_SiteMonth_配对审计.csv"
        )

    @property
    def multi_lmm_summary_csv(self) -> Path:
        return (
            self.project_dir
            / "results"
            / "compare_differences_results"
            / "多等级干旱扩展分析"
            / "05_多等级_LMM结果汇总.csv"
        )

    @property
    def multi_lmm_restricted_csv(self) -> Path:
        return (
            self.project_dir
            / "results"
            / "compare_differences_results"
            / "多等级干旱扩展分析"
            / "16_多等级_限制样本LMM结果汇总.csv"
        )

    @property
    def multi_trend_test_csv(self) -> Path:
        return (
            self.project_dir
            / "results"
            / "compare_differences_results"
            / "多等级干旱扩展分析"
            / "09_多等级_DeltaCBI有序趋势检验.csv"
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

    # Fig. 1A：多等级 Delta_CBI 森林图参数。
    panel_a_marker_size: float = 72
    panel_a_restricted_marker_size: float = 28
    panel_a_ci_linewidth: float = 1.4
    panel_a_restricted_alpha: float = 0.35
    panel_a_zero_line_color: str = "#333333"
    panel_a_text_x: float = 0.985
    panel_a_text_size: int = 7
    panel_a_xlim_left_padding_fraction: float = 0.14
    panel_a_xlim_right_padding_fraction: float = 0.14

    panel_a_legend_loc: str = "center right"
    panel_a_legend_anchor_x: float = 0.99
    panel_a_legend_anchor_y: float = 0.53
    panel_label_x: float = 0.00
    panel_label_y: float = 1.02
    panel_label_size: int = 11

    # Fig. 1B/C：Delta_CBI 发散色带。以 0 为中心，蓝色表示事件期 CBI 低于参考期，
    # 红色表示事件期 CBI 高于参考期。
    delta_cmap: str = "RdBu_r"
    delta_limit_padding: float = 1.15
    site_delta_limit_padding: float = 1.30
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
    panel_b_reference_line_value: float = 1.0
    panel_b_reference_line_color: str = "#333333"
    panel_b_jitter_half_width: float = 0.10
    panel_b_jitter_seed: int = 20260722
    panel_b_median_label_size: int = 7
    panel_b_median_label_color: str = "#333333"
    panel_b_median_label_offset_fraction: float = 0.027
    drought_level_palette: tuple[str, ...] = (
        "#5b8fd9",
        "#f2b94b",
        "#e67f3e",
        "#c74332",
        "#7f1d1d",
    )

    # Fig. 1C：站点中位数 Delta_CBI 地图参数。响应站点点大小固定，只用颜色表达
    # Median_Delta_CBI；N_pairs 保存在输出表和图注，不在地图主视觉中编码。
    panel_c_background_site_size: float = 15
    panel_c_background_site_color: str = "#c7c7c7"
    panel_c_response_size: float = 92
    panel_c_response_edgecolor: str = "#1f1f1f"
    panel_c_response_linewidth: float = 0.45
    panel_c_response_alpha: float = 0.93

    # Fig. 1D：站点排序棒棒糖图。线段从 0 指向站点中位 Delta_CBI，端点颜色按 Delta_CBI 编码。
    panel_d_site_median_size: float = 45
    panel_d_site_median_edgecolor: str = "#111111"
    panel_d_site_median_linewidth: float = 0.9
    panel_d_lollipop_linewidth: float = 1.35
    panel_d_lollipop_alpha: float = 0.90
    panel_d_iqr_linewidth: float = 1.2
    panel_d_iqr_alpha: float = 0.70
    panel_d_iqr_axis_padding: float = 1.08
    panel_d_extreme_label_count: int = 0
    panel_d_zero_line_color: str = "#333333"
    panel_d_zero_linewidth: float = 0.85
    panel_d_zero_linestyle: str = "--"
    panel_d_row_line_color: str = "#e5e5e5"
    panel_d_row_linewidth: float = 0.45
    panel_d_site_label_size: int = 5

    # 色标位置。坐标为相对所在面板的 [x, y, width, height]。
    panel_c_delta_colorbar_x: float = 0.58
    panel_c_delta_colorbar_y: float = 0.075
    panel_c_delta_colorbar_width: float = 0.35
    panel_c_delta_colorbar_height: float = 0.035
    colorbar_tick_size: int = 7
    colorbar_label_size: int = 7

    # Fig.1 绝对 CBI 对比版参数：用于新增候选图，不影响 Delta_CBI 原版和优化版。
    absolute_normal_color: str = "#6d9ed6"
    absolute_drought_color: str = "#d95f3d"
    absolute_cmap: str = "YlOrRd"
    absolute_cbi_limit_padding: float = 0.08
    absolute_panel_a_pair_line_color: str = "#b8b8b8"
    absolute_panel_a_pair_line_alpha: float = 0.22
    absolute_panel_a_pair_linewidth: float = 0.45
    absolute_panel_a_box_width: float = 0.26
    absolute_panel_a_point_size: float = 14
    absolute_panel_a_point_alpha: float = 0.42
    absolute_panel_a_jitter_half_width: float = 0.045
    absolute_panel_a_jitter_seed: int = 20260723
    absolute_panel_c_response_size: float = 92
    absolute_panel_c_colorbar_x: float = 0.58
    absolute_panel_c_colorbar_y: float = 0.075
    absolute_panel_c_colorbar_width: float = 0.35
    absolute_panel_c_colorbar_height: float = 0.035
    absolute_panel_d_line_color: str = "#9a9a9a"
    absolute_panel_d_line_alpha: float = 0.55
    absolute_panel_d_linewidth: float = 0.9
    absolute_panel_d_point_size: float = 42


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


def finite_symmetric_limit_with_padding(
    values: pd.Series,
    fallback: float = 0.2,
    padding: float = 1.15,
) -> float:
    """按指定 padding 计算 0 中心对称范围，用于不同面板独立控制视觉尺度。"""

    arr = pd.to_numeric(values, errors="coerce").to_numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return fallback
    lim = float(np.nanmax(np.abs(arr)))
    if not np.isfinite(lim) or lim == 0:
        return fallback
    return lim * padding


def finite_value_limits(
    values: pd.Series,
    fallback: tuple[float, float] = (0.3, 1.3),
    padding: float = 0.08,
) -> tuple[float, float]:
    """计算绝对 CBI 连续色带/坐标范围，padding 为数据跨度比例。"""

    arr = pd.to_numeric(values, errors="coerce").to_numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return fallback
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    span = vmax - vmin
    if not np.isfinite(span) or span <= 0:
        return fallback
    return vmin - span * padding, vmax + span * padding


def build_absolute_site_response(valid_pairs: pd.DataFrame, sites_summary: pd.DataFrame) -> pd.DataFrame:
    """按站点汇总站点 Normal 与 drought-period 绝对 CBI 中位数。"""

    normal_site = (
        valid_pairs[["Site_ID", "YearMonth", "Normal_CBI"]]
        .drop_duplicates(["Site_ID", "YearMonth"])
        .groupby("Site_ID", as_index=False)
        .agg(Median_Normal_CBI=("Normal_CBI", "median"), N_normal_site_months=("Normal_CBI", "size"))
    )
    drought_site = (
        valid_pairs.groupby("Site_ID", as_index=False)
        .agg(
            Median_Drought_CBI=("Target_CBI", "median"),
            Q25_Drought_CBI=("Target_CBI", lambda x: x.quantile(0.25)),
            Q75_Drought_CBI=("Target_CBI", lambda x: x.quantile(0.75)),
            N_drought_pairs=("Target_CBI", "size"),
            N_drought_classes=("Target_Class", "nunique"),
        )
    )
    return (
        drought_site.merge(normal_site, on="Site_ID", how="left")
        .merge(sites_summary[["Site_ID", "Longitude", "Latitude"]], on="Site_ID", how="left")
        .sort_values("Median_Drought_CBI")
        .reset_index(drop=True)
    )


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


def linear_size_from_count(
    count: pd.Series | np.ndarray,
    min_size: float,
    max_size: float,
    count_min: float | None = None,
    count_max: float | None = None,
) -> np.ndarray:
    """把较窄范围的样本量线性映射为散点面积。

    Fig.1C 的站点有效配对数实际只有约 7-17，如果继续用平方根缩放，
    不同站点和图例圆点大小会很接近。线性缩放用于增强样本量透明度。
    """

    values = np.asarray(count, dtype=float)
    if values.size == 0:
        return values
    min_count = np.nanmin(values) if count_min is None else float(count_min)
    max_count = np.nanmax(values) if count_max is None else float(count_max)
    if not np.isfinite(min_count) or not np.isfinite(max_count) or max_count == min_count:
        return np.full_like(values, (min_size + max_size) / 2, dtype=float)
    return min_size + (values - min_count) / (max_count - min_count) * (max_size - min_size)


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


def format_compact_p(value: float) -> str:
    """统一 q / p 值格式，返回可直接接在 q/p 后面的规范文本。"""

    value = float(value)
    if not np.isfinite(value):
        return "= NA"
    if value < 0.001:
        return " < 0.001"
    return f" = {value:.3f}"


def set_panel_title(ax: plt.Axes, label: str, title: str) -> None:
    """把面板字母放左上角、子标题居中，统一四个子图的标题结构。"""

    ax.text(
        FP.panel_label_x,
        FP.panel_label_y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FP.panel_label_size,
        fontweight="bold",
    )
    ax.set_title(title, loc="center", fontweight="bold")


def classify_drought_class(spi30d: pd.Series) -> pd.Series:
    """按多等级扩展分析的 SPI30d 阈值重建五级干旱状态。

    分级边界必须与“多等级干旱扩展分析”一致，否则 Fig.1B-D 的原始分布
    会和 Fig.1A 的 LMM 结果失去可比性：
        - Normal:   normal_spi_low < SPI30d < normal_spi_high
        - Mild:     moderate_spi_threshold < SPI30d <= normal_spi_low
        - Moderate: severe_spi_threshold < SPI30d <= moderate_spi_threshold
        - Severe:   extreme_spi_threshold < SPI30d <= severe_spi_threshold
        - Extreme:  SPI30d <= extreme_spi_threshold

    其他偏湿或轻微异常状态统一记为 Other，不进入 Target-vs-Normal CBI 配对。
    """

    values = pd.to_numeric(spi30d, errors="coerce")
    conditions = [
        values.gt(CFG.normal_spi_low) & values.lt(CFG.normal_spi_high),
        values.gt(CFG.moderate_spi_threshold) & values.le(CFG.normal_spi_low),
        values.gt(CFG.severe_spi_threshold) & values.le(CFG.moderate_spi_threshold),
        values.gt(CFG.extreme_spi_threshold) & values.le(CFG.severe_spi_threshold),
        values.le(CFG.extreme_spi_threshold),
    ]
    choices = ["Normal", "Mild", "Moderate", "Severe", "Extreme"]
    return pd.Series(np.select(conditions, choices, default="Other"), index=spi30d.index)


def estimate_site_month_status_cbi(group: pd.DataFrame) -> pd.Series:
    """对一个 Site_ID × YearMonth × Drought_Class 子集计算 OLS CBI。

    CBI 定义为 Observed_T15cm_C 对 ERA5_T2m_C 的 OLS 响应斜率：
        Observed_T15cm_C = Intercept + CBI × ERA5_T2m_C + error

    这里是 Fig.1B-D 的描述性 SiteMonth CBI，目的不是替代多等级 LMM，
    而是为“原始分布、空间异质性、站点排序”提供真实五级底层样本。
    为避免由过少小时数或宏观温度变化太小导致斜率不稳定，函数同时输出
    n_hours、Macro_SD 和 CBI_flag，后续再叠加配对审计表的最终 QC 标记。
    """

    data = group[["ERA5_T2m_C", "Observed_T15cm_C"]].dropna()
    n_hours = int(len(data))
    macro_sd = float(data["ERA5_T2m_C"].std(ddof=1)) if n_hours > 1 else np.nan
    if n_hours < CFG.min_site_month_status_cbi_hours:
        return pd.Series(
            {
                "CBI": np.nan,
                "Intercept": np.nan,
                "R2": np.nan,
                "p_slope": np.nan,
                "n_hours": n_hours,
                "Macro_SD": macro_sd,
                "CBI_flag": f"insufficient_hours_lt_{CFG.min_site_month_status_cbi_hours}",
            }
        )
    if not np.isfinite(macro_sd) or macro_sd < CFG.min_site_month_status_macro_sd:
        return pd.Series(
            {
                "CBI": np.nan,
                "Intercept": np.nan,
                "R2": np.nan,
                "p_slope": np.nan,
                "n_hours": n_hours,
                "Macro_SD": macro_sd,
                "CBI_flag": f"macro_sd_lt_{CFG.min_site_month_status_macro_sd:g}",
            }
        )

    fit = linregress(data["ERA5_T2m_C"], data["Observed_T15cm_C"])
    return pd.Series(
        {
            "CBI": float(fit.slope),
            "Intercept": float(fit.intercept),
            "R2": float(fit.rvalue**2),
            "p_slope": float(fit.pvalue),
            "n_hours": n_hours,
            "Macro_SD": macro_sd,
            "CBI_flag": "ok",
        }
    )


# =============================================================================
# 3. 数据读取与汇总
# =============================================================================


def read_inputs() -> tuple[
    gpd.GeoDataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """读取边界、站点、多等级模型结果和五级 CBI 重算底表。

    关键处理：
        - 福建边界统一到 EPSG:4326，保证和站点经纬度一致；
        - Site_ID 全部转为字符串，避免跨表合并失败；
        - 日期列和数值列显式转换，减少 Excel/CSV 自动类型推断造成的隐患。
        - Fig.1A 读取多等级 LMM 结果，不再使用 Extreme-only 事件地图；
        - Fig.1B-D 直接从完整逐小时 SPI 合并审计表重算五级 SiteMonth CBI。
    """

    steps = [
        "检查输入文件",
        "读取福建边界",
        "读取站点坐标",
        "读取完整逐小时SPI合并表",
        "读取多等级SiteMonth配对审计表",
        "读取多等级LMM结果表",
        "统一字段类型",
    ]
    with progress_bar("步骤1/5 读取输入数据", len(steps), "cyan") as bar:
        require_files(
            [
                CFG.sites_csv,
                CFG.hourly_spi_merged_csv,
                CFG.multi_pair_audit_csv,
                CFG.multi_lmm_summary_csv,
                CFG.multi_lmm_restricted_csv,
                CFG.multi_trend_test_csv,
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

        hourly = pd.read_csv(
            CFG.hourly_spi_merged_csv,
            usecols=[
                "Site_ID",
                "YearMonth",
                "Month",
                "Site_Month",
                "ERA5_T2m_C",
                "Observed_T15cm_C",
                "Has_Both_Data",
                "SPI30d",
            ],
            low_memory=False,
        )
        bar.update()

        pair_audit = pd.read_csv(CFG.multi_pair_audit_csv)
        bar.update()

        lmm_summary = pd.read_csv(CFG.multi_lmm_summary_csv)
        lmm_restricted = pd.read_csv(CFG.multi_lmm_restricted_csv)
        trend_test = pd.read_csv(CFG.multi_trend_test_csv)
        bar.update()

        for df in (sites, hourly, pair_audit):
            df["Site_ID"] = df["Site_ID"].map(normalize_site_id)

        for col in ["Longitude", "Latitude"]:
            sites[col] = pd.to_numeric(sites[col], errors="coerce")

        for col in ["ERA5_T2m_C", "Observed_T15cm_C", "SPI30d"]:
            hourly[col] = pd.to_numeric(hourly[col], errors="coerce")

        for col in [
            "Target_n_hours",
            "Normal_n_hours_for_pair",
            "Target_Macro_SD",
            "Normal_Macro_SD_for_pair",
        ]:
            if col in pair_audit.columns:
                pair_audit[col] = pd.to_numeric(pair_audit[col], errors="coerce")

        for col in [
            "Pass_Hours",
            "Pass_Target_Macro_SD",
            "Pass_Normal_Macro_SD",
            "Pass_Hours_Plus_Macro_SD",
        ]:
            pair_audit[col] = pair_audit[col].astype(str).str.lower().isin(["true", "1", "yes"])
        bar.update()

    return boundary, sites, hourly, pair_audit, lmm_summary, lmm_restricted, trend_test


def prepare_data(
    sites: pd.DataFrame,
    hourly: pd.DataFrame,
    pair_audit: pd.DataFrame,
    lmm_summary: pd.DataFrame,
    lmm_restricted: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """生成 Fig. 1A-D 所需的统计表。

    输出含义：
        - sites_summary：全部站点坐标，用于 Fig. 1C 背景点；
        - forest_data：多等级 LMM 主模型和公共支持集模型，用于 Fig. 1A；
        - valid_pairs：五级 SiteMonth Target-vs-Normal CBI 配对，用于 Fig. 1B/D；
        - site_response：按站点汇总的五级 Delta_CBI 中位数和有效配对数，用于 Fig. 1C/D；

    关键科学约束：
        Fig.1B-D 必须展示 Mild/Moderate/Severe/Extreme 四级相对 Normal 的真实
        SiteMonth CBI，而不是旧 03_站点月份状态CBI估计表中的 Normal/Extreme 子集。
        因此这里直接从 00_逐小时温度_SPI合并审计表重新分类并估计 CBI，再用
        03_各等级_SiteMonth_配对审计.csv 的 Pass_Hours_Plus_Macro_SD 作为最终 QC。
    """

    with progress_bar("步骤2/5 清洗并汇总数据", 8, "green") as bar:
        sites_summary = sites[["Site_ID", "Longitude", "Latitude"]].copy()
        bar.update()

        class_order = ["Mild", "Moderate", "Severe", "Extreme"]
        main_forest = lmm_summary.loc[lmm_summary["Target_Class"].isin(class_order)].copy()
        main_forest["Model"] = "Main model"
        restricted_forest = lmm_restricted.loc[
            lmm_restricted["Target_Class"].isin(class_order)
            & lmm_restricted.get("RestrictionVersion", "").eq("All_class_common_month_site")
        ].copy()
        if restricted_forest.empty:
            restricted_forest = lmm_restricted.loc[lmm_restricted["Target_Class"].isin(class_order)].copy()
        restricted_forest["Model"] = "Common support"
        forest_data = pd.concat([main_forest, restricted_forest], ignore_index=True)
        forest_data["Target_Class"] = pd.Categorical(
            forest_data["Target_Class"], categories=class_order, ordered=True
        )
        forest_data = forest_data.sort_values(["Target_Class", "Model"]).reset_index(drop=True)
        bar.update()

        hourly_valid = hourly.loc[
            hourly["Has_Both_Data"].astype(str).str.lower().isin(["true", "1", "yes"])
            & hourly["ERA5_T2m_C"].between(-50, 60)
            & hourly["Observed_T15cm_C"].between(-40, 80)
            & hourly["SPI30d"].notna()
            & hourly["Site_ID"].ne("")
            & hourly["YearMonth"].notna()
        ].copy()
        hourly_valid["Drought_Class"] = classify_drought_class(hourly_valid["SPI30d"])
        hourly_valid = hourly_valid.loc[
            hourly_valid["Drought_Class"].isin(["Normal", "Mild", "Moderate", "Severe", "Extreme"])
        ].copy()
        bar.update()

        site_month_cbi = (
            hourly_valid.groupby(["Site_ID", "YearMonth", "Month", "Drought_Class"], dropna=False)
            .apply(estimate_site_month_status_cbi, include_groups=False)
            .reset_index()
        )
        bar.update()

        normal_cbi = site_month_cbi.loc[
            site_month_cbi["Drought_Class"].eq("Normal") & site_month_cbi["CBI_flag"].eq("ok"),
            ["Site_ID", "YearMonth", "CBI", "n_hours", "Macro_SD"],
        ].rename(
            columns={
                "CBI": "Normal_CBI",
                "n_hours": "Normal_n_hours_recomputed",
                "Macro_SD": "Normal_Macro_SD_recomputed",
            }
        )
        target_cbi = site_month_cbi.loc[
            site_month_cbi["Drought_Class"].isin(["Mild", "Moderate", "Severe", "Extreme"])
            & site_month_cbi["CBI_flag"].eq("ok"),
            ["Site_ID", "YearMonth", "Month", "Drought_Class", "CBI", "n_hours", "Macro_SD"],
        ].rename(
            columns={
                "Drought_Class": "Target_Class",
                "CBI": "Target_CBI",
                "n_hours": "Target_n_hours_recomputed",
                "Macro_SD": "Target_Macro_SD_recomputed",
            }
        )
        valid_pairs = target_cbi.merge(normal_cbi, on=["Site_ID", "YearMonth"], how="inner")
        valid_pairs["Delta_CBI"] = valid_pairs["Target_CBI"] - valid_pairs["Normal_CBI"]
        bar.update()

        qc_cols = [
            "Target_Class",
            "Target_Class_CN",
            "Site_ID",
            "YearMonth",
            "Target_n_hours",
            "Normal_n_hours_for_pair",
            "Target_Macro_SD",
            "Normal_Macro_SD_for_pair",
            "Pass_Hours",
            "Pass_Target_Macro_SD",
            "Pass_Normal_Macro_SD",
            "Pass_Hours_Plus_Macro_SD",
        ]
        qc = pair_audit.loc[
            pair_audit["Pass_Hours_Plus_Macro_SD"], [col for col in qc_cols if col in pair_audit.columns]
        ].drop_duplicates()
        valid_pairs = valid_pairs.merge(qc, on=["Site_ID", "YearMonth", "Target_Class"], how="inner")
        valid_pairs["Target_Class"] = pd.Categorical(
            valid_pairs["Target_Class"], categories=class_order, ordered=True
        )
        valid_pairs = valid_pairs.sort_values(["Target_Class", "Site_ID", "YearMonth"]).reset_index(drop=True)
        bar.update()

        site_response = (
            valid_pairs.groupby("Site_ID", as_index=False)
            .agg(
                Median_Delta_CBI=("Delta_CBI", "median"),
                Mean_Delta_CBI=("Delta_CBI", "mean"),
                Q25_Delta_CBI=("Delta_CBI", lambda x: x.quantile(0.25)),
                Q75_Delta_CBI=("Delta_CBI", lambda x: x.quantile(0.75)),
                N_pairs=("Delta_CBI", "size"),
                N_classes=("Target_Class", "nunique"),
                Median_Target_CBI=("Target_CBI", "median"),
                Median_Normal_CBI=("Normal_CBI", "median"),
            )
            .merge(sites[["Site_ID", "Longitude", "Latitude"]], on="Site_ID", how="left")
        )
        bar.update()

    return sites_summary, forest_data, valid_pairs, site_response


# =============================================================================
# 4. 四个面板绘图函数
# =============================================================================


def draw_panel_a(
    ax: plt.Axes,
    forest_data: pd.DataFrame,
    trend_test: pd.DataFrame,
) -> None:
    """Fig. 1A：多等级 Delta_CBI 森林图。

    面板任务：
        - 用主模型回答 Mild/Moderate/Severe/Extreme 是否相对 Normal 改变 CBI；
        - 用 common-support 限制样本模型提示结果是否受样本构成驱动；
        - 用等级顺序和 q 值呈现非单调模式，而不额外添加说明框干扰主信息。
    """

    class_order = ["Mild", "Moderate", "Severe", "Extreme"]
    y_positions = np.arange(len(class_order))[::-1]
    y_map = dict(zip(class_order, y_positions))
    palette = dict(zip(class_order, FP.drought_level_palette[1:]))
    main = forest_data.loc[forest_data["Model"].eq("Main model")].copy()
    restricted = forest_data.loc[forest_data["Model"].eq("Common support")].copy()

    ax.axvline(0, color=FP.panel_a_zero_line_color, linestyle="--", lw=0.9, zorder=1)
    for _, row in restricted.iterrows():
        drought_class = str(row["Target_Class"])
        y = y_map[drought_class] - 0.12
        x = row["DeltaCBI_Target_minus_Normal"]
        ax.scatter(
            x,
            y,
            marker="s",
            s=FP.panel_a_restricted_marker_size,
            facecolors="none",
            edgecolors=palette[drought_class],
            linewidths=0.9,
            alpha=FP.panel_a_restricted_alpha,
            zorder=2,
        )

    for _, row in main.iterrows():
        drought_class = str(row["Target_Class"])
        y = y_map[drought_class] + 0.12
        x = row["DeltaCBI_Target_minus_Normal"]
        ax.errorbar(
            x,
            y,
            xerr=[
                [x - row["DeltaCBI_CI_low_95"]],
                [row["DeltaCBI_CI_high_95"] - x],
            ],
            fmt="o",
            ms=np.sqrt(FP.panel_a_marker_size),
            color=palette[drought_class],
            markeredgecolor="#1a1a1a",
            ecolor=palette[drought_class],
            elinewidth=FP.panel_a_ci_linewidth,
            capsize=4,
            zorder=3,
        )

    for _, row in main.iterrows():
        drought_class = str(row["Target_Class"])
        y = y_map[drought_class] + 0.12
        q = row.get("DeltaCBI_p_FDR", np.nan)
        ax.text(
            row["DeltaCBI_Target_minus_Normal"],
            y + 0.24,
            f"n_site={int(row['n_sites'])}, q{format_compact_p(q)}",
            ha="center",
            va="bottom",
            fontsize=FP.panel_a_text_size,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(class_order)
    ax.set_xlabel("ΔCBI (drought class - Normal)")
    ax.set_ylabel("Drought class")
    set_panel_title(ax, "A", "Multi-level drought effects on CBI")
    # x 轴范围只按图上实际强调的信息确定：主模型 95% CI + common-support 方块点。
    # 不使用 common-support 的 CI，否则会被未显示的宽置信区间拉出大块空白。
    x_extent_values = pd.concat(
        [
            pd.to_numeric(main["DeltaCBI_CI_low_95"], errors="coerce"),
            pd.to_numeric(main["DeltaCBI_CI_high_95"], errors="coerce"),
            pd.to_numeric(restricted["DeltaCBI_Target_minus_Normal"], errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()
    ci_min = x_extent_values.min()
    ci_max = x_extent_values.max()
    ci_span = ci_max - ci_min
    if not np.isfinite(ci_span) or ci_span <= 0:
        ci_min, ci_max, ci_span = -0.02, 0.08, 0.10
    ax.set_xlim(
        ci_min - ci_span * FP.panel_a_xlim_left_padding_fraction,
        ci_max + ci_span * FP.panel_a_xlim_right_padding_fraction,
    )
    ax.set_ylim(-0.65, len(class_order) - 0.35)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="#555555",
                markeredgecolor="#1a1a1a",
                markersize=6,
                label="Main model (all valid pairs)",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor="#555555",
                markersize=6,
                label="Common support subset",
            ),
        ],
        frameon=False,
        loc=FP.panel_a_legend_loc,
        bbox_to_anchor=(FP.panel_a_legend_anchor_x, FP.panel_a_legend_anchor_y),
        borderaxespad=0,
    )


def draw_panel_b(ax: plt.Axes, valid_pairs: pd.DataFrame, delta_norm: TwoSlopeNorm) -> None:
    """Fig. 1B：五级 SiteMonth 绝对 CBI 分布。

    该面板保留绝对 CBI 尺度，而不是只画 Delta_CBI：
        - CBI 越接近 1，林内 15 cm 温度越跟随 ERA5 2 m 宏气候温度，缓冲越弱；
        - CBI=1 参考线用于帮助读者判断 0.01-0.03 的变化是否具有物理意义；
        - Normal 样本来自所有有效 Target-vs-Normal 配对的参考状态，可能重复出现，
          这是有意保留的配对权重表达，保证和 Delta_CBI 配对底表一致。
    """

    normal_data = (
        valid_pairs[["Site_ID", "YearMonth", "Normal_CBI"]]
        .drop_duplicates(["Site_ID", "YearMonth"])
        .rename(columns={"Normal_CBI": "CBI"})
    )
    normal_data["Status"] = "Normal"
    normal_data["Delta_CBI"] = np.nan
    target_data = valid_pairs[
        ["Site_ID", "YearMonth", "Target_CBI", "Target_Class", "Delta_CBI"]
    ].rename(columns={"Target_CBI": "CBI", "Target_Class": "Status"})
    box_data = pd.concat([normal_data, target_data], ignore_index=True)
    order = ["Normal", "Mild", "Moderate", "Severe", "Extreme"]
    palette = dict(zip(order, FP.drought_level_palette))

    grouped = [box_data.loc[box_data["Status"].eq(status), "CBI"].dropna().to_numpy() for status in order]
    positions = np.arange(len(order))
    box = ax.boxplot(
        grouped,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.2},
        boxprops={"linewidth": 0.8, "color": "#333333"},
        whiskerprops={"linewidth": 0.8, "color": "#333333"},
        capprops={"linewidth": 0.8, "color": "#333333"},
    )
    for patch, status in zip(box["boxes"], order):
        patch.set_facecolor(palette[status])
        patch.set_alpha(0.38 if status == "Normal" else 0.52)

    rng = np.random.default_rng(FP.panel_b_jitter_seed)
    for idx, status in enumerate(order):
        subset = box_data.loc[box_data["Status"].eq(status)].copy()
        if subset.empty:
            continue
        x = idx + rng.uniform(-FP.panel_b_jitter_half_width, FP.panel_b_jitter_half_width, len(subset))
        ax.scatter(
            x,
            subset["CBI"],
            s=FP.panel_b_point_size_min,
            facecolors=palette[status],
            edgecolors=FP.panel_b_point_edgecolor if status != "Normal" else "#9a9a9a",
            linewidths=FP.panel_b_point_linewidth,
            alpha=0.42 if status == "Normal" else FP.panel_b_point_alpha,
            zorder=3,
        )

    label_y_values: list[float] = []
    all_cbi = pd.to_numeric(box_data["CBI"], errors="coerce").dropna()
    y_span = float(all_cbi.max() - all_cbi.min()) if not all_cbi.empty else 1.0
    label_offset = max(y_span * FP.panel_b_median_label_offset_fraction, 0.010)
    for idx, status in enumerate(order):
        values = box_data.loc[box_data["Status"].eq(status), "CBI"].dropna()
        if values.empty:
            continue
        upper_cap = box["caps"][2 * idx + 1]
        upper_anchor = float(np.max(upper_cap.get_ydata()))
        median_value = float(values.median())
        label_y = upper_anchor + label_offset
        label_y_values.append(label_y)
        ax.text(
            idx,
            label_y,
            f"median = {median_value:.2f}",
            ha="center",
            va="bottom",
            fontsize=FP.panel_b_median_label_size,
            color=FP.panel_b_median_label_color,
            zorder=7,
        )

    ax.axhline(
        FP.panel_b_reference_line_value,
        color=FP.panel_b_reference_line_color,
        linestyle="--",
        lw=FP.panel_b_identity_linewidth,
        zorder=1,
    )
    counts = box_data.groupby("Status").size()
    xlabels = [f"{status}\n(n={int(counts.get(status, 0))})" for status in order]
    ax.set_xticks(positions)
    ax.set_xticklabels(xlabels, rotation=0, ha="center")
    ax.set_xlabel("SPI30d class")
    ax.set_ylabel("Site-month CBI")
    if label_y_values:
        ymin, ymax = ax.get_ylim()
        required_top = max(label_y_values) + label_offset * 2.5
        if required_top > ymax:
            ax.set_ylim(ymin, required_top)
    set_panel_title(ax, "B", "Absolute CBI distributions across drought classes")


def draw_panel_c(
    ax: plt.Axes,
    boundary: gpd.GeoDataFrame,
    sites_summary: pd.DataFrame,
    site_response: pd.DataFrame,
    delta_norm: TwoSlopeNorm,
) -> None:
    """Fig. 1C：站点尺度中位数 Delta_CBI 的空间分布图。

    底图灰点显示全部站点；有有效事件-参考期配对的站点按中位数 Delta_CBI 着色，
    点大小固定，避免样本量差异干扰空间结构判断；N_pairs 只写入输出表和图注。
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
    plot_sites = site_response.sort_values("Median_Delta_CBI", ascending=True).copy()
    ax.scatter(
        plot_sites["Longitude"],
        plot_sites["Latitude"],
        s=FP.panel_c_response_size,
        c=plot_sites["Median_Delta_CBI"],
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
    set_panel_title(ax, "C", "Spatial distribution of median responses")


def draw_panel_d(
    ax: plt.Axes,
    site_response: pd.DataFrame,
    site_delta_norm: TwoSlopeNorm,
    site_delta_limit: float,
) -> None:
    """Fig. 1D：站点中位数 Delta_CBI 的棒棒糖排序图。

    每一行代表一个有五级有效配对的站点；横向线段从 Delta_CBI=0 延伸到该站点
    的 Median_Delta_CBI，端点颜色与 Fig.1C 一致。该设计牺牲事件级散点细节，
    换取更清晰的站点差异排序；N_pairs 继续保存在输出表和图注中。
    """

    ordered = site_response.sort_values("Median_Delta_CBI", ascending=True).reset_index(drop=True)
    y_positions = np.arange(len(ordered))
    x_values = pd.to_numeric(ordered["Median_Delta_CBI"], errors="coerce").to_numpy()
    point_cmap = plt.get_cmap(FP.delta_cmap)

    ax.axvline(
        0,
        color=FP.panel_d_zero_line_color,
        lw=FP.panel_d_zero_linewidth,
        linestyle=FP.panel_d_zero_linestyle,
        zorder=1,
    )
    for y, x in zip(y_positions, x_values):
        ax.hlines(
            y,
            min(0, x),
            max(0, x),
            color=point_cmap(site_delta_norm(x)),
            lw=FP.panel_d_lollipop_linewidth,
            alpha=FP.panel_d_lollipop_alpha,
            zorder=2,
        )

    ax.scatter(
        x_values,
        y_positions,
        marker="o",
        s=FP.panel_d_site_median_size,
        c=x_values,
        cmap=FP.delta_cmap,
        norm=site_delta_norm,
        edgecolors=FP.panel_d_site_median_edgecolor,
        linewidths=FP.panel_d_site_median_linewidth,
        zorder=3,
    )
    for y in y_positions:
        ax.hlines(
            y,
            -site_delta_limit,
            site_delta_limit,
            color=FP.panel_d_row_line_color,
            lw=FP.panel_d_row_linewidth,
            zorder=0,
        )

    label_count = max(0, int(FP.panel_d_extreme_label_count))
    label_indices = set(ordered.head(label_count).index).union(set(ordered.tail(label_count).index))
    x_offset = site_delta_limit * 0.025
    for idx in sorted(label_indices):
        row = ordered.loc[idx]
        x = float(row["Median_Delta_CBI"])
        ha = "left" if x >= 0 else "right"
        ax.text(
            x + (x_offset if x >= 0 else -x_offset),
            idx,
            str(row["Site_ID"]),
            va="center",
            ha=ha,
            fontsize=FP.panel_d_site_label_size,
            color="#222222",
        )

    ax.set_xlim(-site_delta_limit, site_delta_limit)
    ax.set_ylim(-0.75, len(ordered) - 0.25)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(i + 1) for i in y_positions], fontsize=FP.tick_label_size)
    ax.set_xlabel("Site median ΔCBI (drought class - Normal)")
    ax.set_ylabel("Site rank")
    set_panel_title(ax, "D", "Ranked site-level response magnitude")


def draw_panel_a_optimized(
    ax: plt.Axes,
    forest_data: pd.DataFrame,
    trend_test: pd.DataFrame,
) -> None:
    """优化版 Fig. 1A：主模型森林图，common-support 仅作为 inset 稳健性对照。

    主面板只展示 Main model 的效应量、95% CI、n_sites 和 FDR q，避免 common-support
    点估计偏大时干扰主结论。inset 只回答“限制公共支持集后方向和量级是否一致”。
    """

    class_order = ["Mild", "Moderate", "Severe", "Extreme"]
    y_positions = np.arange(len(class_order))[::-1]
    y_map = dict(zip(class_order, y_positions))
    palette = dict(zip(class_order, FP.drought_level_palette[1:]))
    main = forest_data.loc[forest_data["Model"].eq("Main model")].copy()
    restricted = forest_data.loc[forest_data["Model"].eq("Common support")].copy()

    ax.axvline(0, color=FP.panel_a_zero_line_color, linestyle="--", lw=0.9, zorder=1)
    for _, row in main.iterrows():
        drought_class = str(row["Target_Class"])
        y = y_map[drought_class]
        x = row["DeltaCBI_Target_minus_Normal"]
        ax.errorbar(
            x,
            y,
            xerr=[
                [x - row["DeltaCBI_CI_low_95"]],
                [row["DeltaCBI_CI_high_95"] - x],
            ],
            fmt="o",
            ms=np.sqrt(FP.panel_a_marker_size),
            color=palette[drought_class],
            markeredgecolor="#1a1a1a",
            ecolor=palette[drought_class],
            elinewidth=FP.panel_a_ci_linewidth,
            capsize=4,
            zorder=3,
        )
        q = row.get("DeltaCBI_p_FDR", np.nan)
        ax.text(
            x,
            y + 0.24,
            f"n_site={int(row['n_sites'])}, q{format_compact_p(q)}",
            ha="center",
            va="bottom",
            fontsize=FP.panel_a_text_size,
        )

    main_extent = pd.concat(
        [
            pd.to_numeric(main["DeltaCBI_CI_low_95"], errors="coerce"),
            pd.to_numeric(main["DeltaCBI_CI_high_95"], errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()
    ci_min = float(main_extent.min()) if not main_extent.empty else -0.02
    ci_max = float(main_extent.max()) if not main_extent.empty else 0.08
    ci_span = ci_max - ci_min
    if not np.isfinite(ci_span) or ci_span <= 0:
        ci_min, ci_max, ci_span = -0.02, 0.08, 0.10
    ax.set_xlim(
        ci_min - ci_span * FP.panel_a_xlim_left_padding_fraction,
        ci_max + ci_span * FP.panel_a_xlim_right_padding_fraction,
    )
    ax.set_ylim(-0.65, len(class_order) - 0.35)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(class_order)
    ax.set_xlabel("ΔCBI (drought class - Normal)")
    ax.set_ylabel("Drought class")
    set_panel_title(ax, "A", "Model-estimated drought effects")

    if not restricted.empty:
        inset = ax.inset_axes([0.64, 0.33, 0.30, 0.34])
        merged = main[["Target_Class", "DeltaCBI_Target_minus_Normal"]].merge(
            restricted[["Target_Class", "DeltaCBI_Target_minus_Normal"]],
            on="Target_Class",
            how="inner",
            suffixes=("_main", "_common"),
        )
        x = np.arange(len(merged))
        inset.axhline(0, color="#777777", linestyle="--", lw=0.7, zorder=1)
        for idx, row in merged.iterrows():
            main_x = row["DeltaCBI_Target_minus_Normal_main"]
            common_x = row["DeltaCBI_Target_minus_Normal_common"]
            inset.plot([idx, idx], [main_x, common_x], color="#b0b0b0", lw=0.8, zorder=1)
        inset.scatter(
            x - 0.05,
            merged["DeltaCBI_Target_minus_Normal_main"],
            marker="o",
            s=14,
            color="#444444",
            edgecolors="#222222",
            linewidths=0.4,
            label="Main",
            zorder=3,
        )
        inset.scatter(
            x + 0.05,
            merged["DeltaCBI_Target_minus_Normal_common"],
            marker="s",
            s=13,
            facecolors="none",
            edgecolors="#999999",
            linewidths=0.8,
            label="Common",
            zorder=2,
        )
        inset.set_xticks(x)
        inset.set_xticklabels(["Mi", "Mo", "Se", "Ex"], fontsize=6)
        inset.tick_params(axis="y", labelsize=6, length=2)
        inset.tick_params(axis="x", length=2)
        inset.set_title("Sensitivity", fontsize=7, pad=2)
        inset.legend(
            frameon=False,
            fontsize=5.8,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=2,
            handlelength=1.4,
            columnspacing=0.8,
            borderaxespad=0,
        )
        for spine in inset.spines.values():
            spine.set_linewidth(0.6)


def draw_panel_d_optimized(
    ax: plt.Axes,
    site_response: pd.DataFrame,
    site_delta_norm: TwoSlopeNorm,
    site_delta_limit: float,
) -> None:
    """优化版 Fig. 1D：站点中位 Delta_CBI + IQR 排序图。

    点表示站点中位响应；横向区间表示站点内有效配对 Delta_CBI 的 25%-75% 分位数。
    相比棒棒糖图，该面板同时传递站点排序和站点内部响应稳定性。
    """

    ordered = site_response.sort_values("Median_Delta_CBI", ascending=True).reset_index(drop=True)
    y_positions = np.arange(len(ordered))
    median = pd.to_numeric(ordered["Median_Delta_CBI"], errors="coerce").to_numpy()
    q25 = pd.to_numeric(ordered["Q25_Delta_CBI"], errors="coerce").to_numpy()
    q75 = pd.to_numeric(ordered["Q75_Delta_CBI"], errors="coerce").to_numpy()
    iqr_limit = finite_symmetric_limit_with_padding(
        pd.Series(np.concatenate([median, q25, q75])),
        fallback=site_delta_limit,
        padding=FP.panel_d_iqr_axis_padding,
    )

    ax.axvline(
        0,
        color=FP.panel_d_zero_line_color,
        lw=FP.panel_d_zero_linewidth,
        linestyle=FP.panel_d_zero_linestyle,
        zorder=1,
    )
    for y, lo, hi, med in zip(y_positions, q25, q75, median):
        ax.hlines(
            y,
            lo,
            hi,
            color=plt.get_cmap(FP.delta_cmap)(site_delta_norm(med)),
            lw=FP.panel_d_iqr_linewidth,
            alpha=FP.panel_d_iqr_alpha,
            zorder=2,
        )
    ax.scatter(
        median,
        y_positions,
        marker="o",
        s=FP.panel_d_site_median_size,
        c=median,
        cmap=FP.delta_cmap,
        norm=site_delta_norm,
        edgecolors=FP.panel_d_site_median_edgecolor,
        linewidths=FP.panel_d_site_median_linewidth,
        zorder=3,
    )
    for y in y_positions:
        ax.hlines(
            y,
            -iqr_limit,
            iqr_limit,
            color=FP.panel_d_row_line_color,
            lw=FP.panel_d_row_linewidth,
            zorder=0,
        )

    ax.set_xlim(-iqr_limit, iqr_limit)
    ax.set_ylim(-0.75, len(ordered) - 0.25)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(i + 1) for i in y_positions], fontsize=FP.tick_label_size)
    ax.set_xlabel("Site-level ΔCBI (median and IQR)")
    ax.set_ylabel("Site rank")
    set_panel_title(ax, "D", "Ranked site-level median and IQR")
    ax.legend(
        handles=[
            Line2D([0], [0], color="#777777", lw=FP.panel_d_iqr_linewidth, label="IQR"),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="#cccccc",
                markeredgecolor=FP.panel_d_site_median_edgecolor,
                markersize=5,
                label="Median",
            ),
        ],
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.72,
        loc="lower right",
        fontsize=FP.legend_font_size,
    )


def add_panel_c_colorbar(fig: plt.Figure, ax_c: plt.Axes, site_delta_norm: TwoSlopeNorm) -> None:
    """给 Fig.1C 添加与 C/D 共享的站点中位 Delta_CBI 色标。"""

    cax_delta_c = ax_c.inset_axes(
        [
            FP.panel_c_delta_colorbar_x,
            FP.panel_c_delta_colorbar_y,
            FP.panel_c_delta_colorbar_width,
            FP.panel_c_delta_colorbar_height,
        ]
    )
    cbar_delta_c = fig.colorbar(
        mpl.cm.ScalarMappable(norm=site_delta_norm, cmap=FP.delta_cmap),
        cax=cax_delta_c,
        orientation="horizontal",
    )
    cbar_delta_c.set_label("Median ΔCBI (class - Normal)", labelpad=1, fontsize=FP.colorbar_label_size)
    cbar_delta_c.ax.xaxis.set_label_position("top")
    cbar_delta_c.ax.tick_params(labelsize=FP.colorbar_tick_size, length=2)


def draw_panel_a_absolute(
    ax: plt.Axes,
    valid_pairs: pd.DataFrame,
    forest_data: pd.DataFrame,
) -> None:
    """绝对 CBI 对比版 Fig.1A：每个干旱等级内的 Normal vs drought 成对对比。"""

    class_order = ["Mild", "Moderate", "Severe", "Extreme"]
    class_to_x = dict(zip(class_order, np.arange(len(class_order), dtype=float)))
    rng = np.random.default_rng(FP.absolute_panel_a_jitter_seed)
    main = forest_data.loc[forest_data["Model"].eq("Main model")].copy()

    for drought_class in class_order:
        subset = valid_pairs.loc[valid_pairs["Target_Class"].astype(str).eq(drought_class)].copy()
        if subset.empty:
            continue
        center = class_to_x[drought_class]
        x_normal = center - 0.18
        x_drought = center + 0.18
        normal_values = pd.to_numeric(subset["Normal_CBI"], errors="coerce").dropna().to_numpy()
        drought_values = pd.to_numeric(subset["Target_CBI"], errors="coerce").dropna().to_numpy()

        pair_sample = subset.sample(
            n=min(len(subset), 80),
            random_state=FP.absolute_panel_a_jitter_seed,
        )
        for _, row in pair_sample.iterrows():
            ax.plot(
                [x_normal, x_drought],
                [row["Normal_CBI"], row["Target_CBI"]],
                color=FP.absolute_panel_a_pair_line_color,
                alpha=FP.absolute_panel_a_pair_line_alpha,
                lw=FP.absolute_panel_a_pair_linewidth,
                zorder=1,
            )

        box = ax.boxplot(
            [normal_values, drought_values],
            positions=[x_normal, x_drought],
            widths=FP.absolute_panel_a_box_width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.0},
            boxprops={"linewidth": 0.75, "color": "#333333"},
            whiskerprops={"linewidth": 0.75, "color": "#333333"},
            capprops={"linewidth": 0.75, "color": "#333333"},
        )
        for patch, color in zip(box["boxes"], [FP.absolute_normal_color, FP.absolute_drought_color]):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)

        ax.scatter(
            x_normal + rng.uniform(-FP.absolute_panel_a_jitter_half_width, FP.absolute_panel_a_jitter_half_width, len(normal_values)),
            normal_values,
            s=FP.absolute_panel_a_point_size,
            color=FP.absolute_normal_color,
            edgecolors="white",
            linewidths=0.25,
            alpha=FP.absolute_panel_a_point_alpha,
            zorder=2,
        )
        ax.scatter(
            x_drought + rng.uniform(-FP.absolute_panel_a_jitter_half_width, FP.absolute_panel_a_jitter_half_width, len(drought_values)),
            drought_values,
            s=FP.absolute_panel_a_point_size,
            color=FP.absolute_drought_color,
            edgecolors="white",
            linewidths=0.25,
            alpha=FP.absolute_panel_a_point_alpha,
            zorder=2,
        )

        model_row = main.loc[main["Target_Class"].astype(str).eq(drought_class)]
        if not model_row.empty:
            row = model_row.iloc[0]
            y_top = max(np.nanmax(normal_values), np.nanmax(drought_values))
            ax.text(
                center,
                y_top + 0.045,
                f"n_site={int(row['n_sites'])}, q{format_compact_p(row.get('DeltaCBI_p_FDR', np.nan))}",
                ha="center",
                va="bottom",
                fontsize=FP.panel_a_text_size,
                color="#222222",
            )

    ax.axhline(
        FP.panel_b_reference_line_value,
        color=FP.panel_b_reference_line_color,
        linestyle="--",
        lw=FP.panel_b_identity_linewidth,
        zorder=0,
    )
    ax.set_xticks(np.arange(len(class_order)))
    ax.set_xticklabels(class_order)
    ax.set_xlabel("Drought class")
    ax.set_ylabel("Site-month CBI")
    y_min, y_max = finite_value_limits(
        pd.concat([valid_pairs["Normal_CBI"], valid_pairs["Target_CBI"]], ignore_index=True),
        fallback=(0.3, 1.3),
        padding=0.14,
    )
    ax.set_ylim(y_min, y_max)
    set_panel_title(ax, "A", "Paired absolute CBI contrasts")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="s", linestyle="None", markerfacecolor=FP.absolute_normal_color,
                   markeredgecolor="#333333", markersize=6, label="Normal reference"),
            Line2D([0], [0], marker="s", linestyle="None", markerfacecolor=FP.absolute_drought_color,
                   markeredgecolor="#333333", markersize=6, label="Drought class"),
        ],
        frameon=False,
        loc="upper left",
        fontsize=FP.legend_font_size,
    )


def draw_panel_c_absolute(
    ax: plt.Axes,
    boundary: gpd.GeoDataFrame,
    sites_summary: pd.DataFrame,
    absolute_site_response: pd.DataFrame,
    cbi_norm: mpl.colors.Normalize,
) -> None:
    """绝对 CBI 对比版 Fig.1C：干旱期站点中位绝对 CBI 空间分布。"""

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
    plot_sites = absolute_site_response.sort_values("Median_Drought_CBI", ascending=True)
    ax.scatter(
        plot_sites["Longitude"],
        plot_sites["Latitude"],
        s=FP.absolute_panel_c_response_size,
        c=plot_sites["Median_Drought_CBI"],
        cmap=FP.absolute_cmap,
        norm=cbi_norm,
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
    set_panel_title(ax, "C", "Spatial distribution of drought-period CBI")


def draw_panel_d_absolute(
    ax: plt.Axes,
    absolute_site_response: pd.DataFrame,
    cbi_norm: mpl.colors.Normalize,
) -> None:
    """绝对 CBI 对比版 Fig.1D：站点 Normal median CBI 到 drought median CBI 的成对迁移。"""

    ordered = absolute_site_response.sort_values("Median_Drought_CBI", ascending=True).reset_index(drop=True)
    y_positions = np.arange(len(ordered))
    normal = pd.to_numeric(ordered["Median_Normal_CBI"], errors="coerce").to_numpy()
    drought = pd.to_numeric(ordered["Median_Drought_CBI"], errors="coerce").to_numpy()
    cbi_min, cbi_max = finite_value_limits(
        pd.Series(np.concatenate([normal, drought])),
        fallback=(0.3, 1.3),
        padding=FP.absolute_cbi_limit_padding,
    )

    ax.axvline(
        FP.panel_b_reference_line_value,
        color=FP.panel_b_reference_line_color,
        linestyle="--",
        lw=FP.panel_b_identity_linewidth,
        zorder=1,
    )
    for y, n_value, d_value in zip(y_positions, normal, drought):
        ax.hlines(
            y,
            n_value,
            d_value,
            color=FP.absolute_panel_d_line_color,
            alpha=FP.absolute_panel_d_line_alpha,
            lw=FP.absolute_panel_d_linewidth,
            zorder=1,
        )
    ax.scatter(
        normal,
        y_positions,
        s=FP.absolute_panel_d_point_size,
        color=FP.absolute_normal_color,
        edgecolors=FP.panel_d_site_median_edgecolor,
        linewidths=FP.panel_d_site_median_linewidth,
        label="Normal median",
        zorder=3,
    )
    ax.scatter(
        drought,
        y_positions,
        s=FP.absolute_panel_d_point_size,
        c=drought,
        cmap=FP.absolute_cmap,
        norm=cbi_norm,
        edgecolors=FP.panel_d_site_median_edgecolor,
        linewidths=FP.panel_d_site_median_linewidth,
        label="Drought median",
        zorder=4,
    )
    for y in y_positions:
        ax.hlines(y, cbi_min, cbi_max, color=FP.panel_d_row_line_color, lw=FP.panel_d_row_linewidth, zorder=0)

    ax.set_xlim(cbi_min, cbi_max)
    ax.set_ylim(-0.75, len(ordered) - 0.25)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(i + 1) for i in y_positions], fontsize=FP.tick_label_size)
    ax.set_xlabel("Site median CBI")
    ax.set_ylabel("Site rank by drought-period CBI")
    set_panel_title(ax, "D", "Site-level Normal-to-drought CBI shift")
    ax.legend(frameon=False, loc="lower right", fontsize=FP.legend_font_size)


def add_absolute_cbi_colorbar(fig: plt.Figure, ax_c: plt.Axes, cbi_norm: mpl.colors.Normalize) -> None:
    """给绝对 CBI 对比版 Fig.1C 添加干旱期 CBI 连续色标。"""

    cax = ax_c.inset_axes(
        [
            FP.absolute_panel_c_colorbar_x,
            FP.absolute_panel_c_colorbar_y,
            FP.absolute_panel_c_colorbar_width,
            FP.absolute_panel_c_colorbar_height,
        ]
    )
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=cbi_norm, cmap=FP.absolute_cmap),
        cax=cax,
        orientation="horizontal",
    )
    cbar.set_label("Drought-period median CBI", labelpad=1, fontsize=FP.colorbar_label_size)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.tick_params(labelsize=FP.colorbar_tick_size, length=2)


# =============================================================================
# 5. 输出表格、参数说明和缓存清理
# =============================================================================


def parameters_to_table() -> pd.DataFrame:
    """把 Config 和 FigureParameters 写成可读表格，便于后续按参数名修改。"""

    rows: list[dict[str, object]] = []
    descriptions = {
        "extreme_spi_threshold": "极端干旱阈值；SPI30d <= 该值被视为 extreme drought。",
        "severe_spi_threshold": "重度干旱边界；-2.0 < SPI30d <= -1.5 被视为 severe drought。",
        "moderate_spi_threshold": "中度干旱边界；-1.5 < SPI30d <= -1.0 被视为 moderate drought。",
        "normal_spi_low": "正常参考状态下界；Normal 为 -0.5 < SPI30d < 0.5。",
        "normal_spi_high": "正常参考状态上界；Normal 为 -0.5 < SPI30d < 0.5。",
        "min_site_month_status_cbi_hours": "重算 SiteMonth 状态 CBI 的最低逐小时样本量。",
        "min_site_month_status_macro_sd": "重算 SiteMonth 状态 CBI 的最低 ERA5 温度标准差。",
        "figure_dpi": "PNG 输出分辨率；投稿图建议 600 dpi。",
        "fig_width": "组合图宽度，单位英寸。",
        "fig_height": "组合图高度，单位英寸。",
        "grid_wspace": "左右面板水平间距；增大可减少面板拥挤。",
        "grid_hspace": "上下两行面板间距；增大可避免色标和标题重叠。",
        "panel_a_marker_size": "Fig.1A 主模型点估计圆点面积。",
        "panel_a_restricted_marker_size": "Fig.1A common-support 对照模型方块面积。",
        "panel_a_ci_linewidth": "Fig.1A 95% CI 误差线线宽。",
        "delta_cmap": "Fig.1B/C ΔCBI 发散色带；当前红色为增加、蓝色为降低。",
        "site_delta_limit_padding": "Fig.1C/D 站点中位 Delta_CBI 色标和 D 横轴的范围扩展倍数。",
        "panel_b_identity_linewidth": "Fig.1B CBI=1 无缓冲参考线宽度。",
        "panel_b_identity_linestyle": "Fig.1B CBI=1 无缓冲参考线线型。",
        "panel_b_median_label_offset_fraction": "Fig.1B median 标签高于箱线图上须横线的距离比例。",
        "panel_c_response_size": "Fig.1C 有效配对站点统一点面积；N_pairs 不再映射为点大小。",
        "panel_c_delta_colorbar_x": "Fig.1C ΔCBI 色标相对位置 x。",
        "panel_c_delta_colorbar_y": "Fig.1C ΔCBI 色标相对位置 y。",
        "panel_c_delta_colorbar_width": "Fig.1C ΔCBI 色标相对宽度。",
        "panel_c_delta_colorbar_height": "Fig.1C ΔCBI 色标相对高度。",
        "panel_d_site_median_size": "Fig.1D 站点中位 Delta_CBI 端点面积。",
        "panel_d_lollipop_linewidth": "Fig.1D 棒棒糖线段宽度。",
        "panel_d_iqr_linewidth": "优化版 Fig.1D 站点内 Delta_CBI IQR 横线宽度。",
        "panel_d_iqr_alpha": "优化版 Fig.1D 站点内 Delta_CBI IQR 横线透明度。",
        "panel_d_iqr_axis_padding": "优化版 Fig.1D 按 Q25/Q75/median 定横轴范围时的扩展倍数。",
        "panel_d_extreme_label_count": "Fig.1D 在高低两端各标注的站点数量。",
        "absolute_normal_color": "绝对 CBI 对比版 Normal reference 的颜色。",
        "absolute_drought_color": "绝对 CBI 对比版 Drought class 的颜色。",
        "absolute_cmap": "绝对 CBI 对比版 C/D 面板干旱期 CBI 连续色带。",
        "absolute_cbi_limit_padding": "绝对 CBI 对比版坐标轴和色标范围扩展比例。",
        "absolute_panel_a_pair_line_alpha": "绝对 CBI 对比版 A 面板配对连线透明度。",
        "absolute_panel_a_box_width": "绝对 CBI 对比版 A 面板箱线图宽度。",
        "absolute_panel_a_point_size": "绝对 CBI 对比版 A 面板散点大小。",
        "absolute_panel_a_jitter_half_width": "绝对 CBI 对比版 A 面板散点水平抖动半宽。",
        "absolute_panel_c_response_size": "绝对 CBI 对比版 C 面板站点点面积。",
        "absolute_panel_c_colorbar_x": "绝对 CBI 对比版 C 面板色标相对位置 x。",
        "absolute_panel_c_colorbar_y": "绝对 CBI 对比版 C 面板色标相对位置 y。",
        "absolute_panel_c_colorbar_width": "绝对 CBI 对比版 C 面板色标相对宽度。",
        "absolute_panel_c_colorbar_height": "绝对 CBI 对比版 C 面板色标相对高度。",
        "absolute_panel_d_line_alpha": "绝对 CBI 对比版 D 面板 Normal-to-drought 连线透明度。",
        "absolute_panel_d_linewidth": "绝对 CBI 对比版 D 面板 Normal-to-drought 连线宽度。",
        "absolute_panel_d_point_size": "绝对 CBI 对比版 D 面板站点中位数点面积。",
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
    fig_optimized: plt.Figure,
    fig_absolute: plt.Figure,
    sites_summary: pd.DataFrame,
    forest_data: pd.DataFrame,
    valid_pairs: pd.DataFrame,
    site_response: pd.DataFrame,
    absolute_site_response: pd.DataFrame,
) -> pd.DataFrame:
    """保存所有图、表和参数说明，文件名全部使用中文。"""

    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    with progress_bar("步骤4/5 保存图表与参数", 15, "magenta") as bar:
        fig.savefig(CFG.output_dir / "图1_现象图_主图.png", dpi=CFG.figure_dpi)
        bar.update()

        fig.savefig(CFG.output_dir / "图1_现象图_主图.pdf")
        bar.update()

        fig_optimized.savefig(CFG.output_dir / "图1_现象图_主图_优化版.png", dpi=CFG.figure_dpi)
        bar.update()

        fig_optimized.savefig(CFG.output_dir / "图1_现象图_主图_优化版.pdf")
        bar.update()

        fig_absolute.savefig(CFG.output_dir / "图1_现象图_主图_绝对CBI对比版.png", dpi=CFG.figure_dpi)
        bar.update()

        fig_absolute.savefig(CFG.output_dir / "图1_现象图_主图_绝对CBI对比版.pdf")
        bar.update()

        sites_summary.to_csv(CFG.output_dir / "图1C_全部站点坐标表.csv", index=False, encoding="utf-8-sig")
        bar.update()

        forest_data.to_csv(CFG.output_dir / "图1A_多等级DeltaCBI森林图底层数据.csv", index=False, encoding="utf-8-sig")
        bar.update()

        valid_pairs.to_csv(CFG.output_dir / "图1B和图1D_五级SiteMonth有效配对表.csv", index=False, encoding="utf-8-sig")
        bar.update()

        site_response.to_csv(CFG.output_dir / "图1C和图1D_五级站点DeltaCBI汇总表.csv", index=False, encoding="utf-8-sig")
        bar.update()

        site_response[
            [
                "Site_ID",
                "Median_Delta_CBI",
                "Q25_Delta_CBI",
                "Q75_Delta_CBI",
                "N_pairs",
                "N_classes",
                "Longitude",
                "Latitude",
            ]
        ].to_csv(CFG.output_dir / "图1D_站点中位数与IQR排序图底层数据.csv", index=False, encoding="utf-8-sig")
        bar.update()

        absolute_site_response.to_csv(
            CFG.output_dir / "图1_绝对CBI对比版_站点Normal与干旱期CBI汇总表.csv",
            index=False,
            encoding="utf-8-sig",
        )
        bar.update()

        parameters_to_table().to_csv(CFG.output_dir / "00_现象图绘图参数说明表.csv", index=False, encoding="utf-8-sig")
        bar.update()

        summary = pd.DataFrame(
            [
                {
                    "总站点数": int(sites_summary["Site_ID"].nunique()),
                    "多等级森林图模型记录数": int(len(forest_data)),
                    "五级SiteMonth有效配对数": int(len(valid_pairs)),
                    "有五级有效配对的站点数": int(valid_pairs["Site_ID"].nunique()),
                    "Mild有效配对数": int(valid_pairs["Target_Class"].astype(str).eq("Mild").sum()),
                    "Moderate有效配对数": int(valid_pairs["Target_Class"].astype(str).eq("Moderate").sum()),
                    "Severe有效配对数": int(valid_pairs["Target_Class"].astype(str).eq("Severe").sum()),
                    "Extreme有效配对数": int(valid_pairs["Target_Class"].astype(str).eq("Extreme").sum()),
                    "Delta_CBI均值": float(valid_pairs["Delta_CBI"].mean()),
                    "Delta_CBI中位数": float(valid_pairs["Delta_CBI"].median()),
                    "Delta_CBI为正的配对数": int((valid_pairs["Delta_CBI"] > 0).sum()),
                    "Delta_CBI为负的配对数": int((valid_pairs["Delta_CBI"] < 0).sum()),
                    "PNG图件": str(CFG.output_dir / "图1_现象图_主图.png"),
                    "PDF图件": str(CFG.output_dir / "图1_现象图_主图.pdf"),
                    "优化版PNG图件": str(CFG.output_dir / "图1_现象图_主图_优化版.png"),
                    "优化版PDF图件": str(CFG.output_dir / "图1_现象图_主图_优化版.pdf"),
                    "绝对CBI对比版PNG图件": str(CFG.output_dir / "图1_现象图_主图_绝对CBI对比版.png"),
                    "绝对CBI对比版PDF图件": str(CFG.output_dir / "图1_现象图_主图_绝对CBI对比版.pdf"),
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

    boundary, sites, hourly, pair_audit, lmm_summary, lmm_restricted, trend_test = read_inputs()
    sites_summary, forest_data, valid_pairs, site_response = prepare_data(
        sites, hourly, pair_audit, lmm_summary, lmm_restricted
    )

    if valid_pairs.empty:
        raise ValueError("No valid multi-level SiteMonth CBI pairs were found.")
    if site_response.empty:
        raise ValueError("No site-level CBI response summaries were found.")

    absolute_site_response = build_absolute_site_response(valid_pairs, sites_summary)

    with progress_bar("步骤3/5 绘制四个面板", 8, "yellow") as bar:
        delta_limit = finite_symmetric_limit(
            pd.concat([valid_pairs["Delta_CBI"], site_response["Median_Delta_CBI"]]),
            fallback=0.25,
        )
        delta_norm = TwoSlopeNorm(vmin=-delta_limit, vcenter=0, vmax=delta_limit)
        site_delta_limit = finite_symmetric_limit_with_padding(
            site_response["Median_Delta_CBI"],
            fallback=0.10,
            padding=FP.site_delta_limit_padding,
        )
        site_delta_norm = TwoSlopeNorm(vmin=-site_delta_limit, vcenter=0, vmax=site_delta_limit)
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

        draw_panel_a(ax_a, forest_data, trend_test)
        bar.update()

        draw_panel_b(ax_b, valid_pairs, delta_norm)
        bar.update()

        draw_panel_c(ax_c, boundary, sites_summary, site_response, site_delta_norm)
        bar.update()

        draw_panel_d(ax_d, site_response, site_delta_norm, site_delta_limit)
        bar.update()

        add_panel_c_colorbar(fig, ax_c, site_delta_norm)
        bar.update()

        fig.suptitle(
            "Spatial heterogeneity in drought-induced microclimate buffering responses",
            fontsize=FP.suptitle_size,
            fontweight="bold",
            y=FP.suptitle_y,
        )
        bar.update()

    with progress_bar("步骤3b/5 绘制优化版四个面板", 8, "cyan") as bar:
        fig_optimized = plt.figure(figsize=(FP.fig_width, FP.fig_height), constrained_layout=False)
        grid_opt = fig_optimized.add_gridspec(
            2,
            2,
            left=FP.grid_left,
            right=FP.grid_right,
            bottom=FP.grid_bottom,
            top=FP.grid_top,
            wspace=FP.grid_wspace,
            hspace=FP.grid_hspace,
        )
        ax_a_opt = fig_optimized.add_subplot(grid_opt[0, 0])
        ax_b_opt = fig_optimized.add_subplot(grid_opt[0, 1])
        ax_c_opt = fig_optimized.add_subplot(grid_opt[1, 0])
        ax_d_opt = fig_optimized.add_subplot(grid_opt[1, 1])
        bar.update()

        draw_panel_a_optimized(ax_a_opt, forest_data, trend_test)
        bar.update()

        draw_panel_b(ax_b_opt, valid_pairs, delta_norm)
        bar.update()

        draw_panel_c(ax_c_opt, boundary, sites_summary, site_response, site_delta_norm)
        bar.update()

        draw_panel_d_optimized(ax_d_opt, site_response, site_delta_norm, site_delta_limit)
        bar.update()

        add_panel_c_colorbar(fig_optimized, ax_c_opt, site_delta_norm)
        bar.update()

        fig_optimized.suptitle(
            "Spatial heterogeneity in drought-induced microclimate buffering responses",
            fontsize=FP.suptitle_size,
            fontweight="bold",
            y=FP.suptitle_y,
        )
        bar.update()

        fig_optimized.canvas.draw_idle()
        bar.update()

    with progress_bar("步骤3c/5 绘制绝对CBI对比版四个面板", 8, "green") as bar:
        absolute_values = pd.concat(
            [
                valid_pairs["Normal_CBI"],
                valid_pairs["Target_CBI"],
                absolute_site_response["Median_Normal_CBI"],
                absolute_site_response["Median_Drought_CBI"],
            ],
            ignore_index=True,
        )
        cbi_min, cbi_max = finite_value_limits(
            absolute_values,
            fallback=(0.3, 1.3),
            padding=FP.absolute_cbi_limit_padding,
        )
        cbi_norm = mpl.colors.Normalize(vmin=cbi_min, vmax=cbi_max)
        bar.update()

        fig_absolute = plt.figure(figsize=(FP.fig_width, FP.fig_height), constrained_layout=False)
        grid_abs = fig_absolute.add_gridspec(
            2,
            2,
            left=FP.grid_left,
            right=FP.grid_right,
            bottom=FP.grid_bottom,
            top=FP.grid_top,
            wspace=FP.grid_wspace,
            hspace=FP.grid_hspace,
        )
        ax_a_abs = fig_absolute.add_subplot(grid_abs[0, 0])
        ax_b_abs = fig_absolute.add_subplot(grid_abs[0, 1])
        ax_c_abs = fig_absolute.add_subplot(grid_abs[1, 0])
        ax_d_abs = fig_absolute.add_subplot(grid_abs[1, 1])
        bar.update()

        draw_panel_a_absolute(ax_a_abs, valid_pairs, forest_data)
        bar.update()

        draw_panel_b(ax_b_abs, valid_pairs, delta_norm)
        bar.update()

        draw_panel_c_absolute(ax_c_abs, boundary, sites_summary, absolute_site_response, cbi_norm)
        bar.update()

        draw_panel_d_absolute(ax_d_abs, absolute_site_response, cbi_norm)
        bar.update()

        add_absolute_cbi_colorbar(fig_absolute, ax_c_abs, cbi_norm)
        bar.update()

        fig_absolute.suptitle(
            "Absolute CBI contrasts between normal and drought conditions",
            fontsize=FP.suptitle_size,
            fontweight="bold",
            y=FP.suptitle_y,
        )
        bar.update()

    summary = write_outputs(
        fig,
        fig_optimized,
        fig_absolute,
        sites_summary,
        forest_data,
        valid_pairs,
        site_response,
        absolute_site_response,
    )
    plt.close(fig)
    plt.close(fig_optimized)
    plt.close(fig_absolute)

    cleanup_runtime_cache()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        make_figure()
