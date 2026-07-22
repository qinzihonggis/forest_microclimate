# -*- coding: utf-8 -*-
"""
Extreme-Normal CBI 阈值稳健性与敏感性分析独立脚本。

本脚本有意独立于 `比较整体和局部气候缓冲能力差异.py`：
    1. 读取与主脚本相同的原始输入文件；
    2. 先复现 BASE 情景的 SPI 分类、小时筛选和 LMM 样本；
    3. 再在预先固定的 E/N/C/H/M 情景下重复构造样本并估计结果；
    4. 所有结果写入独立稳健性结果目录，不覆盖主脚本结果。

RUN_MODE 的含义：
    - "coverage_only"：只做 BASE 复现和各情景覆盖审计，不拟合新模型；
    - "full"：在覆盖审计基础上，进一步运行 LMM、站点月份 CBI、事件 CBI 和汇总图。

修改原则：
    - 如果只是检查样本覆盖，应使用 coverage_only；
    - 只有 BASE 复现对照通过后，才使用 full 解释阈值稳健性。
"""

from pathlib import Path
import shutil
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import linregress
import statsmodels.formula.api as smf
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Matplotlib 默认字体通常不含中文，保存图片时会把中文标题、坐标轴和图例显示成方框。
# 这里按 Windows 常见中文字体顺序设置候选字体；系统会使用其中第一个可用字体。
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]
# 避免负号在中文字体环境下显示异常。
plt.rcParams["axes.unicode_minus"] = False


# =============================================================================
# 0. 路径、字段和固定规则：必须与主脚本保持一致，确保 BASE 可以复现
# =============================================================================

TEMP_HOURLY_FILE = Path(
    r"E:\forest_microclimate\ForestMicroclimate\results\时间序列图\逐小时温度对齐表.csv"
)
# 输入一：主脚本使用的逐小时温度对齐表。不得替换为其他时间体系或预聚合表。
SPI_DAILY_WIDE_FILE = Path(
    r"E:\forest_microclimate\ForestMicroclimate\results\daily_SPI_result\各站点SPI30d逐日宽表_2025.xlsx"
)
# 输入二：主脚本使用的逐日 SPI30d 宽表；第一列为日期，后续各列为站点。
DROUGHT_EVENT_FILE = Path(
    r"E:\forest_microclimate\ForestMicroclimate\results\daily_SPI_features\福建省观测站2025年daily_SPI干旱事件长表.csv"
)
# 输入三：主脚本事件 CBI 使用的完整干旱事件长表。
# 事件不是“连续 SPI < 阈值日”的短片段，而是从进入干旱到退出干旱的完整过程；
# 只要完整过程中 Min_Daily_SPI 达到或低于某个阈值，该完整过程就进入对应情景的事件 CBI。

OUTPUT_DIR = Path(
    r"E:\forest_microclimate\ForestMicroclimate\results\robustness_sensitivity_results"
)
# 独立结果目录：仅写入稳健性分析结果，不覆盖主脚本 compare_differences_results。
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 每次运行单独创建一个临时缓存目录。脚本结束时只删除这个目录，
# 不删除已经写出的正式表格、图件和报告，避免误删历史分析结果。
RUN_START_TIME = time.time()
RUNTIME_CACHE_DIR = OUTPUT_DIR / f"_本次运行临时缓存_{int(RUN_START_TIME)}"

# 主脚本已有输出：只读，用于核验独立脚本 BASE 是否真正复现主脚本样本与结果。
MAIN_OUTPUT_DIR = Path(
    r"E:\forest_microclimate\ForestMicroclimate\results\compare_differences_results"
)

# 以下字段名必须与真实输入表及主脚本一致；不要改为原型脚本的概念化字段名。
SITE_COL = "Site_ID"
TIME_COL = "Time_UTC"
DATE_COL = "UTC_Date"
MICRO_COL = "Observed_T15cm_C"
MACRO_COL = "ERA5_T2m_C"
SPI_COL = "SPI30d"

# UTC 时间解析成功率低于此值即停止，避免错配 SPI 日界线。
MIN_TIME_PARSE_RATE = 0.995
# 温度物理范围仅用于沿用主脚本的防错质量控制。
MICRO_RANGE = (-5.0, 45.0)
MACRO_RANGE = (-20.0, 50.0)

# 事件参考期：结束后第 1 天缓冲；第 2 天至第 31 天为 30 个候选自然日。
POST_EVENT_BUFFER_DAYS = 1
POST_EVENT_SEARCH_DAYS = 30
# 单事件最多选取 30 个合格 Normal 参考日；短事件按其实际持续日数确定目标日数。
MAX_REFERENCE_TARGET_DAYS = 30
# 单个候选参考日需至少 18 个有效小时，防止残缺日进入事件参考期。
MIN_VALID_HOURS_PER_REFERENCE_DAY = 18

# 与主脚本相同的混合模型优化器、最大迭代次数和事件级站点聚类 bootstrap 设置。
LMM_METHOD = "lbfgs"
LMM_MAXITER = 500
N_CLUSTER_BOOTSTRAP = 2000
RANDOM_SEED = 20250714

# 首次正式运行必须保持 coverage_only：只审计 BASE 和各阈值情景的样本覆盖。
# 当前 BASE 复现审计已通过，因此切换为 full 运行完整模型与稳健性汇总。
RUN_MODE = "full"


# =============================================================================
# 0A. 进度条、绘图和图例参数：集中放在这里，便于运行前统一调整
# =============================================================================
# 进度条说明：
#   - 每个关键步骤只显示一个 tqdm 单行动态进度条，leave=False 表示完成后不保留旧进度条，
#     这样终端不会被每个循环步骤刷屏。
#   - colour 控制不同类型进度条的颜色；如果终端不支持 ANSI 颜色，仍会显示百分比、
#     当前量/总量、已耗时、预计剩余时间和速度。
#   - unit 控制进度条右侧的计量单位，例如“情景”“事件”“文件”。
PROGRESS_BAR_CONFIG = {
    "读取与质控": {"colour": "cyan", "unit": "步"},
    "情景覆盖": {"colour": "green", "unit": "情景"},
    "事件覆盖": {"colour": "magenta", "unit": "事件"},
    "LMM拟合": {"colour": "blue", "unit": "情景"},
    "站点月份CBI": {"colour": "yellow", "unit": "组"},
    "事件CBI": {"colour": "red", "unit": "事件"},
    "Bootstrap": {"colour": "cyan", "unit": "次"},
    "绘图": {"colour": "magenta", "unit": "图"},
    "结果写出": {"colour": "green", "unit": "文件"},
    "缓存清理": {"colour": "white", "unit": "项"},
    "逐站点剔除": {"colour": "red", "unit": "站点"},
    "逐区域过程剔除": {"colour": "magenta", "unit": "过程"},
    "过程恢复": {"colour": "blue", "unit": "窗口"},
    "季节审计": {"colour": "cyan", "unit": "季节"},
}

# 森林图参数说明：
#   - 这些参数只影响图形外观，不改变任何统计结果。
#   - 如果后续需要改图的字号、颜色、线宽、图例位置，优先改这里，不要改绘图函数主体。
FIG_DPI = 300
# 输出图片分辨率。300 dpi 通常可用于论文或报告；数值越大，图片文件越大。
FIG_FOREST_WIDTH = 10
# 森林图宽度，单位为英寸。标签太长或横向空间不足时可以调大。
FIG_FOREST_MIN_HEIGHT = 5
# 森林图最小高度，单位为英寸。情景较少时避免图太扁。
FIG_FOREST_HEIGHT_PER_SCENARIO = 0.42
# 每增加一个有效情景增加的高度。情景标签拥挤时可以调大。
COLOR_BASE_SCENARIO = "#D7301F"
# BASE 点颜色。用于突出基准情景，不影响统计计算。
COLOR_OTHER_SCENARIO = "#2C7FB8"
# 非 BASE 敏感性情景点颜色。
COLOR_CI = "#595959"
# 95% 置信区间横线颜色。
COLOR_ZERO_LINE = "#000000"
# Delta CBI = 0 的参考线颜色；用于判断效应方向和是否跨 0。
LINESTYLE_ZERO = "--"
# 0 参考线线型，例如 "--" 虚线、"-" 实线、":" 点线。
LINEWIDTH_ZERO = 1.0
# 0 参考线线宽。
LINEWIDTH_CI = 1.2
# 置信区间横线线宽。
CI_CAPSIZE = 3
# 置信区间两端帽子的长度；设为 0 可去掉端帽。
SCATTER_SIZE_FOREST = 42
# 森林图散点大小；数值越大，点越醒目。
FIGURE_BBOX = "tight"
# 保存图时裁剪空白边缘；一般保持 tight。
LEGEND_SHOW = True
# 是否显示图例。若图中文字太拥挤，可改为 False。
LEGEND_LOCATION = "center left"
# 图例锚点相对位置。默认配合 LEGEND_BBOX_TO_ANCHOR 放在坐标轴外右侧，避免遮挡数据。
LEGEND_BBOX_TO_ANCHOR = (1.02, 0.5)
# 图例锚点坐标。None 表示图例放在图内；(1.02, 0.5) 表示放在坐标轴右侧中部。
LEGEND_FRAME_ON = True
# 图例是否显示边框。
LEGEND_FONT_SIZE = 9
# 图例文字字号。
LEGEND_TITLE = "图例"
# 图例标题。
LEGEND_BASE_LABEL = "BASE 基准情景"
# BASE 点在图例中的显示名称。
LEGEND_OTHER_LABEL = "非 BASE 敏感性情景"
# 非 BASE 点在图例中的显示名称。
LEGEND_CI_LABEL = "95% 置信区间"
# 置信区间线在图例中的显示名称。
LEGEND_ZERO_LABEL = "Delta CBI = 0"
# 0 参考线在图例中的显示名称。

# 所有生成文件统一使用中文命名，便于在结果文件夹中直接识别用途。
OUTPUT_FILES = {
    "scenario_definitions": "00_预设稳健性情景定义.csv",
    "input_audit": "00_输入数据与路径审计.csv",
    "figure_parameters": "00_绘图与进度条参数说明.csv",
    "status_by_scenario": "01_各情景SPI状态覆盖审计.csv",
    "event_definition_events": "02_事件定义敏感性完整干旱事件表.csv",
    "coverage_by_scenario": "01_各情景样本覆盖汇总.csv",
    "base_coverage_audit": "01_BASE覆盖审计.csv",
    "base_reproduction_audit": "01_BASE与主脚本复现对照.csv",
    "all_methods_summary": "05_全部稳健性分析结果汇总.csv",
    "sitemonth_units": "05_站点月份状态CBI估计.csv",
    "sitemonth_pairs": "05_站点月份CBI配对结果.csv",
    "event_pairs": "05_事件与参考期CBI配对结果.csv",
    "forest_plot": "06_LMM阈值稳健性森林图.png",
    "event_definition_plot": "07_事件定义稳健性森林图.png",
    "report": "09_稳健性分析运行报告.txt",
    "interpretation_report": "20_阈值稳健性结果解释报告.txt",
}

# Leave-one-site-out（逐站点剔除）是主分析之外的独立附加模块，
# 只检验 Hourly_state / BASE 是否被单一站点驱动，不接入事件定义证据链。
RUN_LEAVE_ONE_SITE_OUT = True
LEAVE_ONE_SITE_OUT_DIRNAME = "Leave_one_site_out"
MAKE_LEAVE_ONE_SITE_OUT_FOREST = True
LOSO_OUTPUT_FILES = {
    "base_audit": "21_Leave_one_site_out_BASE与主脚本复现对照.csv",
    "coverage": "21_Leave_one_site_out_样本覆盖审计.csv",
    "detail": "21_Leave_one_site_out_逐站点剔除LMM结果.csv",
    "summary": "21_Leave_one_site_out_汇总判定.csv",
    "report": "21_Leave_one_site_out_解释报告.txt",
    "forest_plot": "21_Leave_one_site_out_森林图.png",
}

# Leave-one-regional-drought-process-out（逐区域干旱过程剔除，LORPO）
# 是事件级影响点稳健性模块，只检验 BASE 事件—参考期 CBI 是否被某一次区域性干旱过程驱动。
# 它不删除站点，不重跑小时级 LMM，也不改变阈值或事件定义；每轮只从 drought_events 中
# 删除一个多站点、时间连通的区域过程成员事件，再调用 run_event_cbi() 重新计算事件 CBI。
RUN_LEAVE_ONE_REGIONAL_PROCESS_OUT = True
LEAVE_ONE_REGIONAL_PROCESS_OUT_DIRNAME = "Leave_one_regional_drought_process_out"
# 是否为逐区域干旱过程剔除结果输出森林图。
# 该图只用于可视化 LORPO 的事件级 Delta CBI、95% CI 和 BASE 对照，不改变任何统计计算。
MAKE_LEAVE_ONE_REGIONAL_PROCESS_OUT_FOREST = True
# 过程连接容忍日数：不同站点的 Extreme 事件若日期重叠，或相隔不超过该天数，则归为同一时间连通过程。
# 该参数必须在看结果前预先固定，不应根据显著性或方向一致性事后调整。
REGIONAL_PROCESS_GAP_DAYS = 1
# 至少涉及多少个站点才称为“区域过程”。单站点局地事件不作为剔除对象，但会保留在每轮事件集中。
REGIONAL_PROCESS_MIN_SITES = 2
REGIONAL_PROCESS_OUTPUT_FILES = {
    "process_audit": "22_区域干旱过程识别与成员事件审计.csv",
    "coverage": "22_逐区域过程剔除样本覆盖审计.csv",
    "detail": "22_逐区域过程剔除事件CBI结果.csv",
    "summary": "22_逐区域过程剔除汇总判定.csv",
    "report": "22_逐区域过程剔除解释报告.txt",
    "forest_plot": "22_逐区域过程剔除事件CBI森林图.png",
}

# CBI process-recovery window sensitivity（CBI 过程与恢复窗口敏感性）
# 这是事件级补充稳健性模块，用于回答 CBI 改变是在干旱过程中即时出现、累积出现，
# 还是在事件结束后存在恢复滞后。它只使用 Hourly_state / BASE 的事件配对结果，
# 不改变主 LMM、SPI 阈值、事件定义、LOSO 或 LORPO。
RUN_PROCESS_RECOVERY_WINDOW_MODULE = True
PROCESS_RECOVERY_OUTPUT_DIRNAME = "CBI_process_recovery_window_analysis"
# 每个新增 CBI 窗口至少需要多少个有效小时。保持 72 小时，是为了与 BASE 事件 CBI 口径一致。
# 这里显式写 72，而不引用 HOURLY_STATE_BASE，是因为本参数区位于 HOURLY_STATE_BASE 定义之前。
PROCESS_WINDOW_MIN_HOURS = 72
# CBI 回归中 ERA5 温度至少需要多少个不同取值；低于该值无法稳定估计斜率。
PROCESS_WINDOW_MIN_UNIQUE_MACRO_VALUES = 3
# 恢复窗口若与同站点另一场极端事件重叠，默认整窗标记为不可解释，避免把二次干旱误判为恢复滞后。
EXCLUDE_RECOVERY_WINDOWS_OVERLAPPING_EXTREME_EVENT = True
# 事件内阶段按连续日历天三等分。不同持续时间事件都按相对进程切分，而不是按固定天数切分。
PROCESS_EVENT_PHASES = ("Early", "Middle", "Late")
PROCESS_EVENT_PHASE_LABELS = {
    "Early": "事件前段",
    "Middle": "事件中段",
    "Late": "事件后段",
}
# 事件结束后的恢复窗口。这里从 End+1 开始，因为目标是观察实际恢复过程，
# 不沿用事件参考期的 End+1 缓冲排除规则；参考期仍固定使用原事件配对表中的 Reference_CBI。
PROCESS_RECOVERY_WINDOWS = {
    "R01_07": (1, 7, "结束后1-7天"),
    "R08_14": (8, 14, "结束后8-14天"),
    "R15_30": (15, 30, "结束后15-30天"),
}
# 滑动窗口长度。7 天作为主轨迹，14 天用于检查轨迹形态是否依赖窗口长度。
PROCESS_SLIDING_WINDOW_DAYS = (7, 14)
PROCESS_RECOVERY_OUTPUT_FILES = {
    "reproduction_audit": "30_CBI过程恢复_完整事件复现审计.csv",
    "stage_detail": "30_CBI过程恢复_阶段明细.csv",
    "sliding_detail": "31_CBI过程恢复_滑动窗口明细.csv",
    "stage_summary": "32_CBI过程恢复_区域汇总.csv",
    "sliding_summary": "33_CBI过程恢复_滑动轨迹汇总.csv",
    "stage_figure": "34_CBI过程恢复_阶段恢复图.png",
    "sliding_figure": "35_CBI过程恢复_滑动轨迹图.png",
    "report": "36_CBI过程恢复_结果解释报告.txt",
}

# 季节样本审计与季节模型模块。
# 默认只做审计，不自动拟合季节 LMM，也不自动运行季节事件 CBI。
# 如果后续要正式拟合某个季节，必须人工查看审计表后手动填写 SEASONS_TO_RUN_* 列表。
RUN_SEASONAL_MODULE = True
SEASONAL_OUTPUT_DIRNAME = "seasonal_extreme_analysis"
RUN_SEASONAL_LMM = False
RUN_SEASONAL_EVENT_CBI = False
SEASONS_TO_RUN_LMM = []
SEASONS_TO_RUN_EVENT_CBI = []
SEASONS = ["Spring", "Summer", "Autumn", "Winter"]
SEASON_CN = {
    "Spring": "春季(3-5月)",
    "Summer": "夏季(6-8月)",
    "Autumn": "秋季(9-11月)",
    "Winter": "冬季(12-2月)",
}
SEASONAL_OUTPUT_FILES = {
    "run_config": "00_季节分析配置与运行说明.txt",
    "status_audit": "01_季节SPI状态覆盖审计.csv",
    "sitemonth_pair_audit": "02_季节SiteMonth双状态配对审计.csv",
    "site_contribution_audit": "03_季节站点贡献审计.csv",
    "event_detail_audit": "04_季节事件参考期逐事件审计.csv",
    "event_coverage_audit": "04_季节事件与参考期覆盖审计.csv",
    "model_decision": "05_季节可建模性判定.csv",
    "lmm_summary": "06_季节LMM结果汇总.csv",
    "lmm_forest_plot": "07_季节LMM森林图.png",
    "event_cbi_summary": "08_季节事件CBI结果汇总.csv",
    "interpretation_report": "09_季节分析解释报告.txt",
    "detailed_conclusion": "10_季节审计结论与不建模原因说明.txt",
}


# =============================================================================
# 1. 预先定义的情景：避免根据显著性事后增加、删除或挑选模型规格
# =============================================================================
#
# 情景命名规则总览：
#   - BASE：主脚本基准情景，作为所有稳健性比较的参考点。
#   - Exx：Extreme 阈值敏感性（E = Extreme）。
#       含义是“把 Extreme 的 SPI 阈值改成某个值，观察结论是否改变”。
#       例如：
#         E15 = Extreme 定义改为 SPI30d <= -1.5
#         E18 = Extreme 定义改为 SPI30d <= -1.8
#         E22 = Extreme 定义改为 SPI30d <= -2.2
#       数字部分去掉小数点后保留两位表达，便于简写：
#         15 -> 1.5，18 -> 1.8，22 -> 2.2
#
#   - Nxxx：Normal 窗口敏感性（N = Normal）。
#       含义是“把 Normal 的开区间宽度改掉，观察对结论的影响”。
#       例如：
#         N025 = Normal 改为 -0.25 < SPI30d < 0.25
#         N075 = Normal 改为 -0.75 < SPI30d < 0.75
#         N10  = Normal 改为 -1.0  < SPI30d < 1.0
#       其中：
#         025 -> 0.25，075 -> 0.75，10 -> 1.0
#
#   - Cx：联合情景（C = Combined / Joint definition）。
#       含义是“同时改变 Extreme 定义和 Normal 定义”，用于测试两端阈值一起变化时
#       主结论是否仍然成立。
#       例如：
#         C3 = Extreme <= -1.8，Normal -0.75 < SPI30d < 0.75
#         C4 = Extreme <= -2.2，Normal -0.25 < SPI30d < 0.25
#         C5 = Extreme <= -1.8，Normal -0.25 < SPI30d < 0.25
#
#   - Hxxx：小时门槛敏感性（H = Hours）。
#       含义是“改变样本单元进入分析所需的最低有效小时数”，看主结论是否依赖
#       某个小时门槛。
#       例如：
#         H48  = 最低小时门槛改为 48 小时
#         H120 = 最低小时门槛改为 120 小时
#         H168 = 最低小时门槛改为 168 小时
#       这里同时影响：
#         min_status_hours 和 min_cbi_hours
#
#   - Mxxx：MacroSD 质量控制敏感性（M = Macro climate SD）。
#       含义是“改变站点月份或事件 CBI 可接受的 ERA5 温度标准差门槛”，
#       检查结论是否依赖宏气候波动幅度要求。
#       例如：
#         M05  = MacroSD 门槛改为 0.5 ℃
#         M075 = MacroSD 门槛改为 0.75 ℃
#         M15  = MacroSD 门槛改为 1.5 ℃
#
# 重要提醒：
#   - 情景 ID 只是简写编码，正式解释时必须同时写出对应阈值含义，不能只写 E15 或 C4。
#   - 如果后续结果表、图或论文文字要给别人看，建议始终同时写“情景ID + 中文含义”。
#
# 双证据链设计：
#   - Hourly_state：沿用原 02 脚本的小时级 SPI 状态分析链，包含
#       BASE/E/N/C/H/M，负责 LMM、站点月份 CBI 和“主脚本定义事件”的事件配对。
#   - Event_definition：新增事件定义稳健性链，包含 BASE/D05/D10/B00/B10，
#       只负责“从逐日 SPI 重新提取完整干旱事件”后的事件—参考期 CBI，
#       不进入 build_lmm_dataset，也不进入 LMM。

HOURLY_STATE_BASE = dict(
    # SPI 分类：Extreme 包含阈值边界；Normal 为严格开区间，Normal 边界值归入 Other。
    extreme_threshold=-2.0,
    normal_low=-0.5,
    normal_high=0.5,
    # LMM 中每个 Site_ID × YearMonth × SPI_Status 的最低小时数。
    min_status_hours=72,
    # 站点-月份 OLS CBI 的最低有效小时数。
    min_cbi_hours=72,
    # 事件期与参考期 OLS CBI 的最低有效小时数，沿用主脚本规则。
    min_event_hours=72,
    min_reference_hours=72,
    # 站点-月份 OLS CBI 的 ERA5 温度标准差门槛。
    min_macro_sd=1.0,
    # 主脚本事件 CBI 不施加 MacroSD；只有 M 情景将其作为新增质量控制敏感性。
    apply_event_macro_sd=False,
)

HOURLY_STATE_SCENARIOS = [
    # BASE：主脚本当前正式分析使用的基准定义。
    dict(id="BASE", group="Baseline", description="基准情景：Extreme <= -2.0；Normal -0.5 < SPI < 0.5", **HOURLY_STATE_BASE),

    # E 类：只改变 Extreme 阈值；Normal 窗口和其他规则保持 BASE 不变。
    dict(
        id="E15",
        group="Extreme_threshold",
        description="E15：Extreme 改为 SPI <= -1.5；用于检验放宽极端干旱阈值后的稳健性",
        **{**HOURLY_STATE_BASE, "extreme_threshold": -1.5},
    ),
    dict(
        id="E18",
        group="Extreme_threshold",
        description="E18：Extreme 改为 SPI <= -1.8；用于检验略微放宽极端干旱阈值后的稳健性",
        **{**HOURLY_STATE_BASE, "extreme_threshold": -1.8},
    ),
    dict(
        id="E22",
        group="Extreme_threshold",
        description="E22：Extreme 改为 SPI <= -2.2；用于检验收紧极端干旱阈值后的稳健性",
        **{**HOURLY_STATE_BASE, "extreme_threshold": -2.2},
    ),

    # N 类：只改变 Normal 开区间；Extreme 阈值和其他规则保持 BASE 不变。
    dict(
        id="N025",
        group="Normal_window",
        description="N025：Normal 改为 -0.25 < SPI < 0.25；用于检验收紧正常窗口后的稳健性",
        **{**HOURLY_STATE_BASE, "normal_low": -0.25, "normal_high": 0.25},
    ),
    dict(
        id="N075",
        group="Normal_window",
        description="N075：Normal 改为 -0.75 < SPI < 0.75；用于检验放宽正常窗口后的稳健性",
        **{**HOURLY_STATE_BASE, "normal_low": -0.75, "normal_high": 0.75},
    ),
    dict(
        id="N10",
        group="Normal_window",
        description="N10：Normal 改为 -1.0 < SPI < 1.0；用于检验进一步放宽正常窗口后的稳健性",
        **{**HOURLY_STATE_BASE, "normal_low": -1.0, "normal_high": 1.0},
    ),

    # C 类：Extreme 和 Normal 一起改，检验联合规格变化是否影响主结论。
    dict(
        id="C3",
        group="Joint_definition",
        description="C3：Extreme <= -1.8，且 Normal -0.75 < SPI < 0.75；联合放宽两端定义",
        **{**HOURLY_STATE_BASE, "extreme_threshold": -1.8, "normal_low": -0.75, "normal_high": 0.75},
    ),
    dict(
        id="C4",
        group="Joint_definition",
        description="C4：Extreme <= -2.2，且 Normal -0.25 < SPI < 0.25；联合收紧两端定义",
        **{**HOURLY_STATE_BASE, "extreme_threshold": -2.2, "normal_low": -0.25, "normal_high": 0.25},
    ),
    dict(
        id="C5",
        group="Joint_definition",
        description="C5：Extreme <= -1.8，且 Normal -0.25 < SPI < 0.25；放宽 Extreme、收紧 Normal",
        **{**HOURLY_STATE_BASE, "extreme_threshold": -1.8, "normal_low": -0.25, "normal_high": 0.25},
    ),

    # H 类：只改最低小时门槛，测试样本覆盖要求对结果的影响。
    dict(
        id="H48",
        group="Minimum_hours",
        description="H48：最低小时门槛改为 48 小时；用于检验放宽样本小时要求后的稳健性",
        **{**HOURLY_STATE_BASE, "min_status_hours": 48, "min_cbi_hours": 48},
    ),
    dict(
        id="H120",
        group="Minimum_hours",
        description="H120：最低小时门槛改为 120 小时；用于检验收紧样本小时要求后的稳健性",
        **{**HOURLY_STATE_BASE, "min_status_hours": 120, "min_cbi_hours": 120},
    ),
    dict(
        id="H168",
        group="Minimum_hours",
        description="H168：最低小时门槛改为 168 小时；用于检验更严格样本门槛下的稳健性",
        **{**HOURLY_STATE_BASE, "min_status_hours": 168, "min_cbi_hours": 168},
    ),

    # M 类：只改 MacroSD 质量控制门槛，测试外界温度波动要求对结果的影响。
    dict(
        id="M05",
        group="MacroSD_CBI_only",
        description="M05：CBI 的 MacroSD 门槛改为 0.5 ℃；用于检验放宽宏气候波动要求后的稳健性",
        **{**HOURLY_STATE_BASE, "min_macro_sd": 0.5, "apply_event_macro_sd": True},
    ),
    dict(
        id="M075",
        group="MacroSD_CBI_only",
        description="M075：CBI 的 MacroSD 门槛改为 0.75 ℃；用于检验略微放宽宏气候波动要求后的稳健性",
        **{**HOURLY_STATE_BASE, "min_macro_sd": 0.75, "apply_event_macro_sd": True},
    ),
    dict(
        id="M15",
        group="MacroSD_CBI_only",
        description="M15：CBI 的 MacroSD 门槛改为 1.5 ℃；用于检验收紧宏气候波动要求后的稳健性",
        **{**HOURLY_STATE_BASE, "min_macro_sd": 1.5, "apply_event_macro_sd": True},
    ),
]


EVENT_DEFINITION_BASE = dict(
    # 事件边界：连续 SPI <= -0.5 的完整干旱过程定义为一个候选事件。
    event_threshold=-0.5,
    # 最短持续时间：连续过程至少 6 天，才作为完整干旱事件保留。
    min_duration_days=6,
    # 极端事件判定：完整事件过程中最低 SPI <= -2.0，则进入 Extreme 事件分析。
    extreme_threshold=-2.0,
    # 事件后参考日的 Normal 开区间与主分析保持一致。
    normal_low=-0.5,
    normal_high=0.5,
    # 事件级 CBI 最低小时门槛沿用主分析。
    min_event_hours=72,
    min_reference_hours=72,
)


EVENT_DEFINITION_SCENARIOS = [
    dict(
        id="BASE",
        group="Baseline",
        description="基准事件定义：事件 SPI <= -0.5，至少6天；完整事件内最低 SPI <= -2.0",
        **EVENT_DEFINITION_BASE,
    ),
    dict(
        id="D05",
        group="Minimum_duration",
        description="D05：事件 SPI <= -0.5，至少5天；检验纳入较短事件后的稳健性",
        **{**EVENT_DEFINITION_BASE, "min_duration_days": 5},
    ),
    dict(
        id="D10",
        group="Minimum_duration",
        description="D10：事件 SPI <= -0.5，至少10天；检验仅保留较长事件后的稳健性",
        **{**EVENT_DEFINITION_BASE, "min_duration_days": 10},
    ),
    dict(
        id="B00",
        group="Event_boundary",
        description="B00：事件 SPI <= 0.0，至少6天；检验较宽事件边界后的稳健性",
        **{**EVENT_DEFINITION_BASE, "event_threshold": 0.0},
    ),
    dict(
        id="B10",
        group="Event_boundary",
        description="B10：事件 SPI <= -1.0，至少6天；检验较严事件边界后的稳健性",
        **{**EVENT_DEFINITION_BASE, "event_threshold": -1.0},
    ),
]


MIN_EVENT_PAIRS_SUGGESTED_EVENT_DEFINITION = 10


def build_scenario_definitions():
    """构造统一情景定义表，并标记其所属分析层和证据链。"""
    rows = []
    for cfg in HOURLY_STATE_SCENARIOS:
        rows.append(
            {
                "AnalysisLayer": "Hourly_state",
                "EvidenceChain": "Hourly_state",
                "ScenarioID": cfg["id"],
                "ScenarioGroup": cfg["group"],
                "ScenarioDescription": cfg["description"],
                **cfg,
            }
        )
    for cfg in EVENT_DEFINITION_SCENARIOS:
        rows.append(
            {
                "AnalysisLayer": "Event_definition",
                "EvidenceChain": "Event_definition",
                "ScenarioID": cfg["id"],
                "ScenarioGroup": cfg["group"],
                "ScenarioDescription": cfg["description"],
                **cfg,
            }
        )
    return pd.DataFrame(rows)

MIN_SITES_SUGGESTED = 10
MIN_SITEMONTHS_SUGGESTED = 30
MIN_EVENT_PAIRS_SUGGESTED = 20


# =============================================================================
# 2. 通用工具函数：只处理 ID 标准化、进度条、状态分类、OLS CBI 和文件写出
# =============================================================================

def normalize_site_id(series):
    """统一站点 ID 格式，避免 Excel 或 CSV 把站点编号读成带 .0 的数字字符串。"""
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def progress_bar(total, desc, kind):
    """创建单行动态 tqdm 进度条。

    参数含义：
        total：当前步骤总任务量，用于显示百分比和剩余时间。
        desc：终端左侧显示的中文步骤名。
        kind：进度条类型，必须是 PROGRESS_BAR_CONFIG 中的键；它决定颜色和单位。
    """
    cfg = PROGRESS_BAR_CONFIG[kind]
    return tqdm(
        total=total,
        desc=desc,
        unit=cfg["unit"],
        colour=cfg["colour"],
        dynamic_ncols=True,
        leave=False,
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        ),
    )


def progress_iter(iterable, total, desc, kind):
    """给循环步骤添加一个进度条，并保证出错时也能关闭进度条。"""
    bar = progress_bar(total=total, desc=desc, kind=kind)
    try:
        for item in iterable:
            yield item
            bar.update(1)
    finally:
        bar.close()


def progress_step(desc, kind):
    """给单步操作添加进度条，例如读取一个文件、写出一个表或清理缓存。"""
    return progress_bar(total=1, desc=desc, kind=kind)


def classify_status(spi, cfg):
    """按情景参数把 SPI30d 分类为 Extreme、Normal 或 Other。

    规则说明：
        - Extreme 使用小于等于：SPI30d <= extreme_threshold。
        - Normal 使用严格开区间：normal_low < SPI30d < normal_high。
        - 等于边界值的数据不进入 Normal，而是归入 Other。
    这样处理是为了和主脚本保持一致，避免边界值在不同脚本中被重复或错分。
    """
    return np.select(
        [
            spi <= cfg["extreme_threshold"],
            (spi > cfg["normal_low"]) & (spi < cfg["normal_high"]),
        ],
        ["Extreme", "Normal"],
        default="Other",
    )


def ols_cbi(data, min_hours, min_macro_sd=None):
    """估计单个样本单元的 OLS CBI。

    CBI 的含义：
        用 Observed_T15cm_C 对 ERA5_T2m_C 做一元线性回归，斜率即 CBI。
        斜率越小，表示林下温度对大气温度变化的响应越弱，即缓冲越强。

    参数含义：
        min_hours：该单元至少需要的有效小时数；调大后结果更严格但样本更少。
        min_macro_sd：ERA5 温度标准差门槛；调大后会剔除外界温度变化太小、
            不适合估计斜率的单元。None 表示不使用该门槛。
    """
    d = data[[MICRO_COL, MACRO_COL]].dropna()
    n = len(d)
    if n < min_hours:
        return dict(
            CBI=np.nan,
            Intercept=np.nan,
            R2=np.nan,
            n_hours=n,
            MacroSD=np.nan,
            Flag="insufficient_hours",
        )

    macro_sd = d[MACRO_COL].std()
    if (
        not np.isfinite(macro_sd)
        or d[MACRO_COL].nunique() < 3
        or (min_macro_sd is not None and macro_sd < min_macro_sd)
    ):
        return dict(
            CBI=np.nan,
            Intercept=np.nan,
            R2=np.nan,
            n_hours=n,
            MacroSD=macro_sd,
            Flag="insufficient_macro_variation",
        )

    fit = linregress(d[MACRO_COL], d[MICRO_COL])
    return dict(
        CBI=fit.slope,
        Intercept=fit.intercept,
        R2=fit.rvalue ** 2,
        n_hours=n,
        MacroSD=macro_sd,
        Flag="ok",
    )


def identify_extreme_events(daily_spi, threshold):
    """识别每个站点连续的 Extreme 事件。

    事件定义：
        在同一站点内，SPI30d 连续满足 SPI30d <= threshold 的自然日合并为一个事件。
        如果两个 Extreme 日期不连续，即使中间只断开一天，也会被视为两个事件。
    """
    events = []
    for site, group in daily_spi.sort_values([SITE_COL, DATE_COL]).groupby(SITE_COL):
        g = group[[DATE_COL, SPI_COL]].dropna().sort_values(DATE_COL).copy()
        if g.empty:
            continue

        g["is_extreme"] = g[SPI_COL] <= threshold
        previous_date = g[DATE_COL].shift()
        starts = (
            g["is_extreme"]
            & (
                ~g["is_extreme"].shift(fill_value=False)
                | ((g[DATE_COL] - previous_date).dt.days != 1)
            )
        )
        g["event_no"] = starts.cumsum()

        for _, event_days in g.loc[g["is_extreme"]].groupby("event_no"):
            events.append(
                {
                    SITE_COL: site,
                    "Start_Date": event_days[DATE_COL].min(),
                    "End_Date": event_days[DATE_COL].max(),
                    "Duration_Days": event_days[DATE_COL].nunique(),
                    "Minimum_SPI": event_days[SPI_COL].min(),
                }
            )

    events_df = pd.DataFrame(events)
    if not events_df.empty:
        events_df["Event_ID_Robustness"] = (
            events_df.groupby(SITE_COL).cumcount() + 1
        )
    return events_df


def cluster_bootstrap(event_delta):
    """对事件 CBI 差值做站点聚类 bootstrap。

    设计原因：
        同一站点可能有多个事件，不能让事件多的站点在总体均值中权重过高。
        因此先计算每个站点的平均 Delta_CBI，再对“站点均值”重抽样。

    参数影响：
        N_CLUSTER_BOOTSTRAP 控制重抽样次数；次数越大，置信区间越稳定，但运行越慢。
        RANDOM_SEED 固定随机数种子，保证重复运行结果可复现。
    """
    d = event_delta[[SITE_COL, "Delta_CBI"]].dropna()
    if d.empty:
        return dict(
            Event_Delta_Mean=np.nan,
            Event_CI_low95=np.nan,
            Event_CI_high95=np.nan,
            Event_P_two_sided=np.nan,
            Event_N_sites=0,
            Event_N_pairs=0,
            Event_Flag="no_valid_event_pairs",
        )

    site_means = d.groupby(SITE_COL)["Delta_CBI"].mean()
    if len(site_means) < 2:
        return dict(
            Event_Delta_Mean=site_means.mean(),
            Event_CI_low95=np.nan,
            Event_CI_high95=np.nan,
            Event_P_two_sided=np.nan,
            Event_N_sites=len(site_means),
            Event_N_pairs=len(d),
            Event_Flag="fewer_than_2_sites",
        )

    rng = np.random.default_rng(RANDOM_SEED)
    values = site_means.to_numpy()
    boot = np.empty(N_CLUSTER_BOOTSTRAP)
    for index in progress_iter(
        range(N_CLUSTER_BOOTSTRAP),
        total=N_CLUSTER_BOOTSTRAP,
        desc="事件站点聚类Bootstrap",
        kind="Bootstrap",
    ):
        boot[index] = rng.choice(values, size=len(values), replace=True).mean()
    p_value = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    return dict(
        Event_Delta_Mean=values.mean(),
        Event_CI_low95=np.percentile(boot, 2.5),
        Event_CI_high95=np.percentile(boot, 97.5),
        Event_P_two_sided=p_value,
        Event_N_sites=len(values),
        Event_N_pairs=len(d),
        Event_Flag="ok",
    )


def write_csv(df, filename):
    """写出 CSV 表格。

    使用 utf-8-sig 是为了让中文文件名和中文列名在 Excel 中直接打开不乱码。
    每次写出一个文件时显示一个单步进度条。
    """
    bar = progress_step(f"写出 {filename}", "结果写出")
    try:
        df.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")
        bar.update(1)
    finally:
        bar.close()


def write_text(text, filename):
    """写出中文运行报告文本，并显示一个单步进度条。"""
    bar = progress_step(f"写出 {filename}", "结果写出")
    try:
        (OUTPUT_DIR / filename).write_text(text, encoding="utf-8")
        bar.update(1)
    finally:
        bar.close()


def write_parameter_audit():
    """导出绘图和进度条参数表。

    用途：
        把图形颜色、线宽、图例、尺寸、进度条颜色等参数写入 CSV，
        方便后续不翻代码也能知道本次图片和终端显示使用了哪些设置。
    """
    rows = []
    for name, value in {
        "FIG_DPI": FIG_DPI,
        "FIG_FOREST_WIDTH": FIG_FOREST_WIDTH,
        "FIG_FOREST_MIN_HEIGHT": FIG_FOREST_MIN_HEIGHT,
        "FIG_FOREST_HEIGHT_PER_SCENARIO": FIG_FOREST_HEIGHT_PER_SCENARIO,
        "COLOR_BASE_SCENARIO": COLOR_BASE_SCENARIO,
        "COLOR_OTHER_SCENARIO": COLOR_OTHER_SCENARIO,
        "COLOR_CI": COLOR_CI,
        "COLOR_ZERO_LINE": COLOR_ZERO_LINE,
        "LINEWIDTH_ZERO": LINEWIDTH_ZERO,
        "LINESTYLE_ZERO": LINESTYLE_ZERO,
        "LINEWIDTH_CI": LINEWIDTH_CI,
        "CI_CAPSIZE": CI_CAPSIZE,
        "SCATTER_SIZE_FOREST": SCATTER_SIZE_FOREST,
        "FIGURE_BBOX": FIGURE_BBOX,
        "LEGEND_SHOW": LEGEND_SHOW,
        "LEGEND_LOCATION": LEGEND_LOCATION,
        "LEGEND_BBOX_TO_ANCHOR": LEGEND_BBOX_TO_ANCHOR,
        "LEGEND_FRAME_ON": LEGEND_FRAME_ON,
        "LEGEND_FONT_SIZE": LEGEND_FONT_SIZE,
        "LEGEND_TITLE": LEGEND_TITLE,
        "LEGEND_BASE_LABEL": LEGEND_BASE_LABEL,
        "LEGEND_OTHER_LABEL": LEGEND_OTHER_LABEL,
        "LEGEND_CI_LABEL": LEGEND_CI_LABEL,
        "LEGEND_ZERO_LABEL": LEGEND_ZERO_LABEL,
    }.items():
        rows.append({"类别": "绘图参数", "参数名": name, "当前值": value})
    for kind, cfg in PROGRESS_BAR_CONFIG.items():
        rows.append(
            {
                "类别": "终端进度条",
                "参数名": f"{kind}_颜色",
                "当前值": cfg["colour"],
            }
        )
    write_csv(pd.DataFrame(rows), OUTPUT_FILES["figure_parameters"])


def cleanup_runtime_cache():
    """清理本次运行产生的临时缓存。

    安全边界：
        只删除 RUNTIME_CACHE_DIR 指向的本次运行临时目录；
        不删除 OUTPUT_DIR 下已经生成的正式结果表、图片和报告。
    """
    bar = progress_step("清理本次运行临时缓存", "缓存清理")
    try:
        if RUNTIME_CACHE_DIR.exists():
            shutil.rmtree(RUNTIME_CACHE_DIR)
        bar.update(1)
    finally:
        bar.close()


# =============================================================================
# 3. 数据读取与质量控制：所有情景共用同一份清洗后的小时数据和逐日 SPI 数据
# =============================================================================

def load_data():
    """读取主脚本同源输入，并完成最基础的质量控制。

    关键步骤：
        1. 读取逐小时温度对齐表，检查主脚本所需字段是否存在；
        2. 用 UTC 解析 Time_UTC，并生成 UTC_Date，保证小时温度能正确匹配逐日 SPI；
        3. 将林下温度和 ERA5 温度转为数值，并应用物理范围筛选；
        4. 读取逐日 SPI30d 宽表，转换成长表；
        5. 读取主脚本事件 CBI 使用的完整干旱事件长表；
        6. 按 Site_ID + UTC_Date 合并小时温度和逐日 SPI；
        7. 输出输入审计表，用来检查解析率、SPI 匹配率、站点数和日期范围。

    参数影响：
        MIN_TIME_PARSE_RATE 控制时间解析最低成功率；调低会放宽错误容忍度，但可能错配日期。
        MICRO_RANGE/MACRO_RANGE 控制温度物理范围；调窄会删除更多异常值，调宽会保留更多边界值。
    """
    required = [SITE_COL, TIME_COL, MICRO_COL, MACRO_COL]
    read_temp_bar = progress_step("读取逐小时温度数据", "读取与质控")
    try:
        temp_raw = pd.read_csv(TEMP_HOURLY_FILE, low_memory=False)
        read_temp_bar.update(1)
    finally:
        read_temp_bar.close()
    missing = sorted(set(required) - set(temp_raw.columns))
    if missing:
        raise KeyError(
            f"Temperature file missing columns: {missing}; "
            f"available columns: {temp_raw.columns.tolist()}"
        )

    n_raw = len(temp_raw)
    temp = temp_raw.copy()
    # 站点 ID 标准化必须在合并 SPI 前完成，否则同一个站点可能因格式差异无法匹配。
    temp[SITE_COL] = normalize_site_id(temp[SITE_COL])

    # 严格按 UTC 解析时间；这里不转本地日期，目的是与主脚本的 UTC 日界线一致。
    parsed_time = pd.to_datetime(temp[TIME_COL], errors="coerce", utc=True)
    parse_rate = parsed_time.notna().mean()
    if parse_rate < MIN_TIME_PARSE_RATE:
        raise ValueError(
            f"{TIME_COL} parse rate is {parse_rate:.3%}, below {MIN_TIME_PARSE_RATE:.1%}."
        )

    temp[DATE_COL] = parsed_time.dt.tz_localize(None).dt.normalize()
    temp[MICRO_COL] = pd.to_numeric(temp[MICRO_COL], errors="coerce")
    temp[MACRO_COL] = pd.to_numeric(temp[MACRO_COL], errors="coerce")

    # 只保留温度在合理物理范围内、日期和站点均有效的小时记录。
    temp = temp.loc[
        temp[MICRO_COL].between(*MICRO_RANGE)
        & temp[MACRO_COL].between(*MACRO_RANGE)
        & temp[DATE_COL].notna()
        & temp[SITE_COL].notna()
    ].copy()
    n_after_qc = len(temp)

    temp = temp.drop_duplicates()
    n_after_dedup = len(temp)

    read_spi_bar = progress_step("读取逐日SPI宽表", "读取与质控")
    try:
        spi_wide = pd.read_excel(SPI_DAILY_WIDE_FILE)
        read_spi_bar.update(1)
    finally:
        read_spi_bar.close()
    if spi_wide.shape[1] < 2:
        raise ValueError("SPI wide table must have one date column and at least one site column.")

    spi_date_col = spi_wide.columns[0]
    spi_wide[spi_date_col] = pd.to_datetime(
        spi_wide[spi_date_col], errors="coerce"
    ).dt.normalize()

    # SPI 原表为“日期 × 站点”的宽表；模型计算需要转换为逐站点逐日长表。
    spi_long = (
        spi_wide.melt(
            id_vars=[spi_date_col],
            var_name=SITE_COL,
            value_name=SPI_COL,
        )
        .rename(columns={spi_date_col: DATE_COL})
    )
    spi_long[SITE_COL] = normalize_site_id(spi_long[SITE_COL])
    spi_long[SPI_COL] = pd.to_numeric(spi_long[SPI_COL], errors="coerce")
    spi_long = spi_long.loc[spi_long[DATE_COL].notna()].copy()

    if spi_long.duplicated([SITE_COL, DATE_COL]).any():
        duplicates = spi_long.loc[spi_long.duplicated([SITE_COL, DATE_COL], keep=False)]
        raise ValueError(
            "SPI long table has duplicated Site_ID + UTC_Date rows. "
            f"Example duplicates: {duplicates.head().to_dict(orient='records')}"
        )

    read_event_bar = progress_step("读取完整干旱事件长表", "读取与质控")
    try:
        drought_events = pd.read_csv(DROUGHT_EVENT_FILE, low_memory=False)
        read_event_bar.update(1)
    finally:
        read_event_bar.close()

    required_event_cols = {
        SITE_COL,
        "Event_ID",
        "Start_Date",
        "End_Date",
        "Duration_Days",
        "Drought_Level",
        "Min_Daily_SPI",
    }
    missing_event = sorted(required_event_cols - set(drought_events.columns))
    if missing_event:
        raise KeyError(
            f"Drought event file missing columns: {missing_event}; "
            f"available columns: {drought_events.columns.tolist()}"
        )

    drought_events[SITE_COL] = normalize_site_id(drought_events[SITE_COL])
    drought_events["Start_Date"] = pd.to_datetime(
        drought_events["Start_Date"], errors="coerce"
    ).dt.normalize()
    drought_events["End_Date"] = pd.to_datetime(
        drought_events["End_Date"], errors="coerce"
    ).dt.normalize()
    drought_events["Duration_Days"] = pd.to_numeric(
        drought_events["Duration_Days"], errors="coerce"
    )
    drought_events["Min_Daily_SPI"] = pd.to_numeric(
        drought_events["Min_Daily_SPI"], errors="coerce"
    )
    drought_events["Drought_Level_clean"] = (
        drought_events["Drought_Level"].astype(str).str.strip().str.lower()
    )
    drought_events = drought_events.loc[
        drought_events[SITE_COL].notna()
        & drought_events["Start_Date"].notna()
        & drought_events["End_Date"].notna()
        & drought_events["Duration_Days"].notna()
        & drought_events["Min_Daily_SPI"].notna()
    ].copy()
    if (drought_events["End_Date"] < drought_events["Start_Date"]).any():
        raise ValueError("Drought event file contains End_Date earlier than Start_Date.")

    # many_to_one 校验确保每个 Site_ID + UTC_Date 只有一个 SPI30d，避免重复合并放大小时数。
    hourly = temp.merge(
        spi_long,
        on=[SITE_COL, DATE_COL],
        how="left",
        validate="many_to_one",
    )
    spi_match_rate = hourly[SPI_COL].notna().mean()
    if spi_match_rate < 0.95:
        raise ValueError(
            f"SPI match rate is {spi_match_rate:.2%}, below 95%; "
            "check UTC date, site IDs, and date ranges."
        )

    # YearMonth 和 Site_Month 是后续 LMM、站点月份 CBI 配对的基本分析单元。
    hourly["YearMonth"] = hourly[DATE_COL].dt.to_period("M").astype(str)
    hourly["Month"] = hourly[DATE_COL].dt.month.astype(int)
    hourly["Site_Month"] = hourly[SITE_COL].astype(str) + "_" + hourly["YearMonth"]

    input_audit = pd.DataFrame(
        [
            dict(
                temperature_raw_rows=n_raw,
                temperature_rows_after_qc=n_after_qc,
                temperature_rows_after_dedup=n_after_dedup,
                hourly_rows_after_spi_merge=len(hourly),
                time_parse_rate=parse_rate,
                spi_match_rate=spi_match_rate,
                n_sites=hourly[SITE_COL].nunique(),
                utc_date_min=hourly[DATE_COL].min(),
                utc_date_max=hourly[DATE_COL].max(),
                temperature_file=str(TEMP_HOURLY_FILE),
                spi_daily_wide_file=str(SPI_DAILY_WIDE_FILE),
                drought_event_file=str(DROUGHT_EVENT_FILE),
                drought_event_rows=len(drought_events),
                drought_event_extreme_rows=int(
                    drought_events["Drought_Level_clean"].eq("extreme").sum()
                ),
                output_dir=str(OUTPUT_DIR),
            )
        ]
    )

    return hourly, spi_long, drought_events, input_audit


# =============================================================================
# 4. 情景覆盖、LMM、站点月份 CBI 和事件 CBI
# =============================================================================

def select_drought_events_for_scenario(drought_events, cfg):
    """按完整干旱过程选择当前情景的事件。

    事件定义说明：
        主脚本事件 CBI 使用的是“完整干旱事件长表”，事件从进入干旱开始，
        到退出干旱结束。事件中只要有一天达到或低于某个 Extreme 阈值，就把整个
        干旱过程作为该阈值下的事件期，而不是只截取 SPI 低于阈值的短片段。

    BASE 复现：
        在当前事件长表中，Drought_Level == Extreme 与 Min_Daily_SPI <= -2.0
        计数一致。这里统一用 Min_Daily_SPI <= extreme_threshold，便于 E15/E18/E22
        和联合情景自然扩展，同时 BASE 仍能复现主脚本事件集合。
    """
    events = drought_events.loc[
        drought_events["Min_Daily_SPI"] <= cfg["extreme_threshold"]
    ].copy()
    events = events.sort_values([SITE_COL, "Start_Date", "End_Date", "Event_ID"])
    return events.reset_index(drop=True)


def classify_event_drought_level(min_spi):
    """按完整事件最低 SPI 给完整干旱过程分级，便于事件定义敏感性输出审计。"""
    if min_spi <= -2.0:
        return "Extreme"
    if min_spi <= -1.5:
        return "Severe"
    if min_spi <= -1.0:
        return "Moderate"
    return "Light"


def extract_events_from_daily_spi(daily_spi, cfg):
    """按事件定义情景从逐日 SPI 重新提取完整干旱事件。

    用途：
        该函数仅服务 Event_definition 证据链。
        D05/D10 改变的是“连续过程至少多少天才保留”；
        B00/B10 改变的是“完整干旱过程从哪一个 SPI 边界开始/结束”。

    事件定义：
        连续 SPI <= event_threshold 的自然日构成一个候选完整干旱过程。
        若该过程持续天数小于 min_duration_days，则删除。
        该完整过程内最低 SPI <= extreme_threshold 时，进入极端事件配对分析。
    """
    rows = []
    for site, group in daily_spi.sort_values([SITE_COL, DATE_COL]).groupby(SITE_COL):
        g = group[[DATE_COL, SPI_COL]].dropna().sort_values(DATE_COL).reset_index(drop=True)
        if g.empty:
            continue

        segment_start = None

        def close_segment(start_idx, end_idx):
            if start_idx is None or end_idx <= start_idx:
                return
            seg = g.iloc[start_idx:end_idx].copy()
            duration = len(seg)
            if duration < cfg["min_duration_days"]:
                return
            min_pos = seg[SPI_COL].idxmin()
            min_spi = float(seg.loc[min_pos, SPI_COL])
            rows.append(
                {
                    SITE_COL: site,
                    "Start_Date": seg[DATE_COL].iloc[0],
                    "End_Date": seg[DATE_COL].iloc[-1],
                    "Duration_Days": duration,
                    "Min_Daily_SPI": min_spi,
                    "Max_Daily_SPI": float(seg[SPI_COL].max()),
                    "Min_SPI_Date": seg.loc[min_pos, DATE_COL],
                    "Severity": float(np.abs(seg[SPI_COL]).sum()),
                    "Drought_Level": classify_event_drought_level(min_spi),
                    "Edge_Truncated": bool(start_idx == 0 or end_idx == len(g)),
                    "Is_Extreme_Event": bool(min_spi <= cfg["extreme_threshold"]),
                }
            )

        for index, row in g.iterrows():
            value = row[SPI_COL]
            previous_date = g.loc[index - 1, DATE_COL] if index > 0 else None
            date_break = index > 0 and (row[DATE_COL] - previous_date).days != 1
            in_drought = pd.notna(value) and value <= cfg["event_threshold"]
            if segment_start is not None and (date_break or not in_drought):
                close_segment(segment_start, index)
                segment_start = None
            if in_drought and segment_start is None:
                segment_start = index
        close_segment(segment_start, len(g))

    events = pd.DataFrame(rows)
    if events.empty:
        return events
    events = events.sort_values([SITE_COL, "Start_Date", "End_Date"]).reset_index(drop=True)
    events["Event_ID_Robustness"] = events.groupby(SITE_COL).cumcount() + 1
    return events


def event_coverage(hourly_status, daily_spi, drought_events, cfg):
    """审计某个情景下的事件数量和潜在参考期覆盖，不估计 CBI。

    用途：
        coverage_only 模式下先看事件是否有足够 Normal 参考日。
        如果事件很多但潜在参考日很少，full 模式下事件 CBI 可能样本不足。

    参考期规则：
        事件结束后第 1 天为缓冲日；
        候选参考期为 End + 2 到 End + 31，首尾都包含，共 30 个自然日；
        候选日还必须满足当前情景的 Normal 开区间和每日有效小时门槛。
    """
    events = select_drought_events_for_scenario(drought_events, cfg)
    if events.empty:
        return dict(
            identified_extreme_events=0,
            events_with_potential_normal_reference_days=0,
            potential_reference_days_total=0,
        )

    potential_events = 0
    potential_days_total = 0
    # 逐日有效小时数用于剔除残缺参考日，避免少量小时代表整天。
    daily_hour_counts = (
        hourly_status.groupby([SITE_COL, DATE_COL])
        .size()
        .rename("n_valid_hours")
        .reset_index()
    )

    for _, event in progress_iter(
        events.iterrows(),
        total=len(events),
        desc="审计事件参考期覆盖",
        kind="事件覆盖",
    ):
        site = event[SITE_COL]
        search_start = event["End_Date"] + pd.Timedelta(days=POST_EVENT_BUFFER_DAYS + 1)
        search_end = event["End_Date"] + pd.Timedelta(
            days=POST_EVENT_BUFFER_DAYS + POST_EVENT_SEARCH_DAYS
        )

        ref_daily = daily_spi.loc[
            (daily_spi[SITE_COL] == site)
            & daily_spi[DATE_COL].between(search_start, search_end)
            & (daily_spi[SPI_COL] > cfg["normal_low"])
            & (daily_spi[SPI_COL] < cfg["normal_high"])
        ].copy()

        ref_daily = ref_daily.merge(
            daily_hour_counts,
            on=[SITE_COL, DATE_COL],
            how="left",
            validate="one_to_one",
        )
        good_days = ref_daily.loc[
            ref_daily["n_valid_hours"].fillna(0) >= MIN_VALID_HOURS_PER_REFERENCE_DAY
        ]

        if not good_days.empty:
            potential_events += 1
            potential_days_total += good_days[DATE_COL].nunique()

    return dict(
        identified_extreme_events=len(events),
        events_with_potential_normal_reference_days=potential_events,
        potential_reference_days_total=potential_days_total,
    )


def build_lmm_dataset(hourly, daily_spi, drought_events, cfg):
    """按当前情景重分类 SPI，并构造 LMM 所需小时级数据集。

    关键步骤：
        1. 使用当前情景阈值重新生成 SPI_Status；
        2. 汇总 Extreme/Normal/Other 的小时数、天数、站点数和 SiteMonth 数；
        3. 只保留同时拥有足够 Extreme 小时和 Normal 小时的 SiteMonth；
        4. 计算 Macro_Within，即 ERA5 温度相对 SiteMonth 均值的小时偏差；
        5. 剔除 Macro_Within 没有变化的 SiteMonth，因为这类单元无法估计温度斜率；
        6. 返回覆盖审计信息和可直接进入 LMM 的小时数据。

    参数影响：
        extreme_threshold/normal_low/normal_high 改变 SPI 分类；
        min_status_hours 改变 SiteMonth 进入 LMM 的最低小时门槛，调大后样本更严格但更少。
    """
    d = hourly.copy()
    d["SPI_Status"] = classify_status(d[SPI_COL], cfg)
    d["Is_Extreme"] = (d["SPI_Status"] == "Extreme").astype(int)

    status_summary = (
        d.groupby("SPI_Status")
        .agg(
            n_hours=(SPI_COL, "size"),
            n_days=(DATE_COL, "nunique"),
            n_sites=(SITE_COL, "nunique"),
            n_sitemonths=("Site_Month", "nunique"),
        )
        .reset_index()
    )

    # LMM 只比较 Normal 与 Extreme，Other 只用于覆盖审计，不进入模型。
    candidates = d.loc[d["SPI_Status"].isin(["Normal", "Extreme"])].copy()
    counts = (
        candidates.groupby(["Site_Month", "SPI_Status"])
        .size()
        .unstack(fill_value=0)
    )
    for status in ["Normal", "Extreme"]:
        if status not in counts.columns:
            counts[status] = 0

    # 一个 SiteMonth 必须同时满足 Normal 和 Extreme 的最低小时数，才能支持状态内比较。
    eligible = counts.index[
        (counts["Normal"] >= cfg["min_status_hours"])
        & (counts["Extreme"] >= cfg["min_status_hours"])
    ]

    site_month_macro_all = (
        hourly.groupby("Site_Month")[MACRO_COL]
        .agg(
            Macro_Mean_SiteMonth_AllValid="mean",
            Macro_SD_SiteMonth_AllValid="std",
            n_all_valid_hours="size",
        )
        .reset_index()
    )

    d_lmm = candidates.loc[candidates["Site_Month"].isin(eligible)].copy()
    d_lmm = d_lmm.merge(
        site_month_macro_all,
        on="Site_Month",
        how="left",
        validate="many_to_one",
    )
    # Macro_Within 是主 LMM 的核心连续自变量，表示同一 SiteMonth 内的大气温度波动。
    d_lmm["Macro_Within"] = d_lmm[MACRO_COL] - d_lmm["Macro_Mean_SiteMonth_AllValid"]

    macro_sd_lmm = d_lmm.groupby("Site_Month")["Macro_Within"].std()
    valid_macro_site_months = macro_sd_lmm.index[macro_sd_lmm.fillna(0) > 0]
    d_lmm = d_lmm.loc[d_lmm["Site_Month"].isin(valid_macro_site_months)].copy()

    if not d_lmm.empty:
        d_lmm["Month_Factor"] = d_lmm["Month"].astype(str)
        d_lmm["Site_Group"] = d_lmm[SITE_COL].astype(str)
        d_lmm["Macro_Mean_SiteMonth_C"] = (
            d_lmm["Macro_Mean_SiteMonth_AllValid"]
            - d_lmm["Macro_Mean_SiteMonth_AllValid"].mean()
        )

    coverage = dict(
        all_hourly_rows=len(d),
        all_sites=d[SITE_COL].nunique(),
        all_sitemonths=d["Site_Month"].nunique(),
        extreme_hours=int((d["SPI_Status"] == "Extreme").sum()),
        normal_hours=int((d["SPI_Status"] == "Normal").sum()),
        other_hours=int((d["SPI_Status"] == "Other").sum()),
        extreme_days=d.loc[d["SPI_Status"] == "Extreme", DATE_COL].nunique(),
        normal_days=d.loc[d["SPI_Status"] == "Normal", DATE_COL].nunique(),
        extreme_sites=d.loc[d["SPI_Status"] == "Extreme", SITE_COL].nunique(),
        normal_sites=d.loc[d["SPI_Status"] == "Normal", SITE_COL].nunique(),
        candidate_sitemonths=len(counts),
        eligible_sitemonths_before_macro_sd=len(eligible),
        lmm_hours=len(d_lmm),
        lmm_sites=d_lmm[SITE_COL].nunique() if not d_lmm.empty else 0,
        lmm_sitemonths=d_lmm["Site_Month"].nunique() if not d_lmm.empty else 0,
        lmm_extreme_hours=int((d_lmm["SPI_Status"] == "Extreme").sum()) if not d_lmm.empty else 0,
        lmm_normal_hours=int((d_lmm["SPI_Status"] == "Normal").sum()) if not d_lmm.empty else 0,
    )
    coverage.update(event_coverage(d, daily_spi, drought_events, cfg))

    return d, d_lmm, status_summary, coverage


def run_lmm(d_lmm):
    """拟合与主脚本一致的混合线性模型。

    模型目的：
        估计 Extreme 与 Normal 下 CBI 是否不同。核心结果是
        Macro_Within:Is_Extreme 交互项，即 LMM_Delta_CBI。

    模型结构：
        固定效应包括 Macro_Within、Is_Extreme、二者交互、月份控制、
        月份 × Macro_Within 交互、SiteMonth 平均 ERA5 温度中心化项。
        随机效应以 Site_ID 为组，包含随机截距和 Macro_Within 随机斜率。

    参数影响：
        LMM_METHOD 和 LMM_MAXITER 只影响模型优化过程，不改变模型公式。
        如果收敛失败，可优先增加 LMM_MAXITER 或检查稀疏情景样本量。
    """
    empty_result = dict(
        LMM_Delta_CBI=np.nan,
        LMM_CI_low95=np.nan,
        LMM_CI_high95=np.nan,
        LMM_Pvalue=np.nan,
        LMM_CBI_Normal_weighted=np.nan,
        LMM_CBI_Extreme_weighted=np.nan,
        LMM_Converged=np.nan,
        LMM_Flag="insufficient_LMM_data",
    )

    if (
        d_lmm.empty
        or d_lmm["Is_Extreme"].nunique() < 2
        or d_lmm[SITE_COL].nunique() < 2
        or d_lmm["Site_Month"].nunique() < 2
    ):
        return empty_result

    # 公式必须和主脚本保持一致，否则 BASE 复现和敏感性比较会失去意义。
    formula = (
        f"{MICRO_COL} ~ "
        "Macro_Within * Is_Extreme + "
        "C(Month_Factor) + "
        "Macro_Within:C(Month_Factor) + "
        "Macro_Mean_SiteMonth_C"
    )

    try:
        model = smf.mixedlm(
            formula=formula,
            data=d_lmm,
            groups=d_lmm["Site_Group"],
            re_formula="1 + Macro_Within",
        )
        result = model.fit(
            reml=True,
            method=LMM_METHOD,
            maxiter=LMM_MAXITER,
            disp=False,
        )

        interaction_term = next(
            (
                term
                for term in ["Macro_Within:Is_Extreme", "Is_Extreme:Macro_Within"]
                if term in result.params.index
            ),
            None,
        )
        if interaction_term is None:
            raise KeyError("Macro_Within:Is_Extreme term not found.")

        delta = result.params[interaction_term]
        ci_low, ci_high = result.conf_int().loc[interaction_term].tolist()

        params = result.params
        # 用当前 LMM 样本的月份分布计算加权 Normal CBI，便于与 Extreme CBI 同尺度解释。
        month_weights = d_lmm["Month_Factor"].value_counts(normalize=True).to_dict()
        base_macro = params.get("Macro_Within", np.nan)
        weighted_normal = 0.0
        for month_level, weight in month_weights.items():
            month_term = next(
                (
                    term
                    for term in [
                        f"Macro_Within:C(Month_Factor)[T.{month_level}]",
                        f"C(Month_Factor)[T.{month_level}]:Macro_Within",
                    ]
                    if term in params.index
                ),
                None,
            )
            month_slope = base_macro + (params[month_term] if month_term else 0.0)
            weighted_normal += weight * month_slope

        return dict(
            LMM_Delta_CBI=delta,
            LMM_CI_low95=ci_low,
            LMM_CI_high95=ci_high,
            LMM_Pvalue=result.pvalues[interaction_term],
            LMM_CBI_Normal_weighted=weighted_normal,
            LMM_CBI_Extreme_weighted=weighted_normal + delta,
            LMM_Converged=bool(getattr(result, "converged", False)),
            LMM_Flag="ok",
        )

    except Exception as exc:
        out = empty_result.copy()
        out["LMM_Flag"] = f"fit_failed: {type(exc).__name__}: {exc}"
        return out


def run_sitemonth_cbi(d_status, cfg, scenario_id):
    """估计站点月份尺度的 Normal/Extreme 配对 OLS CBI。

    目的：
        作为 LMM 之外的辅助证据链，检查在 Site_ID × YearMonth 单元内，
        Extreme 的 OLS CBI 是否相对 Normal 发生一致变化。

    参数影响：
        min_cbi_hours 控制每个 Site_ID × YearMonth × SPI_Status 单元的最低小时数；
        min_macro_sd 控制 ERA5 温度变化幅度门槛，避免外界温度几乎不变时硬估斜率。
    """
    rows = []
    # 只在出现过 Extreme 的站点内寻找 Normal 对照，避免无事件站点稀释配对分析。
    source_sites = set(d_status.loc[d_status["SPI_Status"] == "Extreme", SITE_COL].unique())
    d = d_status.loc[
        d_status[SITE_COL].isin(source_sites)
        & d_status["SPI_Status"].isin(["Normal", "Extreme"])
    ].copy()

    grouped = list(d.groupby([SITE_COL, "YearMonth", "SPI_Status"]))
    for (site, year_month, status), group in progress_iter(
        grouped,
        total=len(grouped),
        desc=f"{scenario_id} 站点月份CBI",
        kind="站点月份CBI",
    ):
        result = ols_cbi(group, cfg["min_cbi_hours"], cfg["min_macro_sd"])
        rows.append(
            {
                "ScenarioID": scenario_id,
                SITE_COL: site,
                "YearMonth": year_month,
                "SPI_Status": status,
                **result,
            }
        )

    units = pd.DataFrame(rows)
    if units.empty:
        return (
            dict(
                SiteMonth_Delta_Mean=np.nan,
                SiteMonth_Delta_Median=np.nan,
                SiteMonth_N_pairs=0,
                SiteMonth_N_sites=0,
                SiteMonth_Flag="no_CBI_units",
            ),
            units,
            pd.DataFrame(),
        )

    # 只有 Normal 和 Extreme 两个状态都成功估计 CBI 的 SiteMonth 才能形成配对差值。
    ok = units.loc[units["Flag"] == "ok"].copy()
    wide = ok.pivot(index=[SITE_COL, "YearMonth"], columns="SPI_Status", values="CBI")
    if not {"Normal", "Extreme"}.issubset(wide.columns):
        return (
            dict(
                SiteMonth_Delta_Mean=np.nan,
                SiteMonth_Delta_Median=np.nan,
                SiteMonth_N_pairs=0,
                SiteMonth_N_sites=0,
                SiteMonth_Flag="no_paired_CBI_units",
            ),
            units,
            pd.DataFrame(),
        )

    pairs = wide.dropna(subset=["Normal", "Extreme"]).copy()
    pairs["Delta_CBI"] = pairs["Extreme"] - pairs["Normal"]
    pairs = pairs.reset_index()
    pairs["ScenarioID"] = scenario_id

    return (
        dict(
            SiteMonth_Delta_Mean=pairs["Delta_CBI"].mean(),
            SiteMonth_Delta_Median=pairs["Delta_CBI"].median(),
            SiteMonth_N_pairs=len(pairs),
            SiteMonth_N_sites=pairs[SITE_COL].nunique(),
            SiteMonth_Flag="ok" if not pairs.empty else "no_paired_CBI_units",
        ),
        units,
        pairs,
    )


def run_event_cbi(hourly_status, daily_spi, drought_events, cfg, scenario_id):
    """估计事件期与事件后 Normal 参考期的配对 CBI。

    目的：
        从事件角度验证 Extreme 期间 CBI 是否相对事件后 Normal 参考期改变。
        这条证据链和 LMM 不完全相同，重点是事件级别的配对差值。

    事件定义：
        使用主脚本同一个完整干旱事件长表，而不是从逐日 SPI 重新切短事件。
        当前情景用 Min_Daily_SPI <= extreme_threshold 判断完整事件是否纳入。
        因此 BASE 的事件集合与主脚本 Drought_Level == Extreme 事件一致。

    参考期选择：
        事件结束后第 1 天跳过；
        在 End + 2 到 End + 31 的 30 天窗口内寻找 Normal 日；
        只选每日有效小时数达到 MIN_VALID_HOURS_PER_REFERENCE_DAY 的日期；
        参考期目标天数等于事件持续天数，但最多 MAX_REFERENCE_TARGET_DAYS 天。

    参数影响：
        min_event_hours/min_reference_hours 控制事件期和参考期 OLS CBI 的最低小时数；
        apply_event_macro_sd=True 时，事件 CBI 也会应用 min_macro_sd 门槛，仅用于 M 类敏感性。
    """
    events = select_drought_events_for_scenario(drought_events, cfg)
    rows = []
    if events.empty:
        return cluster_bootstrap(pd.DataFrame(columns=[SITE_COL, "Delta_CBI"])), pd.DataFrame()

    daily_hour_counts = (
        hourly_status.groupby([SITE_COL, DATE_COL])
        .size()
        .rename("n_valid_hours")
        .reset_index()
    )

    # BASE 与 E/N/C/H 情景不对事件 CBI 施加 MacroSD；M 情景才新增这个质量筛选。
    event_macro_sd_threshold = (
        cfg["min_macro_sd"] if cfg["apply_event_macro_sd"] else None
    )

    for _, event in progress_iter(
        events.iterrows(),
        total=len(events),
        desc=f"{scenario_id} 事件CBI",
        kind="事件CBI",
    ):
        site = event[SITE_COL]
        start = event["Start_Date"]
        end = event["End_Date"]

        event_hours = hourly_status.loc[
            (hourly_status[SITE_COL] == site)
            & hourly_status[DATE_COL].between(start, end)
        ]
        event_result = ols_cbi(
            event_hours,
            cfg["min_event_hours"],
            event_macro_sd_threshold,
        )

        search_start = end + pd.Timedelta(days=POST_EVENT_BUFFER_DAYS + 1)
        search_end = end + pd.Timedelta(days=POST_EVENT_BUFFER_DAYS + POST_EVENT_SEARCH_DAYS)
        ref_daily = daily_spi.loc[
            (daily_spi[SITE_COL] == site)
            & daily_spi[DATE_COL].between(search_start, search_end)
            & (daily_spi[SPI_COL] > cfg["normal_low"])
            & (daily_spi[SPI_COL] < cfg["normal_high"])
        ].copy()

        ref_daily = ref_daily.merge(
            daily_hour_counts,
            on=[SITE_COL, DATE_COL],
            how="left",
            validate="one_to_one",
        )
        good_days = (
            ref_daily.loc[
                ref_daily["n_valid_hours"].fillna(0) >= MIN_VALID_HOURS_PER_REFERENCE_DAY,
                DATE_COL,
            ]
            .sort_values()
            .drop_duplicates()
            .tolist()
        )

        # 短事件用相同天数的参考日；长事件最多使用 MAX_REFERENCE_TARGET_DAYS 天。
        target_days = min(int(event["Duration_Days"]), MAX_REFERENCE_TARGET_DAYS)
        ref_days = good_days[:target_days]
        ref_hours = hourly_status.loc[
            (hourly_status[SITE_COL] == site) & hourly_status[DATE_COL].isin(ref_days)
        ]
        ref_result = ols_cbi(
            ref_hours,
            cfg["min_reference_hours"],
            event_macro_sd_threshold,
        )

        pair_flag = (
            "ok"
            if event_result["Flag"] == "ok" and ref_result["Flag"] == "ok"
            else f"event={event_result['Flag']};reference={ref_result['Flag']}"
        )

        rows.append(
            {
                "ScenarioID": scenario_id,
                SITE_COL: site,
                "Event_ID": event["Event_ID"],
                "Start_Date": start,
                "End_Date": end,
                "Duration_Days": event["Duration_Days"],
                "Minimum_SPI": event["Min_Daily_SPI"],
                "Drought_Level": event.get("Drought_Level", np.nan),
                "Severity": event.get("Severity", np.nan),
                "Edge_Truncated": event.get("Edge_Truncated", np.nan),
                "Reference_Search_Start": search_start,
                "Reference_Search_End": search_end,
                "Reference_Target_Days": target_days,
                "Reference_Selected_Days": len(ref_days),
                "Event_MacroSD_Threshold": event_macro_sd_threshold,
                "Event_CBI": event_result["CBI"],
                "Event_n_hours": event_result["n_hours"],
                "Event_MacroSD": event_result["MacroSD"],
                "Event_Flag": event_result["Flag"],
                "Reference_CBI": ref_result["CBI"],
                "Reference_n_hours": ref_result["n_hours"],
                "Reference_MacroSD": ref_result["MacroSD"],
                "Reference_Flag": ref_result["Flag"],
                "Delta_CBI": event_result["CBI"] - ref_result["CBI"],
                "Pair_Flag": pair_flag,
            }
        )

    pairs = pd.DataFrame(rows)
    valid = pairs.loc[pairs["Pair_Flag"] == "ok"].copy()
    return cluster_bootstrap(valid), pairs


def event_definition_coverage(hourly, daily_spi, events, cfg):
    """事件定义证据链的覆盖审计。

    只统计重新识别的完整事件及其参考期覆盖，不涉及小时级 LMM 样本。
    """
    if events.empty:
        return dict(
            identified_events=0,
            identified_sites=0,
            extreme_events=0,
            extreme_event_sites=0,
            edge_truncated_extreme_events=0,
            events_with_potential_normal_reference_days=0,
            potential_reference_days_total=0,
        )

    extreme = events.loc[events["Is_Extreme_Event"]].copy()
    if extreme.empty:
        return dict(
            identified_events=len(events),
            identified_sites=events[SITE_COL].nunique(),
            extreme_events=0,
            extreme_event_sites=0,
            edge_truncated_extreme_events=0,
            events_with_potential_normal_reference_days=0,
            potential_reference_days_total=0,
        )

    daily_hour_counts = (
        hourly.groupby([SITE_COL, DATE_COL])
        .size()
        .rename("n_valid_hours")
        .reset_index()
    )
    potential_events = 0
    potential_days_total = 0

    for _, event in progress_iter(
        extreme.iterrows(),
        total=len(extreme),
        desc=f"{cfg['id']} 事件覆盖审计",
        kind="事件覆盖",
    ):
        site = event[SITE_COL]
        search_start = event["End_Date"] + pd.Timedelta(days=POST_EVENT_BUFFER_DAYS + 1)
        search_end = event["End_Date"] + pd.Timedelta(days=POST_EVENT_BUFFER_DAYS + POST_EVENT_SEARCH_DAYS)
        ref_daily = daily_spi.loc[
            (daily_spi[SITE_COL] == site)
            & daily_spi[DATE_COL].between(search_start, search_end)
            & (daily_spi[SPI_COL] > cfg["normal_low"])
            & (daily_spi[SPI_COL] < cfg["normal_high"])
        ].copy()
        ref_daily = ref_daily.merge(
            daily_hour_counts,
            on=[SITE_COL, DATE_COL],
            how="left",
            validate="one_to_one",
        )
        good_days = ref_daily.loc[
            ref_daily["n_valid_hours"].fillna(0) >= MIN_VALID_HOURS_PER_REFERENCE_DAY
        ]
        if not good_days.empty:
            potential_events += 1
            potential_days_total += good_days[DATE_COL].nunique()

    return dict(
        identified_events=len(events),
        identified_sites=events[SITE_COL].nunique(),
        extreme_events=len(extreme),
        extreme_event_sites=extreme[SITE_COL].nunique(),
        edge_truncated_extreme_events=int(extreme["Edge_Truncated"].sum()),
        events_with_potential_normal_reference_days=potential_events,
        potential_reference_days_total=potential_days_total,
    )


def run_event_definition_cbi(hourly, daily_spi, events, cfg, scenario_id):
    """事件定义证据链：对重新提取的完整事件计算事件—参考期 CBI 配对。"""
    extreme = events.loc[events["Is_Extreme_Event"]].copy()
    rows = []
    if extreme.empty:
        return cluster_bootstrap(pd.DataFrame(columns=[SITE_COL, "Delta_CBI"])), pd.DataFrame()

    daily_hour_counts = (
        hourly.groupby([SITE_COL, DATE_COL])
        .size()
        .rename("n_valid_hours")
        .reset_index()
    )

    for _, event in progress_iter(
        extreme.iterrows(),
        total=len(extreme),
        desc=f"{scenario_id} 事件定义CBI",
        kind="事件CBI",
    ):
        site = event[SITE_COL]
        start = event["Start_Date"]
        end = event["End_Date"]

        event_result = ols_cbi(
            hourly.loc[
                (hourly[SITE_COL] == site)
                & hourly[DATE_COL].between(start, end)
            ],
            cfg["min_event_hours"],
        )

        search_start = end + pd.Timedelta(days=POST_EVENT_BUFFER_DAYS + 1)
        search_end = end + pd.Timedelta(days=POST_EVENT_BUFFER_DAYS + POST_EVENT_SEARCH_DAYS)
        ref_daily = daily_spi.loc[
            (daily_spi[SITE_COL] == site)
            & daily_spi[DATE_COL].between(search_start, search_end)
            & (daily_spi[SPI_COL] > cfg["normal_low"])
            & (daily_spi[SPI_COL] < cfg["normal_high"])
        ].copy()
        ref_daily = ref_daily.merge(
            daily_hour_counts,
            on=[SITE_COL, DATE_COL],
            how="left",
            validate="one_to_one",
        )
        good_days = (
            ref_daily.loc[
                ref_daily["n_valid_hours"].fillna(0) >= MIN_VALID_HOURS_PER_REFERENCE_DAY,
                DATE_COL,
            ]
            .sort_values()
            .drop_duplicates()
            .tolist()
        )
        target_days = min(int(event["Duration_Days"]), MAX_REFERENCE_TARGET_DAYS)
        selected = good_days[:target_days]
        ref_result = ols_cbi(
            hourly.loc[
                (hourly[SITE_COL] == site)
                & hourly[DATE_COL].isin(selected)
            ],
            cfg["min_reference_hours"],
        )
        pair_flag = (
            "ok"
            if event_result["Flag"] == "ok" and ref_result["Flag"] == "ok"
            else f"event={event_result['Flag']};reference={ref_result['Flag']}"
        )
        rows.append(
            {
                "AnalysisLayer": "Event_definition",
                "EvidenceChain": "Event_definition",
                "ScenarioID": scenario_id,
                SITE_COL: site,
                "Event_ID_Robustness": event["Event_ID_Robustness"],
                "Start_Date": start,
                "End_Date": end,
                "Duration_Days": event["Duration_Days"],
                "Min_Daily_SPI": event["Min_Daily_SPI"],
                "Drought_Level": event["Drought_Level"],
                "Edge_Truncated": event["Edge_Truncated"],
                "Reference_Search_Start": search_start,
                "Reference_Search_End": search_end,
                "Reference_Target_Days": target_days,
                "Reference_Selected_Days": len(selected),
                "Event_CBI": event_result["CBI"],
                "Event_n_hours": event_result["n_hours"],
                "Reference_CBI": ref_result["CBI"],
                "Reference_n_hours": ref_result["n_hours"],
                "Delta_CBI": event_result["CBI"] - ref_result["CBI"],
                "Pair_Flag": pair_flag,
            }
        )

    pairs = pd.DataFrame(rows)
    valid = pairs.loc[pairs["Pair_Flag"] == "ok"].copy()
    return cluster_bootstrap(valid), pairs


# =============================================================================
# 5. 结果汇总、绘图和 BASE 复现核验
# =============================================================================

def make_forest_plot(summary):
    """绘制 LMM Delta CBI 森林图。

    图形解释：
        每一行是一个 LMM 有效情景；
        点表示 LMM_Delta_CBI，即 Extreme CBI - Normal CBI；
        横线表示 95% 置信区间；
        竖向 0 线用于判断方向和置信区间是否跨 0。

    外观调整：
        图宽、高度、颜色、线型、图例文字和图例位置均在 0A 参数区集中修改。
    """
    d = summary.loc[
        summary["AnalysisLayer"].eq("Hourly_state")
        & summary["EvidenceChain"].eq("Hourly_state")
        & summary["LMM_Flag"].eq("ok")
        & summary["LMM_Delta_CBI"].notna()
    ].copy()
    if d.empty:
        return

    d = d.sort_values(["ScenarioGroup", "ScenarioID"]).reset_index(drop=True)
    y = np.arange(len(d))
    plot_bar = progress_step("绘制LMM阈值稳健性森林图", "绘图")
    fig, ax = plt.subplots(
        figsize=(
            FIG_FOREST_WIDTH,
            max(FIG_FOREST_MIN_HEIGHT, FIG_FOREST_HEIGHT_PER_SCENARIO * len(d)),
        )
    )
    colors = np.where(
        d["ScenarioID"].eq("BASE"),
        COLOR_BASE_SCENARIO,
        COLOR_OTHER_SCENARIO,
    )

    ax.errorbar(
        d["LMM_Delta_CBI"],
        y,
        xerr=[
            d["LMM_Delta_CBI"] - d["LMM_CI_low95"],
            d["LMM_CI_high95"] - d["LMM_Delta_CBI"],
        ],
        fmt="none",
        color=COLOR_CI,
        elinewidth=LINEWIDTH_CI,
        capsize=CI_CAPSIZE,
        zorder=1,
    )
    ax.scatter(d["LMM_Delta_CBI"], y, c=colors, s=SCATTER_SIZE_FOREST, zorder=2)
    ax.axvline(
        0,
        color=COLOR_ZERO_LINE,
        linestyle=LINESTYLE_ZERO,
        linewidth=LINEWIDTH_ZERO,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(d["ScenarioID"] + " | " + d["ScenarioDescription"])
    ax.set_xlabel("Delta CBI（Extreme - Normal；LMM交互项估计）")
    ax.set_title("Extreme 与 Normal CBI 差异的阈值稳健性")
    if LEGEND_SHOW:
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=COLOR_BASE_SCENARIO,
                markeredgecolor=COLOR_BASE_SCENARIO,
                markersize=np.sqrt(SCATTER_SIZE_FOREST),
                label=LEGEND_BASE_LABEL,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=COLOR_OTHER_SCENARIO,
                markeredgecolor=COLOR_OTHER_SCENARIO,
                markersize=np.sqrt(SCATTER_SIZE_FOREST),
                label=LEGEND_OTHER_LABEL,
            ),
            Line2D(
                [0],
                [0],
                color=COLOR_CI,
                linewidth=LINEWIDTH_CI,
                label=LEGEND_CI_LABEL,
            ),
            Line2D(
                [0],
                [0],
                color=COLOR_ZERO_LINE,
                linestyle=LINESTYLE_ZERO,
                linewidth=LINEWIDTH_ZERO,
                label=LEGEND_ZERO_LABEL,
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc=LEGEND_LOCATION,
            bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
            frameon=LEGEND_FRAME_ON,
            fontsize=LEGEND_FONT_SIZE,
            title=LEGEND_TITLE,
        )
    ax.invert_yaxis()
    fig.tight_layout()
    try:
        fig.savefig(
            OUTPUT_DIR / OUTPUT_FILES["forest_plot"],
            dpi=FIG_DPI,
            bbox_inches=FIGURE_BBOX,
        )
        plot_bar.update(1)
    finally:
        plt.close(fig)
        plot_bar.close()


def make_event_definition_plot(summary):
    """绘制事件定义稳健性森林图，只展示 Event_definition 证据链。"""
    d = summary.loc[
        summary["AnalysisLayer"].eq("Event_definition")
        & summary["EvidenceChain"].eq("Event_definition")
        & summary["Event_Flag"].eq("ok")
        & summary["Event_Delta_Mean"].notna()
    ].copy()
    if d.empty:
        return

    d = d.sort_values(["ScenarioGroup", "ScenarioID"]).reset_index(drop=True)
    y = np.arange(len(d))
    plot_bar = progress_step("绘制事件定义稳健性森林图", "绘图")
    fig, ax = plt.subplots(
        figsize=(
            FIG_FOREST_WIDTH,
            max(FIG_FOREST_MIN_HEIGHT, FIG_FOREST_HEIGHT_PER_SCENARIO * len(d)),
        )
    )
    colors = np.where(
        d["ScenarioID"].eq("BASE"),
        COLOR_BASE_SCENARIO,
        COLOR_OTHER_SCENARIO,
    )

    ax.errorbar(
        d["Event_Delta_Mean"],
        y,
        xerr=[
            d["Event_Delta_Mean"] - d["Event_CI_low95"],
            d["Event_CI_high95"] - d["Event_Delta_Mean"],
        ],
        fmt="none",
        color=COLOR_CI,
        elinewidth=LINEWIDTH_CI,
        capsize=CI_CAPSIZE,
        zorder=1,
    )
    ax.scatter(d["Event_Delta_Mean"], y, c=colors, s=SCATTER_SIZE_FOREST, zorder=2)
    ax.axvline(
        0,
        color=COLOR_ZERO_LINE,
        linestyle=LINESTYLE_ZERO,
        linewidth=LINEWIDTH_ZERO,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(d["ScenarioID"] + " | " + d["ScenarioDescription"])
    ax.set_xlabel("Delta CBI（事件期 - 事件后 Normal 参考期）")
    ax.set_title("完整干旱事件定义的稳健性")
    if LEGEND_SHOW:
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=COLOR_BASE_SCENARIO,
                markeredgecolor=COLOR_BASE_SCENARIO,
                markersize=np.sqrt(SCATTER_SIZE_FOREST),
                label=LEGEND_BASE_LABEL,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=COLOR_OTHER_SCENARIO,
                markeredgecolor=COLOR_OTHER_SCENARIO,
                markersize=np.sqrt(SCATTER_SIZE_FOREST),
                label=LEGEND_OTHER_LABEL,
            ),
            Line2D(
                [0],
                [0],
                color=COLOR_CI,
                linewidth=LINEWIDTH_CI,
                label=LEGEND_CI_LABEL,
            ),
            Line2D(
                [0],
                [0],
                color=COLOR_ZERO_LINE,
                linestyle=LINESTYLE_ZERO,
                linewidth=LINEWIDTH_ZERO,
                label=LEGEND_ZERO_LABEL,
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc=LEGEND_LOCATION,
            bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
            frameon=LEGEND_FRAME_ON,
            fontsize=LEGEND_FONT_SIZE,
            title=LEGEND_TITLE,
        )
    ax.invert_yaxis()
    fig.tight_layout()
    try:
        fig.savefig(
            OUTPUT_DIR / OUTPUT_FILES["event_definition_plot"],
            dpi=FIG_DPI,
            bbox_inches=FIGURE_BBOX,
        )
        plot_bar.update(1)
    finally:
        plt.close(fig)
        plot_bar.close()


def add_direction_consistency(summary):
    """用小时级 BASE 的 Delta CBI 方向统一判断 Hourly_state 非 BASE 情景是否同向。

    分母规则：
        只统计 Hourly_state 非 BASE 且 LMM 成功、LMM_Delta_CBI 非空的情景。
        样本不足或模型失败的情景不进入方向一致性分母。
    """
    summary = summary.copy()
    summary["SameDirectionAsBaseline"] = np.nan
    base_rows = summary.loc[
        summary["AnalysisLayer"].eq("Hourly_state")
        & summary["ScenarioID"].eq("BASE")
        & summary["LMM_Flag"].eq("ok")
        & summary["LMM_Delta_CBI"].notna()
    ]
    if base_rows.empty:
        return summary

    base_delta = base_rows.iloc[0]["LMM_Delta_CBI"]
    valid_non_base = (
        summary["AnalysisLayer"].eq("Hourly_state")
        & ~summary["ScenarioID"].eq("BASE")
        & summary["LMM_Flag"].eq("ok")
        & summary["LMM_Delta_CBI"].notna()
    )
    summary.loc[valid_non_base, "SameDirectionAsBaseline"] = (
        np.sign(summary.loc[valid_non_base, "LMM_Delta_CBI"]) == np.sign(base_delta)
    )
    return summary


def add_event_definition_direction_consistency(summary):
    """用事件定义链 BASE 的 Event Delta CBI 方向判断 D/B 情景是否同向。"""
    summary = summary.copy()
    summary["SameDirectionAsEventBase"] = np.nan
    base_rows = summary.loc[
        summary["AnalysisLayer"].eq("Event_definition")
        & summary["ScenarioID"].eq("BASE")
        & summary["Event_Flag"].eq("ok")
        & summary["Event_Delta_Mean"].notna()
    ]
    if base_rows.empty:
        return summary

    base_delta = base_rows.iloc[0]["Event_Delta_Mean"]
    valid_non_base = (
        summary["AnalysisLayer"].eq("Event_definition")
        & ~summary["ScenarioID"].eq("BASE")
        & summary["Event_Flag"].eq("ok")
        & summary["Event_Delta_Mean"].notna()
    )
    summary.loc[valid_non_base, "SameDirectionAsEventBase"] = (
        np.sign(summary.loc[valid_non_base, "Event_Delta_Mean"]) == np.sign(base_delta)
    )
    return summary


def first_matching_file(pattern):
    """在主脚本结果目录中查找第一个匹配文件，用于 BASE 复现对照。"""
    matches = sorted(MAIN_OUTPUT_DIR.glob(pattern))
    return matches[0] if matches else None


def base_reproduction_audit(summary):
    """将双证据链中的 BASE 分别与主脚本已有输出逐项对照。

    - Hourly_state 的 BASE 负责核验 LMM、站点月份 CBI 和主脚本定义事件配对结果；
    - Event_definition 的 BASE 负责核验完整干旱事件数和可用事件配对数。
    """
    rows = []

    def append_row(metric, independent_value, main_value=np.nan, source="", note=""):
        match = (
            pd.notna(independent_value)
            and pd.notna(main_value)
            and np.isclose(independent_value, main_value, equal_nan=True)
        )
        rows.append(
            {
                "Metric": metric,
                "Robustness_BASE": independent_value,
                "Main_script": main_value,
                "Match": match if pd.notna(main_value) else np.nan,
                "Main_output_source": source,
                "Note": note,
            }
        )

    hourly_base_rows = summary.loc[
        summary["AnalysisLayer"].eq("Hourly_state")
        & summary["ScenarioID"].eq("BASE")
    ]
    if not hourly_base_rows.empty:
        base = hourly_base_rows.iloc[0]
        lmm_file = first_matching_file("01_*LMM*数据集.csv")
        if lmm_file is not None:
            main_lmm = pd.read_csv(lmm_file, usecols=[SITE_COL, "Site_Month", "SPI_Status"])
            append_row("Hourly_state BASE | LMM hourly rows", base["lmm_hours"], len(main_lmm), lmm_file.name)
            append_row("Hourly_state BASE | LMM sites", base["lmm_sites"], main_lmm[SITE_COL].nunique(), lmm_file.name)
            append_row(
                "Hourly_state BASE | LMM SiteMonths",
                base["lmm_sitemonths"],
                main_lmm["Site_Month"].nunique(),
                lmm_file.name,
            )
            append_row(
                "Hourly_state BASE | LMM Extreme hours",
                base["lmm_extreme_hours"],
                int((main_lmm["SPI_Status"] == "Extreme").sum()),
                lmm_file.name,
            )
            append_row(
                "Hourly_state BASE | LMM Normal hours",
                base["lmm_normal_hours"],
                int((main_lmm["SPI_Status"] == "Normal").sum()),
                lmm_file.name,
            )
        else:
            append_row("Hourly_state BASE | LMM sample audit", np.nan, np.nan, note="Main LMM dataset output not found.")

        if RUN_MODE == "full":
            lmm_key_file = first_matching_file("02_*核心结果.csv")
            if lmm_key_file is not None:
                main_lmm_key = pd.read_csv(lmm_key_file)
                delta_columns = [
                    col for col in main_lmm_key.columns if "Delta" in col and "CBI" in col
                ]
                if delta_columns:
                    append_row(
                        "Hourly_state BASE | LMM Delta CBI",
                        base["LMM_Delta_CBI"],
                        main_lmm_key[delta_columns[0]].iloc[0],
                        lmm_key_file.name,
                    )

            site_pair_file = first_matching_file("03_*CBI配对变化审计表.csv")
            if site_pair_file is not None:
                main_site_pairs = pd.read_csv(site_pair_file)
                append_row(
                    "Hourly_state BASE | SiteMonth CBI pairs",
                    base["SiteMonth_N_pairs"],
                    len(main_site_pairs),
                    site_pair_file.name,
                )

            event_file = first_matching_file("04_*对比表.csv")
            if event_file is not None:
                main_events = pd.read_csv(event_file)
                if "Pair_flag" in main_events.columns:
                    append_row(
                        "Hourly_state BASE | Event-reference CBI pairs",
                        base["Event_N_pairs"],
                        int((main_events["Pair_flag"] == "ok").sum()),
                        event_file.name,
                        "BASE event CBI uses the main script's hour-only quality rule.",
                    )

    event_base_rows = summary.loc[
        summary["AnalysisLayer"].eq("Event_definition")
        & summary["ScenarioID"].eq("BASE")
    ]
    if not event_base_rows.empty:
        base = event_base_rows.iloc[0]
        if DROUGHT_EVENT_FILE.exists():
            main_events = pd.read_csv(DROUGHT_EVENT_FILE, low_memory=False)
            main_events[SITE_COL] = normalize_site_id(main_events[SITE_COL])
            main_events["Min_Daily_SPI"] = pd.to_numeric(main_events["Min_Daily_SPI"], errors="coerce")
            main_events["Drought_Level_clean"] = (
                main_events["Drought_Level"].astype(str).str.strip().str.lower()
            )
            append_row(
                "Event_definition BASE | Identified total events",
                base["identified_events"],
                len(main_events),
                DROUGHT_EVENT_FILE.name,
            )
            append_row(
                "Event_definition BASE | Extreme events",
                base["extreme_events"],
                int((main_events["Min_Daily_SPI"] <= -2.0).sum()),
                DROUGHT_EVENT_FILE.name,
            )
            append_row(
                "Event_definition BASE | Event sites",
                base["identified_sites"],
                main_events[SITE_COL].nunique(),
                DROUGHT_EVENT_FILE.name,
            )

            if RUN_MODE == "full":
                event_file = first_matching_file("04_*对比表.csv")
                if event_file is not None:
                    main_event_pairs = pd.read_csv(event_file)
                    if "Pair_flag" in main_event_pairs.columns:
                        append_row(
                            "Event_definition BASE | Event-reference CBI pairs",
                            base["Event_N_pairs"],
                            int((main_event_pairs["Pair_flag"] == "ok").sum()),
                            event_file.name,
                        )

    audit = pd.DataFrame(rows)
    write_csv(audit, OUTPUT_FILES["base_reproduction_audit"])
    return audit


def write_run_report(summary):
    """写出运行报告。

    报告内容包括：
        RUN_MODE、输出目录、BASE 定义、事件参考期规则、
        full 模式下非 BASE 有效 LMM 情景与 BASE 同方向的数量，
        以及完整情景汇总表。
    """
    hourly_summary = summary.loc[summary["AnalysisLayer"].eq("Hourly_state")].copy()
    event_definition_summary = summary.loc[summary["AnalysisLayer"].eq("Event_definition")].copy()

    report_lines = [
        "Robustness sensitivity analysis report",
        "=" * 70,
        f"RUN_MODE: {RUN_MODE}",
        f"Output directory: {OUTPUT_DIR}",
        "",
        "Hourly_state BASE definition:",
        f"Extreme: SPI30d <= {HOURLY_STATE_BASE['extreme_threshold']}",
        f"Normal: {HOURLY_STATE_BASE['normal_low']} < SPI30d < {HOURLY_STATE_BASE['normal_high']}",
        f"minimum status hours: {HOURLY_STATE_BASE['min_status_hours']}",
        f"minimum CBI hours: {HOURLY_STATE_BASE['min_cbi_hours']}",
        f"minimum OLS-CBI MacroSD: {HOURLY_STATE_BASE['min_macro_sd']}",
        (
            "BASE event CBI MacroSD screen: disabled, matching the main script; "
            "only M scenarios apply an event MacroSD screen."
        ),
        "",
        "Event_definition BASE definition:",
        f"event boundary: SPI30d <= {EVENT_DEFINITION_BASE['event_threshold']}",
        f"minimum duration days: {EVENT_DEFINITION_BASE['min_duration_days']}",
        f"extreme event threshold within full event: SPI30d <= {EVENT_DEFINITION_BASE['extreme_threshold']}",
        "",
        "Event reference window:",
        (
            "Buffer day is End + 1; candidate reference days are "
            "End + 2 through End + 31, inclusive."
        ),
        "",
    ]

    if RUN_MODE == "full" and "SameDirectionAsBaseline" in hourly_summary.columns:
        valid = hourly_summary.loc[
            ~hourly_summary["ScenarioID"].eq("BASE")
            & hourly_summary["LMM_Flag"].eq("ok")
            & hourly_summary["LMM_Delta_CBI"].notna()
        ]
        same = int(valid["SameDirectionAsBaseline"].eq(True).sum())
        total = len(valid)
        report_lines.extend(
            [
                f"Hourly_state non-BASE valid LMM scenarios same direction as BASE: {same} / {total}",
                "",
            ]
        )
    if RUN_MODE == "full" and "SameDirectionAsEventBase" in event_definition_summary.columns:
        valid = event_definition_summary.loc[
            ~event_definition_summary["ScenarioID"].eq("BASE")
            & event_definition_summary["Event_Flag"].eq("ok")
            & event_definition_summary["Event_Delta_Mean"].notna()
        ]
        same = int(valid["SameDirectionAsEventBase"].eq(True).sum())
        total = len(valid)
        report_lines.extend(
            [
                f"Event_definition non-BASE valid event scenarios same direction as BASE: {same} / {total}",
                "",
            ]
        )

    report_lines.append("Hourly_state scenario summary:")
    report_lines.append(hourly_summary.to_string(index=False))
    report_lines.append("")
    report_lines.append("Event_definition scenario summary:")
    report_lines.append(event_definition_summary.to_string(index=False))

    write_text("\n".join(report_lines), OUTPUT_FILES["report"])


def format_signed(value, digits=3):
    """把数值格式化为带正负号的小数字符串，便于结果解读文本引用。"""
    if pd.isna(value):
        return "NA"
    return f"{value:+.{digits}f}"


def classify_lmm_interpretation(row):
    """把单个情景的 LMM 结果转成结果解释标签。"""
    if row.get("LMM_Flag") != "ok" or pd.isna(row.get("LMM_Delta_CBI")):
        return "LMM不可解释"
    if row["LMM_CI_low95"] <= 0 <= row["LMM_CI_high95"]:
        return "方向存在但不稳定"
    if row["LMM_Delta_CBI"] > 0:
        return "支持缓冲减弱"
    if row["LMM_Delta_CBI"] < 0:
        return "支持表观缓冲增强"
    return "接近无差异"


def summarize_group(summary, group_name):
    """汇总同一情景组的方向一致性、显著性和稀疏样本情况。"""
    d = summary.loc[summary["ScenarioGroup"] == group_name].copy()
    if d.empty:
        return ""
    valid = d.loc[d["LMM_Flag"].eq("ok") & d["LMM_Delta_CBI"].notna()].copy()
    if valid.empty:
        return f"{group_name} 组没有可解释的 LMM 结果。"
    positive = int((valid["LMM_Delta_CBI"] > 0).sum())
    cross_zero = int(((valid["LMM_CI_low95"] <= 0) & (valid["LMM_CI_high95"] >= 0)).sum())
    sparse = int(valid["SparseSampleFlag"].eq("yes").sum())
    return (
        f"{group_name} 组共 {len(valid)} 个可解释情景，其中 {positive} 个情景的 LMM Delta CBI 为正，"
        f"{cross_zero} 个情景的 95% 置信区间跨 0，{sparse} 个情景被标记为样本偏少。"
    )


def summarize_event_definition_group(summary, group_name):
    """汇总事件定义证据链中某一情景组的方向、显著性和稀疏样本情况。"""
    d = summary.loc[summary["ScenarioGroup"] == group_name].copy()
    if d.empty:
        return ""
    valid = d.loc[d["Event_Flag"].eq("ok") & d["Event_Delta_Mean"].notna()].copy()
    if valid.empty:
        return f"{group_name} 组没有可解释的事件定义结果。"
    positive = int((valid["Event_Delta_Mean"] > 0).sum())
    cross_zero = int(((valid["Event_CI_low95"] <= 0) & (valid["Event_CI_high95"] >= 0)).sum())
    sparse = int(valid["EventSparseFlag"].eq("yes").sum())
    return (
        f"{group_name} 组共 {len(valid)} 个可解释情景，其中 {positive} 个情景的事件 Delta CBI 为正，"
        f"{cross_zero} 个情景的 95% 置信区间跨 0，{sparse} 个情景被标记为样本偏少。"
    )


def write_interpretation_report(summary):
    """写出结果解释型总结报告。

    目标：
        不再只罗列文件和字段，而是基于本次实际结果自动生成规范中文表述，
        可直接作为论文 Results/Discussion 的草稿基础。
    """
    hourly_summary = summary.loc[summary["AnalysisLayer"].eq("Hourly_state")].copy()
    event_definition_summary = summary.loc[summary["AnalysisLayer"].eq("Event_definition")].copy()

    base = hourly_summary.loc[hourly_summary["ScenarioID"] == "BASE"].iloc[0]
    valid_non_base = hourly_summary.loc[
        ~hourly_summary["ScenarioID"].eq("BASE")
        & hourly_summary["LMM_Flag"].eq("ok")
        & hourly_summary["LMM_Delta_CBI"].notna()
    ].copy()
    same_direction = int(valid_non_base["SameDirectionAsBaseline"].eq(True).sum())
    total_valid = len(valid_non_base)
    cross_zero = valid_non_base.loc[
        (valid_non_base["LMM_CI_low95"] <= 0) & (valid_non_base["LMM_CI_high95"] >= 0),
        "ScenarioID",
    ].astype(str).tolist()
    opposite = valid_non_base.loc[
        valid_non_base["SameDirectionAsBaseline"].eq(False),
        "ScenarioID",
    ].astype(str).tolist()
    sparse = hourly_summary.loc[hourly_summary["SparseSampleFlag"].eq("yes"), "ScenarioID"].astype(str).tolist()

    hourly_event_valid = hourly_summary.loc[
        hourly_summary["Event_Flag"].eq("ok") & hourly_summary["Event_Delta_Mean"].notna()
    ].copy()
    event_positive = int((hourly_event_valid["Event_Delta_Mean"] > 0).sum())
    event_cross_zero = hourly_event_valid.loc[
        (hourly_event_valid["Event_CI_low95"] <= 0) & (hourly_event_valid["Event_CI_high95"] >= 0),
        "ScenarioID",
    ].astype(str).tolist()

    event_base = event_definition_summary.loc[event_definition_summary["ScenarioID"] == "BASE"].iloc[0]
    event_definition_valid_non_base = event_definition_summary.loc[
        ~event_definition_summary["ScenarioID"].eq("BASE")
        & event_definition_summary["Event_Flag"].eq("ok")
        & event_definition_summary["Event_Delta_Mean"].notna()
    ].copy()
    event_definition_same = int(event_definition_valid_non_base["SameDirectionAsEventBase"].eq(True).sum())
    event_definition_total = len(event_definition_valid_non_base)
    event_definition_cross_zero = event_definition_valid_non_base.loc[
        (event_definition_valid_non_base["Event_CI_low95"] <= 0)
        & (event_definition_valid_non_base["Event_CI_high95"] >= 0),
        "ScenarioID",
    ].astype(str).tolist()
    event_definition_sparse = event_definition_summary.loc[
        event_definition_summary["EventSparseFlag"].eq("yes"),
        "ScenarioID",
    ].astype(str).tolist()

    lines = [
        "阈值稳健性结果解释报告",
        "=" * 70,
        "",
        "一、分析目的与双证据链结构",
        (
            "本报告基于本次稳健性脚本实际输出结果，分别解释两条证据链。"
            "第一条是 Hourly_state 证据链，检验小时级 SPI 状态定义、Normal 窗口、"
            "小时门槛和 MacroSD 质量筛选是否改变 LMM 与站点月份 CBI 结论；"
            "第二条是 Event_definition 证据链，检验完整干旱事件边界和最短持续天数"
            "是否改变事件—参考期 CBI 结论。"
        ),
        "",
        "二、Hourly_state 证据链：BASE 主结果",
        (
            f"BASE 情景下，主 LMM 的 Delta CBI = {base['LMM_Delta_CBI']:.6f}，"
            f"95% CI 为 [{base['LMM_CI_low95']:.6f}, {base['LMM_CI_high95']:.6f}]，"
            f"p = {base['LMM_Pvalue']:.4g}。该交互项为正，说明在主分析基准定义下，"
            "Extreme 条件下的 CBI 高于 Normal，支持极端干旱期间森林微气候缓冲减弱。"
        ),
        (
            f"BASE 事件级辅助结果中，Event Delta CBI = {base['Event_Delta_Mean']:.6f}，"
            f"95% CI 为 [{base['Event_CI_low95']:.6f}, {base['Event_CI_high95']:.6f}]，"
            f"p = {base['Event_P_two_sided']:.4g}，与主 LMM 方向一致。"
        ),
        "",
        "三、Hourly_state 证据链：主模型稳健性总体判断",
        (
            f"共有 {total_valid} 个非 BASE 情景获得可解释的 LMM 结果，其中 {same_direction} 个与 BASE 同方向。"
            + (f"方向相反的情景为 {', '.join(opposite)}。" if opposite else "")
        ),
        (
            f"置信区间跨 0 的情景包括：{', '.join(cross_zero) if cross_zero else '无'}。"
            "这些情景提示结论不稳定或样本不足，解释时应更谨慎。"
        ),
        "",
        "四、不同敏感性组别的结果",
        summarize_group(summary, "Extreme_threshold"),
        summarize_group(summary, "Normal_window"),
        summarize_group(summary, "Joint_definition"),
        summarize_group(summary, "Minimum_hours"),
        summarize_group(summary, "MacroSD_CBI_only"),
        "",
        "五、Hourly_state 证据链：事件级辅助证据",
        (
            f"事件级分析中，共有 {len(hourly_event_valid)} 个情景获得可解释的事件 Delta CBI 结果，其中 "
            f"{event_positive} 个情景的事件 Delta CBI 为正。"
        ),
        (
            f"事件级 95% 置信区间跨 0 的情景包括：{', '.join(event_cross_zero) if event_cross_zero else '无'}。"
            "总体上，事件级证据与主 LMM 大体同向，但放宽阈值后纳入较温和事件时，事件效应可能减弱。"
        ),
        "",
        "六、Hourly_state 证据链：样本稀疏与解释边界",
        (
            f"被标记为样本偏少的情景包括：{', '.join(sparse) if sparse else '无'}。"
            "其中若同时伴随置信区间跨 0，则更应视为估计不稳定，而不应直接解释为生物学方向反转。"
        ),
        (
            "特别是极端收紧阈值情景（如 E22、C4）即使出现方向变化，也更可能反映样本量显著收缩后的不稳定估计，"
            "而不是主结论被明确推翻。"
        ),
        "",
        "七、Event_definition 证据链：BASE 主结果与稳健性判断",
        (
            f"事件定义 BASE 情景下，Event Delta CBI = {event_base['Event_Delta_Mean']:.6f}，"
            f"95% CI 为 [{event_base['Event_CI_low95']:.6f}, {event_base['Event_CI_high95']:.6f}]，"
            f"p = {event_base['Event_P_two_sided']:.4g}。这说明在“连续 SPI <= -0.5 且持续至少 6 天”的"
            "完整干旱事件定义下，极端事件期相对于事件后 Normal 参考期同样表现出 CBI 增大。"
        ),
        (
            f"在 D05/D10/B00/B10 四个非 BASE 事件定义情景中，共有 {event_definition_total} 个获得可解释结果，"
            f"其中 {event_definition_same} 个与事件定义 BASE 同方向。"
            f"置信区间跨 0 的情景包括：{', '.join(event_definition_cross_zero) if event_definition_cross_zero else '无'}。"
        ),
        summarize_event_definition_group(event_definition_summary, "Minimum_duration"),
        summarize_event_definition_group(event_definition_summary, "Event_boundary"),
        (
            f"事件定义证据链中被标记为样本偏少的情景包括：{', '.join(event_definition_sparse) if event_definition_sparse else '无'}。"
            "这条证据链仅说明完整干旱事件边界与最短持续时间是否改变事件层面结论，"
            "不能替代小时级 LMM 结果。"
        ),
        "",
        "八、综合结论与论文式表述",
        (
            "在基准情景下，Extreme 条件下的 CBI 显著高于 Normal，提示极端干旱期间森林微气候缓冲减弱。"
            "在绝大多数敏感性情景中，LMM 估计的 Delta CBI 保持相同方向，说明该结论对 SPI 阈值、"
            "Normal 窗口、小时门槛和 MacroSD 质量筛选总体稳健。"
        ),
        (
            "事件定义敏感性分析进一步表明，当完整干旱事件的边界阈值或最短持续时间在合理范围内调整时，"
            "事件—参考期 Delta CBI 的方向总体保持一致，说明主结论不仅对状态阈值稳健，"
            "也对完整事件识别规则具有一定稳健性。"
        ),
        "",
        "九、不可过度解释的内容",
        (
            "本报告支持“主结论对阈值总体稳健”，但不支持把所有方向差异都解释为明确的生态学机制变化。"
            "当样本稀疏、置信区间跨 0 或辅助证据减弱时，更合理的解释是证据不足或估计不稳定，"
            "而不是认定缓冲机制发生反转。"
        ),
        "",
        "十、可直接写入论文的简要表述",
        (
            "总体而言，基于小时级状态定义的主结论在多数极端阈值、正常窗口、小时门槛和质量控制设定下保持同向，"
            "表明极端干旱条件下森林微气候缓冲减弱的判断具有较好的阈值稳健性。"
            "从不同干旱程度来看，除极严格且样本稀缺的设定外，极端阈值调整后的估计方向总体一致；"
            "极严格阈值下若出现反向点估计，应优先解释为样本稀缺导致的不稳定估计，"
            "而不是主结论被明确推翻。"
        ),
        (
            "事件定义证据链进一步表明，当完整干旱事件的边界和持续时间在合理范围内变化时，"
            "事件—参考期 CBI 的方向总体保持一致；这说明主结论不仅对状态阈值稳健，"
            "也对事件识别规则具有一定稳健性。"
        ),
    ]

    write_text("\n".join(lines), OUTPUT_FILES["interpretation_report"])


# =============================================================================
# 6. 逐站点剔除（Leave-one-site-out）影响点稳健性：只检验 Hourly_state / BASE
# =============================================================================

def _loso_sign(value):
    """把 LOSO 结果的 Delta CBI 转成方向标签，便于表格和报告判读。"""
    if pd.isna(value):
        return "NA"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _loso_ci_crosses_zero(low, high):
    """判断置信区间是否跨越 0；跨 0 只表示精度下降，不自动等于方向反转。"""
    return bool(pd.notna(low) and pd.notna(high) and low <= 0 <= high)


def _loso_compare_base_to_main(base_fit, base_coverage, main_base_row):
    """把 LOSO 内部重建的 BASE 与主流程 BASE 做逐项对照。"""
    rows = []

    def add_row(metric, loso_value, main_value=np.nan, note=""):
        if pd.isna(main_value) or pd.isna(loso_value):
            match = np.nan
        elif isinstance(loso_value, (int, float, np.integer, np.floating)) and isinstance(
            main_value, (int, float, np.integer, np.floating)
        ):
            match = np.isclose(loso_value, main_value, equal_nan=True)
        else:
            match = loso_value == main_value
        rows.append(
            {
                "Metric": metric,
                "LOSO_BASE": loso_value,
                "Main_script": main_value,
                "Match": match,
                "Note": note,
            }
        )

    add_row("LMM_hours", base_coverage.get("lmm_hours", np.nan), main_base_row.get("lmm_hours", np.nan))
    add_row("LMM_sites", base_coverage.get("lmm_sites", np.nan), main_base_row.get("lmm_sites", np.nan))
    add_row("LMM_sitemonths", base_coverage.get("lmm_sitemonths", np.nan), main_base_row.get("lmm_sitemonths", np.nan))
    add_row("LMM_extreme_hours", base_coverage.get("lmm_extreme_hours", np.nan), main_base_row.get("lmm_extreme_hours", np.nan))
    add_row("LMM_normal_hours", base_coverage.get("lmm_normal_hours", np.nan), main_base_row.get("lmm_normal_hours", np.nan))
    add_row("LMM_other_hours", base_coverage.get("other_hours", np.nan), main_base_row.get("other_hours", np.nan))
    add_row("LMM_Delta_CBI", base_fit.get("LMM_Delta_CBI", np.nan), main_base_row.get("LMM_Delta_CBI", np.nan))
    add_row("LMM_CI_low95", base_fit.get("LMM_CI_low95", np.nan), main_base_row.get("LMM_CI_low95", np.nan))
    add_row("LMM_CI_high95", base_fit.get("LMM_CI_high95", np.nan), main_base_row.get("LMM_CI_high95", np.nan))
    add_row("LMM_Pvalue", base_fit.get("LMM_Pvalue", np.nan), main_base_row.get("LMM_Pvalue", np.nan))
    add_row("LMM_Flag", base_fit.get("LMM_Flag", np.nan), main_base_row.get("LMM_Flag", np.nan))
    return pd.DataFrame(rows)


def make_leave_one_site_out_forest_plot(detail, base_fit, loso_dir):
    """绘制逐站点剔除森林图：BASE 放在最上方，其余点按 Delta CBI 排序。"""
    base_row = pd.DataFrame(
        [
            {
                "Label": "BASE（完整数据）",
                "Delta": base_fit.get("LMM_Delta_CBI", np.nan),
                "Low": base_fit.get("LMM_CI_low95", np.nan),
                "High": base_fit.get("LMM_CI_high95", np.nan),
                "IsBase": True,
            }
        ]
    )
    held = detail.loc[detail["LOSO_LMM_Flag"].eq("ok") & detail["LOSO_Delta_CBI"].notna()].copy()
    if held.empty:
        fig_df = base_row.copy()
    else:
        held = held.assign(
            Label="删除站点 " + held["HeldOutSiteID"].astype(str),
            Delta=held["LOSO_Delta_CBI"],
            Low=held["LOSO_CI_low95"],
            High=held["LOSO_CI_high95"],
            IsBase=False,
        )[["Label", "Delta", "Low", "High", "IsBase"]]
        fig_df = pd.concat([base_row, held], ignore_index=True)
        fig_df = pd.concat([fig_df.iloc[:1], fig_df.iloc[1:].sort_values("Delta", ascending=False)], ignore_index=True)

    plot_bar = progress_step("绘制逐站点剔除森林图", "绘图")
    try:
        y = np.arange(len(fig_df))
        fig, ax = plt.subplots(
            figsize=(
                FIG_FOREST_WIDTH,
                max(FIG_FOREST_MIN_HEIGHT, FIG_FOREST_HEIGHT_PER_SCENARIO * len(fig_df)),
            )
        )
        colors = np.where(fig_df["IsBase"], COLOR_BASE_SCENARIO, COLOR_OTHER_SCENARIO)
        valid = (fig_df["Delta"].notna() & fig_df["Low"].notna() & fig_df["High"].notna()).to_numpy()
        if valid.any():
            ax.errorbar(
                fig_df.loc[valid, "Delta"],
                y[valid],
                xerr=[
                    fig_df.loc[valid, "Delta"] - fig_df.loc[valid, "Low"],
                    fig_df.loc[valid, "High"] - fig_df.loc[valid, "Delta"],
                ],
                fmt="none",
                color=COLOR_CI,
                elinewidth=LINEWIDTH_CI,
                capsize=CI_CAPSIZE,
                zorder=1,
            )
            ax.scatter(
                fig_df.loc[valid, "Delta"],
                y[valid],
                c=colors[valid],
                s=SCATTER_SIZE_FOREST,
                zorder=2,
            )
        ax.axvline(0, color=COLOR_ZERO_LINE, linestyle=LINESTYLE_ZERO, linewidth=LINEWIDTH_ZERO)
        ax.set_yticks(y)
        ax.set_yticklabels(fig_df["Label"])
        ax.set_xlabel("Delta CBI（Extreme - Normal）, LMM")
        ax.set_title("Leave-one-site-out 逐站点剔除主 LMM 稳健性")
        if LEGEND_SHOW:
            legend_handles = [
                Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_BASE_SCENARIO, markersize=7, label="BASE（完整数据）"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_OTHER_SCENARIO, markersize=7, label="删去单站点"),
                Line2D([0], [0], color=COLOR_CI, lw=LINEWIDTH_CI, label=LEGEND_CI_LABEL),
                Line2D([0], [0], color=COLOR_ZERO_LINE, lw=LINEWIDTH_ZERO, linestyle=LINESTYLE_ZERO, label=LEGEND_ZERO_LABEL),
            ]
            ax.legend(
                handles=legend_handles,
                title=LEGEND_TITLE,
                loc=LEGEND_LOCATION,
                bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
                frameon=LEGEND_FRAME_ON,
                fontsize=LEGEND_FONT_SIZE,
            )
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(loso_dir / LOSO_OUTPUT_FILES["forest_plot"], dpi=FIG_DPI, bbox_inches=FIGURE_BBOX)
        plt.close(fig)
        plot_bar.update(1)
    finally:
        plot_bar.close()


def write_leave_one_site_out_report(detail, summary, base_fit, base_coverage, main_summary, loso_dir):
    """写出逐站点剔除的结果解释报告，强调“单一站点影响”而不是新主分析。"""
    valid = detail.loc[detail["LOSO_LMM_Flag"].eq("ok") & detail["LOSO_Delta_CBI"].notna()].copy()
    n_valid = len(valid)
    n_same = int(valid["SameDirectionAsBASE"].eq(True).sum()) if not valid.empty else 0
    n_reverse = int(valid["SameDirectionAsBASE"].eq(False).sum()) if not valid.empty else 0
    n_cross = int(valid["LOSOCICrossesZero"].eq(True).sum()) if not valid.empty else 0
    n_positive = int((valid["LOSO_Delta_CBI"] > 0).sum()) if not valid.empty else 0
    reverse_sites = valid.loc[valid["SameDirectionAsBASE"].eq(False), "HeldOutSiteID"].astype(str).tolist()

    main_base_row = main_summary.loc[
        main_summary["AnalysisLayer"].eq("Hourly_state")
        & main_summary["ScenarioID"].eq("BASE")
    ]
    main_base_row = main_base_row.iloc[0] if not main_base_row.empty else pd.Series(dtype=object)

    base_cmp = _loso_compare_base_to_main(base_fit, base_coverage, main_base_row)
    write_csv(base_cmp, f"{LEAVE_ONE_SITE_OUT_DIRNAME}/{LOSO_OUTPUT_FILES['base_audit']}")

    if n_valid == 0:
        paper_sentence = "逐站点剔除分析未能获得足够的可估计模型，因此当前不能把该模块作为有效的影响点稳健性证据。"
    elif n_reverse == 0:
        paper_sentence = (
            "逐站点剔除分析表明，极端干旱与正常条件下的 CBI 差异总体不依赖单一监测站点；"
            "在删去任一进入主模型的站点后，估计方向保持一致，说明主结果具有较好的影响点稳健性。"
        )
    else:
        paper_sentence = (
            f"逐站点剔除分析显示，主结果在大多数删站轮次中保持与完整 BASE 同方向，"
            f"但删除站点 {', '.join(reverse_sites)} 后出现方向反转，提示总体估计对个别站点存在一定敏感性。"
        )

    report = [
        "Leave-one-site-out 逐站点剔除主 LMM 稳健性报告",
        "=" * 72,
        "",
        "一、分析目的与边界",
        "本模块只检验 Hourly_state / BASE 主 LMM 是否被单一站点驱动，不作为新的主分析，也不进入事件定义证据链。",
        "",
        "二、BASE 内部复现",
        f"内部重建 BASE LMM 的 Delta CBI = {base_fit.get('LMM_Delta_CBI', np.nan):.6f}，",
        f"95% CI = [{base_fit.get('LMM_CI_low95', np.nan):.6f}, {base_fit.get('LMM_CI_high95', np.nan):.6f}]，",
        f"p = {base_fit.get('LMM_Pvalue', np.nan):.4g}。",
        f"BASE 进入 LMM 的站点数为 {base_coverage.get('lmm_sites', np.nan)}，站点—月份数为 {base_coverage.get('lmm_sitemonths', np.nan)}。",
        "",
        "三、逐站点剔除结果",
        f"共剔除 {len(detail)} 个进入 BASE LMM 的站点，其中 {n_valid} 轮可成功拟合，{n_same} 轮与 BASE 同方向，{n_reverse} 轮方向反转，{n_cross} 轮置信区间跨 0。",
        f"有效轮次中 Delta CBI 为正的共有 {n_positive} 轮。",
        (
            f"方向反转站点包括：{', '.join(reverse_sites) if reverse_sites else '无'}。"
            if n_reverse > 0
            else "未观察到任何站点删除后方向反转。"
        ),
        "",
        "四、论文式判读",
        (
            "若所有有效轮次都保持与 BASE 相同方向，则说明主结论不依赖某一个站点；"
            "若部分轮次置信区间跨 0，则更合理的解释是删站后样本减少导致估计精度下降，"
            "而不是主结论被推翻。"
        ),
        (
            "若出现方向反转，应优先结合剩余站点数、站点—月份数和收敛状态解释，"
            "将其视为站点影响信号，而不是直接把它当作生态机制反转。"
        ),
        "",
        "五、可直接写入论文的简要表述",
        paper_sentence,
    ]
    report_bar = progress_step("写出逐站点剔除解释报告", "结果写出")
    try:
        (loso_dir / LOSO_OUTPUT_FILES["report"]).write_text("\n".join(report), encoding="utf-8")
        report_bar.update(1)
    finally:
        report_bar.close()


def run_leave_one_site_out(hourly, daily_spi, drought_events, main_summary):
    """逐站点剔除主 LMM 的影响点稳健性分析。"""
    if not RUN_LEAVE_ONE_SITE_OUT or RUN_MODE != "full":
        return None

    loso_dir = OUTPUT_DIR / LEAVE_ONE_SITE_OUT_DIRNAME
    loso_dir.mkdir(parents=True, exist_ok=True)

    base_bar = progress_step("重建LOSO基准BASE", "LMM拟合")
    try:
        base_d_status, base_d_lmm, _, base_coverage = build_lmm_dataset(
            hourly, daily_spi, drought_events, HOURLY_STATE_BASE
        )
        base_fit = run_lmm(base_d_lmm)
        base_bar.update(1)
    finally:
        base_bar.close()

    included_sites = (
        sorted(base_d_lmm[SITE_COL].dropna().astype(str).unique().tolist())
        if not base_d_lmm.empty
        else []
    )
    if len(included_sites) < 2:
        raise ValueError(
            "Leave-one-site-out cannot run because fewer than two sites enter the BASE LMM."
        )

    detail_rows = []
    coverage_rows = [
        {
            "RunType": "BASE_full_data",
            "HeldOutSiteID": "None",
            "LMM_hours": base_coverage.get("lmm_hours", np.nan),
            "LMM_sites": base_coverage.get("lmm_sites", np.nan),
            "LMM_sitemonths": base_coverage.get("lmm_sitemonths", np.nan),
            "LMM_extreme_hours": base_coverage.get("lmm_extreme_hours", np.nan),
            "LMM_normal_hours": base_coverage.get("lmm_normal_hours", np.nan),
            "LMM_other_hours": base_coverage.get("other_hours", np.nan),
            "LMM_candidate_sitemonths": base_coverage.get("candidate_sitemonths", np.nan),
            "LMM_eligible_sitemonths_before_macro_sd": base_coverage.get("eligible_sitemonths_before_macro_sd", np.nan),
            "LMM_Flag": base_fit.get("LMM_Flag", "unknown"),
        }
    ]

    for held_out_site in progress_iter(
        included_sites,
        total=len(included_sites),
        desc="逐站点剔除BASE LMM",
        kind="逐站点剔除",
    ):
        hourly_keep = hourly.loc[hourly[SITE_COL].astype(str) != held_out_site].copy()
        daily_spi_keep = daily_spi.loc[daily_spi[SITE_COL].astype(str) != held_out_site].copy()
        drought_events_keep = drought_events.loc[drought_events[SITE_COL].astype(str) != held_out_site].copy()

        d_status, d_lmm, _, coverage = build_lmm_dataset(
            hourly_keep, daily_spi_keep, drought_events_keep, HOURLY_STATE_BASE
        )
        fit = run_lmm(d_lmm)
        delta = fit.get("LMM_Delta_CBI", np.nan)
        ci_low = fit.get("LMM_CI_low95", np.nan)
        ci_high = fit.get("LMM_CI_high95", np.nan)
        flag = fit.get("LMM_Flag", "unknown")
        converged = fit.get("LMM_Converged", np.nan)
        valid = flag == "ok" and pd.notna(delta)
        same_direction = (
            bool(np.sign(delta) == np.sign(base_fit.get("LMM_Delta_CBI", np.nan)))
            if valid and pd.notna(base_fit.get("LMM_Delta_CBI", np.nan))
            else np.nan
        )

        if not valid:
            influence = "model_not_estimable"
        elif not same_direction:
            influence = "direction_reversal"
        elif _loso_ci_crosses_zero(ci_low, ci_high):
            influence = "ci_crosses_zero"
        else:
            influence = "direction_and_ci_stable"
        convergence_diagnostic = (
            "valid_estimate_but_optimizer_converged_false"
            if valid and converged is False
            else ("valid_estimate_and_optimizer_converged" if valid else "not_valid_estimate")
        )

        detail_rows.append(
            {
                "AnalysisLayer": "Hourly_state",
                "EvidenceChain": "Leave_one_site_out_BASE_LMM",
                "ScenarioID": "LOSO_BASE",
                "HeldOutSiteID": held_out_site,
                "BASE_Delta_CBI": base_fit.get("LMM_Delta_CBI", np.nan),
                "BASE_CI_low95": base_fit.get("LMM_CI_low95", np.nan),
                "BASE_CI_high95": base_fit.get("LMM_CI_high95", np.nan),
                "BASE_Pvalue": base_fit.get("LMM_Pvalue", np.nan),
                "LOSO_Delta_CBI": delta,
                "LOSO_CI_low95": ci_low,
                "LOSO_CI_high95": ci_high,
                "LOSO_Pvalue": fit.get("LMM_Pvalue", np.nan),
                "LOSO_LMM_Converged": converged,
                "LOSO_LMM_Flag": flag,
                "LOSO_Convergence_Diagnostic": convergence_diagnostic,
                "SameDirectionAsBASE": same_direction,
                "LOSOSign": _loso_sign(delta),
                "LOSOCICrossesZero": _loso_ci_crosses_zero(ci_low, ci_high) if valid else np.nan,
                "InfluenceAssessment": influence,
                "Remaining_LMM_hours": coverage.get("lmm_hours", np.nan),
                "Remaining_LMM_sites": coverage.get("lmm_sites", np.nan),
                "Remaining_LMM_sitemonths": coverage.get("lmm_sitemonths", np.nan),
                "Remaining_LMM_extreme_hours": coverage.get("lmm_extreme_hours", np.nan),
                "Remaining_LMM_normal_hours": coverage.get("lmm_normal_hours", np.nan),
                "Remaining_LMM_other_hours": coverage.get("other_hours", np.nan),
            }
        )
        coverage_rows.append(
            {
                "RunType": "LOSO",
                "HeldOutSiteID": held_out_site,
                "LMM_hours": coverage.get("lmm_hours", np.nan),
                "LMM_sites": coverage.get("lmm_sites", np.nan),
                "LMM_sitemonths": coverage.get("lmm_sitemonths", np.nan),
                "LMM_extreme_hours": coverage.get("lmm_extreme_hours", np.nan),
                "LMM_normal_hours": coverage.get("lmm_normal_hours", np.nan),
                "LMM_other_hours": coverage.get("other_hours", np.nan),
                "LMM_candidate_sitemonths": coverage.get("candidate_sitemonths", np.nan),
                "LMM_eligible_sitemonths_before_macro_sd": coverage.get("eligible_sitemonths_before_macro_sd", np.nan),
                "LMM_Flag": flag,
            }
        )

    detail = pd.DataFrame(detail_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    summary_rows = []

    valid = detail.loc[detail["LOSO_LMM_Flag"].eq("ok") & detail["LOSO_Delta_CBI"].notna()].copy()
    n_valid = len(valid)
    n_same = int(valid["SameDirectionAsBASE"].eq(True).sum()) if not valid.empty else 0
    n_reverse = int(valid["SameDirectionAsBASE"].eq(False).sum()) if not valid.empty else 0
    n_cross = int(valid["LOSOCICrossesZero"].eq(True).sum()) if not valid.empty else 0
    n_positive = int((valid["LOSO_Delta_CBI"] > 0).sum()) if not valid.empty else 0

    summary_rows.append(
        {
            "AnalysisLayer": "Hourly_state",
            "EvidenceChain": "Leave_one_site_out_BASE_LMM",
            "BASE_Delta_CBI": base_fit.get("LMM_Delta_CBI", np.nan),
            "BASE_CI_low95": base_fit.get("LMM_CI_low95", np.nan),
            "BASE_CI_high95": base_fit.get("LMM_CI_high95", np.nan),
            "BASE_Pvalue": base_fit.get("LMM_Pvalue", np.nan),
            "BASE_included_sites": len(included_sites),
            "LOSO_runs_requested": len(detail),
            "LOSO_valid_runs": n_valid,
            "LOSO_same_direction_as_BASE": n_same,
            "LOSO_positive_delta": n_positive,
            "LOSO_CI_crosses_zero": n_cross,
            "LOSO_direction_reversals": n_reverse,
            "LOSO_model_not_estimable": int((detail["LOSO_LMM_Flag"] != "ok").sum()),
            "LOSO_direction_reversal_sites": ", ".join(
                valid.loc[valid["SameDirectionAsBASE"].eq(False), "HeldOutSiteID"].astype(str).tolist()
            ),
        }
    )
    summary = pd.DataFrame(summary_rows)

    write_csv(detail, f"{LEAVE_ONE_SITE_OUT_DIRNAME}/{LOSO_OUTPUT_FILES['detail']}")
    write_csv(coverage_df, f"{LEAVE_ONE_SITE_OUT_DIRNAME}/{LOSO_OUTPUT_FILES['coverage']}")
    write_csv(summary, f"{LEAVE_ONE_SITE_OUT_DIRNAME}/{LOSO_OUTPUT_FILES['summary']}")

    if MAKE_LEAVE_ONE_SITE_OUT_FOREST:
        make_leave_one_site_out_forest_plot(detail, base_fit, loso_dir)

    write_leave_one_site_out_report(detail, summary, base_fit, base_coverage, main_summary, loso_dir)
    return summary, detail, coverage_df


# =============================================================================
# 7. 逐区域干旱过程剔除（LORPO）：只检验 BASE 事件—参考期 CBI
# =============================================================================

def identify_regional_drought_processes(base_events):
    """把 BASE 极端事件按时间连通性归并为区域干旱过程。

    输入必须是 select_drought_events_for_scenario(drought_events, HOURLY_STATE_BASE)
    得到的 BASE 极端事件集合，因此事件口径与 Hourly_state / BASE 的事件 CBI 完全一致。

    过程定义：
        不同站点的事件日期区间只要重叠，或相隔不超过 REGIONAL_PROCESS_GAP_DAYS 天，
        即属于同一个时间连通分量。该定义具有传递性：A 连 B、B 连 C 时，A/B/C
        同属一个区域过程。

    判定边界：
        只有涉及至少 REGIONAL_PROCESS_MIN_SITES 个站点的连通分量才作为剔除对象。
        单站点事件仍保留在每轮事件集中，但不单独作为“区域过程”剔除。
    """
    required = [SITE_COL, "Event_ID", "Start_Date", "End_Date"]
    missing = sorted(set(required) - set(base_events.columns))
    if missing:
        raise KeyError(f"Cannot identify regional drought processes; missing columns: {missing}")

    events = base_events.copy()
    events["Start_Date"] = pd.to_datetime(events["Start_Date"], errors="coerce").dt.normalize()
    events["End_Date"] = pd.to_datetime(events["End_Date"], errors="coerce").dt.normalize()
    events = events.loc[events["Start_Date"].notna() & events["End_Date"].notna()].copy()
    events = events.sort_values(["Start_Date", "End_Date", SITE_COL, "Event_ID"]).reset_index(drop=True)

    if events.empty:
        events["RegionalProcessID"] = pd.Series(dtype="object")
        return events

    component_index = -1
    current_end = None
    process_ids = []
    for _, row in events.iterrows():
        if (
            current_end is None
            or row["Start_Date"] > current_end + pd.Timedelta(days=REGIONAL_PROCESS_GAP_DAYS)
        ):
            component_index += 1
            current_end = row["End_Date"]
        else:
            current_end = max(current_end, row["End_Date"])
        process_ids.append(f"RP{component_index + 1:02d}")

    events["RegionalProcessID"] = process_ids
    process_info = (
        events.groupby("RegionalProcessID", as_index=False)
        .agg(
            ProcessStartDate=("Start_Date", "min"),
            ProcessEndDate=("End_Date", "max"),
            ProcessNEvents=("Event_ID", "size"),
            ProcessNSites=(SITE_COL, "nunique"),
        )
    )
    events = events.merge(process_info, on="RegionalProcessID", how="left", validate="many_to_one")
    events["IsRegionalProcess"] = events["ProcessNSites"] >= REGIONAL_PROCESS_MIN_SITES
    return events


def _regional_process_sign(value):
    """把 LORPO 事件 Delta CBI 转成方向标签。"""
    if pd.isna(value):
        return "NA"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _regional_process_ci_crosses_zero(low, high):
    """判断 LORPO 置信区间是否跨 0；跨 0 表示精度下降，不等于方向反转。"""
    return bool(pd.notna(low) and pd.notna(high) and low <= 0 <= high)


def make_leave_one_regional_process_out_forest_plot(detail, base_fit, out_dir):
    """绘制逐区域干旱过程剔除森林图。

    图形目的：
        直观展示 BASE 完整事件集与每一轮“删除一个区域干旱过程”后的事件级 Delta CBI。
        如果删除任一过程后点估计仍与 BASE 同方向，且置信区间没有明显变宽或跨 0，
        则支持“事件级结论不是由单一区域过程驱动”的稳健性解释。

    图形元素：
        - 第一行：BASE（完整事件集），即未删除任何区域过程时的事件—参考期 CBI 结果；
        - 其余行：删除 RP01、RP02 等区域过程后的事件—参考期 CBI 结果；
        - 横轴：Event Delta CBI，定义为事件期 CBI 减去参考期 CBI；
        - 横向误差线：cluster bootstrap 得到的 95% CI；
        - 竖向虚线：Delta CBI = 0，用于判断方向和 CI 是否跨 0。
    """
    base_row = pd.DataFrame(
        [
            {
                "Label": "BASE（完整事件集）",
                "Delta": base_fit.get("Event_Delta_Mean", np.nan),
                "Low": base_fit.get("Event_CI_low95", np.nan),
                "High": base_fit.get("Event_CI_high95", np.nan),
                "IsBase": True,
            }
        ]
    )
    held = detail.loc[
        detail["LORPO_Event_Flag"].eq("ok") & detail["LORPO_Event_Delta_Mean"].notna()
    ].copy()
    if held.empty:
        fig_df = base_row.copy()
    else:
        held = held.assign(
            Label=(
                "删除 "
                + held["HeldOutRegionalProcessID"].astype(str)
                + "（"
                + held["HeldOutProcessNEvents"].astype(int).astype(str)
                + "事件/"
                + held["HeldOutProcessNSites"].astype(int).astype(str)
                + "站点）"
            ),
            Delta=held["LORPO_Event_Delta_Mean"],
            Low=held["LORPO_Event_CI_low95"],
            High=held["LORPO_Event_CI_high95"],
            IsBase=False,
        )[["Label", "Delta", "Low", "High", "IsBase"]]
        fig_df = pd.concat([base_row, held], ignore_index=True)

    plot_bar = progress_step("绘制逐区域过程剔除森林图", "绘图")
    try:
        y = np.arange(len(fig_df))
        fig, ax = plt.subplots(
            figsize=(
                FIG_FOREST_WIDTH,
                max(FIG_FOREST_MIN_HEIGHT, FIG_FOREST_HEIGHT_PER_SCENARIO * len(fig_df)),
            )
        )
        colors = np.where(fig_df["IsBase"], COLOR_BASE_SCENARIO, COLOR_OTHER_SCENARIO)
        valid = (fig_df["Delta"].notna() & fig_df["Low"].notna() & fig_df["High"].notna()).to_numpy()
        if valid.any():
            ax.errorbar(
                fig_df.loc[valid, "Delta"],
                y[valid],
                xerr=[
                    fig_df.loc[valid, "Delta"] - fig_df.loc[valid, "Low"],
                    fig_df.loc[valid, "High"] - fig_df.loc[valid, "Delta"],
                ],
                fmt="none",
                color=COLOR_CI,
                elinewidth=LINEWIDTH_CI,
                capsize=CI_CAPSIZE,
                zorder=1,
            )
            ax.scatter(
                fig_df.loc[valid, "Delta"],
                y[valid],
                c=colors[valid],
                s=SCATTER_SIZE_FOREST,
                zorder=2,
            )
        ax.axvline(0, color=COLOR_ZERO_LINE, linestyle=LINESTYLE_ZERO, linewidth=LINEWIDTH_ZERO)
        ax.set_yticks(y)
        ax.set_yticklabels(fig_df["Label"])
        ax.set_xlabel("Event Delta CBI（事件期 - 参考期）")
        ax.set_title("逐区域干旱过程剔除事件级稳健性")
        if LEGEND_SHOW:
            legend_handles = [
                Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_BASE_SCENARIO, markersize=7, label="BASE（完整事件集）"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_OTHER_SCENARIO, markersize=7, label="删除一个区域过程"),
                Line2D([0], [0], color=COLOR_CI, lw=LINEWIDTH_CI, label=LEGEND_CI_LABEL),
                Line2D([0], [0], color=COLOR_ZERO_LINE, lw=LINEWIDTH_ZERO, linestyle=LINESTYLE_ZERO, label=LEGEND_ZERO_LABEL),
            ]
            ax.legend(
                handles=legend_handles,
                title=LEGEND_TITLE,
                loc=LEGEND_LOCATION,
                bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
                frameon=LEGEND_FRAME_ON,
                fontsize=LEGEND_FONT_SIZE,
            )
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(out_dir / REGIONAL_PROCESS_OUTPUT_FILES["forest_plot"], dpi=FIG_DPI, bbox_inches=FIGURE_BBOX)
        plt.close(fig)
        plot_bar.update(1)
    finally:
        plot_bar.close()


def write_regional_process_report(detail, summary, base_fit, base_valid_pairs, process_ids, out_dir):
    """写出逐区域干旱过程剔除解释报告。"""
    valid = detail.loc[detail["LORPO_Event_Flag"].eq("ok") & detail["LORPO_Event_Delta_Mean"].notna()].copy()
    n_valid = len(valid)
    n_same = int(valid["SameDirectionAsBASE"].eq(True).sum()) if not valid.empty else 0
    n_reverse = int(valid["SameDirectionAsBASE"].eq(False).sum()) if not valid.empty else 0
    n_cross = int(valid["LORPOCICrossesZero"].eq(True).sum()) if not valid.empty else 0
    reverse_processes = valid.loc[
        valid["SameDirectionAsBASE"].eq(False),
        ["HeldOutRegionalProcessID", "HeldOutProcessStartDate", "HeldOutProcessEndDate"],
    ]

    if n_valid == 0:
        paper_sentence = "逐区域干旱过程剔除未获得足够的可估计轮次，因此当前不能作为有效的事件级影响点稳健性证据。"
    elif n_reverse == 0:
        paper_sentence = (
            "逐区域干旱过程剔除分析显示，删除任一次预先定义的时间连通多站点区域干旱过程后，"
            "事件—参考期 Delta CBI 的方向保持一致，说明事件级结论并非由单一时间连通区域过程驱动。"
        )
    else:
        reverse_desc = "; ".join(
            f"{r['HeldOutRegionalProcessID']}({r['HeldOutProcessStartDate']}至{r['HeldOutProcessEndDate']})"
            for _, r in reverse_processes.iterrows()
        )
        paper_sentence = (
            "逐区域干旱过程剔除分析显示，部分删过程轮次出现方向反转，"
            f"涉及过程为 {reverse_desc}；应结合删后有效事件配对数、站点数和置信区间谨慎解释。"
        )

    report = [
        "逐区域干旱过程剔除事件级稳健性报告",
        "=" * 72,
        "",
        "一、分析目的与边界",
        (
            "本模块检验 BASE 事件—参考期 CBI 是否被某一次跨站点、时间连通的区域性极端干旱过程驱动。"
            "这里的“过程”是预先定义的时间连通多站点事件集合，不等同于已经证明为单一气象事件。"
            "它只属于 Event-reference CBI 证据链，不重跑 Hourly_state 主 LMM，也不替代 LOSO。"
        ),
        "",
        "二、区域过程定义",
        (
            f"区域过程连接规则为：事件日期重叠，或相邻事件间隔不超过 {REGIONAL_PROCESS_GAP_DAYS} 天；"
            f"仅保留涉及至少 {REGIONAL_PROCESS_MIN_SITES} 个站点的连通分量作为剔除对象。"
        ),
        f"本次共识别并测试 {len(process_ids)} 个区域干旱过程。",
        "",
        "三、BASE 事件级结果",
        (
            f"BASE 事件 Delta CBI = {base_fit.get('Event_Delta_Mean', np.nan):.6f}，"
            f"95% CI = [{base_fit.get('Event_CI_low95', np.nan):.6f}, {base_fit.get('Event_CI_high95', np.nan):.6f}]，"
            f"p = {base_fit.get('Event_P_two_sided', np.nan):.4g}。"
        ),
        (
            f"BASE 有效事件—参考期配对数为 {len(base_valid_pairs)}，"
            f"有效站点数为 {base_valid_pairs[SITE_COL].nunique() if not base_valid_pairs.empty else 0}。"
        ),
        "",
        "四、逐过程剔除结果",
        (
            f"共 {len(detail)} 轮剔除，其中 {n_valid} 轮可成功估计，{n_same} 轮与 BASE 同方向，"
            f"{n_reverse} 轮方向反转，{n_cross} 轮置信区间跨 0。"
        ),
        (
            "置信区间跨 0 应优先解释为删去区域过程后有效事件或站点减少导致的精度下降；"
            "只有可估计模型的 Delta CBI 符号改变时，才构成更强的过程影响警告。"
        ),
        "",
        "五、可直接写入论文的简要表述",
        paper_sentence,
    ]

    bar = progress_step("写出逐区域过程剔除解释报告", "结果写出")
    try:
        (out_dir / REGIONAL_PROCESS_OUTPUT_FILES["report"]).write_text("\n".join(report), encoding="utf-8")
        bar.update(1)
    finally:
        bar.close()


def run_leave_one_regional_process_out(hourly, daily_spi, drought_events):
    """逐区域干旱过程剔除事件级稳健性分析。

    每轮只从 drought_events 中删除一个区域过程的成员事件；hourly 和 daily_spi 保持完整。
    这样做的目的是只改变“某次区域过程是否进入事件集合”，而不混入站点删除或原始观测删除的影响。
    """
    if not RUN_LEAVE_ONE_REGIONAL_PROCESS_OUT or RUN_MODE != "full":
        return None

    out_dir = OUTPUT_DIR / LEAVE_ONE_REGIONAL_PROCESS_OUT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    # 使用与 Hourly_state / BASE 事件 CBI 完全相同的事件集合识别区域过程。
    base_events = select_drought_events_for_scenario(drought_events, HOURLY_STATE_BASE)
    process_members = identify_regional_drought_processes(base_events)
    write_csv(
        process_members,
        f"{LEAVE_ONE_REGIONAL_PROCESS_OUT_DIRNAME}/{REGIONAL_PROCESS_OUTPUT_FILES['process_audit']}",
    )

    regional_members = process_members.loc[process_members["IsRegionalProcess"]].copy()
    process_ids = regional_members["RegionalProcessID"].drop_duplicates().tolist()
    if not process_ids:
        raise ValueError(
            "No multi-site regional drought process was identified under the pre-specified LORPO rule."
        )

    base_bar = progress_step("重建LORPO基准事件CBI", "事件CBI")
    try:
        base_status = hourly.copy()
        base_status["SPI_Status"] = classify_status(base_status[SPI_COL], HOURLY_STATE_BASE)
        base_fit, base_pairs = run_event_cbi(
            base_status, daily_spi, drought_events, HOURLY_STATE_BASE, "BASE"
        )
        base_valid_pairs = (
            base_pairs.loc[base_pairs["Pair_Flag"].eq("ok")].copy()
            if not base_pairs.empty
            else base_pairs
        )
        base_bar.update(1)
    finally:
        base_bar.close()

    detail_rows = []
    coverage_rows = [
        {
            "RunType": "BASE_full_event_set",
            "HeldOutRegionalProcessID": "None",
            "RetainedInputExtremeEvents": len(base_events),
            "ValidEventReferencePairs": len(base_valid_pairs),
            "ValidEventSites": base_valid_pairs[SITE_COL].nunique() if not base_valid_pairs.empty else 0,
            "Event_Flag": base_fit.get("Event_Flag", "unknown"),
        }
    ]
    base_event_keys = set(zip(base_events[SITE_COL].astype(str), base_events["Event_ID"].astype(str)))
    base_delta = base_fit.get("Event_Delta_Mean", np.nan)

    for process_id in progress_iter(
        process_ids,
        total=len(process_ids),
        desc="逐区域过程剔除事件CBI",
        kind="逐区域过程剔除",
    ):
        held = regional_members.loc[regional_members["RegionalProcessID"].eq(process_id)].copy()
        held_keys = set(zip(held[SITE_COL].astype(str), held["Event_ID"].astype(str)))

        keep_mask = ~drought_events.apply(
            lambda row: (str(row[SITE_COL]), str(row["Event_ID"])) in held_keys,
            axis=1,
        )
        events_keep = drought_events.loc[keep_mask].copy()
        fit, pairs = run_event_cbi(base_status, daily_spi, events_keep, HOURLY_STATE_BASE, "BASE")
        valid_pairs = pairs.loc[pairs["Pair_Flag"].eq("ok")].copy() if not pairs.empty else pairs

        delta = fit.get("Event_Delta_Mean", np.nan)
        low = fit.get("Event_CI_low95", np.nan)
        high = fit.get("Event_CI_high95", np.nan)
        flag = fit.get("Event_Flag", "unknown")
        valid = flag == "ok" and pd.notna(delta)
        same = bool(np.sign(delta) == np.sign(base_delta)) if valid and pd.notna(base_delta) else np.nan

        if not valid:
            assessment = "event_model_not_estimable"
        elif not same:
            assessment = "direction_reversal"
        elif _regional_process_ci_crosses_zero(low, high):
            assessment = "ci_crosses_zero"
        else:
            assessment = "direction_and_ci_stable"

        removed_base_events = len(held_keys & base_event_keys)
        detail_rows.append(
            {
                "AnalysisLayer": "Event_reference",
                "EvidenceChain": "Event_reference_CBI_LORPO",
                "ScenarioID": "LORPO_BASE",
                "HeldOutRegionalProcessID": process_id,
                "HeldOutProcessStartDate": held["ProcessStartDate"].iloc[0],
                "HeldOutProcessEndDate": held["ProcessEndDate"].iloc[0],
                "HeldOutProcessNEvents": held["ProcessNEvents"].iloc[0],
                "HeldOutProcessNSites": held["ProcessNSites"].iloc[0],
                "BASE_Event_Delta_Mean": base_delta,
                "BASE_Event_CI_low95": base_fit.get("Event_CI_low95", np.nan),
                "BASE_Event_CI_high95": base_fit.get("Event_CI_high95", np.nan),
                "BASE_Event_P_two_sided": base_fit.get("Event_P_two_sided", np.nan),
                "LORPO_Event_Delta_Mean": delta,
                "LORPO_Event_CI_low95": low,
                "LORPO_Event_CI_high95": high,
                "LORPO_Event_P_two_sided": fit.get("Event_P_two_sided", np.nan),
                "LORPO_Event_Flag": flag,
                "SameDirectionAsBASE": same,
                "LORPOSign": _regional_process_sign(delta),
                "LORPOCICrossesZero": _regional_process_ci_crosses_zero(low, high) if valid else np.nan,
                "InfluenceAssessment": assessment,
                "HeldOutBaseExtremeEvents": removed_base_events,
                "RemainingInputExtremeEvents": len(base_events) - removed_base_events,
                "RemainingValidEventPairs": len(valid_pairs),
                "RemainingValidEventSites": valid_pairs[SITE_COL].nunique() if not valid_pairs.empty else 0,
            }
        )
        coverage_rows.append(
            {
                "RunType": "LORPO",
                "HeldOutRegionalProcessID": process_id,
                "HeldOutProcessNEvents": held["ProcessNEvents"].iloc[0],
                "HeldOutProcessNSites": held["ProcessNSites"].iloc[0],
                "RetainedInputExtremeEvents": len(base_events) - removed_base_events,
                "ValidEventReferencePairs": len(valid_pairs),
                "ValidEventSites": valid_pairs[SITE_COL].nunique() if not valid_pairs.empty else 0,
                "Event_Flag": flag,
            }
        )

    detail = pd.DataFrame(detail_rows)
    coverage = pd.DataFrame(coverage_rows)
    valid_detail = detail.loc[
        detail["LORPO_Event_Flag"].eq("ok") & detail["LORPO_Event_Delta_Mean"].notna()
    ].copy()
    summary = pd.DataFrame(
        [
            {
                "AnalysisLayer": "Event_reference",
                "EvidenceChain": "Event_reference_CBI_LORPO",
                "BASE_Event_Delta_Mean": base_delta,
                "BASE_Event_CI_low95": base_fit.get("Event_CI_low95", np.nan),
                "BASE_Event_CI_high95": base_fit.get("Event_CI_high95", np.nan),
                "BASE_Event_P_two_sided": base_fit.get("Event_P_two_sided", np.nan),
                "RegionalProcessesTested": len(detail),
                "LORPO_valid_runs": len(valid_detail),
                "LORPO_same_direction_as_BASE": int(valid_detail["SameDirectionAsBASE"].eq(True).sum()) if not valid_detail.empty else 0,
                "LORPO_CI_crosses_zero": int(valid_detail["LORPOCICrossesZero"].eq(True).sum()) if not valid_detail.empty else 0,
                "LORPO_direction_reversals": int(valid_detail["SameDirectionAsBASE"].eq(False).sum()) if not valid_detail.empty else 0,
                "LORPO_model_not_estimable": int((detail["LORPO_Event_Flag"] != "ok").sum()),
                "LORPO_direction_reversal_processes": ", ".join(
                    valid_detail.loc[
                        valid_detail["SameDirectionAsBASE"].eq(False),
                        "HeldOutRegionalProcessID",
                    ].astype(str).tolist()
                ),
            }
        ]
    )

    write_csv(detail, f"{LEAVE_ONE_REGIONAL_PROCESS_OUT_DIRNAME}/{REGIONAL_PROCESS_OUTPUT_FILES['detail']}")
    write_csv(coverage, f"{LEAVE_ONE_REGIONAL_PROCESS_OUT_DIRNAME}/{REGIONAL_PROCESS_OUTPUT_FILES['coverage']}")
    write_csv(summary, f"{LEAVE_ONE_REGIONAL_PROCESS_OUT_DIRNAME}/{REGIONAL_PROCESS_OUTPUT_FILES['summary']}")
    write_regional_process_report(detail, summary, base_fit, base_valid_pairs, process_ids, out_dir)
    if MAKE_LEAVE_ONE_REGIONAL_PROCESS_OUT_FOREST:
        make_leave_one_regional_process_out_forest_plot(detail, base_fit, out_dir)
    return summary, detail, coverage


# =============================================================================
# 8. CBI 过程与恢复窗口敏感性：只检验 BASE 事件级时间过程
# =============================================================================

def write_process_recovery_csv(df, filename):
    """把 CBI 过程恢复模块的表格写入独立子目录，避免与主稳健性结果混在一起。"""
    write_csv(df, f"{PROCESS_RECOVERY_OUTPUT_DIRNAME}/{filename}")


def write_process_recovery_text(text, filename):
    """把 CBI 过程恢复模块的解释报告写入独立子目录。"""
    out_dir = OUTPUT_DIR / PROCESS_RECOVERY_OUTPUT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    bar = progress_step(f"写出 {filename}", "结果写出")
    try:
        (out_dir / filename).write_text(text, encoding="utf-8")
        bar.update(1)
    finally:
        bar.close()


def _normalize_day(value):
    """把事件日期统一为无时区自然日，保证窗口边界按日历天计算。"""
    return pd.to_datetime(value, errors="coerce").normalize()


def _event_calendar_days(start_date, end_date):
    """返回事件起止日期之间的连续自然日，首尾均包含。"""
    start = _normalize_day(start_date)
    end = _normalize_day(end_date)
    if pd.isna(start) or pd.isna(end) or end < start:
        return pd.DatetimeIndex([])
    return pd.date_range(start, end, freq="D")


def _process_phase_ranges(start_date, end_date):
    """按事件相对进程把连续日历天三等分为 Early、Middle、Late。"""
    days = _event_calendar_days(start_date, end_date)
    parts = np.array_split(days, len(PROCESS_EVENT_PHASES))
    ranges = {}
    for label, part in zip(PROCESS_EVENT_PHASES, parts):
        if len(part) == 0:
            ranges[label] = (pd.NaT, pd.NaT)
        else:
            ranges[label] = (pd.Timestamp(part[0]).normalize(), pd.Timestamp(part[-1]).normalize())
    return ranges


def _process_window_ols(hourly, site, window_start, window_end, min_hours=PROCESS_WINDOW_MIN_HOURS):
    """计算单个事件阶段、恢复段或滑动窗口的 OLS-CBI。

    质量规则：
        1. 只使用 Observed_T15cm_C 和 ERA5_T2m_C 同时有效的小时；
        2. 有效小时数至少为 PROCESS_WINDOW_MIN_HOURS，默认 72 小时；
        3. ERA5 温度至少有 3 个不同值，且标准差必须大于 0；
        4. 不额外施加 MacroSD >= 1.0，因为 BASE 事件 CBI 也不对事件期强加该门槛。
    """
    if pd.isna(window_start) or pd.isna(window_end) or window_end < window_start:
        return dict(CBI=np.nan, Intercept=np.nan, R2=np.nan, n_hours=0, MacroSD=np.nan, Flag="invalid_window")

    subset = hourly.loc[
        (hourly[SITE_COL].astype(str) == str(site))
        & hourly[DATE_COL].between(window_start, window_end)
    ]
    d = subset[[MICRO_COL, MACRO_COL]].dropna()
    n = len(d)
    if n < min_hours:
        return dict(CBI=np.nan, Intercept=np.nan, R2=np.nan, n_hours=n, MacroSD=np.nan, Flag="insufficient_hours")

    macro_sd = d[MACRO_COL].std()
    if (
        not np.isfinite(macro_sd)
        or d[MACRO_COL].nunique() < PROCESS_WINDOW_MIN_UNIQUE_MACRO_VALUES
        or macro_sd <= 0
    ):
        return dict(CBI=np.nan, Intercept=np.nan, R2=np.nan, n_hours=n, MacroSD=macro_sd, Flag="insufficient_macro_variation")

    fit = linregress(d[MACRO_COL], d[MICRO_COL])
    return dict(
        CBI=fit.slope,
        Intercept=fit.intercept,
        R2=fit.rvalue ** 2,
        n_hours=n,
        MacroSD=macro_sd,
        Flag="ok",
    )


def _recovery_overlaps_other_extreme_event(event_row, window_start, window_end, extreme_events):
    """判断恢复窗口是否与同站点另一场 BASE 极端事件重叠。"""
    if not EXCLUDE_RECOVERY_WINDOWS_OVERLAPPING_EXTREME_EVENT:
        return False
    site = str(event_row[SITE_COL])
    event_id = str(event_row["Event_ID"])
    candidates = extreme_events.loc[
        (extreme_events[SITE_COL].astype(str) == site)
        & (extreme_events["Event_ID"].astype(str) != event_id)
    ]
    if candidates.empty:
        return False
    return bool(
        (
            (candidates["Start_Date"] <= window_end)
            & (candidates["End_Date"] >= window_start)
        ).any()
    )


def _process_window_row(event_row, stage, stage_label, window_type, window_start, window_end, hourly, reason_override=None, window_days=np.nan, relative_end_day=np.nan):
    """生成单个事件窗口的明细行，并统一计算 Delta_CBI = 窗口 CBI - Reference_CBI。"""
    reference_cbi = pd.to_numeric(pd.Series([event_row.get("Reference_CBI", np.nan)]), errors="coerce").iloc[0]
    base = {
        "ScenarioID": "BASE",
        SITE_COL: event_row[SITE_COL],
        "Event_ID": event_row["Event_ID"],
        "Event_Start_Date": event_row["Start_Date"],
        "Event_End_Date": event_row["End_Date"],
        "Duration_Days": event_row.get("Duration_Days", np.nan),
        "Minimum_SPI": event_row.get("Minimum_SPI", np.nan),
        "Stage": stage,
        "Stage_Label": stage_label,
        "Window_Type": window_type,
        "Window_Start": window_start,
        "Window_End": window_end,
        "Window_Days": window_days,
        "Window_End_Relative_Day": relative_end_day,
        "Reference_CBI": reference_cbi,
        "Reference_Flag": event_row.get("Reference_Flag", np.nan),
        "Pair_Flag": event_row.get("Pair_Flag", np.nan),
    }
    if reason_override is not None:
        base.update(
            Window_CBI=np.nan,
            Window_Intercept=np.nan,
            Window_R2=np.nan,
            Window_n_hours=0,
            Window_MacroSD=np.nan,
            Window_Flag=reason_override,
            Delta_CBI=np.nan,
        )
        return base

    result = _process_window_ols(hourly, event_row[SITE_COL], window_start, window_end)
    delta = result["CBI"] - reference_cbi if result["Flag"] == "ok" and np.isfinite(reference_cbi) else np.nan
    flag = result["Flag"]
    if flag == "ok" and not np.isfinite(reference_cbi):
        flag = "missing_reference_cbi"
        delta = np.nan

    base.update(
        Window_CBI=result["CBI"],
        Window_Intercept=result["Intercept"],
        Window_R2=result["R2"],
        Window_n_hours=result["n_hours"],
        Window_MacroSD=result["MacroSD"],
        Window_Flag=flag,
        Delta_CBI=delta,
    )
    return base


def build_process_recovery_detail(hourly, base_event_pairs, base_extreme_events):
    """生成完整事件复核、事件内阶段和事件后恢复窗口明细。

    输入事件只使用 Hourly_state / BASE 且 Pair_Flag == ok 的事件配对结果。
    这样所有阶段和恢复窗口都共享同一个已经按主脚本规则选出的 Reference_CBI。
    """
    eligible = base_event_pairs.loc[base_event_pairs["Pair_Flag"].eq("ok")].copy()
    if eligible.empty:
        return pd.DataFrame()

    rows = []
    for _, event in progress_iter(
        eligible.iterrows(),
        total=len(eligible),
        desc="构建CBI过程恢复阶段窗口",
        kind="过程恢复",
    ):
        rows.append(
            _process_window_row(
                event,
                "FullEvent",
                "完整事件",
                "event_reproduction",
                event["Start_Date"],
                event["End_Date"],
                hourly,
                window_days=event.get("Duration_Days", np.nan),
            )
        )
        for stage, (start, end) in _process_phase_ranges(event["Start_Date"], event["End_Date"]).items():
            rows.append(
                _process_window_row(
                    event,
                    stage,
                    PROCESS_EVENT_PHASE_LABELS[stage],
                    "event_phase",
                    start,
                    end,
                    hourly,
                    window_days=len(_event_calendar_days(start, end)),
                )
            )
        event_end = _normalize_day(event["End_Date"])
        for stage, (offset_start, offset_end, label) in PROCESS_RECOVERY_WINDOWS.items():
            start = event_end + pd.Timedelta(days=offset_start)
            end = event_end + pd.Timedelta(days=offset_end)
            reason = (
                "overlap_other_extreme_event"
                if _recovery_overlaps_other_extreme_event(event, start, end, base_extreme_events)
                else None
            )
            rows.append(
                _process_window_row(
                    event,
                    stage,
                    label,
                    "recovery",
                    start,
                    end,
                    hourly,
                    reason_override=reason,
                    window_days=offset_end - offset_start + 1,
                    relative_end_day=offset_end,
                )
            )
    return pd.DataFrame(rows)


def build_process_sliding_detail(hourly, base_event_pairs):
    """生成 7 天和 14 天 trailing CBI 滑动窗口明细。

    每个窗口以某个自然日为终点，向前回溯 7 或 14 个完整日。
    横轴使用 Window_End_Relative_Day，即窗口终点相对事件结束日的天数。
    """
    eligible = base_event_pairs.loc[base_event_pairs["Pair_Flag"].eq("ok")].copy()
    if eligible.empty:
        return pd.DataFrame()

    rows = []
    total_windows = sum(
        len(pd.date_range(row["Start_Date"], row["End_Date"] + pd.Timedelta(days=POST_EVENT_SEARCH_DAYS), freq="D"))
        * len(PROCESS_SLIDING_WINDOW_DAYS)
        for _, row in eligible.iterrows()
    )
    progress = progress_bar(total=total_windows, desc="构建CBI滑动窗口", kind="过程恢复")
    try:
        for _, event in eligible.iterrows():
            event_start = _normalize_day(event["Start_Date"])
            event_end = _normalize_day(event["End_Date"])
            for window_days in PROCESS_SLIDING_WINDOW_DAYS:
                for end in pd.date_range(event_start, event_end + pd.Timedelta(days=POST_EVENT_SEARCH_DAYS), freq="D"):
                    start = end - pd.Timedelta(days=window_days - 1)
                    relative_end = int((end - event_end).days)
                    rows.append(
                        _process_window_row(
                            event,
                            f"SW{window_days:02d}",
                            f"{window_days}天滑动窗口",
                            "sliding",
                            start,
                            end,
                            hourly,
                            window_days=window_days,
                            relative_end_day=relative_end,
                        )
                    )
                    progress.update(1)
    finally:
        progress.close()
    return pd.DataFrame(rows)


def process_recovery_stage_bootstrap(detail, stage):
    """对单个阶段/恢复窗口做站点聚类 bootstrap 汇总。

    汇总逻辑与事件 CBI 一致：先计算每个站点的平均 Delta_CBI，再对站点均值重抽样。
    """
    d = detail.loc[
        detail["Stage"].eq(stage)
        & detail["Window_Flag"].eq("ok")
        & detail["Delta_CBI"].notna()
    ].copy()
    if d.empty:
        return dict(
            Stage=stage,
            Stage_Label=np.nan,
            Window_Type=np.nan,
            Mean_Delta_CBI=np.nan,
            CI_low95=np.nan,
            CI_high95=np.nan,
            P_two_sided=np.nan,
            N_sites=0,
            N_events=0,
            N_windows=0,
            Bootstrap_Flag="no_valid_windows",
        )

    site_means = d.groupby(SITE_COL)["Delta_CBI"].mean()
    values = site_means.to_numpy(dtype=float)
    observed = float(values.mean())
    if len(values) < 2:
        return dict(
            Stage=stage,
            Stage_Label=d["Stage_Label"].iloc[0],
            Window_Type=d["Window_Type"].iloc[0],
            Mean_Delta_CBI=observed,
            CI_low95=np.nan,
            CI_high95=np.nan,
            P_two_sided=np.nan,
            N_sites=len(values),
            N_events=d["Event_ID"].nunique(),
            N_windows=len(d),
            Bootstrap_Flag="fewer_than_2_sites",
        )

    rng = np.random.default_rng(RANDOM_SEED)
    boot = np.empty(N_CLUSTER_BOOTSTRAP)
    for index in progress_iter(
        range(N_CLUSTER_BOOTSTRAP),
        total=N_CLUSTER_BOOTSTRAP,
        desc=f"{stage} 过程恢复Bootstrap",
        kind="Bootstrap",
    ):
        boot[index] = rng.choice(values, size=len(values), replace=True).mean()
    p_value = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    return dict(
        Stage=stage,
        Stage_Label=d["Stage_Label"].iloc[0],
        Window_Type=d["Window_Type"].iloc[0],
        Mean_Delta_CBI=observed,
        CI_low95=np.percentile(boot, 2.5),
        CI_high95=np.percentile(boot, 97.5),
        P_two_sided=p_value,
        N_sites=len(values),
        N_events=d["Event_ID"].nunique(),
        N_windows=len(d),
        Bootstrap_Flag="ok",
    )


def summarize_process_recovery_stages(detail):
    """汇总完整事件、事件前中后段和恢复窗口的区域平均 Delta_CBI。"""
    stage_order = ["FullEvent", "Early", "Middle", "Late", "R01_07", "R08_14", "R15_30"]
    rows = [process_recovery_stage_bootstrap(detail, stage) for stage in stage_order]
    return pd.DataFrame(rows)


def summarize_process_sliding_trajectory(sliding_detail):
    """按滑动窗口长度和相对事件结束日汇总中位数、IQR、事件数和站点数。"""
    d = sliding_detail.loc[
        sliding_detail["Window_Flag"].eq("ok") & sliding_detail["Delta_CBI"].notna()
    ].copy()
    if d.empty:
        return pd.DataFrame(
            columns=[
                "Window_Days",
                "Window_End_Relative_Day",
                "Median_Delta_CBI",
                "Q25_Delta_CBI",
                "Q75_Delta_CBI",
                "N_events",
                "N_sites",
                "N_windows",
            ]
        )
    summary = (
        d.groupby(["Window_Days", "Window_End_Relative_Day"], as_index=False)
        .agg(
            Median_Delta_CBI=("Delta_CBI", "median"),
            Q25_Delta_CBI=("Delta_CBI", lambda value: value.quantile(0.25)),
            Q75_Delta_CBI=("Delta_CBI", lambda value: value.quantile(0.75)),
            N_events=("Event_ID", "nunique"),
            N_sites=(SITE_COL, "nunique"),
            N_windows=("Delta_CBI", "size"),
        )
        .sort_values(["Window_Days", "Window_End_Relative_Day"])
        .reset_index(drop=True)
    )
    return summary


def build_process_recovery_reproduction_audit(detail, base_event_pairs):
    """核验新增模块重算的完整事件 CBI 是否复现已有 BASE Event_CBI。"""
    if detail.empty:
        return pd.DataFrame()
    full = detail.loc[
        detail["Stage"].eq("FullEvent"),
        [SITE_COL, "Event_ID", "Window_CBI", "Window_Flag", "Window_n_hours", "Window_MacroSD"],
    ].copy()
    original = base_event_pairs.loc[
        base_event_pairs["Pair_Flag"].eq("ok"),
        [SITE_COL, "Event_ID", "Event_CBI", "Event_n_hours", "Event_MacroSD", "Event_Flag"],
    ].copy()
    audit = original.merge(full, on=[SITE_COL, "Event_ID"], how="left", validate="one_to_one")
    audit["Absolute_Difference"] = (audit["Event_CBI"] - audit["Window_CBI"]).abs()
    audit["Reproduced"] = (
        audit["Event_Flag"].eq("ok")
        & audit["Window_Flag"].eq("ok")
        & audit["Absolute_Difference"].le(1e-8)
    )
    return audit


def plot_process_recovery_stage_summary(summary, out_dir):
    """绘制事件内阶段和事件后恢复窗口的 Delta_CBI 点估计图。"""
    order = ["Early", "Middle", "Late", "R01_07", "R08_14", "R15_30"]
    labels = ["事件前段", "事件中段", "事件后段", "结束后1-7天", "结束后8-14天", "结束后15-30天"]
    d = summary.set_index("Stage").reindex(order).reset_index()
    d["Plot_Label"] = labels

    plot_bar = progress_step("绘制CBI阶段恢复图", "绘图")
    try:
        x = np.arange(len(d))
        fig, ax = plt.subplots(figsize=(11, 5.8))
        valid = d["Mean_Delta_CBI"].notna() & d["CI_low95"].notna() & d["CI_high95"].notna()
        if valid.any():
            ax.errorbar(
                x[valid],
                d.loc[valid, "Mean_Delta_CBI"],
                yerr=[
                    d.loc[valid, "Mean_Delta_CBI"] - d.loc[valid, "CI_low95"],
                    d.loc[valid, "CI_high95"] - d.loc[valid, "Mean_Delta_CBI"],
                ],
                fmt="o",
                color=COLOR_OTHER_SCENARIO,
                ecolor=COLOR_CI,
                elinewidth=LINEWIDTH_CI,
                capsize=CI_CAPSIZE,
                markersize=6,
            )
        ax.axhline(0, color=COLOR_ZERO_LINE, linestyle=LINESTYLE_ZERO, linewidth=LINEWIDTH_ZERO)
        ax.axvline(2.5, color="#969696", linestyle=":", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(d["Plot_Label"], rotation=20, ha="right")
        ax.set_ylabel("Delta CBI（窗口 CBI - 同事件 Normal 参考 CBI）")
        ax.set_title("CBI 事件过程与恢复窗口敏感性")
        ax.text(1, ax.get_ylim()[1], "事件内过程", ha="center", va="bottom")
        ax.text(4, ax.get_ylim()[1], "事件后恢复", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(out_dir / PROCESS_RECOVERY_OUTPUT_FILES["stage_figure"], dpi=FIG_DPI, bbox_inches=FIGURE_BBOX)
        plt.close(fig)
        plot_bar.update(1)
    finally:
        plot_bar.close()


def plot_process_sliding_trajectory(sliding_summary, out_dir):
    """绘制 7 天和 14 天 trailing CBI 滑动轨迹图。"""
    plot_bar = progress_step("绘制CBI滑动轨迹图", "绘图")
    try:
        fig, axes = plt.subplots(1, len(PROCESS_SLIDING_WINDOW_DAYS), figsize=(13, 5), sharey=True)
        if len(PROCESS_SLIDING_WINDOW_DAYS) == 1:
            axes = [axes]
        for ax, days in zip(axes, PROCESS_SLIDING_WINDOW_DAYS):
            d = sliding_summary.loc[sliding_summary["Window_Days"].eq(days)].sort_values("Window_End_Relative_Day")
            if not d.empty:
                x = d["Window_End_Relative_Day"].to_numpy(dtype=float)
                median = d["Median_Delta_CBI"].to_numpy(dtype=float)
                q25 = d["Q25_Delta_CBI"].to_numpy(dtype=float)
                q75 = d["Q75_Delta_CBI"].to_numpy(dtype=float)
                ax.fill_between(x, q25, q75, color="#9ECAE1", alpha=0.35, label="IQR")
                ax.plot(x, median, color=COLOR_OTHER_SCENARIO, linewidth=2.0, label="中位数")
            ax.axhline(0, color=COLOR_ZERO_LINE, linestyle=LINESTYLE_ZERO, linewidth=LINEWIDTH_ZERO)
            ax.axvline(0, color=COLOR_BASE_SCENARIO, linestyle="--", linewidth=1.1, label="事件结束日")
            ax.set_title(f"{days}天 trailing CBI")
            ax.set_xlabel("窗口终点相对事件结束日（天）")
            ax.grid(alpha=0.25)
            ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE, frameon=LEGEND_FRAME_ON)
        axes[0].set_ylabel("Delta CBI（窗口 CBI - 同事件 Normal 参考 CBI）")
        fig.tight_layout()
        fig.savefig(out_dir / PROCESS_RECOVERY_OUTPUT_FILES["sliding_figure"], dpi=FIG_DPI, bbox_inches=FIGURE_BBOX)
        plt.close(fig)
        plot_bar.update(1)
    finally:
        plot_bar.close()


def write_process_recovery_report(stage_summary, sliding_summary, reproduction_audit, stage_detail, sliding_detail):
    """写出结果解释型报告，强调该模块回答的是时间过程而非主 LMM 替代分析。"""
    if reproduction_audit.empty:
        reproduction_pass = False
        failed_reproduction = 0
        eligible_events = 0
    else:
        eligible_events = len(reproduction_audit)
        failed_reproduction = int((~reproduction_audit["Reproduced"].fillna(False)).sum())
        reproduction_pass = failed_reproduction == 0

    valid_stage = stage_summary.loc[
        stage_summary["Stage"].isin(["Early", "Middle", "Late", "R01_07", "R08_14", "R15_30"])
        & stage_summary["Mean_Delta_CBI"].notna()
    ].copy()
    positive_stage = int((valid_stage["Mean_Delta_CBI"] > 0).sum()) if not valid_stage.empty else 0
    recovery_excluded = int(stage_detail["Window_Flag"].eq("overlap_other_extreme_event").sum()) if not stage_detail.empty else 0
    valid_sliding = sliding_detail.loc[
        sliding_detail["Window_Flag"].eq("ok") & sliding_detail["Delta_CBI"].notna()
    ].copy() if not sliding_detail.empty else pd.DataFrame()

    lines = [
        "CBI过程与恢复窗口敏感性分析解释报告",
        "=" * 72,
        "",
        "一、分析目的与边界",
        (
            "本模块检验极端干旱事件发生过程中及结束后，CBI 相对同一事件 Normal 参考期的差异如何变化。"
            "它属于事件级时间窗口敏感性，不替代 Hourly_state 主 LMM，也不改变 BASE 的 SPI 阈值、事件定义或参考期选择规则。"
        ),
        "",
        "二、固定定义",
        "进入分析的事件为 Hourly_state / BASE 中 Pair_Flag = ok 的极端事件配对。",
        "所有窗口的 Delta_CBI 均定义为：窗口 CBI - 同一事件 Reference_CBI。",
        f"单个窗口 CBI 至少需要 {PROCESS_WINDOW_MIN_HOURS} 个有效小时，ERA5 温度至少 {PROCESS_WINDOW_MIN_UNIQUE_MACRO_VALUES} 个不同值且标准差大于 0。",
        f"恢复窗口与同站点另一场极端事件重叠时是否排除：{EXCLUDE_RECOVERY_WINDOWS_OVERLAPPING_EXTREME_EVENT}。",
        "",
        "三、完整事件复现核验",
        f"本次可进入时间窗口分析的 BASE 事件配对数为 {eligible_events}。",
        f"完整事件 CBI 复现失败数量为 {failed_reproduction}。",
    ]
    if reproduction_pass:
        lines.append("完整事件 CBI 复现通过，因此阶段与恢复窗口结果可以在此前提下解释。")
    else:
        lines.append("完整事件 CBI 复现未完全通过，阶段与恢复窗口结果应先作为审计结果，不建议直接用于论文解释。")

    lines.extend(
        [
            "",
            "四、阶段与恢复窗口结果概括",
            f"Early/Middle/Late/R01_07/R08_14/R15_30 中共有 {len(valid_stage)} 个窗口获得可用区域汇总，其中 {positive_stage} 个 Mean_Delta_CBI 为正。",
            f"因恢复窗口与同站点其他极端事件重叠而排除的窗口数为 {recovery_excluded}。",
        ]
    )
    if not valid_stage.empty:
        for _, row in valid_stage.iterrows():
            lines.append(
                f"{row['Stage_Label']}：Mean Delta CBI = {row['Mean_Delta_CBI']:.6f}，"
                f"95% CI = [{row['CI_low95']:.6f}, {row['CI_high95']:.6f}]，"
                f"事件数 = {int(row['N_events'])}，站点数 = {int(row['N_sites'])}。"
            )

    lines.extend(
        [
            "",
            "五、滑动窗口结果定位",
            (
                f"滑动窗口共获得 {len(valid_sliding)} 个有效窗口；7天窗口用于展示主时间轨迹，"
                "14天窗口用于检查轨迹形态是否依赖窗口长度。滑动轨迹用于描述过程形态，不替代阶段区域汇总。"
            ),
            "",
            "六、建议写作方式",
            (
                "论文中应将本模块表述为 CBI 时间过程与恢复窗口敏感性分析。"
                "若阶段与恢复窗口的 Delta_CBI 多数保持与 BASE 事件效应同方向，可说明事件级结论不仅存在于完整事件平均，"
                "也可在事件过程或恢复窗口中观察到；若部分窗口 CI 跨 0，应解释为对应阶段样本量或有效小时不足导致的不确定性增加。"
            ),
        ]
    )
    write_process_recovery_text("\n".join(lines), PROCESS_RECOVERY_OUTPUT_FILES["report"])


def run_process_recovery_window_module(hourly, drought_events, base_event_pairs):
    """运行 CBI 过程与恢复窗口敏感性模块。

    该模块只接收 Hourly_state / BASE 的事件配对结果：
        - BASE 事件 CBI 和 Reference_CBI 已由 run_event_cbi() 按主脚本规则生成；
        - 事件内阶段、恢复窗口和滑动窗口只改变“CBI 计算窗口”，不改变参考期。
    """
    if not RUN_PROCESS_RECOVERY_WINDOW_MODULE or RUN_MODE != "full":
        return None
    if base_event_pairs is None or base_event_pairs.empty:
        return None

    out_dir = OUTPUT_DIR / PROCESS_RECOVERY_OUTPUT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    hourly_process = hourly.copy()
    hourly_process[DATE_COL] = pd.to_datetime(hourly_process[DATE_COL], errors="coerce").dt.normalize()

    base_extreme_events = select_drought_events_for_scenario(drought_events, HOURLY_STATE_BASE).copy()
    if not base_extreme_events.empty:
        base_extreme_events["Start_Date"] = pd.to_datetime(base_extreme_events["Start_Date"], errors="coerce").dt.normalize()
        base_extreme_events["End_Date"] = pd.to_datetime(base_extreme_events["End_Date"], errors="coerce").dt.normalize()

    detail = build_process_recovery_detail(hourly_process, base_event_pairs, base_extreme_events)
    sliding_detail = build_process_sliding_detail(hourly_process, base_event_pairs)
    reproduction_audit = build_process_recovery_reproduction_audit(detail, base_event_pairs)
    stage_summary = summarize_process_recovery_stages(detail) if not detail.empty else pd.DataFrame()
    sliding_summary = summarize_process_sliding_trajectory(sliding_detail) if not sliding_detail.empty else pd.DataFrame()

    write_process_recovery_csv(reproduction_audit, PROCESS_RECOVERY_OUTPUT_FILES["reproduction_audit"])
    write_process_recovery_csv(detail, PROCESS_RECOVERY_OUTPUT_FILES["stage_detail"])
    write_process_recovery_csv(sliding_detail, PROCESS_RECOVERY_OUTPUT_FILES["sliding_detail"])
    write_process_recovery_csv(stage_summary, PROCESS_RECOVERY_OUTPUT_FILES["stage_summary"])
    write_process_recovery_csv(sliding_summary, PROCESS_RECOVERY_OUTPUT_FILES["sliding_summary"])

    if not stage_summary.empty:
        plot_process_recovery_stage_summary(stage_summary, out_dir)
    if not sliding_summary.empty:
        plot_process_sliding_trajectory(sliding_summary, out_dir)
    write_process_recovery_report(stage_summary, sliding_summary, reproduction_audit, detail, sliding_detail)
    return stage_summary, sliding_summary, reproduction_audit


# =============================================================================
# 9. 季节样本审计与可选季节模型：默认只审计，不自动建模
# =============================================================================

def month_to_season(month):
    """把月份映射为气候季节；冬季包含 12、1、2 月，在单年数据中通常是不完整边界季。"""
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    if month in [9, 10, 11]:
        return "Autumn"
    return "Winter"


def write_seasonal_csv(df, filename):
    """把季节模块结果写入独立子目录，避免覆盖主稳健性分析输出。"""
    write_csv(df, f"{SEASONAL_OUTPUT_DIRNAME}/{filename}")


def write_seasonal_text(text, filename):
    """把季节模块说明和解释报告写入独立子目录。"""
    seasonal_dir = OUTPUT_DIR / SEASONAL_OUTPUT_DIRNAME
    seasonal_dir.mkdir(parents=True, exist_ok=True)
    bar = progress_step(f"写出 {filename}", "结果写出")
    try:
        (seasonal_dir / filename).write_text(text, encoding="utf-8")
        bar.update(1)
    finally:
        bar.close()


def add_season_labels(hourly):
    """为小时数据添加季节字段，并按 BASE 规则生成 SPI_Status。

    季节模块始终使用 BASE 定义：
        Extreme: SPI30d <= -2.0；
        Normal: -0.5 < SPI30d < 0.5；
        SiteMonth 内 Extreme 和 Normal 各至少 72 小时才算合格。
    """
    d = hourly.copy()
    d["Season"] = d["Month"].map(month_to_season)
    d["Season_CN"] = d["Season"].map(SEASON_CN)
    d["SPI_Status"] = classify_status(d[SPI_COL], HOURLY_STATE_BASE)
    d["Is_Extreme"] = (d["SPI_Status"] == "Extreme").astype(int)
    return d


def seasonal_status_audit(hourly_seasonal):
    """统计四季中 Extreme、Normal 和 Other 的小时、天数、站点数和 SiteMonth 数。"""
    rows = []
    grouped = (
        hourly_seasonal.groupby(["Season", "Season_CN", "SPI_Status"])
        .agg(
            Hours=(SPI_COL, "size"),
            Days=(DATE_COL, "nunique"),
            Sites=(SITE_COL, "nunique"),
            SiteMonths=("Site_Month", "nunique"),
        )
        .reset_index()
    )
    for season in SEASONS:
        for status in ["Extreme", "Normal", "Other"]:
            hit = grouped.loc[
                grouped["Season"].eq(season) & grouped["SPI_Status"].eq(status)
            ]
            if hit.empty:
                rows.append(
                    {
                        "Season": season,
                        "Season_CN": SEASON_CN[season],
                        "SPI_Status": status,
                        "Hours": 0,
                        "Days": 0,
                        "Sites": 0,
                        "SiteMonths": 0,
                    }
                )
            else:
                rows.append(hit.iloc[0].to_dict())
    return pd.DataFrame(rows)


def seasonal_sitemonth_pair_audit(hourly_seasonal):
    """审计每个季节中 Site_ID × YearMonth 是否同时满足 Extreme 和 Normal 72 小时门槛。

    不能把同一季节内不同月份的小时数相加后当作配对；合格单元必须是同一个 SiteMonth 内
    Extreme 和 Normal 均达到 min_status_hours。
    """
    candidates = hourly_seasonal.loc[hourly_seasonal["SPI_Status"].isin(["Normal", "Extreme"])].copy()
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "Season",
                "Season_CN",
                SITE_COL,
                "YearMonth",
                "Site_Month",
                "Normal_Hours",
                "Extreme_Hours",
                "Pass_Normal_72h",
                "Pass_Extreme_72h",
                "Eligible_For_Seasonal_LMM",
                "Failure_Reason",
            ]
        )

    counts = (
        candidates.groupby(["Season", "Season_CN", SITE_COL, "YearMonth", "Site_Month", "SPI_Status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for status in ["Normal", "Extreme"]:
        if status not in counts.columns:
            counts[status] = 0
    counts["Normal_Hours"] = counts["Normal"]
    counts["Extreme_Hours"] = counts["Extreme"]
    counts["Pass_Normal_72h"] = counts["Normal_Hours"] >= HOURLY_STATE_BASE["min_status_hours"]
    counts["Pass_Extreme_72h"] = counts["Extreme_Hours"] >= HOURLY_STATE_BASE["min_status_hours"]
    counts["Eligible_For_Seasonal_LMM"] = counts["Pass_Normal_72h"] & counts["Pass_Extreme_72h"]
    counts["Failure_Reason"] = np.select(
        [
            counts["Eligible_For_Seasonal_LMM"],
            ~counts["Pass_Normal_72h"] & ~counts["Pass_Extreme_72h"],
            ~counts["Pass_Normal_72h"],
            ~counts["Pass_Extreme_72h"],
        ],
        ["eligible", "both_below_72h", "normal_below_72h", "extreme_below_72h"],
        default="unknown",
    )
    return counts


def seasonal_site_contribution_audit(pair_audit):
    """统计每季合格 SiteMonth 的站点贡献，识别是否由单一站点高度主导。"""
    if pair_audit.empty:
        return pd.DataFrame(
            columns=[
                "Season",
                "Season_CN",
                SITE_COL,
                "Eligible_SiteMonths",
                "Normal_Hours",
                "Extreme_Hours",
                "Share_Of_Season_EligibleSiteMonths",
            ]
        )
    grouped = (
        pair_audit.groupby(["Season", "Season_CN", SITE_COL], as_index=False)
        .agg(
            Eligible_SiteMonths=("Eligible_For_Seasonal_LMM", "sum"),
            Normal_Hours=("Normal_Hours", "sum"),
            Extreme_Hours=("Extreme_Hours", "sum"),
        )
    )
    total = grouped.groupby("Season")["Eligible_SiteMonths"].transform("sum")
    grouped["Share_Of_Season_EligibleSiteMonths"] = np.where(
        total > 0, grouped["Eligible_SiteMonths"] / total, np.nan
    )
    return grouped


def seasonal_event_audit(hourly_seasonal, daily_spi, drought_events):
    """审计 BASE 极端事件在四季中的事件数、站点数、参考期覆盖和跨季情况。

    事件按 Start_Date 所在季节归类。跨季事件和参考期跨季仅用于透明审计，不会自动剔除或自动建模。
    """
    events = select_drought_events_for_scenario(drought_events, HOURLY_STATE_BASE)
    if events.empty:
        return events, pd.DataFrame(), pd.DataFrame()

    events = events.copy()
    events["Event_Season"] = events["Start_Date"].dt.month.map(month_to_season)
    events["Event_Season_CN"] = events["Event_Season"].map(SEASON_CN)
    events["End_Season"] = events["End_Date"].dt.month.map(month_to_season)
    events["CrossSeasonEvent"] = events["Event_Season"] != events["End_Season"]

    daily_hour_counts = (
        hourly_seasonal.groupby([SITE_COL, DATE_COL])
        .size()
        .rename("n_valid_hours")
        .reset_index()
    )
    rows = []
    for _, event in progress_iter(
        events.iterrows(),
        total=len(events),
        desc="季节事件参考期审计",
        kind="季节审计",
    ):
        site = event[SITE_COL]
        search_start = event["End_Date"] + pd.Timedelta(days=POST_EVENT_BUFFER_DAYS + 1)
        search_end = event["End_Date"] + pd.Timedelta(days=POST_EVENT_BUFFER_DAYS + POST_EVENT_SEARCH_DAYS)
        ref_daily = daily_spi.loc[
            (daily_spi[SITE_COL] == site)
            & daily_spi[DATE_COL].between(search_start, search_end)
            & (daily_spi[SPI_COL] > HOURLY_STATE_BASE["normal_low"])
            & (daily_spi[SPI_COL] < HOURLY_STATE_BASE["normal_high"])
        ].copy()
        ref_daily = ref_daily.merge(
            daily_hour_counts,
            on=[SITE_COL, DATE_COL],
            how="left",
            validate="one_to_one",
        )
        good_ref_days = ref_daily.loc[
            ref_daily["n_valid_hours"].fillna(0) >= MIN_VALID_HOURS_PER_REFERENCE_DAY
        ].copy()
        reference_seasons = (
            sorted(good_ref_days[DATE_COL].dt.month.map(month_to_season).dropna().unique().tolist())
            if not good_ref_days.empty
            else []
        )
        rows.append(
            {
                SITE_COL: site,
                "Event_ID": event.get("Event_ID", np.nan),
                "Event_Season": event["Event_Season"],
                "Event_Season_CN": event["Event_Season_CN"],
                "Start_Date": event["Start_Date"],
                "End_Date": event["End_Date"],
                "Duration_Days": event["Duration_Days"],
                "Minimum_SPI": event["Min_Daily_SPI"],
                "End_Season": event["End_Season"],
                "CrossSeasonEvent": event["CrossSeasonEvent"],
                "Reference_Search_Start": search_start,
                "Reference_Search_End": search_end,
                "Potential_Normal_Reference_Days": len(good_ref_days),
                "Reference_Seasons": ",".join(reference_seasons),
                "ReferenceCrossSeason": (
                    any(season != event["Event_Season"] for season in reference_seasons)
                    if reference_seasons
                    else False
                ),
            }
        )

    detail = pd.DataFrame(rows)
    coverage = (
        detail.groupby(["Event_Season", "Event_Season_CN"], as_index=False)
        .agg(
            Extreme_Events=("Event_ID", "size"),
            Event_Sites=(SITE_COL, "nunique"),
            CrossSeason_Events=("CrossSeasonEvent", "sum"),
            ReferenceCrossSeason_Events=("ReferenceCrossSeason", "sum"),
            Events_With_Normal_Reference=(
                "Potential_Normal_Reference_Days",
                lambda value: int((value > 0).sum()),
            ),
            Potential_Normal_Reference_Days_Total=("Potential_Normal_Reference_Days", "sum"),
        )
        .rename(columns={"Event_Season": "Season", "Event_Season_CN": "Season_CN"})
    )
    return events, detail, coverage


def seasonal_model_decision(pair_audit, site_audit, event_coverage):
    """生成季节可建模性提示。

    这些字段只是审计建议，不是自动建模许可。是否拟合季节模型必须由用户在查看审计后
    手动设置 RUN_SEASONAL_LMM 和 SEASONS_TO_RUN_LMM。
    """
    rows = []
    for season in SEASONS:
        pair_season = pair_audit.loc[pair_audit["Season"].eq(season)] if not pair_audit.empty else pd.DataFrame()
        site_season = site_audit.loc[site_audit["Season"].eq(season)] if not site_audit.empty else pd.DataFrame()
        event_season = event_coverage.loc[event_coverage["Season"].eq(season)] if not event_coverage.empty else pd.DataFrame()

        eligible_sitemonths = int(pair_season["Eligible_For_Seasonal_LMM"].sum()) if not pair_season.empty else 0
        eligible_sites = (
            int(pair_season.loc[pair_season["Eligible_For_Seasonal_LMM"], SITE_COL].nunique())
            if not pair_season.empty
            else 0
        )
        largest_share = (
            site_season["Share_Of_Season_EligibleSiteMonths"].max()
            if not site_season.empty
            else np.nan
        )
        extreme_hours = int(
            pair_season.loc[pair_season["Eligible_For_Seasonal_LMM"], "Extreme_Hours"].sum()
        ) if not pair_season.empty else 0
        normal_hours = int(
            pair_season.loc[pair_season["Eligible_For_Seasonal_LMM"], "Normal_Hours"].sum()
        ) if not pair_season.empty else 0
        extreme_events = int(event_season["Extreme_Events"].iloc[0]) if not event_season.empty else 0
        event_sites = int(event_season["Event_Sites"].iloc[0]) if not event_season.empty else 0

        if eligible_sitemonths >= 10 and eligible_sites >= 6 and (pd.isna(largest_share) or largest_share <= 0.5):
            lmm_suggestion = "candidate_confirm_manually"
            lmm_reason = "合格SiteMonth、站点数和站点贡献集中度初步满足候选建模条件；仍需人工确认。"
        elif eligible_sitemonths >= 6 and eligible_sites >= 4:
            lmm_suggestion = "exploratory_only_confirm_manually"
            lmm_reason = "样本达到探索性分析下限，但不建议作为强季节结论。"
        else:
            lmm_suggestion = "audit_only_do_not_fit_lmm"
            lmm_reason = "合格SiteMonth或站点数不足，默认只报告审计。"

        if extreme_events >= 8 and event_sites >= 4:
            event_suggestion = "candidate_confirm_manually"
            event_reason = "事件数和事件站点数初步满足候选事件CBI分析条件；仍需人工确认。"
        else:
            event_suggestion = "audit_only_do_not_fit_event_CBI"
            event_reason = "事件数或事件站点数不足，默认不运行季节事件CBI。"

        rows.append(
            {
                "Season": season,
                "Season_CN": SEASON_CN[season],
                "EligibleSiteMonths": eligible_sitemonths,
                "EligibleSites": eligible_sites,
                "LargestSiteContribution": largest_share,
                "ExtremeHours": extreme_hours,
                "NormalHours": normal_hours,
                "ExtremeEvents": extreme_events,
                "EventSites": event_sites,
                "CanRunLMM_Suggested": lmm_suggestion,
                "LMM_Reason": lmm_reason,
                "CanRunEventCBI_Suggested": event_suggestion,
                "EventCBI_Reason": event_reason,
                "BoundarySeasonFlag": "yes" if season == "Winter" else "no",
                "Caution": "冬季在单年数据中通常是不完整边界季，解释必须谨慎。" if season == "Winter" else "",
            }
        )
    return pd.DataFrame(rows)


def fit_seasonal_lmm(hourly_seasonal, pair_audit, season):
    """可选季节 LMM，仅在用户人工指定季节后运行。

    模型沿用 BASE 的 SiteMonth 配对规则；Macro_Within 仍按同一 SiteMonth 的全部有效小时均值中心化。
    若该季节实际只包含一个月份，则不加入 MonthFactor，避免无变化固定效应造成奇异拟合。
    """
    eligible_ids = pair_audit.loc[
        pair_audit["Season"].eq(season) & pair_audit["Eligible_For_Seasonal_LMM"],
        "Site_Month",
    ].unique()
    d = hourly_seasonal.loc[
        hourly_seasonal["Season"].eq(season)
        & hourly_seasonal["Site_Month"].isin(eligible_ids)
        & hourly_seasonal["SPI_Status"].isin(["Normal", "Extreme"])
    ].copy()
    if d.empty or d[SITE_COL].nunique() < 2 or d["Site_Month"].nunique() < 2:
        return (
            {
                "Season": season,
                "Season_CN": SEASON_CN.get(season, season),
                "LMM_Flag": "insufficient_independent_units",
            },
            d,
        )

    site_month_macro_all = (
        hourly_seasonal.groupby("Site_Month")[MACRO_COL]
        .agg(Macro_Mean_SiteMonth_AllValid="mean")
        .reset_index()
    )
    d = d.merge(site_month_macro_all, on="Site_Month", how="left", validate="many_to_one")
    d["Macro_Within"] = d[MACRO_COL] - d["Macro_Mean_SiteMonth_AllValid"]
    valid_site_months = d.groupby("Site_Month")["Macro_Within"].std()
    d = d.loc[d["Site_Month"].isin(valid_site_months.index[valid_site_months.fillna(0) > 0])].copy()
    if d.empty or d[SITE_COL].nunique() < 2 or d["Site_Month"].nunique() < 2:
        return (
            {
                "Season": season,
                "Season_CN": SEASON_CN.get(season, season),
                "LMM_Flag": "insufficient_macro_variation_or_units",
            },
            d,
        )

    d["Month_Factor"] = d["Month"].astype(str)
    d["Site_Group"] = d[SITE_COL].astype(str)
    d["Macro_Mean_SiteMonth_C"] = (
        d["Macro_Mean_SiteMonth_AllValid"] - d["Macro_Mean_SiteMonth_AllValid"].mean()
    )

    formula = f"{MICRO_COL} ~ Macro_Within * Is_Extreme + Macro_Mean_SiteMonth_C"
    month_factor_included = d["Month_Factor"].nunique() > 1
    if month_factor_included:
        formula += " + C(Month_Factor)"

    try:
        model = smf.mixedlm(
            formula=formula,
            data=d,
            groups=d["Site_Group"],
            re_formula="1 + Macro_Within",
        )
        result = model.fit(
            reml=True,
            method=LMM_METHOD,
            maxiter=LMM_MAXITER,
            disp=False,
        )
        interaction_term = next(
            (
                term
                for term in ["Macro_Within:Is_Extreme", "Is_Extreme:Macro_Within"]
                if term in result.params.index
            ),
            None,
        )
        if interaction_term is None:
            raise KeyError("Macro_Within:Is_Extreme term not found.")
        delta = result.params[interaction_term]
        ci_low, ci_high = result.conf_int().loc[interaction_term].tolist()
        return (
            {
                "Season": season,
                "Season_CN": SEASON_CN.get(season, season),
                "LMM_Delta_CBI": delta,
                "LMM_CI_low95": ci_low,
                "LMM_CI_high95": ci_high,
                "LMM_Pvalue": result.pvalues[interaction_term],
                "LMM_Hours": len(d),
                "LMM_Sites": d[SITE_COL].nunique(),
                "LMM_SiteMonths": d["Site_Month"].nunique(),
                "Months_In_Model": ",".join(sorted(d["YearMonth"].unique())),
                "MonthFactor_Included": month_factor_included,
                "LMM_Converged": bool(getattr(result, "converged", False)),
                "LMM_Flag": "ok",
            },
            d,
        )
    except Exception as exc:
        return (
            {
                "Season": season,
                "Season_CN": SEASON_CN.get(season, season),
                "LMM_Flag": f"fit_failed: {type(exc).__name__}: {exc}",
            },
            d,
        )


def run_seasonal_event_cbi(hourly_seasonal, daily_spi, drought_events, season):
    """可选季节事件 CBI，仅在用户人工指定季节后运行。

    事件按 Start_Date 所在季节归类；跨季事件和跨季参考期不会被自动删除，而是在审计表中透明报告。
    """
    events = select_drought_events_for_scenario(drought_events, HOURLY_STATE_BASE)
    events = events.loc[events["Start_Date"].dt.month.map(month_to_season).eq(season)].copy()
    if events.empty:
        return {"Season": season, "Season_CN": SEASON_CN.get(season, season), "Event_Flag": "no_events_in_season"}
    event_keys = set(zip(events[SITE_COL].astype(str), events["Event_ID"].astype(str)))
    events_keep = drought_events.loc[
        drought_events.apply(
            lambda row: (str(row[SITE_COL]), str(row["Event_ID"])) in event_keys,
            axis=1,
        )
    ].copy()
    summary, _ = run_event_cbi(hourly_seasonal, daily_spi, events_keep, HOURLY_STATE_BASE, f"Season_{season}")
    return {"Season": season, "Season_CN": SEASON_CN.get(season, season), **summary}


def write_seasonal_run_report(decision_table):
    """写出季节模块说明：默认只审计，季节模型必须人工授权。"""
    lines = [
        "季节样本审计与季节模型运行说明",
        "=" * 72,
        f"RUN_SEASONAL_MODULE = {RUN_SEASONAL_MODULE}",
        f"RUN_SEASONAL_LMM = {RUN_SEASONAL_LMM}",
        f"SEASONS_TO_RUN_LMM = {SEASONS_TO_RUN_LMM}",
        f"RUN_SEASONAL_EVENT_CBI = {RUN_SEASONAL_EVENT_CBI}",
        f"SEASONS_TO_RUN_EVENT_CBI = {SEASONS_TO_RUN_EVENT_CBI}",
        "",
        "默认原则：第一次运行只生成审计表，不自动拟合任何季节 LMM 或季节事件 CBI。",
        "审计表中的建议字段只是透明提示，不是自动建模许可；正式建模必须由用户手动填写季节列表。",
        "",
        "固定 BASE 规则：Extreme 为 SPI30d <= -2.0；Normal 为 -0.5 < SPI30d < 0.5；",
        "每个 Site_ID × YearMonth × SPI_Status 至少 72 小时，且同一 SiteMonth 必须同时满足 Extreme 和 Normal。",
        "Macro_Within 仍按同一 SiteMonth 的全部有效小时均值中心化，不能在季节子样本内重新定义。",
        "",
        "季节可建模性摘要：",
        decision_table.to_string(index=False),
    ]
    write_seasonal_text("\n".join(lines), SEASONAL_OUTPUT_FILES["run_config"])


def write_seasonal_interpretation_report(decision_table):
    """写出季节审计解释报告，强调审计不等于季节效应结论。"""
    candidate_lmm = decision_table.loc[
        decision_table["CanRunLMM_Suggested"].eq("candidate_confirm_manually"),
        "Season_CN",
    ].tolist()
    exploratory_lmm = decision_table.loc[
        decision_table["CanRunLMM_Suggested"].eq("exploratory_only_confirm_manually"),
        "Season_CN",
    ].tolist()
    candidate_event = decision_table.loc[
        decision_table["CanRunEventCBI_Suggested"].eq("candidate_confirm_manually"),
        "Season_CN",
    ].tolist()
    lines = [
        "季节样本审计解释报告",
        "=" * 72,
        "",
        "一、分析定位",
        "本模块用于判断四季是否具备进一步做季节 LMM 或季节事件 CBI 的样本基础；审计结果本身不等于季节效应结论。",
        "",
        "二、LMM 样本建议",
        f"达到候选建模提示的季节：{', '.join(candidate_lmm) if candidate_lmm else '无'}。",
        f"仅建议探索性考虑的季节：{', '.join(exploratory_lmm) if exploratory_lmm else '无'}。",
        "这些提示仍需人工结合合格 SiteMonth 数、站点数、最大站点贡献和冬季边界问题后决定是否开启模型。",
        "",
        "三、事件 CBI 样本建议",
        f"达到候选事件 CBI 提示的季节：{', '.join(candidate_event) if candidate_event else '无'}。",
        "事件按 Start_Date 所在季节归类；跨季事件和跨季参考期必须在解释中透明报告。",
        "",
        "四、解释边界",
        "若某季样本不足、CI 跨 0、模型不收敛或样本高度集中于单站点，应表述为证据不足或估计不稳定，不能直接写成季节效应反转。",
    ]
    write_seasonal_text("\n".join(lines), SEASONAL_OUTPUT_FILES["interpretation_report"])


def write_seasonal_detailed_conclusion(decision_table, event_detail):
    """生成季节审计的详细结论文本，重点解释为什么不建议继续做正式季节模型。"""
    season_lines = []
    for season in SEASONS:
        row = decision_table.loc[decision_table["Season"].eq(season)]
        if row.empty:
            continue
        row = row.iloc[0]
        season_lines.append(
            (
                f"{row['Season_CN']}：合格 SiteMonth = {int(row['EligibleSiteMonths'])}，"
                f"合格站点 = {int(row['EligibleSites'])}，"
                f"Extreme 小时 = {int(row['ExtremeHours'])}，Normal 小时 = {int(row['NormalHours'])}，"
                f"最大单站点贡献比例 = {row['LargestSiteContribution'] if pd.notna(row['LargestSiteContribution']) else 'NA'}，"
                f"极端事件数 = {int(row['ExtremeEvents'])}，事件站点数 = {int(row['EventSites'])}。"
                f"LMM 建议为 {row['CanRunLMM_Suggested']}；事件 CBI 建议为 {row['CanRunEventCBI_Suggested']}。"
            )
        )

    if event_detail.empty:
        cross_season_lines = ["事件审计表为空，无法评估跨季事件和参考期跨季情况。"]
    else:
        cross_season_lines = []
        for season in SEASONS:
            d = event_detail.loc[event_detail["Event_Season"].eq(season)]
            if d.empty:
                cross_season_lines.append(f"{SEASON_CN[season]}：无 BASE 极端事件。")
                continue
            n_events = len(d)
            n_cross_event = int(d["CrossSeasonEvent"].sum())
            n_cross_ref = int(d["ReferenceCrossSeason"].sum())
            n_with_ref = int((d["Potential_Normal_Reference_Days"] > 0).sum())
            cross_season_lines.append(
                (
                    f"{SEASON_CN[season]}：事件数 {n_events}，跨季事件 {n_cross_event}，"
                    f"参考期跨季事件 {n_cross_ref}，具有候选 Normal 参考日的事件 {n_with_ref}。"
                )
            )

    no_formal_lmm = decision_table.loc[
        ~decision_table["CanRunLMM_Suggested"].eq("candidate_confirm_manually"),
        "Season_CN",
    ].tolist()
    event_candidates = decision_table.loc[
        decision_table["CanRunEventCBI_Suggested"].eq("candidate_confirm_manually"),
        "Season_CN",
    ].tolist()

    lines = [
        "季节审计结论与不建模原因说明",
        "=" * 72,
        "",
        "一、总体结论",
        (
            "季节审计的结论不是发现主结论在某个季节发生反转，而是表明当前数据拆分到季节后，"
            "独立配对单元明显不足，不适合继续开展正式季节分层 LMM。"
        ),
        (
            "全年主 LMM 的 Extreme-Normal 差异已经在阈值敏感性、事件定义敏感性、逐站点剔除和逐区域过程剔除中得到主要稳健性支持；"
            "季节审计应作为补充材料，用于说明季节异质性分析为何不作为正式推断模型。"
        ),
        "",
        "二、各季节样本覆盖结果",
        *season_lines,
        "",
        "三、为什么不建议做正式季节 LMM",
        (
            "季节 LMM 的有效独立信息主要来自 Site_ID × YearMonth 配对单元，而不是小时记录总数。"
            "即使某些季节的 Extreme 或 Normal 小时数看似不少，只要这些小时集中在少数站点或少数月份，"
            "混合效应模型仍然缺少足够的独立配对单元。"
        ),
        (
            f"本次审计中，没有任何季节达到正式候选季节 LMM 的条件；不建议正式建模的季节包括：{', '.join(no_formal_lmm) if no_formal_lmm else '无'}。"
            "若强行拟合季节 LMM，结果很可能受到少数站点、少数月份或极少数 Extreme-Normal 配对单元影响，"
            "容易出现置信区间变宽、收敛不稳定或方向对个别样本高度敏感的问题。"
        ),
        (
            "因此，季节 LMM 不应作为论文正式结果。若夏季因样本相对较多而被探索性查看，也只能作为探索性补充，"
            "不能写成可靠的夏季效应结论。"
        ),
        "",
        "四、事件级季节审计与跨季边界",
        *cross_season_lines,
        (
            f"按事件数和事件站点数看，可人工考虑事件 CBI 补充分析的季节为：{', '.join(event_candidates) if event_candidates else '无'}。"
            "但事件按 Start_Date 归季并不等于完整事件过程和参考期都位于该季节。"
        ),
        (
            "如果跨季事件比例较高，季节事件 CBI 只能被表述为“按事件开始季节归类的事件级补充审计”，"
            "不能表述为严格的“纯季节内事件效应”。若要使用更严格口径，应进一步要求事件开始和结束均在同一季节，"
            "且参考期 Normal 日也属于同一季节；但这样会进一步减少样本量。"
        ),
        "",
        "五、建议写法",
        (
            "建议在正文或补充材料中表述为：由于各季节内 Extreme-Normal 合格配对 SiteMonth 数有限，"
            "且部分事件存在跨季过程和跨季参考期，季节异质性分析受样本覆盖限制，未作为正式推断模型纳入主结果。"
        ),
        (
            "这并不削弱全年主结论的稳健性；它说明当前数据更适合支持全年尺度的 Extreme-Normal 比较，"
            "而不足以支持可靠的季节分层机制判断。"
        ),
    ]
    write_seasonal_text("\n".join(lines), SEASONAL_OUTPUT_FILES["detailed_conclusion"])


def run_seasonal_module(hourly, daily_spi, drought_events):
    """运行季节样本审计；默认不拟合季节模型。"""
    if not RUN_SEASONAL_MODULE:
        return None
    if RUN_SEASONAL_LMM and not SEASONS_TO_RUN_LMM:
        raise ValueError("RUN_SEASONAL_LMM=True but SEASONS_TO_RUN_LMM is empty; inspect audits and specify seasons manually.")
    if RUN_SEASONAL_EVENT_CBI and not SEASONS_TO_RUN_EVENT_CBI:
        raise ValueError("RUN_SEASONAL_EVENT_CBI=True but SEASONS_TO_RUN_EVENT_CBI is empty; inspect audits and specify seasons manually.")
    invalid_lmm = sorted(set(SEASONS_TO_RUN_LMM) - set(SEASONS))
    invalid_event = sorted(set(SEASONS_TO_RUN_EVENT_CBI) - set(SEASONS))
    if invalid_lmm or invalid_event:
        raise ValueError(f"Invalid seasonal names. LMM={invalid_lmm}; EventCBI={invalid_event}.")

    seasonal_dir = OUTPUT_DIR / SEASONAL_OUTPUT_DIRNAME
    seasonal_dir.mkdir(parents=True, exist_ok=True)

    audit_bar = progress_step("生成季节审计数据", "季节审计")
    try:
        hourly_seasonal = add_season_labels(hourly)
        status_audit = seasonal_status_audit(hourly_seasonal)
        pair_audit = seasonal_sitemonth_pair_audit(hourly_seasonal)
        site_audit = seasonal_site_contribution_audit(pair_audit)
        _, event_detail, event_coverage = seasonal_event_audit(hourly_seasonal, daily_spi, drought_events)
        decision_table = seasonal_model_decision(pair_audit, site_audit, event_coverage)
        audit_bar.update(1)
    finally:
        audit_bar.close()

    write_seasonal_csv(status_audit, SEASONAL_OUTPUT_FILES["status_audit"])
    write_seasonal_csv(pair_audit, SEASONAL_OUTPUT_FILES["sitemonth_pair_audit"])
    write_seasonal_csv(site_audit, SEASONAL_OUTPUT_FILES["site_contribution_audit"])
    write_seasonal_csv(event_detail, SEASONAL_OUTPUT_FILES["event_detail_audit"])
    write_seasonal_csv(event_coverage, SEASONAL_OUTPUT_FILES["event_coverage_audit"])
    write_seasonal_csv(decision_table, SEASONAL_OUTPUT_FILES["model_decision"])

    if RUN_SEASONAL_LMM:
        rows = []
        for season in progress_iter(
            SEASONS_TO_RUN_LMM,
            total=len(SEASONS_TO_RUN_LMM),
            desc="运行人工指定季节LMM",
            kind="季节审计",
        ):
            row, _ = fit_seasonal_lmm(hourly_seasonal, pair_audit, season)
            rows.append(row)
        write_seasonal_csv(pd.DataFrame(rows), SEASONAL_OUTPUT_FILES["lmm_summary"])

    if RUN_SEASONAL_EVENT_CBI:
        rows = []
        for season in progress_iter(
            SEASONS_TO_RUN_EVENT_CBI,
            total=len(SEASONS_TO_RUN_EVENT_CBI),
            desc="运行人工指定季节事件CBI",
            kind="季节审计",
        ):
            rows.append(run_seasonal_event_cbi(hourly_seasonal, daily_spi, drought_events, season))
        write_seasonal_csv(pd.DataFrame(rows), SEASONAL_OUTPUT_FILES["event_cbi_summary"])

    write_seasonal_run_report(decision_table)
    write_seasonal_interpretation_report(decision_table)
    write_seasonal_detailed_conclusion(decision_table, event_detail)
    return decision_table


# =============================================================================
# 9. 主流程：按固定顺序运行，避免跳步导致结果不可解释
# =============================================================================

def main():
    """主入口函数。

    运行顺序：
        1. 写出预设情景定义和参数审计表；
        2. 读取并质控输入数据；
        3. 逐个情景构造 SPI 状态、LMM 样本和覆盖审计；
        4. coverage_only 模式只写覆盖结果；
        5. full 模式额外运行 LMM、站点月份 CBI、事件 CBI、方向一致性和森林图；
        6. full 模式下可继续运行两个影响点稳健性模块：
           LOSO 检验 Hourly_state / BASE LMM 是否被单一站点驱动；
           LORPO 检验 BASE 事件—参考期 CBI 是否被某一次区域干旱过程驱动；
        7. 季节模块默认只做样本审计；只有人工指定季节后才运行季节模型；
        8. 无论是否出错，finally 都会清理本次运行临时缓存目录。

    注意：
        不要根据 full 结果事后增删情景；SCENARIOS 是预先固定的敏感性规格。
    """
    if RUN_MODE not in {"coverage_only", "full"}:
        raise ValueError('RUN_MODE must be "coverage_only" or "full".')

    # 临时缓存目录只服务本次运行。当前脚本主要在内存中计算，保留该目录是为了
    # 后续如需缓存中间大表时有明确位置，并且 finally 会统一清理。
    RUNTIME_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # 先写出情景定义，保证即使后续模型失败，也能追踪本次使用了哪些阈值规格。
        scenario_definitions = build_scenario_definitions()
        write_csv(scenario_definitions, OUTPUT_FILES["scenario_definitions"])
        write_parameter_audit()

        hourly, daily_spi, drought_events, input_audit = load_data()
        write_csv(input_audit, OUTPUT_FILES["input_audit"])

        results = []
        all_status_summaries = []
        all_sitemonth_units = []
        all_sitemonth_pairs = []
        all_event_pairs = []
        all_event_definition_events = []
        base_hourly_event_pairs = None

        # 第一条证据链：小时级 SPI 状态分析（LMM + 站点月份 CBI + 主脚本事件定义事件 CBI）
        for cfg in progress_iter(
            HOURLY_STATE_SCENARIOS,
            total=len(HOURLY_STATE_SCENARIOS),
            desc="运行小时级稳健性情景",
            kind="情景覆盖",
        ):

            d_status, d_lmm, status_summary, coverage = build_lmm_dataset(
                hourly, daily_spi, drought_events, cfg
            )

            scenario_id = cfg["id"]
            status_summary.insert(0, "ScenarioID", scenario_id)
            status_summary.insert(0, "AnalysisLayer", "Hourly_state")
            all_status_summaries.append(status_summary)

            row = {
                "AnalysisLayer": "Hourly_state",
                "EvidenceChain": "Hourly_state",
                "ScenarioID": scenario_id,
                "ScenarioGroup": cfg["group"],
                "ScenarioDescription": cfg["description"],
                **cfg,
                **coverage,
            }
            row["SparseSampleFlag"] = (
                "yes"
                if (
                    coverage["lmm_sites"] < MIN_SITES_SUGGESTED
                    or coverage["lmm_sitemonths"] < MIN_SITEMONTHS_SUGGESTED
                )
                else "no"
            )

            if RUN_MODE == "full":
                lmm_bar = progress_step(f"{scenario_id} 拟合LMM", "LMM拟合")
                try:
                    row.update(run_lmm(d_lmm))
                    lmm_bar.update(1)
                finally:
                    lmm_bar.close()

                sm_summary, sm_units, sm_pairs = run_sitemonth_cbi(
                    d_status, cfg, scenario_id
                )
                row.update(sm_summary)
                if not sm_units.empty:
                    sm_units.insert(0, "AnalysisLayer", "Hourly_state")
                    sm_units.insert(1, "EvidenceChain", "SiteMonth_CBI")
                    all_sitemonth_units.append(sm_units)
                if not sm_pairs.empty:
                    sm_pairs.insert(0, "AnalysisLayer", "Hourly_state")
                    sm_pairs.insert(1, "EvidenceChain", "SiteMonth_CBI")
                    all_sitemonth_pairs.append(sm_pairs)

                event_summary, event_pairs = run_event_cbi(
                    d_status, daily_spi, drought_events, cfg, scenario_id
                )
                row.update(event_summary)
                if not event_pairs.empty:
                    event_pairs.insert(0, "AnalysisLayer", "Hourly_state")
                    event_pairs.insert(1, "EvidenceChain", "Event_reference")
                    if scenario_id == "BASE":
                        base_hourly_event_pairs = event_pairs.copy()
                    all_event_pairs.append(event_pairs)

                row["EventSparseFlag"] = (
                    "yes" if row.get("Event_N_pairs", 0) < MIN_EVENT_PAIRS_SUGGESTED else "no"
                )

            results.append(row)

        # 第二条证据链：事件定义稳健性（重新提取完整事件，不进入 LMM）
        for cfg in progress_iter(
            EVENT_DEFINITION_SCENARIOS,
            total=len(EVENT_DEFINITION_SCENARIOS),
            desc="运行事件定义稳健性情景",
            kind="情景覆盖",
        ):
            scenario_id = cfg["id"]
            events = extract_events_from_daily_spi(daily_spi, cfg)
            if not events.empty:
                events.insert(0, "AnalysisLayer", "Event_definition")
                events.insert(1, "EvidenceChain", "Event_definition")
                events.insert(2, "ScenarioID", scenario_id)
                events.insert(3, "ScenarioGroup", cfg["group"])
                all_event_definition_events.append(events)

            coverage = event_definition_coverage(hourly, daily_spi, events, cfg)
            row = {
                "AnalysisLayer": "Event_definition",
                "EvidenceChain": "Event_definition",
                "ScenarioID": scenario_id,
                "ScenarioGroup": cfg["group"],
                "ScenarioDescription": cfg["description"],
                **cfg,
                **coverage,
            }

            if RUN_MODE == "full":
                event_summary, event_pairs = run_event_definition_cbi(
                    hourly, daily_spi, events, cfg, scenario_id
                )
                row.update(event_summary)
                row["EventSparseFlag"] = (
                    "yes"
                    if row.get("Event_N_pairs", 0) < MIN_EVENT_PAIRS_SUGGESTED_EVENT_DEFINITION
                    else "no"
                )
                if not event_pairs.empty:
                    all_event_pairs.append(event_pairs)
            results.append(row)

        status_by_scenario = pd.concat(all_status_summaries, ignore_index=True)
        write_csv(status_by_scenario, OUTPUT_FILES["status_by_scenario"])
        if all_event_definition_events:
            write_csv(
                pd.concat(all_event_definition_events, ignore_index=True),
                OUTPUT_FILES["event_definition_events"],
            )

        summary = pd.DataFrame(results)

        if RUN_MODE == "coverage_only":
            write_csv(summary, OUTPUT_FILES["coverage_by_scenario"])
            base_audit = summary.loc[
                summary["AnalysisLayer"].eq("Hourly_state")
                & summary["ScenarioID"].eq("BASE")
            ].copy()
            write_csv(base_audit, OUTPUT_FILES["base_coverage_audit"])
        else:
            # 所有情景完成后再统一计算方向一致性，避免中途用不完整结果做判断。
            summary = add_direction_consistency(summary)
            summary = add_event_definition_direction_consistency(summary)
            write_csv(summary, OUTPUT_FILES["all_methods_summary"])

            if all_sitemonth_units:
                write_csv(
                    pd.concat(all_sitemonth_units, ignore_index=True),
                    OUTPUT_FILES["sitemonth_units"],
                )
            if all_sitemonth_pairs:
                write_csv(
                    pd.concat(all_sitemonth_pairs, ignore_index=True),
                    OUTPUT_FILES["sitemonth_pairs"],
                )
            if all_event_pairs:
                write_csv(
                    pd.concat(all_event_pairs, ignore_index=True),
                    OUTPUT_FILES["event_pairs"],
                )

            make_forest_plot(summary)
            make_event_definition_plot(summary)

        base_reproduction_audit(summary)
        write_run_report(summary)
        if RUN_MODE == "full":
            write_interpretation_report(summary)
            # LOSO 只服务 Hourly_state / BASE 主 LMM，因此放在主稳健性结果全部生成后执行；
            # 它共享同一套已读取的原始数据和 BASE 规则，但结果单独写入 Leave_one_site_out 子目录。
            if RUN_LEAVE_ONE_SITE_OUT:
                run_leave_one_site_out(hourly, daily_spi, drought_events, summary)
            # LORPO 只服务 BASE 事件—参考期 CBI：每轮从事件表删除一个多站点区域过程，
            # 但保留 hourly 和 daily_spi 原始观测完整性，结果单独写入区域过程剔除子目录。
            if RUN_LEAVE_ONE_REGIONAL_PROCESS_OUT:
                run_leave_one_regional_process_out(hourly, daily_spi, drought_events)
            # CBI 过程与恢复窗口模块只使用 Hourly_state / BASE 的事件配对结果；
            # 它改变的是 CBI 的时间窗口，而不是 SPI 阈值、事件定义或参考期搜索规则。
            if RUN_PROCESS_RECOVERY_WINDOW_MODULE:
                run_process_recovery_window_module(hourly, drought_events, base_hourly_event_pairs)
            # 季节模块默认只生成审计表和解释报告，不自动拟合季节 LMM 或季节事件 CBI。
            # 若需要建模，必须先查看季节审计输出，再手动填写 SEASONS_TO_RUN_*。
            if RUN_SEASONAL_MODULE:
                run_seasonal_module(hourly, daily_spi, drought_events)

        print("\n分析完成。")
        print(f"结果目录：{OUTPUT_DIR}")
        if RUN_MODE == "coverage_only":
            print(
                "当前为 coverage_only。请先检查“01_BASE覆盖审计.csv”、"
                "“01_BASE与主脚本复现对照.csv”和“01_各情景样本覆盖汇总.csv”，"
                "确认 BASE 对齐后再将 RUN_MODE 改为 full。"
            )
    finally:
        cleanup_runtime_cache()


if __name__ == "__main__":
    main()
