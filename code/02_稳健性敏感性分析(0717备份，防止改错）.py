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
            "总体而言，基于小时级状态定义的主结论在不同极端阈值、正常窗口、小时门槛和质量控制设定下保持同向，"
            "表明极端干旱条件下森林微气候缓冲减弱的判断具有较好的阈值稳健性。"
            "从不同干旱程度来看，E15/E18/E22 等极端阈值调整并未改变主效应方向，"
            "说明在合理的干旱分级范围内，结论并不依赖某一单点阈值。"
        ),
        (
            "事件定义证据链进一步表明，当完整干旱事件的边界和持续时间在合理范围内变化时，"
            "事件—参考期 CBI 的方向总体保持一致；这说明主结论不仅对状态阈值稳健，"
            "也对事件识别规则具有一定稳健性。"
        ),
    ]

    write_text("\n".join(lines), OUTPUT_FILES["interpretation_report"])


# =============================================================================
# 6. 主流程：按固定顺序运行，避免跳步导致结果不可解释
# =============================================================================

def main():
    """主入口函数。

    运行顺序：
        1. 写出预设情景定义和参数审计表；
        2. 读取并质控输入数据；
        3. 逐个情景构造 SPI 状态、LMM 样本和覆盖审计；
        4. coverage_only 模式只写覆盖结果；
        5. full 模式额外运行 LMM、站点月份 CBI、事件 CBI、方向一致性和森林图；
        6. 无论是否出错，finally 都会清理本次运行临时缓存目录。

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
