# -*- coding: utf-8 -*-
"""
===============================================================================
极端干旱发生 vs 未发生时的森林微气候缓冲能力差异
主分析：逐小时线性混合效应模型（LMM）
辅助分析：极端事件期 CBI vs 事件后有限窗口参考期 CBI
描述分析：月度 CBI 与 Extreme_Ratio >= 0.30 的极端占比较高月份

适用数据：
    1. 逐小时温度对齐表：
       - Site_ID
       - Time_UTC（已处理好的UTC时间列；用于生成唯一分析日期 UTC_Date）
       - ERA5_T2m_C
       - Observed_T15cm_C
       - SM（仅保留，不进入本版主模型）
    2. 各站点逐日 SPI30d 宽表：
       - 第 1 列：Date
       - 后续每列：站点 Site_ID
       - 单元格：该站点当天的 daily SPI30d
    3. 逐日 SPI 干旱事件长表：
       - Site_ID, Event_ID, Start_Date, End_Date, Duration_Days,
         Severity, Drought_Level 等

===============================================================================
研究主问题
===============================================================================

在控制月份/季节背景与站点固有差异后：

    极端干旱日（SPI30d <= -2.0）是否改变森林15 cm微气候温度
    对宏气候温度（ERA5 2 m温度）的响应斜率，即 CBI？

CBI 的定义：
    Observed_T15cm_C = intercept + CBI × ERA5_T2m_C + error

CBI 越低：微气候随宏气候变化越弱，缓冲更强。
CBI 越高：微气候随宏气候变化越强，缓冲更弱。

===============================================================================
总体流程
===============================================================================

Step 0. 读取、字段审计与质量控制
    - 读取三个输入表
    - 检查必须字段、站点匹配、时间范围、缺失值、重复记录
    - 过滤不满足温度质量阈值的数据
    - 仅保留同时具有 ERA5 与实测温度的小时记录

Step 1. SPI 宽表转长表并合并到逐小时温度数据
    - 将“日期 × 站点”的 SPI 宽表转为 Site_ID + Date + SPI30d 的长表
    - 按 Site_ID + UTC_Date 合并到每小时温度记录
    - 将每个小时标记为 Extreme / Normal / Other

Step 2. 主分析 LMM（本研究的核心）
    - 仅使用 Extreme 与 Normal 小时记录
    - 只保留同一站点-月份内同时存在足量 Extreme 和 Normal 小时的月份
      目的：在同站点、同月份背景内比较，最大限度缓解季节混淆
    - 宏气候温度使用该站点-月份全部有效小时的月均ERA5温度进行中心化
    - 模型通过 Macro × Extreme 交互项直接估计 CBI 改变量
    - Site_ID 使用随机截距和宏气候随机斜率

Step 3. 主模型结果与图形
    - 输出 β3（Macro × Extreme）及95%置信区间、p值
    - 输出 Normal CBI、Extreme CBI、Delta_CBI
    - 绘制模型预测回归线：Normal vs Extreme

Step 4. 事件级辅助验证
    - 绝不依赖小时表内的 Event_ID 标签
    - 逐行读取“极端干旱事件长表”，以 Site_ID + Start_Date + End_Date
      直接从小时温度表切出事件期温度
    - 事件结束后第2日起，在未来30个自然日内搜索 Normal SPI 日期
    - 不足目标天数时：找到几天用几天，不向30天外延伸
    - 参考期资料少于最低小时数时，不删除事件，但标记为参考期不足

Step 5. 事件级显著性检验
    - 对有可靠参考期的事件计算：
          Delta_CBI = CBI_event - CBI_reference
    - Delta_CBI > 0：极端期间CBI提高，缓冲减弱
    - 用“站点聚类 bootstrap”估计总体平均 Delta_CBI 的95% CI与p值
      目的：避免将同一站点的多个事件错误当作彼此完全独立

Step 6. 月度描述性分析（不是主结论）
    - 每个站点-月份计算一个 CBI
    - Extreme_Ratio >= 0.30 表示该月中至少30%有效小时处于极端干旱状态
    - 仅用于展示季节轨迹和敏感性背景，不表示统计显著，也不承担主显著性结论

===============================================================================
重要的可调参数
===============================================================================

1. SPI 阈值：
   EXTREME_SPI_THRESHOLD = -2.0
   NORMAL_SPI_LOW = -0.5
   NORMAL_SPI_HIGH = 0.5

   主 LMM 仅比较：
       Extreme: SPI <= -2.0
       Normal : -0.5 < SPI < 0.5
   其他日（重度/中度/轻度干旱、轻度/中度/严重/极端湿润日）均不进入主模型。

2. 参考期：
   POST_EVENT_BUFFER_DAYS = 1
       事件结束后的第1天不作为参考期，以避免紧邻事件边界的不稳定影响。

   POST_EVENT_SEARCH_DAYS = 30
       从事件结束后第2天开始，最多搜索未来30个自然日。

   REF_TARGET_DAYS = min(Duration_Days, 30)
       事件持续小于30天：目标参考日数等于事件持续天数；
       事件持续超过30天：最多只寻找30天参考日。

   MIN_REF_HOURS = 72
       至少3天完整小时记录才计算参考期CBI。
       若不足72小时，保留事件记录，但不计算可靠的 Delta_CBI。

3. 主模型样本门槛：
   MIN_STATUS_HOURS_PER_SITE_MONTH = 72
       每个站点-月份内 Extreme 和 Normal 均至少72小时，才进入主LMM。

4. 参考期候选日：
   MIN_VALID_HOURS_PER_REFERENCE_DAY = 18
       参考期中某个Normal SPI日必须至少有18个有效小时，才视为合格参考日。

5. 月度描述：
   EXTREME_RATIO_MONTHLY = 0.30
       某站点某月中，Extreme 小时比例 >= 30%，定义为“极端占比较高月份”。

===============================================================================
运行前请修改的内容
===============================================================================

只需修改“路径设置”中的三个输入路径和一个输出目录。
如果文件名与当前文件名相同、代码与数据在同一目录，可以保持相对路径。

===============================================================================
"""

# =============================================================================
# 0. 导入库
# =============================================================================

from pathlib import Path
import shutil
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import seaborn as sns
from tqdm import tqdm

from scipy import stats
from scipy.stats import linregress
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 记录脚本启动时间，用于结束时只清理本次运行新产生或更新过的临时缓存。
SCRIPT_START_TIME = time.time()

# 图形中文字体候选列表：
# matplotlib默认字体通常不含中文字形，会导致标题、坐标轴和图例显示为方框。
# 脚本会按顺序查找本机已安装字体；Windows通常可用 Microsoft YaHei、SimHei 或 SimSun。
CHINESE_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "KaiTi",
    "FangSong",
    "Noto Sans CJK SC",
    "Source Han Sans SC"
]


def choose_available_chinese_font(font_candidates):
    """
    从候选中文字体中选择当前系统可用的字体。

    返回：
    - 找到字体时返回字体名；
    - 找不到时返回None，此时matplotlib仍可运行，但中文可能显示为方框。
    """
    for font_name in font_candidates:
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=font_name),
                fallback_to_default=False
            )
            return font_name
        except ValueError:
            continue
    return None


SELECTED_CHINESE_FONT = choose_available_chinese_font(CHINESE_FONT_CANDIDATES)

# 图形基础设置：
# 1. seaborn会重置部分matplotlib样式，因此字体rc参数需要显式传入set_theme；
# 2. axes.unicode_minus=False 用于避免中文字体下负号显示异常；
# 3. 若本机没有候选中文字体，运行时会打印警告，图仍会生成但中文可能仍是方框。
if SELECTED_CHINESE_FONT is not None:
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        font=SELECTED_CHINESE_FONT,
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": [SELECTED_CHINESE_FONT] + CHINESE_FONT_CANDIDATES,
            "axes.unicode_minus": False
        }
    )
else:
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        rc={"axes.unicode_minus": False}
    )
    print(
        "警告：未在当前环境中找到候选中文字体，图片中文可能显示为方框。"
        "请在 CHINESE_FONT_CANDIDATES 中加入本机已安装的中文字体名。"
    )


# =============================================================================
# 1. 路径设置 —— 运行前重点检查
# =============================================================================

# 如果脚本与三个数据文件放在同一文件夹，保持下列文件名即可。
# 如果不在同一文件夹，请替换为完整路径，例如 r"E:\forest_microclimate\...\文件.csv"

TEMP_HOURLY_FILE = r"E:\forest_microclimate\ForestMicroclimate\results\时间序列图\逐小时温度对齐表.csv"
SPI_DAILY_WIDE_FILE = r"E:\forest_microclimate\ForestMicroclimate\results\daily_SPI_result\各站点SPI30d逐日宽表_2025.xlsx"
DROUGHT_EVENT_FILE = r"E:\forest_microclimate\ForestMicroclimate\results\daily_SPI_features\福建省观测站2025年daily_SPI干旱事件长表.csv"

# 建议使用的Python解释器环境。脚本不会自动切换解释器；
# 该路径写入最终审计报告，便于复现实验时确认运行环境。
PYTHON_ENV_DIR = r"D:\ProgramData\anaconda3\envs\gee"
PYTHON_INTERPRETER = str(Path(PYTHON_ENV_DIR) / "python.exe")

# 所有结果均写入此文件夹；不存在时脚本自动创建
OUTPUT_DIR = Path(r"E:\forest_microclimate\ForestMicroclimate\results\compare_differences_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. 参数设置 —— 改参数即可做敏感性分析，不要在主体代码中反复改数字
# =============================================================================

# ----------------------------- 时间与数据列 -----------------------------
# SPI逐日数据以UTC日界线定义。
# 因此所有SPI合并、干旱状态分类、事件切片和事件后参考期搜索，
# 均使用逐小时温度表的 Time_UTC 生成 UTC_Date。
#
# 不提供任何非UTC日期备用入口；否则UTC日界线附近的小时会被归到错误的SPI日期。
ANALYSIS_TIME_COL = "Time_UTC"
UTC_DATE_COL = "UTC_Date"
MIN_UTC_PARSE_RATE = 0.995

SITE_COL = "Site_ID"
MICRO_TEMP_COL = "Observed_T15cm_C"
MACRO_TEMP_COL = "ERA5_T2m_C"
SOIL_MOISTURE_COL = "SM"      # 本版不建模，仅保留在审计输出中

# ----------------------------- 温度质量控制 -----------------------------
# 你已说明实测温度完成异常值处理，合理范围为 -5 到 45 °C。
# 此处仍保留防错过滤，避免极少数漏网异常值影响回归。
MICRO_TEMP_MIN_C = -5.0
MICRO_TEMP_MAX_C = 45.0

# ERA5温度同样设置一个宽泛物理范围；通常不会触发。
MACRO_TEMP_MIN_C = -20.0
MACRO_TEMP_MAX_C = 50.0

# ----------------------------- SPI状态阈值 -----------------------------
EXTREME_SPI_THRESHOLD = -2.0
NORMAL_SPI_LOW = -0.5
NORMAL_SPI_HIGH = 0.5

# ------------------------- 事件期与参考期参数 ---------------------------
POST_EVENT_BUFFER_DAYS = 1
POST_EVENT_SEARCH_DAYS = 30
MAX_REF_TARGET_DAYS = 30

# CBI OLS最低有效小时数：
# 事件期：至少72小时（约3天），才计算可靠的事件期CBI。
# 参考期：至少72小时，才计算可靠的事件后参考CBI。
MIN_EVENT_HOURS = 72
MIN_REF_HOURS = 72
MIN_VALID_HOURS_PER_REFERENCE_DAY = 18

# 主LMM站点-月份状态覆盖门槛：
# 每个站点-月份内，Extreme和Normal都至少达到该小时数，才进入主模型。
MIN_STATUS_HOURS_PER_SITE_MONTH = 72
STATUS_HOUR_SENSITIVITY_THRESHOLDS = [24, 48, 72, 120]

# ---------------------------- 月度描述参数 -----------------------------
MIN_MONTHLY_HOURS = 200
EXTREME_RATIO_MONTHLY = 0.30

# ----------------------- 站点聚类Bootstrap参数 -------------------------
# 用于事件级 Delta_CBI 的辅助显著性检验。
# 2000次通常较稳；如电脑较慢可先设为1000进行测试。
N_CLUSTER_BOOTSTRAP = 2000
RANDOM_SEED = 20250714

# ------------------------- LMM模型收敛参数 ------------------------------
# 混合模型可能需要较长计算时间。lbfgs通常稳定；maxiter可根据报错提高。
LMM_METHOD = "lbfgs"
LMM_MAXITER = 500

# ---------------------------- 进度条参数 -----------------------------
# 使用 tqdm 的单行动态进度条：leave=False 表示当前步骤结束后清除进度条，
# 避免日志被每个循环刷屏；colour 用于区分不同类型的步骤。
PROGRESS_BAR_CONFIG = {
    "读取与审计": {"colour": "cyan"},
    "样本筛选": {"colour": "green"},
    "模型预测": {"colour": "blue"},
    "事件分析": {"colour": "magenta"},
    "Bootstrap": {"colour": "yellow"},
    "绘图": {"colour": "red"},
    "月度分析": {"colour": "white"},
    "模型拟合": {"colour": "blue"},
    "报告输出": {"colour": "green"},
    "站点月份CBI": {"colour": "yellow"},
    "清理缓存": {"colour": "cyan"}
}

# ---------------------------- 图形参数 -----------------------------
# 后续若要调图，不需要在绘图代码中到处找数字，只改这里即可。
FIG_DPI = 300
FIG_MAIN_PREDICT_SIZE = (9, 6)
FIG_SITE_MONTH_CBI_BOX_SIZE = (8, 6)
FIG_SITE_MONTH_CBI_PAIR_SIZE = (8, 6)
FIG_REGIONAL_CBI_BOX_SIZE = (8, 6)
FIG_MAIN_CBI_COMPOSITE_SIZE = (18, 6)
FIG_EVENT_PAIR_SIZE = (10, 7)
FIG_EVENT_DELTA_WIDTH = 10
FIG_EVENT_DELTA_MIN_HEIGHT = 6
FIG_EVENT_DELTA_HEIGHT_PER_EVENT = 0.35
FIG_MONTHLY_TRAJECTORY_SIZE = (10, 6)
FIG_BOTTOM_NOTE_Y = 0.02

COLOR_NORMAL = "#2C7FB8"
COLOR_EXTREME = "#D7301F"
COLOR_MONTHLY_MEDIAN = "#1B7837"
COLOR_SITE_MONTH_LINE = "grey"
COLOR_REFERENCE_LINE = "grey"
COLOR_ZERO_LINE = "black"
COLOR_HIGHLIGHT_TEXT = "#B2182B"

LINEWIDTH_MAIN_PREDICT = 3
LINEWIDTH_REFERENCE = 0.9
LINEWIDTH_EVENT_PAIR = 1.4
LINEWIDTH_MONTHLY_SITE = 0.9
LINEWIDTH_MONTHLY_MEDIAN = 3
LINEWIDTH_MONTHLY_EXTREME_BAR = 7
LINEWIDTH_CBI_PAIR = 1.0
LINEWIDTH_CBI_MEDIAN_PAIR = 3.0
ALPHA_EVENT_PAIR = 0.65
ALPHA_SCATTER = 0.60
ALPHA_EVENT_DELTA_SCATTER = 0.85
ALPHA_CBI_POINTS = 0.65
ALPHA_CBI_PAIR_LINES = 0.45
ALPHA_MONTHLY_SITE = 0.25
ALPHA_MONTHLY_IQR = 0.18
ALPHA_MONTHLY_EXTREME_BAR = 0.80
SCATTER_SIZE_EVENT = 5
SCATTER_SIZE_DELTA = 50
SCATTER_SIZE_CBI = 45
MARKER_MONTHLY_MEDIAN = "o"
LEGEND_LOCATION_BEST = "best"
ANNOTATION_FONTSIZE = 9
TICKLABEL_FONTSIZE_EVENT_DELTA = 8
JITTER_WIDTH_CBI_BOX = 0.08

# --------------------- 站点-月份-状态CBI补充图参数 ----------------------
# 这些图用于描述性展示，不替代主LMM显著性检验。
# 每个CBI点由一个“站点 × UTC月份 × SPI状态”的逐小时OLS斜率估计得到。
MIN_SITE_MONTH_STATUS_CBI_HOURS = 72
MIN_SITE_MONTH_STATUS_MACRO_SD = 1.0
NEAR_ZERO_DELTA_CBI = 0.01

# ---------------------------- 输出文件命名 -----------------------------
# 所有输出文件均使用中文命名。键名只供代码内部引用，值为实际写出的文件名。
OUTPUT_FILES = {
    "time_audit_summary": "00_UTC时间解析审计汇总.csv",
    "time_invalid_by_site": "00_UTC时间解析失败站点分布.csv",
    "utc_range_by_site": "00_各站点UTC日期范围.csv",
    "hourly_merged": "00_逐小时温度_SPI合并审计表.csv",
    "spi_status_audit": "00_SPI状态分布审计表.csv",
    "plot_progress_parameters": "00_绘图与进度条参数说明表.csv",
    "lmm_sample_audit": "01_主LMM样本_月份状态审计表.csv",
    "site_month_macro_audit": "01_站点月份状态小时数与宏气候背景审计表.csv",
    "status_threshold_sensitivity": "01_状态小时门槛敏感性审计表.csv",
    "lmm_dataset": "01_主LMM逐小时分析数据集.csv",
    "main_lmm_summary": "02_主LMM完整模型结果.txt",
    "main_lmm_monthly_cbi": "02_主LMM各月份模型预测CBI.csv",
    "main_lmm_key": "02_主LMM核心结果.csv",
    "main_prediction_lines": "03_主LMM正常与极端预测线数据.csv",
    "main_prediction_plot": "03_主LMM正常与极端预测线图.png",
    "main_prediction_plot_split": "03A_主LMM正常与极端预测线图.png",
    "extreme_site_filter_audit": "03_全年极端干旱站点筛选审计表.csv",
    "site_month_status_cbi": "03_站点月份状态CBI估计表.csv",
    "site_month_cbi_pair_audit": "03_站点月份CBI配对变化审计表.csv",
    "site_month_cbi_pair_summary": "03_站点月份CBI配对变化汇总表.csv",
    "event_to_site_month_pair_audit": "03_极端事件是否进入站点月份CBI配对审计表.csv",
    "regional_cbi_summary": "03_研究区总体CBI分布汇总表.csv",
    "site_month_cbi_box_plot": "03B_站点月份CBI箱线散点图.png",
    "site_month_cbi_pair_plot": "03C_站点月份CBI配对变化图.png",
    "regional_cbi_box_plot": "03D_研究区总体CBI箱线图.png",
    "main_cbi_composite_plot": "03_主LMM与站点月份CBI综合图.png",
    "event_cbi_reference": "04_极端事件CBI与事件后正常参考期对比表.csv",
    "event_unpaired_audit": "04_未进入事件CBI配对检验原因审计表.csv",
    "event_reference_dates": "04_事件后参考期候选日期审计表.csv",
    "event_bootstrap": "05_事件CBI变化站点聚类自助法结果.csv",
    "event_pair_plot": "05_事件CBI与参考期CBI配对图.png",
    "event_delta_plot": "05_事件CBI变化点图.png",
    "monthly_cbi": "06_月度CBI描述性结果.csv",
    "monthly_summary": "06_月度CBI汇总表.csv",
    "monthly_plot": "06_月度CBI季节轨迹图.png",
    "final_report": "07_最终运行审计报告.txt"
}

# ---------------------------- 临时缓存清理 -----------------------------
# 脚本本身不主动创建这些缓存；但运行 pandas/matplotlib/statsmodels 时可能产生
# Python字节码缓存。运行结束后仅删除脚本启动后新产生或更新过的缓存，不碰输入数据。
CACHE_DIR_NAMES_TO_CLEAN = ["__pycache__"]
CACHE_FILE_SUFFIXES_TO_CLEAN = [".tmp", ".temp", ".cache"]


# =============================================================================
# 3. 工具函数
# =============================================================================

def normalize_site_id(series):
    """
    将站点编号统一为字符串，解决以下常见不匹配问题：
    - 温度表为整数：95332217
    - 事件表读入后变为浮点数：95332217.0
    - SPI宽表列名是字符串：'95332217'

    输出统一格式：'95332217'
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s


def parse_utc_date_with_audit(df, time_col=ANALYSIS_TIME_COL,
                              site_col=SITE_COL,
                              min_parse_rate=MIN_UTC_PARSE_RATE):
    """
    生成 UTC_Date（datetime64[ns]，只保留UTC日期），并返回时间解析审计表。

    逻辑：
    1. 只使用 Time_UTC，不提供任何非UTC日期备用入口；
    2. 解析率 < 99.5% 时停止运行；
    3. 解析率为 99.5% 到 100% 时允许继续，但输出失败记录与站点分布。

    注意：
    SPI30d 使用 UTC 日界线，因此后续 SPI 合并、事件切片和参考期搜索
    必须全部基于这里得到的 UTC_Date。
    """
    if time_col not in df.columns:
        raise KeyError(
            f"找不到 ANALYSIS_TIME_COL = '{time_col}'。"
            f"当前可用列为：{df.columns.tolist()}"
        )

    parsed = pd.to_datetime(df[time_col], errors="coerce")
    parse_ok = parsed.notna()
    valid_ratio = parse_ok.mean()

    invalid_by_site = (
        df.loc[~parse_ok]
        .assign(**{site_col: normalize_site_id(df.loc[~parse_ok, site_col])})
        .groupby(site_col)
        .size()
        .reset_index(name="Time_UTC_invalid_rows")
        if (~parse_ok).any()
        else pd.DataFrame(columns=[site_col, "Time_UTC_invalid_rows"])
    )

    valid_dates = pd.DataFrame({
        site_col: normalize_site_id(df.loc[parse_ok, site_col]),
        UTC_DATE_COL: parsed.loc[parse_ok].dt.normalize()
    })
    utc_date_range_by_site = (
        valid_dates.groupby(site_col)
        .agg(
            UTC_Date_min=(UTC_DATE_COL, "min"),
            UTC_Date_max=(UTC_DATE_COL, "max"),
            Time_UTC_parseable_rows=(UTC_DATE_COL, "size")
        )
        .reset_index()
    )

    time_audit_summary = pd.DataFrame([{
        "Time_UTC_total_rows": len(df),
        "Time_UTC_parseable_rows": int(parse_ok.sum()),
        "Time_UTC_parse_rate": valid_ratio,
        "Time_UTC_invalid_rows": int((~parse_ok).sum()),
        "Minimum_required_parse_rate": min_parse_rate
    }])

    return parsed.dt.normalize(), time_audit_summary, invalid_by_site, utc_date_range_by_site


def calc_ols_cbi(data, micro_col=MICRO_TEMP_COL, macro_col=MACRO_TEMP_COL,
                 min_hours=72):
    """
    对一个数据子集计算 OLS CBI。

    模型：
        Micro = Intercept + CBI × Macro + error

    返回：
        CBI, Intercept, R2, p_slope, n_hours, flag

    说明：
    - 先剔除宏气温或微气温缺失的小时；
    - 必须达到最低小时数；
    - 宏气温必须有足够变异，否则无法可靠估计斜率。
    """
    d = data[[micro_col, macro_col]].dropna().copy()
    n = len(d)

    if n < min_hours:
        return {
            "CBI": np.nan,
            "Intercept": np.nan,
            "R2": np.nan,
            "p_slope": np.nan,
            "n_hours": n,
            "flag": f"insufficient_hours_lt_{min_hours}"
        }

    if d[macro_col].nunique() < 3 or d[macro_col].std() == 0:
        return {
            "CBI": np.nan,
            "Intercept": np.nan,
            "R2": np.nan,
            "p_slope": np.nan,
            "n_hours": n,
            "flag": "insufficient_macro_variation"
        }

    fit = linregress(d[macro_col], d[micro_col])

    return {
        "CBI": fit.slope,
        "Intercept": fit.intercept,
        "R2": fit.rvalue ** 2,
        "p_slope": fit.pvalue,
        "n_hours": n,
        "flag": "ok"
    }


def classify_delta_cbi(delta_value, near_zero_threshold=NEAR_ZERO_DELTA_CBI):
    """
    将配对Delta_CBI转成便于审计和绘图的方向标签。

    Delta_CBI = CBI_Extreme - CBI_Normal：
    - 正值：Extreme下CBI更高，表示缓冲减弱；
    - 负值：Extreme下CBI更低，表示表观缓冲增强；
    - 接近0：两种状态差异很小。
    """
    if pd.isna(delta_value):
        return "无法判断"
    if delta_value > near_zero_threshold:
        return "Extreme更高_缓冲减弱"
    if delta_value < -near_zero_threshold:
        return "Extreme更低_表观缓冲增强"
    return "基本不变"


def draw_main_lmm_prediction_panel(ax, prediction_df, normal_cbi,
                                   extreme_cbi, delta_cbi, p_value,
                                   show_bottom_note=False, fig=None):
    """
    绘制主LMM预测线面板。

    该函数被单独的03A图和三面板综合图复用，保证两处图形含义一致。
    """
    palette = {"Normal": COLOR_NORMAL, "Extreme": COLOR_EXTREME}

    for status in ["Normal", "Extreme"]:
        d_plot = prediction_df[prediction_df["SPI_Status"] == status]
        slope_value = normal_cbi if status == "Normal" else extreme_cbi
        ax.plot(
            d_plot["Macro_Within"],
            d_plot["Predicted_Micro_Temperature"],
            color=palette[status],
            linewidth=LINEWIDTH_MAIN_PREDICT,
            label=f"{status}（斜率 CBI = {slope_value:.3f}）"
        )

    ax.axvline(
        0,
        color=COLOR_REFERENCE_LINE,
        linewidth=LINEWIDTH_REFERENCE,
        linestyle="--"
    )
    ax.set_xlabel("站点-月份内中心化 ERA5 2 m 温度（°C）")
    ax.set_ylabel("LMM预测的15 cm微气候温度（°C）")
    ax.set_title("A. 主LMM预测线")

    legend_handles = [
        Line2D([0], [0], color=COLOR_NORMAL, lw=LINEWIDTH_MAIN_PREDICT,
               label=f"Normal（斜率 CBI = {normal_cbi:.3f}）"),
        Line2D([0], [0], color=COLOR_EXTREME, lw=LINEWIDTH_MAIN_PREDICT,
               label=f"Extreme（斜率 CBI = {extreme_cbi:.3f}）"),
        Line2D([0], [0], color="none", lw=0,
               label=f"ΔCBI = {delta_cbi:.3f}，p = {p_value:.4g}")
    ]
    ax.legend(
        handles=legend_handles,
        title="SPI状态与主结果",
        loc=LEGEND_LOCATION_BEST
    )

    if show_bottom_note and fig is not None:
        fig.text(
            0.5,
            FIG_BOTTOM_NOTE_Y,
            "横轴 Macro_Within = 当前小时 ERA5 2 m 温度 - 该站点该UTC月份全部有效小时的 ERA5 月均温；"
            "这样做是为了去除站点之间和月份之间的整体冷热背景，只比较同一站点-月份内“相对更热/更冷”时微气候如何响应。",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONTSIZE
        )


def cluster_bootstrap_mean_delta(event_delta_df, site_col=SITE_COL,
                                 delta_col="Delta_CBI",
                                 n_boot=2000, random_seed=20250714):
    """
    站点聚类 bootstrap：用于事件级辅助分析的总体 Delta_CBI 显著性检验。

    为什么不是把每个事件直接当独立样本？
    - 一个站点可能有多个极端事件；
    - 同站点事件共享地形、冠层、传感器和局地环境；
    - 因此同一站点事件不完全独立。

    重抽样单位：
    - 每次从“站点集合”中有放回抽取同样数量的站点；
    - 一个站点被抽到几次，就将其全部事件复制几次；
    - 每次先算每个站点的平均 Delta_CBI，再对站点均值取平均；
    - 结果表示“站点层面的总体平均差异”。

    返回：
    observed_mean, bootstrap_ci_low, bootstrap_ci_high,
    bootstrap_p_two_sided, n_sites, n_events
    """
    d = event_delta_df[[site_col, delta_col]].dropna().copy()

    if d.empty:
        return {
            "observed_mean_delta_cbi": np.nan,
            "ci_low_95": np.nan,
            "ci_high_95": np.nan,
            "p_two_sided": np.nan,
            "n_sites": 0,
            "n_events": 0,
            "flag": "no_valid_event_pairs"
        }

    # 先在站点内平均，避免“事件较多站点”在总体结果中被不合理地过度加权
    site_means = d.groupby(site_col, as_index=False)[delta_col].mean()
    sites = site_means[site_col].tolist()
    observed = site_means[delta_col].mean()

    if len(sites) < 2:
        return {
            "observed_mean_delta_cbi": observed,
            "ci_low_95": np.nan,
            "ci_high_95": np.nan,
            "p_two_sided": np.nan,
            "n_sites": len(sites),
            "n_events": len(d),
            "flag": "fewer_than_2_sites"
        }

    rng = np.random.default_rng(random_seed)
    boot_means = np.empty(n_boot)

    site_mean_map = dict(
        zip(site_means[site_col], site_means[delta_col])
    )

    for b in progress_iter(
        range(n_boot),
        total=n_boot,
        desc="事件级站点聚类Bootstrap",
        kind="Bootstrap"
    ):
        sampled_sites = rng.choice(sites, size=len(sites), replace=True)
        boot_means[b] = np.mean([site_mean_map[s] for s in sampled_sites])

    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    # 双侧经验p值：bootstrap分布相对0的极端程度
    p_lower = np.mean(boot_means <= 0)
    p_upper = np.mean(boot_means >= 0)
    p_two_sided = min(1.0, 2 * min(p_lower, p_upper))

    return {
        "observed_mean_delta_cbi": observed,
        "ci_low_95": ci_low,
        "ci_high_95": ci_high,
        "p_two_sided": p_two_sided,
        "n_sites": len(sites),
        "n_events": len(d),
        "flag": "ok"
    }


def write_text(filepath, text):
    """将模型结果、审计信息等写入 UTF-8 文本文件。"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)


def progress_iter(iterable, total=None, desc="", kind="读取与审计"):
    """
    创建统一风格的 tqdm 进度条。

    参数说明：
    - iterable：要遍历的对象。
    - total：总迭代次数；若 iterable 本身没有长度，必须手动传入。
    - desc：进度条左侧显示的步骤名称。
    - kind：进度条类型，用于选择颜色；颜色配置集中在 PROGRESS_BAR_CONFIG。

    设计目的：
    每个关键步骤只显示一个单行动态进度条，结束后原地清除，避免日志刷屏。
    """
    bar_config = PROGRESS_BAR_CONFIG.get(kind, {})
    tqdm_kwargs = {
        "total": total,
        "desc": desc,
        "unit": "项",
        "dynamic_ncols": True,
        "leave": False
    }
    if bar_config.get("colour"):
        tqdm_kwargs["colour"] = bar_config["colour"]

    try:
        return tqdm(iterable, **tqdm_kwargs)
    except TypeError:
        # 少数旧版tqdm不支持colour参数；此时保留单行动态进度条，只取消颜色。
        tqdm_kwargs.pop("colour", None)
        return tqdm(iterable, **tqdm_kwargs)


def progress_bar(total, desc="", kind="读取与审计"):
    """
    创建手动推进的 tqdm 进度条。

    使用场景：
    某些步骤不是简单for循环，而是多个连续代码块，例如“读取、解析、质控、合并”。
    这类进度条不能用 next(tqdm对象) 推进，应在每个代码块完成后调用 update(1)。
    """
    bar_config = PROGRESS_BAR_CONFIG.get(kind, {})
    tqdm_kwargs = {
        "total": total,
        "desc": desc,
        "unit": "项",
        "dynamic_ncols": True,
        "leave": False
    }
    if bar_config.get("colour"):
        tqdm_kwargs["colour"] = bar_config["colour"]

    try:
        return tqdm(**tqdm_kwargs)
    except TypeError:
        # 兼容旧版tqdm：如果不支持colour参数，则取消颜色但保留进度显示。
        tqdm_kwargs.pop("colour", None)
        return tqdm(**tqdm_kwargs)


def cleanup_runtime_cache():
    """
    清理本次脚本运行过程中可能产生的临时缓存。

    清理范围：
    - 当前脚本所在目录；
    - OUTPUT_DIR 输出目录。

    清理对象：
    - __pycache__ 目录；
    - 后缀为 .tmp / .temp / .cache 的临时文件。

    安全边界：
    - 只清理脚本启动后新产生或更新过的缓存；
    - 不删除任何输入数据、正式输出表格或图片；
    - 若某个缓存文件正在被占用，则跳过。
    """
    roots = [Path(__file__).resolve().parent, OUTPUT_DIR.resolve()]
    cleaned_records = []

    for root in progress_iter(
        roots,
        total=len(roots),
        desc="清理运行缓存",
        kind="清理缓存"
    ):
        if not root.exists():
            continue

        for cache_dir_name in CACHE_DIR_NAMES_TO_CLEAN:
            for cache_dir in root.rglob(cache_dir_name):
                if not cache_dir.is_dir():
                    continue

                try:
                    cache_dir_created_or_updated_this_run = (
                        cache_dir.stat().st_mtime >= SCRIPT_START_TIME
                    )
                except OSError:
                    continue

                if cache_dir_created_or_updated_this_run:
                    try:
                        shutil.rmtree(cache_dir)
                        cleaned_records.append(str(cache_dir))
                    except OSError:
                        pass
                    continue

                # 如果__pycache__目录本身早已存在，只删除本次运行后更新的缓存文件。
                # 删除后若目录为空，再删除空目录；若还有旧缓存，则保留。
                for cache_file in cache_dir.rglob("*"):
                    if not cache_file.is_file():
                        continue
                    try:
                        if cache_file.stat().st_mtime >= SCRIPT_START_TIME:
                            cache_file.unlink()
                            cleaned_records.append(str(cache_file))
                    except OSError:
                        pass
                try:
                    if not any(cache_dir.iterdir()):
                        cache_dir.rmdir()
                        cleaned_records.append(str(cache_dir))
                except OSError:
                    pass

        for suffix in CACHE_FILE_SUFFIXES_TO_CLEAN:
            for cache_file in root.rglob(f"*{suffix}"):
                if cache_file.is_file():
                    try:
                        if cache_file.stat().st_mtime < SCRIPT_START_TIME:
                            continue
                        cache_file.unlink()
                        cleaned_records.append(str(cache_file))
                    except OSError:
                        pass

    return cleaned_records


# =============================================================================
# Step 0. 读取、字段审计和温度质量控制
# =============================================================================

print("\n" + "=" * 80)
print("Step 0: 读取数据、字段审计和质量控制")
print("=" * 80)

step0_tasks = progress_bar(
    total=5,
    desc="Step 0 数据读取与质量控制",
    kind="读取与审计"
)

# ------------------------ 0.1 读取逐小时温度表 --------------------------
# 目的：
# 从逐小时温度表读取宏气候温度（ERA5 2m）和微气候温度（15 cm），
# 并在任何筛选前记录 CSV 原始行数，保证最终报告中的样本流失可追溯。
df_temp_raw = pd.read_csv(TEMP_HOURLY_FILE, low_memory=False)
n_raw_csv_rows = len(df_temp_raw)
step0_tasks.update(1)

required_temp_cols = {
    SITE_COL, ANALYSIS_TIME_COL, MICRO_TEMP_COL, MACRO_TEMP_COL
}
missing_temp_cols = required_temp_cols - set(df_temp_raw.columns)

if missing_temp_cols:
    raise KeyError(
        f"逐小时温度表缺少必须字段：{missing_temp_cols}\n"
        f"实际字段：{df_temp_raw.columns.tolist()}"
    )

df_temp = df_temp_raw.copy()
df_temp[SITE_COL] = normalize_site_id(df_temp[SITE_COL])
(
    df_temp[UTC_DATE_COL],
    time_utc_audit_summary,
    invalid_time_by_site,
    utc_date_range_by_site
) = parse_utc_date_with_audit(df_temp, ANALYSIS_TIME_COL)

# 输出UTC时间轴审计：
# 1. 汇总表记录总体解析率；
# 2. 失败站点分布表定位异常集中在哪些站点；
# 3. 日期范围表检查每个站点UTC日期覆盖是否连续合理。
time_utc_audit_summary.to_csv(
    OUTPUT_DIR / OUTPUT_FILES["time_audit_summary"],
    index=False,
    encoding="utf-8-sig"
)
invalid_time_by_site.to_csv(
    OUTPUT_DIR / OUTPUT_FILES["time_invalid_by_site"],
    index=False,
    encoding="utf-8-sig"
)
utc_date_range_by_site.to_csv(
    OUTPUT_DIR / OUTPUT_FILES["utc_range_by_site"],
    index=False,
    encoding="utf-8-sig"
)

utc_parse_rate = time_utc_audit_summary["Time_UTC_parse_rate"].iloc[0]
if utc_parse_rate < MIN_UTC_PARSE_RATE:
    raise ValueError(
        f"Time_UTC解析有效比例仅为 {utc_parse_rate:.2%}，"
        f"低于 {MIN_UTC_PARSE_RATE:.2%}。"
        f"已输出{OUTPUT_FILES['time_audit_summary']}、"
        f"{OUTPUT_FILES['time_invalid_by_site']}和{OUTPUT_FILES['utc_range_by_site']}，"
        "请先检查UTC时间轴。"
    )

# 解析率达到门槛但不是100%时，保留审计文件，并将少量无UTC日期记录排除出后续分析。
df_temp = df_temp.loc[df_temp[UTC_DATE_COL].notna()].copy()
n_after_utc_parse = len(df_temp)
step0_tasks.update(1)

# 数值化温度列：非数值内容自动变为NaN，随后统一过滤
df_temp[MICRO_TEMP_COL] = pd.to_numeric(
    df_temp[MICRO_TEMP_COL], errors="coerce"
)
df_temp[MACRO_TEMP_COL] = pd.to_numeric(
    df_temp[MACRO_TEMP_COL], errors="coerce"
)

# 若SM列存在，仅数值化并保留；本版主模型不使用SM
if SOIL_MOISTURE_COL in df_temp.columns:
    df_temp[SOIL_MOISTURE_COL] = pd.to_numeric(
        df_temp[SOIL_MOISTURE_COL], errors="coerce"
    )

# 质量控制：只保留宏/微温度均存在、且处于合理范围的小时
# 参数意义：
# - MICRO_TEMP_MIN_C / MICRO_TEMP_MAX_C：保护实测15 cm温度不受漏网异常值影响；
# - MACRO_TEMP_MIN_C / MACRO_TEMP_MAX_C：保护ERA5温度不受无效填充值影响。
# 只有宏微温度均有效的小时才可参与CBI斜率估计。
valid_temp_mask = (
    df_temp[MICRO_TEMP_COL].between(MICRO_TEMP_MIN_C, MICRO_TEMP_MAX_C) &
    df_temp[MACRO_TEMP_COL].between(MACRO_TEMP_MIN_C, MACRO_TEMP_MAX_C)
)

df_temp = df_temp.loc[valid_temp_mask].copy()
n_after_temp_qc = len(df_temp)

# 检查站点-日期-小时是否存在完全重复。
# 当前表可能没有有效的Time_UTC，因此用全部行完全重复作为保守检查。
n_exact_duplicates = df_temp.duplicated().sum()
if n_exact_duplicates > 0:
    df_temp = df_temp.drop_duplicates().copy()
n_after_dedup = len(df_temp)
step0_tasks.update(1)

print(f"CSV原始逐小时记录数: {n_raw_csv_rows:,}")
print(f"UTC时间解析有效记录数: {n_after_utc_parse:,}")
print(f"温度质量控制后记录数: {n_after_temp_qc:,}")
print(f"完全重复记录数（已移除）: {n_exact_duplicates:,}")
print(f"去重后进入SPI合并记录数: {n_after_dedup:,}")
print(f"站点数量: {df_temp[SITE_COL].nunique()}")
print(
    f"UTC日期范围: "
    f"{df_temp[UTC_DATE_COL].min().date()} 至 "
    f"{df_temp[UTC_DATE_COL].max().date()}"
)
print(
    "Time_UTC解析率: "
    f"{time_utc_audit_summary['Time_UTC_parse_rate'].iloc[0]:.2%}"
)

# --------------------------- 0.2 读取SPI宽表 ----------------------------
# SPI宽表格式：
# 第一列为UTC日期，后续每列为一个站点的逐日SPI30d。
# 这里会转成长表，便于按 Site_ID + UTC_Date 合并到逐小时温度记录。
df_spi_wide = pd.read_excel(SPI_DAILY_WIDE_FILE)

if df_spi_wide.shape[1] < 2:
    raise ValueError("SPI宽表必须至少包含1列日期和1列站点SPI数据。")

spi_date_col = df_spi_wide.columns[0]

df_spi_wide[spi_date_col] = pd.to_datetime(
    df_spi_wide[spi_date_col], errors="coerce"
).dt.normalize()

if df_spi_wide[spi_date_col].notna().mean() < 0.95:
    raise ValueError(
        f"SPI日期列 '{spi_date_col}' 无法可靠解析，请检查Excel第一列。"
    )

# 所有SPI站点列名标准化，确保与温度表Site_ID一致
spi_site_cols_original = df_spi_wide.columns[1:].tolist()
spi_site_cols_normalized = [
    normalize_site_id(pd.Series([x])).iloc[0]
    for x in spi_site_cols_original
]

# 如果标准化后列名重复，说明原表中站点列存在潜在问题，立即停止。
if len(set(spi_site_cols_normalized)) != len(spi_site_cols_normalized):
    raise ValueError("SPI宽表标准化后的站点列名存在重复，请检查站点编号。")

rename_map = dict(zip(spi_site_cols_original, spi_site_cols_normalized))
df_spi_wide = df_spi_wide.rename(columns=rename_map)

# 宽转长：最终每行含 UTC_Date、Site_ID、SPI30d
df_spi_long = df_spi_wide.melt(
    id_vars=[spi_date_col],
    var_name=SITE_COL,
    value_name="SPI30d"
).rename(columns={spi_date_col: UTC_DATE_COL})

df_spi_long[SITE_COL] = normalize_site_id(df_spi_long[SITE_COL])
df_spi_long["SPI30d"] = pd.to_numeric(df_spi_long["SPI30d"], errors="coerce")

# 避免重复站点-日期记录导致merge行数膨胀
if df_spi_long.duplicated([SITE_COL, UTC_DATE_COL]).any():
    raise ValueError(
        "SPI长表存在重复的 Site_ID + UTC_Date 组合，"
        "请检查SPI宽表是否含重复日期或重复站点。"
    )

print(f"SPI日期范围: {df_spi_long[UTC_DATE_COL].min().date()} 至 "
      f"{df_spi_long[UTC_DATE_COL].max().date()}")
print(f"SPI站点数: {df_spi_long[SITE_COL].nunique()}")
step0_tasks.update(1)

# -------------------------- 0.3 站点匹配审计 ----------------------------
temp_sites = set(df_temp[SITE_COL].unique())
spi_sites = set(df_spi_long[SITE_COL].unique())

sites_only_temp = sorted(temp_sites - spi_sites)
sites_only_spi = sorted(spi_sites - temp_sites)

if sites_only_temp:
    raise ValueError(
        "以下温度表站点未在SPI表中找到，不能继续：\n"
        + ", ".join(sites_only_temp)
    )

if sites_only_spi:
    print(
        "警告：以下SPI表站点未在温度表中出现，将不会参与分析：\n"
        + ", ".join(sites_only_spi)
    )

# -------------------------- 0.4 合并SPI至小时表 --------------------------
# 合并原则：
# 使用 UTC_Date 作为唯一日尺度键，确保SPI30d日界线与逐小时温度记录一致。
# validate="many_to_one" 强制每个站点-日期只能匹配一个SPI值，避免重复行膨胀。
n_before_merge = len(df_temp)

df_hourly = df_temp.merge(
    df_spi_long,
    on=[SITE_COL, UTC_DATE_COL],
    how="left",
    validate="many_to_one"
)

if len(df_hourly) != n_before_merge:
    raise RuntimeError(
        "合并SPI后小时记录行数发生变化，说明存在重复匹配；停止运行。"
    )

spi_match_rate = df_hourly["SPI30d"].notna().mean()

print(f"SPI匹配率: {spi_match_rate:.2%}")

if spi_match_rate < 0.95:
    raise ValueError(
        "SPI匹配率低于95%，可能是日期边界、Site_ID格式或SPI日期范围不一致。"
        "请先检查，不建议继续运行。"
    )

# -------------------------- 0.5 SPI状态分类 ------------------------------
# 注意边界：
# Extreme：小于等于 -2.0
# Normal ：严格位于 -0.5 和 +0.5 之间
# 其余状态归入Other：
# -2.0 < SPI30d <= -1.5 为重度干旱；
# -1.5 < SPI30d <= -1.0 为中度干旱；
# -1.0 < SPI30d <= -0.5 为轻度干旱；
# 0.5 <= SPI30d < 1.0 为轻度湿润；
# 1.0 <= SPI30d < 1.5 为中度湿润；
# 1.5 <= SPI30d < 2.0 为严重湿润；
# SPI30d >= 2.0 为极端湿润。
# 边界值 SPI == -0.5 和 SPI == 0.5 也归入Other。

conditions = [
    df_hourly["SPI30d"] <= EXTREME_SPI_THRESHOLD,
    (
        (df_hourly["SPI30d"] > NORMAL_SPI_LOW) &
        (df_hourly["SPI30d"] < NORMAL_SPI_HIGH)
    )
]

df_hourly["SPI_Status"] = np.select(
    conditions,
    ["Extreme", "Normal"],
    default="Other"
)

df_hourly["Is_Extreme"] = (
    df_hourly["SPI_Status"] == "Extreme"
).astype(int)

df_hourly["Month"] = df_hourly[UTC_DATE_COL].dt.month.astype(int)
df_hourly["YearMonth"] = df_hourly[UTC_DATE_COL].dt.to_period("M").astype(str)
df_hourly["Site_Month"] = (
    df_hourly[SITE_COL].astype(str) + "_" + df_hourly["YearMonth"]
)

# 输出完整小时数据审计表；后续每步均从此表派生。
df_hourly.to_csv(
    OUTPUT_DIR / OUTPUT_FILES["hourly_merged"],
    index=False,
    encoding="utf-8-sig"
)

status_audit = (
    df_hourly.groupby(["SPI_Status"])
    .agg(
        n_hours=(MICRO_TEMP_COL, "size"),
        n_sites=(SITE_COL, "nunique"),
        n_days=(UTC_DATE_COL, "nunique"),
        mean_spi=("SPI30d", "mean"),
        min_spi=("SPI30d", "min"),
        max_spi=("SPI30d", "max")
    )
    .reset_index()
)

status_audit.to_csv(
    OUTPUT_DIR / OUTPUT_FILES["spi_status_audit"],
    index=False,
    encoding="utf-8-sig"
)

print("\nSPI状态审计：")
print(status_audit.to_string(index=False))
step0_tasks.update(1)
step0_tasks.close()

# 输出绘图与进度条参数说明表：
# 该表记录本次运行所有图形颜色、线宽、透明度、尺寸、DPI与进度条颜色。
# 如果后续需要调图例、折线、透明度或进度条颜色，可以直接对照此表修改参数区。
plot_progress_parameter_rows = [
    {"参数名": "CHINESE_FONT_CANDIDATES", "当前值": " | ".join(CHINESE_FONT_CANDIDATES), "用途": "图片中文字体候选列表，按顺序自动选择已安装字体"},
    {"参数名": "SELECTED_CHINESE_FONT", "当前值": SELECTED_CHINESE_FONT or "未找到", "用途": "本次绘图实际使用的中文字体"},
    {"参数名": "FIG_DPI", "当前值": FIG_DPI, "用途": "所有图片输出分辨率"},
    {"参数名": "FIG_MAIN_PREDICT_SIZE", "当前值": str(FIG_MAIN_PREDICT_SIZE), "用途": "主LMM预测线图尺寸"},
    {"参数名": "FIG_SITE_MONTH_CBI_BOX_SIZE", "当前值": str(FIG_SITE_MONTH_CBI_BOX_SIZE), "用途": "站点月份CBI箱线散点图尺寸"},
    {"参数名": "FIG_SITE_MONTH_CBI_PAIR_SIZE", "当前值": str(FIG_SITE_MONTH_CBI_PAIR_SIZE), "用途": "站点月份CBI配对变化图尺寸"},
    {"参数名": "FIG_REGIONAL_CBI_BOX_SIZE", "当前值": str(FIG_REGIONAL_CBI_BOX_SIZE), "用途": "研究区总体CBI箱线图尺寸"},
    {"参数名": "FIG_MAIN_CBI_COMPOSITE_SIZE", "当前值": str(FIG_MAIN_CBI_COMPOSITE_SIZE), "用途": "主LMM与站点月份CBI综合图尺寸"},
    {"参数名": "FIG_EVENT_PAIR_SIZE", "当前值": str(FIG_EVENT_PAIR_SIZE), "用途": "事件CBI配对图尺寸"},
    {"参数名": "FIG_MONTHLY_TRAJECTORY_SIZE", "当前值": str(FIG_MONTHLY_TRAJECTORY_SIZE), "用途": "月度CBI季节轨迹图尺寸"},
    {"参数名": "FIG_BOTTOM_NOTE_Y", "当前值": FIG_BOTTOM_NOTE_Y, "用途": "图底部说明文字的纵向位置"},
    {"参数名": "COLOR_NORMAL", "当前值": COLOR_NORMAL, "用途": "Normal状态线条/点颜色"},
    {"参数名": "COLOR_EXTREME", "当前值": COLOR_EXTREME, "用途": "Extreme状态线条/点颜色"},
    {"参数名": "COLOR_MONTHLY_MEDIAN", "当前值": COLOR_MONTHLY_MEDIAN, "用途": "月度CBI中位数线颜色"},
    {"参数名": "COLOR_HIGHLIGHT_TEXT", "当前值": COLOR_HIGHLIGHT_TEXT, "用途": "重点站点编号或说明文字高亮颜色"},
    {"参数名": "LINEWIDTH_MAIN_PREDICT", "当前值": LINEWIDTH_MAIN_PREDICT, "用途": "主预测线线宽"},
    {"参数名": "LINEWIDTH_EVENT_PAIR", "当前值": LINEWIDTH_EVENT_PAIR, "用途": "事件配对连线线宽"},
    {"参数名": "LINEWIDTH_MONTHLY_MEDIAN", "当前值": LINEWIDTH_MONTHLY_MEDIAN, "用途": "月度中位数线宽"},
    {"参数名": "LINEWIDTH_CBI_PAIR", "当前值": LINEWIDTH_CBI_PAIR, "用途": "站点月份CBI配对细线线宽"},
    {"参数名": "LINEWIDTH_CBI_MEDIAN_PAIR", "当前值": LINEWIDTH_CBI_MEDIAN_PAIR, "用途": "站点月份CBI配对中位数粗线线宽"},
    {"参数名": "ALPHA_EVENT_PAIR", "当前值": ALPHA_EVENT_PAIR, "用途": "事件配对连线透明度"},
    {"参数名": "ALPHA_SCATTER", "当前值": ALPHA_SCATTER, "用途": "事件配对散点透明度"},
    {"参数名": "ALPHA_EVENT_DELTA_SCATTER", "当前值": ALPHA_EVENT_DELTA_SCATTER, "用途": "事件Delta散点透明度"},
    {"参数名": "ALPHA_CBI_POINTS", "当前值": ALPHA_CBI_POINTS, "用途": "站点月份CBI散点透明度"},
    {"参数名": "ALPHA_CBI_PAIR_LINES", "当前值": ALPHA_CBI_PAIR_LINES, "用途": "站点月份CBI配对线透明度"},
    {"参数名": "ALPHA_MONTHLY_SITE", "当前值": ALPHA_MONTHLY_SITE, "用途": "站点月度轨迹线透明度"},
    {"参数名": "ALPHA_MONTHLY_IQR", "当前值": ALPHA_MONTHLY_IQR, "用途": "月度IQR阴影透明度"},
    {"参数名": "SCATTER_SIZE_EVENT", "当前值": SCATTER_SIZE_EVENT, "用途": "事件配对散点大小"},
    {"参数名": "SCATTER_SIZE_DELTA", "当前值": SCATTER_SIZE_DELTA, "用途": "事件Delta散点大小"},
    {"参数名": "SCATTER_SIZE_CBI", "当前值": SCATTER_SIZE_CBI, "用途": "站点月份CBI散点大小"},
    {"参数名": "MARKER_MONTHLY_MEDIAN", "当前值": MARKER_MONTHLY_MEDIAN, "用途": "月度中位数线标记样式"},
    {"参数名": "LEGEND_LOCATION_BEST", "当前值": LEGEND_LOCATION_BEST, "用途": "图例位置参数"},
    {"参数名": "ANNOTATION_FONTSIZE", "当前值": ANNOTATION_FONTSIZE, "用途": "图内解释文字和重点站点编号的字体大小"},
    {"参数名": "TICKLABEL_FONTSIZE_EVENT_DELTA", "当前值": TICKLABEL_FONTSIZE_EVENT_DELTA, "用途": "事件Delta点图纵轴标签字号"},
    {"参数名": "JITTER_WIDTH_CBI_BOX", "当前值": JITTER_WIDTH_CBI_BOX, "用途": "CBI箱线图散点左右抖动宽度"},
    {"参数名": "MIN_SITE_MONTH_STATUS_CBI_HOURS", "当前值": MIN_SITE_MONTH_STATUS_CBI_HOURS, "用途": "站点月份状态CBI估计的最低有效小时数"},
    {"参数名": "MIN_SITE_MONTH_STATUS_MACRO_SD", "当前值": MIN_SITE_MONTH_STATUS_MACRO_SD, "用途": "站点月份状态CBI估计的最低ERA5温度标准差"},
    {"参数名": "NEAR_ZERO_DELTA_CBI", "当前值": NEAR_ZERO_DELTA_CBI, "用途": "配对变化中定义基本不变的Delta_CBI阈值"},
    {"参数名": "PYTHON_ENV_DIR", "当前值": PYTHON_ENV_DIR, "用途": "建议运行脚本的Python环境目录"},
    {"参数名": "PYTHON_INTERPRETER", "当前值": PYTHON_INTERPRETER, "用途": "建议运行脚本的Python解释器路径"},
]

for progress_kind, progress_cfg in PROGRESS_BAR_CONFIG.items():
    plot_progress_parameter_rows.append({
        "参数名": f"进度条颜色_{progress_kind}",
        "当前值": progress_cfg.get("colour", ""),
        "用途": "tqdm终端单行动态进度条颜色"
    })

pd.DataFrame(plot_progress_parameter_rows).to_csv(
    OUTPUT_DIR / OUTPUT_FILES["plot_progress_parameters"],
    index=False,
    encoding="utf-8-sig"
)


# =============================================================================
# Step 1. 主LMM的分析样本构建
# =============================================================================

print("\n" + "=" * 80)
print("Step 1: 构建主LMM分析样本")
print("=" * 80)

# 先使用所有通过温度质量控制的小时记录，计算“完整站点-月份”的宏气候背景。
# 这里不区分SPI状态，也不排除Other；否则月平均ERA5温度会被Extreme/Normal样本结构扭曲。
# 输出用途：
# 后续 Macro_Within = 当前小时ERA5温度 - 完整站点-月份ERA5均值。
# 这个定义把每个小时放回其所在站点和UTC月份的真实宏气候背景中，
# 避免只用Extreme/Normal子样本计算月均温造成偏差。
site_month_macro_all = (
    df_hourly.groupby("Site_Month")
    .agg(
        Site_ID=(SITE_COL, "first"),
        YearMonth=("YearMonth", "first"),
        Month=("Month", "first"),
        n_all_valid_hours=(MACRO_TEMP_COL, "size"),
        Macro_Mean_SiteMonth_AllValid=(MACRO_TEMP_COL, "mean"),
        Macro_SD_SiteMonth_AllValid=(MACRO_TEMP_COL, "std")
    )
    .reset_index()
)

# 主模型候选小时仅使用 Extreme 和 Normal 状态。
df_lmm_all = df_hourly.loc[
    df_hourly["SPI_Status"].isin(["Extreme", "Normal"])
].copy()

# -------------------------------------------------------------------------
# 关键设计：仅保留“同一站点-月份中同时存在足量Extreme与Normal”的站点-月份层
#
# 原因：
# 1. 这样Extreme与Normal是在同一站点、同一月份背景下可比较；
# 2. 若某站点某月只有Extreme或只有Normal，则干旱状态与月份完全重合；
# 3. 将这种单状态站点-月放进含月份项的模型，会放大共线性风险。
# 4. 每种状态至少72小时，避免用极少数小时估计斜率差异。
#
# 此筛选是控制季节混淆的关键步骤，而不是删除“不喜欢的数据”。
# 筛选前后样本构成都会输出，保证过程透明。
# -------------------------------------------------------------------------

site_month_status_n = (
    df_lmm_all.groupby(["Site_Month", "SPI_Status"])
    .size()
    .unstack(fill_value=0)
)

for status_name in ["Extreme", "Normal"]:
    if status_name not in site_month_status_n.columns:
        site_month_status_n[status_name] = 0

site_month_status_n = site_month_status_n.reset_index()

other_hours_by_site_month = (
    df_hourly.loc[df_hourly["SPI_Status"] == "Other"]
    .groupby("Site_Month")
    .size()
    .rename("Other")
    .reset_index()
)

site_month_status_n = site_month_status_n.merge(
    other_hours_by_site_month,
    on="Site_Month",
    how="left"
)
site_month_status_n["Other"] = site_month_status_n["Other"].fillna(0).astype(int)

site_month_status_n = site_month_status_n.merge(
    site_month_macro_all,
    on="Site_Month",
    how="left"
)

extreme_normal_macro_mean = (
    df_lmm_all.groupby("Site_Month")[MACRO_TEMP_COL]
    .mean()
    .rename("Macro_Mean_SiteMonth_ExtremeNormalOnly")
    .reset_index()
)

site_month_status_n = site_month_status_n.merge(
    extreme_normal_macro_mean,
    on="Site_Month",
    how="left"
)
site_month_status_n["Macro_Mean_Diff_ExtremeNormal_minus_AllValid"] = (
    site_month_status_n["Macro_Mean_SiteMonth_ExtremeNormalOnly"] -
    site_month_status_n["Macro_Mean_SiteMonth_AllValid"]
)

sensitivity_rows = []
for threshold in progress_iter(
    STATUS_HOUR_SENSITIVITY_THRESHOLDS,
    total=len(STATUS_HOUR_SENSITIVITY_THRESHOLDS),
    desc="状态小时门槛敏感性审计",
    kind="样本筛选"
):
    keep_mask = (
        (site_month_status_n["Extreme"] >= threshold) &
        (site_month_status_n["Normal"] >= threshold)
    )
    retained = site_month_status_n.loc[keep_mask]
    retained_site_months = set(retained["Site_Month"])
    retained_hours = df_lmm_all.loc[
        df_lmm_all["Site_Month"].isin(retained_site_months)
    ]
    sensitivity_rows.append({
        "MIN_STATUS_HOURS_PER_SITE_MONTH": threshold,
        "retained_site_months": len(retained_site_months),
        "retained_sites": retained[SITE_COL].nunique(),
        "retained_hourly_records": len(retained_hours),
        "retained_extreme_hours": int((retained_hours["SPI_Status"] == "Extreme").sum()),
        "retained_normal_hours": int((retained_hours["SPI_Status"] == "Normal").sum())
    })

status_hour_sensitivity_audit = pd.DataFrame(sensitivity_rows)

# 主分析采用 MIN_STATUS_HOURS_PER_SITE_MONTH 对站点-月份做正式筛选。
# 敏感性审计表同时给出 24/48/72/120 小时阈值下保留样本量，
# 便于运行后直接判断主结论是否可能受阈值选择影响。
eligible_site_months = site_month_status_n.loc[
    (site_month_status_n["Extreme"] >= MIN_STATUS_HOURS_PER_SITE_MONTH) &
    (site_month_status_n["Normal"] >= MIN_STATUS_HOURS_PER_SITE_MONTH),
    "Site_Month"
].tolist()

df_lmm = df_lmm_all.loc[
    df_lmm_all["Site_Month"].isin(eligible_site_months)
].copy()

# 若筛选后极端样本完全消失，必须停止，不能拟合无意义模型。
if df_lmm["Is_Extreme"].sum() == 0:
    raise RuntimeError(
        "筛选后没有Extreme小时记录。请检查SPI阈值、日期合并和数据覆盖。"
    )

if (df_lmm["Is_Extreme"] == 0).sum() == 0:
    raise RuntimeError(
        "筛选后没有Normal小时记录。请检查Normal SPI阈值与数据覆盖。"
    )

# -------------------------------------------------------------------------
# 宏气候温度的站点-月份内中心化
#
# Macro_within = 当前小时ERA5温度 - 该站点该月“全部有效小时”的ERA5平均温度
#
# 该变量表示“该小时相对该站点该月温度背景的升降幅度”。
# 它是估计小时尺度CBI的核心变量。
# 同时保留 Macro_Mean_SiteMonth_C 作为固定效应协变量，
# 用于控制不同站点-月份整体冷暖背景的差异。
# -------------------------------------------------------------------------

df_lmm = df_lmm.merge(
    site_month_macro_all[
        [
            "Site_Month",
            "Macro_Mean_SiteMonth_AllValid",
            "Macro_SD_SiteMonth_AllValid",
            "n_all_valid_hours"
        ]
    ],
    on="Site_Month",
    how="left",
    validate="many_to_one"
)

df_lmm["Macro_Within"] = (
    df_lmm[MACRO_TEMP_COL] - df_lmm["Macro_Mean_SiteMonth_AllValid"]
)

# 检查每个站点-月宏气温是否有足够波动；没有波动不能估计斜率。
macro_sd_site_month = (
    df_lmm.groupby("Site_Month")["Macro_Within"].std()
)

valid_macro_site_months = macro_sd_site_month[
    macro_sd_site_month > 0
].index.tolist()

df_lmm = df_lmm.loc[
    df_lmm["Site_Month"].isin(valid_macro_site_months)
].copy()

# Month作为分类变量；Site_ID作为混合模型分组变量。
df_lmm["Month_Factor"] = df_lmm["Month"].astype(str)
df_lmm["Site_Group"] = df_lmm[SITE_COL].astype(str)

# 为提高模型解释性，整体冷热背景也进行中心化。
# 它控制“这个站点这个月总体偏热/偏冷”的差异。
df_lmm["Macro_Mean_SiteMonth_C"] = (
    df_lmm["Macro_Mean_SiteMonth_AllValid"] -
    df_lmm["Macro_Mean_SiteMonth_AllValid"].mean()
)

# 保存LMM样本构成与筛选后分析表
lmm_sample_audit = (
    df_lmm.groupby(["Month", "SPI_Status"])
    .agg(
        n_hours=(MICRO_TEMP_COL, "size"),
        n_sites=(SITE_COL, "nunique"),
        n_site_months=("Site_Month", "nunique"),
        mean_spi=("SPI30d", "mean"),
        mean_macro=(MACRO_TEMP_COL, "mean"),
        mean_micro=(MICRO_TEMP_COL, "mean")
    )
    .reset_index()
    .sort_values(["Month", "SPI_Status"])
)

lmm_sample_audit.to_csv(
    OUTPUT_DIR / OUTPUT_FILES["lmm_sample_audit"],
    index=False,
    encoding="utf-8-sig"
)

site_month_status_n.to_csv(
    OUTPUT_DIR / OUTPUT_FILES["site_month_macro_audit"],
    index=False,
    encoding="utf-8-sig"
)

status_hour_sensitivity_audit.to_csv(
    OUTPUT_DIR / OUTPUT_FILES["status_threshold_sensitivity"],
    index=False,
    encoding="utf-8-sig"
)

df_lmm.to_csv(
    OUTPUT_DIR / OUTPUT_FILES["lmm_dataset"],
    index=False,
    encoding="utf-8-sig"
)

print(f"LMM候选小时数（Extreme + Normal）: {len(df_lmm_all):,}")
print(
    "符合“同站点-同月份Extreme和Normal均达到"
    f"{MIN_STATUS_HOURS_PER_SITE_MONTH}小时”条件的小时数: {len(df_lmm):,}"
)
print(f"有效站点-月份层数: {df_lmm['Site_Month'].nunique()}")
print(f"有效站点数: {df_lmm[SITE_COL].nunique()}")
print("\nLMM样本按月份与SPI状态分布：")
print(lmm_sample_audit.to_string(index=False))


# =============================================================================
# Step 2. 主分析：逐小时线性混合效应模型（LMM）
# =============================================================================

print("\n" + "=" * 80)
print("Step 2: 拟合主LMM —— Macro × Extreme 交互项")
print("=" * 80)

# -----------------------------------------------------------------------------
# 主模型公式
#
# Observed_T15cm_C ~
#     Macro_Within
#   + Is_Extreme
#   + Macro_Within:Is_Extreme            <- 核心项 β3
#   + C(Month_Factor)                    <- 月份截距差异
#   + Macro_Within:C(Month_Factor)       <- 月份CBI差异
#   + Macro_Mean_SiteMonth_C             <- 站点-月份整体宏气温背景
#
# 其中：
#   β1 = 参考月份、Normal条件下的CBI
#   β3 = Extreme相对于Normal的CBI变化量，即 Delta_CBI
#
# 站点随机效应：
#   re_formula = "1 + Macro_Within"
#
# 即允许不同站点有：
#   - 不同基础微气候温度（随机截距）
#   - 不同基础CBI（随机宏气候斜率）
#
# 注意：
# 由于加入 Macro_Within × Month，CBI会随月份变化。
# 因此“总体Normal CBI”和“总体Extreme CBI”将在模型拟合后
# 通过各月份样本权重加权计算，而不是错误地仅解释β1。
# -----------------------------------------------------------------------------

main_formula = (
    f"{MICRO_TEMP_COL} ~ "
    "Macro_Within * Is_Extreme + "
    "C(Month_Factor) + "
    "Macro_Within:C(Month_Factor) + "
    "Macro_Mean_SiteMonth_C"
)

try:
    lmm_fit_steps = progress_bar(
        total=2,
        desc="主LMM模型拟合",
        kind="模型拟合"
    )

    main_lmm = smf.mixedlm(
        formula=main_formula,
        data=df_lmm,
        groups=df_lmm["Site_Group"],
        re_formula="1 + Macro_Within"
    )
    lmm_fit_steps.update(1)

    main_lmm_result = main_lmm.fit(
        reml=True,
        method=LMM_METHOD,
        maxiter=LMM_MAXITER,
        disp=False
    )
    lmm_fit_steps.update(1)
    lmm_fit_steps.close()

except Exception as e:
    if "lmm_fit_steps" in locals():
        lmm_fit_steps.close()
    raise RuntimeError(
        "主LMM拟合失败。\n"
        "可能原因包括：样本结构不足、随机斜率结构过复杂或模型未收敛。\n"
        f"请保留完整报错并检查 {OUTPUT_FILES['lmm_sample_audit']}。\n"
        f"原始错误：{repr(e)}"
    )

# 保存完整模型summary
write_text(
    OUTPUT_DIR / OUTPUT_FILES["main_lmm_summary"],
    main_lmm_result.summary().as_text()
)

print(main_lmm_result.summary())

# -----------------------------------------------------------------------------
# 关键交互项 Macro_Within:Is_Extreme 的提取
#
# statsmodels产生的参数名可能为：
# "Macro_Within:Is_Extreme" 或 "Is_Extreme:Macro_Within"
# 因此使用候选名稳健搜索，避免因列顺序差异导致取值失败。
# -----------------------------------------------------------------------------

interaction_candidates = [
    "Macro_Within:Is_Extreme",
    "Is_Extreme:Macro_Within"
]

interaction_term = next(
    (x for x in interaction_candidates if x in main_lmm_result.params.index),
    None
)

if interaction_term is None:
    raise KeyError(
        "未找到 Macro_Within × Is_Extreme 交互项。"
        f"模型参数名为：{main_lmm_result.params.index.tolist()}"
    )

interaction_beta = main_lmm_result.params[interaction_term]
interaction_ci = main_lmm_result.conf_int().loc[interaction_term].tolist()
interaction_p = main_lmm_result.pvalues[interaction_term]

# -----------------------------------------------------------------------------
# 计算“总体加权 CBI”
#
# 因为模型允许不同月份具有不同的 Macro_Within 斜率：
# Normal月度斜率 = Macro_Within系数 + 对应月份的交互系数
# Extreme月度斜率 = Normal月度斜率 + Macro_Within:Is_Extreme系数
#
# 这里按主LMM样本中每个月的小时数加权，获得研究期内的平均CBI。
# -----------------------------------------------------------------------------

params = main_lmm_result.params
base_macro_coef = params.get("Macro_Within", np.nan)

month_levels = sorted(df_lmm["Month_Factor"].unique(), key=lambda x: int(x))
reference_month = sorted(month_levels, key=lambda x: int(x))[0]

monthly_cbi_rows = []

for month_level in progress_iter(
    month_levels,
    total=len(month_levels),
    desc="计算主LMM月份CBI",
    kind="样本筛选"
):
    # statsmodels treatment coding: 最小/月度参考水平没有交互项系数，取0
    month_interaction_candidates = [
        f"Macro_Within:C(Month_Factor)[T.{month_level}]",
        f"C(Month_Factor)[T.{month_level}]:Macro_Within"
    ]

    month_slope_adjustment = 0.0

    for term in month_interaction_candidates:
        if term in params.index:
            month_slope_adjustment = params[term]
            break

    cbi_normal_month = base_macro_coef + month_slope_adjustment
    cbi_extreme_month = cbi_normal_month + interaction_beta

    n_month_hours = (df_lmm["Month_Factor"] == month_level).sum()

    monthly_cbi_rows.append({
        "Month": int(month_level),
        "n_hours": n_month_hours,
        "CBI_Normal_model": cbi_normal_month,
        "CBI_Extreme_model": cbi_extreme_month,
        "Delta_CBI_Extreme_minus_Normal": interaction_beta
    })

df_lmm_cbi_month = pd.DataFrame(monthly_cbi_rows)

total_n_hours = df_lmm_cbi_month["n_hours"].sum()
weighted_normal_cbi = np.average(
    df_lmm_cbi_month["CBI_Normal_model"],
    weights=df_lmm_cbi_month["n_hours"]
)
weighted_extreme_cbi = np.average(
    df_lmm_cbi_month["CBI_Extreme_model"],
    weights=df_lmm_cbi_month["n_hours"]
)

df_lmm_cbi_month.to_csv(
    OUTPUT_DIR / OUTPUT_FILES["main_lmm_monthly_cbi"],
    index=False,
    encoding="utf-8-sig"
)

main_lmm_key_results = pd.DataFrame([{
    "interaction_term": interaction_term,
    "Delta_CBI_Extreme_minus_Normal": interaction_beta,
    "Delta_CBI_CI_low_95": interaction_ci[0],
    "Delta_CBI_CI_high_95": interaction_ci[1],
    "Delta_CBI_p_value": interaction_p,
    "weighted_CBI_Normal": weighted_normal_cbi,
    "weighted_CBI_Extreme": weighted_extreme_cbi,
    "n_hourly_records_LMM": len(df_lmm),
    "n_sites_LMM": df_lmm[SITE_COL].nunique(),
    "n_site_months_LMM": df_lmm["Site_Month"].nunique(),
    "formula": main_formula
}])

main_lmm_key_results.to_csv(
    OUTPUT_DIR / OUTPUT_FILES["main_lmm_key"],
    index=False,
    encoding="utf-8-sig"
)

print("\n主LMM核心结果：")
print(main_lmm_key_results.to_string(index=False))

if interaction_beta > 0:
    print(
        "\n解释：Macro × Extreme 交互项为正，"
        "表示极端干旱期间CBI更高，森林微气候缓冲减弱。"
    )
elif interaction_beta < 0:
    print(
        "\n解释：Macro × Extreme 交互项为负，"
        "表示极端干旱期间CBI更低，呈现表观缓冲增强。"
    )
else:
    print("\n解释：交互项为0，Extreme与Normal条件下CBI相同。")


# =============================================================================
# Step 3. 主LMM预测图：Normal 与 Extreme 的斜率差异
# =============================================================================

print("\n" + "=" * 80)
print("Step 3: 绘制主LMM预测线")
print("=" * 80)

# 使用各月份出现频率作为权重，构造“平均月份情形”下的预测。
# 图形重点是两条线斜率差异；不是展示某一个具体站点的绝对温度。

month_weights = (
    df_lmm["Month_Factor"]
    .value_counts(normalize=True)
    .to_dict()
)

macro_grid = np.linspace(
    df_lmm["Macro_Within"].quantile(0.01),
    df_lmm["Macro_Within"].quantile(0.99),
    100
)

prediction_rows = []

# 为每一个月份和干旱状态构建预测，再按月份权重加权。
# Macro_Mean_SiteMonth_C 固定为0：代表平均宏气候背景。
# 预测线只用于展示固定效应下 Normal 与 Extreme 斜率差异；
# 不叠加任何具体站点的随机截距或随机斜率，因此图形反映总体平均关系。
prediction_status_pairs = [("Normal", 0), ("Extreme", 1)]
for drought_status, is_extreme_value in progress_iter(
    prediction_status_pairs,
    total=len(prediction_status_pairs),
    desc="生成主LMM预测线",
    kind="模型预测"
):
    weighted_pred = np.zeros_like(macro_grid, dtype=float)

    for month_level, weight in month_weights.items():
        pred_df = pd.DataFrame({
            "Macro_Within": macro_grid,
            "Is_Extreme": is_extreme_value,
            "Month_Factor": month_level,
            "Macro_Mean_SiteMonth_C": 0.0,
            "Site_Group": df_lmm["Site_Group"].iloc[0]
        })

        # fixed effects prediction：不包含任一具体站点的随机效应
        pred = main_lmm_result.predict(pred_df)
        weighted_pred += weight * np.asarray(pred)

    for x_value, y_value in zip(macro_grid, weighted_pred):
        prediction_rows.append({
            "Macro_Within": x_value,
            "SPI_Status": drought_status,
            "Predicted_Micro_Temperature": y_value
        })

df_lmm_prediction = pd.DataFrame(prediction_rows)

df_lmm_prediction.to_csv(
    OUTPUT_DIR / OUTPUT_FILES["main_prediction_lines"],
    index=False,
    encoding="utf-8-sig"
)

fig, ax = plt.subplots(figsize=FIG_MAIN_PREDICT_SIZE)

draw_main_lmm_prediction_panel(
    ax,
    df_lmm_prediction,
    weighted_normal_cbi,
    weighted_extreme_cbi,
    interaction_beta,
    interaction_p,
    show_bottom_note=True,
    fig=fig
)
ax.set_title(
    "主LMM：Normal 与 Extreme 条件下的宏—微气候响应斜率"
)

fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(
    OUTPUT_DIR / OUTPUT_FILES["main_prediction_plot"],
    dpi=FIG_DPI,
    bbox_inches="tight"
)
fig.savefig(
    OUTPUT_DIR / OUTPUT_FILES["main_prediction_plot_split"],
    dpi=FIG_DPI,
    bbox_inches="tight"
)
plt.close(fig)


# =============================================================================
# Step 3B. 站点-月份-状态CBI补充图：箱线、散点、配对变化
# =============================================================================

print("\n" + "=" * 80)
print("Step 3B: 计算并绘制站点-月份-状态CBI补充图")
print("=" * 80)

# -------------------------------------------------------------------------
# 设计目的：
# 主LMM给出“逐小时层面”的正式推断；本节把数据压缩到
# Site_ID × YearMonth × SPI_Status 的独立CBI估计，用于描述性展示。
#
# 每个CBI点都来自同一站点、同一UTC月份、同一SPI状态下的逐小时OLS：
#     Observed_T15cm_C = Intercept + CBI × ERA5_T2m_C
#
# 注意：
# 这些箱线图和配对图用于展示分布和异质性，不替代主LMM显著性检验。
# -------------------------------------------------------------------------

site_month_status_rows = []

# -------------------------------------------------------------------------
# 先进行站点层面的前置筛选：
# 如果某个站点在研究年份内从未出现Extreme小时，则该站点没有极端干旱背景，
# 它的Normal CBI不能用于“Normal vs Extreme缓冲差异”的描述性比较。
#
# 因此，03B/03C/03D以及综合图的站点-月份CBI只来自全年至少出现过
# 一个Extreme小时的站点。这样Normal样本不会被“全年无Extreme的站点”膨胀。
# -------------------------------------------------------------------------
extreme_sites = set(
    df_lmm_all.loc[
        df_lmm_all["SPI_Status"] == "Extreme",
        SITE_COL
    ].unique()
)

all_lmm_candidate_sites = set(df_lmm_all[SITE_COL].unique())
excluded_no_extreme_sites = sorted(all_lmm_candidate_sites - extreme_sites)

extreme_site_filter_audit = (
    df_lmm_all.groupby(SITE_COL)
    .agg(
        n_extreme_hours=("SPI_Status", lambda x: int((x == "Extreme").sum())),
        n_normal_hours=("SPI_Status", lambda x: int((x == "Normal").sum())),
        n_site_months=("Site_Month", "nunique")
    )
    .reset_index()
)
extreme_site_filter_audit["Site_Has_Extreme_In_Year"] = (
    extreme_site_filter_audit["n_extreme_hours"] > 0
)
extreme_site_filter_audit["Included_In_SiteMonth_CBI_Figures"] = (
    extreme_site_filter_audit["Site_Has_Extreme_In_Year"]
)
extreme_site_filter_audit["Exclusion_Reason"] = np.where(
    extreme_site_filter_audit["Site_Has_Extreme_In_Year"],
    "",
    "全年无Extreme小时，Normal CBI不进入Normal-vs-Extreme比较图"
)

extreme_site_filter_audit.to_csv(
    OUTPUT_DIR / OUTPUT_FILES["extreme_site_filter_audit"],
    index=False,
    encoding="utf-8-sig"
)

df_site_month_cbi_source = df_lmm_all.loc[
    df_lmm_all[SITE_COL].isin(extreme_sites)
].copy()

print(
    "站点-月份CBI补充图站点筛选："
    f"候选站点 {len(all_lmm_candidate_sites)} 个；"
    f"全年有Extreme站点 {len(extreme_sites)} 个；"
    f"全年无Extreme被排除站点 {len(excluded_no_extreme_sites)} 个。"
)

site_month_status_groups = list(
    df_site_month_cbi_source.groupby([SITE_COL, "YearMonth", "SPI_Status"])
)

for (site, yearmonth, status), group in progress_iter(
    site_month_status_groups,
    total=len(site_month_status_groups),
    desc="计算站点月份状态CBI",
    kind="站点月份CBI"
):
    cbi_result = calc_ols_cbi(
        group,
        min_hours=MIN_SITE_MONTH_STATUS_CBI_HOURS
    )

    macro_values = group[MACRO_TEMP_COL].dropna()
    micro_values = group[MICRO_TEMP_COL].dropna()
    macro_sd = macro_values.std()
    macro_min = macro_values.min() if not macro_values.empty else np.nan
    macro_max = macro_values.max() if not macro_values.empty else np.nan
    macro_range = macro_max - macro_min if pd.notna(macro_min) and pd.notna(macro_max) else np.nan

    pass_min_hours = cbi_result["n_hours"] >= MIN_SITE_MONTH_STATUS_CBI_HOURS
    pass_macro_sd = (
        pd.notna(macro_sd) and
        macro_sd >= MIN_SITE_MONTH_STATUS_MACRO_SD
    )
    pass_cbi_filter = (
        cbi_result["flag"] == "ok" and
        pass_min_hours and
        pass_macro_sd
    )

    site_month_status_rows.append({
        SITE_COL: site,
        "YearMonth": yearmonth,
        "Month": int(group["Month"].iloc[0]),
        "SPI_Status": status,
        "CBI": cbi_result["CBI"],
        "Intercept": cbi_result["Intercept"],
        "R2": cbi_result["R2"],
        "p_slope": cbi_result["p_slope"],
        "n_hours": cbi_result["n_hours"],
        "Macro_SD": macro_sd,
        "Macro_Range": macro_range,
        "Macro_Min": macro_min,
        "Macro_Max": macro_max,
        "Micro_Mean": micro_values.mean() if not micro_values.empty else np.nan,
        "Macro_Mean": macro_values.mean() if not macro_values.empty else np.nan,
        "Site_Has_Extreme_In_Year": site in extreme_sites,
        "Pass_Min_Hours": bool(pass_min_hours),
        "Pass_Macro_SD": bool(pass_macro_sd),
        "Pass_CBI_Filter": bool(pass_cbi_filter),
        "CBI_flag": cbi_result["flag"]
    })

df_site_month_status_cbi = pd.DataFrame(site_month_status_rows)
df_site_month_status_cbi.to_csv(
    OUTPUT_DIR / OUTPUT_FILES["site_month_status_cbi"],
    index=False,
    encoding="utf-8-sig"
)

df_site_month_cbi_plot = df_site_month_status_cbi.loc[
    (df_site_month_status_cbi["Pass_CBI_Filter"]) &
    (df_site_month_status_cbi["SPI_Status"].isin(["Normal", "Extreme"]))
].copy()

if df_site_month_cbi_plot.empty:
    print(
        "警告：没有满足站点-月份-状态CBI筛选条件的记录，"
        "跳过03B/03C/03D和综合图。"
    )
    df_site_month_cbi_pair = pd.DataFrame()
    df_site_month_cbi_pair_summary = pd.DataFrame()
    df_regional_cbi_summary = pd.DataFrame()
    df_site_month_cbi_pair.to_csv(
        OUTPUT_DIR / OUTPUT_FILES["site_month_cbi_pair_audit"],
        index=False,
        encoding="utf-8-sig"
    )
    df_site_month_cbi_pair_summary.to_csv(
        OUTPUT_DIR / OUTPUT_FILES["site_month_cbi_pair_summary"],
        index=False,
        encoding="utf-8-sig"
    )
    df_regional_cbi_summary.to_csv(
        OUTPUT_DIR / OUTPUT_FILES["regional_cbi_summary"],
        index=False,
        encoding="utf-8-sig"
    )
else:
    cbi_wide = (
        df_site_month_cbi_plot
        .pivot_table(
            index=[SITE_COL, "YearMonth", "Month"],
            columns="SPI_Status",
            values=["CBI", "R2", "n_hours", "Macro_SD"],
            aggfunc="first"
        )
    )
    cbi_wide.columns = [
        f"{value_name}_{status_name}"
        for value_name, status_name in cbi_wide.columns
    ]
    cbi_wide = cbi_wide.reset_index()

    required_pair_cols = ["CBI_Normal", "CBI_Extreme"]
    if all(col in cbi_wide.columns for col in required_pair_cols):
        df_site_month_cbi_pair = cbi_wide.dropna(
            subset=required_pair_cols
        ).copy()
    else:
        df_site_month_cbi_pair = pd.DataFrame()

    if not df_site_month_cbi_pair.empty:
        df_site_month_cbi_pair["Delta_CBI_Extreme_minus_Normal"] = (
            df_site_month_cbi_pair["CBI_Extreme"] -
            df_site_month_cbi_pair["CBI_Normal"]
        )
        df_site_month_cbi_pair["Direction"] = (
            df_site_month_cbi_pair["Delta_CBI_Extreme_minus_Normal"]
            .apply(classify_delta_cbi)
        )

        df_site_month_cbi_pair.to_csv(
            OUTPUT_DIR / OUTPUT_FILES["site_month_cbi_pair_audit"],
            index=False,
            encoding="utf-8-sig"
        )

        n_pairs = len(df_site_month_cbi_pair)
        n_positive = int(
            (df_site_month_cbi_pair["Delta_CBI_Extreme_minus_Normal"] >
             NEAR_ZERO_DELTA_CBI).sum()
        )
        n_negative = int(
            (df_site_month_cbi_pair["Delta_CBI_Extreme_minus_Normal"] <
             -NEAR_ZERO_DELTA_CBI).sum()
        )
        n_near_zero = int(n_pairs - n_positive - n_negative)

        df_site_month_cbi_pair_summary = pd.DataFrame([{
            "n_pairs": n_pairs,
            "n_positive_delta": n_positive,
            "n_negative_delta": n_negative,
            "n_near_zero_delta": n_near_zero,
            "positive_percent": n_positive / n_pairs if n_pairs > 0 else np.nan,
            "negative_percent": n_negative / n_pairs if n_pairs > 0 else np.nan,
            "near_zero_percent": n_near_zero / n_pairs if n_pairs > 0 else np.nan,
            "median_CBI_Normal": df_site_month_cbi_pair["CBI_Normal"].median(),
            "median_CBI_Extreme": df_site_month_cbi_pair["CBI_Extreme"].median(),
            "median_Delta_CBI": df_site_month_cbi_pair["Delta_CBI_Extreme_minus_Normal"].median(),
            "mean_Delta_CBI": df_site_month_cbi_pair["Delta_CBI_Extreme_minus_Normal"].mean()
        }])
    else:
        df_site_month_cbi_pair_summary = pd.DataFrame([{
            "n_pairs": 0,
            "n_positive_delta": 0,
            "n_negative_delta": 0,
            "n_near_zero_delta": 0,
            "positive_percent": np.nan,
            "negative_percent": np.nan,
            "near_zero_percent": np.nan,
            "median_CBI_Normal": np.nan,
            "median_CBI_Extreme": np.nan,
            "median_Delta_CBI": np.nan,
            "mean_Delta_CBI": np.nan
        }])
        df_site_month_cbi_pair.to_csv(
            OUTPUT_DIR / OUTPUT_FILES["site_month_cbi_pair_audit"],
            index=False,
            encoding="utf-8-sig"
        )

    df_site_month_cbi_pair_summary.to_csv(
        OUTPUT_DIR / OUTPUT_FILES["site_month_cbi_pair_summary"],
        index=False,
        encoding="utf-8-sig"
    )

    regional_summary_rows = []
    for status in ["Normal", "Extreme"]:
        d_status = df_site_month_cbi_plot.loc[
            df_site_month_cbi_plot["SPI_Status"] == status
        ]
        regional_summary_rows.append({
            "SPI_Status": status,
            "n_site_month_status_cbi": len(d_status),
            "n_sites": d_status[SITE_COL].nunique(),
            "n_site_months": d_status[[SITE_COL, "YearMonth"]].drop_duplicates().shape[0],
            "source_sites_rule": "仅统计全年至少有1个Extreme小时的站点",
            "n_source_extreme_sites": len(extreme_sites),
            "n_excluded_no_extreme_sites": len(excluded_no_extreme_sites),
            "median_CBI": d_status["CBI"].median(),
            "mean_CBI": d_status["CBI"].mean(),
            "q25_CBI": d_status["CBI"].quantile(0.25),
            "q75_CBI": d_status["CBI"].quantile(0.75),
            "min_CBI": d_status["CBI"].min(),
            "max_CBI": d_status["CBI"].max()
        })

    df_regional_cbi_summary = pd.DataFrame(regional_summary_rows)
    if {"Normal", "Extreme"}.issubset(set(df_regional_cbi_summary["SPI_Status"])):
        median_map = dict(
            zip(
                df_regional_cbi_summary["SPI_Status"],
                df_regional_cbi_summary["median_CBI"]
            )
        )
        df_regional_cbi_summary["median_Delta_Extreme_minus_Normal"] = (
            median_map.get("Extreme", np.nan) -
            median_map.get("Normal", np.nan)
        )
    else:
        df_regional_cbi_summary["median_Delta_Extreme_minus_Normal"] = np.nan

    df_regional_cbi_summary.to_csv(
        OUTPUT_DIR / OUTPUT_FILES["regional_cbi_summary"],
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------ 极端事件是否进入03B/03C配对审计 -----------------
    # 03B/03C的比较单元是 Site_ID × YearMonth，不是事件ID。
    # 因此这里逐个极端事件检查：该事件涉及的站点-月份是否进入了03B/03C配对图。
    # 如果没有进入，明确列出对应站点-月份的Normal/Extreme小时数、Macro_SD和失败原因。
    df_events_for_site_month_audit = pd.read_csv(DROUGHT_EVENT_FILE, low_memory=False)
    required_event_cols_for_site_month_audit = {
        SITE_COL, "Event_ID", "Start_Date", "End_Date",
        "Duration_Days", "Drought_Level"
    }
    missing_event_cols_for_site_month_audit = (
        required_event_cols_for_site_month_audit -
        set(df_events_for_site_month_audit.columns)
    )
    if missing_event_cols_for_site_month_audit:
        raise KeyError(
            "事件长表缺少站点-月份配对审计所需字段："
            f"{missing_event_cols_for_site_month_audit}"
        )

    df_events_for_site_month_audit[SITE_COL] = normalize_site_id(
        df_events_for_site_month_audit[SITE_COL]
    )
    df_events_for_site_month_audit["Start_Date"] = pd.to_datetime(
        df_events_for_site_month_audit["Start_Date"], errors="coerce"
    ).dt.normalize()
    df_events_for_site_month_audit["End_Date"] = pd.to_datetime(
        df_events_for_site_month_audit["End_Date"], errors="coerce"
    ).dt.normalize()
    df_events_for_site_month_audit["Drought_Level_clean"] = (
        df_events_for_site_month_audit["Drought_Level"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    df_extreme_events_for_site_month_audit = df_events_for_site_month_audit.loc[
        df_events_for_site_month_audit["Drought_Level_clean"] == "extreme"
    ].copy()

    status_cbi_lookup = {
        (row[SITE_COL], row["YearMonth"], row["SPI_Status"]): row
        for _, row in df_site_month_status_cbi.iterrows()
    }
    paired_site_month_keys = set(
        zip(
            df_site_month_cbi_pair[SITE_COL],
            df_site_month_cbi_pair["YearMonth"]
        )
    ) if not df_site_month_cbi_pair.empty else set()

    event_site_month_audit_rows = []
    for _, event in progress_iter(
        df_extreme_events_for_site_month_audit.iterrows(),
        total=len(df_extreme_events_for_site_month_audit),
        desc="审计极端事件是否进入站点月份配对",
        kind="站点月份CBI"
    ):
        site = event[SITE_COL]
        event_id = event["Event_ID"]
        start_date = event["Start_Date"]
        end_date = event["End_Date"]
        event_months = pd.period_range(start_date, end_date, freq="M")

        for event_month in event_months:
            yearmonth = str(event_month)
            normal_row = status_cbi_lookup.get((site, yearmonth, "Normal"))
            extreme_row = status_cbi_lookup.get((site, yearmonth, "Extreme"))
            in_pair = (site, yearmonth) in paired_site_month_keys

            reasons = []
            if normal_row is None:
                reasons.append("该站点-月份无Normal记录或未进入全年有Extreme站点筛选")
            elif not bool(normal_row["Pass_CBI_Filter"]):
                reasons.append(
                    "Normal CBI不合格: "
                    f"n_hours={normal_row['n_hours']}, "
                    f"Macro_SD={normal_row['Macro_SD']}, "
                    f"flag={normal_row['CBI_flag']}"
                )

            if extreme_row is None:
                reasons.append("该站点-月份无Extreme记录")
            elif not bool(extreme_row["Pass_CBI_Filter"]):
                reasons.append(
                    "Extreme CBI不合格: "
                    f"n_hours={extreme_row['n_hours']}, "
                    f"Macro_SD={extreme_row['Macro_SD']}, "
                    f"flag={extreme_row['CBI_flag']}"
                )

            if in_pair:
                reasons_text = "已进入03B/03C配对图"
            else:
                reasons_text = "；".join(reasons) if reasons else "未形成配对，原因需检查站点月份CBI估计表"

            event_site_month_audit_rows.append({
                SITE_COL: site,
                "Event_ID": event_id,
                "Event_Start_Date": start_date,
                "Event_End_Date": end_date,
                "Duration_Days": event["Duration_Days"],
                "Event_YearMonth": yearmonth,
                "Site_Month_Key": f"{site}_{yearmonth}",
                "Entered_03B_03C_SiteMonth_Pair": bool(in_pair),
                "未进入03B_03C原因": reasons_text,
                "Normal_n_hours": normal_row["n_hours"] if normal_row is not None else np.nan,
                "Normal_Macro_SD": normal_row["Macro_SD"] if normal_row is not None else np.nan,
                "Normal_Pass_CBI_Filter": bool(normal_row["Pass_CBI_Filter"]) if normal_row is not None else False,
                "Normal_CBI_flag": normal_row["CBI_flag"] if normal_row is not None else "missing",
                "Extreme_n_hours": extreme_row["n_hours"] if extreme_row is not None else np.nan,
                "Extreme_Macro_SD": extreme_row["Macro_SD"] if extreme_row is not None else np.nan,
                "Extreme_Pass_CBI_Filter": bool(extreme_row["Pass_CBI_Filter"]) if extreme_row is not None else False,
                "Extreme_CBI_flag": extreme_row["CBI_flag"] if extreme_row is not None else "missing",
                "最低小时门槛": MIN_SITE_MONTH_STATUS_CBI_HOURS,
                "最低ERA5_SD门槛": MIN_SITE_MONTH_STATUS_MACRO_SD
            })

    df_event_to_site_month_pair_audit = pd.DataFrame(event_site_month_audit_rows)
    df_event_to_site_month_pair_audit.to_csv(
        OUTPUT_DIR / OUTPUT_FILES["event_to_site_month_pair_audit"],
        index=False,
        encoding="utf-8-sig"
    )

    # ---------------------- 03B 配对样本箱线图 + 散点 -------------------
    # 03B只使用同一Site_ID × YearMonth同时具有合格Normal和Extreme CBI的配对单元。
    # 因此Normal和Extreme两侧样本量严格相等，适合解释同背景下的CBI差异。
    if not df_site_month_cbi_pair.empty:
        df_site_month_cbi_pair_long = df_site_month_cbi_pair.melt(
            id_vars=[SITE_COL, "YearMonth", "Month"],
            value_vars=["CBI_Normal", "CBI_Extreme"],
            var_name="CBI_Type",
            value_name="CBI"
        )
        df_site_month_cbi_pair_long["SPI_Status"] = (
            df_site_month_cbi_pair_long["CBI_Type"]
            .map({"CBI_Normal": "Normal", "CBI_Extreme": "Extreme"})
        )

        fig, ax = plt.subplots(figsize=FIG_SITE_MONTH_CBI_BOX_SIZE)
        sns.boxplot(
            data=df_site_month_cbi_pair_long,
            x="SPI_Status",
            y="CBI",
            order=["Normal", "Extreme"],
            palette={"Normal": COLOR_NORMAL, "Extreme": COLOR_EXTREME},
            showfliers=False,
            width=0.45,
            ax=ax
        )
        sns.stripplot(
            data=df_site_month_cbi_pair_long,
            x="SPI_Status",
            y="CBI",
            order=["Normal", "Extreme"],
            palette={"Normal": COLOR_NORMAL, "Extreme": COLOR_EXTREME},
            alpha=ALPHA_CBI_POINTS,
            size=SCATTER_SIZE_CBI / 10,
            jitter=JITTER_WIDTH_CBI_BOX,
            ax=ax
        )
        ax.axhline(1, color=COLOR_REFERENCE_LINE, linestyle="--", linewidth=LINEWIDTH_REFERENCE)

        # 在03B图上直接标注每组样本量和中位数。
        # n表示配对站点-月份单元数；Normal和Extreme两侧应完全相等。
        for status_index, status in enumerate(["Normal", "Extreme"]):
            d_status = df_site_month_cbi_pair_long.loc[
                df_site_month_cbi_pair_long["SPI_Status"] == status
            ]
            if not d_status.empty:
                ax.text(
                    status_index,
                    d_status["CBI"].max(),
                    f"n={len(d_status)}\n中位数={d_status['CBI'].median():.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=ANNOTATION_FONTSIZE,
                    color=COLOR_ZERO_LINE
                )

        ax.set_xlabel("SPI状态")
        ax.set_ylabel("配对站点-月份 CBI")
        ax.set_title(
            "B. 配对样本站点-月份CBI分布\n"
            "仅同一站点-月份同时有合格Normal和Extreme CBI；"
            f"n小时≥{MIN_SITE_MONTH_STATUS_CBI_HOURS}，ERA5 SD≥{MIN_SITE_MONTH_STATUS_MACRO_SD}°C"
        )
        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / OUTPUT_FILES["site_month_cbi_box_plot"],
            dpi=FIG_DPI,
            bbox_inches="tight"
        )
        plt.close(fig)
    else:
        df_site_month_cbi_pair_long = pd.DataFrame()
        print("警告：没有同时满足Normal和Extreme筛选条件的站点-月份配对，跳过03B配对箱线图。")

    # ---------------------- 03C 配对变化图 -----------------------------
    if not df_site_month_cbi_pair.empty:
        fig, ax = plt.subplots(figsize=FIG_SITE_MONTH_CBI_PAIR_SIZE)
        x_positions = [0, 1]

        for _, row in progress_iter(
            df_site_month_cbi_pair.iterrows(),
            total=len(df_site_month_cbi_pair),
            desc="绘制站点月份CBI配对线",
            kind="绘图"
        ):
            delta = row["Delta_CBI_Extreme_minus_Normal"]
            if delta > NEAR_ZERO_DELTA_CBI:
                line_color = COLOR_EXTREME
            elif delta < -NEAR_ZERO_DELTA_CBI:
                line_color = COLOR_NORMAL
            else:
                line_color = COLOR_REFERENCE_LINE
            ax.plot(
                x_positions,
                [row["CBI_Normal"], row["CBI_Extreme"]],
                color=line_color,
                alpha=ALPHA_CBI_PAIR_LINES,
                linewidth=LINEWIDTH_CBI_PAIR
            )

        median_normal = df_site_month_cbi_pair["CBI_Normal"].median()
        median_extreme = df_site_month_cbi_pair["CBI_Extreme"].median()
        ax.plot(
            x_positions,
            [median_normal, median_extreme],
            color=COLOR_ZERO_LINE,
            linewidth=LINEWIDTH_CBI_MEDIAN_PAIR,
            marker=MARKER_MONTHLY_MEDIAN,
            label="配对单元中位数"
        )

        ax.axhline(1, color=COLOR_REFERENCE_LINE, linestyle="--", linewidth=LINEWIDTH_REFERENCE)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(["Normal", "Extreme"])
        ax.set_ylabel("CBI")
        ax.set_title(
            "C. 同站点-同月份CBI配对变化\n"
            "红线：Extreme更高；蓝线：Extreme更低；黑粗线：中位数变化"
        )
        ax.legend(loc=LEGEND_LOCATION_BEST)
        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / OUTPUT_FILES["site_month_cbi_pair_plot"],
            dpi=FIG_DPI,
            bbox_inches="tight"
        )
        plt.close(fig)
    else:
        print("警告：没有同时满足Normal和Extreme筛选条件的站点-月份配对，跳过03C配对图。")

    # ---------------------- 03D 研究区总体箱线图 ------------------------
    fig, ax = plt.subplots(figsize=FIG_REGIONAL_CBI_BOX_SIZE)
    sns.boxplot(
        data=df_site_month_cbi_plot,
        x="SPI_Status",
        y="CBI",
        order=["Normal", "Extreme"],
        palette={"Normal": COLOR_NORMAL, "Extreme": COLOR_EXTREME},
        showfliers=False,
        width=0.50,
        ax=ax
    )
    sns.stripplot(
        data=df_site_month_cbi_plot,
        x="SPI_Status",
        y="CBI",
        order=["Normal", "Extreme"],
        color=COLOR_ZERO_LINE,
        alpha=0.35,
        size=SCATTER_SIZE_CBI / 12,
        jitter=JITTER_WIDTH_CBI_BOX,
        ax=ax
    )
    ax.axhline(1, color=COLOR_REFERENCE_LINE, linestyle="--", linewidth=LINEWIDTH_REFERENCE)

    for status_index, status in enumerate(["Normal", "Extreme"]):
        d_status = df_site_month_cbi_plot.loc[
            df_site_month_cbi_plot["SPI_Status"] == status
        ]
        if not d_status.empty:
            ax.text(
                status_index,
                d_status["CBI"].max(),
                f"n={len(d_status)}\n中位数={d_status['CBI'].median():.3f}",
                ha="center",
                va="bottom",
                fontsize=ANNOTATION_FONTSIZE
            )

    ax.set_xlabel("SPI状态")
    ax.set_ylabel("CBI")
    ax.set_title(
        "D. 研究区总体CBI分布趋势\n"
        "仅包含全年出现过Extreme的站点；每点=合格站点-月份-状态CBI；描述性结果"
    )
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / OUTPUT_FILES["regional_cbi_box_plot"],
        dpi=FIG_DPI,
        bbox_inches="tight"
    )
    plt.close(fig)

    # ---------------------- 三面板综合图 -------------------------------
    fig, axes = plt.subplots(1, 3, figsize=FIG_MAIN_CBI_COMPOSITE_SIZE)

    draw_main_lmm_prediction_panel(
        axes[0],
        df_lmm_prediction,
        weighted_normal_cbi,
        weighted_extreme_cbi,
        interaction_beta,
        interaction_p,
        show_bottom_note=False
    )

    if not df_site_month_cbi_pair_long.empty:
        sns.boxplot(
            data=df_site_month_cbi_pair_long,
            x="SPI_Status",
            y="CBI",
            order=["Normal", "Extreme"],
            palette={"Normal": COLOR_NORMAL, "Extreme": COLOR_EXTREME},
            showfliers=False,
            width=0.45,
            ax=axes[1]
        )
        sns.stripplot(
            data=df_site_month_cbi_pair_long,
            x="SPI_Status",
            y="CBI",
            order=["Normal", "Extreme"],
            palette={"Normal": COLOR_NORMAL, "Extreme": COLOR_EXTREME},
            alpha=ALPHA_CBI_POINTS,
            size=SCATTER_SIZE_CBI / 12,
            jitter=JITTER_WIDTH_CBI_BOX,
            ax=axes[1]
        )
    axes[1].axhline(1, color=COLOR_REFERENCE_LINE, linestyle="--", linewidth=LINEWIDTH_REFERENCE)
    axes[1].set_xlabel("SPI状态")
    axes[1].set_ylabel("配对站点-月份 CBI")
    axes[1].set_title("B. 配对样本CBI分布\n同站点-同月份")

    if not df_site_month_cbi_pair.empty:
        for _, row in df_site_month_cbi_pair.iterrows():
            delta = row["Delta_CBI_Extreme_minus_Normal"]
            if delta > NEAR_ZERO_DELTA_CBI:
                line_color = COLOR_EXTREME
            elif delta < -NEAR_ZERO_DELTA_CBI:
                line_color = COLOR_NORMAL
            else:
                line_color = COLOR_REFERENCE_LINE
            axes[2].plot(
                [0, 1],
                [row["CBI_Normal"], row["CBI_Extreme"]],
                color=line_color,
                alpha=ALPHA_CBI_PAIR_LINES,
                linewidth=LINEWIDTH_CBI_PAIR
            )

        axes[2].plot(
            [0, 1],
            [
                df_site_month_cbi_pair["CBI_Normal"].median(),
                df_site_month_cbi_pair["CBI_Extreme"].median()
            ],
            color=COLOR_ZERO_LINE,
            linewidth=LINEWIDTH_CBI_MEDIAN_PAIR,
            marker=MARKER_MONTHLY_MEDIAN,
            label="中位数"
        )
        axes[2].legend(loc=LEGEND_LOCATION_BEST)

    axes[2].axhline(1, color=COLOR_REFERENCE_LINE, linestyle="--", linewidth=LINEWIDTH_REFERENCE)
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(["Normal", "Extreme"])
    axes[2].set_ylabel("CBI")
    axes[2].set_title("C. 同站点-同月份CBI配对变化")

    fig.suptitle(
        "主LMM与站点-月份CBI补充分析：机制、分布与配对方向",
        fontsize=14
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(
        OUTPUT_DIR / OUTPUT_FILES["main_cbi_composite_plot"],
        dpi=FIG_DPI,
        bbox_inches="tight"
    )
    plt.close(fig)


# =============================================================================
# Step 4. 事件级辅助分析：逐事件切片 + 事件后30天内Normal参考期
# =============================================================================

print("\n" + "=" * 80)
print("Step 4: 事件级辅助验证")
print("=" * 80)

# ---------------------------- 4.1 读取事件表 ----------------------------
df_events = pd.read_csv(DROUGHT_EVENT_FILE, low_memory=False)

required_event_cols = {
    SITE_COL, "Event_ID", "Start_Date", "End_Date",
    "Duration_Days", "Drought_Level"
}
missing_event_cols = required_event_cols - set(df_events.columns)

if missing_event_cols:
    raise KeyError(
        f"事件长表缺少必须字段：{missing_event_cols}\n"
        f"实际字段：{df_events.columns.tolist()}"
    )

df_events[SITE_COL] = normalize_site_id(df_events[SITE_COL])
df_events["Start_Date"] = pd.to_datetime(
    df_events["Start_Date"], errors="coerce"
).dt.normalize()
df_events["End_Date"] = pd.to_datetime(
    df_events["End_Date"], errors="coerce"
).dt.normalize()

# 仅选取事件等级明确为Extreme的记录。
# 为防止大小写或前后空格问题，先统一字符串。
df_events["Drought_Level_clean"] = (
    df_events["Drought_Level"].astype(str).str.strip().str.lower()
)

df_extreme_events = df_events.loc[
    df_events["Drought_Level_clean"] == "extreme"
].copy()

if df_extreme_events.empty:
    raise RuntimeError(
        "事件表中没有找到 Drought_Level == 'Extreme' 的事件。"
        "请检查事件等级字段内容。"
    )

# 检查事件日期是否有效。
if df_extreme_events[["Start_Date", "End_Date"]].isna().any().any():
    raise ValueError("极端事件存在无法解析的开始或结束日期。")

if (df_extreme_events["End_Date"] < df_extreme_events["Start_Date"]).any():
    raise ValueError("发现 End_Date 早于 Start_Date 的极端事件。")

# -------------------------- 4.2 逐事件直接切片 ---------------------------
#
# 关键原则：
# 不使用小时表中的 Event_ID 标签；
# 而是每一行事件表直接按照 Site_ID、Start_Date、End_Date
# 从 df_hourly 中切出事件期小时温度数据。
#
# 所以即使两个同等级事件相邻，也不会被标签覆盖或错误合并。
# 参考期原则：
# 事件结束后跳过1天缓冲日，然后只在未来30个UTC自然日内寻找Normal SPI日期；
# 每个候选日期至少需要 MIN_VALID_HOURS_PER_REFERENCE_DAY 个有效小时；
# 若合格日期不足目标天数，不向30天窗口外延伸。
# -------------------------------------------------------------------------

event_result_rows = []
event_reference_date_rows = []

for _, event in progress_iter(
    df_extreme_events.iterrows(),
    total=len(df_extreme_events),
    desc="逐事件切片与参考期审计",
    kind="事件分析"
):

    site = event[SITE_COL]
    event_id = event["Event_ID"]
    start_date = event["Start_Date"]
    end_date = event["End_Date"]
    duration_days = int(event["Duration_Days"])

    # ----------------------- 4.2.1 切取事件期 ---------------------------
    # 注意：事件期使用事件表定义的完整日期范围；
    # 不需要事件期所有小时SPI都 < -2，因为事件识别是连续过程级定义。
    event_mask = (
        (df_hourly[SITE_COL] == site) &
        (df_hourly[UTC_DATE_COL] >= start_date) &
        (df_hourly[UTC_DATE_COL] <= end_date)
    )

    df_event_hours = df_hourly.loc[event_mask].copy()
    event_cbi_result = calc_ols_cbi(
        df_event_hours,
        min_hours=MIN_EVENT_HOURS
    )

    # ---------------------- 4.2.2 构建后置搜索窗口 -----------------------
    #
    # 事件结束后第1天是buffer，不使用；
    # 候选起点 = End_Date + 2天；
    # 候选终点 = 起点 + 29天，因此一共30个自然日。
    #
    # 例如 End_Date = 2025-05-10：
    # buffer日       = 2025-05-11
    # 搜索起点       = 2025-05-12
    # 搜索终点       = 2025-06-10
    # ---------------------------------------------------------------------

    ref_start = end_date + pd.Timedelta(days=POST_EVENT_BUFFER_DAYS + 1)
    ref_search_end = ref_start + pd.Timedelta(days=POST_EVENT_SEARCH_DAYS - 1)
    ref_target_days = min(duration_days, MAX_REF_TARGET_DAYS)

    # 只取30天候选范围内、且SPI状态严格为Normal的小时。
    # Other状态（包括重度/中度/轻度干旱、轻度/中度/严重/极端湿润及边界/缺失状态）不能作为参考期。
    candidate_ref_hours = df_hourly.loc[
        (df_hourly[SITE_COL] == site) &
        (df_hourly[UTC_DATE_COL] >= ref_start) &
        (df_hourly[UTC_DATE_COL] <= ref_search_end) &
        (df_hourly["SPI_Status"] == "Normal")
    ].copy()

    # 在30天候选窗口内逐日审计：某日必须是Normal SPI日，且至少有18个有效小时，
    # 才能作为合格参考日。这样避免1-2小时残缺记录被当作完整参考日。
    site_window_hours = df_hourly.loc[
        (df_hourly[SITE_COL] == site) &
        (df_hourly[UTC_DATE_COL] >= ref_start) &
        (df_hourly[UTC_DATE_COL] <= ref_search_end)
    ].copy()

    candidate_dates = pd.date_range(
        ref_start,
        ref_search_end,
        freq="D"
    )

    candidate_day_rows = []
    for candidate_date in candidate_dates:
        day_hours = site_window_hours.loc[
            site_window_hours[UTC_DATE_COL] == candidate_date
        ]
        n_valid_hours_day = len(day_hours)
        spi_values = day_hours["SPI30d"].dropna().unique()
        spi_value = spi_values[0] if len(spi_values) > 0 else np.nan
        is_normal_spi = (
            (spi_value > NORMAL_SPI_LOW) and
            (spi_value < NORMAL_SPI_HIGH)
            if pd.notna(spi_value) else False
        )
        pass_daily_hour_threshold = (
            n_valid_hours_day >= MIN_VALID_HOURS_PER_REFERENCE_DAY
        )

        if not is_normal_spi:
            exclusion_reason = "not_normal_spi"
        elif not pass_daily_hour_threshold:
            exclusion_reason = "valid_hours_lt_daily_threshold"
        else:
            exclusion_reason = ""

        candidate_day_rows.append({
            "Candidate_Date": candidate_date,
            "SPI30d": spi_value,
            "Is_Normal_SPI": bool(is_normal_spi),
            "n_valid_hours": int(n_valid_hours_day),
            "Pass_Daily_Hour_Threshold": bool(pass_daily_hour_threshold),
            "Exclusion_Reason": exclusion_reason
        })

    candidate_day_audit = pd.DataFrame(candidate_day_rows)
    qualified_ref_dates = candidate_day_audit.loc[
        candidate_day_audit["Is_Normal_SPI"] &
        candidate_day_audit["Pass_Daily_Hour_Threshold"],
        "Candidate_Date"
    ].tolist()

    # “找得到几天就用几天”的实现：
    # - 目标是 min(事件持续天数, 30)
    # - 但合格Normal日期少于目标时，不向窗口外搜索
    # - 直接使用当前30天内所有合格Normal日期
    selected_ref_dates = qualified_ref_dates[:ref_target_days]

    df_ref_hours = candidate_ref_hours.loc[
        candidate_ref_hours[UTC_DATE_COL].isin(selected_ref_dates)
    ].copy()

    ref_cbi_result = calc_ols_cbi(
        df_ref_hours,
        min_hours=MIN_REF_HOURS
    )

    # 保存30天候选窗口内每个候选日期的筛选结果，便于审计为什么未选足参考日。
    selected_ref_date_set = set(selected_ref_dates)
    for _, day_row in candidate_day_audit.iterrows():
        ref_date = day_row["Candidate_Date"]
        selected_as_reference = ref_date in selected_ref_date_set
        exclusion_reason = day_row["Exclusion_Reason"]
        if (
            day_row["Is_Normal_SPI"] and
            day_row["Pass_Daily_Hour_Threshold"] and
            not selected_as_reference
        ):
            exclusion_reason = "beyond_reference_target_days"
        event_reference_date_rows.append({
            SITE_COL: site,
            "Event_ID": event_id,
            "Event_Start_Date": start_date,
            "Event_End_Date": end_date,
            "Reference_Search_Start": ref_start,
            "Reference_Search_End": ref_search_end,
            "Reference_Target_Days": ref_target_days,
            "Candidate_Date": ref_date,
            "SPI30d": day_row["SPI30d"],
            "Is_Normal_SPI": day_row["Is_Normal_SPI"],
            "n_valid_hours": day_row["n_valid_hours"],
            "Pass_Daily_Hour_Threshold": day_row["Pass_Daily_Hour_Threshold"],
            "Selected_as_Reference": selected_as_reference,
            "Exclusion_Reason": exclusion_reason
        })

    # 只有事件期与参考期CBI都可靠时，才计算Delta_CBI。
    if event_cbi_result["flag"] == "ok" and ref_cbi_result["flag"] == "ok":
        delta_cbi = event_cbi_result["CBI"] - ref_cbi_result["CBI"]
        pair_flag = "ok"
    else:
        delta_cbi = np.nan
        pair_flag = "event_or_reference_CBI_not_reliable"

    event_row = {
        SITE_COL: site,
        "Event_ID": event_id,
        "Start_Date": start_date,
        "End_Date": end_date,
        "Duration_Days": duration_days,
        "Severity": event.get("Severity", np.nan),
        "Min_Daily_SPI": event.get("Min_Daily_SPI", np.nan),
        "Edge_Truncated": event.get("Edge_Truncated", np.nan),

        # 事件期信息
        "Event_CBI": event_cbi_result["CBI"],
        "Event_Intercept": event_cbi_result["Intercept"],
        "Event_R2": event_cbi_result["R2"],
        "Event_n_hours": event_cbi_result["n_hours"],
        "Event_CBI_flag": event_cbi_result["flag"],

        # 参考期定义与实际覆盖
        "Reference_Search_Start": ref_start,
        "Reference_Search_End": ref_search_end,
        "Reference_Target_Days": ref_target_days,
        "Reference_Available_Normal_Days": int(candidate_day_audit["Is_Normal_SPI"].sum()),
        "Reference_Qualified_Normal_Days": len(qualified_ref_dates),
        "Reference_Selected_Normal_Days": len(selected_ref_dates),
        "Reference_Min_Valid_Hours_Per_Day": MIN_VALID_HOURS_PER_REFERENCE_DAY,

        # 参考期CBI信息
        "Reference_CBI": ref_cbi_result["CBI"],
        "Reference_Intercept": ref_cbi_result["Intercept"],
        "Reference_R2": ref_cbi_result["R2"],
        "Reference_n_hours": ref_cbi_result["n_hours"],
        "Reference_CBI_flag": ref_cbi_result["flag"],

        # 配对结果
        "Delta_CBI_Event_minus_Reference": delta_cbi,
        "Pair_flag": pair_flag
    }

    event_result_rows.append(event_row)

df_event_cbi = pd.DataFrame(event_result_rows)
df_event_ref_dates = pd.DataFrame(event_reference_date_rows)

df_event_cbi.to_csv(
    OUTPUT_DIR / OUTPUT_FILES["event_cbi_reference"],
    index=False,
    encoding="utf-8-sig"
)

df_event_unpaired_audit = df_event_cbi.loc[
    df_event_cbi["Pair_flag"] != "ok"
].copy()

if not df_event_unpaired_audit.empty:
    def explain_unpaired_event(row):
        """
        将事件期/参考期CBI失败标记转换成更容易阅读的原因说明。

        说明：
        - Event_CBI_flag来自事件期OLS；
        - Reference_CBI_flag来自事件后Normal参考期OLS；
        - 只有两者均为ok时才进入事件级Delta_CBI配对检验。
        """
        reasons = []

        if row["Event_CBI_flag"] != "ok":
            reasons.append(f"事件期CBI不可用: {row['Event_CBI_flag']}")

        if row["Reference_CBI_flag"] != "ok":
            reasons.append(f"参考期CBI不可用: {row['Reference_CBI_flag']}")

        if not reasons:
            reasons.append(f"其他原因: {row['Pair_flag']}")

        return "；".join(reasons)

    df_event_unpaired_audit["未进入配对检验原因"] = (
        df_event_unpaired_audit.apply(explain_unpaired_event, axis=1)
    )
    df_event_unpaired_audit["事件期最低小时门槛"] = MIN_EVENT_HOURS
    df_event_unpaired_audit["参考期最低小时门槛"] = MIN_REF_HOURS
    df_event_unpaired_audit["参考候选日最低小时门槛"] = MIN_VALID_HOURS_PER_REFERENCE_DAY
    df_event_unpaired_audit["事件期是否达到小时门槛"] = (
        df_event_unpaired_audit["Event_n_hours"] >= MIN_EVENT_HOURS
    )
    df_event_unpaired_audit["参考期是否达到小时门槛"] = (
        df_event_unpaired_audit["Reference_n_hours"] >= MIN_REF_HOURS
    )
else:
    df_event_unpaired_audit = pd.DataFrame(columns=[
        SITE_COL,
        "Event_ID",
        "Start_Date",
        "End_Date",
        "Duration_Days",
        "Event_n_hours",
        "Event_CBI_flag",
        "Reference_Selected_Normal_Days",
        "Reference_n_hours",
        "Reference_CBI_flag",
        "Pair_flag",
        "未进入配对检验原因",
        "事件期最低小时门槛",
        "参考期最低小时门槛",
        "参考候选日最低小时门槛",
        "事件期是否达到小时门槛",
        "参考期是否达到小时门槛"
    ])

df_event_unpaired_audit.to_csv(
    OUTPUT_DIR / OUTPUT_FILES["event_unpaired_audit"],
    index=False,
    encoding="utf-8-sig"
)

df_event_ref_dates.to_csv(
    OUTPUT_DIR / OUTPUT_FILES["event_reference_dates"],
    index=False,
    encoding="utf-8-sig"
)

print(f"极端事件总数: {len(df_event_cbi)}")
print(
    "有可靠事件期与参考期CBI、可进入辅助配对检验的事件数: "
    f"{(df_event_cbi['Pair_flag'] == 'ok').sum()}"
)


# =============================================================================
# Step 5. 事件级辅助显著性与配对图
# =============================================================================

print("\n" + "=" * 80)
print("Step 5: 事件级辅助显著性检验（站点聚类Bootstrap）")
print("=" * 80)

df_event_pairs = df_event_cbi.loc[
    df_event_cbi["Pair_flag"] == "ok"
].copy()

bootstrap_result = cluster_bootstrap_mean_delta(
    df_event_pairs,
    site_col=SITE_COL,
    delta_col="Delta_CBI_Event_minus_Reference",
    n_boot=N_CLUSTER_BOOTSTRAP,
    random_seed=RANDOM_SEED
)

df_bootstrap_result = pd.DataFrame([bootstrap_result])
df_bootstrap_result.to_csv(
    OUTPUT_DIR / OUTPUT_FILES["event_bootstrap"],
    index=False,
    encoding="utf-8-sig"
)

print("\n事件级辅助分析结果：")
print(df_bootstrap_result.to_string(index=False))

# ----------------------- 5.1 事件期 vs 参考期配对图 ---------------------
if not df_event_pairs.empty:

    plot_long = df_event_pairs.melt(
        id_vars=[SITE_COL, "Event_ID", "Duration_Days",
                 "Delta_CBI_Event_minus_Reference"],
        value_vars=["Event_CBI", "Reference_CBI"],
        var_name="Period",
        value_name="CBI"
    )

    period_name_map = {
        "Event_CBI": "极端事件期",
        "Reference_CBI": "事件后Normal参考期"
    }
    plot_long["Period"] = plot_long["Period"].map(period_name_map)

    fig, ax = plt.subplots(figsize=FIG_EVENT_PAIR_SIZE)

    # 每个事件一条连线
    for _, row in progress_iter(
        df_event_pairs.iterrows(),
        total=len(df_event_pairs),
        desc="绘制事件CBI配对线",
        kind="绘图"
    ):
        ax.plot(
            [0, 1],
            [row["Event_CBI"], row["Reference_CBI"]],
            color=(COLOR_EXTREME if
                   row["Delta_CBI_Event_minus_Reference"] > 0
                   else COLOR_NORMAL),
            alpha=ALPHA_EVENT_PAIR,
            linewidth=LINEWIDTH_EVENT_PAIR
        )

        # 仅当事件期与参考期CBI都大于1时，在连线右侧标注站点编号。
        # 这类事件表示两个阶段都表现为“微气候变化幅度不低于宏气候变化幅度”。
        if (row["Event_CBI"] > 1) and (row["Reference_CBI"] > 1):
            ax.text(
                1.03,
                row["Reference_CBI"],
                str(row[SITE_COL]),
                ha="left",
                va="center",
                fontsize=ANNOTATION_FONTSIZE,
                color=COLOR_HIGHLIGHT_TEXT
            )

    sns.stripplot(
        data=plot_long,
        x="Period",
        y="CBI",
        order=["极端事件期", "事件后Normal参考期"],
        color=COLOR_ZERO_LINE,
        alpha=ALPHA_SCATTER,
        size=SCATTER_SIZE_EVENT,
        ax=ax
    )

    ax.axhline(1, color=COLOR_REFERENCE_LINE, linestyle="--", linewidth=LINEWIDTH_REFERENCE)
    ax.set_xlabel("")
    ax.set_ylabel("CBI（微气候对宏气候温度的回归斜率）")
    ax.set_title(
        "事件级辅助验证：极端事件期 vs 事件后Normal参考期\n"
        "红线：事件期CBI更高（缓冲减弱）；蓝线：事件期CBI更低"
    )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / OUTPUT_FILES["event_pair_plot"],
        dpi=FIG_DPI,
        bbox_inches="tight"
    )
    plt.close(fig)

    # ---------------------- 5.2 Delta_CBI事件图 -------------------------
    plot_delta = df_event_pairs.sort_values(
        "Delta_CBI_Event_minus_Reference"
    ).reset_index(drop=True)

    fig_height = max(FIG_EVENT_DELTA_MIN_HEIGHT, FIG_EVENT_DELTA_HEIGHT_PER_EVENT * len(plot_delta))
    fig, ax = plt.subplots(figsize=(FIG_EVENT_DELTA_WIDTH, fig_height))

    colors = np.where(
        plot_delta["Delta_CBI_Event_minus_Reference"] > 0,
        COLOR_EXTREME,
        COLOR_NORMAL
    )

    y_pos = np.arange(len(plot_delta))

    ax.scatter(
        plot_delta["Delta_CBI_Event_minus_Reference"],
        y_pos,
        s=SCATTER_SIZE_DELTA,
        c=colors,
        alpha=ALPHA_EVENT_DELTA_SCATTER
    )

    ax.axvline(0, color=COLOR_ZERO_LINE, linestyle="--", linewidth=1)

    labels = [
        f"{site} | 事件 {event_id}"
        for site, event_id in zip(
            plot_delta[SITE_COL],
            plot_delta["Event_ID"]
        )
    ]

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=TICKLABEL_FONTSIZE_EVENT_DELTA)

    # 若同一站点既出现正Delta_CBI也出现负Delta_CBI，说明该站点不同极端事件的方向不一致。
    # 为便于快速识别这类“同站点内结果异质性”，将该站点对应的所有纵轴标签标红。
    site_delta_sign = (
        df_event_pairs.assign(
            Delta_Sign=np.sign(df_event_pairs["Delta_CBI_Event_minus_Reference"])
        )
        .groupby(SITE_COL)["Delta_Sign"]
        .agg(lambda x: set(v for v in x if v != 0))
    )
    mixed_direction_sites = {
        site for site, sign_set in site_delta_sign.items()
        if (1 in sign_set) and (-1 in sign_set)
    }

    for tick_label, site in zip(ax.get_yticklabels(), plot_delta[SITE_COL]):
        if site in mixed_direction_sites:
            tick_label.set_color(COLOR_HIGHLIGHT_TEXT)

    ax.set_xlabel("Delta_CBI = 事件期CBI - 事件后参考期CBI")
    ax.set_ylabel("站点 | 极端事件编号")
    ax.set_title(
        "各极端事件的CBI变化\n"
        "正值：极端事件期缓冲减弱；负值：极端事件期表观缓冲增强；红色站点标签：同站点内既有正值也有负值"
    )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / OUTPUT_FILES["event_delta_plot"],
        dpi=FIG_DPI,
        bbox_inches="tight"
    )
    plt.close(fig)

else:
    print(
        "警告：没有可用的事件—参考期CBI配对，"
        f"因此跳过事件级图形。请查看 {OUTPUT_FILES['event_cbi_reference']}。"
    )


# =============================================================================
# Step 6. 月度CBI描述性分析（非主显著性结论）
# =============================================================================

print("\n" + "=" * 80)
print("Step 6: 月度CBI描述性分析")
print("=" * 80)

monthly_rows = []

monthly_groups = list(df_hourly.groupby([SITE_COL, "YearMonth"]))
for (site, yearmonth), group in progress_iter(
    monthly_groups,
    total=len(monthly_groups),
    desc="计算月度CBI",
    kind="月度分析"
):

    cbi_result = calc_ols_cbi(
        group,
        min_hours=MIN_MONTHLY_HOURS
    )

    n_total_hours = len(group)
    n_extreme_hours = (group["SPI_Status"] == "Extreme").sum()

    extreme_ratio = (
        n_extreme_hours / n_total_hours
        if n_total_hours > 0 else np.nan
    )

    monthly_rows.append({
        SITE_COL: site,
        "YearMonth": yearmonth,
        "Month": int(group["Month"].iloc[0]),
        "Monthly_CBI": cbi_result["CBI"],
        "Monthly_Intercept": cbi_result["Intercept"],
        "Monthly_R2": cbi_result["R2"],
        "Monthly_n_hours": cbi_result["n_hours"],
        "Monthly_CBI_flag": cbi_result["flag"],
        "Extreme_Hours": n_extreme_hours,
        "Total_Hours": n_total_hours,
        "Extreme_Ratio": extreme_ratio,
        "Is_Extreme_Month_0p30": int(
            extreme_ratio >= EXTREME_RATIO_MONTHLY
        )
    })

df_monthly_cbi = pd.DataFrame(monthly_rows)

df_monthly_cbi.to_csv(
    OUTPUT_DIR / OUTPUT_FILES["monthly_cbi"],
    index=False,
    encoding="utf-8-sig"
)

# ------------------------- 6.1 月度CBI轨迹图 ----------------------------
df_monthly_plot = df_monthly_cbi.loc[
    df_monthly_cbi["Monthly_CBI_flag"] == "ok"
].copy()

if not df_monthly_plot.empty:

    month_summary = (
        df_monthly_plot.groupby("Month")
        .agg(
            median_CBI=("Monthly_CBI", "median"),
            q25_CBI=("Monthly_CBI", lambda x: x.quantile(0.25)),
            q75_CBI=("Monthly_CBI", lambda x: x.quantile(0.75)),
            n_site_months=("Monthly_CBI", "size"),
            extreme_month_proportion=(
                "Is_Extreme_Month_0p30", "mean"
            )
        )
        .reset_index()
        .sort_values("Month")
    )

    month_summary.to_csv(
        OUTPUT_DIR / OUTPUT_FILES["monthly_summary"],
        index=False,
        encoding="utf-8-sig"
    )

    fig, ax = plt.subplots(figsize=FIG_MONTHLY_TRAJECTORY_SIZE)

    # 各站点的月度CBI轨迹
    monthly_plot_groups = list(df_monthly_plot.groupby(SITE_COL))
    for site, group in progress_iter(
        monthly_plot_groups,
        total=len(monthly_plot_groups),
        desc="绘制站点月度轨迹",
        kind="绘图"
    ):
        group = group.sort_values("Month")
        ax.plot(
            group["Month"],
            group["Monthly_CBI"],
            color=COLOR_SITE_MONTH_LINE,
            alpha=ALPHA_MONTHLY_SITE,
            linewidth=LINEWIDTH_MONTHLY_SITE
        )

    # 全站点月度中位数和IQR
    ax.plot(
        month_summary["Month"],
        month_summary["median_CBI"],
        color=COLOR_MONTHLY_MEDIAN,
        linewidth=LINEWIDTH_MONTHLY_MEDIAN,
        marker=MARKER_MONTHLY_MEDIAN,
        label="全站点月度CBI中位数"
    )

    ax.fill_between(
        month_summary["Month"],
        month_summary["q25_CBI"],
        month_summary["q75_CBI"],
        color=COLOR_MONTHLY_MEDIAN,
        alpha=ALPHA_MONTHLY_IQR,
        label="站点间IQR"
    )

    # 顶部用红色短柱表示：该月中被定义为极端占比较高月的站点比例。
    # 它仅是背景信息，不表示因果效应。
    y_max = max(df_monthly_plot["Monthly_CBI"].max(), 1.0)
    top_y = y_max + 0.05

    for _, row in progress_iter(
        month_summary.iterrows(),
        total=len(month_summary),
        desc="绘制极端月份比例",
        kind="绘图"
    ):
        if row["extreme_month_proportion"] > 0:
            ax.vlines(
                row["Month"],
                top_y,
                top_y + 0.10 * row["extreme_month_proportion"],
                color=COLOR_EXTREME,
                linewidth=LINEWIDTH_MONTHLY_EXTREME_BAR,
                alpha=ALPHA_MONTHLY_EXTREME_BAR
            )

    ax.axhline(1, color=COLOR_REFERENCE_LINE, linestyle="--", linewidth=LINEWIDTH_REFERENCE)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("月份")
    ax.set_ylabel("月度 CBI")
    ax.set_title(
        "月度CBI季节轨迹（描述性结果）\n"
        "顶部红线：Extreme_Ratio >= 0.30 的站点比例（描述性标记，非统计显著）"
    )
    ax.legend(loc=LEGEND_LOCATION_BEST)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / OUTPUT_FILES["monthly_plot"],
        dpi=FIG_DPI,
        bbox_inches="tight"
    )
    plt.close(fig)

else:
    print("警告：没有满足最低小时数要求的月度CBI记录，跳过月度图。")


# =============================================================================
# Step 7. 最终运行审计报告
# =============================================================================

print("\n" + "=" * 80)
print("Step 7: 写入最终审计报告")
print("=" * 80)

final_audit_text = f"""
极端干旱 vs 未发生极端干旱的微气候缓冲能力分析
运行审计报告
===============================================================================

一、输入数据
-------------------------------------------------------------------------------
逐小时温度表: {TEMP_HOURLY_FILE}
逐日SPI宽表: {SPI_DAILY_WIDE_FILE}
干旱事件长表: {DROUGHT_EVENT_FILE}
建议Python环境目录: {PYTHON_ENV_DIR}
建议Python解释器: {PYTHON_INTERPRETER}
绘图中文字体: {SELECTED_CHINESE_FONT or "未找到候选中文字体"}
分析日期字段: {ANALYSIS_TIME_COL}
分析日期体系: UTC_Date = Time_UTC 的日期部分；不提供任何非UTC日期备用入口。

二、质量控制
-------------------------------------------------------------------------------
CSV原始逐小时记录: {n_raw_csv_rows:,}
UTC时间解析有效记录: {n_after_utc_parse:,}
温度质量控制后记录: {n_after_temp_qc:,}
去除完全重复后记录: {n_after_dedup:,}
最终小时记录: {len(df_hourly):,}
温度站点数量: {df_hourly[SITE_COL].nunique()}
Time_UTC解析率: {time_utc_audit_summary["Time_UTC_parse_rate"].iloc[0]:.4%}
Time_UTC解析失败记录数: {int(time_utc_audit_summary["Time_UTC_invalid_rows"].iloc[0])}
SPI匹配率: {spi_match_rate:.2%}
UTC日期范围: {df_hourly[UTC_DATE_COL].min().date()} 至 {df_hourly[UTC_DATE_COL].max().date()}

三、SPI分类规则
-------------------------------------------------------------------------------
Extreme: SPI30d <= {EXTREME_SPI_THRESHOLD}
Normal: {NORMAL_SPI_LOW} < SPI30d < {NORMAL_SPI_HIGH}
Other: -2.0 < SPI30d <= -1.5 重度干旱；-1.5 < SPI30d <= -1.0 中度干旱；
       -1.0 < SPI30d <= -0.5 轻度干旱；0.5 <= SPI30d < 1.0 轻度湿润；
       1.0 <= SPI30d < 1.5 中度湿润；1.5 <= SPI30d < 2.0 严重湿润；
       SPI30d >= 2.0 极端湿润；以及SPI缺失或无法分类记录（不进入主LMM）

四、主LMM样本
-------------------------------------------------------------------------------
Extreme + Normal候选小时: {len(df_lmm_all):,}
进入主LMM小时: {len(df_lmm):,}
进入主LMM站点: {df_lmm[SITE_COL].nunique()}
进入主LMM站点-月份层: {df_lmm["Site_Month"].nunique()}
主LMM状态小时门槛: Extreme >= {MIN_STATUS_HOURS_PER_SITE_MONTH} 且 Normal >= {MIN_STATUS_HOURS_PER_SITE_MONTH}
Macro_Within定义: 当前小时ERA5 - 该站点该UTC月份全部有效小时ERA5均值。
状态小时门槛敏感性审计: {OUTPUT_FILES["status_threshold_sensitivity"]}
站点-月份宏气候背景审计: {OUTPUT_FILES["site_month_macro_audit"]}

主模型公式：
{main_formula}

随机效应结构：
随机截距 + Macro_Within随机斜率，分组为Site_ID。

主结果：
Macro × Extreme交互项（Delta_CBI）: {interaction_beta:.6f}
95% CI: [{interaction_ci[0]:.6f}, {interaction_ci[1]:.6f}]
p值: {interaction_p:.6g}
样本加权Normal CBI: {weighted_normal_cbi:.6f}
样本加权Extreme CBI: {weighted_extreme_cbi:.6f}

解释：
Delta_CBI > 0 表示Extreme期间CBI更高，即微气候缓冲减弱；
Delta_CBI < 0 表示Extreme期间CBI更低，即表观缓冲增强。

五、站点-月份CBI描述性补充图
-------------------------------------------------------------------------------
计算单元: Site_ID × YearMonth × SPI_Status
CBI估计方法: 每个单元内用逐小时数据拟合 OLS:
Observed_T15cm_C = Intercept + CBI × ERA5_T2m_C

站点前置筛选:
- 仅保留全年至少出现过1个Extreme小时的站点；
- 全年无Extreme小时的站点不贡献Normal CBI；
- 目的：避免没有极端干旱背景的站点膨胀Normal样本量。

候选站点数: {len(all_lmm_candidate_sites)}
全年有Extreme站点数: {len(extreme_sites)}
全年无Extreme而被排除站点数: {len(excluded_no_extreme_sites)}

筛选规则:
- SPI_Status仅使用Normal与Extreme；
- 每个站点-月份-状态至少{MIN_SITE_MONTH_STATUS_CBI_HOURS}个有效小时；
- 每个站点-月份-状态ERA5温度标准差至少{MIN_SITE_MONTH_STATUS_MACRO_SD}°C；
- 03B配对箱线图和03C配对变化图仅使用同一Site_ID × YearMonth同时具有合格Normal和Extreme CBI的单元；
- 03D研究区总体箱线图使用全部合格站点-月份-状态CBI，属于非配对总体分布图。

说明:
这些图用于展示站点-月份层面的CBI分布、总体趋势和配对方向，
属于描述性补充结果，不替代主LMM显著性检验。

输出文件:
全年极端干旱站点筛选审计表: {OUTPUT_FILES["extreme_site_filter_audit"]}
站点月份状态CBI估计表: {OUTPUT_FILES["site_month_status_cbi"]}
站点月份CBI配对变化审计表: {OUTPUT_FILES["site_month_cbi_pair_audit"]}
站点月份CBI配对变化汇总表: {OUTPUT_FILES["site_month_cbi_pair_summary"]}
极端事件是否进入站点月份CBI配对审计表: {OUTPUT_FILES["event_to_site_month_pair_audit"]}
研究区总体CBI分布汇总表: {OUTPUT_FILES["regional_cbi_summary"]}
主LMM单图: {OUTPUT_FILES["main_prediction_plot_split"]}
配对样本站点月份CBI箱线散点图: {OUTPUT_FILES["site_month_cbi_box_plot"]}
站点月份CBI配对变化图: {OUTPUT_FILES["site_month_cbi_pair_plot"]}
研究区总体CBI箱线图（非配对）: {OUTPUT_FILES["regional_cbi_box_plot"]}
主LMM与站点月份CBI综合图: {OUTPUT_FILES["main_cbi_composite_plot"]}

站点月份配对汇总:
{df_site_month_cbi_pair_summary.to_string(index=False) if not df_site_month_cbi_pair_summary.empty else "无满足筛选条件的配对单元"}

六、事件级辅助分析
-------------------------------------------------------------------------------
极端事件总数: {len(df_event_cbi)}
有可靠Event-CBI与Reference-CBI配对的事件数:
{(df_event_cbi["Pair_flag"] == "ok").sum()}

参考期规则：
- 事件结束后第{POST_EVENT_BUFFER_DAYS}天作为缓冲日，不使用；
- 从结束后第{POST_EVENT_BUFFER_DAYS + 1}天起；
- 最多搜索{POST_EVENT_SEARCH_DAYS}个自然日；
- 仅选择Normal SPI日；
- 单个参考候选日需至少{MIN_VALID_HOURS_PER_REFERENCE_DAY}个有效小时；
- 目标日数 = min(Event Duration, {MAX_REF_TARGET_DAYS})；
- 找不到目标日数时，仅使用30日搜索窗口内实际找到的合格Normal日；
- 参考期至少{MIN_REF_HOURS}小时才计算可靠CBI；
- 不足时保留事件与审计记录，但不用于正式Delta_CBI配对检验。
未进入配对检验原因审计: {OUTPUT_FILES["event_unpaired_audit"]}
候选日期级审计: {OUTPUT_FILES["event_reference_dates"]}

事件级站点聚类bootstrap结果：
{df_bootstrap_result.to_string(index=False)}

七、月度描述性阈值
-------------------------------------------------------------------------------
Extreme_Ratio >= {EXTREME_RATIO_MONTHLY}
说明：该阈值仅标记“极端状态占比较高的月份”，不表示统计显著，
月度分析不作为本研究主显著性结论。
"""

report_steps = progress_bar(
    total=1,
    desc="写入最终审计报告",
    kind="报告输出"
)
write_text(
    OUTPUT_DIR / OUTPUT_FILES["final_report"],
    final_audit_text
)
report_steps.update(1)
report_steps.close()

print(final_audit_text)

cleaned_cache_records = cleanup_runtime_cache()

print("\n" + "=" * 80)
print("分析完成。主要结果文件：")
print(f"0. {OUTPUT_FILES['time_audit_summary']} —— UTC时间解析审计")
print(f"1. {OUTPUT_FILES['main_lmm_key']} —— 主结论（重点查看）")
print(f"2. {OUTPUT_FILES['main_lmm_summary']} —— 主LMM完整输出")
print(f"3. {OUTPUT_FILES['main_prediction_plot']} —— 主结果图")
print(f"4. {OUTPUT_FILES['main_cbi_composite_plot']} —— 主LMM与站点月份CBI综合图")
print(f"5. {OUTPUT_FILES['site_month_cbi_box_plot']} —— 配对样本站点月份CBI箱线散点图")
print(f"6. {OUTPUT_FILES['site_month_cbi_pair_plot']} —— 站点月份CBI配对变化图")
print(f"7. {OUTPUT_FILES['regional_cbi_box_plot']} —— 研究区总体CBI箱线图（非配对）")
print(f"8. {OUTPUT_FILES['extreme_site_filter_audit']} —— 全年极端干旱站点筛选审计")
print(f"9. {OUTPUT_FILES['site_month_status_cbi']} —— 站点月份状态CBI估计表")
print(f"10. {OUTPUT_FILES['site_month_cbi_pair_summary']} —— 站点月份CBI配对变化汇总")
print(f"11. {OUTPUT_FILES['event_to_site_month_pair_audit']} —— 极端事件是否进入站点月份CBI配对审计")
print(f"12. {OUTPUT_FILES['event_cbi_reference']} —— 事件级辅助结果")
print(f"13. {OUTPUT_FILES['event_unpaired_audit']} —— 未进入事件CBI配对检验原因审计")
print(f"14. {OUTPUT_FILES['event_bootstrap']} —— 事件级辅助显著性")
print(f"15. {OUTPUT_FILES['monthly_cbi']} —— 月度描述性CBI")
print(f"16. {OUTPUT_FILES['final_report']} —— 全流程审计报告")
print(f"17. {OUTPUT_FILES['status_threshold_sensitivity']} —— 状态小时门槛敏感性审计")
print(f"本次运行结束清理缓存/临时文件数量: {len(cleaned_cache_records)}")
print(f"\n全部结果保存位置：{OUTPUT_DIR.resolve()}")
print("=" * 80)
