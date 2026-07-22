"""
================================================================================
第一阶段分析脚本：极端干旱事件期 vs 等长前后对照期 微气候缓冲指数(β)对比
================================================================================

【研究目的】
    比较极端干旱期间与非干旱期间，森林林下微气候缓冲能力（β）的差异。
    β 是 OLS 回归方程 T_micro = α + β × T_macro 中的斜率：
      - β 越小（接近0）：林下温度受宏气候影响越小，缓冲能力越强
      - β 越大（接近1）：林下温度紧随宏气候波动，缓冲能力越弱
      - β > 1：林下温度波动比宏气候更剧烈（放大效应）

【核心方法】
    对照期设计：等长配对（Length-matched paired control design）
      参考文献：Zellweger et al. (2020) Global Change Biology
                De Frenne et al. (2021) Nature Ecology & Evolution
    β 计算方法：普通最小二乘回归（OLS）
      参考文献：Ma et al. (2025) Agricultural and Forest Meteorology
    统计检验：Wilcoxon 符号秩检验（配对，非参数）
      参考文献：Hollander & Wolfe (1999) Nonparametric Statistical Methods

【数据要求】
    1. 27个TOMST传感器 CSV（15分钟频率，T3_15列为林下15cm气温）
    2. ERA5-Land 逐小时 TIF（2m气温，共8760张，代表2025年每小时）
    3. 样地坐标 CSV（Site_ID, Longitude, Latitude）
    4. 干旱事件长表 CSV（含Start_Date, End_Date, Drought_Level_Code等列）

【输出文件】
    事件级β配对结果表.csv        ：每次目标干旱事件的三组β详细表（事件级）
    站点级β汇总表.csv            ：每个站点的汇总β表（站点级）
    对照期质量记录表.csv          ：前/后对照期原始窗口、实际连续非干旱天数和截断说明
    站点事件覆盖记录表.csv        ：每个站点是否有目标等级干旱事件及未进入分析原因
    极端干旱与对照期β箱线图.png   ：干旱期 vs 前/后对照期 箱线图
    统计检验结果报告.txt          ：Wilcoxon检验报告

【作者】根据 Ma et al. 2025 方法框架实现
================================================================================
"""

# ============================================================
# 0. 依赖库导入
# ============================================================
import os
import glob
import re
import shutil
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon
import rasterio               # 用于读取GeoTIFF格式的ERA5栅格数据
import matplotlib
matplotlib.use('Agg')         # 非交互式后端，适用于服务器/无显示器环境
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ============================================================
# ★★★ 用户配置区 ★★★
# 所有需要修改的参数集中在这里，运行前请逐一核对
# ============================================================

# ── 数据路径 ────────────────────────────────────────────────

# TOMST传感器CSV所在文件夹
# 要求：文件夹内只放数字命名的CSV（如95332217.csv），脚本会自动识别
TOMST_DIR = r"E:\forest_microclimate\ForestMicroclimate\Tensor_Data"

# ERA5逐小时TIF所在文件夹
# 要求：共8760张TIF（365天×24小时），每张代表一个UTC整点小时的2m气温
# 文件名格式必须包含类似"2025年01月01日00时"的时间信息供脚本解析
ERA5_TIF_DIR = r"E:\forest_microclimate\ForestMicroclimate\T2m\fujian_T2"

# 样地坐标CSV路径
# 必须包含三列：站点编号列、经度列、纬度列（列名见下方 SITE_*_COL 配置）
SITES_CSV = r"E:\forest_microclimate\ForestMicroclimate\Tensor_LatLong.csv"

# 干旱事件长表CSV路径
# 必须包含列：Site_ID, Start_Date, End_Date, Drought_Level_Code, Event_ID 等
DROUGHT_CSV = (r"E:\forest_microclimate\ForestMicroclimate\results"
               r"\daily_SPI20_features\福建省观测站2025年daily_SPI20干旱事件长表.csv")

# 逐日SPI序列表路径
# 用途：只用于前/后对照期筛选。干旱期仍以 DROUGHT_CSV 中的事件起止日期为准。
# 对照期规则：先取等长窗口；若窗口内遇到干旱日，则截断，只保留最靠近目标事件的连续非干旱日，
# 不再向更远时间补足天数。
DAILY_SPI_CSV = (r"E:\forest_microclimate\ForestMicroclimate\results"
                 r"\daily_SPI20_features\福建省观测站2025年daily_SPI20逐日序列表.csv")

# 所有输出文件的保存文件夹（不存在会自动创建）
OUTPUT_DIR = r"E:\forest_microclimate\ForestMicroclimate\results\compare_differences_20"

# 本脚本若未来启用磁盘缓存或临时中间文件，统一放到该目录。
# 当前主流程主要使用内存DataFrame，不主动写临时文件；结束时仍会清理该目录，防止残留。
TEMP_DIR = os.path.join(OUTPUT_DIR, "_本次运行临时缓存")

# ── TOMST CSV 列名配置 ───────────────────────────────────────

# UTC时间列的列名（格式为 2024.10.31 10:15）
# 注意：这是UTC时间，不是UTC+8（UTC+8时间列是data_time8，此处不使用）
TOMST_DATETIME_COL = "data_time"

# 林下15cm大气温度列的列名
# T3_15 = 距地面15cm处的气温，代表林下微气候温度
# 其他可选列：T1_5（5cm土温）、T2_0（0cm气温）——根据研究需要修改
TOMST_TEMP_COL = "T3_15"

# data_time 列的时间格式字符串（Python strptime格式）
# 当前格式对应 "2024.10.31 10:15"
# 如果你的时间格式不同（如 "2024-10-31 10:15"），需修改为 "%Y-%m-%d %H:%M"
TOMST_TIME_FORMAT = "%Y.%m.%d %H:%M"

# ── 样地坐标 CSV 列名配置 ───────────────────────────────────

SITE_ID_COL  = "Site_ID"    # 站点编号列名（数值或字符串，脚本会统一转整数字符串）
SITE_LON_COL = "Longitude"  # 经度列名（WGS84十进制度，如 117.63227）
SITE_LAT_COL = "Latitude"   # 纬度列名（WGS84十进制度，如 25.27501）

# ── ERA5 单位配置 ────────────────────────────────────────────

# 当前福建T2m TIF已预处理为摄氏度(℃)，因此保持 False。
# 若以后改用原始ERA5/ERA5-Land开尔文(K)栅格，则改为 True，脚本会自动执行 ℃ = K - 273.15。
ERA5_UNIT_K = False

# ── 干旱等级筛选 ─────────────────────────────────────────────

# 只分析哪个等级的干旱事件（对应干旱事件长表中的 Drought_Level_Code 列）。
# 等级对应关系：1=轻度(Light), 2=中度(Moderate), 3=严重(Severe), 4=极端(Extreme)
# 支持两种写法：
#   TARGET_DROUGHT_LEVEL_CODE = 4       # 只分析极端干旱
#   TARGET_DROUGHT_LEVEL_CODE = [3, 4]  # 分析严重及以上
# 脚本会自动判断单值或列表，不需要再修改筛选代码。
TARGET_DROUGHT_LEVEL_CODE = 4

# ── 对照期设计参数 ───────────────────────────────────────────

# 对照期设计方法：等长配对（Length-matched paired control design）
# 参考：Zellweger et al. (2020) Global Change Biology
#       De Frenne et al. (2021) Nature Ecology & Evolution
#
# 逻辑示意图（以干旱持续18天为例，gap=1天）：
#   [前置对照期18天] [1天缓冲] [极端干旱期18天] [1天缓冲] [后置对照期18天]
#
# CONTROL_GAP_DAYS：对照期与干旱事件之间的缓冲间隔天数
# 设置缓冲是为了避免"干旱边界效应"（干旱开始/结束前后几天的过渡状态）污染对照期
# 建议值：1~3天；若干旱期很短（<10天），可设为0避免对照期被大幅压缩
CONTROL_GAP_DAYS = 1

# 短事件等长配对与长事件相对偏差法的分界阈值（单位：天）。
# 规则说明：
#   - 干旱持续天数 <= 28 天：继续使用等长配对法（paired control）
#   - 干旱持续天数 >  28 天：改用相对偏差法（relative deviation）
# 这样做的原因是：长干旱事件若继续强行寻找等长对照期，很容易跨季节、
# 跨背景期，导致“对照期”不再可比。
LONG_DROUGHT_THRESH = 28

# 等长对照法的目标对照期长度上限（单位：天）。
# 当前与 LONG_DROUGHT_THRESH 保持一致，意味着只有短事件才会真正进入
# 等长配对流程；长事件虽然也会记录 capped 信息，但不再以配对对照为主方法。
MAX_CONTROL_DAYS = 28

# 长干旱事件计算“同季节非干旱基线β”时，允许的月份偏移范围（单位：月）。
# 例如事件开始于7月，MAX_MONTH_OFFSET=1，则基线月份允许取 6/7/8 月。
MAX_MONTH_OFFSET = 1

# ── OLS 回归质量控制 ────────────────────────────────────────

# 计算β所需的最少有效逐小时数据点数
# 低于此值则该期间β设为NaN，不参与统计检验
# 设定依据：至少需要1整天(24h)的数据才能建立有意义的T_micro~T_macro线性关系
# 若某些站点数据缺失严重，可适当降低（如12），但会降低β估计的可靠性
MIN_HOURS_FOR_OLS = 24

# ── 逐日SPI表列名配置 ───────────────────────────────────────

DAILY_SPI_SITE_COL = "Site_ID"
DAILY_SPI_DATE_COL = "Date"
DAILY_SPI_VALUE_COL = "Daily_SPI_20d"
DAILY_SPI_DROUGHT_COL = "Is_Drought_Day"
CONTROL_SPI_THRESHOLD = -0.5

# ── tqdm 进度条配置 ─────────────────────────────────────────

TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
TQDM_NCOLS_DYNAMIC = True
TQDM_LEAVE = False
PROGRESS_COLOURS = {
    "tomst": "green",
    "era5": "cyan",
    "beta": "yellow",
    "output": "magenta",
    "diagnostic": "blue",
}

# ── 图形参数配置 ─────────────────────────────────────────────
# 后续调图优先修改这里，不需要到绘图函数内部查找参数。

FIG_SIZE = (7.5, 5.5)
FIG_DPI = 300
BOX_WIDTH = 0.45
BOX_LINEWIDTH = 1.2
BOX_COLORS = ['#2196F3', '#E53935', '#4CAF50']
BOX_COLORS_TWO_PERIOD = ['#26A69A', '#E53935']
LINE_COLORS = {
    'Pre': '#2196F3',
    'Drought': '#E53935',
    'Post': '#4CAF50',
    'Non-drought': '#26A69A',
}
POINT_COLOR = 'black'
POINT_SIZE = 3.5
POINT_ALPHA = 0.4
POINT_JITTER = True
REFERENCE_LINE_Y = 1.0
REFERENCE_LINE_COLOR = 'gray'
REFERENCE_LINE_STYLE = '--'
REFERENCE_LINE_ALPHA = 0.4
SIGNIFICANCE_LINE_COLOR = 'black'
SIGNIFICANCE_LINEWIDTH = 1
SAMPLE_N_COLOR = '#555555'
AXIS_LABEL_FONTSIZE = 11
TITLE_FONTSIZE = 11
LEGEND_FONTSIZE = 9
SITE_FIG_SIZE = (11, 5.5)
BAR_WIDTH = 0.22


# ============================================================
# 工具函数区
# ============================================================

def ensure_dir(path):
    """
    确保输出文件夹存在，不存在则自动创建（包括多级目录）
    参数：
        path: 文件夹路径字符串
    """
    os.makedirs(path, exist_ok=True)


def cleanup_temp_dir(temp_dir):
    """
    清理本次脚本运行产生的临时缓存目录。

    设计说明：
      - 当前脚本不主动生成临时中间文件；
      - 若后续为了加速ERA5提取或调试而启用磁盘缓存，应统一写入 TEMP_DIR；
      - 主流程结束或异常退出时会尝试删除 TEMP_DIR，避免遗留临时文件。
    """
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)


def normalize_site_id(val):
    """
    统一站点编号格式为纯整数字符串，避免不同数据表格式不一致导致合并失败。

    问题背景：
        TOMST文件名：95332217（整数）
        干旱长表 Site_ID：95332217.0（pandas读取CSV时浮点化）
        坐标文件 Site_ID：可能是 "95332217" 或 95332217.0
    本函数统一转为 "95332217"（字符串整数），确保三表能正确匹配。

    参数：
        val: 原始站点编号（任意格式）
    返回：
        字符串整数（如 "95332217"），若输入为NaN则返回 None
    """
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def parse_era5_tif_datetime(path):
    """
    从ERA5 TIF文件名中解析UTC时间戳。

    支持的文件名格式（示例）：
        福建省2米气温_2025年01月01日00时.tif
        T2m_2025年06月15日14时.tif
    只要文件名中包含 "YYYY年MM月DD日HH时" 模式即可解析。

    ★ 如果你的文件名格式不同（如 T2m_20250101_00.tif），
      需要修改下方 re.search() 的正则表达式来匹配你的格式。

    参数：
        path: TIF文件的完整路径
    返回：
        pd.Timestamp（UTC时间）；若解析失败则返回 pd.NaT（会被后续步骤过滤）
    """
    name = os.path.basename(path)
    # 正则匹配：年月日时（均为数字，月日时补零）
    m = re.search(r'(\d{4})年(\d{2})月(\d{2})日(\d{2})时', name)
    if not m:
        return pd.NaT
    y, mo, d, h = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return pd.Timestamp(year=y, month=mo, day=d, hour=h)


def run_ols(index_series, t_micro_series, t_macro_series):
    """
    对给定的逐小时温度数据做普通最小二乘（OLS）回归：
        T_micro = α + β × T_macro

    其中：
        α（截距）：林下温度在宏气候为0℃时的基准值（实际意义较弱）
        β（斜率）：核心指标，即"气候缓冲指数（CBI）"
            β ≈ 0：林内温度几乎不随宏气候变化（极强缓冲）
            β ≈ 1：林内温度完全跟随宏气候（无缓冲）
            β > 1：林内温度波动超过宏气候（放大效应，极少见）

    方法参考：Ma et al. (2025) Agricultural and Forest Meteorology

    参数：
        index_series  : 时间索引（用于Debug，不参与计算）
        t_micro_series: 林下微气候温度序列（T_micro，pd.Series）
        t_macro_series: 宏气候温度序列（T_macro，pd.Series）
    返回：
        dict，包含：
            beta    : OLS斜率（缓冲指数）
            alpha   : OLS截距
            r2      : 决定系数 R²（衡量线性拟合优度，0~1）
            p_value : F检验p值（衡量回归显著性）
            n       : 参与回归的有效数据点数（已去除NaN）
    """
    x = t_macro_series.values
    y = t_micro_series.values
    # 去除任意一列含NaN的行（两列必须同时有效才能建立配对关系）
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)

    # 数据点不足时返回NaN（不强行拟合，避免虚假结果）
    if n < MIN_HOURS_FOR_OLS:
        return dict(beta=np.nan, alpha=np.nan, r2=np.nan, p_value=np.nan, n=n)

    slope, intercept, r, p, _ = stats.linregress(x, y)
    return dict(beta=slope, alpha=intercept, r2=r**2, p_value=p, n=n)


def filter_target_events(drought_df, target_level_code):
    """
    根据 Drought_Level_Code 筛选目标干旱事件。

    target_level_code 支持单值或列表：
      - 4      ：只分析极端干旱
      - [3, 4] ：分析严重及以上
    """
    if isinstance(target_level_code, (list, tuple, set)):
        return drought_df[drought_df['Drought_Level_Code'].isin(target_level_code)].copy()
    return drought_df[drought_df['Drought_Level_Code'] == target_level_code].copy()


def count_target_events(drought_df, target_level_code):
    """统计目标等级干旱事件数量，支持单值和列表配置。"""
    return len(filter_target_events(drought_df, target_level_code))


def load_daily_spi_table(path):
    """
    读取逐日SPI序列表，用于前/后对照期过滤。

    注意：
      1. 该表只用于判断对照期哪些日期是干旱日；
      2. 干旱期仍使用干旱事件长表中的 Start_Date~End_Date，不再按逐日SPI过滤；
      3. Is_Drought_Day=True 的日期会截断对照期，只保留靠近目标事件的一段连续非干旱日。
    """
    daily = pd.read_csv(path)
    required = [DAILY_SPI_SITE_COL, DAILY_SPI_DATE_COL, DAILY_SPI_VALUE_COL, DAILY_SPI_DROUGHT_COL]
    missing = [c for c in required if c not in daily.columns]
    if missing:
        raise KeyError(
            f"逐日SPI表缺少必要列：{missing}\n"
            f"实际列名：{list(daily.columns)}"
        )

    daily = daily[[DAILY_SPI_SITE_COL, DAILY_SPI_DATE_COL, DAILY_SPI_VALUE_COL, DAILY_SPI_DROUGHT_COL]].copy()
    daily.columns = ['site_id', 'date', 'daily_spi', 'is_drought_day']
    daily['site_id'] = daily['site_id'].apply(normalize_site_id)
    daily['date'] = pd.to_datetime(daily['date']).dt.normalize()
    daily['daily_spi'] = pd.to_numeric(daily['daily_spi'], errors='coerce')

    if daily['is_drought_day'].dtype == object:
        daily['is_drought_day'] = (
            daily['is_drought_day'].astype(str).str.strip().str.lower()
            .map({'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False})
        )
    daily['is_drought_day'] = daily['is_drought_day'].fillna(daily['daily_spi'] <= CONTROL_SPI_THRESHOLD).astype(bool)
    return daily


def get_control_period_by_daily_spi(site_daily_spi, raw_start, raw_end, side):
    """
    根据逐日SPI表截断前/后对照期。

    参数：
      site_daily_spi : 单个站点的逐日SPI表，必须包含 date/is_drought_day/daily_spi
      raw_start/raw_end : 等长对照期原始窗口
      side : 'pre' 或 'post'

    搜索规则：
      - 目标是累计找到与原始窗口等长的非干旱日；
      - 前置对照期：从 raw_end 往更早日期搜索；
      - 后置对照期：从 raw_start 往更晚日期搜索；
      - 若遇到干旱日或缺少SPI记录，则跳过该日，继续向外搜索；
      - 直到找到足够的非干旱日，或超出该站点逐日SPI记录范围。
    """
    raw_start = raw_start.normalize()
    raw_end = raw_end.normalize()
    raw_days = len(pd.date_range(raw_start, raw_end, freq='D'))

    if site_daily_spi.empty:
        return {
            'actual_start': None,
            'actual_end': None,
            'raw_days': raw_days,
            'actual_days': 0,
            'shortfall_days': raw_days,
            'full_length': False,
            'truncated': True,
            'note': "逐日SPI表无该站点记录",
            'search_start': None,
            'search_end': None,
            'skipped_drought_days': 0,
            'skipped_missing_days': 0,
            'used_days': [],
        }

    spi_map = site_daily_spi.drop_duplicates(subset=['date']).set_index('date').sort_index()
    min_date = spi_map.index.min()
    max_date = spi_map.index.max()

    if side == 'pre':
        current_day = raw_end
        step = -1
    elif side == 'post':
        current_day = raw_start
        step = 1
    else:
        raise ValueError("side 必须为 'pre' 或 'post'")

    kept_days = []
    skipped_drought_days = 0
    skipped_missing_days = 0
    search_days = []

    while len(kept_days) < raw_days:
        if current_day < min_date or current_day > max_date:
            break
        search_days.append(current_day)
        if current_day in spi_map.index:
            row = spi_map.loc[current_day]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if bool(row['is_drought_day']):
                skipped_drought_days += 1
            else:
                kept_days.append(current_day)
        else:
            skipped_missing_days += 1

        current_day = current_day + pd.Timedelta(days=step)

    kept_days = sorted(kept_days)
    actual_days = len(kept_days)
    actual_start = kept_days[0] if kept_days else None
    actual_end = kept_days[-1] if kept_days else None
    full_length = actual_days == raw_days
    truncated = not full_length
    search_start = min(search_days) if search_days else None
    search_end = max(search_days) if search_days else None

    if full_length:
        note = f"已补足等长非干旱日；跳过干旱日 {skipped_drought_days} 天"
    elif actual_days == 0:
        note = "搜索范围内未找到可用非干旱日"
    else:
        note = (
            f"未补足等长；获得 {actual_days}/{raw_days} 天，"
            f"跳过干旱日 {skipped_drought_days} 天，缺失SPI {skipped_missing_days} 天"
        )

    return {
        'actual_start': actual_start,
        'actual_end': actual_end,
        'raw_days': raw_days,
        'actual_days': actual_days,
        'shortfall_days': raw_days - actual_days,
        'full_length': full_length,
        'truncated': truncated,
        'note': note,
        'search_start': search_start,
        'search_end': search_end,
        'skipped_drought_days': skipped_drought_days,
        'skipped_missing_days': skipped_missing_days,
        'used_days': kept_days,
    }


def validate_required_columns(df, required_cols, table_name):
    """统一检查输入表字段，缺列时立即给出明确错误。"""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{table_name} 缺少必要列：{missing}\n实际列名：{list(df.columns)}")


def sanitize_filename(text):
    """将站点编号/标题文本转换为Windows安全文件名。"""
    return re.sub(r'[\\\\/:*?"<>|]+', '_', str(text))


def format_date_range(start, end):
    """
    将起止日期格式化为英文绘图标签中的月/日范围。
    例如：2025-05-01 ~ 2025-05-10 → 5/1-5/10
    """
    if pd.isna(start) or pd.isna(end):
        return "NA"
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    return f"{start_dt.month}/{start_dt.day}-{end_dt.month}/{end_dt.day}"


def get_allowed_months_around_drought(drought_start, max_offset_months):
    """
    根据事件开始月份，返回允许纳入“同季节窗口”的月份集合。

    例如：
      drought_start 在 7 月，max_offset_months=1
      → 允许月份为 {6, 7, 8}

    使用循环月份差，自动处理跨年情况：
      1 月前一个月视为 12 月，12 月后一个月视为 1 月。
    """
    drought_month = pd.Timestamp(drought_start).month
    allowed_months = set()
    for offset in range(-max_offset_months, max_offset_months + 1):
        allowed_months.add(((drought_month - 1 + offset) % 12) + 1)
    return sorted(allowed_months)


def is_within_month_offset(date, drought_start, max_offset_months):
    """
    检查某日期的月份是否位于干旱事件起始月的允许偏移范围内。

    该约束仅比较“月份背景”是否相似，不比较精确日距。
    例如事件开始于 7 月、max_offset_months=1 时：
      6/7/8 月返回 True，其他月份返回 False。
    """
    drought_month = pd.Timestamp(drought_start).month
    check_month = pd.Timestamp(date).month
    diff = abs(check_month - drought_month)
    circular_diff = min(diff, 12 - diff)
    return circular_diff <= max_offset_months


def compute_event_baseline_beta(site_data, site_daily_spi, drought_start,
                                max_offset_months=MAX_MONTH_OFFSET):
    """
    为“单个站点 × 单次长干旱事件”计算同季节非干旱基线β。

    基线定义：
      1. 与该事件开始月份相同或相邻月份（± max_offset_months）；
      2. 仅使用逐日SPI表判定为非干旱日（is_drought_day=False）的日期；
      3. 将这些日期对应的全部逐小时 T_micro / T_macro 数据汇总后做 OLS。

    设计目的：
      当干旱事件过长（>28天）时，不再强行寻找等长对照期，而是改用
      “同季节的非干旱背景”作为参考基线，再计算相对偏差 Δβ。
    """
    drought_start = pd.Timestamp(drought_start).normalize()
    allowed_months = get_allowed_months_around_drought(drought_start, max_offset_months)

    if site_daily_spi.empty:
        return {
            'beta_baseline': np.nan,
            'r2_baseline': np.nan,
            'n_baseline': 0,
            'baseline_days': [],
            'baseline_day_count': 0,
            'allowed_months': allowed_months,
            'note': "逐日SPI表无该站点记录，无法计算事件级baseline",
        }

    site_daily_spi = site_daily_spi.copy()
    site_daily_spi['month'] = site_daily_spi['date'].dt.month
    baseline_day_mask = (
        (~site_daily_spi['is_drought_day']) &
        (site_daily_spi['month'].isin(allowed_months))
    )
    baseline_days = sorted(site_daily_spi.loc[baseline_day_mask, 'date'].drop_duplicates().tolist())

    if not baseline_days:
        return {
            'beta_baseline': np.nan,
            'r2_baseline': np.nan,
            'n_baseline': 0,
            'baseline_days': [],
            'baseline_day_count': 0,
            'allowed_months': allowed_months,
            'note': "允许月份窗口内无可用非干旱日，无法计算事件级baseline",
        }

    baseline_hours = site_data[site_data.index.normalize().isin(baseline_days)].copy()
    beta_baseline = run_ols(
        baseline_hours.index,
        baseline_hours['T_micro'],
        baseline_hours['T_macro']
    )

    if beta_baseline['n'] < MIN_HOURS_FOR_OLS:
        note = (
            f"允许月份 {allowed_months} 内共找到 {len(baseline_days)} 个非干旱日，"
            f"但有效小时数仅 {beta_baseline['n']}，不足以计算baseline β"
        )
    else:
        note = (
            f"允许月份 {allowed_months} 内共使用 {len(baseline_days)} 个非干旱日，"
            f"有效小时数 {beta_baseline['n']}，已计算事件级baseline β"
        )

    return {
        'beta_baseline': beta_baseline['beta'],
        'r2_baseline': beta_baseline['r2'],
        'n_baseline': beta_baseline['n'],
        'baseline_days': baseline_days,
        'baseline_day_count': len(baseline_days),
        'allowed_months': allowed_months,
        'note': note,
    }


def collect_event_baseline_hours(site_data, site_daily_spi, drought_start,
                                 max_offset_months=MAX_MONTH_OFFSET):
    """
    收集“单个站点 × 单次长干旱事件”的事件级baseline逐小时数据。

    返回：
      baseline_hours : 逐小时DataFrame（index 为 datetime）
      baseline_days  : 纳入baseline的逐日日期列表
      allowed_months : 允许月份窗口
    """
    drought_start = pd.Timestamp(drought_start).normalize()
    allowed_months = get_allowed_months_around_drought(drought_start, max_offset_months)
    if site_daily_spi.empty:
        return site_data.iloc[0:0].copy(), [], allowed_months

    site_daily_spi = site_daily_spi.copy()
    site_daily_spi['month'] = site_daily_spi['date'].dt.month
    baseline_days = sorted(
        site_daily_spi.loc[
            (~site_daily_spi['is_drought_day']) &
            (site_daily_spi['month'].isin(allowed_months)),
            'date'
        ].drop_duplicates().tolist()
    )
    baseline_hours = site_data[site_data.index.normalize().isin(baseline_days)].copy()
    return baseline_hours, baseline_days, allowed_months


def collect_period_hours(site_data, start_date, end_date):
    """
    按日范围收集逐小时温度数据，包含起止日期的全天小时。
    """
    if pd.isna(start_date) or pd.isna(end_date):
        return site_data.iloc[0:0].copy()
    start_dt = pd.Timestamp(start_date).normalize()
    end_dt = pd.Timestamp(end_date).normalize()
    return site_data[
        (site_data.index >= start_dt) &
        (site_data.index <= end_dt + pd.Timedelta(hours=23))
    ].copy()


def plot_event_ols_diagnostic(event_row, site_data, site_daily_spi, output_dir):
    """
    为“单个站点 × 单次极端干旱事件”绘制 OLS 诊断图。

    图形目标：
      - 直接显示用于计算β的逐小时散点；
      - 用回归线展示斜率差异；
      - 便于检查 β 是否受异常点、样本量不足或温度范围偏窄影响。

    诊断图分两类：
      1. 短事件（<= LONG_DROUGHT_THRESH）：Drought vs Non-drought
      2. 长事件（>  LONG_DROUGHT_THRESH）：Drought vs Baseline
    """
    site_id = event_row['site_id']
    event_id = event_row['event_id']
    drought_start = pd.Timestamp(event_row['drought_start']).normalize()
    drought_end = pd.Timestamp(event_row['drought_end']).normalize()
    is_long_drought = bool(event_row['is_long_drought'])

    drought_hours = collect_period_hours(site_data, drought_start, drought_end)
    groups = []

    if is_long_drought:
        baseline_hours, baseline_days, allowed_months = collect_event_baseline_hours(
            site_data, site_daily_spi, drought_start, max_offset_months=MAX_MONTH_OFFSET
        )
        groups = [
            {'label': 'Drought', 'hours': drought_hours, 'color': LINE_COLORS['Drought']},
            {'label': 'Baseline', 'hours': baseline_hours, 'color': '#FB8C00'},
        ]
        method_note = (
            f"Long event baseline method | months={','.join(map(str, allowed_months))} | "
            f"baseline days={len(baseline_days)}"
        )
    else:
        nondrought_parts = []
        nondrought_pre = collect_period_hours(
            site_data,
            event_row['nondrought_pre_start'],
            event_row['nondrought_pre_end']
        )
        nondrought_post = collect_period_hours(
            site_data,
            event_row['nondrought_post_start'],
            event_row['nondrought_post_end']
        )
        if not nondrought_pre.empty:
            nondrought_parts.append(nondrought_pre)
        if not nondrought_post.empty:
            nondrought_parts.append(nondrought_post)
        if nondrought_parts:
            nondrought_hours = pd.concat(nondrought_parts).sort_index()
        else:
            nondrought_hours = site_data.iloc[0:0].copy()
        groups = [
            {'label': 'Drought', 'hours': drought_hours, 'color': LINE_COLORS['Drought']},
            {'label': 'Non-drought', 'hours': nondrought_hours, 'color': LINE_COLORS['Non-drought']},
        ]
        method_note = (
            f"Short event paired method | control days={event_row['control_duration_days']}"
        )

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    all_macro_vals = []
    all_micro_vals = []
    summary_lines = []

    for group in groups:
        hours = group['hours'][['T_macro', 'T_micro']].dropna().copy()
        label = group['label']
        color = group['color']
        if hours.empty:
            summary_lines.append(f"{label}: no valid hourly points")
            continue

        fit = run_ols(hours.index, hours['T_micro'], hours['T_macro'])
        all_macro_vals.extend(hours['T_macro'].tolist())
        all_micro_vals.extend(hours['T_micro'].tolist())

        ax.scatter(
            hours['T_macro'], hours['T_micro'],
            s=16, alpha=0.55, color=color, edgecolors='none', label=label, zorder=2
        )

        if pd.notna(fit['beta']) and fit['n'] >= 2:
            x_min = hours['T_macro'].min()
            x_max = hours['T_macro'].max()
            if x_min == x_max:
                x_line = np.array([x_min - 0.1, x_max + 0.1])
            else:
                x_line = np.linspace(x_min, x_max, 100)
            y_line = fit['alpha'] + fit['beta'] * x_line
            ax.plot(x_line, y_line, color=color, linewidth=2.1, alpha=0.95, zorder=3)
            summary_lines.append(
                f"{label}: β={fit['beta']:.3f}, R²={fit['r2']:.3f}, n={fit['n']}"
            )
        else:
            summary_lines.append(f"{label}: β=NA, R²=NA, n={fit['n']}")

    if all_macro_vals and all_micro_vals:
        macro_min, macro_max = min(all_macro_vals), max(all_macro_vals)
        micro_min, micro_max = min(all_micro_vals), max(all_micro_vals)
        x_pad = (macro_max - macro_min) * 0.06 if macro_max > macro_min else 0.5
        y_pad = (micro_max - micro_min) * 0.08 if micro_max > micro_min else 0.5
        ax.set_xlim(macro_min - x_pad, macro_max + x_pad)
        ax.set_ylim(micro_min - y_pad, micro_max + y_pad)

    ax.set_xlabel("Macroclimate temperature (°C)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Microclimate temperature (°C)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(
        f"Site {site_id} | Event {event_id}\n"
        f"{drought_start.date()} to {drought_end.date()}",
        fontsize=TITLE_FONTSIZE,
        pad=10
    )
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.legend(loc='lower right', fontsize=LEGEND_FONTSIZE, frameon=False)

    annotation_lines = [
        method_note,
        f"Drought days={event_row['duration_days']}",
    ]
    if is_long_drought and pd.notna(event_row['delta_beta_pct']):
        annotation_lines.append(f"Δβ={event_row['delta_beta_pct']:+.2f}%")
    annotation_lines.extend(summary_lines)
    ax.text(
        0.02, 0.98,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=8.8
    )

    plt.tight_layout()
    out_name = (
        f"站点_{sanitize_filename(site_id)}_事件_{sanitize_filename(event_id)}_"
        f"{drought_start.date()}_OLS斜率诊断图.png"
    )
    out_path = os.path.join(output_dir, out_name)
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()


def scan_contiguous_nondrought_side(site_daily_spi, start_day, side):
    """
    扫描干旱事件一侧、紧邻事件的连续非干旱日序列。

    本函数用于“干旱期 vs 综合非干旱期”两期配对分析，遵循用户确认的规则：
      1. 只取与目标干旱事件相邻的一段连续非干旱日；
      2. 搜索过程中不跳过干旱日；
      3. 一旦遇到干旱日或缺失SPI记录，就停止该侧搜索；
      4. 若该侧不足，再交由另一侧补足，以实现“前后尽量平衡”的总非干旱天数。

    参数：
      site_daily_spi : 单站点逐日SPI表，必须包含 date / is_drought_day
      start_day      : 该侧最靠近干旱事件、允许纳入非干旱期的首个日期
      side           : 'pre' 或 'post'

    返回：
      dict，包含：
        side / available_days / available_count / actual_start / actual_end
        search_start / search_end / stop_reason / note
    """
    start_day = pd.Timestamp(start_day).normalize()

    if site_daily_spi.empty:
        return {
            'side': side,
            'available_days': [],
            'available_count': 0,
            'actual_start': None,
            'actual_end': None,
            'search_start': None,
            'search_end': None,
            'stop_reason': 'no_spi_records',
            'note': "逐日SPI表无该站点记录",
        }

    spi_map = site_daily_spi.drop_duplicates(subset=['date']).set_index('date').sort_index()
    min_date = spi_map.index.min()
    max_date = spi_map.index.max()

    if side == 'pre':
        step = -1
    elif side == 'post':
        step = 1
    else:
        raise ValueError("side 必须为 'pre' 或 'post'")

    current_day = start_day
    available_days = []
    search_days = []
    stop_reason = 'reached_data_boundary'

    while True:
        if current_day < min_date or current_day > max_date:
            stop_reason = 'reached_data_boundary'
            break

        search_days.append(current_day)
        if current_day not in spi_map.index:
            stop_reason = 'missing_spi_record'
            break

        row = spi_map.loc[current_day]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        if bool(row['is_drought_day']):
            stop_reason = 'encountered_drought_day'
            break

        available_days.append(current_day)
        current_day = current_day + pd.Timedelta(days=step)

    if side == 'pre':
        actual_days = sorted(available_days)
    else:
        actual_days = available_days.copy()

    actual_start = actual_days[0] if actual_days else None
    actual_end = actual_days[-1] if actual_days else None
    search_start = min(search_days) if search_days else None
    search_end = max(search_days) if search_days else None

    if actual_days:
        note = (
            f"获得连续非干旱日 {len(actual_days)} 天；"
            f"停止原因：{stop_reason}"
        )
    else:
        note = f"该侧未获得可用非干旱日；停止原因：{stop_reason}"

    return {
        'side': side,
        'available_days': actual_days,
        'available_count': len(actual_days),
        'actual_start': actual_start,
        'actual_end': actual_end,
        'search_start': search_start,
        'search_end': search_end,
        'stop_reason': stop_reason,
        'note': note,
    }


def get_balanced_nondrought_period(site_daily_spi, d_start, d_end, gap_days,
                                   target_days_override=None):
    """
    构建“干旱期 vs 综合非干旱期”的等长非干旱配对窗口。

    规则说明：
      1. 综合非干旱期总天数默认为干旱期天数；
         若传入 target_days_override，则改用指定目标天数；
      2. 优先按“前后尽量平衡”分配目标天数；
      3. 每一侧只允许取与干旱事件相邻的一段连续非干旱日；
      4. 一旦该侧继续向外遇到干旱日或缺失SPI记录，就立即停止该侧；
      5. 若一侧不足，允许另一侧继续补足，但仍必须保持该侧自身连续。

    这样得到的综合非干旱期，能够表达“与事件相邻、且不被其他干旱污染”的
    前后非干旱背景，同时又不强行要求前后各自必须等长。
    """
    d_start = pd.Timestamp(d_start).normalize()
    d_end = pd.Timestamp(d_end).normalize()
    raw_duration = len(pd.date_range(d_start, d_end, freq='D'))
    target_days = target_days_override if target_days_override is not None else raw_duration

    pre_anchor = d_start - pd.Timedelta(days=gap_days + 1)
    post_anchor = d_end + pd.Timedelta(days=gap_days + 1)

    pre_scan = scan_contiguous_nondrought_side(site_daily_spi, pre_anchor, side='pre')
    post_scan = scan_contiguous_nondrought_side(site_daily_spi, post_anchor, side='post')

    pre_available = sorted(pre_scan['available_days'])
    post_available = sorted(post_scan['available_days'])

    pre_target = target_days // 2
    post_target = target_days - pre_target
    pre_take = min(len(pre_available), pre_target)
    post_take = min(len(post_available), post_target)
    remaining = target_days - pre_take - post_take

    while remaining > 0:
        pre_can_extend = pre_take < len(pre_available)
        post_can_extend = post_take < len(post_available)
        if not pre_can_extend and not post_can_extend:
            break

        if pre_can_extend and post_can_extend:
            if pre_take <= post_take:
                pre_take += 1
            else:
                post_take += 1
        elif pre_can_extend:
            pre_take += 1
        else:
            post_take += 1
        remaining -= 1

    pre_used = pre_available[-pre_take:] if pre_take > 0 else []
    post_used = post_available[:post_take] if post_take > 0 else []
    used_days = sorted(pre_used + post_used)
    actual_days = len(used_days)
    full_length = actual_days == target_days

    if full_length:
        note = (
            f"综合非干旱期已补足等长 {actual_days}/{target_days} 天；"
            f"前侧 {len(pre_used)} 天，后侧 {len(post_used)} 天"
        )
    else:
        note = (
            f"综合非干旱期未补足，仅获得 {actual_days}/{target_days} 天；"
            f"前侧 {len(pre_used)} 天，后侧 {len(post_used)} 天"
        )

    return {
        'target_days': target_days,
        'actual_days': actual_days,
        'shortfall_days': target_days - actual_days,
        'full_length': full_length,
        'truncated': not full_length,
        'pre_target_days': pre_target,
        'post_target_days': post_target,
        'pre_used_days': pre_used,
        'post_used_days': post_used,
        'used_days': used_days,
        'pre_used_count': len(pre_used),
        'post_used_count': len(post_used),
        'pre_start': pre_used[0] if pre_used else None,
        'pre_end': pre_used[-1] if pre_used else None,
        'post_start': post_used[0] if post_used else None,
        'post_end': post_used[-1] if post_used else None,
        'pre_search_start': pre_scan['search_start'],
        'pre_search_end': pre_scan['search_end'],
        'post_search_start': post_scan['search_start'],
        'post_search_end': post_scan['search_end'],
        'pre_stop_reason': pre_scan['stop_reason'],
        'post_stop_reason': post_scan['stop_reason'],
        'pre_note': pre_scan['note'],
        'post_note': post_scan['note'],
        'note': note,
    }


def build_site_event_long(beta_table):
    """
    将事件级β表展开为站点事件长格式。

    输出列：
      site_id, event_id, drought_start, event_order, event_label, period,
      beta, period_start, period_end, period_days
    用于站点级配对连线图和站点级柱状图。
    """
    rows = []
    for site_id, site_df in beta_table.sort_values(['site_id', 'drought_start']).groupby('site_id'):
        site_df = site_df.reset_index(drop=True)
        for idx, row in site_df.iterrows():
            event_order = idx + 1
            event_label = f"Event {event_order}\n{row['drought_start']}"
            pre_days = (
                (pd.to_datetime(row['pre_end']) - pd.to_datetime(row['pre_start'])).days + 1
                if pd.notna(row['pre_start']) and pd.notna(row['pre_end']) else np.nan
            )
            drought_days = row['duration_days']
            post_days = (
                (pd.to_datetime(row['post_end']) - pd.to_datetime(row['post_start'])).days + 1
                if pd.notna(row['post_start']) and pd.notna(row['post_end']) else np.nan
            )
            rows.extend([
                {
                    'site_id': site_id,
                    'event_id': row['event_id'],
                    'drought_start': row['drought_start'],
                    'event_order': event_order,
                    'event_label': event_label,
                    'period': 'Pre',
                    'beta': row['beta_pre'],
                    'period_start': row['pre_start'],
                    'period_end': row['pre_end'],
                    'period_days': pre_days,
                },
                {
                    'site_id': site_id,
                    'event_id': row['event_id'],
                    'drought_start': row['drought_start'],
                    'event_order': event_order,
                    'event_label': event_label,
                    'period': 'Drought',
                    'beta': row['beta_drought'],
                    'period_start': row['drought_start'],
                    'period_end': row['drought_end'],
                    'period_days': drought_days,
                },
                {
                    'site_id': site_id,
                    'event_id': row['event_id'],
                    'drought_start': row['drought_start'],
                    'event_order': event_order,
                    'event_label': event_label,
                    'period': 'Post',
                    'beta': row['beta_post'],
                    'period_start': row['post_start'],
                    'period_end': row['post_end'],
                    'period_days': post_days,
                },
            ])
    return pd.DataFrame(rows)


def build_site_event_nondrought_long(beta_table):
    """
    将“干旱 vs 综合非干旱”事件级结果展开为长格式。

    输出列用于：
      1. 站点级两期配对连线图

    非干旱期由前后两段拼接，因此会额外输出 pre/post 两段的日期与天数信息，
    便于在图中直接检查每次事件的配对来源。
    """
    rows = []
    for site_id, site_df in beta_table.sort_values(['site_id', 'drought_start']).groupby('site_id'):
        site_df = site_df.reset_index(drop=True)
        for idx, row in site_df.iterrows():
            event_order = idx + 1
            event_label = f"Event {event_order}\n{row['drought_start']}"
            rows.extend([
                {
                    'site_id': site_id,
                    'event_id': row['event_id'],
                    'drought_start': row['drought_start'],
                    'event_order': event_order,
                    'event_label': event_label,
                    'period': 'Non-drought',
                    'beta': row['beta_nondrought'],
                    'period_start': row['nondrought_pre_start'],
                    'period_end': row['nondrought_post_end'],
                    'period_days': row['nondrought_actual_days'],
                    'pre_start': row['nondrought_pre_start'],
                    'pre_end': row['nondrought_pre_end'],
                    'pre_days': row['nondrought_pre_days'],
                    'post_start': row['nondrought_post_start'],
                    'post_end': row['nondrought_post_end'],
                    'post_days': row['nondrought_post_days'],
                },
                {
                    'site_id': site_id,
                    'event_id': row['event_id'],
                    'drought_start': row['drought_start'],
                    'event_order': event_order,
                    'event_label': event_label,
                    'period': 'Drought',
                    'beta': row['beta_drought'],
                    'period_start': row['drought_start'],
                    'period_end': row['drought_end'],
                    'period_days': row['duration_days'],
                    'pre_start': None,
                    'pre_end': None,
                    'pre_days': np.nan,
                    'post_start': None,
                    'post_end': None,
                    'post_days': np.nan,
                },
            ])
    return pd.DataFrame(rows)


# ============================================================
# Step 1：加载TOMST数据，15min → 逐小时均值
# ============================================================

def load_tomst(tomst_dir, datetime_col, temp_col, time_fmt):
    """
    读取27个TOMST传感器CSV文件，将15分钟频率的微气候温度数据
    聚合为逐小时均值，作为林下微气候温度（T_micro）序列。

    【为什么降采样到1小时？】
        ERA5数据的时间分辨率为逐小时，TOMST为15分钟，两者需要统一
        时间分辨率才能做 T_micro ~ T_macro 的配对OLS回归。
        逐小时是两者的最粗公共分辨率。

    【文件命名规则】
        只处理文件名为纯数字的CSV（如95332217.csv），其他文件（如README.csv）
        会被自动跳过，避免误读。

    【质量控制】
        1. 物理范围过滤：-20℃ ~ 60℃ 以外的值设为NaN（剔除传感器异常）
        2. 最少读数要求：每小时内至少2个有效15min读数才输出均值，
           否则该小时设为NaN（避免单个异常读数主导小时均值）

    参数：
        tomst_dir    : TOMST CSV文件夹路径
        datetime_col : UTC时间列名（本研究使用 data_time 列）
        temp_col     : 温度列名（本研究使用 T3_15 列，即15cm气温）
        time_fmt     : 时间字符串解析格式（如 "%Y.%m.%d %H:%M"）
    返回：
        DataFrame，列：site_id（站点编号） | datetime（UTC，无时区）| T_micro（℃）
    """
    print("=" * 65)
    print("Step 1: 加载TOMST微气候数据（15min → 1h均值）")
    print(f"  数据文件夹：{tomst_dir}")
    print(f"  使用列：{temp_col}（林下{temp_col}温度）")

    # 只读取文件名为纯数字的CSV（站点编号命名）
    csv_files = [
        f for f in glob.glob(os.path.join(tomst_dir, "*.csv"))
        if os.path.splitext(os.path.basename(f))[0].isdigit()
    ]
    if not csv_files:
        raise FileNotFoundError(
            f"未在 {tomst_dir} 找到数字命名的CSV文件。\n"
            f"请检查：(1)路径是否正确；(2)文件名是否为纯数字（如95332217.csv）"
        )

    records = []
    for f in tqdm(
        sorted(csv_files),
        desc="Step 1 TOMST",
        colour=PROGRESS_COLOURS["tomst"],
        dynamic_ncols=TQDM_NCOLS_DYNAMIC,
        leave=TQDM_LEAVE,
        bar_format=TQDM_BAR_FORMAT,
    ):
        sid = os.path.splitext(os.path.basename(f))[0]  # 文件名即站点编号
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  [警告] 读取 {f} 失败，已跳过: {e}")
            continue

        # 列名检查：提前报错，避免后续混乱
        if datetime_col not in df.columns:
            raise KeyError(
                f"文件 {os.path.basename(f)} 中找不到时间列 '{datetime_col}'。\n"
                f"实际列名：{list(df.columns)}\n"
                f"请修改配置区的 TOMST_DATETIME_COL"
            )
        if temp_col not in df.columns:
            raise KeyError(
                f"文件 {os.path.basename(f)} 中找不到温度列 '{temp_col}'。\n"
                f"实际列名：{list(df.columns)}\n"
                f"请修改配置区的 TOMST_TEMP_COL"
            )

        # 解析时间和温度
        df['datetime']     = pd.to_datetime(df[datetime_col], format=time_fmt, errors='coerce')
        df['T_micro_raw']  = pd.to_numeric(df[temp_col], errors='coerce')
        df['site_id']      = sid

        # 物理范围过滤：去除明显异常值
        invalid_count = ((df['T_micro_raw'] < -20) | (df['T_micro_raw'] > 60)).sum()
        if invalid_count > 0:
            print(f"  [提示] {os.path.basename(f)}: 去除 {invalid_count} 个超出范围的温度值")
        df.loc[(df['T_micro_raw'] < -20) | (df['T_micro_raw'] > 60), 'T_micro_raw'] = np.nan

        # 去掉时间解析失败的行
        df = df.dropna(subset=['datetime'])

        # 降采样到1小时均值
        # 用 lambda 确保每小时至少2个有效读数才输出，否则为NaN
        df = df.set_index('datetime')
        hourly = (
            df.groupby('site_id')['T_micro_raw']
            .resample('1h')
            .agg(lambda x: x.mean() if x.notna().sum() >= 2 else np.nan)
            .reset_index()
        )
        hourly.columns = ['site_id', 'datetime', 'T_micro']
        records.append(hourly)

    micro = pd.concat(records, ignore_index=True)
    # 去除时区信息（pandas时区处理统一为无时区naive，方便后续merge）
    micro['datetime'] = micro['datetime'].dt.tz_localize(None)

    n_sites = micro['site_id'].nunique()
    n_rows  = len(micro)
    print(f"  ✓ 加载完成：{n_sites} 个站点，{n_rows:,} 条逐小时记录")
    print(f"  时间范围：{micro['datetime'].min()} → {micro['datetime'].max()}")
    return micro


# ============================================================
# Step 2：从ERA5逐小时TIF提取宏气候温度
# ============================================================

def extract_era5(tif_dir, sites_df, unit_is_kelvin):
    """
    遍历ERA5的8760张逐小时TIF文件，对每个站点坐标提取最近像元的2m气温，
    作为宏气候参照温度（T_macro）序列。

    【为什么用ERA5作宏气候？】
        ERA5-Land 是高时空分辨率的再分析数据，被广泛用作林外/宏观气候基准。
        参考：Ma et al. (2025) Agricultural and Forest Meteorology
              Zellweger et al. (2020) Global Change Biology

    【坐标提取方法】
        使用 rasterio.sample() 对每张TIF按站点经纬度坐标取值，
        自动选取最近像元（不进行双线性插值），保证与原始数据一致。

    【注意：TIF文件命名格式】
        脚本用正则表达式 "YYYY年MM月DD日HH时" 解析文件名中的时间。
        若你的文件名格式不同，请修改 parse_era5_tif_datetime() 函数中的正则。

    参数：
        tif_dir       : ERA5逐小时TIF文件夹路径
        sites_df      : 样地坐标DataFrame（含 site_id, lon, lat 列）
        unit_is_kelvin: TIF数值单位是否为开尔文（True=K，False=℃）
                        ERA5原始单位为K，若已预处理为℃则设False
    返回：
        DataFrame，列：site_id | datetime（UTC，无时区）| T_macro（℃）
    """
    print("\nStep 2: 从ERA5逐小时TIF提取宏气候温度（共8760张）")
    print(f"  TIF文件夹：{tif_dir}")
    print(f"  温度单位：{'开尔文K（自动转℃）' if unit_is_kelvin else '摄氏度℃（直接使用）'}")

    # 扫描所有TIF文件并解析时间戳
    tif_files = sorted(glob.glob(os.path.join(tif_dir, "*.tif")))
    tif_meta  = [(f, parse_era5_tif_datetime(f)) for f in tif_files]
    # 过滤掉无法解析时间的文件（parse返回pd.NaT）
    tif_meta  = [(f, dt) for f, dt in tif_meta if pd.notna(dt)]

    if not tif_meta:
        raise FileNotFoundError(
            f"在 {tif_dir} 中未找到可解析时间的TIF文件。\n"
            f"请检查：(1)文件名是否包含 '2025年01月01日00时' 格式的时间；\n"
            f"        (2)若格式不同，请修改 parse_era5_tif_datetime() 函数中的正则表达式"
        )

    print(f"  识别到 {len(tif_meta)} 张有效TIF（预期8760张）")
    if len(tif_meta) != 8760:
        print(f"  [提示] 识别到 {len(tif_meta)} 张，与预期8760不符，"
              f"可能有文件缺失或命名不符规范，请检查")

    # 准备坐标列表，顺序与 sites_df 行顺序一致
    coords   = list(zip(sites_df['lon'].values, sites_df['lat'].values))
    site_ids = sites_df['site_id'].values

    macro_records = []
    for f, dt in tqdm(
        tif_meta,
        desc="Step 2 ERA5",
        colour=PROGRESS_COLOURS["era5"],
        dynamic_ncols=TQDM_NCOLS_DYNAMIC,
        leave=TQDM_LEAVE,
        bar_format=TQDM_BAR_FORMAT,
    ):
        with rasterio.open(f) as src:
            nodata = src.nodata
            # rasterio.sample() 返回每个坐标点的像元值（列表的列表）
            vals = [v[0] for v in src.sample(coords)]

        for sid, val in zip(site_ids, vals):
            # 处理NoData值（通常为极大或极小数）
            if nodata is not None and np.isclose(float(val), nodata, atol=1e-3):
                temp = np.nan
            else:
                temp = float(val) - 273.15 if unit_is_kelvin else float(val)
            macro_records.append({'site_id': sid, 'datetime': dt, 'T_macro': temp})

    macro = pd.DataFrame(macro_records)
    macro['datetime'] = macro['datetime'].dt.tz_localize(None)
    valid_macro = macro['T_macro'].dropna()
    if not valid_macro.empty:
        q01, q50, q99 = valid_macro.quantile([0.01, 0.50, 0.99])
        if q50 < -80 or q50 > 60 or q01 < -100 or q99 > 80:
            print(
                "  [强警告] ERA5宏气候温度分布异常："
                f"1%={q01:.2f}, 中位数={q50:.2f}, 99%={q99:.2f}。"
                "请检查 ERA5_UNIT_K 或上游TIF单位。"
            )
    print(f"  ✓ 提取完成：{macro['site_id'].nunique()} 个站点，{len(macro):,} 条记录")
    return macro


# ============================================================
# Step 3：时间对齐合并（T_micro + T_macro → 主分析表）
# ============================================================

def merge_datasets(micro, macro):
    """
    将TOMST微气候数据（T_micro）与ERA5宏气候数据（T_macro）按
    site_id + datetime（UTC）进行内连接合并，得到逐小时主分析表。

    【时间对齐说明】
        TOMST 的 data_time 列是UTC时间，ERA5 TIF的时间也是UTC，
        两者时区一致，可直接按时间戳对齐合并。
        注意：data_time8（UTC+8）不用于此处匹配。

    【内连接 vs 外连接】
        使用内连接（how='inner'）：只保留两个数据集都有记录的时间点。
        这意味着：若ERA5某小时TIF文件缺失，则该小时全部27个站点的记录被丢弃。
        若TOMST某站点某小时缺数据，则该站点该小时被丢弃（不影响其他站点）。

    参数：
        micro: Step 1 输出的逐小时T_micro DataFrame
        macro: Step 2 输出的逐小时T_macro DataFrame
    返回：
        合并后的DataFrame，列：site_id | datetime | T_micro | T_macro
    """
    print("\nStep 3: 按 site_id + datetime 合并微气候与宏气候数据...")

    merged = pd.merge(micro, macro, on=['site_id', 'datetime'], how='inner')
    before = len(merged)
    # 去除任意温度列含NaN的行（这些行无法参与OLS回归）
    merged = merged.dropna(subset=['T_micro', 'T_macro'])
    removed = before - len(merged)

    print(f"  ✓ 合并完成：{merged['site_id'].nunique()} 个站点，"
          f"{len(merged):,} 条有效记录")
    if removed > 0:
        print(f"  [提示] 去除 {removed} 条含NaN的记录")
    return merged


# ============================================================
# Step 4：等长配对对照期设计 + 重叠检查 + 计算三组β
# ============================================================

def compute_paired_betas(merged, drought_df, daily_spi_df, sites_df, gap_days=CONTROL_GAP_DAYS):
    """
    核心分析步骤：对每个站点的每次极端干旱事件，分别计算：
        β_drought ：干旱期内的气候缓冲指数
        β_pre     ：前置等长对照期的气候缓冲指数
        β_post    ：后置等长对照期的气候缓冲指数

    【等长配对设计原则】
        对照期长度 = 干旱期长度，时间紧邻干旱事件前后，间隔 gap_days 天。
        这样设计可以：
        (1) 保证干旱期和对照期的样本量相等，使两组OLS斜率的估计精度可比
        (2) 时间邻近干旱事件，最大程度控制季节背景变化（避免夏季vs冬季的混淆）
        参考：Zellweger et al. (2020) Global Change Biology
              De Frenne et al. (2021) Nature Ecology & Evolution

    【时间段示意图】
        gap=1天，干旱期18天：
        ┌──────────────────┐ ┌──┐ ┌──────────────────┐ ┌──┐ ┌──────────────────┐
        │  前置对照期18天   │ │缓│ │   极端干旱期18天  │ │缓│ │  后置对照期18天  │
        └──────────────────┘ └──┘ └──────────────────┘ └──┘ └──────────────────┘

    【重叠检查机制】
        前/后置对照期先生成等长原始窗口，再用逐日SPI表从靠近目标事件的一侧
        扫描连续非干旱日。一旦遇到 Is_Drought_Day=True 或缺失SPI记录，就截断；
        截断后不向更远日期补足天数。这样保证对照期尽量接近目标干旱事件，
        同时避免其他干旱日污染对照期。

    【站点聚合策略】
        每个站点可能有多次Extreme事件，Step 5 会对每个站点取中位数β。
        这确保每站点只贡献一个β值，满足Wilcoxon配对检验的独立性假设。

    参数：
        merged    : Step 3 输出的合并主表
        drought_df: 干旱事件长表
        daily_spi_df: 逐日SPI序列表，仅用于前/后对照期过滤
        sites_df   : 站点坐标表，用于生成站点事件覆盖记录
        gap_days  : 对照期与干旱事件之间的缓冲天数（默认1天，见配置区说明）
    返回：
        (beta_table, control_quality, site_coverage)
    """
    print("\nStep 4: 计算极端干旱事件 β（短事件=等长配对；长事件=相对偏差法）...")
    print(f"  筛选条件：Drought_Level_Code = {TARGET_DROUGHT_LEVEL_CODE}")
    print(f"  对照期设计：等长配对，缓冲间隔 {gap_days} 天")
    print(f"  长干旱阈值：>{LONG_DROUGHT_THRESH} 天 → 使用事件级baseline与Δβ")
    print(f"  OLS最少有效数据点：{MIN_HOURS_FOR_OLS} 小时")

    target_events = filter_target_events(drought_df, TARGET_DROUGHT_LEVEL_CODE)

    n_events = len(target_events)
    n_sites  = target_events['site_id'].nunique()
    print(f"  共 {n_sites} 个站点，{n_events} 次目标等级干旱事件")

    results = []
    quality_records = []
    target_event_counts = target_events.groupby('site_id').size().to_dict()
    all_event_counts = drought_df.groupby('site_id').size().to_dict()

    coverage_records = []
    for sid in sites_df['site_id']:
        has_target = target_event_counts.get(sid, 0) > 0
        has_hourly = sid in set(merged['site_id'].unique())
        if not has_target:
            reason = "无目标等级干旱事件"
        elif not has_hourly:
            reason = "温度合并数据中无该站点记录"
        else:
            reason = ""
        coverage_records.append({
            '站点编号': sid,
            '是否有目标等级事件': has_target,
            '目标等级事件数': target_event_counts.get(sid, 0),
            '全部干旱事件数': all_event_counts.get(sid, 0),
            '是否有逐小时温度合并数据': has_hourly,
            '是否进入事件计算': has_target and has_hourly,
            '未进入原因': reason,
        })

    for _, ev in tqdm(
        target_events.iterrows(),
        total=len(target_events),
        desc="Step 4 β计算",
        colour=PROGRESS_COLOURS["beta"],
        dynamic_ncols=TQDM_NCOLS_DYNAMIC,
        leave=TQDM_LEAVE,
        bar_format=TQDM_BAR_FORMAT,
    ):
        site_id = ev['site_id']
        site_daily_spi = daily_spi_df[daily_spi_df['site_id'] == site_id].copy()

        # 该站点的逐小时合并数据，按时间索引便于切片
        site_data = merged[merged['site_id'] == site_id].copy()
        site_data = site_data.set_index('datetime').sort_index()

        ev_id    = ev['Event_ID']
        d_start  = pd.Timestamp(ev['Start_Date'])
        d_end    = pd.Timestamp(ev['End_Date'])
        # 持续天数（含首尾，如1月1日~1月3日为3天）
        drought_duration = (d_end - d_start).days + 1
        is_long_drought = drought_duration > LONG_DROUGHT_THRESH
        control_duration = min(drought_duration, MAX_CONTROL_DAYS)

        # ════════════════════════════════════════════════
        # 4a. 干旱期β
        # 时间范围：[d_start 00:00, d_end 23:00]（含末日全天24小时）
        # ════════════════════════════════════════════════
        drought_slice = site_data[
            (site_data.index >= d_start) &
            (site_data.index <= d_end + pd.Timedelta(hours=23))
        ]
        beta_d = run_ols(drought_slice.index,
                         drought_slice['T_micro'],
                         drought_slice['T_macro'])

        if not is_long_drought:
            # ════════════════════════════════════════════════
            # 4b. 前置对照期β（仅短事件）
            # 初始范围：[d_start - gap - control_duration, d_start - gap - 1]
            # ════════════════════════════════════════════════
            pre_end_raw   = d_start - pd.Timedelta(days=gap_days + 1)
            pre_start_raw = pre_end_raw - pd.Timedelta(days=control_duration - 1)

            pre_info = get_control_period_by_daily_spi(site_daily_spi, pre_start_raw, pre_end_raw, side='pre')
            pre_start, pre_end = pre_info['actual_start'], pre_info['actual_end']

            if pre_start is not None:
                pre_slice = site_data[
                    (site_data.index >= pre_start) &
                    (site_data.index <= pre_end + pd.Timedelta(hours=23))
                ]
                beta_pre = run_ols(pre_slice.index,
                                   pre_slice['T_micro'],
                                   pre_slice['T_macro'])
            else:
                beta_pre = dict(beta=np.nan, alpha=np.nan, r2=np.nan, p_value=np.nan, n=0)

            # ════════════════════════════════════════════════
            # 4c. 后置对照期β（仅短事件）
            # 初始范围：[d_end + gap + 1, d_end + gap + control_duration]
            # ════════════════════════════════════════════════
            post_start_raw = d_end + pd.Timedelta(days=gap_days + 1)
            post_end_raw   = post_start_raw + pd.Timedelta(days=control_duration - 1)

            post_info = get_control_period_by_daily_spi(site_daily_spi, post_start_raw, post_end_raw, side='post')
            post_start, post_end = post_info['actual_start'], post_info['actual_end']

            if post_start is not None:
                post_slice = site_data[
                    (site_data.index >= post_start) &
                    (site_data.index <= post_end + pd.Timedelta(hours=23))
                ]
                beta_post = run_ols(post_slice.index,
                                    post_slice['T_micro'],
                                    post_slice['T_macro'])
            else:
                beta_post = dict(beta=np.nan, alpha=np.nan, r2=np.nan, p_value=np.nan, n=0)

            # ════════════════════════════════════════════════
            # 4d. 综合非干旱期β（仅短事件）
            # 目标总天数 = 对照期目标长度 = control_duration
            # ════════════════════════════════════════════════
            nondrought_info = get_balanced_nondrought_period(
                site_daily_spi, d_start, d_end, gap_days, target_days_override=control_duration
            )
            nondrought_days = nondrought_info['used_days']
            if nondrought_days:
                nondrought_hours = site_data[site_data.index.normalize().isin(nondrought_days)]
                beta_nondrought = run_ols(
                    nondrought_hours.index,
                    nondrought_hours['T_micro'],
                    nondrought_hours['T_macro']
                )
            else:
                beta_nondrought = dict(beta=np.nan, alpha=np.nan, r2=np.nan, p_value=np.nan, n=0)

            baseline_info = {
                'beta_baseline': np.nan,
                'r2_baseline': np.nan,
                'n_baseline': 0,
                'baseline_days': [],
                'baseline_day_count': 0,
                'allowed_months': [],
                'note': "短事件主方法为等长配对，不计算事件级baseline",
            }
            delta_beta_pct = np.nan
        else:
            # ════════════════════════════════════════════════
            # 长事件：不再强行计算等长前/后/综合非干旱对照
            # 改用“该事件的同季节非干旱baseline β”作为参考，再计算Δβ
            # ════════════════════════════════════════════════
            pre_end_raw = pre_start_raw = None
            post_start_raw = post_end_raw = None
            pre_start = pre_end = None
            post_start = post_end = None
            beta_pre = dict(beta=np.nan, alpha=np.nan, r2=np.nan, p_value=np.nan, n=0)
            beta_post = dict(beta=np.nan, alpha=np.nan, r2=np.nan, p_value=np.nan, n=0)
            beta_nondrought = dict(beta=np.nan, alpha=np.nan, r2=np.nan, p_value=np.nan, n=0)
            pre_info = {
                'actual_start': None, 'actual_end': None, 'raw_days': control_duration,
                'actual_days': 0, 'shortfall_days': control_duration, 'full_length': False,
                'truncated': True, 'note': "长事件改用相对偏差法，不计算前置等长对照期",
                'search_start': None, 'search_end': None, 'skipped_drought_days': 0,
                'skipped_missing_days': 0, 'used_days': [],
            }
            post_info = {
                'actual_start': None, 'actual_end': None, 'raw_days': control_duration,
                'actual_days': 0, 'shortfall_days': control_duration, 'full_length': False,
                'truncated': True, 'note': "长事件改用相对偏差法，不计算后置等长对照期",
                'search_start': None, 'search_end': None, 'skipped_drought_days': 0,
                'skipped_missing_days': 0, 'used_days': [],
            }
            nondrought_info = {
                'target_days': control_duration, 'actual_days': 0,
                'shortfall_days': control_duration, 'full_length': False, 'truncated': True,
                'pre_target_days': 0, 'post_target_days': 0,
                'pre_used_days': [], 'post_used_days': [], 'used_days': [],
                'pre_used_count': 0, 'post_used_count': 0,
                'pre_start': None, 'pre_end': None, 'post_start': None, 'post_end': None,
                'pre_search_start': None, 'pre_search_end': None,
                'post_search_start': None, 'post_search_end': None,
                'pre_stop_reason': 'long_drought_use_baseline',
                'post_stop_reason': 'long_drought_use_baseline',
                'pre_note': "长事件改用相对偏差法，不计算综合非干旱前段",
                'post_note': "长事件改用相对偏差法，不计算综合非干旱后段",
                'note': "长事件改用事件级baseline，不计算综合非干旱等长对照",
            }
            baseline_info = compute_event_baseline_beta(
                site_data, site_daily_spi, d_start, max_offset_months=MAX_MONTH_OFFSET
            )
            beta_baseline = baseline_info['beta_baseline']
            if pd.notna(beta_baseline) and beta_baseline != 0:
                delta_beta_pct = (beta_d['beta'] - beta_baseline) / abs(beta_baseline) * 100
            else:
                delta_beta_pct = np.nan

        if not is_long_drought:
            beta_baseline = baseline_info['beta_baseline']

        # 汇总本次事件的所有结果
        results.append({
            'site_id'        : site_id,
            'event_id'       : ev_id,
            'drought_start'  : d_start.date(),
            'drought_end'    : d_end.date(),
            'duration_days'  : drought_duration,
            'control_duration_days': control_duration,
            'control_was_capped': control_duration < drought_duration,
            'is_long_drought': is_long_drought,
            'severity'       : ev['Severity'],
            # ── 干旱期 ──
            'beta_drought'   : beta_d['beta'],
            'r2_drought'     : beta_d['r2'],
            'n_drought'      : beta_d['n'],
            # ── 前置对照期 ──
            'pre_start'      : pre_start.date() if pre_start else None,
            'pre_end'        : pre_end.date()   if pre_end   else None,
            'beta_pre'       : beta_pre['beta'],
            'r2_pre'         : beta_pre['r2'],
            'n_pre'          : beta_pre['n'],
            'pre_truncated'  : pre_info['truncated'],   # True=被截断，需检查
            'pre_full_length': pre_info['full_length'],
            # ── 后置对照期 ──
            'post_start'     : post_start.date() if post_start else None,
            'post_end'       : post_end.date()   if post_end   else None,
            'beta_post'      : beta_post['beta'],
            'r2_post'        : beta_post['r2'],
            'n_post'         : beta_post['n'],
            'post_truncated' : post_info['truncated'],  # True=被截断，需检查
            'post_full_length': post_info['full_length'],
            # ── 综合非干旱期（前后拼接、总天数等长） ──
            'beta_nondrought': beta_nondrought['beta'],
            'r2_nondrought': beta_nondrought['r2'],
            'n_nondrought': beta_nondrought['n'],
            'nondrought_target_days': nondrought_info['target_days'],
            'nondrought_actual_days': nondrought_info['actual_days'],
            'nondrought_shortfall_days': nondrought_info['shortfall_days'],
            'nondrought_full_length': nondrought_info['full_length'],
            'nondrought_truncated': nondrought_info['truncated'],
            'nondrought_pre_target_days': nondrought_info['pre_target_days'],
            'nondrought_post_target_days': nondrought_info['post_target_days'],
            'nondrought_pre_days': nondrought_info['pre_used_count'],
            'nondrought_post_days': nondrought_info['post_used_count'],
            'nondrought_pre_start': nondrought_info['pre_start'].date() if nondrought_info['pre_start'] is not None else None,
            'nondrought_pre_end': nondrought_info['pre_end'].date() if nondrought_info['pre_end'] is not None else None,
            'nondrought_post_start': nondrought_info['post_start'].date() if nondrought_info['post_start'] is not None else None,
            'nondrought_post_end': nondrought_info['post_end'].date() if nondrought_info['post_end'] is not None else None,
            # ── 长事件相对偏差法（事件级baseline） ──
            'beta_baseline': baseline_info['beta_baseline'],
            'r2_baseline': baseline_info['r2_baseline'],
            'n_baseline': baseline_info['n_baseline'],
            'baseline_day_count': baseline_info['baseline_day_count'],
            'baseline_allowed_months': ",".join(map(str, baseline_info['allowed_months'])) if baseline_info['allowed_months'] else "",
            'delta_beta_pct': delta_beta_pct,
        })

        quality_records.append({
            '站点编号': site_id,
            '事件编号': ev_id,
            '干旱开始日期': d_start.date(),
            '干旱结束日期': d_end.date(),
            '干旱期天数': drought_duration,
            '对照目标天数': control_duration,
            '对照是否被上限截断': control_duration < drought_duration,
            '是否为长干旱事件': is_long_drought,
            '干旱期有效小时数': beta_d['n'],
            '前置原始开始日期': pre_start_raw.date() if pre_start_raw is not None else None,
            '前置原始结束日期': pre_end_raw.date() if pre_end_raw is not None else None,
            '前置搜索开始日期': pre_info['search_start'].date() if pre_info['search_start'] is not None else None,
            '前置搜索结束日期': pre_info['search_end'].date() if pre_info['search_end'] is not None else None,
            '前置原始天数': pre_info['raw_days'],
            '前置实际开始日期': pre_start.date() if pre_start is not None else None,
            '前置实际结束日期': pre_end.date() if pre_end is not None else None,
            '前置连续非干旱天数': pre_info['actual_days'],
            '前置不足天数': pre_info['shortfall_days'],
            '前置跳过干旱日天数': pre_info['skipped_drought_days'],
            '前置跳过缺失SPI天数': pre_info['skipped_missing_days'],
            '前置是否等长': pre_info['full_length'],
            '前置是否可计算β': beta_pre['n'] >= MIN_HOURS_FOR_OLS,
            '前置有效小时数': beta_pre['n'],
            '前置截断说明': pre_info['note'],
            '后置原始开始日期': post_start_raw.date() if post_start_raw is not None else None,
            '后置原始结束日期': post_end_raw.date() if post_end_raw is not None else None,
            '后置搜索开始日期': post_info['search_start'].date() if post_info['search_start'] is not None else None,
            '后置搜索结束日期': post_info['search_end'].date() if post_info['search_end'] is not None else None,
            '后置原始天数': post_info['raw_days'],
            '后置实际开始日期': post_start.date() if post_start is not None else None,
            '后置实际结束日期': post_end.date() if post_end is not None else None,
            '后置连续非干旱天数': post_info['actual_days'],
            '后置不足天数': post_info['shortfall_days'],
            '后置跳过干旱日天数': post_info['skipped_drought_days'],
            '后置跳过缺失SPI天数': post_info['skipped_missing_days'],
            '后置是否等长': post_info['full_length'],
            '后置是否可计算β': beta_post['n'] >= MIN_HOURS_FOR_OLS,
            '后置有效小时数': beta_post['n'],
            '后置截断说明': post_info['note'],
            '综合非干旱目标天数': nondrought_info['target_days'],
            '综合非干旱实际天数': nondrought_info['actual_days'],
            '综合非干旱不足天数': nondrought_info['shortfall_days'],
            '综合非干旱是否等长': nondrought_info['full_length'],
            '综合非干旱是否可计算β': beta_nondrought['n'] >= MIN_HOURS_FOR_OLS,
            '综合非干旱有效小时数': beta_nondrought['n'],
            '综合非干旱前侧目标天数': nondrought_info['pre_target_days'],
            '综合非干旱后侧目标天数': nondrought_info['post_target_days'],
            '综合非干旱前侧实际天数': nondrought_info['pre_used_count'],
            '综合非干旱后侧实际天数': nondrought_info['post_used_count'],
            '综合非干旱前侧开始日期': nondrought_info['pre_start'].date() if nondrought_info['pre_start'] is not None else None,
            '综合非干旱前侧结束日期': nondrought_info['pre_end'].date() if nondrought_info['pre_end'] is not None else None,
            '综合非干旱后侧开始日期': nondrought_info['post_start'].date() if nondrought_info['post_start'] is not None else None,
            '综合非干旱后侧结束日期': nondrought_info['post_end'].date() if nondrought_info['post_end'] is not None else None,
            '综合非干旱前侧搜索开始日期': nondrought_info['pre_search_start'].date() if nondrought_info['pre_search_start'] is not None else None,
            '综合非干旱前侧搜索结束日期': nondrought_info['pre_search_end'].date() if nondrought_info['pre_search_end'] is not None else None,
            '综合非干旱后侧搜索开始日期': nondrought_info['post_search_start'].date() if nondrought_info['post_search_start'] is not None else None,
            '综合非干旱后侧搜索结束日期': nondrought_info['post_search_end'].date() if nondrought_info['post_search_end'] is not None else None,
            '综合非干旱前侧停止原因': nondrought_info['pre_stop_reason'],
            '综合非干旱后侧停止原因': nondrought_info['post_stop_reason'],
            '综合非干旱前侧说明': nondrought_info['pre_note'],
            '综合非干旱后侧说明': nondrought_info['post_note'],
            '综合非干旱总体说明': nondrought_info['note'],
            '事件级baseline允许月份': ",".join(map(str, baseline_info['allowed_months'])) if baseline_info['allowed_months'] else "",
            '事件级baseline非干旱天数': baseline_info['baseline_day_count'],
            '事件级baseline有效小时数': baseline_info['n_baseline'],
            '事件级baseline说明': baseline_info['note'],
            '事件级baselineβ': baseline_info['beta_baseline'],
            '长干旱Δβ(%)': delta_beta_pct,
        })

    beta_table = pd.DataFrame(results)
    control_quality = pd.DataFrame(quality_records)
    site_coverage = pd.DataFrame(coverage_records)

    if beta_table.empty:
        raise RuntimeError(
            "计算结果为空：没有任何目标等级干旱事件生成β记录。\n"
            "请检查：(1) TARGET_DROUGHT_LEVEL_CODE 是否匹配干旱长表；\n"
            "        (2) drought_df 与 merged 的 site_id 是否成功匹配；\n"
            "        (3) 事件日期是否落在TOMST和ERA5的时间覆盖范围内。"
        )

    # ── 数据质量报告 ──
    print(f"  ✓ 计算完成：{len(beta_table)} 条 site×event 记录")
    n_nan_d  = beta_table['beta_drought'].isna().sum()
    n_nan_p  = beta_table['beta_pre'].isna().sum()
    n_nan_po = beta_table['beta_post'].isna().sum()
    n_nan_nd = beta_table['beta_nondrought'].isna().sum()
    n_nan_bl = beta_table['beta_baseline'].isna().sum()
    n_long = int(beta_table['is_long_drought'].sum())
    n_short = int((~beta_table['is_long_drought']).sum())
    n_trunc_p  = beta_table['pre_truncated'].sum()
    n_trunc_po = beta_table['post_truncated'].sum()
    n_trunc_nd = beta_table['nondrought_truncated'].sum()
    print(f"  [信息] 短事件（<= {LONG_DROUGHT_THRESH} 天）: {n_short} 条；长事件（> {LONG_DROUGHT_THRESH} 天）: {n_long} 条")
    if n_nan_d  > 0:
        print(f"  [警告] {n_nan_d} 条干旱期β为NaN（数据点<{MIN_HOURS_FOR_OLS}h，请检查TOMST/ERA5数据完整性）")
    if n_nan_p  > 0:
        print(f"  [提示] {n_nan_p} 条前置对照期β为NaN（被完全遮盖或数据不足）")
    if n_nan_po > 0:
        print(f"  [提示] {n_nan_po} 条后置对照期β为NaN（被完全遮盖或数据不足）")
    if n_nan_nd > 0:
        print(f"  [提示] {n_nan_nd} 条综合非干旱期β为NaN（两侧合计非干旱日不足或小时数据不足）")
    if n_nan_bl > 0:
        print(f"  [提示] {n_nan_bl} 条事件级baseline β为NaN（允许月份内非干旱小时不足，或该事件为短事件）")
    if n_trunc_p  > 0:
        print(f"  [提示] {n_trunc_p} 条前置对照期被部分截断（pre_truncated=True）")
    if n_trunc_po > 0:
        print(f"  [提示] {n_trunc_po} 条后置对照期被部分截断（post_truncated=True）")
    if n_trunc_nd > 0:
        print(f"  [提示] {n_trunc_nd} 条综合非干旱期未补足等长（nondrought_truncated=True）")

    return beta_table, control_quality, site_coverage


# ============================================================
# Step 5：站点级聚合 + 统计检验 + 可视化
# ============================================================

def analyze_and_plot(beta_table, control_quality, site_coverage, output_dir,
                     merged=None, daily_spi_df=None):
    """
    在事件级β结果基础上进行站点级聚合和统计检验，并输出可视化图表。

    【为什么要进行站点级聚合？】
        每个站点可能有多次极端事件（如4月一次、8月一次），
        若直接用事件级数据做Wilcoxon检验，同一站点的多次事件相关联（伪重复）。
        解决方案：每个站点取所有极端事件β的中位数（站点代表值），
        得到27个独立配对观测，满足Wilcoxon检验的独立性假设。
        参考：De Frenne et al. (2021) Nature Ecology & Evolution

    【为什么用中位数而非均值？】
        中位数对极端值更稳健。若某次事件β因数据质量差而偏离正常范围，
        中位数不受影响，均值则会被拉偏。

    【统计检验：Wilcoxon符号秩检验】
        非参数配对检验，不要求数据正态性。
        H0：干旱期β = 对照期β（缓冲能力无差异）
        H1：干旱期β ≠ 对照期β（双侧检验）
        若 p < 0.05：拒绝H0，认为干旱显著改变了缓冲能力
        参考：Hollander & Wolfe (1999) Nonparametric Statistical Methods

    【Cohen's d 方向约定】
        d = (干旱期均值β - 对照期均值β) / 合并标准差
        d > 0：干旱期β更大 → 缓冲减弱（干旱削弱了林下缓冲能力）
        d < 0：干旱期β更小 → 缓冲增强（干旱期林下温度更稳定，罕见）
        |d| ≥ 0.2：小效应；|d| ≥ 0.5：中等效应；|d| ≥ 0.8：大效应

    【输出文件说明】
        事件级β配对结果表.csv        ：事件级详细表，含每次事件的三组β和对照期日期
        站点级β汇总表.csv            ：站点级中位数β汇总表，用于统计检验的输入数据
        对照期质量记录表.csv          ：记录原始窗口、实际连续非干旱天数、截断原因和是否可计算β
        站点事件覆盖记录表.csv        ：记录站点是否有目标等级事件，以及未进入分析原因
        极端干旱与对照期β箱线图.png   ：三组β箱线图（前置对照/干旱期/后置对照）
        统计检验结果报告.txt          ：完整统计检验报告

    参数：
        beta_table : Step 4 输出的事件级β结果DataFrame
        output_dir : 输出文件夹路径
    """
    print("\nStep 5: 站点级聚合 + 统计检验 + 可视化...")
    ensure_dir(output_dir)
    overall_plot_dir = os.path.join(output_dir, "总体图")
    site_plot_dir = os.path.join(output_dir, "站点图")
    diagnostic_plot_dir = os.path.join(output_dir, "站点事件OLS诊断图")
    ensure_dir(overall_plot_dir)
    ensure_dir(site_plot_dir)
    ensure_dir(diagnostic_plot_dir)

    if beta_table.empty:
        raise RuntimeError("事件级β结果表为空，无法进行站点聚合、统计检验和绘图。")

    # ── 5a. 保存事件级详细表 ──
    detail_path = os.path.join(output_dir, "事件级β配对结果表.csv")
    beta_table.to_csv(detail_path, index=False, encoding='utf-8-sig')
    print(f"  事件级详细表 → {detail_path}")

    short_event_table = beta_table[beta_table['is_long_drought'] == False].copy()
    long_event_table = beta_table[beta_table['is_long_drought'] == True].copy()
    short_detail_path = os.path.join(output_dir, "事件级β配对结果表_短事件.csv")
    long_detail_path = os.path.join(output_dir, "事件级β结果表_长事件基线法.csv")
    short_event_table.to_csv(short_detail_path, index=False, encoding='utf-8-sig')
    long_event_table.to_csv(long_detail_path, index=False, encoding='utf-8-sig')
    print(f"  短事件结果表 → {short_detail_path}")
    print(f"  长事件基线法结果表 → {long_detail_path}")

    quality_path = os.path.join(output_dir, "对照期质量记录表.csv")
    control_quality.to_csv(quality_path, index=False, encoding='utf-8-sig')
    print(f"  对照期质量记录表 → {quality_path}")

    coverage_path = os.path.join(output_dir, "站点事件覆盖记录表.csv")
    site_coverage.to_csv(coverage_path, index=False, encoding='utf-8-sig')
    print(f"  站点事件覆盖记录表 → {coverage_path}")

    # ── 5b. 站点级聚合：每个站点取所有极端事件的中位数β ──
    # 用中位数而非均值，对极端值更稳健
    # 参考：De Frenne et al. (2021) Nature Ecology & Evolution
    site_summary = (
        short_event_table.groupby('site_id')[['beta_drought', 'beta_pre', 'beta_post', 'beta_nondrought']]
        .median()
        .reset_index()
    )
    site_summary.columns = ['site_id', 'beta_drought', 'beta_pre', 'beta_post', 'beta_nondrought']

    if site_summary.empty:
        print("  [提示] 无可用于等长配对法的短事件站点汇总，后续短事件配对图与检验将跳过。")

    if not site_summary.empty:
        summary_path = os.path.join(output_dir, "站点级β汇总表.csv")
        site_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"  站点级汇总表 → {summary_path}")

        two_period_summary = site_summary[['site_id', 'beta_nondrought', 'beta_drought']].copy()
        two_period_summary_path = os.path.join(output_dir, "站点级_干旱与非干旱β汇总表.csv")
        two_period_summary.to_csv(two_period_summary_path, index=False, encoding='utf-8-sig')
        print(f"  站点级干旱与非干旱汇总表 → {two_period_summary_path}")

    long_delta_site_summary = (
        long_event_table.groupby('site_id')[['delta_beta_pct', 'beta_drought', 'beta_baseline']]
        .median()
        .reset_index()
    )
    long_delta_summary_path = os.path.join(output_dir, "站点级_长干旱Δβ汇总表.csv")
    long_delta_site_summary.to_csv(long_delta_summary_path, index=False, encoding='utf-8-sig')
    print(f"  长干旱Δβ站点汇总表 → {long_delta_summary_path}")

    # ── 额外诊断统计：筛出“干旱期β < 对照期β”的事件 ──
    # 目的：
    #   便于快速定位那些“干旱期反而更稳定/缓冲更强”的事件，供后续逐图核查。
    # 说明：
    #   - 短事件主方法为配对对照，因此分别统计：
    #       1) 干旱 vs 前置对照
    #       2) 干旱 vs 后置对照
    #       3) 干旱 vs 综合非干旱
    #   - 长事件主方法为 baseline，因此另外统计：
    #       4) 干旱 vs baseline
    beta_smaller_records = []

    for _, row in short_event_table.iterrows():
        comparisons = [
            ("Pre-control", 'beta_pre', row.get('pre_start'), row.get('pre_end')),
            ("Post-control", 'beta_post', row.get('post_start'), row.get('post_end')),
            ("Non-drought", 'beta_nondrought', row.get('nondrought_pre_start'), row.get('nondrought_post_end')),
        ]
        for control_label, control_col, control_start, control_end in comparisons:
            drought_beta = row.get('beta_drought', np.nan)
            control_beta = row.get(control_col, np.nan)
            if pd.notna(drought_beta) and pd.notna(control_beta) and drought_beta < control_beta:
                beta_smaller_records.append({
                    '站点编号': row['site_id'],
                    '事件编号': row['event_id'],
                    '事件类型': '短事件',
                    '比较对象': control_label,
                    '干旱开始日期': row['drought_start'],
                    '干旱结束日期': row['drought_end'],
                    '干旱天数': row['duration_days'],
                    '对照开始日期': control_start,
                    '对照结束日期': control_end,
                    '干旱期β': drought_beta,
                    '对照期β': control_beta,
                    'β差值(干旱-对照)': drought_beta - control_beta,
                    '是否建议人工复核': '是',
                })

    for _, row in long_event_table.iterrows():
        drought_beta = row.get('beta_drought', np.nan)
        baseline_beta = row.get('beta_baseline', np.nan)
        if pd.notna(drought_beta) and pd.notna(baseline_beta) and drought_beta < baseline_beta:
            beta_smaller_records.append({
                '站点编号': row['site_id'],
                '事件编号': row['event_id'],
                '事件类型': '长事件',
                '比较对象': 'Baseline',
                '干旱开始日期': row['drought_start'],
                '干旱结束日期': row['drought_end'],
                '干旱天数': row['duration_days'],
                '对照开始日期': None,
                '对照结束日期': None,
                '干旱期β': drought_beta,
                '对照期β': baseline_beta,
                'β差值(干旱-对照)': drought_beta - baseline_beta,
                'Δβ(%)': row.get('delta_beta_pct', np.nan),
                'baseline允许月份': row.get('baseline_allowed_months', ""),
                'baseline有效小时数': row.get('n_baseline', np.nan),
                '是否建议人工复核': '是',
            })

    beta_smaller_df = pd.DataFrame(beta_smaller_records)
    if not beta_smaller_df.empty:
        beta_smaller_df = beta_smaller_df.sort_values(
            by=['站点编号', '干旱开始日期', '事件编号', '比较对象'],
            kind='stable'
        ).reset_index(drop=True)
    beta_smaller_path = os.path.join(output_dir, "干旱期β小于对照期β的事件清单.csv")
    beta_smaller_df.to_csv(beta_smaller_path, index=False, encoding='utf-8-sig')
    print(f"  干旱期β较小事件清单 → {beta_smaller_path}")

    # ── 5c. Wilcoxon符号秩检验（配对） ──
    # 分别检验：干旱期 vs 前置对照期；干旱期 vs 后置对照期
    # 参考：Hollander & Wolfe (1999) Nonparametric Statistical Methods

    def do_wilcoxon(drought_vals, control_vals, label):
        """对一组配对数据执行Wilcoxon检验并生成文字报告"""
        n = len(drought_vals)
        if n < 5:
            return (f"{label}: 有效配对站点数 n={n}（需≥5），"
                    f"样本量不足，跳过检验\n"
                    f"  原因：部分站点的前/后对照期被截断或数据不足导致β为NaN")

        diff = drought_vals.reset_index(drop=True) - control_vals.reset_index(drop=True)
        if np.allclose(diff.values, 0, equal_nan=True):
            return (f"{label}: 有效配对站点数 n={n}，所有配对差值均为0，"
                    f"不执行Wilcoxon检验；结论：干旱期与对照期β完全相同。")

        # Wilcoxon检验（双侧，两组分布是否有显著差异）
        stat, p = wilcoxon(drought_vals, control_vals, alternative='two-sided')

        # Cohen's d 效应量
        d_mean = drought_vals.mean() - control_vals.mean()
        pool_sd = np.sqrt(
            ((n-1)*drought_vals.std(ddof=1)**2 + (n-1)*control_vals.std(ddof=1)**2)
            / (2*n - 2)
        )
        d_val    = d_mean / pool_sd if pool_sd > 0 else np.nan
        diff_pct = d_mean / control_vals.mean() * 100 if control_vals.mean() != 0 else np.nan

        # 判断效应量大小
        if abs(d_val) < 0.2:
            effect_size_label = "微小效应"
        elif abs(d_val) < 0.5:
            effect_size_label = "小效应"
        elif abs(d_val) < 0.8:
            effect_size_label = "中等效应"
        else:
            effect_size_label = "大效应"

        sig   = "★ 显著差异 (p<0.05)" if p < 0.05 else "无显著差异 (p≥0.05)"
        lines = [
            f"【{label}】（n={n} 个站点配对）",
            f"  干旱期   均值β={drought_vals.mean():.4f}，中位数β={drought_vals.median():.4f}，"
            f"SD={drought_vals.std(ddof=1):.4f}",
            f"  对照期   均值β={control_vals.mean():.4f}，中位数β={control_vals.median():.4f}，"
            f"SD={control_vals.std(ddof=1):.4f}",
            f"  均值差（干旱−对照）= {d_mean:+.4f}（相对变化 {diff_pct:+.1f}%）",
            f"  Wilcoxon W统计量 = {stat:.1f}，p值 = {p:.4f}  →  {sig}",
            f"  Cohen's d（干旱−对照）= {d_val:.3f}（{effect_size_label}）",
            f"  生态解读：{'干旱期β显著偏大→极端干旱削弱了林下微气候缓冲能力' if p < 0.05 and d_val > 0 else '干旱期β显著偏小→极端干旱增强了林下微气候缓冲能力' if p < 0.05 and d_val < 0 else '干旱期与对照期缓冲能力无显著差异'}",
        ]
        return "\n".join(lines)

    # 筛选前置和后置的有效配对（去除任一为NaN的站点）
    if not site_summary.empty:
        valid_pre  = site_summary.dropna(subset=['beta_drought', 'beta_pre'])
        valid_post = site_summary.dropna(subset=['beta_drought', 'beta_post'])
        valid_nondrought = site_summary.dropna(subset=['beta_drought', 'beta_nondrought'])

        report_pre  = do_wilcoxon(
            valid_pre['beta_drought'], valid_pre['beta_pre'],
            "短事件：干旱期 vs 前置等长对照期（Pre-drought control）"
        )
        report_post = do_wilcoxon(
            valid_post['beta_drought'], valid_post['beta_post'],
            "短事件：干旱期 vs 后置等长对照期（Post-drought control）"
        )
        report_nondrought = do_wilcoxon(
            valid_nondrought['beta_drought'], valid_nondrought['beta_nondrought'],
            "短事件：干旱期 vs 综合非干旱期（Balanced non-drought control）"
        )
    else:
        valid_pre = pd.DataFrame()
        valid_post = pd.DataFrame()
        valid_nondrought = pd.DataFrame()
        report_pre = "短事件：无可用站点汇总，跳过前置等长对照检验。"
        report_post = "短事件：无可用站点汇总，跳过后置等长对照检验。"
        report_nondrought = "短事件：无可用站点汇总，跳过综合非干旱检验。"

    def do_wilcoxon_against_zero(values, label):
        """对单组Δβ做单样本Wilcoxon检验，H0: median(Δβ)=0。"""
        clean_vals = pd.Series(values).dropna().astype(float)
        n = len(clean_vals)
        if n < 5:
            return (f"{label}: 有效站点数 n={n}（需≥5），样本量不足，跳过检验")
        if np.allclose(clean_vals.values, 0):
            return (f"{label}: 有效站点数 n={n}，所有站点Δβ均为0，"
                    f"不执行Wilcoxon检验；结论：长干旱期与事件级baseline无系统偏差。")

        stat, p = wilcoxon(clean_vals, alternative='two-sided')
        mean_val = clean_vals.mean()
        median_val = clean_vals.median()
        sd_val = clean_vals.std(ddof=1)
        sig = "★ 显著差异 (p<0.05)" if p < 0.05 else "无显著差异 (p≥0.05)"
        interpretation = (
            "长干旱期β显著高于同季节baseline → 缓冲能力下降"
            if p < 0.05 and median_val > 0 else
            "长干旱期β显著低于同季节baseline → 缓冲能力增强"
            if p < 0.05 and median_val < 0 else
            "长干旱期β与同季节baseline无显著差异"
        )
        lines = [
            f"【{label}】（n={n} 个站点）",
            f"  Δβ(%) 均值={mean_val:+.2f}，中位数={median_val:+.2f}，SD={sd_val:.2f}",
            f"  Wilcoxon W统计量 = {stat:.1f}，p值 = {p:.4f}  →  {sig}",
            f"  生态解读：{interpretation}",
        ]
        return "\n".join(lines)

    long_delta_by_site = long_event_table.groupby('site_id')['delta_beta_pct'].median()
    report_long_delta = do_wilcoxon_against_zero(
        long_delta_by_site,
        f"长事件：Δβ 相对偏差法（>{LONG_DROUGHT_THRESH} 天，vs 事件级baseline）"
    )

    report_lines = [
        "=" * 65,
        "极端干旱事件对林下微气候缓冲能力影响 — 统计检验报告",
        "=" * 65,
        "",
        "【方法说明】",
        "  短事件方法 ：等长配对（Length-matched paired control）",
        "    参考文献 ：Zellweger et al. (2020) Global Change Biology",
        "               De Frenne et al. (2021) Nature Ecology & Evolution",
        f"  长事件方法 ：事件级同季节baseline + 相对偏差法（干旱期天数 > {LONG_DROUGHT_THRESH}）",
        "    逻辑说明 ：以事件开始月前后1个月内的非干旱小时数据作为baseline",
        "  β计算方法  ：OLS回归 T_micro = α + β × T_macro",
        "    参考文献 ：Ma et al. (2025) Agricultural and Forest Meteorology",
        "  站点聚合   ：每站点取所有同类事件统计量的中位数（n=27个独立站点）",
        "  统计检验   ：短事件用配对Wilcoxon；长事件Δβ用单样本Wilcoxon against 0",
        "    参考文献 ：Hollander & Wolfe (1999) Nonparametric Statistical Methods",
        "  效应量     ：短事件报告 Cohen's d（干旱期 − 对照期）",
        "",
        "【β含义】",
        "  β越小（接近0）→ 缓冲能力越强（林内温度更稳定）",
        "  β越大（接近1）→ 缓冲能力越弱（林内温度更随宏气候波动）",
        "  β>1           → 放大效应（林内温度波动超过宏气候）",
        "  Δβ(%) > 0     → 干旱期β高于baseline，缓冲能力下降",
        "  Δβ(%) < 0     → 干旱期β低于baseline，缓冲能力增强",
        "",
        "=" * 65,
        "【检验结果】",
        "",
        report_pre,
        "",
        report_post,
        "",
        report_nondrought,
        "",
        report_long_delta,
        "",
        "=" * 65,
    ]
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_path = os.path.join(output_dir, "统计检验结果报告.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n  统计报告 → {report_path}")

    # ── 5d. Overall grouped boxplot（global summary） ──
    order  = ['Pre-control', 'Drought', 'Post-control']
    colors = BOX_COLORS

    if not site_summary.empty:
        pre_vals = site_summary[['site_id', 'beta_pre']].dropna().rename(columns={'beta_pre': 'beta'})
        pre_vals['Period'] = order[0]
        drought_vals = site_summary[['site_id', 'beta_drought']].dropna().rename(columns={'beta_drought': 'beta'})
        drought_vals['Period'] = order[1]
        post_vals = site_summary[['site_id', 'beta_post']].dropna().rename(columns={'beta_post': 'beta'})
        post_vals['Period'] = order[2]
        plot_long = pd.concat([pre_vals, drought_vals, post_vals], ignore_index=True)

        if not plot_long.empty:
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            sns.boxplot(
                data=plot_long, x='Period', y='beta', order=order,
                palette=colors, width=BOX_WIDTH, linewidth=BOX_LINEWIDTH,
                flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5),
                ax=ax
            )
            sns.stripplot(
                data=plot_long, x='Period', y='beta', order=order,
                color=POINT_COLOR, size=POINT_SIZE, alpha=POINT_ALPHA, jitter=POINT_JITTER, ax=ax
            )

            y_max = plot_long['beta'].dropna().max()
            beta_range = plot_long['beta'].dropna().max() - plot_long['beta'].dropna().min()
            offset = beta_range * 0.06 if beta_range > 0 else 0.05

            if len(valid_pre) >= 5:
                _, p_pre = wilcoxon(valid_pre['beta_drought'], valid_pre['beta_pre'],
                                    alternative='two-sided')
                y_line = y_max + offset
                ax.plot([0, 1], [y_line, y_line], color=SIGNIFICANCE_LINE_COLOR, linewidth=SIGNIFICANCE_LINEWIDTH)
                sig_str = f"p={p_pre:.3f}" + (" *" if p_pre < 0.05 else " ns")
                ax.text(0.5, y_line + offset * 0.3, sig_str,
                        ha='center', va='bottom', fontsize=9)

            if len(valid_post) >= 5:
                _, p_post = wilcoxon(valid_post['beta_drought'], valid_post['beta_post'],
                                     alternative='two-sided')
                y_line2 = y_max + offset * 2.5
                ax.plot([1, 2], [y_line2, y_line2], color=SIGNIFICANCE_LINE_COLOR, linewidth=SIGNIFICANCE_LINEWIDTH)
                sig_str = f"p={p_post:.3f}" + (" *" if p_post < 0.05 else " ns")
                ax.text(1.5, y_line2 + offset * 0.3, sig_str,
                        ha='center', va='bottom', fontsize=9)

            ax.axhline(REFERENCE_LINE_Y, color=REFERENCE_LINE_COLOR, linestyle=REFERENCE_LINE_STYLE,
                       alpha=REFERENCE_LINE_ALPHA, linewidth=1)
            ax.text(2.48, 1.002, 'β=1\n(No buffering)', fontsize=8, color='gray', va='bottom', ha='right')

            for i, col in enumerate(['beta_pre', 'beta_drought', 'beta_post']):
                n_valid = site_summary[col].notna().sum()
                ax.text(i, plot_long['beta'].dropna().min() - offset * 0.8,
                        f"n={n_valid}", ha='center', fontsize=9, color=SAMPLE_N_COLOR)

            ax.set_xlabel("", fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_ylabel("Climate Buffering Index (β)", fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_title(
                "Overall Comparison of Climate Buffering Index\n"
                "Pre-control vs Drought vs Post-control (short-event summary)",
                fontsize=TITLE_FONTSIZE, pad=12
            )

            plt.tight_layout()
            box_path = os.path.join(overall_plot_dir, "总体_β分组箱线图.png")
            plt.savefig(box_path, dpi=FIG_DPI, bbox_inches='tight')
            plt.close()
            print(f"  总体箱线图 → {box_path}")

    # ── 5e. Overall delta-beta plot ──
    delta_pre = short_event_table[['site_id', 'event_id', 'drought_start', 'beta_drought', 'beta_pre']].dropna().copy()
    delta_pre['Comparison'] = 'Drought - Pre'
    delta_pre['delta_beta'] = delta_pre['beta_drought'] - delta_pre['beta_pre']
    delta_post = short_event_table[['site_id', 'event_id', 'drought_start', 'beta_drought', 'beta_post']].dropna().copy()
    delta_post['Comparison'] = 'Drought - Post'
    delta_post['delta_beta'] = delta_post['beta_drought'] - delta_post['beta_post']
    delta_df = pd.concat([
        delta_pre[['site_id', 'event_id', 'drought_start', 'Comparison', 'delta_beta']],
        delta_post[['site_id', 'event_id', 'drought_start', 'Comparison', 'delta_beta']]
    ], ignore_index=True)

    if not delta_df.empty:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        sns.boxplot(
            data=delta_df, x='Comparison', y='delta_beta',
            palette=['#E57373', '#64B5F6'], width=BOX_WIDTH, linewidth=BOX_LINEWIDTH, ax=ax
        )
        sns.stripplot(
            data=delta_df, x='Comparison', y='delta_beta',
            color=POINT_COLOR, size=POINT_SIZE, alpha=POINT_ALPHA, jitter=POINT_JITTER, ax=ax
        )
        ax.axhline(0.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel("")
        ax.set_ylabel("Δβ (Drought - Control)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title("Overall Delta-β Comparison", fontsize=TITLE_FONTSIZE, pad=12)
        plt.tight_layout()
        delta_path = os.path.join(overall_plot_dir, "总体_β差值图.png")
        plt.savefig(delta_path, dpi=FIG_DPI, bbox_inches='tight')
        plt.close()
        print(f"  总体差值图 → {delta_path}")

    # ── 5f. Overall boxplot: Non-drought vs Drought ──
    if not site_summary.empty:
        order_two = ['Non-drought', 'Drought']
        nondrought_vals = site_summary[['site_id', 'beta_nondrought']].dropna().rename(columns={'beta_nondrought': 'beta'})
        nondrought_vals['Period'] = order_two[0]
        drought_vals_two = site_summary[['site_id', 'beta_drought']].dropna().rename(columns={'beta_drought': 'beta'})
        drought_vals_two['Period'] = order_two[1]
        plot_long_two = pd.concat([nondrought_vals, drought_vals_two], ignore_index=True)

        if not plot_long_two.empty:
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            sns.boxplot(
                data=plot_long_two, x='Period', y='beta', order=order_two,
                palette=BOX_COLORS_TWO_PERIOD, width=BOX_WIDTH, linewidth=BOX_LINEWIDTH,
                flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5),
                ax=ax
            )
            sns.stripplot(
                data=plot_long_two, x='Period', y='beta', order=order_two,
                color=POINT_COLOR, size=POINT_SIZE, alpha=POINT_ALPHA, jitter=POINT_JITTER, ax=ax
            )
            y_max_two = plot_long_two['beta'].dropna().max()
            beta_range_two = plot_long_two['beta'].dropna().max() - plot_long_two['beta'].dropna().min()
            offset_two = beta_range_two * 0.06 if beta_range_two > 0 else 0.05
            if len(valid_nondrought) >= 5:
                _, p_two = wilcoxon(
                    valid_nondrought['beta_drought'],
                    valid_nondrought['beta_nondrought'],
                    alternative='two-sided'
                )
                y_line = y_max_two + offset_two
                ax.plot([0, 1], [y_line, y_line], color=SIGNIFICANCE_LINE_COLOR, linewidth=SIGNIFICANCE_LINEWIDTH)
                sig_str = f"p={p_two:.3f}" + (" *" if p_two < 0.05 else " ns")
                ax.text(0.5, y_line + offset_two * 0.3, sig_str, ha='center', va='bottom', fontsize=9)
            ax.axhline(REFERENCE_LINE_Y, color=REFERENCE_LINE_COLOR, linestyle=REFERENCE_LINE_STYLE,
                       alpha=REFERENCE_LINE_ALPHA, linewidth=1)
            ax.text(1.48, 1.002, 'β=1\n(No buffering)', fontsize=8, color='gray', va='bottom', ha='right')
            for i, col in enumerate(['beta_nondrought', 'beta_drought']):
                n_valid = site_summary[col].notna().sum()
                ax.text(i, plot_long_two['beta'].dropna().min() - offset_two * 0.8,
                        f"n={n_valid}", ha='center', fontsize=9, color=SAMPLE_N_COLOR)
            ax.set_xlabel("")
            ax.set_ylabel("Climate Buffering Index (β)", fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_title(
                "Overall Comparison of Climate Buffering Index\n"
                "Non-drought vs Drought (short-event summary)",
                fontsize=TITLE_FONTSIZE, pad=12
            )
            plt.tight_layout()
            box_path_two = os.path.join(overall_plot_dir, "总体_干旱与非干旱β箱线图.png")
            plt.savefig(box_path_two, dpi=FIG_DPI, bbox_inches='tight')
            plt.close()
            print(f"  总体干旱与非干旱箱线图 → {box_path_two}")

    # ── 5g. Overall delta-beta plot: Non-drought vs Drought ──
    delta_nondrought = short_event_table[
        ['site_id', 'event_id', 'drought_start', 'beta_drought', 'beta_nondrought']
    ].dropna().copy()
    if not delta_nondrought.empty:
        delta_nondrought['Comparison'] = 'Drought - Non-drought'
        delta_nondrought['delta_beta'] = (
            delta_nondrought['beta_drought'] - delta_nondrought['beta_nondrought']
        )
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        sns.boxplot(
            data=delta_nondrought, x='Comparison', y='delta_beta',
            palette=['#FF8A65'], width=BOX_WIDTH, linewidth=BOX_LINEWIDTH, ax=ax
        )
        sns.stripplot(
            data=delta_nondrought, x='Comparison', y='delta_beta',
            color=POINT_COLOR, size=POINT_SIZE, alpha=POINT_ALPHA, jitter=POINT_JITTER, ax=ax
        )
        ax.axhline(0.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel("")
        ax.set_ylabel("Δβ (Drought - Non-drought)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title("Overall Delta-β Comparison: Drought vs Non-drought",
                     fontsize=TITLE_FONTSIZE, pad=12)
        plt.tight_layout()
        delta_path_two = os.path.join(overall_plot_dir, "总体_干旱与非干旱β差值图.png")
        plt.savefig(delta_path_two, dpi=FIG_DPI, bbox_inches='tight')
        plt.close()
        print(f"  总体干旱与非干旱差值图 → {delta_path_two}")

    # ── 5h. Overall plot: long-drought Δβ (%) relative deviation ──
    long_delta_plot = long_event_table[['site_id', 'event_id', 'drought_start', 'delta_beta_pct']].dropna().copy()
    if not long_delta_plot.empty:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        sns.boxplot(
            data=long_delta_plot, y='delta_beta_pct',
            color='#FFB74D', width=0.35, linewidth=BOX_LINEWIDTH, ax=ax
        )
        sns.stripplot(
            data=long_delta_plot, y='delta_beta_pct',
            color=POINT_COLOR, size=POINT_SIZE, alpha=POINT_ALPHA, jitter=0.12, ax=ax
        )
        ax.axhline(0.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel("")
        ax.set_ylabel("Δβ (%) relative to baseline", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(
            f"Long-drought Relative Deviation of β\n(>{LONG_DROUGHT_THRESH} days, event-level baseline)",
            fontsize=TITLE_FONTSIZE, pad=12
        )
        n_valid_long_events = len(long_delta_plot)
        y_min_long = long_delta_plot['delta_beta_pct'].min()
        y_max_long = long_delta_plot['delta_beta_pct'].max()
        y_span_long = y_max_long - y_min_long
        y_offset_long = y_span_long * 0.08 if y_span_long > 0 else 3.0
        ax.text(0, y_min_long - y_offset_long, f"n={n_valid_long_events}",
                ha='center', fontsize=9, color=SAMPLE_N_COLOR)
        plt.tight_layout()
        long_delta_plot_path = os.path.join(overall_plot_dir, "总体_长干旱Δβ相对偏差图.png")
        plt.savefig(long_delta_plot_path, dpi=FIG_DPI, bbox_inches='tight')
        plt.close()
        print(f"  总体长干旱Δβ图 → {long_delta_plot_path}")

    # ── 5i. Site-level plots: paired line plot and event bar plot（3-period, short events only） ──
    site_event_long = build_site_event_long(short_event_table)
    for site_id, site_plot_df in site_event_long.groupby('site_id'):
        site_plot_df = site_plot_df.sort_values(['drought_start', 'period']).copy()
        event_order = sorted(site_plot_df['event_order'].unique())

        # Paired line plot
        fig, ax = plt.subplots(figsize=SITE_FIG_SIZE)
        x_positions = []
        x_labels = []
        period_offsets = {'Pre': -0.25, 'Drought': 0.0, 'Post': 0.25}
        for event_idx in event_order:
            event_df = site_plot_df[site_plot_df['event_order'] == event_idx]
            base_x = event_idx
            period_x = []
            period_y = []
            for period in ['Pre', 'Drought', 'Post']:
                row = event_df[event_df['period'] == period]
                if row.empty or pd.isna(row['beta'].iloc[0]):
                    continue
                x = base_x + period_offsets[period]
                y = row['beta'].iloc[0]
                ax.scatter(x, y, color=LINE_COLORS[period], s=40, zorder=3)
                period_x.append(x)
                period_y.append(y)
                period_start = row['period_start'].iloc[0]
                period_end = row['period_end'].iloc[0]
                period_days = row['period_days'].iloc[0]
                if pd.notna(period_start) and pd.notna(period_end):
                    start_dt = pd.to_datetime(period_start)
                    end_dt = pd.to_datetime(period_end)
                    date_label = f"{start_dt.month}/{start_dt.day}-{end_dt.month}/{end_dt.day}"
                else:
                    date_label = "NA"
                days_label = f"{int(period_days)} d" if pd.notna(period_days) else "NA d"
                x_positions.append(x)
                x_labels.append(f"{period}\n{date_label}\n{days_label}")
            if period_x:
                ax.plot(period_x, period_y, color='black', linewidth=1.0, alpha=0.8, zorder=2)

        ax.axhline(REFERENCE_LINE_Y, color=REFERENCE_LINE_COLOR, linestyle=REFERENCE_LINE_STYLE,
                   alpha=REFERENCE_LINE_ALPHA, linewidth=1)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylabel("Climate Buffering Index (β)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xlabel("Periods within each event (dates and duration)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(f"Site {site_id}: Event-level Paired β Trajectories", fontsize=TITLE_FONTSIZE, pad=10)
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=LINE_COLORS['Pre'], markersize=7, label='Pre-control'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=LINE_COLORS['Drought'], markersize=7, label='Drought'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=LINE_COLORS['Post'], markersize=7, label='Post-control'),
        ]
        ax.legend(handles=handles, loc='best', fontsize=LEGEND_FONTSIZE)
        plt.tight_layout()
        line_path = os.path.join(site_plot_dir, f"站点_{sanitize_filename(site_id)}_配对连线图.png")
        plt.savefig(line_path, dpi=FIG_DPI, bbox_inches='tight')
        plt.close()

    # ── 5j. Site-level plots: Non-drought vs Drought（2-period, short events only） ──
    site_event_nondrought_long = build_site_event_nondrought_long(short_event_table)
    for site_id, site_plot_df in site_event_nondrought_long.groupby('site_id'):
        site_plot_df = site_plot_df.sort_values(['drought_start', 'period']).copy()
        event_order = sorted(site_plot_df['event_order'].unique())

        fig, ax = plt.subplots(figsize=SITE_FIG_SIZE)
        x_positions = []
        x_labels = []
        period_offsets = {'Non-drought': -0.15, 'Drought': 0.15}
        for event_idx in event_order:
            event_df = site_plot_df[site_plot_df['event_order'] == event_idx]
            base_x = event_idx
            period_x = []
            period_y = []
            for period in ['Non-drought', 'Drought']:
                row = event_df[event_df['period'] == period]
                if row.empty or pd.isna(row['beta'].iloc[0]):
                    continue
                x = base_x + period_offsets[period]
                y = row['beta'].iloc[0]
                ax.scatter(x, y, color=LINE_COLORS[period], s=40, zorder=3)
                period_x.append(x)
                period_y.append(y)
                if period == 'Non-drought':
                    pre_range = format_date_range(row['pre_start'].iloc[0], row['pre_end'].iloc[0])
                    post_range = format_date_range(row['post_start'].iloc[0], row['post_end'].iloc[0])
                    pre_days = row['pre_days'].iloc[0]
                    post_days = row['post_days'].iloc[0]
                    total_days = row['period_days'].iloc[0]
                    label = (
                        f"Non-drought\n"
                        f"Pre {pre_range} ({int(pre_days) if pd.notna(pre_days) else 0} d)\n"
                        f"Post {post_range} ({int(post_days) if pd.notna(post_days) else 0} d)\n"
                        f"Total {int(total_days) if pd.notna(total_days) else 0} d"
                    )
                else:
                    date_label = format_date_range(row['period_start'].iloc[0], row['period_end'].iloc[0])
                    days_label = row['period_days'].iloc[0]
                    label = f"Drought\n{date_label}\n{int(days_label) if pd.notna(days_label) else 0} d"
                x_positions.append(x)
                x_labels.append(label)
            if period_x:
                ax.plot(period_x, period_y, color='black', linewidth=1.0, alpha=0.8, zorder=2)

        ax.axhline(REFERENCE_LINE_Y, color=REFERENCE_LINE_COLOR, linestyle=REFERENCE_LINE_STYLE,
                   alpha=REFERENCE_LINE_ALPHA, linewidth=1)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylabel("Climate Buffering Index (β)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xlabel("Periods within each event (dates and duration)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(f"Site {site_id}: Drought vs Non-drought β", fontsize=TITLE_FONTSIZE, pad=10)
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=LINE_COLORS['Non-drought'], markersize=7, label='Non-drought'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=LINE_COLORS['Drought'], markersize=7, label='Drought'),
        ]
        ax.legend(handles=handles, loc='best', fontsize=LEGEND_FONTSIZE)
        plt.tight_layout()
        line_path_two = os.path.join(site_plot_dir, f"站点_{sanitize_filename(site_id)}_干旱与非干旱配对连线图.png")
        plt.savefig(line_path_two, dpi=FIG_DPI, bbox_inches='tight')
        plt.close()

    # ── 5k. Site-level summary paired plots: all short events collapsed into two groups ──
    # 图形目的：
    #   每个站点一张图，把该站点所有极端干旱事件的 Non-drought β 与 Drought β 放在一起。
    #   不再按事件编号或事件日期展开，避免图面信息过碎；灰色细线仍保留“同一次事件”的配对关系。
    # 图形元素：
    #   - 左侧 Non-drought、右侧 Drought；
    #   - 灰色细线：每次事件的配对变化；
    #   - 彩色散点：每次事件的 β；
    #   - 黑色粗线：该站点所有事件的中位数变化；
    #   - 右上角 n = X events：该站点可用于该图的有效配对事件数。
    summary_plot_data = short_event_table[
        ['site_id', 'event_id', 'beta_nondrought', 'beta_drought']
    ].dropna(subset=['beta_nondrought', 'beta_drought']).copy()

    for site_id, site_df in summary_plot_data.groupby('site_id'):
        site_df = site_df.sort_values('event_id').reset_index(drop=True)
        n_events = len(site_df)
        if n_events == 0:
            continue

        fig, ax = plt.subplots(figsize=(5.2, 5.5))
        x_non = 0
        x_dry = 1

        # 为避免多个事件点完全重叠，给每次事件一个很小的水平偏移；不显示事件编号。
        if n_events == 1:
            jitter_values = [0.0]
        else:
            jitter_values = np.linspace(-0.045, 0.045, n_events)

        for jitter, (_, row) in zip(jitter_values, site_df.iterrows()):
            y_non = row['beta_nondrought']
            y_dry = row['beta_drought']
            ax.plot(
                [x_non + jitter, x_dry + jitter],
                [y_non, y_dry],
                color='gray',
                linewidth=1.0,
                alpha=0.65,
                zorder=1
            )
            ax.scatter(
                x_non + jitter, y_non,
                color=LINE_COLORS['Non-drought'],
                edgecolor='black',
                linewidth=0.4,
                s=46,
                zorder=3
            )
            ax.scatter(
                x_dry + jitter, y_dry,
                color=LINE_COLORS['Drought'],
                edgecolor='black',
                linewidth=0.4,
                s=46,
                zorder=3
            )

        median_non = site_df['beta_nondrought'].median()
        median_dry = site_df['beta_drought'].median()
        ax.plot(
            [x_non, x_dry],
            [median_non, median_dry],
            color='black',
            linewidth=3.0,
            alpha=0.95,
            zorder=4,
            label='Median'
        )
        ax.scatter(
            [x_non, x_dry],
            [median_non, median_dry],
            color='black',
            s=52,
            zorder=5
        )

        ax.axhline(
            REFERENCE_LINE_Y,
            color=REFERENCE_LINE_COLOR,
            linestyle=REFERENCE_LINE_STYLE,
            alpha=REFERENCE_LINE_ALPHA,
            linewidth=1
        )
        ax.set_xlim(-0.35, 1.35)
        ax.set_xticks([x_non, x_dry])
        ax.set_xticklabels(['Non-drought', 'Drought'], fontsize=10)
        ax.set_ylabel("Climate Buffering Index (β)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xlabel("")
        ax.set_title(
            f"Site {site_id}: Drought vs Non-drought\nAll extreme drought events",
            fontsize=TITLE_FONTSIZE,
            pad=10
        )
        ax.text(
            0.98, 0.96,
            f"n = {n_events} events",
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=9,
            color=SAMPLE_N_COLOR
        )
        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=LINE_COLORS['Non-drought'],
                       markeredgecolor='black', markersize=7, label='Non-drought'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=LINE_COLORS['Drought'],
                       markeredgecolor='black', markersize=7, label='Drought'),
            plt.Line2D([0], [0], color='black', linewidth=3, label='Median'),
        ]
        ax.legend(handles=handles, loc='best', fontsize=LEGEND_FONTSIZE)
        plt.tight_layout()
        summary_line_path = os.path.join(
            site_plot_dir,
            f"站点_{sanitize_filename(site_id)}_干旱与非干旱汇总配对图.png"
        )
        plt.savefig(summary_line_path, dpi=FIG_DPI, bbox_inches='tight')
        plt.close()

    # ── 5l. OLS diagnostic plots for every site × extreme drought event ──
    # 每张图直接展示 T_micro ~ T_macro 的逐小时散点和OLS回归线，用于检查β来源。
    # 短事件：Drought vs Non-drought；长事件：Drought vs event-level Baseline。
    if merged is not None and daily_spi_df is not None:
        for _, event_row in tqdm(
            beta_table.iterrows(),
            total=len(beta_table),
            desc="Step 5 OLS诊断图",
            colour=PROGRESS_COLOURS["diagnostic"],
            dynamic_ncols=TQDM_NCOLS_DYNAMIC,
            leave=TQDM_LEAVE,
            bar_format=TQDM_BAR_FORMAT,
        ):
            site_id = event_row['site_id']
            site_data = merged[merged['site_id'] == site_id].copy()
            site_data = site_data.set_index('datetime').sort_index()
            site_daily_spi = daily_spi_df[daily_spi_df['site_id'] == site_id].copy()
            plot_event_ols_diagnostic(
                event_row,
                site_data,
                site_daily_spi,
                diagnostic_plot_dir
            )
        print(f"  OLS诊断图目录 → {diagnostic_plot_dir}")
    else:
        print("  [提示] 未传入 merged/daily_spi_df，跳过OLS诊断图。")

    print(f"  站点级图目录 → {site_plot_dir}")
    print(f"  总体图目录 → {overall_plot_dir}")

    print(f"\n{'='*65}")
    print(f"✓ 所有输出文件已保存至：{output_dir}")
    print(f"{'='*65}")


# ============================================================
# 主函数：按顺序调用各步骤
# ============================================================

def main():
    """
    主程序入口：按步骤顺序执行全部分析流程。

    执行顺序：
        Step 1 → 加载TOMST（~1分钟，取决于CSV数量）
        Step 2 → 提取ERA5（~10-20分钟，需遍历8760张TIF）★最耗时
        Step 3 → 合并数据（秒级）
        Step 4 → 计算三组β（秒级至分钟级，取决于极端事件数量）
        Step 5 → 统计检验与出图（秒级）

    【调试建议】
        若ERA5提取耗时过长，可先用少量TIF文件（如1个月=744张）测试
        其他步骤的逻辑是否正确，确认后再跑全量8760张。
    """
    try:
        print("=" * 65)
        print("第一阶段分析：极端干旱事件 vs 等长对照期 微气候缓冲研究")
        print("参考方法：Ma et al. (2025) & Zellweger et al. (2020)")
        print("=" * 65)

        # ── 读取样地坐标（站点编号统一为字符串整数）──
        sites = pd.read_csv(SITES_CSV)
        for c in [SITE_ID_COL, SITE_LON_COL, SITE_LAT_COL]:
            if c not in sites.columns:
                raise KeyError(
                    f"坐标文件缺少列 '{c}'，实际列名：{list(sites.columns)}\n"
                    f"请检查配置区的 SITE_ID_COL / SITE_LON_COL / SITE_LAT_COL"
                )
        sites = sites[[SITE_ID_COL, SITE_LON_COL, SITE_LAT_COL]].copy()
        sites.columns = ['site_id', 'lon', 'lat']
        sites['site_id'] = sites['site_id'].apply(normalize_site_id)
        print(f"\n站点信息：共 {len(sites)} 个站点（预期27个，缺95332241）")
        print(f"经度范围：{sites['lon'].min():.4f} ~ {sites['lon'].max():.4f}")
        print(f"纬度范围：{sites['lat'].min():.4f} ~ {sites['lat'].max():.4f}")

        # ── 读取干旱事件长表 ──
        drought_df = pd.read_csv(DROUGHT_CSV)
        validate_required_columns(
            drought_df,
            ['Site_ID', 'Start_Date', 'End_Date', 'Event_ID', 'Drought_Level_Code', 'Severity'],
            "干旱事件长表"
        )
        drought_df['site_id']    = drought_df['Site_ID'].apply(normalize_site_id)
        drought_df['Start_Date'] = pd.to_datetime(drought_df['Start_Date'])
        drought_df['End_Date']   = pd.to_datetime(drought_df['End_Date'])
        extreme_count = count_target_events(drought_df, TARGET_DROUGHT_LEVEL_CODE)
        print(f"\n干旱事件长表：共 {len(drought_df)} 条事件，"
              f"其中目标等级（Code={TARGET_DROUGHT_LEVEL_CODE}）{extreme_count} 条")

        # ── 读取逐日SPI序列表 ──
        daily_spi_df = load_daily_spi_table(DAILY_SPI_CSV)
        print(f"逐日SPI序列表：共 {len(daily_spi_df):,} 条记录，"
              f"{daily_spi_df['site_id'].nunique()} 个站点")

        # ── 执行各步骤 ──
        micro   = load_tomst(TOMST_DIR, TOMST_DATETIME_COL, TOMST_TEMP_COL, TOMST_TIME_FORMAT)
        macro   = extract_era5(ERA5_TIF_DIR, sites, ERA5_UNIT_K)
        merged  = merge_datasets(micro, macro)
        beta_tb, control_quality, site_coverage = compute_paired_betas(
            merged, drought_df, daily_spi_df, sites, gap_days=CONTROL_GAP_DAYS
        )
        analyze_and_plot(
            beta_tb, control_quality, site_coverage, OUTPUT_DIR,
            merged=merged, daily_spi_df=daily_spi_df
        )
    finally:
        cleanup_temp_dir(TEMP_DIR)


if __name__ == "__main__":
    main()
