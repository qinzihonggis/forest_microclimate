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
               r"\daily_SPI_features\福建省观测站2025年daily_SPI干旱事件长表.csv")

# 逐日SPI序列表路径
# 用途：只用于前/后对照期筛选。干旱期仍以 DROUGHT_CSV 中的事件起止日期为准。
# 对照期规则：先取等长窗口；若窗口内遇到干旱日，则截断，只保留最靠近目标事件的连续非干旱日，
# 不再向更远时间补足天数。
DAILY_SPI_CSV = (r"E:\forest_microclimate\ForestMicroclimate\results"
                 r"\daily_SPI_features\福建省观测站2025年daily_SPI逐日序列表.csv")

# 所有输出文件的保存文件夹（不存在会自动创建）
OUTPUT_DIR = r"E:\forest_microclimate\ForestMicroclimate\results\compare_differences"

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

# ── OLS 回归质量控制 ────────────────────────────────────────

# 计算β所需的最少有效逐小时数据点数
# 低于此值则该期间β设为NaN，不参与统计检验
# 设定依据：至少需要1整天(24h)的数据才能建立有意义的T_micro~T_macro线性关系
# 若某些站点数据缺失严重，可适当降低（如12），但会降低β估计的可靠性
MIN_HOURS_FOR_OLS = 24

# ── 逐日SPI表列名配置 ───────────────────────────────────────

DAILY_SPI_SITE_COL = "Site_ID"
DAILY_SPI_DATE_COL = "Date"
DAILY_SPI_VALUE_COL = "Daily_SPI_30d"
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
}

# ── 图形参数配置 ─────────────────────────────────────────────
# 后续调图优先修改这里，不需要到绘图函数内部查找参数。

FIG_SIZE = (7.5, 5.5)
FIG_DPI = 300
BOX_WIDTH = 0.45
BOX_LINEWIDTH = 1.2
BOX_COLORS = ['#2196F3', '#E53935', '#4CAF50']
LINE_COLORS = {'Pre': '#2196F3', 'Drought': '#E53935', 'Post': '#4CAF50'}
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

    截断规则：
      - 前置对照期：从 raw_end 往前看，保留最靠近目标事件的连续非干旱日；
      - 后置对照期：从 raw_start 往后看，保留最靠近目标事件的连续非干旱日；
      - 一旦遇到干旱日或缺少逐日SPI记录，就停止，不向更远处补天数。
    """
    all_days = pd.date_range(raw_start.normalize(), raw_end.normalize(), freq='D')
    raw_days = len(all_days)
    spi_map = site_daily_spi.set_index('date')

    if side == 'pre':
        scan_days = list(reversed(all_days))
    elif side == 'post':
        scan_days = list(all_days)
    else:
        raise ValueError("side 必须为 'pre' 或 'post'")

    kept_days = []
    stop_reason = "完整等长非干旱窗口"
    for day in scan_days:
        if day not in spi_map.index:
            stop_reason = f"遇到缺失逐日SPI记录：{day.date()}"
            break
        row = spi_map.loc[day]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if bool(row['is_drought_day']):
            stop_reason = f"遇到干旱日：{day.date()}"
            break
        kept_days.append(day)

    if side == 'pre':
        kept_days = list(reversed(kept_days))

    actual_days = len(kept_days)
    actual_start = kept_days[0] if kept_days else None
    actual_end = kept_days[-1] if kept_days else None
    full_length = actual_days == raw_days
    truncated = not full_length
    note = "等长窗口完整可用" if full_length else stop_reason

    return {
        'actual_start': actual_start,
        'actual_end': actual_end,
        'raw_days': raw_days,
        'actual_days': actual_days,
        'shortfall_days': raw_days - actual_days,
        'full_length': full_length,
        'truncated': truncated,
        'note': note,
    }


def validate_required_columns(df, required_cols, table_name):
    """统一检查输入表字段，缺列时立即给出明确错误。"""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{table_name} 缺少必要列：{missing}\n实际列名：{list(df.columns)}")


def sanitize_filename(text):
    """将站点编号/标题文本转换为Windows安全文件名。"""
    return re.sub(r'[\\\\/:*?"<>|]+', '_', str(text))


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
    print("\nStep 4: 计算极端干旱期 vs 等长对照期的 β（OLS缓冲指数）...")
    print(f"  筛选条件：Drought_Level_Code = {TARGET_DROUGHT_LEVEL_CODE}")
    print(f"  对照期设计：等长配对，缓冲间隔 {gap_days} 天")
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
        duration = (d_end - d_start).days + 1

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

        # ════════════════════════════════════════════════
        # 4b. 前置对照期β
        # 初始范围：[d_start - gap - duration, d_start - gap - 1]
        # 即：干旱开始前 gap 天作为缓冲，往前取等长天数
        # ════════════════════════════════════════════════
        pre_end_raw   = d_start - pd.Timedelta(days=gap_days + 1)
        pre_start_raw = pre_end_raw - pd.Timedelta(days=duration - 1)

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
            # 对照期被完全遮盖，无法使用
            beta_pre = dict(beta=np.nan, alpha=np.nan, r2=np.nan,
                            p_value=np.nan, n=0)

        # ════════════════════════════════════════════════
        # 4c. 后置对照期β
        # 初始范围：[d_end + gap + 1, d_end + gap + duration]
        # 即：干旱结束后 gap 天作为缓冲，往后取等长天数
        # ════════════════════════════════════════════════
        post_start_raw = d_end + pd.Timedelta(days=gap_days + 1)
        post_end_raw   = post_start_raw + pd.Timedelta(days=duration - 1)

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
            beta_post = dict(beta=np.nan, alpha=np.nan, r2=np.nan,
                             p_value=np.nan, n=0)

        # 汇总本次事件的所有结果
        results.append({
            'site_id'        : site_id,
            'event_id'       : ev_id,
            'drought_start'  : d_start.date(),
            'drought_end'    : d_end.date(),
            'duration_days'  : duration,
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
        })

        quality_records.append({
            '站点编号': site_id,
            '事件编号': ev_id,
            '干旱开始日期': d_start.date(),
            '干旱结束日期': d_end.date(),
            '干旱期天数': duration,
            '干旱期有效小时数': beta_d['n'],
            '前置原始开始日期': pre_start_raw.date(),
            '前置原始结束日期': pre_end_raw.date(),
            '前置原始天数': pre_info['raw_days'],
            '前置实际开始日期': pre_start.date() if pre_start is not None else None,
            '前置实际结束日期': pre_end.date() if pre_end is not None else None,
            '前置连续非干旱天数': pre_info['actual_days'],
            '前置不足天数': pre_info['shortfall_days'],
            '前置是否等长': pre_info['full_length'],
            '前置是否可计算β': beta_pre['n'] >= MIN_HOURS_FOR_OLS,
            '前置有效小时数': beta_pre['n'],
            '前置截断说明': pre_info['note'],
            '后置原始开始日期': post_start_raw.date(),
            '后置原始结束日期': post_end_raw.date(),
            '后置原始天数': post_info['raw_days'],
            '后置实际开始日期': post_start.date() if post_start is not None else None,
            '后置实际结束日期': post_end.date() if post_end is not None else None,
            '后置连续非干旱天数': post_info['actual_days'],
            '后置不足天数': post_info['shortfall_days'],
            '后置是否等长': post_info['full_length'],
            '后置是否可计算β': beta_post['n'] >= MIN_HOURS_FOR_OLS,
            '后置有效小时数': beta_post['n'],
            '后置截断说明': post_info['note'],
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
    n_trunc_p  = beta_table['pre_truncated'].sum()
    n_trunc_po = beta_table['post_truncated'].sum()
    if n_nan_d  > 0:
        print(f"  [警告] {n_nan_d} 条干旱期β为NaN（数据点<{MIN_HOURS_FOR_OLS}h，请检查TOMST/ERA5数据完整性）")
    if n_nan_p  > 0:
        print(f"  [提示] {n_nan_p} 条前置对照期β为NaN（被完全遮盖或数据不足）")
    if n_nan_po > 0:
        print(f"  [提示] {n_nan_po} 条后置对照期β为NaN（被完全遮盖或数据不足）")
    if n_trunc_p  > 0:
        print(f"  [提示] {n_trunc_p} 条前置对照期被部分截断（pre_truncated=True）")
    if n_trunc_po > 0:
        print(f"  [提示] {n_trunc_po} 条后置对照期被部分截断（post_truncated=True）")

    return beta_table, control_quality, site_coverage


# ============================================================
# Step 5：站点级聚合 + 统计检验 + 可视化
# ============================================================

def analyze_and_plot(beta_table, control_quality, site_coverage, output_dir):
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
    ensure_dir(overall_plot_dir)
    ensure_dir(site_plot_dir)

    if beta_table.empty:
        raise RuntimeError("事件级β结果表为空，无法进行站点聚合、统计检验和绘图。")

    # ── 5a. 保存事件级详细表 ──
    detail_path = os.path.join(output_dir, "事件级β配对结果表.csv")
    beta_table.to_csv(detail_path, index=False, encoding='utf-8-sig')
    print(f"  事件级详细表 → {detail_path}")

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
        beta_table.groupby('site_id')[['beta_drought', 'beta_pre', 'beta_post']]
        .median()
        .reset_index()
    )
    site_summary.columns = ['site_id', 'beta_drought', 'beta_pre', 'beta_post']

    if site_summary.empty:
        raise RuntimeError("站点级β汇总表为空，无法继续统计检验和绘图。")

    summary_path = os.path.join(output_dir, "站点级β汇总表.csv")
    site_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"  站点级汇总表 → {summary_path}")

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
    valid_pre  = site_summary.dropna(subset=['beta_drought', 'beta_pre'])
    valid_post = site_summary.dropna(subset=['beta_drought', 'beta_post'])

    report_pre  = do_wilcoxon(
        valid_pre['beta_drought'], valid_pre['beta_pre'],
        "干旱期 vs 前置等长对照期（Pre-drought control）"
    )
    report_post = do_wilcoxon(
        valid_post['beta_drought'], valid_post['beta_post'],
        "干旱期 vs 后置等长对照期（Post-drought control）"
    )

    report_lines = [
        "=" * 65,
        "极端干旱事件对林下微气候缓冲能力影响 — 统计检验报告",
        "=" * 65,
        "",
        "【方法说明】",
        "  对照期设计 ：等长配对（Length-matched paired control）",
        "    参考文献 ：Zellweger et al. (2020) Global Change Biology",
        "               De Frenne et al. (2021) Nature Ecology & Evolution",
        "  β计算方法  ：OLS回归 T_micro = α + β × T_macro",
        "    参考文献 ：Ma et al. (2025) Agricultural and Forest Meteorology",
        "  站点聚合   ：每站点取所有极端事件β的中位数（n=27个独立站点）",
        "  统计检验   ：Wilcoxon符号秩检验（双侧，配对非参数）",
        "    参考文献 ：Hollander & Wolfe (1999) Nonparametric Statistical Methods",
        "  效应量     ：Cohen's d（干旱期 − 对照期）",
        "",
        "【β含义】",
        "  β越小（接近0）→ 缓冲能力越强（林内温度更稳定）",
        "  β越大（接近1）→ 缓冲能力越弱（林内温度更随宏气候波动）",
        "  β>1           → 放大效应（林内温度波动超过宏气候）",
        "",
        "=" * 65,
        "【检验结果】",
        "",
        report_pre,
        "",
        report_post,
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

    # 构建长格式DataFrame用于seaborn绘图。逐组dropna但保留site_id，避免不同组长度不同导致报错。
    pre_vals = site_summary[['site_id', 'beta_pre']].dropna().rename(columns={'beta_pre': 'beta'})
    pre_vals['Period'] = order[0]
    drought_vals = site_summary[['site_id', 'beta_drought']].dropna().rename(columns={'beta_drought': 'beta'})
    drought_vals['Period'] = order[1]
    post_vals = site_summary[['site_id', 'beta_post']].dropna().rename(columns={'beta_post': 'beta'})
    post_vals['Period'] = order[2]
    plot_long = pd.concat([pre_vals, drought_vals, post_vals], ignore_index=True)

    if plot_long.empty:
        raise RuntimeError("绘图数据为空：三组β均无有效值。请检查对照期质量记录表和事件级β结果表。")

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # 箱线图：显示分布四分位数
    sns.boxplot(
        data=plot_long, x='Period', y='beta', order=order,
        palette=colors, width=BOX_WIDTH, linewidth=BOX_LINEWIDTH,
        flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5),
        ax=ax
    )
    # 叠加散点：展示每个站点的实际值
    sns.stripplot(
        data=plot_long, x='Period', y='beta', order=order,
        color=POINT_COLOR, size=POINT_SIZE, alpha=POINT_ALPHA, jitter=POINT_JITTER, ax=ax
    )

    # 显著性标注（干旱 vs 前置、干旱 vs 后置）
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

    # β=1 reference line（no buffering baseline）
    ax.axhline(REFERENCE_LINE_Y, color=REFERENCE_LINE_COLOR, linestyle=REFERENCE_LINE_STYLE,
               alpha=REFERENCE_LINE_ALPHA, linewidth=1)
    ax.text(2.48, 1.002, 'β=1\n(No buffering)', fontsize=8, color='gray', va='bottom', ha='right')

    # Add sample size labels
    for i, col in enumerate(['beta_pre', 'beta_drought', 'beta_post']):
        n_valid = site_summary[col].notna().sum()
        ax.text(i, plot_long['beta'].dropna().min() - offset * 0.8,
                f"n={n_valid}", ha='center', fontsize=9, color=SAMPLE_N_COLOR)

    ax.set_xlabel("", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Climate Buffering Index (β)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(
        "Overall Comparison of Climate Buffering Index\n"
        "Pre-control vs Drought vs Post-control (site-level summary)",
        fontsize=TITLE_FONTSIZE, pad=12
    )

    plt.tight_layout()
    box_path = os.path.join(overall_plot_dir, "总体_β分组箱线图.png")
    plt.savefig(box_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"  总体箱线图 → {box_path}")

    # ── 5e. Overall delta-beta plot ──
    delta_pre = beta_table[['site_id', 'event_id', 'drought_start', 'beta_drought', 'beta_pre']].dropna().copy()
    delta_pre['Comparison'] = 'Drought - Pre'
    delta_pre['delta_beta'] = delta_pre['beta_drought'] - delta_pre['beta_pre']
    delta_post = beta_table[['site_id', 'event_id', 'drought_start', 'beta_drought', 'beta_post']].dropna().copy()
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

    # ── 5f. Site-level plots: paired line plot and event bar plot ──
    site_event_long = build_site_event_long(beta_table)
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

        # Event bar plot
        fig, ax = plt.subplots(figsize=SITE_FIG_SIZE)
        for event_idx in event_order:
            event_df = site_plot_df[site_plot_df['event_order'] == event_idx]
            base_x = event_idx
            for period, offset in period_offsets.items():
                row = event_df[event_df['period'] == period]
                if row.empty or pd.isna(row['beta'].iloc[0]):
                    continue
                ax.bar(
                    base_x + offset,
                    row['beta'].iloc[0],
                    width=BAR_WIDTH,
                    color=LINE_COLORS[period],
                    alpha=0.9,
                    edgecolor='black',
                    linewidth=0.6,
                    label=period if event_idx == event_order[0] else None,
                )

        ax.axhline(REFERENCE_LINE_Y, color=REFERENCE_LINE_COLOR, linestyle=REFERENCE_LINE_STYLE,
                   alpha=REFERENCE_LINE_ALPHA, linewidth=1)
        ax.set_xticks(event_order)
        ax.set_xticklabels([site_plot_df[site_plot_df['event_order'] == i]['event_label'].iloc[0] for i in event_order], fontsize=9)
        ax.set_ylabel("Climate Buffering Index (β)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xlabel("Extreme Drought Events (chronological order)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(f"Site {site_id}: Event-level β Bars", fontsize=TITLE_FONTSIZE, pad=10)
        ax.legend(title="Period", fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE, loc='best')
        plt.tight_layout()
        bar_path = os.path.join(site_plot_dir, f"站点_{sanitize_filename(site_id)}_事件柱状图.png")
        plt.savefig(bar_path, dpi=FIG_DPI, bbox_inches='tight')
        plt.close()

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
        analyze_and_plot(beta_tb, control_quality, site_coverage, OUTPUT_DIR)
    finally:
        cleanup_temp_dir(TEMP_DIR)


if __name__ == "__main__":
    main()
