# -*- coding: utf-8 -*-
"""
极端干旱森林微气候缓冲能力 Why 阶段：解释 Delta_CBI 空间-事件异质性
================================================================================

一、脚本目标
------------
本脚本承接已经完成的 What 阶段结果，不重新识别干旱事件、不重新计算 CBI、
不重新选择正常参考期，而是直接复用 What 阶段已经通过质量控制的事件-参考期
配对结果，回答第二部分 Why 问题：

    为什么同样经历极端干旱后，不同站点、不同干旱事件的森林微气候缓冲
    能力变化幅度和方向不同？

Why 阶段的唯一核心响应变量是事件尺度缓冲变化：

    Delta_CBI = Event_CBI - Reference_CBI

其中 Delta_CBI > 0 表示极端干旱事件期 CBI 高于事件后正常参考期，即 15 cm
林下温度更随 2 m 宏气温同步变化，缓冲能力减弱；Delta_CBI < 0 表示事件期
缓冲相对增强或维持。脚本使用的分析单位固定为：

    一行 = 一个通过 Pair_flag=ok 质量控制的“站点 x 极端干旱事件”配对。

二、研究设计
------------
完整研究故事分为 What、Why、How 三层：

1. What：已经完成。检验 Extreme 相对 Normal 的 CBI 是否总体改变。
2. Why：本脚本负责。以站点 x 事件的 Delta_CBI 为响应，解释哪些干旱暴露、
   大气需水、能量、水分供给、冠层、地形和土壤过程条件对应更大的缓冲衰减。
3. How：后续阶段。只把 Why 阶段筛选后仍稳健、有机制意义的变量带入路径模型
   或分段 SEM，分解干旱经由土壤水分、表层热过程、冠层状态影响 Delta_CBI
   的直接、间接和总效应。

三、因果与解释边界
------------------
本脚本严格区分三类变量：

1. 上游暴露/背景变量：
   - Drought_Intensity = -Min_Daily_SPI，数值越大代表干旱越强。
   - Duration_Days，表示干旱历时。
   - PET_Anomaly、Srad_Anomaly、Precip_Anomaly，表示事件期相对参考期的大气
     需水、能量输入和降水供给变化。
   - LAI_Anomaly、FAPAR_Anomaly，表示事件期相对参考期的冠层状态变化。
   - Elevation、Slope、Northness、Eastness、Canopy_Height，表示稳定站点背景。

2. 过程/潜在中介变量：
   - SoilMoisture_Anomaly 使用 MicroTandSoilT.csv 中原始 VWC_Daily。
   - T0_Anomaly 使用原始传感器 T2_0，表示空气中 0 cm 的林下大气温度变化。
   这些变量很可能位于“干旱 -> 土壤/热过程 -> CBI 变化”的路径上，因此在
   Why 阶段只作为过程调整模型或机制补充模型，不与上游变量混合后宣称总效应。

3. CBI 原始输入审计变量：
   - T15_Anomaly 使用原始传感器 T3_15 聚合，仅用于过程描述和质量审计。
   - 因为 T3_15 已经用于计算 CBI，所以禁止作为 Why 主模型解释变量。

四、时间窗口原则
----------------
动态变量必须与 Delta_CBI 使用同一事件窗口和同一批实际选中的参考日期：

1. 事件期：直接使用 04_极端事件CBI与事件后正常参考期对比表.csv 中的
   Start_Date 至 End_Date。
2. 参考期：直接使用 04_事件后参考期候选日期审计表.csv 中
   Selected_as_Reference=True 的 Candidate_Date。
3. 日尺度宽表变量 PET、Srad、Precip、T2m：按日期和站点取窗口均值。
4. 8 日尺度 LAI/FAPAR：默认把每个产品日期视为 8 日窗口起始日，展开为日尺度
   后按事件/参考日期取均值，相当于按重叠天数加权。
5. 原始 15 分钟传感器 T2_0/T3_15：使用 UTC 时间列 data_time 生成 UTC_Date，
   聚合为站点逐日均值后再做事件/参考期汇总。这个 UTC 口径与 What 阶段一致。

五、模型策略
------------
当前有效事件结构为小样本事件级数据，且多数站点只有 1 个事件。因此脚本采用
“极简、预注册式候选模型 + 审计 + 稳健性”的策略，不做全变量逐步筛选：

1. 先输出样本结构、缺失率、每站点事件数、参考期天数、CBI 拟合 R2 审计。
2. 对候选变量做 Spearman 相关矩阵。
3. 对每个候选模型分别计算 VIF，避免把所有变量一次性完整案例删除后误判。
4. 候选模型使用共同完整样本计算 AICc；最大可用样本结果作为补充。
5. 主分析采用站点级聚合 OLS/HC3 稳健标准误模型，解释站点平均 Delta_CBI
   的空间差异，避免少数多事件站点在事件级模型中获得过高权重。
6. 事件级 OLS/HC3 保留为补充稳健模型，回答 27 个站点 x 事件配对中的环境关联。
7. 随机截距 LMM 仅作为诊断/补充尝试；若 LMM 不收敛、随机效应方差接近 0
   或矩阵奇异，则明确标记为“不作为主证据”，不强行解释其系数。

六、主要输出
------------
所有输出默认写入：

    E:\\forest_microclimate\\ForestMicroclimate\\results\\Modeling_Explaining

文件均使用中文命名，主要包括：

1. 输入路径审计、参数说明、绘图与进度条参数说明。
2. 有效事件响应表、参考期日期表、事件环境协变量表。
3. 建模主表、缺失率与样本结构审计、相关矩阵、VIF 表。
4. 站点级主分析模型 AICc/R2 和系数表。
5. 事件级补充模型与 LMM 随机效应诊断表。
6. 站点级主模型标准化系数森林图、观测-预测图、残差诊断图。
7. 逐站点剔除、替代变量敏感性结果。
7. 运行摘要和本次运行缓存清理记录。

七、运行方式
------------
直接运行本脚本即可：

    python 建模_解释空间差异.py

运行前通常只需要检查 CONFIG 中的路径、列名和候选模型参数。脚本会显示 tqdm
单行动态彩色进度条，包含百分比、当前量/总量、耗时、剩余时间和速度。每个关键
步骤只显示一个进度条，避免刷屏。

八、重要注意事项
----------------
1. 不要把 15 cm 林下温度 T3_15 作为解释 Delta_CBI 的主模型自变量。
2. 不要把 LAI 与 FAPAR、PET 与 Srad、SPI 与 Precip、Intensity 与 Severity
   同时塞进同一个主模型。
3. 标准化系数只用于同一模型内比较相对关联强弱，不能写成因果权重。
4. AICc 只应在共同完整样本下比较；最大可用样本模型不能直接按 AICc 机械排序。
5. 本脚本提供 Why 阶段环境关联证据；正式因果路径留给 How 阶段。
"""

from __future__ import annotations

import shutil
import tempfile
import time
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# 0. 全局路径、字段、建模与绘图参数
# =============================================================================


RUN_START_TIME = time.time()


@dataclass(frozen=True)
class Config:
    """集中管理路径和关键列名。

    修改建议：
    - 如果上游 What 阶段输出目录改变，只需要修改 event_pair_csv 和 reference_audit_csv。
    - 如果环境宽表路径改变，只需要修改对应变量路径。
    - 如果传感器列名改变，只需要修改 sensor_*_col。
    """

    project_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate")
    output_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\results\Modeling_Explaining")

    event_pair_csv: Path = Path(
        r"E:\forest_microclimate\ForestMicroclimate\results\compare_differences_results\04_极端事件CBI与事件后正常参考期对比表.csv"
    )
    reference_audit_csv: Path = Path(
        r"E:\forest_microclimate\ForestMicroclimate\results\compare_differences_results\04_事件后参考期候选日期审计表.csv"
    )

    static_site_csv: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\Tensor_LatLong.csv")
    fapar_csv: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\FAPAR\站点FAPAR_8日尺度提取结果.csv")
    lai_csv: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\LAI\站点LAI_8日尺度提取结果.csv")
    pet_csv: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\PET_Estimate_era5\站点PET逐日时间序列.csv")
    srad_csv: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\Srad\站点Srad逐日累计提取结果.csv")
    precip_csv: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\Precipitation_CHIRPS\站点CHIRPS逐日降雨提取结果.csv")
    t2m_csv: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\T2m\fujian_T2\站点2米气温逐日平均提取结果.csv")
    micro_soil_csv: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\MicroTandSoilT.csv")
    tensor_data_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\Tensor_Data")

    site_col: str = "Site_ID"
    event_col: str = "Event_ID"
    date_col: str = "datetime"
    valid_pair_value: str = "ok"

    sensor_utc_time_col: str = "data_time"
    sensor_t0_col: str = "T2_0"
    sensor_t15_col: str = "T3_15"
    soil_date_col: str = "Date"
    soil_vwc_col: str = "VWC_Daily"
    soil_tminus5_col: str = "T-5cm_Daily"

    min_sensor_records_per_day: int = 48
    min_window_valid_days: int = 3
    lai_fapar_window_days: int = 8
    srad_scale_to_mj_m2_day: float = 1e-6
    vif_threshold: float = 5.0
    correlation_threshold: float = 0.70
    near_zero_random_effect_threshold: float = 1e-8
    random_seed: int = 20260721


CFG = Config()
RUNTIME_CACHE_DIR = CFG.output_dir / f"_本次运行临时缓存_{int(RUN_START_TIME)}"


OUTPUT_FILES = {
    "path_audit": "00_输入路径审计表.csv",
    "parameter_table": "00_建模参数_绘图参数_进度条说明表.csv",
    "event_response": "01_有效事件Delta_CBI响应表.csv",
    "reference_dates": "02_实际选中参考期日期表.csv",
    "covariates": "03_事件环境协变量表.csv",
    "master": "04_建模主表_含标准化变量.csv",
    "missing_audit": "05_变量缺失率与唯一值审计表.csv",
    "sample_audit": "05_样本结构审计表.csv",
    "correlation": "06_Spearman候选变量相关矩阵.csv",
    "high_correlation": "06_高相关变量对审计表.csv",
    "vif": "07_候选模型VIF审计表.csv",
    "site_master": "08_站点级聚合主分析表.csv",
    "site_model_comparison": "09_站点级主分析模型比较_AICc_R2.csv",
    "site_coefficients": "10_站点级主分析系数表.csv",
    "event_model_comparison": "11_事件级补充模型比较_AICc_R2_收敛诊断.csv",
    "event_coefficients": "12_事件级补充模型系数表.csv",
    "lmm_diagnostic": "13_LMM随机效应诊断表.csv",
    "final_forest_png": "14_站点级主模型标准化系数森林图.png",
    "final_forest_pdf": "14_站点级主模型标准化系数森林图.pdf",
    "pred_obs_png": "15_站点级主模型观测值与预测值对照图.png",
    "pred_obs_pdf": "15_站点级主模型观测值与预测值对照图.pdf",
    "residual_png": "16_站点级主模型残差诊断图.png",
    "residual_pdf": "16_站点级主模型残差诊断图.pdf",
    "leave_one_site": "17_逐站点剔除敏感性分析表.csv",
    "alternative_models": "18_替代变量敏感性模型表.csv",
    "run_summary_csv": "20_运行摘要表.csv",
    "run_summary_txt": "20_运行摘要说明.txt",
    "cache_cleanup": "19_本次运行缓存清理记录.csv",
}


# 进度条参数：
# - leave=False 让每个步骤完成后清除旧进度条，避免日志刷屏。
# - colour 控制不同任务类型的颜色；旧版 tqdm 不支持 colour 时会自动降级。
# - bar_format 显示百分比、当前量/总量、耗时、剩余时间和速度。
PROGRESS_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
)
PROGRESS_BAR_CONFIG = {
    "读取": {"colour": "cyan", "unit": "项"},
    "整理": {"colour": "green", "unit": "项"},
    "汇总": {"colour": "yellow", "unit": "事件"},
    "建模": {"colour": "magenta", "unit": "模型"},
    "绘图": {"colour": "red", "unit": "图"},
    "输出": {"colour": "blue", "unit": "文件"},
    "清理": {"colour": "white", "unit": "项"},
}


# 绘图参数集中放在这里，并会输出到参数说明表。
PLOT_STYLE = {
    "figure_dpi": 300,
    "forest_width": 8.0,
    "forest_height": 5.8,
    "diagnostic_width": 10.0,
    "diagnostic_height": 4.6,
    "point_color": "#1f6f8b",
    "ci_color": "#173f5f",
    "zero_line_color": "#444444",
    "pred_point_color": "#2a9d8f",
    "residual_point_color": "#e76f51",
    "point_alpha": 0.82,
    "point_size": 46,
    "line_width": 1.8,
    "annotation_fontsize": 10,
    "title_fontsize": 15,
    "axis_label_fontsize": 12,
    "significant_color": "#0f5c75",
    "nonsignificant_color": "#9aa6ac",
    "weak_effect_band": 0.02,
    "weak_effect_band_color": "#eef3f5",
    "outlier_label_count": 3,
    "font_family": "SimHei",
    "font_file_candidates": (
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\Deng.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
    ),
}

# Matplotlib 的全局中文字体对象。脚本启动时由 setup_plot_style() 初始化，
# 保存图件前再由 apply_chinese_font_to_figure() 强制应用到整张图，避免 seaborn、
# scipy.stats.probplot 或 Matplotlib 默认文本对象没有继承中文字体而出现方框乱码。
CHINESE_FONT_PROP = None
CHINESE_FONT_NAME = ""
CHINESE_FONT_FILE = ""


# 绘图中文标签映射。模型仍使用英文变量名，图件输出使用中文标签，避免读图时再回查字段含义。
VARIABLE_LABELS = {
    "Drought_Intensity": "干旱强度",
    "Duration_Days": "干旱历时",
    "Severity": "干旱严重度",
    "Min_Daily_SPI": "最低日SPI",
    "PET_Anomaly": "PET异常",
    "Srad_Anomaly": "太阳辐射异常",
    "Precip_Anomaly": "降水异常",
    "T2m_Anomaly": "2米气温异常",
    "LAI_Anomaly": "LAI异常",
    "FAPAR_Anomaly": "FAPAR异常",
    "SoilMoisture_Anomaly": "土壤含水量异常",
    "T0_Anomaly": "0cm林下气温异常",
    "T15_AUDIT_ONLY": "15cm林下气温审计",
    "Canopy_Height": "冠层高度",
    "Elevation": "海拔",
    "Slope": "坡度",
    "Northness": "坡向北向分量",
    "Eastness": "坡向东向分量",
}


# 候选模型说明：
# - 主模型优先控制变量数量，避免 27 个事件上过拟合。
# - dynamic/process 变量会用标准化版本拟合，便于系数比较。
# - M5 加入土壤水分，是过程调整模型，不解释为总效应模型。
BASE_CANDIDATE_MODELS = {
    "M0_月份空模型": {
        "terms": ["C(Event_Month)"],
        "role": "基线模型；估计月份控制和站点随机截距后的剩余变异。",
    },
    "M1_干旱暴露": {
        "terms": ["z_Drought_Intensity", "z_Duration_Days"],
        "role": "检验更强、更久的极端干旱是否对应更大的 Delta_CBI。",
    },
    "M2_干旱加PET": {
        "terms": ["z_Drought_Intensity", "z_Duration_Days", "z_PET_Anomaly"],
        "role": "检验大气蒸散需求异常是否提供额外解释。",
    },
    "M3_冠层结构简化": {
        "terms": ["z_Drought_Intensity", "z_PET_Anomaly", "z_LAI_Anomaly", "z_Canopy_Height"],
        "role": "检验短期冠层状态和长期结构是否解释响应差异。",
    },
    "M4_地形背景简化": {
        "terms": ["z_Drought_Intensity", "z_PET_Anomaly", "z_LAI_Anomaly", "z_Elevation", "z_Northness"],
        "role": "检验地形背景在干旱、需水和冠层之外是否仍有关联。",
    },
    "M5_土壤过程调整": {
        "terms": ["z_Drought_Intensity", "z_PET_Anomaly", "z_LAI_Anomaly", "z_SoilMoisture_Anomaly"],
        "role": "过程调整模型；检验实测 VWC 失水是否与 Delta_CBI 相关。",
    },
    "M6_表层热过程补充": {
        "terms": ["z_Drought_Intensity", "z_SoilMoisture_Anomaly", "z_T0_Anomaly"],
        "role": "机制补充模型；检验 0 cm 林下空气温度异常是否同步变化。",
    },
}


SITE_LEVEL_MODELS = {
    "S0_站点空模型": {
        "terms": [],
        "role": "站点级截距模型；作为空间差异解释的基线。",
    },
    "S1_干旱暴露空间模型": {
        "terms": ["z_Drought_Intensity", "z_Duration_Days"],
        "role": "站点平均响应是否随平均干旱强度和历时变化。",
    },
    "S2_大气需水空间模型": {
        "terms": ["z_Drought_Intensity", "z_PET_Anomaly"],
        "role": "站点平均响应是否与干旱强度和 PET 异常相关。",
    },
    "S3_冠层结构空间模型": {
        "terms": ["z_Drought_Intensity", "z_LAI_Anomaly", "z_Canopy_Height"],
        "role": "站点平均响应是否与冠层状态变化和长期冠层高度相关。",
    },
    "S4_土壤过程空间模型": {
        "terms": ["z_Drought_Intensity", "z_SoilMoisture_Anomaly", "z_T0_Anomaly"],
        "role": "站点平均响应是否与 VWC 变化和 0 cm 林下空气温度变化相关。",
    },
}


ALTERNATIVE_MODELS = {
    "A1_Severity替代强度历时": ["z_Severity", "z_PET_Anomaly", "z_LAI_Anomaly"],
    "A2_Srad替代PET": ["z_Drought_Intensity", "z_Duration_Days", "z_Srad_Anomaly", "z_LAI_Anomaly"],
    "A3_FAPAR替代LAI": ["z_Drought_Intensity", "z_PET_Anomaly", "z_FAPAR_Anomaly", "z_Canopy_Height"],
    "A4_降水供给替代SPI": ["z_Duration_Days", "z_Precip_Anomaly", "z_PET_Anomaly", "z_LAI_Anomaly"],
}


CONTINUOUS_VARIABLES = [
    "Drought_Intensity",
    "Duration_Days",
    "Severity",
    "Min_Daily_SPI",
    "PET_Event",
    "PET_Reference",
    "PET_Anomaly",
    "Srad_Event",
    "Srad_Reference",
    "Srad_Anomaly",
    "Precip_Event",
    "Precip_Reference",
    "Precip_Anomaly",
    "T2m_Event",
    "T2m_Reference",
    "T2m_Anomaly",
    "LAI_Event",
    "LAI_Reference",
    "LAI_Anomaly",
    "FAPAR_Event",
    "FAPAR_Reference",
    "FAPAR_Anomaly",
    "SoilMoisture_Event",
    "SoilMoisture_Reference",
    "SoilMoisture_Anomaly",
    "Tminus5_Event",
    "Tminus5_Reference",
    "Tminus5_Anomaly",
    "T0_Event",
    "T0_Reference",
    "T0_Anomaly",
    "T15_AUDIT_ONLY_Event",
    "T15_AUDIT_ONLY_Reference",
    "T15_AUDIT_ONLY_Anomaly",
    "Elevation",
    "Slope",
    "Aspect",
    "Northness",
    "Eastness",
    "Canopy_Height",
]


# =============================================================================
# 1. 通用工具函数
# =============================================================================


def progress_bar(total: int, desc: str, kind: str) -> tqdm:
    """创建单行动态彩色 tqdm 进度条。

    参数说明：
    - total：当前步骤的总任务数。
    - desc：进度条左侧显示的步骤名称。
    - kind：进度条类型，决定颜色和计量单位。
    """

    cfg = PROGRESS_BAR_CONFIG.get(kind, PROGRESS_BAR_CONFIG["整理"])
    kwargs = {
        "total": total,
        "desc": desc,
        "unit": cfg["unit"],
        "leave": False,
        "dynamic_ncols": True,
        "bar_format": PROGRESS_BAR_FORMAT,
    }
    try:
        return tqdm(**kwargs, colour=cfg["colour"])
    except TypeError:
        return tqdm(**kwargs)


def ensure_output_dir() -> None:
    """创建输出目录和本次运行临时缓存目录。"""

    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    RUNTIME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(RUNTIME_CACHE_DIR)


def normalise_site_id(series: pd.Series) -> pd.Series:
    """统一站点编号，避免 95332217 与 95332217.0 合并失败。"""

    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def parse_date_series(series: pd.Series) -> pd.Series:
    """把日期列转为标准化日期；无法解析的值返回 NaT。"""

    return pd.to_datetime(series, errors="coerce").dt.normalize()


def require_columns(df: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    """检查表格是否包含必须字段，缺字段时立即报错。"""

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"{table_name} 缺少必要字段：{missing}；现有字段：{df.columns.tolist()}")


def read_csv(path: Path, table_name: str, **kwargs) -> pd.DataFrame:
    """读取 CSV，并在路径不存在时给出清楚错误。"""

    if not path.exists():
        raise FileNotFoundError(f"{table_name} 不存在：{path}")
    return pd.read_csv(path, low_memory=False, **kwargs)


def write_csv(df: pd.DataFrame, filename: str) -> Path:
    """写出 UTF-8-SIG CSV，文件名统一由 OUTPUT_FILES 提供中文名称。"""

    path = CFG.output_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_text(text: str, filename: str) -> Path:
    """写出中文说明文本。"""

    path = CFG.output_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def zscore(series: pd.Series) -> pd.Series:
    """对连续变量做 Z 标准化；缺失值保留为缺失，不做隐式插补。"""

    x = pd.to_numeric(series, errors="coerce")
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=series.index)
    return (x - x.mean()) / sd


def aicc(aic: float, n_obs: int, k_params: int) -> float:
    """计算小样本修正 AICc。

    AICc = AIC + 2k(k+1)/(n-k-1)。当 n <= k+1 时无法稳定计算，返回 NaN。
    """

    if not np.isfinite(aic) or n_obs <= k_params + 1:
        return np.nan
    return aic + (2 * k_params * (k_params + 1)) / (n_obs - k_params - 1)


def mixed_r2(result) -> tuple[float, float]:
    """计算 Nakagawa 风格近似边际 R2 和条件 R2。"""

    fixed = np.asarray(result.model.exog) @ np.asarray(result.fe_params)
    var_fixed = float(np.var(fixed, ddof=1)) if len(fixed) > 1 else np.nan
    try:
        var_random = float(np.trace(np.asarray(result.cov_re)))
    except Exception:
        var_random = 0.0
    var_resid = float(result.scale)
    total = var_fixed + var_random + var_resid
    if not np.isfinite(total) or total <= 0:
        return np.nan, np.nan
    return var_fixed / total, (var_fixed + var_random) / total


def ols_r2(result) -> tuple[float, float]:
    """返回 OLS R2 与调整 R2，用于小样本补充模型。"""

    return float(result.rsquared), float(result.rsquared_adj)


# =============================================================================
# 2. 输入审计与参数输出
# =============================================================================


def build_path_audit() -> pd.DataFrame:
    """检查所有输入路径是否存在，并记录输出目录。"""

    rows = []
    for field in fields(CFG):
        value = getattr(CFG, field.name)
        if isinstance(value, Path):
            rows.append({
                "参数名": field.name,
                "路径": str(value),
                "是否存在": value.exists(),
                "类型": "目录" if value.exists() and value.is_dir() else "文件",
            })
    return pd.DataFrame(rows)


def build_parameter_table() -> pd.DataFrame:
    """导出建模、绘图和进度条参数，方便后续调整图例、颜色和线型。"""

    rows = []
    for name, value in Config().__dict__.items():
        rows.append({
            "参数类别": "路径与字段",
            "参数名": name,
            "当前值": str(value),
            "用途说明": "输入输出路径、关键列名或质量控制阈值。",
        })
    for name, value in PLOT_STYLE.items():
        rows.append({
            "参数类别": "绘图",
            "参数名": name,
            "当前值": str(value),
            "用途说明": "控制图件尺寸、DPI、颜色、点大小、透明度、线宽或字体。",
        })
    rows.extend([
        {
            "参数类别": "绘图",
            "参数名": "实际使用中文字体名",
            "当前值": CHINESE_FONT_NAME or "未初始化；main() 中 setup_plot_style() 后会更新",
            "用途说明": "脚本实际注册并传给 Matplotlib 的中文字体名称，用于排查中文乱码。",
        },
        {
            "参数类别": "绘图",
            "参数名": "实际使用中文字体文件",
            "当前值": CHINESE_FONT_FILE or "未找到字体文件；将依赖系统字体名称匹配",
            "用途说明": "脚本实际使用的字体文件路径；若图片中文为方框，优先检查该文件是否存在。",
        },
    ])
    for kind, cfg in PROGRESS_BAR_CONFIG.items():
        rows.append({
            "参数类别": "进度条",
            "参数名": f"{kind}_进度条",
            "当前值": str(cfg),
            "用途说明": "控制 tqdm 单行动态进度条的颜色和计量单位。",
        })
    for model_id, meta in BASE_CANDIDATE_MODELS.items():
        rows.append({
            "参数类别": "事件级补充候选模型",
            "参数名": model_id,
            "当前值": " + ".join(meta["terms"]),
            "用途说明": meta["role"],
        })
    for model_id, meta in SITE_LEVEL_MODELS.items():
        rows.append({
            "参数类别": "站点级主分析候选模型",
            "参数名": model_id,
            "当前值": " + ".join(meta["terms"]) if meta["terms"] else "1",
            "用途说明": meta["role"],
        })
    return pd.DataFrame(rows)


# =============================================================================
# 3. 构建 Why 阶段响应表和参考期日期表
# =============================================================================


def prepare_event_response() -> pd.DataFrame:
    """读取 What 阶段事件配对表，只保留 Pair_flag=ok 的有效事件。

    关键输出字段：
    - Delta_CBI：事件期 CBI - 参考期 CBI。
    - Drought_Intensity：-Min_Daily_SPI，正值越大代表干旱越强。
    - Event_Month：事件开始月份，用于季节控制或审计。
    """

    df = read_csv(CFG.event_pair_csv, "What阶段事件-参考期CBI配对表")
    require_columns(
        df,
        [
            CFG.site_col,
            CFG.event_col,
            "Start_Date",
            "End_Date",
            "Duration_Days",
            "Severity",
            "Min_Daily_SPI",
            "Event_CBI",
            "Reference_CBI",
            "Pair_flag",
        ],
        "What阶段事件-参考期CBI配对表",
    )
    df = df.copy()
    df[CFG.site_col] = normalise_site_id(df[CFG.site_col])
    df[CFG.event_col] = df[CFG.event_col].astype(str)
    df["Start_Date"] = parse_date_series(df["Start_Date"])
    df["End_Date"] = parse_date_series(df["End_Date"])
    df = df.loc[df["Pair_flag"].astype(str).str.lower().eq(CFG.valid_pair_value)].copy()

    if "Delta_CBI_Event_minus_Reference" in df.columns:
        df["Delta_CBI"] = pd.to_numeric(df["Delta_CBI_Event_minus_Reference"], errors="coerce")
    else:
        df["Delta_CBI"] = pd.to_numeric(df["Event_CBI"], errors="coerce") - pd.to_numeric(
            df["Reference_CBI"], errors="coerce"
        )

    df["Drought_Intensity"] = -pd.to_numeric(df["Min_Daily_SPI"], errors="coerce")
    df["Duration_Days"] = pd.to_numeric(df["Duration_Days"], errors="coerce")
    df["Severity"] = pd.to_numeric(df["Severity"], errors="coerce")
    df["Event_Month"] = df["Start_Date"].dt.month
    df["Event_YearMonth"] = df["Start_Date"].dt.to_period("M").astype(str)

    df = df.dropna(subset=["Delta_CBI", "Start_Date", "End_Date"])
    df["Event_Month"] = df["Event_Month"].astype(int)
    if df.duplicated([CFG.site_col, CFG.event_col]).any():
        dup = df.loc[df.duplicated([CFG.site_col, CFG.event_col], keep=False), [CFG.site_col, CFG.event_col]]
        raise ValueError(f"有效事件响应表存在重复 Site_ID + Event_ID：\n{dup}")

    return df


def prepare_reference_dates(valid_events: pd.DataFrame) -> pd.DataFrame:
    """读取参考期候选日期审计表，只保留实际选中的参考日期。"""

    ref = read_csv(CFG.reference_audit_csv, "事件后参考期候选日期审计表")
    require_columns(
        ref,
        [CFG.site_col, CFG.event_col, "Candidate_Date", "Selected_as_Reference"],
        "事件后参考期候选日期审计表",
    )
    ref = ref.copy()
    ref[CFG.site_col] = normalise_site_id(ref[CFG.site_col])
    ref[CFG.event_col] = ref[CFG.event_col].astype(str)
    ref["Candidate_Date"] = parse_date_series(ref["Candidate_Date"])

    selected = ref.loc[ref["Selected_as_Reference"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    selected = selected.dropna(subset=["Candidate_Date"])
    selected = selected.merge(
        valid_events[[CFG.site_col, CFG.event_col]],
        on=[CFG.site_col, CFG.event_col],
        how="inner",
        validate="many_to_one",
    )
    return selected


# =============================================================================
# 4. 读取和汇总环境协变量
# =============================================================================


def load_wide_daily_table(path: Path, value_name: str) -> pd.DataFrame:
    """读取 datetime + 站点列的宽表，转为 Site_ID + Date + value 的长表。

    适用数据：
    - PET、Srad、Precip、T2m 等日尺度站点宽表。
    """

    wide = read_csv(path, value_name)
    require_columns(wide, [CFG.date_col], value_name)
    wide = wide.copy()
    wide[CFG.date_col] = parse_date_series(wide[CFG.date_col])
    long = wide.melt(id_vars=CFG.date_col, var_name=CFG.site_col, value_name=value_name)
    long[CFG.site_col] = normalise_site_id(long[CFG.site_col])
    long = long.rename(columns={CFG.date_col: "Date"})
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    if value_name == "Srad":
        # ERA5-Land ssrd 常见单位为 J/m2/day；转为 MJ/m2/day 后系数更易解释。
        long[value_name] = long[value_name] * CFG.srad_scale_to_mj_m2_day
    return long.dropna(subset=["Date"])


def load_8day_table_as_daily(path: Path, value_name: str) -> pd.DataFrame:
    """把 8 日尺度 LAI/FAPAR 宽表展开为日尺度长表。

    方法说明：
    - 文件中 datetime 被视为 8 日产品窗口起始日。
    - 每个产品值复制到起始日及之后 7 天。
    - 对事件/参考日期求平均时，就等价于按窗口重叠天数加权。
    """

    wide = read_csv(path, value_name)
    require_columns(wide, [CFG.date_col], value_name)
    wide = wide.copy()
    wide[CFG.date_col] = parse_date_series(wide[CFG.date_col])
    value_cols = [c for c in wide.columns if c != CFG.date_col]
    rows = []
    for _, row in wide.dropna(subset=[CFG.date_col]).iterrows():
        dates = pd.date_range(row[CFG.date_col], periods=CFG.lai_fapar_window_days, freq="D")
        repeated = pd.DataFrame({CFG.date_col: dates})
        for col in value_cols:
            repeated[col] = row[col]
        rows.append(repeated)
    if not rows:
        return pd.DataFrame(columns=["Date", CFG.site_col, value_name])
    expanded = pd.concat(rows, ignore_index=True)
    long = expanded.melt(id_vars=CFG.date_col, var_name=CFG.site_col, value_name=value_name)
    long = long.rename(columns={CFG.date_col: "Date"})
    long[CFG.site_col] = normalise_site_id(long[CFG.site_col])
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    return long.dropna(subset=["Date"])


def load_micro_soil_daily() -> pd.DataFrame:
    """读取 MicroTandSoilT.csv 中的 VWC_Daily 和 -5 cm 土温。

    VWC_Daily 是用户确认的土壤水分主变量；SMI 是派生指数，不进入主模型。
    """

    df = read_csv(CFG.micro_soil_csv, "MicroTandSoilT逐日土壤过程表")
    require_columns(df, [CFG.site_col, CFG.soil_date_col, CFG.soil_vwc_col, CFG.soil_tminus5_col], "MicroTandSoilT")
    df = df[[CFG.site_col, CFG.soil_date_col, CFG.soil_vwc_col, CFG.soil_tminus5_col]].copy()
    df[CFG.site_col] = normalise_site_id(df[CFG.site_col])
    df["Date"] = parse_date_series(df[CFG.soil_date_col])
    df = df.rename(columns={CFG.soil_vwc_col: "SoilMoisture", CFG.soil_tminus5_col: "Tminus5"})
    df["SoilMoisture"] = pd.to_numeric(df["SoilMoisture"], errors="coerce")
    df["Tminus5"] = pd.to_numeric(df["Tminus5"], errors="coerce")
    return df[[CFG.site_col, "Date", "SoilMoisture", "Tminus5"]].dropna(subset=["Date"])


def load_raw_sensor_daily(valid_sites: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取 Tensor_Data 原始 15 分钟传感器，聚合 T2_0/T3_15 为 UTC 日均值。

    输出：
    - daily：Site_ID + Date + T0 + T15 + 每日有效记录数。
    - audit：每个站点的文件、时间解析率、有效记录数。
    """

    site_set = set(str(x) for x in valid_sites)
    rows = []
    audit_rows = []
    files = sorted(CFG.tensor_data_dir.glob("953322*.csv"))
    with progress_bar(len(files), "步骤3/9 聚合原始传感器日尺度数据", "汇总") as bar:
        for path in files:
            site_id = path.stem
            if site_id not in site_set:
                bar.update(1)
                continue
            try:
                df = pd.read_csv(path, low_memory=False)
                require_columns(df, [CFG.sensor_utc_time_col, CFG.sensor_t0_col, CFG.sensor_t15_col], path.name)
                parsed = pd.to_datetime(df[CFG.sensor_utc_time_col], format="%Y.%m.%d %H:%M", errors="coerce")
                df["Date"] = parsed.dt.normalize()
                parse_rate = float(df["Date"].notna().mean()) if len(df) else np.nan
                grouped = (
                    df.dropna(subset=["Date"])
                    .assign(
                        T0=pd.to_numeric(df[CFG.sensor_t0_col], errors="coerce"),
                        T15=pd.to_numeric(df[CFG.sensor_t15_col], errors="coerce"),
                    )
                    .groupby("Date", as_index=False)
                    .agg(
                        T0=("T0", "mean"),
                        T15=("T15", "mean"),
                        T0_n=("T0", "count"),
                        T15_n=("T15", "count"),
                    )
                )
                grouped[CFG.site_col] = site_id
                grouped.loc[grouped["T0_n"] < CFG.min_sensor_records_per_day, "T0"] = np.nan
                grouped.loc[grouped["T15_n"] < CFG.min_sensor_records_per_day, "T15"] = np.nan
                rows.append(grouped)
                audit_rows.append({
                    CFG.site_col: site_id,
                    "文件名": path.name,
                    "原始行数": len(df),
                    "UTC时间解析率": parse_rate,
                    "有效UTC日期数": grouped["Date"].nunique(),
                    "T0有效记录数": int(grouped["T0_n"].sum()),
                    "T15有效记录数": int(grouped["T15_n"].sum()),
                    "状态": "ok",
                })
            except Exception as exc:
                audit_rows.append({
                    CFG.site_col: site_id,
                    "文件名": path.name,
                    "原始行数": np.nan,
                    "UTC时间解析率": np.nan,
                    "有效UTC日期数": np.nan,
                    "T0有效记录数": np.nan,
                    "T15有效记录数": np.nan,
                    "状态": f"error: {exc}",
                })
            bar.update(1)
    daily = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["Date", "T0", "T15", "T0_n", "T15_n", CFG.site_col]
    )
    daily[CFG.site_col] = normalise_site_id(daily[CFG.site_col])
    return daily[[CFG.site_col, "Date", "T0", "T15", "T0_n", "T15_n"]], pd.DataFrame(audit_rows)


def summarize_window(
    data: pd.DataFrame,
    site_id: str,
    dates: Iterable[pd.Timestamp],
    value_col: str,
    n_col: str | None = None,
) -> dict[str, float]:
    """按指定站点和日期集合汇总变量均值、有效天数和原始记录数。

    参数说明：
    - data：Site_ID + Date + value 的长表。
    - site_id：当前事件对应站点。
    - dates：事件期日期或实际选中的参考期日期。
    - value_col：需要汇总的变量。
    - n_col：若提供，则额外汇总原始记录数，例如 15 分钟传感器日有效记录数。
    """

    date_index = pd.to_datetime(pd.Series(list(dates)), errors="coerce").dt.normalize()
    date_index = set(date_index.dropna())
    if not date_index or data.empty or value_col not in data.columns:
        return {"mean": np.nan, "n_days": 0, "n_records": np.nan}
    d = data.loc[(data[CFG.site_col].astype(str) == str(site_id)) & (data["Date"].isin(date_index))].copy()
    values = pd.to_numeric(d[value_col], errors="coerce")
    n_days = int(values.notna().sum())
    n_records = float(pd.to_numeric(d[n_col], errors="coerce").sum()) if n_col and n_col in d.columns else np.nan
    return {
        "mean": float(values.mean()) if n_days else np.nan,
        "n_days": n_days,
        "n_records": n_records,
    }


def add_event_reference_summary(
    out: dict,
    prefix: str,
    data: pd.DataFrame,
    site_id: str,
    event_dates: Iterable[pd.Timestamp],
    reference_dates: Iterable[pd.Timestamp],
    value_col: str,
    n_col: str | None = None,
) -> None:
    """把某个变量的事件期、参考期和异常值写入当前事件输出字典。"""

    event = summarize_window(data, site_id, event_dates, value_col, n_col=n_col)
    ref = summarize_window(data, site_id, reference_dates, value_col, n_col=n_col)
    out[f"{prefix}_Event"] = event["mean"]
    out[f"{prefix}_Reference"] = ref["mean"]
    out[f"{prefix}_Anomaly"] = event["mean"] - ref["mean"] if np.isfinite(event["mean"]) and np.isfinite(ref["mean"]) else np.nan
    out[f"{prefix}_Event_n_days"] = event["n_days"]
    out[f"{prefix}_Reference_n_days"] = ref["n_days"]
    if n_col:
        out[f"{prefix}_Event_n_records"] = event["n_records"]
        out[f"{prefix}_Reference_n_records"] = ref["n_records"]


def build_environment_covariates(events: pd.DataFrame, ref_dates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构建一行一个 Site_ID + Event_ID 的事件环境协变量表。"""

    loaded = {}
    with progress_bar(8, "步骤2/9 读取环境与过程输入表", "读取") as bar:
        loaded["PET"] = load_wide_daily_table(CFG.pet_csv, "PET")
        bar.update(1)
        loaded["Srad"] = load_wide_daily_table(CFG.srad_csv, "Srad")
        bar.update(1)
        loaded["Precip"] = load_wide_daily_table(CFG.precip_csv, "Precip")
        bar.update(1)
        loaded["T2m"] = load_wide_daily_table(CFG.t2m_csv, "T2m")
        bar.update(1)
        loaded["LAI"] = load_8day_table_as_daily(CFG.lai_csv, "LAI")
        bar.update(1)
        loaded["FAPAR"] = load_8day_table_as_daily(CFG.fapar_csv, "FAPAR")
        bar.update(1)
        loaded["Soil"] = load_micro_soil_daily()
        bar.update(1)
        static = read_csv(CFG.static_site_csv, "站点静态地形冠层表")
        require_columns(static, [CFG.site_col, "Elevation", "Slope", "Aspect", "Canopy_Height"], "站点静态地形冠层表")
        static[CFG.site_col] = normalise_site_id(static[CFG.site_col])
        radians = np.deg2rad(pd.to_numeric(static["Aspect"], errors="coerce"))
        static["Northness"] = np.cos(radians)
        static["Eastness"] = np.sin(radians)
        bar.update(1)

    sensor_daily, sensor_audit = load_raw_sensor_daily(events[CFG.site_col].unique())

    ref_grouped = {
        (site, event): group["Candidate_Date"].tolist()
        for (site, event), group in ref_dates.groupby([CFG.site_col, CFG.event_col])
    }

    rows = []
    with progress_bar(len(events), "步骤4/9 汇总事件-参考期环境协变量", "汇总") as bar:
        for _, event in events.iterrows():
            site_id = str(event[CFG.site_col])
            event_id = str(event[CFG.event_col])
            event_dates = pd.date_range(event["Start_Date"], event["End_Date"], freq="D")
            reference_dates = ref_grouped.get((site_id, event_id), [])
            out = {
                CFG.site_col: site_id,
                CFG.event_col: event_id,
                "Event_Date_Count": len(event_dates),
                "Reference_Date_Count": len(reference_dates),
            }

            add_event_reference_summary(out, "PET", loaded["PET"], site_id, event_dates, reference_dates, "PET")
            add_event_reference_summary(out, "Srad", loaded["Srad"], site_id, event_dates, reference_dates, "Srad")
            add_event_reference_summary(out, "Precip", loaded["Precip"], site_id, event_dates, reference_dates, "Precip")
            add_event_reference_summary(out, "T2m", loaded["T2m"], site_id, event_dates, reference_dates, "T2m")
            add_event_reference_summary(out, "LAI", loaded["LAI"], site_id, event_dates, reference_dates, "LAI")
            add_event_reference_summary(out, "FAPAR", loaded["FAPAR"], site_id, event_dates, reference_dates, "FAPAR")
            add_event_reference_summary(out, "SoilMoisture", loaded["Soil"], site_id, event_dates, reference_dates, "SoilMoisture")
            add_event_reference_summary(out, "Tminus5", loaded["Soil"], site_id, event_dates, reference_dates, "Tminus5")
            add_event_reference_summary(out, "T0", sensor_daily, site_id, event_dates, reference_dates, "T0", n_col="T0_n")
            add_event_reference_summary(out, "T15_AUDIT_ONLY", sensor_daily, site_id, event_dates, reference_dates, "T15", n_col="T15_n")
            rows.append(out)
            bar.update(1)

    cov = pd.DataFrame(rows)
    cov["Covariate_QC_Flag"] = np.where(
        (cov["Event_Date_Count"] >= CFG.min_window_valid_days)
        & (cov["Reference_Date_Count"] >= CFG.min_window_valid_days),
        "ok",
        "review",
    )
    cov["Covariate_QC_Reason"] = np.where(
        cov["Covariate_QC_Flag"].eq("ok"),
        "",
        "事件期或参考期有效日期数低于阈值，需人工核查。",
    )
    cov = cov.merge(
        static[[CFG.site_col, "Longitude", "Latitude", "Elevation", "Slope", "Aspect", "Canopy_Height", "Northness", "Eastness"]],
        on=CFG.site_col,
        how="left",
        validate="many_to_one",
    )
    return cov, sensor_audit


# =============================================================================
# 5. 主表审计、相关性与 VIF
# =============================================================================


def build_master_table(events: pd.DataFrame, covariates: pd.DataFrame) -> pd.DataFrame:
    """合并响应表和协变量表，并生成标准化变量。"""

    master = events.merge(covariates, on=[CFG.site_col, CFG.event_col], how="left", validate="one_to_one")
    for variable in CONTINUOUS_VARIABLES:
        if variable in master.columns:
            master[f"z_{variable}"] = zscore(master[variable])
    return master


def build_audit_tables(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成变量缺失审计和样本结构审计。"""

    missing = pd.DataFrame({
        "变量名": master.columns,
        "非缺失数量": [int(master[col].notna().sum()) for col in master.columns],
        "缺失率": [float(master[col].isna().mean()) for col in master.columns],
        "唯一值数量": [int(master[col].nunique(dropna=True)) for col in master.columns],
    })

    site_counts = master.groupby(CFG.site_col).size().reset_index(name="有效事件数")
    sample_rows = [
        {"审计项": "有效事件总数", "数值": len(master), "说明": "Pair_flag=ok 的站点 x 事件配对数。"},
        {"审计项": "有效站点数", "数值": master[CFG.site_col].nunique(), "说明": "至少有一个有效事件的站点数。"},
        {"审计项": "Delta_CBI正值事件数", "数值": int((master["Delta_CBI"] > 0).sum()), "说明": "干旱期缓冲减弱事件数。"},
        {"审计项": "Delta_CBI负值事件数", "数值": int((master["Delta_CBI"] < 0).sum()), "说明": "干旱期缓冲增强或维持事件数。"},
        {"审计项": "仅1个事件的站点数", "数值": int((site_counts["有效事件数"] == 1).sum()), "说明": "随机截距可识别性的重要限制。"},
        {"审计项": "2个及以上事件的站点数", "数值": int((site_counts["有效事件数"] >= 2).sum()), "说明": "真正提供站点内重复信息的站点数。"},
        {"审计项": "最多单站点事件数", "数值": int(site_counts["有效事件数"].max()), "说明": "随机斜率模型不建议使用。"},
    ]
    sample = pd.concat([pd.DataFrame(sample_rows), site_counts.rename(columns={CFG.site_col: "审计项", "有效事件数": "数值"}).assign(说明="各站点有效事件数")], ignore_index=True)
    return missing, sample


def correlation_audit(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算候选连续变量 Spearman 相关矩阵和高相关变量对。"""

    variables = [v for v in CONTINUOUS_VARIABLES if v in master.columns]
    numeric = master[variables].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(method="spearman")
    rows = []
    for i, left in enumerate(variables):
        for right in variables[i + 1:]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(value) >= CFG.correlation_threshold:
                rows.append({
                    "变量1": left,
                    "变量2": right,
                    "Spearman_r": value,
                    "阈值": CFG.correlation_threshold,
                    "建议": "不建议同时进入同一主模型；需按生态机制保留一个或作为替代模型。",
                })
    return corr, pd.DataFrame(rows)


def calculate_vif(df: pd.DataFrame, terms: list[str], model_id: str) -> pd.DataFrame:
    """对某个候选模型的固定效应计算 VIF。

    只计算连续数值项；分类项 C(Event_Month) 不参与 VIF。
    """

    numeric_terms = [term for term in terms if not term.startswith("C(") and term in df.columns]
    if not numeric_terms:
        return pd.DataFrame([{"模型": model_id, "变量": "(无连续变量)", "VIF": np.nan, "说明": "无需计算"}])
    x = df[numeric_terms].apply(pd.to_numeric, errors="coerce").dropna()
    if len(x) <= len(numeric_terms) + 2:
        return pd.DataFrame([{
            "模型": model_id,
            "变量": ",".join(numeric_terms),
            "VIF": np.nan,
            "说明": f"完整案例不足，n={len(x)}，无法稳定估计 VIF。",
        }])
    rows = []
    x_const = x.assign(constant=1.0)
    for i, term in enumerate(numeric_terms):
        vif = variance_inflation_factor(x_const.values, i)
        rows.append({
            "模型": model_id,
            "变量": term,
            "VIF": float(vif),
            "说明": "超过阈值，需删除或替代" if vif > CFG.vif_threshold else "可接受",
        })
    return pd.DataFrame(rows)


# =============================================================================
# 6. 候选模型拟合
# =============================================================================


def available_terms(df: pd.DataFrame, terms: list[str]) -> list[str]:
    """剔除当前主表中不存在或全缺失的变量。"""

    available = []
    for term in terms:
        if term.startswith("C("):
            raw = term[2:-1]
            if raw in df.columns and df[raw].notna().any():
                available.append(term)
        elif term in df.columns and df[term].notna().any():
            available.append(term)
    return available


def model_required_columns(terms: list[str]) -> list[str]:
    """把公式项转换成 dropna 所需原始字段。"""

    cols = ["Delta_CBI", CFG.site_col]
    for term in terms:
        if term.startswith("C("):
            cols.append(term[2:-1])
        else:
            cols.append(term)
    return list(dict.fromkeys(cols))


def fit_one_model(df: pd.DataFrame, model_id: str, terms: list[str], sample_label: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """拟合一个候选模型的 LMM 和 OLS 补充模型。"""

    terms = available_terms(df, terms)
    if not terms:
        return (
            pd.DataFrame([{"模型": model_id, "样本口径": sample_label, "方法": "未拟合", "错误": "无可用固定效应"}]),
            pd.DataFrame(),
            {"LMM": None, "OLS_HC3": None},
        )

    required = model_required_columns(terms)
    d = df.dropna(subset=required).copy()
    formula = "Delta_CBI ~ " + " + ".join(terms)
    summaries = []
    coefficients = []
    fitted_lmm = None
    fitted_ols = None

    if len(d) <= len(terms) + 4 or d[CFG.site_col].nunique() < 3:
        summaries.append({
            "模型": model_id,
            "样本口径": sample_label,
            "方法": "LMM",
            "公式": formula,
            "n_obs": len(d),
            "n_sites": d[CFG.site_col].nunique(),
            "错误": "完整案例或站点数不足，跳过 LMM。",
        })
    else:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                lmm = smf.mixedlm(formula, d, groups=d[CFG.site_col], re_formula="1").fit(
                    reml=False, method="lbfgs", maxiter=1000, disp=False
                )
            warning_text = " | ".join(str(w.message) for w in caught)
            r2m, r2c = mixed_r2(lmm)
            random_var = float(np.trace(np.asarray(lmm.cov_re))) if lmm.cov_re is not None else np.nan
            k = int(len(lmm.params))
            summaries.append({
                "模型": model_id,
                "样本口径": sample_label,
                "方法": "LMM",
                "分析定位": "事件级补充诊断；随机效应奇异时不作为主证据",
                "公式": formula,
                "n_obs": len(d),
                "n_sites": d[CFG.site_col].nunique(),
                "AIC": float(lmm.aic),
                "AICc": aicc(float(lmm.aic), len(d), k),
                "BIC": float(lmm.bic),
                "边际R2": r2m,
                "条件R2": r2c,
                "随机截距方差": random_var,
                "随机效应近零": bool(np.isfinite(random_var) and random_var <= CFG.near_zero_random_effect_threshold),
                "收敛": bool(lmm.converged),
                "警告信息": warning_text,
                "错误": "",
            })
            ci = lmm.conf_int()
            for term in lmm.params.index:
                coefficients.append({
                    "模型": model_id,
                    "样本口径": sample_label,
                    "方法": "LMM",
                    "变量": term,
                    "估计值": float(lmm.params[term]),
                    "标准误": float(lmm.bse[term]) if term in lmm.bse.index else np.nan,
                    "P值": float(lmm.pvalues[term]) if term in lmm.pvalues.index else np.nan,
                    "CI95下限": float(ci.loc[term, 0]) if term in ci.index else np.nan,
                    "CI95上限": float(ci.loc[term, 1]) if term in ci.index else np.nan,
                })
            fitted_lmm = lmm
        except Exception as exc:
            summaries.append({
                "模型": model_id,
                "样本口径": sample_label,
                "方法": "LMM",
                "分析定位": "事件级补充诊断；随机效应奇异时不作为主证据",
                "公式": formula,
                "n_obs": len(d),
                "n_sites": d[CFG.site_col].nunique(),
                "警告信息": "",
                "错误": str(exc),
            })

    try:
        ols = smf.ols(formula, d).fit(cov_type="HC3")
        r2, r2_adj = ols_r2(ols)
        k = int(len(ols.params))
        summaries.append({
            "模型": model_id,
            "样本口径": sample_label,
            "方法": "OLS_HC3",
            "分析定位": "事件级补充稳健模型",
            "公式": formula,
            "n_obs": len(d),
            "n_sites": d[CFG.site_col].nunique(),
            "AIC": float(ols.aic),
            "AICc": aicc(float(ols.aic), len(d), k),
            "BIC": float(ols.bic),
            "R2": r2,
            "调整R2": r2_adj,
            "收敛": True,
            "警告信息": "",
            "错误": "",
        })
        ci = ols.conf_int()
        for term in ols.params.index:
            coefficients.append({
                "模型": model_id,
                "样本口径": sample_label,
                "方法": "OLS_HC3",
                "变量": term,
                "估计值": float(ols.params[term]),
                "标准误": float(ols.bse[term]) if term in ols.bse.index else np.nan,
                "P值": float(ols.pvalues[term]) if term in ols.pvalues.index else np.nan,
                "CI95下限": float(ci.loc[term, 0]) if term in ci.index else np.nan,
                "CI95上限": float(ci.loc[term, 1]) if term in ci.index else np.nan,
            })
        fitted_ols = ols
    except Exception as exc:
        summaries.append({
            "模型": model_id,
            "样本口径": sample_label,
            "方法": "OLS_HC3",
            "分析定位": "事件级补充稳健模型",
            "公式": formula,
            "n_obs": len(d),
            "n_sites": d[CFG.site_col].nunique(),
            "警告信息": "",
            "错误": str(exc),
        })

    return pd.DataFrame(summaries), pd.DataFrame(coefficients), {"LMM": fitted_lmm, "OLS_HC3": fitted_ols}


def fit_candidate_models(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], object]]:
    """拟合预设候选模型，并同时输出 VIF。"""

    summaries = []
    coefficients = []
    fitted = {}

    all_terms = []
    for meta in BASE_CANDIDATE_MODELS.values():
        all_terms.extend([t for t in meta["terms"] if not t.startswith("C(")])
    common_terms = sorted(set([t for t in all_terms if t in master.columns]))
    common_cols = ["Delta_CBI", CFG.site_col] + common_terms
    common_sample = master.dropna(subset=common_cols).copy()
    if len(common_sample) < 8:
        common_sample = master.copy()
        common_label = "最大可用样本"
    else:
        common_label = "共同完整样本"

    with progress_bar(len(BASE_CANDIDATE_MODELS), "步骤6/9 拟合预设候选模型", "建模") as bar:
        for model_id, meta in BASE_CANDIDATE_MODELS.items():
            summary, coef, results = fit_one_model(common_sample, model_id, meta["terms"], common_label)
            summaries.append(summary)
            if not coef.empty:
                coefficients.append(coef)
            for method, result in results.items():
                if result is not None:
                    fitted[(model_id, method)] = result
            bar.update(1)

    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    coef_df = pd.concat(coefficients, ignore_index=True) if coefficients else pd.DataFrame()
    if "AICc" in summary_df.columns and summary_df["AICc"].notna().any():
        for method in summary_df["方法"].dropna().unique():
            idx = summary_df["方法"].eq(method) & summary_df["AICc"].notna()
            summary_df.loc[idx, "Delta_AICc"] = summary_df.loc[idx, "AICc"] - summary_df.loc[idx, "AICc"].min()
    return summary_df, coef_df, fitted


def build_lmm_diagnostic(model_summary: pd.DataFrame) -> pd.DataFrame:
    """从事件级模型比较表中提取 LMM 可用性诊断。

    该表用于方法部分说明：当前数据中多数站点只有一个事件，LMM 随机截距
    经常不可稳定估计，因此 LMM 只作为补充诊断，不作为 Why 主证据。
    """

    if model_summary.empty or "方法" not in model_summary.columns:
        return pd.DataFrame()
    lmm = model_summary.loc[model_summary["方法"].eq("LMM")].copy()
    if lmm.empty:
        return pd.DataFrame()
    lmm["LMM可作为主证据"] = False
    lmm["诊断结论"] = np.select(
        [
            lmm.get("错误", pd.Series("", index=lmm.index)).fillna("").str.contains("Singular|singular", case=False, regex=True),
            lmm.get("警告信息", pd.Series("", index=lmm.index)).fillna("").str.contains("singular|boundary", case=False, regex=True),
            lmm.get("随机效应近零", pd.Series(False, index=lmm.index)).fillna(False).astype(bool),
        ],
        [
            "LMM矩阵奇异，随机截距不可稳定估计。",
            "LMM警告随机效应奇异或位于参数边界。",
            "随机截距方差接近0，LMM退化风险高。",
        ],
        default="LMM仅作为补充诊断；主证据仍使用站点级OLS_HC3。",
    )
    keep = [
        col for col in [
            "模型", "样本口径", "公式", "n_obs", "n_sites", "随机截距方差",
            "随机效应近零", "收敛", "警告信息", "错误", "LMM可作为主证据", "诊断结论",
        ]
        if col in lmm.columns
    ]
    return lmm[keep]


def build_vif_table(master: pd.DataFrame) -> pd.DataFrame:
    """对每个候选模型分别计算 VIF。"""

    rows = []
    for model_id, meta in BASE_CANDIDATE_MODELS.items():
        rows.append(calculate_vif(master, available_terms(master, meta["terms"]), model_id))
    for model_id, terms in ALTERNATIVE_MODELS.items():
        rows.append(calculate_vif(master, available_terms(master, terms), model_id))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_site_level_table(master: pd.DataFrame) -> pd.DataFrame:
    """把事件级主表聚合为站点级主分析表。

    聚合原则：
    - Delta_CBI：同站点有效事件的算术均值，代表站点平均缓冲响应。
    - 动态异常变量：同站点事件异常均值，代表该站点在观测极端事件中的平均环境变化。
    - 静态变量：同站点固定背景取第一个非缺失值。
    - 同时保留每站点事件数、正负响应事件数，用于解释站点聚合的不确定性。
    """

    numeric_mean_cols = [
        col for col in CONTINUOUS_VARIABLES + ["Delta_CBI"]
        if col in master.columns and col not in {"Elevation", "Slope", "Aspect", "Northness", "Eastness", "Canopy_Height"}
    ]
    static_cols = [
        col for col in ["Longitude", "Latitude", "Elevation", "Slope", "Aspect", "Northness", "Eastness", "Canopy_Height"]
        if col in master.columns
    ]
    agg_spec = {col: (col, "mean") for col in numeric_mean_cols}
    for col in static_cols:
        agg_spec[col] = (col, "first")
    site = (
        master
        .groupby(CFG.site_col, as_index=False)
        .agg(
            有效事件数=("Delta_CBI", "size"),
            Delta_CBI正值事件数=("Delta_CBI", lambda x: int((x > 0).sum())),
            Delta_CBI负值事件数=("Delta_CBI", lambda x: int((x < 0).sum())),
            **agg_spec,
        )
    )
    for variable in CONTINUOUS_VARIABLES:
        if variable in site.columns:
            site[f"z_{variable}"] = zscore(site[variable])
    site["z_Delta_CBI"] = zscore(site["Delta_CBI"])
    return site


def fit_site_level_models(site_master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], object]]:
    """拟合站点级主分析 OLS_HC3 模型。

    这里不再使用 LMM，因为站点级数据每行已经是一个站点，研究目标是解释空间差异。
    """

    summaries = []
    coefficients = []
    fitted = {}
    with progress_bar(len(SITE_LEVEL_MODELS), "步骤6/9 拟合站点级主分析模型", "建模") as bar:
        for model_id, meta in SITE_LEVEL_MODELS.items():
            terms = available_terms(site_master, meta["terms"])
            formula = "Delta_CBI ~ " + " + ".join(terms) if terms else "Delta_CBI ~ 1"
            required = ["Delta_CBI"] + model_required_columns(terms)
            required = list(dict.fromkeys([col for col in required if col != CFG.site_col]))
            d = site_master.dropna(subset=required).copy()
            try:
                ols = smf.ols(formula, d).fit(cov_type="HC3")
                r2, r2_adj = ols_r2(ols)
                k = int(len(ols.params))
                summaries.append({
                    "模型": model_id,
                    "分析层级": "站点级主分析",
                    "方法": "OLS_HC3",
                    "公式": formula,
                    "n_obs": len(d),
                    "n_sites": len(d),
                    "AIC": float(ols.aic),
                    "AICc": aicc(float(ols.aic), len(d), k),
                    "BIC": float(ols.bic),
                    "R2": r2,
                    "调整R2": r2_adj,
                    "错误": "",
                })
                ci = ols.conf_int()
                for term in ols.params.index:
                    coefficients.append({
                        "模型": model_id,
                        "分析层级": "站点级主分析",
                        "方法": "OLS_HC3",
                        "变量": term,
                        "估计值": float(ols.params[term]),
                        "标准误": float(ols.bse[term]) if term in ols.bse.index else np.nan,
                        "P值": float(ols.pvalues[term]) if term in ols.pvalues.index else np.nan,
                        "CI95下限": float(ci.loc[term, 0]) if term in ci.index else np.nan,
                        "CI95上限": float(ci.loc[term, 1]) if term in ci.index else np.nan,
                    })
                fitted[(model_id, "OLS_HC3")] = ols
            except Exception as exc:
                summaries.append({
                    "模型": model_id,
                    "分析层级": "站点级主分析",
                    "方法": "OLS_HC3",
                    "公式": formula,
                    "n_obs": len(d),
                    "n_sites": len(d),
                    "错误": str(exc),
                })
            bar.update(1)

    summary_df = pd.DataFrame(summaries)
    coef_df = pd.DataFrame(coefficients)
    if "AICc" in summary_df.columns and summary_df["AICc"].notna().any():
        idx = summary_df["AICc"].notna()
        summary_df.loc[idx, "Delta_AICc"] = summary_df.loc[idx, "AICc"] - summary_df.loc[idx, "AICc"].min()
    return summary_df, coef_df, fitted


def fit_alternative_models(master: pd.DataFrame) -> pd.DataFrame:
    """拟合替代变量敏感性模型。"""

    rows = []
    with progress_bar(len(ALTERNATIVE_MODELS), "步骤8/9 拟合替代变量敏感性模型", "建模") as bar:
        for model_id, terms in ALTERNATIVE_MODELS.items():
            summary, _, _ = fit_one_model(master, model_id, terms, "最大可用样本")
            rows.append(summary)
            bar.update(1)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def leave_one_site_out_site_level(site_master: pd.DataFrame, final_terms: list[str]) -> pd.DataFrame:
    """站点级逐站点剔除敏感性分析。

    该函数与主分析层级保持一致：输入是一行一个站点的聚合表，每次删除一个站点后
    重新拟合站点级 OLS_HC3 主模型，检查主要标准化系数方向是否由单个站点驱动。
    """

    rows = []
    sites = sorted(site_master[CFG.site_col].dropna().astype(str).unique())
    with progress_bar(len(sites), "步骤7/9 逐站点剔除敏感性分析", "建模") as bar:
        for site in sites:
            d = site_master.loc[site_master[CFG.site_col].astype(str) != site].copy()
            terms = available_terms(d, final_terms)
            formula = "Delta_CBI ~ " + " + ".join(terms) if terms else "Delta_CBI ~ 1"
            required = ["Delta_CBI"] + terms
            d = d.dropna(subset=required).copy()
            try:
                result = smf.ols(formula, d).fit(cov_type="HC3")
                ci = result.conf_int()
                target_terms = [term for term in terms if term in result.params.index]
                if not target_terms:
                    rows.append({
                        "删除站点": site,
                        "状态": "ok",
                        "变量": "(截距模型)",
                        "估计值": np.nan,
                        "P值": np.nan,
                        "CI95下限": np.nan,
                        "CI95上限": np.nan,
                        "n_sites": len(d),
                        "公式": formula,
                    })
                for term in target_terms:
                    rows.append({
                        "删除站点": site,
                        "状态": "ok",
                        "变量": term,
                        "估计值": float(result.params[term]),
                        "P值": float(result.pvalues[term]),
                        "CI95下限": float(ci.loc[term, 0]),
                        "CI95上限": float(ci.loc[term, 1]),
                        "n_sites": len(d),
                        "公式": formula,
                    })
            except Exception as exc:
                rows.append({
                    "删除站点": site,
                    "状态": "失败或无系数",
                    "说明": str(exc),
                    "n_sites": len(d),
                    "公式": formula,
                })
            bar.update(1)
    return pd.DataFrame(rows)


# =============================================================================
# 7. 图件输出
# =============================================================================


def setup_plot_style() -> None:
    """设置绘图字体和主题。"""

    global CHINESE_FONT_PROP, CHINESE_FONT_NAME, CHINESE_FONT_FILE
    sns.set_theme(style="whitegrid")
    CHINESE_FONT_PROP = get_chinese_font()
    if CHINESE_FONT_PROP is not None:
        CHINESE_FONT_FILE = CHINESE_FONT_PROP.get_file() or ""
        font_manager.fontManager.addfont(CHINESE_FONT_FILE)
        CHINESE_FONT_NAME = CHINESE_FONT_PROP.get_name()
        plt.rcParams["font.family"] = [CHINESE_FONT_NAME]
        plt.rcParams["font.sans-serif"] = [CHINESE_FONT_NAME, PLOT_STYLE["font_family"], "Microsoft YaHei", "DengXian", "SimSun"]
    else:
        CHINESE_FONT_NAME = PLOT_STYLE["font_family"]
        CHINESE_FONT_FILE = ""
        plt.rcParams["font.family"] = [PLOT_STYLE["font_family"]]
        plt.rcParams["font.sans-serif"] = [PLOT_STYLE["font_family"], "Microsoft YaHei", "DengXian", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def get_chinese_font():
    """显式查找 Windows 中文字体文件，避免 Matplotlib 只靠字体名匹配失败。

    返回值可传给 set_title/set_xlabel/set_yticklabels 的 fontproperties 参数。
    如果候选字体文件都不存在，则返回 None，脚本仍可运行，但可能出现缺字警告。
    """

    for font_path in PLOT_STYLE["font_file_candidates"]:
        path = Path(font_path)
        if path.exists():
            return font_manager.FontProperties(fname=str(path))
    return None


def apply_chinese_font_to_figure(fig) -> None:
    """把中文字体强制应用到整张图的所有文本对象。

    为什么需要这一步：
    - seaborn.set_theme() 会重置部分 rcParams；
    - scipy.stats.probplot 会自动创建英文坐标轴和刻度文本；
    - 仅在 set_title/set_xlabel 传 fontproperties，不能保证刻度、图例、suptitle 都继承。

    因此每张图保存前统一遍历 figure/axes/text/legend/ticklabel，最大限度避免中文显示为方框。
    """

    font_prop = CHINESE_FONT_PROP or get_chinese_font()
    if font_prop is None:
        return

    for text in fig.findobj(match=plt.Text):
        text.set_fontproperties(font_prop)
    for ax in fig.axes:
        ax.title.set_fontproperties(font_prop)
        ax.xaxis.label.set_fontproperties(font_prop)
        ax.yaxis.label.set_fontproperties(font_prop)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(font_prop)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontproperties(font_prop)
            if legend.get_title() is not None:
                legend.get_title().set_fontproperties(font_prop)


def term_to_chinese_label(term: str) -> str:
    """把模型项名称转换为图中显示的中文标签。

    统计模型使用 z_ 前缀表示标准化变量，绘图时去掉前缀并映射为中文，避免图上出现
    Drought_Intensity、LAI_Anomaly 这类需要读者二次理解的字段名。
    """

    raw = str(term).replace("z_", "", 1)
    return VARIABLE_LABELS.get(raw, raw)


def format_p_value(value: float) -> str:
    """把 P 值格式化为适合图内标注的短文本。"""

    if not np.isfinite(value):
        return "P=NA"
    if value < 0.001:
        return "P<0.001"
    return f"P={value:.3f}"


def significance_mark(value: float) -> str:
    """根据 P 值返回常见显著性星号，仅作为视觉提示，正式解释仍看系数表。"""

    if not np.isfinite(value):
        return ""
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    if value < 0.1:
        return "·"
    return "ns"


def get_model_stats(result) -> dict[str, float]:
    """提取图内需要显示的模型拟合统计量。"""

    observed = np.asarray(result.model.endog, dtype=float)
    predicted = np.asarray(result.fittedvalues, dtype=float)
    residual = observed - predicted
    rmse = float(np.sqrt(np.mean(residual ** 2))) if len(residual) else np.nan
    mae = float(np.mean(np.abs(residual))) if len(residual) else np.nan
    slope, intercept, _, _, _ = stats.linregress(predicted, observed) if len(observed) >= 2 else (np.nan, np.nan, np.nan, np.nan, np.nan)
    return {
        "n": int(len(observed)),
        "R2": float(getattr(result, "rsquared", np.nan)),
        "Adj_R2": float(getattr(result, "rsquared_adj", np.nan)),
        "AICc": aicc(float(getattr(result, "aic", np.nan)), int(len(observed)), int(len(result.params))),
        "RMSE": rmse,
        "MAE": mae,
        "slope": float(slope),
        "intercept": float(intercept),
    }


def add_text_box(ax, text: str, loc: str = "upper left") -> None:
    """在图内添加半透明统计说明框。"""

    xy = {
        "upper left": (0.03, 0.97, "left", "top"),
        "upper right": (0.97, 0.97, "right", "top"),
        "lower left": (0.03, 0.03, "left", "bottom"),
        "lower right": (0.97, 0.03, "right", "bottom"),
        "below center": (0.50, -0.17, "center", "top"),
    }[loc]
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha=xy[2],
        va=xy[3],
        fontsize=PLOT_STYLE["annotation_fontsize"],
        fontproperties=CHINESE_FONT_PROP,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#c8c8c8", "alpha": 0.88},
    )


def add_small_text_box(ax, text: str, loc: str = "lower left") -> None:
    """添加字号更小的诊断信息框，专门用于残差图，减少遮挡散点。"""

    xy = {
        "lower left": (0.03, 0.03, "left", "bottom"),
        "lower right": (0.97, 0.03, "right", "bottom"),
    }[loc]
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha=xy[2],
        va=xy[3],
        fontsize=max(7, PLOT_STYLE["annotation_fontsize"] - 2),
        fontproperties=CHINESE_FONT_PROP,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#c8c8c8", "alpha": 0.82},
    )


def choose_final_model(summary: pd.DataFrame) -> tuple[str | None, str | None]:
    """选择用于绘图的最终模型。

    选择原则：
    - 优先 LMM 中 AICc 可用且 Delta_AICc 最小的模型。
    - 若 LMM 全部失败，则退回 OLS_HC3 中 AICc 最小的模型。
    """

    if summary.empty or "AICc" not in summary.columns:
        return None, None
    for method in ["OLS_HC3"]:
        d = summary.loc[summary["方法"].eq(method) & summary["AICc"].notna()].copy()
        if not d.empty:
            row = d.sort_values("AICc").iloc[0]
            return str(row["模型"]), method
    return None, None


def plot_forest(coefficients: pd.DataFrame, model_summary: pd.DataFrame, final_model: str, final_method: str) -> None:
    """绘制最终模型标准化系数森林图。"""

    d = coefficients.loc[
        (coefficients["模型"].eq(final_model))
        & (coefficients["方法"].isin(["LMM", "OLS_HC3"]))
        & (coefficients["变量"].str.startswith("z_", na=False))
    ].copy()
    if d.empty:
        return
    method = final_method
    d = d.loc[d["方法"].eq(method)].copy()
    if d.empty:
        return
    d = d.sort_values("估计值")
    font_prop = get_chinese_font()
    d["中文变量"] = d["变量"].map(term_to_chinese_label)
    d["显著性"] = d["P值"].map(significance_mark)
    d["标注"] = d.apply(lambda row: f"β={row['估计值']:.3f}，{format_p_value(row['P值'])}，{row['显著性']}", axis=1)
    d["颜色"] = np.where(pd.to_numeric(d["P值"], errors="coerce") < 0.05, PLOT_STYLE["significant_color"], PLOT_STYLE["nonsignificant_color"])

    fig, ax = plt.subplots(figsize=(PLOT_STYLE["forest_width"] + 1.8, PLOT_STYLE["forest_height"] + 0.25))
    y = np.arange(len(d))
    band = float(PLOT_STYLE["weak_effect_band"])
    ax.axvspan(-band, band, color=PLOT_STYLE["weak_effect_band_color"], zorder=0, label="弱效应参考区")
    for i, row in d.reset_index(drop=True).iterrows():
        ax.errorbar(
            row["估计值"],
            i,
            xerr=[[row["估计值"] - row["CI95下限"]], [row["CI95上限"] - row["估计值"]]],
            fmt="o",
            color=row["颜色"],
            ecolor=row["颜色"],
            elinewidth=PLOT_STYLE["line_width"],
            capsize=4,
            markersize=7,
            alpha=PLOT_STYLE["point_alpha"],
        )
        y_offset = -18 if row["中文变量"] == "冠层高度" else (12 if i % 2 == 0 else -18)
        ax.annotate(
            row["标注"],
            xy=(row["估计值"], i),
            xytext=(0, y_offset),
            textcoords="offset points",
            va="bottom" if y_offset > 0 else "top",
            ha="center",
            fontsize=PLOT_STYLE["annotation_fontsize"],
            fontproperties=font_prop,
            color="#222222",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            clip_on=False,
        )
    ax.axvline(0, color=PLOT_STYLE["zero_line_color"], linestyle="--", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(d["中文变量"], fontproperties=font_prop)
    ax.set_xlabel("标准化回归系数（正值表示 Delta_CBI 更大，缓冲减弱更强）", fontproperties=font_prop)
    ax.set_title(f"站点级主模型标准化系数：{final_model}（{method}）", fontproperties=font_prop)
    chosen = model_summary.loc[model_summary["模型"].eq(final_model) & model_summary["方法"].eq(method)].copy()
    if not chosen.empty:
        row = chosen.iloc[0]
        stat_text = (
            f"n={int(row.get('n_obs', np.nan))} 个站点；R2={row.get('R2', np.nan):.3f}，调整R2={row.get('调整R2', np.nan):.3f}；"
            f"AICc={row.get('AICc', np.nan):.2f}，ΔAICc={row.get('Delta_AICc', np.nan):.2f}；误差线为95%CI；灰带为弱效应区"
        )
        add_text_box(ax, stat_text, loc="below center")
    apply_chinese_font_to_figure(fig)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(CFG.output_dir / OUTPUT_FILES["final_forest_png"], dpi=PLOT_STYLE["figure_dpi"])
    fig.savefig(CFG.output_dir / OUTPUT_FILES["final_forest_pdf"])
    plt.close(fig)


def plot_prediction_and_residual(fitted: dict[tuple[str, str], object], final_model: str, final_method: str) -> None:
    """绘制最终模型的观测-预测图和残差诊断图。"""

    key = (final_model, final_method)
    if key not in fitted:
        return
    result = fitted[key]
    d = result.model.data.frame.copy()
    d["预测Delta_CBI"] = result.fittedvalues
    d["残差"] = result.resid
    model_stats = get_model_stats(result)
    font_prop = get_chinese_font()

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.scatter(
        d["预测Delta_CBI"],
        d["Delta_CBI"],
        s=PLOT_STYLE["point_size"],
        color=PLOT_STYLE["pred_point_color"],
        alpha=PLOT_STYLE["point_alpha"],
        label="站点",
        zorder=3,
    )
    lim_min = min(d["预测Delta_CBI"].min(), d["Delta_CBI"].min())
    lim_max = max(d["预测Delta_CBI"].max(), d["Delta_CBI"].max())
    pad = (lim_max - lim_min) * 0.08 if lim_max > lim_min else 0.02
    lim_min -= pad
    lim_max += pad
    if len(d) >= 2:
        fit_x = np.linspace(lim_min, lim_max, 100)
        fit_y = model_stats["intercept"] + model_stats["slope"] * fit_x
        ax.plot(fit_x, fit_y, color="#d1495b", linewidth=2.0, alpha=0.72, label="观测-预测拟合线", zorder=2)
    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        color="#222222",
        linestyle=(0, (4, 3)),
        linewidth=2.0,
        label="1:1参考线",
        zorder=4,
    )
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("模型预测 Delta_CBI", fontproperties=font_prop)
    ax.set_ylabel("观测 Delta_CBI", fontproperties=font_prop)
    ax.set_title(f"观测值与预测值对照：{final_model}", fontproperties=font_prop)
    stat_text = (
        f"n={model_stats['n']} 个站点\n"
        f"R2={model_stats['R2']:.3f}，调整R2={model_stats['Adj_R2']:.3f}\n"
        f"RMSE={model_stats['RMSE']:.3f}，MAE={model_stats['MAE']:.3f}\n"
        f"观测={model_stats['intercept']:.3f}+{model_stats['slope']:.3f}×预测"
    )
    add_text_box(ax, stat_text, loc="upper left")
    ax.legend(prop=font_prop, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=True)
    apply_chinese_font_to_figure(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(CFG.output_dir / OUTPUT_FILES["pred_obs_png"], dpi=PLOT_STYLE["figure_dpi"])
    fig.savefig(CFG.output_dir / OUTPUT_FILES["pred_obs_pdf"])
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(PLOT_STYLE["diagnostic_width"], PLOT_STYLE["diagnostic_height"]))
    axes[0].scatter(d["预测Delta_CBI"], d["残差"], s=PLOT_STYLE["point_size"], color=PLOT_STYLE["residual_point_color"], alpha=PLOT_STYLE["point_alpha"])
    axes[0].axhline(0, color=PLOT_STYLE["zero_line_color"], linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("拟合值", fontproperties=font_prop)
    axes[0].set_ylabel("残差", fontproperties=font_prop)
    axes[0].set_title("残差-拟合值", fontproperties=font_prop)
    stats.probplot(d["残差"], dist="norm", plot=axes[1])
    axes[1].set_title("正态 Q-Q图", fontproperties=font_prop)
    resid = pd.to_numeric(d["残差"], errors="coerce").dropna()
    shapiro_p = stats.shapiro(resid).pvalue if 3 <= len(resid) <= 5000 else np.nan
    try:
        bp_stat, bp_p, _, _ = het_breuschpagan(result.resid, result.model.exog)
    except Exception:
        bp_p = np.nan
    qq_theoretical, qq_ordered = stats.probplot(d["残差"], dist="norm", fit=False)
    qq_slope, qq_intercept, qq_r, _, _ = stats.linregress(qq_theoretical, qq_ordered) if len(qq_theoretical) >= 2 else (np.nan, np.nan, np.nan, np.nan, np.nan)
    resid_text = (
        f"残差均值={resid.mean():.3f}；残差标准差={resid.std(ddof=1):.3f}\n"
        f"最大|残差|={resid.abs().max():.3f}；Shapiro P={shapiro_p:.3f}\n"
        f"Breusch-Pagan P={bp_p:.3f}"
    )
    add_small_text_box(axes[0], resid_text, loc="lower left")
    qq_text = (
        f"slope={qq_slope:.3f}\n"
        f"intercept={qq_intercept:.3f}, r={qq_r:.3f}\n"
        "closer to red line = more normal"
    )
    add_small_text_box(axes[1], qq_text, loc="lower right")
    fig.suptitle(f"残差诊断：{final_model}", fontproperties=font_prop)
    apply_chinese_font_to_figure(fig)
    fig.tight_layout()
    fig.savefig(CFG.output_dir / OUTPUT_FILES["residual_png"], dpi=PLOT_STYLE["figure_dpi"])
    fig.savefig(CFG.output_dir / OUTPUT_FILES["residual_pdf"])
    plt.close(fig)


# =============================================================================
# 8. 输出、摘要与缓存清理
# =============================================================================


def build_run_summary(master: pd.DataFrame, model_summary: pd.DataFrame, final_model: str | None, final_method: str | None) -> tuple[pd.DataFrame, str]:
    """生成运行摘要表和中文说明文本。"""

    rows = [
        {"项目": "有效事件数", "结果": len(master)},
        {"项目": "有效站点数", "结果": master[CFG.site_col].nunique()},
        {"项目": "Delta_CBI均值", "结果": float(master["Delta_CBI"].mean())},
        {"项目": "Delta_CBI中位数", "结果": float(master["Delta_CBI"].median())},
        {"项目": "Delta_CBI正值事件数", "结果": int((master["Delta_CBI"] > 0).sum())},
        {"项目": "Delta_CBI负值事件数", "结果": int((master["Delta_CBI"] < 0).sum())},
        {"项目": "主分析层级", "结果": "站点级聚合主分析"},
        {"项目": "主分析方法", "结果": "OLS_HC3稳健标准误"},
        {"项目": "LMM定位", "结果": "事件级补充诊断，不作为主证据"},
        {"项目": "最终主模型", "结果": final_model or "未选择"},
        {"项目": "最终主模型方法", "结果": final_method or "未选择"},
    ]
    if model_summary is not None and not model_summary.empty and final_model:
        chosen = model_summary.loc[model_summary["模型"].eq(final_model)].copy()
        if not chosen.empty:
            rows.append({"项目": "最终模型候选记录", "结果": chosen.to_json(force_ascii=False, orient="records")})

    text = f"""Why 阶段建模运行摘要
========================

有效事件数：{len(master)}
有效站点数：{master[CFG.site_col].nunique()}
Delta_CBI 均值：{master['Delta_CBI'].mean():.6f}
Delta_CBI 中位数：{master['Delta_CBI'].median():.6f}
Delta_CBI > 0 事件数：{int((master['Delta_CBI'] > 0).sum())}
Delta_CBI < 0 事件数：{int((master['Delta_CBI'] < 0).sum())}

主分析层级：站点级聚合主分析
主分析方法：OLS_HC3 稳健标准误
最终主模型：{final_model or '未选择'}
最终主模型方法：{final_method or '未选择'}

解释边界：
1. 本脚本解释的是站点 x 事件尺度 Delta_CBI 的环境关联，不重新证明 What 阶段总体效应。
2. SoilMoisture_Anomaly 使用 VWC_Daily，是过程/潜在中介变量。
3. T0_Anomaly 使用原始传感器 T2_0，是表层热过程补充变量。
4. T15_AUDIT_ONLY 使用原始传感器 T3_15，仅用于审计和过程描述，不进入主模型。
5. 由于多数站点只有一个事件，随机截距 LMM 容易奇异；LMM 结果只用于诊断和补充说明。
6. 主证据采用站点级聚合 OLS_HC3；事件级 OLS_HC3 用于检查环境关联是否一致。
7. 因样本量较小，模型结论必须结合 AICc、VIF、残差和敏感性分析共同解释。
"""
    return pd.DataFrame(rows), text


def cleanup_runtime_cache() -> pd.DataFrame:
    """清理本次脚本运行产生的临时缓存目录。

    只删除 RUNTIME_CACHE_DIR，不清理其他结果目录，避免误删用户数据。
    """

    rows = []
    with progress_bar(1, "步骤9/9 清理本次运行临时缓存", "清理") as bar:
        existed = RUNTIME_CACHE_DIR.exists()
        if existed:
            shutil.rmtree(RUNTIME_CACHE_DIR, ignore_errors=True)
        rows.append({
            "缓存路径": str(RUNTIME_CACHE_DIR),
            "运行前是否存在": existed,
            "清理后是否存在": RUNTIME_CACHE_DIR.exists(),
            "说明": "仅清理本次运行专用临时缓存目录。",
        })
        bar.update(1)
    return pd.DataFrame(rows)


# =============================================================================
# 9. 主流程
# =============================================================================


def main() -> None:
    """按固定顺序执行 Why 阶段建模流程。"""

    np.random.seed(CFG.random_seed)
    setup_plot_style()
    ensure_output_dir()

    with progress_bar(4, "步骤1/9 输入路径与What结果审计", "读取") as bar:
        path_audit = build_path_audit()
        bar.update(1)
        parameter_table = build_parameter_table()
        bar.update(1)
        events = prepare_event_response()
        bar.update(1)
        reference_dates = prepare_reference_dates(events)
        bar.update(1)

    covariates, sensor_audit = build_environment_covariates(events, reference_dates)

    with progress_bar(7, "步骤5/9 合并主表并输出审计", "整理") as bar:
        master = build_master_table(events, covariates)
        bar.update(1)
        missing_audit, sample_audit = build_audit_tables(master)
        bar.update(1)
        corr, high_corr = correlation_audit(master)
        bar.update(1)
        vif_table = build_vif_table(master)
        bar.update(1)
        write_csv(path_audit, OUTPUT_FILES["path_audit"])
        bar.update(1)
        write_csv(parameter_table, OUTPUT_FILES["parameter_table"])
        bar.update(1)
        write_csv(events, OUTPUT_FILES["event_response"])
        bar.update(1)

    with progress_bar(8, "步骤5/9 写出协变量与审计表", "输出") as bar:
        write_csv(reference_dates, OUTPUT_FILES["reference_dates"])
        bar.update(1)
        write_csv(covariates, OUTPUT_FILES["covariates"])
        bar.update(1)
        write_csv(sensor_audit, "03_原始传感器T0_T15日聚合审计表.csv")
        bar.update(1)
        write_csv(master, OUTPUT_FILES["master"])
        bar.update(1)
        write_csv(missing_audit, OUTPUT_FILES["missing_audit"])
        bar.update(1)
        write_csv(sample_audit, OUTPUT_FILES["sample_audit"])
        bar.update(1)
        corr.to_csv(CFG.output_dir / OUTPUT_FILES["correlation"], encoding="utf-8-sig")
        bar.update(1)
        write_csv(high_corr, OUTPUT_FILES["high_correlation"])
        bar.update(1)

    site_master = build_site_level_table(master)
    site_model_summary, site_coefficients, site_fitted = fit_site_level_models(site_master)
    event_model_summary, event_coefficients, event_fitted = fit_candidate_models(master)
    lmm_diagnostic = build_lmm_diagnostic(event_model_summary)

    with progress_bar(7, "步骤6/9 写出主分析与补充模型结果", "输出") as bar:
        write_csv(vif_table, OUTPUT_FILES["vif"])
        bar.update(1)
        write_csv(site_master, OUTPUT_FILES["site_master"])
        bar.update(1)
        write_csv(site_model_summary, OUTPUT_FILES["site_model_comparison"])
        bar.update(1)
        write_csv(site_coefficients, OUTPUT_FILES["site_coefficients"])
        bar.update(1)
        write_csv(event_model_summary, OUTPUT_FILES["event_model_comparison"])
        bar.update(1)
        write_csv(event_coefficients, OUTPUT_FILES["event_coefficients"])
        bar.update(1)
        write_csv(lmm_diagnostic, OUTPUT_FILES["lmm_diagnostic"])
        bar.update(1)

    final_model, final_method = choose_final_model(site_model_summary)
    final_terms = SITE_LEVEL_MODELS.get(final_model, {}).get("terms", ["z_Drought_Intensity", "z_PET_Anomaly"])

    loso = leave_one_site_out_site_level(site_master, available_terms(site_master, final_terms))
    alternatives = fit_alternative_models(master)

    with progress_bar(3, "步骤8/9 绘制最终模型图件", "绘图") as bar:
        if final_model:
            plot_forest(site_coefficients, site_model_summary, final_model, final_method or "OLS_HC3")
        bar.update(1)
        if final_model:
            plot_prediction_and_residual(site_fitted, final_model, final_method or "OLS_HC3")
        bar.update(1)
        bar.update(1)

    run_summary, run_text = build_run_summary(master, site_model_summary, final_model, final_method)

    with progress_bar(4, "步骤8/9 写出敏感性与运行摘要", "输出") as bar:
        write_csv(loso, OUTPUT_FILES["leave_one_site"])
        bar.update(1)
        write_csv(alternatives, OUTPUT_FILES["alternative_models"])
        bar.update(1)
        write_csv(run_summary, OUTPUT_FILES["run_summary_csv"])
        bar.update(1)
        write_text(run_text, OUTPUT_FILES["run_summary_txt"])
        bar.update(1)

    cleanup_records = cleanup_runtime_cache()
    write_csv(cleanup_records, OUTPUT_FILES["cache_cleanup"])

    print(f"Why阶段建模脚本运行完成。输出目录：{CFG.output_dir}")
    print(f"最终绘图模型：{final_model or '未选择'}")
    print(f"最终绘图方法：{final_method or '未选择'}")


if __name__ == "__main__":
    main()
