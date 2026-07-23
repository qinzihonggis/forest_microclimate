# -*- coding: utf-8 -*-
"""
森林微气候缓冲能力空间差异（Why部分）机器学习解释脚本
====================================================

一、脚本目标
------------
本脚本用于论文 Why 部分，回答：

    哪些环境因子能够解释不同站点、不同干旱等级下森林微气候缓冲能力变化
    幅度（DeltaCBI）的空间差异？

研究背景：
    What 部分已经完成干旱状态识别、CBI/DeltaCBI 计算、Wilcoxon 检验、
    Kruskal-Wallis 梯度检验以及多轮稳健性/敏感性分析。Why 部分不再重复
    证明“干旱是否改变 CBI”，而是进一步解释“为什么不同站点或不同干旱事件
    的 DeltaCBI 大小不同”。

方法定位：
    早期方案曾考虑 LMM + VIF，但站点样本量只有 n=13~27，且站点-月重复结构
    不均衡。LMM 的随机效应方差估计和 VIF 驱动的变量筛选在这种小样本下容易
    不稳定。因此本脚本采用 Random Forest 作为探索性解释工具，用交叉验证、
    基线比较、缺失值审计、相关性审计和重要性稳定性检查约束小样本不确定性。

解释边界：
    本脚本结果定位为 exploratory explanatory modeling，用于提示变量重要性
    排序和可能的方向趋势，不作为严格因果推断，也不写成确定性统计结论。
    论文表述中应避免“稳定证明”“广泛认可”“确定驱动因素”等过强说法。

为什么首选 Random Forest：
    Random Forest 是 bagging 机制，通过 bootstrap 抽样和随机特征子集训练多棵
    树后平均，超参数相对少，在小样本场景下比 XGBoost 这类 boosting 方法风险
    更低。XGBoost 对学习率、树深度、正则化和子采样等超参数更敏感，小样本下
    调参本身容易造成结果偏乐观。因此当前脚本把随机森林作为主分析；XGBoost
    不纳入默认流程，最多可作为后续补充稳健性对照。

二、总体设计
------------
脚本按“两步走”运行。

Step 1：构建四等级站点-月 DeltaCBI 基础表
    从逐小时温度表和逐日 SPI30d 宽表重新计算 Mild / Moderate / Severe /
    Extreme 四个干旱等级在 Site_ID x YearMonth 层面的 CBI 变化，并输出：

        01_四等级站点月缓冲变化基础表.csv

    该表是后续机器学习的核心输入，也可供以后复用。

Step 2：机器学习解释
    基于 Step 1 的基础表分别构建两个分析层级：

    1. 站点级（site）
       一行 = Site_ID x DroughtLevel。
       这是主分析，用于回答“哪里的森林缓冲能力对干旱更敏感”。

    2. 站点-月级（site_month）
       一行 = Site_ID x YearMonth x DroughtLevel。
       这是互补分析，用于回答“某次干旱发生时，动态环境因子如何调节缓冲表现”。

两层级关系：
    站点级是正文主分析，直接对应“空间差异”；站点-月级不是替代分析，而是
    平行互补分析，用于保留事件发生时间点的信息。若两层级重要性排序一致，
    可说明结论更稳健；若不一致，可解释为“地形/冠层等静态因子塑造长期空间
    格局，而土壤水分/土壤温度等动态因子调节具体事件表现强度”。

三、核心响应变量
----------------
CBI 定义为逐小时线性回归斜率：

    Observed_T15cm_C = intercept + CBI x ERA5_T2m_C

在每个 Site_ID x YearMonth x DroughtLevel 中：

    Target_CBI = 该干旱等级小时数据估计的 CBI
    Normal_CBI = 同一站点、同一月份 Normal 小时数据估计的 CBI
    DeltaCBI   = Target_CBI - Normal_CBI

解释方向：

    DeltaCBI > 0：干旱期 CBI 高于 Normal，表示林下温度更随宏气温波动，缓冲减弱。
    DeltaCBI < 0：干旱期 CBI 低于 Normal，表示表观缓冲增强或维持。

进入机器学习的记录必须满足 Pair_flag == "ok"。

四、干旱等级和样本筛选
----------------------
SPI30d 分级边界与现有多等级分析保持一致：

    Normal   : -0.5 < SPI30d < 0.5
    Mild     : -1.0 < SPI30d <= -0.5
    Moderate : -1.5 < SPI30d <= -1.0
    Severe   : -2.0 < SPI30d <= -1.5
    Extreme  : SPI30d <= -2.0

每个 Site_ID x YearMonth x DroughtLevel 要形成有效配对，默认要求：

    Target 小时数 >= 72
    Normal 小时数 >= 72
    Target_CBI 和 Normal_CBI 均成功估计

MacroSD 默认只作为审计字段保留，不参与 Pair_flag 筛选。若需要把 MacroSD 也
作为硬筛选条件，可将 Config.use_macro_sd_for_pair_flag 改为 True。

同月多次同等级干旱的处理：
    本脚本在 Site_ID x YearMonth x DroughtLevel 层面直接按小时集合估计
    Target_CBI，因此 Target_CBI/DeltaCBI 本身已经按该等级在该月内实际出现的
    小时数自然加权。对应的动态环境变量按该等级实际目标日期窗口聚合；站点级
    再按 DurationDays 加权平均，避免把较短事件和较长事件简单等权处理。

注意：
    EventStartDate / EventEndDate 在基础表中记录的是该站点-月-等级目标日期的
    最早和最晚日期；若同月存在多段不连续同等级干旱，N_events_in_site_month
    会记录连续片段数。该字段用于审计，不表示中间所有日期都是同一个连续事件。

五、实际输入数据
----------------
1. hourly_temperature_csv
   E:/forest_microclimate/ForestMicroclimate/results/时间序列图/逐小时温度对齐表.csv
   用途：提供逐小时 ERA5_T2m_C 和 Observed_T15cm_C，用于计算 CBI。

2. spi_daily_wide_xlsx
   E:/forest_microclimate/ForestMicroclimate/results/daily_SPI_result/各站点SPI30d逐日宽表_2025.xlsx
   用途：给每个站点、每个 UTC 日期匹配 SPI30d，并划分干旱等级。

3. static_site_csv
   E:/forest_microclimate/ForestMicroclimate/Tensor_LatLong.csv
   用途：提供 Longitude、Latitude、Elevation、Slope、Aspect、Canopy_Height。
   其中 Aspect 不直接入模，会转换为 aspect_sin 和 aspect_cos。

4. lai_8day_csv / fapar_8day_csv
   用途：提供 8 日尺度 LAI 和 FAPAR。脚本会把每个 8 日产品日期展开为 8 个
   日值，再按目标干旱日期窗口取均值。

5. micro_soil_daily_csv
   E:/forest_microclimate/ForestMicroclimate/MicroTandSoilT.csv
   用途：当前使用 VWC_Daily 作为 soil_moisture，使用 T-5cm_Daily 作为
   soil_temperature。

6. ntl_csv / built_up_distance_csv
   当前只是人类活动变量占位路径。如果文件不存在或列全缺失，脚本会在缺失值
   筛查中自动剔除，不会因为这两个变量缺失而删除样本。

当前尚未真正纳入的方案变量：

    root_zone_soil_moisture：脚本中尚未配置真实输入源。
    soil_type：若 static_site_csv 中有 soil_type 字段会自动 one-hot；当前表没有该字段。
    nighttime_light / built_up_distance：当前为占位，因全缺失默认不入模。

为什么这些数据未输入也能跑出结果：
    机器学习主表允许候选变量缺失。脚本会先生成缺失率报告，再删除缺失率超过
    missing_drop_threshold 的列；因此人类活动变量等全 NaN 占位列不会进入模型，
    也不会导致整行样本被删除。当前结果代表“使用已有干旱、地形、植被、土壤
    水分/土壤温度变量”的一版可运行探索性分析，不代表方案中所有变量都已完备。

六、站点级聚合规则
------------------
站点级主分析由站点-月级有效记录聚合得到：

    DeltaCBI：按 DurationDays 加权平均
    Duration_days：该等级下累计干旱天数
    SPI_intensity：取该站点该等级中最小 SPI 的相反数，数值越大表示最强干旱越强
    LAI / FAPAR / soil_moisture / soil_temperature：按 DurationDays 加权平均
    elevation / slope / aspect_sin / aspect_cos / canopy_height：站点静态值直接沿用

七、当前候选特征
----------------
默认候选特征为：

    SPI_intensity
    Duration_days
    elevation
    slope
    aspect_sin
    aspect_cos
    LAI
    FAPAR
    canopy_height
    soil_moisture
    soil_temperature
    nighttime_light
    built_up_distance

实际入模前会执行：

    1. 删除 FEATURES_TO_DROP 中人工指定的变量。
    2. 删除缺失率超过 missing_drop_threshold（默认 50%）的变量。
    3. 删除有效值少于 3 个或唯一值不足的变量。
    4. 对剩余少量缺失用列中位数填补。
    5. 若存在 soil_type，则 one-hot 编码。
    6. 输出“变量缺失率与填补审计表”和“高相关变量对审计表”供审计。

高相关变量不会被脚本自动删除。正式解读变量重要性前，应查看
“高相关变量对审计表”，并在 FEATURES_TO_DROP 中手动二选一后重跑。

共线性说明：
    随机森林预测本身对共线性不像 OLS/LMM 那样敏感，但 SHAP、Permutation 或
    RF importance 的变量重要性排序仍会受高相关变量影响，表现为重要性被分摊
    或排名不稳定。因此相关性审计不是形式步骤；正式论文解释前应人工处理高
    相关变量组，例如 LAI/FAPAR 或 soil_moisture/其他水分指标只保留一个代表。

八、交叉验证策略
----------------
站点级：

    Mild / Moderate：5-fold CV
    Severe / Extreme：Leave-One-Out CV

站点-月级：

    所有等级均使用 Leave-One-Site-Out CV。
    这样同一站点不会同时出现在训练集和验证集，避免信息泄漏。

基线模型：

    每个 CV fold 内用训练集 y 均值预测验证集，计算 Baseline_RMSE_cv。
    这样不会把验证集信息泄漏进基线。

可信度判断：
    小样本机器学习不应只看 R2 高低。更重要的是：
    1. CV-RMSE 是否优于训练折均值基线；
    2. 前几位重要变量是否在不同随机种子下稳定；
    3. 站点级和站点-月级结论是否一致或能否被生态学机制合理解释；
    4. 不同干旱等级下的重要性变化是否符合生态学预期。

九、随机森林和重要性输出
------------------------
默认随机森林参数：

    n_estimators = 100
    max_depth = 4
    min_samples_leaf = 2
    random_seed = 20250714

默认关闭：

    enable_shap = False
    enable_cv_permutation = False

原因：四等级 x 两层级全流程包含大量 Leave-One-Site-Out 训练，SHAP 和
CV permutation 在完整运行中非常耗时。默认版本优先保证完整流程可跑通，并使用
验证折训练模型的 Random Forest impurity importance 作为快速 fallback。

如果只想深度解释某一个等级/层级，可以修改：

    run_levels = ("Extreme",)
    run_layers = ("site",)
    enable_shap = True
    enable_cv_permutation = True

十、主要输出
------------
所有输出写入：

    E:/forest_microclimate/ForestMicroclimate/results/Modeling_Machine_Learning_Explanation

关键文件：

    00_输入路径审计表.csv
    00_参数配置表.csv
    00_绘图与进度条参数说明表.csv
    00_机器学习解释建模方法说明书.docx
    01_四等级站点月缓冲变化基础表.csv
    01_四等级站点月缓冲变化基础表审计表.csv
    01_四等级基础表有效样本量统计.csv
    02_站点级机器学习分析表.csv
    02_站点月级机器学习分析表.csv
    03_{干旱等级}_{分析层级}_变量缺失率与填补审计表.csv
    03_{干旱等级}_{分析层级}_高相关变量对审计表.csv
    03_{干旱等级}_{分析层级}_斯皮尔曼候选变量相关矩阵.csv
    04_{干旱等级}_{分析层级}_随机森林交叉验证模型表现表.csv
    05_{干旱等级}_{分析层级}_交叉验证重要性排序表.csv
    05_{干旱等级}_{分析层级}_交叉验证变量重要性图.png
    06_{干旱等级}_{分析层级}_全样本重要性稳定性表.csv
    07_{干旱等级}_{分析层级}_交叉验证预测结果表.csv
    08_跨等级跨层级模型表现汇总表.csv
    08_跨等级跨层级交叉验证重要性汇总表.csv
    08_跨等级跨层级交叉验证置换重要性汇总表.csv
    08_跨等级跨层级全样本稳定性汇总表.csv
    20_运行摘要说明.txt
    21_本次运行临时缓存清理记录.csv

十一、运行界面和可调参数
------------------------
推荐解释器路径：

    D:/ProgramData/anaconda3/envs/gee/python.exe

该路径只用于脚本审计和参数记录，脚本不会在运行时自动切换 Python 解释器。

关键步骤使用 tqdm 单行动态进度条。进度条显示百分比、当前量/总量、耗时、
预计剩余时间和速度；读取、构建、聚合、建模、绘图、输出、清理使用不同颜色。

图像尺寸、柱形颜色、透明度、误差线颜色、DPI、字体候选，以及进度条格式都集中
放在 PLOT_PARAMS / PROGRESS_PARAMS / PROGRESS_COLORS 中，并会输出到
00_绘图与进度条参数说明表.csv，便于后续直接修改。

脚本运行结束会删除本次运行创建的临时缓存目录，只保留正式输出表格、图像和说明。
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, LeaveOneOut
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    shap = import_module("shap")
    HAS_SHAP = True
except Exception:
    shap = None
    HAS_SHAP = False

try:
    Document = import_module("docx").Document
    HAS_DOCX = True
except Exception:
    Document = None
    HAS_DOCX = False


# =============================================================================
# 0. 全局配置
# =============================================================================


@dataclass(frozen=True)
class Config:
    # 项目根目录和输出目录。所有正式结果只写入 output_dir，不散落到输入数据目录。
    project_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate")
    output_dir: Path = Path(
        r"E:\forest_microclimate\ForestMicroclimate\results\Modeling_Machine_Learning_Explanation"
    )

    # 推荐解释器路径，只用于参数审计和复现记录；脚本不会自动切换解释器。
    python_env_dir: Path = Path(r"D:\ProgramData\anaconda3\envs\gee")
    python_interpreter: Path = Path(r"D:\ProgramData\anaconda3\envs\gee\python.exe")

    # Step 1 输入：逐小时温度和逐日 SPI，用于重建 Site_ID x YearMonth x DroughtLevel 的 DeltaCBI。
    hourly_temperature_csv: Path = Path(
        r"E:\forest_microclimate\ForestMicroclimate\results\时间序列图\逐小时温度对齐表.csv"
    )
    spi_daily_wide_xlsx: Path = Path(
        r"E:\forest_microclimate\ForestMicroclimate\results\daily_SPI_result\各站点SPI30d逐日宽表_2025.xlsx"
    )
    static_site_csv: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\Tensor_LatLong.csv")
    lai_8day_csv: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\LAI\站点LAI_8日尺度提取结果.csv")
    fapar_8day_csv: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\FAPAR\站点FAPAR_8日尺度提取结果.csv")
    micro_soil_daily_csv: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\MicroTandSoilT.csv")

    # Optional placeholders. If files are absent or all-NaN, columns are dropped by missing-value rules.
    ntl_csv: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\HumanActivity\nighttime_light.csv")
    built_up_distance_csv: Path = Path(
        r"E:\forest_microclimate\ForestMicroclimate\HumanActivity\built_up_distance.csv"
    )

    # 输入表中的关键字段名。如上游表字段改名，只需要优先检查这里。
    site_col: str = "Site_ID"
    time_col: str = "Time_UTC"
    macro_temp_col: str = "ERA5_T2m_C"
    micro_temp_col: str = "Observed_T15cm_C"
    has_both_col: str = "Has_Both_Data"

    # 干旱和 Normal 的 SPI30d 阈值；必须与 What 部分保持一致。
    normal_spi_low: float = -0.5
    normal_spi_high: float = 0.5
    extreme_spi_threshold: float = -2.0

    # CBI 配对有效性阈值。min_status_hours 控制每个 Target/Normal 至少需要多少小时。
    min_status_hours: int = 72
    min_macro_sd: float = 1.0
    use_macro_sd_for_pair_flag: bool = False

    # 特征预处理和模型参数。缺失率高的列会被剔除，剩余少量缺失用中位数填补。
    missing_drop_threshold: float = 0.50
    high_corr_threshold: float = 0.85
    random_seed: int = 20250714
    n_estimators: int = 100
    max_depth: int = 4
    min_samples_leaf: int = 2
    n_stability_seeds: int = 5
    n_permutation_repeats: int = 5

    # SHAP / CV permutation 较耗时，默认关闭；若只跑单个等级/层级，可打开做深度解释。
    enable_shap: bool = False
    enable_cv_permutation: bool = False
    run_levels: tuple[str, ...] = ("Mild", "Moderate", "Severe", "Extreme")
    run_layers: tuple[str, ...] = ("site", "site_month")


CFG = Config()
RUN_START_TIME = time.time()
RUNTIME_CACHE_DIR = CFG.output_dir / f"本次运行临时缓存_{int(RUN_START_TIME)}"

DROUGHT_LEVELS = ["Mild", "Moderate", "Severe", "Extreme"]
DROUGHT_LEVELS_CN = {
    "Mild": "轻度干旱",
    "Moderate": "中度干旱",
    "Severe": "重度干旱",
    "Extreme": "极端干旱",
}

CV_STRATEGY_BY_LEVEL = {
    "Mild": {"method": "kfold", "k": 5},
    "Moderate": {"method": "kfold", "k": 5},
    "Severe": {"method": "loo"},
    "Extreme": {"method": "loo"},
}

# 查看“高相关变量对审计表”后，可在这里手动写入需要剔除的变量名。
# 脚本只提示高相关风险，不自动替研究者决定保留哪个变量。
FEATURES_TO_DROP: list[str] = []


BASE_FEATURE_COLS = [
    "SPI_intensity",
    "Duration_days",
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "LAI",
    "FAPAR",
    "canopy_height",
    "soil_moisture",
    "soil_temperature",
    "nighttime_light",
    "built_up_distance",
]

LAYER_CN = {"site": "站点级", "site_month": "站点月级"}

OUTPUT_NAMES = {
    "path_audit": "00_输入路径审计表.csv",
    "parameter_table": "00_参数配置表.csv",
    "plot_progress_parameter_table": "00_绘图与进度条参数说明表.csv",
    "method_docx": "00_机器学习解释建模方法说明书.docx",
    "site_month_delta": "01_四等级站点月缓冲变化基础表.csv",
    "site_month_delta_audit": "01_四等级站点月缓冲变化基础表审计表.csv",
    "sample_counts": "01_四等级基础表有效样本量统计.csv",
    "site_level_ml_table": "02_站点级机器学习分析表.csv",
    "site_month_ml_table": "02_站点月级机器学习分析表.csv",
    "cross_summary": "08_跨等级跨层级模型表现汇总表.csv",
    "cross_importance": "08_跨等级跨层级交叉验证重要性汇总表.csv",
    "cross_permutation": "08_跨等级跨层级交叉验证置换重要性汇总表.csv",
    "cross_stability": "08_跨等级跨层级全样本稳定性汇总表.csv",
    "run_summary": "20_运行摘要说明.txt",
    "cache_cleanup": "21_本次运行临时缓存清理记录.csv",
}

# 绘图参数集中放在这里，方便后续调整变量重要性图的尺寸、颜色、透明度、字体和清晰度。
PLOT_PARAMS = {
    "importance_fig_width": 8.0,
    "importance_base_height": 4.0,
    "importance_height_per_feature": 0.35,
    "importance_bar_color": "#3b7a8f",
    "importance_bar_alpha": 0.85,
    "importance_errorbar_color": "#264653",
    "importance_dpi": 300,
    "title_fontsize": 13,
    "axis_label_fontsize": 11,
    "tick_label_fontsize": 10,
    "legend_fontsize": 10,
    "line_width": 2.0,
    "line_alpha": 0.90,
    "font_candidates": "Microsoft YaHei, SimHei, SimSun, Arial Unicode MS",
}

# tqdm 进度条参数。bar_format 控制显示内容；leave=False 表示步骤完成后不堆叠刷屏。
PROGRESS_PARAMS = {
    "bar_format": "{l_bar}{bar}| {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    "leave": False,
    "dynamic_ncols": True,
}

# 不同类型步骤使用不同颜色，便于在终端里快速区分当前正在做什么。
PROGRESS_COLORS = {
    "读取": "cyan",
    "构建": "green",
    "聚合": "yellow",
    "建模": "magenta",
    "绘图": "red",
    "输出": "blue",
    "清理": "white",
}


# =============================================================================
# 1. 通用工具
# =============================================================================


def ensure_output_dir() -> None:
    """Create the formal output directory and a per-run temporary cache directory."""

    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    RUNTIME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(RUNTIME_CACHE_DIR)


def cleanup_runtime_cache(start_time: float) -> pd.DataFrame:
    """Remove temporary files created during this run and return a cleanup audit table."""

    existed_before_cleanup = RUNTIME_CACHE_DIR.exists()
    removed = False
    error = ""
    try:
        if existed_before_cleanup:
            with progress_bar(1, "步骤10/10 清理本次运行临时缓存", "清理") as bar:
                shutil.rmtree(RUNTIME_CACHE_DIR, ignore_errors=False)
                removed = True
                bar.update(1)
    except Exception as exc:
        error = repr(exc)
        shutil.rmtree(RUNTIME_CACHE_DIR, ignore_errors=True)
    return pd.DataFrame(
        [
            {
                "缓存目录": str(RUNTIME_CACHE_DIR),
                "清理前是否存在": existed_before_cleanup,
                "是否已删除": removed or not RUNTIME_CACHE_DIR.exists(),
                "运行耗时_秒": round(time.time() - start_time, 3),
                "错误信息": error,
            }
        ]
    )


def progress_bar(total: int, desc: str, kind: str) -> tqdm:
    """Create a single-line colored tqdm progress bar for one key step.

    Parameters:
        total: Number of units in this step.
        desc: Short Chinese description shown on the left of the bar.
        kind: Progress type. Different kinds use different colors so users can
            distinguish reading, building, modeling, plotting, output, and cleanup.
    """

    kwargs = {
        "total": max(int(total), 1),
        "desc": desc,
        "leave": PROGRESS_PARAMS["leave"],
        "dynamic_ncols": PROGRESS_PARAMS["dynamic_ncols"],
        "bar_format": PROGRESS_PARAMS["bar_format"],
    }
    try:
        return tqdm(**kwargs, colour=PROGRESS_COLORS.get(kind, "green"))
    except TypeError:
        return tqdm(**kwargs)


def level_layer_label(level_name: str, layer_name: str) -> str:
    return f"{DROUGHT_LEVELS_CN.get(level_name, level_name)}_{LAYER_CN.get(layer_name, layer_name)}"


def output_name_for(kind: str, level_name: str, layer_name: str, suffix: str = "csv") -> str:
    label = level_layer_label(level_name, layer_name)
    names = {
        "missing": f"03_{label}_变量缺失率与填补审计表.{suffix}",
        "high_corr": f"03_{label}_高相关变量对审计表.{suffix}",
        "corr": f"03_{label}_斯皮尔曼候选变量相关矩阵.{suffix}",
        "model_summary": f"04_{label}_随机森林交叉验证模型表现表.{suffix}",
        "importance": f"05_{label}_交叉验证重要性排序表.{suffix}",
        "permutation": f"05_{label}_交叉验证置换重要性排序表.{suffix}",
        "stability": f"06_{label}_全样本重要性稳定性表.{suffix}",
        "predictions": f"07_{label}_交叉验证预测结果表.{suffix}",
        "importance_plot": f"05_{label}_交叉验证变量重要性图.{suffix}",
        "permutation_plot": f"05_{label}_交叉验证置换重要性图.{suffix}",
    }
    return names[kind]


def normalise_site_id(series: pd.Series) -> pd.Series:
    """Normalize station IDs read as numeric/string into stable string IDs."""

    def _one(value):
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        if text.endswith(".0"):
            text = text[:-2]
        return text

    return series.map(_one)


def write_csv(df: pd.DataFrame, name: str) -> None:
    path = CFG.output_dir / name
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(text: str, name: str) -> None:
    (CFG.output_dir / name).write_text(text, encoding="utf-8")


def setup_plot_style() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    # 按候选顺序设置中文字体；若系统缺少某字体，matplotlib 会继续使用可用字体。
    font_candidates = [font.strip() for font in str(PLOT_PARAMS["font_candidates"]).split(",")]
    for font_name in font_candidates:
        try:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
        except Exception:
            continue


def build_plot_progress_parameter_table() -> pd.DataFrame:
    """Record plot and progress-bar parameters for later manual adjustment."""

    rows = []
    for key, value in PLOT_PARAMS.items():
        rows.append({"参数类别": "绘图参数", "参数名": key, "当前值": str(value), "用途": "控制变量重要性图的尺寸、颜色、透明度、字体或DPI"})
    for key, value in PROGRESS_PARAMS.items():
        rows.append({"参数类别": "进度条参数", "参数名": key, "当前值": str(value), "用途": "控制tqdm单行动态进度条显示格式"})
    for key, value in PROGRESS_COLORS.items():
        rows.append({"参数类别": "进度条颜色", "参数名": key, "当前值": str(value), "用途": "不同类型关键步骤的彩色进度条颜色"})
    return pd.DataFrame(rows)


def classify_drought_level(spi: pd.Series) -> pd.Series:
    """Use the same SPI class boundaries as the existing multi-level analysis."""

    conditions = [
        (spi > CFG.normal_spi_low) & (spi < CFG.normal_spi_high),
        (spi <= CFG.normal_spi_low) & (spi > -1.0),
        (spi <= -1.0) & (spi > -1.5),
        (spi <= -1.5) & (spi > CFG.extreme_spi_threshold),
        spi <= CFG.extreme_spi_threshold,
    ]
    choices = ["Normal", "Mild", "Moderate", "Severe", "Extreme"]
    return pd.Series(np.select(conditions, choices, default="Other"), index=spi.index)


def calc_ols_cbi(df: pd.DataFrame) -> dict:
    """Estimate CBI as slope of microclimate temperature against macro temperature."""

    d = df[[CFG.micro_temp_col, CFG.macro_temp_col]].dropna()
    if len(d) < CFG.min_status_hours:
        return {
            "CBI": np.nan,
            "Intercept": np.nan,
            "R2": np.nan,
            "p_slope": np.nan,
            "n_hours": int(len(d)),
            "Macro_SD": float(d[CFG.macro_temp_col].std(ddof=1)) if len(d) > 1 else np.nan,
            "flag": "too_few_hours",
        }
    macro_sd = float(d[CFG.macro_temp_col].std(ddof=1))
    if not np.isfinite(macro_sd) or macro_sd <= 0:
        return {
            "CBI": np.nan,
            "Intercept": np.nan,
            "R2": np.nan,
            "p_slope": np.nan,
            "n_hours": int(len(d)),
            "Macro_SD": macro_sd,
            "flag": "zero_macro_sd",
        }
    fit = stats.linregress(d[CFG.macro_temp_col].to_numpy(), d[CFG.micro_temp_col].to_numpy())
    return {
        "CBI": float(fit.slope),
        "Intercept": float(fit.intercept),
        "R2": float(fit.rvalue**2),
        "p_slope": float(fit.pvalue),
        "n_hours": int(len(d)),
        "Macro_SD": macro_sd,
        "flag": "ok",
    }


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


# =============================================================================
# 2. Step 1: 构建四等级站点-月 DeltaCBI 基础表
# =============================================================================


def read_hourly_with_spi() -> pd.DataFrame:
    # 读取并合并逐小时温度和逐日 SPI。这里用一个进度条表示整个“读取输入并分级”关键步骤。
    with progress_bar(5, "步骤1/10 读取逐小时温度和逐日SPI", "读取") as bar:
        hourly = pd.read_csv(CFG.hourly_temperature_csv, encoding="utf-8-sig")
        bar.update(1)

        hourly[CFG.site_col] = normalise_site_id(hourly[CFG.site_col])
        hourly[CFG.time_col] = pd.to_datetime(hourly[CFG.time_col], errors="coerce")
        hourly["UTC_Date"] = hourly[CFG.time_col].dt.floor("D")
        hourly["YearMonth"] = hourly[CFG.time_col].dt.to_period("M").astype(str)
        if CFG.has_both_col in hourly.columns:
            hourly = hourly.loc[hourly[CFG.has_both_col].astype(bool)].copy()
        hourly = hourly.loc[
            hourly[CFG.time_col].notna()
            & hourly[CFG.macro_temp_col].notna()
            & hourly[CFG.micro_temp_col].notna()
        ].copy()
        bar.update(1)

        spi_wide = pd.read_excel(CFG.spi_daily_wide_xlsx)
        bar.update(1)

        date_col = spi_wide.columns[0]
        spi_long = spi_wide.melt(id_vars=[date_col], var_name=CFG.site_col, value_name="SPI30d")
        spi_long = spi_long.rename(columns={date_col: "UTC_Date"})
        spi_long[CFG.site_col] = normalise_site_id(spi_long[CFG.site_col])
        spi_long["UTC_Date"] = pd.to_datetime(spi_long["UTC_Date"], errors="coerce")
        bar.update(1)

        hourly = hourly.merge(spi_long, on=[CFG.site_col, "UTC_Date"], how="left", validate="many_to_one")
        hourly["DroughtLevel"] = classify_drought_level(hourly["SPI30d"])
        hourly["DroughtLevel_CN"] = hourly["DroughtLevel"].map(DROUGHT_LEVELS_CN).fillna(hourly["DroughtLevel"])
        hourly["Site_Month"] = hourly[CFG.site_col].astype(str) + "_" + hourly["YearMonth"]
        bar.update(1)
    return hourly


def count_contiguous_runs(dates: Iterable[pd.Timestamp]) -> int:
    clean = sorted(pd.to_datetime(pd.Series(list(dates)).dropna()).dt.floor("D").unique())
    if not clean:
        return 0
    n_runs = 1
    for prev, cur in zip(clean[:-1], clean[1:]):
        if (cur - prev).days > 1:
            n_runs += 1
    return n_runs


def build_site_month_delta_cbi_by_level(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    audit_rows = []
    group_cols = [CFG.site_col, "YearMonth"]
    grouped = list(hourly.groupby(group_cols, sort=True))

    with progress_bar(len(grouped), "步骤2/10 构建四等级站点月DeltaCBI基础表", "构建") as bar:
        for (site_id, year_month), site_month in grouped:
            normal = site_month.loc[site_month["DroughtLevel"].eq("Normal")]
            normal_result = calc_ols_cbi(normal)

            for level in DROUGHT_LEVELS:
                target = site_month.loc[site_month["DroughtLevel"].eq(level)]
                target_result = calc_ols_cbi(target)

                pass_hours = (
                    target_result["n_hours"] >= CFG.min_status_hours
                    and normal_result["n_hours"] >= CFG.min_status_hours
                )
                pass_macro_sd = (
                    pd.notna(target_result["Macro_SD"])
                    and pd.notna(normal_result["Macro_SD"])
                    and target_result["Macro_SD"] >= CFG.min_macro_sd
                    and normal_result["Macro_SD"] >= CFG.min_macro_sd
                )
                if CFG.use_macro_sd_for_pair_flag:
                    pair_ok = pass_hours and pass_macro_sd and target_result["flag"] == "ok" and normal_result["flag"] == "ok"
                else:
                    pair_ok = pass_hours and target_result["flag"] == "ok" and normal_result["flag"] == "ok"

                target_dates = pd.to_datetime(target["UTC_Date"].dropna().unique())
                min_spi = float(target["SPI30d"].min()) if not target.empty else np.nan
                duration_days = int(len(pd.Series(target_dates).dropna().unique())) if len(target_dates) else 0
                start_date = pd.Series(target_dates).min() if len(target_dates) else pd.NaT
                end_date = pd.Series(target_dates).max() if len(target_dates) else pd.NaT
                n_events = count_contiguous_runs(target_dates)

                delta = (
                    target_result["CBI"] - normal_result["CBI"]
                    if pair_ok and pd.notna(target_result["CBI"]) and pd.notna(normal_result["CBI"])
                    else np.nan
                )
                if pair_ok:
                    pair_flag = "ok"
                elif not pass_hours:
                    pair_flag = "too_few_hours"
                elif CFG.use_macro_sd_for_pair_flag and not pass_macro_sd:
                    pair_flag = "low_macro_sd"
                else:
                    pair_flag = "cbi_failed"

                row = {
                    "Site_ID": site_id,
                    "YearMonth": year_month,
                    "DroughtLevel": level,
                    "DroughtLevel_CN": DROUGHT_LEVELS_CN[level],
                    "Target_CBI": target_result["CBI"],
                    "Normal_CBI": normal_result["CBI"],
                    "DeltaCBI": delta,
                    "Target_Intercept": target_result["Intercept"],
                    "Normal_Intercept": normal_result["Intercept"],
                    "Target_R2": target_result["R2"],
                    "Normal_R2": normal_result["R2"],
                    "Target_n_hours": target_result["n_hours"],
                    "Normal_n_hours": normal_result["n_hours"],
                    "Target_Macro_SD": target_result["Macro_SD"],
                    "Normal_Macro_SD": normal_result["Macro_SD"],
                    "Pass_Hours": bool(pass_hours),
                    "Pass_Macro_SD": bool(pass_macro_sd),
                    "Pair_flag": pair_flag,
                    "MinDailySPI": min_spi,
                    "SPI_intensity": -min_spi if pd.notna(min_spi) else np.nan,
                    "DurationDays": duration_days,
                    "EventStartDate": start_date,
                    "EventEndDate": end_date,
                    "N_events_in_site_month": n_events,
                }
                rows.append(row)
                audit_rows.append(
                    {
                        "Site_ID": site_id,
                        "YearMonth": year_month,
                        "DroughtLevel": level,
                        "Target_n_hours": target_result["n_hours"],
                        "Normal_n_hours": normal_result["n_hours"],
                        "Target_flag": target_result["flag"],
                        "Normal_flag": normal_result["flag"],
                        "Pair_flag": pair_flag,
                    }
                )
            bar.update(1)

    out = pd.DataFrame(rows)
    audit = pd.DataFrame(audit_rows)
    return out, audit


# =============================================================================
# 3. 协变量构建
# =============================================================================


def wide_time_series_to_long(path: Path, value_name: str, date_col: str = "datetime") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=[CFG.site_col, "Date", value_name])
    df = pd.read_csv(path, encoding="utf-8-sig")
    if date_col not in df.columns:
        date_col = df.columns[0]
    long = df.melt(id_vars=[date_col], var_name=CFG.site_col, value_name=value_name)
    long = long.rename(columns={date_col: "Date"})
    long[CFG.site_col] = normalise_site_id(long[CFG.site_col])
    long["Date"] = pd.to_datetime(long["Date"], errors="coerce")
    return long


def expand_8day_to_daily(long_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if long_df.empty:
        return long_df
    rows = []
    # 8日产品需要展开到逐日，后续才能按每个干旱窗口精确取均值。
    valid_rows = long_df.dropna(subset=["Date"])
    for _, row in valid_rows.iterrows():
        for offset in range(8):
            rows.append(
                {
                    CFG.site_col: row[CFG.site_col],
                    "Date": row["Date"] + pd.Timedelta(days=offset),
                    value_col: row[value_col],
                }
            )
    return pd.DataFrame(rows)


def load_static_attributes() -> pd.DataFrame:
    static = pd.read_csv(CFG.static_site_csv, encoding="utf-8-sig")
    static[CFG.site_col] = normalise_site_id(static[CFG.site_col])
    rename_map = {
        "Elevation": "elevation",
        "Slope": "slope",
        "Aspect": "aspect",
        "Canopy_Height": "canopy_height",
        "DEM": "elevation",
    }
    static = static.rename(columns={k: v for k, v in rename_map.items() if k in static.columns})
    keep = [CFG.site_col] + [c for c in ["Longitude", "Latitude", "elevation", "slope", "aspect", "canopy_height", "soil_type"] if c in static.columns]
    static = static[keep].copy()
    if "aspect" in static.columns:
        radians = np.deg2rad(pd.to_numeric(static["aspect"], errors="coerce"))
        static["aspect_sin"] = np.sin(radians)
        static["aspect_cos"] = np.cos(radians)
        static = static.drop(columns=["aspect"])
    return static


def load_daily_soil() -> pd.DataFrame:
    if not CFG.micro_soil_daily_csv.exists():
        return pd.DataFrame(columns=[CFG.site_col, "Date", "soil_moisture", "soil_temperature"])
    soil = pd.read_csv(CFG.micro_soil_daily_csv, encoding="utf-8-sig")
    soil[CFG.site_col] = normalise_site_id(soil[CFG.site_col])
    soil["Date"] = pd.to_datetime(soil["Date"], errors="coerce")
    rename = {}
    if "VWC_Daily" in soil.columns:
        rename["VWC_Daily"] = "soil_moisture"
    if "T-5cm_Daily" in soil.columns:
        rename["T-5cm_Daily"] = "soil_temperature"
    soil = soil.rename(columns=rename)
    keep = [CFG.site_col, "Date"] + [c for c in ["soil_moisture", "soil_temperature"] if c in soil.columns]
    return soil[keep].copy()


def aggregate_daily_window(
    daily_df: pd.DataFrame,
    site_id: str,
    dates: Iterable[pd.Timestamp],
    value_col: str,
) -> float:
    if daily_df.empty or value_col not in daily_df.columns:
        return np.nan
    date_values = pd.to_datetime(pd.Series(list(dates)).dropna()).dt.floor("D").unique()
    if len(date_values) == 0:
        return np.nan
    sub = daily_df.loc[
        (daily_df[CFG.site_col].eq(site_id)) & (daily_df["Date"].isin(date_values)),
        value_col,
    ]
    if sub.dropna().empty:
        return np.nan
    return float(pd.to_numeric(sub, errors="coerce").mean())


def add_time_varying_covariates(site_month_delta: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    # 动态协变量按每个站点-月-干旱等级的目标日期窗口聚合，避免混入非目标干旱日。
    with progress_bar(5, "步骤3/10 构建动态与静态解释变量", "聚合") as setup_bar:
        lai_daily = expand_8day_to_daily(wide_time_series_to_long(CFG.lai_8day_csv, "LAI"), "LAI")
        setup_bar.update(1)
        fapar_daily = expand_8day_to_daily(wide_time_series_to_long(CFG.fapar_8day_csv, "FAPAR"), "FAPAR")
        setup_bar.update(1)
        soil_daily = load_daily_soil()
        setup_bar.update(1)

        key_cols = [CFG.site_col, "YearMonth", "DroughtLevel"]
        target_dates = (
            hourly.loc[hourly["DroughtLevel"].isin(DROUGHT_LEVELS)]
            .groupby(key_cols)["UTC_Date"]
            .apply(lambda x: sorted(pd.to_datetime(x.dropna()).dt.floor("D").unique()))
            .reset_index(name="Target_Dates")
        )
        base = site_month_delta.merge(target_dates, on=key_cols, how="left")
        setup_bar.update(1)
        static_attributes = load_static_attributes()
        setup_bar.update(1)

    rows = []
    with progress_bar(len(base), "步骤4/10 按干旱窗口聚合逐日协变量", "聚合") as bar:
        for _, row in base.iterrows():
            site_id = row[CFG.site_col]
            dates = row["Target_Dates"] if isinstance(row["Target_Dates"], list) else []
            rows.append(
                {
                    "Site_ID": site_id,
                    "YearMonth": row["YearMonth"],
                    "DroughtLevel": row["DroughtLevel"],
                    "LAI": aggregate_daily_window(lai_daily, site_id, dates, "LAI"),
                    "FAPAR": aggregate_daily_window(fapar_daily, site_id, dates, "FAPAR"),
                    "soil_moisture": aggregate_daily_window(soil_daily, site_id, dates, "soil_moisture"),
                    "soil_temperature": aggregate_daily_window(soil_daily, site_id, dates, "soil_temperature"),
                }
            )
            bar.update(1)

    cov = pd.DataFrame(rows)
    out = site_month_delta.merge(cov, on=key_cols, how="left")
    out = out.merge(static_attributes, on=CFG.site_col, how="left")
    out["nighttime_light"] = np.nan
    out["built_up_distance"] = np.nan
    return out


# =============================================================================
# 4. 分析表与预处理
# =============================================================================


def build_site_level_table(site_month_df: pd.DataFrame) -> pd.DataFrame:
    valid = site_month_df.loc[site_month_df["Pair_flag"].eq("ok")].copy()
    if valid.empty:
        return pd.DataFrame()

    rows = []
    groups = list(valid.groupby([CFG.site_col, "DroughtLevel"], sort=True))
    with progress_bar(len(groups), "步骤5/10 聚合站点级机器学习主表", "聚合") as bar:
        for (site_id, level), g in groups:
            w = pd.to_numeric(g["DurationDays"], errors="coerce").fillna(0)
            first = g.iloc[0]
            row = {
                "Site_ID": site_id,
                "DroughtLevel": level,
                "DroughtLevel_CN": DROUGHT_LEVELS_CN[level],
                "n_site_months": int(len(g)),
                "DeltaCBI": weighted_mean(g["DeltaCBI"], w),
                "SPI_intensity": -float(g["MinDailySPI"].min()) if g["MinDailySPI"].notna().any() else np.nan,
                "Duration_days": float(w.sum()),
            }
            for col in [
                "LAI",
                "FAPAR",
                "soil_moisture",
                "soil_temperature",
            ]:
                row[col] = weighted_mean(g[col], w) if col in g.columns else np.nan
            for col in [
                "Longitude",
                "Latitude",
                "elevation",
                "slope",
                "aspect_sin",
                "aspect_cos",
                "canopy_height",
                "soil_type",
                "nighttime_light",
                "built_up_distance",
            ]:
                if col in g.columns:
                    row[col] = first[col]
            rows.append(row)
            bar.update(1)
    return pd.DataFrame(rows)


def build_site_month_ml_table(site_month_df: pd.DataFrame) -> pd.DataFrame:
    d = site_month_df.loc[site_month_df["Pair_flag"].eq("ok")].copy()
    d["Duration_days"] = d["DurationDays"]
    # Use positive intensity so larger means stronger drought.
    d["SPI_intensity"] = -pd.to_numeric(d["MinDailySPI"], errors="coerce")
    return d


def apply_soil_type_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    if "soil_type" not in df.columns:
        return df
    if df["soil_type"].notna().sum() == 0:
        return df.drop(columns=["soil_type"])
    dummies = pd.get_dummies(df["soil_type"].astype("category"), prefix="soil_type", dummy_na=False)
    return pd.concat([df.drop(columns=["soil_type"]), dummies], axis=1)


def candidate_features(df: pd.DataFrame) -> list[str]:
    cols = [c for c in BASE_FEATURE_COLS if c in df.columns]
    soil_dummy_cols = [c for c in df.columns if c.startswith("soil_type_")]
    return cols + soil_dummy_cols


def prepare_features(
    df: pd.DataFrame,
    level_name: str,
    layer_name: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    d = apply_soil_type_one_hot(df.copy())
    y = pd.to_numeric(d["DeltaCBI"], errors="coerce")
    cols = [c for c in candidate_features(d) if c not in FEATURES_TO_DROP]
    X_raw = d[cols].apply(pd.to_numeric, errors="coerce")

    report_rows = []
    keep_cols = []
    for col in X_raw.columns:
        missing_rate = float(X_raw[col].isna().mean())
        non_missing = int(X_raw[col].notna().sum())
        unique_values = int(X_raw[col].nunique(dropna=True))
        drop_reason = ""
        fill_value = np.nan
        if missing_rate > CFG.missing_drop_threshold:
            drop_reason = f"missing_rate>{CFG.missing_drop_threshold}"
        elif non_missing < 3:
            drop_reason = "non_missing<3"
        elif unique_values <= 1:
            drop_reason = "zero_or_one_unique_value"
        else:
            keep_cols.append(col)
            fill_value = float(X_raw[col].median())
        report_rows.append(
            {
                "DroughtLevel": level_name,
                "Layer": layer_name,
                "feature": col,
                "missing_rate": missing_rate,
                "non_missing": non_missing,
                "unique_values": unique_values,
                "dropped": bool(drop_reason),
                "drop_reason": drop_reason,
                "impute_median": fill_value,
            }
        )

    X = X_raw[keep_cols].copy()
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())

    corr_rows = []
    if len(X.columns) >= 2:
        corr = X.corr(method="spearman")
        corr.to_csv(CFG.output_dir / output_name_for("corr", level_name, layer_name), encoding="utf-8-sig")
        for i, a in enumerate(X.columns):
            for b in X.columns[i + 1 :]:
                r = corr.loc[a, b]
                if pd.notna(r) and abs(r) > CFG.high_corr_threshold:
                    corr_rows.append(
                        {
                            "DroughtLevel": level_name,
                            "Layer": layer_name,
                            "feature_1": a,
                            "feature_2": b,
                            "spearman_r": float(r),
                            "warning": "正式解读重要性前建议在 FEATURES_TO_DROP 中人工二选一后重跑",
                        }
                    )
    high_corr = pd.DataFrame(corr_rows)
    return X, y, pd.DataFrame(report_rows), high_corr


# =============================================================================
# 5. 交叉验证与机器学习解释
# =============================================================================


def get_splits(X: pd.DataFrame, y: pd.Series, level_name: str, groups: pd.Series | None):
    n = len(X)
    if groups is not None and groups.nunique() < len(groups):
        splitter = LeaveOneGroupOut()
        return list(splitter.split(X, y, groups=groups)), "Leave-One-Site-Out"
    strategy = CV_STRATEGY_BY_LEVEL[level_name]
    if strategy["method"] == "kfold" and n >= 5:
        k = min(strategy["k"], n)
        splitter = KFold(n_splits=k, shuffle=True, random_state=CFG.random_seed)
        return list(splitter.split(X, y)), f"{k}-fold CV"
    splitter = LeaveOneOut()
    return list(splitter.split(X, y)), "Leave-One-Out"


def make_rf(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=CFG.n_estimators,
        max_depth=CFG.max_depth,
        min_samples_leaf=CFG.min_samples_leaf,
        random_state=seed,
        n_jobs=-1,
    )


def cv_predict_and_importance(
    X: pd.DataFrame,
    y: pd.Series,
    level_name: str,
    layer_name: str,
    groups: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = y.notna()
    Xv = X.loc[valid].reset_index(drop=True)
    yv = y.loc[valid].reset_index(drop=True)
    gv = groups.loc[valid].reset_index(drop=True) if groups is not None else None

    if len(Xv) < 5 or Xv.shape[1] < 1:
        summary = pd.DataFrame(
            [
                {
                    "DroughtLevel": level_name,
                    "Layer": layer_name,
                    "n_samples": len(Xv),
                    "n_features": Xv.shape[1],
                    "CV_method": "not_run",
                    "R2_cv": np.nan,
                    "RMSE_cv": np.nan,
                    "Baseline_RMSE_cv": np.nan,
                    "RMSE_improvement_vs_baseline": np.nan,
                    "note": "n<5 or no features",
                }
            ]
        )
        return summary, pd.DataFrame(), pd.DataFrame()

    splits, cv_desc = get_splits(Xv, yv, level_name, gv)
    y_pred = np.full(len(yv), np.nan, dtype=float)
    baseline_pred = np.full(len(yv), np.nan, dtype=float)
    shap_records = []
    perm_records = []

    label = level_layer_label(level_name, layer_name)
    with progress_bar(len(splits), f"{label} 交叉验证fold建模", "建模") as bar:
        for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
            X_train, X_test = Xv.iloc[train_idx], Xv.iloc[test_idx]
            y_train, y_test = yv.iloc[train_idx], yv.iloc[test_idx]
            model = make_rf(CFG.random_seed + fold_id)
            model.fit(X_train, y_train)
            y_pred[test_idx] = model.predict(X_test)
            baseline_pred[test_idx] = float(y_train.mean())

            if shap is not None and CFG.enable_shap:
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test)
                    mean_abs = np.abs(shap_values).mean(axis=0)
                    for feature, value in zip(Xv.columns, mean_abs):
                        shap_records.append(
                            {
                                "DroughtLevel": level_name,
                                "Layer": layer_name,
                                "fold": fold_id,
                                "feature": feature,
                                "cv_mean_abs_shap": float(value),
                                "n_validation": len(test_idx),
                            }
                        )
                except Exception:
                    pass
            else:
                for feature, value in zip(Xv.columns, model.feature_importances_):
                    shap_records.append(
                        {
                            "DroughtLevel": level_name,
                            "Layer": layer_name,
                            "fold": fold_id,
                            "feature": feature,
                            "cv_mean_abs_shap": float(value),
                            "n_validation": len(test_idx),
                            "note": "SHAP disabled or unavailable; used RF impurity importance as CV fallback",
                        }
                    )

            # Permutation on validation folds is expensive and is disabled by default
            # for full 4-level x 2-layer runs. Enable CFG.enable_cv_permutation for
            # single-combination deep runs if needed.
            if CFG.enable_cv_permutation and len(test_idx) >= 3:
                try:
                    perm = permutation_importance(
                        model,
                        X_test,
                        y_test,
                        n_repeats=CFG.n_permutation_repeats,
                        random_state=CFG.random_seed + fold_id,
                        scoring="neg_mean_squared_error",
                        n_jobs=-1,
                    )
                    for feature, mean, std in zip(Xv.columns, perm.importances_mean, perm.importances_std):
                        perm_records.append(
                            {
                                "DroughtLevel": level_name,
                                "Layer": layer_name,
                                "fold": fold_id,
                                "feature": feature,
                                "cv_permutation_mse_increase_mean": float(mean),
                                "cv_permutation_mse_increase_std": float(std),
                                "n_validation": len(test_idx),
                            }
                        )
                except Exception:
                    pass
            bar.update(1)

    rmse = float(np.sqrt(mean_squared_error(yv, y_pred)))
    baseline_rmse = float(np.sqrt(mean_squared_error(yv, baseline_pred)))
    r2 = float(r2_score(yv, y_pred)) if len(yv) >= 2 else np.nan
    summary = pd.DataFrame(
        [
            {
                "DroughtLevel": level_name,
                "Layer": layer_name,
                "n_samples": len(Xv),
                "n_features": Xv.shape[1],
                "CV_method": cv_desc,
                "R2_cv": r2,
                "RMSE_cv": rmse,
                "Baseline_RMSE_cv": baseline_rmse,
                "RMSE_improvement_vs_baseline": baseline_rmse - rmse,
                "note": "Baseline uses training-fold mean for each validation fold.",
            }
        ]
    )

    shap_df = pd.DataFrame(shap_records)
    if not shap_df.empty:
        shap_df = (
            shap_df.groupby(["DroughtLevel", "Layer", "feature"], as_index=False)
            .agg(
                importance_mean=("cv_mean_abs_shap", "mean"),
                importance_std=("cv_mean_abs_shap", "std"),
                folds_used=("fold", "nunique"),
            )
            .sort_values("importance_mean", ascending=False)
        )

    perm_df = pd.DataFrame(perm_records)
    if not perm_df.empty:
        perm_df = (
            perm_df.groupby(["DroughtLevel", "Layer", "feature"], as_index=False)
            .agg(
                importance_mean=("cv_permutation_mse_increase_mean", "mean"),
                importance_std=("cv_permutation_mse_increase_mean", "std"),
                folds_used=("fold", "nunique"),
            )
            .sort_values("importance_mean", ascending=False)
        )

    pred_df = pd.DataFrame(
        {
            "DroughtLevel": level_name,
            "Layer": layer_name,
            "observed": yv,
            "predicted_cv": y_pred,
            "baseline_predicted_cv": baseline_pred,
        }
    )
    write_csv(pred_df, output_name_for("predictions", level_name, layer_name))
    return summary, shap_df, perm_df


def stability_importance(
    X: pd.DataFrame,
    y: pd.Series,
    level_name: str,
    layer_name: str,
) -> pd.DataFrame:
    valid = y.notna()
    Xv = X.loc[valid].reset_index(drop=True)
    yv = y.loc[valid].reset_index(drop=True)
    if len(Xv) < 5 or Xv.shape[1] < 1:
        return pd.DataFrame()

    rows = []
    label = level_layer_label(level_name, layer_name)
    with progress_bar(CFG.n_stability_seeds, f"{label} 随机种子稳定性", "建模") as bar:
        for seed in range(CFG.n_stability_seeds):
            model = make_rf(seed)
            model.fit(Xv, yv)
            # This is deliberately an auxiliary full-sample stability diagnostic.
            # CV-based SHAP / permutation outputs are the primary interpretable evidence.
            # Using RF impurity importance here avoids very slow repeated full-sample
            # permutation runs across 8 level-layer combinations.
            values = model.feature_importances_
            values_by_feature = pd.Series(values, index=Xv.columns)
            ranks = values_by_feature.rank(ascending=False, method="average")
            for feature in Xv.columns:
                rows.append(
                    {
                        "DroughtLevel": level_name,
                        "Layer": layer_name,
                        "seed": seed,
                        "feature": feature,
                        "importance": float(values_by_feature[feature]),
                        "rank": float(ranks[feature]),
                    }
                )
            bar.update(1)
    out = pd.DataFrame(rows)
    return (
        out.groupby(["DroughtLevel", "Layer", "feature"], as_index=False)
        .agg(importance_mean=("importance", "mean"), importance_std=("importance", "std"), rank_mean=("rank", "mean"), rank_std=("rank", "std"))
        .sort_values(["rank_mean", "rank_std"])
    )


def plot_importance(df: pd.DataFrame, level_name: str, layer_name: str, source: str) -> None:
    if df.empty:
        return
    d = df.head(12).sort_values("importance_mean", ascending=True)
    width = float(PLOT_PARAMS["importance_fig_width"])
    height = max(
        float(PLOT_PARAMS["importance_base_height"]),
        float(PLOT_PARAMS["importance_height_per_feature"]) * len(d) + 1.5,
    )
    fig, ax = plt.subplots(figsize=(width, height))
    ax.barh(
        d["feature"],
        d["importance_mean"],
        xerr=d["importance_std"],
        color=str(PLOT_PARAMS["importance_bar_color"]),
        alpha=float(PLOT_PARAMS["importance_bar_alpha"]),
        error_kw={"ecolor": str(PLOT_PARAMS["importance_errorbar_color"])},
    )
    ax.set_xlabel(source, fontsize=int(PLOT_PARAMS["axis_label_fontsize"]))
    ax.set_title(
        f"{DROUGHT_LEVELS_CN.get(level_name, level_name)} - {LAYER_CN.get(layer_name, layer_name)} 变量重要性",
        fontsize=int(PLOT_PARAMS["title_fontsize"]),
    )
    ax.tick_params(axis="both", labelsize=int(PLOT_PARAMS["tick_label_fontsize"]))
    fig.tight_layout()
    output_kind = "permutation_plot" if ("permutation" in source or "置换" in source) else "importance_plot"
    with progress_bar(1, f"{level_layer_label(level_name, layer_name)} 变量重要性图", "绘图") as bar:
        fig.savefig(
            CFG.output_dir / output_name_for(output_kind, level_name, layer_name, suffix="png"),
            dpi=int(PLOT_PARAMS["importance_dpi"]),
        )
        bar.update(1)
    plt.close(fig)


def run_ml_for_one(df: pd.DataFrame, level_name: str, layer_name: str, groups: pd.Series | None):
    X, y, missing_report, high_corr = prepare_features(df, level_name, layer_name)
    write_csv(missing_report, output_name_for("missing", level_name, layer_name))
    write_csv(high_corr, output_name_for("high_corr", level_name, layer_name))

    summary, shap_imp, perm_imp = cv_predict_and_importance(X, y, level_name, layer_name, groups)
    stability = stability_importance(X, y, level_name, layer_name)

    write_csv(summary, output_name_for("model_summary", level_name, layer_name))
    write_csv(shap_imp, output_name_for("importance", level_name, layer_name))
    write_csv(perm_imp, output_name_for("permutation", level_name, layer_name))
    write_csv(stability, output_name_for("stability", level_name, layer_name))
    plot_importance(shap_imp, level_name, layer_name, "交叉验证重要性")
    plot_importance(perm_imp, level_name, layer_name, "交叉验证置换重要性")
    return summary, shap_imp, perm_imp, stability


# =============================================================================
# 6. 主流程
# =============================================================================


def build_path_audit() -> pd.DataFrame:
    paths = {
        "python_env_dir": CFG.python_env_dir,
        "python_interpreter": CFG.python_interpreter,
        "hourly_temperature_csv": CFG.hourly_temperature_csv,
        "spi_daily_wide_xlsx": CFG.spi_daily_wide_xlsx,
        "static_site_csv": CFG.static_site_csv,
        "lai_8day_csv": CFG.lai_8day_csv,
        "fapar_8day_csv": CFG.fapar_8day_csv,
        "micro_soil_daily_csv": CFG.micro_soil_daily_csv,
        "output_dir": CFG.output_dir,
    }
    return pd.DataFrame(
        [{"name": name, "path": str(path), "exists": Path(path).exists()} for name, path in paths.items()]
    )


def build_parameter_table() -> pd.DataFrame:
    rows = []
    for key, value in CFG.__dict__.items():
        rows.append({"parameter": key, "value": str(value)})
    rows.append({"parameter": "FEATURES_TO_DROP", "value": ", ".join(FEATURES_TO_DROP)})
    rows.append({"parameter": "HAS_SHAP", "value": str(HAS_SHAP)})
    rows.append({"parameter": "HAS_DOCX", "value": str(HAS_DOCX)})
    rows.append({"parameter": "RUNTIME_CACHE_DIR", "value": str(RUNTIME_CACHE_DIR)})
    return pd.DataFrame(rows)


def add_doc_table(document, headers: list[str], rows: list[list[str]]) -> None:
    """Add a simple Word table. Values are converted to strings to avoid docx type errors."""

    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for idx, header in enumerate(headers):
        table.rows[0].cells[idx].text = str(header)
    for row_values in rows:
        cells = table.add_row().cells
        for idx, value in enumerate(row_values):
            cells[idx].text = str(value)


def add_doc_paragraphs(document, paragraphs: list[str]) -> None:
    """Add several explanatory paragraphs to the Word method document."""

    for text in paragraphs:
        document.add_paragraph(text)


def add_doc_bullets(document, bullets: list[str]) -> None:
    """Add bullet points to the Word method document."""

    for text in bullets:
        document.add_paragraph(text, style="List Bullet")


def add_workflow_step(
    document,
    step_no: str,
    title: str,
    purpose: str,
    inputs: str,
    actions: str,
    formula_or_method: str,
    parameters: str,
    outputs: str,
    checks: str,
    troubleshooting: str,
    next_step: str,
) -> None:
    """Add one hand-holding workflow step to the Word document."""

    document.add_heading(f"步骤 {step_no}：{title}", level=2)
    add_doc_table(
        document,
        ["说明项", "具体内容"],
        [
            ["这一步为什么做", purpose],
            ["输入是什么", inputs],
            ["具体怎么做", actions],
            ["公式或方法", formula_or_method],
            ["关键参数", parameters],
            ["输出结果", outputs],
            ["检查点：没问题看什么", checks],
            ["如果出问题怎么办", troubleshooting],
            ["没问题后进入哪一步", next_step],
        ],
    )


def build_method_document() -> None:
    """Generate a detailed Word method note explaining the full modeling design.

    The top script comment is intentionally concise. This document is the detailed
    version for advisor review, thesis-method writing, and later reproducibility.
    Formulas are written as plain mathematical text so they remain readable in Word.
    """

    if not HAS_DOCX or Document is None:
        write_text(
            "未生成 Word 方法说明书：当前 Python 环境无法导入 python-docx。"
            "请确认解释器 D:/ProgramData/anaconda3/envs/gee 中已安装 python-docx。",
            "00_机器学习解释建模方法说明书_未生成原因.txt",
        )
        return

    doc = Document()
    doc.add_heading("森林微气候缓冲能力空间差异机器学习解释建模方法说明书", level=0)
    doc.add_paragraph(f"推荐解释器：{CFG.python_interpreter}")
    doc.add_paragraph(f"结果输出目录：{CFG.output_dir}")
    doc.add_paragraph(
        "本文档由脚本自动生成，目标是让研究者在向导师汇报时，能够完整讲清楚脚本为什么这样设计、"
        "用了什么数据、每个数据起什么作用、每个关键步骤如何计算、如何筛选样本、如何设计机器学习模型、"
        "模型按什么顺序运行、结果应如何判断，以及哪些结论不能过度解释。"
    )
    doc.add_paragraph(
        "文档采用“文字解释为主、表格归纳为辅”的结构。表格用于快速查阅变量、路径、参数和输出；"
        "正文段落用于解释方法学理由和每一步的生态学含义。"
    )

    doc.add_heading("1. 为什么做这个脚本", level=1)
    add_doc_paragraphs(
        doc,
        [
            "本脚本服务于论文 Why 部分。What 部分已经回答了“干旱是否改变森林微气候缓冲能力”，"
            "包括干旱状态识别、CBI 和 DeltaCBI 计算、配对检验、等级差异检验以及敏感性分析。"
            "因此 Why 部分不再重复证明干旱是否产生影响，而是进一步解释影响大小为什么在不同站点之间不同。",
            "更具体地说，本脚本要回答的问题是：在 Mild、Moderate、Severe、Extreme 四个干旱等级下，"
            "哪些环境因子能够解释不同站点或不同站点-月份的 DeltaCBI 差异？如果某个站点在干旱期 DeltaCBI 明显增大，"
            "是因为干旱更强、持续更久，还是因为地形、冠层、植被状态、土壤水分或土壤温度不同？",
            "这里的响应变量是 DeltaCBI，而不是原始温度，也不是 SPI。DeltaCBI 表示同一站点、同一月份下，"
            "干旱状态相对于 Normal 状态的 CBI 改变量。因此它已经尽可能控制了站点本身和季节月份的一部分背景差异，"
            "更适合作为 Why 部分的解释对象。",
        ],
    )

    doc.add_heading("2. 方法定位：为什么是探索性解释分析", level=1)
    add_doc_paragraphs(
        doc,
        [
            "本研究的样本规模是方法选择的核心约束。不同干旱等级下的有效站点数很小，约为 n=13 到 n=27。"
            "在这种极端小样本条件下，如果直接使用复杂的线性混合模型，随机效应方差估计可能不稳定；"
            "如果依赖 VIF 或逐步回归筛选变量，变量保留结果也可能对个别样本非常敏感。",
            "因此脚本采用 Random Forest 作为探索性解释工具，而不是把它作为强因果推断工具。"
            "随机森林可以处理非线性关系和变量交互，对变量尺度不敏感，超参数相对少；与 XGBoost 等 boosting 方法相比，"
            "默认随机森林在小样本下对调参依赖更低，作为第一版 Why 解释框架更稳妥。",
            "需要强调的是，随机森林在这里不是为了证明某个变量“决定”DeltaCBI，而是用于提示变量重要性排序和可能机制方向。"
            "论文表述应使用“可能解释”“提示”“探索性结果”“需要谨慎解读”等限定语，不能写成确定性因果结论。",
        ],
    )

    doc.add_heading("3. 核心响应变量和公式", level=1)
    add_doc_paragraphs(
        doc,
        [
            "脚本的核心响应变量是 DeltaCBI。CBI 的含义是林下温度对宏气候温度变化的敏感程度。"
            "如果 CBI 较大，说明林下温度更容易随宏气候温度波动；如果 CBI 较小，说明森林内部对外界气温波动有更强的缓冲表现。",
            "CBI 不是简单温差，而是逐小时回归斜率。这样做的好处是能够利用小时尺度波动信息，"
            "衡量林下温度对宏气候温度变化的响应强度。脚本对 Normal 和各干旱等级分别估计 CBI，再计算二者差值。",
        ],
    )
    add_doc_table(
        doc,
        ["对象", "公式或定义", "解释"],
        [
            [
                "CBI",
                "Observed_T15cm_C = intercept + CBI x ERA5_T2m_C",
                "CBI 是逐小时林下温度对宏气候温度的一元线性回归斜率。",
            ],
            [
                "DeltaCBI",
                "DeltaCBI = Target_CBI - Normal_CBI",
                "Target_CBI 为某干旱等级下的 CBI；Normal_CBI 为同站点、同月份 Normal 条件下的 CBI。",
            ],
            [
                "SPI_intensity",
                "SPI_intensity = - min(SPI30d)",
                "数值越大表示该站点或站点-月经历的最强干旱越强。",
            ],
            [
                "站点级加权均值",
                "weighted mean = sum(value_i x DurationDays_i) / sum(DurationDays_i)",
                "站点级 DeltaCBI 和动态变量按干旱持续天数加权，避免短事件和长事件等权。",
            ],
            [
                "基线 RMSE",
                "Baseline prediction in each fold = mean(y_train)",
                "每个交叉验证折只用训练集均值预测验证集，避免验证集信息泄漏。",
            ],
        ],
    )
    add_doc_paragraphs(
        doc,
        [
            "DeltaCBI 的解释方向必须明确。DeltaCBI > 0 表示干旱期 CBI 高于同站点同月份 Normal 条件下的 CBI，"
            "即林下温度在干旱期更随宏气候温度波动，通常解释为缓冲能力表观减弱。"
            "DeltaCBI < 0 表示干旱期 CBI 低于 Normal 条件下的 CBI，通常解释为缓冲能力表观增强或维持。",
            "由于 DeltaCBI 是 Target_CBI 与 Normal_CBI 的差值，脚本要求 Target 和 Normal 都有足够小时数，"
            "否则该站点-月-等级的 DeltaCBI 不进入机器学习。这样可以避免用不可靠的 CBI 差值训练模型。",
        ],
    )

    doc.add_heading("4. 手把手建模流程：从第一步到最后一步", level=1)
    doc.add_paragraph(
        "本节是给导师或初学者讲脚本时最重要的流程说明。它不是方法摘要，而是按照脚本真实运行顺序，"
        "逐步说明每一步为什么做、用什么输入、怎么计算、会输出什么、如何判断是否正常，以及异常时应该怎么处理。"
    )
    add_workflow_step(
        doc,
        "0",
        "初始化配置、输出目录和运行审计",
        "在正式计算前固定所有路径、阈值、模型参数、绘图参数和进度条参数，保证本次运行可复现、可追溯。",
        f"脚本顶部 Config 配置；输出目录 {CFG.output_dir}；推荐解释器 {CFG.python_interpreter}。",
        "创建输出目录；创建本次运行临时缓存目录；设置 matplotlib 中文字体；固定 numpy 随机种子；输出路径审计表、参数配置表、绘图与进度条参数说明表，并生成本 Word 说明书。",
        "这一阶段不做统计计算，属于运行环境和参数审计。核心思想是先记录“用什么配置运行”，再开始产生分析结果。",
        f"random_seed={CFG.random_seed}; output_dir={CFG.output_dir}; RUNTIME_CACHE_DIR=本次运行临时缓存目录。",
        f"{OUTPUT_NAMES['path_audit']}；{OUTPUT_NAMES['parameter_table']}；{OUTPUT_NAMES['plot_progress_parameter_table']}；{OUTPUT_NAMES['method_docx']}。",
        "检查路径审计表中核心输入文件是否存在；检查参数配置表中的解释器路径、阈值、run_levels、run_layers 是否符合本次分析目的。",
        "如果核心输入路径 exists=False，先检查文件是否移动、文件名是否变化、路径中的中文是否写错。如果 Word 未生成，检查 python-docx 是否能在 gee 环境中导入。",
        "路径和参数无误后，进入步骤 1 读取逐小时温度和 SPI 数据。",
    )
    add_workflow_step(
        doc,
        "1",
        "读取逐小时温度数据并清理有效小时",
        "CBI 必须由逐小时宏气温和林下温度回归得到，所以第一类核心输入是逐小时温度对齐表。",
        f"{CFG.hourly_temperature_csv}，关键字段为 Site_ID、Time_UTC、ERA5_T2m_C、Observed_T15cm_C、Has_Both_Data。",
        "读取 CSV；统一 Site_ID 格式；把 Time_UTC 转为日期时间；生成 UTC_Date 和 YearMonth；如果存在 Has_Both_Data，则只保留宏气温和林下温度同时存在的小时；继续删除时间或温度缺失的小时。",
        "时间字段处理：UTC_Date = floor(Time_UTC, day)；YearMonth = Time_UTC 的年月。有效小时条件是 Time_UTC、ERA5_T2m_C、Observed_T15cm_C 均非缺失。",
        "site_col、time_col、macro_temp_col、micro_temp_col、has_both_col。如果上游表字段名改变，需要优先改 Config 中这些字段。",
        "内存中的 hourly 表，包含清理后的逐小时温度和站点、日期、月份字段。",
        "检查小时数据读取后是否为空；检查 Site_ID 是否正常；检查 UTC_Date 和 YearMonth 是否生成；检查温度列是否为数值。",
        "如果报字段不存在，说明上游 CSV 列名和 Config 不一致；如果清理后数据为空，检查 Has_Both_Data 是否全 False 或温度列是否被读成异常文本。",
        "小时数据正常后，进入步骤 2 读取 SPI 并划分干旱等级。",
    )
    add_workflow_step(
        doc,
        "2",
        "读取逐日 SPI30d 并给每个小时标记干旱等级",
        "脚本需要知道每个站点每一天属于 Normal、Mild、Moderate、Severe 还是 Extreme，才能分别计算 Normal_CBI 和 Target_CBI。",
        f"{CFG.spi_daily_wide_xlsx}，第一列为日期，其余列为各站点 SPI30d。",
        "读取 Excel 宽表；把各站点列 melt 成长表；统一 Site_ID；把日期列转为 UTC_Date；按 Site_ID + UTC_Date 合并到小时数据；根据 SPI30d 阈值生成 DroughtLevel 和 DroughtLevel_CN。",
        f"Normal: {CFG.normal_spi_low} < SPI30d < {CFG.normal_spi_high}; Mild: -1.0 < SPI30d <= -0.5; Moderate: -1.5 < SPI30d <= -1.0; Severe: {CFG.extreme_spi_threshold} < SPI30d <= -1.5; Extreme: SPI30d <= {CFG.extreme_spi_threshold}。",
        "normal_spi_low、normal_spi_high、extreme_spi_threshold，以及 classify_drought_level 函数中的 Mild/Moderate/Severe 边界。",
        "带 SPI30d、DroughtLevel、DroughtLevel_CN、Site_Month 的 hourly 表。",
        "检查 SPI30d 是否成功合并；检查 DroughtLevel 中是否有 Mild、Moderate、Severe、Extreme；检查 Other 或缺失比例是否异常。",
        "如果大量 SPI30d 缺失，检查 SPI 表站点列名是否和温度表 Site_ID 对不上，或日期时区/日期格式是否不一致。如果某等级完全没有数据，先确认研究区和年份是否确实没有该等级。",
        "干旱等级标记正常后，进入步骤 3 构建四等级站点-月 DeltaCBI 基础表。",
    )
    add_workflow_step(
        doc,
        "3",
        "构建 Site_ID × YearMonth × DroughtLevel 的 DeltaCBI 基础表",
        "这是整个脚本的核心响应变量构建步骤。后续机器学习不直接使用原始温度，而是解释这里计算出来的 DeltaCBI。",
        "步骤 2 得到的 hourly 表，其中必须包含 Site_ID、YearMonth、DroughtLevel、ERA5_T2m_C、Observed_T15cm_C、SPI30d。",
        "按 Site_ID 和 YearMonth 分组；在每个站点-月内部先取 Normal 小时计算 Normal_CBI；再分别取 Mild、Moderate、Severe、Extreme 小时计算 Target_CBI；每个等级生成一行；计算 DeltaCBI、小时数、R2、MacroSD、MinDailySPI、DurationDays、事件开始结束日期和事件片段数。",
        "CBI 回归公式：Observed_T15cm_C = intercept + CBI x ERA5_T2m_C。DeltaCBI = Target_CBI - Normal_CBI。DurationDays = 目标等级实际出现的唯一日期数。SPI_intensity = -MinDailySPI。",
        f"min_status_hours={CFG.min_status_hours}; min_macro_sd={CFG.min_macro_sd}; use_macro_sd_for_pair_flag={CFG.use_macro_sd_for_pair_flag}。",
        f"{OUTPUT_NAMES['site_month_delta']}；{OUTPUT_NAMES['site_month_delta_audit']}。",
        "检查基础表是否包含四个干旱等级；检查 Pair_flag=ok 的记录数量；检查 Target_n_hours 和 Normal_n_hours 是否大多达到阈值；检查 DeltaCBI 是否不是全缺失。",
        "如果 Pair_flag 大量为 too_few_hours，说明该等级或 Normal 小时不足，可检查干旱等级窗口是否太少，或考虑是否需要调整 min_status_hours。如果 cbi_failed 多，检查温度列数值质量或宏气温是否几乎不变。",
        "基础 DeltaCBI 表正常后，进入步骤 4 构建动态和静态解释变量。",
    )
    add_workflow_step(
        doc,
        "4",
        "读取 LAI、FAPAR、土壤水热和站点静态属性",
        "DeltaCBI 是响应变量，接下来要准备解释变量。解释变量必须与站点和事件窗口对齐。",
        f"LAI: {CFG.lai_8day_csv}; FAPAR: {CFG.fapar_8day_csv}; 土壤: {CFG.micro_soil_daily_csv}; 静态属性: {CFG.static_site_csv}。",
        "读取 LAI 和 FAPAR 8日尺度宽表并转成长表；把每个 8日产品日期展开成 8 个逐日值；读取土壤日尺度表并把 VWC_Daily 改名为 soil_moisture、T-5cm_Daily 改名为 soil_temperature；读取站点静态属性并统一字段名。",
        "8日产品展开方法：产品日期 d 的值复制到 d、d+1、...、d+7。坡向转换：aspect_sin = sin(radians(Aspect)); aspect_cos = cos(radians(Aspect))。",
        "LAI/FAPAR 的 date_col 默认 datetime；土壤字段 VWC_Daily 和 T-5cm_Daily；静态属性重命名 Elevation/DEM、Slope、Aspect、Canopy_Height。",
        "内存中的 lai_daily、fapar_daily、soil_daily、static_attributes。",
        "检查 LAI、FAPAR、soil_daily 是否为空；检查 Date 是否成功解析；检查静态属性是否包含 Site_ID 和地形/冠层字段。",
        "如果 LAI/FAPAR 文件不存在，相关变量会后续全缺失并被剔除。如果土壤字段名变化，需要在 load_daily_soil 中补充重命名规则。如果 Aspect 缺失，则不会生成 aspect_sin/cos。",
        "解释变量源数据读取正常后，进入步骤 5 按目标干旱窗口聚合动态变量。",
    )
    add_workflow_step(
        doc,
        "5",
        "按每条 DeltaCBI 的目标干旱日期窗口聚合动态解释变量",
        "动态变量必须对应干旱发生时的实际窗口，不能用全年均值或站点长期均值，否则解释变量和响应变量时间口径不一致。",
        "步骤 3 的 site_month_delta；步骤 2 的 hourly；步骤 4 的 lai_daily、fapar_daily、soil_daily。",
        "从 hourly 中按 Site_ID、YearMonth、DroughtLevel 提取目标干旱日期列表 Target_Dates；把 Target_Dates 合并到 DeltaCBI 基础表；对每一行，在该站点和这些目标日期内分别计算 LAI、FAPAR、soil_moisture、soil_temperature 的均值；再合并站点静态属性。",
        "动态变量窗口均值：mean(value on Target_Dates for one Site_ID)。这里只在目标干旱日期内取均值，不混入 Normal 日期或其他等级日期。",
        "目标窗口由 DroughtLevel in Mild/Moderate/Severe/Extreme 的小时数据生成；aggregate_daily_window 负责按站点和日期筛选。",
        "带解释变量的站点-月 DeltaCBI 表；后续会写出为基础表正式文件。",
        "检查 LAI、FAPAR、soil_moisture、soil_temperature 是否存在合理非缺失值；检查静态变量是否按 Site_ID 合并成功。",
        "如果动态变量全缺失，检查日期是否对齐、站点 ID 是否一致、8日产品日期列是否正确。如果只有部分缺失，后续缺失审计会处理，不要手动删除样本。",
        "协变量合并正常后，进入步骤 6 输出基础表和样本量统计。",
    )
    add_workflow_step(
        doc,
        "6",
        "输出基础表、审计表和有效样本量统计",
        "在建模前必须先确认响应变量和解释变量基础表是否合理，尤其要确认各干旱等级还有多少有效样本。",
        "步骤 5 得到的带协变量 site_month_delta，以及步骤 3 的 delta_audit。",
        "保存四等级站点-月缓冲变化基础表；保存基础表审计表；筛选 Pair_flag=ok 的记录，按 DroughtLevel 统计有效站点数 n_sites 和有效站点-月记录数 n_site_months。",
        "有效样本统计：n_sites = nunique(Site_ID); n_site_months = count(YearMonth rows) within Pair_flag=ok。",
        "Pair_flag=ok 是进入机器学习的硬条件。",
        f"{OUTPUT_NAMES['site_month_delta']}；{OUTPUT_NAMES['site_month_delta_audit']}；{OUTPUT_NAMES['sample_counts']}。",
        "检查 Mild、Moderate、Severe、Extreme 是否都有有效样本；重点看 Extreme 的 n_sites 是否过低；检查基础表是否有合理的 DeltaCBI 分布。",
        "如果某等级 n_sites 太低，后续模型仍可能跑，但只能作为弱探索结果。如果某等级没有有效样本，需要回到步骤 3 检查小时数阈值、SPI 等级或输入数据覆盖。",
        "样本量可以接受后，进入步骤 7 构建站点级和站点-月级机器学习表。",
    )
    add_workflow_step(
        doc,
        "7",
        "构建两个机器学习输入表",
        "Why 分析需要同时解释长期空间差异和事件窗口动态调节，所以脚本构建站点级和站点-月级两个层级。",
        "步骤 6 的 site_month_delta，其中只使用 Pair_flag=ok 的记录。",
        "站点-月级表直接保留 Site_ID x YearMonth x DroughtLevel 记录，并把 DurationDays 改为 Duration_days，把 -MinDailySPI 作为 SPI_intensity。站点级表按 Site_ID x DroughtLevel 聚合：DeltaCBI 和动态变量按 DurationDays 加权平均，Duration_days 累计，SPI_intensity 取最强干旱。",
        "站点级加权均值：sum(value_i x DurationDays_i) / sum(DurationDays_i)。站点级 SPI_intensity = - min(MinDailySPI)。",
        "DurationDays 是权重；DroughtLevel 分组；Site_ID 分组。",
        f"{OUTPUT_NAMES['site_level_ml_table']}；{OUTPUT_NAMES['site_month_ml_table']}。",
        "检查站点级表每个等级的行数是否等于该等级有效站点数；检查站点-月级表行数是否等于有效站点-月记录数；检查 Duration_days、SPI_intensity 是否生成。",
        "如果站点级表为空，说明 Pair_flag=ok 记录为空。如果某些变量缺失，不要在这一步删除，后续步骤 8 会按列审计和处理。",
        "两个建模输入表正常后，进入步骤 8 针对每个等级和层级做特征预处理。",
    )
    add_workflow_step(
        doc,
        "8",
        "每个等级和层级单独做特征预处理、缺失审计和相关审计",
        "不同干旱等级和不同分析层级的样本量、缺失情况、变量相关结构不同，必须分别预处理，不能用全体数据统一筛选。",
        "某一个 DroughtLevel 和 layer 对应的建模数据表；候选特征列表 BASE_FEATURE_COLS；人工剔除列表 FEATURES_TO_DROP。",
        "如果有 soil_type，则先 one-hot；再从候选特征中删除 FEATURES_TO_DROP；逐列计算缺失率、非缺失数量和唯一值数量；剔除高缺失、有效值太少或无变化变量；剩余变量用中位数填补；计算 Spearman 相关矩阵并输出高相关变量对。",
        f"缺失率阈值：missing_drop_threshold={CFG.missing_drop_threshold}; 高相关阈值：high_corr_threshold={CFG.high_corr_threshold}; 中位数填补：X[col].fillna(median)。",
        "FEATURES_TO_DROP、missing_drop_threshold、high_corr_threshold。",
        "每个等级-层级的变量缺失率与填补审计表、高相关变量对审计表、斯皮尔曼候选变量相关矩阵。",
        "检查是否有足够特征保留下来；检查 nighttime_light、built_up_distance 等占位变量是否因全缺失被剔除；检查高相关变量对是否需要人工处理。",
        "如果保留特征为 0，模型不会运行，需要补充数据或放宽不合理的剔除设置。如果发现高相关变量对，正式解释前应在 FEATURES_TO_DROP 手动二选一后重跑。",
        "特征预处理正常后，进入步骤 9 交叉验证随机森林建模。",
    )
    add_workflow_step(
        doc,
        "9",
        "交叉验证训练随机森林并生成验证集预测",
        "这一步检验模型是否能在未参与训练的数据上预测 DeltaCBI，避免只看全样本拟合造成过度乐观。",
        "步骤 8 得到的 X 和 y；站点-月级还需要 groups=Site_ID。",
        "根据层级和等级选择交叉验证切分；每个 fold 用训练集训练随机森林；对验证集预测 DeltaCBI；同时用训练集 y 均值作为该 fold 的基线预测；记录所有验证集预测结果。",
        "模型：RandomForestRegressor。基线预测：baseline_pred(test fold) = mean(y_train)。CV-RMSE = sqrt(mean_squared_error(y, y_pred))。Baseline_RMSE_cv = sqrt(mean_squared_error(y, baseline_pred))。",
        f"n_estimators={CFG.n_estimators}; max_depth={CFG.max_depth}; min_samples_leaf={CFG.min_samples_leaf}; random_seed={CFG.random_seed}; CV_STRATEGY_BY_LEVEL={CV_STRATEGY_BY_LEVEL}。",
        "每个等级-层级的随机森林交叉验证模型表现表和交叉验证预测结果表。",
        "检查 n_samples、n_features、CV_method；检查 RMSE_cv 是否小于 Baseline_RMSE_cv；检查 RMSE_improvement_vs_baseline 是否为正。",
        "如果 n<5 或没有特征，模型标记 not_run。如果 RMSE 没有优于基线，不要强解释变量重要性，应回看变量质量、样本量和缺失审计。",
        "交叉验证预测正常后，进入步骤 10 计算变量重要性。",
    )
    add_workflow_step(
        doc,
        "10",
        "计算交叉验证重要性、可选置换重要性和全样本稳定性",
        "模型表现只能说明预测是否有用，变量重要性用于回答哪些环境因子可能解释 DeltaCBI 差异。",
        "步骤 9 中每个 fold 训练得到的模型；步骤 8 的 X；响应变量 y。",
        "默认情况下，若 SHAP 关闭或不可用，脚本记录每个 fold 模型的随机森林 impurity importance 作为交叉验证重要性 fallback。若 enable_shap=True 且 shap 可用，则计算每个 fold 验证集 SHAP 值。若 enable_cv_permutation=True 且验证集样本数足够，则计算验证集置换重要性。随后用多个随机种子在全样本上重复训练，记录重要性排名稳定性。",
        "SHAP 重要性：mean(abs(SHAP value))。置换重要性：打乱某变量后验证误差增加量。稳定性：不同 seed 下 importance 和 rank 的均值、标准差。",
        f"enable_shap={CFG.enable_shap}; enable_cv_permutation={CFG.enable_cv_permutation}; n_permutation_repeats={CFG.n_permutation_repeats}; n_stability_seeds={CFG.n_stability_seeds}。",
        "交叉验证重要性排序表、交叉验证置换重要性排序表、全样本重要性稳定性表、变量重要性图。",
        "检查排名前几位变量是否合理；检查 importance_std 或 rank_std 是否过大；检查高相关变量是否分摊重要性。",
        "如果 SHAP 或置换重要性为空，先确认开关是否为 True、shap 是否可用、验证集样本数是否足够。默认空置换重要性不是错误，而是因为 enable_cv_permutation=False。",
        "重要性结果生成后，进入步骤 11 汇总跨等级跨层级结果。",
    )
    add_workflow_step(
        doc,
        "11",
        "汇总跨等级、跨层级模型结果",
        "单个模型只能说明一个干旱等级和一个层级，论文需要横向比较不同等级和层级下的结果是否一致。",
        "所有等级-层级组合产生的 summary、importance、permutation、stability 表。",
        "把每个组合的模型表现拼接成跨等级汇总表；把各组合重要性结果拼接成跨等级重要性汇总表；把置换重要性和稳定性结果分别汇总。",
        "表格拼接，不再重新训练模型。核心分组字段为 DroughtLevel、Layer、feature。",
        "run_levels 和 run_layers 控制哪些组合进入汇总。",
        f"{OUTPUT_NAMES['cross_summary']}；{OUTPUT_NAMES['cross_importance']}；{OUTPUT_NAMES['cross_permutation']}；{OUTPUT_NAMES['cross_stability']}。",
        "检查每个预期等级和层级是否都出现在汇总表；检查没有运行的组合是否因为 n<5 或无特征被标记。",
        "如果汇总表缺少某个组合，检查 run_levels/run_layers、该等级样本量、特征预处理结果和模型表现表中的 note。",
        "汇总正常后，进入步骤 12 写运行摘要并清理缓存。",
    )
    add_workflow_step(
        doc,
        "12",
        "写运行摘要并清理本次临时缓存",
        "最后一步把关键方法设置写成摘要，并删除本次运行产生的临时缓存，保持结果目录干净。",
        "前面所有输出结果；本次运行临时缓存目录 RUNTIME_CACHE_DIR。",
        "写入运行摘要说明；在 finally 中调用 cleanup_runtime_cache 删除临时缓存目录；写入缓存清理记录。",
        "清理逻辑：如果 RUNTIME_CACHE_DIR 存在，则 shutil.rmtree 删除；无论主流程是否报错，finally 都会尝试清理。",
        "RUNTIME_CACHE_DIR；OUTPUT_NAMES['run_summary']；OUTPUT_NAMES['cache_cleanup']。",
        f"{OUTPUT_NAMES['run_summary']}；{OUTPUT_NAMES['cache_cleanup']}。",
        "检查运行摘要是否记录输出目录、核心基础表、Pair_flag 规则、模型定位和耗时；检查缓存清理记录中“是否已删除”为 True。",
        "如果缓存清理失败，查看清理记录中的错误信息，通常是文件被其他程序占用。关闭占用程序后可手动删除该缓存目录。",
        "流程结束。此时应按“样本量统计 -> 缺失审计 -> 高相关审计 -> 模型表现 -> 重要性排序 -> 图”的顺序解读结果。",
    )

    doc.add_heading("5. 脚本实际使用的数据", level=1)
    add_doc_paragraphs(
        doc,
        [
            "脚本实际使用的数据分为五类：逐小时温度数据、逐日 SPI30d 数据、站点静态属性、植被动态数据、"
            "土壤水热数据。另有人类活动变量路径作为占位，但如果文件不存在或变量全缺失，会在缺失值筛查阶段被自动剔除。",
            "逐小时温度数据和逐日 SPI30d 是构建 DeltaCBI 的核心数据。没有这两类数据，就无法重新构建四等级站点-月 DeltaCBI 基础表。"
            "站点静态属性、LAI、FAPAR、土壤水分、土壤温度则是解释变量，用于回答为什么 DeltaCBI 在空间上或事件之间不同。",
        ],
    )
    add_doc_table(
        doc,
        ["数据", "路径或来源", "关键字段", "作用"],
        [
            [
                "逐小时温度对齐表",
                str(CFG.hourly_temperature_csv),
                "Site_ID, Time_UTC, ERA5_T2m_C, Observed_T15cm_C, Has_Both_Data",
                "计算 Target_CBI 和 Normal_CBI，是 DeltaCBI 的直接来源。",
            ],
            [
                "逐日 SPI30d 宽表",
                str(CFG.spi_daily_wide_xlsx),
                "日期列，各站点 SPI30d 列",
                "按 Site_ID 和 UTC_Date 合并到小时数据，用于划分 Normal 和四个干旱等级。",
            ],
            [
                "站点静态属性表",
                str(CFG.static_site_csv),
                "Longitude, Latitude, Elevation/DEM, Slope, Aspect, Canopy_Height, soil_type",
                "提供地形、冠层高度和可选土壤类型；Aspect 转换为 sin/cos 后入模。",
            ],
            [
                "LAI 8日尺度表",
                str(CFG.lai_8day_csv),
                "datetime, 各站点 LAI",
                "展开为逐日值后，按目标干旱日期窗口取均值。",
            ],
            [
                "FAPAR 8日尺度表",
                str(CFG.fapar_8day_csv),
                "datetime, 各站点 FAPAR",
                "展开为逐日值后，按目标干旱日期窗口取均值。",
            ],
            [
                "微气候和土壤日尺度表",
                str(CFG.micro_soil_daily_csv),
                "Site_ID, Date, VWC_Daily, T-5cm_Daily",
                "VWC_Daily 作为 soil_moisture；T-5cm_Daily 作为 soil_temperature。",
            ],
            [
                "人类活动占位变量",
                f"{CFG.ntl_csv}; {CFG.built_up_distance_csv}",
                "nighttime_light, built_up_distance",
                "当前若文件不存在或全缺失，会按缺失率规则剔除，不影响模型继续运行。",
            ],
        ],
    )
    add_doc_paragraphs(
        doc,
        [
            "这里要特别说明为什么有些方案变量没有真实输入时脚本仍能跑出结果。脚本的候选特征列表允许变量缺失，"
            "但在正式建模前会按列检查缺失率。若某个变量全是 NaN 或缺失率超过阈值，它会被剔除，而不是导致整行样本被删除。"
            "因此当前结果代表“已有变量版本”的探索性解释模型，不代表方案里的全部变量都已经完整纳入。",
            "如果后续补齐 root_zone_soil_moisture、nighttime_light 或 built_up_distance 等数据，只需把真实路径和字段处理逻辑接入，"
            "这些变量就可以进入同一套缺失审计、相关审计和建模流程。"
        ],
    )

    doc.add_heading("6. 干旱等级如何划分", level=1)
    add_doc_paragraphs(
        doc,
        [
            "脚本使用 SPI30d 对每个站点、每一天划分干旱等级，并把逐日等级合并到逐小时温度数据中。"
            "SPI30d 代表 30 日尺度的水分异常状态，适合描述短期到月尺度干旱背景。"
            "所有等级边界与 What 部分保持一致，避免 Why 部分和前文统计检验使用不同干旱定义。",
        ],
    )
    add_doc_table(
        doc,
        ["等级", "SPI30d 条件", "中文标签"],
        [
            ["Normal", f"{CFG.normal_spi_low} < SPI30d < {CFG.normal_spi_high}", "正常"],
            ["Mild", "-1.0 < SPI30d <= -0.5", "轻度干旱"],
            ["Moderate", "-1.5 < SPI30d <= -1.0", "中度干旱"],
            ["Severe", f"{CFG.extreme_spi_threshold} < SPI30d <= -1.5", "重度干旱"],
            ["Extreme", f"SPI30d <= {CFG.extreme_spi_threshold}", "极端干旱"],
        ],
    )

    doc.add_heading("7. 四等级站点-月 DeltaCBI 基础表如何构建", level=1)
    add_doc_paragraphs(
        doc,
        [
            "脚本第一步不是直接读取已有 What 结果表，而是从逐小时温度数据和逐日 SPI30d 重新构建"
            " Site_ID x YearMonth x DroughtLevel 层面的基础表。这样设计的原因是基础表粒度最清楚，"
            "并且能够确保 DeltaCBI 的计算口径与本脚本后续协变量聚合口径一致。",
            "具体做法是：先把逐小时温度表按 Site_ID 和 UTC_Date 合并 SPI30d；再根据 SPI30d 给每个小时标记 Normal、"
            "Mild、Moderate、Severe、Extreme 或 Other；然后按 Site_ID 和 YearMonth 分组。"
            "在每个站点-月内部，脚本先提取 Normal 小时，计算 Normal_CBI；再分别提取四个干旱等级小时，计算 Target_CBI。",
            "每个 Site_ID x YearMonth x DroughtLevel 都会形成一行。行内既保留 Target_CBI、Normal_CBI、DeltaCBI，"
            "也保留小时数、回归 R2、宏气温标准差、最小 SPI、持续天数和事件片段数。这样后续不仅能建模，也能回头审计某条记录为什么有效或无效。",
        ],
    )
    add_doc_table(
        doc,
        ["基础表字段", "含义", "后续用途"],
        [
            ["Target_CBI", "该站点-月-干旱等级下小时回归斜率", "计算 DeltaCBI"],
            ["Normal_CBI", "同站点同月份 Normal 小时回归斜率", "作为月内基线"],
            ["DeltaCBI", "Target_CBI - Normal_CBI", "机器学习响应变量"],
            ["Target_n_hours / Normal_n_hours", "Target 和 Normal 的有效小时数", "判断 Pair_flag 是否 ok"],
            ["Target_R2 / Normal_R2", "小时回归拟合度", "审计 CBI 估计质量"],
            ["Target_Macro_SD / Normal_Macro_SD", "宏气温标准差", "审计温度波动是否足够"],
            ["MinDailySPI", "目标干旱窗口内最小 SPI", "构建 SPI_intensity"],
            ["DurationDays", "目标等级实际出现的日数", "站点级加权和暴露时长变量"],
            ["N_events_in_site_month", "同月同等级连续片段数", "审计多段干旱情况"],
            ["Pair_flag", "有效配对标记", "只有 ok 进入机器学习"],
        ],
    )

    doc.add_heading("8. 同月多次同等级干旱如何处理", level=1)
    add_doc_paragraphs(
        doc,
        [
            "一个关键边界情况是：同一站点同一月份内可能出现多段不连续但等级相同的干旱。"
            "如果简单把每段事件的 DeltaCBI 算出来再算术平均，短事件和长事件会被赋予相同权重，这不符合生态暴露逻辑。",
            "因此脚本在 Site_ID x YearMonth x DroughtLevel 层面直接集合该等级在该月内所有目标小时，用这些小时一次性估计 Target_CBI。"
            "这种做法相当于按小时数自然加权：持续更久的干旱片段对斜率估计贡献更大，持续很短的片段贡献更小。",
            "EventStartDate 和 EventEndDate 只是记录目标日期窗口的最早和最晚日期。若 N_events_in_site_month 大于 1，"
            "说明中间可能存在断裂，这两个日期之间并不代表每天都是同一个连续事件。这个字段用于审计，不作为连续事件长度解释。",
        ],
    )

    doc.add_heading("9. 有效样本筛选条件", level=1)
    add_doc_paragraphs(
        doc,
        [
            "机器学习只使用 Pair_flag 等于 ok 的记录。这样做的目的是确保响应变量 DeltaCBI 本身可信。"
            "如果 Target 或 Normal 小时数不足，回归斜率会受到少数小时强烈影响；如果宏气温几乎没有波动，"
            "斜率也可能缺乏解释意义。因此脚本保留小时数、MacroSD 和回归状态作为审计字段。",
            "当前默认筛选条件相对克制：Target 和 Normal 均至少 72 个有效小时，且 OLS 回归成功。"
            "MacroSD 默认不参与硬筛选，只做审计字段，这是为了避免在 Extreme 等小样本等级下进一步损失样本。"
            "如果后续导师要求更严格，可以把 use_macro_sd_for_pair_flag 改为 True。",
        ],
    )
    add_doc_table(
        doc,
        ["条件", "默认值", "作用"],
        [
            ["Target 小时数", f">= {CFG.min_status_hours}", "保证干旱等级下有足够小时数据估计 CBI。"],
            ["Normal 小时数", f">= {CFG.min_status_hours}", "保证同站点同月份 Normal 基线 CBI 可靠。"],
            ["Target_CBI / Normal_CBI", "均成功估计", "线性回归失败时不计算 DeltaCBI。"],
            ["MacroSD", f">= {CFG.min_macro_sd}", f"当前是否参与硬筛选：{CFG.use_macro_sd_for_pair_flag}。默认只做审计。"],
            ["Pair_flag", "ok", "只有 Pair_flag=ok 的记录进入机器学习表。"],
        ],
    )

    doc.add_heading("10. 解释变量如何构建", level=1)
    add_doc_paragraphs(
        doc,
        [
            "解释变量分为静态变量和动态变量。静态变量描述站点长期空间背景，例如海拔、坡度、坡向、冠层高度。"
            "动态变量描述某次干旱发生时的生态环境状态，例如 LAI、FAPAR、土壤水分和土壤温度。",
            "动态变量不能简单使用全年均值或站点长期均值，因为本脚本解释的是某个站点、某个月、某个干旱等级下的 DeltaCBI。"
            "因此 LAI、FAPAR、soil_moisture 和 soil_temperature 都按该行对应的目标干旱日期窗口聚合，保证解释变量与响应变量在时间窗口上对齐。",
            "坡向 Aspect 是环形变量，不能直接作为 0 到 360 的连续数值输入模型。脚本将其转换为 aspect_sin 和 aspect_cos。"
            "这样 0 度和 360 度在特征空间中接近，避免模型误以为它们相距很远。",
        ],
    )
    add_doc_table(
        doc,
        ["变量类别", "变量", "计算或处理方式"],
        [
            ["干旱暴露", "SPI_intensity", "站点-月级取 -MinDailySPI；站点级取该等级下最小 SPI 的相反数。"],
            ["干旱暴露", "Duration_days", "站点-月级等于 DurationDays；站点级为该等级累计干旱天数。"],
            ["地形", "elevation, slope", "从站点静态属性表直接合并。"],
            ["坡向", "aspect_sin, aspect_cos", "Aspect 角度转换为 sin/cos，避免 0度 和 360度 被误认为相距很远。"],
            ["冠层", "canopy_height", "从站点静态属性表直接合并。"],
            ["植被动态", "LAI, FAPAR", "8日尺度展开为逐日值，再按目标干旱日期窗口取均值。"],
            ["土壤动态", "soil_moisture, soil_temperature", "按目标干旱日期窗口取均值。"],
            ["类别变量", "soil_type", "若存在且非全缺失，做 one-hot 编码。"],
            ["人类活动", "nighttime_light, built_up_distance", "当前为占位变量，高缺失时自动剔除。"],
        ],
    )

    doc.add_heading("11. 为什么同时做站点级和站点-月级", level=1)
    add_doc_paragraphs(
        doc,
        [
            "脚本设计了两个分析层级：站点级和站点-月级。二者不是重复分析，而是回答不同层面的问题。",
            "站点级是主分析，粒度为 Site_ID x DroughtLevel。它把同一站点同一干旱等级下多个有效月份聚合起来，"
            "用于回答长期空间差异问题：哪些站点背景因素解释了缓冲能力变化更强或更弱。",
            "站点-月级是互补分析，粒度为 Site_ID x YearMonth x DroughtLevel。它保留具体月份和事件窗口信息，"
            "用于回答动态调节问题：当某次干旱发生时，植被状态和土壤水热条件是否影响 DeltaCBI。",
            "如果两个层级的重要变量排序一致，说明结果更稳健；如果不一致，也不是错误，而可能说明长期空间格局由地形、冠层等静态因子塑造，"
            "具体事件强度则更多受土壤水分、土壤温度或植被状态调节。",
        ],
    )
    add_doc_table(
        doc,
        ["层级", "数据粒度", "定位", "主要聚合规则"],
        [
            [
                "站点级",
                "Site_ID x DroughtLevel",
                "主分析，解释长期空间差异。",
                "DeltaCBI 和动态变量按 DurationDays 加权；Duration_days 累计；SPI_intensity 取最强干旱。",
            ],
            [
                "站点-月级",
                "Site_ID x YearMonth x DroughtLevel",
                "互补分析，解释具体干旱发生时的动态调节。",
                "保留每个站点-月-等级记录，动态变量对应该次目标日期窗口。",
            ],
        ],
    )

    doc.add_heading("12. 站点级聚合规则", level=1)
    add_doc_paragraphs(
        doc,
        [
            "站点级表由站点-月级有效记录聚合得到。聚合时不能简单算术平均，因为不同月份同一等级干旱持续时间可能不同。"
            "持续时间越长，理论上对站点该等级暴露特征和缓冲变化的代表性越强，因此 DeltaCBI 和动态变量使用 DurationDays 加权平均。",
            "干旱持续时间 Duration_days 在站点级不是平均值，而是累计值，表示该站点在该等级下累计暴露了多少天。"
            "SPI_intensity 则取最小 SPI 的相反数，而不是加权平均，因为强度变量更关注该站点经历过的最严重干旱程度。",
        ],
    )
    add_doc_table(
        doc,
        ["站点级变量", "聚合方式", "理由"],
        [
            ["DeltaCBI", "按 DurationDays 加权平均", "长事件对站点等级响应更有代表性。"],
            ["Duration_days", "该等级下 DurationDays 累计和", "表示暴露总时长。"],
            ["SPI_intensity", "- min(MinDailySPI)", "表示最强干旱强度，避免均值稀释极端暴露。"],
            ["LAI / FAPAR", "按 DurationDays 加权平均", "代表该等级暴露期间的平均植被状态。"],
            ["soil_moisture / soil_temperature", "按 DurationDays 加权平均", "代表该等级暴露期间的平均土壤水热状态。"],
            ["elevation / slope / aspect_sin / aspect_cos / canopy_height", "直接沿用站点值", "静态空间属性不随月份变化。"],
        ],
    )

    doc.add_heading("13. 缺失值如何处理", level=1)
    add_doc_paragraphs(
        doc,
        [
            "缺失值处理是本脚本的重要方法学改动。早期脚本如果整行删除，会在 Extreme 等级中损失大量样本，"
            "因为一个候选变量缺失就会删除整条记录。对于 n=13 到 n=27 的极小样本，这是不可接受的。",
            "当前脚本采用列级剔除加中位数填补。每个干旱等级、每个分析层级单独计算每个候选变量的缺失率。"
            f"缺失率超过 {CFG.missing_drop_threshold:.0%} 的变量直接剔除；非缺失值少于 3 个或唯一值不足的变量也剔除；"
            "剩余变量如果有少量缺失，则用该列中位数填补。",
            "这种做法的好处是最大限度保留样本，同时让哪些变量被剔除、哪些变量被填补都有审计记录。"
            "人类活动变量如果当前全缺失，会被列级剔除，不会导致整行样本被删除。",
        ],
    )
    add_doc_table(
        doc,
        ["处理对象", "规则", "输出审计"],
        [
            ["人工指定变量", "若出现在 FEATURES_TO_DROP 中，则先剔除", "参数配置表记录 FEATURES_TO_DROP"],
            ["高缺失变量", f"缺失率 > {CFG.missing_drop_threshold:.0%} 时剔除", "变量缺失率与填补审计表"],
            ["有效值太少", "非缺失值少于 3 个时剔除", "变量缺失率与填补审计表"],
            ["无变化变量", "唯一值少于或等于 1 时剔除", "变量缺失率与填补审计表"],
            ["少量缺失变量", "用该变量中位数填补", "记录 impute_median"],
        ],
    )

    doc.add_heading("14. 共线性如何处理", level=1)
    add_doc_paragraphs(
        doc,
        [
            "随机森林的预测能力本身不像 OLS 或 LMM 那样直接受到共线性破坏，但变量重要性会受到共线性影响。"
            "如果 LAI 和 FAPAR 高度相关，模型可能把重要性分摊到两个变量上，导致单个变量的重要性被低估；"
            "也可能在不同随机种子下两个变量排名互换，使解释不稳定。",
            f"因此脚本计算候选变量的 Spearman 相关矩阵，并输出绝对相关系数超过 {CFG.high_corr_threshold} 的变量对。"
            "但脚本不自动删除变量，因为保留哪个变量需要结合生态学意义、数据质量和论文解释重点来判断。",
            "正式解释变量重要性前，应先查看高相关变量对审计表。如果存在明显同义或高度重叠的变量，"
            "应在 FEATURES_TO_DROP 中手动剔除其中一个，然后重跑脚本。这样比自动删除更可解释，也更符合论文方法透明性要求。",
        ],
    )

    doc.add_heading("15. 机器学习模型如何设计", level=1)
    add_doc_paragraphs(
        doc,
        [
            "脚本对每个干旱等级、每个分析层级分别建立随机森林回归模型。默认情况下，一共运行 4 个干旱等级 x 2 个层级，"
            "即 Mild、Moderate、Severe、Extreme 分别在站点级和站点-月级建模。",
            "每个模型的响应变量都是 DeltaCBI，解释变量来自候选特征筛选后的 X。模型不会把 DroughtLevel 当作一个普通特征混在一起训练，"
            "而是分等级建模。这样可以比较不同干旱等级下解释变量重要性是否发生变化，也避免不同等级机制混在同一个模型中。",
            "随机森林参数设置偏保守。n_estimators 控制树数量，max_depth 限制单棵树最大深度，min_samples_leaf 限制叶节点最小样本数。"
            "这些设置的目的不是追求训练集拟合最高，而是在小样本下限制模型复杂度，降低过拟合风险。",
        ],
    )
    add_doc_table(
        doc,
        ["项目", "设置", "原因"],
        [
            ["模型", "RandomForestRegressor", "用于探索非线性关系和变量交互，作为小样本解释工具。"],
            ["树数量", str(CFG.n_estimators), "保证集成模型稳定，同时控制运行时间。"],
            ["最大深度", str(CFG.max_depth), "限制树复杂度，降低小样本过拟合风险。"],
            ["叶节点最小样本数", str(CFG.min_samples_leaf), "避免树把个别样本完全记住。"],
            ["站点级 Mild/Moderate", "5-fold CV", "样本相对较多，可做 K 折验证。"],
            ["站点级 Severe/Extreme", "Leave-One-Out CV", "样本更少，尽量利用训练数据。"],
            ["站点-月级", "Leave-One-Site-Out CV", "同一站点不能同时出现在训练集和验证集，避免站点泄漏。"],
            ["基线模型", "每折训练集 y 均值", "避免使用全样本均值造成验证集信息泄漏。"],
        ],
    )

    doc.add_heading("16. 交叉验证和基线模型为什么这样设计", level=1)
    add_doc_paragraphs(
        doc,
        [
            "交叉验证是本脚本判断模型是否有基本外推能力的核心环节。小样本机器学习不能只看全样本拟合，"
            "因为全样本拟合容易过度乐观。脚本采用按等级和层级区分的交叉验证策略。",
            "站点级 Mild 和 Moderate 样本相对多一些，使用 5-fold CV；Severe 和 Extreme 样本更少，使用 Leave-One-Out CV，"
            "以最大化每一折中的训练样本量。站点-月级虽然记录数更多，但同一站点可能贡献多个站点-月记录，"
            "因此必须使用 Leave-One-Site-Out CV，避免同一站点同时出现在训练集和验证集造成信息泄漏。",
            "脚本还计算基线 RMSE。每个 fold 内只用训练集 y 均值预测验证集，而不是用全样本均值。"
            "这是为了避免验证集信息泄漏进基线模型。若随机森林的 CV-RMSE 不能优于这个训练折均值基线，"
            "说明模型解释能力很弱，对该等级的变量重要性应谨慎处理。",
        ],
    )

    doc.add_heading("17. 变量重要性如何计算和解释", level=1)
    add_doc_paragraphs(
        doc,
        [
            "脚本输出三类重要性相关结果。第一类是交叉验证下的重要性排序。如果 enable_shap=True 且 shap 可用，"
            "脚本会在每个 CV fold 的模型上计算 SHAP，并汇总各 fold 的平均绝对 SHAP 值。"
            "如果默认 enable_shap=False，则使用每个验证折训练模型的随机森林 impurity importance 作为快速 fallback。",
            "第二类是交叉验证置换重要性。置换重要性更接近“打乱某变量后模型误差增加多少”的思想，解释性较强，"
            "但在 4 等级 x 2 层级 x 多折 CV 下很耗时。因此默认关闭，建议只在重点等级或重点层级上打开。",
            "第三类是全样本随机种子稳定性。它用不同随机种子重复训练全样本随机森林，检查变量排名是否稳定。"
            "但这只是辅助诊断，不能替代交叉验证结果。正式论文中更应优先引用交叉验证下的重要性排序。",
            "解释变量重要性时要同时看模型表现、基线比较、相关性审计和稳定性。如果某个变量排名第一，"
            "但模型 RMSE 没有优于基线，或者该变量与另一个变量高度相关，或者不同随机种子下排名大幅变化，"
            "那么该变量只能作为弱提示，而不能写成明确驱动因素。",
        ],
    )

    doc.add_heading("18. 脚本运行流程总览表", level=1)
    add_doc_paragraphs(
        doc,
        [
            "脚本运行顺序是先构建响应变量，再构建解释变量，再做建模解释。这个顺序不能颠倒。"
            "如果先准备解释变量而没有明确 DeltaCBI 粒度，就容易出现变量时间窗口和响应变量时间窗口不一致的问题。",
            "脚本首先输出路径和参数审计表，并生成本 Word 方法说明书。这样即使后续模型运行时间较长，"
            "用户也能先得到本次运行的配置说明。随后读取逐小时温度和 SPI，构建四等级站点-月 DeltaCBI 基础表。"
            "基础表构建完成后，再按目标干旱日期窗口聚合 LAI、FAPAR 和土壤变量，并合并站点静态属性。",
            "机器学习阶段先输出站点级和站点-月级分析表，然后循环运行每个干旱等级和层级组合。"
            "每个组合内部先做特征预处理和审计，再做交叉验证建模，再做稳定性检验和重要性图输出。"
            "最后汇总所有等级和层级的模型表现、重要性和稳定性，并清理本次运行产生的临时缓存。",
        ],
    )
    add_doc_table(
        doc,
        ["顺序", "步骤", "产物或作用"],
        [
            ["0", "输出路径、模型参数、绘图参数和进度条参数审计表", "保证每次运行可追溯。"],
            ["1", "读取逐小时温度和逐日 SPI", "生成带 SPI30d 和 DroughtLevel 的小时数据。"],
            ["2", "构建四等级站点月 DeltaCBI 基础表", "得到核心响应变量表。"],
            ["3", "构建动态与静态解释变量", "读取 LAI、FAPAR、土壤和站点属性。"],
            ["4", "按干旱窗口聚合逐日协变量", "为每条 DeltaCBI 记录匹配事件窗口解释变量。"],
            ["5", "聚合站点级机器学习主表", "得到 Site_ID x DroughtLevel 主分析表。"],
            ["6", "输出基础表和样本量统计", "检查各等级有效样本量。"],
            ["7", "输出机器学习分析表", "保存站点级和站点-月级建模输入。"],
            ["8", "运行全部等级层级模型", "输出每个组合的模型表现、预测和重要性。"],
            ["9", "输出跨等级汇总表和运行摘要", "便于横向比较等级和层级。"],
            ["10", "清理本次运行临时缓存", "删除临时目录，只保留正式结果。"],
        ],
    )

    doc.add_heading("19. 如何判断模型做得怎样", level=1)
    add_doc_paragraphs(
        doc,
        [
            "判断模型做得怎样，不能只看 R2，也不能只看变量重要性图。尤其在小样本条件下，R2 可能不稳定，"
            "变量重要性也可能被少数样本或共线性变量影响。因此脚本提供多类证据共同判断。",
            "第一，看样本量。每个等级有多少有效站点、多少有效站点-月记录，决定了该等级结果能解释到什么程度。"
            "Extreme 等级如果只有十几个站点，即使跑出了模型，也必须谨慎。",
            "第二，看 CV-RMSE 是否优于 Baseline_RMSE_cv。如果随机森林不能优于训练折均值基线，说明这些解释变量没有提供明显预测增益。",
            "第三，看交叉验证预测值和观测值是否大致对应。如果预测明显收缩到均值附近，说明模型可能只学到很弱的信号。",
            "第四，看重要性排序是否有生态学合理性。例如干旱强度、持续时间、土壤水分、冠层结构或地形变量是否在相应等级下表现出合理的重要性。",
            "第五，看高相关审计表。如果高重要性变量与另一个变量高度相关，需要人工判断是否存在重要性分摊或替代解释。",
            "第六，看全样本随机种子稳定性。如果前几位变量在不同种子下排名基本一致，说明结果相对稳健；如果大幅摇摆，应弱化解释。",
            "第七，对比站点级和站点-月级。如果二者结论一致，说明长期空间格局和事件窗口分析相互支持；如果不一致，需要从静态背景与动态调节的差异来解释。",
        ],
    )

    doc.add_heading("20. 主要输出文件如何使用", level=1)
    add_doc_table(
        doc,
        ["文件", "内容"],
        [
            [OUTPUT_NAMES["method_docx"], "本方法说明书。"],
            [OUTPUT_NAMES["path_audit"], "输入路径和输出路径是否存在的审计。"],
            [OUTPUT_NAMES["parameter_table"], "模型、阈值、解释器路径和运行缓存路径。"],
            [OUTPUT_NAMES["plot_progress_parameter_table"], "绘图、图例、折线预留和进度条参数。"],
            [OUTPUT_NAMES["site_month_delta"], "Site_ID x YearMonth x DroughtLevel 的 DeltaCBI 基础表。"],
            [OUTPUT_NAMES["site_month_delta_audit"], "每个配对的小时数、回归状态和 Pair_flag 审计。"],
            [OUTPUT_NAMES["sample_counts"], "各干旱等级有效站点数和站点-月记录数。"],
            [OUTPUT_NAMES["site_level_ml_table"], "站点级机器学习输入表。"],
            [OUTPUT_NAMES["site_month_ml_table"], "站点-月级机器学习输入表。"],
            ["03_...变量缺失率与填补审计表.csv", "每个等级层级的特征缺失、剔除和填补记录。"],
            ["03_...高相关变量对审计表.csv", "高相关特征对及人工处理提醒。"],
            ["04_...随机森林交叉验证模型表现表.csv", "CV R2、CV RMSE、基线 RMSE 和改进量。"],
            ["05_...交叉验证重要性排序表.csv", "CV-SHAP 或 RF fallback 重要性。"],
            ["06_...全样本重要性稳定性表.csv", "不同随机种子下的重要性和排名稳定性。"],
            ["07_...交叉验证预测结果表.csv", "观测值、CV 预测值和基线预测值。"],
            [OUTPUT_NAMES["cross_summary"], "跨等级、跨层级模型表现汇总。"],
            [OUTPUT_NAMES["cache_cleanup"], "本次运行临时缓存清理记录。"],
        ],
    )
    add_doc_paragraphs(
        doc,
        [
            "建议实际查看顺序是：先看样本量统计，确认每个等级是否有足够样本；再看缺失值审计和高相关审计，"
            "确认进入模型的变量是否合理；然后看模型表现汇总，判断模型是否优于基线；最后再看重要性排序和图。",
            "如果某个等级模型表现很弱，不建议直接解释该等级的变量重要性。应在结果中说明该等级样本有限、模型外推能力不足，"
            "因此仅作为补充探索结果。"
        ],
    )

    doc.add_heading("21. 当前限制和论文表述边界", level=1)
    add_doc_paragraphs(
        doc,
        [
            "第一，小样本是最大限制。随机森林可以给出变量重要性，但不能消除 n=13 到 n=27 带来的统计不确定性。"
            "因此结论应定位为探索性解释，而不是确定性推断。",
            "第二，高相关变量会影响重要性解释。即使模型预测表现尚可，如果高相关变量没有人工处理，"
            "重要性排序也可能反映变量分摊，而不是单个变量的真实独立贡献。",
            "第三，当前有些方案变量可能尚未真正纳入，例如 root_zone_soil_moisture、nighttime_light、built_up_distance。"
            "若这些变量因文件不存在或全缺失被剔除，当前结果只能代表已有变量条件下的一版分析。",
            "第四，站点-月级记录虽然数量多，但不是完全独立样本，因为同一站点可以贡献多个月份。"
            "脚本用 Leave-One-Site-Out CV 降低泄漏风险，但解释时仍应承认重复测量结构的存在。",
            "第五，变量重要性不等于因果效应方向。随机森林重要性告诉我们变量对预测 DeltaCBI 有多大贡献，"
            "但不能单独说明变量增加会导致 DeltaCBI 增加还是减少。若需要方向性解释，后续应结合 SHAP 图、偏依赖图或分组可视化。",
            "建议论文表述为：本研究采用随机森林进行探索性解释建模，用于提示不同干旱等级下森林微气候缓冲变化的潜在解释变量及其相对重要性。"
            "由于样本量有限，结果不作为确定性因果推断，而需结合交叉验证表现、基线比较、重要性稳定性、相关性审计和生态学机制综合解释。",
        ],
    )

    doc.save(CFG.output_dir / OUTPUT_NAMES["method_docx"])


def main() -> None:
    start = time.time()
    ensure_output_dir()
    try:
        setup_plot_style()
        np.random.seed(CFG.random_seed)

        # 先输出路径、模型参数、绘图参数和进度条参数，保证每次运行都可追溯。
        with progress_bar(4, "步骤0/10 输出路径参数和方法说明", "输出") as bar:
            write_csv(build_path_audit(), OUTPUT_NAMES["path_audit"])
            bar.update(1)
            write_csv(build_parameter_table(), OUTPUT_NAMES["parameter_table"])
            bar.update(1)
            write_csv(build_plot_progress_parameter_table(), OUTPUT_NAMES["plot_progress_parameter_table"])
            bar.update(1)
            build_method_document()
            bar.update(1)

        hourly = read_hourly_with_spi()
        site_month_delta, delta_audit = build_site_month_delta_cbi_by_level(hourly)
        site_month_delta = add_time_varying_covariates(site_month_delta, hourly)

        with progress_bar(3, "步骤6/10 输出基础表和样本量统计", "输出") as bar:
            write_csv(site_month_delta, OUTPUT_NAMES["site_month_delta"])
            bar.update(1)
            write_csv(delta_audit, OUTPUT_NAMES["site_month_delta_audit"])
            bar.update(1)
            sample_counts = (
                site_month_delta.loc[site_month_delta["Pair_flag"].eq("ok")]
                .groupby("DroughtLevel")
                .agg(n_sites=("Site_ID", "nunique"), n_site_months=("YearMonth", "size"))
                .reindex(DROUGHT_LEVELS)
                .reset_index()
            )
            write_csv(sample_counts, OUTPUT_NAMES["sample_counts"])
            bar.update(1)

        site_level = build_site_level_table(site_month_delta)
        site_month_ml = build_site_month_ml_table(site_month_delta)
        with progress_bar(2, "步骤7/10 输出机器学习分析表", "输出") as bar:
            write_csv(site_level, OUTPUT_NAMES["site_level_ml_table"])
            bar.update(1)
            write_csv(site_month_ml, OUTPUT_NAMES["site_month_ml_table"])
            bar.update(1)

        all_summaries = []
        all_shap = []
        all_perm = []
        all_stability = []

        model_tasks = []
        for level in CFG.run_levels:
            if "site" in CFG.run_layers:
                model_tasks.append((level, "site"))
            if "site_month" in CFG.run_layers:
                model_tasks.append((level, "site_month"))

        # 外层进度条表示 4 个干旱等级 x 2 个分析层级的组合进度；每个组合内部还有 CV fold 进度条。
        with progress_bar(len(model_tasks), "步骤8/10 运行全部等级层级模型", "建模") as model_bar:
            for level, layer in model_tasks:
                if layer == "site":
                    task_df = site_level.loc[site_level["DroughtLevel"].eq(level)].copy()
                    groups = None
                else:
                    task_df = site_month_ml.loc[site_month_ml["DroughtLevel"].eq(level)].copy()
                    groups = task_df["Site_ID"] if "Site_ID" in task_df.columns else None
                result = run_ml_for_one(task_df, level, layer, groups=groups)
                all_summaries.append(result[0])
                all_shap.append(result[1])
                all_perm.append(result[2])
                all_stability.append(result[3])
                model_bar.update(1)

        summary = pd.concat([d for d in all_summaries if not d.empty], ignore_index=True)
        shap_all = pd.concat([d for d in all_shap if not d.empty], ignore_index=True) if any(not d.empty for d in all_shap) else pd.DataFrame()
        perm_all = pd.concat([d for d in all_perm if not d.empty], ignore_index=True) if any(not d.empty for d in all_perm) else pd.DataFrame()
        stability_all = (
            pd.concat([d for d in all_stability if not d.empty], ignore_index=True)
            if any(not d.empty for d in all_stability)
            else pd.DataFrame()
        )

        with progress_bar(4, "步骤9/10 输出跨等级汇总表", "输出") as bar:
            write_csv(summary, OUTPUT_NAMES["cross_summary"])
            bar.update(1)
            write_csv(shap_all, OUTPUT_NAMES["cross_importance"])
            bar.update(1)
            write_csv(perm_all, OUTPUT_NAMES["cross_permutation"])
            bar.update(1)
            write_csv(stability_all, OUTPUT_NAMES["cross_stability"])
            bar.update(1)

        notes = f"""机器学习解释分析运行摘要
========================

输出目录：
{CFG.output_dir}

核心基础表：
{OUTPUT_NAMES["site_month_delta"]}

基础表粒度：
Site_ID x YearMonth x DroughtLevel

DeltaCBI定义：
DeltaCBI = Target_CBI - Normal_CBI

Pair_flag说明：
当前 Pair_flag=ok 默认要求 Target 和 Normal 均至少 {CFG.min_status_hours} 个有效小时，
且 OLS CBI 估计成功。MacroSD 是否参与 Pair_flag：{CFG.use_macro_sd_for_pair_flag}。
MacroSD 仍保留为审计字段。

机器学习定位：
本分析是小样本探索性解释建模，目标是提示变量重要性排序和方向趋势，
不是严格因果推断或确定性统计结论。

重要实现：
1. 站点级为主分析，站点-月级为互补分析。
2. 站点-月级交叉验证使用 Leave-One-Site-Out，避免同一站点泄漏。
3. 基线 RMSE 在每个 CV fold 内使用训练集均值计算。
4. 缺失值按列剔除和中位数填补，不再整行删除。
5. 坡向转换为 aspect_sin / aspect_cos。
6. soil_type 若存在则 one-hot。
7. 高相关变量对输出到“高相关变量对审计表”；正式解读前建议在 FEATURES_TO_DROP 中人工筛选后重跑。
8. SHAP 默认关闭以保证全流程可运行；若 CFG.enable_shap=True 且 shap 可用，则使用 CV-SHAP。
   默认使用验证折模型的 RF impurity importance 作为 CV fallback。
9. CV permutation importance 默认关闭以保证四等级全流程可运行；
   如需单个组合深度解释，可将 CFG.enable_cv_permutation=True 后重跑。
10. 图像和进度条参数记录在：{OUTPUT_NAMES["plot_progress_parameter_table"]}
11. 完整方法说明书记录在：{OUTPUT_NAMES["method_docx"]}
12. 本次运行临时缓存清理记录在：{OUTPUT_NAMES["cache_cleanup"]}

运行耗时：
{time.time() - start:.1f} 秒
"""
        write_text(notes, OUTPUT_NAMES["run_summary"])
        print(notes)
    finally:
        cleanup_report = cleanup_runtime_cache(start)
        try:
            write_csv(cleanup_report, OUTPUT_NAMES["cache_cleanup"])
        except Exception:
            pass


if __name__ == "__main__":
    main()
