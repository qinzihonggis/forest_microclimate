# -*- coding: utf-8 -*-
"""
森林微气候缓冲能力空间差异（Why部分）机器学习解释脚本
====================================================

本脚本按“先构建基础表，再做机器学习解释”的两步流程运行。

Step 1 先从逐小时温度表和逐日 SPI30d 宽表重新计算四个干旱等级
（Mild / Moderate / Severe / Extreme）在 Site_ID x YearMonth 层面的
Target_CBI、Normal_CBI 和 DeltaCBI，并输出可复用基础表：

    site_month_delta_cbi_by_level.csv

Step 2 读取该基础表，分别构建站点级主分析和站点-月级互补分析，使用
Random Forest + 交叉验证 + SHAP/Permutation/稳定性检验进行探索性解释。

重要定位：
    由于样本量很小，尤其 Severe / Extreme 的站点级样本有限，本脚本结果
    定位为 exploratory explanatory modeling，用于提示变量重要性排序和
    方向趋势，不作为严格因果推断或确定性统计结论。
"""

from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, LeaveOneOut
from sklearn.utils import check_random_state

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import shap

    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# =============================================================================
# 0. 全局配置
# =============================================================================


@dataclass(frozen=True)
class Config:
    project_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate")
    output_dir: Path = Path(
        r"E:\forest_microclimate\ForestMicroclimate\results\Modeling_Machine_Learning_Explanation"
    )

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

    site_col: str = "Site_ID"
    time_col: str = "Time_UTC"
    macro_temp_col: str = "ERA5_T2m_C"
    micro_temp_col: str = "Observed_T15cm_C"
    has_both_col: str = "Has_Both_Data"

    normal_spi_low: float = -0.5
    normal_spi_high: float = 0.5
    extreme_spi_threshold: float = -2.0
    min_status_hours: int = 72
    min_macro_sd: float = 1.0
    use_macro_sd_for_pair_flag: bool = False

    missing_drop_threshold: float = 0.50
    high_corr_threshold: float = 0.85
    random_seed: int = 20250714
    n_estimators: int = 100
    max_depth: int = 4
    min_samples_leaf: int = 2
    n_stability_seeds: int = 5
    n_permutation_repeats: int = 5
    enable_shap: bool = False
    enable_cv_permutation: bool = False
    run_levels: tuple[str, ...] = ("Mild", "Moderate", "Severe", "Extreme")
    run_layers: tuple[str, ...] = ("site", "site_month")


CFG = Config()

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

# Strongly correlated variables should be manually removed here after reviewing
# high_corr_pairs_*.csv. The script only warns; it does not silently decide for you.
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


# =============================================================================
# 1. 通用工具
# =============================================================================


def ensure_output_dir() -> None:
    CFG.output_dir.mkdir(parents=True, exist_ok=True)


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
    for font_name in ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"]:
        try:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
        except Exception:
            continue


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
    hourly = pd.read_csv(CFG.hourly_temperature_csv, encoding="utf-8-sig")
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

    spi_wide = pd.read_excel(CFG.spi_daily_wide_xlsx)
    date_col = spi_wide.columns[0]
    spi_long = spi_wide.melt(id_vars=[date_col], var_name=CFG.site_col, value_name="SPI30d")
    spi_long = spi_long.rename(columns={date_col: "UTC_Date"})
    spi_long[CFG.site_col] = normalise_site_id(spi_long[CFG.site_col])
    spi_long["UTC_Date"] = pd.to_datetime(spi_long["UTC_Date"], errors="coerce")

    hourly = hourly.merge(spi_long, on=[CFG.site_col, "UTC_Date"], how="left", validate="many_to_one")
    hourly["DroughtLevel"] = classify_drought_level(hourly["SPI30d"])
    hourly["DroughtLevel_CN"] = hourly["DroughtLevel"].map(DROUGHT_LEVELS_CN).fillna(hourly["DroughtLevel"])
    hourly["Site_Month"] = hourly[CFG.site_col].astype(str) + "_" + hourly["YearMonth"]
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

    for (site_id, year_month), site_month in hourly.groupby(group_cols, sort=True):
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
    for _, row in long_df.dropna(subset=["Date"]).iterrows():
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
    lai_daily = expand_8day_to_daily(wide_time_series_to_long(CFG.lai_8day_csv, "LAI"), "LAI")
    fapar_daily = expand_8day_to_daily(wide_time_series_to_long(CFG.fapar_8day_csv, "FAPAR"), "FAPAR")
    soil_daily = load_daily_soil()

    rows = []
    key_cols = [CFG.site_col, "YearMonth", "DroughtLevel"]
    target_dates = (
        hourly.loc[hourly["DroughtLevel"].isin(DROUGHT_LEVELS)]
        .groupby(key_cols)["UTC_Date"]
        .apply(lambda x: sorted(pd.to_datetime(x.dropna()).dt.floor("D").unique()))
        .reset_index(name="Target_Dates")
    )
    base = site_month_delta.merge(target_dates, on=key_cols, how="left")

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

    cov = pd.DataFrame(rows)
    out = site_month_delta.merge(cov, on=key_cols, how="left")
    out = out.merge(load_static_attributes(), on=CFG.site_col, how="left")
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
    for (site_id, level), g in valid.groupby([CFG.site_col, "DroughtLevel"], sort=True):
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
        corr.to_csv(CFG.output_dir / f"correlation_matrix_{level_name}_{layer_name}.csv", encoding="utf-8-sig")
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

    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, X_test = Xv.iloc[train_idx], Xv.iloc[test_idx]
        y_train, y_test = yv.iloc[train_idx], yv.iloc[test_idx]
        model = make_rf(CFG.random_seed + fold_id)
        model.fit(X_train, y_train)
        y_pred[test_idx] = model.predict(X_test)
        baseline_pred[test_idx] = float(y_train.mean())

        if HAS_SHAP and CFG.enable_shap:
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
    pred_df.to_csv(CFG.output_dir / f"cv_predictions_{level_name}_{layer_name}.csv", index=False, encoding="utf-8-sig")
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
    for seed in range(CFG.n_stability_seeds):
        model = make_rf(seed)
        model.fit(Xv, yv)
        # This is deliberately an auxiliary full-sample stability diagnostic.
        # CV-based SHAP / permutation outputs are the primary interpretable evidence.
        # Using RF impurity importance here avoids very slow repeated full-sample
        # permutation runs across 8 level-layer combinations.
        values = model.feature_importances_
        ranks = pd.Series(values, index=Xv.columns).rank(ascending=False, method="average")
        for feature in Xv.columns:
            rows.append(
                {
                    "DroughtLevel": level_name,
                    "Layer": layer_name,
                    "seed": seed,
                    "feature": feature,
                    "importance": float(pd.Series(values, index=Xv.columns)[feature]),
                    "rank": float(ranks[feature]),
                }
            )
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
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(d) + 1.5)))
    ax.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"], color="#3b7a8f", alpha=0.85)
    ax.set_xlabel(source)
    ax.set_title(f"{DROUGHT_LEVELS_CN.get(level_name, level_name)} - {layer_name} 变量重要性")
    fig.tight_layout()
    fig.savefig(CFG.output_dir / f"importance_{source}_{level_name}_{layer_name}.png", dpi=300)
    plt.close(fig)


def run_ml_for_one(df: pd.DataFrame, level_name: str, layer_name: str, groups: pd.Series | None):
    X, y, missing_report, high_corr = prepare_features(df, level_name, layer_name)
    write_csv(missing_report, f"missing_value_report_{level_name}_{layer_name}.csv")
    write_csv(high_corr, f"high_corr_pairs_{level_name}_{layer_name}.csv")

    summary, shap_imp, perm_imp = cv_predict_and_importance(X, y, level_name, layer_name, groups)
    stability = stability_importance(X, y, level_name, layer_name)

    write_csv(summary, f"model_summary_{level_name}_{layer_name}.csv")
    write_csv(shap_imp, f"cv_shap_or_fallback_importance_{level_name}_{layer_name}.csv")
    write_csv(perm_imp, f"cv_permutation_importance_{level_name}_{layer_name}.csv")
    write_csv(stability, f"full_sample_rank_stability_{level_name}_{layer_name}.csv")
    plot_importance(shap_imp, level_name, layer_name, "cv_shap_or_fallback")
    plot_importance(perm_imp, level_name, layer_name, "cv_permutation_mse")
    return summary, shap_imp, perm_imp, stability


# =============================================================================
# 6. 主流程
# =============================================================================


def build_path_audit() -> pd.DataFrame:
    paths = {
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
    return pd.DataFrame(rows)


def main() -> None:
    start = time.time()
    ensure_output_dir()
    setup_plot_style()
    np.random.seed(CFG.random_seed)

    write_csv(build_path_audit(), "00_输入路径审计表.csv")
    write_csv(build_parameter_table(), "00_参数配置表.csv")

    print("Step 1/2: 构建四等级站点-月 DeltaCBI 基础表...")
    hourly = read_hourly_with_spi()
    site_month_delta, delta_audit = build_site_month_delta_cbi_by_level(hourly)
    site_month_delta = add_time_varying_covariates(site_month_delta, hourly)
    write_csv(site_month_delta, "site_month_delta_cbi_by_level.csv")
    write_csv(delta_audit, "site_month_delta_cbi_by_level_audit.csv")

    sample_counts = (
        site_month_delta.loc[site_month_delta["Pair_flag"].eq("ok")]
        .groupby("DroughtLevel")
        .agg(n_sites=("Site_ID", "nunique"), n_site_months=("YearMonth", "size"))
        .reindex(DROUGHT_LEVELS)
        .reset_index()
    )
    write_csv(sample_counts, "01_四等级基础表有效样本量统计.csv")

    print("Step 2/2: 运行站点级与站点-月级机器学习解释...")
    site_level = build_site_level_table(site_month_delta)
    site_month_ml = build_site_month_ml_table(site_month_delta)
    write_csv(site_level, "site_level_ml_table_by_level.csv")
    write_csv(site_month_ml, "site_month_ml_table_by_level.csv")

    all_summaries = []
    all_shap = []
    all_perm = []
    all_stability = []

    for level in CFG.run_levels:
        if "site" in CFG.run_layers:
            site_df = site_level.loc[site_level["DroughtLevel"].eq(level)].copy()
            result = run_ml_for_one(site_df, level, "site", groups=None)
            all_summaries.append(result[0])
            all_shap.append(result[1])
            all_perm.append(result[2])
            all_stability.append(result[3])

        if "site_month" in CFG.run_layers:
            sm_df = site_month_ml.loc[site_month_ml["DroughtLevel"].eq(level)].copy()
            groups = sm_df["Site_ID"] if "Site_ID" in sm_df.columns else None
            result = run_ml_for_one(sm_df, level, "site_month", groups=groups)
            all_summaries.append(result[0])
            all_shap.append(result[1])
            all_perm.append(result[2])
            all_stability.append(result[3])

    summary = pd.concat([d for d in all_summaries if not d.empty], ignore_index=True)
    shap_all = pd.concat([d for d in all_shap if not d.empty], ignore_index=True) if any(not d.empty for d in all_shap) else pd.DataFrame()
    perm_all = pd.concat([d for d in all_perm if not d.empty], ignore_index=True) if any(not d.empty for d in all_perm) else pd.DataFrame()
    stability_all = (
        pd.concat([d for d in all_stability if not d.empty], ignore_index=True)
        if any(not d.empty for d in all_stability)
        else pd.DataFrame()
    )

    write_csv(summary, "cross_level_model_summary.csv")
    write_csv(shap_all, "cross_level_cv_shap_or_fallback_importance.csv")
    write_csv(perm_all, "cross_level_cv_permutation_importance.csv")
    write_csv(stability_all, "cross_level_full_sample_rank_stability.csv")

    notes = f"""机器学习解释分析运行摘要
========================

输出目录：
{CFG.output_dir}

核心基础表：
site_month_delta_cbi_by_level.csv

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
7. 高相关变量对输出到 high_corr_pairs_*.csv；正式解读前建议在 FEATURES_TO_DROP 中人工筛选后重跑。
8. SHAP 默认关闭以保证全流程可运行；若 CFG.enable_shap=True 且 shap 可用，则使用 CV-SHAP。
   默认使用验证折模型的 RF impurity importance 作为 CV fallback。
9. CV permutation importance 默认关闭以保证四等级全流程可运行；
   如需单个组合深度解释，可将 CFG.enable_cv_permutation=True 后重跑。

运行耗时：
{time.time() - start:.1f} 秒
"""
    write_text(notes, "20_运行摘要说明.txt")
    print(notes)


if __name__ == "__main__":
    main()
