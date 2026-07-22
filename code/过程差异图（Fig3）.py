from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


# =============================================================================
# 0. 论文主线与后续图件路线图
# =============================================================================
# 本脚本负责 Fig. 3“过程差异图”。Fig. 3 必须服务于 Fig. 1 提出的核心现象：
# 极端干旱下 Delta_CBI 在站点间呈现空间异质性。Fig. 2 已经检验“干旱强度
# 和持续时间是否足以解释这种异质性”；Fig. 3 的任务是进一步检查不同
# Delta_CBI 响应组是否经历了不同的水分、植被、能量或大气过程。
#
# 全文闭环：
#   Fig. 1：发现问题。极端干旱诱发的 CBI 响应具有空间异质性。
#   Fig. 2：排除直接解释。干旱等级、强度和持续时间不能充分解释响应分化。
#   Fig. 3：识别过程差异。不同响应组对应不同水分-植被-能量变化模式。
#   Fig. 4：解释机制。Spatial background -> Drought exposure ->
#           Hydrological / vegetation / energy changes -> Delta_CBI。
#   Fig. 5：证明可信。主结论不依赖单一阈值、站点或区域干旱过程。
#   Fig. 6：可选扩展。若论文讨论恢复/韧性，则展示干旱过程和恢复轨迹。
#
# Fig. 3 计划：
#   Fig. 3A：环境变化热图。
#       每行是有效“事件 x 站点”样本；
#       每列是候选环境变量；
#       单元格是事件期相对参考期变化 Delta_X，经 z-score 标准化后显示；
#       行按 Delta_CBI 或响应组排序。
#   Fig. 3B：关键变量响应组比较图。
#       从 Fig. 3A 中筛选理论明确、缺失率可接受、非高度重复的 4-6 个变量；
#       比较 strengthened/stable/weakened 三类响应组的 Delta_X。
#
# 当前脚本先建立标准化流程和输入审计，不强行计算所有 Delta_X。
# 后续需要补充：
#   1. 把事件期和参考期边界与 Tensor_Data 的 15 分钟记录对齐；
#   2. 对 SMC、T1_5、T2_0、T3_15、unknown_data1/2/3 等候选变量计算 Delta_X；
#   3. 输出变量缺失率、样本量、组间差异和候选变量筛选表；
#   4. 用筛选结果正式绘制 Fig. 3A-B，并为 Fig. 4 提供输入。


@dataclass(frozen=True)
class Config:
    project_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate")
    tensor_data_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\Tensor_Data")
    output_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\results\论文主图\Fig3_过程差异图")
    event_pairs_csv: Path = Path(
        r"E:\forest_microclimate\ForestMicroclimate\results\compare_differences_results\04_极端事件CBI与事件后正常参考期对比表.csv"
    )
    drought_events_csv: Path = Path(
        r"E:\forest_microclimate\ForestMicroclimate\results\daily_SPI_features\福建省观测站2025年daily_SPI干旱事件长表.csv"
    )
    response_near_zero_threshold: float = 0.05
    figure_dpi: int = 600


@dataclass(frozen=True)
class FigureParameters:
    fig_width: float = 10.5
    fig_height: float = 5.6
    suptitle_size: int = 12
    axis_label_size: int = 9
    title_size: int = 10
    placeholder_facecolor: str = "#f7f7f7"
    placeholder_edgecolor: str = "#4d4d4d"


CFG = Config()
FP = FigureParameters()


PROGRESS_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed}, remaining: {remaining}, {rate_fmt}]"


def progress_bar(desc: str, total: int, colour: str) -> tqdm:
    return tqdm(total=total, desc=desc, colour=colour, dynamic_ncols=True, leave=False, ncols=100, bar_format=PROGRESS_BAR_FORMAT)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "axes.unicode_minus": False,
            "axes.labelsize": FP.axis_label_size,
            "axes.titlesize": FP.title_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def audit_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """审计 Fig. 3 输入，不计算正式过程变量。"""

    with progress_bar("步骤1/5 审计Fig3输入", 4, "cyan") as bar:
        tensor_files = sorted(CFG.tensor_data_dir.glob("953322*.csv"))
        bar.update()

        rows = []
        for path in tensor_files:
            sample = pd.read_csv(path, nrows=3)
            rows.append(
                {
                    "文件名": path.name,
                    "站点ID": path.stem,
                    "文件大小字节": path.stat().st_size,
                    "列名": ",".join(sample.columns),
                    "样例行数": len(sample),
                }
            )
        tensor_audit = pd.DataFrame(rows)
        bar.update()

        required = [CFG.event_pairs_csv, CFG.drought_events_csv, CFG.tensor_data_dir]
        path_audit = pd.DataFrame(
            [{"路径": str(path), "是否存在": path.exists()} for path in required]
        )
        bar.update()

        if CFG.event_pairs_csv.exists():
            pairs = pd.read_csv(CFG.event_pairs_csv)
            path_audit.loc[path_audit["路径"].eq(str(CFG.event_pairs_csv)), "行数或文件数"] = len(pairs)
        path_audit.loc[path_audit["路径"].eq(str(CFG.tensor_data_dir)), "行数或文件数"] = len(tensor_files)
        bar.update()

    return tensor_audit, path_audit


def make_placeholder_figure(tensor_audit: pd.DataFrame) -> plt.Figure:
    """生成 Fig. 3 占位草稿，说明后续正式图件内容。"""

    with progress_bar("步骤2/5 绘制Fig3占位草稿", 3, "yellow") as bar:
        fig, ax = plt.subplots(figsize=(FP.fig_width, FP.fig_height))
        ax.set_facecolor(FP.placeholder_facecolor)
        bar.update()

        text = (
            "Fig. 3 will compare hydro-vegetation-energy changes among Delta CBI response groups.\n\n"
            f"Tensor_Data files detected: {len(tensor_audit)} site tables\n"
            "Available candidate variables: SMC, T1_5, T2_0, T3_15, unknown_data1/2/3\n\n"
            "TODO:\n"
            "1. Align event/reference windows to 15-min Tensor_Data records.\n"
            "2. Compute Delta_X = X_event - X_reference for candidate variables.\n"
            "3. Build response-group heatmap and key-variable comparison panels.\n"
            "4. Export selected process variables for Fig. 4 pathway analysis."
        )
        ax.text(0.04, 0.92, text, transform=ax.transAxes, ha="left", va="top", fontsize=10)
        bar.update()

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Fig. 3 scaffold: contrasting process changes by response group", loc="left", fontweight="bold")
        fig.suptitle("Process differences behind heterogeneous buffering responses", fontsize=FP.suptitle_size, fontweight="bold")
        bar.update()

    return fig


def parameters_to_table() -> pd.DataFrame:
    rows = []
    for name, value in asdict(CFG).items():
        rows.append({"参数组": "路径与阈值", "参数名": name, "当前值": str(value), "用途说明": "Fig.3 输入输出路径、响应分组阈值或图像分辨率。"})
    for name, value in asdict(FP).items():
        rows.append({"参数组": "图形参数", "参数名": name, "当前值": value, "用途说明": "控制 Fig.3 占位图或后续正式图件外观。"})
    return pd.DataFrame(rows)


def write_outputs(fig: plt.Figure, tensor_audit: pd.DataFrame, path_audit: pd.DataFrame) -> pd.DataFrame:
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    with progress_bar("步骤3/5 保存Fig3输出", 6, "magenta") as bar:
        fig.savefig(CFG.output_dir / "图3_过程差异图_占位草稿.png", dpi=CFG.figure_dpi)
        bar.update()
        fig.savefig(CFG.output_dir / "图3_过程差异图_占位草稿.pdf")
        bar.update()
        tensor_audit.to_csv(CFG.output_dir / "图3_TensorData输入审计表.csv", index=False, encoding="utf-8-sig")
        bar.update()
        path_audit.to_csv(CFG.output_dir / "图3_输入路径审计表.csv", index=False, encoding="utf-8-sig")
        bar.update()
        parameters_to_table().to_csv(CFG.output_dir / "00_Fig3绘图参数说明表.csv", index=False, encoding="utf-8-sig")
        bar.update()
        summary = pd.DataFrame([{"TensorData站点表数量": len(tensor_audit), "状态": "脚手架已建立，Delta_X 计算和正式图件待补充。"}])
        summary.to_csv(CFG.output_dir / "图3_运行摘要表.csv", index=False, encoding="utf-8-sig")
        bar.update()
    return summary


def cleanup_runtime_cache() -> None:
    with progress_bar("步骤4/5 清理运行缓存", 3, "blue") as bar:
        pycache_dir = Path(__file__).resolve().parent / "__pycache__"
        if pycache_dir.exists():
            for pyc in pycache_dir.glob(f"{Path(__file__).stem}*.pyc"):
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


def make_figure() -> None:
    setup_style()
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_audit, path_audit = audit_inputs()
    fig = make_placeholder_figure(tensor_audit)
    summary = write_outputs(fig, tensor_audit, path_audit)
    plt.close(fig)
    cleanup_runtime_cache()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        make_figure()
