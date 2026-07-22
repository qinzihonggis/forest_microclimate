from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import pandas as pd
from tqdm import tqdm


# =============================================================================
# 0. 论文主线与后续图件路线图
# =============================================================================
# 本脚本负责 Fig. 4“路径机制图”。Fig. 4 是全文机制核心，但必须建立在
# Fig. 1-Fig. 3 的结果之上：Fig. 1 提出现象，Fig. 2 排除干旱暴露强度
# 的充分解释，Fig. 3 识别差异化环境过程。Fig. 4 不能强行证明因果，
# 应表述为 observational pathway framework 或 pathway analysis。
#
# 固定路径框架：
#   Spatial background -> Drought exposure ->
#   Hydrological / vegetation / energy changes -> Delta_CBI
#
# 当前脚本先绘制机制框架草稿和输入审计，不运行 SEM。
# 后续待 Fig. 3 输出候选过程变量后，再补充：
#   1. 标准化效应模型或路径模型；
#   2. 直接效应、间接效应、总效应分解；
#   3. 共线性、缺失率、样本量和随机效应结构审计；
#   4. 正式 Fig. 4A 路径图和 Fig. 4B 效应分解图。


@dataclass(frozen=True)
class Config:
    output_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\results\论文主图\Fig4_路径机制图")
    fig3_candidate_table: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\results\论文主图\Fig3_过程差异图\图3_候选过程变量筛选表.csv")
    event_exposure_table: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\results\论文主图\Fig2_暴露解释图\图2_干旱暴露与DeltaCBI分析基础表.csv")
    figure_dpi: int = 600


@dataclass(frozen=True)
class FigureParameters:
    fig_width: float = 10.0
    fig_height: float = 5.8
    node_facecolor: str = "#f4efe6"
    node_edgecolor: str = "#343434"
    arrow_color: str = "#5b5b5b"
    title_size: int = 12
    text_size: int = 10


CFG = Config()
FP = FigureParameters()
PROGRESS_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed}, remaining: {remaining}, {rate_fmt}]"


def progress_bar(desc: str, total: int, colour: str) -> tqdm:
    return tqdm(total=total, desc=desc, colour=colour, dynamic_ncols=True, leave=False, ncols=100, bar_format=PROGRESS_BAR_FORMAT)


def setup_style() -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight"})


def audit_inputs() -> pd.DataFrame:
    with progress_bar("步骤1/5 审计Fig4输入", 2, "cyan") as bar:
        paths = [CFG.fig3_candidate_table, CFG.event_exposure_table]
        rows = []
        for path in paths:
            rows.append({"路径": str(path), "是否存在": path.exists(), "说明": "Fig4 正式路径模型输入" if path.exists() else "待 Fig3/Fig2 生成或确认"})
        bar.update()
        audit = pd.DataFrame(rows)
        bar.update()
    return audit


def draw_framework() -> plt.Figure:
    with progress_bar("步骤2/5 绘制Fig4路径框架", 4, "yellow") as bar:
        fig, ax = plt.subplots(figsize=(FP.fig_width, FP.fig_height))
        ax.set_axis_off()
        nodes = {
            "Spatial background\n(topography, canopy,\nposition)": (0.12, 0.62),
            "Drought exposure\n(Min SPI30d,\nseverity, duration)": (0.38, 0.62),
            "Hydrological /\nvegetation /\nenergy changes\n(Delta_X)": (0.64, 0.62),
            "Delta CBI\n(event - reference)": (0.88, 0.62),
        }
        for label, (x, y) in nodes.items():
            ax.text(x, y, label, transform=ax.transAxes, ha="center", va="center", fontsize=FP.text_size, bbox={"boxstyle": "round,pad=0.45", "fc": FP.node_facecolor, "ec": FP.node_edgecolor, "lw": 1.0})
        bar.update()
        xs = [0.24, 0.50, 0.76]
        xe = [0.30, 0.56, 0.82]
        for start, end in zip(xs, xe):
            ax.add_patch(FancyArrowPatch((start, 0.62), (end, 0.62), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=16, lw=1.6, color=FP.arrow_color))
        bar.update()
        ax.text(0.5, 0.22, "TODO: replace framework arrows with standardized path/effect estimates after Fig. 3 variable screening.", transform=ax.transAxes, ha="center", va="center", fontsize=10, color="#555555")
        bar.update()
        fig.suptitle("Spatial background shaped drought exposure and environmental pathways leading to Delta CBI", fontsize=FP.title_size, fontweight="bold")
        bar.update()
    return fig


def parameters_to_table() -> pd.DataFrame:
    rows = []
    for name, value in asdict(CFG).items():
        rows.append({"参数组": "路径与输入", "参数名": name, "当前值": str(value), "用途说明": "Fig4 输出路径或路径模型输入。"})
    for name, value in asdict(FP).items():
        rows.append({"参数组": "图形参数", "参数名": name, "当前值": value, "用途说明": "控制 Fig4 框架图外观。"})
    return pd.DataFrame(rows)


def write_outputs(fig: plt.Figure, audit: pd.DataFrame) -> pd.DataFrame:
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    with progress_bar("步骤3/5 保存Fig4输出", 5, "magenta") as bar:
        fig.savefig(CFG.output_dir / "图4_路径机制图_框架草稿.png", dpi=CFG.figure_dpi)
        bar.update()
        fig.savefig(CFG.output_dir / "图4_路径机制图_框架草稿.pdf")
        bar.update()
        audit.to_csv(CFG.output_dir / "图4_输入路径审计表.csv", index=False, encoding="utf-8-sig")
        bar.update()
        parameters_to_table().to_csv(CFG.output_dir / "00_Fig4绘图参数说明表.csv", index=False, encoding="utf-8-sig")
        bar.update()
        summary = pd.DataFrame([{"状态": "路径机制框架脚手架已建立；正式路径系数待 Fig3 候选变量确定后补充。"}])
        summary.to_csv(CFG.output_dir / "图4_运行摘要表.csv", index=False, encoding="utf-8-sig")
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
        bar.update()


def make_figure() -> None:
    setup_style()
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_inputs()
    fig = draw_framework()
    summary = write_outputs(fig, audit)
    plt.close(fig)
    cleanup_runtime_cache()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        make_figure()
