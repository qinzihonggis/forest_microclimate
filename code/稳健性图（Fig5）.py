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
# 本脚本负责 Fig. 5“稳健性图”。Fig. 5 不提出新机制，而是回答审稿人
# 对 Fig. 1-Fig. 4 的可信度质疑：主结论是否依赖某个阈值、某个站点、
# 某个区域干旱过程或某个样本质量门槛。
#
# 计划面板：
#   Fig. 5A：不同事件定义、正常期窗口和质量阈值下总体 Delta_CBI 森林图。
#   Fig. 5B：Leave-one-site-out 逐站点删除森林图。
#   Fig. 5C：Leave-one-regional-process-out 逐区域干旱过程删除森林图。
#
# 当前脚本先读取并审计已有稳健性结果路径，绘制占位草稿。后续将把已有
# 06/07 森林图逻辑整合成统一 Fig. 5 三面板。


@dataclass(frozen=True)
class Config:
    output_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\results\论文主图\Fig5_稳健性图")
    robustness_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\results\robustness_sensitivity_results")
    loso_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\results\robustness_sensitivity_results\Leave_one_site_out")
    lorpo_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\results\robustness_sensitivity_results\Leave_one_regional_drought_process_out")
    multilevel_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\results\compare_differences_results\多等级干旱扩展分析")
    figure_dpi: int = 600


@dataclass(frozen=True)
class FigureParameters:
    fig_width: float = 11.5
    fig_height: float = 5.6
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
    with progress_bar("步骤1/5 审计Fig5稳健性输入", 4, "cyan") as bar:
        key_paths = [
            CFG.robustness_dir / "05_全部稳健性分析结果汇总.csv",
            CFG.loso_dir / "21_Leave_one_site_out_逐站点剔除LMM结果.csv",
            CFG.lorpo_dir / "22_逐区域过程剔除事件CBI结果.csv",
            CFG.multilevel_dir / "05_多等级_LMM结果汇总.csv",
        ]
        rows = []
        for path in key_paths:
            rows.append({"路径": str(path), "是否存在": path.exists(), "文件大小字节": path.stat().st_size if path.exists() else None})
        bar.update()
        audit = pd.DataFrame(rows)
        bar.update()
        for i, path in enumerate(key_paths):
            if path.exists() and path.suffix.lower() == ".csv":
                audit.loc[i, "行数"] = len(pd.read_csv(path))
        bar.update()
        audit["用途"] = ["Fig5A 阈值/事件定义敏感性", "Fig5B 逐站点删除", "Fig5C 逐区域过程删除", "多等级响应辅助证据"]
        bar.update()
    return audit


def draw_placeholder(audit: pd.DataFrame) -> plt.Figure:
    with progress_bar("步骤2/5 绘制Fig5占位草稿", 3, "yellow") as bar:
        fig, axes = plt.subplots(1, 3, figsize=(FP.fig_width, FP.fig_height))
        titles = ["A  Definition sensitivity", "B  Leave-one-site-out", "C  Leave-one-process-out"]
        notes = [
            "TODO: forest plot of LMM/Event Delta CBI across thresholds.",
            "TODO: forest plot of Delta CBI after deleting each site.",
            "TODO: forest plot after deleting each regional drought process.",
        ]
        bar.update()
        for ax, title, note in zip(axes, titles, notes):
            ax.set_axis_off()
            ax.text(0.05, 0.8, title, transform=ax.transAxes, fontsize=FP.text_size, fontweight="bold")
            ax.text(0.05, 0.6, note, transform=ax.transAxes, fontsize=9, va="top")
            ax.text(0.05, 0.35, "Input files detected:\n" + "\n".join(audit["是否存在"].astype(str).tolist()), transform=ax.transAxes, fontsize=8)
        bar.update()
        fig.suptitle("Main conclusions were robust to definitions and influence-point tests", fontsize=FP.title_size, fontweight="bold")
        bar.update()
    return fig


def parameters_to_table() -> pd.DataFrame:
    rows = []
    for name, value in asdict(CFG).items():
        rows.append({"参数组": "路径", "参数名": name, "当前值": str(value), "用途说明": "Fig5 输入输出路径。"})
    for name, value in asdict(FP).items():
        rows.append({"参数组": "图形参数", "参数名": name, "当前值": value, "用途说明": "控制 Fig5 占位图或后续森林图外观。"})
    return pd.DataFrame(rows)


def write_outputs(fig: plt.Figure, audit: pd.DataFrame) -> pd.DataFrame:
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    with progress_bar("步骤3/5 保存Fig5输出", 5, "magenta") as bar:
        fig.savefig(CFG.output_dir / "图5_稳健性图_占位草稿.png", dpi=CFG.figure_dpi)
        bar.update()
        fig.savefig(CFG.output_dir / "图5_稳健性图_占位草稿.pdf")
        bar.update()
        audit.to_csv(CFG.output_dir / "图5_稳健性输入审计表.csv", index=False, encoding="utf-8-sig")
        bar.update()
        parameters_to_table().to_csv(CFG.output_dir / "00_Fig5绘图参数说明表.csv", index=False, encoding="utf-8-sig")
        bar.update()
        summary = pd.DataFrame([{"状态": "稳健性图脚手架已建立；三面板森林图待整合。", "可用输入文件数": int(audit["是否存在"].sum())}])
        summary.to_csv(CFG.output_dir / "图5_运行摘要表.csv", index=False, encoding="utf-8-sig")
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
    fig = draw_placeholder(audit)
    summary = write_outputs(fig, audit)
    plt.close(fig)
    cleanup_runtime_cache()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        make_figure()
