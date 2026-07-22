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
# 本脚本负责可选 Fig. 6“恢复过程图”。只有当论文主问题扩展到“干旱结束后
# 的恢复/韧性”时，Fig. 6 才建议进入正文；否则可作为补充材料。
#
# 计划面板：
#   Fig. 6A：干旱过程 Early/Middle/Late 阶段 CBI。
#   Fig. 6B：干旱结束后 1-7、8-14、15-30 天恢复窗口 CBI。
#   Fig. 6C：7 天或 14 天滑动窗口恢复轨迹。
#
# 当前脚本先审计已有 CBI_process_recovery_window_analysis 输出并生成占位草稿。


@dataclass(frozen=True)
class Config:
    output_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\results\论文主图\Fig6_恢复过程图_可选")
    recovery_dir: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\results\robustness_sensitivity_results\CBI_process_recovery_window_analysis")
    stage_detail_csv: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\results\robustness_sensitivity_results\CBI_process_recovery_window_analysis\30_CBI过程恢复_阶段明细.csv")
    sliding_detail_csv: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\results\robustness_sensitivity_results\CBI_process_recovery_window_analysis\31_CBI过程恢复_滑动窗口明细.csv")
    sliding_summary_csv: Path = Path(r"E:\forest_microclimate\ForestMicroclimate\results\robustness_sensitivity_results\CBI_process_recovery_window_analysis\33_CBI过程恢复_滑动轨迹汇总.csv")
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
    with progress_bar("步骤1/5 审计Fig6恢复输入", 3, "cyan") as bar:
        paths = [CFG.stage_detail_csv, CFG.sliding_detail_csv, CFG.sliding_summary_csv]
        rows = []
        for path in paths:
            rows.append({"路径": str(path), "是否存在": path.exists(), "文件大小字节": path.stat().st_size if path.exists() else None, "行数": len(pd.read_csv(path)) if path.exists() else None})
        bar.update()
        audit = pd.DataFrame(rows)
        bar.update()
        audit["用途"] = ["阶段 CBI 明细", "滑动窗口 CBI 明细", "滑动轨迹汇总"]
        bar.update()
    return audit


def draw_placeholder(audit: pd.DataFrame) -> plt.Figure:
    with progress_bar("步骤2/5 绘制Fig6占位草稿", 3, "yellow") as bar:
        fig, axes = plt.subplots(1, 3, figsize=(FP.fig_width, FP.fig_height))
        titles = ["A  Within-drought stages", "B  Post-drought recovery windows", "C  Sliding recovery trajectory"]
        notes = [
            "TODO: compare Early/Middle/Late Delta CBI.",
            "TODO: compare 1-7, 8-14, 15-30 day recovery windows.",
            "TODO: draw 7-day or 14-day sliding recovery curve.",
        ]
        bar.update()
        for ax, title, note in zip(axes, titles, notes):
            ax.set_axis_off()
            ax.text(0.05, 0.8, title, transform=ax.transAxes, fontsize=FP.text_size, fontweight="bold")
            ax.text(0.05, 0.6, note, transform=ax.transAxes, fontsize=9, va="top")
            ax.text(0.05, 0.35, f"Input rows available: {audit['行数'].fillna(0).astype(int).sum()}", transform=ax.transAxes, fontsize=8)
        bar.update()
        fig.suptitle("Optional recovery trajectories after extreme drought", fontsize=FP.title_size, fontweight="bold")
        bar.update()
    return fig


def parameters_to_table() -> pd.DataFrame:
    rows = []
    for name, value in asdict(CFG).items():
        rows.append({"参数组": "路径", "参数名": name, "当前值": str(value), "用途说明": "Fig6 输入输出路径。"})
    for name, value in asdict(FP).items():
        rows.append({"参数组": "图形参数", "参数名": name, "当前值": value, "用途说明": "控制 Fig6 占位图或后续恢复轨迹图外观。"})
    return pd.DataFrame(rows)


def write_outputs(fig: plt.Figure, audit: pd.DataFrame) -> pd.DataFrame:
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    with progress_bar("步骤3/5 保存Fig6输出", 5, "magenta") as bar:
        fig.savefig(CFG.output_dir / "图6_恢复过程图_占位草稿.png", dpi=CFG.figure_dpi)
        bar.update()
        fig.savefig(CFG.output_dir / "图6_恢复过程图_占位草稿.pdf")
        bar.update()
        audit.to_csv(CFG.output_dir / "图6_恢复过程输入审计表.csv", index=False, encoding="utf-8-sig")
        bar.update()
        parameters_to_table().to_csv(CFG.output_dir / "00_Fig6绘图参数说明表.csv", index=False, encoding="utf-8-sig")
        bar.update()
        summary = pd.DataFrame([{"状态": "恢复过程图脚手架已建立；是否进入正文待论文主线决定。", "可用输入文件数": int(audit["是否存在"].sum())}])
        summary.to_csv(CFG.output_dir / "图6_运行摘要表.csv", index=False, encoding="utf-8-sig")
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
