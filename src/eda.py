"""
Mô-đun phân tích mô tả (EDA) và trực quan hóa phân bố công cụ, mục đích AI.
"""

from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import pandas as pd


def plot_frequencies(
    X: pd.DataFrame,
    output_dir: Path,
    show_plots: bool = False
) -> pd.Series:
    """
    Thống kê tần số xuất hiện của từng item và vẽ biểu đồ thanh ngang
    cho Công cụ AI và Mục đích sử dụng.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = X.sum().sort_values(ascending=False)

    charts = [
        ("Công cụ=", "cong_cu", "Công cụ AI trong tệp dữ liệu"),
        ("Mục đích=", "muc_dich", "Mục đích sử dụng AI trong tệp dữ liệu")
    ]

    for prefix, filename, title in charts:
        subset = counts[counts.index.str.startswith(prefix)].sort_values()
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [k.split("=", 1)[1] for k in subset.index]
        ax.barh(labels, subset.values, color="#157F86")

        max_val = subset.max()
        for idx, val in enumerate(subset):
            pct_str = f"{val:,} ({val / len(X):.1%})"
            ax.text(val + (len(X) * 0.008), idx, pct_str, va="center", fontsize=9)

        ax.set_xlabel("Số bản ghi", fontsize=10)
        ax.set_title(f"{title} (n = {len(X):,})", fontsize=12, pad=12)
        ax.set_xlim(0, max_val * 1.28)
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        save_path = output_dir / f"{filename}.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[+] Đã lưu biểu đồ: {save_path}")

        if show_plots:
            plt.show()
        plt.close(fig)

    return counts
