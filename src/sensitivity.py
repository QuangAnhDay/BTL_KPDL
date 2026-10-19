"""
Mô-đun phân tích độ nhạy (Sensitivity Analysis) và so sánh hiệu năng (Benchmark).
"""

import time
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth

from .mining import generate_association_rules


def run_sensitivity_analysis(
    train_df: pd.DataFrame,
    support_levels: List[float],
    min_conf: float = 0.60,
    min_lift: float = 1.20,
    max_len: int = 3,
    output_dir: Path = Path("ket_qua"),
    show_plots: bool = False
) -> pd.DataFrame:
    """
    Thử nghiệm các mức support khác nhau để kiểm tra độ nhạy của số lượng luật và tập mục.
    """
    sensitivity_records = []
    n_train = len(train_df)

    for ms in support_levels:
        f = fpgrowth(train_df, min_support=ms, use_colnames=True, max_len=max_len)
        r = generate_association_rules(f, min_conf=min_conf, min_lift=min_lift)
        sensitivity_records.append({
            "min_support": ms,
            "min_count": int(np.ceil(ms * n_train)),
            "itemsets": len(f),
            "rules": len(r)
        })

    sensitivity_df = pd.DataFrame(sensitivity_records)

    # Vẽ đồ thị độ nhạy
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        sensitivity_df["min_support"] * 100,
        sensitivity_df["rules"],
        marker="o",
        color="#157F86",
        linewidth=2,
        markersize=6
    )
    ax.set(
        xlabel="Support tối thiểu (%)",
        ylabel="Số luật",
        title="Độ nhạy theo ngưỡng support",
        ylim=(0, None)
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    fig_path = output_dir / "do_nhay.png"
    plt.savefig(fig_path, dpi=150)
    print(f"[+] Đã lưu biểu đồ độ nhạy: {fig_path}")

    if show_plots:
        plt.show()
    plt.close(fig)

    return sensitivity_df


def run_performance_benchmark(
    train_df: pd.DataFrame,
    min_support: float = 0.05,
    max_len: int = 3,
    n_runs: int = 5
) -> pd.DataFrame:
    """
    Đo thời gian thực thi (ms) trung vị qua các lượt xen kẽ giữa Apriori và FP-Growth.
    """
    timing = {"Apriori": [], "FP-Growth": []}

    for i in range(n_runs):
        methods = [("Apriori", apriori), ("FP-Growth", fpgrowth)]
        order = methods if (i % 2 == 0) else methods[::-1]

        for name, fn in order:
            start_time = time.perf_counter()
            fn(train_df, min_support=min_support, use_colnames=True, max_len=max_len)
            duration_ms = (time.perf_counter() - start_time) * 1000
            timing[name].append(duration_ms)

    benchmark_rows = []
    for algo, times in timing.items():
        benchmark_rows.append({
            "algorithm": algo,
            "median_ms": float(np.median(times)),
            "min_ms": float(min(times)),
            "max_ms": float(max(times))
        })

    benchmark_df = pd.DataFrame(benchmark_rows)
    return benchmark_df
