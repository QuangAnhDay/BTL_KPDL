"""
Mô-đun đánh giá và kiểm định luật kết hợp trên tập kiểm tra độc lập (Hold-out test).
"""

from typing import Any, Dict, Set, Tuple
import numpy as np
import pandas as pd


def compute_metrics(frame: pd.DataFrame, A: Set[str], B: Set[str]) -> Dict[str, Any]:
    """
    Tính trực tiếp các chỉ số support, confidence và lift của luật A => B trên tập dữ liệu.
    """
    a_vec = frame[list(A)].all(axis=1)
    b_vec = frame[list(B)].all(axis=1)
    n = len(frame)
    na = int(a_vec.sum())
    nb = int(b_vec.sum())
    nab = int((a_vec & b_vec).sum())
    conf = nab / na if na > 0 else np.nan
    lift = conf / (nb / n) if nb > 0 and pd.notna(conf) else np.nan

    return {
        "n_A": na,
        "n_AB": nab,
        "support": nab / n,
        "confidence": conf,
        "lift": lift
    }


def wilson_score_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Tính khoảng tin cậy Wilson 95% cho tỷ lệ k/n (áp dụng cho confidence trên tập test).
    """
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + (z * z / n)
    mid = (p + (z * z / (2 * n))) / denom
    half = (z * np.sqrt((p * (1 - p) / n) + ((z * z) / (4 * n * n)))) / denom
    return mid - half, mid + half


def validate_rules_on_test(
    concise_rules: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    min_support: float = 0.05,
    min_conf: float = 0.60,
    min_lift: float = 1.20
) -> pd.DataFrame:
    """
    Kiểm định từng luật rút gọn trên tập kiểm tra (test set 30%), tính độ suy giảm
    và khoảng tin cậy Wilson.
    """
    if concise_rules.empty:
        return pd.DataFrame()

    rows = []
    for r in concise_rules.itertuples():
        tr = compute_metrics(train_df, r.antecedents, r.consequents)
        ho = compute_metrics(test_df, r.antecedents, r.consequents)

        # Đảm bảo chỉ số đếm trực tiếp trên train khớp hoàn hảo với mlxtend
        assert np.isclose(tr["support"], r.support) and \
               np.isclose(tr["confidence"], r.confidence) and \
               np.isclose(tr["lift"], r.lift), "Chỉ số trên tập train không khớp với luật gốc!"

        lo, hi = wilson_score_interval(ho["n_AB"], ho["n_A"])
        test_pass = bool(
            ho["support"] >= min_support and
            ho["confidence"] >= min_conf and
            ho["lift"] >= min_lift
        )

        row_data = {
            "A": " & ".join(sorted(r.antecedents)),
            "B": " & ".join(sorted(r.consequents)),
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"test_{k}": v for k, v in ho.items()},
            "test_ci_low": lo,
            "test_ci_high": hi,
            "test_pass": test_pass
        }
        rows.append(row_data)

    validation_df = pd.DataFrame(rows)
    return validation_df
