"""
Mô-đun khai phá tập mục phổ biến (Apriori & FP-Growth) và sinh luật kết hợp.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth


def mine_frequent_itemsets(
    train_df: pd.DataFrame,
    min_support: float = 0.05,
    max_len: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Khai phá tập mục phổ biến bằng cả hai thuật toán Apriori và FP-Growth.
    Kiểm tra đối chiếu tính tương đương giữa kết quả của hai thuật toán.
    
    Returns:
        Tuple: (DataFrame kết quả Apriori, DataFrame kết quả FP-Growth)
    """
    fa = apriori(train_df, min_support=min_support, use_colnames=True, max_len=max_len)
    ff = fpgrowth(train_df, min_support=min_support, use_colnames=True, max_len=max_len)

    # Đối chiếu tính chính xác toán học giữa Apriori và FP-Growth
    smap_fa = {r.itemsets: r.support for r in fa.itertuples()}
    smap_ff = {r.itemsets: r.support for r in ff.itertuples()}

    assert smap_fa.keys() == smap_ff.keys(), (
        "Tập mục phổ biến sinh ra bởi Apriori và FP-Growth không đồng nhất!"
    )
    assert all(np.isclose(smap_fa[k], smap_ff[k]) for k in smap_fa), (
        "Độ hỗ trợ (support) giữa Apriori và FP-Growth có sai số vượt ngưỡng!"
    )

    return fa, ff


def generate_association_rules(
    frequent_itemsets: pd.DataFrame,
    min_conf: float = 0.60,
    min_lift: float = 1.20
) -> pd.DataFrame:
    """
    Sinh luật kết hợp từ tập mục phổ biến, lọc các luật có vế phải 1 item,
    lift >= min_lift và liên quan đến chủ đề hành vi (Công cụ hoặc Mục đích).
    """
    if frequent_itemsets.empty:
        return pd.DataFrame()

    raw_rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_conf
    )

    if raw_rules.empty:
        return pd.DataFrame()

    def is_relevant(row) -> bool:
        items = row.antecedents | row.consequents
        return any(v.startswith(("Công cụ=", "Mục đích=")) for v in items)

    filtered = raw_rules[
        (raw_rules.consequents.map(len) == 1) &
        (raw_rules.lift >= min_lift) &
        raw_rules.apply(is_relevant, axis=1)
    ].copy()

    return filtered.reset_index(drop=True)


def prune_redundant_rules(
    rules: pd.DataFrame,
    train_df: pd.DataFrame,
    gain_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Lọc bỏ các luật dư thừa: Với vế trái > 1 item, chỉ giữ lại nếu độ tin cậy
    tăng ít nhất `gain_threshold` so với luật con đơn giản hơn có độ tin cậy lớn nhất.
    """
    if rules.empty:
        return rules.copy()

    gains = []
    for r in rules.itertuples():
        if len(r.antecedents) == 1:
            gains.append(np.nan)
            continue

        conf_subsets = []
        for item in r.antecedents:
            subset_items = list(r.antecedents - {item})
            mask = train_df[subset_items].all(axis=1)
            consequent_items = list(r.consequents)
            sub_conf = train_df.loc[mask, consequent_items].all(axis=1).mean()
            conf_subsets.append(sub_conf)

        max_sub_conf = max(conf_subsets) if conf_subsets else 0.0
        gains.append(r.confidence - max_sub_conf)

    rules_with_gain = rules.copy()
    rules_with_gain["confidence_gain"] = gains

    concise = rules_with_gain[
        rules_with_gain.confidence_gain.isna() |
        (rules_with_gain.confidence_gain >= gain_threshold)
    ].sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)

    return concise


def format_readable_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuyển đổi frozenset trong các cột itemsets, antecedents, consequents
    thành chuỗi text thân thiện để hiển thị và lưu CSV.
    """
    formatted = df.copy()
    for col in ["itemsets", "antecedents", "consequents"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(lambda s: " & ".join(sorted(s)))
    return formatted
