"""
Mô-đun tiền xử lý dữ liệu, làm sạch, mã hóa giao dịch và chia tập Train/Test.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from sklearn.model_selection import train_test_split

from .config import CATEGORICAL_MAP, NUMERIC_CONSTRAINTS, PURPOSE_ALIAS


def split_multi(value: Any, aliases: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Tách chuỗi chứa nhiều lựa chọn cách nhau bởi dấu phẩy và chuẩn hóa tên mục.
    """
    if pd.isna(value):
        return []
    items = {
        (aliases or {}).get(s.strip(), s.strip())
        for s in str(value).split(",")
        if s.strip()
    }
    return sorted(items)


def clean_data(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Làm sạch dữ liệu và thực hiện kiểm tra chất lượng (data audit).
    
    Returns:
        Tuple: (DataFrame đã làm sạch, Từ điển audit)
    """
    df = raw_df.copy()

    # Chuẩn hóa khoảng trắng cho các cột chuỗi
    for col in df.select_dtypes(include=["object", "string"]):
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x).replace("", np.nan)

    # Loại bỏ dòng trùng lặp toàn bộ
    duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # Loại bỏ các dòng trống toàn bộ hành vi dùng AI
    empty_behavior = df[["AI_Tools_Used", "Use_Cases"]].isna().all(axis=1)
    empty_removed = int(empty_behavior.sum())
    df = df.loc[~empty_behavior].reset_index(drop=True)

    # Kiểm tra miền giá trị số hợp lệ
    for col, low, high in NUMERIC_CONSTRAINTS:
        num = pd.to_numeric(df[col], errors="coerce")
        if (df[col].notna() & num.isna()).any() or (~num.dropna().between(low, high)).any():
            raise ValueError(f"Dữ liệu chứa giá trị không hợp lệ tại cột: {col}")
        df[col] = num

    # Kiểm tra công cụ ưa thích có nằm trong các công cụ đã sử dụng
    used_tools = df["AI_Tools_Used"].map(split_multi)
    preferred_not_in_used = sum(
        pd.notna(p) and p not in tools
        for p, tools in zip(df["Preferred_AI_Tool"], used_tools)
    )

    audit = {
        "raw_rows": len(raw_df),
        "raw_columns": len(raw_df.columns),
        "duplicates_removed": duplicates_removed,
        "empty_behavior_removed": empty_removed,
        "valid_rows": len(df),
        "state_missing": int(raw_df["State"].isna().sum()),
        "grade_outside_minus3_plus3": int((~df["Impact_on_Grades"].between(-3, 3)).sum()),
        "preferred_not_in_used": int(preferred_not_in_used)
    }

    return df, audit


def build_transactions(df: pd.DataFrame) -> List[List[str]]:
    """
    Chuyển đổi mỗi dòng dữ liệu thành một tập giao dịch (transaction) gồm các item nhị phân.
    """
    transactions: List[List[str]] = []

    for _, row in df.iterrows():
        items: List[str] = []

        # Nhóm công cụ AI đã dùng
        items.extend(f"Công cụ={tool}" for tool in split_multi(row["AI_Tools_Used"]))

        # Nhóm mục đích sử dụng AI
        items.extend(
            f"Mục đích={purpose}"
            for purpose in split_multi(row["Use_Cases"], PURPOSE_ALIAS)
        )

        # Các biến phân loại
        for col, prefix in CATEGORICAL_MAP.items():
            val = row[col]
            if pd.notna(val):
                val_str = str(int(val) if col == "Year_of_Study" else val)
                items.append(f"{prefix}={val_str}")

        # Rời rạc hóa biến thời gian sử dụng hàng ngày
        if pd.notna(row["Daily_Usage_Hours"]):
            h = row["Daily_Usage_Hours"]
            label = "Dưới 1h" if h < 1 else ("1 đến dưới 3h" if h < 3 else "Từ 3h")
            items.append(f"Giờ dùng={label}")

        # Rời rạc hóa mức độ tin tưởng
        if pd.notna(row["Trust_in_AI_Tools"]):
            t = row["Trust_in_AI_Tools"]
            label = "Thấp" if t <= 2 else ("Trung bình" if t == 3 else "Cao")
            items.append(f"Tin tưởng={label}")

        # Rời rạc hóa mức độ nhận biết
        if pd.notna(row["Awareness_Level"]):
            a = row["Awareness_Level"]
            label = "Thấp" if a <= 3 else ("Trung bình" if a <= 7 else "Cao")
            items.append(f"Nhận biết={label}")

        # Rời rạc hóa ảnh hưởng điểm số
        if pd.notna(row["Impact_on_Grades"]):
            g = row["Impact_on_Grades"]
            label = "Âm" if g < 0 else ("Bằng 0" if g == 0 else "Dương")
            items.append(f"Ảnh hưởng điểm={label}")

        transactions.append(sorted(set(items)))

    return transactions


def encode_and_split(
    transactions: List[List[str]],
    test_size: float = 0.30,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Mã hóa danh sách giao dịch sang ma trận One-Hot Boolean và chia Train/Test.
    
    Returns:
        Tuple: (X, train, test, train_idx, test_idx)
    """
    te = TransactionEncoder()
    matrix_arr = te.fit_transform(transactions)
    X = pd.DataFrame(matrix_arr, columns=te.columns_, dtype=bool)

    # Chia tập ngẫu nhiên trước khi khai phá
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)

    train = X.iloc[train_idx].copy()
    test = X.iloc[test_idx].copy()

    # Kiểm tra tính toàn vẹn của việc chia mẫu
    assert set(train_idx).isdisjoint(test_idx), "Tập khám phá và tập kiểm tra bị giao nhau!"
    assert len(train) + len(test) == len(X), "Tổng số dòng train + test không khớp với dữ liệu gốc!"
    assert X.dtypes.eq(bool).all(), "Ma trận giao dịch phải toàn bộ là kiểu boolean!"

    return X, train, test, train_idx, test_idx
