"""
Mô-đun xuất báo cáo kết quả, tệp dữ liệu CSV, JSON tóm tắt và nén file ZIP.
"""

import json
from pathlib import Path
import platform
from typing import Any, Dict
import zipfile
import numpy as np
import pandas as pd

from .mining import format_readable_df


def export_all_results(
    output_dir: Path,
    cleaned_df: pd.DataFrame,
    X: pd.DataFrame,
    train_idx: np.ndarray,
    ff_itemsets: pd.DataFrame,
    rules: pd.DataFrame,
    concise_rules: pd.DataFrame,
    validation_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    audit: Dict[str, Any],
    sha256_hash: str,
    counts: pd.Series,
    export_zip: bool = True
) -> Path:
    """
    Xuất toàn bộ kết quả phân tích ra thư mục chỉ định và tùy chọn đóng gói ZIP.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dữ liệu đã chọn (loại bỏ thông tin định danh và State bị thiếu nhiều)
    selected_cols_df = cleaned_df.drop(
        columns=["Student_Name", "College_Name", "State"],
        errors="ignore"
    )
    selected_cols_df.to_csv(
        output_dir / "du_lieu_da_chon.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 2. Ma trận giao dịch nhị phân
    X.to_csv(output_dir / "ma_tran_giao_dich.csv", index=False, encoding="utf-8-sig")

    # 3. Bản phân chia mẫu
    split_col = np.where(np.isin(np.arange(len(X)), train_idx), "kham_pha", "kiem_tra")
    pd.DataFrame({
        "row_index": np.arange(len(X)),
        "split": split_col
    }).to_csv(output_dir / "chia_mau.csv", index=False)

    # 4. Từ điển item
    pd.DataFrame({"item": X.columns}).to_csv(
        output_dir / "tu_dien_item.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 5. Tập mục phổ biến
    format_readable_df(ff_itemsets).to_csv(
        output_dir / "tap_muc_pho_bien.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 6. Luật ban đầu
    format_readable_df(rules).to_csv(
        output_dir / "luat_ban_dau.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 7. Luật rút gọn
    format_readable_df(concise_rules).to_csv(
        output_dir / "luat_rut_gon.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 8. Luật và kiểm tra
    validation_df.to_csv(
        output_dir / "luat_va_kiem_tra.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 9. Độ nhạy
    sensitivity_df.to_csv(output_dir / "do_nhay.csv", index=False)

    # 10. Thời gian benchmark
    benchmark_df.to_csv(output_dir / "thoi_gian.csv", index=False)

    # 11. Tóm tắt JSON
    summary_data = {
        **audit,
        "items": int(X.shape[1]),
        "train_rows": len(train_idx),
        "test_rows": len(X) - len(train_idx),
        "itemsets": len(ff_itemsets),
        "rules": len(rules),
        "concise_rules": len(concise_rules),
        "test_pass": int(validation_df["test_pass"].sum()) if not validation_df.empty else 0,
        "sha256": sha256_hash,
        "python": platform.python_version(),
        "tool_counts": {
            k: int(v) for k, v in counts.items() if k.startswith("Công cụ=")
        },
        "purpose_counts": {
            k: int(v) for k, v in counts.items() if k.startswith("Mục đích=")
        }
    }

    summary_file = output_dir / "tom_tat.json"
    summary_file.write_text(
        json.dumps(summary_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"[+] Đã xuất tóm tắt JSON: {summary_file}")

    # 12. Đóng gói ZIP
    zip_path = output_dir / "ket_qua_Kaggle.zip"
    if export_zip:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for item in output_dir.glob("*"):
                if item.name != "ket_qua_Kaggle.zip" and item.is_file():
                    z.write(item, arcname=item.name)
        print(f"[+] Đã đóng gói toàn bộ kết quả vào: {zip_path}")

    return zip_path
