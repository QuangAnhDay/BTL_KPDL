"""
Mô-đun tải dữ liệu, kiểm tra toàn vẹn và xác thực cấu trúc bảng.
"""

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from .config import KAGGLE_URL, REQUIRED_COLUMNS


def find_dataset_file(preferred_path: Optional[Path] = None) -> Optional[Path]:
    """Tìm kiếm file Students.csv trong các vị trí khả dĩ."""
    candidates = []
    if preferred_path:
        candidates.append(Path(preferred_path))
    
    # Các vị trí mặc định phổ biến
    base_dir = Path(__file__).resolve().parent.parent
    candidates.extend([
        base_dir / "data" / "Students.csv",
        base_dir / "Students.csv",
        base_dir / "KhaiPhaDuLieu_AI" / "Students.csv",
        Path("Students.csv"),
        Path("data/Students.csv"),
        Path("KhaiPhaDuLieu_AI/Students.csv")
    ])
    
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_dataset(preferred_path: Optional[Path] = None) -> Tuple[pd.DataFrame, bytes, str]:
    """
    Nạp dữ liệu từ file cục bộ hoặc tải dự phòng từ Kaggle.
    
    Returns:
        Tuple: (DataFrame dữ liệu thô, bytes nội dung file gốc, mã SHA256)
    """
    file_path = find_dataset_file(preferred_path)
    content: Optional[bytes] = None

    if file_path and file_path.is_file():
        print(f"[+] Tìm thấy tệp dữ liệu tại: {file_path}")
        content = file_path.read_bytes()
    else:
        print("[!] Không tìm thấy file cục bộ, đang thử tải tự động từ Kaggle...")
        try:
            req = urllib.request.Request(
                KAGGLE_URL,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            blob = urllib.request.urlopen(req, timeout=60).read()
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                content = z.read("Students.csv")
            print("[+] Tải dữ liệu từ Kaggle thành công!")
        except Exception as exc:
            raise FileNotFoundError(
                "Không tìm thấy file Students.csv và không thể tải tự động từ Kaggle. "
                "Vui lòng đảm bảo Students.csv nằm trong thư mục data/ hoặc KhaiPhaDuLieu_AI/."
            ) from exc

    sha256_hash = hashlib.sha256(content).hexdigest()
    raw_df = pd.read_csv(io.BytesIO(content))

    # Kiểm tra các cột bắt buộc
    missing_cols = set(REQUIRED_COLUMNS) - set(raw_df.columns)
    if missing_cols:
        raise ValueError(f"Dữ liệu CSV thiếu các cột bắt buộc: {missing_cols}")

    print(f"[+] Nạp thành công: {len(raw_df):,} dòng, {len(raw_df.columns)} cột | SHA256: {sha256_hash}")
    return raw_df, content, sha256_hash
