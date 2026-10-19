"""
Cấu hình và các hằng số dùng chung trong toàn bộ pipeline khai phá luật kết hợp.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class MiningConfig:
    """Tham số cấu hình thuật toán và đường dẫn lưu trữ."""
    min_support: float = 0.05
    min_confidence: float = 0.60
    min_lift: float = 1.20
    max_len: int = 3
    test_size: float = 0.30
    seed: int = 42
    confidence_gain_threshold: float = 0.05
    output_dir: Path = field(default_factory=lambda: Path("ket_qua"))
    data_path: Optional[Path] = None
    show_plots: bool = False
    export_zip: bool = True


# Danh sách các cột bắt buộc trong tập dữ liệu Kaggle
REQUIRED_COLUMNS: List[str] = [
    "Student_Name", "College_Name", "Stream", "Year_of_Study",
    "AI_Tools_Used", "Daily_Usage_Hours", "Use_Cases", "Trust_in_AI_Tools",
    "Impact_on_Grades", "Do_Professors_Allow_Use", "Preferred_AI_Tool",
    "Awareness_Level", "Willing_to_Pay_for_Access", "State",
    "Device_Used", "Internet_Access"
]

# Chuẩn hóa tên mục đích dùng AI
PURPOSE_ALIAS: Dict[str, str] = {
    "Exam Preparation": "Exam Prep",
    "Project Work": "Projects"
}

# Tiền tố mã hóa giao dịch cho các biến danh mục
CATEGORICAL_MAP: Dict[str, str] = {
    "Stream": "Ngành",
    "Year_of_Study": "Năm học",
    "Do_Professors_Allow_Use": "Cho phép",
    "Preferred_AI_Tool": "Ưa thích",
    "Willing_to_Pay_for_Access": "Trả phí",
    "Device_Used": "Thiết bị",
    "Internet_Access": "Internet"
}

# Ràng buộc miền giá trị hợp lệ cho các biến số
NUMERIC_CONSTRAINTS: List[Tuple[str, float, float]] = [
    ("Year_of_Study", 1, 4),
    ("Daily_Usage_Hours", 0, 24),
    ("Trust_in_AI_Tools", 1, 5),
    ("Awareness_Level", 1, 10),
    ("Impact_on_Grades", -5, 5)
]

# URL tải dự phòng từ Kaggle
KAGGLE_URL: str = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "rakeshkapilavai/ai-tool-usage-by-indian-college-students-2025"
)

# Các ngưỡng support trong phân tích độ nhạy
SENSITIVITY_SUPPORTS: List[float] = [0.03, 0.05, 0.08, 0.10, 0.15]
