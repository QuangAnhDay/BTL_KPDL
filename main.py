"""
Chương trình chính thực hiện bài toán Khai phá Luật Kết Hợp (Association Rule Mining)
Dữ liệu: AI Tool Usage by Indian College Students 2025 (Kaggle).

Sử dụng:
    python main.py
    python main.py --help
    python main.py --min-support 0.05 --min-confidence 0.60 --min-lift 1.20
"""

import argparse
import io
from pathlib import Path
import sys

# Đảm bảo in tiếng Việt mượt mà trên console Windows mà không bị lỗi cp1252
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib.pyplot as plt

from src.config import MiningConfig, SENSITIVITY_SUPPORTS
from src.data_loader import load_dataset
from src.eda import plot_frequencies
from src.evaluation import validate_rules_on_test
from src.exporter import export_all_results
from src.mining import (
    format_readable_df,
    generate_association_rules,
    mine_frequent_itemsets,
    prune_redundant_rules,
)
from src.preprocessing import build_transactions, clean_data, encode_and_split
from src.sensitivity import run_performance_benchmark, run_sensitivity_analysis


def setup_matplotlib():
    """Cấu hình hiển thị font và độ phân giải hình vẽ."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.dpi": 120
    })


def parse_arguments() -> MiningConfig:
    """Xử lý các tham số dòng lệnh (CLI arguments)."""
    parser = argparse.ArgumentParser(
        description="Chương trình Khai phá Luật Kết Hợp (Association Rule Mining) với Apriori & FP-Growth"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Đường dẫn tới file Students.csv (mặc định tự động tìm trong data/ hoặc KhaiPhaDuLieu_AI/)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ket_qua"),
        help="Thư mục xuất kết quả (mặc định: ket_qua/)"
    )
    parser.add_argument(
        "--min-support",
        type=float,
        default=0.05,
        help="Ngưỡng hỗ trợ tối thiểu min_support (mặc định: 0.05)"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.60,
        help="Ngưỡng tin cậy tối thiểu min_confidence (mặc định: 0.60)"
    )
    parser.add_argument(
        "--min-lift",
        type=float,
        default=1.20,
        help="Ngưỡng nâng (lift) tối thiểu min_lift (mặc định: 1.20)"
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=3,
        help="Độ dài tối đa của tập mục phổ biến (mặc định: 3)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
        help="Tỷ lệ tập kiểm tra giữ lại (mặc định: 0.30)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed ngẫu nhiên để tái lập kết quả (mặc định: 42)"
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Bật hiển thị cửa sổ biểu đồ khi đang chạy"
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Không tạo tệp nén ket_qua_Kaggle.zip"
    )

    args = parser.parse_args()

    return MiningConfig(
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
        max_len=args.max_len,
        test_size=args.test_size,
        seed=args.seed,
        output_dir=args.output_dir,
        data_path=args.data,
        show_plots=args.show_plots,
        export_zip=not args.no_zip
    )


def main():
    print("=" * 80)
    print(" BÀI TẬP LỚN MÔN KHAI PHÁ DỮ LIỆU - ASSOCIATION RULE MINING")
    print(" Đề tài: Khai phá luật kết hợp trong hành vi dùng AI phục vụ học tập")
    print("=" * 80)

    setup_matplotlib()
    cfg = parse_arguments()

    print(f"\n[1/8] Đang tải dữ liệu...")
    raw_df, content_bytes, sha256_hash = load_dataset(cfg.data_path)

    print(f"\n[2/8] Đang làm sạch và kiểm tra chất lượng dữ liệu...")
    cleaned_df, audit = clean_data(raw_df)
    print(f"      - Tổng số bản ghi hợp lệ: {audit['valid_rows']:,}")
    print(f"      - Trùng lặp bị loại bỏ: {audit['duplicates_removed']}")
    print(f"      - Dòng trống hành vi bị loại: {audit['empty_behavior_removed']}")
    print(f"      - Giá trị State bị thiếu: {audit['state_missing']:,}")
    print(f"      - Ưa thích nhưng không nằm trong dùng: {audit['preferred_not_in_used']:,}")

    print(f"\n[3/8] Đang mã hóa giao dịch và chia tập Train/Test (seed={cfg.seed})...")
    transactions = build_transactions(cleaned_df)
    X, train, test, train_idx, test_idx = encode_and_split(
        transactions,
        test_size=cfg.test_size,
        seed=cfg.seed
    )
    print(f"      - Kích thước ma trận One-Hot: {X.shape[0]:,} dòng x {X.shape[1]} items")
    print(f"      - Tập khám phá (Train 70%): {len(train):,} dòng")
    print(f"      - Tập kiểm tra  (Test 30%): {len(test):,} dòng")

    print(f"\n[4/8] Đang phân tích mô tả và xuất biểu đồ phân bố...")
    counts = plot_frequencies(X, cfg.output_dir, show_plots=cfg.show_plots)

    print(f"\n[5/8] Đang khai phá tập mục phổ biến và sinh luật kết hợp...")
    fa_itemsets, ff_itemsets = mine_frequent_itemsets(
        train,
        min_support=cfg.min_support,
        max_len=cfg.max_len
    )
    print(f"      - Số tập mục phổ biến tìm được (Apriori & FP-Growth): {len(ff_itemsets):,}")
    print(f"      - Số hỗ trợ tối thiểu (min_count): {int(-(-cfg.min_support * len(train) // 1))}")

    rules = generate_association_rules(
        ff_itemsets,
        min_conf=cfg.min_confidence,
        min_lift=cfg.min_lift
    )
    concise_rules = prune_redundant_rules(
        rules,
        train,
        gain_threshold=cfg.confidence_gain_threshold
    )
    print(f"      - Số luật ban đầu thỏa mãn ngưỡng: {len(rules)}")
    print(f"      - Số luật sau khi lọc rút gọn (gain >= {cfg.confidence_gain_threshold}): {len(concise_rules)}")

    print(f"\n[6/8] Đang kiểm định trên tập kiểm tra giữ lại (Hold-out Test 30%)...")
    validation_df = validate_rules_on_test(
        concise_rules,
        train,
        test,
        min_support=cfg.min_support,
        min_conf=cfg.min_confidence,
        min_lift=cfg.min_lift
    )
    passed_count = int(validation_df["test_pass"].sum()) if not validation_df.empty else 0
    print(f"      - Số luật giữ vững ngưỡng trên tập test: {passed_count}/{len(concise_rules)}")

    if not validation_df.empty:
        print("\n--- Top 5 luật kết hợp tiêu biểu trên tập kiểm tra ---")
        top5 = validation_df.head(5)
        for r in top5.itertuples():
            print(
                f" * {r.A} => {r.B}\n"
                f"   [Train] Support: {r.train_support:.1%} | Conf: {r.train_confidence:.1%} | Lift: {r.train_lift:.2f}\n"
                f"   [Test ] Hỗ trợ: {r.test_n_AB}/{r.test_n_A} | Conf: {r.test_confidence:.1%} (CI 95%: [{r.test_ci_low:.1%}, {r.test_ci_high:.1%}]) | Lift: {r.test_lift:.2f}\n"
            )

    print(f"\n[7/8] Đang phân tích độ nhạy và đo chuẩn hiệu năng (Benchmark)...")
    sensitivity_df = run_sensitivity_analysis(
        train,
        support_levels=SENSITIVITY_SUPPORTS,
        min_conf=cfg.min_confidence,
        min_lift=cfg.min_lift,
        max_len=cfg.max_len,
        output_dir=cfg.output_dir,
        show_plots=cfg.show_plots
    )
    benchmark_df = run_performance_benchmark(
        train,
        min_support=cfg.min_support,
        max_len=cfg.max_len,
        n_runs=5
    )
    print("      - Kết quả so sánh thời gian thực thi (5 lần đo):")
    for b in benchmark_df.itertuples():
        print(f"        * {b.algorithm:10s}: Trung vị {b.median_ms:8.2f} ms (Min: {b.min_ms:8.2f} ms, Max: {b.max_ms:8.2f} ms)")

    print(f"\n[8/8] Đang xuất toàn bộ kết quả ra thư mục: {cfg.output_dir.resolve()}...")
    zip_path = export_all_results(
        output_dir=cfg.output_dir,
        cleaned_df=cleaned_df,
        X=X,
        train_idx=train_idx,
        ff_itemsets=ff_itemsets,
        rules=rules,
        concise_rules=concise_rules,
        validation_df=validation_df,
        sensitivity_df=sensitivity_df,
        benchmark_df=benchmark_df,
        audit=audit,
        sha256_hash=sha256_hash,
        counts=counts,
        export_zip=cfg.export_zip
    )

    print("\n" + "=" * 80)
    print(" HOÀN THÀNH QUY TRÌNH KHAI PHÁ THÀNH CÔNG!")
    print(f" Tất cả kết quả đã được lưu tại thư mục: {cfg.output_dir.resolve()}")
    if cfg.export_zip:
        print(f" File nén đóng gói đầy đủ: {zip_path.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
