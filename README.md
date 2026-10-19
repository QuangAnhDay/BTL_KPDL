# Khai Phá Luật Kết Hợp (Association Rule Mining)
## Đề tài: Phân tích hành vi sử dụng công cụ AI phục vụ học tập của sinh viên

Dự án này là bài tập lớn môn **Khai phá dữ liệu (Data Mining)**, chuyển thể từ phiên bản thực nghiệm trên Google Colab thành một chương trình Python hoàn chỉnh, mô-đun hóa, chạy độc lập trên máy cục bộ (Local) và sẵn sàng đưa lên GitHub.

---

## 📌 1. Giới thiệu đề tài & Bộ dữ liệu

- **Bộ dữ liệu:** [AI Tool Usage by Indian College Students 2025](https://www.kaggle.com/datasets/rakeshkapilavai/ai-tool-usage-by-indian-college-students-2025) trên Kaggle.
- **Tác giả dữ liệu:** Rakesh Kapilavayi (Giấy phép: CC BY-NC-SA 4.0).
- **Quy mô:** 3.614 bản ghi khảo sát, 16 trường thông tin.
- **Mục tiêu nghiên cứu:** Khai phá các quy luật kết hợp giữa các công cụ AI (ChatGPT, Copilot, Gemini, Midjourney,...), mục đích học tập (Coding Help, Assignments, Projects, Exam Prep, Notes,...), mức độ tin tưởng, thời gian dùng và ngành học của sinh viên.
- **Thuật toán sử dụng:** **Apriori** và **FP-Growth** kết hợp kiểm định chéo trên tập kiểm tra giữ lại (Hold-out Test 30%) và khoảng tin cậy Wilson 95%.

---

## 📂 2. Cấu trúc thư mục dự án

```text
BTL_KPDL/
├── data/
│   └── Students.csv               # Tệp dữ liệu khảo sát gốc
├── ket_qua/                       # Thư mục lưu kết quả xuất ra (tự động tạo)
│   ├── du_lieu_da_chon.csv        # Dữ liệu sạch đã bỏ định danh
│   ├── ma_tran_giao_dich.csv      # Ma trận One-Hot Boolean (3.614 dòng x 59 cột)
│   ├── chia_mau.csv               # Danh sách phân chia Train (70%) và Test (30%)
│   ├── tu_dien_item.csv           # Danh mục 59 item mã hóa
│   ├── cong_cu.png                # Biểu đồ tần suất công cụ AI
│   ├── muc_dich.png               # Biểu đồ tần suất mục đích sử dụng AI
│   ├── tap_muc_pho_bien.csv       # 2.122 tập mục phổ biến tìm được
│   ├── luat_ban_dau.csv           # 166 luật ban đầu thỏa mãn ngưỡng
│   ├── luat_rut_gon.csv           # 49 luật sau khi lọc bỏ dư thừa
│   ├── luat_va_kiem_tra.csv       # Kết quả kiểm định trên tập test & Wilson CI 95%
│   ├── do_nhay.csv & do_nhay.png  # Phân tích độ nhạy theo min_support
│   ├── thoi_gian.csv              # Đo lường thời gian chạy Apriori vs FP-Growth
│   ├── tom_tat.json               # Tổng kết các chỉ số quan trọng dưới dạng JSON
│   └── ket_qua_Kaggle.zip         # Gói nén chứa toàn bộ kết quả phân tích
├── src/                           # Mã nguồn mô-đun hóa
│   ├── __init__.py
│   ├── config.py                  # Tham số cấu hình, ràng buộc miền giá trị và hằng số
│   ├── data_loader.py             # Nạp dữ liệu, kiểm tra SHA-256, tải dự phòng từ Kaggle
│   ├── preprocessing.py           # Làm sạch dữ liệu, mã hóa One-Hot, chia Train/Test
│   ├── eda.py                     # Thống kê mô tả và vẽ biểu đồ tần số
│   ├── mining.py                  # Apriori, FP-Growth, sinh luật và lọc rút gọn
│   ├── evaluation.py              # Kiểm định luật trên tập kiểm tra & khoảng tin cậy Wilson
│   ├── sensitivity.py             # Phân tích độ nhạy và benchmark hiệu năng
│   └── exporter.py                # Xuất các file CSV, JSON và file nén ZIP
├── main.py                        # Điểm khởi chạy chính kèm giao diện dòng lệnh (CLI)
├── run_analysis.py                # Script chạy nhanh 1 click
├── requirements.txt               # Danh sách thư viện phụ thuộc
├── .gitignore                     # Cấu hình bỏ qua các file tạm/cache khi đưa lên Git
└── README.md                      # Tài liệu hướng dẫn sử dụng chi tiết
```

---

## ⚙️ 3. Cài đặt môi trường

Chương trình yêu cầu **Python 3.10 trở lên** (khuyến nghị Python 3.11 hoặc 3.12).

### Bước 1: Mở Terminal / PowerShell tại thư mục dự án
```bash
cd "F:\OneDrive\Đính kèm\hoc hanh\khaiphadl\BTL_KPDL"
```

### Bước 2: Tạo môi trường ảo (tùy chọn nhưng khuyến nghị)
```bash
python -m venv .venv
# Trên Windows:
.venv\Scripts\activate
# Trên Linux/macOS:
source .venv/bin/activate
```

### Bước 3: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

---

## 🚀 4. Hướng dẫn chạy chương trình

### Cách 1: Chạy nhanh bằng cấu hình chuẩn
Chỉ cần thực thi lệnh sau:
```bash
python run_analysis.py
```
hoặc:
```bash
python main.py
```

### Cách 2: Tùy chỉnh các tham số qua dòng lệnh (CLI)
Chương trình hỗ trợ cấu hình linh hoạt mọi siêu tham số:
```bash
python main.py --min-support 0.05 --min-confidence 0.60 --min-lift 1.20 --max-len 3
```

Các tùy chọn tham số:
- `--data <path>`: Đường dẫn tới file `Students.csv` nếu để ở vị trí khác (mặc định tự tìm trong `data/` hoặc `KhaiPhaDuLieu_AI/`).
- `--output-dir <path>`: Thư mục lưu kết quả (mặc định: `ket_qua`).
- `--min-support <float>`: Độ hỗ trợ tối thiểu (mặc định: `0.05` tức 5%).
- `--min-confidence <float>`: Độ tin cậy tối thiểu (mặc định: `0.60` tức 60%).
- `--min-lift <float>`: Độ nâng tối thiểu (mặc định: `1.20`).
- `--max-len <int>`: Độ dài tối đa của tập mục (mặc định: `3`).
- `--test-size <float>`: Tỷ lệ tập kiểm tra giữ lại (mặc định: `0.30` tức 30%).
- `--seed <int>`: Random state để tái lập kết quả (mặc định: `42`).
- `--show-plots`: Hiển thị các cửa sổ biểu đồ khi chạy (mặc định tắt để chạy nền nhanh).
- `--no-zip`: Không đóng gói file nén `ket_qua_Kaggle.zip`.

Xem chi tiết trợ giúp:
```bash
python main.py --help
```

---

## 🔬 5. Quy trình thực nghiệm (8 bước)

1. **Nạp dữ liệu & Bảo toàn bản gốc:** Kiểm tra mã băm SHA-256 (`78444e3e...`) và xác thực 16 cột bắt buộc.
2. **Làm sạch & Kiểm tra chất lượng:**
   - Xóa bỏ các định danh nhạy cảm (`Student_Name`, `College_Name`).
   - Xử lý cột `State` (bị thiếu 1.614 dòng, không dùng để khai phá).
   - Chuẩn hóa các mục chọn nhiều lựa chọn (ví dụ gộp `Exam Preparation` thành `Exam Prep`).
3. **Mã hóa giao dịch:** Rời rạc hóa các thuộc tính liên tục (giờ dùng, tin tưởng, nhận biết, ảnh hưởng điểm) và tạo ma trận One-Hot Boolean qua `TransactionEncoder`.
4. **Chia mẫu trước khi khai phá:** Phân chia tập Khám phá 70% (2.529 dòng) và tập Kiểm tra độc lập 30% (1.085 dòng) với seed 42.
5. **Khai phá trên tập khám phá:**
   - Chạy song song cả hai thuật toán **Apriori** và **FP-Growth**, xác minh kết quả giống nhau 100%.
   - Lọc bỏ các luật dư thừa bằng tiêu chí **mức tăng độ tin cậy (Confidence Gain $\ge 0.05$)**.
6. **Kiểm định chéo (Internal Hold-out Validation):**
   - Đánh giá lại từng luật trên 30% tập kiểm tra chưa từng dùng để chọn luật.
   - Tính khoảng tin cậy Wilson 95% cho Confidence.
7. **Phân tích độ nhạy & So sánh hiệu năng (Benchmark):**
   - Đánh giá sự biến thiên số lượng luật qua các mức support: 3%, 5%, 8%, 10%, 15%.
   - Đo thời gian thực thi trung vị qua 5 lượt chạy xen kẽ giữa Apriori và FP-Growth.
8. **Đóng gói & Báo cáo:** Xuất toàn bộ 10 file CSV, 3 biểu đồ trực quan, file JSON tổng kết và file ZIP hoàn chỉnh.

---

## 📊 6. Kết quả nổi bật

- **Tập mục phổ biến:** 2.122 tập mục ($support \ge 5\%$).
- **Luật ban đầu:** 166 luật ($conf \ge 60\%, lift \ge 1.20$).
- **Luật rút gọn:** 49 luật (sau khi loại trừ luật dư thừa).
- **Luật bền vững trên tập kiểm tra:** 40 / 49 luật (81.6%) vẫn vượt qua cả 3 ngưỡng $support, confidence, lift$.

### Một số luật tiêu biểu tìm được:
1. $\{Công cụ=ChatGPT, Mục đích=Exam Prep\} \Rightarrow \{Mục đích=Notes\}$
   - Khám phá: Support 7.7%, Confidence 89.4%, Lift 7.25.
   - Kiểm định: Confidence 89.1% (Wilson 95%: [81.1%, 94.0%]), Lift 6.96.
2. $\{Công cụ=Gemini, Mục đích=Exam Prep\} \Rightarrow \{Mục đích=Notes\}$
   - Khám phá: Support 5.5%, Confidence 85.2%, Lift 6.90.
   - Kiểm định: Confidence 85.1% (Wilson 95%: [75.3%, 91.5%]), Lift 6.65.
3. $\{Mục đích=Exam Prep\} \Rightarrow \{Mục đích=Notes\}$
   - Khám phá: Support 12.3%, Confidence 69.6%, Lift 5.65.
   - Kiểm định: Confidence 67.8% (Wilson 95%: [61.1%, 73.8%]), Lift 5.29.

---

## 🐙 7. Hướng dẫn đưa lên GitHub (Khi bạn đã tạo Repo)

Khi bạn đã sẵn sàng và tạo xong một repository mới trên GitHub (ví dụ: `https://github.com/<username>/BTL_KhaiPhaDuLieu.git`), hãy mở terminal tại thư mục này và chạy lần lượt các lệnh sau:

```bash
# 1. Khởi tạo Git repository cục bộ
git init

# 2. Thêm tất cả các file mã nguồn (file .gitignore sẽ tự động bỏ qua rác/cache)
git add .

# 3. Tạo commit đầu tiên
git commit -m "feat: Hoan thien chuong trinh khai pha luat ket hop (Apriori & FP-Growth)"

# 4. Đổi nhánh chính thành main
git branch -M main

# 5. Liên kết với kho lưu trữ từ xa trên GitHub (thay URL của bạn vào)
git remote add origin https://github.com/<tai-khoan-cua-ban>/<ten-repo>.git

# 6. Đẩy mã nguồn lên GitHub
git push -u origin main
```

---

## ⚖️ 8. Đạo đức nghiên cứu & Giới hạn dữ liệu

- Dữ liệu thu thập từ sinh viên tại Ấn Độ, không đại diện trực tiếp cho sinh viên Việt Nam.
- Kết quả phản ánh tương quan thống kê trong mẫu khảo sát công khai, không khẳng định quan hệ nhân quả.
- Đã loại bỏ các cột định danh cá nhân trước khi xử lý nhằm bảo vệ tính riêng tư.
