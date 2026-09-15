---

**Role:** Senior Machine Learning Engineer & Python OOP Expert
**Objective:** Triển khai một thư viện Machine Learning hướng đối tượng (OOP) từ đầu (from scratch) để xây dựng mô hình Gaussian Naive Bayes phân loại hành vi mua hàng của khách hàng.

**Core Constraint:** TUYỆT ĐỐI KHÔNG SỬ DỤNG thư viện `scikit-learn` (`sklearn`). Tất cả các bước từ tiền xử lý, thuật toán, kiểm định chéo đến đánh giá đều phải tự code 100% bằng Python thuần, `numpy` và `pandas`.

**References:**

1. Tham khảo các yêu cầu kỹ thuật và công thức toán học trong file `naive-bayes-sdd-spec-v2.md`.


2. Dữ liệu đầu vào: `customer_purchase_data.csv`.

---

### 1. KIẾN TRÚC THƯ MỤC DỰ ÁN

Bạn hãy sinh ra mã nguồn tuân thủ chính xác cấu trúc project sau:

```text
ML_NaiveBayes/
├── data/
│   └── customer_purchase_data.csv        # Bộ dữ liệu gốc
├── src/
│   ├── __init__.py
│   ├── tien_xu_ly.py               # Split, imputer, encoder, IQR, scaler, oversampling
│   ├── thuat_toan_nb.py            # CustomGaussianNB và var_smoothing
│   └── danh_gia.py                 # K-Fold, evaluator và CustomGridSearchCV
├── chay_thu_nghiem.py              # File chính chạy toàn bộ pipeline
└── README.md                       # Hướng dẫn chạy code và giải thích pipeline

```

### 2. QUY TRÌNH PIPELINE 6 BƯỚC BẮT BUỘC

File `chay_thu_nghiem.py` phải thực thi một luồng pipeline hoàn chỉnh tuân thủ nghiêm ngặt 6 nguyên tắc sau để chống rò rỉ dữ liệu (Data Leakage):

1. **Thu thập & Tải dữ liệu thô (Raw Data Ingestion):** Tải tập dữ liệu gốc vào Pandas DataFrame. Tuyệt đối chưa thực hiện bất kỳ phép biến đổi toán học hay tính toán thống kê nào ở bước này.
2. **Phân tách tập dữ liệu ngay từ đầu (Immutable Data Splitting):** Sử dụng hàm `custom_train_test_split` (tự viết) để chia ngay dữ liệu thô thành Train set (80%) và Test set (20%). Nguyên tắc bất biến: Tập Test được "đóng băng" hoàn toàn và đóng vai trò đại diện cho dữ liệu thực tế.
3. **Khám phá & Trực quan hóa dữ liệu (EDA trên tập Train):** Thực hiện tính toán thống kê (xem phân bố, tìm độ tương quan, phát hiện dị biệt) CHỈ dựa trên tập Train.


4. **Tiền xử lý (Fit trên Train, Transform trên Test):** Mọi class xử lý phải có `fit`, `transform`, `fit_transform`. Imputer tính mean/median/mode chỉ trên train và điền cho cả hai tập. Encoder học vocabulary từ train; nhãn lạ trên test phải được ánh xạ về `unknown_value` (mặc định `-1`). IQR/Z-score chỉ được tìm và xóa dòng ngoại lệ trên train; tuyệt đối không xóa dòng test. `CustomStandardScaler` tính mean/std trên train và chuẩn hóa cả hai tập bằng mẫu số `std + epsilon`, với `epsilon = 1e-9`. Oversampling/undersampling (nếu dùng) chỉ được áp dụng cho train sau split.

5. **Huấn luyện & Tinh chỉnh mô hình (Training, Tuning & Cross-Validation):** Sử dụng `CustomStratifiedKFold(n_splits=5)` bên trong train. Trước khi chia fold, phải kiểm tra số mẫu của mọi lớp lớn hơn hoặc bằng `n_splits`; nếu không thì ném `ValueError` rõ ràng. `CustomGridSearchCV` phải thử `[1e-9, 1e-8, 1e-7, 1e-5, 1e-3, 1e-1]` cho `var_smoothing`. Mỗi fold tính Accuracy, Precision, Recall, F1-Score và ROC-AUC; báo cáo mean và std. Chọn tham số có F1 trung bình cao nhất, dùng ROC-AUC làm tiêu chí phụ và ưu tiên Recall/F1 khi dữ liệu mất cân bằng.

6. **Đánh giá cuối cùng trên tập Test (Final Evaluation & Edge Cases):** Fit mô hình tốt nhất trên toàn bộ train đã xử lý, sau đó dùng test đã `transform()` để gọi `predict` và `predict_proba`. `CustomEvaluator` phải kiểm tra độ dài mọi mảng, không được dùng `zip()` để âm thầm bỏ phần dư; dữ liệu rỗng phải được chặn bằng lỗi rõ ràng hoặc giá trị 0 an toàn. Các metric nhị phân mặc định `positive_label=1`; Precision/Recall/F1 bằng 0 khi mẫu số bằng 0. Báo cáo bắt buộc gồm Accuracy, Precision, Recall, F1-Score và ROC-AUC; lưu `confusion_matrix.png` và `roc_curve.png`.



Hãy bắt đầu bằng việc viết nội dung cho các file trong thư mục `src/` trước, đảm bảo bám sát các phương thức (`fit`, `transform`, `predict_proba`) như mô tả trong file spec.

---

