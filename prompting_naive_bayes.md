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
│   ├── tien_xu_ly.py               # Chứa hàm custom_train_test_split và class CustomStandardScaler
│   ├── thuat_toan_nb.py            # Chứa class CustomGaussianNB (tính Log-Probability, var_smoothing)
│   └── danh_gia.py                 # Chứa class CustomStratifiedKFold và CustomEvaluator
├── chay_thu_nghiem.py              # File chính chạy toàn bộ pipeline
└── README.md                       # Hướng dẫn chạy code và giải thích pipeline

```

### 2. QUY TRÌNH PIPELINE 6 BƯỚC BẮT BUỘC

File `chay_thu_nghiem.py` phải thực thi một luồng pipeline hoàn chỉnh tuân thủ nghiêm ngặt 6 nguyên tắc sau để chống rò rỉ dữ liệu (Data Leakage):

1. **Thu thập & Tải dữ liệu thô (Raw Data Ingestion):** Tải tập dữ liệu gốc vào Pandas DataFrame. Tuyệt đối chưa thực hiện bất kỳ phép biến đổi toán học hay tính toán thống kê nào ở bước này.
2. **Phân tách tập dữ liệu ngay từ đầu (Immutable Data Splitting):** Sử dụng hàm `custom_train_test_split` (tự viết) để chia ngay dữ liệu thô thành Train set (80%) và Test set (20%). Nguyên tắc bất biến: Tập Test được "đóng băng" hoàn toàn và đóng vai trò đại diện cho dữ liệu thực tế.
3. **Khám phá & Trực quan hóa dữ liệu (EDA trên tập Train):** Thực hiện tính toán thống kê (xem phân bố, tìm độ tương quan, phát hiện dị biệt) CHỈ dựa trên tập Train.


4. **Tiền xử lý (Fit trên Train, Transform trên Test):** Mọi phép xử lý được đóng gói theo cơ chế Học tham số trên tập Train (`.fit()`) và Áp dụng tham số đó sang tập Test (`.transform()`). Sử dụng `CustomStandardScaler` để tính mean và std trên tập Train, sau đó áp dụng biến đổi cho cả hai tập.
5. **Huấn luyện & Tinh chỉnh mô hình (Training & Cross-Validation):** Sử dụng `CustomStratifiedKFold` (chia K-Fold) ngay bên trong tập Train để huấn luyện `CustomGaussianNB`. Mọi thử nghiệm đánh giá nội bộ chỉ diễn ra trong phạm vi tập Train để tránh quá khớp (overfitting).


6. **Đánh giá cuối cùng trên tập Test (Final Evaluation):** Đưa tập Test (đã qua biến đổi `.transform()`) vào mô hình `CustomGaussianNB` để tính toán các chỉ số thực tế qua `CustomEvaluator` (Accuracy, Precision, Recall, F1-Score, vẽ ROC Curve, Confusion Matrix). Kết quả này phản ánh chính xác khả năng tổng quát hóa của mô hình.



Hãy bắt đầu bằng việc viết nội dung cho các file trong thư mục `src/` trước, đảm bảo bám sát các phương thức (`fit`, `transform`, `predict_proba`) như mô tả trong file spec.

---

