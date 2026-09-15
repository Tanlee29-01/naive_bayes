# ML_NaiveBayes

Thư viện Gaussian Naive Bayes hướng đối tượng, tự cài đặt bằng Python, NumPy và Pandas. Project không sử dụng `scikit-learn`.

## Cấu trúc

- `data/customer_purchase_data.csv`: dữ liệu gốc.
- `src/tien_xu_ly.py`: chia dữ liệu và `CustomStandardScaler`.
- `src/thuat_toan_nb.py`: `CustomGaussianNB`, gồm prior, Gaussian log-probability, `predict`, `predict_proba` và `var_smoothing`.
- `src/danh_gia.py`: stratified K-fold, confusion matrix, Accuracy, Precision, Recall, F1, ROC và AUC.
- `chay_thu_nghiem.py`: pipeline 6 bước và lưu biểu đồ vào `outputs/`.

## Chạy

Từ thư mục `naive_bayes`:

```text
.venv\Scripts\python.exe ML_NaiveBayes\chay_thu_nghiem.py
```

Pipeline đọc dữ liệu thô, chia train/test 80/20 trước mọi thống kê, chỉ EDA trên train, fit scaler theo train (và theo từng fold trong cross-validation), huấn luyện Gaussian NB, rồi đánh giá đúng một lần trên test đã được giữ kín.
