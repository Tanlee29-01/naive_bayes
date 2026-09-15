Role: Senior Data Scientist & Data Analyst
Task: Viết code hoàn chỉnh cho một file Jupyter Notebook (eda_truc_quan_hoa.ipynb) để thực hiện Khám phá và Trực quan hóa dữ liệu (EDA) cho tập dữ liệu customer_purchase_data.csv.

Core Constraints:

TUYỆT ĐỐI KHÔNG SỬ DỤNG sklearn trong bất kỳ bước nào (kể cả việc chia tập train/test).

Chỉ sử dụng pandas, numpy, matplotlib và seaborn.

NGUYÊN TẮC BẤT BIẾN (ANTI-LEAKAGE): Mọi phân tích, tính toán thống kê và vẽ biểu đồ CHỈ được thực hiện trên tập Train. Tập Test phải được tách ra ngay từ đầu và "đóng băng" hoàn toàn.

Yêu cầu cấu trúc các Cell trong Jupyter Notebook:

Cell 1: Import thư viện & Cấu hình đồ thị

Import pandas, numpy, matplotlib.pyplot, seaborn.

Set style cho seaborn để biểu đồ đẹp mắt hơn.

Cell 2: Load dữ liệu thô

Đọc file customer_purchase_data.csv.

Hiển thị thông tin tổng quan (.info()) và 5 dòng đầu tiên.

Cell 3: Tự build hàm Train/Test Split (Không dùng sklearn)

Viết một hàm custom_train_test_split(df, test_size=0.2, random_state=42) sử dụng numpy.random để xáo trộn (shuffle) index và chia tỷ lệ 80/20.

Áp dụng hàm này để tạo ra df_train và df_test.

In ra màn hình số lượng mẫu của tập Train và tập Test để xác nhận.

Lưu ý: Từ Cell 4 trở đi, chỉ được phép gọi biến df_train, tuyệt đối không dùng đến df_test hay biến df ban đầu.

Cell 4: Thống kê mô tả trên tập Train

Dùng .describe() để xem các đại lượng thống kê cơ bản (mean, std, min, max, tứ phân vị).

Kiểm tra tỷ lệ cân bằng của biến mục tiêu PurchaseStatus (số lượng và phần trăm).

Cell 5: Phân tích phân bố đặc trưng (Feature Distribution)

Vẽ các biểu đồ Histogram kết hợp đường cong KDE (sử dụng sns.histplot) cho các biến số học.

Tách màu (hue) theo cột PurchaseStatus để xem sự khác biệt trong phân bố dữ liệu giữa nhóm khách hàng Rời đi và nhóm Mua hàng.

Cell 6: Phân tích tương quan (Correlation Analysis)

Tính toán ma trận tương quan Pearson trên df_train.

Vẽ Heatmap (sử dụng sns.heatmap) hiển thị rõ các hệ số tương quan để xem biến nào ảnh hưởng mạnh nhất đến PurchaseStatus và phát hiện đa cộng tuyến (nếu có).

Cell 7: Phát hiện dị biệt (Outlier Detection)

Vẽ biểu đồ Boxplot (sns.boxplot) cho các biến số học, phân tách theo PurchaseStatus để nhận diện các điểm dữ liệu dị biệt.