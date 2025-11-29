# NUMPY FOR DATA SCIENCE - PHÂN TÍCH KHẢ NĂNG RỜI ĐI CỦA KHÁCH HÀNG

## 🔍Mục lục
1.  [Tiêu đề và Mô tả ngắn gọn](#1-tiêu-đề-và-mô-tả-ngắn-gọn)
2.  [Giới thiệu](#2-giới-thiệu)
    *   [Mô tả Bài toán](#mô-tả-bài-toán)
    *   [Động lực và Ứng dụng thực tế](#động-lực-và-ứng-dụng-thực-tế)
    *   [Mục tiêu cụ thể](#mục-tiêu-cụ-thể)
3.  [Dataset](#3-dataset)
    *   [Nguồn Dữ liệu](#nguồn-dữ-liệu)
    *   [Mô tả Features](#mô-tả-features)
    *   [Kích thước và Đặc điểm Dữ liệu](#kích-thước-và-đặc-điểm-dữ-liệu)
4.  [Method](#4-method)
    *   [Quy trình Xử lý Dữ liệu](#quy-trình-xử-lý-dữ-liệu)
    *   [Thuật toán sử dụng](#thuật-toán-sử-dụng)
    *   [Giải thích Implement bằng NumPy](#giải-thích-implement-bằng-numpy)
5.  [Installation & Setup](#5-installation--setup)
    * [Tạo môi trường ảo](#tạo-môi-trường-ảo)
    * [Kích hoạt môi trường](#kích-hoạt-môi-trường)
    * [Cài đặt các thư viện cần thiết](#cài-đặt-các-thư-viện-cần-thiết)
6.  [Usage](#6-usage)
7.  [Results](#7-results)
    * [Các độ đo đánh giá (Metrics)](#các-độ-đo-đánh-giá-metrics)
    * [Đường cong Receiver Operating Characteristic (ROC)](#đường-cong-receiver-operating-characteristic-roc)
8.  [Project Structure](#8-project-structure)
9.  [Challenges & Solutions](#9-challenges--solutions)
10. [Future Improvements](#10-future-improvements)
11. [Contributors](#11-contributors)
12. [License](#12-license)

---

## 1. Tiêu đề và Mô tả ngắn gọn

**Tiêu đề:** Xây dựng Mô hình Dự đoán Khách hàng Rời đi đầu vào là thông tin khách hàng tín dụng.

**Mô tả ngắn gọn:** Project này nhằm mục đích xây dựng một mô hình học máy đơn giản để dự đoán khả năng một khách hàng thẻ tín dụng sẽ rời bỏ ngân hàng dựa trên dữ liệu khách hàng và lịch sử giao dịch. Toàn bộ quá trình tiền xử lý, tính toán và xây dựng mô hình. Không được sử dụng `Pandas` mà chỉ sử dụng `Numpy` cho các thao tác liên quan.

## 2. Giới thiệu

### Mô tả Bài toán
Bài toán là một nhiệm vụ **phân loại (Classification)**: Dựa trên các thông tin về khách hàng (giới tính, độ tuổi, thu nhập, tình trạng hôn nhân, chỉ số tín dụng, v.v.) và hoạt động thẻ (số lượng giao dịch, tổng chi tiêu), dự đoán xem tài khoản của khách hàng đó **đã rời đi (Attrited)** hay **còn tồn tại (Existing)**.

### Động lực và Ứng dụng thực tế
Việc dự đoán khách hàng rời đi (Attrited Customer Prediction) là tối quan trọng trong ngành Ngân hàng và Tài chính.
*   **Động lực:** Chi phí để giữ chân một khách hàng hiện tại thấp hơn nhiều so với chi phí thu hút một khách hàng mới.
*   **Ứng dụng:** Mô hình dự đoán giúp ngân hàng xác định sớm những khách hàng có nguy cơ rời đi cao, từ đó triển khai các chiến lược giữ chân (Retention Strategies) kịp thời như cung cấp ưu đãi đặc biệt, cải thiện dịch vụ chăm sóc khách hàng.

### Mục tiêu cụ thể
1.  **Thành thạo NumPy:** Sử dụng `NumPy` hiệu quả cho tất cả các tác vụ xử lý dữ liệu dạng bảng.
2.  **Phân tích Dữ liệu:** Đặt và trả lời các câu hỏi về dữ liệu thông qua thống kê mô tả và trực quan hóa (Matplotlib/Seaborn).
3.  **Xây dựng Mô hình (Core/Nâng cao):**
    *   **Core:** Chỉ sử dụng `Numpy` (hoặc sử dụng Scikit-learn) để xây dựng mô hình phân loại (ví dụ: Logistic Regression).
4.  **Trực quan hóa:** Minh họa kết quả phân tích và hiệu suất mô hình bằng Matplotlib và Seaborn.

## 3. Dataset

### Nguồn Dữ liệu
*   **Tên:** Credit Card customers
*   **Nguồn:** [Kaggle - Credit Card customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
*   **Mô tả:** Dữ liệu chứa thông tin chi tiết về khách hàng thẻ tín dụng của một ngân hàng, bao gồm các chỉ số nhân khẩu học, chỉ số tín dụng, và tình trạng tài khoản.

### Mô tả Features
Các đặc trưng chính:
*   `CLIENTNUM`: Số khách hàng (ID duy nhất).
*   `Attrition_Flag`: Biến mục tiêu (Existing Customer / Attrited Customer).
*   `Customer_Age`: Tuổi của khách hàng.
*   `Gender`: Giới tính.
*   `Dependent_count`: Số người phụ thuộc.
*   `Income_Category`: Mức thu nhập hàng năm.
*   `Card_Category`: Loại thẻ tín dụng (Blue, Silver, Gold, Platinum).
*   `Credit_Limit`: Hạn mức tín dụng.
*   `Total_Trans_Amt`: Tổng số tiền giao dịch (trong 12 tháng).
*   `Total_Trans_Ct`: Tổng số lượng giao dịch (trong 12 tháng).
*   ...

### Kích thước và Đặc điểm Dữ liệu
*   **Kích thước:** Ví dụ: 10127 hàng, 23 cột.
*   **Đặc điểm:** Dữ liệu bao gồm các cột dạng số (Age, Credit Limit, Total Trans Amt, ...) và cột dạng phân loại (Gender, Income Category, Card Category). Cần xử lý Missing Values, mã hóa dữ liệu phân loại và chuẩn hóa/điều chuẩn dữ liệu số.

## 4. Method

### Quy trình Xử lý Dữ liệu
1.  **Đọc/Load Dữ liệu:** Sử dụng hàm của NumPy (`np.genfromtxt`) để đọc dữ liệu từ file.
2.  **Tiền xử lý:**
    *   Kiểm tra và xử lý Missing Values.
    *   Xử lý giá trị ngoại lai (Outliers) bằng các kỹ thuật thống kê (ví dụ: Z-score, IQR).
    *   Mã hóa biến phân loại (Label Encoding, One-Hot Encoding, Ordinal Encoding) bằng các thao tác mảng của NumPy.
    *   Thêm các đặc trưng mới có mối quan hệ liên quan (Feature engineering)
3.  **Tách Dữ liệu:** Chia dữ liệu thành tập huấn luyện (Training set) và tập kiểm tra (Testing set).

### Thuật toán sử dụng
*   **Thuật toán:** `Logistic Regression`.
*   **Công thức toán học cho Logistic Regression:**
    *   Hàm tuyến tính: $z = \mathbf{w}^T \mathbf{x} + b$
    *   Hàm Sigmoid (Hàm kích hoạt): $\sigma(z) = \frac{1}{1 + e^{-z}}$
    *   Hàm mất mát (Binary Cross-Entropy): 
    
    $L(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})]$

    *   Thuật toán tối ưu: Gradient Descent (tính đạo hàm và cập nhật $\mathbf{w}, b$)

### Giải thích Implement bằng NumPy
*   Toàn bộ các phép toán vector, ma trận (như phép nhân ma trận `np.dot`, tính tổng `np.sum`, tính lũy thừa `np.exp`, tránh tràn số `np.clip`) được sử dụng để cài đặt các công thức toán học trong mô hình `Logistic Regression` đã nói đến ở trên.
*   Sử dụng broadcasting trong cài đặt hàm `One-hot Encoding` để thực hiện tạo ma trận có các mới có số cột là các giá trị duy nhất (unique values) của đặc trưng cần mã hóa và số dòng bằng số dòng dữ liệu.
    - Mảng unique có kích thước $(k, )$
    - Cột dữ liệu của nó có dạng $(n, 1)$
    - Do cơ chế `broadcasting` thì kích thước của mảng unique sẽ biến thành $(1, k)$ và cuối cùng cả 2 sẽ có cùng kích thước $(n, k)$
*   Sử dụng `np.vectorize` để thao tác trên toàn bộ phần tử của mà ma trận không cần vòng lặp.
*   Trong cài đặt hàm `Ordinal Encoding` sử dụng fancy indexing, dùng các con số index trong danh sách `inv` để lấy ra Value tương ứng, ...

## 5. Installation & Setup

### Tạo môi trường ảo
python -m venv venv

### Kích hoạt môi trường

>> Đối với Windows (Command Prompt):
```bash
venv\Scripts\activate.bat
```
>> Đối với Windows (PowerShell):
```bash
.\venv\Scripts\Activate.ps1
```

>> Đối với macOS / Linux:
```bash
source venv/bin/activate
```

### Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```
## 6. Usage

Hướng dẫn cách chạy project theo từng bước
- Khám phá Dữ liệu: Chạy file `notebooks/01_data_exploration.ipynb` để hiểu về dữ liệu, thống kê mô tả, và các biểu đồ phân tích ban đầu.
- Tiền xử lý: Chạy file `notebooks/02_preprocessing.ipynb` để làm sạch, xử lý Missing Values, mã hóa và chuẩn hóa dữ liệu.
- Xây dựng Mô hình: Chạy file `notebooks/03_modeling.ipynb` để huấn luyện mô hình, đánh giá hiệu suất và đưa ra dự đoán.

**Lưu ý:** Ở mỗi file chỉ cần chọn `Run All` hoặc `Restart & Run All` để chạy toàn bộ notebook.

## 7. Results

### Các độ đo đánh giá (Metrics)

![Stratified 5-Fold Cross-Validation Metrics](notebooks/metrics_plot.png)

- `Logistic Regression` kết hợp với `Stratified 5-Fold Cross-Validation` cho thấy mô hình hoạt động ổn định và đáng tin cậy trong dự đoán khách hàng rời đi tuy mô hình phân loại tốt nhóm khách hàng hiện tại (Existing), nhưng vẫn có sự nhầm lẫn nhỏ ở nhóm khách hàng rời bỏ (Attrited).

- `Precision` phù hợp giúp tránh báo động nhầm, trong khi `Recall` rất quan trọng nhất đối với bài toán giữ chân khách hàng – đạt mức khả quan, thể hiện khả năng phát hiện các khách hàng có nguy cơ rời đi. 

- `F1-score` cao củng cố thêm tính hiệu quả của mô hình trong điều kiện dữ liệu mất cân bằng. 

### Đường cong **Receiver Operating Characteristic** (ROC)

![Receiver Operating Characteristic (ROC) Curve](notebooks/roc_curve_plot.png)

- **Hiệu năng xuất sắc:** Chỉ số **Mean AUC đạt ~0.94** cho thấy mô hình có khả năng phân loại giữa khách hàng "Rời đi" và "Ở lại" tốt hơn gấp nhiều lần so với đoán mò. Điều này củng cố cho kết quả `F1-score` cao ở trên, chứng tỏ mô hình không chỉ đúng tại một điểm mà hoạt động tốt trên toàn miền dữ liệu.

- **Tính ổn định cao:** Các đường ROC của từng Fold (nét mờ) nằm rất sát đường trung bình và độ lệch chuẩn nhỏ. Điều này đồng nhất với sự ổn định của các cột `Accuracy` và `Recall` trong biểu đồ **Metrics**, khẳng định mô hình không bị hiện tượng quá khớp (overfitting).

- **Tối ưu hóa Recall:** Đường cong có dạng dốc đứng ngay từ đầu trục hoành, cho thấy mô hình có thể đạt được `Recall` cao (bắt đúng khách rời bỏ) mà vẫn giữ tỷ lệ báo động giả (False Positive) ở mức thấp.

**Kết luận chung:** Mô hình **Logistic Regression** là lựa chọn phù hợp và hiệu quả cho bài toán.

## 8. Project Structure

```
Credits_Card_customers-Numpy-DS/
├── README.md                   # Tổng quan dự án, hướng dẫn cài đặt & License
├── requirements.txt            # Danh sách các thư viện cần thiết (numpy, pandas...)
├── data/
│   ├── raw/                    # Dữ liệu thô ban đầu
│   └── processed/              # Dữ liệu sau khi đã làm sạch và xử lý
├── notebooks/                  # Nơi chạy thử nghiệm và phân tích
│   ├── 01_data_exploration.ipynb # Phân tích khám phá dữ liệu (EDA)
│   ├── 02_preprocessing.ipynb  # Tiền xử lý, chuẩn hóa và chia tập dữ liệu
│   ├── 03_modeling.ipynb       # Huấn luyện mô hình và tính toán các độ đo đánh giá
│   ├── metrics_plot.png        # Trực quán hóa các độ đo đánh giá bằng biểu đồ
│   └── roc_curve_plot.png      # Đường cong đánh giá khả năng phân loại của mô hình
├── src/                        # Mã nguồn chính (dùng để tái sử dụng)
│   ├── __init__.py             # Đánh dấu thư mục này là một Python Package
│   ├── data_processing.py      # Các hàm tải và xử lý dữ liệu
│   ├── visualization.py        # Các hàm vẽ các biểu đồ Pie, Box, Histogram...
│   └── models.py               # Cài đặt thuật toán Logistic Regression (chỉ NumPy)
```

## 9. Challenges & Solutions

- Khó khăn 1: Xử lý các biến phân loại/chuỗi chỉ bằng NumPy (không dùng Pandas).
    - Giải pháp: Sử dụng các kỹ thuật như ánh xạ (mapping) thủ công sang giá trị số và sử dụng masking/fancy indexing của NumPy để thực hiện One-Hot Encoding và Ordinal encoding.
- Khó khăn 2: Đảm bảo Vectorization hoàn toàn, tránh vòng lặp for trong các thao tác trên ma trận.
    - Giải pháp: Tận dụng tối đa các hàm và kỹ thuật broadcasting, fancy indexing, vectorize của NumPy.

## 10. Future Improvements

- Thử nghiệm các kỹ thuật giảm chiều dữ liệu (ví dụ: PCA - tự implement bằng NumPy).
- Cải tiến thuật toán tối ưu (ví dụ: dùng Adam thay vì Gradient Descent đơn thuần).
- Thử nghiệm các mô hình phức tạp hơn (ví dụ: Linear Discriminant Analysis - LDA).

## 11. Contributors

Tên: Cao Tiến Thành

MSSV: 23120088

Contact: caotienthanh1103@gmail.com

## 12. License

- MIT License

Copyright (c) 2025 [Cao Tien Thanh]