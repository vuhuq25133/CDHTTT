# Music Recommendation System (Hệ thống gợi ý âm nhạc)

Dự án Hệ thống gợi ý âm nhạc sử dụng các thuật toán học máy (Machine Learning) để cung cấp các gợi ý bài hát cá nhân hóa cho người dùng. Ứng dụng web được xây dựng bằng **Flask** và sử dụng cơ sở dữ liệu **MySQL**, tích hợp cùng với các mô hình gợi ý đa dạng như Content-Based, Collaborative Filtering (SVD, KNN).

## Cấu trúc dự án

Dự án được tổ chức thành các thành phần chính như sau:

```text
├── app.py                      # Tệp tin chạy chính của ứng dụng web (Flask).
├── models/                     # Thư mục chứa các mô hình học máy đã huấn luyện (.pkl).
├── processed_data/             # Thư mục chứa dữ liệu đã qua tiền xử lý.
├── dataset_filter/             # Các kịch bản python để lọc và chuẩn bị dữ liệu (train_filter, song_filter...).
├── static/                     # Các tệp tĩnh (CSS, JS, hình ảnh) cho giao diện web.
├── templates/                  # Các file HTML (Jinja2) cho giao diện web.
├── 01_Data_Preparation.ipynb   # Notebook tiền xử lý dữ liệu đầu vào.
├── 02_1_Content_Based.ipynb    # Notebook huấn luyện mô hình Content-Based.
├── 02_2_KNN.ipynb              # Notebook huấn luyện mô hình KNN.
├── 02_3_SVD.ipynb              # Notebook huấn luyện mô hình SVD.
├── import_db.py                # Script hỗ trợ nạp dữ liệu vào cơ sở dữ liệu MySQL.
├── migrate.sql                 # Câu lệnh SQL để khởi tạo các bảng cơ sở dữ liệu.
├── requirements.txt            # Danh sách các thư viện cần thiết.
└── README.md                   # Tài liệu hướng dẫn dự án.
```

## Các mô hình sử dụng

1. **Content-Based Filtering**: Sử dụng TF-IDF để trích xuất đặc trưng từ dữ liệu của bài hát và dùng KNN cosine distance để tìm bài hát tương tự. Giải quyết bài toán gợi ý các bài hát có đặc điểm nội dung giống với bài hát người dùng đang nghe.
2. **Hybrid SVD**: Gợi ý cá nhân hóa dùng Matrix Factorization kết hợp với hồ sơ sở thích người dùng (User Taste Profile). Hỗ trợ giải quyết bài toán cold-start cho người dùng mới thông qua trung bình hóa véc-tơ dựa trên ngôn ngữ/thể loại yêu thích.
3. **Hybrid KNN**: Gợi ý dạng Item-Based, kết hợp thông tin cá nhân của người dùng để tăng cường (Boost) điểm số gợi ý. Giải quyết câu chuyện "Những người nghe tương tự cũng thích nghe...".

## Hướng dẫn Cài đặt và Chạy ứng dụng

### 1. Yêu cầu hệ thống
- Python 3.8+
- MySQL Server

### 2. Cài đặt môi trường và thư viện
Khuyến khích sử dụng môi trường ảo (Virtual Environment) để cài đặt các thư viện:

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows:
venv\Scripts\activate
# Trên macOS/Linux:
source venv/bin/activate

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

### 3. Thiết lập Cơ sở dữ liệu
1. Tạo một cơ sở dữ liệu có tên `music_recommendation_db` trong MySQL của bạn.
2. Cập nhật thông tin kết nối (Tài khoản, Mật khẩu MySQL) trong các file `app.py` và `import_db.py`. Ví dụ:
   ```python
   USERNAME = "root"
   PASSWORD = "your_password"
   HOST = "localhost"
   PORT = "3306"
   DB_NAME = "music_recommendation_db"
   ```
3. Chạy file `migrate.sql` trong giao diện quản trị MySQL (như MySQL Workbench) hoặc bằng dòng lệnh để tạo các bảng.
4. Chạy script `import_db.py` để nhập dữ liệu mẫu từ các file CSV (sau quá trình trích xuất) vào database.
   ```bash
   python import_db.py
   ```

### 4. Huấn luyện mô hình (Tùy chọn)
Các mô hình đã được huấn luyện sẵn và lưu ở định dạng `.pkl` (đặt tại thư mục `models/` và thư mục gốc). Nếu bạn cần huấn luyện lại mô hình từ dữ liệu mới, hãy chạy tuần tự các file Jupyter Notebook từ bước `01_Data_Preparation.ipynb` cho đến `02_3_SVD.ipynb`.

### 5. Khởi chạy Ứng dụng
Để chạy ứng dụng web Flask, thực thi lệnh sau:

```bash
python app.py
```
Ứng dụng sẽ được khởi động tại địa chỉ: `http://localhost:5000` hoặc `http://127.0.0.1:5000`. Bạn có thể truy cập bằng trình duyệt web để sử dụng.
