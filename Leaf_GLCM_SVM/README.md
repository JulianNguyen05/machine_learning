# Leaf Recognition System (GLCM & SVM)

Hệ thống nhận diện lá cây thuốc bằng phương pháp trích xuất đặc trưng kết cấu bề mặt **GLCM (Gray-Level Co-occurrence Matrix)** thủ công kết hợp thuật toán phân lớp **SVM (Support Vector Machine)**. Giao diện người dùng được xây dựng bằng **Tkinter**.

## 📂 Cấu trúc mã nguồn tối giản công khai
* `gui/desktop_app.py`: Giao diện chính của ứng dụng Desktop.
* `src/feature_extraction.py`: Thuật toán tính toán ma trận GLCM thủ công.
* `src/predictor.py`: Hàm xử lý trung gian phục vụ dự đoán ảnh mới.
* `models/`: Chứa bộ não AI đã được huấn luyện sẵn (Scaler, Model, Mapping).

---

## 🚀 Hướng dẫn Cài đặt & Chạy App (Dành cho người dùng)

Nếu bạn chỉ muốn mở App lên để nhận diện ảnh lá cây ngay lập tức mà không cần cài đặt data nặng, hãy làm theo các bước sau:

### Bước 1: Khởi tạo môi trường
```bash
# Clone dự án về máy
git clone <Link_Github_Của_Bạn_Sau_Khi_Tạo>
cd Leaf_GLCM_SVM

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt