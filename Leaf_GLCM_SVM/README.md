# 🌿 LeafVision AI - Hệ Thống Nhận Diện Lá Cây

LeafVision AI là một ứng dụng Desktop hoạt động offline 100%, được phát triển bằng Python. Hệ thống kết hợp thuật toán xử lý ảnh truyền thống **GLCM (Gray-Level Co-occurrence Matrix)** để trích xuất đặc trưng kết cấu bề mặt lá và mô hình học máy **SVM (Support Vector Machine)** để phân loại loài cây. 

Dự án sở hữu giao diện người dùng (UI) hiện đại, mang phong cách "Dark Tech" được xây dựng bằng `CustomTkinter`.

---

## ✨ Tính Năng Nổi Bật
* **🔍 Quét Lá Phân Tích:** Tải ảnh lên (hỗ trợ ảnh macro độ phân giải siêu cao mà không gây tràn RAM) để AI phân tích và đưa ra Top 3 dự đoán loài cây cùng các chỉ số độ tin cậy.
* **📐 Trích xuất Vector GLCM:** Tự động tính toán và hiển thị 10 đặc trưng kết cấu bề mặt (Độ tương phản, Năng lượng, Entropy, v.v.).
* **📚 Từ Điển Thực Vật:** Tra cứu nhanh (0ms delay) thông tin khoa học của 15 loài cây thuốc với kho dữ liệu nội bộ (Hard Data), kèm ảnh minh họa tự động trích xuất từ Dataset.
* **⚡ 100% Offline & Bảo mật:** Không cần kết nối Internet, không gửi dữ liệu qua API bên thứ ba.

---

## 🛠 Công Nghệ Sử Dụng (Tech Stack)
* **Ngôn ngữ:** Python 3.9+
* **Machine Learning:** `scikit-learn`, `pandas`, `numpy`
* **Xử lý ảnh (Computer Vision):** `opencv-python` (cv2), `Pillow` (PIL)
* **Giao diện (GUI):** `customtkinter` (Mô hình MVC)

---

## 📂 Cấu Trúc Thư Mục (Project Structure)

```text
Leaf_GLCM_SVM/
├── dataset/                    # Chứa tập dữ liệu gốc và file features CSV
├── models/                     # Nơi lưu trữ các file mô hình (.pkl) sau khi Train
├── src/
│   ├── data/                   # Tầng Dữ liệu: Chứa hard-data (plants_db.py)
│   ├── gui/                    # Tầng Giao diện (View): main_window.py (CustomTkinter)
│   └── ml/                     # Tầng AI/ML: Code trích xuất GLCM & Huấn luyện SVM
├── app.py                      # Tầng Điều khiển (Controller): File chạy chính của App
├── download_dataset.py         # Script tự động tải tập dữ liệu Swedish Leaf
└── requirements.txt            # Danh sách thư viện cần thiết

```

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy (Dành cho máy mới Clone về)

Vui lòng thực hiện tuần tự các bước dưới đây để xây dựng lại mô hình AI và khởi chạy ứng dụng từ đầu.

### Bước 1: Khởi tạo môi trường

Khuyến nghị sử dụng môi trường ảo (Virtual Environment) để tránh xung đột thư viện.

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường (Trên Windows)
.venv\Scripts\activate
# (Hoặc trên macOS/Linux)
source .venv/bin/activate

# Cài đặt các thư viện lõi
pip install -r requirements.txt

```

*(Nếu chưa có file `requirements.txt`, hãy chạy lệnh: `pip install customtkinter opencv-python scikit-learn pandas pillow joblib tqdm`)*

### Bước 2: Tải tập dữ liệu hình ảnh (Dataset)

Chạy script sau để tự động tải và giải nén bộ dữ liệu Swedish Leaf Dataset về thư mục `dataset/`:

```bash
python download_dataset.py

```

### Bước 3: Trích xuất đặc trưng hình thái (GLCM)

Lệnh này sẽ quét toàn bộ ảnh trong tập dữ liệu, tính toán ma trận GLCM và lưu kết quả vào file `dataset/leaf_features_training.csv`. Đồng thời lưu các bộ tiền xử lý `scaler.pkl` và `label_mapping.pkl` vào thư mục `models/`.

```bash
python src/ml/feature_extraction.py

```

*(Lưu ý: Quá trình này có thể mất vài phút tùy vào cấu hình máy tính. Hãy đợi thanh tiến trình chạy đến 100%).*

### Bước 4: Huấn luyện mô hình phân lớp (Train SVM)

Sau khi đã có file CSV ở Bước 3, tiến hành huấn luyện mô hình học máy:

```bash
python src/ml/train_svm.py

```

*(Sau khi chạy xong, hệ thống sẽ in ra độ chính xác (Accuracy) và lưu file `leaf_svm_model.pkl` vào thư mục `models/`).*

### Bước 5: Khởi chạy Ứng dụng Desktop

Khi trong thư mục `models/` đã có đủ 3 file `.pkl`, hệ thống đã sẵn sàng làm việc. Khởi động giao diện chính:

```bash
python app.py

```

---

## 👨‍💻 Tác giả / Nguồn dữ liệu

* **Dataset:** [Swedish Leaf Dataset](https://www.cvl.isy.liu.se/en/research/datasets/swedish-leaf/) (15 loài thực vật).
* Dự án được phát triển phục vụ mục đích nghiên cứu học thuật về nhận dạng mẫu sinh trắc học và xử lý ảnh (Pattern Recognition & Image Processing).