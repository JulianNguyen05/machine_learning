import os
import joblib
import pandas as pd
from src.feature_extraction import extract_features_from_image

# Xác định thư mục gốc của dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'leaf_svm_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
MAPPING_PATH = os.path.join(BASE_DIR, 'models', 'label_mapping.pkl')


def predict_new_image(img_path):
    """Hàm nhận vào đường dẫn ảnh, trả về thông tin loài cây hoặc thông báo lỗi"""
    # 1. Kiểm tra sự tồn tại của các file model
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(MAPPING_PATH)):
        return None, "⚠️ Hệ thống thiếu file cấu hình model trong thư mục models!"

    # 2. Trích xuất đặc trưng GLCM từ ảnh mới thông qua hàm có sẵn
    features = extract_features_from_image(img_path)
    if features is None:
        return None, "❌ Không thể đọc hoặc xử lý bức ảnh này!"

    # 3. Nạp model và scaler lên để xử lý
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    mapping = joblib.load(MAPPING_PATH)

    # Convert đặc trưng thành DataFrame để khớp tên cột lúc train
    df_features = pd.DataFrame([features])

    # 4. Chuẩn hóa đặc trưng theo đúng chuẩn của tập Train
    features_scaled = scaler.transform(df_features)

    # 5. Đưa vào SVM để dự đoán ra mã số (Label ID)
    prediction_id = model.predict(features_scaled)[0]

    # 6. Tra từ điển để lấy tên Tiếng Anh và Tiếng Việt
    leaf_info = mapping.get(str(prediction_id), {"en": "Unknown", "vn": "Không rõ"})

    return leaf_info, None