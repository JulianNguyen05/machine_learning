import os
import joblib
import pandas as pd
from src.feature_extraction import extract_features_from_image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'leaf_svm_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
MAPPING_PATH = os.path.join(BASE_DIR, 'models', 'label_mapping.pkl')


def predict_new_image(img_path):
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(MAPPING_PATH)):
        return None, None, None, "⚠️ Thiếu file cấu hình model!"

    # 1. Trích xuất đặc trưng GLCM thô từ ảnh
    features = extract_features_from_image(img_path)
    if features is None:
        return None, None, None, "❌ Không thể xử lý bức ảnh này!"

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    mapping = joblib.load(MAPPING_PATH)

    # 2. Chuẩn hóa đặc trưng
    df_features = pd.DataFrame([features])
    features_scaled = scaler.transform(df_features)

    # 3. Tính toán tỷ lệ phần trăm (%) cho TẤT CẢ các lớp
    prob_dist = model.predict_proba(features_scaled)[0]

    # Ghép cặp: (Mã_Lớp, Tỷ_Lệ_%) và sắp xếp từ cao xuống thấp
    class_ids = model.classes_
    prob_pairs = sorted(zip(class_ids, prob_dist), key=lambda x: x[1], reverse=True)

    # Lấy ra Top 3 cây có tỷ lệ phần trăm cao nhất
    top_probabilities = []
    for cid, prob in prob_pairs[:3]:
        name_vn = mapping.get(str(cid), {}).get("vn", "Không rõ")
        top_probabilities.append({
            "vn": name_vn,
            "percent": prob * 100
        })

    # Lấy thông tin cây Cao Nhất để hiển thị riêng
    highest_cid = prob_pairs[0][0]
    leaf_info = mapping.get(str(highest_cid), {"en": "Unknown", "vn": "Không rõ"})

    return leaf_info, top_probabilities, features, None