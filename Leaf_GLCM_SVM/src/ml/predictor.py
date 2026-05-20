# src/ml/predictor.py

import os
import joblib
import pandas as pd
from src.ml.feature_extraction import extract_features_from_image
from src.data.plants_db import PLANTS_DB

# ĐÃ SỬA LẠI ĐƯỜNG DẪN: Lùi 3 cấp
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'leaf_svm_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

def predict_new_image(img_path):
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        return None, None, None, "⚠️ Thiếu file cấu hình model! (model.pkl hoặc scaler.pkl)"

    # 1. Trích xuất đặc trưng
    features = extract_features_from_image(img_path)
    if features is None:
        return None, None, None, "❌ Không thể xử lý bức ảnh này!"

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 2. Chuẩn hóa đặc trưng
    df_features = pd.DataFrame([features])
    features_scaled = scaler.transform(df_features)

    # 3. Tính toán tỷ lệ phần trăm (%)
    prob_dist = model.predict_proba(features_scaled)[0]
    class_ids = model.classes_
    prob_pairs = sorted(zip(class_ids, prob_dist), key=lambda x: x[1], reverse=True)

    # 4. Trả về kết quả Top 3
    top_probabilities = []
    for cid, prob in prob_pairs[:3]:
        name_vn = PLANTS_DB.get(str(cid), {}).get("name_vn", "Không rõ")
        top_probabilities.append({
            "label_id": str(cid),
            "vn": name_vn,
            "percent": round(prob * 100, 2)
        })

    # Lấy thông tin cây giống nhất (Top 1)
    highest_cid = prob_pairs[0][0]
    best_match = {
        "label_id": str(highest_cid),
        "vn": PLANTS_DB.get(str(highest_cid), {}).get("name_vn", "Không rõ"),
        "percent": round(prob_pairs[0][1] * 100, 2)
    }

    return best_match, top_probabilities, features, None