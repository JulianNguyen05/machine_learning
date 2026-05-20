# src/ml/train_svm.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ĐÃ SỬA LẠI ĐƯỜNG DẪN: Lùi 3 cấp
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CSV_PATH = os.path.join(BASE_DIR, 'dataset', 'leaf_features_training.csv')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'leaf_svm_model.pkl')
MAPPING_PATH = os.path.join(BASE_DIR, 'models', 'label_mapping.pkl')

def train_and_evaluate_svm():
    print("📂 1. Đang nạp dữ liệu đặc trưng GLCM...")
    if not os.path.exists(CSV_PATH):
        print(f"❌ Lỗi: Không tìm thấy file dữ liệu tại '{CSV_PATH}'. Vui lòng chạy file feature_extraction.py trước!")
        return

    df = pd.read_csv(CSV_PATH)
    X = df.drop(columns=['Label_ID'])
    y = df['Label_ID']

    print("⚖️ 2. Đồng bộ hóa dữ liệu bằng bộ Scaler đã lưu...")
    if not os.path.exists(SCALER_PATH):
        print(f"❌ Lỗi: Không tìm thấy file '{SCALER_PATH}'.")
        return

    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X)

    print("✂️ 3. Chia tập dữ liệu thành Train (80%) và Test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🧠 4. Đang khởi tạo và huấn luyện mô hình SVM...")
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
    svm_model.fit(X_train, y_train)
    print("✅ Huấn luyện mô hình thành công!")

    print("\n📊 5. ĐÁNH GIÁ ĐỘ CHÍNH XÁC CỦA MÔ HÌNH:")
    y_pred = svm_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Độ chính xác tổng thể (Accuracy): {accuracy * 100:.2f}%")

    print("\n📝 Báo cáo chi tiết từng loại lá cây:")
    if os.path.exists(MAPPING_PATH):
        leaf_mapping = joblib.load(MAPPING_PATH)
        target_names = [leaf_mapping[str(cid)]["vn"] for cid in sorted(y_test.unique())]
        print(classification_report(y_test, y_pred, target_names=target_names))
    else:
        print(classification_report(y_test, y_pred))

    print("💾 6. Đang đóng gói và lưu trữ mô hình SVM...")
    joblib.dump(svm_model, MODEL_PATH)
    print(f"🎉 Đã lưu file mô hình hoàn chỉnh tại: '{MODEL_PATH}'")

if __name__ == "__main__":
    train_and_evaluate_svm()