# ==========================================
# 1. CẤU HÌNH & IMPORT
# ==========================================
import os
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

import cv2
import glob
import numpy as np
import pandas as pd
import joblib  # Dùng để lưu scaler và mapping
from tqdm import tqdm  # Thư viện tạo thanh tiến trình chuyên nghiệp

# 🎯 TỰ ĐỘNG TÌM THƯ MỤC GỐC DỰ ÁN (Leaf_GLCM_SVM)
# os.path.abspath(__file__) lấy đường dẫn file này -> đi ngược lên 2 cấp thư mục sẽ ra thư mục gốc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Định nghĩa các đường dẫn chuẩn theo thư mục gốc
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'swedish_leaf_dataset', 'Swedish', 'Train')
OUTPUT_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_CSV = os.path.join(BASE_DIR, 'dataset', 'leaf_features_training.csv')
MAPPING_FILE = os.path.join(OUTPUT_DIR, 'label_mapping.pkl')

# Tạo thư mục models ở gốc nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SIZE = (256, 256)
LEVELS = 8
ANGLES = [0, 45, 90, 135]

LEAF_MAPPING = {
    "0": {"en": "Ulmus carpinifolia", "vn": "Cây Du"},
    "1": {"en": "Acer", "vn": "Cây Phong"},
    "2": {"en": "Salix aurita", "vn": "Cây Liễu tai chuột"},
    "3": {"en": "Quercus", "vn": "Cây Sồi"},
    "4": {"en": "Alnus incana", "vn": "Cây Tống quán sủ xám"},
    "5": {"en": "Betula pubescens", "vn": "Cây Bạch dương lông"},
    "6": {"en": "Salix alba 'Tristis'", "vn": "Cây Liễu rủ"},
    "7": {"en": "Populus tremula", "vn": "Cây Bạch dương rung"},
    "8": {"en": "Ulmus glabra", "vn": "Cây Du núi"},
    "9": {"en": "Sorbus aucuparia", "vn": "Cây Thanh lương trà"},
    "10": {"en": "Salix cinerea", "vn": "Cây Liễu xám"},
    "11": {"en": "Populus", "vn": "Cây Dương / Bạch dương"},
    "12": {"en": "Tilia", "vn": "Cây Đoạn"},
    "13": {"en": "Sorbus intermedia", "vn": "Cây Thanh lương trà Thụy Điển"},
    "14": {"en": "Fagus silvatica", "vn": "Cây Dẻ gai châu Âu"}
}


# ==========================================
# 2. HÀM TÍNH TOÁN (GIỮ NGUYÊN LOGIC CỦA BẠN)
# ==========================================
def compute_glcm_manual(image, distance=1, angle=0, symmetric=True):
    glcm = np.zeros((LEVELS, LEVELS), dtype=float)
    rows, cols = image.shape

    if angle == 0:
        for i in range(rows):
            for j in range(cols - distance):
                glcm[image[i, j], image[i, j + distance]] += 1
    elif angle == 45:
        for i in range(rows - distance):
            for j in range(cols - distance):
                glcm[image[i, j], image[i + distance, j + distance]] += 1
    elif angle == 90:
        for i in range(rows - distance):
            for j in range(cols):
                glcm[image[i, j], image[i + distance, j]] += 1
    elif angle == 135:
        for i in range(rows - distance):
            for j in range(distance, cols):
                glcm[image[i, j], image[i + distance, j - distance]] += 1

    if symmetric:
        glcm = glcm + glcm.T
    return glcm


def extract_features_from_image(img_path):
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None: return None

    img_resized = cv2.resize(img_gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    # Gom mức xám về LEVELS (VD: 8 mức)
    img_quantized = (img_resized // (256 // LEVELS)).astype('uint8')

    glcm_combined = np.zeros((LEVELS, LEVELS))
    for angle in ANGLES:
        glcm_combined += compute_glcm_manual(img_quantized, distance=1, angle=angle)

    P = glcm_combined / (np.sum(glcm_combined) + 1e-9)
    i, j = np.indices((LEVELS, LEVELS))

    features = {}
    features['Contrast'] = np.sum(P * (i - j) ** 2)
    features['Dissimilarity'] = np.sum(P * np.abs(i - j))
    features['Homogeneity'] = np.sum(P / (1.0 + (i - j) ** 2))
    features['ASM'] = np.sum(P ** 2)
    features['Energy'] = np.sqrt(features['ASM'])
    features['Max_Probability'] = np.max(P)

    P_nonzero = P[P > 0]
    features['Entropy'] = -np.sum(P_nonzero * np.log2(P_nonzero))

    m_i = np.sum(i * P)
    m_j = np.sum(j * P)
    var_i = np.sum(P * (i - m_i) ** 2)
    std_i = np.sqrt(var_i)

    features['GLCM_Mean'] = m_i
    features['GLCM_Variance'] = var_i
    if std_i == 0:
        features['Correlation'] = 1.0
    else:
        features['Correlation'] = np.sum(P * (i - m_i) * (j - m_j)) / (std_i ** 2 + 1e-9)

    return features


# ==========================================
# 3. QUY TRÌNH TRÍCH XUẤT ĐỂ TRAIN MODEL
# ==========================================
# 🎯 THÊM DÒNG NÀY ĐỂ BẢO VỆ: Chỉ chạy khi bấm RUN trực tiếp file này
if __name__ == "__main__":
    print("🚀 Bắt đầu quá trình chuẩn bị dữ liệu huấn luyện...")

    all_data = []
    image_paths = glob.glob(os.path.join(DATASET_PATH, '**/*.*'), recursive=True)
    image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for path in tqdm(image_paths, desc="Đang trích xuất GLCM"):
        features = extract_features_from_image(path)
        if features is not None:
            label_dir = os.path.basename(os.path.dirname(path))
            clean_id = ''.join(filter(str.isdigit, label_dir))

            if clean_id in LEAF_MAPPING:
                features['Label_ID'] = int(clean_id)
                all_data.append(features)

    # ==========================================
    # 4. LƯU TRỮ & CHUẨN HÓA (DÀNH RIÊNG CHO MODEL)
    # ==========================================
    if all_data:
        df = pd.DataFrame(all_data)
        joblib.dump(LEAF_MAPPING, MAPPING_FILE)
        print(f"✅ Đã lưu label mapping vào: {MAPPING_FILE}")

        X = df.drop(columns=['Label_ID'])
        y = df['Label_ID']

        df.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Đã lưu bộ dữ liệu huấn luyện ({len(df)} mẫu) vào: {OUTPUT_CSV}")

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"✅ Đã lưu bộ chuẩn hóa Scaler vào: {scaler_path}")
    else:
        print("⚠️ Không tìm thấy dữ liệu ảnh hợp lệ.")