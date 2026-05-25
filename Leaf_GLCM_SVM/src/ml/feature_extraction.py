# src/ml/feature_extraction.py
"""
GLCM-based Texture Feature Extraction for Leaf Recognition
===========================================================
Trích xuất 10 đặc trưng kết cấu Haralick từ ma trận đồng xuất hiện cấp độ xám
(Gray-Level Co-occurrence Matrix - GLCM) hoàn toàn bằng NumPy thuần.

Các đặc trưng được trích xuất (Haralick, 1973):
    1.  Contrast         – Đo mức độ tương phản cục bộ
    2.  Dissimilarity    – Tương tự contrast nhưng tuyến tính
    3.  Homogeneity      – Đo tính đồng nhất của kết cấu
    4.  ASM              – Angular Second Moment (độ đồng đều)
    5.  Energy           – Căn bậc hai của ASM
    6.  Entropy          – Độ hỗn loạn / phức tạp của kết cấu
    7.  Max Probability  – Cặp pixel phổ biến nhất
    8.  Correlation      – Tương quan tuyến tính giữa các cặp pixel
    9.  GLCM Mean        – Giá trị trung bình phân phối GLCM
    10. GLCM Variance    – Phương sai phân phối GLCM

Author  : Your Name
Version : 2.0.0
"""

from __future__ import annotations

import glob
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Đường dẫn dự án
# ---------------------------------------------------------------------------
# feature_extraction.py nằm tại: src/ml/feature_extraction.py
# BASE_DIR = thư mục gốc dự án (Leaf_GLCM_SVM)
BASE_DIR   = Path(__file__).resolve().parents[2]
DATASET_PATH  = BASE_DIR / "dataset" / "swedish_leaf_dataset" / "Swedish" / "Train"
OUTPUT_DIR    = BASE_DIR / "models"
OUTPUT_CSV    = BASE_DIR / "dataset" / "leaf_features_training.csv"
MAPPING_FILE  = OUTPUT_DIR / "label_mapping.pkl"
SCALER_FILE   = OUTPUT_DIR / "scaler.pkl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hằng số GLCM
# ---------------------------------------------------------------------------
TARGET_SIZE: Tuple[int, int] = (256, 256)   # (width, height)
LEVELS: int = 8                              # số cấp độ xám sau lượng tử hoá
DISTANCES: List[int] = [1]                  # khoảng cách pixel
ANGLES_DEG: List[int] = [0, 45, 90, 135]   # 4 hướng để bất biến hướng

# ---------------------------------------------------------------------------
# Bản đồ nhãn lá cây (Swedish Leaf Dataset)
# ---------------------------------------------------------------------------
LEAF_MAPPING: Dict[str, Dict[str, str]] = {
    "0":  {"en": "Ulmus carpinifolia",    "vn": "Cây Du"},
    "1":  {"en": "Acer",                  "vn": "Cây Phong"},
    "2":  {"en": "Salix aurita",          "vn": "Cây Liễu tai chuột"},
    "3":  {"en": "Quercus",               "vn": "Cây Sồi"},
    "4":  {"en": "Alnus incana",          "vn": "Cây Tống quán sủ xám"},
    "5":  {"en": "Betula pubescens",      "vn": "Cây Bạch dương lông"},
    "6":  {"en": "Salix alba 'Tristis'",  "vn": "Cây Liễu rủ"},
    "7":  {"en": "Populus tremula",       "vn": "Cây Bạch dương rung"},
    "8":  {"en": "Ulmus glabra",          "vn": "Cây Du núi"},
    "9":  {"en": "Sorbus aucuparia",      "vn": "Cây Thanh lương trà"},
    "10": {"en": "Salix cinerea",         "vn": "Cây Liễu xám"},
    "11": {"en": "Populus",               "vn": "Cây Dương / Bạch dương"},
    "12": {"en": "Tilia",                 "vn": "Cây Đoạn"},
    "13": {"en": "Sorbus intermedia",     "vn": "Cây Thanh lương trà Thụy Điển"},
    "14": {"en": "Fagus silvatica",       "vn": "Cây Dẻ gai châu Âu"},
}

# ---------------------------------------------------------------------------
# Dataclass kết quả đặc trưng
# ---------------------------------------------------------------------------
@dataclass
class GLCMFeatures:
    """
    Chứa 10 đặc trưng Haralick trích xuất từ GLCM.
    Tất cả giá trị được tính trên GLCM chuẩn hoá (xác suất),
    tổng hợp qua nhiều hướng để đạt tính bất biến hướng.
    """
    contrast:        float = 0.0   # 1. Tương phản
    dissimilarity:   float = 0.0   # 2. Độ bất tương đồng
    homogeneity:     float = 0.0   # 3. Tính đồng nhất
    asm:             float = 0.0   # 4. Angular Second Moment
    energy:          float = 0.0   # 5. Năng lượng
    entropy:         float = 0.0   # 6. Entropy
    max_probability: float = 0.0   # 7. Xác suất tối đa
    correlation:     float = 0.0   # 8. Hệ số tương quan
    glcm_mean:       float = 0.0   # 9. Giá trị trung bình
    glcm_variance:   float = 0.0   # 10. Phương sai

    def to_dict(self) -> Dict[str, float]:
        return {
            "Contrast":        self.contrast,
            "Dissimilarity":   self.dissimilarity,
            "Homogeneity":     self.homogeneity,
            "ASM":             self.asm,
            "Energy":          self.energy,
            "Entropy":         self.entropy,
            "Max_Probability": self.max_probability,
            "Correlation":     self.correlation,
            "GLCM_Mean":       self.glcm_mean,
            "GLCM_Variance":   self.glcm_variance,
        }

# ---------------------------------------------------------------------------
# Xây dựng GLCM thuần NumPy
# ---------------------------------------------------------------------------

def _angle_to_offset(angle_deg: int, distance: int) -> Tuple[int, int]:
    """
    Chuyển góc (độ) và khoảng cách thành vector dịch chuyển pixel (dr, dc).

    Quy ước:
        0°   → (0,  +d)   – hướng ngang phải
        45°  → (-d, +d)   – hướng chéo phải-trên
        90°  → (-d,  0)   – hướng thẳng đứng lên
        135° → (-d, -d)   – hướng chéo trái-trên
    """
    angle_map = {
        0:   ( 0,  distance),
        45:  (-distance,  distance),
        90:  (-distance,  0),
        135: (-distance, -distance),
    }
    if angle_deg not in angle_map:
        raise ValueError(
            f"Góc không hợp lệ: {angle_deg}°. Chọn trong {list(angle_map.keys())}."
        )
    return angle_map[angle_deg]


def build_glcm(
    image: np.ndarray,
    levels: int,
    distance: int = 1,
    angle_deg: int = 0,
    symmetric: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Xây dựng ma trận đồng xuất hiện cấp độ xám (GLCM) thuần NumPy.

    Thay vì dùng vòng lặp for (chậm), ta khai thác indexing vector hoá:
    lấy toàn bộ cặp pixel (p, q) theo hướng xác định trong một phép toán
    mảng duy nhất, rồi dùng np.add.at để tích luỹ vào GLCM.

    Parameters
    ----------
    image       : ảnh xám đã lượng tử hoá, dtype uint8, giá trị ∈ [0, levels-1]
    levels      : số cấp độ xám (GLCM sẽ là ma trận levels×levels)
    distance    : khoảng cách giữa hai pixel
    angle_deg   : hướng xét (0, 45, 90, 135)
    symmetric   : nếu True, cộng thêm GLCM chuyển vị (GLCM đối xứng)
    normalize   : nếu True, chuẩn hoá GLCM thành phân phối xác suất

    Returns
    -------
    glcm : np.ndarray shape (levels, levels), dtype float64
    """
    rows, cols = image.shape
    dr, dc = _angle_to_offset(angle_deg, distance)

    # Xác định vùng pixel nguồn hợp lệ sao cho pixel đích không ra ngoài biên
    r_start = max(0, -dr)
    r_end   = min(rows, rows - dr)
    c_start = max(0, -dc)
    c_end   = min(cols, cols - dc)

    # Trích cặp pixel (p, q) bằng slicing
    src = image[r_start:r_end, c_start:c_end]                   # pixel nguồn
    dst = image[r_start + dr : r_end + dr, c_start + dc : c_end + dc]  # pixel đích

    # Tích luỹ vào GLCM bằng np.add.at (tránh race condition)
    glcm = np.zeros((levels, levels), dtype=np.float64)
    flat_src = src.ravel()
    flat_dst = dst.ravel()
    np.add.at(glcm, (flat_src, flat_dst), 1)

    if symmetric:
        glcm = glcm + glcm.T

    if normalize:
        total = glcm.sum()
        if total > 0:
            glcm /= total

    return glcm


def aggregate_glcm(
    image: np.ndarray,
    levels: int,
    distances: List[int],
    angles_deg: List[int],
) -> np.ndarray:
    """
    Tổng hợp GLCM trên nhiều khoảng cách và nhiều hướng.
    Trả về GLCM trung bình chuẩn hoá – cách tiếp cận chuẩn để
    đạt tính bất biến hướng (rotation-invariant) trong phân loại kết cấu.

    Parameters
    ----------
    image      : ảnh xám đã lượng tử hoá
    levels     : số cấp độ xám
    distances  : danh sách khoảng cách pixel
    angles_deg : danh sách góc (độ)

    Returns
    -------
    glcm_avg : np.ndarray shape (levels, levels), dtype float64, tổng = 1.0
    """
    glcm_sum = np.zeros((levels, levels), dtype=np.float64)
    count = 0

    for d in distances:
        for angle in angles_deg:
            glcm_sum += build_glcm(
                image,
                levels=levels,
                distance=d,
                angle_deg=angle,
                symmetric=True,
                normalize=False,   # chuẩn hoá sau khi tổng hợp
            )
            count += 1

    total = glcm_sum.sum()
    if total > 0:
        glcm_sum /= total

    return glcm_sum

# ---------------------------------------------------------------------------
# Tính 10 đặc trưng Haralick từ GLCM
# ---------------------------------------------------------------------------

def compute_haralick_features(P: np.ndarray) -> GLCMFeatures:
    """
    Tính 10 đặc trưng Haralick từ ma trận GLCM đã chuẩn hoá P.

    Tất cả công thức theo Haralick et al. (1973) và
    Soh & Tsatsoulis (1999).

    Parameters
    ----------
    P : np.ndarray shape (L, L), float64, ∑P = 1.0

    Returns
    -------
    GLCMFeatures
    """
    L = P.shape[0]
    eps = 1e-10  # tránh chia zero và log(0)

    # Chỉ số i, j (lưới toàn bộ GLCM) — tính sẵn để tái sử dụng
    i_idx, j_idx = np.indices((L, L))   # shape (L, L)
    diff = i_idx - j_idx                # (i - j)

    # ------------------------------------------------------------------
    # 1. Contrast  ∑ P(i,j) · (i-j)²
    # ------------------------------------------------------------------
    contrast = float(np.sum(P * (diff ** 2)))

    # ------------------------------------------------------------------
    # 2. Dissimilarity  ∑ P(i,j) · |i-j|
    # ------------------------------------------------------------------
    dissimilarity = float(np.sum(P * np.abs(diff)))

    # ------------------------------------------------------------------
    # 3. Homogeneity (Inverse Difference Moment)  ∑ P(i,j) / (1 + (i-j)²)
    # ------------------------------------------------------------------
    homogeneity = float(np.sum(P / (1.0 + diff ** 2)))

    # ------------------------------------------------------------------
    # 4. Angular Second Moment (ASM)  ∑ P(i,j)²
    # ------------------------------------------------------------------
    asm = float(np.sum(P ** 2))

    # ------------------------------------------------------------------
    # 5. Energy  √ASM
    # ------------------------------------------------------------------
    energy = float(np.sqrt(asm))

    # ------------------------------------------------------------------
    # 6. Entropy  -∑ P(i,j) · log₂(P(i,j))   [chỉ tính trên P > 0]
    # ------------------------------------------------------------------
    mask = P > 0
    entropy = float(-np.sum(P[mask] * np.log2(P[mask] + eps)))

    # ------------------------------------------------------------------
    # 7. Max Probability  max(P)
    # ------------------------------------------------------------------
    max_probability = float(np.max(P))

    # ------------------------------------------------------------------
    # 8. Correlation
    #    μᵢ = ∑ i · P(i,j),  σᵢ² = ∑ P(i,j)·(i - μᵢ)²
    #    (tương tự j nhưng do GLCM đối xứng nên μᵢ = μⱼ, σᵢ = σⱼ)
    #    Corr = [∑ P(i,j)·(i-μᵢ)·(j-μⱼ)] / (σᵢ · σⱼ)
    # ------------------------------------------------------------------
    mu_i = float(np.sum(i_idx * P))
    mu_j = float(np.sum(j_idx * P))

    var_i = float(np.sum(P * (i_idx - mu_i) ** 2))
    var_j = float(np.sum(P * (j_idx - mu_j) ** 2))

    std_i = np.sqrt(var_i)
    std_j = np.sqrt(var_j)

    if std_i < eps or std_j < eps:
        # Kết cấu hoàn toàn đồng nhất → tương quan hoàn hảo
        correlation = 1.0
    else:
        cov = float(np.sum(P * (i_idx - mu_i) * (j_idx - mu_j)))
        correlation = cov / (std_i * std_j + eps)

    # ------------------------------------------------------------------
    # 9. GLCM Mean  μᵢ  (giá trị trung bình theo chiều i)
    # ------------------------------------------------------------------
    glcm_mean = mu_i

    # ------------------------------------------------------------------
    # 10. GLCM Variance  σᵢ²
    # ------------------------------------------------------------------
    glcm_variance = var_i

    return GLCMFeatures(
        contrast=contrast,
        dissimilarity=dissimilarity,
        homogeneity=homogeneity,
        asm=asm,
        energy=energy,
        entropy=entropy,
        max_probability=max_probability,
        correlation=correlation,
        glcm_mean=glcm_mean,
        glcm_variance=glcm_variance,
    )

# ---------------------------------------------------------------------------
# Pipeline trích xuất đặc trưng từ một ảnh
# ---------------------------------------------------------------------------

def preprocess_image(img_path: str | Path) -> Optional[np.ndarray]:
    """
    Đọc ảnh → chuyển xám → resize → lượng tử hoá về [0, LEVELS-1].

    Parameters
    ----------
    img_path : đường dẫn ảnh (jpg / jpeg / png)

    Returns
    -------
    img_quantized : np.ndarray uint8, hoặc None nếu đọc thất bại
    """
    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        logger.warning("Không đọc được ảnh: %s", img_path)
        return None

    img_resized = cv2.resize(img_gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # Lượng tử hoá đồng đều: [0,255] → [0, LEVELS-1]
    # Dùng right-shift bit: chia 256 cho LEVELS=8 → shift 5 bit
    bin_width = 256 // LEVELS
    img_quantized = (img_resized // bin_width).clip(0, LEVELS - 1).astype(np.uint8)

    return img_quantized


def extract_features_from_image(
    img_path: str | Path,
    levels: int = LEVELS,
    distances: List[int] = DISTANCES,
    angles_deg: List[int] = ANGLES_DEG,
) -> Optional[Dict[str, float]]:
    """
    Pipeline đầy đủ: đọc ảnh → tiền xử lý → xây GLCM → tính 10 đặc trưng.

    Parameters
    ----------
    img_path   : đường dẫn ảnh
    levels     : số cấp độ xám
    distances  : danh sách khoảng cách GLCM
    angles_deg : danh sách góc (độ)

    Returns
    -------
    dict chứa 10 đặc trưng, hoặc None nếu thất bại
    """
    img_quantized = preprocess_image(img_path)
    if img_quantized is None:
        return None

    # Xây GLCM tổng hợp bất biến hướng
    P = aggregate_glcm(img_quantized, levels=levels,
                       distances=distances, angles_deg=angles_deg)

    # Tính 10 đặc trưng Haralick
    features = compute_haralick_features(P)

    return features.to_dict()

# ---------------------------------------------------------------------------
# Chuẩn hoá StandardScaler thuần NumPy (không dùng sklearn)
# ---------------------------------------------------------------------------

class StandardScalerNumPy:
    """
    Chuẩn hoá z-score thuần NumPy.
    z = (x - μ) / σ    với σ tính trên tập huấn luyện.

    Tương đương sklearn.preprocessing.StandardScaler nhưng không phụ thuộc.
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_:  Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScalerNumPy":
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0, ddof=0)
        self.std_  = np.where(self.std_ == 0, 1.0, self.std_)  # tránh chia 0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler chưa được fit. Gọi fit() trước.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        return X_scaled * self.std_ + self.mean_

# ---------------------------------------------------------------------------
# Chương trình chính
# ---------------------------------------------------------------------------

def collect_image_paths(dataset_path: Path) -> List[Path]:
    """Quét đệ quy và lọc ảnh hợp lệ."""
    valid_exts = {".jpg", ".jpeg", ".png"}
    paths = [
        p for p in dataset_path.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_exts
    ]
    return paths


def extract_label_id(img_path: Path) -> Optional[str]:
    """
    Lấy ID nhãn từ tên thư mục cha (chỉ giữ lại ký tự số).
    Ví dụ: thư mục "class_03" → "3"
    """
    raw = img_path.parent.name
    digits = "".join(filter(str.isdigit, raw))
    return digits if digits in LEAF_MAPPING else None


def main() -> None:
    logger.info("=" * 60)
    logger.info("TRÍCH XUẤT ĐẶC TRƯNG GLCM – NHẬN DIỆN KẾT CẤU LÁ CÂY")
    logger.info("=" * 60)
    logger.info("Bộ dữ liệu : %s", DATASET_PATH)
    logger.info("Cấp độ xám : %d | Khoảng cách: %s | Góc: %s°",
                LEVELS, DISTANCES, ANGLES_DEG)

    # 1. Thu thập ảnh
    image_paths = collect_image_paths(DATASET_PATH)
    logger.info("Tìm thấy %d ảnh.", len(image_paths))

    if not image_paths:
        logger.error("Không có ảnh hợp lệ trong: %s", DATASET_PATH)
        sys.exit(1)

    # 2. Trích xuất đặc trưng
    records: List[Dict] = []
    skipped = 0

    for img_path in tqdm(image_paths, desc="Trích xuất GLCM", unit="ảnh"):
        label_id = extract_label_id(img_path)
        if label_id is None:
            skipped += 1
            continue

        feat = extract_features_from_image(img_path)
        if feat is None:
            skipped += 1
            continue

        feat["Label_ID"] = int(label_id)
        records.append(feat)

    logger.info("Thành công: %d ảnh | Bỏ qua: %d ảnh", len(records), skipped)

    if not records:
        logger.error("Không có dữ liệu hợp lệ để lưu.")
        sys.exit(1)

    # 3. Tạo DataFrame
    df = pd.DataFrame(records)

    feature_cols = [c for c in df.columns if c != "Label_ID"]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["Label_ID"].to_numpy()

    logger.info("Các đặc trưng: %s", feature_cols)
    logger.info("Phân phối nhãn:\n%s",
                df["Label_ID"].value_counts().sort_index().to_string())

    # 4. Chuẩn hoá z-score (thuần NumPy)
    scaler = StandardScalerNumPy()
    X_scaled = scaler.fit_transform(X)

    df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    df_scaled["Label_ID"] = y

    # 5. Lưu kết quả
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info("✅ Đã lưu CSV đặc trưng (%d mẫu × %d đặc trưng): %s",
                len(df), len(feature_cols), OUTPUT_CSV)

    joblib.dump(LEAF_MAPPING, MAPPING_FILE)
    logger.info("✅ Đã lưu label mapping: %s", MAPPING_FILE)

    joblib.dump(scaler, SCALER_FILE)
    logger.info("✅ Đã lưu StandardScalerNumPy: %s", SCALER_FILE)

    logger.info("=" * 60)
    logger.info("HOÀN THÀNH — sẵn sàng cho bước huấn luyện SVM.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
