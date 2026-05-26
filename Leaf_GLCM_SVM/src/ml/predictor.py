# src/ml/predictor.py
"""
LeafVision AI — Prediction Engine v2.0
=======================================
Nâng cấp lớn so với v1.0:

  1. ModelCache  – load model/scaler một lần duy nhất, tái sử dụng cho mọi lần predict
  2. PredictionResult (dataclass) – trả về đối tượng cấu trúc thay vì tuple 4 phần tử thô
  3. ConfidenceLevel  – enum rõ ràng (HIGH / MEDIUM / LOW) thay vì để UI tự tính
  4. ConfusionWarning – tính ngay trong ML layer, không phân tán lên Controller
  5. FeatureInsights  – diễn giải GLCM thành câu tiếng Việt ngay trong predictor
  6. PipelineStatus   – expose từng bước xử lý để UI hiển thị real-time loading
  7. ModelInfo        – metadata model (kernel, C, gamma, dataset) cho Information Panel
  8. Error handling   – phân loại 5 loại lỗi khác nhau, message rõ ràng
  9. Tích hợp LEAF_MAPPING từ feature_extraction để đồng bộ nguồn dữ liệu

Author  : LeafVision AI
Version : 2.0.0
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import nội bộ
# ---------------------------------------------------------------------------
# Thêm thư mục gốc vào sys.path để import hoạt động khi chạy trực tiếp
_BASE_DIR = Path(__file__).resolve().parents[2]
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from src.ml.feature_extraction import (
    extract_features_from_image,
    preprocess_image,
    aggregate_glcm,
    compute_haralick_features,
    StandardScalerNumPy,               # ← import để pickle resolve đúng class
    LEVELS,
    DISTANCES,
    ANGLES_DEG,
    LEAF_MAPPING as FE_LEAF_MAPPING,   # nguồn dữ liệu gốc từ training
)
from src.data.plants_db import PLANTS_DB

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Đường dẫn
# ---------------------------------------------------------------------------
MODEL_PATH  = _BASE_DIR / "models" / "leaf_svm_model.pkl"
SCALER_PATH = _BASE_DIR / "models" / "scaler.pkl"

# ---------------------------------------------------------------------------
# Hằng số ngưỡng
# ---------------------------------------------------------------------------
CONFUSION_GAP_THRESHOLD = 10.0   # % — khoảng cách top1-top2 dưới ngưỡng này = cảnh báo
HIGH_CONF_THRESHOLD     = 85.0   # % trở lên = HIGH
MEDIUM_CONF_THRESHOLD   = 60.0   # % trở lên = MEDIUM, dưới = LOW
TOP_K                   = 3      # số kết quả trả về


# ===========================================================================
#  ENUMS & DATACLASSES
# ===========================================================================

class ConfidenceLevel(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"

    @classmethod
    def from_percent(cls, pct: float) -> "ConfidenceLevel":
        if pct >= HIGH_CONF_THRESHOLD:
            return cls.HIGH
        if pct >= MEDIUM_CONF_THRESHOLD:
            return cls.MEDIUM
        return cls.LOW

    @property
    def label_vn(self) -> str:
        return {
            ConfidenceLevel.HIGH:   "Độ tin cậy cao",
            ConfidenceLevel.MEDIUM: "Độ tin cậy trung bình",
            ConfidenceLevel.LOW:    "Độ tin cậy thấp",
        }[self]

    @property
    def color_hex(self) -> str:
        return {
            ConfidenceLevel.HIGH:   "#10b981",
            ConfidenceLevel.MEDIUM: "#f59e0b",
            ConfidenceLevel.LOW:    "#f43f5e",
        }[self]


@dataclass
class ConfusionWarning:
    """Cảnh báo khi top1 và top2 quá gần nhau."""
    gap:            float   # khoảng cách % giữa top1 và top2
    runner_up_id:   str     # label_id của top2
    runner_up_name: str     # tên tiếng Việt của top2
    runner_up_pct:  float   # % của top2


@dataclass
class FeatureInsight:
    """Một lý do AI đưa ra quyết định, dùng để hiển thị trong AI Explanation Panel."""
    feature:     str   # tên đặc trưng GLCM
    value:       float # giá trị thực
    description: str   # mô tả tiếng Việt ngắn gọn


@dataclass
class PipelineStep:
    """Một bước trong quá trình xử lý — dùng cho real-time loading UI."""
    name:    str
    status:  str  # "pending" | "running" | "done" | "error"
    elapsed: float = 0.0  # giây


@dataclass
class ModelInfo:
    """Thông tin về model đã huấn luyện — dùng cho Model Information Panel."""
    classifier:        str = "SVM (RBF Kernel)"
    feature_type:      str = "GLCM Texture Features (Haralick)"
    num_features:      int = 10
    glcm_levels:       int = LEVELS
    glcm_distances:    list = field(default_factory=lambda: DISTANCES)
    glcm_angles:       list = field(default_factory=lambda: ANGLES_DEG)
    dataset:           str = "Swedish Leaf Dataset"
    num_classes:       int = 15
    kernel:            str = "rbf"
    svm_C:             float = 10.0
    svm_gamma:         str = "scale"
    is_loaded:         bool = False
    num_support_vectors: int = 0


@dataclass
class TopResult:
    """Một kết quả trong Top-K."""
    rank:       int
    label_id:   str
    name_vn:    str
    name_sci:   str
    percent:    float


@dataclass
class PredictionResult:
    """
    Toàn bộ kết quả một lần predict — đối tượng cấu trúc thay vì tuple thô.
    UI chỉ cần nhận object này và render, không cần tính toán thêm gì.
    """
    # ── Kết quả chính ──────────────────────────────────────
    best_label_id:   str
    best_name_vn:    str
    best_name_sci:   str
    best_percent:    float

    # ── Confidence ─────────────────────────────────────────
    confidence:      ConfidenceLevel

    # ── Top K ──────────────────────────────────────────────
    top_results:     List[TopResult]

    # ── Dữ liệu thực vật đầy đủ ───────────────────────────
    plant_details:   Dict

    # ── GLCM Features ──────────────────────────────────────
    features:        Dict[str, float]

    # ── AI Explanation ─────────────────────────────────────
    feature_insights: List[FeatureInsight]

    # ── Cảnh báo ───────────────────────────────────────────
    confusion_warning: Optional[ConfusionWarning]

    # ── Pipeline trace ─────────────────────────────────────
    pipeline_steps:  List[PipelineStep]

    # ── Model info ─────────────────────────────────────────
    model_info:      ModelInfo

    # ── Thời gian xử lý ────────────────────────────────────
    elapsed_ms:      float = 0.0


# ===========================================================================
#  MODEL CACHE  (singleton pattern — load một lần, dùng mãi)
# ===========================================================================

def _safe_load_scaler(path: Path):
    """
    Load scaler.pkl an toàn khi StandardScalerNumPy bị pickle với namespace
    '__main__' thay vì 'src.ml.feature_extraction'.

    TẠI SAO KHÔNG dùng pickle.Unpickler trực tiếp:
        joblib.dump() không phải pickle thuần — nó dùng format nén riêng
        (zlib/lz4). Mở bằng pickle.Unpickler → "invalid load key '\x02'"
        vì đọc nhầm header nén thành opcode pickle.

    GIẢI PHÁP — inject class vào __main__ trước khi joblib.load():
        joblib vẫn dùng pickle nội bộ để deserialize, và pickle resolve
        class qua sys.modules[module]. Ta tạm thời gắn StandardScalerNumPy
        vào sys.modules['__main__'] trước khi load, dọn sạch sau.
    """
    import sys

    _main_module = sys.modules.get("__main__")
    _had_attr    = hasattr(_main_module, "StandardScalerNumPy")
    _old_val     = getattr(_main_module, "StandardScalerNumPy", None)

    try:
        setattr(_main_module, "StandardScalerNumPy", StandardScalerNumPy)
        return joblib.load(path)   # joblib đọc đúng format nén
    finally:
        if _had_attr:
            setattr(_main_module, "StandardScalerNumPy", _old_val)
        else:
            try:
                delattr(_main_module, "StandardScalerNumPy")
            except AttributeError:
                pass


class _ModelCache:
    """
    Giữ model và scaler trong bộ nhớ sau lần load đầu tiên.
    Tránh joblib.load() tốn I/O mỗi lần predict.
    """

    def __init__(self) -> None:
        self._model  = None
        self._scaler = None
        self._info   = ModelInfo()
        self._loaded = False

    def load(self) -> Optional[str]:
        """
        Load model + scaler. Trả về None nếu thành công, chuỗi lỗi nếu thất bại.
        """
        if self._loaded:
            return None  # đã load rồi, không cần load lại

        if not MODEL_PATH.exists():
            return (
                f"⚠️  Không tìm thấy file model:\n"
                f"  {MODEL_PATH}\n\n"
                f"→ Hãy chạy train_svm.py để tạo model trước."
            )

        if not SCALER_PATH.exists():
            return (
                f"⚠️  Không tìm thấy file scaler:\n"
                f"  {SCALER_PATH}\n\n"
                f"→ Hãy chạy feature_extraction.py trước."
            )

        try:
            self._model  = joblib.load(MODEL_PATH)
            self._scaler = _safe_load_scaler(SCALER_PATH)
        except Exception as exc:
            return f"❌  Lỗi load model: {exc}"

        # Thu thập metadata từ model thực tế
        try:
            self._info.kernel             = getattr(self._model, "kernel", "rbf")
            self._info.svm_C              = getattr(self._model, "C", 10.0)
            self._info.svm_gamma          = str(getattr(self._model, "gamma", "scale"))
            self._info.num_support_vectors = int(
                sum(self._model.n_support_)
                if hasattr(self._model, "n_support_") else 0
            )
            self._info.num_classes        = len(self._model.classes_)
            self._info.is_loaded          = True
        except Exception:
            pass  # metadata không quan trọng, không crash

        self._loaded = True
        logger.info(
            "✅ Model loaded — kernel=%s | C=%s | classes=%d | SVs=%d",
            self._info.kernel,
            self._info.svm_C,
            self._info.num_classes,
            self._info.num_support_vectors,
        )
        return None

    @property
    def model(self):
        return self._model

    @property
    def scaler(self):
        return self._scaler

    @property
    def info(self) -> ModelInfo:
        return self._info

    def is_ready(self) -> bool:
        return self._loaded and self._model is not None


# Module-level singleton
_cache = _ModelCache()


def get_model_info() -> ModelInfo:
    """Trả về ModelInfo — gọi từ UI để show Model Information Panel."""
    return _cache.info


# ===========================================================================
#  FEATURE INTERPRETATION ENGINE
# ===========================================================================

def _build_feature_insights(features: Dict[str, float]) -> List[FeatureInsight]:
    """
    Diễn giải các giá trị GLCM thành câu mô tả tiếng Việt.
    Trả về tối đa 5 insight quan trọng nhất.

    Logic dựa trên ngưỡng thực nghiệm cho Swedish Leaf Dataset
    với GLCM 8-level, distance=1.
    """
    insights: List[FeatureInsight] = []

    # ── Entropy ─────────────────────────────────────────────────────────────
    entropy = features.get("Entropy", 0.0)
    if entropy < 1.5:
        desc = "Entropy rất thấp → texture cực kỳ đồng nhất và ổn định"
    elif entropy < 2.5:
        desc = "Entropy thấp → texture ổn định, ít nhiễu"
    elif entropy < 4.0:
        desc = "Entropy trung bình → texture lá có độ phức tạp vừa phải"
    else:
        desc = "Entropy cao → texture lá phức tạp, nhiều biến thiên"
    insights.append(FeatureInsight("Entropy", entropy, desc))

    # ── Contrast ────────────────────────────────────────────────────────────
    contrast = features.get("Contrast", 0.0)
    if contrast < 1.0:
        desc = "Contrast thấp → bề mặt lá mịn, gân lá ít nổi bật"
    elif contrast < 5.0:
        desc = "Contrast trung bình → gân lá nổi vừa phải"
    else:
        desc = "Contrast cao → gân lá rất nổi bật, mép lá sắc nét"
    insights.append(FeatureInsight("Contrast", contrast, desc))

    # ── Homogeneity ─────────────────────────────────────────────────────────
    homog = features.get("Homogeneity", 0.0)
    if homog > 0.85:
        desc = "Homogeneity rất cao → cấu trúc lá cực kỳ đồng đều"
    elif homog > 0.65:
        desc = "Homogeneity cao → cấu trúc lá đồng đều"
    else:
        desc = "Homogeneity thấp → cấu trúc lá không đồng đều, nhiều chi tiết"
    insights.append(FeatureInsight("Homogeneity", homog, desc))

    # ── Energy ──────────────────────────────────────────────────────────────
    energy = features.get("Energy", 0.0)
    if energy > 0.3:
        desc = "Energy cao → texture đơn giản, lặp lại đều đặn"
    elif energy > 0.1:
        desc = "Energy trung bình → texture có sự đa dạng vừa phải"
    else:
        desc = "Energy thấp → texture rất đa dạng, không lặp lại"
    insights.append(FeatureInsight("Energy", energy, desc))

    # ── Correlation ─────────────────────────────────────────────────────────
    corr = features.get("Correlation", 0.0)
    if corr > 0.9:
        desc = "Correlation rất cao → gân lá song song, có hướng rõ ràng"
    elif corr > 0.7:
        desc = "Correlation cao → gân lá có hướng nhất định"
    else:
        desc = "Correlation thấp → gân lá phân nhánh phức tạp"
    insights.append(FeatureInsight("Correlation", corr, desc))

    return insights


# ===========================================================================
#  PIPELINE EXECUTOR  (với callback để UI cập nhật từng bước)
# ===========================================================================

def _run_pipeline(
    img_path: str,
    on_step: Optional[Callable[[PipelineStep], None]] = None,
) -> Tuple[Optional[Dict[str, float]], List[PipelineStep], Optional[str]]:
    """
    Chạy toàn bộ pipeline xử lý ảnh và trả về:
      - features dict  (None nếu lỗi)
      - danh sách PipelineStep đã hoàn thành
      - error message  (None nếu thành công)

    Parameters
    ----------
    img_path  : đường dẫn ảnh đầu vào
    on_step   : callback tùy chọn, gọi sau mỗi bước hoàn thành.
                Dùng để UI cập nhật trạng thái real-time.
    """
    steps: List[PipelineStep] = []

    def _step(name: str, fn):
        """Chạy một bước, đo thời gian, cập nhật status và gọi callback."""
        step = PipelineStep(name=name, status="running")
        t0 = time.perf_counter()
        try:
            result = fn()
            step.status  = "done"
            step.elapsed = round(time.perf_counter() - t0, 4)
            steps.append(step)
            if on_step:
                on_step(step)
            return result, None
        except Exception as exc:
            step.status  = "error"
            step.elapsed = round(time.perf_counter() - t0, 4)
            steps.append(step)
            if on_step:
                on_step(step)
            return None, str(exc)

    # ── Bước 1: Đọc và chuyển ảnh sang Grayscale ────────────────────────────
    img_quantized, err = _step(
        "Chuyển Grayscale & Lượng tử hoá",
        lambda: preprocess_image(img_path),
    )
    if err or img_quantized is None:
        return None, steps, f"❌ Không thể đọc hoặc xử lý ảnh:\n{img_path}"

    # ── Bước 2: Xây dựng ma trận GLCM ──────────────────────────────────────
    glcm_matrix, err = _step(
        "Xây dựng ma trận GLCM",
        lambda: aggregate_glcm(
            img_quantized,
            levels=LEVELS,
            distances=DISTANCES,
            angles_deg=ANGLES_DEG,
        ),
    )
    if err or glcm_matrix is None:
        return None, steps, f"❌ Lỗi tạo GLCM matrix: {err}"

    # ── Bước 3: Trích xuất 10 đặc trưng Haralick ───────────────────────────
    haralick_obj, err = _step(
        "Trích xuất 10 đặc trưng Haralick",
        lambda: compute_haralick_features(glcm_matrix),
    )
    if err or haralick_obj is None:
        return None, steps, f"❌ Lỗi tính đặc trưng GLCM: {err}"

    features_dict = haralick_obj.to_dict()

    # ── Bước 4: Chuẩn hoá đặc trưng ────────────────────────────────────────
    df_raw = pd.DataFrame([features_dict])
    scaled_arr, err = _step(
        "Chuẩn hoá z-score (StandardScaler)",
        lambda: _cache.scaler.transform(df_raw),
    )
    if err or scaled_arr is None:
        return None, steps, f"❌ Lỗi chuẩn hoá đặc trưng: {err}"

    # Lưu scaled features vào features_dict để predictor dùng
    features_dict["_scaled"] = scaled_arr   # internal, UI không thấy

    return features_dict, steps, None


# ===========================================================================
#  CORE PREDICT FUNCTION
# ===========================================================================

def predict_new_image(
    img_path: str,
    on_step: Optional[Callable[[PipelineStep], None]] = None,
) -> Tuple[Optional[PredictionResult], Optional[str]]:
    """
    Pipeline đầy đủ: đọc ảnh → GLCM → chuẩn hoá → SVM → trả kết quả.

    Parameters
    ----------
    img_path  : đường dẫn ảnh đầu vào
    on_step   : callback tùy chọn được gọi sau mỗi bước pipeline.
                Signature: fn(PipelineStep) -> None

    Returns
    -------
    (PredictionResult, None)  nếu thành công
    (None, error_message)     nếu thất bại

    Notes
    -----
    Thay đổi so với v1.0:
    - Trả về (result, error) thay vì (best, top_probs, features, error)
    - result là PredictionResult — đối tượng cấu trúc đầy đủ
    - Model chỉ load một lần nhờ _ModelCache
    """
    t_start = time.perf_counter()

    # ── 0. Load model (no-op nếu đã load) ──────────────────────────────────
    err = _cache.load()
    if err:
        return None, err

    # ── 1. Chạy pipeline trích xuất đặc trưng ──────────────────────────────
    features_dict, pipeline_steps, err = _run_pipeline(img_path, on_step)
    if err:
        return None, err

    # Lấy scaled array ra khỏi dict trước khi trả cho UI
    features_scaled = features_dict.pop("_scaled")

    # ── 2. Phân loại SVM ─────────────────────────────────────────────────────
    svm_step = PipelineStep(name="Phân loại SVM (RBF Kernel)", status="running")
    t_svm = time.perf_counter()
    try:
        prob_dist = _cache.model.predict_proba(features_scaled)[0]
        class_ids = _cache.model.classes_
        svm_step.status  = "done"
        svm_step.elapsed = round(time.perf_counter() - t_svm, 4)
    except Exception as exc:
        svm_step.status = "error"
        pipeline_steps.append(svm_step)
        return None, f"❌ Lỗi SVM predict: {exc}"
    pipeline_steps.append(svm_step)
    if on_step:
        on_step(svm_step)

    # ── 3. Sắp xếp kết quả Top-K ─────────────────────────────────────────────
    prob_pairs = sorted(
        zip(class_ids, prob_dist),
        key=lambda x: x[1],
        reverse=True,
    )

    top_results: List[TopResult] = []
    for rank, (cid, prob) in enumerate(prob_pairs[:TOP_K], start=1):
        sid = str(cid)
        db_entry  = PLANTS_DB.get(sid, {})
        fe_entry  = FE_LEAF_MAPPING.get(sid, {})
        name_vn   = db_entry.get("name_vn") or fe_entry.get("vn") or "Không rõ"
        name_sci  = db_entry.get("scientific_name") or fe_entry.get("en") or ""
        top_results.append(TopResult(
            rank     = rank,
            label_id = sid,
            name_vn  = name_vn,
            name_sci = name_sci,
            percent  = round(prob * 100, 2),
        ))

    # ── 4. Lấy thông tin top 1 ───────────────────────────────────────────────
    best         = top_results[0]
    plant_details = PLANTS_DB.get(best.label_id, {
        "name_vn":              best.name_vn,
        "scientific_name":      best.name_sci,
        "family":               "—",
        "description":          "Chưa cập nhật dữ liệu chi tiết.",
        "habitat":              "—",
        "leaf_type":            "—",
        "leaf_shape":           "—",
        "leaf_margin":          "—",
        "leaf_texture":         "—",
        "medical_uses":         "—",
        "toxicity":             "—",
        "average_height":       "—",
        "lifespan":             "—",
        "conservation_status":  "—",
        "ai_features":          {},
    })

    # ── 5. Confidence level ──────────────────────────────────────────────────
    confidence = ConfidenceLevel.from_percent(best.percent)

    # ── 6. Confusion warning ─────────────────────────────────────────────────
    confusion_warning: Optional[ConfusionWarning] = None
    if len(top_results) >= 2:
        gap = best.percent - top_results[1].percent
        if gap < CONFUSION_GAP_THRESHOLD:
            runner = top_results[1]
            confusion_warning = ConfusionWarning(
                gap            = round(gap, 2),
                runner_up_id   = runner.label_id,
                runner_up_name = runner.name_vn,
                runner_up_pct  = runner.percent,
            )

    # ── 7. Feature insights (AI Explanation) ────────────────────────────────
    feature_insights = _build_feature_insights(features_dict)

    # ── 8. Tổng hợp kết quả ─────────────────────────────────────────────────
    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

    result = PredictionResult(
        best_label_id    = best.label_id,
        best_name_vn     = best.name_vn,
        best_name_sci    = best.name_sci,
        best_percent     = best.percent,
        confidence       = confidence,
        top_results      = top_results,
        plant_details    = plant_details,
        features         = features_dict,
        feature_insights = feature_insights,
        confusion_warning= confusion_warning,
        pipeline_steps   = pipeline_steps,
        model_info       = _cache.info,
        elapsed_ms       = elapsed_ms,
    )

    logger.info(
        "✅ Predict hoàn thành — %s (%.1f%%) | confidence=%s | %.1fms",
        best.name_vn,
        best.percent,
        confidence.value,
        elapsed_ms,
    )

    return result, None


# ===========================================================================
#  BACKWARD COMPATIBILITY  (giữ để app.py cũ không bị crash ngay)
# ===========================================================================

def predict_new_image_legacy(
    img_path: str,
) -> Tuple[Optional[dict], Optional[list], Optional[dict], Optional[str]]:
    """
    Wrapper tương thích ngược với API cũ (v1.0):
    Trả về (best_match, top_probabilities, features, error)

    ⚠️  Deprecated — dùng predict_new_image() cho code mới.
    """
    result, error = predict_new_image(img_path)
    if error:
        return None, None, None, error

    best_match = {
        "label_id": result.best_label_id,
        "vn":       result.best_name_vn,
        "percent":  result.best_percent,
    }
    top_probabilities = [
        {
            "label_id": r.label_id,
            "vn":       r.name_vn,
            "percent":  r.percent,
        }
        for r in result.top_results
    ]
    return best_match, top_probabilities, result.features, None