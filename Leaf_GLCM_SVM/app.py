# Leaf_GLCM_SVM/app.py
# ============================================================
# Controller (MVC) — Nâng cấp v2.0
# Sử dụng PredictionResult từ predictor v2.0
# ============================================================

import os
from tkinter import filedialog, messagebox
from PIL import Image

import customtkinter as ctk

from src.gui.main_window import LeafVisionUI
from src.data.plants_db import PLANTS_DB
from src.ml.predictor import predict_new_image, get_model_info


class LeafVisionController:
    def __init__(self):
        self.ui = LeafVisionUI(
            on_select_image=self.handle_select_image,
            on_run_ai=self.handle_run_ai,
            on_search=self.handle_search_filter
        )

        self.current_file_path = None
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        self.ui.render_db_list(PLANTS_DB, self.base_dir)

    # ----------------------------------------------------------
    def handle_select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp")]
        )
        if not file_path:
            return

        self.current_file_path = file_path

        try:
            pil_img = Image.open(file_path)
            pil_img.thumbnail((400, 350))
            ctk_img = ctk.CTkImage(
                light_image=pil_img,
                dark_image=pil_img,
                size=pil_img.size
            )
            self.ui.update_image_preview(ctk_img, os.path.basename(file_path))
        except Exception as e:
            messagebox.showerror("Lỗi đọc tập tin",
                                 f"Không thể giải mã hình ảnh này: {str(e)}")

    # ----------------------------------------------------------
    def handle_run_ai(self):
        if not self.current_file_path:
            return

        self.ui.show_loading_status()

        # ── Gọi predict với callback cập nhật từng bước ──────
        def on_step(step):
            """Được gọi sau mỗi bước pipeline — cập nhật UI real-time."""
            self.ui.update_pipeline_step(step)
            self.ui.update()   # force redraw

        result, error = predict_new_image(
            self.current_file_path,
            on_step=on_step
        )

        if error:
            self.ui.show_error_status(error)
            return

        # ── Đẩy toàn bộ PredictionResult xuống UI ────────────
        self.ui.display_ai_results(result)

    # ----------------------------------------------------------
    def handle_search_filter(self, search_text):
        query = search_text.strip().lower()
        if not query:
            self.ui.render_db_list(PLANTS_DB, self.base_dir)
            return

        filtered_dict = {}
        for cid, info in PLANTS_DB.items():
            if (
                query in info.get("name_vn", "").lower()
                or query == cid
                or query in info.get("scientific_name", "").lower()
                or query in info.get("family", "").lower()
                or query in info.get("habitat", "").lower()
                or query in info.get("medical_uses", "").lower()
            ):
                filtered_dict[cid] = info

        self.ui.render_db_list(filtered_dict, self.base_dir)

    # ----------------------------------------------------------
    def run(self):
        self.ui.mainloop()


if __name__ == "__main__":
    app = LeafVisionController()
    app.run()