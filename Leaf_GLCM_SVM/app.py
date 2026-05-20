# app.py

import os
from tkinter import filedialog, messagebox
from PIL import Image

import customtkinter as ctk

# Import Layer Giao Diện (View) và Dữ Liệu/AI (Model)
from src.gui.main_window import LeafVisionUI
from src.data.plants_db import PLANTS_DB
from src.ml.predictor import predict_new_image


class LeafVisionController:
    def __init__(self):
        # Khởi tạo giao diện độc lập và đẩy các hàm xử lý sự kiện qua tham số
        self.ui = LeafVisionUI(
            on_select_image=self.handle_select_image,
            on_run_ai=self.handle_run_ai,
            on_search=self.handle_search_filter
        )

        self.current_file_path = None
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # Đổ dữ liệu từ điển mặc định lên giao diện lúc khởi động app
        self.ui.render_db_list(PLANTS_DB, self.base_dir)

    def handle_select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp")])
        if not file_path:
            return

        self.current_file_path = file_path

        try:
            # Đoạn xử lý RAM ảnh lớn an toàn
            pil_img = Image.open(file_path)
            pil_img.thumbnail((400, 350))

            # Gửi đối tượng ảnh sạch sang cho UI hiển thị
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
            self.ui.update_image_preview(ctk_img, os.path.basename(file_path))
        except Exception as e:
            messagebox.showerror("Lỗi đọc tập tin", f"Không thể giải mã hình ảnh này: {str(e)}")

    def handle_run_ai(self):
        if not self.current_file_path:
            return

        # Ra lệnh cho UI hiển thị màn hình chờ loading
        self.ui.show_loading_status()

        # Thực thi xử lý thuật toán Máy học (Offline Core AI)
        best_match, top_probs, features, error = predict_new_image(self.current_file_path)

        if error:
            self.ui.show_error_status(error)
            return

        # Ánh xạ từ ID AI sang thông tin khoa học trong Hard Data
        label_id = str(best_match.get('label_id', ''))
        plant_details = PLANTS_DB.get(label_id, {
            "name_vn": "Không rõ tên loài",
            "description": "Chưa cập nhật dữ liệu",
            "habitat": "Không rõ"
        })

        # Đẩy toàn bộ cục dữ liệu đã tính toán xong xuôi ra UI để render
        self.ui.display_ai_results(
            name_vn=plant_details['name_vn'],
            accuracy=best_match['percent'],
            habitat=plant_details['habitat'],
            description=plant_details['description'],
            top_probs=top_probs,
            features=features
        )

    def handle_search_filter(self, search_text):
        query = search_text.strip().lower()
        if not query:
            self.ui.render_db_list(PLANTS_DB, self.base_dir)
            return

        # Xử lý thuật toán tìm kiếm cục bộ (0ms Delay)
        filtered_dict = {}
        for cid, info in PLANTS_DB.items():
            if query in info['name_vn'].lower() or query == cid:
                filtered_dict[cid] = info

        # Đẩy kết quả lọc được ra lại màn hình
        self.ui.render_db_list(filtered_dict, self.base_dir)

    def run(self):
        # Chạy vòng lặp ứng dụng Desktop
        self.ui.mainloop()


if __name__ == "__main__":
    app = LeafVisionController()
    app.run()