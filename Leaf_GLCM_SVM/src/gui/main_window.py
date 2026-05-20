# src/gui/main_window.py

import os
import customtkinter as ctk
from PIL import Image

# =========================================================
# GLOBAL UI CONFIG
# =========================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class LeafVisionUI(ctk.CTk):

    def __init__(self, on_select_image, on_run_ai, on_search):
        super().__init__()

        # =====================================================
        # CALLBACKS
        # =====================================================

        self.on_select_image = on_select_image
        self.on_run_ai = on_run_ai
        self.on_search = on_search

        # =====================================================
        # WINDOW
        # =====================================================

        self.title("LeafVision AI - Hệ Thống Phân Tích Hình Thái Lá Cây")
        self.geometry("1180x760")
        self.minsize(1050, 680)

        self.configure(fg_color="#020617")

        # =====================================================
        # TAB VIEW
        # =====================================================

        self.tabview = ctk.CTkTabview(
            self,
            fg_color="#020617",
            segmented_button_fg_color="#0f172a",
            segmented_button_selected_color="#10b981",
            segmented_button_selected_hover_color="#059669",
            segmented_button_unselected_color="#111827",
            segmented_button_unselected_hover_color="#1e293b",
            corner_radius=18,
            border_width=1,
            border_color="#1e293b"
        )

        self.tabview._segmented_button.configure(
            font=("Inter", 14, "bold"),
            height=42
        )

        self.tabview.pack(
            padx=18,
            pady=18,
            fill="both",
            expand=True
        )

        self.tab_scan = self.tabview.add("🔍 Quét Lá Phân Tích")
        self.tab_db = self.tabview.add("📚 Từ Điển Thực Vật")

        self.tab_scan.configure(fg_color="#020617")
        self.tab_db.configure(fg_color="#020617")

        self._init_scan_tab_layout()
        self._init_db_tab_layout()

    # =========================================================
    # TAB QUÉT LÁ
    # =========================================================

    def _init_scan_tab_layout(self):

        self.tab_scan.grid_columnconfigure(0, weight=4, minsize=420)
        self.tab_scan.grid_columnconfigure(1, weight=5, minsize=500)
        self.tab_scan.grid_rowconfigure(0, weight=1)

        # =====================================================
        # LEFT PANEL
        # =====================================================

        left_frame = ctk.CTkFrame(
            self.tab_scan,
            fg_color="transparent"
        )

        left_frame.grid(
            row=0,
            column=0,
            padx=(18, 10),
            pady=18,
            sticky="nsew"
        )

        # TITLE
        ctk.CTkLabel(
            left_frame,
            text="AI Leaf Scanner",
            font=("Inter", 28, "bold"),
            text_color="#f8fafc"
        ).pack(anchor="w", pady=(0, 4))

        ctk.CTkLabel(
            left_frame,
            text="Phân tích hình thái và kết cấu lá cây bằng AI",
            font=("Inter", 13),
            text_color="#94a3b8"
        ).pack(anchor="w", pady=(0, 18))

        # BUTTON SELECT
        self.btn_select = ctk.CTkButton(
            left_frame,
            text="📁 Chọn Ảnh Lá Cây",
            command=self.on_select_image,
            font=("Inter", 14, "bold"),
            height=48,
            corner_radius=16,
            fg_color="#0f172a",
            hover_color="#1e293b",
            border_width=1,
            border_color="#1e293b",
            text_color="#f8fafc"
        )

        self.btn_select.pack(fill="x", pady=(0, 14))

        # PREVIEW CONTAINER
        self.preview_container = ctk.CTkFrame(
            left_frame,
            fg_color="#0b1120",
            border_width=2,
            border_color="#1e293b",
            corner_radius=22
        )

        self.preview_container.pack(
            fill="both",
            expand=True,
            pady=6
        )

        self.lbl_preview_img = ctk.CTkLabel(
            self.preview_container,
            text="🌿\n\nKéo thả hoặc tải ảnh lá cây\n\nHỗ trợ ảnh macro độ phân giải cao",
            font=("Inter", 15),
            text_color="#64748b",
            justify="center"
        )

        self.lbl_preview_img.pack(expand=True)

        # FILE NAME
        self.lbl_file_name = ctk.CTkLabel(
            left_frame,
            text="",
            font=("Inter", 12),
            text_color="#94a3b8"
        )

        self.lbl_file_name.pack(pady=10)

        # ANALYZE BUTTON
        self.btn_analyze = ctk.CTkButton(
            left_frame,
            text="⚡ Chạy AI Phân Tích",
            command=self.on_run_ai,
            font=("Inter", 14, "bold"),
            fg_color="#10b981",
            hover_color="#059669",
            height=52,
            corner_radius=16,
            text_color="#ffffff"
        )

        self.btn_analyze.pack(fill="x", pady=(6, 0))
        self.btn_analyze.configure(state="disabled")

        # =====================================================
        # RIGHT PANEL
        # =====================================================

        self.right_frame = ctk.CTkScrollableFrame(
            self.tab_scan,
            fg_color="#0f172a",
            border_width=1,
            border_color="#1e293b",
            corner_radius=22
        )

        self.right_frame.grid(
            row=0,
            column=1,
            padx=(10, 18),
            pady=18,
            sticky="nsew"
        )

        self.lbl_status = ctk.CTkLabel(
            self.right_frame,
            text="🧠\n\nHệ thống AI đã sẵn sàng.\nVui lòng tải ảnh ở cột bên trái.",
            font=("Inter", 16),
            text_color="#94a3b8",
            justify="center"
        )

        self.lbl_status.pack(
            pady=120,
            expand=True
        )

    # =========================================================
    # TAB DATABASE
    # =========================================================

    def _init_db_tab_layout(self):

        # SEARCH BAR
        search_frame = ctk.CTkFrame(
            self.tab_db,
            fg_color="transparent"
        )

        search_frame.pack(
            fill="x",
            padx=18,
            pady=(18, 10)
        )

        ctk.CTkLabel(
            search_frame,
            text="🔍 Tìm kiếm:",
            font=("Inter", 14, "bold"),
            text_color="#f8fafc"
        ).pack(side="left", padx=(4, 12))

        self.search_var = ctk.StringVar()

        self.search_var.trace_add(
            "write",
            lambda *args: self.on_search(self.search_var.get())
        )

        self.entry_search = ctk.CTkEntry(
            search_frame,
            placeholder_text="Nhập tên cây thuốc hoặc Label ID...",
            textvariable=self.search_var,
            height=42,
            corner_radius=16,
            fg_color="#0f172a",
            border_color="#1e293b",
            text_color="#f8fafc",
            placeholder_text_color="#64748b",
            font=("Inter", 13)
        )

        self.entry_search.pack(
            side="left",
            fill="x",
            expand=True
        )

        # DATABASE SCROLL
        self.scroll_db = ctk.CTkScrollableFrame(
            self.tab_db,
            fg_color="#0b1120",
            border_width=1,
            border_color="#1e293b",
            corner_radius=22
        )

        self.scroll_db.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=(6, 18)
        )

    # =========================================================
    # UPDATE PREVIEW
    # =========================================================

    def update_image_preview(self, ctk_img, file_name):

        self.lbl_preview_img.configure(
            image=ctk_img,
            text=""
        )

        self.lbl_file_name.configure(
            text=f"📄 {file_name}"
        )

        self.btn_analyze.configure(state="normal")

    # =========================================================
    # CLEAR RESULTS
    # =========================================================

    def clear_results_panel(self):

        for widget in self.right_frame.winfo_children():
            widget.destroy()

    # =========================================================
    # LOADING
    # =========================================================

    def show_loading_status(self):

        self.clear_results_panel()

        loading_frame = ctk.CTkFrame(
            self.right_frame,
            fg_color="#111827",
            corner_radius=18,
            border_width=1,
            border_color="#1e293b"
        )

        loading_frame.pack(
            fill="x",
            padx=12,
            pady=12
        )

        ctk.CTkLabel(
            loading_frame,
            text="🧠",
            font=("Segoe UI Emoji", 52)
        ).pack(pady=(28, 10))

        ctk.CTkLabel(
            loading_frame,
            text="Đang phân tích hình thái lá...",
            font=("Inter", 18, "bold"),
            text_color="#34d399"
        ).pack()

        ctk.CTkLabel(
            loading_frame,
            text="Extracting texture matrix • Running AI classifier",
            font=("Inter", 12),
            text_color="#94a3b8"
        ).pack(pady=(8, 18))

        progress = ctk.CTkProgressBar(
            loading_frame,
            height=10,
            progress_color="#10b981"
        )

        progress.pack(fill="x", padx=40, pady=(0, 28))
        progress.set(0.7)

        self.update()

    # =========================================================
    # ERROR
    # =========================================================

    def show_error_status(self, error_message):

        self.clear_results_panel()

        err_frame = ctk.CTkFrame(
            self.right_frame,
            fg_color="#111827",
            corner_radius=18,
            border_width=1,
            border_color="#7f1d1d"
        )

        err_frame.pack(
            fill="x",
            padx=12,
            pady=12
        )

        ctk.CTkLabel(
            err_frame,
            text="❌",
            font=("Segoe UI Emoji", 48)
        ).pack(pady=(26, 10))

        ctk.CTkLabel(
            err_frame,
            text=error_message,
            font=("Inter", 14),
            text_color="#ef4444",
            wraplength=480
        ).pack(pady=(0, 26))

    # =========================================================
    # DISPLAY AI RESULTS
    # =========================================================

    def display_ai_results(self, name_vn, accuracy, habitat, description, top_probs, features):

        self.clear_results_panel()

        # =====================================================
        # TOP RESULT CARD
        # =====================================================

        fr_top1 = ctk.CTkFrame(
            self.right_frame,
            fg_color="#111827",
            corner_radius=20,
            border_width=1,
            border_color="#1e293b"
        )

        fr_top1.pack(
            fill="x",
            padx=12,
            pady=12
        )

        ctk.CTkLabel(
            fr_top1,
            text="🎯 LOÀI NHẬN DIỆN CAO NHẤT",
            font=("Inter", 12, "bold"),
            text_color="#34d399"
        ).pack(anchor="w", padx=18, pady=(18, 6))

        ctk.CTkLabel(
            fr_top1,
            text=name_vn,
            font=("Inter", 28, "bold"),
            text_color="#ffffff"
        ).pack(anchor="w", padx=18)

        progress = ctk.CTkProgressBar(
            fr_top1,
            height=12,
            progress_color="#10b981"
        )

        progress.pack(fill="x", padx=18, pady=(18, 8))
        progress.set(accuracy / 100.0)

        ctk.CTkLabel(
            fr_top1,
            text=f"Độ chính xác AI: {accuracy}%",
            font=("Inter", 13, "bold"),
            text_color="#d1fae5"
        ).pack(anchor="w", padx=18)

        desc_txt = (
            f"\n🌍 Môi trường sinh thái: {habitat}\n\n"
            f"📝 Đặc điểm dược lý / hình thái:\n{description}"
        )

        lbl_desc = ctk.CTkLabel(
            fr_top1,
            text=desc_txt,
            font=("Inter", 13),
            justify="left",
            wraplength=500,
            text_color="#cbd5e1"
        )

        lbl_desc.pack(anchor="w", padx=18, pady=(8, 18))

        # =====================================================
        # TOP PROBS
        # =====================================================

        fr_probs = ctk.CTkFrame(
            self.right_frame,
            fg_color="#111827",
            border_width=1,
            border_color="#1e293b",
            corner_radius=20
        )

        fr_probs.pack(
            fill="x",
            padx=12,
            pady=12
        )

        ctk.CTkLabel(
            fr_probs,
            text="📊 TOP 3 ĐỘ TIN CẬY MÔ HÌNH",
            font=("Inter", 14, "bold"),
            text_color="#60a5fa"
        ).pack(anchor="w", padx=18, pady=(18, 14))

        for idx, item in enumerate(top_probs):

            row = ctk.CTkFrame(
                fr_probs,
                fg_color="#0f172a",
                corner_radius=14
            )

            row.pack(
                fill="x",
                padx=14,
                pady=6
            )

            row.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                row,
                text=f"Top {idx + 1}: {item['vn']}",
                font=("Inter", 13, "bold"),
                text_color="#f8fafc"
            ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 4))

            ctk.CTkLabel(
                row,
                text=f"{item['percent']}%",
                font=("Inter", 13, "bold"),
                text_color="#38bdf8"
            ).grid(row=0, column=1, sticky="e", padx=14)

            progress = ctk.CTkProgressBar(
                row,
                height=8,
                progress_color="#38bdf8"
            )

            progress.grid(
                row=1,
                column=0,
                columnspan=2,
                sticky="ew",
                padx=14,
                pady=(0, 14)
            )

            progress.set(item['percent'] / 100.0)

        # =====================================================
        # GLCM FEATURES
        # =====================================================

        fr_glcm = ctk.CTkFrame(
            self.right_frame,
            fg_color="#111827",
            border_width=1,
            border_color="#1e293b",
            corner_radius=20
        )

        fr_glcm.pack(
            fill="x",
            padx=12,
            pady=12
        )

        ctk.CTkLabel(
            fr_glcm,
            text="📐 THÔNG SỐ TEXTURE KẾT CẤU (GLCM)",
            font=("Inter", 14, "bold"),
            text_color="#a78bfa"
        ).pack(anchor="w", padx=18, pady=(18, 14))

        GLCM_VN_MAPPING = {
            "Contrast": "Độ tương phản",
            "Dissimilarity": "Độ không tương đồng",
            "Homogeneity": "Độ đồng nhất",
            "ASM": "Mô-men góc bậc hai",
            "Energy": "Năng lượng",
            "Max_Probability": "Xác suất cực đại",
            "Entropy": "Độ hỗn loạn",
            "GLCM_Mean": "Trung bình mức xám",
            "GLCM_Variance": "Phương sai",
            "Correlation": "Độ tương quan"
        }

        for key, val in features.items():

            row_f = ctk.CTkFrame(
                fr_glcm,
                fg_color="#0f172a",
                corner_radius=12
            )

            row_f.pack(
                fill="x",
                padx=14,
                pady=4
            )

            vn_name = GLCM_VN_MAPPING.get(key, key)

            display_name = f"{vn_name} ({key})"

            ctk.CTkLabel(
                row_f,
                text=display_name,
                font=("Inter", 12),
                text_color="#cbd5e1"
            ).pack(side="left", padx=14, pady=12)

            ctk.CTkLabel(
                row_f,
                text=f"{val:.5f}",
                font=("Consolas", 12, "bold"),
                text_color="#c084fc"
            ).pack(side="right", padx=14)

    # =========================================================
    # RENDER DATABASE
    # =========================================================

    def render_db_list(self, data_source, base_dir):

        for widget in self.scroll_db.winfo_children():
            widget.destroy()

        if not data_source:

            lbl_empty = ctk.CTkLabel(
                self.scroll_db,
                text="⚠️ Không tìm thấy loài cây phù hợp.",
                font=("Inter", 14),
                text_color="#94a3b8"
            )

            lbl_empty.pack(pady=40)
            return

        for cid, info in data_source.items():

            card = ctk.CTkFrame(
                self.scroll_db,
                fg_color="#111827",
                border_width=1,
                border_color="#1e293b",
                corner_radius=18
            )

            card.pack(
                fill="x",
                pady=8,
                padx=10
            )

            card.grid_columnconfigure(0, weight=0)
            card.grid_columnconfigure(1, weight=1)

            # =================================================
            # IMAGE FRAME
            # =================================================

            img_frame = ctk.CTkFrame(
                card,
                fg_color="#020617",
                width=130,
                height=130,
                corner_radius=14
            )

            img_frame.grid(
                row=0,
                column=0,
                padx=14,
                pady=14,
                sticky="nsew"
            )

            img_frame.pack_propagate(False)

            first_image_path = None

            train_folder = os.path.join(
                base_dir,
                "dataset",
                "swedish_leaf_dataset",
                "Swedish",
                "Train",
                f"Leaf {cid}"
            )

            test_folder = os.path.join(
                base_dir,
                "dataset",
                "swedish_leaf_dataset",
                "Swedish",
                "Test",
                f"Leaf {cid}"
            )

            folder_to_search = (
                train_folder
                if os.path.exists(train_folder)
                else test_folder
            )

            if os.path.exists(folder_to_search):

                images = [
                    f for f in os.listdir(folder_to_search)
                    if f.lower().endswith(
                        ('.png', '.jpg', '.jpeg', '.tif')
                    )
                ]

                if images:
                    first_image_path = os.path.join(
                        folder_to_search,
                        images[0]
                    )

            if first_image_path:

                try:

                    pil_img = Image.open(first_image_path)
                    pil_img.thumbnail((120, 120))

                    ctk_img = ctk.CTkImage(
                        light_image=pil_img,
                        dark_image=pil_img,
                        size=pil_img.size
                    )

                    lbl_img = ctk.CTkLabel(
                        img_frame,
                        image=ctk_img,
                        text=""
                    )

                    lbl_img.place(
                        relx=0.5,
                        rely=0.5,
                        anchor="center"
                    )

                except:

                    ctk.CTkLabel(
                        img_frame,
                        text="Lỗi ảnh",
                        font=("Inter", 11),
                        text_color="#94a3b8"
                    ).place(relx=0.5, rely=0.5, anchor="center")

            else:

                ctk.CTkLabel(
                    img_frame,
                    text="Chưa có ảnh",
                    font=("Inter", 11),
                    text_color="#64748b"
                ).place(relx=0.5, rely=0.5, anchor="center")

            # =================================================
            # INFO FRAME
            # =================================================

            info_frame = ctk.CTkFrame(
                card,
                fg_color="transparent"
            )

            info_frame.grid(
                row=0,
                column=1,
                padx=(0, 14),
                pady=14,
                sticky="nsew"
            )

            header_row = ctk.CTkFrame(
                info_frame,
                fg_color="transparent"
            )

            header_row.pack(
                fill="x",
                pady=(0, 8)
            )

            lbl_name = ctk.CTkLabel(
                header_row,
                text=info['name_vn'],
                font=("Inter", 18, "bold"),
                text_color="#34d399"
            )

            lbl_name.pack(side="left")

            lbl_tag = ctk.CTkLabel(
                header_row,
                text=f"Label ID: {cid}",
                font=("Consolas", 11, "bold"),
                fg_color="#1e293b",
                text_color="#cbd5e1",
                corner_radius=8,
                height=24,
                padx=10
            )

            lbl_tag.pack(side="right")

            body_txt = (
                f"🌍 Phân bố sinh thái: {info['habitat']}\n\n"
                f"📝 Đặc điểm mô tả:\n{info['description']}"
            )

            lbl_body = ctk.CTkLabel(
                info_frame,
                text=body_txt,
                font=("Inter", 12),
                justify="left",
                wraplength=700,
                text_color="#cbd5e1"
            )

            lbl_body.pack(anchor="w")