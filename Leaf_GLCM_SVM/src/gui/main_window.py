# src/gui/main_window.py
# ============================================================
# LeafVision AI — Main Window v3.0
# Light theme · Botanical aesthetic · Full DB integration
# ============================================================

import os
import customtkinter as ctk
from PIL import Image

# ============================================================
# GLOBAL UI CONFIG  —  Light / Botanical
# ============================================================

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")

# ── Palette ──────────────────────────────────────────────────
BG           = "#F5F7F2"          # nền tổng: trắng ngà xanh
SURFACE      = "#FFFFFF"          # thẻ chính
SURFACE_ALT  = "#EDF2EC"          # thẻ phụ / zebra
SURFACE_DEEP = "#E4EDE3"          # header section / ô đặc biệt
BORDER       = "#C8D9C5"          # viền nhẹ
BORDER_MED   = "#A8C4A4"          # viền medium
BORDER_DARK  = "#7EA87A"          # viền đậm / accent line

PRIMARY      = "#2D7A3A"          # xanh lá chủ đạo
PRIMARY_LT   = "#3D9E4C"          # hover
PRIMARY_SOFT = "#E8F5E9"          # nền badge xanh nhạt
PRIMARY_TEXT = "#1B5E20"          # text xanh đậm

BLUE         = "#1565C0"          # xanh dương (tên khoa học)
BLUE_SOFT    = "#E3F2FD"
AMBER        = "#F57F17"          # cảnh báo / conservation
AMBER_SOFT   = "#FFF8E1"
ROSE         = "#C62828"          # lỗi / độc
ROSE_SOFT    = "#FFEBEE"
PURPLE       = "#6A1B9A"          # GLCM / AI
PURPLE_SOFT  = "#F3E5F5"
TEAL         = "#00695C"          # dược liệu
TEAL_SOFT    = "#E0F2F1"
SAND         = "#795548"          # địa lý / origin
SAND_SOFT    = "#EFEBE9"

TEXT         = "#1A2E1A"          # text chính – xanh đen đậm
TEXT_SEC     = "#4A6741"          # text phụ – xanh rêu
TEXT_DIM     = "#7A9E76"          # text mờ
TEXT_FAINT   = "#A5C0A1"          # placeholder

FONT_HEAD    = "Georgia"          # tiêu đề: serif thanh lịch
FONT_BODY    = "Segoe UI"         # nội dung: rõ ràng, dễ đọc
FONT_MONO    = "Consolas"         # số liệu kỹ thuật


# ── Badge / Progress màu theo mức độ ─────────────────────────
AI_LEVEL_COLOR = {
    "Rất cao":   ("#B71C1C", "#FFCDD2"),
    "Cao":       ("#E65100", "#FFE0B2"),
    "Trung bình":("#F57F17", "#FFF9C4"),
    "Thấp":      ("#2E7D32", "#C8E6C9"),
}

# ── Màu vòng tròn radar AI features ──────────────────────────
_RADAR_LEVEL = {
    "Rất cao": 1.0, "Cao": 0.75,
    "Trung bình": 0.50, "Thấp": 0.25
}


# ============================================================
class LeafVisionUI(ctk.CTk):

    def __init__(self, on_select_image, on_run_ai, on_search):
        super().__init__()

        self.on_select_image = on_select_image
        self.on_run_ai       = on_run_ai
        self.on_search       = on_search
        self._pipeline_step_labels = {}

        self.title("LeafVision AI  ·  Hệ Thống Phân Tích Hình Thái Lá Cây")
        self.geometry("1360x860")
        self.minsize(1100, 700)
        self.configure(fg_color=BG)

        # ── Tab container ────────────────────────────────────
        self.tabview = ctk.CTkTabview(
            self,
            fg_color=BG,
            segmented_button_fg_color=SURFACE_DEEP,
            segmented_button_selected_color=PRIMARY,
            segmented_button_selected_hover_color=PRIMARY_LT,
            segmented_button_unselected_color=SURFACE_DEEP,
            segmented_button_unselected_hover_color=BORDER,
            corner_radius=14,
            border_width=1,
            border_color=BORDER,
            text_color=TEXT,
            text_color_disabled=TEXT_DIM,
        )
        self.tabview._segmented_button.configure(
            font=(FONT_BODY, 14, "bold"), height=44
        )
        self.tabview.pack(padx=16, pady=16, fill="both", expand=True)

        self.tab_scan = self.tabview.add("🔍  Quét Lá Phân Tích")
        self.tab_db   = self.tabview.add("📖  Từ Điển Thực Vật")
        self.tab_scan.configure(fg_color=BG)
        self.tab_db.configure(fg_color=BG)

        self._init_scan_tab()
        self._init_db_tab()

    # ========================================================
    # SCAN TAB
    # ========================================================

    def _init_scan_tab(self):
        self.tab_scan.grid_columnconfigure(0, weight=38, minsize=380)
        self.tab_scan.grid_columnconfigure(1, weight=62, minsize=540)
        self.tab_scan.grid_rowconfigure(0, weight=1)

        # ── LEFT panel ──────────────────────────────────────
        left = ctk.CTkFrame(self.tab_scan, fg_color="transparent")
        left.grid(row=0, column=0, padx=(14, 8), pady=14, sticky="nsew")

        # Header
        ctk.CTkLabel(left, text="AI Leaf Scanner",
                     font=(FONT_HEAD, 26, "bold"), text_color=PRIMARY
                     ).pack(anchor="w", pady=(0, 2))
        ctk.CTkLabel(left,
                     text="Phân tích kết cấu & hình thái lá bằng GLCM · SVM",
                     font=(FONT_BODY, 12), text_color=TEXT_SEC
                     ).pack(anchor="w", pady=(0, 14))

        # Nút chọn ảnh
        self.btn_select = ctk.CTkButton(
            left, text="📁  Chọn Ảnh Lá Cây",
            command=self.on_select_image,
            font=(FONT_BODY, 13, "bold"), height=46, corner_radius=12,
            fg_color=SURFACE, hover_color=SURFACE_DEEP,
            border_width=1.5, border_color=BORDER_MED, text_color=TEXT
        )
        self.btn_select.pack(fill="x", pady=(0, 10))

        # Khung xem trước ảnh
        self.preview_container = ctk.CTkFrame(
            left, fg_color=SURFACE,
            border_width=2, border_color=BORDER, corner_radius=20
        )
        self.preview_container.pack(fill="both", expand=True, pady=4)
        self.lbl_preview_img = ctk.CTkLabel(
            self.preview_container,
            text="🌿\n\nTải ảnh lá cây lên\n\nHỗ trợ ảnh macro độ phân giải cao",
            font=(FONT_BODY, 14), text_color=TEXT_FAINT, justify="center"
        )
        self.lbl_preview_img.pack(expand=True)

        self.lbl_file_name = ctk.CTkLabel(
            left, text="", font=(FONT_BODY, 11), text_color=TEXT_DIM
        )
        self.lbl_file_name.pack(pady=8)

        # Status bar
        status_bar = ctk.CTkFrame(left, fg_color=PRIMARY_SOFT,
                                  corner_radius=12, border_width=1,
                                  border_color=BORDER)
        status_bar.pack(fill="x", pady=(0, 10))
        for dot, label in [
            ("●", "Model Ready"),
            ("●", "Dataset Loaded"),
            ("●", "GLCM Active"),
        ]:
            r = ctk.CTkFrame(status_bar, fg_color="transparent")
            r.pack(side="left", expand=True, padx=8, pady=8)
            ctk.CTkLabel(r, text=dot, font=(FONT_BODY, 13),
                         text_color=PRIMARY).pack(side="left", padx=(0, 4))
            ctk.CTkLabel(r, text=label, font=(FONT_BODY, 11),
                         text_color=PRIMARY_TEXT).pack(side="left")

        # Nút phân tích
        self.btn_analyze = ctk.CTkButton(
            left, text="⚡  Chạy AI Phân Tích",
            command=self.on_run_ai,
            font=(FONT_BODY, 14, "bold"), fg_color=PRIMARY,
            hover_color=PRIMARY_LT, height=50, corner_radius=14,
            text_color="#ffffff"
        )
        self.btn_analyze.pack(fill="x", pady=(4, 0))
        self.btn_analyze.configure(state="disabled")

        # ── RIGHT panel ──────────────────────────────────────
        self.right_frame = ctk.CTkScrollableFrame(
            self.tab_scan, fg_color=SURFACE,
            border_width=1, border_color=BORDER, corner_radius=20,
            scrollbar_button_color=BORDER_MED,
            scrollbar_button_hover_color=BORDER_DARK,
        )
        self.right_frame.grid(
            row=0, column=1, padx=(8, 14), pady=14, sticky="nsew"
        )
        self.lbl_status = ctk.CTkLabel(
            self.right_frame,
            text="🌱\n\nHệ thống AI đã sẵn sàng.\nVui lòng tải ảnh ở cột bên trái.",
            font=(FONT_BODY, 15), text_color=TEXT_DIM, justify="center"
        )
        self.lbl_status.pack(pady=100, expand=True)

    # ========================================================
    # DATABASE TAB
    # ========================================================

    def _init_db_tab(self):
        # Header + search bar
        header_bar = ctk.CTkFrame(self.tab_db, fg_color=SURFACE,
                                  corner_radius=14, border_width=1,
                                  border_color=BORDER)
        header_bar.pack(fill="x", padx=14, pady=(14, 8))

        title_row = ctk.CTkFrame(header_bar, fg_color="transparent")
        title_row.pack(fill="x", padx=18, pady=(14, 8))
        ctk.CTkLabel(title_row,
                     text="Từ Điển Thực Vật  ·  Swedish Leaf Dataset",
                     font=(FONT_HEAD, 20, "bold"), text_color=PRIMARY
                     ).pack(side="left")
        ctk.CTkLabel(title_row, text="15 loài · 600+ ảnh scan",
                     font=(FONT_BODY, 12), text_color=TEXT_DIM
                     ).pack(side="right")

        search_row = ctk.CTkFrame(header_bar, fg_color="transparent")
        search_row.pack(fill="x", padx=18, pady=(0, 14))
        ctk.CTkLabel(search_row, text="🔍",
                     font=(FONT_BODY, 16), text_color=TEXT_SEC
                     ).pack(side="left", padx=(0, 8))
        self.search_var = ctk.StringVar()
        self.search_var.trace_add(
            "write", lambda *_: self.on_search(self.search_var.get())
        )
        self.entry_search = ctk.CTkEntry(
            search_row,
            placeholder_text="Tìm tên cây, tên khoa học, họ thực vật, công dụng, môi trường...",
            textvariable=self.search_var,
            height=40, corner_radius=12,
            fg_color=BG, border_color=BORDER_MED,
            text_color=TEXT, placeholder_text_color=TEXT_FAINT,
            font=(FONT_BODY, 13)
        )
        self.entry_search.pack(side="left", fill="x", expand=True)

        # Scrollable list
        self.scroll_db = ctk.CTkScrollableFrame(
            self.tab_db, fg_color=BG,
            border_width=1, border_color=BORDER, corner_radius=16,
            scrollbar_button_color=BORDER_MED,
            scrollbar_button_hover_color=BORDER_DARK,
        )
        self.scroll_db.pack(fill="both", expand=True, padx=14, pady=(0, 14))

    # ========================================================
    # UI HELPERS
    # ========================================================

    def update_image_preview(self, ctk_img, file_name):
        self.lbl_preview_img.configure(image=ctk_img, text="")
        self.lbl_file_name.configure(text=f"📄  {file_name}")
        self.btn_analyze.configure(state="normal")

    def clear_results_panel(self):
        for w in self.right_frame.winfo_children():
            w.destroy()
        self._pipeline_step_labels.clear()

    def _card(self, parent, border_color=BORDER, bg=SURFACE):
        """Card nền trắng với viền nhẹ."""
        return ctk.CTkFrame(
            parent, fg_color=bg, corner_radius=16,
            border_width=1, border_color=border_color
        )

    def _section_label(self, parent, text, color=None, icon_color=None):
        """Tiêu đề section — có dải màu bên trái."""
        wrapper = ctk.CTkFrame(parent, fg_color=SURFACE_DEEP, corner_radius=0)
        wrapper.pack(fill="x", pady=(10, 0))
        # Dải màu trái — dùng border_color thay vì frame riêng để tránh height bug
        inner = ctk.CTkFrame(wrapper, fg_color="transparent",
                             border_width=0)
        inner.pack(fill="x", padx=0, pady=0)
        # Accent line trái
        ctk.CTkFrame(inner, fg_color=color or PRIMARY,
                     width=4, height=32, corner_radius=0
                     ).pack(side="left")
        ctk.CTkLabel(inner, text=text,
                     font=(FONT_BODY, 12, "bold"),
                     text_color=color or PRIMARY
                     ).pack(side="left", padx=10, pady=8)

    def _kv_row(self, parent, label, value,
                label_color=None, value_color=None, monospace=False):
        row = ctk.CTkFrame(parent, fg_color=SURFACE_ALT, corner_radius=10)
        row.pack(fill="x", padx=14, pady=2)
        ctk.CTkLabel(row, text=label, font=(FONT_BODY, 12),
                     text_color=label_color or TEXT_SEC
                     ).pack(side="left", padx=14, pady=9)
        ctk.CTkLabel(row, text=str(value),
                     font=(FONT_MONO if monospace else FONT_BODY, 12,
                           "bold" if monospace else "normal"),
                     text_color=value_color or TEXT
                     ).pack(side="right", padx=14)

    def _badge(self, parent, text, bg, tc=TEXT):
        ctk.CTkLabel(
            parent, text=text,
            font=(FONT_BODY, 11, "bold"),
            fg_color=bg, text_color=tc,
            corner_radius=8, height=26, padx=12
        ).pack(side="left", padx=3, pady=2)

    def _divider(self, parent, pady=4):
        ctk.CTkFrame(parent, height=1, fg_color=BORDER
                     ).pack(fill="x", padx=14, pady=pady)

    # ── AI feature level badge (texture / edge / vein) ──────
    def _ai_level_badge(self, parent, label, level_str):
        bg_dark, bg_light = AI_LEVEL_COLOR.get(
            level_str, ("#555555", "#EEEEEE")
        )
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(side="left", padx=4)
        ctk.CTkLabel(row, text=label, font=(FONT_BODY, 10),
                     text_color=TEXT_DIM).pack()
        ctk.CTkLabel(row, text=level_str,
                     font=(FONT_BODY, 11, "bold"),
                     fg_color=bg_light, text_color=bg_dark,
                     corner_radius=6, height=22, padx=10
                     ).pack()

    # ── Mini progress bar với nhãn ───────────────────────────
    def _feature_bar(self, parent, label, value, max_val, color=PRIMARY):
        row = ctk.CTkFrame(parent, fg_color=SURFACE_ALT, corner_radius=10)
        row.pack(fill="x", padx=14, pady=2)
        row.grid_columnconfigure(0, weight=1)
        # Row trên: tên + giá trị
        ctk.CTkLabel(row, text=label, font=(FONT_BODY, 11),
                     text_color=TEXT_SEC
                     ).grid(row=0, column=0, sticky="w", padx=14, pady=(8, 2))
        ctk.CTkLabel(row, text=f"{value:.5f}",
                     font=(FONT_MONO, 11, "bold"),
                     text_color=PURPLE
                     ).grid(row=0, column=1, sticky="e", padx=14)
        # Bar
        bar = ctk.CTkProgressBar(row, height=6, corner_radius=3,
                                  progress_color=color, fg_color=BORDER)
        bar.grid(row=1, column=0, columnspan=2,
                 sticky="ew", padx=14, pady=(0, 8))
        bar.set(min(abs(value) / max_val, 1.0))

    # ========================================================
    # LOADING
    # ========================================================

    def show_loading_status(self):
        self.clear_results_panel()
        card = self._card(self.right_frame)
        card.pack(fill="x", padx=12, pady=12)

        ctk.CTkLabel(card, text="🌿",
                     font=("Segoe UI Emoji", 52)).pack(pady=(28, 8))
        ctk.CTkLabel(card, text="Đang phân tích hình thái lá...",
                     font=(FONT_HEAD, 17, "bold"),
                     text_color=PRIMARY).pack()
        ctk.CTkLabel(card,
                     text="Trích xuất ma trận kết cấu · Phân loại SVM",
                     font=(FONT_BODY, 12), text_color=TEXT_DIM
                     ).pack(pady=(6, 12))

        bar = ctk.CTkProgressBar(card, height=8, progress_color=PRIMARY,
                                  fg_color=BORDER)
        bar.pack(fill="x", padx=40, pady=(0, 14))
        bar.set(0.3)
        bar.configure(mode="indeterminate")
        bar.start()

        steps_frame = ctk.CTkFrame(card, fg_color=SURFACE_DEEP,
                                   corner_radius=12)
        steps_frame.pack(fill="x", padx=18, pady=(0, 22))

        step_names = [
            "Chuyển Grayscale & Lượng tử hoá",
            "Xây dựng ma trận GLCM",
            "Trích xuất 10 đặc trưng Haralick",
            "Chuẩn hoá z-score (StandardScaler)",
            "Phân loại SVM (RBF Kernel)",
        ]
        for name in step_names:
            row = ctk.CTkFrame(steps_frame, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=4)
            lbl = ctk.CTkLabel(
                row, text=f"◌  {name}",
                font=(FONT_MONO, 12), text_color=TEXT_FAINT
            )
            lbl.pack(side="left")
            self._pipeline_step_labels[name] = lbl
        self.update()

    def update_pipeline_step(self, step):
        lbl = self._pipeline_step_labels.get(step.name)
        if not lbl:
            return
        if step.status == "done":
            lbl.configure(
                text=f"✓  {step.name}  ({step.elapsed*1000:.0f} ms)",
                text_color=PRIMARY
            )
        elif step.status == "error":
            lbl.configure(text=f"✗  {step.name}", text_color=ROSE)

    # ========================================================
    # ERROR
    # ========================================================

    def show_error_status(self, error_message):
        self.clear_results_panel()
        card = self._card(self.right_frame, border_color=ROSE, bg=ROSE_SOFT)
        card.pack(fill="x", padx=12, pady=12)
        ctk.CTkLabel(card, text="⚠",
                     font=("Segoe UI Emoji", 44),
                     text_color=ROSE).pack(pady=(22, 8))
        ctk.CTkLabel(card, text=error_message,
                     font=(FONT_BODY, 13), text_color=ROSE,
                     wraplength=480, justify="left"
                     ).pack(pady=(0, 22), padx=20)

    # ========================================================
    # DISPLAY AI RESULTS
    # ========================================================

    def display_ai_results(self, result):
        self.clear_results_panel()

        pd_   = result.plant_details
        conf  = result.confidence

        # ── (0) Confusion warning ────────────────────────────
        if result.confusion_warning:
            cw = result.confusion_warning
            warn = ctk.CTkFrame(
                self.right_frame, fg_color=AMBER_SOFT,
                corner_radius=14, border_width=1.5, border_color=AMBER
            )
            warn.pack(fill="x", padx=12, pady=(12, 4))
            ctk.CTkLabel(
                warn,
                text=f"⚠  AI chưa chắc chắn  (khoảng cách chỉ {cw.gap}%)",
                font=(FONT_BODY, 13, "bold"), text_color=AMBER
            ).pack(anchor="w", padx=16, pady=(12, 4))
            ctk.CTkLabel(
                warn,
                text=getattr(cw, "hint", ""),
                font=(FONT_BODY, 12), text_color=SAND
            ).pack(anchor="w", padx=16, pady=(0, 12))

        # ── (1) Top Result ───────────────────────────────────
        fr_top = self._card(self.right_frame, border_color=BORDER_DARK)
        fr_top.pack(fill="x", padx=12, pady=(12, 8))

        # Dải nhãn "NHẬN DIỆN CAO NHẤT"
        tag = ctk.CTkFrame(fr_top, fg_color=PRIMARY_SOFT,
                           corner_radius=0, height=36)
        tag.pack(fill="x")
        ctk.CTkLabel(tag, text="🎯  LOÀI NHẬN DIỆN CAO NHẤT",
                     font=(FONT_BODY, 11, "bold"), text_color=PRIMARY_TEXT
                     ).pack(side="left", padx=18, pady=8)
        ctk.CTkLabel(tag, text=f"⏱ {result.elapsed_ms} ms",
                     font=(FONT_MONO, 11), text_color=TEXT_DIM
                     ).pack(side="right", padx=18)

        # Tên cây
        name_row = ctk.CTkFrame(fr_top, fg_color="transparent")
        name_row.pack(fill="x", padx=18, pady=(14, 4))
        ctk.CTkLabel(name_row, text=result.best_name_vn,
                     font=(FONT_HEAD, 24, "bold"), text_color=TEXT
                     ).pack(side="left")

        if result.best_name_sci:
            ctk.CTkLabel(fr_top, text=f"  {result.best_name_sci}",
                         font=("Georgia", 13, "italic"), text_color=BLUE
                         ).pack(anchor="w", padx=18, pady=(0, 8))

        # Badges họ / độc tính / dược liệu
        badge_row = ctk.CTkFrame(fr_top, fg_color="transparent")
        badge_row.pack(anchor="w", padx=18, pady=(0, 8))
        fam = pd_.get("family", "")
        if fam:
            self._badge(badge_row, f"🌿 {fam}", PRIMARY_SOFT, PRIMARY_TEXT)
        tox = pd_.get("toxicity", "")
        if "⚠️" in tox or ("ĐỘC" in tox.upper() and "KHÔNG" not in tox.upper()):
            self._badge(badge_row, "☠ Có độc", ROSE_SOFT, ROSE)
        else:
            self._badge(badge_row, "✓ Không độc", PRIMARY_SOFT, PRIMARY_TEXT)
        if pd_.get("medical_uses", "—") not in ("—", "Không rõ"):
            self._badge(badge_row, "💊 Dược liệu", TEAL_SOFT, TEAL)
        cs = pd_.get("conservation_status", "")
        if any(w in cs for w in ["Vulnerable", "Endangered", "Threatened", "Đe dọa", "Sắp"]):
            self._badge(badge_row, f"🔴 {cs}", AMBER_SOFT, AMBER)

        # Confidence bar
        bar = ctk.CTkProgressBar(fr_top, height=10, corner_radius=5,
                                  progress_color=conf.color_hex, fg_color=BORDER)
        bar.pack(fill="x", padx=18, pady=(8, 4))
        bar.set(result.best_percent / 100.0)

        conf_row = ctk.CTkFrame(fr_top, fg_color="transparent")
        conf_row.pack(fill="x", padx=18, pady=(0, 14))
        ctk.CTkLabel(conf_row,
                     text=f"Độ chính xác AI: {result.best_percent}%",
                     font=(FONT_BODY, 13, "bold"), text_color=PRIMARY_TEXT
                     ).pack(side="left")
        ctk.CTkLabel(conf_row, text=conf.value,
                     font=(FONT_BODY, 11, "bold"),
                     text_color=conf.color_hex,
                     fg_color=PRIMARY_SOFT, corner_radius=6,
                     padx=12, height=26
                     ).pack(side="right")

        # ── (2) Phân loại khoa học ───────────────────────────
        fr_sci = self._card(self.right_frame)
        fr_sci.pack(fill="x", padx=12, pady=6)
        self._section_label(fr_sci, "🧬  Phân Loại Khoa Học", BLUE)
        self._kv_row(fr_sci, "Giới",          "Plantae",
                     value_color=PRIMARY_TEXT)
        self._kv_row(fr_sci, "Họ",            pd_.get("family", "—"),
                     value_color=BLUE)
        self._kv_row(fr_sci, "Chi",           pd_.get("genus", "—"),
                     value_color=BLUE)
        self._kv_row(fr_sci, "Loài",          pd_.get("scientific_name", "—"),
                     value_color=BLUE, monospace=True)
        self._kv_row(fr_sci, "Tên tiếng Anh", pd_.get("english_name", "—"),
                     value_color=TEXT)
        # Tên thông thường
        cn = pd_.get("common_names", [])
        if cn:
            self._kv_row(fr_sci, "Tên khác",
                         " · ".join(cn), value_color=TEXT_SEC)
        self._kv_row(fr_sci, "Nguồn gốc",    pd_.get("origin", "—"),
                     value_color=SAND)
        self._kv_row(fr_sci, "Bảo tồn",      pd_.get("conservation_status", "—"),
                     value_color=AMBER)
        ctk.CTkFrame(fr_sci, height=10, fg_color="transparent").pack()

        # ── (3) Thông tin sinh học ───────────────────────────
        fr_bio = self._card(self.right_frame)
        fr_bio.pack(fill="x", padx=12, pady=6)
        self._section_label(fr_bio, "📏  Thông Tin Sinh Học", TEAL)
        self._kv_row(fr_bio, "Chiều cao trung bình",
                     pd_.get("average_height", "—"), value_color=TEAL)
        self._kv_row(fr_bio, "Tuổi thọ",
                     pd_.get("lifespan", "—"), value_color=TEAL)
        self._kv_row(fr_bio, "Môi trường sống",
                     pd_.get("habitat", "—"), value_color=TEXT_SEC)
        ctk.CTkFrame(fr_bio, height=10, fg_color="transparent").pack()

        # ── (4) Hình thái lá ─────────────────────────────────
        fr_morph = self._card(self.right_frame)
        fr_morph.pack(fill="x", padx=12, pady=6)
        self._section_label(fr_morph, "🌱  Hình Thái Lá", PRIMARY)
        self._kv_row(fr_morph, "Dạng lá",   pd_.get("leaf_type", "—"))
        self._kv_row(fr_morph, "Hình dạng", pd_.get("leaf_shape", "—"))
        self._kv_row(fr_morph, "Mép lá",    pd_.get("leaf_margin", "—"))
        self._kv_row(fr_morph, "Kết cấu",   pd_.get("leaf_texture", "—"))
        self._kv_row(fr_morph, "Vân gân",   pd_.get("vein_pattern", "—"),
                     value_color=TEXT_SEC)
        # Mô tả đặc điểm nhận diện
        desc = pd_.get("description", "")
        if desc and desc != "—":
            desc_frame = ctk.CTkFrame(fr_morph, fg_color=SURFACE_DEEP,
                                      corner_radius=10)
            desc_frame.pack(fill="x", padx=14, pady=(4, 14))
            ctk.CTkLabel(desc_frame,
                         text="🔍 Đặc điểm nhận diện trong dataset:",
                         font=(FONT_BODY, 11, "bold"),
                         text_color=PRIMARY_TEXT
                         ).pack(anchor="w", padx=14, pady=(10, 4))
            ctk.CTkLabel(desc_frame, text=desc,
                         font=(FONT_BODY, 12), justify="left",
                         wraplength=520, text_color=TEXT_SEC
                         ).pack(anchor="w", padx=14, pady=(0, 12))

        # ── (5) AI Features (texture/edge/vein) ─────────────
        ai_f = pd_.get("ai_features", {})
        if ai_f:
            fr_aif = self._card(self.right_frame)
            fr_aif.pack(fill="x", padx=12, pady=6)
            self._section_label(fr_aif, "🤖  Đặc Trưng AI Của Loài", PURPLE)
            ai_row = ctk.CTkFrame(fr_aif, fg_color="transparent")
            ai_row.pack(anchor="w", padx=18, pady=(4, 14))
            self._ai_level_badge(ai_row, "Kết cấu",
                                 ai_f.get("texture_level", "—"))
            self._ai_level_badge(ai_row, "Độ phức tạp mép",
                                 ai_f.get("edge_complexity", "—"))
            self._ai_level_badge(ai_row, "Mật độ gân",
                                 ai_f.get("vein_density", "—"))

        # ── (6) Dược liệu & An toàn ──────────────────────────
        fr_med = self._card(self.right_frame)
        fr_med.pack(fill="x", padx=12, pady=6)
        self._section_label(fr_med, "💊  Dược Liệu & An Toàn", TEAL)
        med = pd_.get("medical_uses", "—")
        if med and med != "—":
            med_box = ctk.CTkFrame(fr_med, fg_color=TEAL_SOFT, corner_radius=10)
            med_box.pack(fill="x", padx=14, pady=2)
            ctk.CTkLabel(med_box, text="Công dụng dược liệu:",
                         font=(FONT_BODY, 11, "bold"), text_color=TEAL
                         ).pack(anchor="w", padx=14, pady=(10, 4))
            ctk.CTkLabel(med_box, text=med, font=(FONT_BODY, 12),
                         text_color=TEXT, wraplength=520, justify="left"
                         ).pack(anchor="w", padx=14, pady=(0, 12))
        tox = pd_.get("toxicity", "—")
        is_toxic = "⚠️" in tox or ("ĐỘC" in tox.upper() and "KHÔNG" not in tox.upper())
        tox_box = ctk.CTkFrame(
            fr_med,
            fg_color=ROSE_SOFT if is_toxic else PRIMARY_SOFT,
            corner_radius=10
        )
        tox_box.pack(fill="x", padx=14, pady=(2, 14))
        ctk.CTkLabel(tox_box,
                     text=("☠  " if is_toxic else "✓  ") + f"Độc tính: {tox}",
                     font=(FONT_BODY, 12, "bold"),
                     text_color=ROSE if is_toxic else PRIMARY_TEXT
                     ).pack(anchor="w", padx=14, pady=10)

        # ── (7) TOP 3 xác suất ───────────────────────────────
        fr_probs = self._card(self.right_frame)
        fr_probs.pack(fill="x", padx=12, pady=6)
        self._section_label(fr_probs, "📊  TOP 3 Độ Tin Cậy Mô Hình", BLUE)

        colors_top = [PRIMARY, BLUE, SAND]
        for r in result.top_results:
            row = ctk.CTkFrame(fr_probs, fg_color=SURFACE_ALT, corner_radius=12)
            row.pack(fill="x", padx=14, pady=4)
            row.grid_columnconfigure(0, weight=1)
            clr = colors_top[(r.rank - 1) % len(colors_top)]
            ctk.CTkLabel(row, text=f"#{r.rank}  {r.name_vn}",
                         font=(FONT_BODY, 13, "bold"), text_color=TEXT
                         ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 2))
            ctk.CTkLabel(row, text=f"{r.percent}%",
                         font=(FONT_MONO, 13, "bold"), text_color=clr
                         ).grid(row=0, column=1, sticky="e", padx=14)
            if r.name_sci:
                ctk.CTkLabel(row, text=f"  {r.name_sci}",
                             font=("Georgia", 11, "italic"),
                             text_color=TEXT_DIM
                             ).grid(row=1, column=0, sticky="w", padx=14)
            bar2 = ctk.CTkProgressBar(row, height=6, corner_radius=3,
                                       progress_color=clr, fg_color=BORDER)
            bar2.grid(row=2, column=0, columnspan=2,
                      sticky="ew", padx=14, pady=(4, 12))
            bar2.set(r.percent / 100.0)

        # ── (8) AI Analysis Panel ────────────────────────────
        fr_ai = self._card(self.right_frame)
        fr_ai.pack(fill="x", padx=12, pady=6)
        self._section_label(fr_ai, "🧠  AI Analysis Panel", PURPLE)
        self._kv_row(fr_ai, "Prediction Confidence",
                     conf.label_vn, value_color=conf.color_hex)
        stability = ("GOOD" if result.features.get("Entropy", 99) < 3
                     else "MODERATE")
        self._kv_row(fr_ai, "Model Stability", stability,
                     value_color=PRIMARY if stability == "GOOD" else AMBER)
        homog_pct = round(result.features.get("Homogeneity", 0) * 100, 1)
        self._kv_row(fr_ai, "Texture Similarity",
                     f"{homog_pct}%", value_color=BLUE)
        if result.feature_insights:
            ctk.CTkLabel(fr_ai,
                         text="  Lý do nhận diện loài này:",
                         font=(FONT_BODY, 12, "bold"),
                         text_color=PURPLE
                         ).pack(anchor="w", padx=14, pady=(8, 2))
            for ins in result.feature_insights:
                ins_row = ctk.CTkFrame(fr_ai, fg_color=PURPLE_SOFT,
                                       corner_radius=8)
                ins_row.pack(fill="x", padx=14, pady=2)
                ctk.CTkLabel(ins_row, text=f"· {ins.description}",
                             font=(FONT_BODY, 12), text_color=PURPLE
                             ).pack(side="left", padx=14, pady=7)
                ctk.CTkLabel(ins_row, text=f"{ins.value:.4f}",
                             font=(FONT_MONO, 11), text_color=TEXT_DIM
                             ).pack(side="right", padx=14)
        ctk.CTkFrame(fr_ai, height=10, fg_color="transparent").pack()

        # ── (9) GLCM Feature Bars ─────────────────────────────
        fr_glcm = self._card(self.right_frame)
        fr_glcm.pack(fill="x", padx=12, pady=6)
        self._section_label(fr_glcm, "📐  Thông Số Kết Cấu GLCM", PURPLE)

        GLCM_VN = {
            "Contrast":        "Độ tương phản  (Contrast)",
            "Dissimilarity":   "Độ bất tương đồng  (Dissimilarity)",
            "Homogeneity":     "Độ đồng nhất  (Homogeneity)",
            "ASM":             "Mô-men góc bậc hai  (ASM)",
            "Energy":          "Năng lượng  (Energy)",
            "Entropy":         "Độ hỗn loạn  (Entropy)",
            "Max_Probability": "Xác suất cực đại  (Max Probability)",
            "Correlation":     "Độ tương quan  (Correlation)",
            "GLCM_Mean":       "Trung bình mức xám  (GLCM Mean)",
            "GLCM_Variance":   "Phương sai  (GLCM Variance)",
        }
        BAR_MAX = {
            "Contrast": 20, "Dissimilarity": 10, "Entropy": 8,
            "GLCM_Variance": 100, "GLCM_Mean": 255,
        }
        # Màu bar theo nhóm đặc trưng
        BAR_COLOR = {
            "Contrast": ROSE, "Dissimilarity": AMBER,
            "Homogeneity": TEAL, "ASM": PURPLE,
            "Energy": PURPLE, "Entropy": SAND,
            "Max_Probability": BLUE, "Correlation": BLUE,
            "GLCM_Mean": PRIMARY, "GLCM_Variance": AMBER,
        }
        for key, val in result.features.items():
            self._feature_bar(
                fr_glcm,
                label=f"{GLCM_VN.get(key, key)}",
                value=val,
                max_val=BAR_MAX.get(key, 1.0),
                color=BAR_COLOR.get(key, PRIMARY)
            )
        ctk.CTkFrame(fr_glcm, height=10, fg_color="transparent").pack()

        # ── (10) Model Information ────────────────────────────
        mi = result.model_info
        fr_model = self._card(self.right_frame)
        fr_model.pack(fill="x", padx=12, pady=6)
        self._section_label(fr_model, "⚙  Model Information", TEXT_SEC)
        self._kv_row(fr_model, "Classifier",  mi.classifier,
                     value_color=BLUE)
        self._kv_row(fr_model, "Kernel",      mi.kernel,
                     value_color=PURPLE, monospace=True)
        self._kv_row(fr_model, "SVM C / γ",
                     f"{mi.svm_C}  ·  {mi.svm_gamma}",
                     value_color=TEXT_SEC)
        self._kv_row(fr_model, "Features",    mi.feature_type)
        self._kv_row(fr_model, "GLCM Levels", str(mi.glcm_levels))
        self._kv_row(fr_model, "Distances",   str(mi.glcm_distances))
        self._kv_row(fr_model, "Angles",      str(mi.glcm_angles))
        self._kv_row(fr_model, "Support Vectors",
                     str(mi.num_support_vectors), value_color=PURPLE)
        self._kv_row(fr_model, "Dataset",     mi.dataset)
        ctk.CTkFrame(fr_model, height=10, fg_color="transparent").pack()

        # ── (11) Pipeline Trace ───────────────────────────────
        fr_pipe = self._card(self.right_frame)
        fr_pipe.pack(fill="x", padx=12, pady=(6, 18))
        self._section_label(fr_pipe, "🔄  Processing Pipeline", TEXT_DIM)
        for step in result.pipeline_steps:
            ok = step.status == "done"
            row = ctk.CTkFrame(fr_pipe,
                               fg_color=PRIMARY_SOFT if ok else ROSE_SOFT,
                               corner_radius=8)
            row.pack(fill="x", padx=14, pady=2)
            ctk.CTkLabel(row,
                         text=f"{'✓' if ok else '✗'}  {step.name}",
                         font=(FONT_MONO, 12),
                         text_color=PRIMARY_TEXT if ok else ROSE
                         ).pack(side="left", padx=14, pady=7)
            ctk.CTkLabel(row,
                         text=f"{step.elapsed*1000:.0f} ms",
                         font=(FONT_MONO, 11), text_color=TEXT_DIM
                         ).pack(side="right", padx=14)
        ctk.CTkFrame(fr_pipe, height=6, fg_color="transparent").pack()

    # ========================================================
    # RENDER DATABASE  —  full plants_db fields
    # ========================================================

    def render_db_list(self, data_source, base_dir):
        for w in self.scroll_db.winfo_children():
            w.destroy()

        if not data_source:
            ctk.CTkLabel(self.scroll_db,
                         text="⚠  Không tìm thấy loài cây phù hợp.",
                         font=(FONT_BODY, 14), text_color=TEXT_DIM
                         ).pack(pady=40)
            return

        for cid, info in data_source.items():
            self._render_db_card(cid, info, base_dir)

    def _render_db_card(self, cid, info, base_dir):
        """Render 1 thẻ cây — dùng đầy đủ tất cả fields trong plants_db."""

        card = ctk.CTkFrame(
            self.scroll_db, fg_color=SURFACE,
            border_width=1, border_color=BORDER, corner_radius=18
        )
        card.pack(fill="x", pady=7, padx=6)

        # ── Dải màu header ───────────────────────────────────
        hdr = ctk.CTkFrame(card, fg_color=PRIMARY_SOFT,
                           corner_radius=0, height=40)
        hdr.pack(fill="x")
        hdr.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(hdr, text=info.get("name_vn", "?"),
                     font=(FONT_HEAD, 15, "bold"),
                     text_color=PRIMARY_TEXT
                     ).pack(side="left", padx=14, pady=8)

        # Badge ID + badge họ thực vật
        badge_hdr = ctk.CTkFrame(hdr, fg_color="transparent")
        badge_hdr.pack(side="right", padx=10)
        ctk.CTkLabel(badge_hdr,
                     text=info.get("family", ""),
                     font=(FONT_BODY, 11),
                     fg_color=BORDER, text_color=TEXT_SEC,
                     corner_radius=6, height=22, padx=10
                     ).pack(side="left", padx=4)
        ctk.CTkLabel(badge_hdr,
                     text=f"  #{cid}  ",
                     font=(FONT_MONO, 11, "bold"),
                     fg_color=PRIMARY, text_color="#fff",
                     corner_radius=6, height=22, padx=6
                     ).pack(side="left", padx=4)

        # ── Body: ảnh + thông tin ────────────────────────────
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="x", padx=14, pady=10)
        body.grid_columnconfigure(1, weight=1)

        # Khung ảnh
        img_frame = ctk.CTkFrame(body, fg_color=BG,
                                  width=120, height=120, corner_radius=12,
                                  border_width=1, border_color=BORDER)
        img_frame.grid(row=0, column=0, rowspan=3,
                       padx=(0, 16), pady=0, sticky="n")
        img_frame.pack_propagate(False)

        # Tìm ảnh trong dataset
        first_image_path = None
        for folder in [
            os.path.join(base_dir, "dataset", "swedish_leaf_dataset",
                         "Swedish", "Train", f"Leaf {cid}"),
            os.path.join(base_dir, "dataset", "swedish_leaf_dataset",
                         "Swedish", "Test",  f"Leaf {cid}"),
        ]:
            if os.path.exists(folder):
                imgs = [f for f in os.listdir(folder)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
                if imgs:
                    first_image_path = os.path.join(folder, sorted(imgs)[0])
                    break

        if first_image_path:
            try:
                pil_img = Image.open(first_image_path).convert("RGBA")
                pil_img.thumbnail((110, 110))
                ctk_img = ctk.CTkImage(
                    light_image=pil_img, dark_image=pil_img, size=pil_img.size
                )
                ctk.CTkLabel(img_frame, image=ctk_img, text=""
                             ).place(relx=0.5, rely=0.5, anchor="center")
            except Exception:
                ctk.CTkLabel(img_frame, text="Lỗi ảnh",
                             font=(FONT_BODY, 10), text_color=TEXT_DIM
                             ).place(relx=0.5, rely=0.5, anchor="center")
        else:
            ctk.CTkLabel(img_frame, text="📷\nChưa có ảnh",
                         font=(FONT_BODY, 10), text_color=TEXT_FAINT,
                         justify="center"
                         ).place(relx=0.5, rely=0.5, anchor="center")

        # ── Cột phải: thông tin ──────────────────────────────
        info_col = ctk.CTkFrame(body, fg_color="transparent")
        info_col.grid(row=0, column=1, sticky="nsew")

        # Tên khoa học + chi
        sci_row = ctk.CTkFrame(info_col, fg_color="transparent")
        sci_row.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(sci_row,
                     text=f"  {info.get('scientific_name', '')}",
                     font=("Georgia", 13, "italic"), text_color=BLUE
                     ).pack(side="left")
        genus = info.get("genus", "")
        if genus:
            ctk.CTkLabel(sci_row, text=f"· {genus}",
                         font=(FONT_BODY, 11), text_color=TEXT_DIM
                         ).pack(side="left", padx=6)

        # Badges: độc tính / medicinal / AI features
        b_row = ctk.CTkFrame(info_col, fg_color="transparent")
        b_row.pack(anchor="w", pady=(0, 6))
        tox = info.get("toxicity", "")
        if "⚠️" in tox or ("ĐỘC" in tox.upper() and "KHÔNG" not in tox.upper()):
            self._badge(b_row, "☠ Độc", ROSE_SOFT, ROSE)
        else:
            self._badge(b_row, "✓ Không độc", PRIMARY_SOFT, PRIMARY_TEXT)
        if info.get("medical_uses", "—") not in ("—", "Không rõ"):
            self._badge(b_row, "💊 Dược liệu", TEAL_SOFT, TEAL)
        cs = info.get("conservation_status", "")
        if any(w in cs for w in ["Endangered", "Vulnerable", "Đe dọa", "Sắp"]):
            self._badge(b_row, f"🔴 {cs}", AMBER_SOFT, AMBER)

        # AI features level badges
        ai_f = info.get("ai_features", {})
        if ai_f:
            ai_badge_row = ctk.CTkFrame(info_col, fg_color="transparent")
            ai_badge_row.pack(anchor="w", pady=(0, 6))
            self._ai_level_badge(ai_badge_row, "Kết cấu",
                                 ai_f.get("texture_level", "—"))
            self._ai_level_badge(ai_badge_row, "Mép lá",
                                 ai_f.get("edge_complexity", "—"))
            self._ai_level_badge(ai_badge_row, "Gân lá",
                                 ai_f.get("vein_density", "—"))

        # Thông tin nhanh: chiều cao · tuổi thọ · origin
        meta_row = ctk.CTkFrame(info_col, fg_color="transparent")
        meta_row.pack(anchor="w", pady=(0, 6))
        for icon, key in [("↕", "average_height"), ("🕐", "lifespan"),
                          ("🌍", "origin")]:
            val = info.get(key, "")
            if val:
                ctk.CTkLabel(meta_row,
                             text=f"{icon} {val}",
                             font=(FONT_BODY, 11),
                             text_color=TEXT_DIM
                             ).pack(side="left", padx=(0, 14))

        # Mô tả ngắn
        desc = info.get("description", "")
        if desc:
            ctk.CTkLabel(info_col, text=desc,
                         font=(FONT_BODY, 11), justify="left",
                         wraplength=680, text_color=TEXT_SEC
                         ).pack(anchor="w")

        # ── Chi tiết mở rộng: hình thái + dược liệu ──────────
        detail_row = ctk.CTkFrame(card, fg_color=SURFACE_DEEP,
                                  corner_radius=0)
        detail_row.pack(fill="x", padx=0, pady=(6, 0))

        # 3 cột: hình thái | gân+kết cấu | công dụng
        detail_row.grid_columnconfigure(0, weight=1)
        detail_row.grid_columnconfigure(1, weight=1)
        detail_row.grid_columnconfigure(2, weight=1)

        # Cột 1: Hình thái lá
        col1 = ctk.CTkFrame(detail_row, fg_color="transparent")
        col1.grid(row=0, column=0, padx=14, pady=10, sticky="nsew")
        ctk.CTkLabel(col1, text="🌿  Hình thái lá",
                     font=(FONT_BODY, 11, "bold"),
                     text_color=PRIMARY_TEXT
                     ).pack(anchor="w", pady=(0, 4))
        for lbl, key in [
            ("Dạng:", "leaf_type"),
            ("Hình:", "leaf_shape"),
            ("Mép:",  "leaf_margin"),
        ]:
            r = ctk.CTkFrame(col1, fg_color="transparent")
            r.pack(anchor="w", fill="x")
            ctk.CTkLabel(r, text=lbl, font=(FONT_BODY, 10),
                         text_color=TEXT_FAINT, width=40
                         ).pack(side="left")
            ctk.CTkLabel(r, text=info.get(key, "—"),
                         font=(FONT_BODY, 10), text_color=TEXT_SEC
                         ).pack(side="left", padx=4)

        # Cột 2: Kết cấu & gân
        col2 = ctk.CTkFrame(detail_row, fg_color="transparent")
        col2.grid(row=0, column=1, padx=14, pady=10, sticky="nsew")
        ctk.CTkLabel(col2, text="🔬  Kết cấu & Gân lá",
                     font=(FONT_BODY, 11, "bold"),
                     text_color=PRIMARY_TEXT
                     ).pack(anchor="w", pady=(0, 4))
        for lbl, key in [
            ("Bề mặt:", "leaf_texture"),
            ("Gân:",    "vein_pattern"),
            ("Môi trường:", "habitat"),
        ]:
            r = ctk.CTkFrame(col2, fg_color="transparent")
            r.pack(anchor="w", fill="x")
            ctk.CTkLabel(r, text=lbl, font=(FONT_BODY, 10),
                         text_color=TEXT_FAINT, width=70,
                         wraplength=70
                         ).pack(side="left")
            ctk.CTkLabel(r, text=info.get(key, "—"),
                         font=(FONT_BODY, 10), text_color=TEXT_SEC,
                         wraplength=200, justify="left"
                         ).pack(side="left", padx=4)

        # Cột 3: Dược liệu
        col3 = ctk.CTkFrame(detail_row, fg_color="transparent")
        col3.grid(row=0, column=2, padx=14, pady=10, sticky="nsew")
        ctk.CTkLabel(col3, text="💊  Công dụng & Độc tính",
                     font=(FONT_BODY, 11, "bold"),
                     text_color=PRIMARY_TEXT
                     ).pack(anchor="w", pady=(0, 4))
        med = info.get("medical_uses", "—")
        ctk.CTkLabel(col3, text=med,
                     font=(FONT_BODY, 10), text_color=TEAL,
                     wraplength=220, justify="left"
                     ).pack(anchor="w")
        ctk.CTkFrame(col3, height=6, fg_color="transparent").pack()
        tox = info.get("toxicity", "—")
        is_toxic = "⚠️" in tox or ("ĐỘC" in tox.upper() and "KHÔNG" not in tox.upper())
        ctk.CTkLabel(col3,
                     text=("☠ " if is_toxic else "✓ ") + tox,
                     font=(FONT_BODY, 10, "bold"),
                     text_color=ROSE if is_toxic else PRIMARY_TEXT
                     ).pack(anchor="w")

        # Bottom border spacing
        ctk.CTkFrame(card, height=2, fg_color=BORDER).pack(fill="x")
