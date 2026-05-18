import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.predictor import predict_new_image


class LeafApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ Thống Phân Tích & Nhận Diện Lá Cây - GLCM & SVM")
        self.root.geometry("1200x720")
        self.root.configure(bg="#eef3f8")
        self.root.minsize(1100, 680)

        self.selected_img_path = None

        # ================= STYLE =================
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "Modern.TLabelframe",
            background="#ffffff",
            borderwidth=0,
            relief="flat"
        )

        style.configure(
            "Modern.TLabelframe.Label",
            background="#ffffff",
            foreground="#1b4332",
            font=("Segoe UI", 11, "bold")
        )

        # ================= HEADER =================
        header_frame = tk.Frame(root, bg="#0f172a", height=90)
        header_frame.pack(fill="x")

        title_lbl = tk.Label(
            header_frame,
            text="🌿 HỆ THỐNG NHẬN DIỆN LÁ CÂY THUỐC",
            font=("Segoe UI", 22, "bold"),
            fg="white",
            bg="#0f172a"
        )
        title_lbl.pack(pady=(18, 2))

        sub_lbl = tk.Label(
            header_frame,
            text="Ứng dụng phân tích đặc trưng GLCM và phân loại bằng SVM",
            font=("Segoe UI", 10),
            fg="#cbd5e1",
            bg="#0f172a"
        )
        sub_lbl.pack()

        # ================= MAIN =================
        main_frame = tk.Frame(root, bg="#eef3f8")
        main_frame.pack(fill="both", expand=True, padx=25, pady=20)

        # ==================================================
        # LEFT PANEL
        # ==================================================
        left_card = tk.Frame(
            main_frame,
            bg="white",
            bd=0,
            highlightthickness=1,
            highlightbackground="#dbe4ee"
        )
        left_card.pack(side="left", fill="both", padx=(0, 12), pady=5)

        left_inner = tk.Frame(left_card, bg="white")
        left_inner.pack(padx=20, pady=20)

        image_title = tk.Label(
            left_inner,
            text="🖼  Ảnh Lá Cây",
            font=("Segoe UI", 14, "bold"),
            bg="white",
            fg="#1e293b"
        )
        image_title.pack(anchor="w", pady=(0, 12))

        self.canvas = tk.Canvas(
            left_inner,
            width=360,
            height=360,
            bg="#f8fafc",
            highlightthickness=2,
            highlightbackground="#dbe4ee"
        )
        self.canvas.pack()

        self.canvas.create_text(
            180,
            180,
            text="Chưa có ảnh nào được chọn",
            fill="#94a3b8",
            font=("Segoe UI", 11, "italic")
        )

        # BUTTONS
        btn_frame = tk.Frame(left_inner, bg="white")
        btn_frame.pack(fill="x", pady=22)

        upload_btn = tk.Button(
            btn_frame,
            text="📁 Tải Ảnh",
            font=("Segoe UI", 11, "bold"),
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            relief="flat",
            padx=18,
            pady=10,
            cursor="hand2",
            command=self.upload_image
        )
        upload_btn.grid(row=0, column=0, padx=6)

        self.predict_btn = tk.Button(
            btn_frame,
            text="🧠 Phân Tích",
            font=("Segoe UI", 11, "bold"),
            bg="#16a34a",
            fg="white",
            activebackground="#15803d",
            activeforeground="white",
            relief="flat",
            padx=18,
            pady=10,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.classify_image
        )
        self.predict_btn.grid(row=0, column=1, padx=6)

        # ==================================================
        # RIGHT PANEL
        # ==================================================
        right_frame = tk.Frame(main_frame, bg="#eef3f8")
        right_frame.pack(side="right", fill="both", expand=True)

        # ---------- FEATURE FRAME ----------
        feat_frame = ttk.LabelFrame(
            right_frame,
            text=" 📊 Đặc Trưng GLCM ",
            style="Modern.TLabelframe"
        )
        feat_frame.pack(fill="both", expand=True, pady=(0, 14))

        feat_container = tk.Frame(feat_frame, bg="white")
        feat_container.pack(fill="both", expand=True, padx=12, pady=12)

        self.txt_features = tk.Text(
            feat_container,
            height=14,
            font=("Consolas", 10),
            bg="#f8fafc",
            fg="#0f172a",
            relief="flat",
            padx=12,
            pady=12,
            insertbackground="#0f172a",
            state=tk.DISABLED
        )
        self.txt_features.pack(fill="both", expand=True)

        # ---------- PROB FRAME ----------
        prob_frame = ttk.LabelFrame(
            right_frame,
            text=" 📈 Xác Suất Dự Đoán ",
            style="Modern.TLabelframe"
        )
        prob_frame.pack(fill="x", pady=(0, 14))

        prob_container = tk.Frame(prob_frame, bg="white")
        prob_container.pack(fill="x", padx=12, pady=12)

        self.lbl_prob_result = tk.Label(
            prob_container,
            text="Chưa có dữ liệu phân tích.",
            font=("Segoe UI", 11),
            bg="white",
            fg="#64748b",
            justify="left",
            anchor="w"
        )
        self.lbl_prob_result.pack(fill="x")

        # ---------- FINAL RESULT ----------
        final_frame = ttk.LabelFrame(
            right_frame,
            text=" 🎯 Kết Luận ",
            style="Modern.TLabelframe"
        )
        final_frame.pack(fill="x")

        final_container = tk.Frame(final_frame, bg="white")
        final_container.pack(fill="x", padx=12, pady=16)

        self.lbl_final = tk.Label(
            final_container,
            text="Hệ thống đang chờ ảnh...",
            font=("Segoe UI", 16, "bold"),
            bg="white",
            fg="#475569"
        )
        self.lbl_final.pack(anchor="w")

    # ==================================================
    # UPLOAD IMAGE
    # ==================================================
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
        )

        if file_path:
            self.selected_img_path = file_path

            img = Image.open(file_path)
            img = img.resize((360, 360), Image.Resampling.LANCZOS)

            self.tk_img = ImageTk.PhotoImage(img)

            self.canvas.delete("all")
            self.canvas.create_image(180, 180, image=self.tk_img)

            self.predict_btn.config(state=tk.NORMAL)

            self.lbl_final.config(
                text="Hệ thống đang chờ phân tích...",
                fg="#475569"
            )

            self.lbl_prob_result.config(
                text="Chưa có dữ liệu phân tích.",
                fg="#64748b",
                font=("Segoe UI", 11)
            )

            self.clear_text_box()

    # ==================================================
    # CLEAR TEXT BOX
    # ==================================================
    def clear_text_box(self):
        self.txt_features.config(state=tk.NORMAL)
        self.txt_features.delete("1.0", tk.END)
        self.txt_features.config(state=tk.DISABLED)

    # ==================================================
    # CLASSIFY IMAGE
    # ==================================================
    def classify_image(self):
        if not self.selected_img_path:
            return

        leaf_info, top_probs, features, error = predict_new_image(self.selected_img_path)

        if error:
            messagebox.showerror("Lỗi", error)
            return

        # ================= FEATURES =================
        self.txt_features.config(state=tk.NORMAL)
        self.txt_features.delete("1.0", tk.END)

        glcm_translation = {
            'Contrast': 'Độ tương phản',
            'Dissimilarity': 'Độ khác biệt',
            'Homogeneity': 'Tính đồng nhất',
            'ASM': 'Mô-men góc (ASM)',
            'Energy': 'Năng lượng',
            'Max_Probability': 'Xác suất lớn nhất',
            'Entropy': 'Độ hỗn loạn (Entropy)',
            'GLCM_Mean': 'Giá trị trung bình',
            'GLCM_Variance': 'Phương sai GLCM',
            'Correlation': 'Độ tương quan'
        }

        self.txt_features.insert(
            tk.END,
            "===== THÔNG TIN ĐẶC TRƯNG GLCM =====\n\n"
        )

        for key, value in features.items():
            vi_name = glcm_translation.get(key, 'Không rõ')
            display_label = f"{key} ({vi_name})"

            self.txt_features.insert(
                tk.END,
                f"➤ {display_label:<42}: {value:.6f}\n"
            )

        self.txt_features.config(state=tk.DISABLED)

        # ================= TOP PROBS =================
        prob_text = ""

        for idx, item in enumerate(top_probs, 1):
            prob_text += (
                f"🏷  Top {idx}: {item['vn']}\n"
                f"     ➜ Độ chính xác dự kiến: {item['percent']:.2f}%\n\n"
            )

        self.lbl_prob_result.config(
            text=prob_text,
            font=("Segoe UI", 11, "bold"),
            fg="#14532d"
        )

        # ================= FINAL RESULT =================
        self.lbl_final.config(
            text=f"🌿 Loại lá cây: {leaf_info['vn']} ({leaf_info['en']})",
            fg="#dc2626"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = LeafApp(root)
    root.mainloop()
