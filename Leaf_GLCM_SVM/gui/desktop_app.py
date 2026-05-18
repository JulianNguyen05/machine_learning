import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Thêm thư mục gốc vào hệ thống để gọi được các file trong src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.predictor import predict_new_image


class LeafApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ Thống Nhận Diện Lá Cây - GLCM & SVM")
        self.root.geometry("600x650")
        self.root.config(bg="#f0f4f1")

        # Biến lưu đường dẫn ảnh người dùng chọn
        self.selected_img_path = None

        # --- GIAO DIỆN CÁC TIÊU ĐỀ ---
        title_lbl = tk.Label(root, text="NHẬN DIỆN LÁ CÂY THUỐC", font=("Helvetica", 18, "bold"), fg="#1e522f",
                             bg="#f0f4f1")
        title_lbl.pack(pady=15)

        # Khung hiển thị ảnh (Canvas)
        self.canvas = tk.Canvas(root, width=300, height=300, bg="white", highlightthickness=1,
                                highlightbackground="#cccccc")
        self.canvas.pack(pady=10)
        self.canvas.create_text(150, 150, text="Chưa có ảnh nào được chọn", fill="gray", font=("Arial", 10, "italic"))

        # Khung chứa các nút bấm
        btn_frame = tk.Frame(root, bg="#f0f4f1")
        btn_frame.pack(pady=15)

        # Nút bấm 1: Chọn ảnh
        upload_btn = tk.Button(btn_frame, text="📁 Tải Ảnh Lên", font=("Arial", 11, "bold"), bg="#4a90e2", fg="white",
                               padx=15, pady=5, command=self.upload_image)
        upload_btn.grid(row=0, column=0, padx=10)

        # Nút bấm 2: Nhận diện
        self.predict_btn = tk.Button(btn_frame, text="🧠 Nhận Diện Lá", font=("Arial", 11, "bold"), bg="#2e7d32",
                                     fg="white", padx=15, pady=5, state=tk.DISABLED, command=self.classify_image)
        self.predict_btn.grid(row=0, column=1, padx=10)

        # --- KHUNG HIỂN THỊ KẾT QUẢ ---
        result_frame = tk.LabelFrame(root, text=" Kết quả phân lớp ", font=("Arial", 11, "bold"), bg="#f0f4f1",
                                     fg="#1e522f", padx=15, pady=15)
        result_frame.pack(fill="x", padx=40, pady=10)

        self.lbl_vn = tk.Label(result_frame, text="Tên Tiếng Việt: ---", font=("Arial", 12), bg="#f0f4f1", anchor="w")
        self.lbl_vn.pack(fill="x", pady=2)

        self.lbl_en = tk.Label(result_frame, text="Tên Khoa Học: ---", font=("Arial", 12, "italic"), bg="#f0f4f1",
                               anchor="w")
        self.lbl_en.pack(fill="x", pady=2)

    def upload_image(self):
        """Hàm xử lý khi người dùng ấn chọn ảnh từ máy tính"""
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.selected_img_path = file_path

            # Đọc và hiển thị ảnh lên giao diện Canvas
            img = Image.open(file_path)
            img = img.resize((300, 300), Image.Resampling.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(img)

            self.canvas.delete("all")
            self.canvas.create_image(150, 150, image=self.tk_img)

            # Kích hoạt nút bấm nhận diện lá cây
            self.predict_btn.config(state=tk.NORMAL)

            # Reset lại kết quả hiển thị cũ
            self.lbl_vn.config(text="Tên Tiếng Việt: ---")
            self.lbl_en.config(text="Tên Khoa Học: ---")

    def classify_image(self):
        """Hàm kết nối tới bộ não predictor để chạy dự đoán"""
        if not self.selected_img_path:
            return

        # Gọi hàm xử lý từ file predictor.py
        result, error = predict_new_image(self.selected_img_path)

        if error:
            messagebox.showerror("Lỗi hệ thống", error)
        else:
            # Hiển thị thông tin tên cây thuốc lên màn hình giao diện
            self.lbl_vn.config(text=f"Tên Tiếng Việt: {result['vn']}", fg="#d32f2f")
            self.lbl_en.config(text=f"Tên Khoa Học: {result['en']}", fg="#1976d2")


# Khởi chạy chương trình GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = LeafApp(root)
    root.mainloop()