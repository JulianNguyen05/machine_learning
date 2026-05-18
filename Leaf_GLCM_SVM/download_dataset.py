import os
import zipfile
from pathlib import Path


def download_and_extract_dataset():
    # 1. Xác định đường dẫn đích trong dự án
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, 'dataset')
    extract_to = os.path.join(target_dir, 'swedish_leaf_dataset')
    zip_path = os.path.join(target_dir, 'swedish-leaf-dataset.zip')

    os.makedirs(target_dir, exist_ok=True)

    print("⏳ Đang kết nối và tải dataset từ Kaggle... (Vui lòng chờ)")
    try:
        # Cài đặt thư viện kaggle trực tiếp bằng code nếu chưa có
        try:
            import kaggle
        except ImportError:
            print("📦 Chưa tìm thấy thư viện 'kaggle', đang tự động cài đặt...")
            os.system('pip install kaggle')
            import kaggle

        # Gọi API Kaggle để tải file zip
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('majorproject24/swedish-leaf-dataset', path=target_dir, unzip=False)

        # 2. Tiến hành giải nén
        print("📦 Tải xong file Zip! Đang tiến hành giải nén vào thư mục dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        # 3. Dọn dẹp file zip sau khi xong
        if os.path.exists(zip_path):
            os.remove(zip_path)

        print(f"🎉 Hoàn thành xuất sắc! Dataset đã sẵn sàng tại: {extract_to}")

    except Exception as e:
        print(f"❌ Lỗi không thể tải dữ liệu: {e}")
        print("\n💡 HƯỚNG DẪN SỬA LỖI:")
        print("Để tính năng này chạy được, bạn cần để file 'kaggle.json' lấy từ tài khoản Kaggle của bạn vào thư mục:")
        print(" - Windows: C:\\Users\\<Tên_User>\\.kaggle\\kaggle.json")
        print(" - Linux/Mac: ~/.kaggle/kaggle.json")


if __name__ == "__main__":
    download_and_extract_dataset()