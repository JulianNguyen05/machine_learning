import os


def generate_tree(dir_path, exclude_dirs, prefix=""):
    """Hàm đệ quy để tạo sơ đồ cây thư mục"""
    tree_str = ""
    try:
        entries = os.listdir(dir_path)
    except PermissionError:
        return ""

    # Lọc bỏ các thư mục không cần thiết
    entries = [e for e in entries if e not in exclude_dirs]
    entries.sort()

    entries_count = len(entries)
    for i, entry in enumerate(entries):
        is_last = i == (entries_count - 1)
        full_path = os.path.join(dir_path, entry)

        connector = "└── " if is_last else "├── "
        tree_str += prefix + connector + entry + "\n"

        if os.path.isdir(full_path):
            extension = "    " if is_last else "│   "
            tree_str += generate_tree(full_path,
                                      exclude_dirs, prefix + extension)

    return tree_str


def extract_project_code(output_filename="project_source_code.txt"):
    root_dir = "."

    # Các thư mục ẩn hoặc chứa file nặng cần bỏ qua khi duyệt
    exclude_dirs = {
        '.venv', 'venv', 'env', '__pycache__',
        '.git', '.idea', '.vscode'
    }

    # Các định dạng file muốn lấy nội dung (đã thêm .txt để lấy requirements.txt)
    allowed_extensions = {'.py', '.md', '.txt', '.gitignore'}

    # Các file đặc biệt cần bỏ qua (để không lấy nhầm dữ liệu thô)
    exclude_files = {'leaf_features_training.csv'}

    print("Đang tiến hành gom mã nguồn và vẽ cây thư mục...")

    with open(output_filename, 'w', encoding='utf-8') as outfile:
        # 1. In sơ đồ cây thư mục lên đầu file
        outfile.write(
            "============================================================\n")
        outfile.write("--- CẤU TRÚC CÂY THƯ MỤC DỰ ÁN ---\n")
        outfile.write(
            "============================================================\n")
        outfile.write(".\n")
        outfile.write(generate_tree(root_dir, exclude_dirs))
        outfile.write("\n\n")

        # 2. Duyệt để lấy mã nguồn
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Cắt tỉa các thư mục không cần duyệt
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

            for filename in filenames:
                # Bỏ qua các file dữ liệu, model hoặc script này
                if filename in exclude_files or filename == output_filename or filename == "extract_code.py":
                    continue

                # Chỉ lấy các file có đuôi cho phép
                if any(filename.endswith(ext) for ext in allowed_extensions) or filename == ".gitignore":
                    filepath = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(filepath, root_dir)

                    # Ghi tiêu đề phân cách
                    outfile.write(f"\n{'=' * 60}\n")
                    outfile.write(f"--- FILE: {rel_path} ---\n")
                    outfile.write(f"{'=' * 60}\n\n")

                    # Đọc và ghi nội dung
                    try:
                        with open(filepath, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(
                            f"# [CẢNH BÁO] Không thể đọc file này. Lỗi: {e}\n")

                    outfile.write("\n")
                    print(f"Đã thêm code: {rel_path}")


if __name__ == "__main__":
    output_file = "project_source_code.txt"
    extract_project_code(output_file)
    print(f"\nHoàn tất! File tổng hợp đã được lưu tại: {output_file}")
