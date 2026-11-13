"""
Cách sử dụng:
    1. Chạy chuẩn Single Source of Truth (toàn bộ biểu đồ lưu tập trung tại `reports/figures/`):
       python run_lab04.py
    2. Tùy chọn đóng gói riêng thư mục `docs/` độc lập (tự động copy ảnh sang `docs/images/` khi cần nộp lẻ thư mục `docs/`):
       python run_lab04.py --bundle-docs
"""

import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import main


def bundle_standalone_docs() -> None:
    """Sao chép bộ biểu đồ từ `reports/figures/` sang `docs/images/` khi người dùng cần tách riêng thư mục `docs/`."""
    fig_dir = ROOT_DIR / "reports" / "figures"
    docs_img_dir = ROOT_DIR / "docs" / "images"
    docs_img_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for png in fig_dir.glob("*.png"):
        shutil.copy2(png, docs_img_dir / png.name)
        count += 1
    print(f"[BUNDLE-DOCS] Đã đồng bộ {count} biểu đồ vào docs/images/ để có thể copy riêng thư mục docs/.")


if __name__ == "__main__":
    main()
    if "--bundle-docs" in sys.argv:
        bundle_standalone_docs()
