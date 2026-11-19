"""
Module tiện ích dùng chung (`src/utils.py`) cho dự án LAB04: MLP Regression.

Trách nhiệm của module:
1. Quản lý đường dẫn và khởi tạo thư mục tự động (Directory creation).
2. Sinh chuỗi thời gian chuẩn ISO-8601 và mã định danh thực nghiệm (Timestamp & Run ID generation).
3. Thiết lập hạt giống ngẫu nhiên toàn cục (Reproducibility utilities) cho Python, NumPy và TensorFlow.
4. Nạp và lưu tệp cấu hình YAML, đối tượng dữ liệu/mô hình (.pkl/.joblib) và tệp JSON.
5. Cung cấp tiện ích ghi nhận thông báo chuẩn hóa trên màn hình console và xử lý ngoại lệ.

Lưu ý thiết kế: Module này chỉ chứa các hàm tiện ích hạ tầng, tuyệt đối không chứa
logic nghiệp vụ của EDA, Preprocessing, Model hay Evaluation.
"""
import torch
import json
import os
import pickle
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import yaml

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import torch
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.yaml"


def get_project_root() -> Path:
    """
    What: Trả về đường dẫn tuyệt đối đến thư mục gốc của dự án LAB04.
    Why: Đảm bảo mọi module trong `src/` hoặc `notebooks/` đều tham chiếu thống nhất
         tới các thư mục `data/`, `models/`, `reports/` bất kể thư mục làm việc hiện tại.
    Input: Không có tham số đầu vào.
    Output: Đối tượng `pathlib.Path` trỏ tới thư mục gốc của dự án.
    How: Xác định vị trí tệp `src/utils.py` bằng `Path(__file__).resolve()` và lấy
         thư mục cha cấp 2 (`.parent.parent`).
    """
    return ROOT_DIR


def ensure_dir(directory_path: Union[str, Path]) -> Path:
    """
    What: Kiểm tra và tự động tạo thư mục (cùng các thư mục cha) nếu chưa tồn tại.
    Why: Ngăn ngừa lỗi `FileNotFoundError` khi các module lưu dữ liệu, biểu đồ, mô hình
         hoặc tệp nhật ký vào đĩa.
    Input:
        - directory_path (Union[str, Path]): Đường dẫn thư mục cần khởi tạo.
    Output:
        - Path: Đối tượng `pathlib.Path` của thư mục đã được đảm bảo tồn tại.
    How: Chuyển đổi đầu vào sang `Path` và gọi `mkdir(parents=True, exist_ok=True)`.
    """
    path_obj = Path(directory_path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_current_timestamp(fmt: str = "iso") -> str:
    """
    What: Tạo chuỗi thời gian hiện tại theo định dạng chuẩn.
    Why: Phục vụ việc đóng dấu thời gian (timestamp) cho nhật ký thực thi (`logs/`),
         tệp tiến độ (`progress_results/`) và bảng lịch sử thực nghiệm.
    Input:
        - fmt (str): Định dạng mong muốn ("iso" cho ISO-8601 có múi giờ, "compact" cho
          YYYYMMDD_HHMMSS, hoặc "readable" cho YYYY-MM-DD HH:MM:SS).
    Output:
        - str: Chuỗi thời gian đã định dạng.
    How: Lấy thời gian hệ thống kèm múi giờ hiện tại qua `datetime.now().astimezone()`
         và áp dụng định dạng tương ứng.
    """
    now = datetime.now().astimezone()
    if fmt == "compact":
        return now.strftime("%Y%m%d_%H%M%S")
    if fmt == "readable":
        return now.strftime("%Y-%m-%d %H:%M:%S")
    return now.isoformat(timespec="seconds")


def generate_run_id() -> str:
    """
    What: Sinh mã định danh duy nhất (Run ID) cho mỗi lần thực thi Pipeline.
    Why: Giúp phân biệt rõ ràng các lần chạy thực nghiệm khác nhau và không ghi đè
         lịch sử kết quả trước đó (ví dụ: `20261112_170000_a1b2c3`).
    Input: Không có tham số đầu vào.
    Output:
        - str: Chuỗi Run ID theo định dạng `YYYYMMDD_HHMMSS_<6_ky_tu_hex>`.
    How: Kết hợp chuỗi thời gian `compact` từ `get_current_timestamp()` với 6 ký tự
         đầu của mã UUIDv4 ngẫu nhiên (`uuid.uuid4().hex[:6]`).
    """
    timestamp_part = get_current_timestamp(fmt="compact")
    random_hex = uuid.uuid4().hex[:6]
    return f"{timestamp_part}_{random_hex}"


def set_global_seeds(seed: int = 42) -> Dict[str, Any]:
    """
    What: Cố định toàn bộ hạt giống ngẫu nhiên (random seed) cho môi trường thực thi.
    Why: Đảm bảo tính tái lập kết quả thực nghiệm (Reproducibility - Rule 6) khi chia
         dữ liệu Train/Test/Validation, khởi tạo trọng số MLP và huấn luyện Random Forest.
    Input:
        - seed (int): Giá trị hạt giống ngẫu nhiên (mặc định là 42).
    Output:
        - Dict[str, Any]: Từ điển ghi nhận trạng thái thiết lập seed trên từng thư viện.
    How: Thiết lập biến môi trường `PYTHONHASHSEED`, gọi `random.seed(seed)`,
         `np.random.seed(seed)` và `tf.random.set_seed(seed)` (nếu có TensorFlow).
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    torch_seeded = False
    if HAS_PYTORCH:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch_seeded = True
        except Exception:
            torch_seeded = False

    return {
        "seed": seed,
        "python_random": True,
        "numpy_random": True,
        "pytorch_random": torch_seeded,
    }


def load_config(config_path: Union[str, Path, None] = None) -> Dict[str, Any]:
    """
    What: Đọc và phân tích nội dung tệp cấu hình trung tâm `config.yaml`.
    Why: Tập trung toàn bộ siêu tham số (hyperparameters) và đường dẫn tại một nơi duy nhất,
         tránh việc hardcode tham số rải rác trong mã nguồn.
    Input:
        - config_path (Union[str, Path, None]): Đường dẫn tới tệp YAML (mặc định dùng
          `config.yaml` tại thư mục gốc dự án).
    Output:
        - Dict[str, Any]: Từ điển chứa toàn bộ thông số cấu hình.
    How: Mở tệp với mã hóa UTF-8 và dùng `yaml.safe_load()` để chuyển đổi sang `dict`.
    """
    target_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp cấu hình tại: {target_path}")

    with open(target_path, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file) or {}
    return config_data


def save_object(obj: Any, file_path: Union[str, Path]) -> Path:
    """
    What: Lưu một đối tượng Python (ma trận dữ liệu, StandardScaler, mô hình ML) xuống đĩa.
    Why: Cung cấp giao diện lưu trữ thống nhất giữa các giai đoạn Preprocessing, Training
         và Evaluation; tự động tương thích với cả `joblib` và `pickle` chuẩn.
    Input:
        - obj (Any): Đối tượng Python cần lưu trữ.
        - file_path (Union[str, Path]): Đường dẫn tệp đích (`.pkl` hoặc `.joblib`).
    Output:
        - Path: Đường dẫn tệp đã lưu thành công.
    How: Đảm bảo thư mục cha tồn tại, sau đó ưu tiên dùng `joblib.dump()` nếu thư viện
         `joblib` khả dụng, hoặc dùng `pickle.dump()` với giao thức cao nhất.
    """
    target_path = Path(file_path)
    ensure_dir(target_path.parent)

    if HAS_JOBLIB:
        joblib.dump(obj, target_path)
    else:
        with open(target_path, "wb") as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    return target_path


def load_object(file_path: Union[str, Path]) -> Any:
    """
    What: Nạp lại đối tượng Python đã lưu từ đĩa (`.pkl` hoặc `.joblib`).
    Why: Giúp các module `train.py`, `evaluate.py` và `notebooks/lab04.ipynb` đọc lại
         dữ liệu đã tiền xử lý, bộ chuẩn hóa `StandardScaler` và các mô hình đã huấn luyện.
    Input:
        - file_path (Union[str, Path]): Đường dẫn tới tệp cần đọc.
    Output:
        - Any: Đối tượng Python được khôi phục.
    How: Kiểm tra sự tồn tại của tệp, ưu tiên đọc bằng `joblib.load()` và tự động
         chuyển sang `pickle.load()` nếu môi trường không cài đặt `joblib`.
    """
    target_path = Path(file_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp đối tượng tại: {target_path}")

    if HAS_JOBLIB:
        try:
            return joblib.load(target_path)
        except Exception:
            pass

    with open(target_path, "rb") as file:
        return pickle.load(file)


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> Path:
    """
    What: Lưu từ điển dữ liệu dưới định dạng JSON có định dạng thụt lề rõ ràng.
    Why: Phục vụ lưu trữ siêu dữ liệu huấn luyện (`training_metadata.json`), lịch sử
         huấn luyện (`mlp_training_history.json`) và kết quả đánh giá.
    Input:
        - data (Dict[str, Any]): Từ điển dữ liệu cần lưu.
        - file_path (Union[str, Path]): Đường dẫn tệp JSON đích.
    Output:
        - Path: Đường dẫn tệp JSON đã lưu.
    How: Tạo thư mục cha nếu cần và gọi `json.dump(..., ensure_ascii=False, indent=4)`.
    """
    target_path = Path(file_path)
    ensure_dir(target_path.parent)
    with open(target_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4, default=str)
    return target_path


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    What: Đọc nội dung tệp JSON từ đĩa và chuyển thành từ điển Python.
    Why: Phục vụ việc truy xuất kết quả thực nghiệm và tiến trình từ `progress_results/`.
    Input:
        - file_path (Union[str, Path]): Đường dẫn tệp JSON cần đọc.
    Output:
        - Dict[str, Any]: Nội dung JSON dưới dạng từ điển.
    How: Mở tệp với mã hóa UTF-8 và phân tích bằng `json.load()`.
    """
    target_path = Path(file_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp JSON tại: {target_path}")
    with open(target_path, "r", encoding="utf-8") as file:
        return json.load(file)


def configure_console_utf8() -> None:
    """
    What: Cấu hình luồng xuất chuẩn (`stdout`, `stderr`) hỗ trợ bảng mã UTF-8.
    Why: Tránh lỗi `UnicodeEncodeError` khi hiển thị tiếng Việt trên Windows PowerShell
         hoặc Command Prompt.
    Input: Không có tham số đầu vào.
    Output: Không trả về giá trị (`None`).
    How: Gọi phương thức `reconfigure(encoding='utf-8')` trên `sys.stdout` và `sys.stderr`
         nếu phiên bản Python hỗ trợ.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass
