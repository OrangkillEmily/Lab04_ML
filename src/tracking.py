"""
Module quản lý theo dõi thực nghiệm và nhật ký hệ thống (`src/tracking.py`).

Trách nhiệm của module:
1. Khởi tạo phiên thực nghiệm mới với mã định danh `Run_ID` duy nhất.
2. Ghi nhận chi tiết trạng thái, dữ liệu đầu vào, đầu ra và thời gian thực thi của
   từng bước trong Pipeline vào thư mục `progress_results/progress_<run_id>.json`.
3. Ghi nhật ký hoạt động tuần tự theo dòng thời gian vào `logs/pipeline_<run_id>.log`.
4. Lưu trữ bảng lịch sử chạy thực nghiệm và bảng chỉ số đánh giá (MSE, RMSE, MAE, MAPE, R²)
   vào `reports/metrics/` và `docs/experiment_logs.csv` mà không ghi đè dữ liệu của các lần chạy trước.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils import (
    ensure_dir,
    generate_run_id,
    get_current_timestamp,
    get_project_root,
    load_json,
    save_json,
)

ROOT_DIR = get_project_root()
LOG_DIR = ROOT_DIR / "logs"
PROGRESS_DIR = ROOT_DIR / "progress_results"
METRICS_DIR = ROOT_DIR / "reports" / "metrics"
EXPERIMENT_LOG_CSV = ROOT_DIR / "docs" / "experiment_logs.csv"

ensure_dir(LOG_DIR)
ensure_dir(PROGRESS_DIR)
ensure_dir(METRICS_DIR)


def create_run(run_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    What: Khởi tạo một phiên chạy thực nghiệm mới và tạo tệp JSON theo dõi tiến độ.
    Why: Đảm bảo mọi lần thực thi Pipeline hoặc từng module độc lập đều có hồ sơ
         theo dõi riêng biệt, phục vụ đối chiếu và kiểm toán chất lượng (QA Audit).
    Input:
        - run_id (Optional[str]): Mã Run ID tùy chọn (nếu `None` sẽ tự động sinh mới).
        - metadata (Optional[Dict[str, Any]]): Thông tin bổ sung về môi trường hoặc cấu hình.
    Output:
        - str: Chuỗi `run_id` vừa được khởi tạo.
    How: Sinh `run_id`, tạo cấu trúc từ điển chứa `run_id`, `start_time`, `status="RUNNING"`,
         `metadata` và danh sách `steps=[]`, sau đó lưu vào `progress_results/progress_<run_id>.json`.
    """
    active_run_id = run_id or generate_run_id()
    run_data = {
        "run_id": active_run_id,
        "project": "LAB04_MLP_Regression",
        "start_time": get_current_timestamp("iso"),
        "end_time": None,
        "status": "RUNNING",
        "metadata": metadata or {},
        "steps": [],
    }
    progress_path = PROGRESS_DIR / f"progress_{active_run_id}.json"
    save_json(run_data, progress_path)
    return active_run_id


def write_log(run_id: str, step_name: str, status: str, message: str) -> Path:
    """
    What: Ghi một dòng nhật ký sự kiện vào tệp `logs/pipeline_<run_id>.log`.
    Why: Cung cấp lịch sử hoạt động chi tiết theo thời gian thực để dễ dàng kiểm tra
         tiến độ hoặc truy vết nguyên nhân lỗi khi hệ thống gặp sự cố.
    Input:
        - run_id (str): Mã định danh của lần chạy hiện tại.
        - step_name (str): Tên giai đoạn (ví dụ: `DATA_LOADING`, `EDA`, `PREPROCESSING`, `TRAINING`).
        - status (str): Trạng thái sự kiện (`START`, `INFO`, `SUCCESS`, `WARNING`, `FAILED`).
        - message (str): Nội dung thông điệp chi tiết.
    Output:
        - Path: Đường dẫn tới tệp log đã ghi.
    How: Mở tệp `logs/pipeline_<run_id>.log` ở chế độ nối thêm (`append`) với mã hóa UTF-8
         và ghi dòng định dạng `[timestamp] [status] [step_name] message`.
    """
    ensure_dir(LOG_DIR)
    log_path = LOG_DIR / f"pipeline_{run_id}.log"
    timestamp = get_current_timestamp("iso")
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(f"[{timestamp}] [{status}] [{step_name}] {message}\n")
    return log_path


def update_progress(
    run_id: str,
    step_name: str,
    status: str,
    input_data: Optional[Dict[str, Any]] = None,
    output_data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Path:
    """
    What: Cập nhật kết quả thực thi của một bước vào tệp `progress_results/progress_<run_id>.json`.
    Why: Lưu trữ có cấu trúc toàn bộ thông số đầu vào, đầu ra, kích thước dữ liệu và
         metrics của từng giai đoạn để module `report.py` tự động tổng hợp báo cáo.
    Input:
        - run_id (str): Mã định danh của lần chạy hiện tại.
        - step_name (str): Tên bước thực hiện.
        - status (str): Trạng thái hoàn thành (`SUCCESS`, `SKIPPED`, `FAILED`).
        - input_data (Optional[Dict[str, Any]]): Thông tin đầu vào của bước.
        - output_data (Optional[Dict[str, Any]]): Kết quả đầu ra của bước.
        - error (Optional[str]): Thông báo lỗi nếu bước thực thi thất bại.
    Output:
        - Path: Đường dẫn tệp JSON tiến độ đã cập nhật.
    How: Đọc tệp JSON hiện tại của `run_id`, nối bản ghi bước mới vào danh sách `steps`,
         cập nhật trạng thái tổng nếu gặp lỗi `FAILED`, và ghi lại xuống đĩa.
    """
    progress_path = PROGRESS_DIR / f"progress_{run_id}.json"
    if not progress_path.exists():
        create_run(run_id=run_id)

    run_data = load_json(progress_path)
    step_record = {
        "step": step_name,
        "timestamp": get_current_timestamp("iso"),
        "status": status,
        "input": input_data,
        "output": output_data,
        "error": error,
    }
    run_data.setdefault("steps", []).append(step_record)

    if status == "FAILED":
        run_data["status"] = "FAILED"

    save_json(run_data, progress_path)
    return progress_path


def finish_run(
    run_id: str,
    status: str = "SUCCESS",
    summary_data: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    What: Đóng phiên chạy thực nghiệm, ghi nhận thời gian kết thúc và trạng thái cuối cùng.
    Why: Đánh dấu một chu trình Pipeline đã hoàn tất trọn vẹn (`SUCCESS`) hoặc dừng do lỗi (`FAILED`).
    Input:
        - run_id (str): Mã định danh của lần chạy.
        - status (str): Trạng thái kết thúc (`SUCCESS` hoặc `FAILED`).
        - summary_data (Optional[Dict[str, Any]]): Tóm tắt kết quả chung cuộc của lần chạy.
    Output:
        - Path: Đường dẫn tệp JSON tiến độ sau khi hoàn tất.
    How: Cập nhật trường `end_time`, `status` và `summary` trong tệp `progress_<run_id>.json`.
    """
    progress_path = PROGRESS_DIR / f"progress_{run_id}.json"
    if not progress_path.exists():
        create_run(run_id=run_id)

    run_data = load_json(progress_path)
    run_data["end_time"] = get_current_timestamp("iso")
    run_data["status"] = status
    if summary_data is not None:
        run_data["summary"] = summary_data

    save_json(run_data, progress_path)
    return progress_path


def append_experiment_log(
    run_id: str,
    dataset_name: str,
    dataset_shape: List[int],
    model_name: str,
    hyperparameters: Dict[str, Any],
    training_time_sec: float,
    metrics: Dict[str, float],
    status: str = "SUCCESS",
) -> Path:
    """
    What: Ghi nối thêm một dòng kết quả thực nghiệm của từng mô hình vào `docs/experiment_logs.csv`
          và `reports/metrics/experiment_history.csv`.
    Why: Lưu trữ bảng theo dõi xuyên suốt các lần chạy (Run ID, Timestamp, Dataset, Model,
         Hyperparameters, Training Time, MSE, RMSE, MAE, MAPE (%), R², Status) mà không ghi đè lần chạy cũ.
    Input:
        - run_id (str): Mã định danh lần chạy.
        - dataset_name (str): Tên bộ dữ liệu sử dụng.
        - dataset_shape (List[int]): Kích thước bộ dữ liệu `[n_samples, n_columns]`.
        - model_name (str): Tên mô hình (MLP Regression, Linear Regression, Decision Tree, Random Forest).
        - hyperparameters (Dict[str, Any]): Siêu tham số của mô hình.
        - training_time_sec (float): Thời gian huấn luyện (giây).
        - metrics (Dict[str, float]): Từ điển chứa `MSE`, `RMSE`, `MAE`, `MAPE`, `R2` trên tập Test.
        - status (str): Trạng thái thực thi.
    Output:
        - Path: Đường dẫn tới tệp `docs/experiment_logs.csv`.
    How: Chuẩn bị bản ghi theo schema chuẩn và ghi nối thêm (`append`) vào cả hai tệp CSV
         trong `docs/` và `reports/metrics/`.
    """
    fieldnames = [
        "Run_ID",
        "Timestamp",
        "Dataset_Name",
        "Dataset_Shape",
        "Model_Name",
        "Hyperparameters",
        "Training_Time_Sec",
        "Test_MSE",
        "Test_RMSE",
        "Test_MAE",
        "Test_MAPE_Pct",
        "Test_R2",
        "Execution_Status",
    ]

    row = {
        "Run_ID": run_id,
        "Timestamp": get_current_timestamp("readable"),
        "Dataset_Name": dataset_name,
        "Dataset_Shape": f"{dataset_shape[0]}x{dataset_shape[1]}",
        "Model_Name": model_name,
        "Hyperparameters": json.dumps(hyperparameters, ensure_ascii=False),
        "Training_Time_Sec": round(float(training_time_sec), 4),
        "Test_MSE": round(float(metrics.get("MSE", 0.0)), 4),
        "Test_RMSE": round(float(metrics.get("RMSE", 0.0)), 4),
        "Test_MAE": round(float(metrics.get("MAE", 0.0)), 4),
        "Test_MAPE_Pct": round(float(metrics.get("MAPE", 0.0)), 4),
        "Test_R2": round(float(metrics.get("R2", 0.0)), 4),
        "Execution_Status": status,
    }

    for target_csv in [EXPERIMENT_LOG_CSV, METRICS_DIR / "experiment_history.csv"]:
        ensure_dir(target_csv.parent)
        file_exists = target_csv.exists() and target_csv.stat().st_size > 0
        with open(target_csv, "a", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    return EXPERIMENT_LOG_CSV
