"""
Module điều phối quy trình thực thi tổng thể (`src/pipeline.py`) cho dự án LAB04.

Trách nhiệm của module:
1. Kết nối tuần tự toàn bộ các giai đoạn của bài thực hành LAB04 trong một luồng thống nhất:
   Data Loading → EDA → Preprocessing → Training (MLP + Baselines) → Evaluation & Visualization → Reporting.
2. Tạo và truyền một mã định danh `Run_ID` duy nhất xuyên suốt tất cả các bước để nhật ký
   (`logs/pipeline_<run_id>.log`), tiến độ (`progress_results/progress_<run_id>.json`) và
   báo cáo (`reports/experiment_summary.md`) đồng bộ tuyệt đối.
3. Hỗ trợ các cờ dòng lệnh (`--skip-eda`, `--skip-preprocess`, `--skip-train`) khi người dùng
   chỉ muốn chạy lại bước đánh giá và xuất báo cáo trên các mô hình đã lưu.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import data_loader, eda, evaluate, preprocess, report, train
from src.model import get_backend_info
from src.tracking import create_run, finish_run, update_progress, write_log
from src.utils import configure_console_utf8, load_config, set_global_seeds


def run_pipeline(
    skip_eda: bool = False,
    skip_preprocess: bool = False,
    skip_train: bool = False,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    What: Chạy toàn bộ quy trình Machine Learning của dự án LAB04 từ đầu đến cuối.
    Why: Cho phép thực thi toàn bộ dự án chỉ bằng một lệnh duy nhất (`python run_lab04.py`),
         đảm bảo tính nhất quán, tái lập kết quả và chống rò rỉ dữ liệu.
    Input:
        - skip_eda (bool): Bỏ qua bước phân tích khám phá dữ liệu nếu đặt `True`.
        - skip_preprocess (bool): Bỏ qua bước tiền xử lý nếu đặt `True`.
        - skip_train (bool): Bỏ qua bước huấn luyện nếu đặt `True`.
        - run_id (Optional[str]): Mã định danh phiên chạy tùy chọn.
    Output:
        - Dict[str, Any]: Tóm tắt kết quả thực thi toàn trình (Run ID, thời gian, kết quả đánh giá).
    How: Khởi tạo `run_id`, thiết lập random seed toàn cục từ `config.yaml`, thực thi tuần tự
         các module, đóng phiên chạy bằng `finish_run()` và xuất báo cáo tổng hợp.
    """
    configure_console_utf8()
    config = load_config()
    seed = int(config.get("data", {}).get("random_state", 42))
    seed_info = set_global_seeds(seed)
    backend_info = get_backend_info()

    active_run_id = create_run(
        run_id=run_id,
        metadata={
            "seed_info": seed_info,
            "backend_info": backend_info,
            "config_path": "config.yaml",
        },
    )

    core_steps = [
        ("DATA_LOADING", data_loader.main, False),
        ("EDA", eda.main, skip_eda),
        ("PREPROCESSING", preprocess.main, skip_preprocess),
        ("TRAINING", train.main, skip_train),
        ("EVALUATION", evaluate.main, False),
    ]

    completed_steps: List[str] = []
    skipped_steps: List[str] = []
    step_outputs: Dict[str, Any] = {}
    pipeline_start = time.perf_counter()

    print("\n" + "#" * 68)
    print("   KHỞI ĐỘNG PIPELINE LAB04: MULTILAYER PERCEPTRON REGRESSION")
    print("   Bài toán: Dự đoán Giá Nhà (House Price Prediction — Chapter 5)")
    print("#" * 68)
    print(f"[INFO] Run ID        : {active_run_id}")
    print(f"[INFO] Random Seed   : {seed}")
    print(f"[INFO] MLP Backend   : {backend_info['mlp_backend']}")

    write_log(active_run_id, "PIPELINE", "START", f"Bắt đầu thực thi Pipeline LAB04 (Run_ID={active_run_id}).")

    try:
        for step_name, step_fn, should_skip in core_steps:
            if should_skip:
                print(f"\n[SKIP] Bỏ qua bước {step_name} theo cấu hình tham số dòng lệnh.")
                skipped_steps.append(step_name)
                update_progress(
                    active_run_id,
                    step_name,
                    "SKIPPED",
                    output_data={"message": "Bước này được bỏ qua theo yêu cầu người dùng."},
                )
                write_log(active_run_id, step_name, "SKIPPED", "Bỏ qua theo cờ dòng lệnh.")
                continue

            step_outputs[step_name] = step_fn(active_run_id)
            completed_steps.append(step_name)

        completed_steps.append("REPORTING")
        total_elapsed = round(time.perf_counter() - pipeline_start, 4)

        pipeline_summary = {
            "run_id": active_run_id,
            "status": "SUCCESS",
            "total_execution_time_seconds": total_elapsed,
            "completed_steps": completed_steps,
            "skipped_steps": skipped_steps,
            "best_models": step_outputs.get("EVALUATION", {}).get("best_models", {}),
        }

        update_progress(active_run_id, "PIPELINE", "SUCCESS", output_data=pipeline_summary)
        write_log(active_run_id, "PIPELINE", "SUCCESS", f"Pipeline LAB04 hoàn tất thành công trong {total_elapsed}s.")
        finish_run(active_run_id, status="SUCCESS", summary_data=pipeline_summary)

        # Xuất báo cáo sau khi trạng thái SUCCESS và end_time đã được cập nhật
        step_outputs["REPORTING"] = report.main(active_run_id)
        finish_run(active_run_id, status="SUCCESS", summary_data=pipeline_summary)

        print("\n" + "#" * 68)
        print("   HOÀN TẤT TOÀN BỘ PIPELINE LAB04 THÀNH CÔNG!")
        print("#" * 68)
        print(f"[INFO] Run ID              : {active_run_id}")
        print(f"[INFO] Các bước hoàn thành : {', '.join(completed_steps)}")
        print(f"[INFO] Tổng thời gian chạy : {total_elapsed:.4f} giây")
        print(f"[INFO] Báo cáo tổng hợp    : reports/experiment_summary.md")
        print(f"[INFO] Nhật ký chi tiết    : logs/pipeline_{active_run_id}.log")
        print(f"[INFO] Tiến độ JSON        : progress_results/progress_{active_run_id}.json")
        print("#" * 68 + "\n")

        return pipeline_summary

    except Exception as exc:
        total_elapsed = round(time.perf_counter() - pipeline_start, 4)
        update_progress(
            active_run_id,
            "PIPELINE",
            "FAILED",
            output_data={
                "total_execution_time_seconds": total_elapsed,
                "completed_steps": completed_steps,
                "skipped_steps": skipped_steps,
            },
            error=str(exc),
        )
        write_log(active_run_id, "PIPELINE", "FAILED", str(exc))
        finish_run(active_run_id, status="FAILED")
        print(f"\n[ERROR] Pipeline dừng do lỗi: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise


def main() -> None:
    """
    What: Hàm phân tích tham số dòng lệnh (`argparse`) và kích hoạt `run_pipeline()`.
    Why: Cho phép chạy `python -m src.pipeline` hoặc `python run_lab04.py` với các tùy chọn linh hoạt.
    Input: Đọc tham số từ `sys.argv`.
    Output: Không trả về giá trị (`None`).
    How: Khởi tạo `ArgumentParser` và truyền các cờ vào `run_pipeline()`.
    """
    parser = argparse.ArgumentParser(
        description="Điều phối toàn bộ quy trình thực nghiệm LAB04: Multilayer Perceptron Regression."
    )
    parser.add_argument("--skip-eda", action="store_true", help="Bỏ qua bước phân tích khám phá dữ liệu (EDA).")
    parser.add_argument("--skip-preprocess", action="store_true", help="Bỏ qua bước tiền xử lý dữ liệu.")
    parser.add_argument("--skip-train", action="store_true", help="Bỏ qua bước huấn luyện mô hình.")
    args = parser.parse_args()

    run_pipeline(
        skip_eda=args.skip_eda,
        skip_preprocess=args.skip_preprocess,
        skip_train=args.skip_train,
    )


if __name__ == "__main__":
    main()
