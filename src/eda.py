"""
Module Phân tích Khám phá Dữ liệu (`src/eda.py`) cho dự án LAB04: MLP Regression.

Trách nhiệm của module:
1. Tải bộ dữ liệu gốc thông qua `src/data_loader.py`.
2. Thực hiện phân tích toàn diện theo yêu cầu Part 4.2 & Part 6.2:
   - Tổng quan kích thước dữ liệu (số mẫu, số đặc trưng, kiểu dữ liệu).
   - Phân tích giá trị thiếu (Missing value analysis).
   - Phân tích bản ghi trùng lặp (Duplicate analysis).
   - Thống kê mô tả (Descriptive statistics: mean, std, min, 25%, 50%, 75%, max, skewness).
   - Phân tích phân phối đặc trưng và biến mục tiêu (`MEDV`).
   - Phân tích ma trận tương quan Pearson và xác định các đặc trưng ảnh hưởng mạnh nhất tới giá nhà.
3. Xuất các bảng thống kê ra `reports/metrics/` và tạo các biểu đồ EDA (Biểu đồ 1, 2, 3 và
   Feature-Target Relationships) lưu vào `reports/figures/`.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.data_loader import FEATURE_DESCRIPTIONS, inspect_dataset, load_housing_dataframe
from src.tracking import create_run, finish_run, update_progress, write_log
from src.utils import ensure_dir, get_project_root, load_config, save_json
from src.visualize import (
    create_stage_figure_dir,
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_feature_target_relationships,
    plot_target_distribution,
)

ROOT_DIR = get_project_root()
METRICS_DIR = ROOT_DIR / "reports" / "metrics"


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    What: Thực thi toàn bộ quy trình Phân tích Khám phá Dữ liệu (EDA), lưu bảng thống kê CSV/JSON
          và sinh bộ biểu đồ EDA.
    Why: Giúp hiểu sâu cấu trúc phân phối, độ lệch và mối tương quan giữa 13 đặc trưng kinh tế - xã hội
         với giá nhà `MEDV` trước khi xây dựng mô hình MLP Regression.
    Input:
        - run_id (Optional[str]): Mã định danh phiên chạy hiện tại.
    Output:
        - Dict[str, Any]: Báo cáo kết quả EDA và danh sách đường dẫn các tệp biểu đồ/bảng biểu đã lưu.
    How:
        1. Nạp dữ liệu qua `load_housing_dataframe()`.
        2. Chạy `inspect_dataset()` để tính thống kê mô tả, độ lệch (`skew`) và tương quan.
        3. Lưu các tệp CSV thống kê vào `reports/metrics/`.
        4. Gọi các hàm vẽ biểu đồ trong `src/visualize.py` và cập nhật `progress_results/`.
    """
    own_run = run_id is None
    if own_run:
        run_id = create_run()

    start_time = time.perf_counter()
    write_log(run_id, "EDA", "START", "Bắt đầu Phân tích Khám phá Dữ liệu (Exploratory Data Analysis).")

    try:
        config = load_config()
        df, source_info = load_housing_dataframe(config)
        target_col = source_info["target_column"]
        inspection = inspect_dataset(df, target_col=target_col)

        ensure_dir(METRICS_DIR)
        stage_fig_dir = create_stage_figure_dir(run_id, "eda")

        # Tính bảng thống kê mô tả mở rộng (kèm skewness & kurtosis)
        desc_df = df.describe().T
        desc_df["skewness"] = df.skew(numeric_only=True)
        desc_df["kurtosis"] = df.kurtosis(numeric_only=True)
        desc_df["missing_count"] = df.isnull().sum()
        desc_df["description"] = [FEATURE_DESCRIPTIONS.get(col, "") for col in desc_df.index]
        desc_df = desc_df.round(4)

        desc_csv_path = METRICS_DIR / "eda_descriptive_statistics.csv"
        desc_df.to_csv(desc_csv_path, encoding="utf-8-sig")
        
        corr_matrix = df.corr(numeric_only=True).round(4)
        corr_csv_path = METRICS_DIR / "eda_correlation_matrix.csv"
        corr_matrix.to_csv(corr_csv_path, encoding="utf-8-sig")

        # Phân tích điểm ngoại lai (Outlier Analysis) bằng IQR và Z-score (chỉ thống kê, không xóa mẫu)
        outlier_rows = []
        for col in df.select_dtypes(include=["number"]).columns:
            series = df[col].dropna()
            q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
            iqr = q3 - q1
            low_b, high_b = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            iqr_cnt = int(((series < low_b) | (series > high_b)).sum())
            std_val = float(series.std(ddof=0))
            z_cnt = int((series.sub(series.mean()).abs().div(std_val if std_val > 1e-12 else 1.0) > 3.0).sum())
            outlier_rows.append({
                "Feature": col,
                "Q1": round(q1, 4),
                "Q3": round(q3, 4),
                "IQR": round(iqr, 4),
                "Lower_Bound": round(low_b, 4),
                "Upper_Bound": round(high_b, 4),
                "IQR_Outlier_Count": iqr_cnt,
                "IQR_Outlier_Pct": round(iqr_cnt / len(series) * 100.0, 2),
                "ZScore_gt3_Count": z_cnt,
            })
        outlier_df = pd.DataFrame(outlier_rows)
        outlier_df.to_csv(METRICS_DIR / "eda_outlier_analysis.csv", index=False, encoding="utf-8-sig")


        # Tạo 4 bộ biểu đồ EDA
        saved_figures = []
        saved_figures.extend(plot_target_distribution(df, target_col=target_col, stage_dir=stage_fig_dir))
        saved_figures.extend(plot_correlation_heatmap(df, stage_dir=stage_fig_dir))
        saved_figures.extend(plot_feature_distributions(df, target_col=target_col, stage_dir=stage_fig_dir))
        saved_figures.extend(plot_feature_target_relationships(df, target_col=target_col, stage_dir=stage_fig_dir))

        eda_summary = {
            "dataset_name": source_info["dataset_name"],
            "n_samples": inspection["n_samples"],
            "n_features": inspection["n_features"],
            "target_column": target_col,
            "total_missing_values": inspection["total_missing_values"],
            "duplicate_records": inspection["duplicate_records"],
            "target_summary": inspection["target_summary"],
            "correlation_with_target": inspection["correlation_with_target"],
            "censoring_count_at_50k": int((df[target_col] >= 50.0).sum()),
        }
        save_json(eda_summary, METRICS_DIR / "eda_summary.json")

        duration = round(time.perf_counter() - start_time, 4)

        print("\n" + "=" * 65)
        print("BƯỚC 2: PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EXPLORATORY DATA ANALYSIS)")
        print("=" * 65)
        print(f"[INFO] Số lượng mẫu (Samples)     : {inspection['n_samples']}")
        print(f"[INFO] Số lượng đặc trưng         : {inspection['n_features']} ({', '.join(inspection['feature_names'])})")
        print(f"[INFO] Biến mục tiêu ({target_col})       : Mean = ${inspection['target_summary']['mean']:.2f}k, "
              f"Std = ${inspection['target_summary']['std']:.2f}k, Median = ${inspection['target_summary']['50%']:.2f}k")
        print(f"[INFO] Số mẫu chạm trần $50,000   : {eda_summary['censoring_count_at_50k']} mẫu")
        print("[INFO] Top 5 đặc trưng tương quan mạnh nhất với MEDV:")
        for feat, r_val in list(inspection["correlation_with_target"].items())[:3]:
            print(f"   + Tương quan thuận : {feat:<8} -> r = {r_val:+.4f}")
        for feat, r_val in list(inspection["correlation_with_target"].items())[-3:]:
            print(f"   - Tương quan nghịch: {feat:<8} -> r = {r_val:+.4f}")

        output_payload = {
            "eda_summary": eda_summary,
            "saved_tables": [
                "reports/metrics/eda_descriptive_statistics.csv",
                "reports/metrics/eda_correlation_matrix.csv",
                "reports/metrics/eda_outlier_analysis.csv",
                "reports/metrics/eda_summary.json",
            ],
            "saved_figures": saved_figures,
            "duration_seconds": duration,
        }

        update_progress(
            run_id,
            "EDA",
            "SUCCESS",
            input_data={"dataset": source_info["file_path"]},
            output_data=output_payload,
        )
        write_log(run_id, "EDA", "SUCCESS", f"Hoàn thành phân tích EDA và xuất biểu đồ trong {duration}s.")

        if own_run:
            finish_run(run_id, "SUCCESS")

        return output_payload

    except Exception as exc:
        update_progress(run_id, "EDA", "FAILED", error=str(exc))
        write_log(run_id, "EDA", "FAILED", str(exc))
        if own_run:
            finish_run(run_id, "FAILED")
        raise


if __name__ == "__main__":
    main()
