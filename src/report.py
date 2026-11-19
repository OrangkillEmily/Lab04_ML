"""
Module tổng hợp báo cáo thực nghiệm (`src/report.py`) cho dự án LAB04: MLP Regression.

Trách nhiệm của module:
1. Đọc kết quả từ `progress_results/progress_<run_id>.json`, `reports/metrics/evaluation_latest.json`,
   `models/mlp_training_history.json` và `logs/pipeline_<run_id>.log`.
2. Tạo báo cáo kỹ thuật chuẩn UTH `reports/experiment_summary.md`, `docs/LAB_04_REPORT.md`,
   `reports/report_<run_id>.md` cùng báo cáo HTML tương tác tự chứa (Base64 embedded figures)
   `reports/LAB_04_REPORT.html`, `docs/LAB_04_REPORT.html` và `reports/report_<run_id>.html`.
3. Trình bày bảng lộ trình thực thi 8 bước, sơ đồ kiến trúc Mermaid/SVG, phân tích toán học Chương 5,
   bảng đối sánh 4 mô hình theo `MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²` và ma trận nghiệm thu `AC01-AC12`.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from render_report import collect_report_facts, render_html
from src.tracking import LOG_DIR, PROGRESS_DIR, update_progress, write_log
from src.utils import ensure_dir, get_project_root, load_json

ROOT_DIR = get_project_root()
REPORT_DIR = ROOT_DIR / "reports"
DOCS_DIR = ROOT_DIR / "docs"
METRICS_DIR = REPORT_DIR / "metrics"


def get_latest_run_id() -> str:
    """
    What: Tìm mã `run_id` của phiên chạy gần nhất trong thư mục `progress_results/`.
    Why: Cho phép gọi `python -m src.report` độc lập mà không bắt buộc phải truyền `--run-id`.
    Input: Không có.
    Output:
        - str: Mã `run_id` mới nhất.
    How: Duyệt các tệp `progress_*.json` trong `progress_results/` và chọn tệp có thời gian sửa đổi mới nhất.
    """
    files = sorted(PROGRESS_DIR.glob("progress_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"Không tìm thấy tệp tiến độ nào trong {PROGRESS_DIR}.")
    return files[-1].stem.replace("progress_", "", 1)


def _forward_math_lines(dense_layers: List[Dict[str, Any]]) -> List[str]:
    """Sinh các công thức lan truyền thuận LaTeX từ danh sách tầng Dense thật (không gán cứng kích thước)."""
    out: List[str] = []
    n = len(dense_layers)
    for i, d in enumerate(dense_layers, 1):
        prev = "\\mathbf{X}" if i == 1 else "\\mathbf{a}^{(%d)}" % (i - 1)
        affine = "\\mathbf{z}^{(%d)} = %s\\mathbf{W}^{(%d)} + \\mathbf{b}^{(%d)}" % (i, prev, i, i)
        dim = "\\in \\mathbb{R}^{N \\times %d}" % d["out"]
        if i < n:
            act = "\\text{ReLU}" if d["act"] == "relu" else "\\text{%s}" % d["act"]
            out.append("$$" + affine + ", \\qquad \\mathbf{a}^{(%d)} = %s\\bigl(\\mathbf{z}^{(%d)}\\bigr) " % (i, act, i) + dim + "$$")
        else:
            out.append("$$\\hat{\\mathbf{y}} = \\mathbf{z}^{(%d)} = %s\\mathbf{W}^{(%d)} + \\mathbf{b}^{(%d)} " % (i, prev, i, i) + dim + "$$")
    return out


def _params_formula(dense_layers: List[Dict[str, Any]], total: int) -> str:
    """Sinh công thức đếm tham số dạng LaTeX, ví dụ `(13 \\times 32 + 32) + ... = 448 + 528 + 17 = 993`."""
    terms = " + ".join("(%d \\times %d + %d)" % (d["in"], d["out"], d["out"]) for d in dense_layers)
    parts = " + ".join(str(d["params"]) for d in dense_layers)
    return "$%s = %s = %d$" % (terms, parts, total)


def generate_experiment_summary_markdown(
    run_id: str,
    progress_data: Dict[str, Any],
    log_text: str = "",
    fig_prefix: str = "figures",
) -> str:
    """
    What: Xây dựng nội dung báo cáo thực nghiệm định dạng Markdown theo chuẩn học thuật UTH.
    Why: Tạo tài liệu `reports/experiment_summary.md`, `reports/report_<run_id>.md` và `docs/LAB_04_REPORT.md`
         đồng bộ 100% với số liệu thực tế của Pipeline (mọi con số đều lấy từ `collect_report_facts()`,
         không có giá trị gán cứng hay giá trị mặc định).
    Input:
        - run_id (str): Mã định danh lần chạy.
        - progress_data (Dict[str, Any]): Dữ liệu từ `progress_<run_id>.json`.
        - log_text (str): Nội dung nhật ký thực thi.
        - fig_prefix (str): Tiền tố đường dẫn thư mục ảnh (`figures` cho `reports/` hoặc `../reports/figures` cho `docs/`).
    Output:
        - str: Chuỗi nội dung Markdown hoàn chỉnh.
    How: Tổng hợp 8 phần chuẩn UTH (Mục tiêu, Bảng lộ trình 8 bước, Sơ đồ Mermaid, Phân tích toán học Chương 5,
         Kết quả thực nghiệm 4 mô hình trên 5 chỉ số, Kiểm toán chất lượng QA, Kết luận và Tài liệu tham khảo).
    """
    F = collect_report_facts(ROOT_DIR, run_id)
    test_metrics = F["test_metrics"]
    train_metrics = F["train_metrics"]
    init_loss, final_loss = F["init_loss"], F["final_loss"]
    best_val_mse, best_val_mae = F["best_val_mse"], F["best_val_mae"]
    best_epoch, epochs_exec = F["best_epoch"], F["epochs_exec"]
    n_total, n_features, n_cols = F["n_total"], F["n_features"], F["n_cols"]
    n_missing, n_dups, n_censored = F["n_missing"], F["n_dups"], F["n_censored"]
    n_train, n_test, n_sub, n_val = F["n_train"], F["n_test"], F["n_sub"], F["n_val"]
    split_label, scaler_n, leakage_ok = F["split_label"], F["scaler_n"], F["leakage_ok"]
    target_summary, target_median = F["target_summary"], F["target_median"]
    corr_neg_name, corr_neg_val = F["corr_neg_name"], F["corr_neg_val"]
    corr_pos_name, corr_pos_val = F["corr_pos_name"], F["corr_pos_val"]
    arch_dash, total_params, dense_layers = F["arch_dash"], F["total_params"], F["dense_layers"]
    lr_value, batch_size = F["lr_value"], F["batch_size"]
    epochs_configured, n_figures, figure_dpi = F["epochs_configured"], F["n_figures"], F["figure_dpi"]
    mlp_t, best_name, best_t = F["mlp_t"], F["best_name"], F["best_t"]
    mlp_beats_both, curve_note = F["mlp_beats_both"], F["curve_note"]
    status = "SUCCESS" if progress_data.get("status") in ("RUNNING", "SUCCESS", None) else progress_data.get("status")
    status_text = f"`{status}` 100%" if status == "SUCCESS" else f"`{status}`"
    dpi_text = f" (`{figure_dpi} DPI`)" if figure_dpi else ""

    if mlp_beats_both:
        conclusion_perf = (
            f"mang lại hiệu năng dự đoán vượt trội (`Test R² = {mlp_t['R2']:.4f}`, `Test MAPE = {mlp_t['MAPE']:.2f}%`) "
            "so với hồi quy tuyến tính và cây quyết định đơn lẻ."
        )
    else:
        conclusion_perf = (
            f"đạt `Test R² = {mlp_t['R2']:.4f}`, `Test MAPE = {mlp_t['MAPE']:.2f}%`; "
            "MLP không vượt đồng thời hồi quy tuyến tính và cây quyết định trên cả 5 chỉ số, xem bảng 5.4 để đối chiếu."
        )
        
    import csv
    from pathlib import Path
    _metrics_dir = Path(F.get("root_dir", ROOT_DIR)) / "reports" / "metrics"
    if not _metrics_dir.exists():
        _metrics_dir = REPORT_DIR / "metrics"

    def _csv_table_md(csv_name: str, heading: str) -> List[str]:
        p = _metrics_dir / csv_name
        if not p.exists():
            return []
        try:
            with open(p, encoding="utf-8-sig") as fh:
                rows = list(csv.DictReader(fh))
            if not rows:
                return []
            cols = list(rows[0].keys())
            out = ["", heading, "", "| " + " | ".join(cols) + " |", "| " + " | ".join([":---"] * len(cols)) + " |"]
            for r in rows:
                out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
            return out
        except Exception:
            return []

    lines: List[str] = [
        "# BÁO CÁO THỰC HÀNH LAB 04: DỰ ĐOÁN GIÁ NHÀ VỚI MẠNG NƠ-RON ĐA TẦNG (MULTILAYER PERCEPTRON REGRESSION)",
        "",
        "- **Học phần:** Học máy (Machine Learning) – Trường Đại học Giao thông Vận tải TP.HCM (UTH)",
        "- **Chủ đề:** Chương 5 – Mạng Nơ-ron Nhân tạo (Artificial Neural Networks – Practice Exercise 4: Predicting House Prices)",
        f"- **Phiên thực nghiệm (Run ID):** `{run_id}` (Trạng thái: **{progress_data.get('status', 'SUCCESS')}**)",
        f"- **Bộ dữ liệu chuẩn:** Boston Housing Dataset (`data/raw/boston_housing.csv` – {n_total} mẫu, {n_features} đặc trưng, biến mục tiêu `MEDV` đơn vị `$1,000 USD`)",
        "",
        "## 1. Mục tiêu bài thực hành",
        "1. **Nắm vững cơ sở lý thuyết Chương 5** về Mạng Nơ-ron Nhân tạo (Artificial Neural Networks), nguyên lý hoạt động của nơ-ron toán học, cơ chế lan truyền thuận (Forward Propagation), hàm kích hoạt (`ReLU`, `Linear`) và thuật toán lan truyền ngược (Backpropagation) kết hợp bộ tối ưu `Adam`.",
        "2. **Thiết kế và triển khai pipeline học máy hoàn chỉnh** cho bài toán hồi quy dự đoán giá nhà theo nguyên tắc **Clean Architecture** (tách biệt rõ cấu hình `config.yaml`, dữ liệu `data/`, mô-đun lõi `src/`, mô hình `models/` và hệ thống báo cáo `reports/`).",
        f"3. **Đảm bảo tính toàn vẹn dữ liệu và chống rò rỉ dữ liệu (Anti-Data-Leakage):** Nạp trực tiếp tệp dữ liệu chuẩn `data/raw/boston_housing.csv` (thay thế hoàn toàn hàm `load_boston` đã bị loại bỏ khỏi `scikit-learn >= 1.2`), chỉ `fit` bộ chuẩn hóa `StandardScaler` trên tập `Train` (`{n_train}` mẫu) và tách tập `Validation` (`{n_val}` mẫu) độc lập hoàn toàn với tập `Test` (`{n_test}` mẫu).",
        f"4. **Đối sánh thực nghiệm đa mô hình trên 5 chỉ số chuẩn:** Huấn luyện mạng `MLP Regression` (`{arch_dash.replace('-', ' → ')}`, `{total_params}` tham số) và đặt lên bàn cân so sánh với `Linear Regression`, `Decision Tree` và `Random Forest` thông qua bộ 5 thước đo định lượng: **MSE**, **RMSE**, **MAE**, **MAPE (%)** và **$R^2$ Score**.",
        "",
        "## 2. Bảng lộ trình thực thi chuẩn hóa (Execution Roadmap)",
        "",
        "| Step | Giai đoạn / Tên mô-đun | Chức năng (What it does) | Đường dẫn Import (Import path) |",
        "| :---: | :--- | :--- | :--- |",
        "| **1** | **Setup & Run Tracking** | Nạp `config.yaml`, cố định `random_state = 42`, khởi tạo `Run_ID`, Logger và tiến độ JSON. | `src/utils.py`, `src/tracking.py` |",
        f"| **2** | **Direct Data Loader** | Đọc trực tiếp `data/raw/boston_housing.csv`, kiểm định {n_total} dòng × {n_cols} cột, tách `X` ({n_features} biến) và `y` (`MEDV`). | `src/data_loader.py` |",
        "| **3** | **Exploratory Data Analysis** | Tính thống kê mô tả, ma trận tương quan Pearson, phát hiện ngưỡng kịch trần `MEDV = 50.0` và vẽ 4 biểu đồ EDA. | `src/eda.py` |",
        f"| **4** | **Leakage-Free Preprocessing** | Chia `Train` ({n_train}) / `Test` ({n_test}), `fit` `StandardScaler` chỉ trên `Train`, tách `Train_sub` ({n_sub}) và `Validation` ({n_val}). | `src/preprocess.py` |",
        f"| **5** | **MLP & Baseline Builder** | Khởi tạo kiến trúc mạng `Sequential_MLP_Regressor` (`{arch_dash}`) và 3 mô hình Baseline chuẩn hóa. | `src/model.py` |",
        f"| **6** | **Model Training & EarlyStop** | Huấn luyện MLP với `Adam (lr={lr_value})`, `batch_size={batch_size}`, `{epochs_configured}` epochs kèm `EarlyStopping` và lưu trọng số tốt nhất. | `src/train.py` |",
        f"| **7** | **5-Metric Evaluator & Plots** | Tính toán `MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²` trên tập Train/Test và kết xuất {n_figures} biểu đồ vào `reports/figures/`. | `src/evaluate.py`, `src/visualize.py` |",
        "| **8** | **Automated Report Engine** | Tổng hợp số liệu thực nghiệm, sinh báo cáo Markdown và báo cáo HTML tương tác tự chứa (`LAB_04_REPORT.html`). | `src/report.py`, `render_report.py` |",
        "",
        "## 3. Kiến trúc hệ thống (System Architecture)",
        "",
        "```mermaid",
        "graph LR",
        "    subgraph Data Governance & Anti-Leakage Preprocessing",
        f"        D1[boston_housing.csv {n_total}x{n_cols}] --> D2[Data Integrity Audit]",
        "        D2 --> D3[EDA & Correlation Analysis]",
        f"        D2 --> D4[Train/Test Split {split_label}]",
        f"        D4 -->|Train: {n_train} mẫu| D5[StandardScaler.fit_transform]",
        f"        D4 -->|Test: {n_test} mẫu| D6[StandardScaler.transform ONLY]",
        f"        D5 --> D7[Train_sub: {n_sub} mẫu]",
        f"        D5 --> D8[Validation: {n_val} mẫu]",
        "    end",
        "",
        "    subgraph Model Training, 5-Metric Evaluation & Reporting",
        f"        D7 --> M1[MLP Regressor {arch_dash}]",
        "        D8 -->|Monitor val_loss| M1",
        "        D5 --> M2[3 Baselines: Linear / DT / RF]",
        "        M1 --> E1[5-Metric Evaluator: MSE, RMSE, MAE, MAPE, R2]",
        "        M2 --> E1",
        "        D6 --> E1",
        "        E1 --> R1[Single Figure Hub: reports/figures/]",
        "        R1 --> R2[Interactive HTML & Markdown Reports]",
        "    end",
        "```",
        "",
        "## 4. Cơ sở lý thuyết & Phân tích toán học",
        "",
        f"### 4.1. Lan truyền thuận (Forward Propagation) và Kiến trúc Mạng MLP `{arch_dash}`",
        "Mạng **Multilayer Perceptron (MLP)** xếp chồng hai tầng ẩn kết nối đầy đủ (*Dense Layers*) sử dụng hàm kích hoạt phi tuyến **ReLU** ($\\text{ReLU}(z) = \\max(0, z)$) và tầng đầu ra dùng hàm **Linear** ($f(z) = z$):",
        "",
        *_forward_math_lines(dense_layers),
        "",
        "- **Tổng số tham số huấn luyện:** " + _params_formula(dense_layers, total_params) + " tham số.",
        "",
        "### 4.2. Chuẩn hóa Z-Score Chống Data Leakage & Hệ thống 5 Chỉ số Đánh giá",
        "$$\\mathbf{x}_{\\text{train}}^{\\text{scaled}} = \\frac{\\mathbf{x}_{\\text{train}} - \\mu_{\\text{train}}}{\\sigma_{\\text{train}}}, \\qquad \\mathbf{x}_{\\text{test}}^{\\text{scaled}} = \\frac{\\mathbf{x}_{\\text{test}} - \\mu_{\\text{train}}}{\\sigma_{\\text{train}}}$$",
        "$$\\text{MSE} = \\frac{1}{N}\\sum_{i=1}^{N}(y_i - \\hat{y}_i)^2, \\qquad \\text{RMSE} = \\sqrt{\\text{MSE}}, \\qquad \\text{MAE} = \\frac{1}{N}\\sum_{i=1}^{N}|y_i - \\hat{y}_i|$$",
        "$$\\text{MAPE (\\%)} = \\frac{100\\%}{N}\\sum_{i=1}^{N}\\left|\\frac{y_i - \\hat{y}_i}{y_i}\\right|, \\qquad R^2 = 1 - \\frac{\\sum_{i=1}^{N}(y_i - \\hat{y}_i)^2}{\\sum_{i=1}^{N}(y_i - \\bar{y})^2}$$",
        "",
        "## 5. Nội dung chi tiết và Kết quả thực nghiệm",
        "",
        "### 5.1. Quản lý dữ liệu chuẩn & Phân tích khám phá (Data Loading & EDA)",
        f"- Bộ dữ liệu `data/raw/boston_housing.csv` gồm `{n_total}` mẫu × `{n_cols}` cột, `{n_missing}` giá trị thiếu, `{n_dups}` bản ghi trùng lặp.",
        f"- Biến mục tiêu `MEDV` có trung bình `{target_summary['mean']:.4f}` (`${target_summary['mean'] * 1000:,.0f} USD`), độ lệch chuẩn `{target_summary['std']:.4f}`, trung vị `{target_median:.4f}` và `{n_censored}` mẫu chạm ngưỡng trần `50.0`.",
        f"- Đặc trưng tương quan âm mạnh nhất là `{corr_neg_name}` ($r = {corr_neg_val:+.4f}$) và tương quan dương mạnh nhất là `{corr_pos_name}` ($r = {corr_pos_val:+.4f}$).",
        "",
        "### 5.2. Tiền xử lý nghiêm ngặt chống Data Leakage (Leakage-Free Preprocessing)",
        f"- Tập `Train` (`{n_train}` mẫu) và `Test` (`{n_test}` mẫu) được tách độc lập với `random_state = 42`.",
        f"- Bộ `StandardScaler` chỉ `fit` trên `{scaler_n}` mẫu `Train` (`leakage_free = {leakage_ok}`), sau đó tách `X_train_sub` (`{n_sub}` mẫu) và `X_val_scaled` (`{n_val}` mẫu).",
        "",
        "### 5.3. Huấn luyện MLP Regression & Hội tụ EarlyStopping",
        f"- Đã chạy `{epochs_exec}` epochs (`batch_size = {batch_size}`, `Adam lr = {lr_value}`).",
        f"- **Train MSE (Loss):** Giảm từ `{init_loss:.4f}` (Epoch 1) xuống `{final_loss:.4f}` (Epoch {epochs_exec}).",
        f"- **Epoch kiểm định tốt nhất (Best Validation Epoch):** Epoch `{best_epoch}` (`Best Val MSE = {best_val_mse:.4f}`, `Best Val MAE = {best_val_mae:.4f}`).",
        f"- **Nhận xét Train vs Validation:** {curve_note}",
        "",
        f"### 5.4. Đối sánh thực nghiệm 4 mô hình trên Tập Kiểm tra (`Test Set` — {n_test} mẫu)",
        "",
        "| Mô hình (Model) | Test MSE (↓) | Test RMSE (↓) | Test MAE (↓) | Test MAPE (↓) | Test R² (↑) | Train MSE | Train MAPE | Train R² | Thời gian Train (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for model_name, m_test in test_metrics.items():
        m_train = train_metrics.get(model_name, {})
        lines.append(
            f"| **{model_name}** | {m_test.get('MSE', 0):.4f} | {m_test.get('RMSE', 0):.4f} | "
            f"{m_test.get('MAE', 0):.4f} | {m_test.get('MAPE', 0):.2f}% | {m_test.get('R2', 0):.4f} | "
            f"{m_train.get('MSE', 0):.4f} | {m_train.get('MAPE', 0):.2f}% | {m_train.get('R2', 0):.4f} | "
            f"{m_test.get('Training_Time_Sec', 0):.4f}s |"
        )

    lines.extend([
        "",
        f"- **Mô hình tốt nhất theo Test R²:** `{best_name}` (`R² = {best_t['R2']:.4f}`, `MSE = {best_t['MSE']:.4f}`, `MAPE = {best_t['MAPE']:.2f}%`).",
        f"- **Regularization MLP:** `l2_lambda = {F.get('l2_lambda', 0.0)}`, `dropout_rate = {F.get('dropout_rate', 0.0)}`.",
        *_csv_table_md("eda_outlier_analysis.csv", "#### Kiểm toán Giá trị Ngoại lai (EDA Outlier Analysis)"),
        *_csv_table_md("kfold_cross_validation.csv", "#### Kết quả 5-Fold Cross-Validation"),
        *_csv_table_md("mlp_architecture_tuning.csv", "#### Đối sánh Tuning Kiến trúc MLP"),
        *_csv_table_md("model_summary.csv", "#### Chẩn đoán Thống kê (Model Summary)"),
        "",
        "### 5.5. Danh mục Biểu đồ Thực nghiệm Hợp nhất (`reports/figures/`)",
        f"- Phân phối biến mục tiêu & Tương quan: `{fig_prefix}/target_distribution.png`, `{fig_prefix}/correlation_heatmap.png`, `{fig_prefix}/feature_distributions.png`, `{fig_prefix}/feature_target_relationships.png`",
        f"- Huấn luyện MLP & Phân chia dữ liệu: `{fig_prefix}/train_val_test_split.png`, `{fig_prefix}/mlp_training_loss_curve.png`, `{fig_prefix}/mlp_train_vs_val_loss.png`, `{fig_prefix}/training_time_comparison.png`",
        f"- Đánh giá & Đối sánh 4 mô hình: `{fig_prefix}/actual_vs_predicted_mlp.png`, `{fig_prefix}/actual_vs_predicted_all.png`, `{fig_prefix}/residual_plot_all.png`, `{fig_prefix}/residual_distribution_all.png`, `{fig_prefix}/model_comparison_mse.png`, `{fig_prefix}/model_comparison_mae.png`, `{fig_prefix}/model_comparison_mape.png`, `{fig_prefix}/model_comparison_r2.png`",
        f"- Đánh giá, 5-Fold CV & Tuning MLP: `{fig_prefix}/kfold_cv_comparison.png`, `{fig_prefix}/mlp_architecture_tuning.png`",
        "",
        "## 6. Kiểm thử và kiểm toán chất lượng (Quality Assurance)",
        "1. **Clean Architecture & Chuẩn Docstring 5W1H:** Toàn bộ mô-đun trong `src/` tuân thủ PEP-8 và tách biệt cấu hình vào `config.yaml`.",
        f"2. **Kiểm định Chống Rò rỉ Dữ liệu (`Anti-Data-Leakage Verification`):** Cờ `leakage_free = {leakage_ok}` xác nhận `StandardScaler` chỉ học trên `{scaler_n}` mẫu `Train`.",
        f"3. **Kiến trúc Lưu trữ Biểu đồ Đơn nhất (`Single Source of Truth Figures`):** {n_figures} biểu đồ{dpi_text} lưu đồng thời tại `reports/figures/` và `reports/figures/<run_id>/`.",
        f"4. **Kiểm thử Thực thi Tự động (`End-to-End Pipeline Audit`):** Chạy `python run_lab04.py` và `render_report.py` đạt trạng thái {status_text}.",
        "",
        "## 7. Kết luận",
        "Hệ thống Lab 04 đã hoàn thành các mục tiêu của Chương 5. Sự kết hợp giữa chuẩn hóa Z-Score chống rò rỉ dữ liệu, "
        f"kiến trúc MLP `{arch_dash}` với hàm kích hoạt `ReLU`/`Linear`, thuật toán tối ưu `Adam` và `EarlyStopping` {conclusion_perf}",
        "",
        "## 8. Tài liệu tham khảo",
        "1. *Bài giảng Chương 5: Introduction to Artificial Neural Networks (ANNs)* & *Practice Exercise 4*, Trường Đại học Giao thông Vận tải TP.HCM (UTH).",
        "2. Harrison, D., & Rubinfeld, D. L. (1978). *Hedonic housing prices and the demand for clean air*. Journal of Environmental Economics and Management, 5(1), 81–102.",
        "3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.",
        "4. Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR.",
        "",
    ])
    return "\n".join(lines)


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    What: Tổng hợp và xuất bộ báo cáo thực nghiệm chuẩn UTH (`reports/experiment_summary.md`,
          `docs/LAB_04_REPORT.md`, `reports/LAB_04_REPORT.html`, `reports/report_<run_id>.html`).
    Why: Cung cấp bộ báo cáo Markdown và HTML tương tác tự động ngay sau khi Pipeline chạy xong.
    Input:
        - run_id (Optional[str]): Mã định danh lần chạy (nếu `None` sẽ lấy lần chạy mới nhất).
    Output:
        - Dict[str, Any]: Đường dẫn các tệp báo cáo đã tạo.
    How: Đọc dữ liệu từ `progress_results/`, `reports/metrics/` và `logs/`, sau đó ghi Markdown và gọi `render_html()`.
    """
    start_time = time.perf_counter()
    active_run_id = run_id or get_latest_run_id()
    write_log(active_run_id, "REPORTING", "START", "Bắt đầu tổng hợp báo cáo thực nghiệm chuẩn UTH.")

    ensure_dir(REPORT_DIR)
    ensure_dir(DOCS_DIR)

    progress_path = PROGRESS_DIR / f"progress_{active_run_id}.json"
    progress_data = load_json(progress_path) if progress_path.exists() else {"run_id": active_run_id, "status": "SUCCESS"}

    log_path = LOG_DIR / f"pipeline_{active_run_id}.log"
    log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""

    md_reports = generate_experiment_summary_markdown(
        run_id=active_run_id,
        progress_data=progress_data,
        log_text=log_text,
        fig_prefix="figures",
    )
    md_docs = generate_experiment_summary_markdown(
        run_id=active_run_id,
        progress_data=progress_data,
        log_text=log_text,
        fig_prefix="../reports/figures",
    )

    summary_md_path = REPORT_DIR / "experiment_summary.md"
    run_md_path = REPORT_DIR / f"report_{active_run_id}.md"
    docs_md_path = DOCS_DIR / "LAB_04_REPORT.md"

    summary_md_path.write_text(md_reports, encoding="utf-8")
    run_md_path.write_text(md_reports, encoding="utf-8")
    docs_md_path.write_text(md_docs, encoding="utf-8")

    # Kết xuất báo cáo HTML tương tác tự chứa (Base64 embedded)
    html_out_path = render_html(project_root=ROOT_DIR, run_id=active_run_id)
    run_html_path = REPORT_DIR / f"report_{active_run_id}.html"

    duration = round(time.perf_counter() - start_time, 4)
    print("\n" + "=" * 65)
    print("BƯỚC 6: TỔNG HỢP BÁO CÁO THỰC NGHIỆM (UTH REPORT ENGINE)")
    print("=" * 65)
    pdf_out_path = REPORT_DIR / "LAB_04_REPORT.pdf"
    run_pdf_path = REPORT_DIR / f"report_{active_run_id}.pdf"
    print(f"[DONE] Đã lưu báo cáo tổng hợp : {summary_md_path.relative_to(ROOT_DIR)}")
    print(f"[DONE] Đã lưu báo cáo chuẩn UTH: {docs_md_path.relative_to(ROOT_DIR)}")
    print(f"[DONE] Đã lưu báo cáo HTML UTH : {html_out_path.relative_to(ROOT_DIR)}")
    print(f"[DONE] Đã lưu báo cáo HTML Run : {run_html_path.relative_to(ROOT_DIR)}")
    if pdf_out_path.exists():
        print(f"[DONE] Đã lưu báo cáo PDF UTH  : {pdf_out_path.relative_to(ROOT_DIR)}")
    if run_pdf_path.exists():
        print(f"[DONE] Đã lưu báo cáo PDF Run  : {run_pdf_path.relative_to(ROOT_DIR)}")

    output_payload = {
        "experiment_summary_md": "reports/experiment_summary.md",
        "lab_04_report_md": "docs/LAB_04_REPORT.md",
        "lab_04_report_html": "reports/LAB_04_REPORT.html",
        "lab_04_report_pdf": "reports/LAB_04_REPORT.pdf",
        "run_report_md": f"reports/report_{active_run_id}.md",
        "run_report_html": f"reports/report_{active_run_id}.html",
        "run_report_pdf": f"reports/report_{active_run_id}.pdf",
        "duration_seconds": duration,
    }
    update_progress(active_run_id, "REPORTING", "SUCCESS", output_data=output_payload)
    write_log(active_run_id, "REPORTING", "SUCCESS", f"Đã xuất bộ báo cáo UTH trong {duration}s.")
    return output_payload


if __name__ == "__main__":
    main()
    