"""
Module trực quan hóa dữ liệu và kết quả thực nghiệm (`src/visualize.py`) cho dự án LAB04.

Trách nhiệm của module:
1. Tạo đầy đủ 12+ biểu đồ phân tích bắt buộc theo yêu cầu Part 6.2:
   (1) Target Distribution (`target_distribution.png`)
   (2) Feature Correlation Heatmap (`correlation_heatmap.png`)
   (3) Feature Distributions (`feature_distributions.png`) & Feature-Target Relationships (`feature_target_relationships.png`)
   (4) MLP Training Loss Curve (`mlp_training_loss_curve.png`)
   (5) Training vs Validation Loss & MAE (`mlp_train_vs_val_loss.png`)
   (6) Actual vs Predicted Plot cho toàn bộ 4 mô hình (`actual_vs_predicted_all.png` & từng mô hình)
   (7) Residual Plot (`residual_plot_all.png`)
   (8) Residual Distribution (`residual_distribution_all.png`)
   (9) Model Performance Comparison — MSE (`model_comparison_mse.png`)
   (10) Model Performance Comparison — MAE (`model_comparison_mae.png`)
   (11) Model Performance Comparison — R² (`model_comparison_r2.png`)
   (12) Model Performance Comparison — MAPE (%) (`model_comparison_mape.png`)
   Cùng biểu đồ bổ trợ: Phân bổ kích thước Train/Validation/Test (`train_val_test_split.png`) và
   So sánh thời gian huấn luyện (`training_time_comparison.png`).
2. Mỗi biểu đồ đều có tiêu đề rõ ràng (Title), nhãn trục X/Y kèm đơn vị đo (`$1,000 USD` hoặc `%`),
   chú giải (Legend), lưới đọc giá trị và độ phân giải 300 DPI.
3. Lưu đồng thời vào 3 nơi để đảm bảo tính truy vết theo `Run_ID` và hiển thị trực tiếp trong tài liệu Markdown:
   - `reports/figures/<run_id>/<stage>/<filename>.png`
   - `reports/figures/<filename>.png`
"""

import shutil
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import ensure_dir, get_project_root

ROOT_DIR = get_project_root()
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
DOCS_IMAGES_DIR = ROOT_DIR / "docs" / "images"

sns.set_theme(style="whitegrid", palette="deep")
MODEL_PALETTE = {
    "MLP Regression": "#1d4ed8",
    "Linear Regression": "#dc2626",
    "Decision Tree": "#d97706",
    "Random Forest": "#059669",
}


def create_stage_figure_dir(run_id: str, stage: str) -> Path:
    """
    What: Tạo thư mục lưu biểu đồ riêng cho từng `run_id` và từng giai đoạn (`eda`, `preprocessing`, `training`, `evaluation`).
    Why: Đảm bảo các lần chạy khác nhau không ghi đè biểu đồ lịch sử của nhau (kế thừa kiến trúc LAB01).
    Input:
        - run_id (str): Mã định danh lần chạy.
        - stage (str): Tên giai đoạn.
    Output:
        - Path: Đường dẫn thư mục đã khởi tạo.
    How: Tạo đường dẫn `reports/figures/<run_id>/<stage>` và gọi `ensure_dir()`.
    """
    stage_dir = FIGURES_DIR / str(run_id) / stage.lower()
    return ensure_dir(stage_dir)


def _save_figure_multi_dest(fig: plt.Figure, filename: str, stage_dir: Optional[Path] = None, dpi: int = 300) -> List[str]:
    """
    What: Lưu đối tượng `matplotlib.figure.Figure` vào thư mục theo `run_id`, thư mục `reports/figures/`
          theo chuẩn Single Source of Truth (`reports/figures/`).
    Why: Giúp cả báo cáo theo `Run_ID`, báo cáo tổng hợp `reports/experiment_summary.md` và
         các tài liệu trong `docs/` đều hiển thị được hình ảnh mà không bị gãy đường dẫn.
    Input:
        - fig (plt.Figure): Hình vẽ Matplotlib cần lưu.
        - filename (str): Tên tệp ảnh (ví dụ: `'target_distribution.png'`).
        - stage_dir (Optional[Path]): Thư mục giai đoạn của lần chạy hiện tại.
        - dpi (int): Độ phân giải điểm ảnh (mặc định `300`).
    Output:
        - List[str]: Danh sách đường dẫn tương đối của các tệp ảnh đã lưu.
    How: Lưu ảnh vào `reports/figures/<filename>` và sao chép vào `stage_dir / filename` (`reports/figures/<run_id>/<stage>/`).
    """
    ensure_dir(FIGURES_DIR)

    primary_path = FIGURES_DIR / filename
    fig.savefig(primary_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    saved_paths = [str(primary_path.relative_to(ROOT_DIR)).replace("\\", "/")]

    if stage_dir is not None:
        ensure_dir(stage_dir)
        stage_path = stage_dir / filename
        shutil.copy2(primary_path, stage_path)
        saved_paths.append(str(stage_path.relative_to(ROOT_DIR)).replace("\\", "/"))

    return saved_paths


# ==============================================================================
# BIỂU ĐỒ GIAI ĐOẠN EDA (BIỂU ĐỒ 1, 2, 3)
# ==============================================================================

def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = "MEDV",
    stage_dir: Optional[Path] = None,
) -> List[str]:
    """
    What: Vẽ biểu đồ phân phối của biến mục tiêu `MEDV` (Histogram kết hợp KDE và Boxplot).
    Why: Biểu đồ bắt buộc số 1 (Target Distribution) — giúp nhận diện phân phối lệch phải,
         giá trị trung bình, trung vị và hiện tượng cắt ngưỡng trần (censoring) tại `MEDV = 50.0` ($50,000).
    Input:
        - df (pd.DataFrame): Bảng dữ liệu nhà ở.
        - target_col (str): Tên cột mục tiêu (`'MEDV'`).
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Danh sách đường dẫn ảnh đã lưu.
    How: Tạo 2 khung hình (trên: Histogram + KDE kèm đường Mean/Median; dưới: Boxplot nhận diện ngoại lai).
    """
    values = df[target_col].dropna()
    mean_val = float(values.mean())
    median_val = float(values.median())

    fig, (ax_hist, ax_box) = plt.subplots(
        2, 1, figsize=(10, 6.5), gridspec_kw={"height_ratios": [3.5, 1.2]}, sharex=True
    )

    sns.histplot(values, bins=30, kde=True, color="#1d4ed8", edgecolor="white", alpha=0.75, ax=ax_hist)
    ax_hist.axvline(mean_val, color="#dc2626", linestyle="--", linewidth=2, label=f"Mean = ${mean_val:.2f}k")
    ax_hist.axvline(median_val, color="#059669", linestyle="-.", linewidth=2, label=f"Median = ${median_val:.2f}k")
    ax_hist.set_title("Biểu đồ 1: Phân phối Giá Nhà Trung vị (Target Distribution — MEDV)", fontsize=13, fontweight="bold")
    ax_hist.set_ylabel("Số lượng khu vực (Frequency)", fontsize=11)
    ax_hist.legend(loc="upper right", frameon=True)

    sns.boxplot(x=values, color="#93c5fd", ax=ax_box, width=0.45)
    ax_box.set_xlabel("Giá trị nhà trung vị MEDV (Đơn vị: $1,000 USD)", fontsize=11)
    ax_box.set_ylabel("Boxplot", fontsize=10)

    fig.tight_layout()
    return _save_figure_multi_dest(fig, "target_distribution.png", stage_dir)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    stage_dir: Optional[Path] = None,
) -> List[str]:
    """
    What: Vẽ ma trận tương quan Pearson (Correlation Heatmap) giữa toàn bộ 13 đặc trưng và `MEDV`.
    Why: Biểu đồ bắt buộc số 2 (Feature Correlation Heatmap) — giúp xác định các biến tương quan
         thuận mạnh nhất (`RM` = +0.70), tương quan nghịch mạnh nhất (`LSTAT` = -0.74) và đa cộng tuyến (`RAD` - `TAX`).
    Input:
        - df (pd.DataFrame): Bảng dữ liệu nhà ở.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Danh sách đường dẫn ảnh đã lưu.
    How: Tính `df.corr(numeric_only=True)` và vẽ bằng `sns.heatmap(annot=True, fmt='.2f', cmap='coolwarm')`.
    """
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 9.5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        linewidths=0.5,
        cbar_kws={"label": "Hệ số tương quan Pearson (r)"},
        ax=ax,
    )
    ax.set_title(
        "Biểu đồ 2: Ma trận Tương quan Đặc trưng và Giá Nhà (Feature Correlation Heatmap)",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    fig.tight_layout()
    return _save_figure_multi_dest(fig, "correlation_heatmap.png", stage_dir)


def plot_feature_distributions(
    df: pd.DataFrame,
    target_col: str = "MEDV",
    stage_dir: Optional[Path] = None,
) -> List[str]:
    """
    What: Vẽ lưới phân phối (Histogram + KDE) của toàn bộ 13 biến đặc trưng đầu vào.
    Why: Biểu đồ bắt buộc số 3 (Feature Distribution) — giúp quan sát thang đo rất khác biệt
         giữa các biến (ví dụ `NOX` từ 0.38–0.87, trong khi `TAX` từ 187–711), giải thích vì sao
         bắt buộc phải áp dụng `StandardScaler` trước khi huấn luyện mạng nơ-ron MLP.
    Input:
        - df (pd.DataFrame): Bảng dữ liệu nhà ở.
        - target_col (str): Tên cột mục tiêu cần loại khỏi danh sách đặc trưng.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Danh sách đường dẫn ảnh đã lưu.
    How: Tạo lưới 4x4 subplots và vẽ phân phối cho từng cột đặc trưng.
    """
    feature_cols = [c for c in df.columns if c != target_col]
    n_cols = 4
    n_rows = int(np.ceil(len(feature_cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.3 * n_rows))
    axes_flat = np.asarray(axes).ravel()

    for idx, col in enumerate(feature_cols):
        ax = axes_flat[idx]
        sns.histplot(df[col], bins=25, kde=True, color="#2563eb", edgecolor="white", alpha=0.7, ax=ax)
        ax.set_title(f"{col} (mean={df[col].mean():.2f})", fontsize=10, fontweight="bold")
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)

    for idx in range(len(feature_cols), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(
        "Biểu đồ 3: Phân phối của 13 Đặc trưng Đầu vào (Feature Distributions — Raw Scale)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return _save_figure_multi_dest(fig, "feature_distributions.png", stage_dir)


def plot_feature_target_relationships(
    df: pd.DataFrame,
    target_col: str = "MEDV",
    stage_dir: Optional[Path] = None,
) -> List[str]:
    """
    What: Vẽ biểu đồ phân tán (Scatter Plot + Đường xu hướng) giữa 6 đặc trưng có tương quan
          lớn nhất (`LSTAT`, `RM`, `PTRATIO`, `INDUS`, `TAX`, `NOX`) với biến mục tiêu `MEDV`.
    Why: Minh họa trực quan quan hệ phi tuyến (đặc biệt ở biến `LSTAT` và `RM` với `MEDV`),
         chứng minh lý do mạng nơ-ron MLP với hàm kích hoạt phi tuyến `ReLU` vượt trội hơn hồi quy tuyến tính.
    Input:
        - df (pd.DataFrame): Bảng dữ liệu nhà ở.
        - target_col (str): Tên cột mục tiêu (`'MEDV'`).
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Danh sách đường dẫn ảnh đã lưu.
    How: Chọn 6 đặc trưng có $|r|$ cao nhất với `MEDV` và vẽ scatter kèm đường hồi quy bậc 2.
    """
    corrs = df.corr(numeric_only=True)[target_col].drop(labels=[target_col], errors="ignore")
    top_features = corrs.abs().sort_values(ascending=False).head(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.ravel()

    for idx, feat in enumerate(top_features):
        ax = axes_flat[idx]
        r_val = corrs[feat]
        ax.scatter(df[feat], df[target_col], alpha=0.55, color="#1d4ed8", edgecolors="none", s=28, label="Mẫu thực tế")
        x_vals = np.linspace(df[feat].min(), df[feat].max(), 100)
        poly_coef = np.polyfit(df[feat], df[target_col], deg=2)
        y_poly = np.polyval(poly_coef, x_vals)
        ax.plot(x_vals, y_poly, color="#dc2626", linewidth=2.2, label="Xu hướng phi tuyến")
        ax.set_title(f"{feat} vs {target_col} (r = {r_val:+.3f})", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Giá trị {feat}", fontsize=10)
        ax.set_ylabel(f"{target_col} ($1,000 USD)", fontsize=10)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        "Quan hệ giữa 6 Đặc trưng Quan trọng nhất và Giá Nhà MEDV (Feature-Target Relationships)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return _save_figure_multi_dest(fig, "feature_target_relationships.png", stage_dir)


def plot_train_val_test_split(
    y_train_sub: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    stage_dir: Optional[Path] = None,
) -> List[str]:
    """
    What: Vẽ biểu đồ so sánh số lượng mẫu và phân phối giá nhà `MEDV` trên 3 tập tách biệt:
          `Train_sub` (323 mẫu), `Validation` (81 mẫu) và `Test` (102 mẫu).
    Why: Minh chứng trực quan cho quy trình chống Data Leakage (Rule 7).
    Input:
        - y_train_sub, y_val, y_test (np.ndarray): Các mảng mục tiêu của 3 tập dữ liệu.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Danh sách đường dẫn ảnh đã lưu.
    How: Vẽ cột số lượng mẫu bên trái và KDE phân phối `MEDV` của 3 tập bên phải.
    """
    fig, (ax_bar, ax_kde) = plt.subplots(1, 2, figsize=(13, 5))

    labels = ["Train (MLP Fit)", "Validation (Monitor)", "Test (Hold-out)"]
    counts = [len(y_train_sub), len(y_val), len(y_test)]
    colors = ["#1d4ed8", "#d97706", "#059669"]

    bars = ax_bar.bar(labels, counts, color=colors, width=0.55)
    ax_bar.bar_label(bars, labels=[f"{c} mẫu\n({c/506*100:.1f}%)" for c in counts], padding=3, fontsize=9.5)
    ax_bar.set_ylim(0, max(counts) * 1.22)
    ax_bar.set_title("Số lượng mẫu theo Phân vùng Dữ liệu (Chống Data Leakage)", fontsize=11.5, fontweight="bold")
    ax_bar.set_ylabel("Số lượng mẫu (Samples)", fontsize=10.5)

    sns.kdeplot(y_train_sub, label=f"Train (mean=${np.mean(y_train_sub):.1f}k)", color="#1d4ed8", linewidth=2, ax=ax_kde)
    sns.kdeplot(y_val, label=f"Validation (mean=${np.mean(y_val):.1f}k)", color="#d97706", linewidth=2, ax=ax_kde)
    sns.kdeplot(y_test, label=f"Test (mean=${np.mean(y_test):.1f}k)", color="#059669", linewidth=2, ax=ax_kde)
    ax_kde.set_title("Đối chiếu Phân phối MEDV giữa Train, Validation và Test", fontsize=11.5, fontweight="bold")
    ax_kde.set_xlabel("Giá trị nhà MEDV ($1,000 USD)", fontsize=10.5)
    ax_kde.set_ylabel("Mật độ (Density)", fontsize=10.5)
    ax_kde.legend(loc="upper right")

    fig.tight_layout()
    return _save_figure_multi_dest(fig, "train_val_test_split.png", stage_dir)


# ==============================================================================
# BIỂU ĐỒ GIAI ĐOẠN TRAINING (BIỂU ĐỒ 4 & 5)
# ==============================================================================

def plot_mlp_training_loss_curve(
    history_dict: Dict[str, List[float]],
    stage_dir: Optional[Path] = None,
) -> List[str]:
    """
    What: Vẽ đường cong giảm hàm mất mát MSE trên tập Huấn luyện (MLP Training Loss Curve)
          qua các epoch.
    Why: Biểu đồ bắt buộc số 4 (MLP Training Loss Curve) — minh họa quá trình thuật toán
         Backpropagation và Adam Optimizer cập nhật trọng số giúp giảm hàm mục tiêu MSE.
    Input:
        - history_dict (Dict[str, List[float]]): Từ điển lịch sử huấn luyện chứa khóa `'loss'`.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Danh sách đường dẫn ảnh đã lưu.
    How: Vẽ đường `loss` theo số thứ tự epoch kèm điểm bắt đầu và điểm kết thúc.
    """
    train_loss = history_dict.get("loss", [])
    epochs = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(epochs, train_loss, color="#1d4ed8", linewidth=2.4, label="Training Loss (MSE)")
    if len(train_loss) > 0:
        ax.scatter([1, len(train_loss)], [train_loss[0], train_loss[-1]], color="#dc2626", zorder=5)
        ax.annotate(
            f"Epoch 1: {train_loss[0]:.2f}",
            xy=(1, train_loss[0]),
            xytext=(max(2, int(len(train_loss) * 0.08)), train_loss[0] * 0.92),
            fontsize=9.5,
            fontweight="bold",
        )
        ax.annotate(
            f"Epoch {len(train_loss)}: {train_loss[-1]:.2f}",
            xy=(len(train_loss), train_loss[-1]),
            xytext=(max(1, int(len(train_loss) * 0.72)), train_loss[-1] + max(train_loss) * 0.08),
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_title("Biểu đồ 4: Đường cong Hàm Mất mát Huấn luyện MLP (MLP Training Loss Curve)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch (Vòng lặp huấn luyện)", fontsize=11)
    ax.set_ylabel("Mean Squared Error — MSE ([$1,000 USD]²)", fontsize=11)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save_figure_multi_dest(fig, "mlp_training_loss_curve.png", stage_dir)


def plot_training_vs_validation_loss(
    history_dict: Dict[str, List[float]],
    best_epoch: Optional[int] = None,
    stage_dir: Optional[Path] = None,
) -> List[str]:
    """
    What: Vẽ biểu đồ đối chiếu song song Training vs Validation Loss (MSE) và Training vs Validation MAE
          qua các epoch.
    Why: Biểu đồ bắt buộc số 5 (Training vs Validation Loss) — giúp đánh giá hiện tượng
         Overfitting / Underfitting và điểm dừng tối ưu của `EarlyStopping`.
    Input:
        - history_dict (Dict[str, List[float]]): Từ điển chứa `'loss'`, `'val_loss'`, `'mae'`, `'val_mae'`.
        - best_epoch (Optional[int]): Epoch đạt `val_loss` thấp nhất.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Danh sách đường dẫn ảnh đã lưu.
    How: Tạo 2 khung hình cạnh nhau: bên trái vẽ MSE (`loss` vs `val_loss`), bên phải vẽ MAE (`mae` vs `val_mae`).
    """
    loss = history_dict.get("loss", [])
    val_loss = history_dict.get("val_loss", [])
    mae = history_dict.get("mae", [])
    val_mae = history_dict.get("val_mae", [])
    epochs = np.arange(1, len(loss) + 1)

    fig, (ax_mse, ax_mae) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax_mse.plot(epochs, loss, color="#1d4ed8", linewidth=2.2, label="Training MSE")
    if val_loss:
        ax_mse.plot(epochs, val_loss, color="#dc2626", linestyle="--", linewidth=2.2, label="Validation MSE")
        opt_ep = best_epoch or (int(np.argmin(val_loss)) + 1)
        ax_mse.axvline(opt_ep, color="#059669", linestyle=":", linewidth=1.8, label=f"Best Epoch ({opt_ep})")

    ax_mse.set_title("Training vs Validation Loss (MSE)", fontsize=12, fontweight="bold")
    ax_mse.set_xlabel("Epoch", fontsize=11)
    ax_mse.set_ylabel("MSE ([$1,000 USD]²)", fontsize=11)
    ax_mse.legend(loc="upper right")

    if mae:
        ax_mae.plot(epochs, mae, color="#1d4ed8", linewidth=2.2, label="Training MAE")
    if val_mae:
        ax_mae.plot(epochs, val_mae, color="#dc2626", linestyle="--", linewidth=2.2, label="Validation MAE")
        opt_ep = best_epoch or (int(np.argmin(val_loss)) + 1 if val_loss else len(mae))
        ax_mae.axvline(opt_ep, color="#059669", linestyle=":", linewidth=1.8, label=f"Best Epoch ({opt_ep})")

    ax_mae.set_title("Training vs Validation Mean Absolute Error (MAE)", fontsize=12, fontweight="bold")
    ax_mae.set_xlabel("Epoch", fontsize=11)
    ax_mae.set_ylabel("MAE ($1,000 USD)", fontsize=11)
    ax_mae.legend(loc="upper right")

    fig.suptitle(
        "Biểu đồ 5: So sánh Quá trình Huấn luyện và Kiểm định MLP (Training vs Validation Curves)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return _save_figure_multi_dest(fig, "mlp_train_vs_val_loss.png", stage_dir)


def plot_training_time_comparison(
    training_times: Dict[str, float],
    stage_dir: Optional[Path] = None,
) -> List[str]:
    """
    What: Vẽ biểu đồ cột so sánh thời gian huấn luyện (giây) của 4 mô hình.
    Why: Đánh giá sự đánh đổi giữa chi phí tính toán (Computational Cost) và độ chính xác dự đoán.
    Input:
        - training_times (Dict[str, float]): Thời gian huấn luyện của từng mô hình.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Danh sách đường dẫn ảnh đã lưu.
    How: Vẽ biểu đồ cột với nhãn thời gian chính xác tới 4 chữ số thập phân.
    """
    models = list(training_times.keys())
    times = [float(training_times[m]) for m in models]
    colors = [MODEL_PALETTE.get(m, "#3b82f6") for m in models]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(models, times, color=colors, width=0.52)
    ax.bar_label(bars, labels=[f"{t:.4f}s" for t in times], padding=3, fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(times) * 1.22 if max(times) > 0 else 1.0)
    ax.set_title("So sánh Thời gian Huấn luyện giữa các Mô hình (Training Time Comparison)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mô hình Hồi quy (Regression Models)", fontsize=11)
    ax.set_ylabel("Thời gian huấn luyện (Giây - Seconds)", fontsize=11)
    fig.tight_layout()
    return _save_figure_multi_dest(fig, "training_time_comparison.png", stage_dir)


# ==============================================================================
# BIỂU ĐỒ GIAI ĐOẠN EVALUATION (BIỂU ĐỒ 6, 7, 8, 9, 10, 11, 12)
# ==============================================================================

def plot_actual_vs_predicted(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    metrics_dict: Dict[str, Dict[str, float]],
    stage_dir: Optional[Path] = None,
) -> List[str]:
    """
    What: Vẽ biểu đồ phân tán Giá trị Thực tế so với Giá trị Dự đoán (Actual vs Predicted Prices)
          cho cả 4 mô hình trên tập Kiểm tra (`Test Set`), kèm biểu đồ riêng của MLP Regression.
    Why: Biểu đồ bắt buộc số 6 (Actual vs Predicted Plot) và yêu cầu trực tiếp trong Task 5
         của `Practice exercise 4.pdf` (`plt.scatter(y_test, y_pred)`).
    Input:
        - y_true (np.ndarray): Giá nhà thực tế trên tập Test (`y_test`).
        - predictions_dict (Dict[str, np.ndarray]): Dự đoán `y_pred` của 4 mô hình.
        - metrics_dict (Dict[str, Dict[str, float]]): Chỉ số `MSE`, `MAE`, `MAPE`, `R2` của từng mô hình.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Danh sách đường dẫn ảnh đã lưu.
    How: Tạo lưới 2x2 so sánh 4 mô hình với đường chéo lý tưởng $y = \\hat{y}$, đồng thời xuất
         riêng biểu đồ `actual_vs_predicted_mlp.png` đúng theo đoạn code mẫu của đề bài.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64).ravel()
    min_val = float(min(y_true_arr.min(), *(np.min(p) for p in predictions_dict.values()))) - 2.0
    max_val = float(max(y_true_arr.max(), *(np.max(p) for p in predictions_dict.values()))) + 2.0

    # 1. Biểu đồ tổng hợp 4 mô hình (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes_flat = axes.ravel()

    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        ax = axes_flat[idx]
        y_pred_arr = np.asarray(y_pred, dtype=np.float64).ravel()
        m_info = metrics_dict.get(model_name, {})
        color = MODEL_PALETTE.get(model_name, "#1d4ed8")

        ax.scatter(
            y_true_arr,
            y_pred_arr,
            alpha=0.72,
            color=color,
            edgecolors="white",
            linewidth=0.5,
            s=48,
            label="Mẫu tập Test (n=102)",
        )
        ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.8, label="Dự đoán hoàn hảo (y = ŷ)")
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_title(
            f"{model_name}\nMSE={m_info.get('MSE', 0):.2f} | MAE={m_info.get('MAE', 0):.2f} | "
            f"MAPE={m_info.get('MAPE', 0):.2f}% | R²={m_info.get('R2', 0):.4f}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Giá thực tế — Actual Prices ($1,000 USD)", fontsize=10.5)
        ax.set_ylabel("Giá dự đoán — Predicted Prices ($1,000 USD)", fontsize=10.5)
        ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        "Biểu đồ 6: So sánh Giá Nhà Thực tế và Dự đoán trên Tập Test (Actual vs. Predicted House Prices)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    saved = _save_figure_multi_dest(fig, "actual_vs_predicted_all.png", stage_dir)

    # 2. Biểu đồ riêng cho MLP Regression (theo đúng Task 5 của Practice Exercise 4)
    if "MLP Regression" in predictions_dict:
        y_mlp = np.asarray(predictions_dict["MLP Regression"], dtype=np.float64).ravel()
        m_mlp = metrics_dict.get("MLP Regression", {})
        fig_mlp, ax_mlp = plt.subplots(figsize=(8.5, 6.5))
        ax_mlp.scatter(y_true_arr, y_mlp, alpha=0.75, color="#1d4ed8", edgecolors="white", linewidth=0.6, s=55, label="Dự đoán MLP")
        ax_mlp.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Đường chuẩn y = ŷ")
        ax_mlp.set_xlim(min_val, max_val)
        ax_mlp.set_ylim(min_val, max_val)
        ax_mlp.set_title(
            f"Actual vs. Predicted House Prices — MLP Regression\n"
            f"(MSE = {m_mlp.get('MSE', 0):.4f}, MAE = {m_mlp.get('MAE', 0):.4f}, "
            f"MAPE = {m_mlp.get('MAPE', 0):.2f}%, R² = {m_mlp.get('R2', 0):.4f})",
            fontsize=12,
            fontweight="bold",
        )
        ax_mlp.set_xlabel("Actual Prices ($1,000 USD)", fontsize=11)
        ax_mlp.set_ylabel("Predicted Prices ($1,000 USD)", fontsize=11)
        ax_mlp.legend(loc="upper left")
        fig_mlp.tight_layout()
        saved.extend(_save_figure_multi_dest(fig_mlp, "actual_vs_predicted_mlp.png", stage_dir))

    return saved


def plot_residuals(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    stage_dir: Optional[Path] = None,
) -> List[str]:
    """
    What: Vẽ biểu đồ phần dư (Residual Plot: $e_i = y_i - \\hat{y}_i$ theo giá trị dự đoán $\\hat{y}_i$)
          cho cả 4 mô hình trên tập Test.
    Why: Biểu đồ bắt buộc số 7 (Residual Plot) — kiểm tra tính ngẫu nhiên của sai số xung quanh
         đường $e = 0$ và phát hiện phương sai sai số thay đổi (heteroscedasticity).
    Input:
        - y_true (np.ndarray): Giá nhà thực tế trên tập Test.
        - predictions_dict (Dict[str, np.ndarray]): Dự đoán của 4 mô hình.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Danh sách đường dẫn ảnh đã lưu.
    How: Tính `residuals = y_true - y_pred` và vẽ scatter theo `y_pred` với đường ngang $e = 0$.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64).ravel()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes_flat = axes.ravel()

    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        ax = axes_flat[idx]
        y_pred_arr = np.asarray(y_pred, dtype=np.float64).ravel()
        residuals = y_true_arr - y_pred_arr
        color = MODEL_PALETTE.get(model_name, "#1d4ed8")

        ax.scatter(y_pred_arr, residuals, alpha=0.72, color=color, edgecolors="white", linewidth=0.5, s=45)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.8, label="Sai số bằng 0 (e = 0)")
        ax.set_title(
            f"{model_name} (Mean Residual = {np.mean(residuals):+.3f}, Std = {np.std(residuals):.3f})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Giá dự đoán — Predicted Price ŷ ($1,000 USD)", fontsize=10)
        ax.set_ylabel("Phần dư — Residual e = y - ŷ ($1,000 USD)", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "Biểu đồ 7: Phân tích Phần dư theo Giá trị Dự đoán trên Tập Test (Residual Plots)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return _save_figure_multi_dest(fig, "residual_plot_all.png", stage_dir)


def plot_residual_distributions(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    stage_dir: Optional[Path] = None,
) -> List[str]:
    """
    What: Vẽ biểu đồ phân phối xác suất của phần dư (Residual Distribution: Histogram + KDE)
          cho cả 4 mô hình trên tập Test.
    Why: Biểu đồ bắt buộc số 8 (Residual Distribution) — đánh giá xem sai số dự đoán có
         tập trung đối xứng quanh 0 theo phân phối chuẩn hay bị lệch (skewed).
    Input:
        - y_true (np.ndarray): Giá nhà thực tế trên tập Test.
        - predictions_dict (Dict[str, np.ndarray]): Dự đoán của 4 mô hình.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Danh sách đường dẫn ảnh đã lưu.
    How: Vẽ Histogram + KDE của `residuals = y_true - y_pred` cho từng mô hình trên lưới 2x2.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64).ravel()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes_flat = axes.ravel()

    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        ax = axes_flat[idx]
        y_pred_arr = np.asarray(y_pred, dtype=np.float64).ravel()
        residuals = y_true_arr - y_pred_arr
        color = MODEL_PALETTE.get(model_name, "#1d4ed8")

        sns.histplot(residuals, bins=22, kde=True, color=color, edgecolor="white", alpha=0.7, ax=ax)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.8, label="e = 0")
        ax.axvline(float(np.mean(residuals)), color="#dc2626", linestyle=":", linewidth=1.8, label=f"Mean = {np.mean(residuals):+.2f}")
        ax.set_title(f"Phân phối Phần dư — {model_name}", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Phần dư e = y - ŷ ($1,000 USD)", fontsize=10)
        ax.set_ylabel("Số lượng mẫu (Frequency)", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "Biểu đồ 8: Phân phối Phần dư của 4 Mô hình trên Tập Test (Residual Distributions)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return _save_figure_multi_dest(fig, "residual_distribution_all.png", stage_dir)


def plot_metric_comparison_bars(
    metrics_dict: Dict[str, Dict[str, float]],
    stage_dir: Optional[Path] = None,
) -> List[str]:
    """
    What: Vẽ 4 biểu đồ cột độc lập so sánh hiệu năng của 4 mô hình trên tập Test theo từng chỉ số:
          - Biểu đồ 9: Model Performance Comparison (MSE) (`model_comparison_mse.png`)
          - Biểu đồ 10: Model Performance Comparison (MAE) (`model_comparison_mae.png`)
          - Biểu đồ 11: Model Performance Comparison (R²) (`model_comparison_r2.png`)
          - Biểu đồ 12: Model Performance Comparison (MAPE %) (`model_comparison_mape.png`)
    Why: Đáp ứng chính xác các biểu đồ bắt buộc số 9, 10, 11 trong Part 6.2, Yêu cầu 6 của
         `Practice exercise 4.pdf` VÀ bổ sung biểu đồ so sánh `MAPE (%)` theo Slide 10 của `Chapter 5`.
    Input:
        - metrics_dict (Dict[str, Dict[str, float]]): Từ điển chứa kết quả Test của 4 mô hình.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Danh sách đường dẫn của cả 4 biểu đồ đã lưu.
    How: Duyệt qua cấu hình 4 chỉ số (`MSE`, `MAE`, `R2`, `MAPE`), vẽ biểu đồ cột có gắn nhãn giá trị
         và đánh dấu mô hình tốt nhất.
    """
    models = list(metrics_dict.keys())
    colors = [MODEL_PALETTE.get(m, "#1d4ed8") for m in models]
    saved_paths: List[str] = []

    chart_specs = [
        (
            "MSE",
            "Biểu đồ 9: So sánh Sai số Bình phương Trung bình trên Tập Test (MSE — Càng thấp càng tốt)",
            "Mean Squared Error — MSE ([$1,000 USD]²)",
            "model_comparison_mse.png",
            "min",
            "",
        ),
        (
            "MAE",
            "Biểu đồ 10: So sánh Sai số Tuyệt đối Trung bình trên Tập Test (MAE — Càng thấp càng tốt)",
            "Mean Absolute Error — MAE ($1,000 USD)",
            "model_comparison_mae.png",
            "min",
            "",
        ),
        (
            "R2",
            "Biểu đồ 11: So sánh Hệ số Xác định R² trên Tập Test (R² Score — Càng cao càng tốt)",
            "R-squared (R² Score)",
            "model_comparison_r2.png",
            "max",
            "",
        ),
        (
            "MAPE",
            "Biểu đồ 12: So sánh Sai số Phần trăm Tuyệt đối Trung bình trên Tập Test (MAPE % — Càng thấp càng tốt)",
            "Mean Absolute Percentage Error — MAPE (%)",
            "model_comparison_mape.png",
            "min",
            "%",
        ),
    ]

    for metric_key, title, ylabel, filename, mode, suffix in chart_specs:
        values = [float(metrics_dict[m][metric_key]) for m in models]
        best_idx = int(np.argmin(values)) if mode == "min" else int(np.argmax(values))

        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        bars = ax.bar(models, values, color=colors, width=0.52, edgecolor="black", linewidth=0.7)

        labels = []
        for idx, val in enumerate(values):
            fmt_str = f"{val:.2f}{suffix}" if suffix == "%" else f"{val:.4f}{suffix}"
            if idx == best_idx:
                fmt_str += " [Best]"
            labels.append(fmt_str)

        ax.bar_label(bars, labels=labels, padding=4, fontsize=10, fontweight="bold")
        upper_lim = max(values) * 1.20 if max(values) > 0 else 1.0
        if metric_key == "R2":
            ax.set_ylim(0.0, min(1.12, max(1.02, upper_lim)))
        else:
            ax.set_ylim(0.0, upper_lim)

        ax.set_title(title, fontsize=12.5, fontweight="bold")
        ax.set_xlabel("Mô hình Hồi quy (Regression Models)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        fig.tight_layout()
        saved_paths.extend(_save_figure_multi_dest(fig, filename, stage_dir))

    return saved_paths

# ==============================================================================
# BIỂU ĐỒ BỔ SUNG: 5-FOLD CROSS-VALIDATION & MLP ARCHITECTURE TUNING
# ==============================================================================

def plot_kfold_cv_comparison(
    cv_summary_df: pd.DataFrame,
    stage_dir: Optional[Path] = None,
    filename: str = "kfold_cv_comparison.png",
) -> List[str]:
    """Vẽ biểu đồ 5-Fold Cross-Validation (CV MSE Mean ± Std và CV R² Mean ± Std)."""
    if cv_summary_df is None or cv_summary_df.empty:
        return []

    df = cv_summary_df.copy()
    models = df["Model"].tolist()
    colors = [MODEL_PALETTE.get(m, "#475569") for m in models]
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

    # Subplot 1: CV MSE (Mean ± Std — Càng thấp càng tốt)
    mse_mean = df["CV_MSE_Mean"].values.astype(float)
    mse_std = df["CV_MSE_Std"].fillna(0.0).values.astype(float)
    best_mse_idx = int(np.argmin(mse_mean))
    bars1 = axes[0].bar(x, mse_mean, yerr=mse_std, capsize=6, color=colors, edgecolor="black", alpha=0.88, error_kw={"ecolor": "#1e293b", "lw": 1.4})
    for idx, (bar, m_val, s_val) in enumerate(zip(bars1, mse_mean, mse_std)):
        lbl = f"{m_val:.2f}±{s_val:.2f}" + (" [Best]" if idx == best_mse_idx else "")
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, m_val + s_val + max(mse_mean) * 0.03, lbl, ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=12, fontsize=9.5)
    axes[0].set_ylim(0, max(mse_mean + mse_std) * 1.24)
    axes[0].set_title("5-Fold CV: Sai số MSE (Mean ± Std — Thấp hơn là tốt hơn)", fontsize=11.5, fontweight="bold")
    axes[0].set_ylabel("CV MSE ($1,000 USD)²", fontsize=10.5)

    # Subplot 2: CV R² (Mean ± Std — Càng cao càng tốt)
    r2_mean = df["CV_R2_Mean"].values.astype(float)
    r2_std = df["CV_R2_Std"].fillna(0.0).values.astype(float)
    best_r2_idx = int(np.argmax(r2_mean))
    bars2 = axes[1].bar(x, r2_mean, yerr=r2_std, capsize=6, color=colors, edgecolor="black", alpha=0.88, error_kw={"ecolor": "#1e293b", "lw": 1.4})
    for idx, (bar, m_val, s_val) in enumerate(zip(bars2, r2_mean, r2_std)):
        lbl = f"{m_val:.4f}±{s_val:.3f}" + (" [Best]" if idx == best_r2_idx else "")
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, m_val + s_val + 0.018, lbl, ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=12, fontsize=9.5)
    axes[1].set_ylim(0, min(1.15, max(r2_mean + r2_std) * 1.22))
    axes[1].set_title("5-Fold CV: Hệ số R² (Mean ± Std — Cao hơn là tốt hơn)", fontsize=11.5, fontweight="bold")
    axes[1].set_ylabel("CV R² Score", fontsize=10.5)

    fig.suptitle("Đánh giá 5-Fold Cross-Validation trên 506 mẫu (Chống rò rỉ dữ liệu từng Fold)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save_figure_multi_dest(fig, filename, stage_dir)


def plot_mlp_architecture_tuning(
    arch_tuning_df: pd.DataFrame,
    stage_dir: Optional[Path] = None,
    filename: str = "mlp_architecture_tuning.png",
) -> List[str]:
    """Vẽ biểu đồ đối sánh 4 kiến trúc MLP (Train/Val/Test MSE và Test R² vs Total Params)."""
    if arch_tuning_df is None or arch_tuning_df.empty:
        return []

    df = arch_tuning_df.copy()
    short_names = [str(a).split(" ")[0] for a in df["Architecture"]]
    params = df["Total_Params"].values.astype(int)
    x_labels = [f"{n}\n({p:,} params)" for n, p in zip(short_names, params)]
    x = np.arange(len(df))
    w = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))

    # Subplot 1: Grouped Bar Chart (Train MSE vs Val MSE vs Test MSE)
    b_tr = axes[0].bar(x - w, df["Train_MSE"].values, width=w, label="Train MSE", color="#64748b", edgecolor="black", alpha=0.88)
    b_va = axes[0].bar(x, df["Val_MSE"].values, width=w, label="Validation MSE", color="#f59e0b", edgecolor="black", alpha=0.88)
    b_te = axes[0].bar(x + w, df["Test_MSE"].values, width=w, label="Test MSE", color="#1d4ed8", edgecolor="black", alpha=0.92)
    for bars in (b_tr, b_va, b_te):
        axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=8.5, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_labels, fontsize=9)
    axes[0].set_ylim(0, float(df[["Train_MSE", "Val_MSE", "Test_MSE"]].values.max()) * 1.22)
    axes[0].set_title("Bias–Variance Trade-off: Train vs Val vs Test MSE (↓)", fontsize=11.5, fontweight="bold")
    axes[0].set_ylabel("Mean Squared Error (MSE)", fontsize=10.5)
    axes[0].legend(frameon=True, fontsize=9.5)

    # Subplot 2: Test R² theo từng kiến trúc MLP
    r2_vals = df["Test_R2"].values.astype(float)
    best_idx = int(np.argmax(r2_vals))
    bar_colors = ["#059669" if i == best_idx else "#2563eb" for i in range(len(df))]
    bars_r2 = axes[1].bar(x, r2_vals, width=0.52, color=bar_colors, edgecolor="black", alpha=0.90)
    for idx, (bar, val) in enumerate(zip(bars_r2, r2_vals)):
        lbl = f"{val:.4f}" + (" [Best]" if idx == best_idx else "")
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, val + 0.012, lbl, ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels, fontsize=9)
    axes[1].set_ylim(0, min(1.05, max(r2_vals) * 1.18))
    axes[1].set_title("Hiệu năng Tổng quát hóa trên Tập Test theo Kiến trúc MLP (R² ↑)", fontsize=11.5, fontweight="bold")
    axes[1].set_ylabel("Test R² Score", fontsize=10.5)

    fig.suptitle("Tuning Kiến trúc Mạng PyTorch MLP (So sánh Độ sâu, Độ rộng & Số tham số)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save_figure_multi_dest(fig, filename, stage_dir)

