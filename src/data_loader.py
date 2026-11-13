"""
Module quản lý việc tải, xác thực và đọc dữ liệu (`src/data_loader.py`).

Trách nhiệm của module:
1. Đọc trực tiếp bộ dữ liệu dự đoán giá nhà **Boston Housing Dataset** (506 mẫu, 13 đặc trưng,
   biến mục tiêu `MEDV`) từ tệp dữ liệu chuẩn có sẵn `data/raw/boston_housing.csv` mà KHÔNG
   sử dụng hàm `sklearn.datasets.load_boston()` đã bị loại bỏ từ phiên bản scikit-learn 1.2.
2. Kiểm tra tính toàn vẹn của dữ liệu (số dòng, số cột, tên cột, kiểu dữ liệu, giá trị thiếu,
   bản ghi trùng lặp) và tách ma trận đặc trưng `X` cùng vector mục tiêu `y`.
"""

import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.tracking import create_run, finish_run, update_progress, write_log
from src.utils import get_project_root, load_config

ROOT_DIR = get_project_root()

EXPECTED_BOSTON_FEATURES = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]
EXPECTED_BOSTON_TARGET = "MEDV"

FEATURE_DESCRIPTIONS = {
    "CRIM": "Tỷ lệ tội phạm bình quân đầu người theo thị trấn (Per capita crime rate by town)",
    "ZN": "Tỷ lệ đất ở được quy hoạch cho các lô trên 25.000 sq.ft (%)",
    "INDUS": "Tỷ lệ diện tích kinh doanh phi bán lẻ (công nghiệp) trên mỗi thị trấn (%)",
    "CHAS": "Biến giả sông Charles (1 nếu khu vực giáp sông Charles, 0 nếu ngược lại)",
    "NOX": "Nồng độ khí Nitơ Oxit (phần chục triệu - parts per 10 million)",
    "RM": "Số lượng phòng trung bình trên mỗi căn hộ (Average number of rooms per dwelling)",
    "AGE": "Tỷ lệ căn hộ có chủ sở hữu được xây dựng trước năm 1940 (%)",
    "DIS": "Khoảng cách có trọng số tới 5 trung tâm việc làm lớn tại Boston",
    "RAD": "Chỉ số khả năng tiếp cận các trục đường cao tốc hướng tâm",
    "TAX": "Thuế suất bất động sản toàn phần trên mỗi $10,000",
    "PTRATIO": "Tỷ lệ học sinh / giáo viên theo từng thị trấn",
    "B": "Chỉ số 1000(Bk - 0.63)^2 trong thống kê lịch sử năm 1978 (cần lưu ý khía cạnh đạo đức dữ liệu)",
    "LSTAT": "Tỷ lệ dân số có thu nhập/địa vị kinh tế xã hội thấp (%)",
    "MEDV": "Giá trị trung vị của nhà có chủ sở hữu (Đơn vị: $1,000 USD - Biến mục tiêu)",
}


# Tên khác (alias) thường gặp của các cột trong các phiên bản Boston Housing trên Kaggle/GitHub.
COLUMN_ALIASES = {
    "BLACK": "B",
    "PRICE": "MEDV",
    "TARGET": "MEDV",
}

# Tên cột chỉ mục (index) tự sinh mà một số tệp Kaggle chèn thêm ở đầu bảng.
_INDEX_LIKE_NAMES = {"", "ID", "INDEX", "ROWNAMES", "ROW", "NO", "NO.", "STT"}


def _is_number(token: str) -> bool:
    """Trả về True nếu chuỗi `token` có thể chuyển thành số thực (coi `NA`/`NaN` cũng là số khuyết)."""
    token = token.strip().strip('"').strip("'")
    if token.upper() in {"NA", "NAN", "NULL", "N/A", ""}:
        return True
    try:
        float(token)
        return True
    except ValueError:
        return False


def _read_raw_boston_csv(raw_path: Path) -> pd.DataFrame:
    """
    What: Đọc tệp Boston Housing gốc (giữ nguyên 100% tệp trên đĩa) và trả về DataFrame đúng 14 cột
          `CRIM..LSTAT, MEDV` theo đúng thứ tự chuẩn.
    Why: Các phiên bản Boston Housing trên Kaggle không thống nhất định dạng: có tệp có header chữ
         thường (`crim`, `medv`), có tệp dùng `black` thay cho `B`, có tệp thừa cột chỉ mục
         (`Unnamed: 0`), có tệp phân tách bằng khoảng trắng và KHÔNG có dòng header (`housing.csv`).
         Thay vì sửa tay tệp dữ liệu, loader tự nhận diện và chuẩn hóa ngay trong bộ nhớ.
    How:
        1. Đọc dòng đầu tiên để phát hiện dấu phân tách (`,` `;` tab hoặc khoảng trắng).
        2. Nếu mọi ô ở dòng đầu đều là số -> tệp không có header -> gán tên cột theo vị trí chuẩn.
        3. Nếu có header -> chuẩn hóa tên (viết hoa, alias `BLACK->B`, `PRICE->MEDV`) và bỏ cột chỉ mục.
        4. Chỉ giữ đúng 14 cột chuẩn (loại cột thừa như `CAT. MEDV` để tránh rò rỉ nhãn), ép kiểu số.
    """
    expected_cols = EXPECTED_BOSTON_FEATURES + [EXPECTED_BOSTON_TARGET]

    with open(raw_path, "r", encoding="utf-8-sig", errors="replace") as fh:
        first_line = ""
        for line in fh:
            if line.strip():
                first_line = line.strip()
                break
    if not first_line:
        raise ValueError(f"Tệp {raw_path} rỗng.")

    # 1. Phát hiện dấu phân tách
    if "," in first_line:
        sep = ","
    elif ";" in first_line:
        sep = ";"
    elif "\t" in first_line:
        sep = "\t"
    else:
        sep = r"\s+"

    # 2. Phát hiện có header hay không
    first_tokens = re.split(sep, first_line)
    has_header = not all(_is_number(tok) for tok in first_tokens)

    if has_header:
        df = pd.read_csv(raw_path, sep=sep, engine="python", encoding="utf-8-sig",
                         na_values=["NA", "N/A", "NaN", "null", ""])
        df.columns = [str(c).strip().strip('"').strip("'").upper() for c in df.columns]
        # Bỏ cột chỉ mục như "Unnamed: 0", "ID", "rownames"
        drop_idx = [c for c in df.columns if c in _INDEX_LIKE_NAMES or c.startswith("UNNAMED")]
        df = df.drop(columns=drop_idx)
        df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns and v not in df.columns})
    else:
        df = pd.read_csv(raw_path, sep=sep, engine="python", encoding="utf-8-sig", header=None,
                         na_values=["NA", "N/A", "NaN", "null", ""])
        if df.shape[1] == len(expected_cols) + 1:      # có thêm cột số thứ tự ở đầu
            df = df.iloc[:, 1:]
        if df.shape[1] != len(expected_cols):
            raise ValueError(
                f"Tệp {raw_path} không có header và có {df.shape[1]} cột, "
                f"không khớp {len(expected_cols)} cột chuẩn của Boston Housing."
            )
        df.columns = expected_cols

    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Tệp {raw_path} thiếu các cột bắt buộc: {missing_cols}. "
            f"Các cột hiện có trong tệp: {df.columns.tolist()}"
        )

    df = df[expected_cols].apply(pd.to_numeric, errors="coerce")
    return df.reset_index(drop=True)


def load_housing_dataframe(config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    What: Đọc trực tiếp bộ dữ liệu Boston Housing từ `data/raw/boston_housing.csv` và trả về
          DataFrame cùng thông tin nguồn gốc dữ liệu.
    Why: Thay thế hoàn toàn API `load_boston()` đã bị gỡ bỏ khỏi scikit-learn >= 1.2 bằng tệp
         dữ liệu chuẩn có sẵn trong dự án (`data/raw/boston_housing.csv`), giúp code gọn nhẹ,
         chạy offline 100% và không phụ thuộc vào kết nối mạng bên ngoài.
    Input:
        - config (Optional[Dict[str, Any]]): Từ điển cấu hình từ `config.yaml` (nếu `None` sẽ tự nạp).
    Output:
        - Tuple[pd.DataFrame, Dict[str, Any]]:
            + `df`: DataFrame chứa toàn bộ 13 cột đặc trưng và cột mục tiêu `MEDV` (506 × 14).
            + `source_info`: Từ điển mô tả tên dataset, nguồn tệp, đường dẫn và ghi chú đạo đức.
    How:
        1. Lấy đường dẫn `data/raw/boston_housing.csv` từ `config.yaml`.
        2. Kiểm tra sự tồn tại của tệp và nạp trực tiếp bằng `pd.read_csv()`.
        3. Tự nhận diện định dạng tệp (có/không header, dấu phân tách, cột chỉ mục, alias tên cột) qua
           `_read_raw_boston_csv()` rồi xác thực đầy đủ 14 cột bắt buộc `CRIM..LSTAT, MEDV`.
    """
    if config is None:
        config = load_config()

    data_cfg = config.get("data", {})
    raw_rel_path = data_cfg.get("raw_path", "data/raw/boston_housing.csv")
    raw_path = ROOT_DIR / raw_rel_path

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy tệp dữ liệu gốc tại '{raw_path}'. "
            "Vui lòng đảm bảo tệp data/raw/boston_housing.csv tồn tại trong thư mục dự án."
        )

    df = _read_raw_boston_csv(raw_path)

    source_info = {
        "dataset_name": "Boston Housing Dataset (Harrison & Rubinfeld, 1978)",
        "source_type": "Local Raw CSV (data/raw/boston_housing.csv — tải từ Kaggle, giữ nguyên tệp gốc)",
        "file_path": str(raw_path.relative_to(ROOT_DIR)).replace("\\", "/"),
        "target_column": EXPECTED_BOSTON_TARGET,
        "feature_columns": [c for c in df.columns if c != EXPECTED_BOSTON_TARGET],
        "ethical_note": (
            "API sklearn.datasets.load_boston() đã bị loại bỏ từ scikit-learn 1.2 do biến B "
            "được thiết kế vào năm 1978 có hàm ý phân biệt chủng tộc. Dự án này nạp trực tiếp "
            "dữ liệu gốc từ tệp CSV lưu trữ cục bộ (data/raw/boston_housing.csv) để phục vụ mục đích "
            "học thuật và đối chiếu đúng yêu cầu của đề bài Practice Exercise 4."
        ),
    }
    return df, source_info


def inspect_dataset(df: pd.DataFrame, target_col: str = EXPECTED_BOSTON_TARGET) -> Dict[str, Any]:
    """
    What: Kiểm tra toàn diện cấu trúc, kiểu dữ liệu, giá trị khuyết thiếu và thống kê của DataFrame.
    Why: Đáp ứng yêu cầu kiểm toán dữ liệu (Dataset Audit - Part 4.2 & Audit 2) trước khi đưa vào
         bước tiền xử lý và huấn luyện mô hình.
    Input:
        - df (pd.DataFrame): Bảng dữ liệu cần kiểm tra.
        - target_col (str): Tên cột mục tiêu (mặc định là `"MEDV"`).
    Output:
        - Dict[str, Any]: Từ điển chứa số mẫu, số đặc trưng, danh sách tên đặc trưng, kiểu dữ liệu,
          số giá trị thiếu, số bản ghi trùng lặp, thống kê mô tả và hệ số tương quan với biến mục tiêu.
    How: Sử dụng các phép toán vector hóa của `pandas` (`shape`, `dtypes`, `isnull().sum()`,
         `duplicated().sum()`, `describe()`, `corr()`) để trích xuất báo cáo kiểm tra.
    """
    if target_col not in df.columns:
        raise ValueError(f"Cột mục tiêu '{target_col}' không tồn tại trong DataFrame: {df.columns.tolist()}")

    feature_cols = [col for col in df.columns if col != target_col]
    missing_counts = df.isnull().sum().astype(int).to_dict()
    total_missing = int(sum(missing_counts.values()))
    duplicate_rows = int(df.duplicated().sum())

    desc_df = df.describe().round(4)
    corr_with_target = (
        df.corr(numeric_only=True)[target_col]
        .drop(labels=[target_col], errors="ignore")
        .sort_values(ascending=False)
        .round(4)
        .to_dict()
    )

    return {
        "n_samples": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_features": int(len(feature_cols)),
        "feature_names": feature_cols,
        "target_name": target_col,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values_per_column": missing_counts,
        "total_missing_values": total_missing,
        "duplicate_records": duplicate_rows,
        "target_summary": desc_df[target_col].to_dict(),
        "descriptive_statistics": desc_df.to_dict(),
        "correlation_with_target": corr_with_target,
    }


def extract_features_and_target(
    df: pd.DataFrame, target_col: str = EXPECTED_BOSTON_TARGET
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    What: Tách DataFrame tổng thể thành ma trận đặc trưng đầu vào `X` và vector biến mục tiêu `y`.
    Why: Chuẩn bị đúng định dạng đầu vào cho bước chia tập dữ liệu `train_test_split(X, y)`
         theo yêu cầu Task 2 của Practice Exercise 4.
    Input:
        - df (pd.DataFrame): Bảng dữ liệu hoàn chỉnh.
        - target_col (str): Tên cột mục tiêu cần dự đoán (`"MEDV"`).
    Output:
        - Tuple[pd.DataFrame, pd.Series]: Ma trận đặc trưng `X` (506 x 13) và biến mục tiêu `y` (506,).
    How: Sao chép các cột khác `target_col` vào `X` và trích xuất cột `target_col` vào `y`
         dưới kiểu số thực `float64`.
    """
    if target_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột mục tiêu '{target_col}' trong dữ liệu.")

    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].astype(np.float64).copy()
    y = df[target_col].astype(np.float64).copy()
    return X, y


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    What: Hàm điều phối thực thi độc lập bước tải và kiểm tra bộ dữ liệu.
    Why: Cho phép kiểm tra nhanh nguồn dữ liệu từ dòng lệnh (`python -m src.data_loader`)
         hoặc được gọi từ `pipeline.py`.
    Input:
        - run_id (Optional[str]): Mã định danh phiên chạy hiện tại.
    Output:
        - Dict[str, Any]: Thông tin tổng hợp về bộ dữ liệu và kết quả kiểm tra.
    How: Tải cấu hình, gọi `load_housing_dataframe()`, chạy `inspect_dataset()`,
         in báo cáo tóm tắt ra màn hình và ghi nhận vào hệ thống tracking.
    """
    own_run = run_id is None
    if own_run:
        run_id = create_run()

    start_time = time.perf_counter()
    write_log(run_id, "DATA_LOADING", "START", "Bắt đầu tải và kiểm tra bộ dữ liệu từ data/raw/boston_housing.csv.")

    try:
        config = load_config()
        df, source_info = load_housing_dataframe(config)
        inspection = inspect_dataset(df, target_col=source_info["target_column"])
        duration = round(time.perf_counter() - start_time, 4)

        print("\n" + "=" * 65)
        print("BƯỚC 1: TẢI VÀ KIỂM TRA DỮ LIỆU (DATA LOADING & VERIFICATION)")
        print("=" * 65)
        print(f"[INFO] Tên bộ dữ liệu : {source_info['dataset_name']}")
        print(f"[INFO] Nguồn dữ liệu  : {source_info['source_type']}")
        print(f"[INFO] Đường dẫn tệp  : {source_info['file_path']}")
        print(f"[INFO] Kích thước     : {inspection['n_samples']} mẫu x {inspection['n_columns']} cột "
              f"({inspection['n_features']} đặc trưng + 1 mục tiêu '{inspection['target_name']}')")
        print(f"[INFO] Giá trị thiếu  : {inspection['total_missing_values']}")
        print(f"[INFO] Dòng trùng lặp : {inspection['duplicate_records']}")

        output_payload = {
            "source_info": source_info,
            "inspection": {
                "n_samples": inspection["n_samples"],
                "n_features": inspection["n_features"],
                "feature_names": inspection["feature_names"],
                "target_name": inspection["target_name"],
                "total_missing_values": inspection["total_missing_values"],
                "duplicate_records": inspection["duplicate_records"],
                "target_summary": inspection["target_summary"],
                "top_correlations": inspection["correlation_with_target"],
            },
            "duration_seconds": duration,
        }

        update_progress(
            run_id,
            "DATA_LOADING",
            "SUCCESS",
            input_data={"raw_path": source_info["file_path"]},
            output_data=output_payload,
        )
        write_log(
            run_id,
            "DATA_LOADING",
            "SUCCESS",
            f"Đã tải thành công {inspection['n_samples']} mẫu, {inspection['n_features']} đặc trưng trong {duration}s.",
        )

        if own_run:
            finish_run(run_id, "SUCCESS")

        return output_payload

    except Exception as exc:
        update_progress(run_id, "DATA_LOADING", "FAILED", error=str(exc))
        write_log(run_id, "DATA_LOADING", "FAILED", str(exc))
        if own_run:
            finish_run(run_id, "FAILED")
        raise


if __name__ == "__main__":
    main()