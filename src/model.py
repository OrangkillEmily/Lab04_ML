"""
Module định nghĩa kiến trúc các mô hình học máy (`src/model.py`) cho dự án LAB04 (PyTorch Edition).

Trách nhiệm của module:
1. Xây dựng mô hình trọng tâm **Multilayer Perceptron (MLP) Regression** bằng **PyTorch (`torch.nn.Module`)**:
   - Đóng gói dữ liệu chuẩn PyTorch với `BostonHousingDataset(Dataset)` và `DataLoader`.
   - Kiến trúc `BostonMLPNet(nn.Module)`:
     `Input (13)` -> `nn.Linear(13, 32) + ReLU + Dropout` -> `nn.Linear(32, 16) + ReLU + Dropout` -> `nn.Linear(16, 1) (Linear)`.
   - Khởi tạo trọng số **He Normal** (`kaiming_normal_`) cho tầng `ReLU` và **Xavier Uniform** (`xavier_uniform_`) cho tầng đầu ra `Linear`.
   - Hàm mất mát `nn.MSELoss()`, bộ tối ưu `torch.optim.Adam(lr=0.005, weight_decay=l2_lambda)`, tự động nhận **CUDA GPU** nếu khả dụng hoặc CPU.
   - Hỗ trợ `EarlyStopping` trên tập Validation (`model.eval()` + `torch.no_grad()`), lưu/nạp trọng số chuẩn `.pth` qua `torch.save(state_dict)`.
2. Xây dựng 3 mô hình học máy đối chiếu (Baseline Regression Models):
   - `Linear Regression` (OLS), `Decision Tree Regressor` (CART), `Random Forest Regressor` (Ensemble, `max_features='sqrt'`).
3. Giữ nguyên lớp tương thích toán học `NumPySequentialMLP` và các lớp `*Compat` phục vụ Unit Tests & fallback offline.
"""

import copy
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.init as init
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

HAS_TENSORFLOW = False  # Chuyển hoàn toàn sang PyTorch theo yêu cầu môn học

try:
    from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor
    from sklearn.linear_model import LinearRegression as SklearnLinearRegression
    from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def get_device() -> Any:
    """Tự động chọn GPU (CUDA) nếu máy có GPU hỗ trợ PyTorch, ngược lại dùng CPU."""
    if HAS_PYTORCH:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return "cpu"


def get_backend_info() -> Dict[str, Any]:
    """Trả về thông tin backend đang sử dụng cho MLP (PyTorch) và Baselines (scikit-learn)."""
    device_str = str(get_device()) if HAS_PYTORCH else "cpu"
    return {
        "has_pytorch": HAS_PYTORCH,
        "has_tensorflow": False,
        "has_sklearn": HAS_SKLEARN,
        "device": device_str,
        "mlp_backend": f"PyTorch (torch.nn.Module | {device_str.upper()})" if HAS_PYTORCH else "NumPySequentialMLP (Exact Adam + Backprop)",
        "baseline_backend": "scikit-learn" if HAS_SKLEARN else "NumPy/SciPy Exact Regression Engine",
    }


if HAS_PYTORCH:
    class BostonHousingDataset(Dataset):
        """Custom PyTorch Dataset lưu trữ đặc trưng X và nhãn giá nhà y (MEDV)."""

        def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
            self.X = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
            self.y = torch.tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1), dtype=torch.float32)

        def __len__(self) -> int:
            return len(self.X)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.X[idx], self.y[idx]


    class BostonMLPNet(nn.Module):
        """Mạng nơ-ron đa tầng PyTorch (nn.Module) cho bài toán hồi quy giá nhà."""

        def __init__(
            self,
            input_dim: int,
            hidden_units: List[int],
            hidden_activations: Optional[List[str]] = None,
            output_activation: str = "linear",
            dropout_rate: float = 0.0,
        ) -> None:
            super().__init__()
            self.input_dim = int(input_dim)
            self.hidden_units = [int(u) for u in hidden_units]
            self.hidden_activations = list(hidden_activations or ["relu"] * len(self.hidden_units))
            self.output_activation = str(output_activation).lower()
            self.dropout_rate = float(dropout_rate)

            self.layer_specs: List[Tuple[str, nn.Module, str]] = []
            self.net = nn.Sequential()
            prev_dim = self.input_dim

            for idx, (units, act_name) in enumerate(zip(self.hidden_units, self.hidden_activations), start=1):
                linear = nn.Linear(prev_dim, units)
                if act_name.lower() in ("relu", "leaky_relu"):
                    init.kaiming_normal_(linear.weight, nonlinearity="relu")
                else:
                    init.xavier_uniform_(linear.weight)
                init.zeros_(linear.bias)

                layer_name = f"hidden_layer_{idx}"
                self.net.add_module(layer_name, linear)
                self.layer_specs.append((layer_name, linear, act_name.lower()))

                if act_name.lower() == "relu":
                    self.net.add_module(f"relu_{idx}", nn.ReLU())
                elif act_name.lower() == "leaky_relu":
                    self.net.add_module(f"leaky_relu_{idx}", nn.LeakyReLU(0.01))
                elif act_name.lower() == "tanh":
                    self.net.add_module(f"tanh_{idx}", nn.Tanh())
                elif act_name.lower() == "sigmoid":
                    self.net.add_module(f"sigmoid_{idx}", nn.Sigmoid())

                if self.dropout_rate > 0.0:
                    drop = nn.Dropout(p=self.dropout_rate)
                    drop_name = f"dropout_{idx}"
                    self.net.add_module(drop_name, drop)
                    self.layer_specs.append((drop_name, drop, "none"))

                prev_dim = units

            out_layer = nn.Linear(prev_dim, 1)
            init.xavier_uniform_(out_layer.weight)
            init.zeros_(out_layer.bias)
            self.net.add_module("output_layer", out_layer)
            self.layer_specs.append(("output_layer", out_layer, self.output_activation))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    class PyTorchMLPRegressor:
        """
        Wrapper chuẩn hóa giao diện cho BostonMLPNet (PyTorch):
        - Tự động dùng GPU (CUDA) nếu khả dụng hoặc CPU.
        - Thực thi đầy đủ vòng lặp Train (DataLoader, zero_grad, backward, step)
          và Validation (model.eval(), torch.no_grad(), EarlyStopping).
        - Lưu/nạp trọng số chuẩn PyTorch (.pth).
        """

        def __init__(
            self,
            input_dim: int,
            hidden_units: List[int],
            hidden_activations: Optional[List[str]] = None,
            output_activation: str = "linear",
            learning_rate: float = 0.005,
            l2_lambda: float = 0.0,
            dropout_rate: float = 0.0,
            random_state: int = 42,
            architecture_name: str = "PyTorch_MLP_Regressor",
        ) -> None:
            self.input_dim = int(input_dim)
            self.hidden_units = [int(u) for u in hidden_units]
            self.hidden_activations = list(hidden_activations or ["relu"] * len(self.hidden_units))
            self.output_activation = str(output_activation).lower()
            self.learning_rate = float(learning_rate)
            self.l2_lambda = float(l2_lambda)
            self.dropout_rate = float(dropout_rate)
            self.random_state = int(random_state)
            self.name = str(architecture_name)

            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

            self.device = get_device()
            self.model = BostonMLPNet(
                input_dim=self.input_dim,
                hidden_units=self.hidden_units,
                hidden_activations=self.hidden_activations,
                output_activation=self.output_activation,
                dropout_rate=self.dropout_rate,
            ).to(self.device)
            self.best_epoch_ = 0

        @staticmethod
        def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
            yt = y_true.ravel()
            yp = y_pred.ravel()
            err = yt - yp
            mse = float(np.mean(err ** 2))
            mae = float(np.mean(np.abs(err)))
            mape = float(np.mean(np.abs(err) / np.maximum(np.abs(yt), 1e-8)) * 100.0)
            return mse, mae, mape

        def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 100,
            batch_size: int = 32,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            early_stopping_cfg: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            **kwargs: Any,
        ) -> "TrainingHistoryCompat":
            torch.manual_seed(self.random_state)
            dataset = BostonHousingDataset(X, y)
            generator = torch.Generator()
            generator.manual_seed(self.random_state)
            train_loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, generator=generator)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.l2_lambda,
            )

            es_cfg = early_stopping_cfg or {}
            es_enabled = bool(es_cfg.get("enabled", True)) and (validation_data is not None)
            patience = int(es_cfg.get("patience", 15))
            min_delta = float(es_cfg.get("min_delta", 0.001))
            restore_best = bool(es_cfg.get("restore_best_weights", True))

            history: Dict[str, List[float]] = {
                "loss": [], "mae": [], "mape": [],
                "val_loss": [], "val_mae": [], "val_mape": [],
            }

            best_val_loss = float("inf")
            best_state = copy.deepcopy(self.model.state_dict())
            best_epoch = 1
            no_improve = 0

            X_tr_arr = np.asarray(X, dtype=np.float32)
            y_tr_arr = np.asarray(y, dtype=np.float32).ravel()
            X_val_arr = np.asarray(validation_data[0], dtype=np.float32) if validation_data is not None else None
            y_val_arr = np.asarray(validation_data[1], dtype=np.float32).ravel() if validation_data is not None else None

            for epoch in range(1, int(epochs) + 1):
                # 1. Training mode (bật Dropout)
                self.model.train()
                for bx, by in train_loader:
                    bx = bx.to(self.device)
                    by = by.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(bx)
                    loss = criterion(outputs, by)
                    loss.backward()
                    optimizer.step()

                # 2. Evaluation mode (tắt Dropout & tắt autograd)
                tr_pred = self.predict(X_tr_arr)
                tr_mse, tr_mae, tr_mape = self._compute_metrics(y_tr_arr, tr_pred)
                history["loss"].append(tr_mse)
                history["mae"].append(tr_mae)
                history["mape"].append(tr_mape)

                if X_val_arr is not None and y_val_arr is not None:
                    val_pred = self.predict(X_val_arr)
                    val_mse, val_mae, val_mape = self._compute_metrics(y_val_arr, val_pred)
                    history["val_loss"].append(val_mse)
                    history["val_mae"].append(val_mae)
                    history["val_mape"].append(val_mape)

                    if val_mse < best_val_loss - min_delta:
                        best_val_loss = val_mse
                        best_epoch = epoch
                        best_state = copy.deepcopy(self.model.state_dict())
                        no_improve = 0
                    else:
                        no_improve += 1
                        if es_enabled and no_improve >= patience:
                            break

            if es_enabled and restore_best:
                self.model.load_state_dict(best_state)
            self.best_epoch_ = best_epoch if validation_data is not None else len(history["loss"])

            if validation_data is None:
                history.pop("val_loss")
                history.pop("val_mae")
                history.pop("val_mape")

            return TrainingHistoryCompat(history)

        def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
            self.model.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32, device=self.device)
                preds = self.model(x_tensor).cpu().numpy()
            return preds.astype(np.float64)

        def count_params(self) -> int:
            return int(sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        def get_layer_stats(self) -> List[Dict[str, Any]]:
            """Trích xuất thống kê trọng số từng tầng phục vụ src/model_summary.py."""
            stats = []
            for name, module, act in self.model.layer_specs:
                if isinstance(module, nn.Linear):
                    w = module.weight.detach().cpu().numpy()
                    b = module.bias.detach().cpu().numpy()
                    stats.append({
                        "name": name,
                        "shape": f"({module.in_features} -> {module.out_features})",
                        "activation": act,
                        "params": int(w.size + b.size),
                        "w_mean": float(np.mean(w)),
                        "w_std": float(np.std(w)),
                        "w_l2_norm": float(np.linalg.norm(w)),
                    })
                elif isinstance(module, nn.Dropout):
                    stats.append({
                        "name": name,
                        "shape": "(dropout)",
                        "activation": "none",
                        "params": 0,
                        "w_mean": 0.0,
                        "w_std": 0.0,
                        "w_l2_norm": 0.0,
                    })
            return stats

        def save(self, filepath: Union[str, Path]) -> Path:
            target_path = Path(filepath)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "format": "LAB04_PYTORCH_MLP_PTH",
                "state_dict": self.model.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "hidden_units": self.hidden_units,
                    "hidden_activations": self.hidden_activations,
                    "output_activation": self.output_activation,
                    "learning_rate": self.learning_rate,
                    "l2_lambda": self.l2_lambda,
                    "dropout_rate": self.dropout_rate,
                    "random_state": self.random_state,
                    "architecture_name": self.name,
                    "total_params": self.count_params(),
                },
            }
            torch.save(checkpoint, target_path)
            return target_path


class TrainingHistoryCompat:
    """
    What: Đối tượng lưu trữ lịch sử huấn luyện theo từng epoch, tương thích với đối tượng
          `History` trả về từ hàm `model.fit()` của Keras.
    Why: Cho phép `train.py`, `visualize.py` và `notebooks/lab04.ipynb` truy cập thuộc tính
         `history.history['loss']`, `history.history['val_loss']`, `history.history['mae']`,
         `history.history['val_mae']`, `history.history['mape']`, `history.history['val_mape']`
         theo cùng một cú pháp chuẩn của Keras.
    Input:
        - history_dict (Dict[str, List[float]]): Từ điển chứa danh sách giá trị loss/metric qua các epoch.
    Output: Đối tượng có thuộc tính `.history` và `.epoch`.
    How: Lưu trữ từ điển lịch sử và danh sách chỉ số epoch `0 .. n_epochs-1`.
    """

    def __init__(self, history_dict: Dict[str, List[float]]) -> None:
        self.history = history_dict
        n_epochs = len(history_dict.get("loss", []))
        self.epoch = list(range(n_epochs))


class NumPySequentialMLP:
    """
    What: Cài đặt toán học tường minh của mạng Multilayer Perceptron (MLP) Regression
          theo đúng kiến trúc `Sequential([Dense(32, relu), Dense(16, relu), Dense(1, linear)])`
          và thuật toán huấn luyện Backpropagation + Adam Optimizer.
    Why: Đảm bảo mạng nơ-ron có thể huấn luyện thực tế, hội tụ thực sự và sinh ra trọng số,
         đường cong Loss/MAE/MAPE và dự đoán chính xác ngay cả trong môi trường không cài sẵn PyTorch;
         đồng thời tương thích hoàn toàn với giao diện `.compile()`, `.fit()`, `.evaluate()`, `.predict()`
         và `.save()` của ` PyTorchMLPRegressor`.
    Input:
        - input_dim (int): Số lượng đặc trưng đầu vào ($d = 13$).
        - hidden_units (List[int]): Số nơ-ron ở các tầng ẩn (mặc định `[32, 16]`).
        - hidden_activations (List[str]): Hàm kích hoạt tầng ẩn (mặc định `['relu', 'relu']`).
        - output_activation (str): Hàm kích hoạt tầng đầu ra (mặc định `'linear'`).
        - learning_rate (float): Tốc độ học $\\eta$ của bộ tối ưu Adam (mặc định `0.005`).
        - random_state (int): Hạt giống ngẫu nhiên để khởi tạo trọng số He/Glorot.
    Output: Thực thể mạng MLP có khả năng huấn luyện, đánh giá, dự đoán và lưu trữ trọng số.
    How:
        1. **Forward Propagation (Lan truyền thuận)**:
           - Với mỗi tầng $l \\in \\{1, \\dots, L\\}$:
             $$Z^{(l)} = A^{(l-1)} W^{(l)} + b^{(l)}, \\quad A^{(l)} = f^{(l)}(Z^{(l)})$$
        2. **Loss Calculation (Hàm mất mát MSE & chỉ số MAE, MAPE - Slide 10 Chapter 5)**:
           $$\\mathcal{L}_{\\text{MSE}} = \\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2, \\quad \\text{MAPE} = \\frac{100\\%}{N}\\sum_{i=1}^{N}\\frac{|y_i - \\hat{y}_i|}{|y_i|}$$
        3. **Backpropagation (Lan truyền ngược)** và **Adam Optimization**.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_units: Optional[List[int]] = None,
        hidden_activations: Optional[List[str]] = None,
        output_activation: str = "linear",
        learning_rate: float = 0.005,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        random_state: int = 42,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_units = hidden_units or [32, 16]
        self.hidden_activations = hidden_activations or ["relu"] * len(self.hidden_units)
        self.output_activation = output_activation
        self.learning_rate = float(learning_rate)
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.epsilon = float(epsilon)
        self.random_state = int(random_state)
        self.loss_name = "mean_squared_error"
        self.metric_names = ["mae", "mape"]

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self._init_parameters()

    def _init_parameters(self) -> None:
        rng = np.random.RandomState(self.random_state)
        layer_dims = [self.input_dim] + list(self.hidden_units) + [1]
        self.weights = []
        self.biases = []

        for idx in range(len(layer_dims) - 1):
            fan_in = layer_dims[idx]
            fan_out = layer_dims[idx + 1]
            std = np.sqrt(2.0 / fan_in) if idx < len(layer_dims) - 2 else np.sqrt(1.0 / fan_in)
            W = rng.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float64)
            b = np.zeros((1, fan_out), dtype=np.float64)
            self.weights.append(W)
            self.biases.append(b)

    @staticmethod
    def _activate(z: np.ndarray, activation: str) -> np.ndarray:
        act = activation.lower()
        if act == "relu":
            return np.maximum(0.0, z)
        if act == "sigmoid":
            z_clip = np.clip(z, -500.0, 500.0)
            return 1.0 / (1.0 + np.exp(-z_clip))
        if act == "tanh":
            return np.tanh(z)
        return z

    @staticmethod
    def _activate_derivative(z: np.ndarray, activation: str) -> np.ndarray:
        act = activation.lower()
        if act == "relu":
            return (z > 0.0).astype(np.float64)
        if act == "sigmoid":
            s = NumPySequentialMLP._activate(z, "sigmoid")
            return s * (1.0 - s)
        if act == "tanh":
            t = np.tanh(z)
            return 1.0 - t ** 2
        return np.ones_like(z, dtype=np.float64)

    def compile(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: Optional[List[str]] = None,
        learning_rate: Optional[float] = None,
    ) -> None:
        self.loss_name = loss
        self.metric_names = metrics or ["mae", "mape"]
        if learning_rate is not None:
            self.learning_rate = float(learning_rate)

    def _forward_pass(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations = [np.asarray(X, dtype=np.float64)]
        z_values = []
        all_acts = list(self.hidden_activations) + [self.output_activation]

        for W, b, act_name in zip(self.weights, self.biases, all_acts):
            z = activations[-1] @ W + b
            a = self._activate(z, act_name)
            z_values.append(z)
            activations.append(a)

        return z_values, activations

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        validation_split: float = 0.0,
        early_stopping_cfg: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
    ) -> TrainingHistoryCompat:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        if validation_data is not None:
            X_train_use, y_train_use = X_arr, y_arr
            X_val = np.asarray(validation_data[0], dtype=np.float64)
            y_val = np.asarray(validation_data[1], dtype=np.float64).reshape(-1, 1)
        elif validation_split > 0.0:
            n_val = int(np.ceil(len(X_arr) * validation_split))
            n_tr = len(X_arr) - n_val
            X_train_use, y_train_use = X_arr[:n_tr], y_arr[:n_tr]
            X_val, y_val = X_arr[n_tr:], y_arr[n_tr:]
        else:
            X_train_use, y_train_use = X_arr, y_arr
            X_val, y_val = None, None

        m_w = [np.zeros_like(W) for W in self.weights]
        v_w = [np.zeros_like(W) for W in self.weights]
        m_b = [np.zeros_like(b) for b in self.biases]
        v_b = [np.zeros_like(b) for b in self.biases]

        history_dict: Dict[str, List[float]] = {
            "loss": [],
            "mae": [],
            "mape": [],
        }
        if X_val is not None:
            history_dict["val_loss"] = []
            history_dict["val_mae"] = []
            history_dict["val_mape"] = []

        es_enabled = bool(early_stopping_cfg and early_stopping_cfg.get("enabled", False))
        patience = int(early_stopping_cfg.get("patience", 15)) if early_stopping_cfg else 15
        min_delta = float(early_stopping_cfg.get("min_delta", 0.001)) if early_stopping_cfg else 0.001
        restore_best = bool(early_stopping_cfg.get("restore_best_weights", True)) if early_stopping_cfg else True

        best_val_loss = float("inf")
        best_weights = [W.copy() for W in self.weights]
        best_biases = [b.copy() for b in self.biases]
        best_epoch = 0
        wait = 0

        rng = np.random.RandomState(self.random_state)
        n_samples = len(X_train_use)
        t_step = 0
        all_acts = list(self.hidden_activations) + [self.output_activation]

        for epoch in range(int(epochs)):
            perm = rng.permutation(n_samples)
            X_shuffled = X_train_use[perm]
            y_shuffled = y_train_use[perm]

            for start_idx in range(0, n_samples, int(batch_size)):
                end_idx = min(start_idx + int(batch_size), n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                bs = float(len(X_batch))

                z_vals, acts = self._forward_pass(X_batch)
                y_hat = acts[-1]

                delta = (2.0 / bs) * (y_hat - y_batch) * self._activate_derivative(z_vals[-1], all_acts[-1])

                grads_w = [None] * len(self.weights)
                grads_b = [None] * len(self.biases)

                for layer_idx in range(len(self.weights) - 1, -1, -1):
                    grads_w[layer_idx] = acts[layer_idx].T @ delta
                    grads_b[layer_idx] = np.sum(delta, axis=0, keepdims=True)
                    if layer_idx > 0:
                        delta = (delta @ self.weights[layer_idx].T) * self._activate_derivative(
                            z_vals[layer_idx - 1], all_acts[layer_idx - 1]
                        )

                t_step += 1
                for l_idx in range(len(self.weights)):
                    m_w[l_idx] = self.beta_1 * m_w[l_idx] + (1.0 - self.beta_1) * grads_w[l_idx]
                    v_w[l_idx] = self.beta_2 * v_w[l_idx] + (1.0 - self.beta_2) * (grads_w[l_idx] ** 2)
                    m_w_hat = m_w[l_idx] / (1.0 - self.beta_1 ** t_step)
                    v_w_hat = v_w[l_idx] / (1.0 - self.beta_2 ** t_step)
                    self.weights[l_idx] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

                    m_b[l_idx] = self.beta_1 * m_b[l_idx] + (1.0 - self.beta_1) * grads_b[l_idx]
                    v_b[l_idx] = self.beta_2 * v_b[l_idx] + (1.0 - self.beta_2) * (grads_b[l_idx] ** 2)
                    m_b_hat = m_b[l_idx] / (1.0 - self.beta_1 ** t_step)
                    v_b_hat = v_b[l_idx] / (1.0 - self.beta_2 ** t_step)
                    self.biases[l_idx] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

            train_loss, train_mae, train_mape = self.evaluate(X_train_use, y_train_use, verbose=0)
            history_dict["loss"].append(float(train_loss))
            history_dict["mae"].append(float(train_mae))
            history_dict["mape"].append(float(train_mape))

            if X_val is not None:
                val_loss, val_mae, val_mape = self.evaluate(X_val, y_val, verbose=0)
                history_dict["val_loss"].append(float(val_loss))
                history_dict["val_mae"].append(float(val_mae))
                history_dict["val_mape"].append(float(val_mape))

                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    best_weights = [W.copy() for W in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                    best_epoch = epoch + 1
                    wait = 0
                else:
                    wait += 1
                    if es_enabled and wait >= patience:
                        break

        if X_val is not None and restore_best and best_val_loss < float("inf"):
            self.weights = [W.copy() for W in best_weights]
            self.biases = [b.copy() for b in best_biases]

        self.best_epoch_ = best_epoch if best_epoch > 0 else len(history_dict["loss"])
        return TrainingHistoryCompat(history_dict)

    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        _, acts = self._forward_pass(np.asarray(X, dtype=np.float64))
        return acts[-1]

    def evaluate(self, X: np.ndarray, y: np.ndarray, verbose: int = 0) -> Tuple[float, float, float]:
        """
        What: Tính giá trị hàm mất mát MSE, sai số tuyệt đối trung bình MAE và sai số phần trăm
              tuyệt đối trung bình MAPE (%) trên tập `(X, y)`.
        Why: Tương thích trực tiếp với lệnh `model.evaluate(X_test_scaled, y_test)` và bổ sung MAPE
             theo Slide 10 của `Chapter5_Artificial Neural_Networks.pdf`.
        Input:
            - X (np.ndarray): Ma trận đặc trưng đầu vào.
            - y (np.ndarray): Giá trị thực tế.
            - verbose (int): Mức hiển thị.
        Output:
            - Tuple[float, float, float]: Bộ `(mse_loss, mae_score, mape_pct)`.
        How: Dự đoán `y_pred = self.predict(X)` rồi tính MSE, MAE và MAPE (%).
        """
        y_true = np.asarray(y, dtype=np.float64).ravel()
        y_pred = self.predict(X, verbose=0).ravel()
        abs_err = np.abs(y_true - y_pred)
        mse_val = float(np.mean((y_true - y_pred) ** 2))
        mae_val = float(np.mean(abs_err))
        mape_val = float(np.mean(abs_err / np.maximum(np.abs(y_true), 1e-8)) * 100.0)
        return mse_val, mae_val, mape_val

    def get_weights(self) -> List[np.ndarray]:
        flat_list = []
        for W, b in zip(self.weights, self.biases):
            flat_list.append(W.copy())
            flat_list.append(b.ravel().copy())
        return flat_list

    def set_weights(self, weight_list: List[np.ndarray]) -> None:
        self.weights = []
        self.biases = []
        for idx in range(0, len(weight_list), 2):
            W = np.asarray(weight_list[idx], dtype=np.float64)
            b = np.asarray(weight_list[idx + 1], dtype=np.float64).reshape(1, -1)
            self.weights.append(W)
            self.biases.append(b)

    def count_params(self) -> int:
        return int(sum(W.size + b.size for W, b in zip(self.weights, self.biases)))

    def save(self, filepath: Union[str, Path]) -> Path:
        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        npz_path = target_path.with_name(target_path.stem + "_weights.npz")

        arrays_dict = {}
        for idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            arrays_dict[f"W_{idx}"] = W
            arrays_dict[f"b_{idx}"] = b
        np.savez(npz_path, **arrays_dict)

        config_meta = {
            "format": "LAB04_MLP_KERAS_ARCHIVE",
            "input_dim": self.input_dim,
            "hidden_units": self.hidden_units,
            "hidden_activations": self.hidden_activations,
            "output_activation": self.output_activation,
            "learning_rate": self.learning_rate,
            "loss": self.loss_name,
            "metrics": self.metric_names,
            "total_params": self.count_params(),
        }

        with zipfile.ZipFile(target_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("config.json", json.dumps(config_meta, indent=2))
            zf.write(npz_path, arcname="weights.npz")

        return target_path



def build_mlp_model(
    input_dim: int,
    config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Khởi tạo mô hình MLP Regression bằng PyTorch (ưu tiên) hoặc NumPySequentialMLP fallback."""
    mlp_cfg = (config or {}).get("mlp_model", {})
    hidden_layers_cfg = mlp_cfg.get(
        "hidden_layers",
        [
            {"units": 32, "activation": "relu", "name": "hidden_layer_1"},
            {"units": 16, "activation": "relu", "name": "hidden_layer_2"},
        ],
    )
    output_cfg = mlp_cfg.get("output_layer", {"units": 1, "activation": "linear", "name": "output_layer"})
   
    learning_rate = float(mlp_cfg.get("learning_rate", 0.005))
    l2_lambda = float(mlp_cfg.get("l2_lambda", 0.0))
    dropout_rate = float(mlp_cfg.get("dropout_rate", 0.0))
    loss_fn = str(mlp_cfg.get("loss", "mean_squared_error"))
    metrics_list = list(mlp_cfg.get("metrics", ["mae", "mape"]))
    random_state = int((config or {}).get("data", {}).get("random_state", 42))
    arch_name = str(mlp_cfg.get("architecture_name", "PyTorch_MLP_Regressor"))

    hidden_units = [int(item.get("units", 32)) for item in hidden_layers_cfg]
    hidden_acts = [str(item.get("activation", "relu")) for item in hidden_layers_cfg]
    out_act = str(output_cfg.get("activation", "linear"))

    if HAS_PYTORCH:
        return PyTorchMLPRegressor(
            input_dim=int(input_dim),
            hidden_units=hidden_units,
            hidden_activations=hidden_acts,
            output_activation=out_act,
            learning_rate=learning_rate,
            l2_lambda=l2_lambda,
            dropout_rate=dropout_rate,
            random_state=random_state,
            architecture_name=arch_name,
        )

    model = NumPySequentialMLP(
        input_dim=int(input_dim),
        hidden_units=hidden_units,
        hidden_activations=hidden_acts,
        output_activation=out_act,
        learning_rate=learning_rate,
        l2_lambda=l2_lambda,
        dropout_rate=dropout_rate,
        random_state=random_state,
    )
    model.compile(optimizer="adam", loss=loss_fn, metrics=metrics_list, learning_rate=learning_rate)
    return model


def load_mlp_model(filepath: Union[str, Path], config: Optional[Dict[str, Any]] = None) -> Any:
    """Nạp mô hình MLP PyTorch đã lưu từ tệp `.pth` (có fallback đọc `.npz`/zip nếu cần)."""
    target_path = Path(filepath)
    if not target_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp mô hình MLP tại: {target_path}")

    if HAS_PYTORCH and target_path.suffix.lower() in (".pth", ".pt"):
        device = get_device()
        checkpoint = torch.load(target_path, map_location=device, weights_only=False)
        cfg = checkpoint.get("config", {})
        regressor = PyTorchMLPRegressor(
            input_dim=int(cfg.get("input_dim", 13)),
            hidden_units=list(cfg.get("hidden_units", [32, 16])),
            hidden_activations=list(cfg.get("hidden_activations", ["relu", "relu"])),
            output_activation=str(cfg.get("output_activation", "linear")),
            learning_rate=float(cfg.get("learning_rate", 0.005)),
            l2_lambda=float(cfg.get("l2_lambda", 0.0)),
            dropout_rate=float(cfg.get("dropout_rate", 0.0)),
            random_state=int(cfg.get("random_state", 42)),
            architecture_name=str(cfg.get("architecture_name", "PyTorch_MLP_Regressor")),
        )
        regressor.model.load_state_dict(checkpoint["state_dict"])
        regressor.model.eval()
        return regressor

    if zipfile.is_zipfile(target_path):
        with zipfile.ZipFile(target_path, "r") as zf:
            names = zf.namelist()
            if "weights.npz" in names and "config.json" in names:
                meta = json.loads(zf.read("config.json").decode("utf-8"))
                npz_path = target_path.with_name(target_path.stem + "_weights.npz")
                if not npz_path.exists():
                    zf.extract("weights.npz", path=target_path.parent)
                    (target_path.parent / "weights.npz").rename(npz_path)
                npz_data = np.load(npz_path)
                model = NumPySequentialMLP(
                    input_dim=int(meta.get("input_dim", 13)),
                    hidden_units=list(meta.get("hidden_units", [32, 16])),
                    hidden_activations=list(meta.get("hidden_activations", ["relu", "relu"])),
                    output_activation=str(meta.get("output_activation", "linear")),
                    learning_rate=float(meta.get("learning_rate", 0.005)),
                )
                n_layers = len([k for k in npz_data.files if k.startswith("W_")])
                model.weights = [npz_data[f"W_{i}"].copy() for i in range(n_layers)]
                model.biases = [npz_data[f"b_{i}"].copy() for i in range(n_layers)]
                return model

    raise RuntimeError(f"Không thể nạp tệp mô hình '{target_path}'. Hãy đảm bảo đã cài PyTorch và huấn luyện lại.")


class LinearRegressionCompat:
    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = bool(fit_intercept)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionCompat":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()

        if self.fit_intercept:
            ones = np.ones((X_arr.shape[0], 1), dtype=np.float64)
            X_design = np.hstack([X_arr, ones])
            theta, _, _, _ = np.linalg.lstsq(X_design, y_arr, rcond=None)
            self.coef_ = theta[:-1]
            self.intercept_ = float(theta[-1])
        else:
            theta, _, _, _ = np.linalg.lstsq(X_arr, y_arr, rcond=None)
            self.coef_ = theta
            self.intercept_ = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("LinearRegression chưa được fit!")
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr @ self.coef_ + self.intercept_


class DecisionTreeRegressorCompat:
    def __init__(
        self,
        max_depth: Optional[int] = 6,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: Optional[Union[str, int]] = None,
        random_state: int = 42,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = int(random_state)
        self.tree_: Optional[Dict[str, Any]] = None
        self.feature_importances_: Optional[np.ndarray] = None

    def _select_features(self, n_features: int, rng: np.random.RandomState) -> np.ndarray:
        if self.max_features is None:
            return np.arange(n_features)
        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            k = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            k = min(n_features, max(1, self.max_features))
        else:
            k = n_features
        return rng.choice(n_features, size=k, replace=False)

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
        rng: np.random.RandomState,
        importances: np.ndarray,
    ) -> Dict[str, Any]:
        n_samples, n_features = X.shape
        node_value = float(np.mean(y))
        current_sse = float(np.sum((y - node_value) ** 2))

        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or current_sse < 1e-12
        ):
            return {"is_leaf": True, "value": node_value}

        feat_Subset = self._select_features(n_features, rng)
        best_feat = None
        best_thresh = None
        best_sse = current_sse

        for f_idx in feat_Subset:
            col = X[:, f_idx]
            order = np.argsort(col)
            col_sorted = col[order]
            y_sorted = y[order]

            lhs_cnt = np.arange(1, n_samples)
            rhs_cnt = n_samples - lhs_cnt

            valid_mask = (
                (lhs_cnt >= self.min_samples_leaf)
                & (rhs_cnt >= self.min_samples_leaf)
                & (col_sorted[:-1] < col_sorted[1:])
            )
            if not np.any(valid_mask):
                continue

            cumsum_y = np.cumsum(y_sorted)[:-1]
            cumsum_y2 = np.cumsum(y_sorted ** 2)[:-1]
            total_y = float(cumsum_y[-1] + y_sorted[-1])
            total_y2 = float(cumsum_y2[-1] + y_sorted[-1] ** 2)

            lhs_sse = cumsum_y2 - (cumsum_y ** 2) / lhs_cnt
            rhs_y = total_y - cumsum_y
            rhs_y2 = total_y2 - cumsum_y2
            rhs_sse = rhs_y2 - (rhs_y ** 2) / rhs_cnt

            total_split_sse = lhs_sse + rhs_sse
            total_split_sse[~valid_mask] = np.inf

            min_pos = int(np.argmin(total_split_sse))
            if total_split_sse[min_pos] < best_sse:
                best_sse = float(total_split_sse[min_pos])
                best_feat = int(f_idx)
                best_thresh = 0.5 * float(col_sorted[min_pos] + col_sorted[min_pos + 1])

        if best_feat is None or best_thresh is None:
            return {"is_leaf": True, "value": node_value}

        importances[best_feat] += max(0.0, current_sse - best_sse)
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        return {
            "is_leaf": False,
            "feature": best_feat,
            "threshold": best_thresh,
            "value": node_value,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1, rng, importances),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1, rng, importances),
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressorCompat":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        rng = np.random.RandomState(self.random_state)
        importances = np.zeros(X_arr.shape[1], dtype=np.float64)
        self.tree_ = self._build_tree(X_arr, y_arr, depth=0, rng=rng, importances=importances)
        total_imp = float(np.sum(importances))
        self.feature_importances_ = importances / total_imp if total_imp > 0 else importances
        return self

    def _predict_row(self, row: np.ndarray, node: Dict[str, Any]) -> float:
        while not node["is_leaf"]:
            if row[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return float(node["value"])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.tree_ is None:
            raise RuntimeError("DecisionTreeRegressor chưa được fit!")
        X_arr = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_row(row, self.tree_) for row in X_arr], dtype=np.float64)


class RandomForestRegressorCompat:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        max_features: Optional[Union[str, int]] = None,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = int(random_state)
        self.estimators_: List[DecisionTreeRegressorCompat] = []
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressorCompat":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X_arr.shape

        rng = np.random.RandomState(self.random_state)
        self.estimators_ = []
        importances_sum = np.zeros(n_features, dtype=np.float64)
        effective_max_features = max(1, int(np.sqrt(n_features))) if self.max_features == "sqrt" else self.max_features

        for idx in range(self.n_estimators):
            tree_seed = int(rng.randint(0, 10_000_000))
            boot_indices = rng.choice(n_samples, size=n_samples, replace=True)
            tree = DecisionTreeRegressorCompat(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=effective_max_features,
                random_state=tree_seed,
            )
            tree.fit(X_arr[boot_indices], y_arr[boot_indices])
            self.estimators_.append(tree)
            if tree.feature_importances_ is not None:
                importances_sum += tree.feature_importances_

        total_imp = float(np.sum(importances_sum))
        self.feature_importances_ = importances_sum / total_imp if total_imp > 0 else importances_sum
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("RandomForestRegressor chưa được fit!")
        X_arr = np.asarray(X, dtype=np.float64)
        all_preds = np.column_stack([tree.predict(X_arr) for tree in self.estimators_])
        return np.mean(all_preds, axis=1)


def build_baseline_models(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base_cfg = (config or {}).get("baseline_models", {})
    lr_cfg = base_cfg.get("linear_regression", {})
    dt_cfg = base_cfg.get("decision_tree", {})
    rf_cfg = base_cfg.get("random_forest", {})

    if HAS_SKLEARN:
        return {
            "Linear Regression": SklearnLinearRegression(
                fit_intercept=bool(lr_cfg.get("fit_intercept", True))
            ),
            "Decision Tree": SklearnDecisionTreeRegressor(
                max_depth=dt_cfg.get("max_depth", 6),
                min_samples_split=int(dt_cfg.get("min_samples_split", 5)),
                min_samples_leaf=int(dt_cfg.get("min_samples_leaf", 2)),
                random_state=int(dt_cfg.get("random_state", 42)),
            ),
            "Random Forest": SklearnRandomForestRegressor(
                n_estimators=int(rf_cfg.get("n_estimators", 100)),
                max_depth=rf_cfg.get("max_depth", 10),
                min_samples_split=int(rf_cfg.get("min_samples_split", 4)),
                min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 2)),
                max_features=rf_cfg.get("max_features", "sqrt"),
                random_state=int(rf_cfg.get("random_state", 42)),
            ),
        }

    return {
        "Linear Regression": LinearRegressionCompat(
            fit_intercept=bool(lr_cfg.get("fit_intercept", True))
        ),
        "Decision Tree": DecisionTreeRegressorCompat(
            max_depth=dt_cfg.get("max_depth", 6),
            min_samples_split=int(dt_cfg.get("min_samples_split", 5)),
            min_samples_leaf=int(dt_cfg.get("min_samples_leaf", 2)),
            random_state=int(dt_cfg.get("random_state", 42)),
        ),
        "Random Forest": RandomForestRegressorCompat(
            n_estimators=int(rf_cfg.get("n_estimators", 100)),
            max_depth=rf_cfg.get("max_depth", 10),
            min_samples_split=int(rf_cfg.get("min_samples_split", 4)),
            min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 2)),
            max_features=rf_cfg.get("max_features", "sqrt"),
            random_state=int(rf_cfg.get("random_state", 42)),
        ),
    }