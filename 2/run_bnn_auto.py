from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


EPS = 1e-12


@dataclass
class ExperimentConfig:
    data_path: str
    sheet_name: str
    output_dir: str
    device: str = "cuda:0"
    n_splits: int = 10
    n_repeats: int = 4
    seed: int = 42
    batch_size: int = 32
    max_epochs: int = 180
    pretrain_epochs: int = 140
    bnn_epochs: int = 260
    mc_samples_probe: int = 30
    mc_samples_eval: int = 20
    lr: float = 1e-4
    pretrain_lr: float = 1e-4
    bnn_lr: float = 1e-4
    lr_schedule: str = "linear_to_floor"
    lr_floor: float = 5e-5
    plateau_factor: float = 0.5
    plateau_cooldown: int = 1
    weight_decay: float = 1e-6
    hidden_dim: int = 128
    latent_dim: int = 8
    kl_weight: float = 3e-8
    kl_anneal_fraction: float = 0.6
    moped_delta: float = 0.05
    bayesian_scope: str = "predictor_only"
    inner_val_fraction: float = 0.15
    probe_every: int = 8
    early_stop_patience: int = 64
    scheduler_patience: int = 20
    min_lr: float = 1e-7
    loss_kind: str = "mse"
    smooth_l1_beta: float = 0.05
    bnn_auto_beta: float = 0.2
    bnn_auto_use_kl: int = 1
    models: str = "all"
    repeat_start: int = 2
    torch_threads: int = 1
    rf_estimators: int = 600
    rf_min_samples_leaf: int = 2
    gp_restarts_optimizer: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_threads(n_threads: int) -> None:
    torch.set_num_threads(max(1, int(n_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def resolve_device(device_str: str) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        device = torch.device(device_str)
        if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
            return torch.device("cuda:0")
        return device
    except Exception:
        return torch.device("cuda:0")


def ensure_output_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def build_features(df: pd.DataFrame) -> np.ndarray:
    a = df["A"].to_numpy(dtype=np.float64)
    z = df["Z"].to_numpy(dtype=np.float64)
    q = df["Q"].to_numpy(dtype=np.float64)
    l = df["L"].to_numpy(dtype=np.float64)
    n = a - z
    x1 = np.power(a, 1.0 / 3.0)
    x2 = z / np.sqrt(q)
    x3 = np.sqrt(l * (l + 1.0))
    return np.column_stack([a, z, q, n, l, x1, x2, x3]).astype(np.float32)


def validate_dataframe(df: pd.DataFrame) -> None:
    required_columns = {"A", "Z", "Q", "L", "logPa_exp"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing)
        )


def load_main_dataset(
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(config.data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Place BNN_input2.xlsx under the data directory "
            "or pass --data-path explicitly."
        )

    df = pd.read_excel(data_path, sheet_name=config.sheet_name).copy()
    validate_dataframe(df)
    df["N"] = df["A"] - df["Z"]
    features = build_features(df)
    targets_log = df["logPa_exp"].to_numpy(dtype=np.float32)
    targets_prob = np.power(10.0, targets_log).astype(np.float32)
    return df, features, targets_prob, targets_log


def compute_metrics_logprob(preds_log, preds_prob, targets_log, targets_prob) -> dict[str, float]:
    preds_log_t = torch.as_tensor(preds_log, dtype=torch.float32)
    preds_prob_t = torch.as_tensor(preds_prob, dtype=torch.float32)
    targets_log_t = torch.as_tensor(targets_log, dtype=torch.float32)
    targets_prob_t = torch.as_tensor(targets_prob, dtype=torch.float32)
    preds_prob_t = torch.clamp(preds_prob_t, min=EPS, max=1.0)
    targets_prob_t = torch.clamp(targets_prob_t, min=EPS)
    rmse_log = torch.sqrt(torch.mean((preds_log_t - targets_log_t) ** 2)).item()
    rmse_prob = torch.sqrt(torch.mean((preds_prob_t - targets_prob_t) ** 2)).item()
    return {"rmse_log": float(rmse_log), "rmse_prob": float(rmse_prob)}


class CurrentBNNAuto(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        mid_dim = hidden_dim // 2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ELU(),
            nn.Linear(mid_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ELU(),
            nn.Linear(mid_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.skip_proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ELU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim * 2 + input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ELU(),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        skip_latent = self.skip_proj(x)
        fused = torch.cat([latent, skip_latent, x], dim=-1)
        pred_raw = self.predictor(fused).squeeze(-1)
        pred_log = -F.softplus(pred_raw)
        pred_prob = torch.pow(pred_log.new_tensor(10.0), pred_log)
        return {"pred_log": pred_log, "pred_prob": pred_prob, "recon": recon}


class StructuralSBNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.output_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        return self.output_act(self.net(x))


class FFNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred_raw = self.net(x).squeeze(-1)
        pred_log = -F.softplus(pred_raw)
        pred_prob = torch.pow(pred_log.new_tensor(10.0), pred_log)
        return pred_prob


def make_predictor_bnn(model: nn.Module, config: ExperimentConfig) -> nn.Module:
    params = {
        "prior_mu": 0.0,
        "prior_sigma": 0.1,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",
        "moped_enable": True,
        "moped_delta": config.moped_delta,
    }
    # Keep the encoder, decoder, skip projection, and their normalization layers
    # deterministic; only the predictor is converted to Bayesian Linear layers.
    dnn_to_bnn(model.predictor, params)
    return model


def make_full_bnn(model: nn.Module, config: ExperimentConfig) -> nn.Module:
    params = {
        "prior_mu": 0.0,
        "prior_sigma": 0.1,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",
        "moped_enable": True,
        "moped_delta": config.moped_delta,
    }
    dnn_to_bnn(model, params)
    return model


def safe_kl_loss(model: nn.Module, device: torch.device) -> torch.Tensor:
    kl = get_kl_loss(model)
    if kl is None:
        return torch.tensor(0.0, device=device)
    return kl


def predict_mc(model: nn.Module, x_tensor: torch.Tensor, n_samples: int):
    model.eval()
    preds_log = []
    preds_prob = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(x_tensor)
            preds_log.append(out["pred_log"])
            preds_prob.append(out["pred_prob"])
    stack_log = torch.stack(preds_log, dim=0)
    stack_prob = torch.stack(preds_prob, dim=0)
    return stack_log.mean(dim=0), stack_prob.mean(dim=0), stack_log.std(dim=0, unbiased=False)


def predict_mc_prob(model: nn.Module, x_tensor: torch.Tensor, n_samples: int):
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x_tensor)
            preds.append(pred.squeeze(-1))
    stack = torch.stack(preds, dim=0)
    mean_prob = stack.mean(dim=0)
    mean_log = torch.log10(torch.clamp(mean_prob, min=EPS))
    return mean_log, mean_prob, stack.std(dim=0, unbiased=False)


def train_model_phase(
    model: nn.Module,
    x_train: np.ndarray,
    y_train_log: np.ndarray,
    x_val: np.ndarray,
    y_val_prob: np.ndarray,
    y_val_log: np.ndarray,
    epochs: int,
    beta: float,
    use_kl: bool,
    use_recon_loss: bool,
    learning_rate: float,
    config: ExperimentConfig,
    device: torch.device,
):
    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train_log, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    if len(train_loader) == 0:
        raise ValueError("Training split is smaller than the configured batch size.")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.weight_decay,
    )
    if config.lr_schedule == "constant":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=config.scheduler_patience,
            min_lr=config.min_lr,
        )
    elif config.lr_schedule == "plateau_validation":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.plateau_factor,
            patience=config.scheduler_patience,
            min_lr=config.lr_floor,
            cooldown=config.plateau_cooldown,
        )
    elif config.lr_schedule == "linear_to_floor":
        scheduler = None
    else:
        raise ValueError(f"Unknown learning-rate schedule: {config.lr_schedule}")

    best_state = None
    best_probe = float("inf")
    best_epoch = 0
    stale = 0
    epochs_ran = 0

    for epoch in range(epochs):
        epochs_ran = epoch + 1
        if config.lr_schedule == "linear_to_floor":
            progress = 0.0 if epochs <= 1 else epoch / float(epochs - 1)
            phase_lr = learning_rate + (config.lr_floor - learning_rate) * progress
            for param_group in optimizer.param_groups:
                param_group["lr"] = phase_lr
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y_log in train_loader:
            batch_x = batch_x.to(device)
            batch_y_log = batch_y_log.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch_x)
            if config.loss_kind == "mse":
                loss_log = nn.MSELoss()(out["pred_log"], batch_y_log)
            else:
                loss_log = nn.SmoothL1Loss(beta=config.smooth_l1_beta)(
                    out["pred_log"], batch_y_log
                )
            if use_recon_loss:
                recon_loss = nn.MSELoss()(out["recon"], batch_x)
                data_loss = (1.0 - beta) * loss_log + beta * recon_loss
            else:
                data_loss = loss_log
            if use_kl:
                anneal_epochs = max(1, int(round(epochs * config.kl_anneal_fraction)))
                kl_scale = min(1.0, float(epoch + 1) / anneal_epochs)
                kl_loss = safe_kl_loss(model, device) / len(train_dataset)
                loss = data_loss + kl_scale * config.kl_weight * kl_loss
            else:
                loss = data_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
        if scheduler is not None and config.lr_schedule == "constant":
            scheduler.step(epoch_loss / max(1, len(train_loader)))

        if (epoch + 1) % config.probe_every == 0 or epoch == epochs - 1:
            x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
            pred_log, pred_prob, _pred_std = predict_mc(
                model,
                x_val_tensor,
                config.mc_samples_probe if use_kl else 1,
            )
            metrics = compute_metrics_logprob(pred_log.cpu(), pred_prob.cpu(), y_val_log, y_val_prob)
            if scheduler is not None and config.lr_schedule == "plateau_validation":
                scheduler.step(metrics["rmse_log"])
            if metrics["rmse_log"] + 1e-6 < best_probe:
                best_probe = metrics["rmse_log"]
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += config.probe_every
            if stale >= config.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    model.phase_diagnostics = {
        "epochs_ran": int(epochs_ran),
        "best_epoch": int(best_epoch),
        "best_inner_rmse_log": float(best_probe),
        "final_lr": float(optimizer.param_groups[0]["lr"]),
    }
    return model


def train_current_bnn_auto_model(
    x_train,
    y_train_log,
    x_val,
    y_val_prob,
    y_val_log,
    config,
    device,
):
    model = CurrentBNNAuto(
        x_train.shape[1], config.hidden_dim, config.latent_dim
    ).to(device)
    model = train_model_phase(
        model,
        x_train,
        y_train_log,
        x_val,
        y_val_prob,
        y_val_log,
        config.pretrain_epochs,
        config.bnn_auto_beta,
        False,
        True,
        config.pretrain_lr,
        config,
        device,
    )
    pretrain_diagnostics = dict(model.phase_diagnostics)
    model = make_predictor_bnn(model, config).to(device)
    model = train_model_phase(
        model,
        x_train,
        y_train_log,
        x_val,
        y_val_prob,
        y_val_log,
        config.bnn_epochs,
        config.bnn_auto_beta,
        bool(int(config.bnn_auto_use_kl)),
        True,
        config.bnn_lr,
        config,
        device,
    )
    return model, {
        "pretrain": pretrain_diagnostics,
        "bnn": dict(model.phase_diagnostics),
    }


def train_structural_sbnn_model(
    x_train,
    y_train_prob,
    x_val,
    config,
    device,
):
    model = StructuralSBNN(x_train.shape[1], 64).to(device)
    model = make_full_bnn(model, config).to(device)
    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train_prob, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=34,
        shuffle=True,
        drop_last=True,
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=80,
        min_lr=config.min_lr,
    )
    kl_anneal_epochs = max(1, int(round(100 * 0.2)))
    for epoch in range(100):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y_prob in train_loader:
            batch_x = batch_x.to(device)
            batch_y_prob = batch_y_prob.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred_prob = model(batch_x).squeeze(-1)
            data_loss = nn.MSELoss()(pred_prob, batch_y_prob)
            kl_scale = min(1.0, float(epoch + 1) / kl_anneal_epochs)
            kl_loss = safe_kl_loss(model, device) / len(train_dataset)
            loss = data_loss + kl_scale * 1e-6 * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
        scheduler.step(epoch_loss / max(1, len(train_loader)))
    model.eval()
    model.phase_diagnostics = {
        "epochs_ran": 100,
        "final_lr": float(optimizer.param_groups[0]["lr"]),
    }
    return model, {
        "bnn": dict(model.phase_diagnostics),
    }


def train_ffnn(x_train, y_train_prob, x_val, config, device):
    model = FFNN(input_dim=x_train.shape[1], hidden_dim=config.hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=60)
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train_prob, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    for _epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred_prob = model(batch_x).squeeze(-1)
            loss = nn.MSELoss()(pred_prob, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        scheduler.step(epoch_loss / max(1, len(train_loader)))

    model.eval()
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
    pred_log, pred_prob, pred_std = predict_mc_prob(model, x_val_tensor, 1)
    return pred_log.cpu().numpy(), pred_prob.cpu().numpy(), float(pred_std.mean().cpu().item())


def fit_random_forest(x_train_raw, y_train_prob, x_val_raw, config):
    model = RandomForestRegressor(
        n_estimators=config.rf_estimators,
        random_state=config.seed,
        min_samples_leaf=config.rf_min_samples_leaf,
        n_jobs=-1,
    )
    model.fit(x_train_raw, y_train_prob)
    pred_prob = np.asarray(model.predict(x_val_raw), dtype=np.float32)
    pred_log = np.log10(np.clip(pred_prob, EPS, 1.0))
    return pred_log, pred_prob


def fit_linear_ls(x_train, y_train_prob, x_val):
    model = LinearRegression()
    model.fit(x_train, y_train_prob)
    pred_prob = np.asarray(model.predict(x_val), dtype=np.float32)
    pred_log = np.log10(np.clip(pred_prob, EPS, 1.0))
    return pred_log, pred_prob


def fit_gaussian_process(x_train, y_train_prob, x_val, config):
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=np.ones(x_train.shape[1]), length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e0))
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        random_state=config.seed,
        n_restarts_optimizer=config.gp_restarts_optimizer,
    )
    model.fit(x_train, y_train_prob)
    pred_prob, pred_std = model.predict(x_val, return_std=True)
    pred_prob = np.asarray(pred_prob, dtype=np.float32)
    pred_log = np.log10(np.clip(pred_prob, EPS, 1.0))
    return pred_log, pred_prob, float(np.mean(pred_std))


def run_experiment(config: ExperimentConfig):
    device = resolve_device(config.device)
    input_df, features, targets_prob, targets_log = load_main_dataset(config)
    source_columns = list(input_df.columns[:8])
    if len(source_columns) != 8:
        raise ValueError("The input workbook must provide at least 8 source feature columns.")

    available_models = {
        "BNN-Auto",
        "S-BNN",
        "FFNN",
        "Random Forest",
        "Gaussian Process",
        "Linear LS",
    }
    if config.models.strip().lower() == "all":
        selected_models = available_models
    else:
        selected_models = {
            name.strip() for name in config.models.split(",") if name.strip()
        }
        unknown_models = selected_models - available_models
        if unknown_models:
            raise ValueError(
                "Unknown model name(s): " + ", ".join(sorted(unknown_models))
            )

    model_meta = {
        "BNN-Auto": {
            "beta": config.bnn_auto_beta,
            "use_kl": config.bnn_auto_use_kl,
            "source": "arch12 BNN-Auto",
        },
        "S-BNN": {
            "beta": None,
            "use_kl": config.bnn_auto_use_kl,
            "source": "April Standard BNN",
        },
        "FFNN": {
            "beta": None,
            "use_kl": 0,
            "source": "standard feedforward neural network",
        },
        "Random Forest": {
            "beta": None,
            "use_kl": 0,
            "source": "sklearn RandomForestRegressor",
        },
        "Gaussian Process": {
            "beta": None,
            "use_kl": 0,
            "source": "sklearn GaussianProcessRegressor",
        },
        "Linear LS": {
            "beta": None,
            "use_kl": 0,
            "source": "ordinary least-squares regression on the same engineered features",
        },
    }

    fold_records = []
    oof_log = [[] for _ in range(len(input_df))]
    oof_prob = [[] for _ in range(len(input_df))]

    if "BNN-Auto" not in selected_models:
        raise ValueError("BNN-Auto must be included so that BNN_output2.xlsx can be written.")

    repeat_stop = config.repeat_start - 1 + config.n_repeats
    for repeat_idx in range(config.repeat_start - 1, repeat_stop):
        split_seed = config.seed + repeat_idx
        kf = KFold(
            n_splits=config.n_splits,
            shuffle=True,
            random_state=split_seed,
        )

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(features), start=1):
            fold_seed = 100000 + split_seed * 100 + fold_idx
            x_outer_train_raw = features[train_idx]
            x_outer_val_raw = features[val_idx]
            y_outer_train_prob = targets_prob[train_idx]
            y_outer_val_prob = targets_prob[val_idx]
            y_outer_train_log = targets_log[train_idx]
            y_outer_val_log = targets_log[val_idx]

            inner_train_idx, inner_val_idx = train_test_split(
                np.arange(len(train_idx)),
                test_size=config.inner_val_fraction,
                shuffle=True,
                random_state=fold_seed,
            )
            nn_scaler = StandardScaler()
            x_inner_train = nn_scaler.fit_transform(
                x_outer_train_raw[inner_train_idx]
            ).astype(np.float32)
            x_inner_val = nn_scaler.transform(
                x_outer_train_raw[inner_val_idx]
            ).astype(np.float32)
            x_outer_val_nn = nn_scaler.transform(x_outer_val_raw).astype(np.float32)
            y_inner_train_prob = y_outer_train_prob[inner_train_idx]
            y_inner_train_log = y_outer_train_log[inner_train_idx]
            y_inner_val_log = y_outer_train_log[inner_val_idx]
            y_inner_val_prob = y_outer_train_prob[inner_val_idx]

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_outer_train_raw).astype(np.float32)
            x_val = scaler.transform(x_outer_val_raw).astype(np.float32)
            fold_predictions = {}
            fold_diagnostics = {}

            if "BNN-Auto" in selected_models:
                set_seed(fold_seed)
                model_auto, diagnostics = train_current_bnn_auto_model(
                    x_inner_train,
                    y_inner_train_log,
                    x_inner_val,
                    y_inner_val_prob,
                    y_inner_val_log,
                    config,
                    device,
                )
                pred_log, pred_prob, pred_std = predict_mc(
                    model_auto,
                    torch.tensor(x_outer_val_nn, dtype=torch.float32).to(device),
                    config.mc_samples_eval,
                )
                fold_predictions["BNN-Auto"] = (
                    pred_log.cpu().numpy(),
                    pred_prob.cpu().numpy(),
                    float(pred_std.mean().cpu().item()),
                )
                fold_diagnostics["BNN-Auto"] = diagnostics
                for original_idx, log_value, prob_value in zip(
                    val_idx,
                    fold_predictions["BNN-Auto"][0],
                    fold_predictions["BNN-Auto"][1],
                ):
                    oof_log[int(original_idx)].append(float(log_value))
                    oof_prob[int(original_idx)].append(float(prob_value))

            if "S-BNN" in selected_models:
                set_seed(fold_seed + 1)
                model_sbnn, diagnostics = train_structural_sbnn_model(
                    x_inner_train,
                    y_inner_train_prob,
                    x_inner_val,
                    config,
                    device,
                )
                pred_log, pred_prob, pred_std = predict_mc_prob(
                    model_sbnn,
                    torch.tensor(x_outer_val_nn, dtype=torch.float32).to(device),
                    config.mc_samples_eval,
                )
                fold_predictions["S-BNN"] = (
                    pred_log.cpu().numpy(),
                    pred_prob.cpu().numpy(),
                    float(pred_std.mean().cpu().item()),
                )
                fold_diagnostics["S-BNN"] = diagnostics

            if "FFNN" in selected_models:
                set_seed(fold_seed + 2)
                pred_log, pred_prob, pred_std = train_ffnn(
                    x_train,
                    y_outer_train_prob,
                    x_val,
                    config,
                    device,
                )
                fold_predictions["FFNN"] = (pred_log, pred_prob, pred_std)

            if "Random Forest" in selected_models:
                pred_log, pred_prob = fit_random_forest(
                    x_outer_train_raw,
                    y_outer_train_prob,
                    x_outer_val_raw,
                    config,
                )
                fold_predictions["Random Forest"] = (pred_log, pred_prob, 0.0)

            if "Gaussian Process" in selected_models:
                pred_log, pred_prob, pred_std = fit_gaussian_process(
                    x_train,
                    y_outer_train_prob,
                    x_val,
                    config,
                )
                fold_predictions["Gaussian Process"] = (pred_log, pred_prob, pred_std)

            if "Linear LS" in selected_models:
                pred_log, pred_prob = fit_linear_ls(
                    x_train,
                    y_outer_train_prob,
                    x_val,
                )
                fold_predictions["Linear LS"] = (pred_log, pred_prob, 0.0)

            metrics_snapshot = []
            for model_name, (pred_log, pred_prob, pred_std) in fold_predictions.items():
                metrics = compute_metrics_logprob(
                    pred_log,
                    pred_prob,
                    y_outer_val_log,
                    y_outer_val_prob,
                )
                diagnostics = fold_diagnostics.get(model_name, {})
                pretrain_diagnostics = diagnostics.get("pretrain", {})
                bnn_diagnostics = diagnostics.get("bnn", {})
                fold_records.append(
                    {
                        "repeat": repeat_idx + 1,
                        "split_seed": split_seed,
                        "fold": fold_idx,
                        "model": model_name,
                        "rmse_prob": metrics["rmse_prob"],
                        "rmse_log": metrics["rmse_log"],
                        "mean_pred_std": float(pred_std),
                        "beta": model_meta[model_name]["beta"],
                        "use_kl": model_meta[model_name]["use_kl"],
                        "pretrain_epochs_ran": pretrain_diagnostics.get("epochs_ran"),
                        "pretrain_best_epoch": pretrain_diagnostics.get("best_epoch"),
                        "pretrain_best_inner_rmse_log": pretrain_diagnostics.get(
                            "best_inner_rmse_log"
                        ),
                        "bnn_epochs_ran": bnn_diagnostics.get("epochs_ran"),
                        "bnn_best_epoch": bnn_diagnostics.get("best_epoch"),
                        "bnn_best_inner_rmse_log": bnn_diagnostics.get(
                            "best_inner_rmse_log"
                        ),
                        "bnn_final_lr": bnn_diagnostics.get("final_lr"),
                    }
                )
                metrics_snapshot.append(f"{model_name}={metrics['rmse_log']:.6f}")

            print(
                f"repeat {repeat_idx + 1}, "
                f"fold {fold_idx}/{config.n_splits}: "
                + ", ".join(metrics_snapshot),
                flush=True,
            )

    fold_df = pd.DataFrame(fold_records)
    summary_df = (
        fold_df.groupby("model", as_index=False)
        .agg(
            mean_val_rmse_prob=("rmse_prob", "mean"),
            std_val_rmse_prob=("rmse_prob", lambda s: float(np.std(s, ddof=0))),
            mean_val_rmse_log=("rmse_log", "mean"),
            std_val_rmse_log=("rmse_log", lambda s: float(np.std(s, ddof=0))),
            mean_pred_std=("mean_pred_std", "mean"),
        )
        .sort_values("mean_val_rmse_log", ascending=True)
        .reset_index(drop=True)
    )

    summary_df["beta"] = summary_df["model"].map(lambda x: model_meta[x]["beta"])
    summary_df["use_kl"] = summary_df["model"].map(lambda x: model_meta[x]["use_kl"])
    summary_df["source"] = summary_df["model"].map(lambda x: model_meta[x]["source"])

    missing_oof = [idx for idx, values in enumerate(oof_log) if not values]
    if missing_oof:
        raise RuntimeError(
            "Some input rows did not receive an out-of-fold BNN-Auto prediction: "
            + ", ".join(map(str, missing_oof[:10]))
        )

    prediction_df = input_df.loc[:, source_columns].copy()
    prediction_df["pred_log"] = [float(np.mean(values)) for values in oof_log]
    prediction_df["pred_prob"] = [float(np.mean(values)) for values in oof_prob]
    return fold_df, summary_df, prediction_df


def plot_log_only(summary_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = summary_df.sort_values("mean_val_rmse_log", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(plot_df))
    display_map = {
        "BNN-Auto": "BNN-Auto",
        "S-BNN": "S-BNN",
        "Random Forest": "RF",
        "Gaussian Process": "GP",
        "FFNN": "SFNN",
        "Linear LS": "L-LSR",
    }
    y_labels = [display_map.get(name, name) for name in plot_df["model"]]

    fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=220)
    ax.barh(
        y_pos,
        plot_df["mean_val_rmse_log"],
        xerr=plot_df["std_val_rmse_log"].fillna(0.0),
        color="#e76f51",
        alpha=0.92,
        capsize=3,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"RMS deviation $\sigma_{\mathrm{post}}$ of $\log_{10} P_{\alpha}$")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "log_only_comparison.png", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> ExperimentConfig:
    release_root = Path(__file__).resolve().parents[1]
    default_data_path = release_root / "data" / "BNN_input2.xlsx"
    default_output_dir = release_root / "outputs" / "current_model_compare_run"

    parser = argparse.ArgumentParser(
        description="Compare the current BNN-Auto model with S-BNN and other machine-learning baselines."
    )
    parser.add_argument("--data-path", default=str(default_data_path))
    parser.add_argument("--sheet-name", default="Sheet1")
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--n-repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=180)
    parser.add_argument("--pretrain-epochs", type=int, default=140)
    parser.add_argument("--bnn-epochs", type=int, default=260)
    parser.add_argument("--mc-samples-probe", type=int, default=30)
    parser.add_argument("--mc-samples-eval", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pretrain-lr", type=float, default=1e-4)
    parser.add_argument("--bnn-lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "linear_to_floor", "plateau_validation"],
        default="linear_to_floor",
        help="Use a training-loss plateau scheduler, linear decay, or validation-RMS plateau decay.",
    )
    parser.add_argument("--lr-floor", type=float, default=5e-5)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-cooldown", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--kl-weight", type=float, default=3e-8)
    parser.add_argument("--kl-anneal-fraction", type=float, default=0.6)
    parser.add_argument("--moped-delta", type=float, default=0.05)
    parser.add_argument("--inner-val-fraction", type=float, default=0.15)
    parser.add_argument("--probe-every", type=int, default=8)
    parser.add_argument("--early-stop-patience", type=int, default=64)
    parser.add_argument("--scheduler-patience", type=int, default=20)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument(
        "--loss-kind",
        choices=["smooth_l1", "mse"],
        default="mse",
    )
    parser.add_argument("--smooth-l1-beta", type=float, default=0.05)
    parser.add_argument("--bnn-auto-beta", type=float, default=0.2)
    parser.add_argument("--bnn-auto-use-kl", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model names, or 'all'.",
    )
    parser.add_argument("--repeat-start", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--rf-estimators", type=int, default=600)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=2)
    parser.add_argument("--gp-restarts-optimizer", type=int, default=0)
    args = parser.parse_args()

    return ExperimentConfig(
        data_path=args.data_path,
        sheet_name=args.sheet_name,
        output_dir=args.output_dir,
        device=args.device,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        seed=args.seed,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        pretrain_epochs=args.pretrain_epochs,
        bnn_epochs=args.bnn_epochs,
        mc_samples_probe=args.mc_samples_probe,
        mc_samples_eval=args.mc_samples_eval,
        lr=args.lr,
        pretrain_lr=args.pretrain_lr,
        bnn_lr=args.bnn_lr,
        lr_schedule=args.lr_schedule,
        lr_floor=args.lr_floor,
        plateau_factor=args.plateau_factor,
        plateau_cooldown=args.plateau_cooldown,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        kl_weight=args.kl_weight,
        kl_anneal_fraction=args.kl_anneal_fraction,
        moped_delta=args.moped_delta,
        inner_val_fraction=args.inner_val_fraction,
        probe_every=args.probe_every,
        early_stop_patience=args.early_stop_patience,
        scheduler_patience=args.scheduler_patience,
        min_lr=args.min_lr,
        loss_kind=args.loss_kind,
        smooth_l1_beta=args.smooth_l1_beta,
        bnn_auto_beta=args.bnn_auto_beta,
        bnn_auto_use_kl=args.bnn_auto_use_kl,
        models=args.models,
        repeat_start=args.repeat_start,
        torch_threads=args.torch_threads,
        rf_estimators=args.rf_estimators,
        rf_min_samples_leaf=args.rf_min_samples_leaf,
        gp_restarts_optimizer=args.gp_restarts_optimizer,
    )


def main() -> None:
    config = parse_args()
    configure_torch_threads(config.torch_threads)
    set_seed(config.seed)
    output_dir = ensure_output_dir(Path(config.output_dir))
    _fold_df, summary_df, prediction_df = run_experiment(config)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    prediction_df.to_excel(
        output_dir / "BNN_output2.xlsx",
        index=False,
        sheet_name="predictions",
    )
    plot_log_only(summary_df, output_dir)
    print(summary_df.to_csv(index=False), flush=True)


if __name__ == "__main__":
    main()
