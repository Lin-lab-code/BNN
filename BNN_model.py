#!/usr/bin/env python3
"""Public release script for the current BNN-Auto comparison benchmark.

This script compares the current BNN-Auto model against several machine
learning baselines on the `BNN_input2.xlsx` dataset.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import asdict, dataclass
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
from sklearn.model_selection import KFold
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
    seed: int = 42
    batch_size: int = 34
    max_epochs: int = 180
    mc_samples_probe: int = 10
    mc_samples_eval: int = 24
    lr: float = 8e-5
    weight_decay: float = 1e-6
    hidden_dim: int = 128
    latent_dim: int = 8
    expert_hidden: int = 40
    kl_weight: float = 5e-8
    moped_delta: float = 0.10
    probe_every: int = 8
    early_stop_patience: int = 32
    bnn_auto_beta: float = 0.0
    bnn_auto_use_kl: int = 1
    rf_estimators: int = 600
    rf_min_samples_leaf: int = 2
    gp_restarts_optimizer: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def ensure_output_dirs(base_dir: Path) -> dict[str, Path]:
    paths = {
        "base": base_dir,
        "code": base_dir / "code",
        "results": base_dir / "results",
        "figures": base_dir / "figures",
        "logs": base_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def snapshot_script(paths: dict[str, Path]) -> None:
    shutil.copy2(Path(__file__), paths["code"] / Path(__file__).name)


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


def build_groups(df: pd.DataFrame) -> np.ndarray:
    a = df["A"].to_numpy(dtype=np.float64)
    z = df["Z"].to_numpy(dtype=np.float64)
    n = a - z
    group = np.zeros(len(df), dtype=np.int64)
    is_even_even = ((z % 2) == 0) & ((n % 2) == 0)
    is_odd_odd = ((z % 2) == 1) & ((n % 2) == 1)
    group[is_even_even] = 0
    group[~is_even_even & ~is_odd_odd] = 1
    group[is_odd_odd] = 2
    return group


def validate_dataframe(df: pd.DataFrame) -> None:
    required_columns = {"A", "Z", "Q", "L", "logPa_exp"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing)
        )


def load_main_dataset(config: ExperimentConfig) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    groups = build_groups(df)
    targets_log = df["logPa_exp"].to_numpy(dtype=np.float32)
    targets_prob = np.power(10.0, targets_log).astype(np.float32)
    return df, features, targets_prob, targets_log, groups


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


class ExpertHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        half_dim = max(hidden_dim // 2, 4)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, half_dim),
            nn.ELU(),
            nn.Linear(half_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CurrentBNNAuto(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, expert_hidden: int):
        super().__init__()
        mid_dim = max(hidden_dim // 2, latent_dim * 2)
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
        self.shared = nn.Sequential(
            nn.Linear(latent_dim + input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ELU(),
        )
        self.expert_heads = nn.ModuleList([ExpertHead(mid_dim, expert_hidden) for _ in range(3)])

    def forward(self, x: torch.Tensor, group_idx: torch.Tensor):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        hidden = self.shared(torch.cat([latent, recon], dim=-1))
        pred_raw = torch.zeros(x.shape[0], device=x.device, dtype=hidden.dtype)
        for g, head in enumerate(self.expert_heads):
            mask = group_idx == g
            if mask.any():
                pred_raw[mask] = head(hidden[mask])
        pred_log = -F.softplus(pred_raw)
        pred_prob = torch.pow(pred_log.new_tensor(10.0), pred_log)
        return {"pred_log": pred_log, "pred_prob": pred_prob, "recon": recon}


class StructuralSBNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, expert_hidden: int):
        super().__init__()
        mid_dim = max(hidden_dim // 2, latent_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ELU(),
            nn.Linear(mid_dim, latent_dim),
        )
        self.shared = nn.Sequential(
            nn.Linear(latent_dim + input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ELU(),
        )
        self.expert_heads = nn.ModuleList([ExpertHead(mid_dim, expert_hidden) for _ in range(3)])

    def forward(self, x: torch.Tensor, group_idx: torch.Tensor):
        latent = self.encoder(x)
        hidden = self.shared(torch.cat([latent, x], dim=-1))
        pred_raw = torch.zeros(x.shape[0], device=x.device, dtype=hidden.dtype)
        for g, head in enumerate(self.expert_heads):
            mask = group_idx == g
            if mask.any():
                pred_raw[mask] = head(hidden[mask])
        pred_log = -F.softplus(pred_raw)
        pred_prob = torch.pow(pred_log.new_tensor(10.0), pred_log)
        return {"pred_log": pred_log, "pred_prob": pred_prob}


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


def make_bnn_expert_model(model: nn.Module, config: ExperimentConfig) -> nn.Module:
    params = {
        "prior_mu": 0.0,
        "prior_sigma": 0.1,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",
        "moped_enable": True,
        "moped_delta": config.moped_delta,
    }
    for head in model.expert_heads:
        dnn_to_bnn(head, params)
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


def predict_mc_group(model: nn.Module, x_tensor: torch.Tensor, g_tensor: torch.Tensor, n_samples: int):
    preds_log = []
    preds_prob = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(x_tensor, g_tensor)
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


def train_group_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train_prob: np.ndarray,
    y_train_log: np.ndarray,
    g_train: np.ndarray,
    x_val: np.ndarray,
    y_val_prob: np.ndarray,
    y_val_log: np.ndarray,
    g_val: np.ndarray,
    beta: float,
    use_kl: bool,
    use_recon_loss: bool,
    config: ExperimentConfig,
    device: torch.device,
):
    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train_prob, dtype=torch.float32),
        torch.tensor(y_train_log, dtype=torch.float32),
        torch.tensor(g_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=14)

    best_state = None
    best_probe = float("inf")
    stale = 0

    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, _batch_y_prob, batch_y_log, batch_g in train_loader:
            batch_x = batch_x.to(device)
            batch_y_log = batch_y_log.to(device)
            batch_g = batch_g.to(device)
            optimizer.zero_grad()
            out = model(batch_x, batch_g)
            loss_log = nn.SmoothL1Loss(beta=0.02)(out["pred_log"], batch_y_log)
            if use_recon_loss:
                recon_loss = nn.MSELoss()(out["recon"], batch_x)
                data_loss = (1.0 - beta) * loss_log + beta * recon_loss
            else:
                data_loss = loss_log
            if use_kl:
                kl_scale = min(1.0, float(epoch + 1) / max(1, int(config.max_epochs * 0.2)))
                kl_loss = safe_kl_loss(model, device) / len(train_dataset)
                loss = data_loss + kl_scale * config.kl_weight * kl_loss
            else:
                loss = data_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
        scheduler.step(epoch_loss / max(1, len(train_loader)))

        if (epoch + 1) % config.probe_every == 0 or epoch == config.max_epochs - 1:
            model.eval()
            x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
            g_val_tensor = torch.tensor(g_val, dtype=torch.long).to(device)
            pred_log, pred_prob, _pred_std = predict_mc_group(model, x_val_tensor, g_val_tensor, config.mc_samples_probe)
            metrics = compute_metrics_logprob(pred_log.cpu(), pred_prob.cpu(), y_val_log, y_val_prob)
            if metrics["rmse_log"] + 1e-6 < best_probe:
                best_probe = metrics["rmse_log"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += config.probe_every
            if stale >= config.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def train_current_bnn_auto_model(
    x_train,
    y_train_prob,
    y_train_log,
    g_train,
    x_val,
    y_val_prob,
    y_val_log,
    g_val,
    config,
    device,
):
    model = CurrentBNNAuto(x_train.shape[1], config.hidden_dim, config.latent_dim, config.expert_hidden).to(device)
    model = make_bnn_expert_model(model, config).to(device)
    return train_group_model(
        model,
        x_train,
        y_train_prob,
        y_train_log,
        g_train,
        x_val,
        y_val_prob,
        y_val_log,
        g_val,
        config.bnn_auto_beta,
        bool(int(config.bnn_auto_use_kl)),
        True,
        config,
        device,
    )


def train_structural_sbnn_model(
    x_train,
    y_train_prob,
    y_train_log,
    g_train,
    x_val,
    y_val_prob,
    y_val_log,
    g_val,
    config,
    device,
):
    model = StructuralSBNN(x_train.shape[1], config.hidden_dim, config.latent_dim, config.expert_hidden).to(device)
    model = make_bnn_expert_model(model, config).to(device)
    return train_group_model(
        model,
        x_train,
        y_train_prob,
        y_train_log,
        g_train,
        x_val,
        y_val_prob,
        y_val_log,
        g_val,
        0.0,
        bool(int(config.bnn_auto_use_kl)),
        False,
        config,
        device,
    )


def train_ffnn(x_train, y_train_prob, x_val, config, device):
    model = FFNN(input_dim=x_train.shape[1], hidden_dim=config.hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=60)
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train_prob, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)

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
    _df, features, targets_prob, targets_log, groups = load_main_dataset(config)

    model_meta = {
        "BNN-Auto": {
            "beta": config.bnn_auto_beta,
            "use_kl": config.bnn_auto_use_kl,
            "source": "latent+recon BNN-Auto with parity-aware expert heads",
        },
        "S-BNN": {
            "beta": 0.0,
            "use_kl": config.bnn_auto_use_kl,
            "source": "same encoder/shared/expert structure as BNN-Auto but without decoder/reconstruction branch",
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

    kf = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    fold_records = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(features), start=1):
        x_train_raw = features[train_idx]
        x_val_raw = features[val_idx]
        y_train_prob = targets_prob[train_idx]
        y_val_prob = targets_prob[val_idx]
        y_train_log = targets_log[train_idx]
        y_val_log = targets_log[val_idx]
        g_train = groups[train_idx]
        g_val = groups[val_idx]

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train_raw).astype(np.float32)
        x_val = scaler.transform(x_val_raw).astype(np.float32)

        fold_predictions = {}

        model_auto = train_current_bnn_auto_model(
            x_train,
            y_train_prob,
            y_train_log,
            g_train,
            x_val,
            y_val_prob,
            y_val_log,
            g_val,
            config,
            device,
        )
        pred_log, pred_prob, pred_std = predict_mc_group(
            model_auto,
            torch.tensor(x_val, dtype=torch.float32).to(device),
            torch.tensor(g_val, dtype=torch.long).to(device),
            config.mc_samples_eval,
        )
        fold_predictions["BNN-Auto"] = (
            pred_log.cpu().numpy(),
            pred_prob.cpu().numpy(),
            float(pred_std.mean().cpu().item()),
        )

        model_sbnn = train_structural_sbnn_model(
            x_train,
            y_train_prob,
            y_train_log,
            g_train,
            x_val,
            y_val_prob,
            y_val_log,
            g_val,
            config,
            device,
        )
        pred_log, pred_prob, pred_std = predict_mc_group(
            model_sbnn,
            torch.tensor(x_val, dtype=torch.float32).to(device),
            torch.tensor(g_val, dtype=torch.long).to(device),
            config.mc_samples_eval,
        )
        fold_predictions["S-BNN"] = (
            pred_log.cpu().numpy(),
            pred_prob.cpu().numpy(),
            float(pred_std.mean().cpu().item()),
        )

        pred_log, pred_prob, pred_std = train_ffnn(x_train, y_train_prob, x_val, config, device)
        fold_predictions["FFNN"] = (pred_log, pred_prob, pred_std)

        pred_log, pred_prob = fit_random_forest(x_train_raw, y_train_prob, x_val_raw, config)
        fold_predictions["Random Forest"] = (pred_log, pred_prob, 0.0)

        pred_log, pred_prob, pred_std = fit_gaussian_process(x_train, y_train_prob, x_val, config)
        fold_predictions["Gaussian Process"] = (pred_log, pred_prob, pred_std)

        pred_log, pred_prob = fit_linear_ls(x_train, y_train_prob, x_val)
        fold_predictions["Linear LS"] = (pred_log, pred_prob, 0.0)

        metrics_snapshot = []
        for model_name, (pred_log, pred_prob, pred_std) in fold_predictions.items():
            metrics = compute_metrics_logprob(pred_log, pred_prob, y_val_log, y_val_prob)
            fold_records.append(
                {
                    "fold": fold_idx,
                    "model": model_name,
                    "rmse_prob": metrics["rmse_prob"],
                    "rmse_log": metrics["rmse_log"],
                    "mean_pred_std": float(pred_std),
                    "beta": model_meta[model_name]["beta"],
                    "use_kl": model_meta[model_name]["use_kl"],
                }
            )
            metrics_snapshot.append(f"{model_name}={metrics['rmse_log']:.6f}")

        print(f"fold {fold_idx}/{config.n_splits}: " + ", ".join(metrics_snapshot), flush=True)

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

    metadata = {
        "release_note": "Public release version cleaned from the paper-revision experiment folder.",
        "device_request": config.device,
        "resolved_device": str(device),
        "data_path": config.data_path,
        "sheet_name": config.sheet_name,
        "n_nuclei": int(len(features)),
        "n_features": int(features.shape[1]),
        "bnn_auto_setting": {
            "beta": config.bnn_auto_beta,
            "use_kl": config.bnn_auto_use_kl,
        },
    }
    return fold_df, summary_df, metadata


def plot_log_only(summary_df: pd.DataFrame, figures_dir: Path) -> None:
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
    fig.savefig(figures_dir / "log_only_comparison_current.png", bbox_inches="tight")
    fig.savefig(figures_dir / "log_only_comparison_current.pdf", bbox_inches="tight")
    plt.close(fig)


def save_run_artifacts(paths: dict[str, Path], config: ExperimentConfig, metadata: dict) -> None:
    config_payload = asdict(config)
    config_payload["torch_version"] = torch.__version__
    config_payload["cuda_available"] = bool(torch.cuda.is_available())
    config_payload["gpu_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    with open(paths["results"] / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=True)
    with open(paths["results"] / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=True)


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=34)
    parser.add_argument("--max-epochs", type=int, default=180)
    parser.add_argument("--mc-samples-probe", type=int, default=10)
    parser.add_argument("--mc-samples-eval", type=int, default=24)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--expert-hidden", type=int, default=40)
    parser.add_argument("--kl-weight", type=float, default=5e-8)
    parser.add_argument("--moped-delta", type=float, default=0.10)
    parser.add_argument("--probe-every", type=int, default=8)
    parser.add_argument("--early-stop-patience", type=int, default=32)
    parser.add_argument("--bnn-auto-beta", type=float, default=0.0)
    parser.add_argument("--bnn-auto-use-kl", type=int, choices=[0, 1], default=1)
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
        seed=args.seed,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        mc_samples_probe=args.mc_samples_probe,
        mc_samples_eval=args.mc_samples_eval,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        expert_hidden=args.expert_hidden,
        kl_weight=args.kl_weight,
        moped_delta=args.moped_delta,
        probe_every=args.probe_every,
        early_stop_patience=args.early_stop_patience,
        bnn_auto_beta=args.bnn_auto_beta,
        bnn_auto_use_kl=args.bnn_auto_use_kl,
        rf_estimators=args.rf_estimators,
        rf_min_samples_leaf=args.rf_min_samples_leaf,
        gp_restarts_optimizer=args.gp_restarts_optimizer,
    )


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    paths = ensure_output_dirs(Path(config.output_dir))
    snapshot_script(paths)
    fold_df, summary_df, metadata = run_experiment(config)
    fold_df.to_csv(paths["results"] / "fold_metrics.csv", index=False)
    summary_df.to_csv(paths["results"] / "summary_metrics.csv", index=False)
    save_run_artifacts(paths, config, metadata)
    plot_log_only(summary_df, paths["figures"])
    print(summary_df.to_csv(index=False), flush=True)


if __name__ == "__main__":
    main()
