# Author: ZengWenquan
# https://github.com/chaosbull
# License: Apache-2.0

"""Train DRG-MDN-U, plus the linear and gradient-boosting baselines."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import config as cfg
from calibration_utils import enrich_metrics, expected_calibration_error, coverage_profile
from model import (
    DRGMDN, calibration_coverage_loss, mdn_entropy_penalty, mdn_nll, vat_loss,
)

MSE_AUX = getattr(cfg, "MSE_AUX_WEIGHT", 0.0)
MSE_POINT = getattr(cfg, "MSE_POINT_WEIGHT", 0.0)


@dataclass
class TrainConfig:
    seed: int = cfg.SEED
    lr: float = cfg.LR
    weight_decay: float = cfg.WEIGHT_DECAY
    max_epochs: int = cfg.MAX_EPOCHS
    early_stop_patience: int = cfg.EARLY_STOP_PATIENCE
    early_stop_metric: str = cfg.EARLY_STOP_METRIC
    entropy_gamma: float = cfg.ENTROPY_GAMMA
    mse_weight: float = cfg.MSE_LOSS_WEIGHT
    calib_weight: float = cfg.CALIBRATION_WEIGHT
    calib_target: float = cfg.CALIBRATION_TARGET
    adv_eta: float = cfg.ADV_ETA
    adv_eps: float = cfg.ADV_EPS
    adv_warmup_epochs: int = cfg.ADV_WARMUP_EPOCHS
    grad_clip: float = cfg.GRAD_CLIP
    verbose: bool = True
    save_checkpoint: bool = True


def set_seed(seed: int = cfg.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collect_loader(loader) -> Dict[str, np.ndarray]:
    cats, conts, ys = [], [], []
    for batch in loader:
        cats.append(batch["cat"].numpy())
        conts.append(batch["cont"].numpy())
        ys.append(batch["y"].numpy())
    return {"cat": np.vstack(cats), "cont": np.vstack(conts), "y": np.concatenate(ys)}


def evaluate_drg_mdn(model: DRGMDN, loader, device: torch.device) -> Dict:
    model.eval()
    ys, preds, stds, nlls, entropies = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            cat = batch["cat"].to(device)
            cont = batch["cont"].to(device)
            y = batch["y"].to(device)
            out = model.forward_from_z(model.build_z(cat, cont))
            pi, mu, sigma = out["pi"], out["mu"], out["sigma"]
            pred = out["y_pred"]
            var = model.mixture_variance(pi, mu, sigma)
            nll = mdn_nll(y, pi, mu, sigma)
            ent = model.mixture_entropy(pi)

            ys.append(y.cpu().numpy())
            preds.append(pred.cpu().numpy())
            stds.append(torch.sqrt(var).cpu().numpy())
            nlls.append(nll.cpu().numpy())
            entropies.append(ent.cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    y_std = np.concatenate(stds)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    nll = float(np.mean(np.concatenate(nlls)))
    mix_entropy = float(np.mean(np.concatenate(entropies)))

    lower = y_pred - 1.645 * y_std
    upper = y_pred + 1.645 * y_std
    coverage_90 = float(np.mean((y_true >= lower) & (y_true <= upper)))

    return {
        "rmse": rmse, "mae": mae, "r2": r2, "nll": nll,
        "coverage_90": coverage_90, "mixture_entropy": mix_entropy,
        "y_true": y_true, "y_pred": y_pred, "y_std": y_std,
    }


def _val_score(val_m: Dict, metric: str) -> float:
    prof = coverage_profile(val_m["y_true"], val_m["y_pred"], val_m["y_std"])
    ece = expected_calibration_error(prof)
    cov_gap = abs(val_m["coverage_90"] - cfg.CALIBRATION_TARGET)

    if metric == "rmse":
        return val_m["rmse"]
    if metric == "nll":
        return val_m["nll"]
    if metric == "ece":
        return ece
    if metric == "composite":
        return val_m["nll"] + 1.2 * cov_gap + 0.6 * ece + 0.08 * val_m["rmse"]
    if metric == "hybrid":
        return 0.72 * val_m["rmse"] + 0.18 * val_m["nll"] + 0.06 * ece + 0.04 * cov_gap
    if metric == "universal":
        return (
            0.20 * val_m["nll"]
            + 0.18 * ece
            + 0.17 * cov_gap
            + 0.45 * val_m["rmse"]
        )
    raise ValueError(f"Unknown early-stop metric: {metric}")


def _huber_mse(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.30) -> torch.Tensor:
    return torch.nn.functional.huber_loss(pred, target, delta=delta)


def _compute_loss(model: DRGMDN, cat, cont, y, tc: TrainConfig) -> torch.Tensor:
    z = model.build_z(cat, cont)
    out = model.forward_from_z(z)
    pi, mu, sigma = out["pi"], out["mu"], out["sigma"]
    y_pred, y_point, mix_mean = out["y_pred"], out["y_point"], out["mix_mean"]

    loss_nll = mdn_nll(y, pi, mu, sigma).mean()
    loss_mse = _huber_mse(mix_mean, y)
    loss_fused = _huber_mse(y_pred, y)
    loss_point = _huber_mse(y_point, y) if y_point is not None else loss_mse
    loss_cal = calibration_coverage_loss(y, y_pred, pi, mu, sigma, tc.calib_target)

    if model.task_weights is not None and cfg.USE_MULTITASK_WEIGHTS:
        loss = (
            model.task_weights.loss(0, loss_nll)
            + model.task_weights.loss(1, loss_mse)
            + model.task_weights.loss(2, loss_cal)
            + MSE_AUX * loss_mse
            + MSE_POINT * loss_point
            + tc.entropy_gamma * mdn_entropy_penalty(pi)
        )
    else:
        loss = (
            loss_nll
            + tc.mse_weight * loss_mse
            + MSE_POINT * loss_point
            + tc.calib_weight * loss_cal
            + MSE_AUX * loss_fused
            + tc.entropy_gamma * mdn_entropy_penalty(pi)
        )
    return loss


def train_drg_mdn(
    model: DRGMDN,
    loaders: Dict,
    device: torch.device,
    train_cfg: Optional[TrainConfig] = None,
) -> Tuple[DRGMDN, Dict, Dict]:
    tc = train_cfg or TrainConfig()
    set_seed(tc.seed)

    optimizer = AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=tc.max_epochs, eta_min=1e-5)

    history = {
        "train_loss": [], "val_loss": [], "val_nll": [], "val_rmse": [],
        "val_ece": [], "val_coverage_90": [], "val_score": [],
    }
    best_val = float("inf")
    best_state = None
    patience = 0
    swa_state = None
    swa_n = 0
    swa_start = getattr(cfg, "SWA_START_EPOCH", 0)
    swa_enabled = getattr(cfg, "SWA_ENABLED", False)

    for epoch in range(1, tc.max_epochs + 1):
        model.train()
        epoch_losses = []

        for batch in loaders["train"]:
            cat = batch["cat"].to(device)
            cont = batch["cont"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            loss = _compute_loss(model, cat, cont, y, tc)

            if tc.adv_eta > 0 and epoch > tc.adv_warmup_epochs:
                loss = loss + tc.adv_eta * vat_loss(model, cat, cont, y, tc.adv_eps)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        if swa_enabled and epoch >= swa_start:
            swa_n += 1
            cur = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if swa_state is None:
                swa_state = cur
            else:
                for k in swa_state:
                    if not torch.is_floating_point(swa_state[k]):
                        swa_state[k] = cur[k]
                    else:
                        swa_state[k] += (cur[k] - swa_state[k]) / swa_n

        val_m = enrich_metrics(evaluate_drg_mdn(model, loaders["val"], device))
        score = _val_score(val_m, tc.early_stop_metric)

        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_loss"].append(val_m["nll"])
        history["val_nll"].append(val_m["nll"])
        history["val_rmse"].append(val_m["rmse"])
        history["val_ece"].append(val_m["ece"])
        history["val_coverage_90"].append(val_m["coverage_90"])
        history["val_score"].append(score)

        if score < best_val:
            best_val = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if tc.verbose and (epoch % 20 == 0 or epoch == 1):
            cal_mean = 1.0
            if model.calibrator is not None:
                with torch.no_grad():
                    batch = next(iter(loaders["val"]))
                    out = model.forward_from_z(model.build_z(
                        batch["cat"].to(device), batch["cont"].to(device)
                    ))
                    cal_mean = float(out["cal_scale"].mean().item())
            tw = ""
            if model.task_weights is not None:
                tw = " w=[" + ",".join(f"{v:.2f}" for v in model.task_weights.log_vars.detach().cpu().tolist()) + "]"
            print(
                f"Epoch {epoch:3d} | train={history['train_loss'][-1]:.4f} "
                f"| nll={val_m['nll']:.4f} rmse={val_m['rmse']:.4f} "
                f"cov90={val_m['coverage_90']:.1%} ece={val_m['ece']:.4f} "
                f"cal={cal_mean:.2f}{tw} score={score:.4f}",
                flush=True,
            )

        if patience >= tc.early_stop_patience:
            if tc.verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_best = enrich_metrics(evaluate_drg_mdn(model, loaders["val"], device))
    best_rmse = val_best["rmse"]
    final_state = best_state

    if swa_enabled and swa_state is not None:
        # keep the average only when it actually beats the best checkpoint on val RMSE
        model.load_state_dict(swa_state)
        val_swa = enrich_metrics(evaluate_drg_mdn(model, loaders["val"], device))
        if val_swa["rmse"] < best_rmse:
            val_best = val_swa
            final_state = swa_state
        elif best_state is not None:
            model.load_state_dict(best_state)

    if final_state is not None and final_state is not best_state:
        model.load_state_dict(final_state)
    elif final_state is None and swa_state is not None:
        model.load_state_dict(swa_state)
        val_best = enrich_metrics(evaluate_drg_mdn(model, loaders["val"], device))
    summary = {
        "best_val_rmse": val_best["rmse"],
        "best_val_nll": val_best["nll"],
        "best_val_ece": val_best["ece"],
        "best_val_coverage_90": val_best["coverage_90"],
        "best_val_score": _val_score(val_best, tc.early_stop_metric),
        "epochs_trained": len(history["train_loss"]),
        "early_stop_metric": tc.early_stop_metric,
        "swa_epochs": swa_n if swa_enabled else 0,
        "train_config": asdict(tc),
    }
    if model.task_weights is not None:
        summary["task_log_vars"] = model.task_weights.log_vars.detach().cpu().tolist()

    if tc.save_checkpoint:
        cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), cfg.RESULTS_DIR / "drg_mdn_best.pt")
        with open(cfg.RESULTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    return model, history, summary


def train_baselines(loaders: Dict, seed: int = cfg.SEED) -> Dict[str, Dict]:
    train = _collect_loader(loaders["train"])
    test = _collect_loader(loaders["test"])

    def flat(d):
        return np.hstack([d["cat"].astype(float), d["cont"]])

    x_train, y_train = flat(train), train["y"]
    x_test, y_test = flat(test), test["y"]

    models = {
        "Linear Regression": LinearRegression(),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.9, random_state=seed
        ),
    }

    results = {}
    for name, m in models.items():
        m.fit(x_train, y_train)
        pred = m.predict(x_test)
        resid = y_test - pred
        var = float(resid.var()) if resid.var() > 0 else 1.0
        results[name] = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "mae": float(mean_absolute_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
            "nll": float(np.mean(0.5 * np.log(2 * np.pi * var) + resid ** 2 / (2 * var))),
            "coverage_90": float(np.mean(np.abs(resid) <= 1.645 * np.sqrt(var))),
            "y_pred": pred,
        }

    with open(cfg.RESULTS_DIR / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: {m: v for m, v in d.items() if m != "y_pred"} for k, d in results.items()}, f, indent=2)

    return results


def load_dbian_baseline(cardinalities: Dict, loader, device: torch.device) -> Dict | None:
    ckpt = cfg.DBIAN_CKPT
    if not ckpt.exists():
        return None
    from dbian_legacy import DBIAN
    model = DBIAN(cardinalities).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for batch in loader:
            cat = batch["cat"].to(device)
            cont = batch["cont"].to(device)
            pred = model.predict(cat, cont)
            ys.append(batch["y"].numpy())
            preds.append(pred.cpu().numpy())
    y_true, y_pred = np.concatenate(ys), np.concatenate(preds)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_final_metrics(test_metrics, baseline_metrics, model_config, train_summary) -> None:
    skip = ("y_true", "y_pred", "y_std", "coverage_profile")
    payload = {
        "drg_mdn_test": {k: v for k, v in test_metrics.items() if k not in skip},
        "drg_mdn_config": model_config,
        "train_summary": train_summary,
        "baselines": {k: {m: v for m, v in d.items() if m != "y_pred"} for k, d in baseline_metrics.items()},
    }
    with open(cfg.RESULTS_DIR / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
