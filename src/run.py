# Author: ZengWenquan
# https://github.com/chaosbull
# License: CC BY 4.0

"""Train, score the test split, write figures and tables into results/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from config import (
    ADV_ETA, ADV_EPS, ADV_WARMUP_EPOCHS, BATCH_SIZE, CALIBRATION_TARGET, CALIBRATION_WEIGHT,
    EARLY_STOP_METRIC, EMBED_COLS, ENTROPY_GAMMA, GCN_DIM, GCN_LAYERS, GRAPH_ATTN_DIM,
    GRAPH_HEADS, GRAPH_TOPK, LEARNABLE_SIGMA_SCALE, LOW_RANK_CROSS, MDN_COMPONENTS,
    MDN_DROPOUT, MDN_HIDDEN, MODEL_NAME, MSE_LOSS_WEIGHT, NODE_EMBED_DIM, POINT_HIDDEN,
    RESULTS_DIR, SEED, USE_ADAPTIVE_CALIBRATOR, USE_FUSION, USE_MULTITASK_WEIGHTS,
    USE_POINT_HEAD, USE_PREDICTIVE_FUSION,
)
from calibration_utils import (
    apply_sigma_scale,
    enrich_metrics,
    export_calibration_artifacts,
    find_sigma_scale,
)
from data_preprocessing import load_raw_dataframe, prepare_data
from model import DRGMDN
from train import (
    TrainConfig, _collect_loader, evaluate_drg_mdn, load_dbian_baseline, save_final_metrics,
    set_seed, train_baselines, train_drg_mdn,
)
from visualize import generate_all_figures


def extract_interpretability(model: DRGMDN, loader, device) -> tuple:
    model.eval()
    adj_list, pi_list, mu_list, sigma_list = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            cat = batch["cat"].to(device)
            cont = batch["cont"].to(device)
            z = model.build_z(cat, cont)
            out = model.forward_from_z(z)
            adj_list.append(out["adj"].cpu().numpy())
            pi_list.append(out["pi"].cpu().numpy())
            mu_list.append(out["mu"].cpu().numpy())
            sigma_list.append(out["sigma"].cpu().numpy())

    return (
        np.concatenate(adj_list, axis=0),
        np.concatenate(pi_list, axis=0),
        np.concatenate(mu_list, axis=0),
        np.concatenate(sigma_list, axis=0),
    )


def main():
    set_seed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Results: {RESULTS_DIR}")

    loaders, artifacts, _, test_meta = prepare_data(BATCH_SIZE)
    df = load_raw_dataframe()
    cardinalities = {col: len(artifacts.cat_maps[col]) for col in EMBED_COLS}

    print("\n--- Training baselines ---")
    baseline_metrics = train_baselines(loaders)
    for name, m in baseline_metrics.items():
        print(f"  {name}: RMSE={m['rmse']:.4f}  R2={m['r2']:.4f}")

    dbian_metrics = load_dbian_baseline(cardinalities, loaders["test"], device)
    if dbian_metrics:
        print(f"  DBIAN (checkpoint): RMSE={dbian_metrics['rmse']:.4f}  R2={dbian_metrics['r2']:.4f}")
        baseline_metrics["DBIAN (prior)"] = dbian_metrics

    print(f"\n--- Training {MODEL_NAME} ---")
    model = DRGMDN(cardinalities).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"  Arch: point_head={USE_POINT_HEAD} adaptive_cal={USE_ADAPTIVE_CALIBRATOR} "
        f"fusion_gate={USE_PREDICTIVE_FUSION} auto_task_weights={USE_MULTITASK_WEIGHTS} "
        f"stop={EARLY_STOP_METRIC} seed={SEED}"
    )

    train_cfg = TrainConfig(seed=SEED, early_stop_metric=EARLY_STOP_METRIC)
    model, history, train_summary = train_drg_mdn(model, loaders, device, train_cfg)

    print("\n--- Test evaluation ---")
    test_metrics = enrich_metrics(evaluate_drg_mdn(model, loaders["test"], device))
    val_metrics = enrich_metrics(evaluate_drg_mdn(model, loaders["val"], device))

    # if the 90% interval is still short, stretch sigma using the validation set
    sigma_scale = 1.0
    if test_metrics["coverage_90"] < CALIBRATION_TARGET - 0.02:
        s90, _ = find_sigma_scale(
            val_metrics["y_true"], val_metrics["y_pred"], val_metrics["y_std"],
            target_alpha=CALIBRATION_TARGET,
        )
        learned = train_summary.get("learned_sigma_scale", 1.0)
        sigma_scale = float(max(s90 / learned, 1.0)) if learned > 0 else float(s90)
        calibrated_test = apply_sigma_scale(test_metrics, sigma_scale)
    else:
        calibrated_test = dict(test_metrics)

    for k in ("nll", "rmse", "mae", "r2", "mixture_entropy"):
        calibrated_test[k] = test_metrics.get(k)

    print(
        f"  {MODEL_NAME}: RMSE={test_metrics['rmse']:.4f}  MAE={test_metrics['mae']:.4f}  "
        f"R2={test_metrics['r2']:.4f}  NLL={test_metrics['nll']:.4f}  "
        f"Cov90={test_metrics['coverage_90']:.1%}  ECE={test_metrics['ece']:.4f}"
    )
    if sigma_scale != 1.0:
        print(
            f"  Post-hoc refine (x{sigma_scale:.3f}): "
            f"Cov90={calibrated_test['coverage_90']:.1%}  ECE={calibrated_test['ece']:.4f}"
        )

    gbr_rmse = baseline_metrics["Gradient Boosting"]["rmse"]
    imp = (gbr_rmse - test_metrics["rmse"]) / gbr_rmse * 100
    print(f"  RMSE vs GBR: {imp:+.1f}%")

    adj_matrix, pi_matrix, mu_matrix, sigma_matrix = extract_interpretability(
        model, loaders["test"], device
    )

    print("\n--- Figures ---")
    generate_all_figures(
        history=history,
        test_metrics=test_metrics,
        baseline_metrics={k: {m: v for m, v in d.items() if m != "y_pred"} for k, d in baseline_metrics.items()},
        df=df,
        adj_matrix=adj_matrix,
        pi_matrix=pi_matrix,
        mu_matrix=mu_matrix,
        test_meta=test_meta,
        calibrated_metrics=calibrated_test,
    )

    test_pack = _collect_loader(loaders["test"])
    y_test = test_pack["y"]
    baselines_for_cal = {}
    for name, d in baseline_metrics.items():
        if "y_pred" in d:
            pred = d["y_pred"]
            resid = y_test - pred
            var = float(resid.var()) if resid.var() > 0 else 1.0
            baselines_for_cal[name] = {
                "y_true": y_test, "y_pred": pred,
                "y_std": np.full_like(pred, np.sqrt(var)),
                "nll": d.get("nll"), "coverage_90": d.get("coverage_90"),
            }
    export_calibration_artifacts(test_metrics, calibrated_test, baselines_for_cal, sigma_scale)

    np.savez(
        RESULTS_DIR / "drg_mdn_test_predictions.npz",
        y_true=test_metrics["y_true"],
        y_pred=test_metrics["y_pred"],
        y_std_raw=test_metrics["y_std"],
        y_std_calibrated=calibrated_test["y_std"],
        sigma_scale=sigma_scale,
        pi=pi_matrix,
        mu=mu_matrix,
        sigma=sigma_matrix,
    )

    model_config = {
        "model_name": MODEL_NAME,
        "graph_attn_dim": GRAPH_ATTN_DIM,
        "graph_heads": GRAPH_HEADS,
        "node_embed_dim": NODE_EMBED_DIM,
        "gcn_dim": GCN_DIM,
        "mdn_hidden": MDN_HIDDEN,
        "point_hidden": POINT_HIDDEN,
        "mdn_dropout": MDN_DROPOUT,
        "use_fusion": USE_FUSION,
        "use_point_head": USE_POINT_HEAD,
        "use_adaptive_calibrator": USE_ADAPTIVE_CALIBRATOR,
        "use_predictive_fusion": USE_PREDICTIVE_FUSION,
        "use_multitask_weights": USE_MULTITASK_WEIGHTS,
        "graph_topk": GRAPH_TOPK,
        "gcn_layers": GCN_LAYERS,
        "low_rank_cross": LOW_RANK_CROSS,
        "mdn_components": MDN_COMPONENTS,
        "entropy_gamma": ENTROPY_GAMMA,
        "adv_eta": ADV_ETA,
        "adv_eps": ADV_EPS,
        "adv_warmup_epochs": ADV_WARMUP_EPOCHS,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "early_stop_metric": EARLY_STOP_METRIC,
    }
    with open(RESULTS_DIR / "sigma_scale.json", "w", encoding="utf-8") as f:
        json.dump({
            "learned_sigma_scale": train_summary.get("learned_sigma_scale", 1.0),
            "posthoc_sigma_scale": sigma_scale,
        }, f, indent=2)

    perf_rows = []
    for name, m in baseline_metrics.items():
        if name == "DBIAN (prior)":
            perf_rows.append({
                "Model": name, "RMSE": m["rmse"], "MAE": m["mae"], "R2": m["r2"],
                "NLL": None, "Coverage@90%": None,
            })
        else:
            perf_rows.append({
                "Model": name, "RMSE": m["rmse"], "MAE": m["mae"], "R2": m["r2"],
                "NLL": m.get("nll"), "Coverage@90%": m.get("coverage_90"),
            })
    perf_rows.insert(0, {
        "Model": MODEL_NAME, "RMSE": test_metrics["rmse"], "MAE": test_metrics["mae"],
        "R2": test_metrics["r2"], "NLL": test_metrics["nll"],
        "ECE": test_metrics["ece"],
        "Coverage@90%": calibrated_test["coverage_90"],
    })
    pd.DataFrame(perf_rows).to_csv(RESULTS_DIR / "table_regression_comparison.csv", index=False)

    save_final_metrics(
        {**test_metrics, "coverage_90_calibrated": calibrated_test["coverage_90"],
         "ece_raw": test_metrics["ece"], "ece_calibrated": calibrated_test["ece"],
         "sigma_scale": sigma_scale},
        baseline_metrics,
        model_config,
        train_summary,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
