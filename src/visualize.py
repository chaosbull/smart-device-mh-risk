# Author: ZengWenquan
# https://github.com/chaosbull
# License: Apache-2.0

"""Plots written into results/ when you run the training script."""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy import stats

from config import (
    CMAP_DIV, CMAP_SEQ, CMAP_UNCERT, C_ACCENT, C_LIGHT, C_NEUTRAL, C_PRIMARY,
    C_QUATERNARY, C_SECONDARY, C_TERTIARY, FONT_FAMILY, FONT_SIZE, GRID, LABEL_SIZE,
    PALETTE, RESULTS_DIR, SAVE_DPI, SPINE, TEXT, TICK_SIZE, TITLE_SIZE,
)
from plot_style import BG, CARD
from calibration_utils import (
    coverage_profile,
    expected_calibration_error,
    maximum_calibration_error,
)

FEATURE_LABELS = (
    [f"Country-{i+1}" for i in range(8)]
    + [f"Acad-{i+1}" for i in range(8)]
    + [f"Plat-{i+1}" for i in range(8)]
    + [f"Purp-{i+1}" for i in range(8)]
    + ["Age", "Unlocks", "Study", "Activity", "Sleep", "Usage", "Gender", "Stress"]
    + ["DigSat", "RestRatio", "BehBal", "StrUsage"]
)

STRESS_ORDER = ["Low", "Medium", "High", "Very High"]


def apply_theme():
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": CARD,
        "axes.edgecolor": SPINE,
        "axes.labelcolor": TEXT,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.8,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "text.color": TEXT,
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZE,
        "grid.color": GRID,
        "grid.linestyle": "-",
        "grid.alpha": 0.55,
        "legend.frameon": True,
        "legend.edgecolor": GRID,
        "legend.facecolor": BG,
        "legend.fontsize": TICK_SIZE,
        "figure.dpi": 120,
        "savefig.dpi": SAVE_DPI,
        "savefig.facecolor": BG,
        "savefig.edgecolor": BG,
        "savefig.bbox": "tight",
    })


def _style_ax(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, axis="y", alpha=0.45)
        ax.set_axisbelow(True)


def _save(fig, name: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / name, facecolor=BG, edgecolor="none")
    plt.close(fig)


def plot_training_history(history: Dict):
    apply_theme()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], color=C_PRIMARY, lw=1.8, label="Training")
    axes[0].plot(epochs, history["val_loss"], color=C_SECONDARY, lw=1.8, ls="--", label="Validation")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("(a) Optimization")
    axes[0].legend(); _style_ax(axes[0])

    axes[1].plot(epochs, history["val_rmse"], color=C_TERTIARY, lw=1.8)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("RMSE"); axes[1].set_title("(b) Validation RMSE")
    _style_ax(axes[1])

    axes[2].plot(epochs, history["val_nll"], color=C_QUATERNARY, lw=1.8)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("NLL"); axes[2].set_title("(c) Validation NLL")
    _style_ax(axes[2])

    fig.tight_layout()
    _save(fig, "01_training_history.png")


def plot_actual_vs_predicted(metrics: Dict):
    apply_theme()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    y, pred, std = metrics["y_true"], metrics["y_pred"], metrics["y_std"]
    sc = ax.scatter(y, pred, c=std, cmap=CMAP_UNCERT, s=22, alpha=0.75,
                    edgecolors="white", linewidths=0.3, vmin=np.percentile(std, 5),
                    vmax=np.percentile(std, 95))
    lims = [min(y.min(), pred.min()) - 0.2, max(y.max(), pred.max()) + 0.2]
    ax.plot(lims, lims, ls="--", color=C_NEUTRAL, lw=1.5, label="Ideal fit")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Observed mental health score")
    ax.set_ylabel("Predicted score (mixture mean)")
    ax.set_title(f"Parity plot  ($R^2$={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f})")
    cb = fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label("Predictive uncertainty ($\\sigma$)")
    ax.legend(loc="upper left", framealpha=0.95)
    _style_ax(ax)
    _save(fig, "02_actual_vs_predicted.png")


def plot_prediction_intervals(metrics: Dict, n_show: int = 100):
    apply_theme()
    idx = np.argsort(metrics["y_true"])[:n_show]
    y, pred, std = metrics["y_true"][idx], metrics["y_pred"][idx], metrics["y_std"][idx]
    x = np.arange(n_show)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(x, pred - 1.645 * std, pred + 1.645 * std, color=C_LIGHT, alpha=0.9, label="90% PI")
    ax.plot(x, pred, color=C_PRIMARY, lw=1.6, label="Predicted mean")
    ax.scatter(x, y, color=C_TERTIARY, s=14, zorder=3, label="Observed", edgecolors="white", linewidths=0.3)
    ax.set_xlabel("Sample index (sorted by observed score)")
    ax.set_ylabel("Mental health score")
    ax.set_title(f"Prediction intervals  (90% coverage = {metrics['coverage_90']:.1%})")
    ax.legend(ncol=3, loc="upper left"); _style_ax(ax)
    _save(fig, "03_prediction_intervals.png")


def plot_model_comparison(main_metrics: Dict, baseline_metrics: Dict):
    apply_theme()
    models = ["DRG-MDN"] + list(baseline_metrics.keys())
    rmse = [main_metrics["rmse"]] + [baseline_metrics[k]["rmse"] for k in baseline_metrics]
    r2 = [main_metrics["r2"]] + [baseline_metrics[k]["r2"] for k in baseline_metrics]
    mae = [main_metrics["mae"]] + [baseline_metrics[k]["mae"] for k in baseline_metrics]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = [C_PRIMARY if m == "DRG-MDN" else PALETTE[(i % (len(PALETTE)-1)) + 1]
              for i, m in enumerate(models)]

    for ax, vals, ylab, panel in zip(axes, [rmse, mae, r2], ["RMSE", "MAE", "$R^2$"], ["a", "b", "c"]):
        bars = ax.bar(range(len(models)), vals, color=colors, edgecolor=SPINE, linewidth=0.6, width=0.62)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=28, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(f"({panel}) {ylab} comparison")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        _style_ax(ax)

    fig.tight_layout()
    _save(fig, "04_model_comparison.png")


def plot_residual_analysis(metrics: Dict):
    apply_theme()
    resid = metrics["y_true"] - metrics["y_pred"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.histplot(resid, kde=True, color=C_PRIMARY, ax=axes[0], edgecolor="white",
                 line_kws={"color": C_SECONDARY, "lw": 1.5})
    axes[0].axvline(0, color=C_NEUTRAL, ls="--", lw=1.2)
    axes[0].set_xlabel("Residual"); axes[0].set_ylabel("Count")
    axes[0].set_title("(a) Residual distribution")
    _style_ax(axes[0])

    axes[1].scatter(metrics["y_pred"], resid, c=metrics["y_std"], cmap=CMAP_UNCERT,
                    alpha=0.65, s=16, edgecolors="none")
    axes[1].axhline(0, color=C_NEUTRAL, ls="--", lw=1.2)
    axes[1].set_xlabel("Predicted score"); axes[1].set_ylabel("Residual")
    axes[1].set_title("(b) Residuals vs. prediction")
    _style_ax(axes[1])

    fig.tight_layout()
    _save(fig, "05_residual_analysis.png")


def plot_calibration(
    metrics: Dict,
    calibrated_metrics: Optional[Dict] = None,
):
    apply_theme()
    y, pred, std = metrics["y_true"], metrics["y_pred"], metrics["y_std"]
    raw_prof = metrics["coverage_profile"] if "coverage_profile" in metrics else coverage_profile(y, pred, std)

    fig, ax = plt.subplots(figsize=(5.5, 5))

    nominal = raw_prof["nominal_coverage"].values
    perfect = nominal

    ax.plot(
        [0.45, 1.0], [0.45, 1.0], ls="--", color=C_NEUTRAL, lw=1.5,
        label="Perfect calibration", zorder=1,
    )
    ax.scatter(
        nominal, raw_prof["empirical_coverage"].values,
        s=60, color=C_SECONDARY, zorder=3, label="DRG-MDN (raw)", edgecolors="white",
    )
    ax.plot(
        nominal, raw_prof["empirical_coverage"].values,
        color=C_SECONDARY, lw=1.4, alpha=0.7, zorder=2,
    )

    ece_raw = metrics.get("ece", expected_calibration_error(raw_prof))
    subtitle = f"raw ECE={ece_raw:.3f}"

    if calibrated_metrics is not None:
        cal_prof = (
            calibrated_metrics["coverage_profile"]
            if "coverage_profile" in calibrated_metrics
            else coverage_profile(
                calibrated_metrics["y_true"],
                calibrated_metrics["y_pred"],
                calibrated_metrics["y_std"],
            )
        )
        ax.scatter(
            cal_prof["nominal_coverage"], cal_prof["empirical_coverage"],
            s=60, color=C_PRIMARY, zorder=4, label="DRG-MDN (calibrated)", edgecolors="white",
        )
        ax.plot(
            cal_prof["nominal_coverage"], cal_prof["empirical_coverage"],
            color=C_PRIMARY, lw=1.6, zorder=3,
        )
        ece_cal = calibrated_metrics.get("ece", expected_calibration_error(cal_prof))
        subtitle = f"raw ECE={ece_raw:.3f}  �? calibrated ECE={ece_cal:.3f}"

    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("Nominal interval coverage")
    ax.set_ylabel("Empirical interval coverage")
    ax.set_title(f"Interval calibration reliability diagram\n({subtitle})")
    ax.legend(loc="lower right", fontsize=9)
    _style_ax(ax)
    _save(fig, "06_calibration_curve.png")


def plot_bland_altman(metrics: Dict):
    apply_theme()
    y, pred = metrics["y_true"], metrics["y_pred"]
    mean = (y + pred) / 2
    diff = y - pred
    md = diff.mean()
    sd = diff.std()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(mean, diff, c=C_PRIMARY, alpha=0.55, s=18, edgecolors="white", linewidths=0.3)
    ax.axhline(md, color=C_SECONDARY, lw=1.5, label=f"Mean bias = {md:.3f}")
    ax.axhline(md + 1.96 * sd, color=C_NEUTRAL, ls="--", lw=1.2, label=f"+1.96 SD = {md+1.96*sd:.3f}")
    ax.axhline(md - 1.96 * sd, color=C_NEUTRAL, ls="--", lw=1.2)
    ax.set_xlabel("Mean of observed and predicted")
    ax.set_ylabel("Observed $-$ predicted")
    ax.set_title("Bland-Altman agreement analysis")
    ax.legend(fontsize=9); _style_ax(ax)
    _save(fig, "07_bland_altman.png")


def plot_uncertainty_quality(metrics: Dict):
    apply_theme()
    error = np.abs(metrics["y_true"] - metrics["y_pred"])
    std = metrics["y_std"]

    bins = pd.qcut(std, q=5, duplicates="drop")
    grp = (
        pd.DataFrame({"bin": bins, "error": error})
        .groupby("bin", observed=True)["error"]
        .agg(["mean", "std", "count"])
    )
    rho, pval = stats.spearmanr(
        np.arange(len(grp)), grp["mean"].values, nan_policy="omit"
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(grp))
    ax.bar(
        x, grp["mean"].values, yerr=grp["std"].values / np.sqrt(grp["count"].values),
        color=PALETTE[: len(grp)], edgecolor=SPINE, width=0.62,
        capsize=3, error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i+1}\n(low $\\sigma$)" if i == 0 else (f"Q{i+1}\n(high $\\sigma$)" if i == len(grp) - 1 else f"Q{i+1}") for i in range(len(grp))])
    ax.set_xlabel("Predictive uncertainty quintile ($\\sigma$)")
    ax.set_ylabel("Mean absolute error")
    ax.set_title("Uncertainty stratification quality")
    ax.text(
        0.98, 0.04,
        f"Spearman $\\rho$={rho:.2f}  ($p$={pval:.3g})\n"
        f"Q1→Q5 mean |error|: {grp['mean'].values[0]:.3f}→{grp['mean'].values[-1]:.3f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=BG, edgecolor=GRID, alpha=0.92),
    )
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, "08_uncertainty_quality.png")


def plot_dynamic_adjacency(adj_matrix: np.ndarray):
    apply_theme()
    avg = adj_matrix.mean(axis=0)
    n = avg.shape[0]
    labels = FEATURE_LABELS if n == len(FEATURE_LABELS) else [f"F{i}" for i in range(n)]

    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(avg, cmap=CMAP_SEQ, ax=ax, square=True, linewidths=0.15, linecolor=GRID,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"shrink": 0.75, "label": "Edge weight"})
    ax.set_title("Dynamic relational graph (mean adjacency, test set)")
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    _save(fig, "09_dynamic_graph_heatmap.png")


def plot_graph_node_importance(adj_matrix: np.ndarray):
    apply_theme()
    importance = adj_matrix.mean(axis=0).sum(axis=1)
    importance = importance / importance.max()
    n = len(importance)
    labels = FEATURE_LABELS if n == len(FEATURE_LABELS) else [f"F{i}" for i in range(n)]
    top = np.argsort(importance)[::-1][:12]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([labels[i] for i in top[::-1]], importance[top[::-1]], color=C_PRIMARY, edgecolor=SPINE, height=0.7)
    ax.set_xlabel("Normalized graph centrality")
    ax.set_title("Top feature nodes by dynamic graph influence")
    _style_ax(ax, grid=True)
    _save(fig, "10_graph_node_importance.png")


def plot_mixture_analysis(pi_matrix: np.ndarray, mu_matrix: np.ndarray, y_true: np.ndarray):
    apply_theme()
    mean_pi = pi_matrix.mean(axis=0)
    dominant = pi_matrix.argmax(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    comp_labels = [f"Component {k+1}" for k in range(len(mean_pi))]
    axes[0].bar(comp_labels, mean_pi, color=PALETTE[:len(mean_pi)], edgecolor=SPINE, width=0.55)
    axes[0].set_ylabel("Average mixture weight $\\pi_k$")
    axes[0].set_title("(a) Global mixture composition")
    _style_ax(axes[0])

    for k in range(mu_matrix.shape[1]):
        mask = dominant == k
        if mask.sum() == 0:
            continue
        axes[1].scatter(y_true[mask], mu_matrix[mask, k], alpha=0.55, s=18,
                        color=PALETTE[k], label=f"C{k+1} dominant (n={mask.sum()})", edgecolors="none")
    axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                 ls="--", color=C_NEUTRAL, lw=1.2)
    axes[1].set_xlabel("Observed score"); axes[1].set_ylabel("Component mean $\\mu_k$")
    axes[1].set_title("(b) Component means vs. observed (by dominant component)")
    axes[1].legend(fontsize=8); _style_ax(axes[1])

    fig.tight_layout()
    _save(fig, "11_mixture_components.png")


def plot_mixture_by_stress(pi_matrix: np.ndarray, test_meta: pd.DataFrame):
    if test_meta.empty or "Stress_Level" not in test_meta.columns:
        return
    apply_theme()
    df = test_meta.copy()
    df["dominant"] = pi_matrix.argmax(axis=1)
    for k in range(pi_matrix.shape[1]):
        df[f"pi_{k}"] = pi_matrix[:, k]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    stress_pi = df.groupby("Stress_Level")[[f"pi_{k}" for k in range(pi_matrix.shape[1])]].mean()
    stress_pi = stress_pi.reindex(STRESS_ORDER).dropna(how="all")
    bottom = np.zeros(len(stress_pi))
    for k in range(pi_matrix.shape[1]):
        axes[0].bar(stress_pi.index, stress_pi[f"pi_{k}"], bottom=bottom,
                    color=PALETTE[k], label=f"Component {k+1}", edgecolor=SPINE, width=0.6)
        bottom += stress_pi[f"pi_{k}"].values
    axes[0].set_ylabel("Mixture weight"); axes[0].set_title("(a) Mixture weights by stress level")
    axes[0].legend(fontsize=8); _style_ax(axes[0])

    sns.boxplot(data=df, x="Stress_Level", y="dominant", order=STRESS_ORDER,
                hue="Stress_Level", palette=PALETTE[:4], ax=axes[1],
                linewidth=0.8, legend=False)
    axes[1].set_xlabel("Stress level"); axes[1].set_ylabel("Dominant component index")
    axes[1].set_title("(b) Dominant mixture component by stress")
    _style_ax(axes[1])

    fig.tight_layout()
    _save(fig, "12_mixture_by_stress.png")


def plot_error_by_stress(metrics: Dict, test_meta: pd.DataFrame):
    if test_meta.empty:
        return
    apply_theme()
    df = test_meta.copy()
    df["error"] = np.abs(metrics["y_true"] - metrics["y_pred"])
    df["observed"] = metrics["y_true"]
    df["predicted"] = metrics["y_pred"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.boxplot(data=df, x="Stress_Level", y="error", order=STRESS_ORDER,
                color=C_LIGHT, ax=axes[0], linewidth=0.8,
                flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
    sns.stripplot(data=df, x="Stress_Level", y="error", order=STRESS_ORDER,
                  color=C_PRIMARY, alpha=0.25, size=2.5, ax=axes[0])
    axes[0].set_xlabel("Stress level"); axes[0].set_ylabel("Absolute error")
    axes[0].set_title("(a) Prediction error by stress stratum")
    _style_ax(axes[0])

    sns.violinplot(data=df, x="Stress_Level", y="observed", order=STRESS_ORDER,
                   color=C_TERTIARY, inner="box", ax=axes[1], linewidth=0.8, alpha=0.75)
    axes[1].set_xlabel("Stress level"); axes[1].set_ylabel("Observed MH score")
    axes[1].set_title("(b) Observed score distribution by stress")
    _style_ax(axes[1])

    fig.tight_layout()
    _save(fig, "13_error_by_stress.png")


def plot_risk_stratification(metrics: Dict):
    apply_theme()
    std = metrics["y_std"]
    tertiles = pd.qcut(std, q=3, labels=["Low", "Medium", "High"])
    df = pd.DataFrame({"tier": tertiles, "actual": metrics["y_true"], "pred": metrics["y_pred"],
                       "error": np.abs(metrics["y_true"] - metrics["y_pred"])})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    order = ["Low", "Medium", "High"]
    colors = [C_TERTIARY, C_ACCENT, C_SECONDARY]

    parts = axes[0].violinplot(
        [df[df["tier"] == t]["actual"].values for t in order],
        positions=range(3), showmedians=True, widths=0.75,
    )
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c); pc.set_alpha(0.65); pc.set_edgecolor(SPINE)
    parts["cmedians"].set_color(SPINE)
    axes[0].set_xticks(range(3)); axes[0].set_xticklabels([f"{t} $\\sigma$" for t in order])
    axes[0].set_ylabel("Observed MH score")
    axes[0].set_title("(a) Observed score by uncertainty tier")
    _style_ax(axes[0])

    mae = df.groupby("tier", observed=True)["error"].mean().reindex(order)
    axes[1].bar(order, mae.values, color=colors, edgecolor=SPINE, width=0.55)
    axes[1].set_xlabel("Uncertainty tier"); axes[1].set_ylabel("MAE")
    axes[1].set_title("(b) MAE by uncertainty tier")
    for i, v in enumerate(mae.values):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    _style_ax(axes[1])

    fig.tight_layout()
    _save(fig, "14_risk_stratification.png")


def plot_factor_scatter(df: pd.DataFrame):
    apply_theme()
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    panels = [
        ("Avg_Daily_Usage_Hours", "Daily usage (h)", "(a)"),
        ("Sleep_Hours_Per_Night", "Sleep duration (h)", "(b)"),
        ("Daily_Unlocks", "Daily unlocks", "(c)"),
        ("Study_Hours", "Study hours", "(d)"),
    ]
    stress_colors = dict(zip(STRESS_ORDER, PALETTE[:4]))

    for ax, (col, xlab, tag) in zip(axes.flat, panels):
        for stress in STRESS_ORDER:
            sub = df[df["Stress_Level"] == stress]
            ax.scatter(sub[col], sub["Mental_Health_Score"], s=10, alpha=0.35,
                       c=stress_colors[stress], label=stress, edgecolors="none")
        r, _ = stats.pearsonr(df[col], df["Mental_Health_Score"])
        ax.set_xlabel(xlab); ax.set_ylabel("MH score")
        ax.set_title(f"{tag} {xlab.split('(')[0].strip()}  ($r$={r:.2f})")
        _style_ax(ax)

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=stress_colors[s],
                      markersize=6, label=s) for s in STRESS_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    _save(fig, "15_behavior_mh_scatter.png")


def plot_eda_dashboard(df: pd.DataFrame):
    apply_theme()
    fig = plt.figure(figsize=(13, 8.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.42, width_ratios=[1.0, 1.0, 1.25])

    ax = fig.add_subplot(gs[0, 0])
    sns.histplot(df["Mental_Health_Score"], kde=True, color=C_PRIMARY, ax=ax,
                 edgecolor="white", line_kws={"color": C_SECONDARY})
    ax.axvline(df["Mental_Health_Score"].mean(), color=C_NEUTRAL, ls="--", lw=1.2)
    ax.set_title("(a) MH score distribution", pad=10)
    _style_ax(ax)

    ax = fig.add_subplot(gs[0, 1])
    stress_mh = df.groupby("Stress_Level")["Mental_Health_Score"].mean().reindex(STRESS_ORDER)
    ax.bar(STRESS_ORDER, stress_mh.values, color=PALETTE[:4], edgecolor=SPINE, width=0.62)
    ax.set_title("(b) Mean MH by stress", pad=10)
    ax.tick_params(axis="x", rotation=25)
    _style_ax(ax)

    ax = fig.add_subplot(gs[0, 2])
    num = ["Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night", "Daily_Unlocks",
           "Study_Hours", "Physical_Activity_Hours", "Mental_Health_Score"]
    short = ["Usage", "Sleep", "Unlocks", "Study", "Activity", "MH"]
    corr = df[num].corr()
    corr.index = short
    corr.columns = short
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap=CMAP_DIV, ax=ax,
        vmin=-1, vmax=1, linewidths=0.4, linecolor=GRID,
        annot_kws={"size": 7.5}, cbar_kws={"shrink": 0.72, "pad": 0.02},
    )
    ax.set_title("(c) Correlation matrix", pad=10)
    ax.tick_params(axis="x", rotation=40, labelsize=7.5)
    ax.tick_params(axis="y", rotation=0, labelsize=7.5)

    ax = fig.add_subplot(gs[1, 0])
    plat = df.groupby("Most_Used_Platform")["Mental_Health_Score"].mean().sort_values()
    ax.barh(plat.index, plat.values, color=C_PRIMARY, edgecolor=SPINE, height=0.7)
    ax.set_xlabel("Mean MH score")
    ax.set_title("(d) Platform comparison", pad=10)
    _style_ax(ax)

    ax = fig.add_subplot(gs[1, 1])
    purpose = df.groupby("Purpose_Of_Use")["Mental_Health_Score"].mean().sort_values()
    ax.barh(purpose.index, purpose.values, color=C_TERTIARY, edgecolor=SPINE, height=0.7)
    ax.set_xlabel("Mean MH score")
    ax.set_title("(e) Usage purpose", pad=10)
    _style_ax(ax)

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    stats_txt = (
        f"Sample size: {len(df):,}\n"
        f"Countries: {df['Country'].nunique()}\n"
        f"Mean usage: {df['Avg_Daily_Usage_Hours'].mean():.1f} h/day\n"
        f"Mean MH score: {df['Mental_Health_Score'].mean():.2f}\n"
        f"Very high stress: {(df['Stress_Level']=='Very High').mean()*100:.1f}%"
    )
    ax.text(
        0.5, 0.5, stats_txt, ha="center", va="center", fontsize=11,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=C_LIGHT, edgecolor=GRID, linewidth=0.8),
    )
    ax.set_title("(f) Dataset summary", pad=10)

    fig.subplots_adjust(top=0.94)
    _save(fig, "16_eda_dashboard.png")


def plot_nll_comparison(main_metrics: Dict, baseline_metrics: Dict):
    apply_theme()
    models, nlls = ["DRG-MDN"], [main_metrics["nll"]]
    for k, v in baseline_metrics.items():
        if v.get("nll") is not None:
            models.append(k); nlls.append(v["nll"])

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [C_PRIMARY] + [PALETTE[i+1] for i in range(len(models)-1)]
    ax.bar(models, nlls, color=colors, edgecolor=SPINE, width=0.55)
    ax.set_ylabel("Negative log-likelihood")
    ax.set_title("Probabilistic fit quality (NLL)")
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(nlls):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    _style_ax(ax)
    _save(fig, "17_nll_comparison.png")


def generate_all_figures(
    history: Dict,
    test_metrics: Dict,
    baseline_metrics: Dict,
    df: pd.DataFrame,
    adj_matrix: Optional[np.ndarray] = None,
    pi_matrix: Optional[np.ndarray] = None,
    mu_matrix: Optional[np.ndarray] = None,
    test_meta: Optional[pd.DataFrame] = None,
    calibrated_metrics: Optional[Dict] = None,
):
    apply_theme()
    test_meta = test_meta if test_meta is not None else pd.DataFrame()

    plot_training_history(history)
    plot_actual_vs_predicted(test_metrics)
    plot_prediction_intervals(test_metrics)
    plot_model_comparison(test_metrics, baseline_metrics)
    plot_residual_analysis(test_metrics)
    plot_calibration(test_metrics, calibrated_metrics)
    plot_bland_altman(test_metrics)
    plot_uncertainty_quality(test_metrics)
    if adj_matrix is not None:
        plot_dynamic_adjacency(adj_matrix)
        plot_graph_node_importance(adj_matrix)
    if pi_matrix is not None and mu_matrix is not None:
        plot_mixture_analysis(pi_matrix, mu_matrix, test_metrics["y_true"])
        plot_mixture_by_stress(pi_matrix, test_meta)
    plot_error_by_stress(test_metrics, test_meta)
    plot_risk_stratification(test_metrics)
    plot_factor_scatter(df)
    plot_eda_dashboard(df)
    plot_nll_comparison(test_metrics, baseline_metrics)

    summary = {
        "drg_mdn_test": {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in test_metrics.items()
            if k not in ("y_true", "y_pred", "y_std", "coverage_profile")
        },
        "baselines": baseline_metrics,
    }
    with open(RESULTS_DIR / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nAll figures saved to: {RESULTS_DIR}")
