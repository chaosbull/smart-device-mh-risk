# Author: ZengWenquan
# https://github.com/chaosbull
# License: CC BY 4.0

"""Coverage, ECE, and the csv/tex tables that go with the calibration figure."""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from config import RESULTS_DIR

NOMINAL_LEVELS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def z_critical(alpha: float) -> float:
    return float(stats.norm.ppf((1.0 + alpha) / 2.0))


def empirical_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    alpha: float,
    scale: float = 1.0,
) -> float:
    z = z_critical(alpha)
    return float(np.mean(np.abs(y_true - y_pred) <= z * scale * y_std))


def coverage_profile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    scale: float = 1.0,
    levels: List[float] | None = None,
) -> pd.DataFrame:
    levels = levels or NOMINAL_LEVELS
    rows = []
    for alpha in levels:
        emp = empirical_coverage(y_true, y_pred, y_std, alpha, scale)
        rows.append({
            "nominal_coverage": alpha,
            "empirical_coverage": emp,
            "calibration_gap": emp - alpha,
            "abs_gap": abs(emp - alpha),
            "z_score": z_critical(alpha),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(profile: pd.DataFrame) -> float:
    return float(profile["abs_gap"].mean())


def maximum_calibration_error(profile: pd.DataFrame) -> float:
    return float(profile["abs_gap"].max())


def interval_sharpness(y_std: np.ndarray, alpha: float = 0.90, scale: float = 1.0) -> float:
    return float(np.mean(z_critical(alpha) * scale * y_std))


def find_sigma_scale(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    target_alpha: float = 0.90,
    s_min: float = 1.0,
    s_max: float = 2.5,
    n_grid: int = 151,
) -> Tuple[float, float]:
    best_s, best_cov, best_gap = 1.0, 0.0, 1.0
    for s in np.linspace(s_min, s_max, n_grid):
        cov = empirical_coverage(y_true, y_pred, y_std, target_alpha, s)
        gap = abs(cov - target_alpha)
        if gap < best_gap:
            best_gap = gap
            best_s = float(s)
            best_cov = cov
    return best_s, best_cov


def find_sigma_scale_min_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    s_min: float = 1.0,
    s_max: float = 2.5,
    n_grid: int = 151,
) -> Tuple[float, float]:
    best_s, best_ece = 1.0, 1.0
    for s in np.linspace(s_min, s_max, n_grid):
        prof = coverage_profile(y_true, y_pred, y_std, scale=s)
        ece = expected_calibration_error(prof)
        if ece < best_ece:
            best_ece = ece
            best_s = float(s)
    return best_s, best_ece


def apply_sigma_scale(metrics: Dict, scale: float) -> Dict:
    out = dict(metrics)
    out["y_std"] = metrics["y_std"] * scale
    out["sigma_scale"] = scale
    for alpha in (0.90,):
        z = z_critical(alpha)
        lower = out["y_pred"] - z * out["y_std"]
        upper = out["y_pred"] + z * out["y_std"]
        out[f"coverage_{int(alpha * 100)}"] = float(
            np.mean((out["y_true"] >= lower) & (out["y_true"] <= upper))
        )
    prof = coverage_profile(out["y_true"], out["y_pred"], out["y_std"])
    out["ece"] = expected_calibration_error(prof)
    out["mce"] = maximum_calibration_error(prof)
    out["coverage_profile"] = prof
    return out


def quintile_coverage_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    alpha: float = 0.90,
    scale: float = 1.0,
) -> pd.DataFrame:
    z = z_critical(alpha)
    in_interval = np.abs(y_true - y_pred) <= z * scale * y_std
    bins = pd.qcut(y_std, q=5, duplicates="drop")
    rows = []
    for i, b in enumerate(bins.unique()):
        mask = bins == b
        rows.append({
            "uncertainty_quintile": f"Q{i + 1}",
            "mean_sigma": float(y_std[mask].mean() * scale),
            "n_samples": int(mask.sum()),
            f"empirical_{int(alpha*100)}": float(in_interval[mask].mean()),
            "mean_abs_error": float(np.abs(y_true[mask] - y_pred[mask]).mean()),
        })
    return pd.DataFrame(rows)


def model_comparison_calibration_table(
    models: Dict[str, Dict],
) -> pd.DataFrame:
    rows = []
    for name, m in models.items():
        rows.append({
            "Model": name,
            "ECE": m.get("ece", np.nan),
            "MCE": m.get("mce", np.nan),
            "Cov@50%": m.get("cov_50", np.nan),
            "Cov@80%": m.get("cov_80", np.nan),
            "Cov@90%": m.get("cov_90", np.nan),
            "Cov@95%": m.get("cov_95", np.nan),
            "Mean PI width@90%": m.get("sharpness_90", np.nan),
            "NLL": m.get("nll", np.nan),
        })
    return pd.DataFrame(rows)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}\\%"


def profile_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Nominal & Empirical & Gap & $|Gap|$ & $z$ \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{_pct(r['nominal_coverage'])} & {_pct(r['empirical_coverage'])} & "
            f"{r['calibration_gap']:+.3f} & {r['abs_gap']:.3f} & {r['z_score']:.3f} \\\\"
        )
    lines += [
        "\\midrule",
        f"ECE & \\multicolumn{{4}}{{l}}{{{df['abs_gap'].mean():.4f}}} \\\\",
        f"MCE & \\multicolumn{{4}}{{l}}{{{df['abs_gap'].max():.4f}}} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def comparison_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Model & ECE & MCE & Cov@90\\% & Cov@95\\% & PI Width & NLL \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        nll = f"{r['NLL']:.3f}" if not np.isnan(r["NLL"]) else "--"
        lines.append(
            f"{r['Model']} & {r['ECE']:.4f} & {r['MCE']:.4f} & "
            f"{r['Cov@90%']*100:.1f}\\% & {r['Cov@95%']*100:.1f}\\% & "
            f"{r['Mean PI width@90%']:.3f} & {nll} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def export_calibration_artifacts(
    raw_metrics: Dict,
    calibrated_metrics: Dict,
    baseline_models: Dict[str, Dict],
    sigma_scale: float,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_prof = raw_metrics["coverage_profile"]
    cal_prof = calibrated_metrics["coverage_profile"]

    raw_prof.to_csv(RESULTS_DIR / "table_calibration_profile_raw.csv", index=False)
    cal_prof.to_csv(RESULTS_DIR / "table_calibration_profile_calibrated.csv", index=False)

    quint_raw = quintile_coverage_table(
        raw_metrics["y_true"], raw_metrics["y_pred"], raw_metrics["y_std"]
    )
    quint_cal = quintile_coverage_table(
        calibrated_metrics["y_true"],
        calibrated_metrics["y_pred"],
        calibrated_metrics["y_std"] / sigma_scale,
        scale=sigma_scale,
    )
    quint_raw.to_csv(RESULTS_DIR / "table_quintile_coverage_raw.csv", index=False)
    quint_cal.to_csv(RESULTS_DIR / "table_quintile_coverage_calibrated.csv", index=False)

    cmp_rows = {
        "DRG-MDN (raw)": _summary_from_metrics(raw_metrics),
        "DRG-MDN (calibrated)": _summary_from_metrics(calibrated_metrics),
    }
    for name, bm in baseline_models.items():
        if "y_true" in bm and "y_pred" in bm:
            cmp_rows[name] = _summary_from_arrays(
                np.asarray(bm["y_true"]),
                np.asarray(bm["y_pred"]),
                np.asarray(bm["y_std"]) if bm.get("y_std") is not None else None,
                bm.get("nll"),
            )
        elif "coverage_90" in bm:
            cmp_rows[name] = {
                "ece": np.nan,
                "mce": np.nan,
                "cov_90": bm.get("coverage_90"),
                "cov_95": np.nan,
                "sharpness_90": np.nan,
                "nll": bm.get("nll"),
            }

    cmp_df = model_comparison_calibration_table(cmp_rows)
    cmp_df.to_csv(RESULTS_DIR / "table_model_calibration_comparison.csv", index=False)

    latex_parts = [
        "% Calibration tables\n",
        profile_to_latex(
            cal_prof,
            "Prediction interval calibration profile (DRG-MDN, post-hoc scaled).",
            "tab:calibration_profile",
        ),
        "",
        comparison_to_latex(
            cmp_df.dropna(subset=["ECE"], how="all"),
            "Uncertainty calibration comparison across models (test set).",
            "tab:calibration_comparison",
        ),
    ]
    (RESULTS_DIR / "tables_calibration.tex").write_text("\n".join(latex_parts), encoding="utf-8")

    summary = {
        "sigma_scale": sigma_scale,
        "raw": {
            "ece": raw_metrics["ece"],
            "mce": raw_metrics["mce"],
            "coverage_90": raw_metrics["coverage_90"],
            "coverage_50": empirical_coverage(
                raw_metrics["y_true"], raw_metrics["y_pred"], raw_metrics["y_std"], 0.50
            ),
            "coverage_95": empirical_coverage(
                raw_metrics["y_true"], raw_metrics["y_pred"], raw_metrics["y_std"], 0.95
            ),
            "sharpness_90": interval_sharpness(raw_metrics["y_std"]),
        },
        "calibrated": {
            "ece": calibrated_metrics["ece"],
            "mce": calibrated_metrics["mce"],
            "coverage_90": calibrated_metrics["coverage_90"],
            "coverage_50": empirical_coverage(
                calibrated_metrics["y_true"],
                calibrated_metrics["y_pred"],
                calibrated_metrics["y_std"],
                0.50,
            ),
            "coverage_95": empirical_coverage(
                calibrated_metrics["y_true"],
                calibrated_metrics["y_pred"],
                calibrated_metrics["y_std"],
                0.95,
            ),
            "sharpness_90": interval_sharpness(calibrated_metrics["y_std"]),
        },
        "ece_improvement_pct": (
            (raw_metrics["ece"] - calibrated_metrics["ece"]) / raw_metrics["ece"] * 100
            if raw_metrics["ece"] > 0 else 0.0
        ),
        "coverage_90_improvement_pp": (
            (calibrated_metrics["coverage_90"] - raw_metrics["coverage_90"]) * 100
        ),
    }
    with open(RESULTS_DIR / "calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md = _markdown_report(summary, cal_prof, cmp_df)
    (RESULTS_DIR / "tables_calibration.md").write_text(md, encoding="utf-8")


def _summary_from_metrics(m: Dict) -> Dict:
    prof = m["coverage_profile"]
    return {
        "ece": m["ece"],
        "mce": m["mce"],
        "cov_50": float(prof.loc[prof["nominal_coverage"] == 0.50, "empirical_coverage"].iloc[0]),
        "cov_80": float(prof.loc[prof["nominal_coverage"] == 0.80, "empirical_coverage"].iloc[0]),
        "cov_90": m["coverage_90"],
        "cov_95": float(prof.loc[prof["nominal_coverage"] == 0.95, "empirical_coverage"].iloc[0]),
        "sharpness_90": interval_sharpness(m["y_std"]),
        "nll": m.get("nll"),
    }


def _summary_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray | None,
    nll: float | None,
) -> Dict:
    if y_std is None:
        resid = y_true - y_pred
        y_std = np.full_like(y_pred, max(float(resid.std()), 1e-6))
    prof = coverage_profile(y_true, y_pred, y_std)
    return {
        "ece": expected_calibration_error(prof),
        "mce": maximum_calibration_error(prof),
        "cov_50": empirical_coverage(y_true, y_pred, y_std, 0.50),
        "cov_80": empirical_coverage(y_true, y_pred, y_std, 0.80),
        "cov_90": empirical_coverage(y_true, y_pred, y_std, 0.90),
        "cov_95": empirical_coverage(y_true, y_pred, y_std, 0.95),
        "sharpness_90": interval_sharpness(y_std),
        "nll": nll,
    }


def enrich_metrics(metrics: Dict) -> Dict:
    prof = coverage_profile(metrics["y_true"], metrics["y_pred"], metrics["y_std"])
    out = dict(metrics)
    out["coverage_profile"] = prof
    out["ece"] = expected_calibration_error(prof)
    out["mce"] = maximum_calibration_error(prof)
    if "coverage_90" not in out:
        out["coverage_90"] = empirical_coverage(
            metrics["y_true"], metrics["y_pred"], metrics["y_std"], 0.90
        )
    return out


def _df_to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(format(v, floatfmt))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def _markdown_report(summary: Dict, profile: pd.DataFrame, comparison: pd.DataFrame) -> str:
    lines = [
        "# Calibration tables",
        "",
        "Horizontal axis of the reliability plot is the nominal coverage, vertical axis is what we actually got.",
        "A point on y = x means the interval is about as wide as it claims.",
        "",
        f"Scale fit on the validation set: lambda = {summary['sigma_scale']:.3f}",
        "",
        "### Before and after the scale",
        "",
        "| | raw | scaled | change |",
        "|------|--------|--------|------|",
        f"| ECE | {summary['raw']['ece']:.4f} | {summary['calibrated']['ece']:.4f} | "
        f"{summary['ece_improvement_pct']:+.1f}% |",
        f"| 90% coverage | {summary['raw']['coverage_90']*100:.1f}% | "
        f"{summary['calibrated']['coverage_90']*100:.1f}% | "
        f"{summary['coverage_90_improvement_pp']:+.1f} pp |",
        f"| mean 90% interval width | {summary['raw']['sharpness_90']:.3f} | "
        f"{summary['calibrated']['sharpness_90']:.3f} | |",
        "",
        "### Nominal vs empirical coverage (after scaling)",
        "",
        _df_to_markdown(profile),
        "",
        "### Model comparison",
        "",
        _df_to_markdown(comparison),
    ]
    return "\n".join(lines)
