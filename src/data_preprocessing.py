# Author: ZengWenquan
# https://github.com/chaosbull
# License: CC BY 4.0

"""Read the csv, build the 12 continuous columns, split 70/15/15."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from config import (
    BASE_CONTINUOUS_COLS,
    CONTINUOUS_COLS,
    DATA_PATH,
    DERIVED_COLS,
    EMBED_COLS,
    GENDER_MAP,
    RESULTS_DIR,
    SEED,
    STRESS_MAP,
    TARGET_COL,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


@dataclass
class PreprocessArtifacts:
    boxcox_lambda: Dict[str, float]
    continuous_mean: np.ndarray
    continuous_std: np.ndarray
    cat_maps: Dict[str, Dict[str, int]]
    country_names: List[str]


class MentalHealthDataset(Dataset):
    def __init__(
        self,
        cat_indices: np.ndarray,
        continuous: np.ndarray,
        targets: np.ndarray,
        countries: np.ndarray,
    ):
        self.cat_indices = torch.as_tensor(cat_indices, dtype=torch.long)
        self.continuous = torch.as_tensor(continuous, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.countries = torch.as_tensor(countries, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return {
            "cat": self.cat_indices[idx],
            "cont": self.continuous[idx],
            "y": self.targets[idx],
            "country": self.countries[idx],
        }


def _boxcox_transform(series: pd.Series, lam: float | None = None) -> Tuple[np.ndarray, float]:
    shifted = series - series.min() + 1e-3
    if lam is None:
        transformed, fitted_lam = stats.boxcox(shifted)
        return transformed, float(fitted_lam)
    if lam == 0:
        return np.log(shifted), 0.0
    return stats.boxcox(shifted, lmbda=lam), lam


def load_raw_dataframe() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, PreprocessArtifacts]:
    df = df.copy()
    df["Gender_num"] = df["Gender"].map(GENDER_MAP)
    df["Stress_num"] = df["Stress_Level"].map(STRESS_MAP)

    sleep_bc, lam_sleep = _boxcox_transform(df["Sleep_Hours_Per_Night"])
    usage_bc, lam_usage = _boxcox_transform(df["Avg_Daily_Usage_Hours"])
    df["Sleep_Hours_Per_Night_bc"] = sleep_bc
    df["Avg_Daily_Usage_Hours_bc"] = usage_bc

    # usage x unlocks, sleep / usage, study+activity-usage, stress x usage
    df["Digital_Saturation"] = df["Avg_Daily_Usage_Hours_bc"] * df["Daily_Unlocks"]
    df["Restorative_Ratio"] = df["Sleep_Hours_Per_Night_bc"] / (df["Avg_Daily_Usage_Hours_bc"].abs() + 1.0)
    df["Behavioral_Balance"] = (
        df["Study_Hours"] + df["Physical_Activity_Hours"] - df["Avg_Daily_Usage_Hours_bc"]
    )
    df["Stress_Usage_Coupling"] = df["Stress_num"] * df["Avg_Daily_Usage_Hours_bc"]

    cat_maps: Dict[str, Dict[str, int]] = {}
    for col in EMBED_COLS:
        cats = sorted(df[col].astype(str).unique())
        cat_maps[col] = {c: i for i, c in enumerate(cats)}

    country_names = sorted(df["Country"].astype(str).unique())
    country_map = {c: i for i, c in enumerate(country_names)}
    df["Country_idx"] = df["Country"].map(country_map)

    cont_matrix = df[CONTINUOUS_COLS].astype(float).values
    cont_mean = cont_matrix.mean(axis=0)
    cont_std = cont_matrix.std(axis=0)
    cont_std[cont_std < 1e-8] = 1.0
    df["_cont_scaled"] = list((cont_matrix - cont_mean) / cont_std)

    artifacts = PreprocessArtifacts(
        boxcox_lambda={"Sleep_Hours_Per_Night": lam_sleep, "Avg_Daily_Usage_Hours": lam_usage},
        continuous_mean=cont_mean,
        continuous_std=cont_std,
        cat_maps=cat_maps,
        country_names=country_names,
    )
    return df, artifacts


def build_arrays(df: pd.DataFrame, artifacts: PreprocessArtifacts):
    cat_indices = np.column_stack([
        df[col].astype(str).map(artifacts.cat_maps[col]).astype(int).values for col in EMBED_COLS
    ])
    continuous = np.vstack(df["_cont_scaled"].values).astype(np.float32)
    targets = df[TARGET_COL].values.astype(np.float32)
    countries = df["Country_idx"].values.astype(np.int64)
    return cat_indices, continuous, targets, countries


def split_data(
    cat: np.ndarray,
    cont: np.ndarray,
    y: np.ndarray,
    countries: np.ndarray,
    df: pd.DataFrame | None = None,
) -> Dict[str, Tuple]:
    idx = np.arange(len(y))
    train_idx, temp_idx = train_test_split(idx, test_size=(1 - TRAIN_RATIO), random_state=SEED)
    val_size = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_idx, test_idx = train_test_split(temp_idx, test_size=(1 - val_size), random_state=SEED)

    def pack(idxs):
        base = cat[idxs], cont[idxs], y[idxs], countries[idxs]
        if df is not None:
            meta = df.iloc[idxs][["Stress_Level", "Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night"]].reset_index(drop=True)
            return (*base, meta)
        return base

    return {"train": pack(train_idx), "val": pack(val_idx), "test": pack(test_idx)}


def make_loaders(splits: Dict[str, Tuple], batch_size: int) -> Dict[str, DataLoader]:
    loaders = {}
    for name, packed in splits.items():
        cat, cont, y, c = packed[:4]
        ds = MentalHealthDataset(cat, cont, y, c)
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            drop_last=(name == "train"),
        )
    return loaders


def get_test_meta(splits: Dict) -> pd.DataFrame:
    if len(splits["test"]) > 4:
        return splits["test"][4]
    return pd.DataFrame()


def save_artifacts(artifacts: PreprocessArtifacts) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "preprocess_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    meta = {
        "boxcox_lambda": artifacts.boxcox_lambda,
        "continuous_cols": CONTINUOUS_COLS,
        "base_continuous_cols": BASE_CONTINUOUS_COLS,
        "derived_cols": DERIVED_COLS,
        "embed_cols": EMBED_COLS,
        "n_countries": len(artifacts.country_names),
        "category_counts": {c: len(artifacts.cat_maps[c]) for c in EMBED_COLS},
    }
    with open(RESULTS_DIR / "preprocess_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def prepare_data(batch_size: int):
    df_raw = load_raw_dataframe()
    df, artifacts = engineer_features(df_raw)
    cat, cont, y, countries = build_arrays(df, artifacts)
    splits = split_data(cat, cont, y, countries, df=df)
    loaders = make_loaders(splits, batch_size)
    save_artifacts(artifacts)
    test_meta = get_test_meta(splits)
    return loaders, artifacts, df_raw, test_meta
