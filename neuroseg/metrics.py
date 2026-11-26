"""Patient-cluster bootstrap: resample patients, not correlated slices."""
import numpy as np
import pandas as pd

METRICS = ("dice", "iou", "precision", "recall", "specificity")

def patient_summary(rows, seed=42, draws=2000):
    if rows.empty or draws < 1:
        raise ValueError("Non-empty rows and positive draws required")
    patients = rows.groupby("patient_group")[list(METRICS)].mean()
    values = patients.to_numpy()
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    lo, hi = np.quantile(boot, [.025, .975], axis=0)
    return {
        "slice_macro": rows[list(METRICS)].mean().to_dict(),
        "patient_macro": patients.mean().to_dict(),
        "patient_bootstrap_95_ci": {m: [float(a), float(b)] for m, a, b in zip(METRICS, lo, hi)},
        "by_tumor_type": rows.groupby("tumor_type")[list(METRICS)].mean().to_dict(orient="index"),
        "by_tumor_type_patient_macro": rows.groupby(["tumor_type", "patient_group"])[list(METRICS)].mean().groupby(level=0).mean().to_dict(orient="index"),
    }
