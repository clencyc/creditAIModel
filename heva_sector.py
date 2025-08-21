# heva_sector.py
from dataclasses import dataclass
import pandas as pd

# 1) Define sector feature weights (edit these as you learn from data/policy)
SECTOR_FEATURE_WEIGHTS = {
    "Film":               {"Amount": 0.9, "Balance": 1.0, "Punctuality_Score": 1.35},
    "Fashion":            {"Amount": 1.0, "Balance": 1.15, "Punctuality_Score": 1.1},
    "Design":             {"Amount": 1.0, "Balance": 1.0, "Punctuality_Score": 1.2},
    "Music":              {"Amount": 1.05, "Balance": 1.0, "Punctuality_Score": 1.15},
    "Media/Broadcast":    {"Amount": 0.95, "Balance": 1.05, "Punctuality_Score": 1.2},
    "General":            {"Amount": 1.0, "Balance": 1.0, "Punctuality_Score": 1.0},
}

# 2) Simple sector probability calibration (post-model)
#    p' = clip(alpha * p + beta, 0, 1)
SECTOR_PROBA_CALIB = {
    "Film":            {"alpha": 1.05, "beta": -0.02},
    "Fashion":         {"alpha": 1.00, "beta":  0.00},
    "Design":          {"alpha": 1.02, "beta": -0.01},
    "Music":           {"alpha": 1.03, "beta": -0.01},
    "Media/Broadcast": {"alpha": 1.00, "beta":  0.00},
    "General":         {"alpha": 1.00, "beta":  0.00},
}

def weight_features(df: pd.DataFrame, sector_col: str = "Sector") -> pd.DataFrame:
    """Create sector-weighted features alongside raw ones."""
    required = ["Amount", "Balance", "Punctuality_Score", sector_col]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df = df.copy()
    # Create weighted versions (suffix _w)
    df["Amount_w"] = df.apply(lambda r: r["Amount"] * SECTOR_FEATURE_WEIGHTS.get(
        str(r[sector_col]), SECTOR_FEATURE_WEIGHTS["General"])["Amount"], axis=1)

    df["Balance_w"] = df.apply(lambda r: r["Balance"] * SECTOR_FEATURE_WEIGHTS.get(
        str(r[sector_col]), SECTOR_FEATURE_WEIGHTS["General"])["Balance"], axis=1)

    df["Punctuality_Score_w"] = df.apply(lambda r: r["Punctuality_Score"] * SECTOR_FEATURE_WEIGHTS.get(
        str(r[sector_col]), SECTOR_FEATURE_WEIGHTS["General"])["Punctuality_Score"], axis=1)

    return df

def calibrate_probability(p: float, sector: str) -> float:
    """Apply simple sector calibration to model probability."""
    cal = SECTOR_PROBA_CALIB.get(str(sector), SECTOR_PROBA_CALIB["General"])
    p_new = cal["alpha"] * float(p) + cal["beta"]
    return max(0.0, min(1.0, p_new))
