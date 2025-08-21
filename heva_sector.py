# heva_sector.py
from dataclasses import dataclass
import pandas as pd

# ---------------------------
# 1) Define sector feature weights (tweak these as you learn from data/policy)
# ---------------------------
SECTOR_FEATURE_WEIGHTS = {
    "Film":               {"Amount": 0.9,  "Balance": 1.0,  "Punctuality_Score": 1.35},
    "Fashion":            {"Amount": 1.0,  "Balance": 1.15, "Punctuality_Score": 1.1},
    "Design":             {"Amount": 1.0,  "Balance": 1.0,  "Punctuality_Score": 1.2},
    "Music":              {"Amount": 1.05, "Balance": 1.0,  "Punctuality_Score": 1.15},
    "Media/Broadcast":    {"Amount": 0.95, "Balance": 1.05, "Punctuality_Score": 1.2},
    "General":            {"Amount": 1.0,  "Balance": 1.0,  "Punctuality_Score": 1.0},
}

# ---------------------------
# 2) Define sector probability calibration (post-model adjustment)
#    Formula: p' = clip(alpha * p + beta, 0, 1)
# ---------------------------
SECTOR_PROBA_CALIB = {
    "Film":            {"alpha": 1.05, "beta": -0.02},
    "Fashion":         {"alpha": 1.00, "beta":  0.00},
    "Design":          {"alpha": 1.02, "beta": -0.01},
    "Music":           {"alpha": 1.03, "beta": -0.01},
    "Media/Broadcast": {"alpha": 1.00, "beta":  0.00},
    "General":         {"alpha": 1.00, "beta":  0.00},
}

# ---------------------------
# 3) Feature weighting
# ---------------------------
def weight_features(df: pd.DataFrame, sector_col: str = "Sector") -> pd.DataFrame:
    """
    Create sector-weighted features alongside raw ones.
    Safely handles duplicate columns and missing values.
    """
    # Defensive: remove duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()].copy()

    required = ["Amount", "Balance", "Punctuality_Score", sector_col]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Helper to fetch sector-specific weight (fallback = General)
    def get_weight(sector, feature):
        return SECTOR_FEATURE_WEIGHTS.get(str(sector), SECTOR_FEATURE_WEIGHTS["General"]).get(feature, 1.0)

    # Create weighted versions (suffix _w)
    df["Amount_w"] = df.apply(lambda r: r["Amount"] * get_weight(r[sector_col], "Amount"), axis=1)
    df["Balance_w"] = df.apply(lambda r: r["Balance"] * get_weight(r[sector_col], "Balance"), axis=1)
    df["Punctuality_Score_w"] = df.apply(lambda r: r["Punctuality_Score"] * get_weight(r[sector_col], "Punctuality_Score"), axis=1)

    # Final cleanup: ensure unique column names
    df = df.loc[:, ~df.columns.duplicated()].copy()

        # --- Assign sector (placeholder until real mappings exist) ---
    if "Sector" not in df.columns:
        df["Sector"] = "General"   # Default fallback sector

    # Encode Sector → Sector_Code
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df["Sector_Code"] = le.fit_transform(df["Sector"].astype(str))

def add_features(df):
    # (your existing engineered features code …)

    # Punctuality scores…
    punctuality_scores = df.groupby("Account_Number")["Date"].apply(calc_punctuality).to_dict()
    df["Punctuality_Score"] = df["Account_Number"].map(punctuality_scores)

    # --- Assign sector (default General if missing) ---
    if "Sector" not in df.columns:
        df["Sector"] = "General"

    # Encode sector
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df["Sector_Code"] = le.fit_transform(df["Sector"].astype(str))

    return df, le

# ---------------------------
# 4) Probability calibration
# ---------------------------
def calibrate_probability(p: float, sector: str) -> float:
    """
    Apply simple sector calibration to model probability.
    p' = alpha * p + beta, clipped to [0,1].
    """
    cal = SECTOR_PROBA_CALIB.get(str(sector), SECTOR_PROBA_CALIB["General"])
    p_new = cal["alpha"] * float(p) + cal["beta"]
    return max(0.0, min(1.0, p_new))
