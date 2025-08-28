# app.py
import io
import random
import streamlit as st
import pandas as pd

from heva_data import read_mpesa_csv, read_bills_csv, read_bank_zip, add_features
from heva_model import train_sector_model
from heva_sector import weight_features, calibrate_probability

from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError

# ---------------------------
# Streamlit Config
# ---------------------------
st.set_page_config(page_title="HEVA Integrated Credit Intelligence", layout="wide")
st.title("🌍 HEVA Credit Intelligence – Sector-Aware AI (Calibrated)")

# ---------------------------
# Helpers (robustness)
# ---------------------------
REQUIRED_BASE_COLS = ["Account_Number", "Amount", "Balance", "Date"]

def _ensure_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce critical columns to expected dtypes and fill safe defaults."""
    if "Amount" in df.columns:
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    else:
        df["Amount"] = 0.0

    if "Balance" in df.columns:
        df["Balance"] = pd.to_numeric(df["Balance"], errors="coerce").fillna(0.0)
    else:
        df["Balance"] = 0.0

    if "Punctuality_Score" not in df.columns:
        # Safe placeholder until you compute it properly in heva_data
        df["Punctuality_Score"] = 0.5

    if "Account_Number" not in df.columns:
        # Provide a fallback ID to avoid map failures later
        df["Account_Number"] = "UNKNOWN"

    # Standardize/parse Date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        df["Date"] = pd.NaT

    # Make sure the index is clean to avoid InvalidIndexError
    return df.reset_index(drop=True)

def _safe_read(fn, file_obj, label: str) -> pd.DataFrame | None:
    """Wrap your heva_data readers and guarantee a DataFrame or None."""
    if not file_obj:
        return None
    try:
        df = fn(file_obj)
        if df is None:
            st.warning(f"⚠️ {label} reader returned None. Skipping.")
            return None
        if not isinstance(df, pd.DataFrame):
            st.warning(f"⚠️ {label} reader did not return a DataFrame. Skipping.")
            return None
        if df.empty:
            st.warning(f"⚠️ {label} is empty after parsing. Skipping.")
            return None
        return _ensure_dtypes(df)
    except Exception as e:
        st.error(f"❌ Failed to read {label}: {e}")
        return None

def _concat_non_empty(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [d for d in dfs if d is not None and not d.empty]
    if not valid:
        return pd.DataFrame()
    # Drop duplicate columns if any helper returns overlapping names weirdly
    # (Prevents InvalidIndexError during later reindex/merge operations)
    out = pd.concat(valid, ignore_index=True)
    out = out.loc[:, ~out.columns.duplicated(keep="first")]
    return out.reset_index(drop=True)

def _make_sector_template(account_numbers: list[str]) -> bytes:
    sectors = ["Music", "Film", "Fashion", "Design", "Gaming", "Photography", "General"]
    mapping = pd.DataFrame({
        "Account_Number": account_numbers,
        "Sector": [random.choice(sectors) for _ in account_numbers]
    })
    buf = io.StringIO()
    mapping.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

# ---------------------------
# Upload Section + Sidebar tools
# ---------------------------
col1, col2 = st.columns(2)
mpesa = col1.file_uploader("Upload M-Pesa CSV", type=["csv"])
bills = col1.file_uploader("Upload Bills CSV", type=["csv"])
bankz = col2.file_uploader("Upload Bank ZIP (PDFs)", type=["zip"])
sector_file = col2.file_uploader("Upload Sector Mapping CSV (Account_Number → Sector)", type=["csv"])

with st.sidebar:
    st.header("🧰 Tools")
    if st.button("Generate Sector Mapping Template from Uploaded Data"):
        # Build a quick preview df from whatever is uploaded so far
        preview_parts = []
        preview_parts.append(_safe_read(read_mpesa_csv, mpesa, "M-Pesa")) if mpesa else None
        preview_parts.append(_safe_read(read_bills_csv, bills, "Bills")) if bills else None
        preview_parts.append(_safe_read(read_bank_zip, bankz, "Bank ZIP")) if bankz else None

        merged_preview = _concat_non_empty(preview_parts)
        if merged_preview.empty:
            st.warning("Upload at least one source (M-Pesa/Bills/Bank) first to build a template.")
        else:
            accs = (
                merged_preview["Account_Number"]
                .astype(str)
                .fillna("UNKNOWN")
                .drop_duplicates()
                .tolist()
            )
            payload = _make_sector_template(accs)
            st.download_button(
                "⬇️ Download Sector Mapping CSV",
                data=payload,
                file_name="sector_mapping_template.csv",
                mime="text/csv"
            )

# ---------------------------
# Read Uploaded Data
# ---------------------------
pieces = []
pieces.append(_safe_read(read_mpesa_csv, mpesa, "M-Pesa"))
pieces.append(_safe_read(read_bills_csv, bills, "Bills"))
pieces.append(_safe_read(read_bank_zip, bankz, "Bank ZIP"))

df = _concat_non_empty(pieces)
if df.empty:
    st.info("Upload at least one data source to proceed.")
    st.stop()

# ---------------------------
# Basic Label (placeholder)
# ---------------------------
# NOTE: Replace with real labels once you have repayment outcomes.
try:
    df["Risk_Label"] = (df["Balance"] > 500).astype(int)
except KeyError:
    df["Risk_Label"] = 0

# ---------------------------
# Feature Engineering
# ---------------------------
try:
    df, feature_label_encoder = add_features(df)  # your function
except Exception as e:
    st.error(f"❌ add_features failed: {e}")
    # Fail-safe: keep going with minimal features
    df = _ensure_dtypes(df)
    feature_label_encoder = LabelEncoder().fit(["General"])

# ---------------------------
# Sector Mapping
# ---------------------------
if sector_file:
    try:
        mapping_df = pd.read_csv(sector_file)
        if {"Account_Number", "Sector"}.issubset(mapping_df.columns):
            sector_map = mapping_df.set_index("Account_Number")["Sector"].to_dict()
            df["Sector"] = df["Account_Number"].map(sector_map).fillna("General")
            st.success("✅ Sector mapping applied from uploaded file.")
        else:
            st.warning("⚠️ Mapping CSV must have columns: Account_Number, Sector. Defaulting to 'General'.")
            df["Sector"] = "General"
    except Exception as e:
        st.error(f"❌ Failed to read mapping file: {e}")
        df["Sector"] = "General"
else:
    df["Sector"] = "General"

# Encode sector with its own encoder (don't reuse the one from add_features)
le_sector = LabelEncoder()
try:
    df["Sector_Code"] = le_sector.fit_transform(df["Sector"].astype(str))
except Exception:
    # If something odd happens, default everything to a single code
    df["Sector_Code"] = 0

# ---------------------------
# Sector Feature Weighting
# ---------------------------
try:
    df = weight_features(df, sector_col="Sector")  # your function
except Exception as e:
    st.warning(f"⚠️ weight_features failed, using unweighted features. Details: {e}")
    # Fallback: create *_w columns by copying the originals
    for col in ["Amount", "Balance", "Punctuality_Score"]:
        df[f"{col}_w"] = df.get(col, pd.Series([0.0]*len(df)))

st.success(f"✅ {len(df)} records processed across sectors: {', '.join(sorted(map(str, df['Sector'].unique())))}")

# ---------------------------
# Train Model
# ---------------------------
feature_cols = ["Amount_w", "Balance_w", "Punctuality_Score_w", "Sector_Code"]

# Make sure all required columns exist
for c in feature_cols:
    if c not in df.columns:
        df[c] = 0.0

# Guard against single-class labels (sklearn needs ≥2 classes)
if df["Risk_Label"].nunique() < 2:
    st.warning("⚠️ Only one class present in Risk_Label. Adding a synthetic minority class for demo training.")
    # naive synthetic flip for 1% of rows
    flip_n = max(1, int(0.01 * len(df)))
    df.loc[df.sample(flip_n, random_state=42).index, "Risk_Label"] = 1 - df["Risk_Label"].iloc[0]

try:
    model = train_sector_model(df, feature_cols)  # your function
except Exception as e:
    st.error(f"❌ Model training failed: {e}")
    st.stop()

# ---------------------------
# Inference UI
# ---------------------------
st.subheader("📊 Predict Credit Risk (Sector-Calibrated)")

left, right = st.columns([2, 1])
with left:
    amt = st.number_input("Transaction Amount", value=1000.0, step=100.0, help="KES")
    bal = st.number_input("Balance", value=500.0, step=50.0, help="KES")
    sector_choice = st.selectbox("Sector", options=sorted(map(str, df["Sector"].unique())))
    punct = st.slider("Punctuality Score", 0.0, 1.0, 0.5, 0.01,
                      help="1.0 = always on time; 0.0 = always late")

with right:
    st.write("**Feature Preview**")
    st.dataframe(df[["Amount_w", "Balance_w", "Punctuality_Score_w", "Sector", "Risk_Label"]].head(8), use_container_width=True)

try:
    sec_code = int(le_sector.transform([sector_choice])[0])
except (NotFittedError, ValueError):
    sec_code = 0

row = pd.DataFrame([{
    "Amount": amt,
    "Balance": bal,
    "Punctuality_Score": punct,
    "Sector": sector_choice,
    "Sector_Code": sec_code
}])

try:
    row_w = weight_features(row, sector_col="Sector")
except Exception:
    # Fallback mirrors above
    row_w = row.copy()
    for col in ["Amount", "Balance", "Punctuality_Score"]:
        row_w[f"{col}_w"] = row_w.get(col, 0.0)

X_user = row_w[feature_cols]

try:
    proba = float(model.predict_proba(X_user)[0].max())
    pred = int(model.predict(X_user)[0])
except Exception as e:
    st.error(f"❌ Inference failed: {e}")
    st.stop()

# Apply sector probability calibration (α/β)
try:
    proba_cal = float(calibrate_probability(proba, sector_choice))
except Exception:
    proba_cal = proba

pred_cal = 1 if proba_cal >= 0.5 else 0

st.markdown(
    f"**Raw Model:** {'🟢 Low Risk' if pred==1 else '🔴 High Risk'} "
    f"({round(proba*100,2)}%)  \n"
    f"**Calibrated ({sector_choice}):** {'🟢 Low Risk' if pred_cal==1 else '🔴 High Risk'} "
    f"({round(proba_cal*100,2)}%)"
)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption(
    "© 2025 HEVA Credit Intelligence | Built with ❤️ to democratize access "
    "to credit for creative enterprises. Date fields are auto-standardized and "
    "missing columns are safely defaulted to avoid crashes."
)
