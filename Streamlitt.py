# app.py
import streamlit as st
import pandas as pd
from heva_data import read_mpesa_csv, read_bills_csv, read_bank_zip, add_features
from heva_model import train_sector_model
from heva_sector import weight_features, calibrate_probability

# ---------------------------
# Streamlit Config
# ---------------------------
st.set_page_config(page_title="HEVA Integrated Credit Intelligence", layout="wide")
st.title("🌍 HEVA Credit Intelligence – Sector-Aware AI (Calibrated)")

# ---------------------------
# Upload Section
# ---------------------------
col1, col2 = st.columns(2)
mpesa = col1.file_uploader("Upload M-Pesa CSV", type=["csv"])
bills = col1.file_uploader("Upload Bills CSV", type=["csv"])
bankz = col2.file_uploader("Upload Bank ZIP (PDFs)", type=["zip"])
sector_file = col2.file_uploader("Upload Sector Mapping CSV (Account_Number → Sector)", type=["csv"])

pieces = []
if mpesa:
    pieces.append(read_mpesa_csv(mpesa))
if bills:
    pieces.append(read_bills_csv(bills))
if bankz:
    pieces.append(read_bank_zip(bankz))

if not pieces:
    st.info("Upload at least one data source to proceed.")
    st.stop()

# ---------------------------
# Merge & Label
# ---------------------------
df = pd.concat(pieces, ignore_index=True)

# Add a simple placeholder target (replace with real labels later)
df["Risk_Label"] = df["Balance"].apply(lambda x: 1 if x > 500 else 0)

# ---------------------------
# Feature Engineering
# ---------------------------
df, le = add_features(df)

# ---------------------------
# Sector Mapping
# ---------------------------
if sector_file:
    try:
        mapping_df = pd.read_csv(sector_file)
        if "Account_Number" in mapping_df.columns and "Sector" in mapping_df.columns:
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

# Re-encode sector
df["Sector_Code"] = le.fit_transform(df["Sector"].astype(str))

# ---------------------------
# Sector Feature Weighting
# ---------------------------
df = weight_features(df, sector_col="Sector")

st.success(f"✅ {len(df)} records processed across sectors: {', '.join(sorted(df['Sector'].unique()))}")

# ---------------------------
# Train Model
# ---------------------------
feature_cols = ["Amount_w", "Balance_w", "Punctuality_Score_w", "Sector_Code"]
model = train_sector_model(df, feature_cols)

# ---------------------------
# Inference UI
# ---------------------------
st.subheader("📊 Predict Credit Risk (Sector-Calibrated)")
amt = st.number_input("Transaction Amount", value=1000.0, step=100.0)
bal = st.number_input("Balance", value=500.0, step=50.0)
sector = st.selectbox("Sector", options=sorted(df["Sector"].unique()))
sec_code = int(le.transform([sector])[0])
punct = st.slider("Punctuality Score", 0.0, 1.0, 0.5, 0.01)

row = pd.DataFrame([{
    "Amount": amt,
    "Balance": bal,
    "Punctuality_Score": punct,
    "Sector": sector,
    "Sector_Code": sec_code
}])

row_w = weight_features(row, sector_col="Sector")
X_user = row_w[feature_cols]

proba = model.predict_proba(X_user)[0].max()
pred = model.predict(X_user)[0]

# Apply sector probability calibration (α/β)
proba_cal = calibrate_probability(proba, sector)
pred_cal = 1 if proba_cal >= 0.5 else 0

st.markdown(
    f"**Raw Model:** {'🟢 Low Risk' if pred==1 else '🔴 High Risk'} "
    f"({round(proba*100,2)}%)\n\n"
    f"**Calibrated ({sector}):** {'🟢 Low Risk' if pred_cal==1 else '🔴 High Risk'} "
    f"({round(proba_cal*100,2)}%)"
)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption(
    "© 2025 HEVA Credit Intelligence | Built with ❤️ to democratize access "
    "to credit for creative enterprises. Date column is auto-standardized "
    "to prevent errors."
)
