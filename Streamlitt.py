# app.py
import io
import os
import joblib
import random
import tempfile
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
from typing import Optional

# PDF creation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# Your project modules
from heva_data import read_mpesa_csv, read_bills_csv, read_bank_zip, add_features
from heva_model import train_sector_model
from heva_sector import weight_features, calibrate_probability

from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError

# ---------------------------
# Config & constants
# ---------------------------
st.set_page_config(page_title="HEVA Integrated Credit Intelligence", layout="wide")
st.title("🌍 HEVA Credit Intelligence – Sector-Aware AI (Calibrated)")

MODEL_FILENAME = "heva_trained_model.joblib"  # saved bundle (model + encoders)
REQUIRED_BASE_COLS = ["Account_Number", "Amount", "Balance", "Date"]

# ---------------------------
# Helpers
# ---------------------------
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
        df["Punctuality_Score"] = 0.5

    if "Account_Number" not in df.columns:
        df["Account_Number"] = "UNKNOWN"

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        df["Date"] = pd.NaT

    return df.reset_index(drop=True)

def _safe_read(fn, file_obj, label: str) -> Optional[pd.DataFrame]:
    """Wrap your heva_data readers and guarantee a DataFrame or None."""
    if not file_obj:
        return None
    try:
        df = fn(file_obj)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            st.warning(f"⚠️ {label} reader returned no usable data. Skipping.")
            return None
        return _ensure_dtypes(df)
    except Exception as e:
        st.error(f"❌ Failed to read {label}: {e}")
        return None

def _concat_non_empty(dfs: list[Optional[pd.DataFrame]]) -> pd.DataFrame:
    valid = [d for d in dfs if d is not None and not d.empty]
    if not valid:
        return pd.DataFrame()
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

def fig_to_png_bytes(fig):
    """Convert Plotly figure to PNG bytes (tries to use kaleido)."""
    try:
        img_bytes = fig.to_image(format="png", scale=1)
        return img_bytes
    except Exception:
        try:
            # fallback using write_image to temporary file
            with tempfile.TemporaryDirectory() as td:
                tmp = os.path.join(td, "tmp.png")
                fig.write_image(tmp)
                with open(tmp, "rb") as f:
                    return f.read()
        except Exception:
            return None

def generate_pdf_report(df: pd.DataFrame, figs: list, prediction_text: str) -> io.BytesIO:
    """Create a simple PDF report with summary, figures (png), and prediction text."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("HEVA Credit Intelligence - Report", styles['Title']))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(f"Total Records: {len(df)}", styles['Normal']))
    elements.append(Paragraph(f"Sectors: {', '.join(sorted(map(str, df['Sector'].unique())))}", styles['Normal']))
    elements.append(Spacer(1, 12))

    for fig in figs:
        png = fig_to_png_bytes(fig)
        if png:
            img_buf = io.BytesIO(png)
            # scale images to fit reasonably in the PDF
            elements.append(Image(img_buf, width=450, height=280))
            elements.append(Spacer(1, 12))
        else:
            elements.append(Paragraph("⚠️ Could not render a chart image for PDF.", styles['Normal']))
            elements.append(Spacer(1, 8))

    elements.append(Paragraph("User Prediction", styles['Heading2']))
    elements.append(Paragraph(prediction_text, styles['Normal']))

    doc.build(elements)
    buf.seek(0)
    return buf

# ---------------------------
# Sidebar: tools and model upload
# ---------------------------
with st.sidebar:
    st.header("🧰 Tools & Model")
    st.write("Upload accepted file types: CSV, XLSX (for M-Pesa/Bills) and ZIP (for bank statements).")
    st.markdown("---")

    # Model import
    uploaded_model = st.file_uploader("Upload Trained Model (.joblib/.pkl)", type=["joblib", "pkl"])
    if uploaded_model is not None:
        try:
            loaded = joblib.load(uploaded_model)
            # Expect loaded to be dict with keys: model, le_sector, feature_label_encoder
            st.session_state["model_bundle"] = loaded
            st.success("✅ Uploaded model loaded into session.")
        except Exception as e:
            st.error(f"❌ Failed to load uploaded model: {e}")

    st.markdown("---")
    st.write("If you need a sector mapping template, upload some data and use the button on the main page.")

# ---------------------------
# Main uploads
# ---------------------------
col1, col2 = st.columns(2)
with col1:
    mpesa = st.file_uploader("Upload M-Pesa (CSV/XLSX)", type=["csv", "xlsx"])
    bills = st.file_uploader("Upload Bills (CSV/XLSX)", type=["csv", "xlsx"])
with col2:
    bankz = st.file_uploader("Upload Bank ZIP (PDFs inside)", type=["zip"])
    sector_file = st.file_uploader("Upload Sector Mapping CSV (Account_Number → Sector)", type=["csv"])

# Sector template generator button
if st.button("Generate Sector Mapping Template from Uploaded Data"):
    preview_parts = []
    preview_parts.append(_safe_read(read_mpesa_csv, mpesa, "M-Pesa") if mpesa else None)
    preview_parts.append(_safe_read(read_bills_csv, bills, "Bills") if bills else None)
    preview_parts.append(_safe_read(read_bank_zip, bankz, "Bank ZIP") if bankz else None)
    merged_preview = _concat_non_empty(preview_parts)
    if merged_preview.empty:
        st.warning("Upload at least one source (M-Pesa/Bills/Bank) first to build a template.")
    else:
        accs = merged_preview["Account_Number"].astype(str).fillna("UNKNOWN").drop_duplicates().tolist()
        payload = _make_sector_template(accs)
        st.download_button("⬇️ Download Sector Mapping CSV", data=payload, file_name="sector_mapping_template.csv", mime="text/csv")

# ---------------------------
# Read uploaded data
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
# Placeholder label and features
# ---------------------------
try:
    df["Risk_Label"] = (df["Balance"] > 500).astype(int)
except Exception:
    df["Risk_Label"] = 0

try:
    df, feature_label_encoder = add_features(df)
except Exception as e:
    st.error(f"❌ add_features failed: {e} — proceeding with minimal features.")
    df = _ensure_dtypes(df)
    feature_label_encoder = LabelEncoder().fit(["General"])

# ---------------------------
# Sector mapping
# ---------------------------
if sector_file:
    try:
        mapping_df = pd.read_csv(sector_file)
        if {"Account_Number", "Sector"}.issubset(mapping_df.columns):
            sector_map = mapping_df.set_index("Account_Number")["Sector"].to_dict()
            df["Sector"] = df["Account_Number"].map(sector_map).fillna("General")
            st.success("✅ Sector mapping applied.")
        else:
            st.warning("⚠️ Mapping CSV must have columns: Account_Number, Sector. Defaulting to 'General'.")
            df["Sector"] = "General"
    except Exception as e:
        st.error(f"❌ Failed to read mapping file: {e}")
        df["Sector"] = "General"
else:
    df["Sector"] = df.get("Sector", "General")

# encode sector
le_sector = LabelEncoder()
try:
    df["Sector_Code"] = le_sector.fit_transform(df["Sector"].astype(str))
except Exception:
    df["Sector_Code"] = 0

# ---------------------------
# Sector weighting
# ---------------------------
try:
    df = weight_features(df, sector_col="Sector")
except Exception as e:
    st.warning(f"⚠️ weight_features failed: {e}. Creating fallback weighted columns.")
    for col in ["Amount", "Balance", "Punctuality_Score"]:
        df[f"{col}_w"] = df.get(col, 0.0)

st.success(f"✅ {len(df)} records processed across sectors: {', '.join(sorted(map(str, df['Sector'].unique())))}")

# ---------------------------
# Quick interactive charts (Plotly)
# ---------------------------
st.subheader("📈 Data Insights (Interactive)")

fig1 = px.histogram(df, x="Balance", nbins=30, title="Balance Distribution", labels={"Balance": "Balance"})
fig2 = px.bar(df["Sector"].value_counts().reset_index(), x="index", y="Sector", labels={"index": "Sector", "Sector": "Count"}, title="Records per Sector")
fig3 = px.histogram(df, x="Sector", color="Risk_Label", barmode="stack", title="Risk Labels by Sector", labels={"Risk_Label": "Risk Label"})

st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)
st.plotly_chart(fig3, use_container_width=True)

# ---------------------------
# Training / Load model logic
# ---------------------------
feature_cols = ["Amount_w", "Balance_w", "Punctuality_Score_w", "Sector_Code"]
for c in feature_cols:
    if c not in df.columns:
        df[c] = 0.0

# Ensure at least two classes for training
if df["Risk_Label"].nunique() < 2:
    st.warning("⚠️ Only one class present in Risk_Label. Adding a tiny synthetic minority class for demo training.")
    flip_n = max(1, int(0.01 * len(df)))
    df.loc[df.sample(flip_n, random_state=42).index, "Risk_Label"] = 1 - df["Risk_Label"].iloc[0]

# Train model (cached)
@st.cache_resource
def _train_cached(df_snapshot, feature_cols_snapshot):
    # df_snapshot is pickled by Streamlit cache system; ensure train_sector_model accepts DataFrame
    return train_sector_model(df_snapshot, feature_cols_snapshot)

# Option: load model bundle from session (if user uploaded)
model_bundle = st.session_state.get("model_bundle", None)

col_train_left, col_train_right = st.columns([2,1])
with col_train_left:
    retrain = st.checkbox("🔄 Retrain Model (force)", value=False)
    if st.button("🧠 Train / (Re)Train Model"):
        try:
            model = _train_cached(df, feature_cols)
            # Build bundle with encoders to make predictions portable
            bundle = {
                "model": model,
                "le_sector": le_sector,
                "feature_label_encoder": feature_label_encoder
            }
            joblib.dump(bundle, MODEL_FILENAME)
            st.session_state["model_bundle"] = bundle
            st.success("✅ Model trained and saved to disk & session.")
        except Exception as e:
            st.error(f"❌ Model training failed: {e}")
with col_train_right:
    if os.path.exists(MODEL_FILENAME) and st.button("📂 Load Saved Model From Disk"):
        try:
            bundle = joblib.load(MODEL_FILENAME)
            st.session_state["model_bundle"] = bundle
            st.success("✅ Loaded saved model into session.")
        except Exception as e:
            st.error(f"❌ Failed to load saved model: {e}")

# If a model bundle wasn't uploaded or loaded, train automatically (unless user wants manual control)
if "model_bundle" not in st.session_state:
    try:
        model_auto = _train_cached(df, feature_cols)
        st.session_state["model_bundle"] = {"model": model_auto, "le_sector": le_sector, "feature_label_encoder": feature_label_encoder}
        # Save to disk for portability
        try:
            joblib.dump(st.session_state["model_bundle"], MODEL_FILENAME)
        except Exception:
            pass
    except Exception as e:
        st.error(f"❌ Automatic model training failed: {e}")
        st.stop()

# Expose model
bundle = st.session_state["model_bundle"]
model = bundle.get("model")
le_sector_saved = bundle.get("le_sector", le_sector)
feature_label_encoder_saved = bundle.get("feature_label_encoder", feature_label_encoder)

# Download model button
with st.expander("📥 Export / Share Model"):
    if model is not None:
        try:
            with open(MODEL_FILENAME, "rb") as mf:
                st.download_button("⬇️ Download Trained Model Bundle", data=mf, file_name="heva_model_bundle.joblib", mime="application/octet-stream")
        except Exception as e:
            st.warning(f"Could not provide model file for download: {e}")
    else:
        st.info("No model available to download yet.")

# ---------------------------
# Inference UI
# ---------------------------
st.subheader("📊 Predict Credit Risk (Sector-Calibrated)")

left, right = st.columns([2,1])
with left:
    amt = st.number_input("Transaction Amount", value=1000.0, step=100.0, help="KES")
    bal = st.number_input("Balance", value=500.0, step=50.0, help="KES")
    sector_choice = st.selectbox("Sector", options=sorted(map(str, df["Sector"].unique())))
    punct = st.slider("Punctuality Score", 0.0, 1.0, 0.5, 0.01, help="1.0 = always on time; 0.0 = always late")

with right:
    st.write("**Feature preview (first rows)**")
    preview_cols = [c for c in ["Amount_w", "Balance_w", "Punctuality_Score_w", "Sector", "Risk_Label"] if c in df.columns]
    st.dataframe(df[preview_cols].head(8), use_container_width=True)

# encode sector using saved encoder (fallback to local if not saved)
try:
    sec_code = int(le_sector_saved.transform([sector_choice])[0])
except Exception:
    try:
        sec_code = int(le_sector.transform([sector_choice])[0])
    except Exception:
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
    row_w = row.copy()
    for col in ["Amount", "Balance", "Punctuality_Score"]:
        row_w[f"{col}_w"] = row_w.get(col, 0.0)

X_user = row_w[feature_cols]

# Do inference
try:
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X_user)[0].max())
    else:
        # If no predict_proba, fallback to predict (class only)
        pred_tmp = int(model.predict(X_user)[0])
        proba = 1.0 if pred_tmp == 1 else 0.0
    pred = int(model.predict(X_user)[0])
except Exception as e:
    st.error(f"❌ Inference failed: {e}")
    st.stop()

# Calibration
try:
    proba_cal = float(calibrate_probability(proba, sector_choice))
except Exception:
    proba_cal = proba

pred_cal = 1 if proba_cal >= 0.5 else 0

st.markdown(
    f"**Raw Model:** {'🟢 Low Risk' if pred==1 else '🔴 High Risk'} ({round(proba*100,2)}%)  \n\n"
    f"**Calibrated ({sector_choice}):** {'🟢 Low Risk' if pred_cal==1 else '🔴 High Risk'} ({round(proba_cal*100,2)}%)"
)

# ---------------------------
# Download processed data & PDF report
# ---------------------------
st.subheader("📥 Download Reports & Data")

# CSV
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Processed Data (CSV)", data=csv_bytes, file_name="heva_processed.csv", mime="text/csv")

# PDF report (generate on-demand)
prediction_text = (
    f"Raw Model: {'Low Risk' if pred==1 else 'High Risk'} ({round(proba*100,2)}%) | "
    f"Calibrated ({sector_choice}): {'Low Risk' if pred_cal==1 else 'High Risk'} ({round(proba_cal*100,2)}%)"
)

if st.button("Generate & Download PDF Report"):
    with st.spinner("Generating PDF..."):
        pdf_buf = generate_pdf_report(df, [fig1, fig2, fig3], prediction_text)
        st.download_button("⬇️ Download PDF Report", data=pdf_buf, file_name="heva_report.pdf", mime="application/pdf")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption(
    "© 2025 HEVA Credit Intelligence | Built with ❤️ to democratize access to credit "
    "for creative enterprises. This app accepts CSV/XLSX/ZIP inputs; features are defaulted safely if missing."
)
