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

def _sanitize_df_before_concat(d: pd.DataFrame) -> pd.DataFrame:
    """Make column names unique, drop duplicate columns, reset index."""
    d = d.copy()
    # Drop exact duplicate columns, keep first occurrence
    d = d.loc[:, ~d.columns.duplicated(keep="first")]

    # If columns still not unique (rare), deduplicate by appending suffixes
    if len(d.columns) != len(set(d.columns)):
        new_cols = []
        seen = {}
        for col in d.columns:
            if col in seen:
                seen[col] += 1
                new_cols.append(f"{col}__dup{seen[col]}")
            else:
                seen[col] = 0
                new_cols.append(col)
        d.columns = new_cols

    d.reset_index(drop=True, inplace=True)
    return d

def _concat_non_empty(dfs: list[Optional[pd.DataFrame]]) -> pd.DataFrame:
    """
    Sanitize each valid DataFrame before concatenation to avoid InvalidIndexError.
    Returns an empty DataFrame if nothing valid.
    """
    valid = []
    for d in dfs:
        if d is None:
            continue
        if not isinstance(d, pd.DataFrame):
            continue
        if d.empty:
            continue
        try:
            sanitized = _sanitize_df_before_concat(d)
            valid.append(sanitized)
        except Exception:
            # if sanitization fails for this df, skip it but log a warning
            st.warning("⚠️ Skipping a dataset because it couldn't be sanitized before concat.")
            continue

    if not valid:
        return pd.DataFrame()

    # Now safe to concat
    out = pd.concat(valid, ignore_index=True)
    # final pass to drop any duplicated columns after concat
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
    df_piece = _safe_read(read_mpesa_csv, mpesa, "M-Pesa") if mpesa else None
    if df_piece is not None:
        preview_parts.append(df_piece)
    df_piece = _safe_read(read_bills_csv, bills, "Bills") if bills else None
    if df_piece is not None:
        preview_parts.append(df_piece)
    df_piece = _safe_read(read_bank_zip, bankz, "Bank ZIP") if bankz else None
    if df_piece is not None:
        preview_parts.append(df_piece)

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
df_piece = _safe_read(read_mpesa_csv, mpesa, "M-Pesa") if mpesa else None
if df_piece is not None:
    pieces.append(df_piece)
df_piece = _safe_read(read_bills_csv, bills, "Bills") if bills else None
if df_piece is not None:
    pieces.append(df_piece)
df_piece = _safe_read(read_bank_zip, bankz, "Bank ZIP") if bankz else None
if df_piece is not None:
    pieces.append(df_piece)

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
    st.error(
