<<<<<<< HEAD
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
=======
import os
import zipfile
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Scoring Data Tool", layout="wide")
st.title("📊 Credit Scoring Data Analyzer")

# -----------------------
# Helpers
# -----------------------
def unzip_file(zip_file, extract_dir="unzipped_pdfs"):
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_file, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir

def parse_bank_statement_text(text):
    lines = text.split("\n")
    records = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            try:
                date = pd.to_datetime(parts[0], errors="coerce")
                if pd.notnull(date):
                    desc = " ".join(parts[1:-3])
                    debit = float(parts[-3]) if parts[-3].replace(".", "", 1).isdigit() else 0
                    credit = float(parts[-2]) if parts[-2].replace(".", "", 1).isdigit() else 0
                    balance = float(parts[-1]) if parts[-1].replace(".", "", 1).isdigit() else 0
                    records.append([date, desc, debit, credit, balance])
            except:
                pass
    return pd.DataFrame(records, columns=["Date", "Description", "Debit", "Credit", "Balance"])

def standardize_date_column(df, possible_names):
    """Rename any matching column to 'Date', or create if missing."""
    for name in possible_names:
        if name in df.columns:
            df = df.rename(columns={name: "Date"})
            break
    if "Date" not in df.columns:
>>>>>>> parent of 0c6368d (Update Streamlitt.py)
        df["Date"] = pd.NaT

<<<<<<< HEAD
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

# Safely display success message with sector information
try:
    if 'Sector' in df.columns and not df['Sector'].empty:
        # Handle NaN values and convert to string safely
        unique_sectors = df['Sector'].dropna().astype(str).unique()
        if len(unique_sectors) > 0:
            sectors_str = ', '.join(sorted(unique_sectors))
            st.success(f"✅ {len(df)} records processed across sectors: {sectors_str}")
        else:
            st.success(f"✅ {len(df)} records processed (no sector information available)")
    else:
        st.success(f"✅ {len(df)} records processed (no sector column found)")
except Exception as e:
    st.success(f"✅ {len(df)} records processed (sector display error: {str(e)})")

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
=======
def read_mpesa_csv(file):
    df = pd.read_csv(file)
    df = standardize_date_column(df, ["Date", "Transaction Date", "Date & Time"])
    df = df.rename(columns={
        "Transaction_Type": "Description",
        "Transaction_Amount": "Amount",
        "Account": "Account_Number"
    })
    if "Balance" not in df.columns:
        df["Balance"] = None
    df["Source_Type"] = "M-Pesa"
    return df[["Account_Number", "Date", "Description", "Amount", "Balance", "Source_Type"]]

def read_bills_csv(file):
    df = pd.read_csv(file)
    df = standardize_date_column(df, ["Date", "Billing_Date", "Payment_Date"])
    df = df.rename(columns={
        "Provider": "Description",
        "Final_Amount_KSh": "Amount",
        "User_ID": "Account_Number"
    })
    if "Balance" not in df.columns:
        df["Balance"] = None
    df["Source_Type"] = "Utility Bill"
    return df[["Account_Number", "Date", "Description", "Amount", "Balance", "Source_Type"]]

# -----------------------
# File Upload Section
# -----------------------
st.header("📂 Upload Your Files")

bank_zip_file = st.file_uploader("Upload Bank Statements ZIP", type=["zip"])
mpesa_file = st.file_uploader("Upload M-Pesa CSV", type=["csv"])
bills_file = st.file_uploader("Upload Utility Bills CSV", type=["csv"])

merged_df = pd.DataFrame()

# -----------------------
# Bank Statements Parsing
# -----------------------
if bank_zip_file:
    pdf_dir = unzip_file(bank_zip_file)
    bank_records = []
    for fname in os.listdir(pdf_dir):
        if fname.lower().endswith(".pdf"):
            pdf = PdfReader(os.path.join(pdf_dir, fname))
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            df = parse_bank_statement_text(text)
            if not df.empty:
                df["Account_Number"] = fname.split(".")[0]
                df["Source_Type"] = "Bank Statement"
                df["Amount"] = df["Credit"] - df["Debit"]
                bank_records.append(df)
    if bank_records:
        bank_df = pd.concat(bank_records, ignore_index=True)
        bank_df = standardize_date_column(bank_df, ["Date", "Statement_Date", "Txn Date"])
        bank_df = bank_df[["Account_Number", "Date", "Description", "Amount", "Balance", "Source_Type"]]
    else:
        bank_df = pd.DataFrame()

# -----------------------
# Merge Data
# -----------------------
dfs = []
if bank_zip_file and not bank_df.empty:
    dfs.append(bank_df)
if mpesa_file:
    dfs.append(read_mpesa_csv(mpesa_file))
if bills_file:
    dfs.append(read_bills_csv(bills_file))

if dfs:
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df["Date"] = pd.to_datetime(merged_df["Date"], errors="coerce")
    merged_df = merged_df.sort_values("Date")

# -----------------------
# Display & Download
# -----------------------
if not merged_df.empty:
    st.subheader("📋 Merged Data Preview")
    st.dataframe(merged_df.head(50))

    csv_data = merged_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Merged CSV", csv_data, "merged_data.csv", "text/csv")

    # -----------------------
    # Plot
    # -----------------------
    st.subheader("📈 Transaction Trend")
    fig, ax = plt.subplots(figsize=(10, 4))
    merged_df.groupby("Date")["Amount"].sum().plot(ax=ax)
    ax.set_title("Daily Net Amount")
    ax.set_ylabel("Amount")
    st.pyplot(fig)
else:
    st.info("Upload at least one file to see results.")
>>>>>>> parent of 0c6368d (Update Streamlitt.py)
