# app.py
import os
import re
import zipfile
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="HEVA Credit Intelligence", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
def find_date_col(df):
    """Return the first column name that likely contains dates, or None."""
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ("date", "time", "billing", "payment", "paid", "txn", "transaction")):
            return c
    return None

def standardize_date_column(df):
    """Rename an obvious date column to 'Date', or create 'Date' as NaT."""
    if df is None or df.shape[0] == 0:
        df = pd.DataFrame()
    col = find_date_col(df)
    if col:
        df = df.rename(columns={col: "Date"})
    if "Date" not in df.columns:
        df["Date"] = pd.NaT
    return df

def safe_to_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors="coerce")

# --------------- Bank parsing ---------------
date_line_re = re.compile(r'^\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})')

def parse_bank_statement_text(text):
    """
    Lightweight parser:
    - looks for lines starting with a date (dd/mm/yyyy or yyyy-mm-dd)
    - attempts to extract numeric tokens at the end for debit/credit/balance
    """
    records = []
    for line in text.splitlines():
        m = date_line_re.match(line)
        if not m:
            continue
        date_str = m.group(1)
        try:
            date = pd.to_datetime(date_str, dayfirst=True, errors="coerce")
        except Exception:
            date = pd.to_datetime(date_str, errors="coerce")

        rest = line[m.end():].strip()
        # find numbers (like 1,234.56 or 1234 or 1234.00)
        nums = re.findall(r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\d+', rest)
        # description = rest with numbers removed
        desc = re.sub(r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\d+', '', rest).strip()
        debit = nums[-3] if len(nums) >= 3 else (nums[-2] if len(nums) == 2 else None)
        credit = nums[-2] if len(nums) >= 2 else None
        balance = nums[-1] if len(nums) >= 1 else None

        records.append({
            "Date": date,
            "Description": desc if desc else None,
            "Debit": safe_to_numeric(pd.Series([debit]))[0] if debit is not None else np.nan,
            "Credit": safe_to_numeric(pd.Series([credit]))[0] if credit is not None else np.nan,
            "Balance": safe_to_numeric(pd.Series([balance]))[0] if balance is not None else np.nan,
        })
    return pd.DataFrame(records)

def unzip_and_parse_bank_zip(uploaded_zip):
    """
    Read uploaded ZIP (UploadedFile) in-memory, parse each PDF inside,
    return concatenated DataFrame (or empty DataFrame).
    """
    bank_records = []
    try:
        z = zipfile.ZipFile(BytesIO(uploaded_zip.read()))
    except Exception as e:
        st.error(f"Could not read ZIP: {e}")
        return pd.DataFrame()

    for name in z.namelist():
        if not name.lower().endswith(".pdf"):
            continue
        try:
            pdf_bytes = z.read(name)
            reader = PdfReader(BytesIO(pdf_bytes))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            df = parse_bank_statement_text(text)
            if not df.empty:
                # use filename (without extension) as Account_Number if none found
                df["Account_Number"] = os.path.splitext(os.path.basename(name))[0]
                df["Source_Type"] = "Bank Statement"
                df["Amount"] = df["Credit"].fillna(0) - df["Debit"].fillna(0)
                bank_records.append(df[["Account_Number", "Date", "Description", "Amount", "Balance", "Source_Type"]])
        except Exception as e:
            st.warning(f"Failed parsing {name}: {e}")

    return pd.concat(bank_records, ignore_index=True) if bank_records else pd.DataFrame()

# --------------- CSV readers ---------------
def read_mpesa_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = standardize_date_column(df)
    rename_map = {}
    if "Transaction_Type" in df.columns:
        rename_map["Transaction_Type"] = "Description"
    if "Transaction_Amount" in df.columns:
        rename_map["Transaction_Amount"] = "Amount"
    if "Account" in df.columns:
        rename_map["Account"] = "Account_Number"
    if rename_map:
        df = df.rename(columns=rename_map)
    if "Amount" not in df.columns and "Transaction Amount" in df.columns:
        df = df.rename(columns={"Transaction Amount": "Amount"})
    if "Account_Number" not in df.columns and "User_ID" in df.columns:
        df = df.rename(columns={"User_ID": "Account_Number"})
    if "Balance" not in df.columns:
        df["Balance"] = np.nan

    df["Amount"] = safe_to_numeric(df.get("Amount", pd.Series(dtype=float))).fillna(0)
    df["Balance"] = safe_to_numeric(df.get("Balance", pd.Series(dtype=float))).fillna(0)
    df["Source_Type"] = df.get("Source_Type", "M-Pesa")
    if "Account_Number" not in df.columns:
        df["Account_Number"] = "UNKNOWN"
    return df[["Account_Number", "Date", "Description", "Amount", "Balance", "Source_Type"]]

def read_bills_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = standardize_date_column(df)
    rename_map = {}
    if "Provider" in df.columns:
        rename_map["Provider"] = "Description"
    if "Final_Amount_KSh" in df.columns:
        rename_map["Final_Amount_KSh"] = "Amount"
    if "User_ID" in df.columns:
        rename_map["User_ID"] = "Account_Number"
    if rename_map:
        df = df.rename(columns=rename_map)
    if "Amount" not in df.columns and "Total_Bill_KSh" in df.columns:
        df = df.rename(columns={"Total_Bill_KSh": "Amount"})
    if "Balance" not in df.columns:
        df["Balance"] = np.nan

    df["Amount"] = safe_to_numeric(df.get("Amount", pd.Series(dtype=float))).fillna(0)
    df["Balance"] = safe_to_numeric(df.get("Balance", pd.Series(dtype=float))).fillna(0)
    df["Source_Type"] = df.get("Source_Type", "Utility Bill")
    if "Account_Number" not in df.columns:
        df["Account_Number"] = "UNKNOWN"
    return df[["Account_Number", "Date", "Description", "Amount", "Balance", "Source_Type"]]

# ---------------------------
# UI & Main Flow
# ---------------------------
st.title("HEVA Credit Intelligence Dashboard")
st.write("Upload M-Pesa CSV, Utility Bills CSV, and a ZIP of bank PDFs (optional). App standardizes columns and trains a simple risk model.")

col1, col2 = st.columns(2)
with col1:
    mpesa_file = st.file_uploader("Upload M-Pesa CSV", type=["csv"])
    bills_file = st.file_uploader("Upload Utility Bills CSV", type=["csv"])
with col2:
    bank_zip_file = st.file_uploader("Upload Bank Statements ZIP (contains PDFs)", type=["zip"])

# Try to build merged_df from uploaded files
merged_df = pd.DataFrame()
pieces = []

if mpesa_file:
    try:
        mdf = read_mpesa_csv(mpesa_file)
        pieces.append(mdf)
    except Exception as e:
        st.error(f"Error reading M-Pesa CSV: {e}")

if bills_file:
    try:
        bdf = read_bills_csv(bills_file)
        pieces.append(bdf)
    except Exception as e:
        st.error(f"Error reading Bills CSV: {e}")

if bank_zip_file:
    try:
        bank_df = unzip_and_parse_bank_zip(bank_zip_file)
        if not bank_df.empty:
            pieces.append(bank_df)
    except Exception as e:
        st.error(f"Error handling bank ZIP: {e}")

if pieces:
    merged_df = pd.concat(pieces, ignore_index=True)
    # Ensure Date column exists and is datetime
    merged_df = standardize_date_column(merged_df)
    merged_df["Date"] = pd.to_datetime(merged_df["Date"], errors="coerce")
    # fill missing description
    merged_df["Description"] = merged_df["Description"].astype(str).fillna("").replace("nan", "")
    # normalize Source_Type strings
    merged_df["Source_Type"] = merged_df["Source_Type"].astype(str).fillna("UNKNOWN")
    # Ensure Amount/Balance numeric
    merged_df["Amount"] = safe_to_numeric(merged_df.get("Amount", pd.Series(dtype=float))).fillna(0)
    merged_df["Balance"] = safe_to_numeric(merged_df.get("Balance", pd.Series(dtype=float))).fillna(0)
    # Save a cleaned CSV so your original load_data block can reuse it
    cleaned_path = "cleaned_credit_data.csv"
    merged_df.to_csv(cleaned_path, index=False)
    st.success(f"✅ Merged dataset created and saved to `{cleaned_path}` ({len(merged_df)} rows).")
else:
    st.info("Upload at least one file (M-Pesa or Bills or Bank ZIP) to build the dataset. If you already have cleaned_credit_data.csv in the working directory, the app will load it below.")

# ---------------------------
# Load / Prepare data for dashboard & model
# ---------------------------
@st.cache_data
def load_data(path="cleaned_credit_data.csv"):
    if not os.path.exists(path):
        return pd.DataFrame(), None  # empty + no encoder
    df = pd.read_csv(path)
    df = standardize_date_column(df)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # coerce numeric, fill NaN
    df["Amount"] = safe_to_numeric(df.get("Amount", pd.Series(dtype=float))).fillna(0)
    df["Balance"] = safe_to_numeric(df.get("Balance", pd.Series(dtype=float))).fillna(0)
    df["Description"] = df.get("Description", "").astype(str).str.lower()
    if "Source_Type" not in df.columns:
        df["Source_Type"] = "UNKNOWN"
    # label encode source type for model features but keep original text column too
    le = LabelEncoder()
    df["Source_Code"] = le.fit_transform(df["Source_Type"].astype(str))
    return df, le

df, le = load_data()

if df.empty:
    st.warning("No data loaded. Upload files or place 'cleaned_credit_data.csv' in the app folder.")
    st.stop()

# Add a simple risk label if missing (1 => Low Risk per your original code)
if "Risk_Label" not in df.columns:
    df["Risk_Label"] = df["Balance"].apply(lambda x: 0 if x < 500 else 1)

# ---------------------------
# Sidebar filters & display
# ---------------------------
st.sidebar.header("Filter Options")
source_options = list(df["Source_Type"].unique())
source_filter = st.sidebar.multiselect("Select Source Type", options=source_options, default=source_options)
min_amt, max_amt = float(df["Amount"].min()), float(df["Amount"].max())
amount_range = st.sidebar.slider("Transaction Amount Range", min_amt, max_amt, (min_amt, max_amt))

filtered_df = df[(df["Source_Type"].isin(source_filter)) & (df["Amount"] >= amount_range[0]) & (df["Amount"] <= amount_range[1])]

st.subheader("Filtered Transactions (sample)")
st.dataframe(filtered_df.head(25))

# ---------------
# Plots
# ---------------
st.subheader("Transaction Amounts Over Time")
fig1, ax1 = plt.subplots(figsize=(10, 4))
# resample daily (drop missing dates)
time_df = filtered_df.dropna(subset=["Date"]).set_index("Date").resample("D")["Amount"].sum()
time_df.plot(ax=ax1)
ax1.set_ylabel("Total Amount")
ax1.set_title("Daily Transaction Amounts")
st.pyplot(fig1)

st.subheader("Transaction Source Breakdown")
fig2, ax2 = plt.subplots(figsize=(6, 3))
sns.countplot(data=filtered_df, x="Source_Type", order=filtered_df["Source_Type"].value_counts().index, ax=ax2)
ax2.set_title("Transactions by Source")
plt.xticks(rotation=30)
st.pyplot(fig2)

st.subheader("Summary Statistics")
st.write(filtered_df[["Amount", "Balance"]].describe())

# ---------------------------
# Simple Credit Risk Predictor
# ---------------------------
st.subheader("📊 Predict Credit Risk (toy model)")
input_amount = st.number_input("Transaction Amount", value=1000.0)
input_balance = st.number_input("Account Balance", value=500.0)
input_source = st.selectbox("Source Type", options=source_options, index=0)

# Prepare training data
feature_cols = ["Amount", "Balance", "Source_Code"]
X = df[feature_cols].copy()
y = df["Risk_Label"].copy()

if y.nunique() < 2 or len(df) < 20:
    st.warning("Not enough labeled data to train a meaningful model. Need at least 2 classes and ~20 rows.")
else:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    if st.button("Predict Risk"):
        # encode selected source to code
        try:
            src_code = int(le.transform([input_source])[0])
        except Exception:
            # fallback if unseen
            src_code = 0
        user_input = pd.DataFrame([[input_amount, input_balance, src_code]], columns=feature_cols)
        user_scaled = scaler.transform(user_input)
        pred = model.predict(user_scaled)[0]
        proba = model.predict_proba(user_scaled)[0].max()
        st.markdown(f"### Risk Prediction: {'🟢 Low Risk' if pred == 1 else '🔴 High Risk'}")
        st.write(f"Confidence: {round(proba * 100, 2)}%")

# Footer
st.markdown("---")
st.caption("Built with ❤️ by HEVA Credit Analytics — date column is auto-standardized so KeyError('Date') is avoided.")
