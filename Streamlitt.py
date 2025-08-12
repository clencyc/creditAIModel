# heva_credit_ai.py
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

# Streamlit App Config
st.set_page_config(page_title="HEVA Credit Intelligence – AI Risk Predictor", layout="wide")

# ---------------------------
# Helper Functions
# ---------------------------
def find_date_col(df):
    for c in df.columns:
        if any(k in c.lower() for k in ("date", "time", "billing", "payment", "paid", "txn", "transaction")):
            return c
    return None

def standardize_date_column(df):
    col = find_date_col(df)
    if col:
        df = df.rename(columns={col: "Date"})
    if "Date" not in df.columns:
        df["Date"] = pd.NaT
    return df

def safe_to_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors="coerce")

# ---------------------------
# Bank Statement Parsing
# ---------------------------
date_line_re = re.compile(r'^\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})')

def parse_bank_statement_text(text):
    records = []
    for line in text.splitlines():
        m = date_line_re.match(line)
        if not m:
            continue
        date = pd.to_datetime(m.group(1), dayfirst=True, errors="coerce")
        rest = line[m.end():].strip()
        nums = re.findall(r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\d+', rest)
        desc = re.sub(r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\d+', '', rest).strip()
        debit = nums[-3] if len(nums) >= 3 else None
        credit = nums[-2] if len(nums) >= 2 else None
        balance = nums[-1] if len(nums) >= 1 else None
        records.append({
            "Date": date,
            "Description": desc or None,
            "Debit": safe_to_numeric(pd.Series([debit]))[0] if debit else np.nan,
            "Credit": safe_to_numeric(pd.Series([credit]))[0] if credit else np.nan,
            "Balance": safe_to_numeric(pd.Series([balance]))[0] if balance else np.nan,
        })
    return pd.DataFrame(records)

def unzip_and_parse_bank_zip(uploaded_zip):
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
            reader = PdfReader(BytesIO(z.read(name)))
            text = "".join(page.extract_text() + "\n" for page in reader.pages if page.extract_text())
            df = parse_bank_statement_text(text)
            if not df.empty:
                df["Account_Number"] = os.path.splitext(os.path.basename(name))[0]
                df["Source_Type"] = "Bank Statement"
                df["Amount"] = df["Credit"].fillna(0) - df["Debit"].fillna(0)
                bank_records.append(df[["Account_Number", "Date", "Description", "Amount", "Balance", "Source_Type"]])
        except Exception as e:
            st.warning(f"Failed parsing {name}: {e}")
    return pd.concat(bank_records, ignore_index=True) if bank_records else pd.DataFrame()

# ---------------------------
# CSV Readers
# ---------------------------
def read_mpesa_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = standardize_date_column(df)
    rename_map = {
        "Transaction_Type": "Description",
        "Transaction_Amount": "Amount",
        "Account": "Account_Number",
        "User_ID": "Account_Number",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["Amount"] = safe_to_numeric(df.get("Amount", pd.Series(dtype=float))).fillna(0)
    df["Balance"] = safe_to_numeric(df.get("Balance", pd.Series(dtype=float))).fillna(0)
    df["Source_Type"] = "M-Pesa"
    if "Account_Number" not in df.columns:
        df["Account_Number"] = "UNKNOWN"
    return df[["Account_Number", "Date", "Description", "Amount", "Balance", "Source_Type"]]

def read_bills_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = standardize_date_column(df)
    rename_map = {
        "Provider": "Description",
        "Final_Amount_KSh": "Amount",
        "Total_Bill_KSh": "Amount",
        "User_ID": "Account_Number",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["Amount"] = safe_to_numeric(df.get("Amount", pd.Series(dtype=float))).fillna(0)
    df["Balance"] = safe_to_numeric(df.get("Balance", pd.Series(dtype=float))).fillna(0)
    df["Source_Type"] = "Utility Bill"
    if "Account_Number" not in df.columns:
        df["Account_Number"] = "UNKNOWN"
    return df[["Account_Number", "Date", "Description", "Amount", "Balance", "Source_Type"]]

# ---------------------------
# Load & Feature Engineering
# ---------------------------
@st.cache_data
def load_data(path="cleaned_credit_data.csv"):
    if not os.path.exists(path):
        return pd.DataFrame(), None
    df = pd.read_csv(path)
    df = standardize_date_column(df)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = safe_to_numeric(df.get("Amount", pd.Series(dtype=float))).fillna(0)
    df["Balance"] = safe_to_numeric(df.get("Balance", pd.Series(dtype=float))).fillna(0)
    if "Source_Type" not in df.columns:
        df["Source_Type"] = "UNKNOWN"
    le = LabelEncoder()
    df["Source_Code"] = le.fit_transform(df["Source_Type"].astype(str))

    # NEW: Calculate Punctuality Score
    def calc_punctuality_score(group):
        dates = group.dropna().sort_values()
        if len(dates) < 2:
            return 0.5
        avg_gap = dates.diff().dt.days.dropna().mean()
        expected_gap = 30  # Ideal: monthly reporting
        score = max(0, min(1, 1 - abs(avg_gap - expected_gap) / expected_gap))
        return round(score, 3)

    punctuality_scores = df.groupby("Account_Number")["Date"].apply(calc_punctuality_score).to_dict()
    df["Punctuality_Score"] = df["Account_Number"].map(punctuality_scores)

    return df, le

# ---------------------------
# UI & Upload Section
# ---------------------------
st.title("HEVA Credit Intelligence – AI Risk Predictor")
st.write("Upload your financial data (M-Pesa, Utility Bills, Bank Statements) to generate an AI-powered credit risk assessment.")

col1, col2 = st.columns(2)
with col1:
    mpesa_file = st.file_uploader("Upload M-Pesa CSV", type=["csv"])
    bills_file = st.file_uploader("Upload Utility Bills CSV", type=["csv"])
with col2:
    bank_zip_file = st.file_uploader("Upload Bank Statements ZIP (PDFs)", type=["zip"])

pieces = []
if mpesa_file:
    pieces.append(read_mpesa_csv(mpesa_file))
if bills_file:
    pieces.append(read_bills_csv(bills_file))
if bank_zip_file:
    pieces.append(unzip_and_parse_bank_zip(bank_zip_file))

if pieces:
    merged_df = pd.concat(pieces, ignore_index=True)
    merged_df.to_csv("cleaned_credit_data.csv", index=False)
    st.success(f"✅ Merged dataset created with {len(merged_df)} records.")

df, le = load_data()
if df.empty:
    st.warning("No data available. Upload files above.")
    st.stop()

if "Risk_Label" not in df.columns:
    df["Risk_Label"] = df["Balance"].apply(lambda x: 0 if x < 500 else 1)

# ---------------------------
# Visualization
# ---------------------------
st.subheader("Transaction Amount Trends")
fig, ax = plt.subplots(figsize=(10,4))
df.dropna(subset=["Date"]).set_index("Date").resample("D")["Amount"].sum().plot(ax=ax)
ax.set_ylabel("Total Amount")
st.pyplot(fig)

st.subheader("Transaction Source Breakdown")
fig2, ax2 = plt.subplots(figsize=(6,3))
sns.countplot(data=df, x="Source_Type", order=df["Source_Type"].value_counts().index, ax=ax2)
plt.xticks(rotation=30)
st.pyplot(fig2)

# ---------------------------
# AI Model & Prediction
# ---------------------------
st.subheader("📊 Predict Credit Risk")
input_amount = st.number_input("Transaction Amount", value=1000.0)
input_balance = st.number_input("Account Balance", value=500.0)
input_source = st.selectbox("Source Type", options=list(df["Source_Type"].unique()))
input_punctuality = st.slider("Reporting Punctuality Score", 0.0, 1.0, 0.5, 0.01)

feature_cols = ["Amount", "Balance", "Source_Code", "Punctuality_Score"]
X = df[feature_cols]
y = df["Risk_Label"]

if y.nunique() >= 2 and len(df) >= 20:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    if st.button("Predict Risk"):
        try:
            src_code = int(le.transform([input_source])[0])
        except:
            src_code = 0
        user_input = pd.DataFrame([[input_amount, input_balance, src_code, input_punctuality]], columns=feature_cols)
        user_scaled = scaler.transform(user_input)
        pred = model.predict(user_scaled)[0]
        proba = model.predict_proba(user_scaled)[0].max()
        st.markdown(f"### Risk Prediction: {'🟢 Low Risk' if pred == 1 else '🔴 High Risk'}")
        st.write(f"Confidence: {round(proba * 100, 2)}%")
else:
    st.warning("Not enough data to train the model.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("© 2025 HEVA Credit Intelligence | Built with ❤️ to democratize access to credit for creative enterprises. Date column is auto-standardized to prevent errors.")
