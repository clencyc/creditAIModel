import pandas as pd
import numpy as np
import zipfile, os, re
from io import BytesIO
from PyPDF2 import PdfReader
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Helpers
# ---------------------------
def safe_to_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^\d\.\-]", "", regex=True),
        errors="coerce"
    )

def standardize_date_column(df):
    possible_date_cols = [
        "Date", "Transaction_Date", "Txn_Date", "Billing_Date", "Payment_Date", "date"
    ]
    for c in possible_date_cols:
        if c in df.columns:
            df = df.rename(columns={c: "Date"})
            break
    if "Date" not in df.columns:
        df["Date"] = pd.NaT
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

# ---------------------------
# M-Pesa CSV
# ---------------------------
def read_mpesa_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = standardize_date_column(df)

    col_map = {
        "Description": ["Transaction_Type", "Type", "Details", "Description"],
        "Amount": ["Transaction_Amount", "Amount", "Debit", "Credit"],
        "Balance": ["Balance", "Account_Balance", "Running_Balance"],
        "Account_Number": ["Account", "User_ID", "Account_Number", "MSISDN"],
    }

    for target, candidates in col_map.items():
        for c in candidates:
            if c in df.columns:
                df = df.rename(columns={c: target})
                break
        if target not in df.columns:
            df[target] = "UNKNOWN" if target in ["Description", "Account_Number"] else 0.0

    df["Amount"] = safe_to_numeric(df["Amount"]).fillna(0)
    df["Balance"] = safe_to_numeric(df["Balance"]).fillna(0)
    df["Source_Type"] = "M-Pesa"

    return df[["Account_Number", "Date", "Description", "Amount", "Balance", "Source_Type"]]

# ---------------------------
# Utility Bills CSV
# ---------------------------
def read_bills_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = standardize_date_column(df)

    col_map = {
        "Description": ["Provider", "Service", "Bill_Type", "Description"],
        "Amount": ["Final_Amount_KSh", "Total_Bill_KSh", "Amount", "Charge"],
        "Balance": ["Balance", "Outstanding_Balance", "Remaining_Balance"],
        "Account_Number": ["User_ID", "Account", "Account_Number", "Customer_ID"],
    }

    for target, candidates in col_map.items():
        for c in candidates:
            if c in df.columns:
                df = df.rename(columns={c: target})
                break
        if target not in df.columns:
            df[target] = "UNKNOWN" if target in ["Description", "Account_Number"] else 0.0

    df["Amount"] = safe_to_numeric(df["Amount"]).fillna(0)
    df["Balance"] = safe_to_numeric(df["Balance"]).fillna(0)
    df["Source_Type"] = "Utility Bill"

    return df[["Account_Number", "Date", "Description", "Amount", "Balance", "Source_Type"]]

# ---------------------------
# Bank Statements (ZIP of PDFs)
# ---------------------------
date_line_re = re.compile(r"^\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})")

def parse_bank_statement_text(text):
    records = []
    for line in text.splitlines():
        m = date_line_re.match(line)
        if not m:
            continue
        date = pd.to_datetime(m.group(1), dayfirst=True, errors="coerce")
        rest = line[m.end():].strip()
        nums = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\d+", rest)
        desc = re.sub(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\d+", "", rest).strip()
        debit = nums[-3] if len(nums) >= 3 else None
        credit = nums[-2] if len(nums) >= 2 else None
        balance = nums[-1] if len(nums) >= 1 else None
        records.append({
            "Date": date,
            "Description": desc or "UNKNOWN",
            "Debit": safe_to_numeric(pd.Series([debit]))[0] if debit else np.nan,
            "Credit": safe_to_numeric(pd.Series([credit]))[0] if credit else np.nan,
            "Balance": safe_to_numeric(pd.Series([balance]))[0] if balance else np.nan,
        })
    return pd.DataFrame(records)

def read_bank_zip(uploaded_zip):
    bank_records = []
    try:
        z = zipfile.ZipFile(BytesIO(uploaded_zip.read()))
    except Exception:
        return pd.DataFrame()

    for name in z.namelist():
        if not name.lower().endswith(".pdf"):
            continue
        try:
            reader = PdfReader(BytesIO(z.read(name)))
            text = "".join(
                page.extract_text() + "\n"
                for page in reader.pages if page.extract_text()
            )
            df = parse_bank_statement_text(text)
            if not df.empty:
                df["Account_Number"] = os.path.splitext(os.path.basename(name))[0]
                df["Source_Type"] = "Bank Statement"
                df["Amount"] = df["Credit"].fillna(0) - df["Debit"].fillna(0)
                bank_records.append(
                    df[["Account_Number", "Date", "Description", "Amount", "Balance", "Source_Type"]]
                )
        except Exception:
            continue
    return pd.concat(bank_records, ignore_index=True) if bank_records else pd.DataFrame()

# ---------------------------
# Feature Engineering
# ---------------------------
def add_features(df):
    """
    Add engineered features for credit risk modeling.
    Expects a standardized dataframe with:
    ['Account_Number','Date','Description','Amount','Balance','Source_Type']
    """

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Transaction frequency (per account)
    txn_freq = df.groupby("Account_Number")["Date"].count().to_dict()
    df["Txn_Frequency"] = df["Account_Number"].map(txn_freq)

    # Average balance per account
    avg_balance = df.groupby("Account_Number")["Balance"].mean().to_dict()
    df["Avg_Balance"] = df["Account_Number"].map(avg_balance)

    # Total credit inflow
    inflow = df[df["Amount"] > 0].groupby("Account_Number")["Amount"].sum().to_dict()
    df["Total_Credit"] = df["Account_Number"].map(inflow)

    # Total debit outflow
    outflow = df[df["Amount"] < 0].groupby("Account_Number")["Amount"].sum().to_dict()
    df["Total_Debit"] = df["Account_Number"].map(outflow)

    # Punctuality score = consistency of reporting (gap days)
    def calc_punctuality(group):
        dates = group.dropna().sort_values()
        if len(dates) < 2:
            return 0.5
        avg_gap = dates.diff().dt.days.dropna().mean()
        expected_gap = 30  # assume monthly ideal
        score = max(0, min(1, 1 - abs(avg_gap - expected_gap) / expected_gap))
        return round(score, 3)

    punctuality_scores = df.groupby("Account_Number")["Date"].apply(calc_punctuality).to_dict()
    df["Punctuality_Score"] = df["Account_Number"].map(punctuality_scores)

    # --- Sector assignment (default "General") ---
    if "Sector" not in df.columns:
        df["Sector"] = "General"

    # Encode sector into integers
    le = LabelEncoder()
    df["Sector_Code"] = le.fit_transform(df["Sector"].astype(str))

    return df, le
