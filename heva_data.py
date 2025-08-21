# heva_data.py
import os, re, zipfile
from io import BytesIO
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.preprocessing import LabelEncoder

# ---------- Utilities ----------
def safe_to_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors="coerce")

def find_date_col(df):
    for c in df.columns:
        if any(k in c.lower() for k in ("date","time","billing","payment","txn","transaction")):
            return c
    return None

def standardize_date_column(df):
    col = find_date_col(df)
    if col: df = df.rename(columns={col: "Date"})
    if "Date" not in df.columns: df["Date"] = pd.NaT
    return df

# ---------- Readers ----------
def read_mpesa_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = standardize_date_column(df)
    df["Amount"] = safe_to_numeric(df.get("Transaction_Amount", pd.Series(dtype=float))).fillna(0)
    df["Balance"] = safe_to_numeric(df.get("Balance", pd.Series(dtype=float))).fillna(0)
    df["Source_Type"] = "M-Pesa"
    df["Account_Number"] = df.get("User_ID", "UNKNOWN")
    return df[["Account_Number","Date","Transaction_Type","Amount","Balance","Source_Type"]].rename(
        columns={"Transaction_Type":"Description"}
    )

def read_bills_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = standardize_date_column(df)
    df["Amount"] = safe_to_numeric(df.get("Final_Amount_KSh", df.get("Total_Bill_KSh", pd.Series(dtype=float)))).fillna(0)
    df["Balance"] = safe_to_numeric(df.get("Balance", pd.Series(dtype=float))).fillna(0)
    df["Source_Type"] = "Utility Bill"
    df["Account_Number"] = df.get("User_ID","UNKNOWN")
    return df[["Account_Number","Date","Provider","Amount","Balance","Source_Type"]].rename(
        columns={"Provider":"Description"}
    )

def read_bank_zip(uploaded_file):
    records = []
    z = zipfile.ZipFile(BytesIO(uploaded_file.read()))
    for name in z.namelist():
        if not name.lower().endswith(".pdf"): continue
        reader = PdfReader(BytesIO(z.read(name)))
        text = "".join(page.extract_text() or "" for page in reader.pages)
        for line in text.splitlines():
            if re.match(r'^\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', line):
                parts = line.split()
                try: date = pd.to_datetime(parts[0], errors="coerce", dayfirst=True)
                except: date = pd.NaT
                nums = re.findall(r'[-+]?\d[\d,]*\.?\d*', line)
                balance = nums[-1] if nums else None
                records.append({
                    "Account_Number": name,
                    "Date": date,
                    "Description": line,
                    "Amount": float(nums[-2]) if len(nums)>=2 else 0,
                    "Balance": float(balance.replace(",","")) if balance else 0,
                    "Source_Type":"Bank Statement"
                })
    return pd.DataFrame(records)

# ---------- Feature Engineering ----------
def add_features(df, sector_map=None):
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = safe_to_numeric(df["Amount"]).fillna(0)
    df["Balance"] = safe_to_numeric(df["Balance"]).fillna(0)
    if "Source_Type" not in df.columns: df["Source_Type"] = "UNKNOWN"

    # Example: Punctuality Score per account
    def calc_punctuality(dates):
        dates = dates.dropna().sort_values()
        if len(dates)<2: return 0.5
        avg_gap = dates.diff().dt.days.dropna().mean()
        expected = 30
        return max(0,min(1,1-abs(avg_gap-expected)/expected))

    punctuality = df.groupby("Account_Number")["Date"].apply(calc_punctuality).to_dict()
    df["Punctuality_Score"] = df["Account_Number"].map(punctuality)

    # Map sector (dummy until linked with metadata)
    if sector_map:
        df["Sector"] = df["Account_Number"].map(sector_map).fillna("General")
    else:
        df["Sector"] = "General"

    le = LabelEncoder()
    df["Sector_Code"] = le.fit_transform(df["Sector"])

    return df, le
