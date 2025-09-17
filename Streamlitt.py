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
        df["Date"] = pd.NaT
    return df  # Return the DataFrame

def read_mpesa_csv(file):
    try:
        df = pd.read_csv(file)
        if df.empty:
            st.error("M-Pesa CSV is empty or invalid.")
            return pd.DataFrame()
        df = standardize_date_column(df, ["Date", "Transaction Date", "Date & Time"])
        rename_dict = {
            "Transaction_Type": "Description",
            "Transaction_Amount": "Amount",
            "Account": "Account_Number"
        }
        df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})
        if "Balance" not in df.columns:
            df["Balance"] = None
        df["Source_Type"] = "M-Pesa"
        return df[["Account_Number", "Date", "Description", "Amount", "Balance", "Source_Type"]]
    except Exception as e:
        st.error(f"Error reading M-Pesa CSV: {str(e)}")
        return pd.DataFrame()

def read_bills_csv(file):
    try:
        df = pd.read_csv(file)
        if df.empty:
            st.error("Utility Bills CSV is empty or invalid.")
            return pd.DataFrame()
        df = standardize_date_column(df, ["Date", "Billing_Date", "Payment_Date"])
        rename_dict = {
            "Provider": "Description",
            "Final_Amount_KSh": "Amount",
            "User_ID": "Account_Number"
        }
        df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})
        if "Balance" not in df.columns:
            df["Balance"] = None
        df["Source_Type"] = "Utility Bill"
        return df[["Account_Number", "Date", "Description", "Amount", "Balance", "Source_Type"]]
    except Exception as e:
        st.error(f"Error reading Utility Bills CSV: {str(e)}")
        return pd.DataFrame()

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
    mpesa_df = read_mpesa_csv(mpesa_file)
    if not mpesa_df.empty:
        dfs.append(mpesa_df)
if bills_file:
    bills_df = read_bills_csv(bills_file)
    if not bills_df.empty:
        dfs.append(bills_df)

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
