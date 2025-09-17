import os
import zipfile
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import timedelta

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
    return df

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

def calculate_risk_score(df):
    """Calculate a risk score (0-100) based on financial metrics."""
    if df.empty or "Amount" not in df.columns or "Balance" not in df.columns:
        return 0, {"Transaction Frequency": 0, "Balance Stability": 0, "Credit Ratio": 0}
    
    # Transaction Frequency (30%): Higher frequency of transactions reduces risk
    transaction_count = len(df)
    frequency_score = min(transaction_count / 50, 1) * 30  # Cap at 50 transactions
    
    # Balance Stability (40%): Lower volatility in balance reduces risk
    balance_std = df["Balance"].std() if df["Balance"].notnull().any() else float("inf")
    avg_balance = df["Balance"].mean() if df["Balance"].notnull().any() else 0
    stability_score = (1 - min(balance_std / (avg_balance + 1), 1)) * 40 if avg_balance > 0 else 0
    
    # Credit Ratio (30%): Higher credit-to-debit ratio reduces risk
    credits = df[df["Amount"] > 0]["Amount"].sum()
    debits = abs(df[df["Amount"] < 0]["Amount"].sum())
    credit_ratio = credits / (debits + 1) if debits > 0 else 1
    credit_score = min(credit_ratio, 1) * 30
    
    total_score = round(frequency_score + stability_score + credit_score)
    breakdown = {
        "Transaction Frequency": round(frequency_score),
        "Balance Stability": round(stability_score),
        "Credit Ratio": round(credit_score)
    }
    return min(total_score, 100), breakdown

def calculate_punctuality_score(df):
    """Calculate a punctuality score (0-100) based on payment timeliness."""
    if df.empty or "Source_Type" not in df.columns or "Date" not in df.columns:
        return 0, []
    
    bill_df = df[df["Source_Type"] == "Utility Bill"]
    if bill_df.empty:
        return 0, []
    
    # Assume bills CSV has a 'Due_Date' column; if not, estimate as Date - 30 days
    if "Due_Date" not in bill_df.columns:
        bill_df["Due_Date"] = bill_df["Date"] - timedelta(days=30)
    
    # Calculate timeliness: Payment within 5 days of due date is considered punctual
    bill_df["Days_Late"] = (bill_df["Date"] - pd.to_datetime(bill_df["Due_Date"])).dt.days
    punctual_payments = len(bill_df[bill_df["Days_Late"] <= 5])
    total_payments = len(bill_df)
    
    # Punctuality Score (70%): Ratio of punctual payments
    punctuality_ratio = punctual_payments / total_payments if total_payments > 0 else 0
    punctuality_score = punctuality_ratio * 70
    
    # Regularity Score (30%): Consistency of transaction dates
    transaction_dates = df["Date"].dropna()
    if len(transaction_dates) > 1:
        intervals = (transaction_dates.diff().dt.days).dropna()
        regularity_score = (1 - min(intervals.std() / 30, 1)) * 30 if intervals.std() > 0 else 30
    else:
        regularity_score = 0
    
    total_score = round(punctuality_score + regularity_score)
    payment_details = bill_df[["Description", "Date", "Due_Date", "Days_Late"]].to_dict("records")
    return min(total_score, 100), payment_details

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
    # Risk Score Assessment
    # -----------------------
    st.subheader("📉 Risk Score Assessment")
    risk_score, risk_breakdown = calculate_risk_score(merged_df)
    st.write(f"**Risk Score: {risk_score}/100** (Lower is riskier)")
    
    # Gauge Chart for Risk Score
    fig_risk = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={"text": "Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 33], "color": "red"},
                {"range": [33, 66], "color": "yellow"},
                {"range": [66, 100], "color": "green"}
            ]
        }
    ))
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # Risk Breakdown
    st.write("**Risk Score Breakdown**")
    st.table(risk_breakdown)
    
    # Improvement Insights
    st.write("**How to Improve Risk Score**")
    if risk_breakdown["Transaction Frequency"] < 20:
        st.write("- Increase transaction frequency with consistent deposits.")
    if risk_breakdown["Balance Stability"] < 25:
        st.write("- Maintain higher and more stable account balances.")
    if risk_breakdown["Credit Ratio"] < 20:
        st.write("- Increase credit transactions relative to debits.")

    # -----------------------
    # Punctuality Score Assessment
    # -----------------------
    st.subheader("⏰ Punctuality Score Assessment")
    punctuality_score, payment_details = calculate_punctuality_score(merged_df)
    st.write(f"**Punctuality Score: {punctuality_score}/100** (Higher is better)")
    
    # Gauge Chart for Punctuality Score
    fig_punctuality = go.Figure(go.Indicator(
        mode="gauge+number",
        value=punctuality_score,
        title={"text": "Punctuality Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkgreen"},
            "steps": [
                {"range": [0, 33], "color": "red"},
                {"range": [33, 66], "color": "yellow"},
                {"range": [66, 100], "color": "green"}
            ]
        }
    ))
    st.plotly_chart(fig_punctuality, use_container_width=True)
    
    # Payment Details
    if payment_details:
        st.write("**Payment Details**")
        st.dataframe(pd.DataFrame(payment_details).head(10))
        
        # Improvement Insights
        st.write("**How to Improve Punctuality Score**")
        late_payments = len([p for p in payment_details if p["Days_Late"] > 5])
        if late_payments > 0:
            st.write(f"- {late_payments} payments were late. Aim to pay bills within 5 days of due date.")
        if len(payment_details) < 5:
            st.write("- Increase the number of recorded payments for better assessment.")
        if len(merged_df) > 1:
            intervals = (merged_df["Date"].diff().dt.days).dropna()
            if intervals.std() > 30:
                st.write("- Maintain more consistent transaction intervals.")
    else:
        st.write("No utility bill data available for punctuality analysis.")

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
