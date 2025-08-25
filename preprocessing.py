# preprocessing.py
import pandas as pd

def preprocess_transactions(uploaded_file, sector_mapping_path="data/sector_mapping.csv"):
    """
    Preprocess uploaded M-Pesa/Bank/Utility transactions safely.
    Handles missing files, bad data, and merges with sector mapping.

    Args:
        uploaded_file: Streamlit uploaded file object (.csv or .xlsx)
        sector_mapping_path: path to sector mapping CSV

    Returns:
        Cleaned pandas DataFrame or None if failed
    """
    if uploaded_file is None:
        print("⚠️ No file uploaded")
        return None

    try:
        # Read uploaded file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            print("❌ Unsupported file format")
            return None

        # Clean dataframe
        df = df.reset_index(drop=True)
        df = df.loc[:, ~df.columns.duplicated()]  # remove duplicate cols

        # Check sector mapping
        try:
            sector_map = pd.read_csv(sector_mapping_path)
            if "merchant_name" in df.columns and "merchant_name" in sector_map.columns:
                df = df.merge(sector_map, on="merchant_name", how="left")
            else:
                print("⚠️ No merchant_name column found, skipping sector mapping")
        except Exception as e:
            print(f"⚠️ Could not load sector mapping: {e}")

        # Fill missing values for safety
        df = df.fillna({"sector": "Unknown"})

        return df

    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return None
