🌍 HEVA Credit Intelligence – AI Risk Predictor

An AI-powered platform that democratizes access to credit for creative enterprises by analyzing bank statements, M-Pesa transactions, and utility bills.

This system uses a sector-aware, calibrated credit scoring model that accounts for differences across 9 creative sectors (Film, Fashion, Design, Music, Media/Broadcast, etc.), ensuring fairness, transparency, and scalability.

✨ Features

📊 Unified Data Ingestion: Upload M-Pesa CSVs, utility bills, and zipped PDF bank statements.

⚡ Automated Parsing & Cleaning: Standardizes dates, balances, and amounts; handles inconsistent file formats.

🎯 AI-Powered Risk Prediction: Random Forest + sector-aware calibration for accurate scoring.

🧭 Sector Weighting: Adjusts importance of features like punctuality, balance, and cash flow for each sector.

🔍 Transparency: Outputs both raw model score and calibrated sector-adjusted score.

📈 Visualization: Transaction trends, source breakdowns, and sector-level insights.

🌐 Streamlit Web App: Simple, interactive UI with live risk predictions.

🛠️ Project Structure
heva_credit_ai/
│
├── app.py                # Streamlit UI (sector-aware, calibrated, with footer)
├── heva_data.py          # Data ingestion & cleaning (M-Pesa, bills, banks)
├── heva_model.py         # Model training logic
├── heva_sector.py        # Sector-specific weights & calibration functions
├── requirements.txt      # Python dependencies
│
├── data/                 # Uploaded or sample datasets
│   ├── sample_mpesa.csv
│   ├── sample_bills.csv
│   └── sample_bank.zip
│
├── notebooks/            # Colab-ready notebooks for experiments
│   └── exploratory.ipynb
│
├── docs/                 # Diagrams & PDFs
│   └── heva_integrated_model_architecture.png
│
└── README.md             # Project overview & instructions

🚀 Quick Start
1️⃣ Local Setup

Clone this repo and install requirements:

git clone https://github.com/your-org/heva_credit_ai.git
cd heva_credit_ai
pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py

2️⃣ Run on Google Colab

Upload the repo to Colab, then run:

!pip install -r heva_credit_ai/requirements.txt
!streamlit run heva_credit_ai/app.py --server.port 8080 & npx localtunnel --port 8080


Colab will provide a public URL to access the dashboard.

📂 Data Inputs

The app expects:

M-Pesa CSV → columns: Transaction_Type, Transaction_Amount, Balance, User_ID.

Utility Bills CSV → columns: Provider, Final_Amount_KSh, Balance, User_ID.

Bank ZIP → zipped PDFs of statements, parsed into Date, Amount, Balance.

⚠️ Don’t worry if column names differ slightly — the parser auto-standardizes where possible.

📊 Outputs

Low Risk (🟢) or High Risk (🔴) labels.

Raw Score (from ML model).

Calibrated Score (sector-aware adjustments).

Confidence level (%).

⚖️ Governance & Fairness

📌 Sector-aware calibration prevents bias across different creative industries.

📌 Transparency: Probabilities are displayed before and after calibration.

📌 Extendable: Future modules will include SHAP explanations and automated reporting.

🤝 Contributing

We welcome contributions from developers, data scientists, and creative industry partners.

Fork the repo.

Create a feature branch (feature/new-idea).

Submit a pull request.

📜 License

© 2025 HEVA Credit Intelligence. All rights reserved.
Built with ❤️ to support Africa’s creative economy.
