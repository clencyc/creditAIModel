heva_credit_ai/
│
├── app.py                # Streamlit UI with footer, sector calibration
├── heva_data.py          # Data ingestion + cleaning (M-Pesa, bills, bank statements)
├── heva_model.py         # Model training (shared backbone)
├── heva_sector.py        # Sector weights + probability calibration
├── requirements.txt      # Python dependencies
│
├── data/                 # Uploaded or sample datasets
│   ├── sample_mpesa.csv
│   ├── sample_bills.csv
│   └── sample_bank.zip
│
├── notebooks/            # Colab-ready experiments + analysis
│   └── exploratory.ipynb
│
├── docs/                 # Diagrams, PDFs, sector weighting policies
│   └── heva_integrated_model_architecture.png
│
└── README.md             # Project overview, setup, usage instructions
