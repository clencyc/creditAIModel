# heva_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_sector_model(df: pd.DataFrame, feature_cols, target_col="Risk_Label"):
    X = df[feature_cols]
    y = df[target_col]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42))
    ])
    pipe.fit(X, y)
    return pipe
