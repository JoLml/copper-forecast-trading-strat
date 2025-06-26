import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump

from models.data_preparation import prepare_data
from utils.data_loader import fetch_copper_data
from models.feature_engineering import add_technical_indicators

def train_random_forest():
    """
    Train a Random Forest model on copper technical indicators to predict
    whether the price will go up in N days.
    """
    # 1. Load & prepare data
    df = fetch_copper_data()
    df = add_technical_indicators(df)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    # 2. Initialize model
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

    # 3. Fit model
    rf.fit(X_train, y_train)

    # 4. Evaluate model
    y_pred = rf.predict(X_test)
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("🧾 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 5. Save model and scaler
    dump(rf, "models/random_forest_model.joblib")
    if scaler:
        dump(scaler, "models/scaler.joblib")

    print("✅ Model training complete and saved to disk.")

if __name__ == "__main__":
    train_random_forest()
