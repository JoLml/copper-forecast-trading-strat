# FILE: models/backtest.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from joblib import load

from models.data_preparation import prepare_data
from models.feature_engineering import add_technical_indicators
from utils.data_loader import fetch_copper_data

def naive_strategy(y_true):
    return pd.Series(y_true).shift(1).bfill()

def simple_backtest(preds, returns):
    strat_returns = returns.shift(-1) * preds
    strat_returns = strat_returns.fillna(0)
    return strat_returns.cumsum()

def run_backtest():
    # Load copper data
    original_df = fetch_copper_data().reset_index(drop=True)

    # ✅ Use correct lowercase column name
    close_col = "close"
    if close_col not in original_df.columns:
        print(f"❌ ERROR: '{close_col}' column not found in original_df.")
        print(original_df.columns)
        return

    # Prepare dataset
    df = add_technical_indicators(original_df.copy())
    X_train, X_test, y_train, y_test, _ = prepare_data(df)

    # Load model
    model_path = os.path.join(os.path.dirname(__file__), "random_forest_model.joblib")
    model = load(model_path)

    # Predict
    y_pred = pd.Series(model.predict(X_test))

    # Accuracy
    acc_model = accuracy_score(y_test, y_pred)
    acc_naive = accuracy_score(y_test, naive_strategy(y_test))

    print(f"\n✅ MODEL ACCURACY: {acc_model:.4f}")
    print(f"🪞 NAIVE STRATEGY ACCURACY: {acc_naive:.4f}")

    # Compute returns
    closes = original_df[close_col].iloc[-len(y_test):].reset_index(drop=True)
    returns = closes.pct_change().fillna(0)
    cum_returns = simple_backtest(y_pred, returns)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(cum_returns, label="Model Strategy")
    plt.title(" Cumulative Return of Model Strategy")
    plt.xlabel("Days")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_backtest()
