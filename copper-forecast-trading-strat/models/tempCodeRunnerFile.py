# FILE 1: models/backtest.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from models.data_preparation import prepare_data
from utils.data_loader import fetch_copper_data
from models.feature_engineering import add_technical_indicators
from joblib import load


def naive_strategy(y_true):
    """
    Simple naive strategy: predicts the same as previous day's value
    """
    return y_true.shift(1).fillna(method='bfill')


def simple_backtest(preds, returns):
    """
    Simple backtest: take position (long) when prediction is 1
    """
    strat_returns = returns.shift(-1) * preds  # Next day's return if signal == 1
    strat_returns = strat_returns.fillna(0)
    return strat_returns.cumsum()


def run_backtest():
    # Step 1: Load and prepare data
    df = fetch_copper_data()
    df = add_technical_indicators(df)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    # Step 2: Load model
    model = load("models/random_forest_model.joblib")

    # Step 3: Predict
    y_pred = pd.Series(model.predict(X_test), index=y_test.index)

    # Step 4: Benchmark
    y_naive = naive_strategy(y_test)
    acc_model = accuracy_score(y_test, y_pred)
    acc_naive = accuracy_score(y_test, y_naive)

    print(f"\n✅ MODEL ACCURACY: {acc_model:.4f}")
    print(f"🪞 NAIVE STRATEGY ACCURACY: {acc_naive:.4f}")

    # Step 5: Backtest
    df = df.loc[y_test.index]
    daily_returns = df['Close'].pct_change().fillna(0)
    cum_return = simple_backtest(y_pred, daily_returns)

    # Step 6: Plot
    plt.figure(figsize=(10, 5))
    plt.plot(cum_return, label='Model Strategy')
    plt.title("📈 Cumulative Return of Model Strategy")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_backtest()
