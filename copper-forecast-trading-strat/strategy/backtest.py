# strategy/backtest.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load

from utils.data_loader import fetch_copper_data
from models.feature_engineering import add_technical_indicators
from models.data_preparation import prepare_data


def backtest_strategy():
    # Load and prepare data
    df = fetch_copper_data()
    df = add_technical_indicators(df)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    # Align with X_test dates (since we did a temporal split)
    df = df.iloc[-len(X_test):].copy()

    # Load trained model and scaler
    model = load("models/random_forest_model.joblib")
    if scaler:
        X_test = scaler.transform(X_test)

    # Predict on test set
    df['prediction'] = model.predict(X_test)

    # Simulate simple trading strategy
    df['position'] = df['prediction'].shift(1)  # simulate trade entered next day
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['position'] * df['returns']

    # Calculate cumulative returns
    df['cumulative_market'] = (1 + df['returns']).cumprod()
    df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['cumulative_market'], label='Market (Buy & Hold)')
    plt.plot(df.index, df['cumulative_strategy'], label='Strategy (Model-based)')
    plt.title("📈 Backtest - Copper Price Prediction Strategy")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print final performance
    final_strategy_return = df['cumulative_strategy'].iloc[-1] - 1
    final_market_return = df['cumulative_market'].iloc[-1] - 1

    print(f"\n📊 Final Strategy Return: {final_strategy_return:.2%}")
    print(f"📊 Final Market Return: {final_market_return:.2%}")


if __name__ == "__main__":
    backtest_strategy()
