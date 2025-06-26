"""
Feature engineering functions for copper price forecasting.
Includes technical indicators such as RSI, Bollinger Bands, Momentum, etc.
"""

import pandas as pd
import numpy as np

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data

    Returns:
        pd.DataFrame: DataFrame enriched with technical indicators
    """
    df = df.copy()

    # 20-day Moving Average
    df['ma20'] = df['close'].rolling(window=20).mean()

    # 50-day Moving Average
    df['ma50'] = df['close'].rolling(window=50).mean()

    # 20-day Rolling Volatility (Standard Deviation)
    df['volatility_20d'] = df['close'].rolling(window=20).std()

    # Relative Strength Index (RSI - 14 days)
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=df.index).rolling(window=14).mean()
    avg_loss = pd.Series(loss, index=df.index).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Bollinger Bands (using MA20 and volatility_20d)
    df['bb_upper'] = df['ma20'] + 2 * df['volatility_20d']
    df['bb_lower'] = df['ma20'] - 2 * df['volatility_20d']

    # Momentum (10-day)
    df['momentum_10d'] = df['close'] - df['close'].shift(10)

    # Rate of Change (ROC - 10 days)
    df['roc_10d'] = df['close'].pct_change(periods=10)

    # MACD (Moving Average Convergence Divergence)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Drop initial NaNs caused by rolling/ewm calculations
    df.dropna(inplace=True)

    return df
