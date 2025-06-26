import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data(df, target_shift=5, test_size=0.2, scale=True):
    """
    Prepare features and target for modeling.

    Args:
        df (pd.DataFrame): DataFrame with technical indicators.
        target_shift (int): Number of days ahead to predict.
        test_size (float): Fraction of data to reserve for testing.
        scale (bool): Whether to apply StandardScaler to features.

    Returns:
        X_train, X_test, y_train, y_test (np.arrays)
        scaler (StandardScaler or None)
    """
    df = df.copy()

    # Drop rows with NaN due to indicator calculation
    df.dropna(inplace=True)

    # Target: 1 if price in N days > current price, else 0
    df['target'] = (df['close'].shift(-target_shift) > df['close']).astype(int)
    df.dropna(inplace=True)  # drop last N rows with NaN target

    # Features (excluding raw prices and volume)
    exclude_cols = ['adj close', 'close', 'open', 'high', 'low', 'volume', 'target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].values
    y = df['target'].values

    # Train/test split respecting temporal order
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    from utils.data_loader import fetch_copper_data
    from models.feature_engineering import add_technical_indicators

    df = fetch_copper_data()
    df = add_technical_indicators(df)

    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    print("✅ Data preparation complete:")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
