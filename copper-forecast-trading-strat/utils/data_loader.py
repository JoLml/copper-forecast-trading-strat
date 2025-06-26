import yfinance as yf
import pandas as pd

def fetch_copper_data(start='2015-01-01', end=None):
    """
    Download historical copper data (COMEX continuous contract) from Yahoo Finance.
    Ticker used: 'HG=F'

    Args:
        start (str): Start date in 'YYYY-MM-DD' format
        end (str): End date in 'YYYY-MM-DD' format (defaults to today)

    Returns:
        pd.DataFrame: OHLCV data (Open, High, Low, Close, Volume) for copper
    """
    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')

    # Disable auto_adjust to avoid MultiIndex columns
    df = yf.download('HG=F', start=start, end=end, auto_adjust=False, progress=False)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Normalize column names (handle tuple case)
    df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]

    return df

if __name__ == "__main__":
    df = fetch_copper_data()
    if df.empty:
        print("⚠️ No data fetched. Check your internet connection or the ticker symbol.")
    else:
        print(df.tail())
