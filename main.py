import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import fetch_copper_data
from models.feature_engineering import add_technical_indicators

# Fetch raw copper data
df = fetch_copper_data()

# Add technical indicators
df = add_technical_indicators(df)

# Print last rows to confirm
print(df.tail())

from models.data_preparation import prepare_data
