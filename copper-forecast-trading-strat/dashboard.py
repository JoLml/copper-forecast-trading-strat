import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import os
import sys

# Add path to modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from utils.data_loader import fetch_copper_data
from models.feature_engineering import add_technical_indicators
from models.data_preparation import prepare_data

# UI setup
st.set_page_config(page_title="Copper Forecast Dashboard", layout="centered")
st.title("🔍 Copper Forecast Dashboard")
st.markdown("This app predicts the **short-term copper price direction** using a Random Forest model trained on technical indicators.")

# Load model and scaler files
model_path = "models/random_forest_model.joblib"
scaler_path = "models/scaler.joblib"

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.error("❌ Model not found. Please run `trainer.py` to train the model first.")
    st.stop()

model = load(model_path)
scaler = load(scaler_path)

# Load and prepare data
df = fetch_copper_data()
df = add_technical_indicators(df)
X_train, X_test, y_train, y_test, _ = prepare_data(df, scale=True)

# Latest prediction
latest_features = X_test[-1].reshape(1, -1)
prediction = model.predict(latest_features)[0]
proba = model.predict_proba(latest_features)[0][prediction]

st.subheader("📈 Latest Model Prediction")
if prediction == 1:
    st.success(f"The model predicts an UP trend 📈 with {proba * 100:.2f}% confidence")
else:
    st.warning(f"The model predicts a DOWN trend 📉 with {proba * 100:.2f}% confidence")

# Simple backtest visualization
st.subheader("💹 Simplified Backtest Performance")

# Align data on test set
df_test = df.iloc[-len(y_test):].copy()
df_test["Signal"] = model.predict(X_test)
df_test["Market Return"] = df_test["close"].pct_change()
df_test["Strategy Return"] = df_test["Market Return"] * df_test["Signal"].shift(1)
df_test.dropna(inplace=True)

df_test["Cumulative Market"] = (1 + df_test["Market Return"]).cumprod()
df_test["Cumulative Strategy"] = (1 + df_test["Strategy Return"]).cumprod()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_test.index, df_test["Cumulative Market"], label="🏦 Market (Buy & Hold)")
ax.plot(df_test.index, df_test["Cumulative Strategy"], label="🤖 Model Strategy")
ax.legend()
ax.set_title("Cumulative Returns")
st.pyplot(fig)

# Performance summary
perf_strategy = (df_test["Cumulative Strategy"].iloc[-1] - 1) * 100
perf_market = (df_test["Cumulative Market"].iloc[-1] - 1) * 100

st.write(f"📊 **Strategy Performance:** {perf_strategy:.2f}%")
st.write(f"📊 **Market Performance:** {perf_market:.2f}%")
