# 🟠 Copper Price Forecasting & Trading Strategy with Machine Learning

This project develops and backtests a **predictive trading strategy** for short-term copper price movements using machine learning on technical indicators.

## 🎯 Project Motivation & Objectives

Copper is a key industrial metal, highly sensitive to global economic cycles, making its price forecasting valuable for traders and analysts. This project aims to:

- Collect and analyze historical copper price data  
- Engineer relevant technical indicators to capture market signals  
- Train a robust machine learning classifier (Random Forest) to predict price direction  
- Backtest a trading strategy based on model signals to evaluate profitability  
- Provide an interactive dashboard for real-time visualization and monitoring  

This is a practical example of **financial data science**, **feature engineering**, **machine learning classification**, and **quantitative trading**.

## 🔍 Project Highlights

- ✅ Data acquisition from Yahoo Finance (`yfinance` package)  
- 📊 Feature engineering with popular technical indicators:  
  - Moving averages (MA20, MA50)  
  - RSI (Relative Strength Index)  
  - Bollinger Bands  
  - MACD (Moving Average Convergence Divergence)  
  - Momentum and Rate of Change (ROC)  
  - Volatility measures  
- 🌲 Machine Learning:  
  - Random Forest Classifier for direction prediction (up/down)  
  - Model evaluation with classification reports and feature importance  
- 📈 Strategy backtesting:  
  - Simulated trades based on model-generated buy/sell signals  
  - Cumulative returns comparison against buy-and-hold benchmark  
- 🖥️ Interactive Streamlit dashboard:  
  - Latest prediction with confidence scores  
  - Visual backtest results and performance metrics  
  - Raw data preview for transparency  

## 📌 Key Metrics (Example)

| Metric                        | Value            |
|------------------------------|------------------|
| Model Accuracy (test set)     | ~XX%             |
| Strategy Cumulative Return    | +YY%             |
| Market (Buy & Hold) Return    | +ZZ%             |
| Feature Importance Highlights | MA20, RSI, MACD  |

*(Replace XX, YY, ZZ with your actual results)*

## ⚙️ Technical Indicators Summary

- **MA20 & MA50:** Capture short- and medium-term price trends  
- **RSI (14 days):** Measures momentum and overbought/oversold levels  
- **Bollinger Bands:** Identify volatility and potential price breakouts  
- **MACD:** Trend-following momentum indicator  
- **Momentum & ROC:** Quantify speed and magnitude of price changes  
- **20-day Volatility:** Reflects short-term price fluctuations  

These indicators provide a comprehensive view of trend, momentum, and risk factors.

## 🧰 Tech Stack & Dependencies

- Python 3.13+  
- yfinance  
- pandas, numpy  
- scikit-learn  
- matplotlib, seaborn  
- streamlit  
- xgboost, prophet (optional for future enhancements)  

## 🚀 Getting Started & How to Run

```bash
git clone https://github.com/JoLml/copper-forecast-trading-strat.git
cd copper-forecast-trading-strat

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

# Train the model
python main.py --mode train

# Backtest the strategy
python main.py --mode backtest

# Train and backtest sequentially
python main.py --mode all

# Launch the interactive dashboard
streamlit run dashboard.py
