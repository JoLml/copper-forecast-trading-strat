# 🟠 Copper Price Forecasting & Trading Strategy with Machine Learning

## Overview

This project implements a **machine learning-based predictive trading strategy** targeting short-term price movements of copper, a crucial industrial metal. By leveraging technical indicators and a Random Forest classifier, it forecasts market direction and backtests a rule-based trading strategy to evaluate profitability.

It showcases practical skills in **financial data science**, **feature engineering**, **machine learning classification**, and **quantitative trading**.

---

## Table of Contents

- [Motivation & Objectives](#motivation--objectives)  
- [Data & Feature Engineering](#data--feature-engineering)  
- [Modeling Approach](#modeling-approach)  
- [Strategy Backtesting](#strategy-backtesting)  
- [Interactive Dashboard](#interactive-dashboard)  
- [Results & Metrics](#results--metrics)  
- [Project Structure](#project-structure)  
- [Tech Stack & Dependencies](#tech-stack--dependencies)  
- [Getting Started](#getting-started)  
- [Future Work & Improvements](#future-work--improvements)  

---

## Motivation & Objectives

Copper prices are highly sensitive to global economic cycles, making accurate forecasting valuable for traders and analysts. This project aims to:

- Collect and preprocess historical copper price data  
- Engineer relevant technical indicators to capture market signals  
- Train a Random Forest classifier to predict price direction (up/down)  
- Backtest a trading strategy driven by model signals to assess profitability  
- Provide a user-friendly dashboard for real-time monitoring  

---

## Data & Feature Engineering

- Historical copper price data sourced from Yahoo Finance via `yfinance`  
- Key technical indicators computed include:  
  - Moving Averages (MA20, MA50)  
  - Relative Strength Index (RSI)  
  - Bollinger Bands  
  - Moving Average Convergence Divergence (MACD)  
  - Momentum & Rate of Change (ROC)  
  - 20-day Volatility  

These features capture trend, momentum, and volatility aspects of the market.

---

## Modeling Approach

- Random Forest Classifier trained on labeled data representing price movement direction  
- Model evaluated using classification metrics (accuracy, precision, recall)  
- Feature importance analyzed to identify most predictive indicators  

---

## Strategy Backtesting

- Trading signals derived from model predictions trigger simulated buy/sell actions  
- Performance benchmarked against buy-and-hold strategy  
- Returns and risk metrics computed and visualized  

---

## Interactive Dashboard

Built with Streamlit, the dashboard provides:  

- Latest price prediction with confidence score  
- Historical backtest performance charts  
- Raw data and indicator visualization  

---

## Results & Metrics

| Metric                        | Value      |
|------------------------------|------------|
| Model Accuracy (test set)     | XX %       |
| Strategy Cumulative Return    | +YY %      |
| Buy & Hold Return             | +ZZ %      |
| Top Features                 | MA20, RSI, MACD |

*Note: Replace XX, YY, ZZ with your actual results.*

---

## Project Structure

copper-forecast-trading-strat/
├── data/ # Raw and processed data files
├── models/ # Model code and saved artifacts
├── notebooks/ # Exploratory data analysis and experiments
├── strategy/ # Backtesting engine and trading logic
├── utils/ # Helper functions (data loading, feature engineering)
├── dashboard.py # Streamlit app entrypoint
├── main.py # Main script to train, backtest, or run all
├── requirements.txt # Python dependencies
└── README.md # This file


---

## Tech Stack & Dependencies

- Python 3.13+  
- yfinance, pandas, numpy  
- scikit-learn  
- matplotlib, seaborn  
- streamlit  
- Optional: xgboost, prophet (future extensions)  

---

## Getting Started

```bash
git clone https://github.com/JoLml/copper-forecast-trading-strat.git
cd copper-forecast-trading-strat

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Train the model
python main.py --mode train

# Backtest the strategy
python main.py --mode backtest

# Run training + backtesting sequentially
python main.py --mode all

# Launch the dashboard
streamlit run dashboard.py




## Future Work & Improvements

Incorporate alternative machine learning models (XGBoost, Neural Nets)
Add hyperparameter tuning and cross-validation
Integrate additional data sources (fundamental, macroeconomic indicators)
Develop paper trading or live trading connectivity
Enhance dashboard with real-time data feeds and alerts


Created by Johan Lameul — GitHub
Feel free to explore, raise issues, or contribute!