# Project Deep Dive — Copper Price Forecasting & Trading Strategy

## Overview

This project aims to build a predictive trading strategy for copper prices based on machine learning models applied to technical indicators. It covers data collection, feature engineering, the Random Forest model, backtesting, and a visualization dashboard.

## 🎯 Motivation and Project Objectives

Copper is a key industrial metal, highly sensitive to global economic cycles, making price forecasting valuable for traders and analysts. This project seeks to:

- Collect and analyze historical copper price data  
- Engineer relevant technical indicators to capture market signals  
- Train a robust classifier (Random Forest) to predict price direction  
- Backtest a trading strategy based on model signals to evaluate profitability  
- Provide an interactive dashboard for real-time visualization and monitoring  

A predictive strategy on copper can help market participants anticipate price movements, manage risks, and identify profitable trading opportunities on this volatile and economically important metal.

This project is a practical example of **financial data science**, **feature engineering**, **machine learning**, and **quantitative trading**.

---

### 1. Why Copper?

#### 1.1 Economic and Industrial Importance

Copper is a **fundamental** industrial metal widely used in energy, construction, electronics, and infrastructure sectors. Its demand often reflects overall economic health because:

- It is a **leading indicator** of global economic cycles, especially in emerging and developed economies.  
- Growth in renewable energy, electric vehicles, and urban infrastructure sectors increases copper demand.

#### 1.2 Volatility and Trading Opportunities

- Copper is **highly liquid** in commodity markets, with large trading volumes.  
- Its **moderate to high volatility** makes it attractive for short-term trading strategies.  
- Copper prices are sensitive to geopolitics, trade policies, and supply/demand tensions, providing exploitable signals.

#### 1.3 Data Availability

- Copper price data is easily accessible and reliable from sources like Yahoo Finance (`yfinance`), facilitating quantitative analysis projects.  
- Copper has a long historical data series enabling robust model building.

#### 1.4 Diversified Sector Coverage

- Working on copper touches multiple themes (energy, industry, environment), making the project relevant both for trading and economic insight.

---

#### Cause-Effect Link Between Copper and the Market

As a barometer of global economic activity, copper quickly reacts to macroeconomic changes, trade policies, and geopolitical tensions. For example, an increase in industrial demand or a supply shock directly impacts its price, which in turn influences investor and trader decisions in commodity markets. This dynamic creates a feedback loop between overall economic health and copper price, making it crucial for forecasting and trading strategies.

#### Link to Recent Geopolitical Events

For instance, current geopolitical tensions in the Middle East, notably involving Iran, affect commodity markets through supply chain and energy risks. These disruptions can indirectly impact industrial copper demand (via energy costs and economic confidence), causing increased price volatility. [See recent analysis on geopolitical impact on commodities](https://www.reuters.com/business/energy/trump-iran-oil-policy-impact-2025-06-23/).

---

## 2. Technical and Methodological Choices

### 2.1 Selection of Technical Indicators

- **MA20 & MA50:** Capture short- and medium-term trends, classic in technical analysis.  
- **RSI (14 days):** Measures momentum and identifies overbought/oversold zones.  
- **Bollinger Bands:** Assess volatility and potential reversals or breakouts.  
- **MACD:** Momentum indicator tracking trend-following behavior.  
- **Momentum & ROC:** Quantify speed and magnitude of price moves.  
- **20-day Volatility:** Measures price dispersion, useful for risk management.

These indicators are well-established and provide complementary information: trend, momentum, and volatility.

### 2.2 Why Random Forest?

- **Robustness** against financial data noise.  
- **Non-linear capacity** to capture complex relationships without heavy tuning.  
- **Explainability** through feature importance.  
- **Resistance to overfitting** via ensemble of decision trees.  
- Well-suited for an exploratory project without massive data.

### 2.3 Identified Limitations

- Not yet accounted for:  
  - Transaction costs and slippage  
  - Liquidity impact  
  - Unmodeled macroeconomic risks  
  - Dynamic adaptation to market regime shifts  
- Static model trained on historical data — risk of performance degradation with structural changes.

---

## 3. Challenges and Solutions

- **Data cleaning and quality:** Missing or outlier data detected, imputed or excluded.  
- **Feature engineering:** Multiple tests to avoid redundancy and multicollinearity.  
- **Overfitting:** Cross-validation and test set evaluation to ensure generalization.  
- **Dashboard deployment:** Managing library version compatibility and performance via virtual environments.

---

## 4. Future Improvement Paths

- Adding **hyperparameter tuning** using GridSearch or Bayesian Optimization.  
- Exploring advanced models: XGBoost, LightGBM, neural networks (LSTM, Transformers).  
- Integrating **macroeconomic data**, news sentiment, and fundamentals (stocks, production).  
- Enhanced backtesting including **transaction costs, slippage, and risk management**.  
- Implementing adaptive or online learning strategies.  
- Cloud deployment (Heroku, AWS) with secure dashboard access.  
- Documentation and unit testing for maintainability.

---

## 5. Conclusion

This project represents a solid first step toward quantified trading strategies on copper. It demonstrates an effective combination of technical analysis, machine learning, and backtesting. With proposed evolutions, it can become a powerful tool for a trader assistant or quant analyst.

---
