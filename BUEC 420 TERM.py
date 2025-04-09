import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
import ta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import grangercausalitytests

# Define FAANG tickers and control variables
tickers = ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL']
market_tickers = ['^GSPC', '^VIX']  # S&P 500 and VIX
all_tickers = tickers + market_tickers

# Explicit data collection period (Jan 2020 to Jan 2025)
start_date = '2020-01-01'
end_date = '2025-01-01'

# Download historical data
data = yf.download(all_tickers, start=start_date, end=end_date, auto_adjust=False)

# Primary data extraction
open_prices = data['Open']
close_prices = data['Close']
high_prices = data['High']
low_prices = data['Low']
adj_close = data['Adj Close']
volume = data['Volume']

# Calculate derived data
returns = adj_close[tickers].pct_change().dropna()
historical_volatility = returns.rolling(window=21).std() * np.sqrt(252)
realized_volatility = returns.pow(2).rolling(window=21).sum()

# Risk and Performance Metrics
sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
downside_std = returns[returns < 0].std() * np.sqrt(252)
sortino_ratio = (returns.mean() * 252) / downside_std
cumulative_returns = (1 + returns).cumprod()
rolling_max = cumulative_returns.cummax()
drawdown = (cumulative_returns - rolling_max) / rolling_max
max_drawdown = drawdown.min()

# Value of shares traded (daily turnover in USD)
daily_turnover = volume[tickers] * adj_close[tickers]

# Technical Indicators
technical_indicators = pd.DataFrame(index=adj_close.index)
for ticker in tickers:
    technical_indicators[f'{ticker}_SMA50'] = ta.trend.sma_indicator(adj_close[ticker], window=50)
    technical_indicators[f'{ticker}_RSI'] = ta.momentum.rsi(adj_close[ticker], window=14)
    technical_indicators[f'{ticker}_MACD'] = ta.trend.macd(adj_close[ticker])

# Backtesting Strategy (Simple Moving Average Crossover)
backtest_results = pd.DataFrame(index=returns.index)
for ticker in tickers:
    short_sma = adj_close[ticker].rolling(window=50).mean()
    long_sma = adj_close[ticker].rolling(window=200).mean()
    signal = np.where(short_sma > long_sma, 1, -1)
    signal = pd.Series(signal, index=adj_close[ticker].index)
    backtest_results[f'{ticker}_signal'] = signal
    backtest_results[f'{ticker}_returns'] = returns[ticker] * signal.shift(1)
    backtest_results[f'{ticker}_cumulative'] = (1 + backtest_results[f'{ticker}_returns']).cumprod()

# Visualizations: Cumulative Returns and Correlation Heatmap
plt.figure(figsize=(14, 8))
for ticker in tickers:
    plt.plot(cumulative_returns.index, cumulative_returns[ticker], label=f'{ticker} Cumulative Returns')
plt.title("Cumulative Returns of FAANG Stocks")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(returns.corr(), annot=True, cmap="coolwarm")
plt.title("FAANG Returns Correlation Heatmap")
plt.show()

# Regression Plots with Equation
for ticker in tickers:
    plt.figure(figsize=(8, 6))
    X = np.arange(len(adj_close[ticker])).reshape(-1, 1)
    y = adj_close[ticker].values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    trendline = model.predict(X)
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    equation = f'Price = {slope:.2f} * Day + {intercept:.2f}'
    plt.plot(adj_close.index, adj_close[ticker], label=f'{ticker} Prices')
    plt.plot(adj_close.index, trendline, label=f'Trendline: {equation}', linestyle="--")
    plt.title(f"Regression Line for {ticker} Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Price")
    plt.legend()
    plt.show()

# Save backtest results
backtest_results.to_csv('FAANG_backtest_results.csv')

# Combine all data into a final DataFrame
final_data = pd.concat({
    'Open': open_prices,
    'Close': close_prices,
    'High': high_prices,
    'Low': low_prices,
    'Adj Close': adj_close,
    'Volume': volume,
    'Daily Turnover (USD)': daily_turnover,
    'Returns': returns,
    'Historical Volatility': historical_volatility,
    'Realized Volatility': realized_volatility,
    'Sharpe Ratio': sharpe_ratio,
    'Sortino Ratio': sortino_ratio,
    'Max Drawdown': max_drawdown,
    'Technical Indicators': technical_indicators
}, axis=1)

# Reset index to explicitly list dates
final_data.reset_index(inplace=True)

# Save to CSV and Excel
final_data.to_csv('FAANG_combined_analysis_dataset.csv', index=False)
final_data.to_excel('FAANG_combined_analysis_dataset.xlsx', sheet_name='FAANG_Data', index=False)

print("Combined dataset with advanced analytics and visualizations saved to CSV and Excel!")