import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
import ta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.formula.api as smf
import os




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

# Advanced Volatility Modeling (GARCH) for all FAANG stocks
for ticker in tickers:
    print(f"\nGARCH Model for {ticker} Stock Returns:")
    returns_ticker = returns[ticker].dropna() * 100  # Scale for percentage returns
    try:
        garch_model = arch_model(returns_ticker, vol='Garch', p=1, q=1).fit(disp='off')
        print(garch_model.summary())

        # Plotting the conditional volatility
        plt.figure(figsize=(10, 6))
        plt.plot(garch_model.conditional_volatility, label=f'{ticker} Conditional Volatility')
        plt.title(f"GARCH Conditional Volatility for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Failed to fit GARCH model for {ticker}: {e}")

# Forecasting Accuracy Evaluation for all FAANG stocks
forecast_accuracy = []

print("\n=== Forecast Accuracy Evaluation for GARCH Models ===")
for ticker in tickers:
    print(f"\nEvaluating Forecast Accuracy for {ticker} Stock Returns:")
    returns_ticker = returns[ticker].dropna() * 100  # Scale for percentage returns
    try:
        # Fit the GARCH model
        garch_model = arch_model(returns_ticker, vol='Garch', p=1, q=1).fit(disp='off')

        # Forecasting the next 30 days
        forecast = garch_model.forecast(horizon=30)
        forecast_volatility = forecast.variance[-1:].T

        # Actual volatility for comparison (using realized volatility)
        actual_volatility = realized_volatility[ticker].tail(30)

        # Accuracy Metrics
        mae = np.mean(np.abs(forecast_volatility.values.flatten() - actual_volatility.values))
        rmse = np.sqrt(np.mean((forecast_volatility.values.flatten() - actual_volatility.values) ** 2))
        mape = np.mean(np.abs((forecast_volatility.values.flatten() - actual_volatility.values) / actual_volatility.values)) * 100

        # Log-Likelihood, AIC, BIC
        log_likelihood = garch_model.loglikelihood
        aic = garch_model.aic
        bic = garch_model.bic

        # Print Evaluation Metrics
        print(f"Model: GARCH(1,1) for {ticker}")
        print(f"Log-Likelihood: {log_likelihood:.2f}")
        print(f"AIC: {aic:.2f}")
        print(f"BIC: {bic:.2f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")

        # Save accuracy metrics to list
        forecast_accuracy.append([ticker, log_likelihood, aic, bic, mae, rmse, mape])

        # Get GARCH parameters for the forecast equation
        omega = garch_model.params['omega']
        alpha = garch_model.params['alpha[1]']
        beta = garch_model.params['beta[1]']

        # Forecasted volatility equation representation
        equation_forecast = f'σ² = {omega:.4f} + {alpha:.4f} * r²(t-1) + {beta:.4f} * σ²(t-1)'

        # Plot Actual vs Forecasted Volatility with Forecast Equation
        plt.figure(figsize=(10, 6))
        plt.plot(actual_volatility.index, actual_volatility.values, label="Actual Volatility", color="blue")
        plt.plot(actual_volatility.index, forecast_volatility.values.flatten(), label="Forecasted Volatility",
                 color="red", linestyle="--")

        # Linear Regression between Actual and Forecasted Volatility
        X = forecast_volatility.values.flatten().reshape(-1, 1)  # Forecasted as X
        y = actual_volatility.values  # Actual as Y
        model = LinearRegression().fit(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        equation_regression = f'Y = {slope:.2f}X + {intercept:.2f}'

        # Adding the regression line to the plot
        regression_line = model.predict(X)
        plt.plot(actual_volatility.index, regression_line, label=f'Regression: {equation_regression}', color="green",
                 linestyle="-.")

        # Adding the forecasted volatility equation as text
        plt.text(0.05, 0.95, f'Forecast Eq: {equation_forecast}', transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.title(f"Actual vs Forecasted Volatility for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.legend()
        plt.show()


    except Exception as e:
        print(f"Forecasting failed for {ticker}: {e}")

# Save Forecast Accuracy Results to DataFrame
forecast_df = pd.DataFrame(forecast_accuracy, columns=["Ticker", "Log-Likelihood", "AIC", "BIC", "MAE", "RMSE", "MAPE"])
forecast_df.to_csv("FAANG_forecast_accuracy.csv", index=False)

print("Forecast accuracy metrics saved to FAANG_forecast_accuracy.csv")

# Forecasting Accuracy Evaluation for all FAANG stocks
forecast_accuracy = []

print("\n=== Forecast Accuracy Evaluation for GARCH Models ===")
for ticker in tickers:
    print(f"\nEvaluating Forecast Accuracy for {ticker} Stock Returns:")
    returns_ticker = returns[ticker].dropna() * 100  # Scale for percentage returns
    try:
        # Fit the GARCH model
        garch_model = arch_model(returns_ticker, vol='Garch', p=1, q=1).fit(disp='off')

        # Forecasting the next 30 days
        forecast = garch_model.forecast(horizon=30)
        forecast_volatility = forecast.variance[-1:].T

        # Actual volatility for comparison (using realized volatility)
        actual_volatility = realized_volatility[ticker].tail(30)

        # Accuracy Metrics
        mae = np.mean(np.abs(forecast_volatility.values.flatten() - actual_volatility.values))
        rmse = np.sqrt(np.mean((forecast_volatility.values.flatten() - actual_volatility.values) ** 2))
        mape = np.mean(np.abs((forecast_volatility.values.flatten() - actual_volatility.values) / actual_volatility.values)) * 100

        # Log-Likelihood, AIC, BIC
        log_likelihood = garch_model.loglikelihood
        aic = garch_model.aic
        bic = garch_model.bic

        # Print Evaluation Metrics
        print(f"Model: GARCH(1,1) for {ticker}")
        print(f"Log-Likelihood: {log_likelihood:.2f}")
        print(f"AIC: {aic:.2f}")
        print(f"BIC: {bic:.2f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")

        # Save accuracy metrics to list
        forecast_accuracy.append([ticker, log_likelihood, aic, bic, mae, rmse, mape])

        # Plot Actual vs Forecasted Volatility with Regression Equation
        plt.figure(figsize=(10, 6))
        plt.plot(actual_volatility.index, actual_volatility.values, label="Actual Volatility", color="blue")
        plt.plot(actual_volatility.index, forecast_volatility.values.flatten(), label="Forecasted Volatility",
                 color="red", linestyle="--")

        # Linear Regression between Actual and Forecasted Volatility
        X = forecast_volatility.values.flatten().reshape(-1, 1)  # Forecasted as X
        y = actual_volatility.values  # Actual as Y
        model = LinearRegression().fit(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        equation = f'Y = {slope:.2f}X + {intercept:.2f}'

        # Adding the regression line to the plot
        regression_line = model.predict(X)
        plt.plot(actual_volatility.index, regression_line, label=f'Regression: {equation}', color="green",
                 linestyle="-.")

        plt.title(f"Actual vs Forecasted Volatility for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.legend()
        plt.show()


    except Exception as e:
        print(f"Forecasting failed for {ticker}: {e}")

# Save Forecast Accuracy Results to DataFrame
forecast_df = pd.DataFrame(forecast_accuracy, columns=["Ticker", "Log-Likelihood", "AIC", "BIC", "MAE", "RMSE", "MAPE"])
forecast_df.to_csv("FAANG_forecast_accuracy.csv", index=False)

print("Forecast accuracy metrics saved to FAANG_forecast_accuracy.csv")


# Statistical Testing (Correlation & Granger causality for all FAANG stocks)
print("Correlations between returns and volume:")
for ticker in tickers:
    correlations = returns[ticker].corr(volume[ticker])
    print(f"{ticker}: {correlations:.6f}")

    print(f"Granger causality test for {ticker} returns and volume:")
    try:
        granger_results = grangercausalitytests(
            pd.concat([returns[ticker], volume[ticker]], axis=1).dropna(), maxlag=5
        )
    except Exception as e:
        print(f"Granger causality test failed for {ticker}: {e}")

# Save backtest results for all FAANG stocks
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

# Ensure the index is properly set as dates
if not final_data.index.name == 'Date':
    final_data.reset_index(inplace=True)
    final_data.rename(columns={'index': 'Date'}, inplace=True)

# Flatten MultiIndex columns
final_data.columns = ['_'.join(map(str, col)).strip() if isinstance(col, tuple) else col for col in final_data.columns]

# Set the date as the index before saving
final_data.set_index('Date', inplace=True)

# Save to CSV and Excel
final_data.to_csv('FAANG_combined_analysis_dataset.csv')
final_data.to_excel('FAANG_combined_analysis_dataset.xlsx', sheet_name='FAANG_Data')

print("Combined dataset with dates saved to CSV and Excel!")




# NEW WORKING IN REGRESSION
# Define tickers and date range
tickers = ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL']
start_date = '2020-01-01'
end_date = '2025-01-01'

# Download historical data
data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
adj_close = data['Adj Close']
volume = data['Volume']

# Calculate returns
returns = adj_close.pct_change().dropna()

# -------------------------------
# Regression Analysis Across FAANG Stocks
# -------------------------------
# Stack returns and volume for each ticker
stacked_reg = pd.concat([
    pd.DataFrame({
        "Return": returns[ticker],
        "Volume": volume[ticker],
        "Ticker": ticker
    })
    for ticker in tickers
])
# Feature engineering: log(volume) and high-volume dummy
stacked_reg['LogVolume'] = np.log(stacked_reg['Volume'] + 1)
stacked_reg['HighVolume'] = stacked_reg.groupby('Ticker')['Volume'].transform(lambda x: (x > x.median()).astype(int))
stacked_reg.dropna(inplace=True)

# Run OLS regression with interaction terms
reg_model = smf.ols("Return ~ LogVolume * C(Ticker) + HighVolume", data=stacked_reg).fit()

# Build coefficients table
reg_results = pd.DataFrame({
    "Coefficient": reg_model.params,
    "Std Error": reg_model.bse,
    "t-Statistic": reg_model.tvalues,
    "p-Value": reg_model.pvalues,
    "CI Lower": reg_model.conf_int()[0],
    "CI Upper": reg_model.conf_int()[1]
})

# Extract base coefficients (for the reference stock)
beta_0 = reg_model.params['Intercept']
beta_1 = reg_model.params['LogVolume']
beta_2 = reg_model.params['HighVolume']
print("Beta 0 (Intercept):", beta_0)
print("Beta 1 (LogVolume coefficient):", beta_1)
print("Beta 2 (HighVolume coefficient):", beta_2)

# Build model summary table
reg_summary = pd.DataFrame(list({
    "R-squared": reg_model.rsquared,
    "Adj. R-squared": reg_model.rsquared_adj,
    "F-statistic": reg_model.fvalue,
    "F p-value": reg_model.f_pvalue,
    "No. observations": int(reg_model.nobs),
    "AIC": reg_model.aic,
    "BIC": reg_model.bic,
    "Log-Likelihood": reg_model.llf
}.items()), columns=["Metric", "Value"])

# -------------------------------
# Save Regression Results to Excel in the specified folder
# -------------------------------
# Set the path to your desired folder
save_path = "/Users/zafirkamdar/PycharmProjects/BUEC 420 Term paper"
excel_file = os.path.join(save_path, "FAANG_volume_regression_results.xlsx")

with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
    reg_results.to_excel(writer, sheet_name="Regression Coefficients", index=True)
    reg_summary.to_excel(writer, sheet_name="Model Summary", index=False)

# Confirm file saved
if os.path.exists(excel_file):
    print("✅ Regression results saved to:")
    print(excel_file)
else:
    print("❌ File not saved — check path or permissions.")




#OLS AND R SQUARED
import statsmodels.api as sm
import pandas as pd
import os

# Your tickers and dataframe
tickers = ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL']

# Excel save path
save_path = "/Users/zafirkamdar/PycharmProjects/BUEC 420 Term paper"
excel_output = os.path.join(save_path, "OLS_regression_summary_FAANG.xlsx")

# Check if path exists, if not, print warning
if not os.path.exists(save_path):
    print(f"❌ Directory does not exist: {save_path}")
else:
    print(f"📁 Saving to: {excel_output}")

# List to collect results
ols_summary_list = []

# Run OLS per ticker
for ticker in tickers:
    df = stacked_reg[stacked_reg["Ticker"] == ticker]
    if df.empty:
        print(f"⚠️ No data for {ticker}, skipping...")
        continue

    X = sm.add_constant(df["LogVolume"])
    y = df["Return"]

    try:
        model = sm.OLS(y, X).fit()
        ols_summary_list.append({
            "Ticker": ticker,
            "Intercept": model.params.get('const', float('nan')),
            "LogVolume Coef": model.params.get('LogVolume', float('nan')),
            "p-Value": model.pvalues.get('LogVolume', float('nan')),
            "R-squared": model.rsquared,
            "Adj. R-squared": model.rsquared_adj,
            "AIC": model.aic,
            "BIC": model.bic,
            "F-statistic": model.fvalue
        })
        print(f"✅ Processed {ticker}")
    except Exception as e:
        print(f"❌ OLS failed for {ticker}: {e}")

# Create DataFrame
ols_summary_df = pd.DataFrame(ols_summary_list)

# Save to Excel
try:
    with pd.ExcelWriter(excel_output, engine='openpyxl') as writer:
        ols_summary_df.to_excel(writer, index=False, sheet_name="OLS_Results")
    print("✅ OLS regression results successfully saved to Excel.")
except Exception as e:
    print(f"❌ Failed to save Excel file: {e}")

