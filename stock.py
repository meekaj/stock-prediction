import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Download stock data from Yahoo Finance
stock_symbol = "AAPL"
try:
    data = yf.download(stock_symbol, start="2022-01-01", end="2024-05-07")
except Exception as e:
    print(f"Error fetching data: {e}")
    exit()

data['SMA'] = data['Close'].rolling(window=15).mean()
data['Tomorrow'] = data['Close'].shift(-1)
data.fillna(method='ffill', inplace=True)
data.dropna(inplace=True)

X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA']]
y = data['Tomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

try:
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X)
except Exception as e:
    print(f"Error during model training or prediction: {e}")
    exit()

predictions_series = pd.Series(predictions, index=data.index)

plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Actual Close', color='blue')
plt.plot(predictions_series, label='Predicted Close', color='red')
plt.title(f"Stock Price Prediction for {stock_symbol}")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

future_dates = pd.date_range(start=data.index[-1], periods=8, closed='right')
future_data = pd.DataFrame(index=future_dates, columns=X.columns)
for column in ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA']:
    future_data[column] = data[column].iloc[-1]


# Calculate the trend on the last 30 days of data
last_30_days = data[-30:]
days_since_start = np.arange(len(last_30_days)).reshape(-1, 1)
close_prices = last_30_days['Close'].values.reshape(-1, 1)

# Fit a linear model
trend_model = LinearRegression()
trend_model.fit(days_since_start, close_prices)

# Predict future 'Close' prices
future_days = np.arange(len(last_30_days), len(last_30_days) + 7).reshape(-1, 1)
predicted_close_prices = trend_model.predict(future_days)

# Apply these predictions to future_data
future_data['Close'] = predicted_close_prices.flatten()


if future_data.isnull().values.any() or np.isinf(future_data.values).any():
    print("Future data contains invalid values.")
    future_data.fillna(method='ffill', inplace=True)

try:
    future_predictions = model.predict(future_data)
except Exception as e:
    print(f"Error predicting future data: {e}")
    exit()

# Assume other future data features are constant or use similar methods to predict them
future_data['Open'] = future_data['Close']  # Simplification
future_data['High'] = future_data['Close'] * 1.01  # Assume high is 1% higher than close
future_data['Low'] = future_data['Close'] * 0.99  # Assume low is 1% lower than close
future_data['Volume'] = data['Volume'].iloc[-1]  # Keep volume constant or model as needed
future_data['SMA'] = data['SMA'].iloc[-1]  # Keep SMA constant or model as needed

# Predict using the model
future_predictions = model.predict(future_data)

# Plotting the future predictions
plt.figure(figsize=(14, 7))
plt.plot(future_dates, future_predictions, label='Future Predicted Close', marker='o', color='green')
plt.title(f"Future Stock Price Prediction for {stock_symbol} (Next Week)")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(future_dates, future_predictions_series, label='Future Predicted Close', marker='o', color='green')
plt.title(f"Future Stock Price Prediction for {stock_symbol} (Next Week)")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()