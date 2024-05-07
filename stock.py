import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Fetch stock data from Yahoo Finance
stock_symbol = "AAPL"
data = yf.download(stock_symbol, start="2022-01-01", end="2023-01-01")
data['SMA'] = data['Close'].rolling(window=15).mean()  # Simple Moving Average

# Prepare features
X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA']].fillna(method='bfill')
y = data['Close'].shift(-1).fillna(method='ffill')  # Predict next day close

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Recommendation system based on sentiment and last prediction
last_real_price = data.iloc[-1]['Close']
predicted_price = predictions[-1]
average_sentiment = 0.35  # Example fixed value, replace with your scraped or pre-compiled sentiment data
recommendation = "Buy" if predicted_price > last_real_price and average_sentiment > 0 else "Hold"

# Plot results
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Actual Close', color='blue')
# Create a temporary DataFrame to hold predictions with the same index as X_test for correct plotting
predictions_df = pd.DataFrame(predictions, index=X_test.index, columns=['Predicted Close'])
plt.plot(predictions_df['Predicted Close'], label='Predicted Close', color='red')
plt.title(f"Stock Price Prediction and Sentiment Analysis for {stock_symbol}")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Print the average sentiment and recommendation
print(f"Average sentiment for {stock_symbol}: {average_sentiment:.2f}")
print(f"Recommendation: {recommendation}")
