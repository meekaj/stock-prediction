# Stock Price Prediction

This project aims to predict the future closing prices of Apple Inc. (AAPL) stock using historical data from the last two years, obtained via Yahoo Finance. The prediction model employs simple linear regression and a rolling simple moving average (SMA) to forecast stock prices.

## Project Overview

The script fetches historical stock data for AAPL from Yahoo Finance, spanning from the 8th of May, 2022, to the 7th of May, 2024. It utilizes features such as open, high, low, close prices, and volume to predict the next day's closing price. The model also incorporates a 15-day simple moving average (SMA) to provide a smoothed indicator of the closing price trend.

## Features

- **Data Retrieval**: Downloads historical data for AAPL from Yahoo Finance.
- **Feature Engineering**: Calculates the 15-day SMA and prepares features for model training.
- **Model Training**: Uses linear regression to predict the next day's closing prices.
- **Future Prediction**: Predicts closing prices for the next week based on the model trained on historical data.

## Dependencies

- pandas
- numpy
- yfinance
- scikit-learn
- matplotlib

## Installation

To set up a local development environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone [repository-url]
2. Install the required Python packages:
   ```bash
    pip install pandas numpy matplotlib scikit-learn yfinance

## Usage
Run the script with the following command:

    ```bash
    python stock_prediction.py

The script will output plots showing the actual closing prices alongside the predicted closing prices based on historical data. It will also forecast the next week's closing prices.

## Limitations and Future Work
The current model uses simple linear regression, which assumes a linear relationship between the input features and the target variable. Future enhancements could include:

Implementing more complex models such as LSTM or ARIMA for better prediction accuracy.
Incorporating additional features like economic indicators or market sentiment analysis.
Using advanced feature selection techniques to improve model performance.