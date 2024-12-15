import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import yfinance as yf

# Function to fetch historical stock data
def fetch_stock_data(symbol):
    ticker = yf.Ticker(symbol)
    stock_data = ticker.history(period="max")
    stock_data.reset_index(inplace=True)
    return stock_data

# Function to preprocess stock data
def preprocess_data(stock_data):
    stock_data = stock_data[['Date', 'Close']].dropna()
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.sort_values('Date', inplace=True)

    # Add a Year column to stock data
    stock_data['Year'] = stock_data['Date'].dt.year

    # Keep only the last stock price data for each year
    stock_data = stock_data.groupby('Year').last().reset_index()
    return stock_data

# Function to engineer features
def engineer_features(data):
    data['LogPrice'] = np.log(data['Close'])
    return data

# Function to train the linear regression model
def train_model(data):
    X = data[['Year']]
    y = data['LogPrice']
    model = LinearRegression()
    model.fit(X, y)
    return model

# Function to calculate intrinsic value and Z-score boundaries
def calculate_intrinsic_value(data, model):
    data['PredictedLogPrice'] = model.predict(data[['Year']])
    data['IntrinsicValue'] = np.exp(data['PredictedLogPrice'])
    return data

# Function to calculate IRR from regression parameters
def calculate_irr(model):
    """
    Calculate the IRR (Internal Rate of Return) from the slope of the regression line.
    """
    slope = model.coef_[0]  # Extract the slope from the regression model
    irr = (np.exp(slope) - 1) * 100  # Convert the log growth rate to percentage
    return irr

# Function to display summary values
def display_summary(data, std_dev_log, model):
    # Current year values
    current_data = data.iloc[-1]
    current_price = current_data['Close']
    predicted_value = current_data['IntrinsicValue']

    # Convert log std deviation to linear scale
    upper_boundary = np.exp(current_data['PredictedLogPrice'] + 1.5 * std_dev_log)
    lower_boundary = np.exp(current_data['PredictedLogPrice'] + 0.5 * std_dev_log)

    # Determine buy or sell signal
    if current_price > upper_boundary:
        signal = "Sell"
    elif current_price < lower_boundary:
        signal = "Buy"
    else:
        signal = "Hold"

    # Display IRR, current price, predicted value, and signal
    irr = calculate_irr(model)
    print("\nSummary:\n")
    print(f"IRR (Internal Rate of Return): {irr:.2f}%")
    print(f"Current Actual Price: {current_price:.2f}")
    print(f"Predicted Intrinsic Value: {predicted_value:.2f}")
    print(f"Upper Boundary: {upper_boundary:.2f}")
    print(f"Lower Boundary: {lower_boundary:.2f}")
    print(f"Buy/Sell Signal: {signal}")

# Function to visualize results with regression line and Z-score region
def visualize_with_boundaries(data, std_dev_log, symbol):
    plt.figure(figsize=(12, 6))

    # Convert std deviation to linear scale boundaries
    upper_bound = np.exp(data['PredictedLogPrice'] + 1.5 * std_dev_log)
    lower_bound = np.exp(data['PredictedLogPrice'] + 0.5 * std_dev_log)

    # Plot actual prices (Close)
    plt.plot(data['Year'], data['Close'], label='Actual Close Price', color='black', linewidth=1.5)

    # Plot regression line
    plt.plot(data['Year'], data['IntrinsicValue'], label='Regression Line', color='red', linewidth=2)

    # Shade the Z-score region
    plt.fill_between(data['Year'], lower_bound, upper_bound, color='gray', alpha=0.3, label='Z-Score Region (+1.5 std +0.5 std)')

    # Set logarithmic scale for prices
    plt.yscale('log')

    # Add labels, legend, and grid
    plt.xlabel('Year')
    plt.ylabel('Price (Log Scale)')
    plt.title(f'{symbol} Regression Line with Z-Score Region and Actual Prices (Log Scale)')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Show plot
    plt.show()

# Main function to execute the workflow for a specific symbol
def analyze_stock(symbol):
    stock_data = fetch_stock_data(symbol)
    data = preprocess_data(stock_data)
    data = engineer_features(data)
    model = train_model(data)
    data = calculate_intrinsic_value(data, model)

    # Calculate residuals and standard deviation in log scale
    residuals = data['LogPrice'] - data['PredictedLogPrice']
    std_dev_log = np.std(residuals)

    # Display summary information
    display_summary(data, std_dev_log, model)

    # Visualize the results with boundaries, regression line, and actual price
    visualize_with_boundaries(data, std_dev_log, symbol)

# Analyze multiple stocks
def main():
    stocks = ['GOOGL', 'BLK', 'NVDA','TSM']  # Google, BlackRock, Nifty
    for stock in stocks:
        print(f"\nAnalyzing {stock}...\n")
        analyze_stock(stock)

main()
