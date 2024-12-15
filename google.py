import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import yfinance as yf
from datetime import datetime

# Function to fetch Google stock price and P/E ratio data
def fetch_stock_data_with_pe(symbol='GOOGL'):
    # Fetch stock price data using yfinance
    stock = yf.Ticker(symbol)
    history = stock.history(period="max", interval="1mo")  # Monthly stock data
    history.reset_index(inplace=True)
    history.rename(columns={"Date": "Date", "Close": "Price"}, inplace=True)
    
    # Simulate or estimate P/E ratio using EPS (user-provided or hypothetical)
    eps = 100  # Replace with actual or estimated EPS for Google
    history['EPS'] = eps  # Assume constant EPS for simplification
    history['PE'] = history['Price'] / history['EPS']

    stock_data = history[['Date', 'Price', 'EPS', 'PE']]
    return stock_data

# Function to perform regression with adjusted P/E influence
def process_stock_data(stock_data):
    stock_data['LogPrice'] = np.log(stock_data['Price'])
    stock_data['TimeIndex'] = np.arange(len(stock_data))  # Sequential time as independent variable
    stock_data['LogPE'] = np.log(stock_data['PE'])  # Logarithmic scale of PE

    # Prepare features: Time and LogPE (scaled)
    X = stock_data[['TimeIndex', 'LogPE']].fillna(0).values
    y = stock_data['LogPrice'].values

    # Linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict log prices and calculate residuals
    stock_data['PredictedLogPrice'] = model.predict(X)
    stock_data['IntrinsicValue'] = np.exp(stock_data['PredictedLogPrice'])  # Convert back to actual prices
    return stock_data, model

def compute_irr_from_regression(model):
    # Extract the coefficient for TimeIndex
    slope_time = model.coef_[0]  # Coefficient for the time variable

    # Convert monthly slope to annual growth rate (IRR)
    irr = np.exp(12 * slope_time) - 1  # Annual compounding
    print(f"Internal Rate of Return (IRR): {irr * 100:.2f}%")
    return irr, slope_time, model.intercept_


# Main program function
def main():
    try:
        symbol = 'GOOGL'  # Stock symbol for Google (Alphabet)
        stock_data = fetch_stock_data_with_pe(symbol)
        stock_data, model = process_stock_data(stock_data)

        # Compute IRR from regression slope
        irr, slope, intercept = compute_irr_from_regression(model)

        # Visualization in logarithmic scale with intrinsic value line and regression equation
        plt.figure(figsize=(12, 6))
        plt.plot(stock_data['Date'], stock_data['Price'], label='Actual Price', linestyle='-', marker='o')
        plt.plot(stock_data['Date'], stock_data['IntrinsicValue'], label='Regression Line (Intrinsic Value)', linestyle='--', color='orange')
        plt.yscale('log')  # Logarithmic scale
        plt.title("Google (Alphabet) Stock Price and Regression Line (Log Scale)")
        plt.xlabel("Date")
        plt.ylabel("Price (Log Scale)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        # Add regression equation and IRR to the plot
        regression_eq = f"y = {slope:.4f}x + {intercept:.4f}"
        plt.text(stock_data['Date'].iloc[len(stock_data) // 2], 
                 max(stock_data['Price']) / 2, 
                 f"Regression Line: {regression_eq}\nIRR: {irr * 100:.2f}%", 
                 fontsize=10, color="orange")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


# Run the program
main()
