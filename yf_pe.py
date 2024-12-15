import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import requests
from datetime import datetime

def fetch_stock_data_with_pe(symbol, api_key, data_dir='stock_data'):
    """
    Fetches stock price and P/E ratio data for the given symbol.
    Caches the data locally. If cached data is up-to-date, loads it from file.
    
    Parameters:
    - symbol: str, stock symbol
    - api_key: str, API key for Alpha Vantage
    - data_dir: str, directory to save data files
    
    Returns:
    - stock_data: pandas DataFrame with Date, Price, EPS, PE
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    data_file = os.path.join(data_dir, f"{symbol}_data.csv")
    
    # Check if data file exists
    if os.path.exists(data_file):
        # Load existing data
        stock_data = pd.read_csv(data_file, parse_dates=['Date'])
        # Check if the data is up-to-date
        latest_date = stock_data['Date'].max()
        current_month = datetime.now().replace(day=1)
        if latest_date >= current_month:
            print(f"Loading cached data for {symbol}")
            return stock_data
        else:
            print(f"Data for {symbol} is outdated. Fetching new data.")
    else:
        print(f"No cached data found for {symbol}. Fetching data.")
    
    # Fetch stock price data
    price_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={symbol}&apikey={api_key}'
    price_response = requests.get(price_url)
    if price_response.status_code != 200:
        raise ValueError("Error fetching stock price data.")
    price_data = price_response.json()
    time_series = price_data.get('Monthly Time Series', {})
    if not time_series:
        raise ValueError("Price data is empty or malformed.")
    
    dates = []
    prices = []
    for date, stats in time_series.items():
        dates.append(datetime.strptime(date, "%Y-%m-%d"))
        prices.append(float(stats['4. close']))
    
    # Fetch P/E ratio data
    pe_url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={api_key}'
    pe_response = requests.get(pe_url)
    if pe_response.status_code != 200:
        raise ValueError("Error fetching earnings data.")
    pe_data = pe_response.json()
    annual_reports = pe_data.get('annualEarnings', [])
    if not annual_reports:
        raise ValueError("P/E data is empty or malformed.")
    
    pe_dates = []
    eps = []
    for report in annual_reports:
        pe_dates.append(datetime.strptime(report['fiscalDateEnding'], "%Y-%m-%d"))
        eps.append(float(report['reportedEPS']))
    
    # Merge P/E ratio into stock data
    price_df = pd.DataFrame({'Date': dates, 'Price': prices}).sort_values('Date')
    pe_df = pd.DataFrame({'Date': pe_dates, 'EPS': eps}).sort_values('Date')
    
    # Merge the EPS into the stock data
    stock_data_new = pd.merge_asof(price_df, pe_df, on='Date', direction='backward')
    stock_data_new['PE'] = stock_data_new['Price'] / stock_data_new['EPS']
    
    if os.path.exists(data_file):
        # Merge new data with existing data
        stock_data = pd.concat([stock_data, stock_data_new]).drop_duplicates(subset='Date').sort_values('Date')
    else:
        stock_data = stock_data_new
    
    # Save to file
    stock_data.to_csv(data_file, index=False)
    print(f"Data for {symbol} saved to {data_file}")
    return stock_data

def process_stock_data(stock_data):
    """
    Processes stock data by performing linear regression on log price vs time and log PE.
    
    Parameters:
    - stock_data: pandas DataFrame with Date, Price, EPS, PE
    
    Returns:
    - stock_data: pandas DataFrame with additional columns
    - model: trained LinearRegression model
    """
    stock_data = stock_data.copy()
    stock_data['LogPrice'] = np.log(stock_data['Price'])
    stock_data['TimeIndex'] = np.arange(len(stock_data))  # Sequential time as independent variable
    stock_data['LogPE'] = np.log(stock_data['PE'])  # Logarithmic scale of PE
    
    # Prepare features: Time and LogPE (handle NaN)
    X = stock_data[['TimeIndex', 'LogPE']].fillna(0).values
    y = stock_data['LogPrice'].values
    
    # Linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict log prices and calculate residuals
    stock_data['PredictedLogPrice'] = model.predict(X)
    stock_data['IntrinsicValue'] = np.exp(stock_data['PredictedLogPrice'])  # Convert back to actual prices
    return stock_data, model

def compute_irr_from_regression(model, freq='monthly'):
    """
    Computes Internal Rate of Return (IRR) from the regression model's slope.
    
    Parameters:
    - model: trained LinearRegression model
    - freq: str, frequency of TimeIndex ('monthly', 'daily', etc.)
    
    Returns:
    - irr: float, Internal Rate of Return
    - slope_time: float, slope coefficient for time variable
    - intercept: float, intercept of the model
    """
    # Extract the coefficient for TimeIndex
    slope_time = model.coef_[0]  # Coefficient for the time variable
    
    # Calculate IRR as annualized growth rate
    # Assuming TimeIndex is monthly
    if freq == 'monthly':
        periods_per_year = 12
    elif freq == 'daily':
        periods_per_year = 252
    else:
        periods_per_year = 1  # default
    
    irr = (np.exp(slope_time * periods_per_year)) - 1
    print(f"Internal Rate of Return (IRR): {irr * 100:.2f}%")
    return irr, slope_time, model.intercept_

def visualize_stock_data(stock_data, model, symbol):
    """
    Visualizes the actual price and intrinsic value on a logarithmic scale.
    
    Parameters:
    - stock_data: pandas DataFrame with Date, Price, IntrinsicValue
    - model: trained LinearRegression model
    - symbol: str, stock symbol
    """
    slope = model.coef_[0]
    intercept = model.intercept_
    irr = (np.exp(slope * 12)) - 1  # Assuming monthly data
    
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Date'], stock_data['Price'], label='Actual Price', linestyle='-', marker='o')
    plt.plot(stock_data['Date'], stock_data['IntrinsicValue'], label='Regression Line (Intrinsic Value)', linestyle='--', color='orange')
    plt.yscale('log')  # Logarithmic scale
    plt.title(f"{symbol} Stock Price and Regression Line (Log Scale)")
    plt.xlabel("Date")
    plt.ylabel("Price (Log Scale)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Add regression equation as text on the plot
    regression_eq = f"y = {slope:.4f}x + {intercept:.4f}"
    plt.text(stock_data['Date'].iloc[len(stock_data) // 2], 
             max(stock_data['Price']) / 2, 
             f"Regression Line: {regression_eq}\nIRR: {irr * 100:.2f}%", 
             fontsize=10, color="orange")
    
    plt.tight_layout()
    plt.show()

def get_stock_analysis(symbol, api_key, data_dir='stock_data', visualize=True):
    """
    Fetches, processes, and analyzes stock data for the given symbol.
    
    Parameters:
    - symbol: str, stock symbol
    - api_key: str, API key for Alpha Vantage
    - data_dir: str, directory to save data files
    - visualize: bool, whether to display the plot
    
    Returns:
    - stock_data: pandas DataFrame with processed data
    - model: trained LinearRegression model
    - irr: float, Internal Rate of Return
    """
    stock_data = fetch_stock_data_with_pe(symbol, api_key, data_dir)
    stock_data, model = process_stock_data(stock_data)
    irr, slope, intercept = compute_irr_from_regression(model)
    
    if visualize:
        visualize_stock_data(stock_data, model, symbol)
    
    return stock_data, model, irr

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Data Analysis with P/E Ratio and IRR computation.')
    parser.add_argument('symbol', type=str, help='Stock symbol, e.g., NVDA')
    parser.add_argument('--api_key', type=str, required=True, help='Alpha Vantage API key')
    parser.add_argument('--data_dir', type=str, default='stock_data', help='Directory to save data files')
    parser.add_argument('--no_visual', action='store_true', help='Do not display visualization')
    
    args = parser.parse_args()
    
    stock_data, model, irr = get_stock_analysis(
        symbol=args.symbol,
        api_key=args.api_key,
        data_dir=args.data_dir,
        visualize=not args.no_visual
    )
