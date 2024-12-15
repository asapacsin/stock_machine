import yfinance as yf

# Define the ticker symbol for NVIDIA
ticker = 'NVDA'

# Fetch the NVIDIA stock data
nvda_data = yf.Ticker(ticker)

# Get historical stock prices
hist = nvda_data.history(period="max")  # You can specify a period like "1y", "5y", "max"

# Print historical stock prices
print("Historical Stock Prices for NVIDIA:")
print(hist)

# Get the PE ratio
pe_ratio = nvda_data.info['trailingPE']

# Print the PE ratio
print(f"\nNVIDIA's current PE Ratio: {pe_ratio}")

# If you want to see the historical PE ratios, you might need to calculate them manually
# as yfinance does not provide historical PE ratios directly. You would need the historical
# earnings data and the historical stock prices to calculate this.
