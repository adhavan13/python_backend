import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Get today's date and calculate the start date (10 years ago)
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=10 * 365)).strftime('%Y-%m-%d')

# List of asset symbols (Can be passed dynamically in a real app)
assets = ["AAPL", "MSFT", "BTC-USD", "ETH-USD", "GC=F", "CL=F", "TLT", "VNQ"]  # Stocks, Crypto, Commodities, Bonds, Real Estate

# Loop through each asset and fetch data
for asset in assets:
    print(f"Fetching data for {asset} from {start_date} to {end_date}...")

    # Fetch historical data
    data = yf.download(asset, start=start_date, end=end_date)

    # Check if data is available
    if data.empty:
        print(f"❌ No data found for {asset}")
        continue

    # Calculate additional features
    data["30D_MA"] = data["Close"].rolling(window=30).mean()
    data["Volatility"] = data["Close"].pct_change().rolling(window=30).std()

    # Save to CSV with asset name
    file_name = f"{asset.replace('=', '')}_data.csv"
    data.to_csv(file_name)
    print(f"✅ Data saved to {file_name}!")

print("✅ All assets processed successfully!")