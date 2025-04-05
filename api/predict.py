import json
import joblib
import pandas as pd
import yfinance as yf

# Load trained model globally (loaded only once in serverless)
model = joblib.load("AAPL_model.pkl")

# Asset mapping to ticker symbols
ASSET_TICKERS = {
    "Stock": {"AAPL": "Apple", "MSFT": "Microsoft"},
    "Cryptocurrency": {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum"},
    "Commodity": {"GC=F": "Gold", "CL=F": "Oil"},
    "Bond": {"^TNX": "Treasury Bond", "LQD": "Corporate Bond"},
    "Real Estate": {"VNQ": "REIT"},
}

# Expected annual growth rate per asset type
EXPECTED_ANNUAL_GROWTH = {
    "Stock": 1.084,
    "Cryptocurrency": 1.15,
    "Commodity": 1.055,
    "Bond": 1.03,
    "Real Estate": 1.07
}

def handler(request):
    if request["method"] != "POST":
        return {
            "statusCode": 405,
            "body": json.dumps({"error": "Only POST method is allowed"})
        }

    try:
        data = json.loads(request["body"])
        asset_type = data.get("asset_type")
        asset_name = data.get("asset_name")
        interest_rate = data.get("interest_rate")
        inflation_rate = data.get("inflation_rate")
        forecast_years = data.get("forecast_years", 5)

        if asset_type not in ASSET_TICKERS or asset_name not in ASSET_TICKERS[asset_type]:
            return {"statusCode": 400, "body": json.dumps({"error": "Invalid asset type or name"})}

        ticker = asset_name
        df = yf.download(ticker, period="5y", interval="1mo")

        if df.empty:
            return {"statusCode": 404, "body": json.dumps({"error": "No data found for this asset"})}

        df["30D_MA"] = df["Close"].rolling(window=3).mean()
        df["Volatility"] = df["Close"].pct_change().rolling(window=3).std()
        df["Price Change"] = df["Close"].pct_change()
        latest_data = df.iloc[-1]

        input_data = pd.DataFrame([[latest_data["Close"],
                                    latest_data["30D_MA"],
                                    latest_data["Volatility"],
                                    latest_data["Price Change"]]],
                                  columns=["Close", "30D_MA", "Volatility", "Price Change"])

        base_prediction = model.predict(input_data)[0]

        annual_growth = EXPECTED_ANNUAL_GROWTH.get(asset_type, 1.05)
        growth_applied_prediction = base_prediction * (annual_growth ** forecast_years)
        adjusted_prediction = growth_applied_prediction * (1 + (interest_rate - inflation_rate) / 100)

        response = {
            "asset": ASSET_TICKERS[asset_type][asset_name],
            "current_price": round(latest_data["Close"], 2),
            f"predicted_price_{forecast_years}_years": round(adjusted_prediction, 2),
            "interest_rate": interest_rate,
            "inflation_rate": inflation_rate,
            "forecast_years": forecast_years,
            "growth_applied": f"{int((annual_growth ** forecast_years - 1) * 100)}%"
        }

        return {
            "statusCode": 200,
            "body": json.dumps(response),
            "headers": {"Content-Type": "application/json"}
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
