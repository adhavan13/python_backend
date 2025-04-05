import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset (Change the filename as needed)
asset = "AAPL"  # Change to any asset symbol you want to train on
file_name = f"{asset}_data.csv"
df = pd.read_csv(file_name, index_col=0, parse_dates=True)

# Convert 'Close' column to numeric (force errors to NaN)
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

# Drop rows where Close is NaN (missing or invalid data)
df.dropna(subset=["Close"], inplace=True)

# Feature Engineering: Use past data to predict future prices
df["Price Change"] = df["Close"].pct_change(fill_method=None)  # Fix the FutureWarning
df["Future Price"] = df["Close"].shift(-1)  # Shift to create a prediction target

# Drop NaN values
df.dropna(inplace=True)

# Features (X) and Target (y)
features = ["Close", "30D_MA", "Volatility", "Price Change"]
df[features] = df[features].apply(pd.to_numeric, errors="coerce")  # Ensure all features are numeric
df.dropna(inplace=True)

X = df[features]
y = df["Future Price"]

# Split data into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.4f}")

# Save the trained model
joblib.dump(model, f"{asset}_model.pkl")
print(f"✅ Model saved as {asset}_model.pkl")