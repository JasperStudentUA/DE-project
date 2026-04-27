import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

path = "data/processed/master.parquet"

df = pd.read_parquet(path)
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)

print(df.shape)
print(df.head())

sector_cols = [
    "Energy",
    "Technology",
    "Financials",
    "Utilities",
    "Healthcare",
    "Consumer_Discretionary",
    "Consumer_Staples",
    "Industrials",
    "Materials"
]

base_features = [
    "inflation_lag1",
    "inflation_lag3",
    "inflation_roll3",
    "trend_inflation_lag1",
    "trend_recession_lag1",
    "trend_rates_lag1",
    "trend_oil_lag1",
    "trend_energy_lag1",
    "trend_tech_lag1",
    "trend_banks_lag1",
    "trend_utilities_lag1",
    "trend_healthcare_lag1",
    "trend_retail_lag1",
    "trend_dividend_lag1",
    "trend_manufacturing_lag1",
    "trend_commodity_lag1"
]

df["month_of_year"] = df["DATE"].dt.month
base_features = base_features + ["month_of_year"]

results = []
predictions = {}

for sector in sector_cols:
    model_df = df[["DATE", sector] + base_features].copy()

    # Target: next month's sector return
    model_df["target_next_month_return"] = model_df[sector].shift(-1)

    # Sector-specific lag features
    model_df["return_lag1"] = model_df[sector].shift(1)
    model_df["return_roll3"] = model_df[sector].rolling(3).mean().shift(1)

    feature_cols = base_features + ["return_lag1", "return_roll3"]

    model_df = model_df.dropna().reset_index(drop=True)

    X = model_df[feature_cols]
    y = model_df["target_next_month_return"]

    train_size = int(len(model_df) * 0.8)

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]
    test_dates = model_df["DATE"].iloc[train_size:]

    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    results.append({
        "sector": sector,
        "mae": mae,
        "rmse": rmse,
        "train_rows": len(X_train),
        "test_rows": len(X_test)
    })

    predictions[sector] = pd.DataFrame({
        "DATE": test_dates,
        "actual": y_test.values,
        "predicted": preds
    })

results_df = pd.DataFrame(results).sort_values("mae")
results_df

print("\nModel performance by sector:")
print(results_df.to_string(index=False))

results_df.to_csv("data/processed/xgb_sector_results.csv", index=False)

all_predictions = pd.concat(
    [df.assign(sector=sector) for sector, df in predictions.items()],
    ignore_index=True
)

all_predictions.to_parquet("data/processed/xgb_predictions.parquet", index=False)

print("\nSaved:")
print("data/processed/xgb_sector_results.csv")
print("data/processed/xgb_predictions.parquet")
