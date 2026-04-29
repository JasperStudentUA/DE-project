import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

path = "data/processed/master.parquet"

# Spark reads & validates the data
spark = SparkSession.builder.appName("XGB_Sectors").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

spark_df = spark.read.parquet("data/processed/master.parquet")
print(f"Spark loaded: {spark_df.count()} rows, {len(spark_df.columns)} cols")

# Hand off to pandas for XGBoost
df = spark_df.toPandas()
spark.stop()

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

    # Target: next month sector return
    model_df["target_next_month_return"] = model_df[sector].shift(-1)

    # Sector-specific lag features
    model_df["return_lag1"] = model_df[sector].shift(1)
    model_df["return_roll3"] = model_df[sector].rolling(3).mean().shift(1)

    feature_cols = base_features + ["return_lag1", "return_roll3"]
    
    model_df = model_df.dropna().reset_index(drop=True)

    X = model_df[feature_cols]
    y = model_df["target_next_month_return"]

    train_size = int(len(model_df) * 0.8)

    # WALK-FORWARD CV (3 folds) 
    n = len(model_df)
    fold_size = n // 4 
    cv_maes = []
    for fold in range(1, 4):
        cv_train_end = fold * fold_size
        cv_test_end = cv_train_end + fold_size
        if cv_test_end > n:
            break
        Xcv_train = X.iloc[:cv_train_end]
        ycv_train = y.iloc[:cv_train_end]
        Xcv_test  = X.iloc[cv_train_end:cv_test_end]
        ycv_test  = y.iloc[cv_train_end:cv_test_end]
        cv_model = XGBRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        cv_model.fit(Xcv_train, ycv_train)
        cv_maes.append(mean_absolute_error(ycv_test, cv_model.predict(Xcv_test)))
    cv_mae = np.mean(cv_maes)

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

    # FEATURE IMPORTANCE (top 5) 
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).head(5)
    print(f"\n[{sector}] Top 5 features:")
    print(importance_df.to_string(index=False))

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    baseline_mae = mean_absolute_error(y_test, model_df["return_roll3"].iloc[train_size:])
    baseline_rmse = np.sqrt(mean_squared_error(y_test, model_df["return_roll3"].iloc[train_size:]))

    results.append({
        "sector": sector,
        "mae": mae,
        "rmse": rmse,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "cv_mae": cv_mae
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

results_df.to_csv("data/processed/xgb_sector_results_V2.csv", index=False)

all_predictions = pd.concat(
    [df.assign(sector=sector) for sector, df in predictions.items()],
    ignore_index=True
)

all_predictions.to_parquet("data/processed/xgb_predictions_V2.parquet", index=False)

print("\nSaved:")
print("data/processed/xgb_sector_results_V2.csv")
print("data/processed/xgb_predictions_V2.parquet")
