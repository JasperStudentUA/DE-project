import os
from datetime import date
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()
fred = Fred(api_key=os.environ["FRED_API_KEY"])

# Pull monthly CPI from FRED — start 2004 to align with Google Trends
cpi = fred.get_series(
    "CPIAUCSL",
    observation_start="2004-01-01",
    observation_end=date.today().strftime("%Y-%m-%d")
)

df = cpi.reset_index()
df.columns = ["DATE", "CPI"]
df["DATE"] = pd.to_datetime(df["DATE"]).dt.strftime("%Y-%m-%d")

# YoY inflation: (CPI_t - CPI_t-12) / CPI_t-12 * 100
df["inflation_yoy"] = df["CPI"].pct_change(periods=12) * 100

# Classify inflation regime
def classify_inflation(x):
    if pd.isna(x): return pd.NA
    elif x < 2:    return "low"
    elif x <= 4:   return "moderate"
    else:          return "high"

df["inflation_regime"] = df["inflation_yoy"].apply(classify_inflation)
df = df.dropna(subset=["inflation_yoy"])

os.makedirs("data/raw", exist_ok=True)
df.to_parquet("data/raw/cpi_inflation_regimes.parquet", index=False)
print(f"✅ Saved {len(df)} rows → data/raw/cpi_inflation_regimes.parquet")
print(df.head())