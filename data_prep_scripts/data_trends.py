import os
import time
import pandas as pd
from pytrends.request import TrendReq


pytrends = TrendReq(
    hl="en-US",
    tz=0,
)

keywords = {
    # General macro sentiment
    "inflation":          "inflation",
    "recession":          "recession",
    "interest_rates":     "interest rates",
    "oil_price":          "oil price",

    # Sector-specific search interest
    "energy_stocks":      "energy stocks",
    "tech_stocks":        "tech stocks",
    "bank_stocks":        "bank stocks",
    "utility_stocks":     "utility stocks",
    "healthcare_stocks":  "healthcare stocks",
    "retail_stocks":      "retail stocks",        
    "dividend_stocks":    "dividend stocks",       
    "manufacturing":      "manufacturing stocks",  
    "commodity_stocks":   "commodity stocks",      
}



all_series = []

for col_name, term in keywords.items():
    print(f"Fetching: '{term}'...")
    try:
        pytrends.build_payload(
            kw_list=[term],
            timeframe="2004-01-01 2026-04-30",  # full range → forces monthly
            geo="US"
        )
        df_term = pytrends.interest_over_time()

        if not df_term.empty:
            df_term = df_term[[term]].rename(columns={term: col_name})
            all_series.append(df_term)
            print(f"  ✅ {len(df_term)} rows")
        
        time.sleep(5)

    except Exception as e:
        print(f"  ⚠️  Failed '{term}': {e}")
        time.sleep(30)

# Merge all keywords on date index
df = all_series[0]
for s in all_series[1:]:
    df = df.join(s, how="outer")

df = df.reset_index().rename(columns={"date": "DATE"})
df["DATE"] = pd.to_datetime(df["DATE"]).dt.strftime("%Y-%m-%d")
df = df.sort_values("DATE").reset_index(drop=True)

# Drop duplicate dates that can appear at chunk boundaries
df = df.drop_duplicates(subset="DATE").sort_values("DATE").reset_index(drop=True)

# Save to data/raw/
os.makedirs("data/raw", exist_ok=True)
df.to_parquet("data/raw/google_trends.parquet", index=False)

print(f"\n✅ Saved {len(df)} rows → data/raw/google_trends.parquet")
print(f"   Keywords : {list(keywords.keys())}")
print(f"   Date range: {df['DATE'].min()} → {df['DATE'].max()}")
print(df.head())