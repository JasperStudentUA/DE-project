import os
import yfinance as yf
import pandas as pd

sector_etfs = {
    "Energy":                 "XLE",   
    "Technology":             "XLK",   
    "Financials":             "XLF",   
    "Utilities":              "XLU",   
    "Healthcare":             "XLV",   
    "Consumer_Discretionary": "XLY",   
    "Consumer_Staples":       "XLP",   
    "Industrials":            "XLI",   
    "Materials":              "XLB",   
}

tickers = list(sector_etfs.values())

# Download monthly adjusted closing prices from 2004
# auto_adjust=True corrects for dividends and stock splits
prices = yf.download(
    tickers=tickers,
    start="2004-01-01",
    interval="1mo",
    auto_adjust=True,
    progress=False
)

monthly_close = prices["Close"].copy()

# Rename ticker symbols to readable sector names
ticker_to_sector = {ticker: sector for sector, ticker in sector_etfs.items()}
monthly_close = monthly_close.rename(columns=ticker_to_sector)

# Compute monthly % returns: (P_t - P_t-1) / P_t-1 * 100
monthly_returns = monthly_close.pct_change() * 100

# Reset index and standardise DATE column match FRED output
monthly_returns = monthly_returns.reset_index()
monthly_returns = monthly_returns.rename(columns={"Date": "DATE"})
monthly_returns["DATE"] = pd.to_datetime(monthly_returns["DATE"]).dt.strftime("%Y-%m-%d")

# Drop first row: NaN 
monthly_returns = monthly_returns.dropna(how="all", subset=list(sector_etfs.keys()))

# Save to data/raw/ 
os.makedirs("data/raw", exist_ok=True)
monthly_returns.to_parquet("data/raw/spdr_sector_monthly_returns.parquet", index=False)

print(f"   Saved {len(monthly_returns)} rows → data/raw/spdr_sector_monthly_returns.parquet")
print(f"   Sectors : {list(sector_etfs.keys())}")
print(f"   Date range: {monthly_returns['DATE'].min()} → {monthly_returns['DATE'].max()}")
print(monthly_returns.head())