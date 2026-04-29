import os
import duckdb

os.makedirs("data/processed", exist_ok=True)

# Connect to DuckDB database
con = duckdb.connect("data/processed/analytics.db")

# Step 1: Join all 3 raw sources on DATE (INNER JOIN) 
con.execute("""
    CREATE OR REPLACE TABLE master_raw AS
    SELECT
        f.DATE,

        -- FRED: macro inflation data
        f.CPI,
        f.inflation_yoy,
        f.inflation_regime,

        -- yfinance: monthly ETF returns (9 sectors)
        e.Energy,
        e.Technology,
        e.Financials,
        e.Utilities,
        e.Healthcare,
        e.Consumer_Discretionary,
        e.Consumer_Staples,
        e.Industrials,
        e.Materials,

        -- Google Trends: search interest (renamed to avoid column conflicts)
        t.inflation      AS trend_inflation,
        t.recession      AS trend_recession,
        t.interest_rates AS trend_interest_rates,
        t.oil_price      AS trend_oil_price,
        t.energy_stocks  AS trend_energy,
        t.tech_stocks    AS trend_tech,
        t.bank_stocks    AS trend_banks,
        t.utility_stocks AS trend_utilities,
        t.healthcare_stocks AS trend_healthcare,
        t.retail_stocks  AS trend_retail,
        t.dividend_stocks AS trend_dividend,
        t.manufacturing  AS trend_manufacturing,
        t.commodity_stocks AS trend_commodity

    FROM read_parquet('data/raw/cpi_inflation_regimes.parquet')        f
    INNER JOIN read_parquet('data/raw/spdr_sector_monthly_returns.parquet') e
        ON f.DATE = e.DATE
    INNER JOIN read_parquet('data/raw/google_trends.parquet')          t
        ON f.DATE = t.DATE
    ORDER BY f.DATE
""")

# Step 2: Feature engineering using SQL window functions
# (All features LAGGED by at least 1 month to avoid look-ahead bias: we only use information available before the month whose returns we want to predict)
con.execute("""
    CREATE OR REPLACE TABLE master AS
    SELECT
        *,

        -- Lagged inflation: 1 and 3 months back
        LAG(inflation_yoy, 1) OVER (ORDER BY DATE) AS inflation_lag1,
        LAG(inflation_yoy, 3) OVER (ORDER BY DATE) AS inflation_lag3,

        -- Rolling 3-month average inflation (trend smoothing)
        AVG(inflation_yoy) OVER (
            ORDER BY DATE ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS inflation_roll3,

        -- Lagged Google Trends: 1 month back (search interest → next month returns)
        LAG(trend_inflation, 1)      OVER (ORDER BY DATE) AS trend_inflation_lag1,
        LAG(trend_recession, 1)      OVER (ORDER BY DATE) AS trend_recession_lag1,
        LAG(trend_interest_rates, 1) OVER (ORDER BY DATE) AS trend_rates_lag1,
        LAG(trend_oil_price, 1)      OVER (ORDER BY DATE) AS trend_oil_lag1,
        LAG(trend_energy, 1)         OVER (ORDER BY DATE) AS trend_energy_lag1,
        LAG(trend_tech, 1)           OVER (ORDER BY DATE) AS trend_tech_lag1,
        LAG(trend_banks, 1)          OVER (ORDER BY DATE) AS trend_banks_lag1,
        LAG(trend_utilities, 1)      OVER (ORDER BY DATE) AS trend_utilities_lag1,
        LAG(trend_healthcare, 1)     OVER (ORDER BY DATE) AS trend_healthcare_lag1,
        LAG(trend_retail, 1)         OVER (ORDER BY DATE) AS trend_retail_lag1,
        LAG(trend_dividend, 1)       OVER (ORDER BY DATE) AS trend_dividend_lag1,
        LAG(trend_manufacturing, 1)  OVER (ORDER BY DATE) AS trend_manufacturing_lag1,
        LAG(trend_commodity, 1)      OVER (ORDER BY DATE) AS trend_commodity_lag1

    FROM master_raw
    ORDER BY DATE
""")


# Step 3: output and export
df = con.execute("SELECT * FROM master").df()
df.to_parquet("data/processed/master.parquet", index=False)

print(f"   master table: {len(df)} rows × {len(df.columns)} columns")
print(f"   Date range : {df['DATE'].min()} → {df['DATE'].max()}")
print(f"   Tables in analytics.db: {[r[0] for r in con.execute('SHOW TABLES').fetchall()]}")
print(df.head())

con.close()