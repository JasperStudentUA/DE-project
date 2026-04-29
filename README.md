# DE-project
This project predicts monthly sector ETF returns using inflation data (FRED), Google Trends search interest, and an XGBoost model.

**Research question:** Can inflation data and public search interest predict sector ETF returns?


## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up your API key
- Get a free FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html
- Copy `.env.example` to a new file called `.env` in the root folder
- Fill in your key: "FRED_API_KEY=your_fred_api_key_here"

- Yahoo Finance and Google Trends require no API key.



## Running the project

Run the scripts in this order:

```bash
# Step 1: Fetch data
python data_prep_scripts/data_fred.py
python data_prep_scripts/data_yfinance.py
python data_prep_scripts/data_trends.py

# Step 2: Build master dataset
python data_prep_scripts/build_dataset.py

# Step 3: Train model
python model/XGB_model_V2.py

# Step 4: Generate visualisations
python model/visualisation_xgb_v2.py
```

Output files (parquet, CSV, figures) will be saved to `data/` and `report/figures/`.



## Git workflow (for team members)

Get the latest changes:
```bash
git pull origin main
```

Push your changes:
```bash
git add .
git commit -m "your comment here"
git push
```

