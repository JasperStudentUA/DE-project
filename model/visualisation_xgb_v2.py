from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from xgboost import XGBRegressor


# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "report" / "figures" / "xgb_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PATH = DATA_DIR / "master.parquet"
RESULTS_PATH = DATA_DIR / "xgb_sector_results_V2.csv"
PREDICTIONS_PATH = DATA_DIR / "xgb_predictions_V2.parquet"


# ── Constants ──────────────────────────────────────────────────────────────────
SECTOR_COLS = [
    "Energy", "Technology", "Financials", "Utilities", "Healthcare",
    "Consumer_Discretionary", "Consumer_Staples", "Industrials", "Materials",
]

BASE_FEATURES = [
    "inflation_lag1", "inflation_lag3", "inflation_roll3",
    "trend_inflation_lag1", "trend_recession_lag1", "trend_rates_lag1",
    "trend_oil_lag1", "trend_energy_lag1", "trend_tech_lag1",
    "trend_banks_lag1", "trend_utilities_lag1", "trend_healthcare_lag1",
    "trend_retail_lag1", "trend_dividend_lag1", "trend_manufacturing_lag1",
    "trend_commodity_lag1", "month_of_year",
]

PLOT_TEMPLATE = "plotly_white"
COLOR_SEQUENCE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
]


# ── Helpers ────────────────────────────────────────────────────────────────────
def save_fig(fig, filename: str) -> None:
    html_path = OUTPUT_DIR / filename
    png_path = OUTPUT_DIR / filename.replace(".html", ".png")
    fig.write_html(html_path, include_plotlyjs=True)
    fig.write_image(png_path, width=1400, height=800, scale=2)
    print(f"Saved {html_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved {png_path.relative_to(PROJECT_ROOT)}")


def feature_group(feature: str) -> str:
    if feature.startswith("inflation"):
        return "Inflation"
    if feature.startswith("trend"):
        return "Google Trends"
    if feature.startswith("return"):
        return "Sector return history"
    if feature == "month_of_year":
        return "Seasonality"
    return "Other"


FEATURE_LABELS = {
    "inflation_lag1": "Inflation YoY, 1-month lag",
    "inflation_lag3": "Inflation YoY, 3-month lag",
    "inflation_roll3": "Inflation YoY, 3-month average",
    "trend_inflation_lag1": "Search: inflation",
    "trend_recession_lag1": "Search: recession",
    "trend_rates_lag1": "Search: interest rates",
    "trend_oil_lag1": "Search: oil price",
    "trend_energy_lag1": "Search: energy stocks",
    "trend_tech_lag1": "Search: tech stocks",
    "trend_banks_lag1": "Search: bank stocks",
    "trend_utilities_lag1": "Search: utility stocks",
    "trend_healthcare_lag1": "Search: healthcare stocks",
    "trend_retail_lag1": "Search: retail stocks",
    "trend_dividend_lag1": "Search: dividend stocks",
    "trend_manufacturing_lag1": "Search: manufacturing",
    "trend_commodity_lag1": "Search: commodity stocks",
    "month_of_year": "Month of year",
    "return_lag1": "Sector return, 1-month lag",
    "return_roll3": "Sector return, 3-month average",
}


def readable_feature(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data():
    metrics = pd.read_csv(RESULTS_PATH)
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    master = pd.read_parquet(MASTER_PATH)

    predictions["DATE"] = pd.to_datetime(predictions["DATE"])
    master["DATE"] = pd.to_datetime(master["DATE"])

    predictions["error"] = predictions["predicted"] - predictions["actual"]
    predictions["abs_error"] = predictions["error"].abs()
    predictions["direction_correct"] = (
        np.sign(predictions["actual"]) == np.sign(predictions["predicted"])
    )

    sector_diagnostics = (
        predictions.groupby("sector")
        .agg(
            actual_mean=("actual", "mean"),
            predicted_mean=("predicted", "mean"),
            mean_error=("error", "mean"),
            median_abs_error=("abs_error", "median"),
            directional_accuracy=("direction_correct", "mean"),
            correlation=(
                "actual",
                lambda x: x.corr(predictions.loc[x.index, "predicted"]),
            ),
        )
        .reset_index()
    )

    metrics = metrics.merge(sector_diagnostics, on="sector", how="left")
    metrics["mae_improvement_vs_baseline"] = metrics["baseline_mae"] - metrics["mae"]
    metrics["mae_improvement_pct"] = (
        metrics["mae_improvement_vs_baseline"] / metrics["baseline_mae"] * 100
    )

    return master, metrics, predictions


# ── Feature importance ─────────────────────────────────────────────────────────
def compute_feature_importance(master: pd.DataFrame) -> pd.DataFrame:
    master = master.sort_values("DATE").reset_index(drop=True).copy()
    master["month_of_year"] = master["DATE"].dt.month

    all_importance = []
    for sector in SECTOR_COLS:
        model_df = master[["DATE", sector] + BASE_FEATURES].copy()
        model_df["target_next_month_return"] = model_df[sector].shift(-1)
        model_df["return_lag1"] = model_df[sector].shift(1)
        model_df["return_roll3"] = model_df[sector].rolling(3).mean().shift(1)

        feature_cols = BASE_FEATURES + ["return_lag1", "return_roll3"]
        model_df = model_df.dropna().reset_index(drop=True)

        X = model_df[feature_cols]
        y = model_df["target_next_month_return"]
        train_size = int(len(model_df) * 0.8)

        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(X.iloc[:train_size], y.iloc[:train_size])

        all_importance.append(
            pd.DataFrame({
                "sector": sector,
                "feature": feature_cols,
                "importance": model.feature_importances_,
            })
        )

    importance = pd.concat(all_importance, ignore_index=True)
    importance["feature_label"] = importance["feature"].map(readable_feature)
    importance["feature_group"] = importance["feature"].map(feature_group)
    importance.to_csv(DATA_DIR / "xgb_feature_importance_V2.csv", index=False)
    return importance


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_mae_improvement(metrics: pd.DataFrame) -> None:
    """01 — MAE improvement vs baseline (horizontal bar)."""
    fig = px.bar(
        metrics.sort_values("mae_improvement_pct", ascending=True),
        x="mae_improvement_pct",
        y="sector",
        orientation="h",
        template=PLOT_TEMPLATE,
        color="mae_improvement_pct",
        color_continuous_scale="RdYlGn",
        title="Where XGBoost Improves on the Baseline",
        labels={
            "sector": "Sector",
            "mae_improvement_pct": "MAE improvement vs baseline (%)",
        },
    )
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.update_layout(coloraxis_showscale=False)
    save_fig(fig, "01_mae_improvement_vs_baseline.html")


def plot_actual_vs_predicted_lines(predictions: pd.DataFrame) -> None:
    """02 — Actual vs predicted return time series per sector."""
    long = predictions.melt(
        id_vars=["DATE", "sector"],
        value_vars=["actual", "predicted"],
        var_name="series",
        value_name="monthly_return",
    )
    fig = px.line(
        long,
        x="DATE",
        y="monthly_return",
        color="series",
        facet_col="sector",
        facet_col_wrap=3,
        template=PLOT_TEMPLATE,
        color_discrete_map={"actual": "#1f77b4", "predicted": "#ff7f0e"},
        title="Actual vs Predicted Next-Month Sector Returns",
        labels={"DATE": "Date", "monthly_return": "Monthly return", "series": ""},
    )
    fig.update_yaxes(matches=None)
    fig.update_xaxes(matches=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    save_fig(fig, "02_actual_vs_predicted_by_sector.html")


def plot_top_feature_importance(importance: pd.DataFrame) -> None:
    """03 — Top-12 features averaged across all sectors (horizontal bar)."""
    top_features = (
        importance.groupby(
            ["feature", "feature_label", "feature_group"], as_index=False
        )["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .head(12)
    )
    fig = px.bar(
        top_features.sort_values("importance"),
        x="importance",
        y="feature_label",
        color="feature_group",
        orientation="h",
        template=PLOT_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
        title="Most Important Predictors Across All Sector Models",
        labels={
            "importance": "Average XGBoost feature importance",
            "feature_label": "Feature",
            "feature_group": "Feature type",
        },
    )
    fig.update_layout(legend_title_text="")
    save_fig(fig, "03_top_feature_importance.html")


def plot_feature_importance_heatmap(importance: pd.DataFrame) -> None:
    """04 — Feature importance heatmap (feature × sector)."""
    ordered_features = (
        importance.groupby("feature_label")["importance"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .index
    )
    heatmap_data = (
        importance.pivot_table(
            index="feature_label",
            columns="sector",
            values="importance",
            aggfunc="mean",
            fill_value=0,
        )
        .loc[ordered_features]
    )
    fig = px.imshow(
        heatmap_data,
        template=PLOT_TEMPLATE,
        color_continuous_scale="Viridis",
        aspect="auto",
        title="Which Predictors Matter Most for Each Sector?",
        labels={"x": "Sector", "y": "Feature", "color": "Importance"},
    )
    save_fig(fig, "04_feature_importance_heatmap.html")


def plot_rolling_forecast_error(predictions: pd.DataFrame) -> None:
    """05 — Six-month rolling MAE per sector."""
    rolling = predictions.sort_values(["sector", "DATE"]).copy()
    rolling["rolling_mae_6m"] = rolling.groupby("sector")["abs_error"].transform(
        lambda x: x.rolling(6, min_periods=3).mean()
    )
    fig = px.line(
        rolling,
        x="DATE",
        y="rolling_mae_6m",
        color="sector",
        template=PLOT_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
        title="Six-Month Rolling Forecast Error",
        labels={
            "DATE": "Date",
            "rolling_mae_6m": "Rolling MAE (6-month)",
            "sector": "Sector",
        },
    )
    save_fig(fig, "05_rolling_forecast_error.html")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    master, metrics, predictions = load_data()
    importance = compute_feature_importance(master)

    metrics.to_csv(DATA_DIR / "xgb_sector_diagnostics_V2.csv", index=False)

    plot_mae_improvement(metrics)
    plot_actual_vs_predicted_lines(predictions)
    plot_top_feature_importance(importance)
    plot_feature_importance_heatmap(importance)
    plot_rolling_forecast_error(predictions)

    print("\nKey diagnostics:")
    print(
        f"  XGBoost beats baseline MAE in "
        f"{(metrics['mae'] < metrics['baseline_mae']).sum()} of {len(metrics)} sectors."
    )
    print(f"  Average MAE improvement vs baseline: {metrics['mae_improvement_pct'].mean():.2f}%")
    print(f"  Average directional accuracy:        {metrics['directional_accuracy'].mean():.2%}")


if __name__ == "__main__":
    main()