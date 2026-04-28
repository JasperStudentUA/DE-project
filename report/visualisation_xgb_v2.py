from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "report" / "figures" / "xgb_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PATH = DATA_DIR / "master.parquet"
RESULTS_PATH = DATA_DIR / "xgb_sector_results_V2.csv"
PREDICTIONS_PATH = DATA_DIR / "xgb_predictions_V2.parquet"

SECTOR_COLS = [
    "Energy",
    "Technology",
    "Financials",
    "Utilities",
    "Healthcare",
    "Consumer_Discretionary",
    "Consumer_Staples",
    "Industrials",
    "Materials",
]

BASE_FEATURES = [
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
    "trend_commodity_lag1",
    "month_of_year",
]

PLOT_TEMPLATE = "plotly_white"
COLOR_SEQUENCE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
]


def save_fig(fig, filename):
    html_path = OUTPUT_DIR / filename
    png_path = OUTPUT_DIR / filename.replace(".html", ".png")

    fig.write_html(html_path, include_plotlyjs=True)
    fig.write_image(png_path, width=1400, height=800, scale=2)

    print(f"Saved {html_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved {png_path.relative_to(PROJECT_ROOT)}")



def feature_group(feature):
    if feature.startswith("inflation"):
        return "Inflation"
    if feature.startswith("trend"):
        return "Google Trends"
    if feature.startswith("return"):
        return "Sector return history"
    if feature == "month_of_year":
        return "Seasonality"
    return "Other"


def readable_feature(feature):
    labels = {
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
    return labels.get(feature, feature)


def load_data():
    metrics = pd.read_csv(RESULTS_PATH)
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    master = pd.read_parquet(MASTER_PATH)

    predictions["DATE"] = pd.to_datetime(predictions["DATE"])
    master["DATE"] = pd.to_datetime(master["DATE"])

    predictions["error"] = predictions["predicted"] - predictions["actual"]
    predictions["abs_error"] = predictions["error"].abs()
    predictions["squared_error"] = predictions["error"] ** 2
    predictions["direction_correct"] = np.sign(predictions["actual"]) == np.sign(
        predictions["predicted"]
    )

    sector_diagnostics = (
        predictions.groupby("sector")
        .agg(
            actual_mean=("actual", "mean"),
            predicted_mean=("predicted", "mean"),
            mean_error=("error", "mean"),
            median_abs_error=("abs_error", "median"),
            directional_accuracy=("direction_correct", "mean"),
            correlation=("actual", lambda x: x.corr(predictions.loc[x.index, "predicted"])),
        )
        .reset_index()
    )

    metrics = metrics.merge(sector_diagnostics, on="sector", how="left")
    metrics["mae_improvement_vs_baseline"] = metrics["baseline_mae"] - metrics["mae"]
    metrics["rmse_improvement_vs_baseline"] = metrics["baseline_rmse"] - metrics["rmse"]
    metrics["mae_improvement_pct"] = (
        metrics["mae_improvement_vs_baseline"] / metrics["baseline_mae"] * 100
    )
    metrics["rmse_improvement_pct"] = (
        metrics["rmse_improvement_vs_baseline"] / metrics["baseline_rmse"] * 100
    )

    return master, metrics, predictions


def compute_feature_importance(master):
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

        sector_importance = pd.DataFrame(
            {
                "sector": sector,
                "feature": feature_cols,
                "importance": model.feature_importances_,
            }
        )
        all_importance.append(sector_importance)

    importance = pd.concat(all_importance, ignore_index=True)
    importance["feature_label"] = importance["feature"].map(readable_feature)
    importance["feature_group"] = importance["feature"].map(feature_group)
    importance.to_csv(DATA_DIR / "xgb_feature_importance_V2.csv", index=False)
    return importance


def plot_metric_comparison(metrics):
    metric_long = metrics.melt(
        id_vars="sector",
        value_vars=["mae", "baseline_mae", "rmse", "baseline_rmse"],
        var_name="metric",
        value_name="error",
    )
    metric_long["metric"] = metric_long["metric"].map(
        {
            "mae": "XGBoost MAE",
            "baseline_mae": "Baseline MAE",
            "rmse": "XGBoost RMSE",
            "baseline_rmse": "Baseline RMSE",
        }
    )

    fig = px.bar(
        metric_long,
        x="sector",
        y="error",
        color="metric",
        barmode="group",
        template=PLOT_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
        title="XGBoost Forecast Error Compared With a Simple Rolling-Return Baseline",
        labels={"sector": "Sector", "error": "Prediction error", "metric": "Metric"},
    )
    fig.update_layout(xaxis_tickangle=-35, legend_title_text="")
    save_fig(fig, "01_error_metrics_vs_baseline.html")


def plot_improvement(metrics):
    sorted_metrics = metrics.sort_values("mae_improvement_pct", ascending=True)
    fig = px.bar(
        sorted_metrics,
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
    save_fig(fig, "02_mae_improvement_vs_baseline.html")


def plot_actual_vs_predicted_lines(predictions):
    fig = px.line(
        predictions.melt(
            id_vars=["DATE", "sector"],
            value_vars=["actual", "predicted"],
            var_name="series",
            value_name="monthly_return",
        ),
        x="DATE",
        y="monthly_return",
        color="series",
        facet_col="sector",
        facet_col_wrap=3,
        template=PLOT_TEMPLATE,
        color_discrete_map={"actual": "#1f77b4", "predicted": "#ff7f0e"},
        title="Actual vs Predicted Next-Month Sector Returns",
        labels={
            "DATE": "Date",
            "monthly_return": "Monthly return",
            "series": "",
        },
    )
    fig.update_yaxes(matches=None)
    fig.update_xaxes(matches=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    save_fig(fig, "03_actual_vs_predicted_by_sector.html")


def plot_prediction_scatter(predictions):
    fig = px.scatter(
        predictions,
        x="actual",
        y="predicted",
        color="sector",
        template=PLOT_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
        title="Prediction Quality: Actual vs Predicted Returns",
        labels={"actual": "Actual next-month return", "predicted": "Predicted return"},
        hover_data=["DATE", "sector", "abs_error"],
    )
    min_value = min(predictions["actual"].min(), predictions["predicted"].min())
    max_value = max(predictions["actual"].max(), predictions["predicted"].max())
    fig.add_shape(
        type="line",
        x0=min_value,
        y0=min_value,
        x1=max_value,
        y1=max_value,
        line={"color": "black", "dash": "dash"},
    )
    save_fig(fig, "04_actual_vs_predicted_scatter.html")


def plot_error_distribution(predictions):
    fig = px.box(
        predictions,
        x="sector",
        y="error",
        color="sector",
        template=PLOT_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
        title="Forecast Error Distribution by Sector",
        labels={"sector": "Sector", "error": "Prediction error"},
        points="outliers",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(showlegend=False, xaxis_tickangle=-35)
    save_fig(fig, "05_error_distribution_by_sector.html")


def plot_directional_accuracy(metrics):
    fig = px.bar(
        metrics.sort_values("directional_accuracy"),
        x="directional_accuracy",
        y="sector",
        orientation="h",
        template=PLOT_TEMPLATE,
        color="directional_accuracy",
        color_continuous_scale="Blues",
        title="Directional Accuracy: Does the Model Predict the Correct Sign?",
        labels={
            "directional_accuracy": "Share of months with correct return direction",
            "sector": "Sector",
        },
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="black")
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(coloraxis_showscale=False)
    save_fig(fig, "06_directional_accuracy.html")


def plot_feature_importance(importance):
    top_features = (
        importance.groupby(["feature", "feature_label", "feature_group"], as_index=False)[
            "importance"
        ]
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
    save_fig(fig, "07_top_feature_importance_overall.html")

    heatmap_data = importance.pivot_table(
        index="feature_label",
        columns="sector",
        values="importance",
        aggfunc="mean",
        fill_value=0,
    )
    ordered_features = (
        importance.groupby("feature_label")["importance"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .index
    )
    heatmap_data = heatmap_data.loc[ordered_features]

    fig = px.imshow(
        heatmap_data,
        template=PLOT_TEMPLATE,
        color_continuous_scale="Viridis",
        aspect="auto",
        title="Which Predictors Matter Most for Each Sector?",
        labels={"x": "Sector", "y": "Feature", "color": "Importance"},
    )
    save_fig(fig, "08_feature_importance_heatmap.html")


def plot_group_importance(importance):
    grouped = (
        importance.groupby(["sector", "feature_group"], as_index=False)["importance"]
        .sum()
        .sort_values(["sector", "importance"], ascending=[True, False])
    )

    fig = px.bar(
        grouped,
        x="sector",
        y="importance",
        color="feature_group",
        template=PLOT_TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
        title="Do Inflation, Search Interest, or Return History Drive the Forecasts?",
        labels={
            "sector": "Sector",
            "importance": "Total feature importance",
            "feature_group": "Feature group",
        },
    )
    fig.update_layout(xaxis_tickangle=-35, legend_title_text="")
    save_fig(fig, "09_feature_group_importance.html")


def plot_research_question_summary(metrics):
    summary = pd.DataFrame(
        {
            "Metric": [
                "Sectors where XGBoost beats baseline MAE",
                "Average MAE improvement vs baseline",
                "Average directional accuracy",
                "Best sector by MAE",
                "Hardest sector by MAE",
            ],
            "Value": [
                f"{(metrics['mae'] < metrics['baseline_mae']).sum()} of {len(metrics)}",
                f"{metrics['mae_improvement_pct'].mean():.1f}%",
                f"{metrics['directional_accuracy'].mean():.1%}",
                metrics.sort_values("mae").iloc[0]["sector"],
                metrics.sort_values("mae", ascending=False).iloc[0]["sector"],
            ],
        }
    )

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Research-question diagnostic", "Result"],
                    fill_color="#1f77b4",
                    font=dict(color="white", size=13),
                    align="left",
                ),
                cells=dict(
                    values=[summary["Metric"], summary["Value"]],
                    fill_color=[["#f6f8fa"] * len(summary)],
                    align="left",
                    height=30,
                ),
            )
        ]
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title="Can Inflation and Public Search Interest Predict Sector ETF Returns?",
    )
    save_fig(fig, "10_research_question_summary.html")


def plot_rolling_error(predictions):
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
            "rolling_mae_6m": "Rolling MAE",
            "sector": "Sector",
        },
    )
    save_fig(fig, "11_rolling_forecast_error.html")


def main():
    master, metrics, predictions = load_data()
    importance = compute_feature_importance(master)

    metrics.to_csv(DATA_DIR / "xgb_sector_diagnostics_V2.csv", index=False)

    plot_metric_comparison(metrics)
    plot_improvement(metrics)
    plot_actual_vs_predicted_lines(predictions)
    plot_prediction_scatter(predictions)
    plot_error_distribution(predictions)
    plot_directional_accuracy(metrics)
    plot_feature_importance(importance)
    plot_group_importance(importance)
    plot_research_question_summary(metrics)
    plot_rolling_error(predictions)

    print("\nMain takeaway checks:")
    print(
        f"- XGBoost beats the rolling-return baseline in "
        f"{(metrics['mae'] < metrics['baseline_mae']).sum()} of {len(metrics)} sectors."
    )
    print(
        f"- Average MAE improvement vs baseline: "
        f"{metrics['mae_improvement_pct'].mean():.2f}%"
    )
    print(
        f"- Average directional accuracy: "
        f"{metrics['directional_accuracy'].mean():.2%}"
    )
    print("\nFeature importance note:")
    print(
        "XGBoost importance shows which variables helped reduce prediction error inside "
        "the trees. It supports interpretation, but it should not be read as causal proof."
    )


if __name__ == "__main__":
    main()
