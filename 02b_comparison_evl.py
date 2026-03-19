import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROCESSED_DATA_PATH = "data/processed/weekly_beer_data.csv"
PLOT_OUTPUT_PATH = "data/processed/model_comparison_mae_rmse.png"
RESULTS_OUTPUT_PATH = "data/processed/model_comparison_metrics.csv"

FOCUS_BEERS = [
    "Hoop Bleke Nelis",
    "Hoop Lager",
    "Hoop Kaper Tropical IPA",
]

WEEKLY_LAGS = [1, 2, 3, 4, 8, 12, 26, 52]
MONTHLY_LAGS = [1, 2, 3, 6, 12]
TEST_RATIO = 0.2

MODEL_CONFIGS = [
    {"name": "Weekly + Weather + Events", "freq": "W", "use_events": True, "lags": WEEKLY_LAGS},
    {"name": "Monthly + Weather + Events", "freq": "M", "use_events": True, "lags": MONTHLY_LAGS},
    {"name": "Weekly + Weather (No Events)", "freq": "W", "use_events": False, "lags": WEEKLY_LAGS},
    {"name": "Monthly + Weather (No Events)", "freq": "M", "use_events": False, "lags": MONTHLY_LAGS},
]


def _seasonal_terms(period_number: int, periods_per_year: float) -> tuple[float, float]:
    angle = 2 * np.pi * (period_number / periods_per_year)
    return float(np.sin(angle)), float(np.cos(angle))


def load_data() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["week"])
    df = df[df["beer"].isin(FOCUS_BEERS)].copy()
    return df


def aggregate_by_frequency(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Build one time series per beer, aggregating over all containers.
    - Weekly: one row per beer/week
    - Monthly: one row per beer/month
    """
    if freq == "W":
        out = (
            df.groupby(["beer", "week"]).agg(
                liters=("liters", "sum"),
                temp_mean=("temp_mean", "mean"),
                rain_mm=("rain_mm", "sum"),
                event_flag=("event_week", "max"),
            )
            .reset_index()
            .rename(columns={"week": "date"})
        )

    elif freq == "M":
        tmp = df.copy()
        tmp["date"] = tmp["week"].dt.to_period("M").dt.to_timestamp()
        out = (
            tmp.groupby(["beer", "date"]).agg(
                liters=("liters", "sum"),
                temp_mean=("temp_mean", "mean"),
                rain_mm=("rain_mm", "sum"),
                event_flag=("event_week", "max"),
            )
            .reset_index()
        )
    else:
        raise ValueError(f"Unsupported freq: {freq}")

    return out.sort_values(["beer", "date"]).reset_index(drop=True)


def add_features(series_df: pd.DataFrame, lags: list[int], freq: str, use_events: bool) -> tuple[pd.DataFrame, list[str]]:
    s = series_df.sort_values("date").copy()

    for lag in lags:
        s[f"lag_{lag}"] = s["liters"].shift(lag)

    s["roll_mean_3"] = s["liters"].shift(1).rolling(3, min_periods=1).mean()
    s["roll_mean_6"] = s["liters"].shift(1).rolling(6, min_periods=1).mean()

    s["month"] = s["date"].dt.month.astype(int)
    s["year"] = s["date"].dt.year.astype(int)

    if freq == "W":
        period_id = s["date"].dt.isocalendar().week.astype(int)
        periods_per_year = 52.18
    else:
        period_id = s["date"].dt.month.astype(int)
        periods_per_year = 12.0

    s["season_sin"] = period_id.apply(lambda p: _seasonal_terms(int(p), periods_per_year)[0])
    s["season_cos"] = period_id.apply(lambda p: _seasonal_terms(int(p), periods_per_year)[1])

    temp_thresh = float(np.nanquantile(s["temp_mean"], 0.75)) if s["temp_mean"].notna().any() else np.nan
    rain_thresh = float(np.nanquantile(s["rain_mm"], 0.75)) if s["rain_mm"].notna().any() else np.nan

    if np.isnan(temp_thresh):
        s["hot_flag"] = 0
    else:
        s["hot_flag"] = (s["temp_mean"] >= temp_thresh).astype(int)

    if np.isnan(rain_thresh):
        s["rainy_flag"] = 0
    else:
        s["rainy_flag"] = (s["rain_mm"] >= rain_thresh).astype(int)

    feature_cols = [f"lag_{lag}" for lag in lags] + [
        "roll_mean_3",
        "roll_mean_6",
        "season_sin",
        "season_cos",
        "month",
        "year",
        "temp_mean",
        "rain_mm",
        "hot_flag",
        "rainy_flag",
    ]

    if use_events:
        feature_cols.append("event_flag")

    return s, feature_cols


def evaluate_single_beer(series_df: pd.DataFrame, lags: list[int], freq: str, use_events: bool) -> dict | None:
    n_raw = len(series_df)
    usable_lags = [lag for lag in lags if lag < n_raw]
    if not usable_lags:
        return None

    feat_df, feature_cols = add_features(series_df, usable_lags, freq=freq, use_events=use_events)
    feat_df = feat_df.dropna(subset=feature_cols).copy()

    n = len(feat_df)
    if n < 10:
        return None

    test_size = max(3, int(round(n * TEST_RATIO)))
    if freq == "M":
        test_size = max(2, min(test_size, 6))
    else:
        test_size = max(6, min(test_size, 20))

    if n - test_size < 5:
        return None

    train = feat_df.iloc[:-test_size]
    test = feat_df.iloc[-test_size:]

    X_train = train[feature_cols]
    y_train = train["liters"].clip(lower=0)
    X_test = test[feature_cols]
    y_test = test["liters"].clip(lower=0).values

    loss = "poisson" if float(y_train.sum()) > 0 else "squared_error"

    model = HistGradientBoostingRegressor(
        loss=loss,
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        l2_regularization=0.1,
        random_state=42,
    )

    model.fit(X_train, y_train)
    preds = np.maximum(model.predict(X_test), 0.0)

    return {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "test_periods": int(len(test)),
    }


def evaluate_config(df: pd.DataFrame, config: dict) -> tuple[dict, list[dict]]:
    agg = aggregate_by_frequency(df, config["freq"])

    per_beer = []
    for beer in FOCUS_BEERS:
        series = agg[agg["beer"] == beer].copy()
        if series.empty:
            continue

        res = evaluate_single_beer(
            series_df=series,
            lags=config["lags"],
            freq=config["freq"],
            use_events=config["use_events"],
        )
        if res is None:
            continue

        res_row = {
            "model": config["name"],
            "beer": beer,
            **res,
        }
        per_beer.append(res_row)

    if not per_beer:
        return {
            "model": config["name"],
            "MAE": np.nan,
            "RMSE": np.nan,
            "beers_used": 0,
        }, []

    avg_mae = float(np.mean([r["MAE"] for r in per_beer]))
    avg_rmse = float(np.mean([r["RMSE"] for r in per_beer]))

    summary = {
        "model": config["name"],
        "MAE": avg_mae,
        "RMSE": avg_rmse,
        "beers_used": len(per_beer),
    }
    return summary, per_beer


def make_plot(summary_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_df = summary_df.copy()

    axes[0].bar(plot_df["model"], plot_df["MAE"], color="#1f77b4", alpha=0.85)
    axes[0].set_title("MAE by Model")
    axes[0].set_ylabel("MAE (liters)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(plot_df["model"], plot_df["RMSE"], color="#ff7f0e", alpha=0.85)
    axes[1].set_title("RMSE by Model")
    axes[1].set_ylabel("RMSE (liters)")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Forecast Accuracy Comparison (lower is better)")
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("Loading processed dataset...")
    df = load_data()

    print("\nMAE = Mean Absolute Error")
    print("- Average absolute prediction error in liters")
    print("- Easy to interpret; lower is better")
    print("\nRMSE = Root Mean Squared Error")
    print("- Like MAE but penalizes large misses more")
    print("- Lower is better\n")

    summary_rows = []
    detailed_rows = []

    for cfg in MODEL_CONFIGS:
        print(f"Evaluating: {cfg['name']}")
        summary, per_beer = evaluate_config(df, cfg)
        summary_rows.append(summary)
        detailed_rows.extend(per_beer)

    summary_df = pd.DataFrame(summary_rows)
    detailed_df = pd.DataFrame(detailed_rows)

    summary_df = summary_df.sort_values("RMSE", ascending=True).reset_index(drop=True)

    summary_df.to_csv(RESULTS_OUTPUT_PATH, index=False)
    make_plot(summary_df)

    print("\n=== Overall model comparison (average across beers) ===")
    print(summary_df.to_string(index=False))

    if not detailed_df.empty:
        print("\n=== Per beer details ===")
        print(detailed_df.sort_values(["model", "beer"]).to_string(index=False))

    print(f"\nSaved metrics to: {RESULTS_OUTPUT_PATH}")
    print(f"Saved chart   to: {PLOT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
