import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==============================
# CONFIG
# ==============================

PROCESSED_DATA_PATH = "data/processed/weekly_beer_data.csv"
TEST_RATIO = 0.20
MAX_FOLDS = 5
LAGS = [1, 2, 3, 4, 8, 12, 26, 52]

FOCUS_BEERS = [
    "Hoop Bleke Nelis",
    "Hoop Lager",
    "Hoop Kaper Tropical IPA",
]

BEER_COLORS = {
    "Hoop Bleke Nelis": "#1f77b4",
    "Hoop Lager": "#ff7f0e",
    "Hoop Kaper Tropical IPA": "#2ca02c",
}


def _seasonal_terms(week_of_year: int) -> tuple[float, float]:
    angle = 2 * np.pi * (week_of_year / 52.18)
    return float(np.sin(angle)), float(np.cos(angle))


def smape(y_true, y_pred):
    return np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)
    ) * 100


def create_features(df: pd.DataFrame, lags: list[int], use_events: bool = True) -> tuple[pd.DataFrame, list[str]]:
    df = df.sort_values("week").copy()
    for lag in lags:
        df[f"lag_{lag}"] = df["liters"].shift(lag)

    df["roll_mean_4"] = df["liters"].shift(1).rolling(4, min_periods=1).mean()
    df["roll_mean_12"] = df["liters"].shift(1).rolling(12, min_periods=1).mean()
    df["roll_sum_4"] = df["liters"].shift(1).rolling(4, min_periods=1).sum()
    df["sin_woy"] = df["week_of_year"].apply(lambda w: _seasonal_terms(int(w))[0]).astype(float)
    df["cos_woy"] = df["week_of_year"].apply(lambda w: _seasonal_terms(int(w))[1]).astype(float)

    feature_cols = [f"lag_{lag}" for lag in lags] + [
        "roll_mean_4",
        "roll_mean_12",
        "roll_sum_4",
        "sin_woy",
        "cos_woy",
        "month",
        "year",
        "hot_week",
        "rainy_week",
    ]
    if use_events and "event_week" in df.columns:
        feature_cols.append("event_week")
    if "temp_mean" in df.columns:
        feature_cols.append("temp_mean")
    if "rain_mm" in df.columns:
        feature_cols.append("rain_mm")

    return df, feature_cols


def _prepare_weekly_beer_df(beer_df: pd.DataFrame) -> pd.DataFrame:
    return (
        beer_df.groupby("week")
        .agg(
            liters=("liters", "sum"),
            temp_mean=("temp_mean", "mean"),
            rain_mm=("rain_mm", "sum"),
            event_week=("event_week", "max"),
            week_of_year=("week_of_year", "first"),
            month=("month", "first"),
            year=("year", "first"),
            hot_week=("hot_week", "max"),
            rainy_week=("rainy_week", "max"),
        )
        .reset_index()
        .sort_values("week")
    )


def evaluate_beer_rolling_cv(
    beer_df: pd.DataFrame,
    beer_name: str,
    test_ratio: float = 0.20,
    max_folds: int = 5,
    use_events: bool = True,
) -> tuple[dict | None, pd.DataFrame | None]:
    weekly = _prepare_weekly_beer_df(beer_df)
    usable_lags = [lag for lag in LAGS if lag < len(weekly)]
    if not usable_lags:
        return None, None

    weekly, feature_cols = create_features(weekly, usable_lags, use_events=use_events)
    weekly = weekly.dropna(subset=feature_cols).copy()
    weekly["liters"] = weekly["liters"].clip(lower=0)

    n = len(weekly)
    if n < 30:
        return None, None

    test_size = max(4, int(round(n * test_ratio)))
    test_size = min(test_size, n // 2)

    fold_bounds = []
    offset = 0
    while len(fold_bounds) < max_folds:
        test_end = n - offset
        test_start = test_end - test_size
        if test_start <= max(usable_lags):
            break
        fold_bounds.append((test_start, test_end))
        offset += test_size
    if not fold_bounds:
        return None, None

    model_mae_list, model_rmse_list, model_smape_list = [], [], []
    naive_mae_list, naive_rmse_list, naive_smape_list = [], [], []
    plot_df = None

    for i, (test_start, test_end) in enumerate(fold_bounds):
        train = weekly.iloc[:test_start]
        test = weekly.iloc[test_start:test_end].copy()

        X_train = train[feature_cols]
        y_train = train["liters"]
        X_test = test[feature_cols]
        y_test = test["liters"].values

        model = HistGradientBoostingRegressor(
            loss="poisson" if float(y_train.sum()) > 0 else "squared_error",
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            l2_regularization=0.1,
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds = np.maximum(model.predict(X_test), 0.0)
        naive_preds = test["lag_1"].values

        model_mae_list.append(mean_absolute_error(y_test, preds))
        model_rmse_list.append(np.sqrt(mean_squared_error(y_test, preds)))
        model_smape_list.append(smape(y_test, preds))
        naive_mae_list.append(mean_absolute_error(y_test, naive_preds))
        naive_rmse_list.append(np.sqrt(mean_squared_error(y_test, naive_preds)))
        naive_smape_list.append(smape(y_test, naive_preds))

        if i == 0:
            plot_df = test[["week"]].copy()
            plot_df["actual"] = y_test
            plot_df["predicted"] = preds
            plot_df["beer"] = beer_name

    metrics = {
        "Beer": beer_name,
        "Folds": len(model_mae_list),
        "Test Size (%)": int(round(test_ratio * 100)),
        "Model MAE": float(np.mean(model_mae_list)),
        "Naive MAE": float(np.mean(naive_mae_list)),
        "Model RMSE": float(np.mean(model_rmse_list)),
        "Naive RMSE": float(np.mean(naive_rmse_list)),
        "Model SMAPE (%)": float(np.mean(model_smape_list)),
        "Naive SMAPE (%)": float(np.mean(naive_smape_list)),
    }
    return metrics, plot_df


def evaluate_all_beers(
    processed_df: pd.DataFrame,
    focus_beers: list[str],
    test_ratio: float = 0.20,
    max_folds: int = 5,
    use_events: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    rows = []
    plots = {}
    for beer in focus_beers:
        beer_df = processed_df[processed_df["beer"] == beer].copy()
        metrics, plot_df = evaluate_beer_rolling_cv(
            beer_df=beer_df,
            beer_name=beer,
            test_ratio=test_ratio,
            max_folds=max_folds,
            use_events=use_events,
        )
        if metrics is not None:
            rows.append(metrics)
        if plot_df is not None:
            plots[beer] = plot_df

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        return metrics_df, plots, pd.DataFrame()

    overall_df = pd.DataFrame(
        [
            {
                "Beer": "Overall average",
                "Folds": int(round(metrics_df["Folds"].mean())),
                "Test Size (%)": int(round(metrics_df["Test Size (%)"].mean())),
                "Model MAE": float(metrics_df["Model MAE"].mean()),
                "Naive MAE": float(metrics_df["Naive MAE"].mean()),
                "Model RMSE": float(metrics_df["Model RMSE"].mean()),
                "Naive RMSE": float(metrics_df["Naive RMSE"].mean()),
                "Model SMAPE (%)": float(metrics_df["Model SMAPE (%)"].mean()),
                "Naive SMAPE (%)": float(metrics_df["Naive SMAPE (%)"].mean()),
            }
        ]
    )
    return metrics_df, plots, overall_df

# ==============================
# COMBINED PLOT
# ==============================

def plot_comparison(all_test_results):
    """One subplot per beer using shared 80/20 rolling-CV plot fold."""

    n_beers = len(all_test_results)
    fig, axes = plt.subplots(n_beers, 1, figsize=(14, 5 * n_beers), sharex=False)

    if n_beers == 1:
        axes = [axes]

    for ax, (beer_name, df) in zip(axes, all_test_results.items()):
        color = BEER_COLORS.get(beer_name, None)
        ax.plot(
            df["week"], df["actual"],
            color=color, linewidth=2,
            label="Actual",
        )
        ax.plot(
            df["week"], df["predicted"],
            color=color, linewidth=2, linestyle="--",
            label="Predicted",
        )
        ax.fill_between(
            df["week"], df["actual"], df["predicted"],
            color=color, alpha=0.1,
        )

        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
        ax.tick_params(axis="x", rotation=45)

        ax.set_ylabel("Total Liters")
        ax.set_title(f"{beer_name} — Actual vs Predicted")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Model Evaluation (test set, all containers summed)", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = "data/processed/model_evaluation_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.close()


# ==============================
# MAIN
# ==============================

def main():
    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["week"])
    df = df[df["beer"].isin(FOCUS_BEERS)].copy()
    metrics_df, all_test_results, overall_df = evaluate_all_beers(
        processed_df=df,
        focus_beers=FOCUS_BEERS,
        test_ratio=TEST_RATIO,
        max_folds=MAX_FOLDS,
    )

    if all_test_results:
        plot_comparison(all_test_results)

    if not metrics_df.empty:
        results_df = pd.concat([metrics_df, overall_df], ignore_index=True)
        print("\n" + "=" * 60)
        print("MODEL vs NAIVE BASELINE  (shared rolling 80/20 CV)")
        print("=" * 60 + "\n")
        print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
