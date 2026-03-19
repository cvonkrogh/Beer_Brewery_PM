import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

LAGS = [1, 2, 3, 4, 8, 12, 26, 52]


def _seasonal_terms(week_of_year: int) -> tuple[float, float]:
    angle = 2 * np.pi * (week_of_year / 52.18)
    return float(np.sin(angle)), float(np.cos(angle))


def smape(y_true, y_pred):
    return np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)
    ) * 100


def create_features(df: pd.DataFrame, lags: list[int]) -> tuple[pd.DataFrame, list[str]]:
    df = df.sort_values("week").copy()

    for lag in lags:
        df[f"lag_{lag}"] = df["liters"].shift(lag)

    df["roll_mean_4"] = df["liters"].shift(1).rolling(4, min_periods=1).mean()
    df["roll_mean_12"] = df["liters"].shift(1).rolling(12, min_periods=1).mean()
    df["roll_sum_4"] = df["liters"].shift(1).rolling(4, min_periods=1).sum()

    sin_terms = df["week_of_year"].apply(lambda w: _seasonal_terms(int(w))[0])
    cos_terms = df["week_of_year"].apply(lambda w: _seasonal_terms(int(w))[1])
    df["sin_woy"] = sin_terms.astype(float)
    df["cos_woy"] = cos_terms.astype(float)

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
        "event_week",
    ]

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
) -> tuple[dict | None, pd.DataFrame | None]:
    """
    Rolling-origin time-series CV:
    - each fold uses first 80% as train and next 20% as test in rolling windows
    - returns averaged metrics + most recent fold predictions for plotting
    """
    weekly = _prepare_weekly_beer_df(beer_df)

    usable_lags = [lag for lag in LAGS if lag < len(weekly)]
    if not usable_lags:
        return None, None

    weekly, feature_cols = create_features(weekly, usable_lags)
    weekly = weekly.dropna(subset=feature_cols).copy()
    weekly["liters"] = weekly["liters"].clip(lower=0)

    n = len(weekly)
    if n < 30:
        return None, None

    test_size = max(4, int(round(n * test_ratio)))
    test_size = min(test_size, n // 2)

    # Backward rolling windows of fixed 20% test size
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

    model_mae_list = []
    model_rmse_list = []
    model_smape_list = []
    naive_mae_list = []
    naive_rmse_list = []
    naive_smape_list = []
    plot_df = None

    for fold_idx, (test_start, test_end) in enumerate(fold_bounds):
        train = weekly.iloc[:test_start]
        test = weekly.iloc[test_start:test_end].copy()

        X_train = train[feature_cols]
        y_train = train["liters"]
        X_test = test[feature_cols]
        y_test = test["liters"].values

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

        naive_preds = test["lag_1"].values

        model_mae_list.append(mean_absolute_error(y_test, preds))
        model_rmse_list.append(np.sqrt(mean_squared_error(y_test, preds)))
        model_smape_list.append(smape(y_test, preds))
        naive_mae_list.append(mean_absolute_error(y_test, naive_preds))
        naive_rmse_list.append(np.sqrt(mean_squared_error(y_test, naive_preds)))
        naive_smape_list.append(smape(y_test, naive_preds))

        # First fold is the most recent window (for plotting in dashboard)
        if fold_idx == 0:
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
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    metrics_rows = []
    plot_map: dict[str, pd.DataFrame] = {}

    for beer in focus_beers:
        beer_df = processed_df[processed_df["beer"] == beer].copy()
        metrics, plot_df = evaluate_beer_rolling_cv(
            beer_df,
            beer_name=beer,
            test_ratio=test_ratio,
            max_folds=max_folds,
        )
        if metrics is not None:
            metrics_rows.append(metrics)
        if plot_df is not None:
            plot_map[beer] = plot_df

    metrics_df = pd.DataFrame(metrics_rows)
    if metrics_df.empty:
        overall_df = pd.DataFrame()
    else:
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

    return metrics_df, plot_map, overall_df
