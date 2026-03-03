import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROCESSED_DATA_PATH = "data/processed/weekly_beer_data.csv"
LAGS = [1, 2, 3, 4, 8, 12, 26, 52]
TEST_WEEKS = 4         # length of each test window (a few weeks)
N_FOLDS = 5             # number of rolling test windows

FOCUS_BEERS = [
    "Hoop Bleke Nelis",
    "Hoop Lager",
    "Hoop Kaper Tropical IPA",
]


# ==============================
# CREATE LAGS
# ==============================

def create_lags(df):
    df = df.sort_values("week")
    for lag in LAGS:
        df[f"lag_{lag}"] = df["liters"].shift(lag)
    return df


# ==============================
# SMAPE FUNCTION
# ==============================

def smape(y_true, y_pred):
    return np.mean(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred) + 1e-6)
    ) * 100


# ==============================
# EVALUATE SINGLE BEER
# ==============================

def evaluate_beer(beer_df, beer_name):

    beer_df = create_lags(beer_df)
    beer_df = beer_df.dropna().copy()

    beer_df["liters"] = beer_df["liters"].clip(lower=0)

    feature_cols = [f"lag_{lag}" for lag in LAGS] + [
        "week_of_year",
        "month",
        "year",
        "hot_week",
        "rainy_week"
    ]

    n = len(beer_df)
    if n < TEST_WEEKS * 2:
        # Not enough history for CV: fall back to single holdout at the end
        folds_to_use = 1
    else:
        folds_to_use = min(
            N_FOLDS,
            max(1, (n // TEST_WEEKS) - 1)
        )

    model_mae_list = []
    model_rmse_list = []
    model_smape_list = []

    naive_mae_list = []
    naive_rmse_list = []
    naive_smape_list = []

    # Rolling-origin CV: keep time order, slide test window forward
    train_start = 0
    base_train_size = int(n * 0.8)
    remaining = n - base_train_size - TEST_WEEKS
    step = 0 if folds_to_use == 1 else max(1, remaining // (folds_to_use - 1))

    # Store the *best* fold for plotting (prefer full TEST_WEEKS, otherwise longest)
    best_test = None
    best_preds = None
    best_naive = None
    best_len = 0

    for fold in range(folds_to_use):
        train_end = base_train_size + fold * step
        test_start = train_end
        test_end = min(test_start + TEST_WEEKS, n)

        if test_end - test_start <= 0:
            continue

        train = beer_df.iloc[train_start:train_end]
        test = beer_df.iloc[test_start:test_end]

        X_train = train[feature_cols]
        y_train = np.log1p(train["liters"])

        X_test = test[feature_cols]
        y_test = test["liters"].values

        # -------------------------
        # Train Model
        # -------------------------
        model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )

        model.fit(X_train, y_train)

        preds_log = model.predict(X_test)
        preds = np.expm1(preds_log)

        # -------------------------
        # Metrics
        # -------------------------
        model_mae_list.append(mean_absolute_error(y_test, preds))
        model_rmse_list.append(np.sqrt(mean_squared_error(y_test, preds)))
        model_smape_list.append(smape(y_test, preds))

        # -------------------------
        # Naive Baseline (Last Week)
        # -------------------------
        naive_preds = test["lag_1"].values
        naive_mae_list.append(mean_absolute_error(y_test, naive_preds))
        naive_rmse_list.append(np.sqrt(mean_squared_error(y_test, naive_preds)))
        naive_smape_list.append(smape(y_test, naive_preds))

        # Save best fold for plotting
        cur_len = len(test)
        if cur_len > best_len:
            best_len = cur_len
            best_test = test.copy()
            best_preds = preds
            best_naive = naive_preds

    # -------------------------
    # Plot Actual vs Predicted (best fold)
    # -------------------------
    if best_test is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(best_test["week"], best_test["liters"].values, label="Actual")
        plt.plot(best_test["week"], best_preds, label="Model Forecast")
        plt.plot(best_test["week"], best_naive, label="Naive Forecast", linestyle="--")

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d (W%U)"))

        plt.title(f"{beer_name} – Rolling CV (best fold, {len(best_test)} weeks)")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return {
        "beer": beer_name,
        "folds": len(model_mae_list),
        "Model MAE": np.mean(model_mae_list),
        "Naive MAE": np.mean(naive_mae_list),
        "Model RMSE": np.mean(model_rmse_list),
        "Naive RMSE": np.mean(naive_rmse_list),
        "Model SMAPE (%)": np.mean(model_smape_list),
        "Naive SMAPE (%)": np.mean(naive_smape_list),
    }


# ==============================
# MAIN
# ==============================

def main():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["week"] = pd.to_datetime(df["week"])

    # Focus only on the three beers of interest
    df = df[df["beer"].isin(FOCUS_BEERS)].copy()

    results = []

    for beer in df["beer"].unique():
        print(f"\nEvaluating {beer}...")
        beer_df = df[df["beer"] == beer].copy()
        result = evaluate_beer(beer_df, beer)
        results.append(result)

    results_df = pd.DataFrame(results)

    print("\n===============================================")
    print("MODEL vs NAIVE BASELINE – Rolling 80/20 CV")
    print("Average over folds (per beer)")
    print("===============================================\n")
    print(results_df)


if __name__ == "__main__":
    main()