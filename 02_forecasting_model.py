import pandas as pd
import numpy as np
import os
from sklearn.ensemble import HistGradientBoostingRegressor

# ==============================
# CONFIG
# ==============================

PROCESSED_DATA_PATH = "data/processed/weekly_beer_data.csv"
FORECAST_OUTPUT_PATH = "data/processed/forecast_results.csv"

FORECAST_HORIZON = 52
LAGS = [1, 2, 3, 4, 8, 12, 26, 52]


# ==============================
# LOAD DATA
# ==============================

def load_data():
    print("Loading processed dataset...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["week"] = pd.to_datetime(df["week"])
    return df


# ==============================
# CREATE LAGS
# ==============================

def create_lag_features(df):
    df = df.sort_values("week")

    for lag in LAGS:
        df[f"lag_{lag}"] = df["liters"].shift(lag)

    return df


# ==============================
# FORECAST PER BEER
# ==============================

def forecast_beer(beer_df, beer_name):
    print(f"Training model for {beer_name}...")

    beer_df = create_lag_features(beer_df)
    beer_df = beer_df.dropna().copy()

    beer_df["liters"] = beer_df["liters"].clip(lower=0)

    feature_cols = [f"lag_{lag}" for lag in LAGS] + [
        "week_of_year",
        "month",
        "year",
        "hot_week",
        "rainy_week"
    ]

    X = beer_df[feature_cols]
    y = np.log1p(beer_df["liters"])

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X, y)

    history = beer_df.copy()
    forecasts = []

    for i in range(1, FORECAST_HORIZON + 1):

        next_week = history["week"].max() + pd.Timedelta(weeks=1)

        new_row = history.iloc[-1:].copy()
        new_row["week"] = next_week

        new_row["week_of_year"] = next_week.isocalendar().week
        new_row["month"] = next_week.month
        new_row["year"] = next_week.year

        new_row["hot_week"] = 0
        new_row["rainy_week"] = 0

        for lag in LAGS:
            new_row[f"lag_{lag}"] = history["liters"].iloc[-lag]

        X_new = new_row[feature_cols]

        pred_log = model.predict(X_new)[0]
        pred = np.expm1(pred_log)

        new_row["liters"] = pred

        forecasts.append({
            "week": next_week,
            "beer": beer_name,
            "forecast_liters": pred
        })

        history = pd.concat([history, new_row], ignore_index=True)

    return forecasts


# ==============================
# MAIN
# ==============================

def main():
    df = load_data()

    all_forecasts = []

    for beer in df["beer"].unique():
        beer_df = df[df["beer"] == beer].copy()
        forecasts = forecast_beer(beer_df, beer)
        all_forecasts.extend(forecasts)

    forecast_df = pd.DataFrame(all_forecasts)

    os.makedirs(os.path.dirname(FORECAST_OUTPUT_PATH), exist_ok=True)
    forecast_df.to_csv(FORECAST_OUTPUT_PATH, index=False)

    print("STEP 2 COMPLETE ✅")


if __name__ == "__main__":
    main()