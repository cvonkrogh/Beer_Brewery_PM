import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import os

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

def create_lags(df):
    df = df.sort_values("week")
    for lag in LAGS:
        df[f"lag_{lag}"] = df["liters"].shift(lag)
    return df

# ==============================
# TRAIN MODEL
# ==============================

def train_model(train_df, feature_cols):
    X = train_df[feature_cols]
    y = np.log1p(train_df["liters"].clip(lower=0))

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X, y)
    return model

# ==============================
# FORECAST FUNCTION
# ==============================

def forecast_series(df, beer, container):

    df = create_lags(df)
    df = df.dropna().copy()

    feature_cols = [f"lag_{lag}" for lag in LAGS] + [
        "week_of_year",
        "month",
        "year",
        "hot_week",
        "rainy_week"
    ]

    train_df = df.copy()

    model = train_model(train_df, feature_cols)

    last_row = df.iloc[-1:].copy()
    future_predictions = []

    current_data = df.copy()

    for i in range(FORECAST_HORIZON):

        next_week = current_data["week"].max() + pd.Timedelta(weeks=1)

        new_row = last_row.copy()
        new_row["week"] = next_week

        # Update time features
        new_row["week_of_year"] = next_week.isocalendar()[1]
        new_row["month"] = next_week.month
        new_row["year"] = next_week.year

        # Recompute lags from updated dataset
        temp_df = pd.concat([current_data, new_row], ignore_index=True)

        temp_df = create_lags(temp_df)

        new_row = temp_df.iloc[-1:].copy()

        X_new = new_row[feature_cols]
        pred_log = model.predict(X_new)[0]
        pred = np.expm1(pred_log)

        new_row["liters"] = pred

        future_predictions.append({
            "week": next_week,
            "beer": beer,
            "container": container,
            "forecast_liters": pred
        })

        current_data = pd.concat([current_data, new_row], ignore_index=True)
        last_row = new_row.copy()

    return pd.DataFrame(future_predictions)

# ==============================
# MAIN
# ==============================

def main():

    df = load_data()

    results = []

    grouped = df.groupby(["beer", "container"])

    for (beer, container), group in grouped:

        print(f"Training model for {beer} - {container}...")

        if group["liters"].sum() == 0:
            print("Skipping (no sales)...")
            continue

        forecast_df = forecast_series(group.copy(), beer, container)
        results.append(forecast_df)

    final_forecast = pd.concat(results)

    os.makedirs("data/processed", exist_ok=True)
    final_forecast.to_csv(FORECAST_OUTPUT_PATH, index=False)

    print("STEP 2 COMPLETE ✅")
    print("Forecast shape:", final_forecast.shape)


if __name__ == "__main__":
    main()