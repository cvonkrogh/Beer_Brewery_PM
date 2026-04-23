import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import os

# ==============================
# CONFIG
# ==============================

PROCESSED_DATA_PATH = "data/processed/weekly_beer_data.csv"
EVENTS_DATA_PATH = "data/events.csv"
FORECAST_OUTPUT_PATH = "data/processed/forecast_results.csv"

FORECAST_HORIZON = 52
LAGS = [1, 2, 3, 4, 8, 12, 26, 52]
WEEK_FREQ = "W-MON"
EPS = 1e-9
EXCLUDED_CONTAINER_KEYWORDS = ("can",)

# ==============================
# LOAD DATA
# ==============================


def load_data():
    print("Loading processed dataset...")
    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["week"])
    return df


def load_events():
    """Load recurring event dates to project event_week onto future weeks."""
    events = pd.read_csv(EVENTS_DATA_PATH)
    return events


def is_event_week(date, events):
    """Check whether any recurring event falls in the ISO week of `date`."""
    year = date.year
    week_start = date - pd.Timedelta(days=date.weekday())
    week_end = week_start + pd.Timedelta(days=6)
    for _, row in events.iterrows():
        try:
            evt = pd.Timestamp(year=year, month=int(row["month"]), day=int(row["day"]))
            if week_start <= evt <= week_end:
                return 1
        except ValueError:
            continue
    return 0


# ==============================
# HELPERS
# ==============================


def _seasonal_terms(week_of_year: int) -> tuple[float, float]:
    # 52.18 is closer to the average ISO weeks/year; keeps sin/cos continuous.
    angle = 2 * np.pi * (week_of_year / 52.18)
    return float(np.sin(angle)), float(np.cos(angle))


def ensure_weekly_continuity(group: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex to a full weekly timeline so lags represent true week-to-week history.
    Missing sales become 0 liters; weather/time features are recomputed/fill-forwarded.
    """
    group = group.sort_values("week").copy()
    group = group.drop_duplicates(subset=["week"], keep="last")

    full_weeks = pd.date_range(
        start=group["week"].min(),
        end=group["week"].max(),
        freq=WEEK_FREQ,
    )

    group = group.set_index("week").reindex(full_weeks)
    group.index.name = "week"
    group = group.reset_index()

    # liters: missing => 0 demand
    group["liters"] = pd.to_numeric(group["liters"], errors="coerce").fillna(0.0).clip(lower=0)

    # If temp/rain exist, fill gaps so features remain usable
    if "temp_mean" in group.columns:
        group["temp_mean"] = pd.to_numeric(group["temp_mean"], errors="coerce")
        group["temp_mean"] = group["temp_mean"].interpolate(limit_direction="both")
        group["temp_mean"] = group["temp_mean"].fillna(group["temp_mean"].mean())

    if "rain_mm" in group.columns:
        group["rain_mm"] = pd.to_numeric(group["rain_mm"], errors="coerce")
        group["rain_mm"] = group["rain_mm"].interpolate(limit_direction="both")
        group["rain_mm"] = group["rain_mm"].fillna(group["rain_mm"].mean())

    # Recompute time features from week
    iso_woy = group["week"].dt.isocalendar().week.astype(int)
    group["week_of_year"] = iso_woy
    group["month"] = group["week"].dt.month.astype(int)
    group["year"] = group["week"].dt.year.astype(int)
    group["time_index"] = np.arange(len(group), dtype=int)

    # Recompute hot/rainy flags from filled weather if possible
    if "temp_mean" in group.columns:
        thresh = float(np.nanquantile(group["temp_mean"], 0.75))
        group["hot_week"] = (group["temp_mean"] >= thresh).astype(int)
    else:
        group["hot_week"] = 0

    if "rain_mm" in group.columns:
        thresh = float(np.nanquantile(group["rain_mm"], 0.75))
        group["rainy_week"] = (group["rain_mm"] >= thresh).astype(int)
    else:
        group["rainy_week"] = 0

    sin_terms = group["week_of_year"].apply(lambda w: _seasonal_terms(int(w))[0])
    cos_terms = group["week_of_year"].apply(lambda w: _seasonal_terms(int(w))[1])
    group["sin_woy"] = sin_terms.astype(float)
    group["cos_woy"] = cos_terms.astype(float)

    return group


# ==============================
# CREATE LAGS
# ==============================


def create_features(df: pd.DataFrame, lags: list[int]) -> tuple[pd.DataFrame, list[str]]:
    df = df.sort_values("week").copy()

    for lag in lags:
        df[f"lag_{lag}"] = df["liters"].shift(lag)

    # Rolling summaries based on past data only (shift(1) avoids leakage)
    df["roll_mean_4"] = df["liters"].shift(1).rolling(4, min_periods=1).mean()
    df["roll_mean_12"] = df["liters"].shift(1).rolling(12, min_periods=1).mean()
    df["roll_sum_4"] = df["liters"].shift(1).rolling(4, min_periods=1).sum()

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

    if "event_week" in df.columns:
        feature_cols.append("event_week")

    if "temp_mean" in df.columns:
        feature_cols.append("temp_mean")
    if "rain_mm" in df.columns:
        feature_cols.append("rain_mm")

    return df, feature_cols


# ==============================
# TRAIN MODEL
# ==============================


def train_model(train_df, feature_cols):
    X = train_df[feature_cols]
    y = train_df["liters"].clip(lower=0)

    loss = "poisson" if float(y.sum()) > 0 else "squared_error"

    model = HistGradientBoostingRegressor(
        loss=loss,
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        l2_regularization=0.1,
        random_state=42
    )

    model.fit(X, y)
    return model


# ==============================
# FORECAST FUNCTION
# ==============================


def forecast_series(df, beer, container, events=None):

    df = ensure_weekly_continuity(df)

    # Adapt lag set to series length (avoid dropping everything on short histories)
    series_len = len(df)
    usable_lags = [lag for lag in LAGS if lag < series_len]
    if not usable_lags:
        # Extremely short history: forecast recent average (non-negative)
        avg = float(df["liters"].tail(8).mean())
        last_week = df["week"].max()
        future_weeks = pd.date_range(start=last_week + pd.Timedelta(weeks=1), periods=FORECAST_HORIZON, freq=WEEK_FREQ)
        return pd.DataFrame(
            {
                "week": future_weeks,
                "beer": beer,
                "container": container,
                "forecast_liters": np.maximum(avg, 0.0),
            }
        )

    df_feat, feature_cols = create_features(df, usable_lags)
    train_df = df_feat.dropna(subset=feature_cols).copy()

    if train_df.empty:
        avg = float(df["liters"].tail(8).mean())
        last_week = df["week"].max()
        future_weeks = pd.date_range(start=last_week + pd.Timedelta(weeks=1), periods=FORECAST_HORIZON, freq=WEEK_FREQ)
        return pd.DataFrame(
            {
                "week": future_weeks,
                "beer": beer,
                "container": container,
                "forecast_liters": np.maximum(avg, 0.0),
            }
        )

    model = train_model(train_df, feature_cols)

    # Build a seasonal "typical weather" lookup for future weeks
    weather_by_woy = {}
    if "temp_mean" in df.columns or "rain_mm" in df.columns:
        agg = df.groupby("week_of_year").agg(
            temp_mean=("temp_mean", "mean") if "temp_mean" in df.columns else ("hot_week", "mean"),
            rain_mm=("rain_mm", "mean") if "rain_mm" in df.columns else ("rainy_week", "mean"),
        )
        weather_by_woy = agg.to_dict(orient="index")

    temp_thresh = float(np.nanquantile(df["temp_mean"], 0.75)) if "temp_mean" in df.columns else None
    rain_thresh = float(np.nanquantile(df["rain_mm"], 0.75)) if "rain_mm" in df.columns else None

    history = df["liters"].astype(float).tolist()
    last_week = df["week"].max()
    future_predictions: list[dict] = []

    for _ in range(FORECAST_HORIZON):
        next_week = last_week + pd.Timedelta(weeks=1)
        woy = int(next_week.isocalendar().week)
        sin_woy, cos_woy = _seasonal_terms(woy)

        # Typical weather proxy per week-of-year if available
        temp_mean = None
        rain_mm = None
        if weather_by_woy:
            w = weather_by_woy.get(woy, {})
            temp_mean = w.get("temp_mean")
            rain_mm = w.get("rain_mm")

        hot_week = int(temp_thresh is not None and temp_mean is not None and float(temp_mean) >= temp_thresh)
        rainy_week = int(rain_thresh is not None and rain_mm is not None and float(rain_mm) >= rain_thresh)

        evt_flag = is_event_week(next_week, events) if events is not None else 0

        row = {
            "sin_woy": sin_woy,
            "cos_woy": cos_woy,
            "month": int(next_week.month),
            "year": int(next_week.year),
            "hot_week": hot_week,
            "rainy_week": rainy_week,
            "roll_mean_4": float(np.mean(history[-4:])) if history else 0.0,
            "roll_mean_12": float(np.mean(history[-12:])) if history else 0.0,
            "roll_sum_4": float(np.sum(history[-4:])) if history else 0.0,
        }

        if "event_week" in feature_cols:
            row["event_week"] = evt_flag

        for lag in usable_lags:
            row[f"lag_{lag}"] = float(history[-lag]) if len(history) >= lag else 0.0

        if "temp_mean" in df.columns:
            row["temp_mean"] = float(temp_mean) if temp_mean is not None else float(df["temp_mean"].mean())
        if "rain_mm" in df.columns:
            row["rain_mm"] = float(rain_mm) if rain_mm is not None else float(df["rain_mm"].mean())

        X_new = pd.DataFrame([row], columns=feature_cols).fillna(0.0)
        pred = float(model.predict(X_new)[0])
        pred = max(0.0, pred)  # demand can't be negative

        future_predictions.append(
            {
                "week": next_week,
                "beer": beer,
                "container": container,
                "forecast_liters": pred,
            }
        )

        history.append(pred)
        last_week = next_week

    return pd.DataFrame(future_predictions)


# ==============================
# MAIN
# ==============================


def main():

    df = load_data()
    events = load_events()

    results = []

    grouped = df.groupby(["beer", "container"])

    for (beer, container), group in grouped:
        if any(keyword in str(container).lower() for keyword in EXCLUDED_CONTAINER_KEYWORDS):
            print(f"Skipping {beer} - {container} (container excluded)...")
            continue

        print(f"Training model for {beer} - {container}...")

        if group["liters"].sum() == 0:
            print("Skipping (no sales)...")
            continue

        forecast_df = forecast_series(group.copy(), beer, container, events=events)
        results.append(forecast_df)

    final_forecast = pd.concat(results)

    os.makedirs("data/processed", exist_ok=True)
    final_forecast.to_csv(FORECAST_OUTPUT_PATH, index=False)

    print("STEP 2 COMPLETE ✅")
    print("Forecast shape:", final_forecast.shape)


if __name__ == "__main__":
    main()
