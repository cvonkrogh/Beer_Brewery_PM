import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# ==============================
# CONFIG
# ==============================

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "processed" / "weekly_beer_data.csv"
EVENTS_PATH = BASE / "data" / "events.csv"
DATA_PATH_USED = None
LAGS = [1, 2, 3, 4, 8, 12, 26, 52]
TEST_SIZE = 0.2

FOCUS_BEERS = [
    "Hoop Bleke Nelis",
    "Hoop Lager",
    "Hoop Kaper Tropical IPA",
]

WEEK_FREQ = "W-MON"

# ==============================
# LOAD DATA
# ==============================

@st.cache_data
def load_data_with_events():
    global DATA_PATH_USED

    # Candidate paths (in order)
    candidates = [
        DATA_PATH,
        BASE / "data" / "processed" / "weekly_beer_data.csv",
        BASE / "data" / "raw" / "weekly_beer_data.csv",
        BASE / "weekly_beer_data.csv",
    ]

    chosen = None
    for p in candidates:
        try:
            if p and p.exists():
                chosen = p
                break
        except Exception:
            continue

    # Last-resort: search for files that look like weekly data under data/
    if chosen is None and (BASE / "data").exists():
        files = list((BASE / "data").rglob("weekly*.csv"))
        if files:
            chosen = files[0]

    if chosen is None:
        st.error(f"Could not find weekly data file. Tried {candidates}.")
        return pd.DataFrame(columns=["week", "beer", "container", "liters"]) 

    DATA_PATH_USED = chosen

    try:
        df = pd.read_csv(chosen, parse_dates=["week"])
    except Exception as e:
        st.error(f"Could not load data file: {chosen}. Error: {e}")
        return pd.DataFrame(columns=["week", "beer", "container", "liters"]) 

    # Ensure week is datetime
    if "week" in df.columns:
        df["week"] = pd.to_datetime(df["week"]) 
    else:
        st.error("Loaded data does not contain a 'week' column.")
        return pd.DataFrame(columns=["week", "beer", "container", "liters"]) 

    # Normalize possible volume column names to 'liters'
    possible_volume_cols = ["liters", "liter", "litres", "volume", "sales", "sales_liters", "sales_ltrs", "qty"]
    found_vol = None
    for col in possible_volume_cols:
        if col in df.columns:
            found_vol = col
            break
    if found_vol and found_vol != "liters":
        df = df.rename(columns={found_vol: "liters"})
    if "liters" not in df.columns:
        # If not found, create liters with zeros and warn
        df["liters"] = 0
        st.warning(f"No volume column found in {chosen}. Created 'liters' filled with zeros.")

    # Create date-based features used later
    df["month"] = df["week"].dt.month
    df["year"] = df["week"].dt.year

    # Create simple hot/rain flags if temp/rain columns exist, otherwise default to 0
    if "temp_mean" in df.columns:
        df["hot_week"] = (df["temp_mean"] > 25).astype(int)
    else:
        df["hot_week"] = 0

    if "rain_mm" in df.columns:
        df["rainy_week"] = (df["rain_mm"] > 5).astype(int)
    else:
        df["rainy_week"] = 0

    # Load events if available
    try:
        events = pd.read_csv(EVENTS_PATH)
        if {"month", "day"}.issubset(events.columns):
            event_dates = set(zip(events["month"], events["day"]))
            df["has_event"] = df["week"].apply(
                lambda w: 1 if (w.month, w.day) in event_dates else 0
            )
        else:
            df["has_event"] = 0
    except FileNotFoundError:
        df["has_event"] = 0
    except Exception:
        df["has_event"] = 0

    return df

# ==============================
# CREATE FEATURES
# ==============================

def _seasonal_terms(week_of_year: int) -> tuple[float, float]:
    angle = 2 * np.pi * (week_of_year / 52.18)
    return float(np.sin(angle)), float(np.cos(angle))

def ensure_weekly_continuity(group: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex a single series to a full weekly timeline (W-MON).
    Missing sales -> 0. Interpolate/fill weather and recompute time features.
    """
    group = group.sort_values("week").copy()
    group = group.drop_duplicates(subset=["week"], keep="last")

    full_weeks = pd.date_range(start=group["week"].min(), end=group["week"].max(), freq=WEEK_FREQ)
    group = group.set_index("week").reindex(full_weeks)
    group.index.name = "week"
    group = group.reset_index()

    # liters: missing => 0
    if "liters" in group.columns:
        group["liters"] = pd.to_numeric(group["liters"], errors="coerce").fillna(0.0).clip(lower=0)
    else:
        group["liters"] = 0.0

    # Fill/interpolate weather
    if "temp_mean" in group.columns:
        group["temp_mean"] = pd.to_numeric(group["temp_mean"], errors="coerce")
        group["temp_mean"] = group["temp_mean"].interpolate(limit_direction="both")
        group["temp_mean"] = group["temp_mean"].fillna(group["temp_mean"].mean())

    if "rain_mm" in group.columns:
        group["rain_mm"] = pd.to_numeric(group["rain_mm"], errors="coerce")
        group["rain_mm"] = group["rain_mm"].interpolate(limit_direction="both")
        group["rain_mm"] = group["rain_mm"].fillna(group["rain_mm"].mean())

    # Recompute time features
    iso_woy = group["week"].dt.isocalendar().week.astype(int)
    group["week_of_year"] = iso_woy
    group["month"] = group["week"].dt.month.astype(int)
    group["year"] = group["week"].dt.year.astype(int)
    group["time_index"] = np.arange(len(group), dtype=int)

    # Hot / rainy flags computed from quantiles of filled weather (if present)
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

    # Seasonal sin/cos
    sin_terms = group["week_of_year"].apply(lambda w: _seasonal_terms(int(w))[0])
    cos_terms = group["week_of_year"].apply(lambda w: _seasonal_terms(int(w))[1])
    group["sin_woy"] = sin_terms.astype(float)
    group["cos_woy"] = cos_terms.astype(float)

    return group

def create_lags(df):
    # removed (unused) — lag creation is handled inside create_features to avoid duplication
    return df

def create_features(df, lags):
    df = df.sort_values("week").copy()

    for lag in lags:
        df[f"lag_{lag}"] = df["liters"].shift(lag)

    df["roll_mean_4"] = df["liters"].shift(1).rolling(4, min_periods=1).mean()
    df["roll_mean_12"] = df["liters"].shift(1).rolling(12, min_periods=1).mean()
    df["roll_sum_4"] = df["liters"].shift(1).rolling(4, min_periods=1).sum()

    # Build feature list; include seasonal sin/cos if available
    feature_cols = [f"lag_{lag}" for lag in lags] + [
        "roll_mean_4",
        "roll_mean_12",
        "roll_sum_4",
    ]

    if "sin_woy" in df.columns and "cos_woy" in df.columns:
        feature_cols += ["sin_woy", "cos_woy"]

    feature_cols += [
        "month",
        "year",
        "hot_week",
        "rainy_week",
        "has_event",
    ]

    if "temp_mean" in df.columns:
        feature_cols.append("temp_mean")
    if "rain_mm" in df.columns:
        feature_cols.append("rain_mm")

    return df, feature_cols

# ==============================
# TRAIN AND PREDICT
# ==============================

def train_and_predict(beer_df, beer_name, container, test_weeks=None, forecast_weeks=0):
    # enforce weekly continuity first
    beer_df_cont = ensure_weekly_continuity(beer_df)

    # Adapt lag set to series length
    series_len = len(beer_df_cont)
    usable_lags = [lag for lag in LAGS if lag < series_len]
    if not usable_lags:
        return None

    # Now create features using usable_lags
    df_feat, feature_cols = create_features(beer_df_cont, usable_lags)
    df_feat = df_feat.dropna(subset=feature_cols + ["liters"]).copy()

    if len(df_feat) < 20:
        return None  # Not enough data
    
    # Split train/test: prefer explicit week count when provided
    if test_weeks and test_weeks > 0:
        test_weeks = int(test_weeks)
        split_idx = max(0, len(df_feat) - test_weeks)
    else:
        split_idx = int(len(df_feat) * (1 - TEST_SIZE))

    train = df_feat.iloc[:split_idx]
    test = df_feat.iloc[split_idx:]
    
    # If test is empty, still allow training on all data and produce future forecasts
    if len(train) < 10:
        return None

    # Train model
    X_test = test[feature_cols] if len(test) > 0 else pd.DataFrame(columns=feature_cols)
    y_test = test["liters"].values if len(test) > 0 else np.array([])
    
    # If forecasting into the future, train on the full available history (df_feat)
    if forecast_weeks and forecast_weeks > 0:
        X_train = df_feat[feature_cols]
        y_train = df_feat["liters"].clip(lower=0)
    else:
        X_train = train[feature_cols]
        y_train = train["liters"].clip(lower=0)
    
    model = HistGradientBoostingRegressor(
        loss="poisson" if y_train.sum() > 0 else "squared_error",
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        l2_regularization=0.1,
        random_state=42
    )

    # Time-series cross-validation (rolling-origin) to report CV metrics
    cv_results = {"mae": None, "rmse": None, "smape": None, "naive_mae": None, "naive_rmse": None, "naive_smape": None}
    try:
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)
        maes, rmses, smapes = [], [], []
        n_maes, n_rmses, n_smapes = [], [], []
        X = df_feat[feature_cols].values
        y = df_feat["liters"].values
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            # fit on training fold
            try:
                model_fold = HistGradientBoostingRegressor(
                    loss="poisson" if y_tr.sum() > 0 else "squared_error",
                    max_iter=200,
                    learning_rate=0.05,
                    max_depth=6,
                    l2_regularization=0.1,
                    random_state=42
                )
                model_fold.fit(X_tr, y_tr)
                y_pred = np.maximum(model_fold.predict(X_val), 0)
                maes.append(mean_absolute_error(y_val, y_pred))
                rmses.append(np.sqrt(mean_squared_error(y_val, y_pred)))
                smapes.append(np.mean(2.0 * np.abs(y_pred - y_val) / (np.abs(y_val) + np.abs(y_pred) + 1e-9)) * 100.0)
                # naive baseline
                naive_p = np.concatenate(([y_tr[-1]], y_val[:-1])) if len(y_val)>0 else np.array([])
                if len(naive_p)==len(y_val):
                    n_maes.append(mean_absolute_error(y_val, naive_p))
                    n_rmses.append(np.sqrt(mean_squared_error(y_val, naive_p)))
                    n_smapes.append(np.mean(2.0 * np.abs(naive_p - y_val) / (np.abs(y_val) + np.abs(naive_p) + 1e-9)) * 100.0)
            except Exception:
                continue
        if maes:
            cv_results["mae"] = float(np.mean(maes))
            cv_results["rmse"] = float(np.mean(rmses))
            cv_results["smape"] = float(np.mean(smapes))
        if n_maes:
            cv_results["naive_mae"] = float(np.mean(n_maes))
            cv_results["naive_rmse"] = float(np.mean(n_rmses))
            cv_results["naive_smape"] = float(np.mean(n_smapes))
    except Exception:
        cv_results = {k: None for k in cv_results}

    # Fit final model on chosen training set
    model.fit(X_train, y_train)
    
    results = {
        "week": test["week"].values if len(test) > 0 else np.array([]),
        "actual": y_test,
        "predicted": model.predict(X_test) if len(X_test) > 0 else np.array([]),
        "mae": None,
        "rmse": None,
        "beer": beer_name,
        "container": container,
        "future_weeks": [],
        "future_predicted": [],
        "cv_mae": cv_results["mae"],
        "cv_rmse": cv_results["rmse"],
        "cv_smape": cv_results["smape"],
        "cv_naive_mae": cv_results["naive_mae"],
        "cv_naive_rmse": cv_results["naive_rmse"],
        "cv_naive_smape": cv_results["naive_smape"],
    }

    if len(results["actual"]) > 0:
        results["predicted"] = np.maximum(results["predicted"], 0)
        results["mae"] = mean_absolute_error(results["actual"], results["predicted"]) if len(results["actual"])>0 else None
        results["rmse"] = np.sqrt(mean_squared_error(results["actual"], results["predicted"])) if len(results["actual"])>0 else None

    # If forecast_weeks requested, perform recursive forecasting beyond last known week
    if forecast_weeks and forecast_weeks > 0:
        # Prepare history of known liters (use full beer_df sorted)
        history = list(beer_df.sort_values("week")["liters"].tolist())
        last_week = beer_df["week"].max()

        # Load events to mark has_event for future dates if possible
        event_dates = set()
        try:
            events_df = pd.read_csv(EVENTS_PATH)
            if {"month", "day"}.issubset(events_df.columns):
                event_dates = set(zip(events_df["month"], events_df["day"]))
        except Exception:
            event_dates = set()

        # Build typical-weather lookup (median) by week_of_year from the continuity-ensured series
        typical_weather = {}
        try:
            # Only aggregate columns that exist
            agg_cols = {}
            if "temp_mean" in beer_df_cont.columns:
                agg_cols["temp_mean"] = ("temp_mean", "median")
            if "rain_mm" in beer_df_cont.columns:
                agg_cols["rain_mm"] = ("rain_mm", "median")
            if "hot_week" in beer_df_cont.columns:
                agg_cols["hot_week"] = ("hot_week", "median")
            if "rainy_week" in beer_df_cont.columns:
                agg_cols["rainy_week"] = ("rainy_week", "median")

            if agg_cols:
                tw = beer_df_cont.groupby("week_of_year").agg(**agg_cols).reset_index()
                for _, r in tw.iterrows():
                    w = int(r["week_of_year"]) if "week_of_year" in r else None
                    if w is None:
                        continue
                    typical_weather[w] = {
                        "temp_mean": float(r["temp_mean"]) if "temp_mean" in r and not pd.isna(r["temp_mean"]) else None,
                        "rain_mm": float(r["rain_mm"]) if "rain_mm" in r and not pd.isna(r["rain_mm"]) else None,
                        "hot_week": int(r["hot_week"]) if "hot_week" in r and not pd.isna(r["hot_week"]) else 0,
                        "rainy_week": int(r["rainy_week"]) if "rainy_week" in r and not pd.isna(r["rainy_week"]) else 0,
                    }
        except Exception:
            typical_weather = {}

        future_weeks = []
        future_preds = []

        for i in range(int(forecast_weeks)):
            last_week = last_week + pd.Timedelta(weeks=1)
            future_weeks.append(last_week)

            # week_of_year for seasonal/typical-weather lookup
            woy = int(last_week.isocalendar().week)

            # Build feature vector following the exact order of feature_cols
            feat = []
            for col in feature_cols:
                if col.startswith("lag_"):
                    lag_val = int(col.split("_")[1])
                    if len(history) >= lag_val:
                        feat.append(history[-lag_val])
                    else:
                        feat.append(np.mean(history) if len(history) > 0 else 0.0)
                elif col == "roll_mean_4":
                    feat.append(np.mean(history[-4:]) if len(history) > 0 else 0.0)
                elif col == "roll_mean_12":
                    feat.append(np.mean(history[-12:]) if len(history) > 0 else 0.0)
                elif col == "roll_sum_4":
                    feat.append(np.sum(history[-4:]) if len(history) > 0 else 0.0)
                elif col == "sin_woy":
                    feat.append(_seasonal_terms(woy)[0])
                elif col == "cos_woy":
                    feat.append(_seasonal_terms(woy)[1])
                elif col == "month":
                    feat.append(last_week.month)
                elif col == "year":
                    feat.append(last_week.year)
                elif col == "hot_week":
                    if woy in typical_weather and typical_weather[woy].get("hot_week") is not None:
                        feat.append(int(typical_weather[woy]["hot_week"]))
                    elif "hot_week" in beer_df_cont.columns:
                        feat.append(int(beer_df_cont["hot_week"].iloc[-1]))
                    else:
                        feat.append(0)
                elif col == "rainy_week":
                    if woy in typical_weather and typical_weather[woy].get("rainy_week") is not None:
                        feat.append(int(typical_weather[woy]["rainy_week"]))
                    elif "rainy_week" in beer_df_cont.columns:
                        feat.append(int(beer_df_cont["rainy_week"].iloc[-1]))
                    else:
                        feat.append(0)
                elif col == "has_event":
                    feat.append(1 if (last_week.month, last_week.day) in event_dates else 0)
                elif col == "temp_mean":
                    if woy in typical_weather and typical_weather[woy].get("temp_mean") is not None:
                        feat.append(float(typical_weather[woy]["temp_mean"]))
                    elif "temp_mean" in beer_df_cont.columns:
                        feat.append(float(beer_df_cont["temp_mean"].iloc[-1]))
                    else:
                        feat.append(0.0)
                elif col == "rain_mm":
                    if woy in typical_weather and typical_weather[woy].get("rain_mm") is not None:
                        feat.append(float(typical_weather[woy]["rain_mm"]))
                    elif "rain_mm" in beer_df_cont.columns:
                        feat.append(float(beer_df_cont["rain_mm"].iloc[-1]))
                    else:
                        feat.append(0.0)
                else:
                    # default fallback
                    feat.append(0.0)

            fv = np.array(feat, dtype=float)
            # Safety: ensure length matches
            if fv.shape[0] != len(feature_cols):
                if fv.shape[0] < len(feature_cols):
                    fv = np.pad(fv, (0, len(feature_cols)-fv.shape[0]), constant_values=0.0)
                else:
                    fv = fv[:len(feature_cols)]

            pred = model.predict(fv.reshape(1, -1))[0]
            pred = max(0.0, float(pred))
            future_preds.append(pred)
            history.append(pred)

        results["future_weeks"] = np.array(future_weeks)
        results["future_predicted"] = np.array(future_preds)

    return results

# ==============================
# MAIN APP
# ==============================

st.set_page_config(layout="wide")
st.title("🎯 Prediction vs Actual Sales – Test Sample Analysis")

# Sidebar options
st.sidebar.header("Options")
st.sidebar.write("Control how much historical data is used as the test sample for comparison.")
# 0 means use fraction TEST_SIZE; otherwise use fixed number of weeks
#test_horizon_weeks = st.sidebar.slider("Test horizon (weeks, 0 = use fraction)", min_value=0, max_value=220, value=52, step=1)
# Forecast horizon (future weeks to predict beyond available data)
# Forecast horizon (future weeks to predict beyond available data) - discrete choices
# Options: 12, 26, 52 weeks (removed 104)
forecast_horizon_weeks = st.sidebar.select_slider(
    "Forecast horizon (future weeks)",
    options=[12, 26, 52],
    value=12,
    format_func=lambda x: f"{x} weeks"
)

# Historical test horizon: allow only specific discrete choices (weeks)
# Options: 4 weeks, 16 weeks, 24 weeks
test_horizon_weeks = st.sidebar.select_slider(
    "Test horizon (weeks)",
    options=[4, 16, 24],
    value=16,
    format_func=lambda x: f"{x} weeks"
)

# Load data
df = load_data_with_events()

# Filter to focus beers
df = df[df["beer"].isin(FOCUS_BEERS)].copy()

# Create tabs for each beer
tabs = st.tabs(FOCUS_BEERS)

for tab, beer in zip(tabs, FOCUS_BEERS):
    with tab:
        beer_data = df[df["beer"] == beer]
        containers = beer_data["container"].unique()
        
        if len(containers) == 0:
            st.warning(f"No data available for {beer}")
            continue
        
        # Create subtabs for each container
        container_tabs = st.tabs(list(containers))
        
        for container_tab, container in zip(container_tabs, containers):
            with container_tab:
                beer_container_data = beer_data[beer_data["container"] == container].copy()
                
                if len(beer_container_data) < 20:
                    st.warning(f"Not enough data for {beer} – {container}")
                    continue
                
                # Train and predict (pass test_horizon_weeks; 0 uses fraction)
                results = train_and_predict(
                    beer_container_data,
                    beer,
                    container,
                    test_weeks=(test_horizon_weeks if test_horizon_weeks > 0 else None),
                    forecast_weeks=forecast_horizon_weeks
                )
                
                if results is None:
                    st.error(f"Could not train model for {beer} – {container}")
                    continue

                # create a safe unique key for Streamlit elements
                safe = f"{beer}_{container}".replace(" ", "_").replace("/", "_").replace("\\", "_")

                # Compute additional metrics: SMAPE and naive baselines
                actual = np.array(results.get("actual", [])) if results.get("actual") is not None else np.array([])
                pred = np.array(results.get("predicted", [])) if results.get("predicted") is not None else np.array([])
                n = len(actual)

                def _fmt_num(x, unit=" L"):
                    try:
                        if x is None:
                            return "n/a"
                        if isinstance(x, (float, int)) and np.isnan(x):
                            return "n/a"
                        return f"{float(x):.2f}{unit}"
                    except Exception:
                        return "n/a"

                if n > 0:
                    # SMAPE (percentage)
                    smape = np.mean(2.0 * np.abs(pred - actual) / (np.abs(actual) + np.abs(pred) + 1e-9)) * 100.0

                    # Naive: previous-observation forecast (t -> actual_{t-1}); first value uses actual[0]
                    naive_pred = np.concatenate(([actual[0]], actual[:-1])) if n > 0 else np.array([])
                    naive_mae = mean_absolute_error(actual, naive_pred)
                    naive_rmse = np.sqrt(mean_squared_error(actual, naive_pred))
                    naive_smape = np.mean(2.0 * np.abs(naive_pred - actual) / (np.abs(actual) + np.abs(naive_pred) + 1e-9)) * 100.0
                else:
                    smape = np.nan
                    naive_mae = np.nan
                    naive_rmse = np.nan
                    naive_smape = np.nan

                mae = results.get("mae", np.nan)
                rmse = results.get("rmse", np.nan)

                # Display metrics in requested order: MAE - Naive MAE - RMSE - Naive RMSE - SMAPE % - Naive SMAPE % - Test Sample Size
                cols = st.columns(7)
                with cols[0]:
                    st.metric("MAE", _fmt_num(mae))
                with cols[1]:
                    st.metric("Naive MAE", _fmt_num(naive_mae))
                with cols[2]:
                    st.metric("RMSE", _fmt_num(rmse))
                with cols[3]:
                    st.metric("Naive RMSE", _fmt_num(naive_rmse))
                with cols[4]:
                    st.metric("SMAPE", f"{smape:.1f}%" if not np.isnan(smape) else "n/a")
                with cols[5]:
                    st.metric("Naive SMAPE", f"{naive_smape:.1f}%" if not np.isnan(naive_smape) else "n/a")
                with cols[6]:
                    st.metric("Test Sample Size", n)

                # Create comparison dataframe
                comparison_df = pd.DataFrame({
                    "Week": results["week"],
                    "Actual Sales": results["actual"],
                    "Predicted Sales": results["predicted"],
                    "Difference": results["actual"] - results["predicted"],
                    "Absolute Error": np.abs(results["actual"] - results["predicted"])
                })
                
                # Plot 1: Actual vs Predicted
                st.subheader("Actual vs Predicted Sales")
                fig1 = go.Figure()

                fig1.add_trace(go.Scatter(
                    x=comparison_df["Week"],
                    y=comparison_df["Actual Sales"],
                    mode="lines+markers",
                    name="Actual Sales",
                    line=dict(color="blue", width=2)
                ))
                
                fig1.add_trace(go.Scatter(
                    x=comparison_df["Week"],
                    y=comparison_df["Predicted Sales"],
                    mode="lines+markers",
                    name="Predicted Sales",
                    line=dict(color="red", width=2, dash="dash")
                ))
                
                fig1.update_layout(
                    title=f"{beer} – {container}",
                    xaxis_title="Week",
                    yaxis_title="Liters",
                    hovermode="x unified",
                    height=500
                )

                # If future forecasts exist, add them to the plot
                if results.get("future_weeks") is not None and len(results.get("future_weeks"))>0:
                    fig1.add_trace(go.Scatter(
                        x=results["future_weeks"],
                        y=results["future_predicted"],
                        mode="lines+markers",
                        name="Forecast (future)",
                        line=dict(color="green", width=2, dash="dot")
                    ))

                st.plotly_chart(fig1, use_container_width=True, key=f"fig1_{safe}")
                
                # Plot 2: Prediction Error
                st.subheader("Prediction Error Over Time")
                fig2 = go.Figure()

                fig2.add_trace(go.Bar(
                    x=comparison_df["Week"],
                    y=comparison_df["Difference"],
                    name="Prediction Error",
                    marker=dict(color=comparison_df["Difference"], colorscale="RdBu", showscale=True),
                    showlegend=False
                ))
                
                fig2.add_hline(y=0, line_dash="dash", line_color="gray")
                
                fig2.update_layout(
                    title="Prediction Error (Actual - Predicted)",
                    xaxis_title="Week",
                    yaxis_title="Error (Liters)",
                    height=400
                )

                st.plotly_chart(fig2, use_container_width=True, key=f"fig2_{safe}")
                
                # Data table
                st.subheader("Detailed Comparison")
                st.dataframe(
                    comparison_df.round(2),
                    use_container_width=True,
                    height=400,
                    key=f"df_{safe}"
                )

                # Summary statistics
                st.subheader("Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Actual Sales", f"{results['actual'].mean():.2f} L")
                with col2:
                    st.metric("Mean Predicted Sales", f"{results['predicted'].mean():.2f} L")
                with col3:
                    st.metric("Max Error", f"{comparison_df['Absolute Error'].max():.2f} L")
                with col4:
                    denom = comparison_df['Actual Sales'].sum()
                    accuracy = (1 - (comparison_df['Absolute Error'].sum() / denom)) * 100 if denom > 0 else 0.0
                    st.metric("Overall Accuracy", f"{accuracy:.1f}%")
