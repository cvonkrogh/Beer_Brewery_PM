import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# -----------------------------
# CONFIG
# -----------------------------

FORECAST_HORIZON = 24  # months ahead


# -----------------------------
# LOAD & PREPARE DATA
# -----------------------------

@st.cache_data
def load_and_prepare_data():

    df = pd.read_csv("data/sales.csv", sep=";")

    # Clean columns
    df.columns = df.columns.str.strip()

    # Convert date
    df["Factuurdatum"] = pd.to_datetime(df["Factuurdatum"], dayfirst=True, errors="coerce")

    # Keep only positive liters
    df = df[df["Liter"] > 0].copy()

    # Monthly aggregation per beer (Grondstof)
    monthly = (
        df
        .groupby([df["Factuurdatum"].dt.to_period("M"), "Grondstof"])["Liter"]
        .sum()
        .reset_index()
    )

    monthly["Factuurdatum"] = monthly["Factuurdatum"].dt.to_timestamp()

    return monthly


# -----------------------------
# FORECAST FUNCTION (ROBUST)
# -----------------------------

def forecast_beer_monthly(monthly_data, beer_name, months_ahead=FORECAST_HORIZON):

    beer_data = (
        monthly_data[monthly_data["Grondstof"] == beer_name]
        .sort_values("Factuurdatum")
        .set_index("Factuurdatum")["Liter"]
    )

    # Create full monthly timeline
    full_index = pd.date_range(
        start=beer_data.index.min(),
        end=beer_data.index.max(),
        freq="MS"
    )

    beer_data = beer_data.reindex(full_index).fillna(0)

    n = len(beer_data)

    # Case 1: full seasonal model
    if n >= 24:
        model = ExponentialSmoothing(
            beer_data,
            trend="add",
            seasonal="add",
            seasonal_periods=12
        )

    # Case 2: trend only
    elif n >= 12:
        model = ExponentialSmoothing(
            beer_data,
            trend="add",
            seasonal=None
        )

    # Case 3: very short history
    else:
        avg = beer_data.mean()
        future_dates = pd.date_range(
            start=beer_data.index[-1] + pd.DateOffset(months=1),
            periods=months_ahead,
            freq="MS"
        )
        return pd.Series([avg] * months_ahead, index=future_dates)

    fit = model.fit()
    forecast = fit.forecast(months_ahead)

    # Ensure no negative forecasted volumes
    forecast = forecast.clip(lower=0)

    return forecast


# -----------------------------
# FORECAST ALL BEERS
# -----------------------------

@st.cache_data
def forecast_all_beers(monthly_data):

    forecasts = []

    for beer in monthly_data["Grondstof"].unique():

        f = forecast_beer_monthly(monthly_data, beer)

        df_f = pd.DataFrame({
            "Beer": beer,
            "Date": f.index,
            "Forecast_Liters": f.values
        })

        forecasts.append(df_f)

    return pd.concat(forecasts)


# -----------------------------
# STREAMLIT UI
# -----------------------------

st.set_page_config(layout="wide")
st.title("🍺 Brewery Demand Forecast Dashboard")

monthly_core = load_and_prepare_data()
full_forecast = forecast_all_beers(monthly_core)

# Year & Month selector
available_years = sorted(full_forecast["Date"].dt.year.unique())

year = st.selectbox("Select Year", available_years)
month = st.selectbox("Select Month", list(range(1, 13)))

# Filter forecast
result = full_forecast[
    (full_forecast["Date"].dt.year == year) &
    (full_forecast["Date"].dt.month == month)
].copy()

result = result.sort_values("Forecast_Liters", ascending=False)

# Metrics
total_liters = result["Forecast_Liters"].sum()

col1, col2 = st.columns(2)
col1.metric("Total Forecasted Liters", f"{round(total_liters, 0)} L")
col2.metric("Number of Beers", len(result))

# Top 5
st.subheader("🏆 Top 5 Beers")
st.dataframe(result.head(5), use_container_width=True)

# Bar chart
st.subheader("📊 Demand Distribution")
st.bar_chart(result.set_index("Beer")["Forecast_Liters"])

# Full table
st.subheader("📋 All Beers")
st.dataframe(result, use_container_width=True)
