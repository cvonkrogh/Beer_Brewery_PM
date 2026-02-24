import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# ==============================
# CONFIG
# ==============================

FORECAST_PATH = "data/processed/forecast_results.csv"
PRODUCTION_PATH = "data/processed/production_schedule.csv"

CONTAINER_LITER_MAP = {
    "Can 33cl": 0.33,
    "Bottle 33cl": 0.33,
    "Bottle 75cl": 0.75,
    "Keg 20L": 20,
    "Keg 50L": 50
}

# ==============================
# LOAD DATA
# ==============================

@st.cache_data
def load_data():
    forecast = pd.read_csv(FORECAST_PATH)
    production = pd.read_csv(PRODUCTION_PATH)

    forecast["week"] = pd.to_datetime(forecast["week"])
    production["production_week"] = pd.to_datetime(production["production_week"])
    production["available_week"] = pd.to_datetime(production["available_week"])

    return forecast, production


forecast_df, production_df = load_data()

st.set_page_config(layout="wide")
st.title("🍺 Brewery Production Planning Cockpit")

# ==============================
# DEFAULT TIME WINDOW
# ==============================

today = pd.Timestamp.today().normalize()
default_end = today + pd.Timedelta(weeks=52)

st.sidebar.header("📅 Time Selection")

start_date = st.sidebar.date_input("Start Date", today)
end_date = st.sidebar.date_input("End Date", default_end)

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

filtered_forecast = forecast_df[
    (forecast_df["week"] >= start_date) &
    (forecast_df["week"] <= end_date)
]

filtered_production = production_df[
    (production_df["production_week"] >= start_date) &
    (production_df["production_week"] <= end_date)
]

# ==============================
# BEER SELECTION
# ==============================

st.sidebar.header("🍺 Beer Selection")
selected_beer = st.sidebar.selectbox(
    "Select Beer",
    sorted(filtered_forecast["beer"].unique())
)

beer_forecast = filtered_forecast[
    filtered_forecast["beer"] == selected_beer
]

beer_production = filtered_production[
    filtered_production["beer"] == selected_beer
]

# ==============================
# CONTAINER UNIT CONVERSION
# ==============================

beer_forecast["units"] = beer_forecast.apply(
    lambda row: row["forecast_liters"] / CONTAINER_LITER_MAP.get(row["container"], 1),
    axis=1
)

# ==============================
# KPI SECTION
# ==============================

st.header("📊 Forecast Overview")

total_liters = beer_forecast["forecast_liters"].sum()
total_units = beer_forecast["units"].sum()

col1, col2 = st.columns(2)
col1.metric("Total Forecasted Liters", f"{total_liters:,.0f} L")
col2.metric("Total Forecasted Units", f"{total_units:,.0f}")

# ==============================
# CONTAINER BREAKDOWN
# ==============================

st.subheader("📦 Container Breakdown")

container_summary = (
    beer_forecast.groupby("container")
    .agg({
        "forecast_liters": "sum",
        "units": "sum"
    })
    .reset_index()
)

fig_container = px.bar(
    container_summary,
    x="container",
    y="units",
    title="Forecasted Units per Container"
)

st.plotly_chart(fig_container, width="stretch")

# ==============================
# PRODUCTION SCHEDULE
# ==============================

st.header("🏭 Brewing Schedule")

if beer_production.empty:
    st.warning("No brewing scheduled in selected timeframe.")
else:

    fig_prod = px.bar(
        beer_production,
        x="production_week",
        y="volume",
        title="Scheduled Brewing Volume (Liters)"
    )

    st.plotly_chart(fig_prod, width="stretch")

    st.subheader("🧠 Packaging Strategy")
    st.write(f"Strategy: **{beer_production['packaging_strategy'].iloc[0]}**")

    st.dataframe(
        beer_production[
            ["production_week", "available_week", "volume"]
        ],
        width="stretch"
    )

# ==============================
# STOCKOUT RISK INDICATOR
# ==============================

st.header("⚠ Stockout Risk Indicator")

weekly_demand = (
    beer_forecast.groupby("week")["forecast_liters"]
    .sum()
    .reset_index()
)

avg_demand = weekly_demand["forecast_liters"].mean()

if avg_demand == 0:
    st.success("No demand forecasted.")
else:
    coverage_weeks = total_liters / avg_demand

    if coverage_weeks < 12:
        st.error("⚠ Risk of stockout — insufficient planned coverage.")
    else:
        st.success("✅ Low stockout risk — coverage sufficient.")

# ==============================
# DEMAND HEATMAP
# ==============================

st.header("🔥 Demand Heatmap")

beer_forecast["year_month"] = beer_forecast["week"].dt.to_period("M").astype(str)

heatmap_data = (
    beer_forecast.groupby(["year_month", "container"])["units"]
    .sum()
    .reset_index()
)

fig_heatmap = px.density_heatmap(
    heatmap_data,
    x="year_month",
    y="container",
    z="units",
    color_continuous_scale="Blues",
    title="Monthly Container Demand Intensity"
)

st.plotly_chart(fig_heatmap, width="stretch")