import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ==============================
# CONFIG
# ==============================

FORECAST_PATH = "data/processed/forecast_results.csv"
PRODUCTION_PATH = "data/processed/production_schedule.csv"

SMALL_TANK = 2000
LARGE_TANK = 6000


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

    forecast["month"] = forecast["week"].dt.month
    forecast["year"] = forecast["week"].dt.year

    return forecast, production


forecast_df, production_df = load_data()

st.set_page_config(layout="wide")
st.title("🍺 Brewery Forecast & Production Dashboard")

# ==============================
# YEAR OVERVIEW
# ==============================

st.header("📅 Annual Production Overview")

annual_summary = forecast_df.groupby("beer")["forecast_liters"].sum().reset_index()

fig_total = px.bar(
    annual_summary,
    x="beer",
    y="forecast_liters",
    title="Total Forecasted Sales (Next 52 Weeks)"
)

st.plotly_chart(fig_total, use_container_width=True)

# ==============================
# heatmap view
# ==============================

st.header("🗓 Demand Heatmap (Year Overview)")

heatmap_data = forecast_df.copy()
heatmap_data["year_month"] = heatmap_data["week"].dt.to_period("M").astype(str)

heatmap_grouped = (
    heatmap_data.groupby(["beer", "year_month"])["forecast_liters"]
    .sum()
    .reset_index()
)

fig_heatmap = px.density_heatmap(
    heatmap_grouped,
    x="year_month",
    y="beer",
    z="forecast_liters",
    color_continuous_scale="Blues",
    title="Monthly Demand Intensity per Beer"
)

st.plotly_chart(fig_heatmap, width="stretch")

# ==============================
# MONTHLY DEMAND VIEW
# ==============================

st.header("📊 Monthly Demand Overview")

monthly = forecast_df.groupby(["year", "month", "beer"])["forecast_liters"].sum().reset_index()

fig_month = px.bar(
    monthly,
    x="month",
    y="forecast_liters",
    color="beer",
    barmode="stack",
    title="Monthly Demand per Beer"
)

st.plotly_chart(fig_month, use_container_width=True)

# ==============================
# PRODUCTION CALENDAR
# ==============================

st.header("🏭 Production Schedule Calendar")

prod_calendar = production_df.groupby("production_week")["volume"].sum().reset_index()

fig_prod = px.bar(
    prod_calendar,
    x="production_week",
    y="volume",
    title="Weekly Brewing Volume"
)

st.plotly_chart(fig_prod, use_container_width=True)

# ==============================
# INTERACTIVE BREWING CALCULATOR
# ==============================

st.header("🧮 Brewing Requirement Calculator")

col1, col2, col3 = st.columns(3)

with col1:
    selected_beer = st.selectbox("Select Beer", forecast_df["beer"].unique())

with col2:
    selected_year = st.selectbox("Select Year", sorted(forecast_df["year"].unique()))

with col3:
    selected_month = st.selectbox("Select Month", range(1, 13))

filtered = forecast_df[
    (forecast_df["beer"] == selected_beer) &
    (forecast_df["year"] == selected_year) &
    (forecast_df["month"] == selected_month)
]

if not filtered.empty:

    monthly_demand = filtered["forecast_liters"].sum()

    # Calculate brewing recommendation
    large_tanks_needed = int(monthly_demand // LARGE_TANK)
    remainder = monthly_demand % LARGE_TANK

    small_tanks_needed = int(np.ceil(remainder / SMALL_TANK))

    total_brew_volume = large_tanks_needed * LARGE_TANK + small_tanks_needed * SMALL_TANK

    st.subheader(f"📦 Forecasted Sales for {selected_beer} in {selected_month}/{selected_year}")
    st.write(f"Expected Sales: **{monthly_demand:,.0f} liters**")

    st.subheader("🏭 Brewing Recommendation")

    st.write(f"6000L Tanks Needed: **{large_tanks_needed}**")
    st.write(f"2000L Tanks Needed: **{small_tanks_needed}**")
    st.write(f"Total Brew Volume: **{total_brew_volume:,.0f} liters**")

else:
    st.warning("No forecast data available for this selection.")