import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# ==============================
# CONFIG
# ==============================

FORECAST_PATH = "data/processed/forecast_results.csv"
PRODUCTION_PATH = "data/processed/production_schedule.csv"
EVENTS_PATH = "data/events.csv"

# ==============================
# LOAD DATA
# ==============================

@st.cache_data
def load_data():
    forecast = pd.read_csv(FORECAST_PATH)
    production = pd.read_csv(PRODUCTION_PATH)
    events = pd.read_csv(EVENTS_PATH)

    forecast["week"] = pd.to_datetime(forecast["week"])
    production["production_week"] = pd.to_datetime(production["production_week"])

    return forecast, production, events


forecast_df, production_df, events_df = load_data()

st.set_page_config(layout="wide")
st.title("🍺 Brewery Operational Planning Cockpit")

# ======================================================
# DEFAULT WINDOW = NEXT 52 WEEKS
# ======================================================

today = pd.Timestamp.today().normalize()
end_date = today + pd.Timedelta(weeks=52)


def expand_event_dates(events, start, end):
    """Expand recurring month/day events to concrete dates within the date range."""
    rows = []
    for year in range(start.year, end.year + 1):
        for _, row in events.iterrows():
            try:
                dt = pd.Timestamp(year=year, month=int(row["month"]), day=int(row["day"]))
                if start <= dt <= end:
                    rows.append({"date": dt, "event": row["Event_Name"]})
            except ValueError:
                continue
    return pd.DataFrame(rows)


event_dates_df = expand_event_dates(events_df, today, end_date)

forecast_next_year = forecast_df[
    (forecast_df["week"] >= today) &
    (forecast_df["week"] <= end_date)
]

production_next_year = production_df[
    (production_df["production_week"] >= today) &
    (production_df["production_week"] <= end_date)
]

# ======================================================
# 1️⃣ IMMEDIATE BREWING ACTION BOARD
# ======================================================

st.header("🚨 Immediate Brewing Actions")

next_brews = (
    production_next_year
    .sort_values("production_week")
    .groupby("beer")
    .first()
    .reset_index()
)

if next_brews.empty:
    st.success("No brewing required in next 52 weeks.")
else:
    next_brews["Weeks Until Brew"] = (
        (next_brews["production_week"] - today).dt.days / 7
    ).round(1)

    st.dataframe(
        next_brews[[
            "beer",
            "production_week",
            "volume",
            "packaging_strategy",
            "Weeks Until Brew"
        ]],
        width="stretch"
    )

# ======================================================
# 2️⃣ BREWING CALENDAR (YEAR VIEW)
# ======================================================

st.header("📅 Annual Brewing Calendar")

if not production_next_year.empty:

    fig_calendar = px.scatter(
        production_next_year,
        x="production_week",
        y="beer",
        size="volume",
        color="packaging_strategy",
        title="Brewing Start Weeks",
        hover_data=["volume"]
    )

    if not event_dates_df.empty:
        for _, evt in event_dates_df.drop_duplicates(subset=["date", "event"]).iterrows():
            fig_calendar.add_vline(
                x=evt["date"],
                line_dash="dot",
                line_color="rgba(255,165,0,0.5)",
                annotation_text=evt["event"],
                annotation_position="top",
                annotation_font_size=9,
                annotation_textangle=-45,
            )

    fig_calendar.update_layout(
        yaxis_title="Beer",
        xaxis_title="Week",
        height=500
    )

    st.plotly_chart(fig_calendar, width="stretch")

else:
    st.info("No brewing scheduled.")

# ======================================================
# 3️⃣ NEXT 52 WEEK SALES FORECAST (ALL BEERS)
# ======================================================

st.header("📈 Next 52 Week Sales Forecast (Liters)")

beer_weekly_total = (
    forecast_next_year.groupby(["week", "beer"])["forecast_liters"]
    .sum()
    .reset_index()
)

fig_forecast = px.line(
    beer_weekly_total,
    x="week",
    y="forecast_liters",
    color="beer",
    title="Weekly Sales Forecast per Beer"
)

if not event_dates_df.empty:
    for _, evt in event_dates_df.drop_duplicates(subset=["date", "event"]).iterrows():
        fig_forecast.add_vline(
            x=evt["date"],
            line_dash="dot",
            line_color="rgba(255,165,0,0.6)",
            annotation_text=evt["event"],
            annotation_position="top",
            annotation_font_size=9,
            annotation_textangle=-45,
        )

st.plotly_chart(fig_forecast, width="stretch")

# ======================================================
# 4️⃣ CONTAINER MIX OVERVIEW
# ======================================================

st.header("📦 Weekly Container Demand")

container_weekly = (
    forecast_next_year.groupby(["week", "container"])["forecast_liters"]
    .sum()
    .reset_index()
)

fig_container = px.bar(
    container_weekly,
    x="week",
    y="forecast_liters",
    color="container",
    title="Weekly Container Mix",
    barmode="stack"
)

st.plotly_chart(fig_container, width="stretch")