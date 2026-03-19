import pandas as pd
import plotly.express as px
import streamlit as st
import importlib.util
from pathlib import Path

_eval_module_path = Path(__file__).with_name("02b_model_evaluation.py")
_eval_spec = importlib.util.spec_from_file_location("eval_shared_02b", _eval_module_path)
_eval_module = importlib.util.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(_eval_module)
evaluate_all_beers = _eval_module.evaluate_all_beers

# Step 03 (filename starts with a digit → load by path)
_prod_module_path = Path(__file__).with_name("03_production_planning.py")
_prod_spec = importlib.util.spec_from_file_location("production_planning_03", _prod_module_path)
_prod_module = importlib.util.module_from_spec(_prod_spec)
_prod_spec.loader.exec_module(_prod_module)
plan_production = _prod_module.plan_production
summarize_monthly_production = _prod_module.summarize_monthly_production

# ==============================
# CONFIG
# ==============================

FORECAST_PATH = "data/processed/forecast_results.csv"
PROCESSED_DATA_PATH = "data/processed/weekly_beer_data.csv"

FOCUS_BEERS = [
    "Hoop Bleke Nelis",
    "Hoop Lager",
    "Hoop Kaper Tropical IPA",
]
ALL_BEERS_OPTION = "All beers"

LAGS = [1, 2, 3, 4, 8, 12, 26, 52]
TEST_WEEKS = 12


@st.cache_data
def load_data():
    forecast = pd.read_csv(FORECAST_PATH, parse_dates=["week"])
    forecast = forecast[forecast["beer"].isin(FOCUS_BEERS)].copy()
    processed = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["week"])
    processed = processed[processed["beer"].isin(FOCUS_BEERS)].copy()
    return forecast, processed


@st.cache_data
def load_shared_evaluation(processed_df):
    metrics_df, plot_map, overall_df = evaluate_all_beers(
        processed_df=processed_df,
        focus_beers=FOCUS_BEERS,
        test_ratio=0.20,
        max_folds=5,
    )
    return metrics_df, plot_map, overall_df


def _selected_beers(selected_beer):
    if selected_beer == ALL_BEERS_OPTION:
        return FOCUS_BEERS
    return [selected_beer]


def build_monthly_need_chart(forecast_df, selected_beer, horizon_months):
    monthly = forecast_df[forecast_df["beer"].isin(_selected_beers(selected_beer))].copy()
    monthly["month"] = monthly["week"].dt.to_period("M").dt.to_timestamp()

    start_month = monthly["month"].min()
    target_months = pd.date_range(start=start_month, periods=horizon_months, freq="MS")

    monthly = monthly[monthly["month"].isin(target_months)].copy()
    monthly_need = (
        monthly.groupby(["month", "beer"], as_index=False)
        .agg(liters_needed=("forecast_liters", "sum"))
        .sort_values("month")
    )

    if selected_beer == ALL_BEERS_OPTION:
        fig = px.bar(
            monthly_need,
            x="month",
            y="liters_needed",
            color="beer",
            barmode="stack",
            title=f"Monthly Beer Needed - All Beers ({horizon_months} months)",
            labels={"month": "Month", "liters_needed": "Liters Needed"},
            text="liters_needed",
        )
        fig.update_traces(texttemplate="%{text:.0f}", textposition="inside")
    else:
        fig = px.bar(
            monthly_need.groupby("month", as_index=False)["liters_needed"].sum(),
            x="month",
            y="liters_needed",
            title=f"Monthly Beer Needed - {selected_beer} ({horizon_months} months)",
            labels={"month": "Month", "liters_needed": "Liters Needed"},
            text_auto=".0f",
        )
    fig.update_layout(xaxis_tickformat="%b")

    return fig, monthly_need, target_months


def build_historical_average_chart(processed_df, selected_beer, target_months):
    hist = processed_df[processed_df["beer"].isin(_selected_beers(selected_beer))].copy()
    hist["month_date"] = hist["week"].dt.to_period("M").dt.to_timestamp()
    hist["month_num"] = hist["month_date"].dt.month
    hist["year_num"] = hist["month_date"].dt.year

    # First aggregate to monthly sales per beer/year, then average by month-of-year.
    hist_monthly = (
        hist.groupby(["beer", "year_num", "month_num"], as_index=False)
        .agg(monthly_liters=("liters", "sum"))
    )
    month_avg = (
        hist_monthly.groupby(["beer", "month_num"], as_index=False)
        .agg(avg_liters=("monthly_liters", "mean"))
    )

    horizon_df = pd.DataFrame({"month": target_months})
    horizon_df["month_num"] = horizon_df["month"].dt.month

    if selected_beer == ALL_BEERS_OPTION:
        beer_df = pd.DataFrame({"beer": FOCUS_BEERS})
        horizon_beer = horizon_df.merge(beer_df, how="cross")
        aligned = (
            horizon_beer.merge(month_avg, on=["beer", "month_num"], how="left")
            .fillna({"avg_liters": 0.0})
            .sort_values(["month", "beer"])
        )
        fig = px.bar(
            aligned,
            x="month",
            y="avg_liters",
            color="beer",
            barmode="stack",
            title="Average Past Monthly Sales - All Beers (aligned to selected forecast window)",
            labels={"month": "Month", "avg_liters": "Average Historical Liters"},
            text="avg_liters",
        )
        fig.update_traces(texttemplate="%{text:.0f}", textposition="inside")
    else:
        aligned = (
            horizon_df.assign(beer=selected_beer)
            .merge(month_avg, on=["beer", "month_num"], how="left")
            .fillna({"avg_liters": 0.0})
            .sort_values("month")
        )
        fig = px.bar(
            aligned,
            x="month",
            y="avg_liters",
            title=f"Average Past Monthly Sales - {selected_beer} (aligned to selected forecast window)",
            labels={"month": "Month", "avg_liters": "Average Historical Liters"},
            text_auto=".0f",
        )

    fig.update_layout(xaxis_tickformat="%b")
    return fig, aligned


def build_container_need_chart(forecast_df, selected_beer, target_months):
    scope = forecast_df[forecast_df["beer"].isin(_selected_beers(selected_beer))].copy()
    scope["month"] = scope["week"].dt.to_period("M").dt.to_timestamp()
    scope = scope[scope["month"].isin(target_months)].copy()

    container_need = (
        scope.groupby(["month", "container"], as_index=False)
        .agg(liters_needed=("forecast_liters", "sum"))
        .sort_values(["month", "container"])
    )

    fig = px.bar(
        container_need,
        x="month",
        y="liters_needed",
        color="container",
        barmode="stack",
        title=f"Monthly Container Liters Needed - {selected_beer}",
        labels={"month": "Month", "liters_needed": "Liters Needed"},
        text="liters_needed",
    )
    fig.update_traces(texttemplate="%{text:.0f}", textposition="inside")
    fig.update_layout(xaxis_tickformat="%b")

    return fig, container_need


def build_production_schedule_from_forecast(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Same logic as `python 03_production_planning.py`, but uses the forecast already
    loaded in the app (so the dashboard stays in sync with the CSV you view above).
    """
    parts = []
    for beer in forecast_df["beer"].unique():
        beer_df = forecast_df[forecast_df["beer"] == beer].copy()
        parts.append(plan_production(beer_df, beer))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# ==============================
# APP
# ==============================

st.set_page_config(layout="wide")
st.title("Sales Forecast Performance Dashboard")
st.caption(
    "Sections 1–4: demand & forecast views (tank rules not applied). "
    "Section 5: operational brew plan from Step 03 logic (tank sizes, lead time, safety buffer)."
)

forecast_df, processed_df = load_data()
metrics_df, plots_by_beer, overall_df = load_shared_evaluation(processed_df)
selected_beer = st.selectbox("Select beer", [ALL_BEERS_OPTION] + FOCUS_BEERS, index=0)
horizon_months = st.slider(
    "Forecast horizon to display (months)",
    min_value=3,
    max_value=12,
    value=12,
    step=1,
)

st.header("1) Test Results of Sales Prediction Model (Per Beer)")

if metrics_df.empty:
    st.warning("Not enough data to calculate test metrics.")
else:
    metrics_by_beer = {row["Beer"]: row for _, row in metrics_df.iterrows()}
    if selected_beer == ALL_BEERS_OPTION:
        display_metrics = metrics_df.copy()
        if not overall_df.empty:
            display_metrics = pd.concat([display_metrics, overall_df], ignore_index=True)
    else:
        display_metrics = pd.DataFrame([metrics_by_beer[selected_beer]])
        if not overall_df.empty:
            display_metrics = pd.concat([display_metrics, overall_df], ignore_index=True)

    st.dataframe(display_metrics, width="stretch")
    st.caption("Shared evaluation method: rolling time-series CV with 80% train / 20% test windows.")

    if selected_beer == ALL_BEERS_OPTION:
        combined_parts = []
        for beer in FOCUS_BEERS:
            if beer not in plots_by_beer:
                continue
            beer_plot = plots_by_beer[beer].copy()
            beer_plot["Beer"] = beer_plot["beer"]
            melted = beer_plot.melt(
                id_vars=["week", "Beer"],
                value_vars=["actual", "predicted"],
                var_name="Series",
                value_name="Liters",
            )
            melted["Series"] = melted["Series"].str.capitalize()
            combined_parts.append(melted)

        if combined_parts:
            combined_df = pd.concat(combined_parts, ignore_index=True).sort_values(["Beer", "week"])

            fig_all = px.line(
                combined_df,
                x="week",
                y="Liters",
                color="Beer",
                line_dash="Series",
                title="All Beers - Actual vs Predicted (Most Recent CV Test Window)",
                markers=True,
            )
            fig_all.update_layout(xaxis_title="Week", yaxis_title="Liters")
            st.plotly_chart(fig_all, width="stretch")
    else:
        beer_plot = plots_by_beer[selected_beer]
        beer_plot["Beer"] = beer_plot["beer"]
        melted = beer_plot.melt(
            id_vars=["week", "Beer"],
            value_vars=["actual", "predicted"],
            var_name="Series",
            value_name="Liters",
        )
        melted["Series"] = melted["Series"].str.capitalize()
        fig = px.line(
            melted,
            x="week",
            y="Liters",
            color="Series",
            title=f"{selected_beer} - Actual vs Predicted (Most Recent CV Test Window)",
            markers=True,
        )
        fig.update_layout(xaxis_title="Week", yaxis_title="Liters")
        st.plotly_chart(fig, width="stretch")

st.header("2) Monthly Beer Needed (Forecast Horizon)")
fig_monthly, monthly_need, target_months = build_monthly_need_chart(
    forecast_df,
    selected_beer,
    horizon_months=horizon_months,
)
if monthly_need.empty:
    st.warning("No forecast months available for the selected beer.")
else:
    st.caption(
        f"Showing {horizon_months} predicted month(s): "
        f"{target_months.min().strftime('%b %Y')} to {target_months.max().strftime('%b %Y')}."
    )
    st.plotly_chart(fig_monthly, width="stretch")

st.header("3) Average Past Monthly Sales (Historical Baseline)")
fig_hist, hist_aligned = build_historical_average_chart(processed_df, selected_beer, target_months)
if hist_aligned.empty:
    st.warning("No historical data available to compute monthly averages.")
else:
    st.caption(
        "Historical average is computed by month-of-year and aligned to the selected forecast window."
    )
    st.plotly_chart(fig_hist, width="stretch")

st.header("4) Monthly Container Demand")
fig_container, container_need = build_container_need_chart(forecast_df, selected_beer, target_months)
if container_need.empty:
    st.warning("No container forecast data available for the selected filter and horizon.")
else:
    st.caption("Shows liters needed per container for each displayed month.")
    st.plotly_chart(fig_container, width="stretch")

st.header("5) Production plan (Step 03)")
st.caption(
    "**Why here, not in the model?** Step 02 only predicts *liters demanded*. Step 03 turns that into "
    "*when to brew* and *how much per batch* using tank sizes (2k / 6k L), a 12-week lead time, and a "
    "20% safety factor — that is operations logic, not machine learning. Showing it in the dashboard "
    "links forecast → executable plan for the same beer selection."
)
schedule_df = build_production_schedule_from_forecast(forecast_df)
if schedule_df.empty:
    st.info("No production rows (check `forecast_results.csv` and focus beers).")
else:
    monthly_plan = summarize_monthly_production(schedule_df)
    beers_sel = _selected_beers(selected_beer)
    sched_view = schedule_df[schedule_df["beer"].isin(beers_sel)].copy()
    monthly_view = monthly_plan[monthly_plan["beer"].isin(beers_sel)].copy()

    st.subheader("Brew batches (weekly detail)")
    st.dataframe(
        sched_view.sort_values(["production_week", "beer"]),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Planned brew volume by month")
    if monthly_view.empty:
        st.warning("No monthly rows for the current beer filter.")
    else:
        common_kwargs = dict(
            x="month",
            y="planned_volume_liters",
            title=(
                "Sum of batch volumes scheduled in each month — "
                f"{selected_beer}"
            ),
            labels={
                "month": "Month",
                "planned_volume_liters": "Planned liters (sum of batches)",
                "beer": "Beer",
            },
            text="planned_volume_liters",
        )
        if selected_beer == ALL_BEERS_OPTION:
            fig_plan = px.bar(
                monthly_view,
                color="beer",
                barmode="group",
                **common_kwargs,
            )
            fig_plan.update_layout(showlegend=True)
        else:
            fig_plan = px.bar(monthly_view, **common_kwargs)
            fig_plan.update_layout(showlegend=False)
        fig_plan.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig_plan.update_layout(xaxis_tickformat="%b")
        st.plotly_chart(fig_plan, width="stretch")

        with st.expander("Monthly table (includes packaging strategy)"):
            st.dataframe(monthly_view, width="stretch", hide_index=True)

    st.caption(
        "To persist these tables to disk, run `python 03_production_planning.py` "
        "(writes `production_schedule.csv` and `production_schedule_monthly.csv` from the forecast file)."
    )