import pandas as pd
import plotly.express as px
import streamlit as st
import importlib.util
from pathlib import Path
import numpy as np

_eval_module_path = Path(__file__).with_name("02b_model_evaluation.py")
_eval_spec = importlib.util.spec_from_file_location("eval_shared_02b", _eval_module_path)
_eval_module = importlib.util.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(_eval_module)
evaluate_all_beers = _eval_module.evaluate_all_beers
BEER_CONTAINER_METRICS_CSV = getattr(
    _eval_module, "BEER_CONTAINER_METRICS_CSV", "data/processed/model_evaluation_beer_container.csv"
)

# Step 03 (filename starts with a digit → load by path)
_prod_module_path = Path(__file__).with_name("03_production_planning.py")
_prod_spec = importlib.util.spec_from_file_location("production_planning_03", _prod_module_path)
_prod_module = importlib.util.module_from_spec(_prod_spec)
_prod_spec.loader.exec_module(_prod_module)
plan_production = _prod_module.plan_production
summarize_monthly_production = _prod_module.summarize_monthly_production
build_inventory_projection = _prod_module.build_inventory_projection
estimate_weeks_until_stockout_naive = _prod_module.estimate_weeks_until_stockout_naive

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
    # Same shape as `03.load_processed_weekly_totals` — used for spike + starting inventory
    historical_weekly = (
        processed.groupby(["beer", "week"], as_index=False)
        .agg(liters=("liters", "sum"))
        .sort_values(["beer", "week"])
    )
    return forecast, processed, historical_weekly


@st.cache_data
def load_shared_evaluation(processed_df):
    metrics_df, plot_map, overall_df = evaluate_all_beers(
        processed_df=processed_df,
        focus_beers=FOCUS_BEERS,
        test_ratio=0.20,
        max_folds=5,
    )
    return metrics_df, plot_map, overall_df


@st.cache_data
def load_beer_container_metrics_csv(file_mtime: float):
    """`file_mtime` invalidates cache when the CSV is regenerated."""
    path = Path(__file__).resolve().parent / BEER_CONTAINER_METRICS_CSV
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


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


def build_production_schedule_from_forecast(
    forecast_df: pd.DataFrame, historical_weekly: pd.DataFrame
) -> pd.DataFrame:
    """
    Same logic as `python 03_production_planning.py`, using historical weekly totals
    for spike-aware targets and opening inventory.
    """
    parts = []
    for beer in forecast_df["beer"].unique():
        beer_df = forecast_df[forecast_df["beer"] == beer].copy()
        parts.append(plan_production(beer_df, beer, historical_weekly=historical_weekly))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_batch_container_split(
    schedule_df: pd.DataFrame, forecast_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split each brew batch over containers using forecast-driven shares.

    Primary basis: same beer + available_week container mix.
    Fallback basis: beer-level container mix over full horizon.
    """
    if schedule_df.empty:
        return schedule_df.copy(), pd.DataFrame()

    batches = schedule_df.copy().reset_index(drop=True)
    batches["production_week"] = pd.to_datetime(batches["production_week"])
    batches["available_week"] = pd.to_datetime(batches["available_week"])
    batches["batch_id"] = np.arange(1, len(batches) + 1)

    f = forecast_df.copy()
    f["week"] = pd.to_datetime(f["week"])

    by_week = (
        f.groupby(["beer", "week", "container"], as_index=False)
        .agg(forecast_liters=("forecast_liters", "sum"))
    )
    by_week_totals = (
        by_week.groupby(["beer", "week"], as_index=False)
        .agg(total_week_liters=("forecast_liters", "sum"))
    )
    week_mix = by_week.merge(by_week_totals, on=["beer", "week"], how="left")
    week_mix["share"] = (
        week_mix["forecast_liters"] / week_mix["total_week_liters"].replace(0, pd.NA)
    ).fillna(0.0)
    week_mix = week_mix.rename(columns={"week": "available_week"})

    by_beer = (
        f.groupby(["beer", "container"], as_index=False)
        .agg(total_liters=("forecast_liters", "sum"))
    )
    by_beer_totals = (
        by_beer.groupby("beer", as_index=False)
        .agg(beer_total_liters=("total_liters", "sum"))
    )
    fallback_mix = by_beer.merge(by_beer_totals, on="beer", how="left")
    fallback_mix["share"] = (
        fallback_mix["total_liters"] / fallback_mix["beer_total_liters"].replace(0, pd.NA)
    ).fillna(0.0)

    matched = batches.merge(
        week_mix[["beer", "available_week", "container", "share"]],
        on=["beer", "available_week"],
        how="inner",
    )
    matched["split_basis"] = "available_week_forecast_mix"

    matched_ids = set(matched["batch_id"].tolist())
    unmatched = batches[~batches["batch_id"].isin(matched_ids)].copy()

    fallback = unmatched.merge(
        fallback_mix[["beer", "container", "share"]],
        on="beer",
        how="left",
    )
    fallback["container"] = fallback["container"].fillna("Unknown")
    fallback["share"] = fallback["share"].fillna(1.0)
    fallback["split_basis"] = np.where(
        fallback["container"] == "Unknown",
        "no_forecast_container_data",
        "beer_horizon_container_mix",
    )

    alloc = pd.concat([matched, fallback], ignore_index=True)
    alloc["allocated_liters"] = alloc["volume"] * alloc["share"]
    alloc["share_pct"] = 100.0 * alloc["share"]

    # Build one-line split summary for each batch row.
    parts = alloc.sort_values(["batch_id", "allocated_liters"], ascending=[True, False]).copy()
    parts["split_piece"] = (
        parts["container"].astype(str)
        + " "
        + parts["share_pct"].round(1).map(lambda x: f"{x:.1f}%")
        + " ("
        + parts["allocated_liters"].round(0).astype(int).astype(str)
        + " L)"
    )
    split_summary = (
        parts.groupby("batch_id", as_index=False)
        .agg(container_split=("split_piece", " | ".join))
    )

    basis_summary = (
        alloc.groupby("batch_id", as_index=False)
        .agg(split_basis=("split_basis", "first"))
    )

    batches = batches.merge(split_summary, on="batch_id", how="left")
    batches = batches.merge(basis_summary, on="batch_id", how="left")

    alloc_cols = [
        "batch_id",
        "beer",
        "production_week",
        "available_week",
        "volume",
        "packaging_strategy",
        "container",
        "share_pct",
        "allocated_liters",
        "split_basis",
    ]
    alloc = alloc[alloc_cols].sort_values(["production_week", "beer", "batch_id", "container"])

    batch_cols = [
        "batch_id",
        "beer",
        "production_week",
        "available_week",
        "volume",
        "packaging_strategy",
        "split_basis",
        "container_split",
    ]
    batches = batches[batch_cols].sort_values(["production_week", "beer", "batch_id"])
    return batches, alloc


# ==============================
# APP
# ==============================

st.set_page_config(layout="wide")
st.title("Sales Forecast Performance Dashboard")
st.caption(
    "Sections 1–4: demand & forecast views (tank rules not applied). "
    "Section 5: operational brew plan from Step 03 logic (tank sizes, lead time, safety buffer)."
)

forecast_df, processed_df, historical_weekly = load_data()
metrics_df, plots_by_beer, overall_df = load_shared_evaluation(processed_df)
_bc_path = Path(__file__).resolve().parent / BEER_CONTAINER_METRICS_CSV
_bc_mtime = float(_bc_path.stat().st_mtime) if _bc_path.is_file() else -1.0
beer_container_metrics_df = load_beer_container_metrics_csv(_bc_mtime)
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

st.subheader("Model test metrics by beer and container")
st.caption(
    "Rolling CV error metrics (MAE, RMSE, SMAPE) per beer–container series — same method as section 1. "
    f"Regenerate: `python 02b_model_evaluation.py` → `{BEER_CONTAINER_METRICS_CSV}`."
)
if beer_container_metrics_df.empty:
    st.info(
        "No beer+container metrics file found. Run `python 02b_model_evaluation.py` to create "
        f"`{BEER_CONTAINER_METRICS_CSV}`."
    )
else:
    if selected_beer == ALL_BEERS_OPTION:
        display_bc = beer_container_metrics_df.copy()
    else:
        display_bc = beer_container_metrics_df[
            beer_container_metrics_df["Beer"] == selected_beer
        ].copy()
    if display_bc.empty:
        st.warning("No rows for the selected beer in the beer+container metrics file.")
    else:
        st.dataframe(display_bc, width="stretch", hide_index=True)

st.header("5) Production plan & stock cover")

schedule_df = build_production_schedule_from_forecast(forecast_df, historical_weekly)
if schedule_df.empty:
    st.info("No production rows (check `forecast_results.csv` and focus beers).")
else:
    beers_sel = _selected_beers(selected_beer)

    # --- Stock risk summary (single beer clearer)
    if selected_beer != ALL_BEERS_OPTION:
        bdf = forecast_df[forecast_df["beer"] == selected_beer].copy()
        wk_naive = estimate_weeks_until_stockout_naive(bdf, historical_weekly, selected_beer)
        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                "Naive weeks until stockout (no more brewing)",
                f"{wk_naive:.1f}" if wk_naive is not None else "—",
                help="Starts from modelled opening stock, subtracts forecast demand week by week.",
            )
        with c2:
            st.metric(
                "Brew batches in plan (selected beer)",
                int(len(schedule_df[schedule_df["beer"] == selected_beer])),
            )

    # --- Inventory + cover chart
    inv_parts = []
    for b in beers_sel:
        b_fore = forecast_df[forecast_df["beer"] == b].copy()
        if b_fore.empty:
            continue
        proj = build_inventory_projection(b, b_fore, schedule_df, historical_weekly)
        if proj.empty:
            continue
        proj["beer"] = b
        inv_parts.append(proj)
    if inv_parts:
        inv_df = pd.concat(inv_parts, ignore_index=True)
        if selected_beer == ALL_BEERS_OPTION:
            fig_inv = px.line(
                inv_df,
                x="week",
                y="inventory_end",
                color="beer",
                title="Projected inventory after weekly sales (with scheduled arrivals)",
                labels={"inventory_end": "Inventory (L)", "week": "Week"},
            )
            fig_cov = px.line(
                inv_df,
                x="week",
                y="weeks_cover",
                color="beer",
                title="Weeks of cover (inventory ÷ next week demand)",
                labels={"weeks_cover": "Weeks", "week": "Week"},
            )
        else:
            one = inv_df[inv_df["beer"] == selected_beer]
            fig_inv = px.line(
                one,
                x="week",
                y="inventory_end",
                title=f"Projected inventory — {selected_beer}",
                labels={"inventory_end": "Inventory (L)", "week": "Week"},
            )
            fig_demand = px.line(
                one,
                x="week",
                y="demand_liters",
                title=f"Forecast demand (same horizon) — {selected_beer}",
                labels={"demand_liters": "Liters", "week": "Week"},
            )
            fig_cov = px.line(
                one,
                x="week",
                y="weeks_cover",
                title=f"Weeks of cover — {selected_beer}",
                labels={"weeks_cover": "Weeks", "week": "Week"},
            )
            st.plotly_chart(fig_demand, width="stretch")

        st.plotly_chart(fig_inv, width="stretch")
        st.plotly_chart(fig_cov, width="stretch")
    else:
        st.warning("Could not build inventory projection for the current selection.")

    st.subheader("Planned brew volume by month")
    monthly_plan = summarize_monthly_production(schedule_df)
    monthly_view = monthly_plan[monthly_plan["beer"].isin(beers_sel)].copy()

    if monthly_view.empty:
        st.warning("No monthly rows for the current beer filter.")
    else:
        common_kwargs = dict(
            x="month",
            y="planned_volume_liters",
            title=f"Liters scheduled to be brewed (batch starts) — {selected_beer}",
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

    sched_view = schedule_df[schedule_df["beer"].isin(beers_sel)].copy()
    batch_view, split_view = build_batch_container_split(sched_view, forecast_df)

    st.subheader("Planner action list")
    actions = batch_view.copy()
    actions["is_small_batch"] = (actions["volume"] == 2000).astype(int)
    actions["is_large_batch"] = (actions["volume"] == 6000).astype(int)
    brew_actions = (
        actions.groupby(["production_week", "beer"], as_index=False)
        .agg(
            total_brew_liters=("volume", "sum"),
            total_batches=("batch_id", "count"),
            batches_2000l=("is_small_batch", "sum"),
            batches_6000l=("is_large_batch", "sum"),
        )
        .sort_values(["production_week", "beer"])
    )
    st.caption("Weekly brew starts by beer. This is the direct brew instruction view.")
    st.dataframe(brew_actions, width="stretch", hide_index=True)

    container_plan = (
        split_view.groupby(["available_week", "beer", "container"], as_index=False)
        .agg(planned_container_liters=("allocated_liters", "sum"))
        .sort_values(["available_week", "beer", "container"])
    )
    st.caption(
        "Container split plan on available weeks, allocated from forecast container mix."
    )
    st.dataframe(container_plan, width="stretch", hide_index=True)

    with st.expander("Raw brew batch list (production week → available week)"):
        st.caption(
            "Each batch now includes a forecast-based container split (share and liters)."
        )
        st.dataframe(
            batch_view,
            width="stretch",
            hide_index=True,
        )

    with st.expander("Batch-to-container split detail"):
        st.caption(
            "One row per batch x container allocation. This is the exact split assumption used."
        )
        st.dataframe(
            split_view,
            width="stretch",
            hide_index=True,
        )

    with st.expander("Monthly summary table"):
        st.dataframe(monthly_view, width="stretch", hide_index=True)

    st.caption(
        "Save schedules to CSV: `python 03_production_planning.py` → "
        "`production_schedule.csv` & `production_schedule_monthly.csv`."
    )