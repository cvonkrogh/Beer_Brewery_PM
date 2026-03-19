"""
Step 3 — Production planning from weekly demand forecast.

Fixes vs earlier version:
- Brew volumes are added to inventory only on *available_week* (after lead time),
  not the moment they are scheduled (this was inflating stock and hiding stockouts).
- Optional historical sales are used for (a) starting inventory and (b) spike weeks
  so we build extra cover before predictable seasonal peaks.
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd

# ==============================
# CONFIG
# ==============================

FORECAST_PATH = "data/processed/forecast_results.csv"
PROCESSED_DATA_PATH = "data/processed/weekly_beer_data.csv"
OUTPUT_PATH = "data/processed/production_schedule.csv"
MONTHLY_OUTPUT_PATH = "data/processed/production_schedule_monthly.csv"

LEAD_TIME = 12  # weeks until brewed beer is available to sell
SMALL_TANK = 2000
LARGE_TANK = 6000
SAFETY_FACTOR = 1.20  # base buffer on cumulative lookahead demand
SPIKE_RATIO_FORECAST = 1.45  # max/mean in lookahead window → treat as volatile
SPIKE_MULT_FORECAST = 1.25  # extra cover when forecast window is spiky
SPIKE_RATIO_HIST = 1.50  # forecast / historical median same week-of-year
SPIKE_MULT_HIST = 1.35  # extra cover when forecast >> typical same week in past
INITIAL_COVER_WEEKS = 2.0  # starting inventory ≈ this many weeks of recent avg demand

MAX_SCHEDULE_ROWS = 8000  # guardrail

# ==============================
# LOAD DATA
# ==============================


def load_forecast():
    df = pd.read_csv(FORECAST_PATH)
    df["week"] = pd.to_datetime(df["week"])
    return df


def load_processed_weekly_totals():
    """Beer-level weekly liters (sum containers) for history-driven planning."""
    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["week"])
    return (
        df.groupby(["beer", "week"], as_index=False)
        .agg(liters=("liters", "sum"))
        .sort_values(["beer", "week"])
    )


# ==============================
# PACKAGING STRATEGY
# ==============================


def choose_packaging_strategy(beer_df: pd.DataFrame) -> str:
    containers = [str(c) for c in beer_df["container"].dropna().unique()]

    has_cans = any("Can" in c for c in containers)
    has_bottles = any("Bottle" in c for c in containers)

    volume_by_container = (
        beer_df.groupby("container")["forecast_liters"]
        .sum()
        .sort_values(ascending=False)
    )

    if has_cans and not has_bottles:
        return "CANS_KEGS"
    if has_bottles and not has_cans:
        return "BOTTLES_KEGS"
    if volume_by_container.empty:
        return "CANS_KEGS"
    dominant = str(volume_by_container.index[0])
    if "Can" in dominant:
        return "CANS_KEGS"
    return "BOTTLES_KEGS"


# ==============================
# HISTORY HELPERS (spikes + starting stock)
# ==============================


def _history_series_for_beer(historical_weekly: pd.DataFrame | None, beer_name: str) -> pd.Series:
    if historical_weekly is None or historical_weekly.empty:
        return pd.Series(dtype=float)
    h = historical_weekly[historical_weekly["beer"] == beer_name].sort_values("week")
    if h.empty:
        return pd.Series(dtype=float)
    return h.set_index("week")["liters"].astype(float)


def initial_inventory_from_history(hist_series: pd.Series, forecast_demands: list[float]) -> float:
    """Rough opening stock: recent historical pace × INITIAL_COVER_WEEKS."""
    if hist_series is not None and len(hist_series) >= 4:
        m = float(hist_series.tail(8).mean())
        return max(m * INITIAL_COVER_WEEKS, 0.0)
    if forecast_demands:
        m = float(np.mean(forecast_demands[: min(8, len(forecast_demands))]))
        return max(m * INITIAL_COVER_WEEKS, 0.0)
    return 0.0


def forecast_window_spike_multiplier(demands: np.ndarray) -> float:
    """If the next lookahead window has a sharp peak, ask for more cover."""
    if demands.size == 0:
        return 1.0
    pos = demands[demands > 0]
    mean_w = float(np.mean(pos)) if pos.size else 1.0
    max_w = float(np.max(demands))
    if mean_w > 0 and max_w / mean_w >= SPIKE_RATIO_FORECAST:
        return SPIKE_MULT_FORECAST
    return 1.0


def historical_spike_multiplier(hist_series: pd.Series, week_ts: pd.Timestamp, forecast_demand: float) -> float:
    """If this week is usually much smaller in history, upcoming demand is a spike vs the past."""
    if hist_series is None or hist_series.empty:
        return 1.0
    woy = int(week_ts.isocalendar().week)
    vals = []
    for idx, val in hist_series.items():
        try:
            if int(pd.Timestamp(idx).isocalendar().week) == woy:
                vals.append(float(val))
        except Exception:
            continue
    if len(vals) < 3:
        return 1.0
    med = float(np.median(vals))
    if med > 0 and forecast_demand / med >= SPIKE_RATIO_HIST:
        return SPIKE_MULT_HIST
    return 1.0


# ==============================
# PRODUCTION PLANNING
# ==============================


def _ts(x) -> pd.Timestamp:
    return pd.Timestamp(x)


def _take_arrivals_for_week(arrivals: dict[pd.Timestamp, float], w: pd.Timestamp) -> float:
    """Pop all liters scheduled to arrive exactly on the same calendar day as `w`."""
    wn = _ts(w).normalize()
    acc = 0.0
    for k in list(arrivals.keys()):
        if _ts(k).normalize() == wn:
            acc += float(arrivals.pop(k))
    return acc


def _pipeline_arriving_within(arrivals: dict[pd.Timestamp, float], w_now: pd.Timestamp) -> float:
    """Beer scheduled to arrive strictly after `w_now` and within the next LEAD_TIME weeks."""
    t0 = _ts(w_now)
    t1 = t0 + pd.Timedelta(weeks=LEAD_TIME)
    return sum(float(v) for t, v in arrivals.items() if t0 < _ts(t) <= t1)


def plan_production(
    beer_df: pd.DataFrame,
    beer_name: str,
    historical_weekly: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build brew batches so finished beer lands on *available_week* = production_week + LEAD_TIME.

    `historical_weekly`: columns beer, week, liters (weekly totals). Used for starting inventory
    and spike detection vs same week-of-year in the past.

    Ordering rule: before consuming week *i*, compare (on-hand + beer scheduled to arrive in the
    next LEAD_TIME calendar weeks) to (lookahead demand × safety × spike factors). Schedule tank
    fills until position meets target; scheduled volume only hits inventory on *available_week*.
    """
    strategy = choose_packaging_strategy(beer_df)

    weekly_total = (
        beer_df.groupby("week")["forecast_liters"]
        .sum()
        .reset_index()
        .sort_values("week")
    )
    if weekly_total.empty:
        return pd.DataFrame()

    weeks = weekly_total["week"].tolist()
    demands = weekly_total["forecast_liters"].astype(float).tolist()
    n = len(weeks)
    hist_series = _history_series_for_beer(historical_weekly, beer_name)

    arrivals: dict[pd.Timestamp, float] = defaultdict(float)
    production_schedule: list[dict] = []

    inventory = initial_inventory_from_history(hist_series, demands)

    for i in range(n):
        w = weeks[i]
        d = demands[i]

        inventory += _take_arrivals_for_week(arrivals, w)

        # Lookahead demand in the next LEAD_TIME weeks (including this week)
        window = np.array(demands[i : min(i + LEAD_TIME, n)], dtype=float)
        base_target = float(window.sum()) * SAFETY_FACTOR

        m_forecast = forecast_window_spike_multiplier(window)
        m_hist = historical_spike_multiplier(hist_series, w, d)
        spike_mult = max(m_forecast, m_hist)

        target = base_target * spike_mult

        # Beer already scheduled to land in the next LEAD_TIME weeks (not yet in inventory)
        pipeline = _pipeline_arriving_within(arrivals, w)
        position = inventory + pipeline

        guard = 0
        while position < target and guard < MAX_SCHEDULE_ROWS:
            guard += 1
            if target - position > LARGE_TANK:
                brew_volume = LARGE_TANK
            else:
                brew_volume = SMALL_TANK

            available_week = w + pd.Timedelta(weeks=LEAD_TIME)
            arrivals[available_week] += brew_volume

            production_schedule.append(
                {
                    "beer": beer_name,
                    "production_week": w,
                    "available_week": available_week,
                    "volume": brew_volume,
                    "packaging_strategy": strategy,
                }
            )
            pipeline = _pipeline_arriving_within(arrivals, w)
            position = inventory + pipeline

        # Sell this week's demand
        inventory -= d
        if inventory < 0:
            inventory = 0.0

    return pd.DataFrame(production_schedule)


def summarize_monthly_production(schedule_df: pd.DataFrame) -> pd.DataFrame:
    if schedule_df.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "beer",
                "packaging_strategy",
                "planned_volume_liters",
                "planned_batches",
            ]
        )

    monthly = schedule_df.copy()
    monthly["month"] = monthly["production_week"].dt.to_period("M").dt.to_timestamp()

    monthly_summary = (
        monthly.groupby(["month", "beer", "packaging_strategy"], as_index=False)
        .agg(
            planned_volume_liters=("volume", "sum"),
            planned_batches=("volume", "count"),
        )
        .sort_values(["month", "beer"])
        .reset_index(drop=True)
    )

    return monthly_summary


# ==============================
# INVENTORY PROJECTION (for reporting / dashboard)
# ==============================


def weekly_forecast_totals(beer_df: pd.DataFrame) -> pd.DataFrame:
    return (
        beer_df.groupby("week")["forecast_liters"]
        .sum()
        .reset_index()
        .sort_values("week")
    )


def build_inventory_projection(
    beer_name: str,
    forecast_beer_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    historical_weekly: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Week-by-week: inventory after sales, arrivals from schedule, demand, weeks of cover.
    `weeks_cover` = inventory_end / next_week_demand (cap at 99).
    """
    wk = weekly_forecast_totals(forecast_beer_df)
    if wk.empty:
        return pd.DataFrame()

    weeks = wk["week"].tolist()
    demands = wk["forecast_liters"].astype(float).tolist()
    hist_series = _history_series_for_beer(historical_weekly, beer_name)
    inventory = initial_inventory_from_history(hist_series, demands)

    arrivals: dict[pd.Timestamp, float] = defaultdict(float)
    if schedule_df is not None and not schedule_df.empty:
        sub = schedule_df[schedule_df["beer"] == beer_name]
        for _, r in sub.iterrows():
            aw = pd.Timestamp(r["available_week"])
            arrivals[aw] += float(r["volume"])

    rows = []
    for i, w in enumerate(weeks):
        d = demands[i]
        inv_start = inventory + _take_arrivals_for_week(arrivals, w)
        inv_after = inv_start - d
        if inv_after < 0:
            inv_after = 0.0
        next_d = demands[i + 1] if i + 1 < len(demands) else d
        cover = min(99.0, inv_after / next_d) if next_d > 1e-6 else 99.0
        rows.append(
            {
                "week": w,
                "demand_liters": d,
                "inventory_start": inv_start,
                "inventory_end": inv_after,
                "weeks_cover": cover,
            }
        )
        inventory = inv_after

    return pd.DataFrame(rows)


def simulate_inventory_with_shortages(
    beer_name: str,
    forecast_beer_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    historical_weekly: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Same physics as `build_inventory_projection`, but records **shortages** when demand
    exceeds on-hand + arrivals (for Step 03b quality metrics). Does not hide stockouts.
    """
    wk = weekly_forecast_totals(forecast_beer_df)
    if wk.empty:
        return pd.DataFrame()

    weeks = wk["week"].tolist()
    demands = wk["forecast_liters"].astype(float).tolist()
    hist_series = _history_series_for_beer(historical_weekly, beer_name)
    inventory = initial_inventory_from_history(hist_series, demands)

    arrivals: dict[pd.Timestamp, float] = defaultdict(float)
    if schedule_df is not None and not schedule_df.empty:
        sub = schedule_df[schedule_df["beer"] == beer_name]
        for _, r in sub.iterrows():
            aw = pd.Timestamp(r["available_week"])
            arrivals[aw] += float(r["volume"])

    rows = []
    for i, w in enumerate(weeks):
        d = demands[i]
        inv_start = inventory + _take_arrivals_for_week(arrivals, w)
        raw_end = inv_start - d
        shortage = max(0.0, -raw_end)
        inv_after = max(0.0, raw_end)
        rows.append(
            {
                "week": w,
                "demand_liters": d,
                "inventory_start": inv_start,
                "inventory_end": inv_after,
                "shortage_liters": shortage,
                "stockout": shortage > 1e-6,
            }
        )
        inventory = inv_after

    return pd.DataFrame(rows)


def estimate_weeks_until_stockout_naive(forecast_beer_df: pd.DataFrame, historical_weekly: pd.DataFrame | None, beer_name: str) -> float | None:
    """
    If you never brewed again: weeks until starting inventory is depleted by forecast demand.
    """
    wk = weekly_forecast_totals(forecast_beer_df)
    if wk.empty:
        return None
    demands = wk["forecast_liters"].astype(float).tolist()
    hist_series = _history_series_for_beer(historical_weekly, beer_name)
    inv = initial_inventory_from_history(hist_series, demands)
    for i, d in enumerate(demands):
        inv -= d
        if inv <= 0:
            return float(i + 1)
    return float(len(demands))


# ==============================
# MAIN
# ==============================


def main():
    forecast_df = load_forecast()
    historical_weekly = load_processed_weekly_totals()

    results = []

    for beer in forecast_df["beer"].unique():
        print(f"Planning production for {beer}...")

        beer_df = forecast_df[forecast_df["beer"] == beer].copy()

        production = plan_production(beer_df, beer, historical_weekly=historical_weekly)
        results.append(production)

    final_schedule = pd.concat(results, ignore_index=True)

    os.makedirs("data/processed", exist_ok=True)
    final_schedule.to_csv(OUTPUT_PATH, index=False)

    monthly_schedule = summarize_monthly_production(final_schedule)
    monthly_schedule.to_csv(MONTHLY_OUTPUT_PATH, index=False)

    print("STEP 3 COMPLETE ✅")
    print("Weekly production schedule shape:", final_schedule.shape)
    print("Monthly summary shape:", monthly_schedule.shape)


if __name__ == "__main__":
    main()
