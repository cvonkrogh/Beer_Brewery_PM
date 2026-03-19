"""
Step 03b — Evaluate production planning (Step 03) quality.

What this is for
----------------
- **Not** ML “accuracy” (that’s 02 / 02b on demand forecasts).
- **Operational** quality of the brew plan: stockouts vs forecast, lead-time discipline,
  valid batch sizes, internal consistency, and how much production exceeds demand (safety).

Uses the **same** opening inventory and arrival logic as `03_production_planning.py`
(via `simulate_inventory_with_shortages`), so results match the dashboard inventory path.

Run after: `python 02_forecasting_model.py` then `python 03_production_planning.py`
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

_mod_path = ROOT / "03_production_planning.py"
_spec = importlib.util.spec_from_file_location("production_planning_03", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

LEAD_TIME_WEEKS = _mod.LEAD_TIME
SMALL_TANK = _mod.SMALL_TANK
LARGE_TANK = _mod.LARGE_TANK
TANK_SET = {SMALL_TANK, LARGE_TANK}
simulate_inventory_with_shortages = _mod.simulate_inventory_with_shortages
load_processed_weekly_totals = _mod.load_processed_weekly_totals

FORECAST_PATH = str(ROOT / "data/processed/forecast_results.csv")
PRODUCTION_WEEKLY_PATH = str(ROOT / "data/processed/production_schedule.csv")
PRODUCTION_MONTHLY_PATH = str(ROOT / "data/processed/production_schedule_monthly.csv")
PROCESSED_PATH = str(ROOT / "data/processed/weekly_beer_data.csv")


def load_data():
    forecast = pd.read_csv(FORECAST_PATH, parse_dates=["week"])
    prod_weekly = pd.read_csv(
        PRODUCTION_WEEKLY_PATH,
        parse_dates=["production_week", "available_week"],
    )
    prod_monthly = pd.read_csv(PRODUCTION_MONTHLY_PATH, parse_dates=["month"])
    historical_weekly = load_processed_weekly_totals()
    return forecast, prod_weekly, prod_monthly, historical_weekly


def planning_service_metrics(forecast: pd.DataFrame, prod_weekly: pd.DataFrame, historical_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Per-beer: weeks with shortage, total shortage liters, service level (% of weeks fully covered).
    Aligned with Step 03 inventory simulation (including historical opening stock).
    """
    rows = []
    for beer in sorted(forecast["beer"].unique()):
        fbeer = forecast[forecast["beer"] == beer].copy()
        sim = simulate_inventory_with_shortages(beer, fbeer, prod_weekly, historical_weekly)
        if sim.empty:
            rows.append(
                {
                    "beer": beer,
                    "horizon_weeks": 0,
                    "weeks_with_shortage": 0,
                    "total_shortage_liters": 0.0,
                    "service_level_pct": 100.0,
                    "max_single_week_shortage": 0.0,
                }
            )
            continue
        n = len(sim)
        w_short = int(sim["stockout"].sum())
        tot_short = float(sim["shortage_liters"].sum())
        max_short = float(sim["shortage_liters"].max())
        svc = 100.0 * (1.0 - w_short / n) if n else 100.0
        rows.append(
            {
                "beer": beer,
                "horizon_weeks": n,
                "weeks_with_shortage": w_short,
                "total_shortage_liters": round(tot_short, 2),
                "service_level_pct": round(svc, 2),
                "max_single_week_shortage": round(max_short, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("beer").reset_index(drop=True)


def check_lead_time(prod_weekly: pd.DataFrame) -> pd.DataFrame:
    """Available week must equal production week + LEAD_TIME (calendar)."""
    pw = pd.to_datetime(prod_weekly["production_week"])
    aw = pd.to_datetime(prod_weekly["available_week"])
    expected = pw + pd.Timedelta(weeks=LEAD_TIME_WEEKS)
    bad = aw.dt.normalize() != expected.dt.normalize()
    return prod_weekly[bad].copy()


def check_tank_sizes(prod_weekly: pd.DataFrame):
    invalid = prod_weekly[~prod_weekly["volume"].isin(TANK_SET)].copy()
    summary = (
        prod_weekly["volume"]
        .value_counts()
        .rename_axis("tank_volume")
        .reset_index(name="batch_count")
        .sort_values("tank_volume")
    )
    return invalid, summary


def check_monthly_consistency(prod_weekly: pd.DataFrame, prod_monthly: pd.DataFrame):
    weekly_rollup = prod_weekly.copy()
    weekly_rollup["month"] = weekly_rollup["production_week"].dt.to_period("M").dt.to_timestamp()
    weekly_rollup = (
        weekly_rollup.groupby(["month", "beer", "packaging_strategy"], as_index=False)
        .agg(
            planned_volume_liters=("volume", "sum"),
            planned_batches=("volume", "count"),
        )
    )

    cmp_df = weekly_rollup.merge(
        prod_monthly,
        on=["month", "beer", "packaging_strategy"],
        how="outer",
        suffixes=("_from_weekly", "_from_monthly"),
    ).fillna(0)

    cmp_df["volume_diff"] = (
        cmp_df["planned_volume_liters_from_weekly"] - cmp_df["planned_volume_liters_from_monthly"]
    )
    cmp_df["batches_diff"] = (
        cmp_df["planned_batches_from_weekly"] - cmp_df["planned_batches_from_monthly"]
    )

    mismatches = cmp_df[(cmp_df["volume_diff"] != 0) | (cmp_df["batches_diff"] != 0)].copy()
    return mismatches, cmp_df


def reconcile_forecast_vs_monthly_production(forecast: pd.DataFrame, prod_monthly: pd.DataFrame) -> pd.DataFrame:
    monthly_forecast = forecast.copy()
    monthly_forecast["month"] = monthly_forecast["week"].dt.to_period("M").dt.to_timestamp()
    monthly_forecast = (
        monthly_forecast.groupby(["month", "beer"], as_index=False)
        .agg(forecast_liters=("forecast_liters", "sum"))
    )

    monthly_prod = (
        prod_monthly.groupby(["month", "beer"], as_index=False)
        .agg(planned_volume_liters=("planned_volume_liters", "sum"))
    )

    rec = monthly_forecast.merge(monthly_prod, on=["month", "beer"], how="outer").fillna(0)
    rec["volume_gap_liters"] = rec["planned_volume_liters"] - rec["forecast_liters"]
    rec["coverage_ratio"] = rec["planned_volume_liters"] / rec["forecast_liters"].replace(0, pd.NA)

    return rec.sort_values(["month", "beer"]).reset_index(drop=True)


def overall_planning_score(service_df: pd.DataFrame) -> dict:
    """Single-line summary for quick reading."""
    if service_df.empty:
        return {"mean_service_level_pct": None, "any_shortage": None, "total_shortage_liters": None}
    return {
        "mean_service_level_pct": round(float(service_df["service_level_pct"].mean()), 2),
        "any_shortage": bool((service_df["weeks_with_shortage"] > 0).any()),
        "total_shortage_liters": round(float(service_df["total_shortage_liters"].sum()), 2),
    }


def main():
    if not os.path.isfile(FORECAST_PATH):
        raise FileNotFoundError(f"Missing {FORECAST_PATH} — run 02_forecasting_model.py first.")
    if not os.path.isfile(PRODUCTION_WEEKLY_PATH):
        raise FileNotFoundError(f"Missing {PRODUCTION_WEEKLY_PATH} — run 03_production_planning.py first.")

    forecast, prod_weekly, prod_monthly, historical_weekly = load_data()

    service_metrics = planning_service_metrics(forecast, prod_weekly, historical_weekly)
    lead_time_violations = check_lead_time(prod_weekly)
    invalid_tanks, tank_summary = check_tank_sizes(prod_weekly)
    monthly_mismatches, monthly_compare = check_monthly_consistency(prod_weekly, prod_monthly)
    reconciliation = reconcile_forecast_vs_monthly_production(forecast, prod_monthly)
    score = overall_planning_score(service_metrics)

    print("\n==========================================")
    print("STEP 03b — PRODUCTION PLAN QUALITY (vs forecast + rules)")
    print("==========================================")
    print(
        "\nThis measures whether the brew **plan** keeps up with **forecast demand**, "
        "using the same opening stock as Step 03 (from history).\n"
        "It is NOT the same as forecast model MAE (see 02b / dashboard section 1).\n"
    )

    print("\n0) SERVICE LEVEL & SHORTAGES (inventory sim = Step 03 logic)")
    print(service_metrics.to_string(index=False))
    print(f"\n  → Mean service level across beers: {score['mean_service_level_pct']}%")
    print(f"  → Any week with shortage: {score['any_shortage']}")
    print(f"  → Total shortage volume (L): {score['total_shortage_liters']}")

    print("\n1) LEAD-TIME CHECK (production_week + 12w == available_week)")
    if lead_time_violations.empty:
        print("PASS: no lead-time violations.")
    else:
        print("FAIL: lead-time violations found:")
        print(
            lead_time_violations[
                ["beer", "production_week", "available_week", "volume"]
            ].to_string(index=False)
        )

    print("\n2) TANK / BATCH CHECK (only 2000 L or 6000 L)")
    print("Batch volume distribution:")
    print(tank_summary.to_string(index=False))
    if invalid_tanks.empty:
        print("PASS: all batches use valid tank sizes.")
    else:
        print("FAIL: invalid tank sizes found:")
        print(invalid_tanks[["beer", "production_week", "volume"]].to_string(index=False))

    print("\n3) MONTHLY CONSISTENCY (weekly schedule vs monthly summary file)")
    if monthly_mismatches.empty:
        print("PASS: monthly totals match weekly rollup.")
    else:
        print("FAIL: monthly mismatches found:")
        print(monthly_mismatches.to_string(index=False))

    print("\n4) FORECAST vs PLANNED PRODUCTION (by month)")
    print("(coverage_ratio ≈ planned / forecast; >1 means extra buffer / safety stock)")
    print(reconciliation.to_string(index=False))

    out_dir = ROOT / "data/processed"
    os.makedirs(out_dir, exist_ok=True)
    service_metrics.to_csv(out_dir / "production_eval_service_level.csv", index=False)
    monthly_compare.to_csv(out_dir / "production_eval_monthly_compare.csv", index=False)
    reconciliation.to_csv(out_dir / "production_eval_forecast_vs_monthly.csv", index=False)

    print("\nSaved:")
    print(f"- {out_dir / 'production_eval_service_level.csv'}")
    print(f"- {out_dir / 'production_eval_monthly_compare.csv'}")
    print(f"- {out_dir / 'production_eval_forecast_vs_monthly.csv'}")


if __name__ == "__main__":
    main()
