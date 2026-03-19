import pandas as pd

FORECAST_PATH = "data/processed/forecast_results.csv"
PRODUCTION_WEEKLY_PATH = "data/processed/production_schedule.csv"
PRODUCTION_MONTHLY_PATH = "data/processed/production_schedule_monthly.csv"
LEAD_TIME_WEEKS = 12
SMALL_TANK = 2000
LARGE_TANK = 6000
TANK_SET = {SMALL_TANK, LARGE_TANK}


def load_data():
    forecast = pd.read_csv(FORECAST_PATH, parse_dates=["week"])
    prod_weekly = pd.read_csv(
        PRODUCTION_WEEKLY_PATH,
        parse_dates=["production_week", "available_week"],
    )
    prod_monthly = pd.read_csv(PRODUCTION_MONTHLY_PATH, parse_dates=["month"])
    return forecast, prod_weekly, prod_monthly


def check_inventory_non_negative(forecast, prod_weekly):
    """
    Simulate inventory using forecasted demand vs planned arrivals.
    Negative inventory means a stockout-risk in the plan.
    """
    demand = (
        forecast.groupby(["beer", "week"], as_index=False)
        .agg(demand_liters=("forecast_liters", "sum"))
        .sort_values(["beer", "week"])
    )
    arrivals = (
        prod_weekly.groupby(["beer", "available_week"], as_index=False)
        .agg(arrival_liters=("volume", "sum"))
        .rename(columns={"available_week": "week"})
    )

    merged = demand.merge(arrivals, on=["beer", "week"], how="left")
    merged["arrival_liters"] = merged["arrival_liters"].fillna(0.0)

    rows = []
    for beer, group in merged.groupby("beer"):
        inv = 0.0
        min_inventory = float("inf")
        stockout_weeks = 0
        shortage_total = 0.0

        for _, r in group.sort_values("week").iterrows():
            inv += float(r["arrival_liters"])
            inv -= float(r["demand_liters"])

            if inv < 0:
                stockout_weeks += 1
                shortage_total += abs(inv)
                inv = 0.0

            min_inventory = min(min_inventory, inv)

        rows.append(
            {
                "beer": beer,
                "stockout_weeks": stockout_weeks,
                "shortage_liters": round(shortage_total, 2),
                "ending_inventory_liters": round(inv, 2),
                "min_inventory_liters": round(min_inventory if min_inventory != float("inf") else 0.0, 2),
                "inventory_ok": stockout_weeks == 0,
            }
        )

    return pd.DataFrame(rows).sort_values("beer").reset_index(drop=True)


def check_lead_time(prod_weekly):
    delta_days = (prod_weekly["available_week"] - prod_weekly["production_week"]).dt.days
    violations = prod_weekly[delta_days != LEAD_TIME_WEEKS * 7].copy()
    return violations


def check_tank_sizes(prod_weekly):
    invalid = prod_weekly[~prod_weekly["volume"].isin(TANK_SET)].copy()
    summary = (
        prod_weekly["volume"]
        .value_counts()
        .rename_axis("tank_volume")
        .reset_index(name="batch_count")
        .sort_values("tank_volume")
    )
    return invalid, summary


def check_monthly_consistency(prod_weekly, prod_monthly):
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


def reconcile_forecast_vs_monthly_production(forecast, prod_monthly):
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


def main():
    forecast, prod_weekly, prod_monthly = load_data()

    inventory_check = check_inventory_non_negative(forecast, prod_weekly)
    lead_time_violations = check_lead_time(prod_weekly)
    invalid_tanks, tank_summary = check_tank_sizes(prod_weekly)
    monthly_mismatches, monthly_compare = check_monthly_consistency(prod_weekly, prod_monthly)
    reconciliation = reconcile_forecast_vs_monthly_production(forecast, prod_monthly)

    print("\n==========================================")
    print("STEP 03 PLANNING QUALITY CHECK")
    print("==========================================")

    print("\n1) INVENTORY CHECK (no negative inventory in simulation)")
    print(inventory_check.to_string(index=False))
    print(f"\nInventory pass all beers: {bool(inventory_check['inventory_ok'].all())}")

    print("\n2) LEAD-TIME CHECK (production_week -> available_week = 12 weeks)")
    if lead_time_violations.empty:
        print("PASS: no lead-time violations.")
    else:
        print("FAIL: lead-time violations found:")
        print(
            lead_time_violations[
                ["beer", "production_week", "available_week", "volume"]
            ].to_string(index=False)
        )

    print("\n3) TANK/BATCH CHECK (only 2000L or 6000L)")
    print("Batch volume distribution:")
    print(tank_summary.to_string(index=False))
    if invalid_tanks.empty:
        print("PASS: all batches use valid tank sizes.")
    else:
        print("FAIL: invalid tank sizes found:")
        print(invalid_tanks[["beer", "production_week", "volume"]].to_string(index=False))

    print("\n4) MONTHLY CONSISTENCY CHECK (weekly rollup equals monthly output)")
    if monthly_mismatches.empty:
        print("PASS: monthly totals match weekly totals.")
    else:
        print("FAIL: monthly mismatches found:")
        print(monthly_mismatches.to_string(index=False))

    print("\n5) FORECAST vs MONTHLY PRODUCTION RECONCILIATION")
    print("(Positive gap means planned production above forecast.)")
    print(reconciliation.to_string(index=False))

    # Optional artifacts for deeper inspection
    inventory_check.to_csv("data/processed/production_eval_inventory_check.csv", index=False)
    monthly_compare.to_csv("data/processed/production_eval_monthly_compare.csv", index=False)
    reconciliation.to_csv("data/processed/production_eval_forecast_vs_monthly.csv", index=False)

    print("\nSaved:")
    print("- data/processed/production_eval_inventory_check.csv")
    print("- data/processed/production_eval_monthly_compare.csv")
    print("- data/processed/production_eval_forecast_vs_monthly.csv")


if __name__ == "__main__":
    main()