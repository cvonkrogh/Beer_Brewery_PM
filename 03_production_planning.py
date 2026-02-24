import pandas as pd
import numpy as np
import os

# ==============================
# CONFIG
# ==============================

FORECAST_PATH = "data/processed/forecast_results.csv"
OUTPUT_PATH = "data/processed/production_schedule.csv"

LEAD_TIME = 12
SMALL_TANK = 2000
LARGE_TANK = 6000

NUM_SMALL_TANKS = 4
NUM_LARGE_TANKS = 7


# ==============================
# LOAD FORECAST
# ==============================

def load_forecast():
    print("Loading forecast data...")
    df = pd.read_csv(FORECAST_PATH)
    df["week"] = pd.to_datetime(df["week"])
    return df


# ==============================
# PLAN PRODUCTION PER BEER
# ==============================

def plan_production_for_beer(beer_df, beer_name):
    print(f"Planning production for {beer_name}...")

    beer_df = beer_df.sort_values("week").copy()

    inventory = 0
    production_plan = []

    tank_usage = {}  # Track tanks used per production week

    for i in range(len(beer_df)):

        week = beer_df.iloc[i]["week"]
        demand = beer_df.iloc[i]["forecast_liters"]

        inventory -= demand

        if inventory < 0:

            shortage = abs(inventory)

            # Decide tank size
            if shortage > LARGE_TANK:
                tank_volume = LARGE_TANK
                tank_type = "6000L"
            else:
                tank_volume = SMALL_TANK
                tank_type = "2000L"

            production_week = week - pd.Timedelta(weeks=LEAD_TIME)

            # Track tank usage
            if production_week not in tank_usage:
                tank_usage[production_week] = {"2000L": 0, "6000L": 0}

            # Check capacity
            if tank_type == "2000L":
                if tank_usage[production_week]["2000L"] >= NUM_SMALL_TANKS:
                    continue
                tank_usage[production_week]["2000L"] += 1
            else:
                if tank_usage[production_week]["6000L"] >= NUM_LARGE_TANKS:
                    continue
                tank_usage[production_week]["6000L"] += 1

            inventory += tank_volume

            production_plan.append({
                "production_week": production_week,
                "beer": beer_name,
                "volume": tank_volume,
                "tank_type": tank_type,
                "available_week": week
            })

    return production_plan


# ==============================
# MAIN
# ==============================

def main():
    forecast_df = load_forecast()

    all_plans = []

    for beer in forecast_df["beer"].unique():
        beer_df = forecast_df[forecast_df["beer"] == beer]
        plans = plan_production_for_beer(beer_df, beer)
        all_plans.extend(plans)

    production_df = pd.DataFrame(all_plans)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    production_df.to_csv(OUTPUT_PATH, index=False)

    print("STEP 3 COMPLETE ✅")


if __name__ == "__main__":
    main()