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
SAFETY_FACTOR = 1.20  # 20% safety buffer

# ==============================
# LOAD DATA
# ==============================

def load_forecast():
    df = pd.read_csv(FORECAST_PATH)
    df["week"] = pd.to_datetime(df["week"])
    return df

# ==============================
# PACKAGING STRATEGY
# ==============================

def choose_packaging_strategy(beer_df):

    containers = beer_df["container"].unique()

    has_cans = any("Can" in c for c in containers)
    has_bottles = any("Bottle" in c for c in containers)

    # Prefer dominant volume type
    volume_by_container = (
        beer_df.groupby("container")["forecast_liters"]
        .sum()
        .sort_values(ascending=False)
    )

    if has_cans and not has_bottles:
        return "CANS_KEGS"
    elif has_bottles and not has_cans:
        return "BOTTLES_KEGS"
    else:
        # Choose dominant packaging
        dominant = volume_by_container.index[0]
        if "Can" in dominant:
            return "CANS_KEGS"
        else:
            return "BOTTLES_KEGS"

# ==============================
# PRODUCTION PLANNING
# ==============================

def plan_production(beer_df, beer_name):

    strategy = choose_packaging_strategy(beer_df)

    # Aggregate total liters per week
    weekly_total = (
        beer_df.groupby("week")["forecast_liters"]
        .sum()
        .reset_index()
    )

    inventory = 0
    production_schedule = []

    for i in range(len(weekly_total)):

        current_week = weekly_total.iloc[i]["week"]

        # Cumulative demand next 12 weeks
        lookahead = weekly_total.iloc[i:i+LEAD_TIME]
        cumulative_demand = lookahead["forecast_liters"].sum()

        # Add safety buffer
        cumulative_demand *= SAFETY_FACTOR

        if inventory < cumulative_demand:

            needed = cumulative_demand - inventory

            while needed > 0:

                if needed > LARGE_TANK:
                    brew_volume = LARGE_TANK
                else:
                    brew_volume = SMALL_TANK

                production_week = current_week
                available_week = current_week + pd.Timedelta(weeks=LEAD_TIME)

                production_schedule.append({
                    "beer": beer_name,
                    "production_week": production_week,
                    "available_week": available_week,
                    "volume": brew_volume,
                    "packaging_strategy": strategy
                })

                inventory += brew_volume
                needed -= brew_volume

        # Simulate consumption
        inventory -= weekly_total.iloc[i]["forecast_liters"]
        inventory = max(inventory, 0)

    return pd.DataFrame(production_schedule)

# ==============================
# MAIN
# ==============================

def main():

    forecast_df = load_forecast()

    results = []

    for beer in forecast_df["beer"].unique():

        print(f"Planning production for {beer}...")

        beer_df = forecast_df[forecast_df["beer"] == beer].copy()

        production = plan_production(beer_df, beer)

        results.append(production)

    final_schedule = pd.concat(results)

    os.makedirs("data/processed", exist_ok=True)
    final_schedule.to_csv(OUTPUT_PATH, index=False)

    print("STEP 3 COMPLETE ✅")
    print("Production schedule shape:", final_schedule.shape)


if __name__ == "__main__":
    main()