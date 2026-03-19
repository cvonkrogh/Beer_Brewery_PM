"""
Re-run the same beer-level rolling CV as the Streamlit dashboard.

Uses `evaluate_all_beers` from `02b_model_evaluation.py`: for each focus beer,
all containers are summed to weekly total liters, then HistGradientBoosting
is scored with rolling 80/20 CV (same as dashboard table).
"""
from pathlib import Path
import importlib.util
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED_DATA_PATH = ROOT / "data/processed/weekly_beer_data.csv"

FOCUS_BEERS = [
    "Hoop Bleke Nelis",
    "Hoop Lager",
    "Hoop Kaper Tropical IPA",
]

# Match 04_dashboard_app.py `load_shared_evaluation` exactly
TEST_RATIO = 0.20
MAX_FOLDS = 5


def _load_eval_module():
    path = ROOT / "02b_model_evaluation.py"
    spec = importlib.util.spec_from_file_location("eval_shared_02b", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    mod = _load_eval_module()
    evaluate_all_beers = mod.evaluate_all_beers

    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["week"])
    df = df[df["beer"].isin(FOCUS_BEERS)].copy()

    metrics_df, _plot_map, overall_df = evaluate_all_beers(
        processed_df=df,
        focus_beers=FOCUS_BEERS,
        test_ratio=TEST_RATIO,
        max_folds=MAX_FOLDS,
        # dashboard omits this → default True in evaluate_all_beers
    )

    if metrics_df.empty:
        print("No metrics (check data / beer names).")
        return

    results_df = pd.concat([metrics_df, overall_df], ignore_index=True)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 60)
    print("MODEL vs NAIVE (same as dashboard: beer total / week, rolling CV)")
    print("=" * 60 + "\n")
    print(results_df.to_string(index=False))
    print()


if __name__ == "__main__":
    main()
