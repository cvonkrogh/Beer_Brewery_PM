import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from model_evaluation_shared import evaluate_all_beers

# ==============================
# CONFIG
# ==============================

PROCESSED_DATA_PATH = "data/processed/weekly_beer_data.csv"
TEST_RATIO = 0.20
MAX_FOLDS = 5

FOCUS_BEERS = [
    "Hoop Bleke Nelis",
    "Hoop Lager",
    "Hoop Kaper Tropical IPA",
]

BEER_COLORS = {
    "Hoop Bleke Nelis": "#1f77b4",
    "Hoop Lager": "#ff7f0e",
    "Hoop Kaper Tropical IPA": "#2ca02c",
}

# ==============================
# COMBINED PLOT
# ==============================

def plot_comparison(all_test_results):
    """One subplot per beer using shared 80/20 rolling-CV plot fold."""

    n_beers = len(all_test_results)
    fig, axes = plt.subplots(n_beers, 1, figsize=(14, 5 * n_beers), sharex=False)

    if n_beers == 1:
        axes = [axes]

    for ax, (beer_name, df) in zip(axes, all_test_results.items()):
        color = BEER_COLORS.get(beer_name, None)
        ax.plot(
            df["week"], df["actual"],
            color=color, linewidth=2,
            label="Actual",
        )
        ax.plot(
            df["week"], df["predicted"],
            color=color, linewidth=2, linestyle="--",
            label="Predicted",
        )
        ax.fill_between(
            df["week"], df["actual"], df["predicted"],
            color=color, alpha=0.1,
        )

        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
        ax.tick_params(axis="x", rotation=45)

        ax.set_ylabel("Total Liters")
        ax.set_title(f"{beer_name} — Actual vs Predicted")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Model Evaluation (test set, all containers summed)", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = "data/processed/model_evaluation_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.close()


# ==============================
# MAIN
# ==============================

def main():
    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["week"])
    df = df[df["beer"].isin(FOCUS_BEERS)].copy()
    metrics_df, all_test_results, overall_df = evaluate_all_beers(
        processed_df=df,
        focus_beers=FOCUS_BEERS,
        test_ratio=TEST_RATIO,
        max_folds=MAX_FOLDS,
    )

    if all_test_results:
        plot_comparison(all_test_results)

    if not metrics_df.empty:
        results_df = pd.concat([metrics_df, overall_df], ignore_index=True)
        print("\n" + "=" * 60)
        print("MODEL vs NAIVE BASELINE  (shared rolling 80/20 CV)")
        print("=" * 60 + "\n")
        print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
