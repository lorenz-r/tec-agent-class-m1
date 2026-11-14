import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ..cleaning_model.model import CleaningModel

SCRIPT_LOCATION = Path(__file__).resolve().parent
ROOT = SCRIPT_LOCATION.parent
RESULTS_DIR = ROOT / "results"

WIDTH = 20
HEIGHT = 20
DIRTY_PCT = 0.5
MAX_STEPS = 50_000

N_RUNS = 30
AGENT_CONFIGURATIONS = [1, 2, 4, 8]

def run_experiments():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for n_agents in AGENT_CONFIGURATIONS:
        print(f"\nRunning experiments with {n_agents} robots...")

        for run in range(1, N_RUNS + 1):
            seed = run

            model = CleaningModel(
                width=WIDTH,
                height=HEIGHT,
                n_agents=n_agents,
                max_steps=MAX_STEPS,
                dirty_pct=DIRTY_PCT,
                seed=seed
            )

            model.run()
            summary = model.summary()
            reached_100 = summary["CleaningPercentage"] >= 100

            results.append({
                "Robots": n_agents,
                "Run": run,
                "Steps": model.step_count,
                "CleaningPercentage": summary["CleaningPercentage"],
                "TotalMoves": summary["TotalMoves"],
                "Reached100": reached_100,
                "FullCleaningTime": summary["FullCleaningTime"]
            })

            print(
                f"Run {run}/{N_RUNS} | "
                f"Steps: {model.step_count} | "
                f"Cleaning: {summary['CleaningPercentage']:.2f}% | "
                f"Moves: {summary['TotalMoves']} | "
                f"Completed: {reached_100}"
            )

    df = pd.DataFrame(results)

    df.to_csv(
        RESULTS_DIR / "cleaning_results.csv",
        index=False
    )

    summary_df = df.groupby("Robots").agg(
        AverageSteps=("Steps", "mean"),
        AverageMoves=("TotalMoves", "mean"),
        AverageCleaningPercentage=("CleaningPercentage", "mean"),
        SuccessfulRuns=("Reached100", "sum")
    ).reset_index()

    summary_df.to_csv(
        RESULTS_DIR / "experiment_summary.csv",
        index=False
    )

    print("\nEXPERIMENT SUMMARY")
    print(summary_df.to_string(index=False))

    plt.figure(figsize=(8, 5))

    plt.bar(
        summary_df["Robots"],
        summary_df["AverageSteps"]
    )

    plt.xlabel("Number of Robots")
    plt.ylabel("Average Steps")
    plt.title("Average Steps vs Number of Robots")
    plt.xticks(AGENT_CONFIGURATIONS)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "average_steps.png")
    plt.close()

    plt.figure(figsize=(8, 5))

    plt.bar(
        summary_df["Robots"],
        summary_df["AverageMoves"]
    )

    plt.xlabel("Number of Robots")
    plt.ylabel("Average Movements")
    plt.title("Average Movements vs Number of Robots")
    plt.xticks(AGENT_CONFIGURATIONS)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "average_movements.png")
    plt.close()

if __name__ == "__main__":
    run_experiments()