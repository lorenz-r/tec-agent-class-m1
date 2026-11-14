import mesa
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ..cleaning_model.model import CleaningModel

SCRIPT_LOCATION = Path(__file__).resolve().parent
ROOT = SCRIPT_LOCATION.parent
RESULTS_DIR = ROOT / "results"
AGENT_CONFIGURATIONS = [1, 2, 4, 8]

def run_experiments():
    results = mesa.batch_run(
        CleaningModel,
        parameters={
            "width": 20,
            "height": 20,
            "dirty_pct": 0.5,
            "max_steps": 50_000,
            "n_agents": AGENT_CONFIGURATIONS,
            "seed": range(30),
        },
        iterations=1,
        data_collection_period=-1,
        number_processes=None,
    )

    df = pd.DataFrame(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "sweep.csv", index=False)

    summary_df = df.groupby("n_agents").agg(
        AverageSteps=("step", "mean"),
        AverageMoves=("TotalMoves", "mean"),
        AverageCleaningPercentage=("CleaningPercentage", "mean"),
        SuccessfulRuns=("CleaningPercentage", lambda x: (x >= 100).sum())
    ).reset_index()

    summary_df.to_csv(
        RESULTS_DIR / "experiment_summary.csv",
        index=False
    )

    print("\nEXPERIMENT SUMMARY")
    print(summary_df.to_string(index=False))

    plt.figure(figsize=(8, 5))

    plt.bar(
        summary_df["n_agents"],
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
        summary_df["n_agents"],
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