import argparse

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description="Visualize scaling graphs for scheduling algorithms")
parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON Lines file")
parser.add_argument("--n_robots", type=int, required=True, help="Number of robots to filter for")
args = parser.parse_args()

if __name__ == "__main__":
    input_file = args.input_file
    n_robots = args.n_robots

    df = pd.read_json(input_file, lines=True)

    # Filter for the fixed number of robtos and exclude infeasible solutions (-> have no makespan, since MILP took too long)
    df_filtered = df[(df["n_robots"] == n_robots) & (df["infeasible_count"] == 0)]

    # Group by scheduler and number of tasks to compute stats for average computation time
    comp_stats = (
        df_filtered.groupby(["scheduler", "n_tasks"])["avg_comp_time"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Group by scheduler and number of tasks to compute stats for makespan
    makespan_stats = (
        df_filtered.groupby(["scheduler", "n_tasks"])["makespan"].agg(["mean", "std"]).reset_index()
    )

    # Compute optimality gap for non-MILP schedulers relative to MILP for the same run and configuration
    df_milp = df_filtered[df_filtered["scheduler"] == "milp"][
        ["n_tasks", "n_robots", "run", "makespan"]
    ].rename(columns={"makespan": "milp_makespan"})
    df_non = df_filtered[df_filtered["scheduler"] != "milp"]
    merged = pd.merge(df_non, df_milp, on=["n_tasks", "n_robots", "run"])
    merged["gap"] = (merged["makespan"] - merged["milp_makespan"]) / merged["milp_makespan"]

    # Create a gap dataframe including MILP (with gap=0) and non-MILP results
    df_gap_milp = df_filtered[df_filtered["scheduler"] == "milp"][["scheduler", "n_tasks"]].copy()
    df_gap_milp["gap"] = 0.0
    df_gap_non = merged[["scheduler", "n_tasks", "gap"]]
    df_gap = pd.concat([df_gap_non, df_gap_milp])
    gap_stats = df_gap.groupby(["scheduler", "n_tasks"])["gap"].agg(["mean", "std"]).reset_index()

    # Create plots
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))

    # Plot for Average Computation Time
    for scheduler, group in comp_stats.groupby("scheduler"):
        ax[0].errorbar(
            group["n_tasks"],
            group["mean"],
            yerr=group["std"],
            label=scheduler,
            marker="o",
            capsize=5,
        )
    ax[0].set_xlabel("Number of Tasks")
    ax[0].set_ylabel("Average Computation Time (s)")
    ax[0].set_title(f"Avg Computation Time vs. #Tasks (#Robots = {n_robots})")
    ax[0].legend()
    ax[0].grid(True)
    ax[0].set_xticks(range(4, 14))

    # Plot for Makespan
    for scheduler, group in makespan_stats.groupby("scheduler"):
        ax[1].errorbar(group["n_tasks"], group["mean"], label=scheduler, marker="o", capsize=5)
    ax[1].set_xlabel("Number of Tasks")
    ax[1].set_ylabel("Makespan")
    ax[1].set_title(f"Makespan vs. #Tasks (#Robots = {n_robots})")
    ax[1].legend()
    ax[1].grid(True)
    ax[1].set_xticks(range(4, 14))

    # Plot for Optimality Gap (non-MILP relative to MILP)
    for scheduler, group in gap_stats.groupby("scheduler"):
        ax[2].errorbar(
            group["n_tasks"], group["mean"] * 100, label=scheduler, marker="o", capsize=5
        )
    ax[2].set_xlabel("Number of Tasks")
    ax[2].set_ylabel("Optimality Gap [%]")
    ax[2].set_title(f"Optimality Gap vs. #Tasks (#Robots = {n_robots})")
    ax[2].legend()
    ax[2].grid(True)
    ax[2].set_xticks(range(4, 14))

    plt.tight_layout()
    plt.show()
