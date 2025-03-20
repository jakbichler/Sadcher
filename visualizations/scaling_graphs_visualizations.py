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

    # Read the full dataset
    df = pd.read_json(input_file, lines=True)

    # First, filter by the fixed number of robots.
    df = df[df["n_robots"] == n_robots]

    # Identify instances (unique by n_tasks, n_robots, and run) where all schedulers were feasible.
    instance_keys = ["n_tasks", "n_robots", "run"]
    feasible_instances = (
        df.groupby(instance_keys)["infeasible_count"].apply(lambda x: (x == 0).all()).reset_index()
    )
    # Keep only those instances where all schedulers have a feasible solution.
    feasible_instances = feasible_instances[feasible_instances["infeasible_count"]]
    feasible_instances = feasible_instances.drop(columns=["infeasible_count"])

    # Merge the feasible instance identifiers back to the original dataframe.
    df_filtered = pd.merge(df, feasible_instances, on=instance_keys)

    # Print the count of rows per scheduler and number of tasks for the fully feasible instances.
    print("Counts for feasible instances:")
    print(df_filtered.groupby(["scheduler", "n_tasks"]).size())

    # -------------------------
    # Statistics Computation
    # -------------------------

    # 1. Average Computation Time stats (grouped by scheduler and n_tasks)
    comp_stats = (
        df_filtered[df_filtered["n_tasks"] > 2]
        .groupby(["scheduler", "n_tasks"])["avg_comp_time"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # 2. Makespan stats (grouped by scheduler and n_tasks)
    makespan_stats = (
        df_filtered.groupby(["scheduler", "n_tasks"])["makespan"].agg(["mean", "std"]).reset_index()
    )

    # 3. Optimality gap using MILP as baseline (only for instances with MILP available)
    df_milp = df_filtered[df_filtered["scheduler"] == "milp"][
        ["n_tasks", "n_robots", "run", "makespan"]
    ].rename(columns={"makespan": "milp_makespan"})
    df_non_milp = df_filtered[df_filtered["scheduler"] != "milp"]
    merged = pd.merge(df_non_milp, df_milp, on=["n_tasks", "n_robots", "run"])
    merged["gap"] = (merged["makespan"] - merged["milp_makespan"]) / merged["milp_makespan"]

    # Include MILP rows (with gap = 0) for a complete comparison.
    df_gap_milp = df_filtered[df_filtered["scheduler"] == "milp"][["scheduler", "n_tasks"]].copy()
    df_gap_milp["gap"] = 0.0
    df_gap_non = merged[["scheduler", "n_tasks", "gap"]]
    df_gap = pd.concat([df_gap_non, df_gap_milp])
    gap_stats = df_gap.groupby(["scheduler", "n_tasks"])["gap"].agg(["mean", "std"]).reset_index()

    # 4. Alternative gap: Comparing greedy and random_bipartite against sadcher as baseline.
    # For large instances where MILP is not available, we use sadcher's makespan as the baseline.
    # Filter data for the three non-MILP algorithms.

    df_sadcher = df_non_milp[df_non_milp["scheduler"] == "sadcher"][
        ["n_tasks", "n_robots", "run", "makespan"]
    ].rename(columns={"makespan": "sadcher_makespan"})
    # Merge baseline into the non-sadcher results (greedy and random_bipartite).
    merged_sadcher = pd.merge(df_non_milp, df_sadcher, on=["n_tasks", "n_robots", "run"])
    merged_sadcher["gap_sadcher"] = (
        merged_sadcher["makespan"] - merged_sadcher["sadcher_makespan"]
    ) / merged_sadcher["sadcher_makespan"]

    df_sadcher_gap = df_non_milp[df_non_milp["scheduler"] == "sadcher"][
        ["scheduler", "n_tasks"]
    ].copy()
    df_sadcher_gap["gap_sadcher"] = 0.0
    df_non_sadcher_gap = merged_sadcher[["scheduler", "n_tasks", "gap_sadcher"]]
    df_gap_sadcher = pd.concat([df_non_sadcher_gap, df_sadcher_gap])

    # Group these gap values by scheduler and n_tasks.
    gap_stats_sadcher = (
        merged_sadcher.groupby(["scheduler", "n_tasks"])["gap_sadcher"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # -------------------------
    # Visualization
    # -------------------------

    # Create 4 subplots: computation time, makespan, MILP-based gap, and sadcher-based gap.
    fig, ax = plt.subplots(1, 4, figsize=(18, 6))
    unique_schedulers = comp_stats["scheduler"].unique()
    color_map = {scheduler: plt.cm.tab10(i) for i, scheduler in enumerate(unique_schedulers)}

    # Plot 1: Average Computation Time
    for scheduler, group in comp_stats.groupby("scheduler"):
        ax[0].errorbar(
            group["n_tasks"],
            group["mean"],
            label=scheduler,
            marker="o",
            capsize=5,
            color=color_map[scheduler],
        )
    ax[0].set_xlabel("Number of Tasks")
    ax[0].set_ylabel("Average Computation Time (s)")
    ax[0].set_title(f"Avg Comp Time vs. #Tasks (#Robots = {n_robots})")
    ax[0].legend()
    ax[0].grid(True)
    ax[0].set_xticks(range(0, 110, 10))
    ax[0].set_yscale("log")

    # Plot 2: Makespan
    for scheduler, group in makespan_stats.groupby("scheduler"):
        ax[1].errorbar(
            group["n_tasks"],
            group["mean"],
            label=scheduler,
            marker="o",
            capsize=5,
            color=color_map[scheduler],
        )
    ax[1].set_xlabel("Number of Tasks")
    ax[1].set_ylabel("Makespan")
    ax[1].set_title(f"Makespan vs. #Tasks (#Robots = {n_robots})")
    ax[1].legend()
    ax[1].grid(True)
    ax[1].set_xticks(range(0, 110, 10))

    # Plot 3: Optimality Gap (MILP baseline) in percentage
    for scheduler, group in gap_stats.groupby("scheduler"):
        ax[2].errorbar(
            group["n_tasks"],
            group["mean"] * 100,
            label=scheduler,
            marker="o",
            capsize=5,
            color=color_map[scheduler],
        )
    ax[2].set_xlabel("Number of Tasks")
    ax[2].set_ylabel("Optimality Gap [%]")
    ax[2].set_title(f"Gap vs. MILP Baseline (#Robots = {n_robots})")
    ax[2].legend()
    ax[2].grid(True)
    ax[2].set_xticks(range(0, 20, 2))

    # Plot 4: Optimality Gap (Sadcher baseline) in percentage (for greedy and random_bipartite)
    for scheduler, group in gap_stats_sadcher.groupby("scheduler"):
        ax[3].errorbar(
            group["n_tasks"],
            group["mean"] * 100,
            label=scheduler,
            marker="o",
            capsize=5,
            color=color_map[scheduler],
        )
    ax[3].set_xlabel("Number of Tasks")
    ax[3].set_ylabel("Gap to Sadcher [%]")
    ax[3].set_title(f"Gap vs. Sadcher Baseline (#Robots = {n_robots})")
    ax[3].legend()
    ax[3].grid(True)
    ax[3].set_xticks(range(0, 110, 10))

    plt.tight_layout()
    plt.show()
