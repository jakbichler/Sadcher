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

    # Read dataset and filter by n_robots.
    df = pd.read_json(input_file, lines=True)
    df = df[(df["n_robots"] == n_robots) & (df["n_tasks"] > 2)]

    # Identify instances (unique by n_tasks, n_robots, run) where all schedulers were feasible.
    instance_keys = ["n_tasks", "n_robots", "run"]
    feasible_instances = (
        df.groupby(instance_keys)["infeasible_count"].apply(lambda x: (x == 0).all()).reset_index()
    )
    feasible_instances = feasible_instances[feasible_instances["infeasible_count"]].drop(
        columns=["infeasible_count"]
    )
    df_filtered = pd.merge(df, feasible_instances, on=instance_keys)

    # -------------------------
    # Clip makespan for sampling schedulers to their non-sampling counterpart
    # -------------------------
    df_sad_base = df_filtered[df_filtered["scheduler"] == "sadcher"][
        instance_keys + ["makespan"]
    ].rename(columns={"makespan": "baseline_sadcher_makespan"})
    df_het_base = df_filtered[df_filtered["scheduler"] == "heteromrta"][
        instance_keys + ["makespan"]
    ].rename(columns={"makespan": "baseline_heteromrta_makespan"})
    df_clip = pd.merge(df_filtered, df_sad_base, on=instance_keys)
    df_clip = pd.merge(df_clip, df_het_base, on=instance_keys, how="left")

    # Apply clipping
    mask_sil = df_clip["scheduler"] == "stochastic_IL_sadcher"
    df_clip.loc[mask_sil, "makespan"] = df_clip.loc[
        mask_sil, ["makespan", "baseline_sadcher_makespan"]
    ].min(axis=1)
    mask_het = df_clip["scheduler"] == "heteromrta_sampling"
    df_clip.loc[mask_het, "makespan"] = df_clip.loc[
        mask_het, ["makespan", "baseline_heteromrta_makespan"]
    ].min(axis=1)

    df_filtered = df_clip.copy()

    # Get counts per scheduler and n_tasks
    counts = df.groupby(["scheduler", "n_tasks"]).size().reset_index(name="feasible_count")

    # Print each row with descriptive labels
    for _, row in counts.iterrows():
        print(
            f"Scheduler: {row['scheduler']}, "
            f"Number of Tasks: {row['n_tasks']}, "
            f"Feasible Instances: {row['feasible_count']}"
        )

    # -------------------------
    # Statistics Computation
    # -------------------------

    # 1.1 Average Computation Time stats
    comp_stats = (
        df_filtered.groupby(["scheduler", "n_tasks"])["computation_time_per_decision"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # 1.2 Cumulative comp times for sadcher and greedy
    comp_stats_cum = (
        df_filtered.groupby(["scheduler", "n_tasks"])["computation_time_full_solution"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # 2. Makespan stats (grouped by scheduler and n_tasks)
    makespan_stats = (
        df_filtered.groupby(["scheduler", "n_tasks"])["makespan"].agg(["mean", "std"]).reset_index()
    )

    # 3. Optimality gap using Sadcher as baseline (including MILP)
    # Get baseline makespan from Sadcher for each instance.
    df_sadcher = df_filtered[df_filtered["scheduler"] == "sadcher"][
        ["n_tasks", "n_robots", "run", "makespan"]
    ].rename(columns={"makespan": "sadcher_makespan"})
    # Merge baseline into all schedulers (including MILP)
    merged_sadcher_all = pd.merge(df_filtered, df_sadcher, on=["n_tasks", "n_robots", "run"])
    merged_sadcher_all["gap_sadcher"] = (
        merged_sadcher_all["makespan"] - merged_sadcher_all["sadcher_makespan"]
    ) / merged_sadcher_all["sadcher_makespan"]
    gap_stats_sadcher = (
        merged_sadcher_all.groupby(["scheduler", "n_tasks"])["gap_sadcher"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # -------------------------
    # Visualization
    # -------------------------

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    unique_schedulers = [
        "greedy",
        "heteromrta",
        "heteromrta_sampling",
        "milp",
        "sadcher",
        "stochastic_IL_sadcher",
    ]
    color_map = {scheduler: plt.cm.tab10(i) for i, scheduler in enumerate(unique_schedulers)}

    # identify which schedulers only have full‐solution decisions
    sched_only_full = {
        "milp",
        "stochastic_IL_sadcher",
        "rl_sadcher_sampling",
        "heteromrta_sampling",
    }
    # Plot 1: Computation Time
    for scheduler, group in comp_stats.groupby("scheduler"):
        if scheduler in sched_only_full:
            # Skip schedulers that only have full-solution decisions
            continue
        ax[0].errorbar(
            group["n_tasks"],
            group["mean"],
            label=f"{scheduler} (per decision)",
            marker="o",
            capsize=3,
            color=color_map[scheduler],
        )

    for scheduler, group in comp_stats_cum.groupby("scheduler"):
        if scheduler not in sched_only_full:
            # Skip schedulers that do have per-decision computation times
            continue
        ax[0].errorbar(
            group["n_tasks"],
            group["mean"],
            label=f"{scheduler} (full solution)",
            marker="o",
            capsize=5,
            color=color_map[scheduler],
        )

    ax[0].set_xlabel("Number of Tasks")
    ax[0].set_ylabel("Average Computation Time (s)")
    ax[0].set_title(f"Avg Comp Time vs. #Tasks (#Robots = {n_robots})")
    ax[0].legend()
    ax[0].grid(True)
    ax[0].set_xticks(range(0, 260, 20))
    ax[0].set_yscale("log")

    # Plot 2: Makespan
    for scheduler, group in makespan_stats.groupby("scheduler"):
        ax[1].errorbar(
            group["n_tasks"],
            group["mean"],
            label=scheduler,
            marker="o",
            capsize=3,
            color=color_map[scheduler],
        )
    ax[1].set_xlabel("Number of Tasks")
    ax[1].set_ylabel("Makespan")
    ax[1].set_title(f"Makespan vs. #Tasks (#Robots = {n_robots})")
    ax[1].legend()
    ax[1].grid(True)
    ax[1].set_xticks(range(0, 260, 20))

    # Plot 3: Optimality Gap (Sadcher baseline) in percentage (including MILP)
    # for scheduler, group in gap_stats_sadcher.groupby("scheduler"):
    # ax[2].errorbar(
    # group["n_tasks"],
    # group["mean"] * 100,
    # label=scheduler,
    # marker="o",
    # capsize=3,
    # color=color_map[scheduler],
    # )

    gap_stats_sadcher["mean_smooth"] = gap_stats_sadcher.groupby("scheduler")["mean"].transform(
        lambda x: x.ewm(span=5, min_periods=1).mean()
    )

    for scheduler, group in gap_stats_sadcher.groupby("scheduler"):
        ax[2].errorbar(
            group["n_tasks"],
            group["mean_smooth"] * 100,
            label=f"{scheduler}",
            marker="o",
            capsize=3,
            color=color_map[scheduler],
        )
    ax[2].set_xlabel("Number of Tasks")
    ax[2].set_ylabel("Gap to Sadcher [%]")
    ax[2].set_title(f"Gap vs. Sadcher (#Robots = {n_robots})")
    ax[2].legend()
    ax[2].grid(True)
    ax[2].set_xticks(range(0, 260, 20))
    ax[2].set_ylim(bottom=-16, top=21)
    plt.tight_layout()
    plt.show()
