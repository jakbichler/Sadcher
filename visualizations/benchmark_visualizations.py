import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter

LABEL_MAP = {
    "greedy": "Greedy",
    "milp": "MILP",
    "sadcher": "Sadcher",
    "stochastic_IL_sadcher": "Sample-Sadcher",
    "heteromrta": "HeteroMRTA",
    "heteromrta_sampling": "Sample-HeteroMRTA",
    "rl_sadcher": "RL-Sadcher",
    "rl_sadcher_sampling": "Sample-RL-Sadcher",
}


def plot_results(
    makespans,
    travel_distances,
    computation_times_per_decision,
    computation_times_full_solution,
    feasibility,
    scheduler_names,
    args,
    n_tasks,
    n_robots,
    n_skills,
    n_precedence,
):
    fig, axs = plt.subplots(1, 3, figsize=(12, 10))
    fig.suptitle(
        f"Scheduler Comparison on {args.n_iterations} instances of {n_tasks}t{n_robots}r{n_skills}s{n_precedence}p"
    )
    fig.subplots_adjust(hspace=0.4, wspace=0.8)

    def apply_mask(data_dict):
        return {s: np.array(data_dict[s])[mask] for s in scheduler_names}

    # For the graphs, we only plot runs, where all schedulers were feasible
    mask = np.logical_and.reduce([feasibility[s] for s in scheduler_names])

    makespans = apply_mask(makespans)
    travel_distances = apply_mask(travel_distances)
    computation_times_per_decision = apply_mask(computation_times_per_decision)
    computation_times_full_solution = apply_mask(computation_times_full_solution)

    plot_violin(
        axs[0],
        makespans,
        scheduler_names,
        "makespan",
        "Makespan",
    )

    plot_violin(
        axs[1],
        travel_distances,
        scheduler_names,
        "travel_distance",
        "Travel Distance",
    )

    plot_double_violin_computation_times(
        axs[2],
        computation_times_per_decision,
        computation_times_full_solution,
        scheduler_names,
        title="Computation Time",
    )
    plt.tight_layout()
    plt.show()


def plot_violin(ax, data, scheduler_names, comparison_type, title):
    ax.violinplot(data.values(), showmeans=True)
    ax.set_xticks(range(1, len(scheduler_names) + 1))

    # Build labels in the same order as scheduler_names:
    labels = [LABEL_MAP.get(s, s) for s in scheduler_names]
    ax.set_xticklabels(labels, rotation=45, ha="right")

    if comparison_type == "makespan":
        ylabel = "Makespan"
    elif comparison_type == "travel_distance":
        ylabel = "Travel Distance"
    else:
        raise ValueError(f"Unknown comparison type: {comparison_type}")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    text_offset_x = 0.25
    for i, s in enumerate(scheduler_names):
        avg_value = np.mean(data[s])

        if comparison_type == "makespan":
            ax.text(i + 1 + text_offset_x, avg_value, f"{avg_value:.1f}", ha="center")
        elif comparison_type == "travel_distance":
            ax.text(i + 1 + text_offset_x, avg_value, f"{avg_value:.1f}", ha="center")

    for i, scheduler in enumerate(scheduler_names, start=1):
        x_jitter = np.random.normal(0, 0.03, len(data[scheduler]))
        ax.scatter(
            np.full_like(data[scheduler], i) + x_jitter,
            data[scheduler],
            alpha=0.5,
            s=10,
            color="black",
        )


def plot_double_violin_computation_times(
    ax,
    data1,
    data2,
    scheduler_names,
    title,
):
    label1 = "Per Decision"
    label2 = "Full Solution"

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(-3, 2))

    pos = np.arange(1, len(scheduler_names) + 1)

    # prepare filtered list without MILP -> MILP only has one vilion as per decsion and full soliton is equal
    sched_no_milp = [s for s in scheduler_names if s != "milp"]
    pos_no_milp = [pos[i] for i, s in enumerate(scheduler_names) if s != "milp"]

    # first violin (blue)
    v1 = ax.violinplot([data1[s] for s in sched_no_milp], positions=pos_no_milp, showmeans=True)
    for pc in v1["bodies"]:
        pc.set_facecolor("C0")
        pc.set_edgecolor("k")
        pc.set_alpha(0.3)
    v1["cmeans"].set_color("k")

    # second violin (orange) only for non‐MILP schedulers
    v2 = ax.violinplot(
        [data2[s] for s in scheduler_names],
        positions=pos,
        showmeans=True,
    )
    for pc in v2["bodies"]:
        pc.set_facecolor("C1")
        pc.set_edgecolor("k")
        pc.set_alpha(0.3)
    v2["cmeans"].set_color("k")

    ax.set_xticks(pos)
    labels = [LABEL_MAP.get(s, s) for s in scheduler_names]
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ylabel = "Computation Time (s, log‐scale)"
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    text_offset_x = 0.25
    fmt = "{:.3g}"
    for i, s in enumerate(scheduler_names, start=1):
        m1, m2 = np.mean(data1[s]), np.mean(data2[s])
        ax.text(i + text_offset_x, m1, fmt.format(m1), ha="center", va="bottom")
        if s != "milp":
            ax.text(i + text_offset_x, m2, fmt.format(m2), ha="center", va="bottom")

    for i, s in enumerate(scheduler_names, start=1):
        j1 = np.random.normal(0, 0.03, len(data1[s]))
        ax.scatter(i + j1, data1[s], s=10, alpha=0.5, color="C0", edgecolor="k")
        if s != "milp":
            j2 = np.random.normal(0, 0.03, len(data2[s]))
            ax.scatter(i + j2, data2[s], s=10, alpha=0.5, color="C1", edgecolor="k")

    ax.legend(
        [
            Patch(facecolor="C0", edgecolor="k"),
            Patch(facecolor="C1", edgecolor="k"),
        ],
        [label1, label2],
        loc="upper right",
    )


def compare_makespans_1v1(ax, makespans1, makespans2, scheduler1, scheduler2):
    makespans1 = np.array(makespans1)
    makespans2 = np.array(makespans2)

    min_value = min(min(makespans1), min(makespans2))
    max_value = max(max(makespans1), max(makespans2))

    # Fill regions
    ax.fill_between(
        [min_value, max_value],
        [min_value, max_value],
        max_value,
        color="red",
        alpha=0.15,
        label=f"{scheduler1} Wins",
    )
    ax.fill_between(
        [min_value, max_value],
        min_value,
        [min_value, max_value],
        color="green",
        alpha=0.15,
        label=f"{scheduler2} Wins",
    )

    # Scatter plot
    ax.scatter(makespans1, makespans2, color="black", alpha=0.7, edgecolor="k")

    # Parity line
    x_vals = np.linspace(min_value, max_value, 100)
    ax.plot(x_vals, x_vals, color="black", linestyle="--", label="Parity Line")

    # Labels and legend
    ax.set_xlabel(f"{scheduler1} Makespan")
    ax.set_ylabel(f"{scheduler2} Makespan")
    ax.legend()


def print_final_results(scheduler_names, n_iter, makespans, feasibility, computation_times):
    # Averages computed only over feasible samples
    avg_makespans = {s: np.mean(makespans[s]) for s in scheduler_names}
    avg_computation_times = {s: np.mean(computation_times[s]) for s in scheduler_names}
    infeasible_count = {s: len([f for f in feasibility[s] if not f]) for s in scheduler_names}

    print(f"\nSummary of Results after {n_iter} runs:")
    for scheduler in scheduler_names:
        print(f"{scheduler.capitalize()}:")
        print(f"  Average Makespan (feasible only): {avg_makespans[scheduler]:.2f}")
        print(f"  Average Computation Time: {avg_computation_times[scheduler]:.4f} seconds")
        print(f"  Infeasible Count: {infeasible_count[scheduler]}\n")
