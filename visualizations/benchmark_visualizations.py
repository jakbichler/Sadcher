import numpy as np


def plot_violin(ax, data, scheduler_names, comparison_type, title):
    ax.violinplot(data.values(), showmeans=True)
    ax.set_xticks(range(1, len(scheduler_names) + 1))
    ax.set_xticklabels(scheduler_names)

    if comparison_type == "makespan":
        ylabel = "Makespan"
    elif comparison_type == "computation_time":
        ylabel = "Computation Time (s)"
    else:
        raise ValueError(f"Unknown comparison type: {comparison_type}")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    text_offset_x = 0.25
    for i, s in enumerate(scheduler_names):
        avg_value = np.mean(data[s])

        if comparison_type == "makespan":
            ax.text(i + 1 + text_offset_x, avg_value, f"{avg_value:.1f}", ha="center")
        elif comparison_type == "computation_time":
            ax.text(i + 1 + text_offset_x, avg_value, f"{avg_value:.4f}", ha="center")

    for i, scheduler in enumerate(scheduler_names, start=1):
        x_jitter = np.random.normal(0, 0.03, len(data[scheduler]))
        ax.scatter(
            np.full_like(data[scheduler], i) + x_jitter,
            data[scheduler],
            alpha=0.5,
            s=10,
            color="black",
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


def print_final_results(
    scheduler_names, n_iter, feasible_makespans, infeasible_count, computation_times
):
    # Averages computed only over feasible samples
    avg_makespans = {
        s: np.mean(feasible_makespans[s]) if feasible_makespans[s] else float("nan")
        for s in scheduler_names
    }
    avg_computation_times = {s: np.mean(computation_times[s]) for s in scheduler_names}
    print(f"\nSummary of Results after {n_iter} runs:")
    for scheduler in scheduler_names:
        print(f"{scheduler.capitalize()}:")
        print(f"  Average Makespan (feasible only): {avg_makespans[scheduler]:.2f}")
        print(f"  Average Computation Time: {avg_computation_times[scheduler]:.4f} seconds")
        print(f"  Infeasible Count: {infeasible_count[scheduler]}\n")
