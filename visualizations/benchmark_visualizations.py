import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import wilcoxon

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
    n_iterations,
    n_tasks,
    n_robots,
    n_skills,
    n_precedence,
    return_fig=False,
    paper_format=False,
):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5) if paper_format else (16, 10))
    if not paper_format:
        fig.suptitle(
            f"Scheduler Comparison on {n_iterations} instances of {n_tasks}t{n_robots}r{n_skills}s{n_precedence}p"
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
        paper_format=paper_format,
    )

    plot_violin(
        axs[1],
        travel_distances,
        scheduler_names,
        "travel_distance",
        "Travel Distance",
        paper_format=paper_format,
    )

    plot_double_violin_computation_times(
        axs[2],
        computation_times_per_decision,
        computation_times_full_solution,
        scheduler_names,
        title="Computation Time",
        paper_format=paper_format,
    )
    # Horizontal padding for edgbe violin labels
    for ax in axs:
        ax.set_xlim(0.5, len(scheduler_names) + 1)

    if return_fig:
        return fig

    plt.tight_layout()
    plt.show()


def _sig_symbol(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def plot_violin(ax, data, scheduler_names, comparison_type, title, paper_format=False):
    ax.violinplot(data.values(), showmeans=True)
    ax.set_xticks(range(1, len(scheduler_names) + 1))
    labels = [LABEL_MAP.get(s, s) for s in scheduler_names]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)

    # ── Wilcoxon statistical signifcance  ───────────────────────────────────────
    PAIRS = [
        ("sadcher", "heteromrta"),
        ("sadcher", "heteromrta_sampling"),
        ("stochastic_IL_sadcher", "heteromrta_sampling"),
    ]
    base_max = max(*(np.max(data[a]) for pair in PAIRS for a in pair if a in data))
    offsets = [1.05 + i * 0.1 for i in range(len(PAIRS))]

    for idx, (A, B) in enumerate(PAIRS):
        arrA = data.get(A)
        arrB = data.get(B)
        if arrA is None or arrB is None or len(arrA) != len(arrB):
            continue

        W, p = wilcoxon(arrA, arrB, alternative="two-sided")
        print(f"Wilcoxon {A} vs {B}: W={W}, p={p:.4g}")

        iA = scheduler_names.index(A) + 1
        iB = scheduler_names.index(B) + 1

        y0 = base_max * offsets[idx]
        y1 = y0 * 1.02

        # draw the “T-shaped” bar
        ax.plot([iA, iA, iB, iB], [y0, y1, y1, y0], color="k")
        ax.text(
            (iA + iB) / 2,
            y1 * 1.005,
            f"{_sig_symbol(p)}, p={p:.2g}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    # ── Scatter plot of individual data points ────────────────────────────────
    if comparison_type == "makespan":
        ylabel = "Makespan (time steps)"
    elif comparison_type == "travel_distance":
        ylabel = "Travel Distance (simulation units)"
    else:
        raise ValueError(f"Unknown comparison type: {comparison_type}")

    ax.set_ylabel(ylabel, fontsize=12)
    if not paper_format:
        ax.set_title(title, fontsize=14)

    text_offset_x = 0.4
    fontsize = 10
    for i, s in enumerate(scheduler_names):
        avg_value = np.mean(data[s])
        ax.text(
            i + 1 + text_offset_x,
            avg_value,
            f"{avg_value:.1f}",
            ha="center",
            fontsize=fontsize,
            fontweight="bold",
        )

    for i, scheduler in enumerate(scheduler_names, start=1):
        x_jitter = np.random.normal(0, 0.02, len(data[scheduler]))
        ax.scatter(
            np.full_like(data[scheduler], i) + x_jitter,
            data[scheduler],
            alpha=0.1,
            s=10,
            color="black",
        )


def plot_double_violin_computation_times(
    ax,
    data1,
    data2,
    scheduler_names,
    title,
    paper_format=False,
):
    import numpy as np

    label1 = "Per Decision"
    label2 = "Full Solution"

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))

    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
    )
    ax.grid(which="minor", axis="y", linestyle=":", color="gray", alpha=0.5)

    pos = np.arange(1, len(scheduler_names) + 1)

    # identify which schedulers only have full‐solution decisions
    sched_only_full = {
        "milp",
        "stochastic_IL_sadcher",
        "rl_sadcher_sampling",
        "heteromrta_sampling",
    }
    instant_scheds = [s for s in scheduler_names if s not in sched_only_full]
    instant_pos = [pos[i] for i, s in enumerate(scheduler_names) if s in instant_scheds]

    # 1) per-decision violins (blue) — only for instant schedulers
    v1 = ax.violinplot(
        [data1[s] for s in instant_scheds],
        positions=instant_pos,
        showmeans=True,
    )
    for pc in v1["bodies"]:
        pc.set_facecolor("C0")
        pc.set_edgecolor("k")
        pc.set_alpha(0.3)
    v1["cmeans"].set_color("k")

    # 2) full-solution violins (orange) — for all schedulers
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

    # x-axis
    ax.set_xticks(pos)
    labels = [LABEL_MAP.get(s, s) for s in scheduler_names]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("Computation Time (s, log‐scale)", fontsize=12)
    ax.set_title(title, fontsize=14) if not paper_format else None

    # annotate means
    text_offset_x = 0.5
    fontsize = 10
    fmt = "{:.3g}"
    for i, s in enumerate(scheduler_names, start=1):
        if s in instant_scheds:
            # show per-decision mean
            m1 = np.mean(data1[s])
            ax.text(
                i + text_offset_x,
                m1,
                fmt.format(m1),
                ha="center",
                va="bottom",
                fontsize=fontsize,
                fontweight="bold",
            )
        # show full-solution mean for everyone
        m2 = np.mean(data2[s])
        ax.text(
            i + text_offset_x,
            m2,
            fmt.format(m2),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
        )

    # scatter data points
    for i, s in enumerate(scheduler_names, start=1):
        if s in instant_scheds:
            # per-decision points
            j1 = np.random.normal(0, 0.015, len(data1[s]))
            ax.scatter(i + j1, data1[s], s=10, alpha=0.2, color="C0", edgecolor="k")
        # full-solution points for all
        j2 = np.random.normal(0, 0.015, len(data2[s]))
        ax.scatter(i + j2, data2[s], s=10, alpha=0.2, color="C1", edgecolor="k")

    # legend
    ax.legend(
        [
            Patch(facecolor="C0", edgecolor="k"),
            Patch(facecolor="C1", edgecolor="k"),
        ],
        [label1, label2],
        loc="upper right",
    )


def compare_makespans_1v1(ax, makespans1, makespans2, scheduler1, scheduler2, legend=True):
    makespans1 = np.array(makespans1)
    makespans2 = np.array(makespans2)

    min_value = min(min(makespans1), min(makespans2))
    max_value = max(max(makespans1), max(makespans2))

    # Fill regions
    ax.fill_between(
        [min_value, max_value],
        [min_value, max_value],
        max_value,
        color="tab:blue",
        alpha=0.15,
        label=f"{scheduler1} Wins",
    )
    ax.fill_between(
        [min_value, max_value],
        min_value,
        [min_value, max_value],
        color="tab:orange",
        alpha=0.15,
        label=f"{scheduler2} Wins",
    )

    # Scatter plot
    ax.scatter(makespans1, makespans2, color="black", alpha=0.4, edgecolor="k")

    # Parity line
    x_vals = np.linspace(min_value, max_value, 100)
    ax.plot(x_vals, x_vals, color="black", linestyle="-", label="Parity Line", linewidth=3)

    # Labels and legend
    ax.set_xlabel(f"{scheduler1} Makespan", fontsize=12)
    ax.set_ylabel(f"{scheduler2} Makespan", fontsize=12)
    ax.legend(fontsize=12, loc="upper left") if legend else None


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
