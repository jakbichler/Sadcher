import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge


def show_problem_instance(problem_instance):
    Q = np.array(problem_instance["Q"])
    R = np.array(problem_instance["R"])
    robots = range(Q.shape[0])
    skills = range(Q.shape[1])
    n_tasks = R.shape[0] - 2

    """Displays a table with robot capabilities and task requirements."""
    fig, ax = plt.subplots(figsize=(8, 4))  # Smaller figure for the table
    ax.axis("off")

    # Build table data
    robot_info = [
        f"Robot {i}: " + ", ".join([f"Skill {s}" for s in skills if Q[i][s] == 1]) for i in robots
    ]
    task_info = [
        f"Task {k}: " + ", ".join([f"Skill {s}" for s in skills if R[k][s] == 1])
        for k in range(1, n_tasks + 1)
    ]
    max_length = max(len(robot_info), len(task_info))
    robot_info += [""] * (
        max_length - len(robot_info)
    )  # Pad with empty strings if robots are fewer
    task_info += [""] * (max_length - len(task_info))  # Pad with empty strings if tasks are fewer

    table_data = [["Robots Capabilities", "Tasks Requirements"]] + list(zip(robot_info, task_info))

    # Create the table
    table = ax.table(cellText=table_data, colLabels=None, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)  # Smaller font size for better fit
    table.scale(1, 1.5)  # Scale the table for better layout

    plt.tight_layout()
    plt.show()  # Displays the table


def plot_task_map(task_locations, T_execution, R):
    n_skills = R.shape[1]
    colors = plt.cm.Set1(np.linspace(0, 1, n_skills))  # Generate a color palette
    marker_sizes = T_execution[1:-1] * 3

    def draw_pie(ax, x, y, sizes, radius):
        start_angle = 0
        for size, color in zip(sizes, colors):
            end_angle = start_angle + size * 360
            if size > 0:
                wedge = Wedge(
                    (x, y),
                    radius,
                    start_angle,
                    end_angle,
                    facecolor=color,
                    edgecolor="black",
                    lw=0.5,
                )
                ax.add_patch(wedge)
            start_angle = end_angle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot tasks with pie-chart representation
    for idx, (x, y) in enumerate(task_locations[1:-1], start=1):
        skills_required = R[idx]
        total_skills = np.sum(skills_required)
        skill_sizes = (
            skills_required / total_skills if total_skills > 0 else np.zeros_like(skills_required)
        )
        draw_pie(ax, x, y, skill_sizes, marker_sizes[idx - 1] / 50)
        plt.text(x, y, f"Task {idx}", fontsize=10, ha="center")

    # Plot start and end points
    ax.scatter(
        task_locations[0, 0],
        task_locations[0, 1],
        color="green",
        s=150,
        marker="x",
        label="Start (Task 0)",
    )
    plt.text(task_locations[0, 0] + 6, task_locations[0, 1] - 1, "Start", fontsize=15, ha="center")
    ax.scatter(
        task_locations[-1, 0],
        task_locations[-1, 1],
        color="red",
        s=150,
        marker="x",
        label="End (Task -1)",
    )
    plt.text(task_locations[-1, 0] + 6, task_locations[-1, 1] - 1, "End", fontsize=15, ha="center")

    legend_patches = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[i],
            markersize=10,
            label=f"Skill {i}",
        )
        for i in range(n_skills)
    ]
    ax.legend(handles=legend_patches, title="Task Skills", loc="upper right")

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("Task Map (Size corresponds to execution time)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    plt.show()
