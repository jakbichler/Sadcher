import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.patches import Wedge


def plot_gantt_chart(title, schedule):
    """Displays a Gantt chart showing the tasks assigned to each robot over time."""

    # Create task colors
    task_colors = {}  
    task_colors = plt.cm.get_cmap('hsv', schedule.n_tasks + 2)

    fig, ax = plt.subplots(figsize=(10, 6))

    yticks = []
    yticklabels = []
    legend_elements = []

    for idx, robot in enumerate(reversed(range(schedule.n_robots))):
        yticks.append(idx)
        yticklabels.append(f"Robot {robot}")
        for task in schedule.robot_schedules[robot]:
            task, start_time, end_time = task
            ax.barh(
                idx,
                end_time - start_time,
                left=start_time,
                height=0.4,
                align='center',
                color=task_colors(task),
                edgecolor='black',
            )
            ax.text(
                start_time + (end_time - start_time) / 2,
                idx,
                f"Task {task}",
                va='center',
                ha='center',
                color='black',
                fontsize=6,  # Smaller font size for a large number of tasks
            )
            # Add to legend if not already added
            if task not in [e.get_label() for e in legend_elements]:
                legend_elements.append(
                    Patch(facecolor=task_colors(task), edgecolor='black', label=f'Task {task}')
                )


    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel('Time')
    ax.set_title(title)
    ax.grid(True)
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(0, schedule.makespan+10)
    ax.axvline(x=schedule.makespan, color='red', linestyle='--', label='Makespan')

    plt.tight_layout()
    plt.show()  # Displays the Gantt chart without blocking



def plot_robot_trajectories(task_locations, robot_schedules, T_execution, R):
    def draw_arrow(start, end, color, label=""):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        adjusted_dx, adjusted_dy = dx / length * (length - 5), dy / length * (length - 5)
        plt.arrow(start[0], start[1], adjusted_dx, adjusted_dy, head_width=2, head_length=5,
                  fc=color, ec=color, alpha=0.7, label=label)

    def draw_pie(ax, x, y, sizes, radius):
        start_angle = 0
        for size, color in zip(sizes, colors):
            end_angle = start_angle + size * 360
            if size > 0:
                wedge = Wedge((x, y), radius, start_angle, end_angle, facecolor=color, edgecolor="black", lw=0.5)
                ax.add_patch(wedge)
            start_angle = end_angle

    marker_sizes = T_execution[1:-1] * 2
    n_skills = R.shape[1]
    colors = plt.cm.Set1(np.linspace(0, 1, n_skills))  # Generate skill color palette

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot tasks with pie-chart representation
    for idx, (x, y) in enumerate(task_locations[1:-1], start=1):
        skills_required = R[idx]
        total_skills = np.sum(skills_required)
        skill_sizes = skills_required / total_skills if total_skills > 0 else np.zeros_like(skills_required)
        draw_pie(ax, x, y, skill_sizes, marker_sizes[idx - 1] / 100)

    # Plot start and end points
    ax.scatter(task_locations[0, 0], task_locations[0, 1], color='green', s=150, marker='x', label="Start (Task 0)")
    ax.text(task_locations[0, 0] + 6, task_locations[0, 1] - 1, "Start", fontsize=12, ha='center')
    ax.scatter(task_locations[-1, 0], task_locations[-1, 1], color='red', s=150, marker='x', label="End (Task -1)")
    ax.text(task_locations[-1, 0] + 6, task_locations[-1, 1] - 1, "End", fontsize=12, ha='center')

    # Draw arrows for robot trajectories
    trajectory_colors = plt.cm.Set1(np.linspace(0, 1, len(robot_schedules.keys())))
    for idx, (robot_id, tasks) in enumerate(robot_schedules.items()):
        color = trajectory_colors[idx]
        start = task_locations[0]

        # Ensure tasks are sorted by start_time
        tasks_sorted = sorted(tasks, key=lambda x: x[1])

        for task_id, _, _ in tasks_sorted:
            end = task_locations[task_id]
            draw_arrow(start, end, color, label=f"Robot {robot_id}" if start is task_locations[0] else "")
            start = end

        end = task_locations[-1]
        draw_arrow(start, end, color)

    # Add legend for skills
    legend_patches = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10, label=f"Skill {i}")
        for i in range(n_skills)
    ]
    ax.legend(handles=legend_patches, title="Task Skills", loc="upper right")

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("Robot Trajectories with Task Skill Representation")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    plt.show()
