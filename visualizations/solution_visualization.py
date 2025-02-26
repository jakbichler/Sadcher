import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.patches import Wedge


def plot_gantt_chart(title, schedule, travel_times=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    task_colors = plt.cm.get_cmap('hsv', schedule.n_tasks + 2)
    travel_times = np.array(travel_times)

    yticks = []
    yticklabels = []
    legend_elements = []

    for idx, robot in enumerate(reversed(range(schedule.n_robots))):
        yticks.append(idx)
        yticklabels.append(f"Robot {robot}")
        
        tasks_for_robot = schedule.robot_schedules[robot]
        
        for i, (task_i, start_i, end_i) in enumerate(tasks_for_robot):
            ax.barh(
                idx,
                end_i - start_i,
                left=start_i,
                height=0.4,
                align='center',
                color=task_colors(task_i),
                edgecolor='black',
            )
            ax.text(
                start_i + (end_i - start_i) / 2,
                idx,
                f"Task {task_i}",
                va='center',
                ha='center',
                color='black',
                fontsize=6,
            )
            if task_i not in [e.get_label() for e in legend_elements]:
                legend_elements.append(
                    Patch(facecolor=task_colors(task_i), edgecolor='black', label=f'Task {task_i}')
                )

            # Add arrows for travel times between normal tasks
            if i < len(tasks_for_robot)- 1:
                (task_j, start_j, _) = tasks_for_robot[i + 1]
                t_ij = travel_times[task_i, task_j]
                arrow_start = start_j - t_ij
                arrow_len = t_ij
                
                if arrow_len > 0:
                    ax.arrow(
                        arrow_start,
                        idx,
                        arrow_len,
                        0,
                        length_includes_head=True,
                        head_width=0.03,
                        head_length=10,
                        linewidth=3,
                        color='black',
                        shape='full',
                    )

            first_task, first_start, _ = tasks_for_robot[0]
            arrow_start = first_start - travel_times[0, first_task]
            arrow_len = travel_times[0, first_task]
            if arrow_len > 0:
                ax.arrow(
                    arrow_start,            # starting at time 0
                    idx,
                    arrow_len,
                    0,
                    length_includes_head=True,
                    head_width=0.03,
                    head_length=10,
                    linewidth=3,
                    color='black',
                    shape='full',
                )

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel('Time')
    ax.set_title(title)
    ax.grid(True)
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(0, schedule.makespan + 10)
    ax.axvline(x=schedule.makespan, color='red', linestyle='--', label='Makespan')

    if ax is None:
        plt.tight_layout()
        plt.show()



def plot_robot_trajectories(task_locations, robot_schedules, T_execution, R, ax=None, Q=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    def draw_arrow(start, end, color, label=""):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        adjusted_dx, adjusted_dy = dx / length * (length - 5), dy / length * (length - 5)
        ax.arrow(start[0], start[1], adjusted_dx, adjusted_dy, head_width=2, head_length=5,
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
    colors = plt.cm.Set1(np.linspace(0, 1, n_skills))

    for idx, (x, y) in enumerate(task_locations[1:-1], start=1):
        skills_required = R[idx]
        total_skills = np.sum(skills_required)
        skill_sizes = skills_required / total_skills if total_skills > 0 else np.zeros_like(skills_required)
        draw_pie(ax, x, y, skill_sizes, marker_sizes[idx - 1] / 100)
        ax.text(x, y+2, f"Task {idx}", fontsize=10, ha='right')

    ax.scatter(task_locations[0, 0], task_locations[0, 1], color='green', s=150, marker='x', label="Start (Task 0)")
    ax.text(task_locations[0, 0] + 6, task_locations[0, 1] - 1, "Start", fontsize=12, ha='center')
    ax.scatter(task_locations[-1, 0], task_locations[-1, 1], color='red', s=150, marker='x', label="End (Task -1)")
    ax.text(task_locations[-1, 0] + 6, task_locations[-1, 1] - 1, "End", fontsize=12, ha='center')

    trajectory_colors = ["black", 'royalblue', 'darkorange', 'green']
    for idx, (robot_id, tasks) in enumerate(robot_schedules.items()):
        color = trajectory_colors[idx]
        start = task_locations[0]

        tasks_sorted = sorted(tasks, key=lambda x: x[1])

        for task_id, _, _ in tasks_sorted:
            end = task_locations[task_id]
            draw_arrow(start, end, color, label=f"Robot {robot_id}" if start is task_locations[0] else "")
            start = end

        end = task_locations[-1]
        draw_arrow(start, end, color)

    legend_patches_skills = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10, label=f"Skill {i}")
        for i in range(n_skills)
    ]
    legend_patches_robots = [
        plt.Line2D([0], [0], color=trajectory_colors[i], lw=2, label=f"R{robot_id}: {Q[robot_id]}")
        for i, robot_id in enumerate(robot_schedules.keys())
    ]

    legend_patches = legend_patches_skills + legend_patches_robots

    ax.legend(handles=legend_patches, title="Task Skills", loc="upper right")

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("Robot Trajectories with Task Skill Representation")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    if ax is None:
        plt.show()


def add_precedence_constraints_text(fig, precedence_constraints):
    """Display precedence constraints as a single line of text below the robot table."""
    ax_text = plt.axes([0.1, 0.0, 0.8, 0.05])  # Position below the robot table
    ax_text.axis("off")  # Hide the axes

    precedence_text = f"Precedence Constraints: {precedence_constraints}"
    ax_text.text(0.5, 0.5, precedence_text, ha='center', va='center', fontsize=10, wrap=True)


def plot_gantt_and_trajectories(title, schedule, travel_times, task_locations, T_execution, R, Q, precedence_constraints = None):
    fig, axs = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 5]})

    # Use the modified functions with specific axes
    plot_gantt_chart(title, schedule, travel_times, ax=axs[0])
    plot_robot_trajectories(task_locations, schedule.robot_schedules, T_execution, R, ax=axs[1], Q=Q)

    if precedence_constraints:
        add_precedence_constraints_text(fig, precedence_constraints)

    plt.show()
