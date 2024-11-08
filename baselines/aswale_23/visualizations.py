import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def prepare_data_for_gantt_chart(robots, tasks, X, Y_max, T_execution):
    # Prepare data for Gantt chart
    task_colors = {}  
    robot_tasks = {i: [] for i in robots} 
    color_pool = plt.cm.get_cmap('hsv', len(tasks))

    for i in robots:
        for k in tasks[:-1]:
            # Check if robot i visits task k
            if any(X[i][j][k] == 1 for j in tasks if j != k):
                start_time = Y_max[k]
                end_time = start_time + T_execution[k]
                robot_tasks[i].append((start_time, end_time, k))
                if k not in task_colors:
                    task_colors[k] = color_pool(k)

    return robot_tasks, task_colors


def plot_gantt_chart(title, robots, robot_tasks, task_colors):
    """Displays a Gantt chart showing the tasks assigned to each robot over time."""
    fig, ax = plt.subplots(figsize=(10, 6))

    yticks = []
    yticklabels = []
    legend_elements = []

    for idx, i in enumerate(reversed(robots)):
        yticks.append(idx)
        yticklabels.append(f"Robot {i}")
        for task in robot_tasks[i]:
            start_time, end_time, k = task
            ax.barh(
                idx,
                end_time - start_time,
                left=start_time,
                height=0.4,
                align='center',
                color=task_colors[k],
                edgecolor='black',
            )
            ax.text(
                start_time + (end_time - start_time) / 2,
                idx,
                f"Task {k}",
                va='center',
                ha='center',
                color='black',
                fontsize=6,  # Smaller font size for a large number of tasks
            )
            # Add to legend if not already added
            if k not in [e.get_label() for e in legend_elements]:
                legend_elements.append(
                    Patch(facecolor=task_colors[k], edgecolor='black', label=f'Task {k}')
                )

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel('Time')
    ax.set_title(title)
    ax.grid(True)
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show(block=False)  # Displays the Gantt chart without blocking


def show_problem_instance(problem_instance):
    
    Q = problem_instance['Q']
    R = problem_instance['R']
    robots = range(Q.shape[0])
    skills = range(Q.shape[1])
    n_tasks = R.shape[0] - 2


    """Displays a table with robot capabilities and task requirements."""
    fig, ax = plt.subplots(figsize=(8, 4))  # Smaller figure for the table
    ax.axis('off')

    # Build table data
    robot_info = [
        f"Robot {i}: " + ", ".join([f"Skill {s}" for s in skills if Q[i][s] == 1])
        for i in robots
    ]
    task_info = [
        f"Task {k}: " + ", ".join([f"Skill {s}" for s in skills if R[k][s] == 1])
        for k in range(1, n_tasks + 1)
    ]
    max_length = max(len(robot_info), len(task_info))
    robot_info += [""] * (max_length - len(robot_info))  # Pad with empty strings if robots are fewer
    task_info += [""] * (max_length - len(task_info))    # Pad with empty strings if tasks are fewer

    table_data = [["Robots Capabilities", "Tasks Requirements"]] + list(zip(robot_info, task_info))

    # Create the table
    table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)  # Smaller font size for better fit
    table.scale(1, 1.5)    # Scale the table for better layout

    plt.tight_layout()
    plt.show()  # Displays the table


def plot_task_map(task_locations, T_execution, R):
    marker_sizes = T_execution[1:-1] * 2
    
    plt.figure(figsize=(8, 8))
    plt.scatter(task_locations[1:-1, 0], task_locations[1:-1, 1], s=marker_sizes, label="Tasks")
    plt.scatter(task_locations[0, 0], task_locations[0, 1], color='green', s=150, marker = 'x', label="Start (Task 0)")
    plt.scatter(task_locations[-1, 0], task_locations[-1, 1], color='red', s=150, marker = 'x', label="End (Task -1)")
    
    for idx, (x, y) in enumerate(task_locations[1:-1], start=1):
        skills_required = [str(skill) for skill, needed in enumerate(R[idx]) if needed == 1]
        plt.text(x, y+2, f"{', '.join(skills_required)}", fontsize=10, ha='center')
    
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.title("Task Map with Scaled Execution Times")
    plt.show()

def plot_robot_trajectories(task_locations, task_assignments, T_execution, R):
    def draw_arrow(start, end, color, label=""):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        adjusted_dx, adjusted_dy = dx / length * (length - 5), dy / length * (length - 5)
        plt.arrow(start[0], start[1], adjusted_dx, adjusted_dy, head_width=2, head_length=5,
                  fc=color, ec=color, alpha=0.5, label=label)

    marker_sizes = T_execution[1:-1] * 2

    plt.figure(figsize=(10, 10))
    plt.scatter(task_locations[1:-1, 0], task_locations[1:-1, 1], s=marker_sizes, label="Tasks")
    plt.scatter(task_locations[0, 0], task_locations[0, 1], color='green', s=150, marker = 'x', label="Start (Task 0)")
    plt.scatter(task_locations[-1, 0], task_locations[-1, 1], color='red', s=150, marker = 'x', label="End (Task -1)")

    colors = plt.cm.Set1(np.linspace(0, 1, len(task_assignments.keys())))

    for idx, (robot_id, tasks) in enumerate(task_assignments.items()):
        color = colors[idx]
        start = task_locations[0]
        
        for _, _, task_id in tasks:
            end = task_locations[task_id]
            draw_arrow(start, end, color, label=f"Robot {robot_id}" if start is task_locations[0] else "")
            start = end
        
        end = task_locations[-1]
        draw_arrow(start, end, color)

    for idx, (x, y) in enumerate(task_locations[1:-1], start=1):
        skills_required = [str(skill) for skill, needed in enumerate(R[idx]) if needed == 1]
        plt.text(x, y+2, f"{', '.join(skills_required)}", fontsize=10, ha='center')


    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.title("Robot Trajectories for Fulfilled Tasks")
    plt.show()