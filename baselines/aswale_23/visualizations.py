import matplotlib.pyplot as plt
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
