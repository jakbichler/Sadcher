# visualizations.py

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_gantt_chart(robots, tasks, robot_tasks, task_colors, Q, R, n_tasks, skills):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

    yticks = []
    yticklabels = []
    legend_elements = []

    for idx, i in enumerate(robots):
        yticks.append(idx)
        yticklabels.append(f"Robot {i}")
        for task in robot_tasks[i]:
            start_time, end_time, k = task
            ax1.barh(
                idx,
                end_time - start_time,
                left=start_time,
                height=0.4,
                align='center',
                color=task_colors[k],
                edgecolor='black',
            )
            ax1.text(
                start_time + (end_time - start_time) / 2,
                idx,
                f"Task {k}",
                va='center',
                ha='center',
                color='black',
                fontsize=8,
            )
            # Add to legend if not already added
            if k not in [e.get_label() for e in legend_elements]:
                legend_elements.append(
                    Patch(facecolor=task_colors[k], edgecolor='black', label=f'Task {k}')
                )

    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticklabels)
    ax1.set_xlabel('Time')
    ax1.set_title('Gantt Chart of Robot Schedules')
    ax1.grid(True)
    ax1.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add explanation of robot capabilities and task requirements
    ax2.axis('off')  

    # Build table data
    robot_info = ["Robot {}: {}".format(i, ", ".join([f"Skill {s}" for s in skills if Q[i][s] == 1])) for i in robots]
    task_info = ["Task {}: {}".format(k, ", ".join([f"Skill {s}" for s in skills if R[k][s] == 1])) for k in range(1, n_tasks + 1)]
    max_length = max(len(robot_info), len(task_info))
    robot_info += [""] * (max_length - len(robot_info))  # Pad with empty strings if robots are fewer
    task_info += [""] * (max_length - len(task_info))    # Pad with empty strings if tasks are fewer

    table_data = [["Robots Capabilities", "Tasks Requirements"]] + list(zip(robot_info, task_info))

    # Create the table
    table = ax2.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2) 

    plt.tight_layout()
    plt.show()

