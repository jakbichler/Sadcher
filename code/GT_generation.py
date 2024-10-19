import matplotlib.pyplot as plt
import numpy as np
import pulp
from matplotlib.patches import Patch

# Define parameters
n_tasks = 5  # Total number of tasks (m)
n_robots = 3  # Total number of robots (n)
n_skills = 2  # Total number of skills (l)
M = n_robots  # Large constant for if-else constraints

robots = range(n_robots)
skills = range(n_skills)
tasks = range(n_tasks + 2)  # Tasks 0 and m+1 are starting and ending points

# Q[i][s] = 1 if robot i has skill s, 0 otherwise
Q = [
    [1, 1],  # Robot 0
    [0, 1],  # Robot 1
    [1, 0],  # Robot 2
]

# R[k][s] = 1 if task k requires skill s, 0 otherwise
R = [
    [0, 0],  # Task 0 (start)
    [1, 0],  # Task 1
    [1, 0],  # Task 2
    [0, 1],  # Task 3
    [0, 1],  # Task 4
    [1, 1],  # Task 5
    [0, 0],  # Task 6 (end)
]

# Task execution times
T_e = np.array([0, 30, 40, 5, 6, 7, 0])  # Including start and end tasks

# Task travel times
T_t = np.ones((n_tasks + 2, n_tasks + 2))

M_time = np.sum(T_e) * n_robots + np.sum(T_t)

# Problem definition
prob = pulp.LpProblem("Coalition_Formation_and_Scheduling", pulp.LpMinimize)

# Variables
X = pulp.LpVariable.dicts(
    "X", (robots, tasks, tasks), 0, 1, pulp.LpBinary
)  # Xijk --> robot i attends task k after task j
Y = pulp.LpVariable.dicts("Y", (robots, tasks), 0, None)  # Yik (arrival times)
Y_max = pulp.LpVariable.dicts("Y_max", tasks, 0, None)  # Yk (latest arrival time)
Z = pulp.LpVariable.dicts("Z", (tasks, skills), 0, None)  # Zks (skills provided)
Z_b = pulp.LpVariable.dicts(
    "Z_b", (tasks, skills), 0, 1, pulp.LpBinary
)  # if 1 then skill s is excessive for task k

# Define the new variable T_max, which represents the maximum finish time
T_max = pulp.LpVariable("T_max", 0, None)

# Objective: Minimize the maximum time (T_max) at which the last robot finishes
prob += T_max, "MinimizeMaxCompletionTime"

# Constraints: T_max must be greater than or equal to the finish time of each robot at task m+1
for i in robots:
    prob += T_max >= Y[i][n_tasks + 1], f"T_max_Constraint_Robot_{i}"

for i in robots:
    # Each robot starts at task 0 (eq 1)
    prob += (
        pulp.lpSum([X[i][0][k] for k in range(1, n_tasks + 2)]) == 1,
        f"Start_Task_0_Robot_{i}",
    )

    # Each robot finishes at task m+1 (eq 2)
    prob += (
        pulp.lpSum([X[i][j][n_tasks + 1] for j in range(n_tasks + 1)]) == 1,
        f"Finish_Task_{n_tasks + 1}_Robot_{i}",
    )

    # Task 0 is exit only (eq 3)
    prob += (
        pulp.lpSum([X[i][j][0] for j in range(1, n_tasks + 2)]) == 0,
        f"Exit_Only_Task_0_Robot_{i}",
    )

    # Task m+1 is entry only (eq 4)
    prob += (
        pulp.lpSum([X[i][n_tasks + 1][k] for k in range(0, n_tasks + 1)]) == 0,
        f"Entry_Only_Task_{n_tasks + 1}_Robot_{i}",
    )

# Each robot can enter each task k\{0, n_tasks+1} at most once (eq 5)
for i in robots:
    for k in range(1, n_tasks + 1):
        prob += (
            pulp.lpSum([X[i][j][k] for j in range(n_tasks + 2)]) <= 1,
            f"Enter_Task_{k}_Robot_{i}",
        )

# Each robot can exit each task j\{0, n_tasks+1} at most once (eq 6)
for i in robots:
    for j in range(1, n_tasks + 1):
        prob += (
            pulp.lpSum([X[i][j][k] for k in range(n_tasks + 2)]) <= 1,
            f"Exit_Task_{j}_Robot_{i}",
        )

# Each task has to first be entered before being able to leave (eq 7)
for i in robots:
    for j in range(1, n_tasks + 1):
        prob += (
            pulp.lpSum([X[i][k][j] for k in range(n_tasks + 2)])
            == pulp.lpSum([X[i][j][k] for k in range(n_tasks + 2)]),
            f"Enter_Exit_Task_{j}_Robot_{i}",
        )

# A robot cannot revisit a task (eq 8)
for i in robots:
    for j in tasks:
        prob += X[i][j][j] == 0, f"No_Revisit_Task_{j}_Robot_{i}"

# Skill allocation
# Robot i must at least possess 1 of the required skills for task k\{0,m+1} (eq 9)
for i in robots:
    for k in range(1, n_tasks + 1):
        prob += (
            pulp.lpSum([X[i][j][k] for j in range(n_tasks + 2)])
            <= pulp.lpSum([Q[i][s] * R[k][s] for s in range(n_skills)]),
            f"Skill_Allocation_Task_{k}_Robot_{i}",
        )

# Matrix Z for number of robots with skill s for task k
# where Z[k][s] indicates number of robots with skill s for task k
for s in skills:
    for k in range(1, n_tasks + 1):
        prob += (
            Z[k][s]
            == pulp.lpSum(
                [
                    Q[i][s] * pulp.lpSum([X[i][j][k] for j in range(n_tasks + 2)])
                    for i in robots
                ]
            ),
            f"Z_{k}_{s}",
        )

        # All required skills for task k must be provided (eq 11)
        prob += Z[k][s] >= R[k][s], f"Skill_Provided_Task_{k}_Skill_{s}"

        # Identify superfluous robots (eq 12)
        prob += (
            Z[k][s] - R[k][s] - M * Z_b[k][s] <= 0,
            f"Superfluous_Skill_Upper_{k}_{s}",
        )
        prob += (
            Z[k][s] - R[k][s] - 1 + M * (1 - Z_b[k][s]) >= 0,
            f"Superfluous_Skill_Lower_{k}_{s}",
        )

# Arrival times
# If a robot does not visit a task, then the arrival time is 0 (eq 14)
for i in robots:
    for k in tasks:
        prob += (
            Y[i][k]
            <= M_time * pulp.lpSum([X[i][j][k] for j in tasks if j != k]),
            f"ArrivalTimeZero_{i}_{k}",
        )

# Task j starts at the arrival time of the last robot (eq 15)
# Constraint: y_j_max >= Y[i][j] for all robots i
for i in robots:
    for j in tasks:
        prob += Y_max[j] >= Y[i][j], f"Max_Arrival_Time_Task_{j}_Robot_{i}"

# Arrival time of robot i at task k is sum of completion time of previous task j and travel time between j and k (eq 16)
for i in robots:
    for j in tasks:
        for k in tasks:
            if j != k:
                prob += (
                    Y[i][k]
                    >= Y_max[j] + T_e[j] + T_t[j][k] - M_time * (1 - X[i][j][k]),
                    f"ArrivalTime_Update_LB_{i}_{j}_{k}",
                )
                prob += (
                    Y[i][k]
                    <= Y_max[j] + T_e[j] + T_t[j][k] + M_time * (1 - X[i][j][k]),
                    f"ArrivalTime_Update_UB_{i}_{j}_{k}",
                )

# Solve the problem
prob.solve()
# Print the status
print("Status:", pulp.LpStatus[prob.status])

# Check if the problem is feasible
if pulp.LpStatus[prob.status] == 'Optimal':
    # Print the objective value
    print("Total time to complete all tasks:", pulp.value(prob.objective))

    # Prepare data for Gantt chart
    task_colors = {}  # Assign colors to tasks
    robot_tasks = {i: [] for i in robots}  # Tasks for each robot
    color_pool = plt.cm.get_cmap('hsv', n_tasks + 2)

    for i in robots:
        for k in tasks:
            # Check if robot i visits task k
            if any(pulp.value(X[i][j][k]) > 0.5 for j in tasks if j != k):
                start_time = pulp.value(Y_max[k])
                end_time = start_time + T_e[k]
                robot_tasks[i].append((start_time, end_time, k))
                # Assign a color to the task if not already assigned
                if k not in task_colors:
                    task_colors[k] = color_pool(k)

    # Plot Gantt chart
    fig, ax = plt.subplots(figsize=(10, 6))

    yticks = []
    yticklabels = []
    legend_elements = []

    for idx, i in enumerate(robots):
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
                fontsize=8,
            )
            # Add to legend if not already added
            if k not in [e.get_label() for e in legend_elements]:
                legend_elements.append(
                    Patch(facecolor=task_colors[k], edgecolor='black', label=f'Task {k}')
                )

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel('Time')
    ax.set_title('Gantt Chart of Robot Schedules')
    ax.grid(True)

    # Create legend
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()
else:
    print("No feasible solution found.")