"""Implementation inspired by the paper "Heterogeneous Coalition Formation and Scheduling with Multi-Skilled Robots", Aswale 2023



TODO:
- Algorithm 1 to reject looping candidate solutions  (maybe lazy constraint as callback funciton )
- (to align with paper: Stochastic task execution times (maybe not needed))
"""

import matplotlib.pyplot as plt
import numpy as np
import pulp
from problem_generator import ProblemData, generate_random_data
from visualizations import plot_gantt_chart

# Define parameters
n_tasks = 6 # Total number of tasks (m)
n_robots = 3  # Total number of robots (n)
n_skills = 2  # Total number of skills (l)
M = n_robots  # Large constant for if-else constraints
M_skills = n_skills  # Large constant for if-else constraints

robots = range(n_robots)
skills = range(n_skills)
tasks = range(n_tasks + 2)  # Tasks 0 and m+1 are starting and ending points

np.random.seed(44)

# Generate random data
problem_instance: ProblemData = generate_random_data(n_tasks, n_robots, n_skills)

Q = problem_instance['Q']
R = problem_instance['R']
T_e = problem_instance['T_e']
T_t = problem_instance['T_t']

M_time = np.sum(T_e) * n_robots + np.sum(T_t)

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
# where Z[k][s] indicates number of robots with skill s for task k (eq 10)
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

# # Each robot that attends a task must have at leat one skill that is not in excess (eq 13)
for i in robots:
    for k in range(1, n_tasks + 1):
        # x_ik: Whether robot i attends task k
        x_ik = pulp.lpSum([X[i][j][k] for j in tasks if j != k])

        # s_ik: Number of redundant skills robot i has for task k
        s_ik = pulp.lpSum([Z_b[k][s] * Q[i][s] for s in skills])

        # r_ik: Number of required skills robot i has for task k that robot i possesses
        r_ik = pulp.lpSum([Q[i][s] * R[k][s] for s in skills])

        prob += (
            s_ik <= r_ik - 1 + M_skills * (1 - x_ik),
            f"Skill_Not_Excessive_{i}_{k}",
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
prob.solve(pulp.PULP_CBC_CMD(timeLimit=300)) 
print("Status:", pulp.LpStatus[prob.status])

# Check if the problem is feasible
if pulp.LpStatus[prob.status] == 'Optimal':
    # Print the objective value
    print("Total time to complete all tasks:", pulp.value(prob.objective))

    # Prepare data for Gantt chart
    task_colors = {}  
    robot_tasks = {i: [] for i in robots} 
    color_pool = plt.cm.get_cmap('hsv', n_tasks + 2)

    for i in robots:
        for k in tasks[:-1]:
            # Check if robot i visits task k
            if any(pulp.value(X[i][j][k]) > 0.5 for j in tasks if j != k):
                start_time = pulp.value(Y_max[k])
                end_time = start_time + T_e[k]
                robot_tasks[i].append((start_time, end_time, k))
                if k not in task_colors:
                    task_colors[k] = color_pool(k)

    
    plot_gantt_chart(robots, tasks, robot_tasks, task_colors, Q, R, n_tasks, skills)
else:
    print("No feasible solution found.")
