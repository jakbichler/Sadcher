"""Implementation inspired by the paper "Heterogeneous Coalition Formation and Scheduling with Multi-Skilled Robots", Aswale 2023
https://arxiv.org/abs/2306.11936

--> precedence constraints are added, which the paper does not mention


TODO:
- Algorithm 1/subtour elimination to reject looping candidate solutions  (maybe lazy constraint as callback funciton )
- Stochastic task execution times (if needed))
"""

import matplotlib.pyplot as plt
import numpy as np
import pulp
from data_generation.problem_generator import read_problem_instance
from helper_functions.schedules import Full_Horizon_Schedule


def milp_scheduling(problem_instance, n_threads = 2):
    Q, R, T_execution, T_travel, task_locations, precedence_constraints = read_problem_instance(problem_instance)
    n_robots = Q.shape[0]
    n_tasks = R.shape[0] - 2
    n_skills = Q.shape[1]

    robots = range(n_robots)
    tasks = range(n_tasks + 2)
    skills = range(n_skills)

    M_time = 2 * (np.sum(T_execution) + np.sum(T_travel))

    prob = pulp.LpProblem("Coalition_Formation_and_Scheduling", pulp.LpMinimize)
    X = pulp.LpVariable.dicts(
        "X", (robots, tasks, tasks), 0, 1, pulp.LpBinary
    )  # Xijk --> robot i attends task k after task j
    Y = pulp.LpVariable.dicts("Y", (robots, tasks), 0, None)  # Yik (arrival times)
    Y_max = pulp.LpVariable.dicts("Y_max", tasks, 0, None)  # Yk (latest arrival time)
    Z = pulp.LpVariable.dicts("Z", (tasks, skills), 0, None, cat = pulp.LpInteger)  # Zks (skills provided)
    Z_b = pulp.LpVariable.dicts(
        "Z_b", (tasks, skills), 0, 1, pulp.LpBinary
    )  # if 1 then skill s is excessive for task k

    T_max = pulp.LpVariable("T_max", 0, None) # last finish time of task

    # Objective: Minimize makespans
    prob += T_max, "MinimizeMaxCompletionTime"

    # Constraints: T_max must be greater than or equal to the finish time of each robot at task m+1
    for i in robots:
        prob += T_max >= Y[i][n_tasks + 1] + T_execution[n_tasks + 1], f"T_max_Constraint_Robot_{i}"

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
                pulp.lpSum([X[i][j][k] for j in range(n_tasks + 1)]) <= 1,
                f"Enter_Task_{k}_Robot_{i}",
            )

    # Each robot can exit each task j\{0, n_tasks+1} at most once (eq 6)
    for i in robots:
        for j in range(1, n_tasks + 1):
            prob += (
                pulp.lpSum([X[i][j][k] for k in range(1, n_tasks + 2)]) <= 1,
                f"Exit_Task_{j}_Robot_{i}",
            )

    # Each task has to first be entered before being able to leave (eq 7)
    for i in robots:
        for j in range(1, n_tasks + 1):
            prob += (
                pulp.lpSum([X[i][k][j] for k in range(n_tasks + 1)])
                == pulp.lpSum([X[i][j][k] for k in range(1, n_tasks + 2)]),
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
                pulp.lpSum([X[i][j][k] for j in range(n_tasks + 1)])
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
                        Q[i][s] * pulp.lpSum([X[i][j][k] for j in range(n_tasks + 1)])
                        for i in robots
                    ]
                ),
                f"Z_{k}_{s}",
            )

            # All required skills for task k must be provided (eq 11)
            prob += Z[k][s] >= R[k][s], f"Skill_Provided_Task_{k}_Skill_{s}"

            # Identify superfluous robots (eq 12)
            prob += (
                Z[k][s] - R[k][s] - n_robots * Z_b[k][s] <= 0,
                f"Superfluous_Skill_Upper_{k}_{s}",
            )
            prob += (
                Z[k][s] - R[k][s] - 1 + n_robots * (1 - Z_b[k][s]) >= 0,
                f"Superfluous_Skill_Lower_{k}_{s}",
            )

    # Each robot that attends a task must have at least one skill that is not in excess (eq 13)
    for i in robots:
        for k in range(1, n_tasks + 1):
            # x_ik: Whether robot i attends task k
            x_ik = pulp.lpSum([X[i][j][k] for j in tasks if j != k])

            # c_ik: Number of required skills robot i has for task k that are not redundant
            c_ik = pulp.lpSum([Q[i][s] * R[k][s] * (1 - Z_b[k][s]) for s in skills])

            prob += c_ik >= x_ik, f"NonRedundantSkill_{i}_{k}"


    # If a robot does not visit a task, then the arrival time is 0 (eq 14)
    for i in robots:
        for k in range(1, n_tasks+2):
            prob += (
                Y[i][k]
                <= M_time * pulp.lpSum([X[i][j][k] for j in tasks if j != k]),
                f"ArrivalTimeZero_{i}_{k}",
            )

    # Task j starts at the arrival time of the last robot (eq 15)
    # Constraint: y_j_max >= Y[i][j] for all robots i
    for j in range(1, n_tasks + 2):
        for i in robots:
            prob += Y_max[j] >= Y[i][j], f"Max_Arrival_Time_Task_{j}_Robot_{i}"

    # Arrival time of robot i at task k is sum of completion time of previous task j and travel time between j and k (eq 16)
    for i in robots:
        for j in range(0, n_tasks + 1):
            for k in range(1, n_tasks + 2):
                prob += (
                    Y[i][k]
                    >= Y_max[j] + T_execution[j] + T_travel[j][k] - M_time * (1 - X[i][j][k]),
                    f"ArrivalTime_Update_LB_{i}_{j}_{k}",
                )
                prob += (
                    Y[i][k]
                    <= Y_max[j] + T_execution[j] + T_travel[j][k] + M_time * (1 - X[i][j][k]),
                    f"ArrivalTime_Update_UB_{i}_{j}_{k}",
                )

    # Precedence constraints
    for j, k in precedence_constraints:
        prob += Y_max[k] >= Y_max[j] + T_execution[j], f"Precedence_Task_{j}_before_Task_{k}"

    # # Subtour elimination
    # for i in robots:
    #     for j in range(1, n_tasks + 1):
    #         for k in range(1, n_tasks + 1):
    #             if j != k:
    #                 prob += U[i][j] - U[i][k] + (n_tasks + 1) * X[i][j][k] <= n_tasks, f"Subtour_Elimination_{i}_{j}_{k}"


    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(timeLimit=60*10, msg = False, threads = n_threads)) 
    print("Status:", pulp.LpStatus[prob.status])
    # Check if the problem is feasible
    if pulp.LpStatus[prob.status] in ['Optimal', 'Feasible']:
        makespan = pulp.value(prob.objective)
        print(f"MILP time to complete all tasks: {makespan}")

        robot_schedules = {robot: [] for robot in robots}

        for robot in robots:
            for task in tasks:
                # Check if robot i visits task k
                if any(pulp.value(X[robot][previous_task][task]) > 0.5 for previous_task in tasks if previous_task != task):
                    start_time = pulp.value(Y_max[task])
                    end_time = start_time + T_execution[task]

                    # Exclude start and end tasks
                    if task != 0 and task != n_tasks + 1:
                        robot_schedules[robot].append((task, start_time, end_time))

    else:
        print("No feasible solution found.")

    return Full_Horizon_Schedule(makespan, robot_schedules, n_tasks)