
import pulp

def solve_bipartite_matching(R, sim):
    """
    R    : np.array, shape [n_robots, n_tasks], reward matrix
    sim  : Simulation object
    Returns: dict of {(i, j): 0/1} solutions indicating the best A_{i,j}.
    """

    n_robots = len(sim.robots)
    n_tasks = len(sim.tasks)
    n_skills = len(sim.robots[0].capabilities)

    print(f"started bipartite matching with {n_robots} robots and {n_tasks} tasks")

    for task in sim.tasks:
        print(task.task_id, task.requirements)

    problem = pulp.LpProblem("BipartiteMatching", pulp.LpMaximize)

    # Decision variables: A[robot][task] in {0,1}
    A = pulp.LpVariable.dicts("A", (range(n_robots), range(n_tasks)),
                              lowBound=0, upBound=1, cat=pulp.LpBinary)

    # Objective: maximize total reward
    problem += pulp.lpSum(R[robot_idx][task_idx] * A[robot_idx][task_idx]
                          for robot_idx in range(n_robots)
                          for task_idx in range(n_tasks)), "TotalReward"

    for robot_idx, robot in enumerate(sim.robots):
        # Constraint: each available robot can take at most one task
        if robot.available:
            problem += pulp.lpSum(A[robot_idx][task] for task in range(n_tasks)) <= 1

        # unavailable robots cannot take any task
        else:
            for task in range(n_tasks):
                problem += A[robot_idx][task] == 0
        
    # Subteaming constraints 
    for task_idx, task in enumerate(sim.tasks):
        # Only constrain if this task is ready
        if task.ready and task.incomplete:
            # 1) Capability requirement: if c_t[j][p] = 1, subteam must have it
            for cap in range(n_skills):
                if task.requirements[cap] != 0:
                    problem += pulp.lpSum(sim.robots[robot_idx].capabilities[cap] * A[robot_idx][task_idx] for robot_idx in range(n_robots)) >= task.requirements[cap]

        else:
            # If task is not ready, force no assignment
            for robot_idx, robot in enumerate(sim.robots):
                problem += A[robot_idx][task_idx] == 0



    problem.solve(pulp.PULP_CBC_CMD(msg=0))
    print(f"solver status: {pulp.LpStatus[problem.status]}")
    solution = {(robot_idx, task_idx): int(pulp.value(A[robot_idx][task_idx])) for robot_idx in range(n_robots) for task_idx in range(n_tasks)}
    print(f"bipartite matching done with makespan {pulp.value(problem.objective)} and solution {solution}")
    robot_assignments = {robot: task for (robot, task), val in solution.items() if val == 1}
    print(f"robot assignments: {robot_assignments}")
    return solution