
import pulp
import numpy as np
from collections import defaultdict

def solve_bipartite_matching(R, sim):
    """
    R    : np.array, shape [n_robots, n_tasks], reward matrix
    sim  : Simulation object
    Returns: dict of {(i, j): 0/1} solutions indicating the best A_{i,j}.
    """

    n_robots = len(sim.robots)
    n_tasks = len(sim.tasks)
    n_skills = len(sim.robots[0].capabilities)

    problem = pulp.LpProblem("BipartiteMatching", pulp.LpMaximize)

    # Decision variables: A[robot][task] in {0,1}
    A = pulp.LpVariable.dicts("A", (range(n_robots), range(n_tasks)),
                              lowBound=0, upBound=1, cat=pulp.LpBinary)


    # Binary "activated" variables for the bipartite matching --> only checks requirements for tasks that will be scheduled this round
    X = pulp.LpVariable.dicts("X", range(n_tasks),
                              lowBound=0, upBound=1, cat=pulp.LpBinary)

    M_robots = n_robots
    
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

            # Link A to X with big-M constraints
            problem += pulp.lpSum(A[robot_idx][task_idx] for robot_idx in range(n_robots)) <= M_robots * X[task_idx]
            problem += pulp.lpSum(A[robot_idx][task_idx] for robot_idx in range(n_robots)) >= X[task_idx]


            # 1) Capability requirement: if c_t[j][p] = 1, subteam must have it
            for cap in range(n_skills):
                if task.requirements[cap] != 0:
                    problem += pulp.lpSum(sim.robots[robot_idx].capabilities[cap] * A[robot_idx][task_idx] for robot_idx in range(n_robots)) >= task.requirements[cap] * X[task_idx]

        else:
            # If task is not ready, force no assignment
            for robot_idx, robot in enumerate(sim.robots):
                problem += A[robot_idx][task_idx] == 0

    

    problem.solve(pulp.PULP_CBC_CMD(msg=0))
    #print(f"solver status: {pulp.LpStatus[problem.status]}")
    solution = {(robot_idx, task_idx): int(pulp.value(A[robot_idx][task_idx])) for robot_idx in range(n_robots) for task_idx in range(n_tasks)}
    #print(f"bipartite matching done with makespan {pulp.value(problem.objective)} and solution {solution}")
    robot_assignments = {robot: task for (robot, task), val in solution.items() if val == 1}
    #print(f"robot assignments: {robot_assignments}")
    return solution




def filter_redundant_assignments(assignment_solution, sim):
    """
    If a new assignment doesn't add any new skills beyond what's already 
    provided by the *existing set* of assigned robots, remove it.
    """
    filtered_solution = dict(assignment_solution)  # copy so we can modify

    for (robot_id, task_id), val in assignment_solution.items():
        if val == 1:
            # Find any robots currently assigned to this task
            existing_robots = [
                r for r in sim.robots 
                if r.current_task == sim.tasks[task_id]
            ]
            # If at least one robot is already on this task, 
            # check if their combined capabilities cover all requirements.
            if len(existing_robots) > 0:
                task = sim.tasks[task_id]
                combined_capabilities = np.zeros_like(task.requirements, dtype=bool)
                for rb in existing_robots:
                    combined_capabilities = np.logical_or(combined_capabilities, rb.capabilities)
                
                # If all requirements are covered by the existing sub-team:
                if np.all(combined_capabilities[task.requirements]):
                    # Then this new assignment doesn't add value; remove it.
                    filtered_solution[(robot_id, task_id)] = 0

    return filtered_solution


def filter_overassignments(assignment_solution, sim):
    """
    For each task:
    1) Gather the 'existing' robots that already have current_task == that task.
    2) Gather the 'new' robots assigned to that task in assignment_solution.
    3) Iteratively see if we already cover the full skill requirements. If so,
        any additional new robot is unnecessary and removed.
    """
    # Copy so we don’t mutate the original while iterating
    filtered_solution = dict(assignment_solution)

    # 1) Build a dictionary of task -> [list of newly assigned robot_ids]
    task_to_new_assignments = defaultdict(list)
    for (robot_id, task_id), val in assignment_solution.items():
        if val == 1:
            task_to_new_assignments[task_id].append(robot_id)

    # 2) Iterate over each task that got new assignments
    for task_id, new_robot_ids in task_to_new_assignments.items():
        task = sim.tasks[task_id]
        # If task is incomplete and ready, we want to see if the sub-team is needed
        if not (task.ready and task.incomplete):
            continue

        # - Already assigned (existing) robots
        existing_robots = [
            r for r in sim.robots 
            if r.current_task == task
        ]

        # Combine existing coverage
        combined_capabilities = np.zeros_like(task.requirements, dtype=bool)
        for r in existing_robots:
            combined_capabilities = np.logical_or(combined_capabilities, r.capabilities)

        # 3) For each newly assigned robot, check if they add coverage
        # We do this in the order we see them, but you can choose a different strategy if you like
        for robot_id in new_robot_ids:
            # If we already fully cover the task’s requirements, no need for another robot
            if np.all(combined_capabilities[task.requirements]):
                filtered_solution[(robot_id, task_id)] = 0
            else:
                # This robot might add something, so incorporate its skills
                robot_cap = sim.robots[robot_id].capabilities
                combined_capabilities = np.logical_or(combined_capabilities, robot_cap)

    return filtered_solution