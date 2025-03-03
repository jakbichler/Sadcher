import torch
from schedulers.bipartite_matching import solve_bipartite_matching, filter_overassignments, filter_redundant_assignments
from helper_functions.schedules import Instantaneous_Schedule


class RandomBipartiteMatchingScheduler:
    def __init__(self):
        pass

    def calculate_robot_assignment(self, sim):
        n_robots = len(sim.robots)
        n_tasks = len(sim.tasks)
        robot_assignments = {}
        # Special case for the last task
        idle_robots = [r for r in sim.robots if r.current_task is None or r.current_task.status == 'DONE']
        pending_tasks = [t for t in sim.tasks if t.status == 'PENDING']
        # Check if all tasks are done -> send all robots to the exit task
        if len(pending_tasks) == 1:
            for robot in idle_robots:
                robot_assignments[robot.robot_id] = pending_tasks[0].task_id
                robot.current_task = pending_tasks[0]
            return Instantaneous_Schedule(robot_assignments)

        # Create random reward matrix 
        R = torch.randint(0, 10, size=(n_robots, n_tasks-2))
        R = torch.cat((torch.zeros(n_robots, 1), R, torch.zeros(n_robots, 1)), dim=1)
        bipartite_matching_solution = solve_bipartite_matching(R, sim)
        filtered_solution = filter_redundant_assignments(bipartite_matching_solution, sim)
        filtered_solution = filter_overassignments(filtered_solution, sim)
        robot_assignments = {robot: task for (robot, task), val in filtered_solution.items() if val == 1}
        return Instantaneous_Schedule(robot_assignments)