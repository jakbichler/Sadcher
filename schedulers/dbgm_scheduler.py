from collections import defaultdict
import numpy as np
import torch
from schedulers.bigraph_matching import solve_bipartite_matching
from method_explorations.LVWS.transformer_models import TransformerScheduler
from helper_functions.schedules import Instantaneous_Schedule



class DBGMScheduler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trained_model = TransformerScheduler(robot_input_dimensions=3, task_input_dimension=5, embed_dim=64, ff_dim=128, num_layers=2).to(self.device)
        self.trained_model.load_state_dict(torch.load("/home/jakob/thesis/method_explorations/LVWS/checkpoints/checkpoint_epoch_20.pt", weights_only=True)
                                    )
    def assign_tasks_to_robots(self, sim):
        n_robots = len(sim.robots)
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

        task_features = [task.feature_vector() for task in sim.tasks[1:-1]]
        task_features = torch.tensor(task_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        robot_features = [robot.feature_vector() for robot in sim.robots]
        robot_features = torch.tensor(robot_features, dtype= torch.float32).unsqueeze(0).to(self.device)


        
        predicted_reward = self.trained_model(robot_features, task_features).squeeze()


        # Add  negative rewards for for the start and end task --> not to be selected, will be handled by the scheduler
        reward_start_end = torch.ones(n_robots, 1).to(self.device) * (-1000)
        predicted_reward = torch.cat((reward_start_end, predicted_reward, reward_start_end), dim=1)


        for robot_idx, robot in enumerate(sim.robots):
            for task_idx, task in enumerate(sim.tasks):
                print(f"Robot {robot_idx} -> Task {task_idx}: {predicted_reward[robot_idx][task_idx]:.2f}")

            print("\n")

        bipartite_matching_solution = solve_bipartite_matching(predicted_reward, sim)
        print(bipartite_matching_solution)
        filtered_solution = self.filter_redundant_assignments(bipartite_matching_solution, sim)
        print(filtered_solution)
        filtered_solution = self.filter_overassignments(filtered_solution, sim)
        print(filtered_solution)
        robot_assignments = {robot: task for (robot, task), val in filtered_solution.items() if val == 1}

        return Instantaneous_Schedule(robot_assignments)
    

    def filter_redundant_assignments(self, assignment_solution, sim):
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


    def filter_overassignments(self, assignment_solution, sim):
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