from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from schedulers.bigraph_matching import solve_bipartite_matching, filter_redundant_assignments, filter_overassignments
from method_explorations.DBGM.transformer_models import SchedulerNetwork
from helper_functions.schedules import Instantaneous_Schedule



class DBGMScheduler:
    def __init__(self, debugging, checkpoint_path, duration_normalization, location_normalization):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trained_model = SchedulerNetwork(robot_input_dimensions=6, task_input_dimension=8, 
                                              embed_dim=128, ff_dim=256, n_transformer_heads=4, 
                                              n_transformer_layers= 4, n_gatn_heads=4, n_gatn_layers=2).to(self.device)
        self.trained_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        self.debug = debugging
        self.duration_normalization = duration_normalization
        self.location_normalization = location_normalization


    def assign_tasks_to_robots(self, sim):

        IDLE_PROBABILITY_THRESHOLD = 0.8


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
            return torch.zeros((n_robots, len(sim.tasks))), Instantaneous_Schedule(robot_assignments)

        task_features = np.array([task.feature_vector(self.location_normalization, self.duration_normalization) for task in sim.tasks[1:-2]])
        task_features = torch.tensor(task_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        robot_features = np.array([robot.feature_vector(self.location_normalization, self.duration_normalization) for robot in sim.robots])
        robot_features = torch.tensor(robot_features, dtype= torch.float32).unsqueeze(0).to(self.device)

        predicted_reward_raw = self.trained_model(robot_features, task_features).squeeze(0) # remove batch dim

        # Clamp negative values, since BGM will not work with only negative values
        predicted_reward = torch.clamp(predicted_reward_raw, min=1e-6)

        predicted_task_rewards = predicted_reward[:, :-1]
        predicted_idle_probability = predicted_reward[:, -1]

        if self.debug:
            print(predicted_task_rewards)
            print(predicted_idle_probability)


        # Add  negative rewards for for the start and end task --> not to be selected, will be handled by the scheduler
        reward_start_end = torch.ones(n_robots, 1).to(self.device) * (-1000)
        masked_reward_idle = torch.ones(n_robots, 1).to(self.device) * (-1000)

        #predicted_reward = torch.cat((reward_start_end, predicted_reward, reward_start_end), dim=1)
        predicted_reward = torch.cat((reward_start_end, predicted_task_rewards, masked_reward_idle, reward_start_end), dim=1)
        
        #Only for debugging
        predicted_reward_raw = torch.cat((reward_start_end, predicted_reward_raw, reward_start_end), dim=1)

        if self.debug:
            for robot_idx, robot in enumerate(sim.robots):
                for task_idx, task in enumerate(sim.tasks):
                    print(f"Robot {robot_idx} -> Task {task_idx}: {predicted_reward_raw[robot_idx][task_idx]:.6f} -> {predicted_reward[robot_idx][task_idx]:.6f}")
                print("\n")
        
        bipartite_matching_solution = solve_bipartite_matching(predicted_reward, sim)

        if self.debug:
            print(bipartite_matching_solution)
        filtered_solution = replace_assignment_with_idling(bipartite_matching_solution, predicted_idle_probability, IDLE_PROBABILITY_THRESHOLD)
        if self.debug:
            print(filtered_solution)
        filtered_solution = filter_redundant_assignments(filtered_solution, sim)
        if self.debug:
            print(filtered_solution)
        filtered_solution = filter_overassignments(filtered_solution, sim)
        if self.debug:
            print(filtered_solution)
            print("##############################################\n\n")
        
        robot_assignments = {robot: task for (robot, task), val in filtered_solution.items() if val == 1}
        

        return predicted_reward, Instantaneous_Schedule(robot_assignments)
    

def replace_assignment_with_idling(robot_assignments, predicted_idle_probability, idle_threshold):
    """
    If for a given robot the idle_probability is higher than idle_threshold,
    *and* the robot is the only one assigned to that task,
    set all assignments for that robot to 0, then assign 1 to the
    "idle" (second-to-last) task.

    Args:
        robot_assignments (dict): {(robot_id, task_id): 0 or 1}.
        predicted_idle_probability (Tensor or np.ndarray): shape (n_robots,).
        idle_threshold (float): threshold for deciding to force idle.

    Returns:
        dict: A modified dictionary of robot assignments.
    """
    # Copy the assignments to avoid mutating the input directly.
    modified_assignments = dict(robot_assignments)

    if not modified_assignments:
        return modified_assignments

    # Find the maximum task ID to infer "idle" as second-to-last
    max_task_id = max(task for (_, task) in modified_assignments.keys())
    idle_task_id = max_task_id - 1  # "Idle" by convention

    # 1) Determine which task each robot is currently assigned to (if any).
    #    Also tally how many robots are assigned to each task.
    robot_to_task = {}
    task_count = {}

    for (r, t), val in modified_assignments.items():
        if val == 1: 
            robot_to_task[r] = t
            task_count[t] = task_count.get(t, 0) + 1

    n_robots = len(predicted_idle_probability)

    # 2) For each robot:
    #    - If idle_probability[r] > idle_threshold
    #    - AND that robot is the *only* occupant of its assigned task
    #      => reassign to idle.
    for r in range(n_robots):
        if predicted_idle_probability[r] > idle_threshold:
            # Check if the robot has an assigned task, and how many occupy it
            if r in robot_to_task:
                assigned_task = robot_to_task[r]
                num_assignees_for_that_task = task_count.get(assigned_task, 0)

                # Only if exactly one occupant is assigned => reassign to idle
                if num_assignees_for_that_task == 1:
                    # Zero out all assignments for this robot
                    for t in range(max_task_id + 1):
                        if (r, t) in modified_assignments:
                            modified_assignments[(r, t)] = 0
                    # Now set the idle task to 1
                    modified_assignments[(r, idle_task_id)] = 1

    return modified_assignments