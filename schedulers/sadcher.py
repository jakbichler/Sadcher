import numpy as np
import torch

from helper_functions.schedules import Instantaneous_Schedule
from imitation_learning.attention_models import SchedulerNetwork
from schedulers.bipartite_matching import (
    filter_overassignments,
    filter_redundant_assignments,
    solve_bipartite_matching,
)


class SadcherScheduler:
    def __init__(self, debugging, checkpoint_path, duration_normalization, location_normalization):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trained_model = SchedulerNetwork(
            robot_input_dimensions=7,
            task_input_dimension=9,
            embed_dim=128,
            ff_dim=256,
            n_transformer_heads=4,
            n_transformer_layers=4,
            n_gatn_heads=4,
            n_gatn_layers=2,
        ).to(self.device)
        self.trained_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        self.trained_model.eval()
        self.debug = debugging
        self.duration_normalization = duration_normalization
        self.location_normalization = location_normalization

    def calculate_robot_assignment(self, sim):
        n_robots = len(sim.robots)
        robot_assignments = {}
        # Special case for the last task
        available_robots = [robot for robot in sim.robots if robot.available]
        incomplete_tasks = [
            task for task in sim.tasks if task.incomplete and task.status == "PENDING"
        ]

        # Check if all normal tasks are done -> send all robots to the exit task
        if len(incomplete_tasks) == 1:  # Only end task incomplete
            for robot in available_robots:
                robot_assignments[robot.robot_id] = incomplete_tasks[0].task_id
                robot.current_task = incomplete_tasks[0]
            return torch.zeros((n_robots, len(sim.tasks))), Instantaneous_Schedule(
                robot_assignments
            )

        task_features = np.array(
            [
                task.feature_vector(self.location_normalization, self.duration_normalization)
                for task in sim.tasks[1:-2]
            ]
        )
        task_features = (
            torch.tensor(task_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        robot_features = np.array(
            [
                robot.feature_vector(self.location_normalization, self.duration_normalization)
                for robot in sim.robots
            ]
        )
        robot_features = (
            torch.tensor(robot_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            predicted_reward_raw = self.trained_model(
                robot_features, task_features, sim.task_adjacency.to(self.device)
            ).squeeze(0)  # remove batch dim

        predicted_reward = torch.clamp(predicted_reward_raw, min=1e-6)
        predicted_reward[:, -1] = torch.clamp(predicted_reward[:, -1], max=4.0)

        # Add  negative rewards for for the start and end task --> not to be selected, will be handled by the scheduler
        reward_start_end = torch.ones(n_robots, 1).to(self.device) * (-1000)
        predicted_reward = torch.cat((reward_start_end, predicted_reward, reward_start_end), dim=1)

        # Only for debugging
        predicted_reward_raw = torch.cat(
            (reward_start_end, predicted_reward_raw, reward_start_end), dim=1
        )

        if self.debug:
            for robot_idx, robot in enumerate(sim.robots):
                for task_idx, task in enumerate(sim.tasks):
                    print(
                        f"Robot {robot_idx} -> Task {task_idx}: {predicted_reward_raw[robot_idx][task_idx]:.6f} -> {predicted_reward[robot_idx][task_idx]:.6f}"
                    )
                print("\n")

        bipartite_matching_solution = solve_bipartite_matching(predicted_reward, sim)
        if self.debug:
            print(bipartite_matching_solution)
        filtered_solution = filter_redundant_assignments(bipartite_matching_solution, sim)
        if self.debug:
            print(filtered_solution)
        filtered_solution = filter_overassignments(filtered_solution, sim)
        if self.debug:
            print(filtered_solution)
            print("##############################################\n\n")

        robot_assignments = {
            robot: task for (robot, task), val in filtered_solution.items() if val == 1
        }

        return predicted_reward, Instantaneous_Schedule(robot_assignments)
