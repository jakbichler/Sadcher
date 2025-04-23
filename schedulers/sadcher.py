import numpy as np
import torch

from helper_functions.schedules import Instantaneous_Schedule
from models.scheduler_network import SchedulerNetwork
from schedulers.bipartite_matching import CachedBipartiteMatcher
from schedulers.filtering_assignments import filter_overassignments, filter_redundant_assignments


class SadcherScheduler:
    def __init__(
        self,
        debugging,
        checkpoint_path,
        duration_normalization,
        location_normalization,
        model_name="8t3r3s",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_name == "6t2r2s":
            self.trained_model = SchedulerNetwork(
                robot_input_dimensions=6,
                task_input_dimension=8,
                embed_dim=256,
                ff_dim=512,
                n_transformer_heads=8,
                n_transformer_layers=1,
                n_gatn_heads=2,
                n_gatn_layers=1,
            ).to(self.device)

        elif model_name == "8t3r3s":
            self.trained_model = SchedulerNetwork(
                robot_input_dimensions=7,
                task_input_dimension=9,
                embed_dim=256,
                ff_dim=512,
                n_transformer_heads=4,
                n_transformer_layers=2,
                n_gatn_heads=8,
                n_gatn_layers=1,
            ).to(self.device)

        else:
            raise ValueError("Invalid model name")

        self.trained_model.eval()
        self.debug = debugging
        self.duration_normalization = duration_normalization
        self.location_normalization = location_normalization
        self.load_model_weights(checkpoint_path, debugging)
        self.bipartite_matcher = None

    def calculate_robot_assignment(self, sim):
        n_robots = len(sim.robots)
        robot_assignments = {}
        # Special case for the last task
        available_robots = [robot for robot in sim.robots if robot.available]
        available_robot_ids = [robot.robot_id for robot in sim.robots if robot.available]
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
                # print the feature vecotors with explanation what is what
                print(
                    f"Robot {robot_idx} feature vector: {robot_features[0][robot_idx].cpu().numpy()}"
                )

        if self.bipartite_matcher is None:
            self.bipartite_matcher = CachedBipartiteMatcher(sim)
        bipartite_matching_solution = self.bipartite_matcher.solve(
            predicted_reward, n_threads=6, gap=0.0
        )

        if self.debug:
            print(
                *[
                    task_robot_pair
                    for task_robot_pair, assigned in bipartite_matching_solution.items()
                    if assigned == 1
                ]
            )
        filtered_solution = filter_redundant_assignments(bipartite_matching_solution, sim)
        if self.debug:
            print(
                *[
                    task_robot_pair
                    for task_robot_pair, assigned in filtered_solution.items()
                    if assigned == 1
                ]
            )

        filtered_solution = filter_overassignments(filtered_solution, sim)

        if self.debug:
            print(
                *[
                    task_robot_pair
                    for task_robot_pair, assigned in filtered_solution.items()
                    if assigned == 1
                ]
            )
            print("##############################################\n\n")

        robot_assignments = {
            robot: task for (robot, task), val in filtered_solution.items() if val == 1
        }

        return predicted_reward, Instantaneous_Schedule(robot_assignments)

    def load_model_weights(self, checkpoint_path, debugging):
        if checkpoint_path is None:
            raise ValueError("Checkpoint path must be provided")
        if not isinstance(checkpoint_path, str):
            raise ValueError("Checkpoint path must be a string")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        checkpoint_state_dict = checkpoint.get("state_dict", checkpoint).get("policy", checkpoint)

        for prefix in ["scheduler_net."]:
            if all(k.startswith(prefix) for k in checkpoint_state_dict):
                checkpoint_state_dict = {
                    k[len(prefix) :]: v for k, v in checkpoint_state_dict.items()
                }
                break

        current_state_dict = self.trained_model.state_dict()
        filtered_checkpoint_state_dict = {
            k: v
            for k, v in checkpoint_state_dict.items()
            if k in current_state_dict and v.size() == current_state_dict[k].size()
        }

        self.trained_model.load_state_dict(filtered_checkpoint_state_dict, strict=False)

        skipped_layers = [
            k
            for k, v in checkpoint_state_dict.items()
            if k not in current_state_dict or v.size() != current_state_dict[k].size()
        ]

        if debugging:
            if skipped_layers:
                print("Skipped layers due to shape mismatch or missing in the new model:")
                for layer in skipped_layers:
                    print(f"  - {layer}")
            else:
                print("No layers were skipped.")

        print(f"Loaded {len(filtered_checkpoint_state_dict)} matching layers from checkpoint.")
        print(f"Skipped {len(skipped_layers)} layers.")
