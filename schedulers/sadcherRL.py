import numpy as np
import torch
from torch.distributions import Categorical

from helper_functions.schedules import Instantaneous_Schedule
from models.scheduler_network import SchedulerNetwork
from schedulers.filtering_assignments import (
    filter_overassignments,
    filter_redundant_assignments,
    filter_unqualified_assignments,
)


class RLSadcherScheduler:
    """
    Scheduler that uses a neural network to predict the reward for each robot-task pair.
    Softmax over the rows gives a distribution for agent-task assignment -> used for models trained with discrete RL
    """

    def __init__(
        self,
        debugging,
        checkpoint_path,
        duration_normalization,
        location_normalization,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.trained_model = SchedulerNetwork(
            robot_input_dimensions=7,
            task_input_dimension=9,
            embed_dim=256,
            ff_dim=512,
            n_transformer_heads=4,
            n_transformer_layers=2,
            n_gatn_heads=8,
            n_gatn_layers=1,
            use_idle=False,
        ).to(self.device)

        self.trained_model.eval()
        self.debug = debugging
        self.duration_normalization = duration_normalization
        self.location_normalization = location_normalization
        self.load_model_weights(checkpoint_path, debugging)
        self.bipartite_matcher = None

    def calculate_robot_assignment(self, sim, sampling=True):
        n_robots = len(sim.robots)
        robot_assignments = {}
        # Special case for the last task
        available_robots = [robot for robot in sim.robots if robot.available]
        available_robot_ids = [robot.robot_id for robot in available_robots]
        incomplete_tasks = [
            task for task in sim.tasks if task.incomplete and task.status == "PENDING"
        ]

        only_end_task_left = len(incomplete_tasks) == 1
        all_tasks_assigned = all(
            sim.all_skills_assigned(task)
            for task in incomplete_tasks
            if task.task_id != sim.last_task_id
        )

        # Send all robots to end location if no tasks left
        if only_end_task_left or all_tasks_assigned:
            robot_assignments = {robot: sim.tasks[-1].task_id for robot in available_robot_ids}
            return Instantaneous_Schedule(robot_assignments), False

        # Calculate robot assignments
        else:
            # If a robot cannot contribute anymore -> send to end location
            for robot in available_robots:
                if not sim.robot_can_still_contribute_to_other_tasks(
                    robot, only_full_assignments=False
                ):
                    robot_assignments[robot.robot_id] = sim.tasks[-1].task_id

            robot_features, task_features = self.extract_features(sim)
            task_adjacency = torch.tensor(sim.task_adjacency, dtype=torch.float32).to(self.device)

            logits = self.trained_model(
                robot_features, task_features, task_adjacency
            )  # shape: (batch, n_robots, n_tasks + 1)

            mask = self.create_task_robot_mask(task_features, robot_features)
            logits = logits.masked_fill(mask == 0, -1e9).float()
            action_probas = torch.softmax(logits, dim=-1)  # shape: (n_robots, n_tasks)

            if sampling:
                action_distributions = [
                    Categorical(probs=action_proba)
                    for action_proba in torch.split(action_probas, n_robots)
                ]

                actions = [distribution.sample() for distribution in action_distributions]
            else:
                actions = [torch.argmax(probas, dim=-1) for probas in action_probas]

            actions = torch.cat(actions).flatten()
            # Only assign available robots
            action_dict = {
                (robot_id, task_id + 1): 1  # +1 since task 0 is start task (not predicted)
                for (robot_id, task_id) in enumerate(actions)
                if robot_id in available_robot_ids and robot_id not in robot_assignments
            }

            action_dict_filtered = filter_redundant_assignments(action_dict, sim)
            action_dict_filtered = filter_overassignments(action_dict_filtered, sim)
            action_dict_filtered = filter_unqualified_assignments(action_dict_filtered, sim)
            robot_assignments = {
                robot: task for (robot, task), val in action_dict_filtered.items() if val == 1
            }

            removed = [
                (rid, tid)
                for (rid, tid), val in action_dict.items()
                if val == 1 and action_dict_filtered.get((rid, tid), 0) == 0
            ]

            filter_triggered = any(
                sim.robot_can_still_contribute_to_other_tasks(sim.robots[rid]) for rid, _ in removed
            )

            return Instantaneous_Schedule(robot_assignments), filter_triggered

    def extract_features(self, sim):
        """
        Returns robot and task features.
        """
        robot_features = (
            torch.tensor(
                np.array(
                    [
                        r.feature_vector(self.location_normalization, self.duration_normalization)
                        for r in sim.robots
                    ]
                ),
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .to(self.device)
        )

        task_features = np.array(
            [
                t.feature_vector(self.location_normalization, self.duration_normalization)
                for t in sim.tasks[1:-1]
            ]
        )

        task_features = self.subtract_assigned_task_requirements(sim, task_features)

        return robot_features, task_features

    def subtract_assigned_task_requirements(self, sim, task_features):
        # Subtract already covered skills
        first_skill_index = 3  # [0,1,2] is location and duration
        n_skills = 3
        skill_slice = slice(first_skill_index, first_skill_index + n_skills)
        real_tasks = sim.tasks[1:-1]

        for task_index, task in enumerate(real_tasks):
            required = np.array(task.requirements, dtype=bool)
            covered = np.zeros(n_skills, dtype=bool)

            # Task is complete -> all are covered
            if not task.incomplete:
                covered = required
            else:
                for robot in sim.robots:
                    if robot.current_task is task:
                        covered |= np.array(robot.capabilities, dtype=bool)
            remaining = required & ~covered

            task_features[task_index, skill_slice] = torch.tensor(remaining)
        task_features = (
            torch.tensor(task_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        return task_features

    def create_task_robot_mask(self, task_features, robot_features):
        # For robots: last element is availability (1: available, 0: not)
        rx, ry, workload, c0, c1, c2, available = robot_features.unbind(dim=-1)
        robot_caps = torch.stack((c0, c1, c2), dim=-1).bool()  # (n_robots, 3)
        robot_availability = available.bool()  # (n_robots)

        # For tasks
        tx, ty, dur, r0, r1, r2, ready, assigned, incomplete = task_features.unbind(dim=-1)
        reqs = torch.stack((r0, r1, r2), -1).bool()

        can_contribute_mask = (robot_caps.unsqueeze(2) & reqs.unsqueeze(1)).any(
            dim=-1
        )  # (n_robots, n_tasks)

        task_mask = ready.bool() & incomplete.bool()  # (n_tasks)

        # Build a mask of shape (n_robots, n_actions) that indicates valid actions.
        robot_mask = robot_availability.unsqueeze(-1).to(self.device)  # (n_robots, 1)
        task_mask = task_mask.unsqueeze(0).to(self.device)  # (1, n_tasks)
        robot_task_mask = robot_mask & task_mask & can_contribute_mask  # (n_robots, n_tasks)

        return robot_task_mask

    def load_model_weights(self, checkpoint_path, debugging):
        if checkpoint_path is None:
            raise ValueError("Checkpoint path must be provided")
        if not isinstance(checkpoint_path, str):
            raise ValueError("Checkpoint path must be a string")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        checkpoint_state_dict = checkpoint.get("state_dict", checkpoint).get("policy", checkpoint)

        for prefix in ["scheduler_net."]:
            checkpoint_state_dict = {
                (k[len(prefix) :] if k.startswith(prefix) else k): v
                for k, v in checkpoint_state_dict.items()
            }

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
