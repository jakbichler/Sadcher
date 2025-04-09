import sys
import time

import numpy as np
import torch

sys.path.append("..")
from helper_functions.task_robot_classes import Robot, Task


class RL_Simulation:
    def __init__(
        self,
        problem_instance,
        debug=False,
        move_while_waiting=False,
    ):
        self.timestep = 0
        self.sim_done = False
        self.makespan = -1  # Will be set when simulation is done
        self.debugging = debug
        self.move_while_waiting = move_while_waiting
        self.precedence_constraints: list = problem_instance["precedence_constraints"]
        self.duration_normalization = np.max(problem_instance["T_e"])
        self.location_normalization = np.max(problem_instance["task_locations"])
        self.robots: list[Robot] = self.create_robots(problem_instance)
        self.tasks: list[Task] = self.create_tasks(problem_instance)
        self.robot_schedules = {robot.robot_id: [] for robot in self.robots}
        self.update_task_status()  # Initialize task status
        self.task_adjacency = self.create_task_adjacency_matrix()
        self.num_available_robots_in_previous_timestep = -1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_task_adjacency_matrix(self):
        task_adjacency = torch.zeros((self.n_real_tasks, self.n_real_tasks))

        if self.precedence_constraints:
            for precedence in self.precedence_constraints:
                # Precedence is 1-indexed
                task_adjacency[precedence[0] - 1, precedence[1] - 1] = 1

        return task_adjacency

    def create_robots(self, problem_instance):
        robot_capabilities = problem_instance["Q"]
        start_location = problem_instance["task_locations"][0]
        return [
            Robot(robot_id=idx, location=start_location, capabilities=cap)
            for idx, cap in enumerate(robot_capabilities)
        ]

    def create_tasks(self, problem_instance):
        locations = problem_instance["task_locations"]
        durations = problem_instance["T_e"]
        requirements = problem_instance["R"]

        # Insert artificial IDLE task
        central_position = self.location_normalization / 2
        locations = np.insert(locations, -1, np.array([central_position, central_position]), axis=0)
        durations = np.insert(durations, -1, 0, axis=0)
        requirements = np.insert(requirements, -1, np.zeros_like(requirements[0]), axis=0)

        tasks = [
            Task(idx, loc, dur, req)
            for idx, (loc, dur, req) in enumerate(zip(locations, durations, requirements))
        ]
        tasks[-2].status = "DONE"  # Idle task

        self.n_tasks = len(tasks)
        self.last_task_id = self.n_tasks - 1
        self.idle_task_id = self.n_tasks - 2
        self.n_real_tasks = self.n_tasks - 3  # Excluding start, exit and idle task

        return tasks

    def update_task_status(self):
        for task in self.tasks:
            if task.task_id in [0, self.idle_task_id] or task.predecessors_completed(self):
                task.ready = True
            else:
                task.ready = False

            # Special handling of last task
            if task.task_id == self.last_task_id:
                if self.all_robots_at_exit_location(threshold=0.01):
                    task.status = "DONE"
                    self.finish_simulation()
                else:
                    task.status = "PENDING"

    def update_task_duration(self):
        for task in self.tasks:
            if task.status == "DONE":
                continue

            elif task.status == "PENDING":
                if (
                    self.all_skills_assigned(task)
                    and self.all_robots_at_task(task, threshold=0.01)
                    and task.ready
                ):
                    previous_status = task.status
                    task.start()
                    task.decrement_duration()
                    self.log_into_full_horizon_schedule(task, previous_status)

            elif task.status == "IN_PROGRESS":
                task.decrement_duration()
                # If decrementing just switched it to DONE, log the transition for final full horizon schedule:
                if task.status == "DONE":
                    self.log_into_full_horizon_schedule(task, "IN_PROGRESS")

    def finish_simulation(self):
        self.sim_done = True
        self.makespan = self.timestep

    def all_skills_assigned(self, task):
        """
        Returns True if:
        1) The logical OR of all assigned robots' capabilities covers all task requirements.
        2) Every assigned robot is within 1 unit of the task location.
        """
        assigned_robots = [r for r in self.robots if r.current_task == task]

        # Combine capabilities across all assigned robots
        combined_capabilities = np.zeros_like(task.requirements, dtype=bool)
        for robot in assigned_robots:
            robot_cap = np.array(robot.capabilities, dtype=bool)
            combined_capabilities = np.logical_or(combined_capabilities, robot_cap)

        required_skills = np.array(task.requirements, dtype=bool)

        # Check if the combined team covers all required skills
        return np.all(combined_capabilities[required_skills])

    def all_robots_at_task(self, task, threshold=0.01):
        assigned_robots = [r for r in self.robots if r.current_task == task]
        if not assigned_robots:
            return False

        # Filter for robots that are within the distance threshold
        nearby_robots = [
            r for r in assigned_robots if np.linalg.norm(r.location - task.location) <= threshold
        ]
        if not nearby_robots:
            return False

        combined_capabilities = np.zeros_like(task.requirements, dtype=bool)
        for r in nearby_robots:
            combined_capabilities = np.logical_or(
                combined_capabilities, np.array(r.capabilities, dtype=bool)
            )

        # Check if the combined capabilities cover all the task's required skills
        required_skills = np.array(task.requirements, dtype=bool)
        return np.all(combined_capabilities[required_skills])

    def all_robots_at_exit_location(self, threshold=0.01):
        """True if all robots are within 'threshold' distance of the exit location."""
        exit_location = self.tasks[-1].location
        for r in self.robots:
            if np.linalg.norm(r.location - exit_location) > threshold:
                return False
        return True

    def log_into_full_horizon_schedule(self, task, previous_status):
        pass

    def step(self, render=False):
        """Advance the simulation by one timestep, moving robots and updating tasks."""

        for robot in self.robots:
            if robot.current_task:
                if robot.current_task.task_id is not self.idle_task_id:
                    # Move to assigned task location
                    robot.update_position_on_task()

                elif robot.current_task.task_id is self.idle_task_id and self.move_while_waiting:
                    # Robot was assigned IDLE task
                    if self.robot_can_still_contribute_to_other_tasks(robot):
                        task_to_premove_to = self.find_task_to_premove_to(robot)
                        if task_to_premove_to:
                            # Move towards the task
                            robot.position_towards_task(task_to_premove_to)
                    else:
                        # Robot cannot contribute anymore ->  Premove towards exit location
                        robot.position_towards_task(self.tasks[-1])

        self.update_task_status()
        self.update_task_duration()
        # Task durations have been updated, checking task_status again
        self.update_task_status()

        for robot in self.robots:
            robot.check_task_status(self.idle_task_id)

        self.timestep += 1

    def assign_tasks_to_robots(self, instantaneous_schedule, robots):
        for robot in robots:
            task_id = instantaneous_schedule.robot_assignments.get(robot.robot_id)
            if task_id is not None:
                task = self.tasks[task_id]
                robot.current_task = task
                task.assigned = True if task_id != self.idle_task_id else False

    def find_highest_non_idle_reward(self, predicted_rewards):
        predicted_rewards_non_idle = predicted_rewards[:, 1 : self.idle_task_id]
        highest_non_idle_rewards, highest_non_idle_rewards_ids = torch.max(
            predicted_rewards_non_idle, dim=1
        )

        self.highest_non_idle_probas = highest_non_idle_rewards
        self.highest_non_idle_proba_ids = (
            highest_non_idle_rewards_ids + 1
        )  # +1 to account for cutting off the start task

    def find_highest_non_idle_probas(self, predicted_rewards):
        predicted_probas_non_idle = predicted_rewards[:, :-1]
        highest_non_idle_probas, highest_non_idle_proba_ids = torch.max(
            predicted_probas_non_idle, dim=1
        )

        self.highest_non_idle_probas = highest_non_idle_probas
        self.highest_non_idle_proba_ids = (
            highest_non_idle_proba_ids + 1
        )  # +1 to account for cutting off the start task

    def robot_can_still_contribute_to_other_tasks(self, robot):
        pending_tasks = [
            task for task in self.tasks if (task.status == "PENDING" and not task.assigned)
        ]
        if not pending_tasks:
            return False

        pending_tasks_requirements = [task.requirements for task in pending_tasks]
        robot_capabilities = np.array(robot.capabilities)

        can_contribute = np.any(
            np.logical_and(robot_capabilities, np.logical_or.reduce(pending_tasks_requirements))
        )

        return can_contribute

    def find_task_to_premove_to(self, robot):
        """
        Find the task that the robot can move to, which is closest to its current location.
        """
        # Get all tasks that are not assigned and not the IDLE task
        unassigned_tasks = [
            task for task in self.tasks[:-1] if task.incomplete and not task.assigned
        ]
        if not unassigned_tasks:
            return None

        tasks_robot_can_contribute_to = []
        for task in unassigned_tasks:
            if np.any(np.logical_and(robot.capabilities, task.requirements)):
                tasks_robot_can_contribute_to.append(task)

        distances = [
            np.linalg.norm(robot.location - task.location) for task in tasks_robot_can_contribute_to
        ]

        return tasks_robot_can_contribute_to[np.argmin(distances)]

    def return_task_robot_states(self):
        task_features = np.array(
            [
                task.feature_vector(self.location_normalization, self.duration_normalization)
                for task in self.tasks[1:-2]
            ],  # Exclude start, end, and IDLE task
            dtype=np.float32,
        )

        robot_features = np.array(
            [
                robot.feature_vector(self.location_normalization, self.duration_normalization)
                for robot in self.robots
            ],
            dtype=np.float32,
        )

        task_features = torch.tensor(task_features, dtype=torch.float32)
        robot_features = torch.tensor(robot_features, dtype=torch.float32)

        return task_features, robot_features
