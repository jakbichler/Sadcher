import argparse
import json
import sys
import time

import numpy as np
import torch
import yaml

sys.path.append("..")
from data_generation.problem_generator import (
    generate_random_data,
    generate_random_data_with_precedence,
)
from helper_functions.schedules import Full_Horizon_Schedule
from helper_functions.task_robot_classes import Robot, Task
from schedulers.greedy_instantaneous_scheduler import GreedyInstantaneousScheduler
from schedulers.random_bipartite_matching_scheduler import RandomBipartiteMatchingScheduler
from schedulers.sadcher import SadcherScheduler
from simulation_environment.display_simulation import run_video_mode, visualize
from visualizations.solution_visualization import plot_gantt_and_trajectories


def create_scheduler(
    name: str,
    checkpoint_path=None,
    model_name=None,
    duration_normalization=100,
    location_normalization=100,
):
    if name == "greedy":
        return GreedyInstantaneousScheduler()
    elif name == "random_bipartite":
        return RandomBipartiteMatchingScheduler()
    elif name == "sadcher":
        return SadcherScheduler(
            debugging=False,
            checkpoint_path=checkpoint_path,
            duration_normalization=duration_normalization,
            location_normalization=location_normalization,
            model_name=model_name,
        )
    else:
        raise ValueError(f"Unknown scheduler '{name}'")


class Simulation:
    def __init__(
        self,
        problem_instance,
        scheduler_name=None,
        debug=False,
        move_while_waiting=True,
    ):
        self.timestep = 0
        self.sim_done = False
        self.makespan = -1  # Will be set when simulation is done
        self.debugging = debug
        self.scheduler_computation_times = []
        self.move_while_waiting = move_while_waiting
        self.precedence_constraints: list = problem_instance["precedence_constraints"]
        self.duration_normalization = np.max(problem_instance["T_e"])
        self.location_normalization = np.max(problem_instance["task_locations"])
        self.robots: list[Robot] = self.create_robots(problem_instance)
        self.tasks: list[Task] = self.create_tasks(problem_instance)
        self.update_task_status()  # Initialize task status
        self.task_adjacency = self.create_task_adjacency_matrix()
        self.robot_schedules = {robot.robot_id: [] for robot in self.robots}
        self.scheduler_name = scheduler_name
        self.num_available_robots_in_previous_timestep = -1
        self.worst_case_makespan = np.sum(problem_instance["T_e"]) + np.sum(
            [np.max(problem_instance["T_t"][task]) for task in range(len(problem_instance["T_e"]))]
        )
        self.no_new_assignment_steps = 0

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
        # Check for transition from PENDING -> IN_PROGRESS: log start time for all assigned robots
        if previous_status == "PENDING" and task.status == "IN_PROGRESS":
            for r in [rb for rb in self.robots if rb.current_task == task]:
                self.robot_schedules[r.robot_id].append((task.task_id, self.timestep, None))

        # Check for transition from IN_PROGRESS -> DONE: log end time for all assigned robots
        if previous_status == "IN_PROGRESS" and task.status == "DONE":
            for r in [rb for rb in self.robots if rb.current_task == task]:
                tid, start, _ = self.robot_schedules[r.robot_id][-1]
                if tid == task.task_id:
                    self.robot_schedules[r.robot_id][-1] = (tid, start, self.timestep)

    def find_task_to_premove_to(self, robot):
        task_to_premove_to = None

        if self.scheduler_name == "sadcher":
            highest_non_idle_reward = self.highest_non_idle_rewards[robot.robot_id]
            highest_non_idle_reward_id = self.highest_non_idle_reward_ids[robot.robot_id]
            if highest_non_idle_reward > 0.1:
                task_to_premove_to = self.tasks[highest_non_idle_reward_id]

        elif self.scheduler_name == "sadcher_rl":
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
                np.linalg.norm(robot.location - task.location)
                for task in tasks_robot_can_contribute_to
            ]
            task_to_premove_to = tasks_robot_can_contribute_to[np.argmin(distances)]

        return task_to_premove_to

    def step(self):
        """Advance the simulation by one timestep, moving robots and updating tasks."""

        for robot in self.robots:
            if robot.current_task:
                if robot.current_task.task_id is not self.idle_task_id:
                    # Move to assigned task location (Normal task)
                    robot.update_position_on_task()

                elif robot.current_task.task_id is self.idle_task_id and self.move_while_waiting:
                    # Robot was assigned IDLE task
                    if self.robot_can_still_contribute_to_other_tasks(robot):
                        task_to_premove_to = self.find_task_to_premove_to(robot)
                        if task_to_premove_to:
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

    def step_until_next_decision_point(self, max_no_new_assignment_steps=10_000):
        no_new_assignment_steps = 0
        while not self.sim_done:
            self.step()

            current_available = len([r for r in self.robots if r.available])
            previous_available = self.num_available_robots_in_previous_timestep
            self.num_available_robots_in_previous_timestep = current_available

            change_in_available_robots = (current_available > 0) and (
                current_available != previous_available
            )

            maxed_out_time_without_assignments = (
                no_new_assignment_steps >= max_no_new_assignment_steps
            )

            if (self.sim_done or change_in_available_robots) or maxed_out_time_without_assignments:
                no_new_assignment_steps = 0
                break

            # Force termination if timestep exceeds worst-case threshold
            if self.timestep >= self.worst_case_makespan:
                self.finish_simulation()
                self.makespan = self.worst_case_makespan
                print(f"Scheduler did not find a feasible solution at timestep {self.timestep}")
                break

            no_new_assignment_steps += 1

    def assign_tasks_to_robots(self, instantaneous_schedule):
        for robot in self.robots:
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

        self.highest_non_idle_rewards = highest_non_idle_rewards
        self.highest_non_idle_reward_ids = (
            highest_non_idle_rewards_ids + 1
        )  # + 1 to accoutn for cutting off start task

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="Visualize the simulation")
    parser.add_argument("--video", action="store_true", help="Generate a video of the simulation")
    parser.add_argument(
        "--scheduler", type=str, help="Scheduler to use (greedy or random_bipartite)"
    )
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    parser.add_argument(
        "--move_while_waiting",
        action="store_true",
        help="Move robots towards second highest reward task while waiting",
    )
    parser.add_argument("--sadcher_model_name", type=str, help="Name of the model to use")
    args = parser.parse_args()

    with open("simulation_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    n_tasks = config["n_tasks"]
    n_robots = config["n_robots"]
    n_skills = config["n_skills"]
    n_precedence = config["n_precedence"]
    np.random.seed(config["random_seed"])
    precedence_constraints = config["precedence_constraints"]

    # problem_instance = generate_random_data(n_tasks, n_robots, n_skills, precedence_constraints)
    problem_instance = generate_random_data_with_precedence(
        n_tasks, n_robots, n_skills, n_precedence
    )

    sim = Simulation(
        problem_instance,
        scheduler_name=args.scheduler,
        debug=True,
    )

    checkpoint_path = (
        "/home/jakob/thesis/imitation_learning/checkpoints/hyperparam_2_8t3r3s/best_checkpoint.pt"
    )

    scheduler = create_scheduler(
        args.scheduler,
        checkpoint_path,
        args.sadcher_model_name,
        duration_normalization=sim.duration_normalization,
        location_normalization=sim.location_normalization,
    )

    if args.video:
        # Step simulation, saving frames each time, then generate .mp4
        run_video_mode(sim)
    elif args.visualize:
        # Interactive mode
        visualize(sim, scheduler)
    else:
        # Run simulation until completion
        while not sim.sim_done:
            if isinstance(scheduler, SadcherScheduler):
                predicted_reward, instantaneous_schedule = scheduler.calculate_robot_assignment(sim)
                sim.find_highest_non_idle_reward(predicted_reward)
            else:
                instantaneous_schedule = scheduler.calculate_robot_assignment(sim)
            sim.assign_tasks_to_robots(instantaneous_schedule)
            sim.step_until_next_decision_point()

    rolled_out_schedule = Full_Horizon_Schedule(sim.makespan, sim.robot_schedules, n_tasks)
    print(rolled_out_schedule)
    print(f"Sum of computation times: {sum(sim.scheduler_computation_times)}")
    plot_gantt_and_trajectories(
        f"{sim.scheduler_name}: MS, {sim.makespan}, \n nt: {n_tasks}, nr: {n_robots}, sn: {n_skills}, seed: {config['random_seed']}",
        rolled_out_schedule,
        problem_instance,
    )
