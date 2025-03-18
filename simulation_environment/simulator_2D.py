import argparse
import sys

import numpy as np
import torch
import yaml

sys.path.append("..")
from data_generation.problem_generator import generate_random_data_with_precedence
from helper_functions.schedules import Full_Horizon_Schedule
from schedulers.greedy_instantaneous_scheduler import GreedyInstantaneousScheduler
from schedulers.random_bipartite_matching_scheduler import RandomBipartiteMatchingScheduler
from schedulers.sadcher import SadcherScheduler
from simulation_environment.display_simulation import run_video_mode, visualize
from simulation_environment.task_robot_classes import Robot, Task
from visualizations.solution_visualization import plot_gantt_and_trajectories


class Simulation:
    def __init__(
        self,
        problem_instance,
        scheduler_name=None,
        checkpoint_path=None,
        debug=False,
        move_while_waiting=False,
        model_name=None,
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
        self.update_task_status()  # Initialize task status
        self.task_adjacency = self.create_task_adjacency_matrix()
        self.robot_schedules = {robot.robot_id: [] for robot in self.robots}
        self.scheduler_name = scheduler_name
        self.scheduler = self.create_scheduler(scheduler_name, checkpoint_path, model_name)

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

    def create_scheduler(self, name: str, checkpoint_path=None, model_name=None):
        if name == "greedy":
            return GreedyInstantaneousScheduler()
        elif name == "random_bipartite":
            return RandomBipartiteMatchingScheduler()
        elif name == "sadcher":
            return SadcherScheduler(
                debugging=self.debugging,
                checkpoint_path=checkpoint_path,
                duration_normalization=self.duration_normalization,
                location_normalization=self.location_normalization,
                model_name=model_name,
            )
        else:
            raise ValueError(f"Unknown scheduler '{name}'")

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
        """True if all robots are within 'threshold' distance of 'task' location."""
        assigned_robots = [r for r in self.robots if r.current_task == task]
        if not assigned_robots:
            return False

        for r in assigned_robots:
            if np.linalg.norm(r.location - task.location) > threshold:
                return False
        return True

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

    def step(self):
        """Advance the simulation by one timestep, moving robots and updating tasks."""
        for robot in self.robots:
            if robot.current_task:
                if robot.current_task.task_id is not self.idle_task_id:
                    # Move to assigned task location
                    robot.update_position_on_task()

                elif (
                    robot.current_task.task_id is self.idle_task_id
                    and self.move_while_waiting
                    and self.scheduler_name == "sadcher"
                ):
                    # Robot was assigned IDLE task
                    if self.robot_can_still_contribute_to_other_tasks(robot):
                        # Premove robots towards highest reward non-IDLE task
                        highest_non_idle_reward = self.highest_non_idle_rewards[robot.robot_id]
                        highest_non_idle_reward_id = self.highest_non_idle_reward_ids[
                            robot.robot_id
                        ]
                        if highest_non_idle_reward > 0.1:
                            robot.position_towards_task(self.tasks[highest_non_idle_reward_id])

                    else:
                        # Robot cannot contribute anymore ->  Premove towards exit location
                        robot.position_towards_task(self.tasks[-1])

        self.update_task_status()
        self.update_task_duration()
        # Task durations hvae been updated, checking task_status again
        self.update_task_status()

        for robot in self.robots:
            robot.check_task_status()

        available_robots = [robot for robot in self.robots if robot.available]

        if available_robots:
            if self.scheduler_name == "sadcher":
                predicted_reward, instantaneous_assignment = (
                    self.scheduler.calculate_robot_assignment(self)
                )

                if self.move_while_waiting:
                    self.highest_non_idle_rewards, self.highest_non_idle_reward_ids = (
                        self.find_highest_non_idle_reward(predicted_reward)
                    )
            else:
                instantaneous_assignment = self.scheduler.calculate_robot_assignment(self)

            self.assign_tasks_to_robots(instantaneous_assignment, self.robots)

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

        return (
            highest_non_idle_rewards,
            highest_non_idle_rewards_ids + 1,
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
    # problem_instance = json.load(open("/home/jakob/thesis/benchmarking/precedence_6t2r2s2p/problem_instances/problem_instance_000044.json", "r"))

    sim = Simulation(
        problem_instance,
        scheduler_name=args.scheduler,
        checkpoint_path="/home/jakob/thesis/imitation_learning/checkpoints/hyperparam_0_8t3r3s/best_checkpoint.pt",
        debug=True,
        move_while_waiting=args.move_while_waiting,
        model_name=args.sadcher_model_name,
    )

    if args.video:
        # Step simulation, saving frames each time, then generate .mp4
        run_video_mode(sim)
    elif args.visualize:
        # Interactive mode
        visualize(sim)
    else:
        # Run simulation until completion
        while not sim.sim_done:
            sim.step()

    rolled_out_schedule = Full_Horizon_Schedule(sim.makespan, sim.robot_schedules, n_tasks)
    print(rolled_out_schedule)
    plot_gantt_and_trajectories(
        f"{sim.scheduler_name}: MS, {sim.makespan}, \n nt: {n_tasks}, nr: {n_robots}, sn: {n_skills}, seed: {config['random_seed']}",
        rolled_out_schedule,
        problem_instance,
    )
