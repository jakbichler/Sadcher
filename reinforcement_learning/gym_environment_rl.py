import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.append("..")
from rl_simulator import RL_Simulation

from data_generation.problem_generator import generate_random_data_with_precedence
from helper_functions.schedules import Instantaneous_Schedule
from schedulers.bipartite_matching import (
    filter_overassignments,
    filter_redundant_assignments,
    solve_bipartite_matching,
)


class SchedulingRLEnvironment(gym.Env):
    def __init__(self, seed=None):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if seed:
            np.random.seed(seed)
            torch.manual_seed(seed)
            print(f"Seed: {seed}")

        self.N_THREADS = 6

        self.n_robots = 3
        self.n_tasks = 8
        self.n_skills = 3
        self.n_precedence = 3
        self.num_robots_available_in_previous_timestep = -1
        dim_robots = 7  # (x,y,duration,[skill0, skill1, skill2], available)
        dim_tasks = 9  # (x,y,duration,[skill0, skill1, skill2],ready, assigned, incomplete)
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=10.0,
            shape=(
                self.n_robots,
                self.n_tasks + 1,
            ),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "robot_features": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(self.n_robots, dim_robots), dtype=np.float32
                ),
                "task_features": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(self.n_tasks, dim_tasks), dtype=np.float32
                ),
            }
        )

        self.problem_instance = generate_random_data_with_precedence(
            self.n_tasks, self.n_robots, self.n_skills, self.n_precedence
        )

        self.sim = RL_Simulation(
            problem_instance=self.problem_instance,
            debug=False,
            move_while_waiting=True,
        )

    def reset(self, seed=None, options=None):
        self.problem_instance = generate_random_data_with_precedence(
            self.n_tasks, self.n_robots, self.n_skills, self.n_precedence
        )

        self.sim = RL_Simulation(
            problem_instance=self.problem_instance,
            debug=False,
            move_while_waiting=True,
        )

        return self._get_observation()

    def _get_observation(self):
        robot_features, task_features = self.sim.return_task_robot_states()

        return {
            "robot_features": robot_features,
            "task_features": task_features,
        }

    def calculate_reward(self):
        if self.sim.sim_done:
            return -self.sim.makespan
        else:
            return 0.0

    def step(self, action):
        robot_assignments = {}
        available_robots = [robot for robot in self.sim.robots if robot.available]
        incomplete_tasks = [
            task for task in self.sim.tasks if task.incomplete and task.status == "PENDING"
        ]

        # Check if all normal tasks are done -> send all robots to the exit task
        if len(incomplete_tasks) == 1:  # Only end task incomplete
            for robot in available_robots:
                robot_assignments[robot.robot_id] = incomplete_tasks[0].task_id
                robot.current_task = incomplete_tasks[0]
        else:  # Normal assignment
            action = torch.clamp(action, min=1e-6)
            reward_start_end = torch.ones(self.n_robots, 1).to(self.device) * (-1000)
            action = torch.cat((reward_start_end, action, reward_start_end), dim=1)

            bipartite_matching_solution = solve_bipartite_matching(
                action, self.sim, n_threads=self.N_THREADS
            )
            filtered_solution = filter_redundant_assignments(bipartite_matching_solution, self.sim)
            filtered_solution = filter_overassignments(filtered_solution, self.sim)

            robot_assignments = {
                robot: task for (robot, task), val in filtered_solution.items() if val == 1
            }

            self.sim.assign_tasks_to_robots(
                Instantaneous_Schedule(robot_assignments), self.sim.robots
            )

            self.sim.find_highest_non_idle_reward(action)

        while not self.sim.sim_done:
            available = [r for r in self.sim.robots if r.available]
            current_available = len(available)
            # print(f"Current available robots: {current_available}")
            # print(f"Previous available robots: {self.num_robots_available_in_previous_timestep}")
            self.sim.step()
            # Roll out the simulation until we reach the next decision step (new available robots)
            if self.sim.sim_done or (
                current_available > 0
                and current_available != self.num_robots_available_in_previous_timestep
            ):
                self.num_robots_available_in_previous_timestep = current_available
                break

            self.num_robots_available_in_previous_timestep = current_available

        reward = self.calculate_reward()

        done = self.sim.sim_done

        return self._get_observation(), reward, done, {}
