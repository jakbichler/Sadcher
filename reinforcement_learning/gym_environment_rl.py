import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.append("..")
from rl_simulator import RL_Simulation

from baselines.aswale_23.greedy_solver import greedy_scheduling
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

        self.N_THREADS = 4
        self.n_robots = 3
        self.n_tasks = 8
        self.n_skills = 3
        self.n_precedence = 3
        self.num_robots_available_in_previous_timestep = -1
        dim_robots = 7  # (x,y,duration,[skill0, skill1, skill2], available)
        dim_tasks = 9  # (x,y,duration,[skill0, skill1, skill2],ready, assigned, incomplete)

        self.action_space = gym.spaces.Discrete(
            self.n_tasks + 1,
        )

        self.observation_space = gym.spaces.Dict(
            {
                "robot_features": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(self.n_robots, dim_robots), dtype=np.float32
                ),
                "task_features": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(self.n_tasks, dim_tasks), dtype=np.float32
                ),
                "task_adjacency": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(self.n_tasks, self.n_tasks), dtype=np.float32
                ),
            }
        )

        if seed:
            np.random.seed(seed)
            torch.manual_seed(seed)
            print(f"Seed: {seed}")

    def reset(self, seed=None, options=None):
        self.problem_instance = generate_random_data_with_precedence(
            self.n_tasks, self.n_robots, self.n_skills, self.n_precedence
        )

        self.worst_case_makespan = np.sum(self.problem_instance["T_e"]) + np.sum(
            [np.max(self.problem_instance["T_t"][task]) for task in range(self.n_tasks + 1)]
        )

        self.sim = RL_Simulation(
            problem_instance=self.problem_instance,
            debug=False,
            move_while_waiting=True,
        )

        self.greedy_makespan = greedy_scheduling(self.problem_instance).makespan

        return self._get_observation(), {}

    def _get_observation(self):
        robot_features, task_features = self.sim.return_task_robot_states()

        return {
            "robot_features": robot_features,
            "task_features": task_features,
            "task_adjacency": self.sim.task_adjacency,
        }

    def calculate_reward(self):
        if self.sim.sim_done:
            print(
                "Simulation done with makespan:",
                self.sim.makespan,
                "and greedy makespan:",
                self.greedy_makespan,
            )
            return -(self.sim.makespan - self.greedy_makespan)

        else:
            return 0.0

    def step(self, action: Instantaneous_Schedule):
        self.sim.assign_tasks_to_robots(action, self.sim.robots)
        truncated = False

        while not self.sim.sim_done:
            available = [r for r in self.sim.robots if r.available]
            current_available = len(available)
            self.sim.step()
            # Roll out the simulation until we reach the next decision step (new available robots)
            if self.sim.sim_done or (
                current_available > 0
                and current_available != self.num_robots_available_in_previous_timestep
            ):
                self.num_robots_available_in_previous_timestep = current_available
                break

            self.num_robots_available_in_previous_timestep = current_available
            # Force termination if timestep exceeds worst-case threshold
            if self.sim.timestep >= self.worst_case_makespan:
                self.sim.finish_simulation()
                self.sim.makespan = self.worst_case_makespan
                print(f"Scheduler did not find a feasible solution at timestep {self.sim.timestep}")
                truncated = True
                break

        reward = self.calculate_reward()
        terminated = self.sim.sim_done

        return self._get_observation(), reward, terminated, truncated, {}
