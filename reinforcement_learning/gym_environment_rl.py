import sys
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from icecream import ic

sys.path.append("..")
from rl_simulator import RL_Simulation

from baselines.aswale_23.greedy_solver import greedy_scheduling
from data_generation.problem_generator import (
    generate_random_data_all_robots_all_skills,
    generate_random_data_with_precedence,
)
from helper_functions.schedules import Instantaneous_Schedule
from schedulers.bipartite_matching import (
    filter_overassignments,
    filter_redundant_assignments,
    solve_bipartite_matching,
)
from simulation_environment.display_simulation import update_plot


class SchedulingRLEnvironment(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "none"],
        "render_fps": 5,
    }

    def __init__(self, seed=None, problem_type="random_with_precedence", render_mode="human"):
        super().__init__()

        self.n_robots = 3
        self.n_tasks = 8
        self.n_skills = 3
        self.n_precedence = 3
        self.num_robots_available_in_previous_timestep = -1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.RENDER_COUNTER_THRESHOLD = 20
        self.MAX_NO_NEW_ASSIGNMENT_STEPS = 100
        dim_robots = 7  # (x,y,duration,[skill0, skill1, skill2], available)
        dim_tasks = 9  # (x,y,duration,[skill0, skill1, skill2],ready, assigned, incomplete)

        self.action_space = gym.spaces.MultiDiscrete(
            [self.n_tasks + 1] * self.n_robots,
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

        self.render_mode = render_mode  # "human", "rgb_array", or "none"
        self.render_simulation = False
        self.fig, self.ax = None, None
        self.colors = plt.cm.Set1(np.linspace(0, 1, self.n_skills))
        self.problem_type = problem_type

    def reset(self, seed=None, options=None):
        if self.problem_type == "random_with_precedence":
            self.problem_instance = generate_random_data_with_precedence(
                self.n_tasks, self.n_robots, self.n_skills, self.n_precedence
            )
        elif self.problem_type == "random_all_robots_all_skills":
            self.problem_instance = generate_random_data_all_robots_all_skills(
                self.n_tasks, self.n_robots, self.n_skills
            )

        else:
            raise ValueError(f"Unknown problem type: {self.problem_type}")

        self.worst_case_makespan = np.sum(self.problem_instance["T_e"]) + np.sum(
            [np.max(self.problem_instance["T_t"][task]) for task in range(self.n_tasks + 1)]
        )

        self.sim = RL_Simulation(
            problem_instance=self.problem_instance,
            debug=False,
            move_while_waiting=True,
        )

        self.greedy_makespan = greedy_scheduling(self.problem_instance, print_flag=False).makespan

        return self._get_observation(), {}

    def _get_observation(self):
        task_features, robot_features = self.sim.return_task_robot_states()

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
            return -(self.sim.makespan - self.greedy_makespan) / self.greedy_makespan

        else:
            return 0.0

    def step(self, action):
        available_robots = [robot for robot in self.sim.robots if robot.available]
        available_robot_ids = [robot.robot_id for robot in available_robots]
        incomplete_tasks = [
            task for task in self.sim.tasks if task.incomplete and task.status == "PENDING"
        ]
        # Check if all normal tasks are done -> send all robots to the exit task
        if len(incomplete_tasks) == 1:  # Only end task incomplete
            for robot in available_robots:
                robot.current_task = self.sim.tasks[-1]

        else:
            # Only assign available robots
            action_dict = {
                (robot_id, task_id + 1): 1  # +1 since task 0 is start task (not predicted)
                for (robot_id, task_id) in enumerate(action)
                if robot_id in available_robot_ids
            }

            action_dict_filtered = filter_redundant_assignments(action_dict, self.sim)
            action_dict_filtered = filter_overassignments(action_dict_filtered, self.sim)
            robot_assignments = {
                robot: task for (robot, task), val in action_dict_filtered.items() if val == 1
            }
            self.sim.assign_tasks_to_robots(
                Instantaneous_Schedule(robot_assignments), self.sim.robots
            )

        truncated = False
        no_new_assignment_steps = 0
        while not self.sim.sim_done:
            self.sim.step()

            available = [r for r in self.sim.robots if r.available]
            current_available = len(available)

            previous_available = self.num_robots_available_in_previous_timestep
            self.num_robots_available_in_previous_timestep = current_available

            if self.render_simulation and self.sim.timestep % self.RENDER_COUNTER_THRESHOLD == 0:
                self._low_level_render()

            # Roll out the simulation until we reach the next decision step (new available robots)
            simulation_done = self.sim.sim_done

            change_in_available_robots = (current_available > 0) and (
                current_available != previous_available
            )

            maxed_out_time_without_assignments = (
                no_new_assignment_steps >= self.MAX_NO_NEW_ASSIGNMENT_STEPS
            )

            if simulation_done or change_in_available_robots or maxed_out_time_without_assignments:
                no_new_assignment_steps = 0
                break

            # Force termination if timestep exceeds worst-case threshold
            if self.sim.timestep >= self.worst_case_makespan:
                self.sim.finish_simulation()
                self.sim.makespan = self.worst_case_makespan
                print(f"Scheduler did not find a feasible solution at timestep {self.sim.timestep}")
                truncated = True
                break

            no_new_assignment_steps += 1
        reward = self.calculate_reward()
        terminated = self.sim.sim_done

        return self._get_observation(), reward, terminated, truncated, {}

    def render(self, mode=None):
        self.render_mode = mode
        self.render_simulation = True

    def _low_level_render(self):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            plt.ion()
            self.fig.show()
        update_plot(self.sim, self.ax, self.fig, self.colors, self.n_skills, video_mode=True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(1)

    def get_config(self):
        return {
            "n_robots": self.n_robots,
            "n_tasks": self.n_tasks,
            "n_skills": self.n_skills,
            "n_precedence": self.n_precedence,
            "problem_type": self.problem_type,
        }
