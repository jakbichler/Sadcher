import sys
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append("..")

from baselines.aswale_23.greedy_solver import greedy_scheduling
from data_generation.problem_generator import (
    generate_random_data_all_robots_all_skills,
    generate_random_data_with_precedence,
)
from helper_functions.schedules import Instantaneous_Schedule
from schedulers.filtering_assignments import (
    filter_overassignments,
    filter_redundant_assignments,
    filter_unqualified_assignments,
)
from simulation_environment.display_simulation import update_plot
from simulation_environment.simulator_2D import Simulation


class SchedulingRLEnvironment(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "none"],
        "render_fps": 5,
    }

    def __init__(
        self,
        seed=None,
        problem_type="random_with_precedence",
        render_mode="human",
        use_idle=True,
        subtractive_assignment=False,
        **kwargs,
    ):
        super().__init__()

        self.n_robots = 5
        self.n_tasks = 15
        self.n_skills = 3
        self.n_precedence = 3
        self.num_robots_available_in_previous_timestep = -1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.render_counter_threshold = 20
        self.use_idle = use_idle
        self.subtractive_assignment = subtractive_assignment
        dim_robots = 7  # (x,y,duration,[skill0, skill1, skill2], available)
        dim_tasks = 9  # (x,y,duration,[skill0, skill1, skill2],ready, assigned, incomplete)

        if self.use_idle:
            self.action_space = gym.spaces.MultiDiscrete(
                [self.n_tasks + 1] * self.n_robots,
            )
        else:
            self.action_space = gym.spaces.MultiDiscrete(
                [self.n_tasks] * self.n_robots,
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

        self.render_mode = render_mode
        self.render_simulation = False
        self.fig, self.ax = None, None
        self.colors = plt.cm.Set1(np.linspace(0, 1, self.n_skills))
        self.problem_type = problem_type
        self.render_fn = None

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

        self.sim = Simulation(
            self.problem_instance, scheduler_name="sadcher_rl", use_idle=self.use_idle
        )

        self.greedy_makespan = greedy_scheduling(self.problem_instance, print_flag=False).makespan
        self.worst_case_makespan = np.sum(self.problem_instance["T_e"]) + np.sum(
            [np.max(self.problem_instance["T_t"][task]) for task in range(self.n_tasks + 1)]
        )

        return self._get_observation(), {}

    def reset_same_problem_instance(self):
        print("Resetting same problem instance")
        self.sim = Simulation(self.problem_instance, scheduler_name="sadcher_rl")

        self.greedy_makespan = greedy_scheduling(self.problem_instance, print_flag=False).makespan

        return self._get_observation(), {}

    def _get_observation(self):
        task_features, robot_features = self.sim.return_task_robot_states()

        if self.subtractive_assignment:
            # Subtract already covered skills
            first_skill_index = 3  # [0,1,2] is location and duration
            skill_slice = slice(first_skill_index, first_skill_index + self.n_skills)
            real_tasks = self.sim.tasks[1:-2] if self.use_idle else self.sim.tasks[1:-1]

            for task_index, task in enumerate(real_tasks):
                required = np.array(task.requirements, dtype=bool)
                covered = np.zeros(self.n_skills, dtype=bool)

                # Task is complete -> all are covered
                if not task.incomplete:
                    covered = required
                else:
                    for robot in self.sim.robots:
                        if robot.current_task is task:
                            covered |= np.array(robot.capabilities, dtype=bool)
                remaining = required & ~covered

                task_features[task_index, skill_slice] = torch.tensor(remaining)

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
        truncated = False

        robot_assignments, filter_triggered = self.calculate_robot_assignment(action)

        if self.render_simulation:
            self._low_level_render()

        self.sim.assign_tasks_to_robots(Instantaneous_Schedule(robot_assignments))
        self.sim.step_until_next_decision_point(
            render_fn=self.render_fn, filter_triggered=filter_triggered
        )

        # Force termination if timestep exceeds worst-case threshold
        if self.sim.timestep >= self.worst_case_makespan:
            self.sim.finish_simulation()
            self.sim.makespan = self.worst_case_makespan
            # print(f"Scheduler did not find a feasible solution at timestep {self.sim.timestep}")
            truncated = True

        reward = self.calculate_reward()
        terminated = self.sim.sim_done

        return self._get_observation(), reward, terminated, truncated, {}

    def calculate_robot_assignment(self, action):
        robot_assignments = {}
        available_robots = [robot for robot in self.sim.robots if robot.available]
        available_robot_ids = [robot.robot_id for robot in available_robots]
        incomplete_tasks = [
            task for task in self.sim.tasks if task.incomplete and task.status == "PENDING"
        ]
        filter_triggered = False
        only_end_task_left = len(incomplete_tasks) == 1
        all_tasks_assigned = all(
            self.sim.all_skills_assigned(task)
            for task in incomplete_tasks
            if task.task_id != self.sim.last_task_id
        )

        if only_end_task_left or all_tasks_assigned:
            robot_assignments = {robot: self.sim.tasks[-1].task_id for robot in available_robot_ids}
        else:
            # If a robot cannot contribute anymore -> send to end location
            for robot in available_robots:
                if not self.sim.robot_can_still_contribute_to_other_tasks(
                    robot, only_full_assignments=False
                ):
                    robot_assignments[robot.robot_id] = self.sim.tasks[-1].task_id

            # Only assign available robots
            action_dict = {
                (robot_id, task_id + 1): 1  # +1 since task 0 is start task (not predicted)
                for (robot_id, task_id) in enumerate(action)
                if robot_id in available_robot_ids and robot_id not in robot_assignments
            }

            action_dict_filtered = filter_redundant_assignments(action_dict, self.sim)
            action_dict_filtered = filter_overassignments(action_dict_filtered, self.sim)
            action_dict_filtered = filter_unqualified_assignments(action_dict_filtered, self.sim)
            robot_assignments = {
                robot: task for (robot, task), val in action_dict_filtered.items() if val == 1
            }

            removed = [
                (rid, tid)
                for (rid, tid), val in action_dict.items()
                if val == 1 and action_dict_filtered.get((rid, tid), 0) == 0
            ]

            filter_triggered = any(
                self.sim.robot_can_still_contribute_to_other_tasks(self.sim.robots[rid])
                for rid, _ in removed
            )

        return robot_assignments, filter_triggered

    def render(self, mode=None):
        self.render_mode = mode
        self.render_simulation = True
        self._low_level_render()
        self.render_fn = self._low_level_render

    def _low_level_render(self):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            plt.ion()
            self.fig.show()
        update_plot(self.sim, self.ax, self.fig, self.colors, self.n_skills, video_mode=True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(0.1)

    def get_config(self):
        return {
            "n_robots": self.n_robots,
            "n_tasks": self.n_tasks,
            "n_skills": self.n_skills,
            "n_precedence": self.n_precedence,
            "problem_type": self.problem_type,
        }

    def return_greedy_makespan(self):
        return self.greedy_makespan

    def return_final_makespan(self):
        if self.sim.sim_done:
            return self.sim.makespan
        else:
            raise ValueError("Simulation not done yet. Cannot return final makespan.")
