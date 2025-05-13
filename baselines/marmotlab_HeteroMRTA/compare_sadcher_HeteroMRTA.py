import os
import sys
import time

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("/home/jakob/HeteroMRTA/"))
sys.path.append("../..")

import numpy as np
import pandas as pd
import torch
from attention import AttentionNet
from bridge import problem_to_taskenv
from env.task_env import TaskEnv
from parameters import EnvParams, TrainParams
from worker import Worker

from data_generation.problem_generator import generate_random_data_with_precedence
from helper_functions.schedules import Full_Horizon_Schedule
from schedulers.initialize_schedulers import create_scheduler
from simulation_environment.simulator_2D import Simulation

CONFIG = {
    "n_tasks": 10,
    "n_robots": 3,
    "n_skills": 3,
    "n_precedence": 1,
    "random_seed": 42,
    "grid_size": 100,
    "duration_factor": 100 / 5,  # sadcher_max_task_duration / marmot_max_task_duration
    "checkpoint_sadcher": "/home/jakob/thesis/imitation_learning/checkpoints/hyperparam_2_8t3r3s/best_checkpoint.pt",
    "checkpoint_marmotlab": "/home/jakob/HeteroMRTA/model/save/checkpoint.pth",
    "save_dir": "results",
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)
np.random.seed(CONFIG["random_seed"])
torch.manual_seed(CONFIG["random_seed"])

problem_instance = generate_random_data_with_precedence(
    CONFIG["n_tasks"],
    CONFIG["n_robots"],
    CONFIG["n_skills"],
    CONFIG["n_precedence"],
)

# HeteroMRTA can only deal with end/start being same, so
problem_instance["task_locations"][-1] = problem_instance["task_locations"][0]


# Sadcher
sim = Simulation(problem_instance, scheduler_name="sadcher", debug=False)
scheduler = create_scheduler(
    "sadcher",
    CONFIG["checkpoint_sadcher"],
    "8t3r3s",
    duration_normalization=sim.duration_normalization,
    location_normalization=sim.location_normalization,
    debugging=False,
)
start_sim = time.time()
while not sim.sim_done:
    predicted_reward, instantaneous_schedule = scheduler.calculate_robot_assignment(sim)
    sim.find_highest_non_idle_reward(predicted_reward)
    sim.assign_tasks_to_robots(instantaneous_schedule)
    sim.step_until_next_decision_point()
sim_time = time.time() - start_sim
sim_schedule = Full_Horizon_Schedule(sim.makespan, sim.robot_schedules, CONFIG["n_tasks"])
sim_ct = sum(sim.scheduler_computation_times)

# HeteroMRTA
EnvParams.TRAIT_DIM = 5
TrainParams.EMBEDDING_DIM = 128
TrainParams.AGENT_INPUT_DIM = 6 + EnvParams.TRAIT_DIM
TrainParams.TASK_INPUT_DIM = 5 + 2 * EnvParams.TRAIT_DIM

env: TaskEnv = problem_to_taskenv(problem_instance, CONFIG["grid_size"], CONFIG["duration_factor"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = AttentionNet(
    TrainParams.AGENT_INPUT_DIM,
    TrainParams.TASK_INPUT_DIM,
    TrainParams.EMBEDDING_DIM,
).to(device)
ckpt = torch.load(CONFIG["checkpoint_marmotlab"], map_location="cpu")
net.load_state_dict(ckpt["best_model"])
worker = Worker(0, net, net, 0, device)

start_rl = time.time()
env.init_state()
worker.env = env
_, _, rl_results = worker.run_episode(False, sample=False, max_waiting=False)
rl_time = time.time() - start_rl


"""
 Conversion between Sadcher/HeteroMRTA is 20:
 Sadcher: locations are in [0,100], with robot_speed 1 and task durations in [50,100] 
 HeteroMRTA: locations are in [0,1], with robot speed 0.2 and task durations in [0,5]
 -> problem instance input is scaled down in the bridge for the input to have the same
  distribution as  the one HeteroMRTA is trained on
"""
MAKESPAN_CONVERSION = 20
rl_makespan_scaled = [m * MAKESPAN_CONVERSION for m in rl_results["makespan"]]
# 4) Aggregate and save
df = pd.DataFrame(
    [
        {
            "method": "simulation",
            "makespan": sim.makespan,
            "wall_time": sim_time,
        },
        {
            "method": "rl",
            "makespan": rl_makespan_scaled,
            "wall_time": rl_time,
        },
    ]
)
print(df)
