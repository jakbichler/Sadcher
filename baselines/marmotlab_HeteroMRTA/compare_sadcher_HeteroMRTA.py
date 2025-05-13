import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# TO-DO: FIX IMPORTS
sys.path.append(os.path.abspath("/home/jakob/HeteroMRTA/"))
sys.path.append("../..")

from attention import AttentionNet
from bridge import problem_to_taskenv
from env.task_env import TaskEnv
from parameters import EnvParams, TrainParams
from worker import Worker

from data_generation.problem_generator import generate_random_data_with_precedence
from helper_functions.schedules import Full_Horizon_Schedule, calculate_traveled_distance
from schedulers.initialize_schedulers import create_scheduler
from simulation_environment.simulator_2D import Simulation

"""
Scaling between Sadcher and HeteroMRTA (https://github.com/marmotlab/HeteroMRTA):

– In Sadcher, tasks live on a [0,100]×[0,100] grid with robot speed = 1 unit/timestep.
– In HeteroMRTA, tasks live on a [0,1]×[0,1] grid with robot speed = 0.2 units/timestep.

Since 1 unit in the HeteroMRTA grid corresponds to 100 units in Sadcher’s grid, 
and the HeteroMRTA robot moves at 0.2 versus Sadcher’s speed of 1, 
the effective traversal rate in HeteroMRTA is 0.2×100 = 20× faster 
than in Sadcher.

– In Sadcher, durations are in [50,100] units 
– In HeteroMRTA in [0,5]

So the duration factor is again 20x. 

For fair comparison, we first scale all inputs in the bridge between sadcher problem instance and 
HeteroMRTA environment to match the distribution that HeteroMRTA was trained on.
Afterwards, we multiply all HeteroMRTA makespans by 20 to compare the values.
This results in the exact fair comparison (can be verified: instances with same schedule have same makespan).
"""
N_TASKS = 50
N_ROBOTS = 20
N_SKILLS = 3
N_PRECEDENCE = 0
N_RUNS = 100
RANDOM_SEED = 0
GRID_SIZE = 100
DURATION_FACTOR = 100 / 5
MAKESPAN_FACTOR = 20
TRAVEL_DISTANCE_FACTOR = 100
CHECKPOINT_SADCHER = (
    "/home/jakob/thesis/imitation_learning/checkpoints/hyperparam_2_8t3r3s/best_checkpoint.pt"
)
CHECKPOINT_HETEROMRTA = "/home/jakob/HeteroMRTA/model/save/checkpoint.pth"
# ──────────────────────────────────────────────────────────────────────────────

sadcher_makespans = []
sadcher_travel_distances = []
sadcher_times = []
heteromrta_makespans = []
heteromrta_times = []
heteromrta_travel_distances = []

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

for _ in tqdm(range(N_RUNS)):
    problem_instance = generate_random_data_with_precedence(
        N_TASKS, N_ROBOTS, N_SKILLS, N_PRECEDENCE
    )

    # HeteroMRTA requires task/end location being the same:
    problem_instance["task_locations"][-1] = problem_instance["task_locations"][0]

    # ─── Sadcher run ─────────────────────────────────────────────────────────
    sim = Simulation(problem_instance, scheduler_name="sadcher", debug=False)
    scheduler = create_scheduler(
        "sadcher",
        CHECKPOINT_SADCHER,
        "8t3r3s",
        duration_normalization=sim.duration_normalization,
        location_normalization=sim.location_normalization,
        debugging=False,
    )
    t0 = time.time()
    while not sim.sim_done:
        pred_reward, inst_sched = scheduler.calculate_robot_assignment(sim)
        sim.find_highest_non_idle_reward(pred_reward)
        sim.assign_tasks_to_robots(inst_sched)
        sim.step_until_next_decision_point()
    rolled_out_schedule = Full_Horizon_Schedule(sim.makespan, sim.robot_schedules, N_TASKS)
    sadcher_travel_distances.append(
        calculate_traveled_distance(rolled_out_schedule, problem_instance["T_t"])
    )
    t_sad = time.time() - t0
    sadcher_makespans.append(sim.makespan)
    sadcher_times.append(t_sad)

    # ─── HeteroMRTA run ───────────────────────────────────────────────────────
    EnvParams.TRAIT_DIM = 5
    TrainParams.EMBEDDING_DIM = 128
    TrainParams.AGENT_INPUT_DIM = 6 + EnvParams.TRAIT_DIM
    TrainParams.TASK_INPUT_DIM = 5 + 2 * EnvParams.TRAIT_DIM

    env: TaskEnv = problem_to_taskenv(problem_instance, GRID_SIZE, DURATION_FACTOR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AttentionNet(
        TrainParams.AGENT_INPUT_DIM,
        TrainParams.TASK_INPUT_DIM,
        TrainParams.EMBEDDING_DIM,
    ).to(device)
    ckpt = torch.load(CHECKPOINT_HETEROMRTA, map_location=device)
    net.load_state_dict(ckpt["best_model"])
    worker = Worker(0, net, net, 0, device)

    t1 = time.time()
    env.init_state()
    worker.env = env
    _, _, rl_res = worker.run_episode(False, sample=False, max_waiting=False)

    t_het = time.time() - t1
    heteromrta_makespans.append(rl_res["makespan"][-1])
    heteromrta_travel_distances.append(rl_res["travel_dist"][-1])
    heteromrta_times.append(t_het)


# ─── Conversion for comparison  ───────────────────────────────────────────────────────────
heteromrta_makespans = [ms * MAKESPAN_FACTOR for ms in heteromrta_makespans]
heteromrta_travel_distances = [td * TRAVEL_DISTANCE_FACTOR for td in heteromrta_travel_distances]

# ─── Plots ───────────────────────────────────────────────────────────
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle(
    f"On {N_RUNS} randomized instances of {N_TASKS}t, {N_ROBOTS}r, {N_SKILLS}s, {N_PRECEDENCE}p"
)

mean_sad_ms = np.mean(sadcher_makespans)
mean_het_ms = np.mean(heteromrta_makespans)
mean_sad_t = np.mean(sadcher_times)
mean_het_t = np.mean(heteromrta_times)
mean_sad_dist = np.mean(sadcher_travel_distances)
mean_het_dist = np.mean(heteromrta_travel_distances)

# ─ Makespan  ─
ax = axs[0]
ax.violinplot([sadcher_makespans, heteromrta_makespans], positions=[1, 2], showmeans=True)
ax.text(1.25, mean_sad_ms, f"{mean_sad_ms:.1f}", ha="center", va="bottom")
ax.text(2.25, mean_het_ms, f"{mean_het_ms:.1f}", ha="center", va="bottom")
ax.set_xticks([1, 2])
ax.set_xticklabels(["Sadcher", "HeteroMRTA"])
ax.set_ylabel("Makespan")
ax.set_title("Makespan Comparison")

# ─ Runtime  ─
ax = axs[1]
ax.violinplot([sadcher_times, heteromrta_times], positions=[1, 2], showmeans=True)
ax.text(1.25, mean_sad_t, f"{mean_sad_t:.2f}", ha="center", va="bottom")
ax.text(2.25, mean_het_t, f"{mean_het_t:.2f}", ha="center", va="bottom")
ax.set_xticks([1, 2])
ax.set_xticklabels(["Sadcher", "HeteroMRTA"])
ax.set_ylabel("Wall Time (s)")
ax.set_title("Runtime Comparison")

# ─ Travel Distance  ─
ax = axs[2]
ax.violinplot(
    [sadcher_travel_distances, heteromrta_travel_distances], positions=[1, 2], showmeans=True
)
ax.text(1.25, mean_sad_dist, f"{mean_sad_dist:.1f}", ha="center", va="bottom")
ax.text(2.25, mean_het_dist, f"{mean_het_dist:.1f}", ha="center", va="bottom")
ax.set_xticks([1, 2])
ax.set_xticklabels(["Sadcher", "HeteroMRTA"])
ax.set_ylabel("Travel Distance")
ax.set_title("Distance Comparison")

plt.tight_layout()
plt.show()
