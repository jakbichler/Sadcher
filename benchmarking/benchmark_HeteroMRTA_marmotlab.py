import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.append("..")

from baselines.heteromrta.attention import AttentionNet
from baselines.heteromrta.bridge import problem_to_taskenv
from baselines.heteromrta.env.task_env import TaskEnv
from baselines.heteromrta.parameters import EnvParams, TrainParams
from baselines.heteromrta.worker import Worker
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
N_TASKS = 10
N_ROBOTS = 3
N_SKILLS = 3
N_PRECEDENCE = 3
N_RUNS = 100
N_SAMPLING_STOCH = 10
RANDOM_SEED = 123
GRID_SIZE = 100
DURATION_FACTOR = 100 / 5
MAKESPAN_FACTOR = 20
TRAVEL_DISTANCE_FACTOR = 100
CHECKPOINT_SADCHER = (
    "/home/jakob/thesis/imitation_learning/checkpoints/hyperparam_2_8t3r3s/best_checkpoint.pt"
)
CHECKPOINT_HETEROMRTA = "/home/jakob/HeteroMRTA/model/save/checkpoint.pth"


sadcher_makespans, sadcher_travel_distances, sadcher_times = [], [], []
stoch_sadcher_makespans, stoch_sadcher_travel_distances, stoch_sadcher_times = [], [], []
heteromrta_makespans, heteromrta_times, heteromrta_travel_distances = [], [], []
stoch_heteromrta_makespans, stoch_heteromrta_times, stoch_heteromrta_travel_distances = [], [], []


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
    sadcher_travel_distance = calculate_traveled_distance(
        rolled_out_schedule, problem_instance["T_t"]
    )
    t_sad = time.time() - t0
    sadcher_times.append(t_sad)
    sadcher_makespans.append(sim.makespan)
    sadcher_travel_distances.append(sadcher_travel_distance)

    #  ─── Stochastic  Sadcher run ─────────────────────────────────────────────────────────
    best_ms, best_dist = None, None
    t_sto0 = time.time()

    for _ in range(N_SAMPLING_STOCH):
        sim_sto = Simulation(problem_instance, scheduler_name="stochastic_sadcher", debug=False)
        scheduler_sto = create_scheduler(
            "stochastic_IL_sadcher",
            CHECKPOINT_SADCHER,
            duration_normalization=sim_sto.duration_normalization,
            location_normalization=sim_sto.location_normalization,
            debugging=False,
            stddev=1,  # ← tweak if you want a differeqqnt exploration width
        )

        while not sim_sto.sim_done:
            pred_reward, inst_sched = scheduler_sto.calculate_robot_assignment(sim_sto)
            sim_sto.find_highest_non_idle_reward(pred_reward)
            sim_sto.assign_tasks_to_robots(inst_sched)
            sim_sto.step_until_next_decision_point()

        sched = Full_Horizon_Schedule(sim_sto.makespan, sim_sto.robot_schedules, N_TASKS)
        travel = calculate_traveled_distance(sched, problem_instance["T_t"])

        if (best_ms is None) or (sim_sto.makespan < best_ms):
            best_ms, best_dist = sim_sto.makespan, travel

    stoch_sadcher_times.append(time.time() - t_sto0)
    stoch_sadcher_makespans.append(best_ms)
    stoch_sadcher_travel_distances.append(best_dist)

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
    heteromrta_times.append(t_het)
    heteromrta_makespans.append(rl_res["makespan"][-1])
    heteromrta_travel_distances.append(rl_res["travel_dist"][-1])

    # ─── Stochastic/Sampling HeteroMRTA run ───────────────────────────────────────────────────────
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

    heteromrta_results_best = None
    t1 = time.time()
    for _ in range(N_SAMPLING_STOCH):
        env.init_state()
        worker.env = env
        _, _, heteromrta_results = worker.run_episode(False, sample=True, max_waiting=False)
        if heteromrta_results_best is None:
            heteromrta_results_best = heteromrta_results
        else:
            if heteromrta_results_best["makespan"] >= heteromrta_results["makespan"]:
                heteromrta_results_best = heteromrta_results

    t_het = time.time() - t1
    stoch_heteromrta_times.append(t_het)
    stoch_heteromrta_makespans.append(heteromrta_results_best["makespan"][-1])
    stoch_heteromrta_travel_distances.append(heteromrta_results_best["travel_dist"][-1])


# ─── Conversion for comparison  ───────────────────────────────────────────────
heteromrta_makespans = [ms * MAKESPAN_FACTOR for ms in heteromrta_makespans]
heteromrta_travel_distances = [td * TRAVEL_DISTANCE_FACTOR for td in heteromrta_travel_distances]
stoch_heteromrta_makespans = [ms * MAKESPAN_FACTOR for ms in stoch_heteromrta_makespans]
stoch_heteromrta_travel_distances = [
    td * TRAVEL_DISTANCE_FACTOR for td in stoch_heteromrta_travel_distances
]

# ─── Plots (all algorithms) ────────────────────────────────────────────────
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle(
    f"{N_RUNS} random instances, with: {N_TASKS}t | {N_ROBOTS}r | {N_SKILLS}s | {N_PRECEDENCE}p"
)

labels = [
    "Sadcher",
    f"Sto-Sadcher-{N_SAMPLING_STOCH}",
    "HeteroMRTA",
    f"Sto-HeteroMRTA-{N_SAMPLING_STOCH}",
]

means_ms = list(
    map(
        np.mean,
        [
            sadcher_makespans,
            stoch_sadcher_makespans,
            heteromrta_makespans,
            stoch_heteromrta_makespans,
        ],
    )
)
means_time = list(
    map(np.mean, [sadcher_times, stoch_sadcher_times, heteromrta_times, stoch_heteromrta_times])
)
means_dist = list(
    map(
        np.mean,
        [
            sadcher_travel_distances,
            stoch_sadcher_travel_distances,
            heteromrta_travel_distances,
            stoch_heteromrta_travel_distances,
        ],
    )
)

# ─ Makespan  ─
ax = axs[0]
ax.violinplot(
    [sadcher_makespans, stoch_sadcher_makespans, heteromrta_makespans, stoch_heteromrta_makespans],
    positions=[1, 2, 3, 4],
    showmeans=True,
)
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(labels, rotation=12)
ax.set_ylabel("Makespan")
ax.set_title("Makespan")
for x, m in enumerate(means_ms, start=1):
    ax.text(x + 0.25, m, f"{m:.1f}", ha="center", va="bottom")

# ─ Runtime  ─
ax = axs[1]
ax.violinplot(
    [sadcher_times, stoch_sadcher_times, heteromrta_times, stoch_heteromrta_times],
    positions=[1, 2, 3, 4],
    showmeans=True,
)
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(labels, rotation=12)
ax.set_ylabel("Wall-clock time (s)")
ax.set_title("Runtime")
for x, m in enumerate(means_time, start=1):
    ax.text(x + 0.25, m, f"{m:.2f}", ha="center", va="bottom")

# ─ Travel distance  ─
ax = axs[2]
ax.violinplot(
    [
        sadcher_travel_distances,
        stoch_sadcher_travel_distances,
        heteromrta_travel_distances,
        stoch_heteromrta_travel_distances,
    ],
    positions=[1, 2, 3, 4],
    showmeans=True,
)
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(labels, rotation=12)
ax.set_ylabel("Travel distance")
ax.set_title("Distance")
for x, m in enumerate(means_dist, start=1):
    ax.text(x + 0.25, m, f"{m:.1f}", ha="center", va="bottom")

plt.tight_layout()
plt.show()
