import os
import pickle
import sys

import numpy as np

sys.path.append(os.path.abspath("/home/jakob/HeteroMRTA/"))
sys.path.append("../..")

from env.task_env import TaskEnv

from data_generation.problem_generator import generate_random_data_with_precedence


def build_env(task_specs, species_specs):
    task_dic, agent_dic, depot_dic = {}, {}, {}
    species_dict = {"abilities": [], "number": []}

    # ---------- tasks ----------
    for tid, t in enumerate(task_specs):
        task_dic[tid] = {
            "ID": tid,
            "requirements": np.array(t["req"]),
            "members": [],
            "cost": [],
            "location": np.array(t["loc"]),
            "feasible_assignment": False,
            "finished": False,
            "time_start": 0,
            "time_finish": 0,
            "status": np.array(t["req"]),
            "time": float(t["dur"]),
            "sum_waiting_time": 0,
            "efficiency": 0,
            "abandoned_agent": [],
            "optimized_ability": None,
            "optimized_species": [],
        }

    # ---------- species / agents ----------
    aid = 0
    for sid, s in enumerate(species_specs):
        # abilities & counts
        species_dict["abilities"].append(np.array(s["abil"]))
        species_dict["number"].append(s["n"])
        species_dict[sid] = []

        # depot
        depot_dic[sid] = {"location": np.array(s["depot"]), "members": [], "ID": -sid - 1}

        # agents
        for _ in range(s["n"]):
            agent_dic[aid] = {
                "ID": aid,
                "species": sid,
                "abilities": np.array(s["abil"]),
                "location": np.array(s["depot"]),
                "route": [-sid - 1],
                "current_task": -sid - 1,
                "contributed": False,
                "arrival_time": [0.0],
                "cost": [0.0],
                "travel_time": 0.0,
                "velocity": 0.2,
                "next_decision": 0.0,
                "depot": np.array(s["depot"]),
                "travel_dist": 0.0,
                "sum_waiting_time": 0.0,
                "current_action_index": 0,
                "decision_step": 0,
                "task_waiting_ratio": 1,
                "trajectory": [],
                "angle": 0.0,
                "returned": False,
                "assigned": False,
                "pre_set_route": None,
                "no_choice": False,
            }
            depot_dic[sid]["members"].append(aid)
            species_dict[sid].append(aid)  # <- NEW
            aid += 1

    species_dict["abilities"] = np.vstack(species_dict["abilities"])
    return task_dic, agent_dic, depot_dic, species_dict


def _pad(v, dim=5):
    "Pad 2-D array v with trailing zeros so it has exactly <dim> columns."
    if v.shape[1] == dim:
        return v
    pad = dim - v.shape[1]
    return np.hstack([v, np.zeros((v.shape[0], pad), dtype=v.dtype)])


def problem_to_taskenv(pb, grid_size, duration_factor):
    """
    Convert a ProblemData instance (possibly using only 3 skills)
    into a TaskEnv that always has 5 skill slots.
    """
    Q_raw, R_raw = pb["Q"], pb["R"][1:-1]  # strip start/end dummy tasks
    Q, R = _pad(Q_raw), _pad(R_raw)  # <- **padding step**
    T_e = pb["T_e"][1:-1]
    loc = pb["task_locations"][1:-1] / grid_size  # scale to [0,1]

    # ----------- build species list (unique ability patterns) -----------------
    uniq, inv = np.unique(Q, axis=0, return_inverse=True)
    species_specs = [
        {
            "depot": pb["task_locations"][0] / grid_size,
            "abil": uniq[s],
            "n": int(sum(inv == s)),
        }
        for s in range(len(uniq))
    ]

    # ----------- task list ----------------------------------------------------
    tasks = [
        {"loc": loc[k], "req": R[k].astype(int), "dur": float(T_e[k]) / duration_factor}
        for k in range(len(R))
    ]

    # ----------- instantiate TaskEnv -----------------------------------------
    env = TaskEnv(traits_dim=5)  # ranges don't matter; we overwrite
    env.reset(test_env=build_env(tasks, species_specs))
    return env


# --------------- example ---------------
if __name__ == "__main__":
    np.random.seed(4)
    grid_size = 100
    sadcher_max_task_duration = 100
    marmot_max_task_duration = 5
    duration_factor = sadcher_max_task_duration / marmot_max_task_duration

    problem_instance = generate_random_data_with_precedence(
        n_tasks=10, n_robots=3, n_skills=3, n_precedence=0
    )

    problem_instance["task_locations"][-1] = problem_instance["task_locations"][
        0
    ]  # make start end depot same to comply with MARMOTLAB code

    env = problem_to_taskenv(problem_instance, grid_size, duration_factor)

    import os

    pkl_path = "SadcherTestSet/env_0.pkl"
    os.makedirs("SadcherTestSet/env_0", exist_ok=True)
    pickle.dump(env, open(pkl_path, "wb"))
    print("✅ custom env saved")
