#!/usr/bin/env python3
"""Optuna hyper‑parameter search targeting **best average reward** within
150 k episodes on the Scheduling PPO problem.

Key points
~~~~~~~~~~
* **Episode budget**: each trial trains for ≤ 150 000 episodes (across all parallel envs).
* **Objective**: maximum evaluation mean‑reward observed during training.
* **Entropy schedule**: choose *constant* or *linear decay* (no exponential).
* **Models**: "simple" and "IL_pretrained" only.
* **Pruning**: MedianPruner + automatic CUDA‑OOM pruning.
* **GPU safety**:   `mini_batches = ceil((n_envs × n_rollouts) / 2304)`
* **Faster sweep**: learning‑rate sampled from **{1e‑4, 2.5e‑4, 5e‑4}** and learning‑epochs from **{2, 4, 6}** only.

Run examples
------------
```bash
python optuna_scheduler_study.py                     # fresh study
python optuna_scheduler_study.py --resume            # resume existing
```
Adjust `--storage`, `--study-name`, `--n-trials`, or `--n-jobs` on the CLI.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import optuna
import torch

# ── project‑local imports ------------------------------------------------------
from gym_environment_rl import SchedulingRLEnvironment  # env registration happens inside
from optuna.exceptions import TrialPruned

# ── SKRL imports ───────────────────────────────────────────────────────────────
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory

from models.policy_value_discrete import SchedulerPolicy, SchedulerValue

SAFE_BATCH_SAMPLES = 2_304  # 96 rollouts * 96 envs with big model fit on laptop GPU
MAX_EPISODES = 200_000  # hard cap per trial
REPORT_EVERY = 1_000
PLATEAU_DELTA = 0.02  # min absolute improvement considered “progress”
PLATEAU_PATIENCE = 20  # number of report intervals with no progress
SMOOTH_ALPHA = 0.20  # 0<α≤1
###############################################################################
#  Storage helpers                                                             #
###############################################################################


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Optuna study driver (episode‑budget)")
    p.add_argument("--storage", default="sqlite:///optuna_sched.db")
    p.add_argument("--study_name", default="scheduler_search")
    p.add_argument("--n_trials", type=int, default=1_000)
    p.add_argument("--n_jobs", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def get_study(storage_url: str, study_name: str, resume: bool) -> optuna.Study:
    return optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=resume,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=75_000),
    )


###############################################################################
#  Search space                                                                #
###############################################################################


def suggest_params(trial: optuna.Trial) -> Dict:
    cfg: Dict = {}
    cfg["n_envs"] = trial.suggest_categorical("n_envs", [16, 32, 64, 96, 128, 256])
    cfg["n_rollouts"] = trial.suggest_categorical("n_rollouts", [16, 32, 64, 96, 128, 256])
    cfg["policy_type"] = trial.suggest_categorical("policy_type", ["simple", "IL_pretrained"])
    cfg["learning_epochs"] = trial.suggest_categorical("learning_epochs", [2, 4, 6, 8])
    cfg["learning_rate"] = trial.suggest_categorical("learning_rate", [1e-4, 2.5e-4, 5e-4])
    cfg["kl_threshold"] = trial.suggest_categorical("kl_threshold", [0.01, 0.025, 0.03])
    cfg["clip_ratio"] = trial.suggest_categorical("clip_ratio", [0.1, 0.2, 0.3])
    cfg["entropy_mode"] = trial.suggest_categorical("entropy_mode", ["const", "linear_decay"])
    cfg["entropy_init"] = trial.suggest_float("entropy_init", 0.0, 0.02)

    # derived ------------------------------------------------------------------
    batch_size = cfg["n_envs"] * cfg["n_rollouts"]
    cfg["mini_batches"] = math.ceil(batch_size / SAFE_BATCH_SAMPLES)
    cfg["mini_batch_size"] = batch_size // cfg["mini_batches"]
    if cfg["mini_batch_size"] < 256 or cfg["mini_batches"] > 16:
        raise TrialPruned("mini‑batch size impractical")
    return cfg


###############################################################################
#  Model helpers                                                               #
###############################################################################


def get_model_cfgs(policy_type: str) -> Tuple[Dict, Dict, bool]:
    if policy_type == "simple":
        pol_cfg = {
            "robot_input_dimensions": 7,
            "task_input_dimension": 9,
            "embed_dim": 32,
            "ff_dim": 64,
            "n_transformer_heads": 2,
            "n_transformer_layers": 2,
            "n_gatn_heads": 4,
            "n_gatn_layers": 1,
        }
        val_cfg = {
            "robot_input_dim": 7,
            "task_input_dim": 9,
            "embed_dim": 32,
            "ff_dim": 64,
            "n_transformer_heads": 1,
            "n_transformer_layers": 1,
            "n_gatn_heads": 1,
            "n_gatn_layers": 1,
        }
        return pol_cfg, val_cfg, False
    else:
        pol_cfg = {
            "robot_input_dimensions": 7,
            "task_input_dimension": 9,
            "embed_dim": 256,
            "ff_dim": 512,
            "n_transformer_heads": 4,
            "n_transformer_layers": 2,
            "n_gatn_heads": 8,
            "n_gatn_layers": 1,
        }
        val_cfg = {
            "robot_input_dim": 7,
            "task_input_dim": 9,
            "embed_dim": 128,
            "ff_dim": 256,
            "n_transformer_heads": 2,
            "n_transformer_layers": 1,
            "n_gatn_heads": 1,
            "n_gatn_layers": 1,
        }
        return pol_cfg, val_cfg, True


###############################################################################
#  Environment creator & evaluator                                             #
###############################################################################


def make_vec_env(num_envs: int):
    env_id = "SchedulingRLEnvironment-v0"
    if env_id not in gym.registry:
        gym.register(
            id=env_id,
            entry_point="gym_environment_rl:SchedulingRLEnvironment",
            kwargs={
                "problem_type": "random_with_precedence",
                "use_idle": False,
                "subtractive_assignment": True,
            },
        )
    env = gym.make_vec(
        env_id,
        num_envs=num_envs,
        vectorization_mode="async",
        kwargs={
            "problem_type": "random_with_precedence",
            "use_idle": False,
            "subtractive_assignment": True,
        },
    )
    return wrap_env(env)


###############################################################################
#  Training                                                                   #
###############################################################################


def train_and_report(agent: PPO, env, cfg: Dict, trial: optuna.Trial):
    states, _ = env.reset()
    ep_returns = np.zeros(env.num_envs, dtype=np.float32)
    completed_returns: deque[float] = deque(maxlen=500)  # buffer for mean calc

    episodes_done = 0
    best_mean = -np.inf
    ema_mean = None
    step = 0
    plateau_cnt = 0

    while episodes_done < MAX_EPISODES:
        ent = cfg["entropy_init"]
        if cfg["entropy_mode"] == "linear_decay":
            ent *= 1 - episodes_done / MAX_EPISODES
        agent.cfg["entropy_loss_scale"] = ent

        agent.pre_interaction(step, MAX_EPISODES)
        with torch.no_grad():
            actions, _, _ = agent.act(states, timestep=step, timesteps=MAX_EPISODES)
            next_states, rewards, terminated, truncated, infos = env.step(actions)
            agent.record_transition(
                states,
                actions,
                rewards,
                next_states,
                terminated,
                truncated,
                infos,
                step,
                MAX_EPISODES,
            )
        # accumulate returns
        r_np = rewards.cpu().numpy()
        if r_np.ndim > 1:  # e.g. shape (n_envs, n_agents)
            r_np = r_np.mean(axis=-1)  # collapse to scalar per env
        ep_returns += r_np

        done_mask = (terminated | truncated).cpu().numpy()
        if done_mask.ndim > 1:  # collapse if multi‑agent mask
            done_mask = done_mask.any(axis=-1)
        if done_mask.any():
            finished = ep_returns[done_mask]
            completed_returns.extend(finished.tolist())
            ep_returns[done_mask] = 0.0
            episodes_done += int(done_mask.sum())
        agent.post_interaction(step, MAX_EPISODES, episode_counter=episodes_done)
        states = next_states
        step += 1

        # reporting
        if episodes_done % REPORT_EVERY == 0 or episodes_done >= MAX_EPISODES:
            if completed_returns:
                current_mean = float(np.mean(completed_returns))
                if ema_mean is None:
                    ema_mean = current_mean
                else:
                    ema_mean = SMOOTH_ALPHA * current_mean + (1 - SMOOTH_ALPHA) * ema_mean
                metric = ema_mean

                if metric > best_mean + PLATEAU_DELTA:
                    best_mean = metric
                    plateau_cnt = 0
                else:
                    plateau_cnt += 1
                if plateau_cnt >= PLATEAU_PATIENCE:
                    raise TrialPruned("Stalled (plateau or collapse)")

                # -------- Optuna reporting -----------------------------------------
                trial.report(metric, episodes_done)
                if trial.should_prune():
                    raise TrialPruned()

    return best_mean


###############################################################################
# Objective                                                                    #
###############################################################################


def objective(trial: optuna.Trial):
    cfg = suggest_params(trial)
    trial.set_user_attr("mini_batches", cfg["mini_batches"])
    trial.set_user_attr("mini_batch_size", cfg["mini_batch_size"])
    env = make_vec_env(cfg["n_envs"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    memory = RandomMemory(cfg["n_rollouts"], cfg["n_envs"], device)

    p_cfg, v_cfg, il_pre = get_model_cfgs(cfg["policy_type"])
    policy = SchedulerPolicy(
        env.observation_space,
        env.action_space,
        device,
        p_cfg,
        pretrained=il_pre,
        use_idle=False,
        use_positional_encoding=False,
        debug=False,
    )
    value = SchedulerValue(env.observation_space, env.action_space, v_cfg)

    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg.update(
        dict(
            rollouts=cfg["n_rollouts"],
            learning_epochs=cfg["learning_epochs"],
            mini_batches=cfg["mini_batches"],
            learning_rate=cfg["learning_rate"],
            kl_threshold=cfg["kl_threshold"],
            clip_ratio=cfg["clip_ratio"],
            entropy_loss_scale=cfg["entropy_init"],
            mixed_precision=False,
            experiment={"write_interval": 0, "checkpoint_interval": 0},
        )
    )

    agent = PPO(
        {"policy": policy, "value": value},
        memory,
        env.observation_space,
        env.action_space,
        device,
        ppo_cfg,
    )
    agent.init(idle_task_id=env.call("get_config")[0]["n_tasks"])

    try:
        best = train_and_report(agent, env, cfg, trial)
    except TrialPruned:
        # genuine prune: re‑raise
        raise
    except RuntimeError as e:
        # CUDA OOM → prune
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            raise TrialPruned("CUDA OOM")
        # other runtime errors → mark & prune
        torch.cuda.empty_cache()
        trial.set_user_attr("crashed", True)
        raise TrialPruned(f"crashed during training: {e}")
    except Exception as e:
        # any other exception → mark & prune
        torch.cuda.empty_cache()
        trial.set_user_attr("crashed", True)
        raise TrialPruned(f"crashed during training: {e}")
    finally:
        env.close()
        torch.cuda.empty_cache()

    return best


###############################################################################
# Main                                                                         #
###############################################################################


def main():
    args = parse_args()
    study = get_study(args.storage, args.study_name, args.resume)
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)
    print("Best value:", study.best_value)
    for k, v in study.best_params.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
