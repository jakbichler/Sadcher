from __future__ import annotations

import argparse
import math
import sys

sys.path.append("..")
from collections import deque
from typing import Dict, Tuple

import gymnasium as gym
import optuna
import torch
from optuna.exceptions import TrialPruned
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory

from models.policy_value_continuous import ContinuousSchedulerPolicy, ZeroCritic

SAFE_BATCH_SAMPLES_FROZEN = 1024
SAFE_BATCH_SAMPLES_UNFROZEN = 512
MAX_EPISODES = 150_000  # hard cap per trial
REPORT_EVERY = 500
PLATEAU_DELTA = 0.003  # min absolute improvement considered “progress”
PLATEAU_PATIENCE = 75  # number of report intervals with no progress
SMOOTH_ALPHA = 0.01  # 0<α≤1
EMA_INITIAL = 0.04  # initial value for EMA (for 20 5 we expect roughly 4% advantage over greedy in the beginning)
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
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50_000),
    )


###############################################################################
#  Search space                                                                #
###############################################################################


def suggest_params(trial: optuna.Trial) -> Dict:
    cfg: Dict = {}
    cfg["n_envs"] = trial.suggest_categorical("n_envs", [48, 64, 96, 128, 256])
    cfg["n_rollouts"] = trial.suggest_categorical("n_rollouts", [32, 64, 96, 128, 256])
    cfg["learning_epochs"] = trial.suggest_categorical("learning_epochs", [2, 4, 6])
    cfg["learning_rate"] = trial.suggest_categorical("learning_rate", [5e-5, 1e-4, 5e-4])
    cfg["kl_threshold"] = trial.suggest_categorical("kl_threshold", [0.01, 0.02, 0.03])
    cfg["clip_ratio"] = trial.suggest_categorical("clip_ratio", [0.05, 0.1, 0.2])
    cfg["frozen_encoders"] = trial.suggest_categorical("frozen_encoders", [True, False])

    # derived ------------------------------------------------------------------
    batch_size = cfg["n_envs"] * cfg["n_rollouts"]
    batch_divider = (
        SAFE_BATCH_SAMPLES_FROZEN if cfg["frozen_encoders"] else SAFE_BATCH_SAMPLES_UNFROZEN
    )
    cfg["mini_batches"] = math.ceil(batch_size / batch_divider)
    cfg["mini_batch_size"] = batch_size // cfg["mini_batches"]
    if cfg["mini_batches"] > 32:
        raise TrialPruned("mini‑batch size impractical")
    return cfg


#########################################50####################################
#  Model helpers                                                               #
###############################################################################


def get_model_cfgs() -> Tuple[Dict, Dict, bool]:
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
    return pol_cfg


###############################################################################
#  Environment creator & evaluator                                             #
###############################################################################


def make_vec_env(num_envs: int):
    env_id = "ContinuousSchedulingRLEnvironment-v0"
    if env_id not in gym.registry:
        gym.register(
            id=env_id,
            entry_point="continuous_gym_environment_rl:ContinuousSchedulingRLEnvironment",
            kwargs={
                "problem_type": "random_with_precedence",
                "use_idle": True,
                "subtractive_assignment": False,
            },
        )
    env = gym.make_vec(
        env_id,
        num_envs=num_envs,
        vectorization_mode="async",
        kwargs={
            "problem_type": "random_with_precedence",
            "use_idle": True,
            "subtractive_assignment": False,
        },
    )
    return wrap_env(env)


###############################################################################
#  Training                                                                   #
###############################################################################


def train_and_report(agent: PPO, env, cfg: Dict, trial: optuna.Trial):
    states, _ = env.reset()
    ep_returns = torch.zeros(env.num_envs, device=agent.device)
    completed_returns: deque[float] = deque(maxlen=REPORT_EVERY)  # buffer for mean calc

    episodes_done = 0
    best_mean = -2  # Given our reward formulation, - 2 is the worst possible mean
    ema_mean = None
    step = 0
    plateau_cnt = 0
    next_report_at = REPORT_EVERY

    while episodes_done < MAX_EPISODES:
        agent.pre_interaction(step, MAX_EPISODES)

        with torch.inference_mode():
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
            r = rewards.mean(-1) if rewards.ndim > 1 else rewards
            ep_returns += r

            done_mask = terminated | truncated
            if done_mask.ndim > 1:  # collapse if multi‑agent mask
                done_mask = done_mask.any(-1)
            if done_mask.any():
                finished = ep_returns.masked_select(done_mask)
                completed_returns.extend(finished.tolist())
                ep_returns.masked_fill_(done_mask, 0.0)
                episodes_done += int(done_mask.sum())

        agent.post_interaction(step, MAX_EPISODES, episode_counter=episodes_done)
        states = next_states
        step += 1

        # -------- Optuna reporting -------------------------------------------
        if episodes_done >= next_report_at or episodes_done >= MAX_EPISODES:
            if completed_returns:
                returns_t = torch.tensor(list(completed_returns), device=agent.device)
                current_mean = returns_t.mean()
                ema_mean = (
                    EMA_INITIAL
                    if ema_mean is None
                    else SMOOTH_ALPHA * current_mean + (1 - SMOOTH_ALPHA) * ema_mean
                )
                metric = float(ema_mean)

                if metric > best_mean + PLATEAU_DELTA:
                    best_mean = metric
                    plateau_cnt = 0
                else:
                    plateau_cnt += 1
                if plateau_cnt >= PLATEAU_PATIENCE:
                    trial.set_user_attr("stop reason", "plateau/collapse")
                    break

                trial.report(metric, episodes_done)
                if trial.should_prune():
                    trial.set_user_attr("stop reason", "median_pruned")
                    break

            next_report_at += REPORT_EVERY

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

    p_cfg = get_model_cfgs()
    policy = ContinuousSchedulerPolicy(
        env.observation_space,
        env.action_space,
        device,
        p_cfg,
        pretrained=True,
        use_idle=True,
        debug=False,
        frozen_encoders=cfg["frozen_encoders"],
    )
    value = ZeroCritic(env.observation_space, env.action_space, device)

    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg.update(
        dict(
            rollouts=cfg["n_rollouts"],
            learning_epochs=cfg["learning_epochs"],
            mini_batches=cfg["mini_batches"],
            learning_rate=cfg["learning_rate"],
            kl_threshold=cfg["kl_threshold"],
            clip_ratio=cfg["clip_ratio"],
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
