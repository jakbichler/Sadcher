# SADCHER: Scheduling using Attention-based Dynamic Coalitions of Heterogeneous Robots in Real-Time

<p align="center">
  <img src="media/sadcher_use_case.png" width="1000" />
</p>

This repository accompanies our paper submission to IEEE MRS 2025, called "SADCHER: Scheduling using Attention-based Dynamic Coalitions of Heterogeneous Robots in Real-Time". The project was developed as part of my Master's thesis at Delft University of Technology under the supervision of Prof. Javier Alonso-Mora and Andreu Matoses Gimenez. The [paper](media/Sadcher_IEEE_MRS25.pdf), the [thesis report](media/Bichler_Thesis_Report.pdf) and the [literature review](media/Bichler_Literature_Review.pdf) can  be found in this repo. The released dataset can be downloaded [here](https://data.4tu.nl/datasets/10e28ee0-9ad9-450d-8be7-6e6a91f2931f).

We present Sadcher, a real-time task assignment framework for heterogeneous multi-robot teams that incorporates dynamic coalition formation and task precedence constraints. Sadcher is trained through Imitation Learning and combines graph attention and transformers to predict assignment rewards between robots and tasks. Based on the predicted rewards, a relaxed bipartite matching step generates high-quality schedules with feasibility guarantees. We explicitly model robot and task positions, task durations, and robots’ remaining processing times, enabling advanced temporal and spatial reasoning and generalization to environments with different spatiotemporal distributions compared to training. Trained on optimally solved small-scale instances, our method can scale to larger task sets and team sizes. Sadcher outper- forms other learning-based and heuristic baselines on random- ized, unseen problems for small and medium-sized teams with computation times suitable for real-time operation. We also explore sampling-based variants and evaluate scalability across robot and task counts. In addition, we release our dataset of 250,000 optimal schedules to facilitate future research.

The repository also contains code to benchmark the performance of our method against other methods in the literature, including [HeteroMRTA](https://github.com/marmotlab/HeteroMRTA),  a greedy method and the exact MILP solver.

We also provide code for the experiments with Reinforcement Learning fine-tuning.


## Simple Demo
<p align="center">
  <img src="media/sadcher.gif" width="600" />
</p>

## Network Architecture
<p align="center">
  <img src="media/sadcher_architecture.png" width="1000" />
</p>


## 🔧 Installation

This project uses [`uv`](https://github.com/astral-sh/uv) to manage dependencies and run scripts.  
All requirements are defined in `pyproject.toml`

### 📦 Recommended: Install `uv` - [uv Docs](https://docs.astral.sh/uv/getting-started/installation/)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installing uv, any of the scripts can be ran with 

```bash
uv run script_you_want_to_run.py
```
## 📁 Repository Overview

- **`baselines/`**  
  Contains MILP and greedy solvers for comparison with learning-based methods.

- **`benchmarking/`**  
  Tools to benchmark different schedulers and generate performance statistics and scaling graphs.

- **`data_generation/`**  
  Scripts to generate randomized MRTA problem instances and optimal solution datasets.

- **`helper_functions/`**  
  Core utility classes for tasks, robots, and schedule handling.

- **`imitation_learning/`**  
  Training code, datasets, and pretrained checkpoints for the imitation learning pipeline.

- **`models/`**  
  Neural network components including GATs, Transformers, and the full scheduler architecture.

- **`reinforcement_learning/`**  
  PPO-based reinforcement learning environments and training scripts using [skrl](https://skrl.readthedocs.io/en/latest/) for fine-tuning the scheduler.

- **`schedulers/`**  
  All implemented schedulers including Sadcher (IL), SadcherRL (RL), greedy, and bipartite matchers.

- **`simulation_environment/`**  
  2D simulation engine, configuration, and visualizations of scheduling results.

- **`visualizations/`**  
  Scripts for visualizing benchmark results, scaling behavior, and specific problem instances.

- **`pyproject.toml`**  
  Project metadata and dependencies — all installed automatically when using `uv run`.

## Running the Code
For the most important scripts, we provide an overview of how to run them below. For more details, please refer to the respective script files and their documentation.

### Generate Dataset
To generate a dataset of problem instances and optimal solutions, from within the 
**`data_generation/`** folder, run:

```
uv run generate_dataset.py -n 2000 -o dataset/ -t random_with_precedence --n_robots 3 \
--n_tasks 8 --n_skills 3 --n_precedence 3
```

| Flag                     | Description                                              |
|--------------------------|----------------------------------------------------------|
| `-n`, `--num_instances`  | Number of instances to generate (**required**)           |
| `-o`, `--output_dir`     | Output folder for instances and solutions (**required**) |
| `-t`, `--problem_instance_type` | Type of instance: `random`, `heterogeneous`, etc. (**required**) |
| `--n_robots`             | Number of robots (optional, required for most types)     |
| `--n_tasks`              | Number of tasks (optional, required for most types)      |
| `--n_skills`             | Number of skills (optional, used in heterogeneous types) |
| `--n_precedence`         | Number of precedence constraints (only for `random_with_precedence`) |


### Imitation Learning Training
To train the IL model, from within the  **`imitation_learning/`** folder, run:

```bash
uv run train.py --dataset_dir /path/to/dataset/ --out_checkpoint_dir /path/to/checkpoints/ \
 --continue_training --in_checkpoint_path /path/to/checkpoints/best_checkpoint.pt
```
  

| Flag                    | Description                                                           |
|-------------------------|-----------------------------------------------------------------------|
| `--dataset_dir`         | Path to dataset directory (must contain `problem_instances/` and `solutions/`) |
| `--out_checkpoint_dir`  | Folder to save checkpoints, logs, and loss plots                      |
| `--continue_training`   | Resume training from a checkpoint (optional flag, no value needed)    |
| `--in_checkpoint_path`  | Path to checkpoint to load if continuing training                     |


### RL Finetuning (PPO)

To finetune a PPO agent from an IL-pretrained policy, from the **`reinforcement_learning/`** folder, run:

```bash
uv run ppo_train.py --problem_type random_with_precedence --N_ENVS 32 --N_ROLLOUTS 256 \
--IL_pretrained_policy 
```


| Flag                     | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `--problem_type`         | Type of scheduling problem (e.g. `random_with_precedence`)                  |
| `--N_ENVS`               | Number of parallel environments                                             |
| `--N_ROLLOUTS`           | Environment steps per PPO update                                           |
| `--IL_pretrained_policy` | Use IL-pretrained policy weights                                            |
| `--RL_pretrained`        | Resume PPO training from existing RL checkpoint                             |
| `--RL_pretrained_path`   | Path to the saved PPO model checkpoint                                     |
| `--continuous_RL`        | Use continuous instead of discrete action space                             |
| `--zero_critic`          | Use a dummy critic (no value learning)                                      |
| `--frozen_encoders`      | Freeze encoder layers in policy/value networks                              |
| `--not_use_idle`         | Disable idle task assignment (enabled by default)                           |


### Run Simulation
The file ``simulation_config.yaml`` contains the configuration for the simulation environment. You can modify it to change the number of robots, tasks, skills, and precedence constraints.

To simulate and visualize a scheduling rollout, from the **`simulation_environment/`** folder, run:

```bash
uv run run_simulation.py --scheduler sadcher --visualize
```

Then in the simulation, you can use the following keys to control the simulation: ``n`` to advance the simulation by 1 time step and ``m`` for 10 time steps.

| Flag                     | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `--scheduler`            | Name of the scheduler to use (`sadcher`, `rl_sadcher`, `greedy`, etc.)      |
| `--visualize`            | Run interactive visualization of the simulation                             |
| `--debug`                | Enable debug logging                                                        |
| `--start_end_identical` | Force start and end depot to be the same location (for comparison with HeteroMRTA) |



###  Benchmarking Schedulers

To benchmark different schedulers on randomly generated problems, from the **`benchmarking/`** folder, run:

```bash
uv run benchmark_schedulers.py --n_iterations 100 --include_milp --include_heteromrta \
--include_stochastic_IL_sadcher
```
You can set the number of tasks, robots and precedence constraints in the ``benchmark_schedulers.py`` file directly.

| Flag                                | Description                                                                 |
|-------------------------------------|-----------------------------------------------------------------------------|
| `--n_iterations`                   | Number of benchmark runs to average over (default: 50)                      |
| `--include_milp`                    | Include optimal MILP baseline (very slow for >9 tasks)                          |
| `--include_stochastic_IL_sadcher`  | Include stochastic sampling version of IL-based Sadcher                    |
| `--include_heteromrta`             | Include HeteroMRTA baseline for comparison                                  |


### Generate Scaling Graph Data

To generate runtime and performance data for scaling graphs, from the **`benchmarking/`** folder, run:

```bash
uv run create_data_for_scaling_graphs.py \
  --min_tasks 10 --max_tasks 250 --step_tasks 10 \
  --min_robots 3 --max_robots 7 --step_robots 2 \
  --n_runs 5 --output_file /path/to/scaling.json
```
| Flag                 | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `--min_tasks`        | Minimum number of tasks per instance                                       |
| `--max_tasks`        | Maximum number of tasks per instance                                       |
| `--step_tasks`       | Step size for increasing number of tasks (default: 1)                      |
| `--min_robots`       | Minimum number of robots per instance                                      |
| `--max_robots`       | Maximum number of robots per instance                                      |
| `--step_robots`      | Step size for increasing number of robots (default: 1)                     |
| `--n_runs`           | Number of repeated trials per (tasks, robots) configuration (default: 10) |
| `--include_milp`     | Include MILP baseline (slow for large task sizes)                          |
| `--milp_cutoff_time` | Max MILP solve time in seconds (default: 600)                              |
| `--output_file`      | Path to JSON file to store results                                        |


###  Plot Scaling Graphs

To generate and visualize scaling plots from the JSON results, from the **`benchmarking/`** folder, run:

```bash
uv run scaling_graphs_visualizations.py --input_file /path/to/scaling.json --n_robots 3
```
| Flag            | Description                                                    |
|------------------|----------------------------------------------------------------|
| `--input_file`   | Path to input JSON file generated during scaling runs   |
| `--n_robots`     | Number of robots to filter for in the visualization           |
