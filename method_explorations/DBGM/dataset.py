import os
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from training_helpers import (
    create_robot_features_from_optimal, 
    create_task_features_from_optimal, 
    get_expert_reward, 
    find_decision_points
)
from helper_functions.schedules import Full_Horizon_Schedule  # Ensure this import is correct

class LazyLoadedSchedulingDataset(Dataset):
    def __init__(self, problem_dir, solution_dir, gamma=0.99, immediate_reward=10):
        """
        Lazy loading version of the dataset.
        Instead of storing all samples in memory, we store problem & solution file paths
        and process each sample only when requested.
        """
        self.problem_dir = problem_dir
        self.solution_dir = solution_dir
        self.gamma = gamma
        self.immediate_reward = immediate_reward

        # Collect file paths
        self.problem_files = sorted(os.listdir(problem_dir))
        self.solution_files = sorted(os.listdir(solution_dir))

        # Ensure same number of problems and solutions
        assert len(self.problem_files) == len(self.solution_files), "Mismatch in problem and solution count."

        # Precompute decision points for each problem-solution pair
        self.data_indices = []  # (problem_idx, decision_time) pairs
        for i in tqdm(range(len(self.problem_files)), desc="Indexing Dataset"):
            problem_path = os.path.join(problem_dir, self.problem_files[i])
            solution_path = os.path.join(solution_dir, self.solution_files[i])

            with open(solution_path, "r") as f:
                solution_obj = Full_Horizon_Schedule.from_dict(json.load(f))

            decision_points = find_decision_points(solution_obj)
            self.data_indices.extend([(i, dec_time) for dec_time in decision_points])

        
        first_problem_path = os.path.join(problem_dir, self.problem_files[0])
        first_solution_path = os.path.join(solution_dir, self.solution_files[0])
        with open(first_problem_path, "r") as f:
            first_problem = json.load(f)
        with open(first_solution_path, "r") as f:
            first_solution = Full_Horizon_Schedule.from_dict(json.load(f))

        first_robot_feats = create_robot_features_from_optimal(first_problem, first_solution.robot_schedules, 0, 100, 100)
        first_task_feats = create_task_features_from_optimal(first_problem, first_solution.robot_schedules, 0, 100, 100)

        self.n_robots = first_robot_feats.shape[0]
        self.robot_dim = first_robot_feats.shape[1]
        
        self.n_tasks = first_task_feats.shape[0]
        self.task_dim = first_task_feats.shape[1]



    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        """
        Load only the necessary problem-solution pair and process the features on demand.
        """
        problem_idx, decision_time = self.data_indices[idx]

        # Load problem
        problem_path = os.path.join(self.problem_dir, self.problem_files[problem_idx])
        with open(problem_path, "r") as f:
            problem = json.load(f)

        # Load solution
        solution_path = os.path.join(self.solution_dir, self.solution_files[problem_idx])
        with open(solution_path, "r") as f:
            solution_obj = Full_Horizon_Schedule.from_dict(json.load(f))

        # Normalization factors
        location_normalization = np.max(problem["task_locations"])
        duration_normalization = np.max(problem["T_e"])

        # Generate features
        robot_feats = create_robot_features_from_optimal(problem, solution_obj.robot_schedules, decision_time, location_normalization, duration_normalization)
        task_feats = create_task_features_from_optimal(problem, solution_obj.robot_schedules, decision_time, location_normalization, duration_normalization)

        # Compute expert reward and feasibility mask
        expert_reward, feasibility_mask = get_expert_reward(solution_obj.robot_schedules, decision_time, problem["T_t"], gamma=self.gamma, immediate_reward=self.immediate_reward)

        return robot_feats, task_feats, expert_reward, feasibility_mask


class PrecomputedDataset(Dataset):
    def __init__(self, precomputed_dir):
        self.precomputed_dir = precomputed_dir
        self.files = sorted(os.listdir(precomputed_dir))
        
        first_sample = torch.load(os.path.join(precomputed_dir, self.files[0]))
        
        self.n_robots = first_sample['robot_feats'].shape[0]
        self.robot_dim = first_sample['robot_feats'].shape[1]

        self.n_tasks = first_sample['task_feats'].shape[0]
        self.task_dim = first_sample['task_feats'].shape[1]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        sample = torch.load(os.path.join(self.precomputed_dir, self.files[idx]))
        return sample['robot_feats'], sample['task_feats'], sample['expert_reward'], sample['feasibility_mask']