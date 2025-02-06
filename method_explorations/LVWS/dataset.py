import numpy as np
from torch.utils.data import Dataset
from training_helpers import create_robot_features_from_optimal, create_task_features_from_optimal, get_expert_reward, find_decision_points

class SchedulingDataset(Dataset):
    def __init__(self, problems, solutions, gamma=0.99, immediate_reward=10):
        """
        Builds a list of samples. Each sample is a tuple:
          (robot_feats, task_feats, expert_reward, feasibility_mask)
        for a particular problem-solution pair and decision_time.
        """
        self.samples = []
        for problem, solution_obj in zip(problems, solutions):
            location_normalization = np.max(problem["task_locations"])
            duration_normalization = np.max(problem["T_e"])
            decision_points = find_decision_points(solution_obj)
            for dec_time in decision_points:
                # 1) Robot & task features
                robot_feats = create_robot_features_from_optimal(problem, solution_obj.robot_schedules, dec_time, location_normalization, duration_normalization)
                task_feats = create_task_features_from_optimal(problem, solution_obj.robot_schedules, dec_time, location_normalization, duration_normalization)

                # 2) Expert reward & feasibility mask
                E, X = get_expert_reward(solution_obj.robot_schedules, dec_time, gamma=gamma, immediate_reward= immediate_reward)

                self.samples.append((robot_feats, task_feats, E, X))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
