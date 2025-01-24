from torch.utils.data import Dataset
from training_helpers import create_robot_features, create_task_features, get_expert_reward, find_decision_points

class SchedulingDataset(Dataset):
    def __init__(self, problems, solutions):
        """
        Builds a list of samples. Each sample is a tuple:
          (robot_feats, task_feats, expert_reward, feasibility_mask)
        for a particular problem-solution pair and decision_time.
        """
        self.samples = []
        for problem, solution_obj in zip(problems, solutions):
            decision_points = find_decision_points(solution_obj)
            for dec_time in decision_points:
                # 1) Robot & task features
                robot_feats = create_robot_features(problem, solution_obj.robot_schedules, dec_time)
                task_feats = create_task_features(problem, solution_obj.robot_schedules, dec_time)

                # 2) Expert reward & feasibility mask
                E, X = get_expert_reward(solution_obj.robot_schedules, dec_time)

                self.samples.append((robot_feats, task_feats, E, X))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
