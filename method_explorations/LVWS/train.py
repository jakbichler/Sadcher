import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SchedulingDataset
from training_helpers import load_dataset, create_robot_features, create_task_features, get_expert_reward, find_decision_points
from transformer_models import TransformerScheduler

class LVWS_Loss(nn.Module):
    def __init__(self, weight_factor):
        super(LVWS_Loss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.weight_factor = weight_factor

        
    def forward (self, expert_reward, predicted_reward, feasibility_mask):
        loss_feasible = self.l1_loss(predicted_reward*feasibility_mask, expert_reward * feasibility_mask)
        loss_not_in_expert = self.l1_loss(predicted_reward*(1-feasibility_mask), expert_reward*(1-feasibility_mask))
        return loss_feasible + self.weight_factor * loss_not_in_expert 



if __name__ == "__main__":

    problem_dir = "/home/jakob/thesis/datasets/simple_dataset_1000/problem_instances"
    solution_dir = "/home/jakob/thesis/datasets/simple_dataset_1000/solutions"
    checkpoint_dir = "/home/jakob/thesis/method_explorations/LVWS/checkpoints"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_epochs = 10
    batch_size = 512

    problems, solutions = load_dataset(problem_dir, solution_dir)
    dataset = SchedulingDataset(problems, solutions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Loaded {len(problems)} problems and {len(solutions)} solutions...............")
    
    robot_input_dim = len(problems[0]["Q"][0]) + 1    # e.g., capabilities + 'available'
    task_input_dim = len(problems[0]["R"][0]) + 3     # e.g., skill requirements + (ready, assigned, incomplete)
    
    model = TransformerScheduler(
        robot_input_dimensions=robot_input_dim,
        task_input_dimension=task_input_dim,
        embed_dim=64,
        ff_dim=128,
        num_layers=2,
        dropout=0.1
    ).to(device)

    loss_fn = LVWS_Loss(weight_factor=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("Starting training...............")

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for robot_features, task_features, expert_reward, feasibility_mask in tqdm(dataloader, unit="Batch", desc=f"Epoch {epoch+1}/{n_epochs}"):

            predicted_reward_matrix = model(robot_features.to(device), task_features.to(device))

            loss = loss_fn(expert_reward.to(device), predicted_reward_matrix, feasibility_mask.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, sum(len(find_decision_points(s)) for s in solutions))
        print(f"Epoch {epoch+1}/{n_epochs} - Avg Loss: {avg_loss:.4f}")
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)




