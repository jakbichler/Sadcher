import argparse
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import SchedulingDataset
from training_helpers import load_dataset, find_decision_points
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
    argument_parser = argparse.ArgumentParser(description="Train the LVWS model on a dataset of problem instances.")
    argument_parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing problem instances.")
    argument_parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory to save model checkpoints.")

    args = argument_parser.parse_args()


    problem_dir = os.path.join(args.dataset_dir, "problem_instances/")
    solution_dir = os.path.join(args.dataset_dir, "solutions/")
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_epochs = 200
    batch_size = 512

    problems, solutions = load_dataset(problem_dir, solution_dir)
    dataset = SchedulingDataset(problems, solutions, gamma=0.99, immediate_reward=10)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Train-validation split (80% train, 20% validation)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Dataset split into {train_size} training samples and {val_size} validation samples.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Loaded {len(problems)} problems and {len(solutions)} solutions...............")
    
    robot_input_dim = len(problems[0]["Q"][0]) + 1 + 2   # e.g., capabilities + 'available' + xy_location2 xy_location
    task_input_dim = len(problems[0]["R"][0]) + 3 + 2     # e.g., skill requirements + (ready, assigned, incomplete) + xy_location
    
    model = TransformerScheduler(
        robot_input_dimensions=robot_input_dim,
        task_input_dimension=task_input_dim,
        embed_dim=128,
        ff_dim=256,
        num_heads=4,
        num_layers=4,
        dropout=0.0
    ).to(device)

    loss_fn = LVWS_Loss(weight_factor=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print("Starting training...............")

    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        # Training loop
        model.train()
        total_train_loss = 0.0
        for robot_features, task_features, expert_reward, feasibility_mask in tqdm(train_loader, unit="Batch", desc=f"Epoch {epoch+1}/{n_epochs} - Training"):
           


            predicted_reward_matrix = model(robot_features.to(device), task_features.to(device))

            loss = loss_fn(expert_reward.to(device), predicted_reward_matrix, feasibility_mask.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{n_epochs} - Avg Train Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for robot_features, task_features, expert_reward, feasibility_mask in tqdm(val_loader, unit="Batch", desc=f"Epoch {epoch+1}/{n_epochs} - Validation"):
                predicted_reward_matrix = model(robot_features.to(device), task_features.to(device))

                loss = loss_fn(expert_reward.to(device), predicted_reward_matrix, feasibility_mask.to(device))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{n_epochs} - Avg Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)

    # Plot losses
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()