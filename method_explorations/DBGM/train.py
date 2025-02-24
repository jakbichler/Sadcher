import argparse
from datetime import datetime   
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from icecream import ic 
from dataset import LazyLoadedSchedulingDataset
from training_helpers import load_dataset, find_decision_points
from transformer_models import SchedulerNetwork

class LVWS_Loss(nn.Module):
    def __init__(self, weight_factor):
        super(LVWS_Loss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.weight_factor = weight_factor
        
    def forward(self, expert_reward, predicted_reward, feasibility_mask):
        # Apply loss only on the real task columns.
        loss_feasible = self.l1_loss(predicted_reward * feasibility_mask,
                                     expert_reward * feasibility_mask)
        loss_not_in_expert = self.l1_loss(predicted_reward * (1 - feasibility_mask),
                                          expert_reward * (1 - feasibility_mask))
        return loss_feasible + self.weight_factor * loss_not_in_expert 

def initialize_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) > 1:  # Ensure it's a linear layer
                nn.init.kaiming_uniform_(param, nonlinearity='relu')
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Train the LVWS model on a dataset of problem instances.")
    argument_parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing problem instances.")
    argument_parser.add_argument("--out_checkpoint_dir", type=str, required=True, help="Directory to save model checkpoints.")
    argument_parser.add_argument("--continue_training", action="store_true", default=False, help="Continue training from a checkpoint.")
    argument_parser.add_argument("--in_checkpoint_path", type=str, help="Path to a checkpoint to continue training from.")

    args = argument_parser.parse_args()
    problem_dir = os.path.join(args.dataset_dir, "problem_instances/")
    solution_dir = os.path.join(args.dataset_dir, "solutions/")
    out_checkpoint_dir = args.out_checkpoint_dir
    os.makedirs(out_checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = {
        "batch_size": 512,
        "embedding_dim": 128,
        "ff_dim": 256,
        "n_transformer_heads": 4,
        "n_transformer_layers": 4,
        "n_gatn_heads": 4,
        "n_gatn_layers": 2,
        "dropout": 0.0,
        "loss_weight_factor": 0.1,
        "learning_rate": 1e-3,
        "reward_gamma": 0.99,
        "early_stopping_patience": 3,
    }

    ic("loading dataset")
    dataset = LazyLoadedSchedulingDataset(problem_dir, solution_dir, gamma=config["reward_gamma"])
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Log config
    with open(os.path.join(out_checkpoint_dir, "run_description.txt"), "w") as f:
        f.write(f"Training Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Config: {config}\n")   
        f.write(f"Dataset: {args.dataset_dir}\n")
        f.write(f"N_samples: {dataset_size}\n")

    print(f"Dataset split into {train_size} training samples and {val_size} validation samples.")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    model = SchedulerNetwork(
        robot_input_dimensions=dataset.robot_dim,
        task_input_dimension=dataset.task_dim,
        embed_dim=config["embedding_dim"],
        ff_dim=config["ff_dim"],
        n_transformer_heads=config["n_transformer_heads"],
        n_transformer_layers=config["n_transformer_layers"],
        n_gatn_heads=config["n_gatn_heads"],
        n_gatn_layers=config["n_gatn_layers"],
    ).to(device)

    if args.continue_training:
        model.load_state_dict(torch.load(args.in_checkpoint_path, map_location=device))
    else:
        initialize_weights(model)

    # Create loss functions: LVWS loss for real tasks and BCE loss for idle predictions.
    lvws_loss_fn = LVWS_Loss(weight_factor=config["loss_weight_factor"])
    bce_loss_fn = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    print("Starting training...............")

    # To track losses separately.
    train_lvws_losses, train_bce_losses = [], []
    val_lvws_losses, val_bce_losses = [], []
    overall_train_losses, overall_val_losses = [], []
    
    best_val_loss = float("inf")
    epoch, epochs_without_improvement = 0, 0 

    while epochs_without_improvement < config["early_stopping_patience"]:
        epoch += 1
        
        model.train()
        total_train_lvws_loss = 0.0
        total_train_bce_loss = 0.0
        
        for robot_features, task_features, expert_reward, feasibility_mask in tqdm(train_loader, unit="Batch", desc=f"Epoch {epoch} - Training"):
            # Move data to device.
            robot_features = robot_features.to(device)
            task_features = task_features.to(device)
            expert_reward = expert_reward.to(device)  # Expect shape: (B, N, M+1)
            feasibility_mask = feasibility_mask.to(device)  # Expect shape: (B, N, M+1), but use only real-task cols.
            
            predicted_reward_matrix = model(robot_features, task_features)  # (B, N, M+1)
            # Split predictions: first M columns for tasks, last column for idle.
            predicted_task_rewards = predicted_reward_matrix[:, :, :-1]
            predicted_idle = predicted_reward_matrix[:, :, -1]  # (B, N)

            # Similarly, split expert targets.
            expert_task_rewards = expert_reward[:, :, :-1]
            expert_idle = expert_reward[:, :, -1]  # (B, N)
            # For LVWS loss, use feasibility mask for real tasks.
            feasibility_mask_tasks = feasibility_mask[:, :, :-1]

            lvws_loss = lvws_loss_fn(expert_task_rewards, predicted_task_rewards, feasibility_mask_tasks)
            bce_loss = bce_loss_fn(predicted_idle.float(), expert_idle.float())
            total_loss = lvws_loss + bce_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_train_lvws_loss += lvws_loss.item()
            total_train_bce_loss += bce_loss.item()

        avg_train_lvws = total_train_lvws_loss / len(train_loader)
        avg_train_bce = total_train_bce_loss / len(train_loader)
        overall_train = avg_train_lvws + avg_train_bce
        train_lvws_losses.append(avg_train_lvws)
        train_bce_losses.append(avg_train_bce)
        overall_train_losses.append(overall_train)
        print(f"Epoch {epoch} - Avg Train LVWS Loss: {avg_train_lvws:.5f}, BCE Loss: {avg_train_bce:.5f}, Overall: {overall_train:.5f}")

        # Validation loop.
        model.eval()
        total_val_lvws_loss = 0.0
        total_val_bce_loss = 0.0
        with torch.no_grad():
            for robot_features, task_features, expert_reward, feasibility_mask in tqdm(val_loader, unit="Batch", desc=f"Epoch {epoch} - Validation"):
                robot_features = robot_features.to(device)
                task_features = task_features.to(device)
                expert_reward = expert_reward.to(device)
                feasibility_mask = feasibility_mask.to(device)

                predicted_reward_matrix = model(robot_features, task_features)
                predicted_task_rewards = predicted_reward_matrix[:, :, :-1]
                predicted_idle = predicted_reward_matrix[:, :, -1]
                expert_task_rewards = expert_reward[:, :, :-1]
                expert_idle = expert_reward[:, :, -1]
                feasibility_mask_tasks = feasibility_mask[:, :, :-1]

                lvws_loss = lvws_loss_fn(expert_task_rewards, predicted_task_rewards, feasibility_mask_tasks)
                bce_loss = bce_loss_fn(predicted_idle.float(), expert_idle.float())
                total_val_lvws_loss += lvws_loss.item()
                total_val_bce_loss += bce_loss.item()

        avg_val_lvws = total_val_lvws_loss / len(val_loader)
        avg_val_bce = total_val_bce_loss / len(val_loader)
        overall_val = avg_val_lvws + avg_val_bce
        val_lvws_losses.append(avg_val_lvws)
        val_bce_losses.append(avg_val_bce)
        overall_val_losses.append(overall_val)
        print(f"Epoch {epoch} - Avg Val LVWS Loss: {avg_val_lvws:.5f}, BCE Loss: {avg_val_bce:.5f}, Overall: {overall_val:.5f}")

        # Log losses separately.
        with open(os.path.join(out_checkpoint_dir, "losses.txt"), "a") as f:
            f.write(f"Epoch {epoch} - Train LVWS: {avg_train_lvws:.5f}, Train BCE: {avg_train_bce:.5f}, Overall Train: {overall_train:.5f} | " +
                    f"Val LVWS: {avg_val_lvws:.5f}, Val BCE: {avg_val_bce:.5f}, Overall Val: {overall_val:.5f}\n")

        # Early stopping: save best model if overall validation loss improves.
        if overall_val < (best_val_loss - 1e-4):
            best_val_loss = overall_val
            epochs_without_improvement = 0  
            best_model_path = os.path.join(out_checkpoint_dir, "best_checkpoint.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at epoch {epoch}")
        else:
            epochs_without_improvement += 1

    # Plot training and validation curves for each loss component.
    plt.figure(figsize=(10, 6))
    plt.plot(overall_train_losses, label="Overall Train Loss")
    plt.plot(overall_val_losses, label="Overall Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Overall Loss")
    plt.savefig(os.path.join(out_checkpoint_dir, "overall_loss.png"))
    plt.show()



    plt.figure(figsize=(10, 6))
    plt.plot(train_lvws_losses, label="Train LVWS Loss")
    plt.plot(val_lvws_losses, label="Val LVWS Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("LVWS Loss")
    plt.savefig(os.path.join(out_checkpoint_dir, "lvws_loss.png"))
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(train_bce_losses, label="Train BCE Loss")
    plt.plot(val_bce_losses, label="Val BCE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Idle (BCE) Loss")
    plt.savefig(os.path.join(out_checkpoint_dir, "bce_loss.png"))
    plt.show()

    # Once in a single plot
    plt.figure(figsize=(10, 6))
    plt.plot(overall_train_losses, label="Overall Train Loss")
    plt.plot(overall_val_losses, label="Overall Val Loss")
    plt.plot(train_lvws_losses, label="Train LVWS Loss")
    plt.plot(val_lvws_losses, label="Val LVWS Loss")
    plt.plot(train_bce_losses, label="Train BCE Loss")
    plt.plot(val_bce_losses, label="Val BCE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("All Losses")
    plt.savefig(os.path.join(out_checkpoint_dir, "all_losses.png"))
    plt.show()