import argparse
from datetime import datetime   
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from icecream import ic 
from dataset import LazyLoadedSchedulingDataset, PrecomputedDataset
from training_helpers import load_dataset, find_decision_points
from transformer_models import SchedulerNetwork

class LVWS_Loss(nn.Module):
    def __init__(self, weight_factor):
        super(LVWS_Loss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.weight_factor = weight_factor
        
    def forward (self, expert_reward, predicted_reward, feasibility_mask):
        loss_feasible = self.l1_loss(predicted_reward*feasibility_mask, expert_reward * feasibility_mask)
        loss_not_in_expert = self.l1_loss(predicted_reward*(1-feasibility_mask), expert_reward*(1-feasibility_mask))
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
    argument_parser.add_argument("--continue_training", action="store_true", default= False, help="Continue training from a checkpoint.")
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
    #problems, solutions = load_dataset(problem_dir, solution_dir)
    #dataset = LazyLoadedSchedulingDataset(problem_dir, solution_dir, gamma=config["reward_gamma"], immediate_reward=10)
    dataset = PrecomputedDataset(args.dataset_dir)
    # Train-validation split (80% train, 20% validation)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

     # Log config
    with open(os.path.join(out_checkpoint_dir, "run_description.txt"), "w") as f:
        f.write(f"Training Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Config: {config}\n")   
        f.write(f"Dataset: {args.dataset_dir}\n")
#        f.write(f"N_problems: {len(problems)}\n")
        f.write(f"N_samples: {dataset_size}\n")

    #print(f"Loaded {len(problems)} problems and {len(solutions)} solutions...............")
    print(f"Dataset of {dataset_size} decision points split into {train_size} training samples and {val_size} validation samples.")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size= config["batch_size"], shuffle=False)

    #robot_input_dim = len(problems[0]["Q"][0]) + 4   # e.g., capabilities + xy_location (2) + remaining_workload (1) + 'available' (1)
    #task_input_dim = len(problems[0]["R"][0]) + 6     # e.g., skill requirements + xy_location (2), duration (1), + (ready, assigned, incomplete) (3)
    robot_input_dim = 2+4
    task_input_dim = 2+6
    
    model = SchedulerNetwork(
        robot_input_dimensions=robot_input_dim,
        task_input_dimension=task_input_dim,
        embed_dim=config["embedding_dim"],
        ff_dim=config["ff_dim"],
        n_transformer_heads=config["n_transformer_heads"],
        n_transformer_layers=config["n_transformer_layers"],
        n_gatn_heads=config["n_gatn_heads"],
        n_gatn_layers=config["n_gatn_layers"],
    ).to(device)

    if args.continue_training:
        model.load_state_dict(torch.load(args.in_checkpoint_path, weights_only=True))
    else:
        initialize_weights(model)

    loss_fn = LVWS_Loss(weight_factor=config["loss_weight_factor"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    print("Starting training...............")

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    epoch, epochs_without_improvement = 0, 0 

    while epochs_without_improvement < config["early_stopping_patience"]:
        epoch += 1
        
        # Training loop
        model.train()
        total_train_loss = 0.0
        for robot_features, task_features, expert_reward, feasibility_mask in tqdm(train_loader, unit="Batch", desc=f"Epoch {epoch} - Training"):
            predicted_reward_matrix = model(robot_features.to(device), task_features.to(device))
            loss = loss_fn(expert_reward.to(device), predicted_reward_matrix, feasibility_mask.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch} - Avg Train Loss: {avg_train_loss:.5f}")

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for robot_features, task_features, expert_reward, feasibility_mask in tqdm(val_loader, unit="Batch", desc=f"Epoch {epoch} - Validation"):
                predicted_reward_matrix = model(robot_features.to(device), task_features.to(device))
                loss = loss_fn(expert_reward.to(device), predicted_reward_matrix, feasibility_mask.to(device))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch} - Avg Val Loss: {avg_val_loss:.5f}")

        # Log losses
        with open(os.path.join(out_checkpoint_dir, "losses.txt"), "a") as f:
            f.write(f"Epoch - {epoch}, Train loss {avg_train_loss}, Val loss - {avg_val_loss}\n")

        # Check early stopping condition (has to be at least 0.0001 better)
        if avg_val_loss < (best_val_loss - 1e-4):
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0  
            best_model_path = os.path.join(out_checkpoint_dir, "best_checkpoint.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at epoch {epoch}")
        else:
            epochs_without_improvement += 1

    # Plot losses
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()