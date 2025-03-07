import argparse
from functools import partial

import optuna
import torch
from attention_models import SchedulerNetwork
from dataset import LazyLoadedSchedulingDataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from train import LVWS_Loss, initialize_weights


# -----------------------------
# Optuna Objective Function
# -----------------------------
def objective(trial, args, dataset):
    # Sample hyperparameters

    embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
    ff_dim = 2 * embedding_dim
    config = {
        "batch_size": 512,
        "embedding_dim": embedding_dim,
        "ff_dim": ff_dim,
        "n_transformer_heads": trial.suggest_categorical("n_transformer_heads", [2, 4, 8]),
        "n_transformer_layers": trial.suggest_int("n_transformer_layers", 1, 4),
        "n_gatn_heads": trial.suggest_categorical("n_gatn_heads", [2, 4, 8]),
        "n_gatn_layers": trial.suggest_int("n_gatn_layers", 1, 4),
        "loss_weight_factor": trial.suggest_categorical("loss_weight_factor", [0.1, 0.2]),
        "learning_rate": 1e-3,
        "early_stopping_patience": 3,
        "early_stopping_threshold": 1e-3,  # Aggressive early stopping threshold
        "dropout": 0.0,
    }
    print(f"Trial {trial.number} config: {config}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    robot_input_dim = dataset.robot_dim
    task_input_dim = dataset.task_dim
    total_samples = len(dataset)
    if args.subset_fraction < 1.0:
        subset_size = int(total_samples * args.subset_fraction)
        dataset, _ = random_split(dataset, [subset_size, total_samples - subset_size])
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Initialize model, loss, and optimizer
    model = SchedulerNetwork(
        robot_input_dimensions=robot_input_dim,
        task_input_dimension=task_input_dim,
        embed_dim=config["embedding_dim"],
        ff_dim=config["ff_dim"],
        n_transformer_heads=config["n_transformer_heads"],
        n_transformer_layers=config["n_transformer_layers"],
        n_gatn_heads=config["n_gatn_heads"],
        n_gatn_layers=config["n_gatn_layers"],
        dropout=config["dropout"],
    ).to(device)
    initialize_weights(model)
    loss_fn = LVWS_Loss(weight_factor=config["loss_weight_factor"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    epoch = 0

    # Training loop with early stopping and pruning per trial
    while (
        epochs_without_improvement < config["early_stopping_patience"] and epoch < args.max_epochs
    ):
        epoch += 1
        model.train()
        total_train_loss = 0.0
        for batch in tqdm(
            train_loader, desc=f"Trial {trial.number} Epoch {epoch} - Training", leave=False
        ):
            robot_features, task_features, expert_reward, feasibility_mask, task_adjacency = batch
            predicted_reward = model(
                robot_features.to(device), task_features.to(device), task_adjacency.to(device)
            )
            loss = loss_fn(expert_reward.to(device), predicted_reward, feasibility_mask.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Trial {trial.number} Epoch {epoch} - Validation", leave=False
            ):
                robot_features, task_features, expert_reward, feasibility_mask, task_adjacency = (
                    batch
                )
                predicted_reward = model(
                    robot_features.to(device), task_features.to(device), task_adjacency.to(device)
                )
                loss = loss_fn(
                    expert_reward.to(device), predicted_reward, feasibility_mask.to(device)
                )
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch}.")
            raise optuna.TrialPruned()

        if avg_val_loss < best_val_loss - config["early_stopping_threshold"]:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Trial {trial.number} Epoch {epoch}: Train Loss = {avg_train_loss:.5f}, Val Loss = {avg_val_loss:.5f}"
        )

    return best_val_loss


# -----------------------------
# Main: Parse Arguments and Run Optuna Study
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for LVWS model using Optuna"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing 'problem_instances' and 'solutions' directories",
    )
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna trials to run")
    parser.add_argument("--max_epochs", type=int, default=20, help="Maximum epochs per trial")
    parser.add_argument(
        "--subset_fraction",
        type=float,
        default=0.1,
        help="Fraction of dataset to use for tuning (e.g., 0.1 for 10%)",
    )
    args = parser.parse_args()

    # Load dataset (using a subset for faster runs if desired)
    problem_dir = f"{args.dataset_dir}/problem_instances"
    solution_dir = f"{args.dataset_dir}/solutions"
    dataset = LazyLoadedSchedulingDataset(
        problem_dir, solution_dir, gamma=0.99, immediate_reward=10
    )
    # Create an in-memory study (no external storage)
    study = optuna.create_study(
        direction="minimize",
        study_name="sadcher_hyperparam_tuning_schiermonnikoog",
        storage="sqlite:///optuna_study_schiermonnikoog.db",
        load_if_exists=True,
    )
    objective_func = partial(objective, args=args, dataset=dataset)
    study.optimize(objective_func, n_trials=args.n_trials)

    print("Best trial:")
    print(f"  Trial number: {study.best_trial.number}")
    print(f"  Validation Loss: {study.best_trial.value}")
    print("  Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
