# Python script for training models
import glob
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError, MeanAbsolutePercentageError, MetricCollection
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd

from modeling.SimpleForcaster import SimpleForcaster
from utils.data_processing import TimeSeriesImputationDataset, find_and_convert_date_column


def train_model(csv_file,
                model_save_path,
                mode='many_to_one',
                sequence_length=10,
                target_offset=1,
                batch_size=32,
                num_epochs=50,
                learning_rate=0.001,
                impute_backward=10,
                group_by='24H',
                eval_every=1,
                early_stopping_patience=5,
                use_cuda=True):
    """
    Trains a model using the specified parameters, saves checkpoints, and applies early stopping.

    @param csv_file: Path to the CSV file containing the training data.
    @param model_save_path: Path where the trained model will be saved.
    @param mode: Mode of the model ('many_to_one' or 'many_to_many').
    @param sequence_length: Length of the input sequence.
    @param target_offset: Offset for target values in 'many_to_many' mode.
    @param batch_size: Batch size for training.
    @param num_epochs: Number of epochs for training.
    @param learning_rate: Learning rate for the optimizer.
    @param impute_backward: Number of past steps to impute in the dataset.
    @param group_by: Frequency for grouping time series data (e.g., '24H').
    @param eval_every: Frequency (in epochs) for evaluating the model.
    @param early_stopping_patience: Number of epochs with no improvement to trigger early stopping.
    @param use_cuda: Flag to indicate if GPU should be used if available.
    """
    # Load the data from the CSV
    data = find_and_convert_date_column(pd.read_csv(csv_file))

    # Extract the features and timestamps
    timestamps = data["date"].values
    features = data.drop("date", axis=1)

    # Sort the names of the columns alphabetically
    features = features.reindex(sorted(features.columns), axis=1)

    # Save the order of the columns in a list
    column_order = list(features.columns)

    # Convert features to numpy array
    features = features.values

    # Initialize the dataset
    dataset = TimeSeriesImputationDataset(features,
                                          timestamps,
                                          sequence_length,
                                          column_names=column_order,
                                          group_by=group_by,
                                          target_offset=target_offset,
                                          impute_backward=impute_backward)

    # Split data into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define the model
    input_size = train_dataset[0][0].shape[-1]
    output_size = train_dataset[0][1].shape[-1]

    model = SimpleForcaster(mode=mode,
                            input_size=input_size,
                            hidden_size=128,
                            output_size=output_size,
                            target_offset=target_offset)

    # Move model to GPU if available
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss, optimizer, and scheduler
    criterion = nn.MSELoss()

    # Define metrics using MetricCollection
    metrics = MetricCollection({
        'mse': MeanSquaredError().to(device),
        'mape': MeanAbsolutePercentageError().to(device)
    })

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler(device=device.type)

    # Initialize variables for checkpointing and early stopping
    best_loss = float('inf')
    best_metrics = None
    epochs_no_improve = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        metrics = metrics.to(device)  # Ensure metrics are on the correct device
        metrics.reset()  # Reset metrics at the beginning of the epoch

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            with autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            metrics.update(outputs, targets)  # Update metrics with the current batch

        # Scheduler step
        scheduler.step()

        # Average training loss
        avg_train_loss = running_loss / len(train_loader)

        # Compute metrics
        avg_train_metrics = metrics.compute()

        # Print the epoch and training loss
        print(
            f"Epoch [{epoch + 1}/{num_epochs}],"
            f" Training Loss: {avg_train_loss:.4f}")

        # Print each entry of the training metrics
        for k, v in avg_train_metrics.items():
            print(f" Training {k}: {v:.4f}")

        # Evaluate the model
        if (epoch + 1) % eval_every == 0:
            val_loss, val_metrics = evaluate_model(model, val_loader, criterion, metrics, device)

            # Print the epoch and validation loss
            print(
                f"Epoch [{epoch + 1}/{num_epochs}],"
                f" Validation Loss: {val_loss:.4f}")

            # Print each entry of the validation metrics
            for k, v in val_metrics.items():
                print(f" Validation {k}: {v:.4f}")

            # Checkpointing and early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                best_metrics = val_metrics
                epochs_no_improve = 0
                save_model_checkpoint(model, model_save_path, epoch + 1, best_loss, best_metrics)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print("Early stopping triggered!")
                    break

    save_final_model(best_loss, best_metrics, dataset.feature_stats, os.path.dirname(model_save_path))


def evaluate_model(model, val_loader, criterion, metrics, device):
    """
    Evaluates the model on the validation set.

    @param model: The model to be evaluated.
    @param val_loader: DataLoader for the validation dataset.
    @param criterion: Loss function used for evaluation.
    @param metrics: Metrics for evaluating model performance.
    @param device: Device on which the model and data are located (CPU or GPU).
    @return: Tuple containing average validation loss and metrics.
    """
    model.eval()
    val_loss = 0.0
    metrics = metrics.to(device)  # Ensure metrics are on the correct device
    metrics.reset()  # Reset metrics before evaluation

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            metrics.update(outputs, targets)  # Update metrics with the current batch

    avg_val_loss = val_loss / len(val_loader)
    avg_val_metrics = metrics.compute()  # Compute metrics after evaluation

    return avg_val_loss, avg_val_metrics


def save_model_checkpoint(model, model_save_path, epoch, loss, metrics):
    """
    Saves the model checkpoint and deletes previous checkpoints.

    @param model: The model to be saved.
    @param model_save_path: Path where the checkpoint will be saved.
    @param epoch: Current epoch number.
    @param loss: Loss value at the current epoch.
    @param metrics: Metrics at the current epoch.
    """
    # Define the path for the new checkpoint
    checkpoint_path = model_save_path.replace('.pth', f'_checkpoint_epoch_{epoch}.pth')

    # Define the pattern to search for previous checkpoints, excluding the current one
    checkpoint_pattern = model_save_path.replace('.pth', '_checkpoint_epoch_*.pth')

    # Remove the checkpoint files of previous epochs, if any, but keep the current one
    for f in glob.glob(checkpoint_pattern):
        if f != checkpoint_path:
            os.remove(f)
            print(f"Deleted previous checkpoint: {f}")

    # Save the checkpoint to disk
    model.save_to_checkpoint(epoch=epoch, loss=loss, metrics=metrics, model_save_path=checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")


def save_final_model(loss, metrics, feature_stats, save_dir="checkpoints"):
    """
    Saves the final model and metrics after training.

    @param loss: Best validation loss achieved during training.
    @param metrics: Best validation metrics achieved during training.
    @param feature_stats: Statistics of features used in training.
    @param save_dir: Directory where the final model and metrics will be saved.
    """
    # Locate the checkpoint model
    checkpoint_model = None
    for filename in os.listdir(save_dir):
        if "checkpoint" in filename and filename.endswith(".pth"):
            checkpoint_model = filename
            break

    if not checkpoint_model:
        raise ValueError("No checkpoint model found.")

    # Determine the next index for the final model
    indices = [int(f.split('_')[1].split('.')[0]) for f in os.listdir(save_dir) if
               f.endswith('.pth') and not "checkpoint" in f]
    next_index = max(indices) + 1 if indices else 1

    # Rename the model
    new_model_name = f"model_{next_index}.pth"
    os.rename(os.path.join(save_dir, checkpoint_model), os.path.join(save_dir, new_model_name))

    # Save the metrics to a JSON file
    json_data = {k: v.cpu().item() for k, v in metrics.items()}
    json_data.update(feature_stats)
    json_data.update({"loss": loss})
    json_file_path = os.path.join(save_dir, f"model_{next_index}.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    print(f"Final model saved as {new_model_name} and metrics saved as {json_file_path}")
