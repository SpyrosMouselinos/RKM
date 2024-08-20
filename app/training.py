# Python script for training models
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd

from app.modeling.SimpleForcaster import SimpleForcaster
from .utils.data_processing import TimeSeriesImputationDataset


def train_model(csv_file,
                model_save_path,
                sequence_length=10,
                target_offset=1,
                batch_size=32,
                num_epochs=50,
                learning_rate=0.001,
                impute_backward=10,
                group_by='200ms',
                eval_every=1,
                early_stopping_patience=5,
                use_cuda=True):

    # Load the data from the CSV
    data = pd.read_csv(csv_file)

    # Extract the features and timestamps
    timestamps = data["timestamp_column"].values
    features = data.drop("timestamp_column", axis=1)

    # Write a function that sorts the names of the columns alphabetically
    data = data.reindex(sorted(data.columns), axis=1)

    # Save the order of the columns in a list
    column_order = list(data.columns)

    # Convert it to numpy
    features = features.values

    # Initialize the dataset
    dataset = TimeSeriesImputationDataset(features,
                                          timestamps,
                                          sequence_length,
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
    input_size = train_dataset[0][0].shape[1]
    output_size = input_size  # Assuming the output has the same dimensions as the input
    model = SimpleForcaster(input_size=input_size, hidden_size=256, output_size=output_size)

    # Move model to GPU if available
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss, optimizer, and scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # For mixed precision training
    scaler = GradScaler()

    # Initialize variables for checkpointing and early stopping
    best_loss = float('inf')
    epochs_no_improve = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        # Scheduler step
        scheduler.step()

        # Average training loss
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        # Evaluate the model
        if (epoch + 1) % eval_every == 0:
            val_loss = evaluate_model(model, val_loader, criterion, device)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

            # Checkpointing and early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                save_model_checkpoint(model, optimizer, scheduler, model_save_path, epoch + 1, best_loss)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print("Early stopping triggered!")
                    break

    # Save final model and metrics
    save_final_model(model, model_save_path, dataset)


def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    return val_loss / len(val_loader)


def save_model_checkpoint(model, optimizer, scheduler, model_save_path, epoch, loss):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    checkpoint_path = model_save_path.replace('.pth', f'_checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")


def save_final_model(model, model_save_path, dataset):
    # Save the final model state
    torch.save(model.state_dict(), model_save_path)

    # Save the metrics and normalization statistics
    metrics = {
        "loss": best_loss,  # Best loss from the training loop
    }
    for i, (mean, std) in enumerate(zip(dataset.mean_std['mean'], dataset.mean_std['std'])):
        metrics[f"mean_feat_{i + 1}"] = mean
        metrics[f"std_feat_{i + 1}"] = std

    metrics_path = model_save_path.replace('.pth', '.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

    print(f"Model and metrics saved at {model_save_path} and {metrics_path}")
