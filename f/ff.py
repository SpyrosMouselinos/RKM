import os
import random

import numpy as np
import pytest
import torch
from app.training import train_model
import pandas as pd


# @pytest.fixture
def sample_data(tmpdir):
    # Create a temporary CSV file with sample data
    data = {
        'timestamp_column': pd.date_range(start='1/1/2020', periods=100, freq='200ms'),
        'feature1': range(100),
        'feature2': range(100, 200),
        'feature3': [random.choice([np.nan, np.random.randint(0, 100)]) for _ in range(95)] + [np.nan] * 5
    }
    df = pd.DataFrame(data)
    file_path = os.path.join(tmpdir, "sample.csv")
    df.to_csv(file_path, index=False)
    return file_path


def _train_model(sample_data):
    # Define a temporary directory for saving the model
    model_save_path = "test_model.pth"

    # Run the training function
    train_model(csv_file=sample_data,
                model_save_path=model_save_path,
                sequence_length=10,
                target_offset=1,
                batch_size=4,
                num_epochs=5,
                learning_rate=0.001,
                impute_backward=2,
                group_by='200ms',
                eval_every=1,
                early_stopping_patience=2,
                use_cuda=False)

    # Check if the model file was created
    assert os.path.exists(model_save_path)

    # Optionally, load the model and check its structure
    model = torch.load(model_save_path)
    assert isinstance(model, dict)  # Model should be saved as a state_dict

    # Cleanup: Remove the model file after test
    os.remove(model_save_path)


_train_model(sample_data=sample_data('.'))
