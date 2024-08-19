import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.impute import SimpleImputer


class TimeSeriesImputationDataset(Dataset):
    def __init__(self, data, timestamps, sequence_length, group_by='seconds',
                 target_offset=1, impute_backward=10):
        self.sequence_length = sequence_length
        self.group_by = group_by
        self.target_offset = target_offset
        self.impute_backward = impute_backward

        self.data = data
        self.timestamps = pd.to_datetime(timestamps)

        # Normalize and group the data
        self.mean_std = self.calculate_mean_std()
        self.data = self.normalize_data(self.data)
        self.data = self.group_and_impute()

        # Generate valid sequences
        self.valid_sequences = self.generate_valid_sequences()

    def calculate_mean_std(self):
        means = np.nanmean(self.data, axis=0)
        stds = np.nanstd(self.data, axis=0)
        return {'mean': means, 'std': stds}

    def normalize_data(self, data):
        return (data - self.mean_std['mean']) / self.mean_std['std']

    def group_and_impute(self):
        df = pd.DataFrame(self.data, index=pd.to_datetime(self.timestamps))
        df = df.resample(self.group_by).mean()

        imputer = SimpleImputer(strategy='mean')
        data_imputed = df.copy()

        for col in df.columns:
            for i in range(self.impute_backward):
                data_imputed[col] = data_imputed[col].fillna(method='ffill', limit=self.impute_backward)

        data_imputed = imputer.fit_transform(data_imputed)

        valid_rows = ~np.isnan(data_imputed).any(axis=1)
        return data_imputed[valid_rows]

    def generate_valid_sequences(self):
        sequences = []
        for i in range(len(self.data) - self.sequence_length - self.target_offset + 1):
            x = self.data[i:i + self.sequence_length]
            y = self.data[i + self.sequence_length + self.target_offset - 1]
            if not np.isnan(x).any() and not np.isnan(y).any():
                sequences.append((x, y))
        return sequences

    def __len__(self):
        return len(self.valid_sequences)

    def __getitem__(self, idx):
        return (torch.tensor(self.valid_sequences[idx][0], dtype=torch.float32),
                torch.tensor(self.valid_sequences[idx][1], dtype=torch.float32))


class RealTimeTimeSeriesDataset:
    def __init__(self, sequence_length, mean_std=None):
        self.sequence_length = sequence_length
        self.buffer = []
        self.mean_std = mean_std

    def update_buffer(self, new_data_point):
        self.buffer.append(new_data_point)
        if len(self.buffer) > self.sequence_length:
            self.buffer.pop(0)

    def normalize_data(self, data):
        if self.mean_std:
            return (data - self.mean_std['mean']) / self.mean_std['std']
        else:
            return data

    def get_current_sequence(self):
        if len(self.buffer) == self.sequence_length:
            return torch.tensor(self.normalize_data(np.array(self.buffer)), dtype=torch.float32)
        else:
            return None

    def impute_missing(self, data_point):
        if len(self.buffer) > 0:
            last_valid = self.buffer[-1]
            for i in range(len(data_point)):
                if np.isnan(data_point[i]):
                    data_point[i] = last_valid[i]
        return data_point


