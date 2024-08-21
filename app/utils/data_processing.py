import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class TimeSeriesImputationDataset(Dataset):
    def __init__(self, data, timestamps, sequence_length, column_names=None, group_by='seconds',
                 target_offset=1, impute_backward=10):
        self.column_names = column_names
        self.sequence_length = sequence_length
        self.group_by = group_by
        self.target_offset = target_offset
        self.impute_backward = impute_backward

        self.data = data
        self.timestamps = pd.to_datetime(timestamps)

        # Normalize and group the data
        self.mean_std = self.calculate_mean_std()
        self.data = self.normalize_data(self.data)
        self.data, self.invalid_timestamps = self.group_and_impute()

        # Determine the valid sequence start indices
        self.valid_indices = self.find_valid_sequence_starts()

    def calculate_mean_std(self):
        means = np.nanmean(self.data, axis=0)
        stds = np.nanstd(self.data, axis=0)
        return {'mean': means, 'std': stds}

    def normalize_data(self, data):
        return (data - self.mean_std['mean']) / self.mean_std['std']

    def group_and_impute(self):
        df = pd.DataFrame(self.data, index=self.timestamps)
        df = df.resample(self.group_by).mean()
        for col in df.columns:
            df[col] = df[col].fillna(method='ffill', limit=self.impute_backward)
        # Identify rows with NaN values after imputation
        valid_rows = ~np.isnan(df).any(axis=1)
        invalid_timestamps = df.index[~valid_rows]  # Timestamps with NaN values
        return df[valid_rows], set(invalid_timestamps)

    def find_valid_sequence_starts(self):
        valid_indices = []
        for i in range(len(self.data) - self.sequence_length - self.target_offset + 1):
            # Get the start and end timestamps for the current sequence
            start_timestamp = self.timestamps[i]
            end_timestamp = self.timestamps[i + self.sequence_length + self.target_offset - 1]

            # Check if any of the timestamps in the sequence or target are invalid
            if any(ts in self.invalid_timestamps for ts in pd.date_range(start=start_timestamp, end=end_timestamp)):
                continue

            valid_indices.append(i)
        return valid_indices

    @property
    def feature_stats(self):
        fs = {}
        if self.column_names is None:
            return self.mean_std
        else:
            assert len(self.column_names) == len(self.mean_std['mean']) == len(self.mean_std['std'])
            for i, column_name in enumerate(self.column_names):
                fs[column_name + '_mean'] = self.mean_std['mean'][i]
                fs[column_name + '_std'] = self.mean_std['std'][i]
        return fs

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length

        x = self.data.iloc[start_idx:end_idx].values
        y = self.data.iloc[start_idx + self.sequence_length + self.target_offset - 1].values

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return x_tensor, y_tensor


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


def find_and_convert_date_column(df):
    # List of common date-related keywords
    date_keywords = ['date', 'year', 'month', 'day', 'time']

    # Dictionary to store the likelihood of each column being a date column
    date_column_likelihood = {}

    # Iterate through each column
    for column in df.columns:
        # Check for date-related keywords in the column name
        if any(keyword in column.lower() for keyword in date_keywords):
            date_column_likelihood[column] = 'Keyword match'

    # Find the first column that was identified as a date column
    for col in date_column_likelihood:
        if date_column_likelihood[col] in ['Keyword match', 'Convertible to datetime']:
            df.rename(columns={col: 'date'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])  # Ensure it is converted to datetime
            break  # We rename only the first identified date column

    return df
