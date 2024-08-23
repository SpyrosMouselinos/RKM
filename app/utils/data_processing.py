from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd


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
        y = self.data.iloc[end_idx: end_idx + self.target_offset].values

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return x_tensor, y_tensor


class RealTimeTimeSeriesDataset:
    def __init__(self, feature_names, feature_stats, sequence_length=10, timegroup_factor='24H'):
        self.buffer = []
        self.sequence_length = sequence_length
        self.timegroup_factor = timegroup_factor
        self.feature_names = feature_names
        self.feature_stats = feature_stats
        self.date_feature = "date"  # Assumed name of the date feature

    def update_buffer(self, new_data_point):
        """
        Update the buffer with new data points, group them by the timegroup_factor,
        and safely remove old data.
        """
        # Scale the new data point
        scaled_data_point = self.normalize_data(new_data_point)

        # Handle corrupted data by averaging forward and backward fill
        self.correct_corrupted_data(scaled_data_point)

        # Append the new data point to the buffer
        self.buffer.append(scaled_data_point)

        # Group data by the timegroup_factor
        grouped_data = self.group_data_by_time()

        # Update the buffer with the grouped data
        self.buffer = grouped_data[-self.sequence_length:]  # Keep only the last 'sequence_length' groups

        # Safely remove old data
        self.remove_old_data()

    def remove_old_data(self):
        """
        Remove data points from the buffer that are older than the required range.
        """
        if len(self.buffer) > 0:
            # Convert the buffer to a DataFrame for easier manipulation
            df = pd.DataFrame(self.buffer)
            df[self.date_feature] = pd.to_datetime(df[self.date_feature])

            # Convert the timegroup_factor to a timedelta
            time_delta = pd.to_timedelta(self.timegroup_factor)

            # Calculate the minimum allowed timestamp
            max_time = df[self.date_feature].max()
            min_allowed_time = max_time - (self.sequence_length * time_delta)

            # Filter the buffer to keep only relevant data
            self.buffer = df[df[self.date_feature] >= min_allowed_time].to_dict('records')

    def correct_corrupted_data(self, data):
        """
        Correct corrupted data by averaging the forward fill and backward fill.
        """
        for feature in self.feature_names:
            if pd.isna(data[feature]) or np.isinf(data[feature]):
                # Perform forward fill
                forward_value = self.forward_fill(feature)
                # Perform backward fill
                backward_value = self.backward_fill(feature)

                # Take the average of forward and backward fill
                data[feature] = (forward_value + backward_value) / 2

    def forward_fill(self, feature):
        """
        Perform forward fill for the given feature.
        """
        for i in reversed(range(len(self.buffer))):
            if not pd.isna(self.buffer[i][feature]) and not np.isinf(self.buffer[i][feature]):
                return self.buffer[i][feature]
        # If no valid value is found, return 0 as a default
        return 0.0

    def backward_fill(self, feature):
        """
        Perform backward fill for the given feature.
        """
        for i in range(len(self.buffer)):
            if not pd.isna(self.buffer[i][feature]) and not np.isinf(self.buffer[i][feature]):
                return self.buffer[i][feature]
        # If no valid value is found, return 0 as a default
        return 0.0

    def group_data_by_time(self):
        """
        Group the buffer data by the timegroup_factor.
        """
        df = pd.DataFrame(self.buffer)
        df[self.date_feature] = pd.to_datetime(df[self.date_feature])
        df.set_index(self.date_feature, inplace=True)

        # Group by the timegroup factor using mean
        grouped = df.resample(self.timegroup_factor).mean().dropna()

        return grouped.reset_index().to_dict('records')

    def normalize_data(self, data):
        """
        Normalize the data using the mean and std from the config.
        """
        for feature in self.feature_names:
            mean = self.feature_stats.get(f"{feature}_mean", torch.tensor(0.0))
            std = self.feature_stats.get(f"{feature}_std", torch.tensor(1.0))
            data[feature] = (data[feature] - mean.item()) / std.item()
        return data

    def get_current_sequences(self):
        """
        Get sequences ready for prediction. Returns a list of sequences.
        """
        if len(self.buffer) < self.sequence_length:
            return None, self.sequence_length - len(self.buffer)

        # Normalize the data in the buffer
        normalized_data = [self.normalize_data(data_point) for data_point in self.buffer]

        # Create sequences of length 'sequence_length'
        sequences = []
        for i in range(len(normalized_data) - self.sequence_length + 1):
            sequence = normalized_data[i:i + self.sequence_length]
            sequences.append(torch.tensor([list(dp.values()) for dp in sequence], dtype=torch.float32))

        return sequences, 0

    def impute_missing(self, data_point):
        """
        Impute missing values in a data point using the last valid data point in the buffer.
        """
        if len(self.buffer) > 0:
            last_valid = self.buffer[-1]
            for feature in self.feature_names:
                if pd.isna(data_point[feature]):
                    data_point[feature] = last_valid.get(feature, 0.0)
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
