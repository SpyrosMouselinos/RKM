from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd


class TimeSeriesImputationDataset(Dataset):
    """
    A PyTorch Dataset for time series data imputation.

    @param data: The time series data as a NumPy array.
    @param timestamps: The timestamps corresponding to the data.
    @param sequence_length: The length of sequences used for training.
    @param column_names: Optional list of column names in the data.
    @param group_by: The frequency to group the data (e.g., 'seconds', 'minutes').
    @param target_offset: The offset for the target variable.
    @param impute_backward: The number of periods to fill forward during imputation.
    """
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
        """
        Calculate the mean and standard deviation for normalization.

        @return: A dictionary with 'mean' and 'std' keys containing the mean and standard deviation.
        """
        means = np.nanmean(self.data, axis=0)
        stds = np.nanstd(self.data, axis=0)
        return {'mean': means, 'std': stds}

    def normalize_data(self, data):
        """
        Normalize the data using the mean and standard deviation.

        @param data: The data to normalize.
        @return: The normalized data.
        """
        return (data - self.mean_std['mean']) / self.mean_std['std']

    def group_and_impute(self):
        """
        Group the data by the specified frequency and impute missing values.

        @return: A tuple containing the grouped and imputed data, and a set of timestamps with NaN values.
        """
        df = pd.DataFrame(self.data, index=self.timestamps)
        df = df.resample(self.group_by).mean()
        for col in df.columns:
            df[col] = df[col].fillna(method='ffill', limit=self.impute_backward)
        # Identify rows with NaN values after imputation
        valid_rows = ~np.isnan(df).any(axis=1)
        invalid_timestamps = df.index[~valid_rows]  # Timestamps with NaN values
        return df[valid_rows], set(invalid_timestamps)

    def find_valid_sequence_starts(self):
        """
        Find the start indices of valid sequences in the data.

        @return: A list of valid start indices for sequences.
        """
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
        """
        Get the mean and standard deviation statistics for features.

        @return: A dictionary with feature names and their mean and std values.
        """
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
        """
        Get the number of valid sequences.

        @return: The number of valid sequences.
        """
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        Get a sequence and its corresponding target.

        @param idx: The index of the sequence to retrieve.
        @return: A tuple containing the input tensor and the target tensor.
        """
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length

        x = self.data.iloc[start_idx:end_idx].values
        y = self.data.iloc[end_idx: end_idx + self.target_offset].values

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return x_tensor, y_tensor


class RealTimeTimeSeriesDataset:
    """
    A class for handling real-time time series data, including normalization and buffering.

    @param feature_names: The names of the features in the dataset.
    @param feature_stats: The mean and standard deviation for normalization.
    @param sequence_length: The length of sequences used for prediction.
    @param timegroup_factor: The frequency to group the data (e.g., '24H').
    """
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

        @param new_data_point: The new data point to add to the buffer.
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

        @param data: The data point with possible corrupted values.
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

        @param feature: The feature name to forward fill.
        @return: The forward-filled value for the feature.
        """
        for i in reversed(range(len(self.buffer))):
            if not pd.isna(self.buffer[i][feature]) and not np.isinf(self.buffer[i][feature]):
                return self.buffer[i][feature]
        # If no valid value is found, return 0 as a default
        return 0.0

    def backward_fill(self, feature):
        """
        Perform backward fill for the given feature.

        @param feature: The feature name to backward fill.
        @return: The backward-filled value for the feature.
        """
        for i in range(len(self.buffer)):
            if not pd.isna(self.buffer[i][feature]) and not np.isinf(self.buffer[i][feature]):
                return self.buffer[i][feature]
        # If no valid value is found, return 0 as a default
        return 0.0

    def group_data_by_time(self):
        """
        Group the buffer data by the timegroup_factor.

        @return: A list of dictionaries with the grouped data.
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

        @param data: The data point to normalize.
        @return: The normalized data point.
        """
        for feature in self.feature_names:
            mean = self.feature_stats.get(f"{feature}_mean", torch.tensor(0.0))
            std = self.feature_stats.get(f"{feature}_std", torch.tensor(1.0))
            data[feature] = (data[feature] - mean.item()) / std.item()
        return data

    def get_current_sequences(self):
        """
        Get sequences ready for prediction. Returns a list of sequences.

        @return: A tuple containing a list of sequences and the number of sequences needed to reach the required length.
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

    def format_incoming_values(self, data_point):
        """
        Impute missing values in a data point using the last valid data point in the buffer.

        @param data_point: The data point with possible missing values.
        @return: The data point with imputed values.
        """
        if len(self.buffer) > 0:
            last_valid = self.buffer[-1]
            for feature in self.feature_names:
                if pd.isna(data_point[feature]):
                    data_point[feature] = last_valid.get(feature, 0.0)
        return data_point


def find_and_convert_date_column(df):
    """
    Find and convert the most likely date column in a DataFrame to a 'date' column.

    @param df: The DataFrame to process.
    @return: The DataFrame with the identified date column renamed to 'date'.
    """
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


# Test of incoming values imputation
def test_incoming_values_imputation():
    """
    Test the imputation of incoming values using a sample RealTimeTimeSeriesDataset.

    This function initializes a RealTimeTimeSeriesDataset, reads a sample CSV file,
    and formats the incoming values for testing the imputation functionality.
    """
    import random

    # Initialize a RealTimeDataset
    rttsd = RealTimeTimeSeriesDataset(feature_names=['High', 'Low'],
                                      feature_stats={'High_mean': 0.0,
                                                     'Low_mean': 0.0,
                                                     'High_std': 1000.0,
                                                     'Low_std': 1000.0},
                                      sequence_length=10,
                                      timegroup_factor='24H')

    # Read the CSV file from /data/
    df = pd.read_csv('../data/yahoo_stock.csv')

    # Keep an index of current rows elapsed
    index = 0

    # In a loop, until the end of the dataset
    while index < len(df):
        # Take a single row from the dataset
        data_point = df.iloc[index]

        # Format it as a JSON-like dicitonary
        data_point = data_point.to_dict()

        # Add a random-named feature with a random integer value
        data_point['random_feature'] = random.randint(0, 10)

        # Order the features in reverse alphabetical order
        data_point = {k: v for k, v in sorted(data_point.items(), key=lambda item: item[0], reverse=True)}

        # Format the incoming values
        data_point = rttsd.format_incoming_values(data_point)

        # Add it to the buffer
        rttsd.buffer.append(data_point)

        # Increment the index
        index += 1
