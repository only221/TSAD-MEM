import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
from typing import Union, Tuple

class AnomalyDataset(Dataset):
    def __init__(self, data, seq_len, step=1, mode='train', scaler=None):
        self.data = data
        self.seq_len = seq_len
        self.step = step
        self.mode = mode


        if isinstance(data, pd.DataFrame):
            label_data = data[data['cols'] == 'label']
            feature_data = data[data['cols'] != 'label']
            channels = feature_data['cols'].unique()
            if mode == 'train':  
                print(f"c_in: {len(channels)}")
            
            feature_pivot = feature_data.pivot_table(
                index='date', 
                columns='cols', 
                values='data', 
                fill_value=0
            )
            
            feature_pivot = feature_pivot.reindex(columns=sorted(feature_pivot.columns))
            self.features = feature_pivot.values
            
            if len(label_data) > 0:
                label_pivot = label_data.pivot_table(
                    index='date',
                    values='data',
                    aggfunc='first'  
                )

                label_pivot = label_pivot.reindex(feature_pivot.index, fill_value=0)
                self.labels = label_pivot.values.flatten()
            else:
                self.labels = np.zeros(len(feature_pivot))
            
        else:
            self.features = data
            self.labels = np.zeros(len(data))
        
        if mode == 'train':
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:
            if scaler is None:
                raise ValueError("Scaler must be provided for validation/test datasets")
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)
        
        self.sequences, self.sequence_labels = self._create_sequences()
    
    def _create_sequences(self):

        num_sequences = (len(self.features) - self.seq_len) // self.step + 1

        sequences = np.empty((num_sequences, self.seq_len, self.features.shape[1]), dtype=np.float32)
        labels = np.empty((num_sequences, self.seq_len), dtype=np.float32)
        
        for i in range(num_sequences):
            start_idx = i * self.step
            end_idx = start_idx + self.seq_len
            
            sequences[i] = self.features[start_idx:end_idx]
            labels[i] = self.labels[start_idx:end_idx]
        
        return sequences, labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.FloatTensor(self.sequence_labels[idx])
        return sequence, label

def get_data_loader_with_meta(data_path, meta_path, dataset_name, config):

    meta_info = load_meta_info(meta_path, dataset_name)
    train_lens = meta_info['train_lens']
    data = pd.read_csv(data_path)
    data = process_data_df(data)
    data = data.reset_index(drop=True)
    
    train, test = split_before(data, train_lens)
    train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )
    test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )    

    return train_data, train_label, test_data, test_label


def load_meta_info(meta_path, dataset_name):
    meta_df = pd.read_csv(meta_path)
    dataset_info = meta_df[meta_df['file_name'] == f'{dataset_name}.csv']
    
    if len(dataset_info) == 0:
        raise ValueError(f"Dataset {dataset_name} not found in meta file")
    
    return {
        'train_lens': int(dataset_info['train_lens'].iloc[0]),
        'total_lens': int(dataset_info['time_steps'].iloc[0]) if not pd.isna(dataset_info['time_steps'].iloc[0]) else None
    }



def process_data_df(data: pd.DataFrame, nrows=None) -> pd.DataFrame:
    """
    Read the data file and return DataFrame.

    According to the provided file path, read the data file and return the corresponding DataFrame.

    :param data: Data frame to read.
    :return:  The DataFrame of the content of the data file.
    """
    label_exists = "label" in data["cols"].values

    all_points = data.shape[0]

    columns = data.columns

    if columns[0] == "date":
        n_points = data.iloc[:, 2].value_counts().max()
    else:
        n_points = data.iloc[:, 1].value_counts().max()

    is_univariate = n_points == all_points

    n_cols = all_points // n_points
    df = pd.DataFrame()

    cols_name = data["cols"].unique()

    if columns[0] == "date" and not is_univariate:
        df["date"] = data.iloc[:n_points, 0]
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 1].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    elif columns[0] != "date" and not is_univariate:
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 0].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)

    elif columns[0] == "date" and is_univariate:
        df["date"] = data.iloc[:, 0]
        df[cols_name[0]] = data.iloc[:, 1]

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    else:
        df[cols_name[0]] = data.iloc[:, 0]

    if label_exists:
        # Get the column name of the last column
        last_col_name = df.columns[-1]
        # Renaming the last column as "label"
        df.rename(columns={last_col_name: "label"}, inplace=True)

    if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
        df = df.iloc[:nrows, :]

    return df

def split_before(
    data: Union[pd.DataFrame, np.ndarray], index: int
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[np.ndarray, np.ndarray]]:
    """
    Split time series data into two parts at the specified index.

    :param data: Time series data to be segmented.
                 Can be a pandas DataFrame or a NumPy array.
    :param index: Split index position.
    :return: Tuple containing the first and second parts of the data.
    """
    if isinstance(data, pd.DataFrame):
        return data.iloc[:index, :], data.iloc[index:, :]
    elif isinstance(data, np.ndarray):
        return data[:index, :], data[index:, :]
    else:
        raise TypeError("Input data must be a pandas DataFrame or a NumPy array.")

class SegLoader(object):
    def __init__(self, data, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.data = data
        self.test_labels = data      

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.data.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.data.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.data.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.data.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.data[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.data[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.data[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.data[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

def anomaly_detection_data_provider(data, batch_size, win_size=100, step=100, mode='train'):
    dataset = SegLoader(data, win_size, 1, mode)

    shuffle = False
    if mode == 'train' or mode == 'val':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0,
                             drop_last=False)
    return data_loader