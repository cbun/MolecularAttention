import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

class DescriptorsDataset(Dataset):
    """ Descriptors dataset."""

    # Initialize your data, download, etc.
    def __init__(self, df_X, df_y):

        self.len = df_X.shape[0]
        self.x_data = df_X
        self.y_data = df_y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len



def load_descriptor_data_sets(descriptor_fname, train_idx=None, test_idx=None, read_n_rows=None, random_state=42):
    df = pd.read_csv(descriptor_fname, delimiter="\t", nrows=read_n_rows)

    ## Handle strings and NaNs
    df.replace(regex=r"False|0.0", value=0.0, inplace=True)
    df.replace(regex=r"True|1.0", value=1.0, inplace=True)
    # df = df.fillna(df.mean())
    df = df.fillna(0)

    if train_idx is None and test_idx is None:
        exit('todo handle no indexes')

    ## Take abs() of docking scores for better training
    data_docking_score = abs(np.clip(df.iloc[:, 2], a_min=None, a_max=0))
    data_docking_score = np.asarray(data_docking_score).astype(np.float32)

    ## Standardize
    scaler = StandardScaler()
    data_descriptors = df.iloc[:, 3:]
    data_descriptors = scaler.fit_transform(data_descriptors).astype(np.float32)

    train_dataset = DescriptorsDataset(data_descriptors[train_idx], data_docking_score[train_idx])
    test_dataset = DescriptorsDataset(data_descriptors[test_idx], data_docking_score[test_idx])
    return train_dataset, test_dataset
