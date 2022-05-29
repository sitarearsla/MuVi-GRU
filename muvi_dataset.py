import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os


class MuVi_Dataset(Dataset):
    """Dataset of music videos"""

    def __init__(self, data_root, seq_len, ewe):
        self.opensmileFeatureDir = data_root + 'emobase_features'
        self.urls = pd.read_csv(data_root + 'video_urls.csv')
        self.ewe = ewe # Evaluator Weighted Estimator
        self.seq_len = seq_len
        self.len = len(self.urls)  # Size of data

    def __getitem__(self, index):
        x_data, y_arousal, y_valence = [], [], []
        url = self.urls['video_id'].iloc[index]
        media_id = url.split('_')[0]
        data = self.load_opensmile(url)

        data = data.truncate(after=117)  # length=118 for training
        data = data.iloc[:, 2:(len(data.columns) - 1)]  # drop columns from opensmile not required for model training

        for j in range(len(data) - self.seq_len):
            x_data.append(np.array(data[j:(j + self.seq_len)]))

        # keep last arousal-valence value of each window as the prediction target
        y_arousal = pd.Series([x for x in self.ewe if x[0] == media_id if x[1] == 'music'][0][2])
        y_valence = pd.Series([x for x in self.ewe if x[0] == media_id if x[1] == 'music'][0][3])

        y_arousal = y_arousal.truncate(before=len(y_arousal) - 118 + self.seq_len).reset_index(drop=True)
        y_arousal = y_arousal.truncate(after=len(data) - self.seq_len - 1)

        y_valence = y_valence.truncate(before=len(y_valence) - 118 + self.seq_len).reset_index(drop=True)
        y_valence = y_valence.truncate(after=len(data) - self.seq_len - 1)

        y_arousal = np.array(y_arousal)
        y_valence = np.array(y_valence)

        print(len(x_data))
        print(y_arousal.shape)
        print(y_valence.shape)

        return x_data, y_arousal, y_valence

    def __len__(self):
        return self.len

    def load_opensmile(self, video_id):
        """get audio features of music videos"""
        filename = self.opensmileFeatureDir + '/' + video_id + '.csv'
        data = pd.read_csv(filename)
        return data
