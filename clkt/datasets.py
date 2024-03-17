from os import PathLike
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import os

import pickle

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from clkt.utils import match_seq_len

DATASET_DIR = "datasets/ASSIST2015/"


class ASSIST2015(Dataset):
    """Assistments2015 dataset.

    Parameters:
        q_seqs:
        r_seqs:
        q_list:
        u_list:
        q2idx:
        u2idx

    """

    def __init__(self, seq_len, dataset_dir=DATASET_DIR) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        self.dataset_path = os.path.join(
            self.dataset_dir, "2015_100_skill_builders_main_problems.csv"
        )

        if os.path.exists(os.path.join(self.dataset_dir, "q_seqs.pkl")):
            with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "rb") as f:
                self.q_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "rb") as f:
                self.r_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "q_list.pkl"), "rb") as f:
                self.q_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "u_list.pkl"), "rb") as f:
                self.u_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "q2idx.pkl"), "rb") as f:
                self.q2idx = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "u2idx.pkl"), "rb") as f:
                self.u2idx = pickle.load(f)
        else:
            self.q_seqs, self.r_seqs, self.q_list, self.u_list, self.q2idx, \
                self.u2idx = self.preprocess()

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]

        if seq_len:
            self.q_seqs, self.r_seqs = \
                match_seq_len(self.q_seqs, self.r_seqs, seq_len)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return self.q_seqs[index], np.array(self.r_seqs[index])

    def __len__(self):
        return self.len

    def preprocess(self):
        df = pd.read_csv(self.dataset_path, encoding="ISO-8859-1")
        df = df[(df["correct"] == 0).values + (df["correct"] == 1).values]

        u_list = np.unique(df["user_id"].values)
        q_list = np.unique(df["sequence_id"].values)

        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}

        q_seqs = []
        r_seqs = []

        for u in u_list:
            df_u = df[df["user_id"] == u].sort_values("log_id")

            q_seq = np.array([q2idx[q] for q in df_u["sequence_id"].values])
            r_seq = tuple(df_u["correct"].values.tolist())

            q_seqs.append(q_seq)
            r_seqs.append(r_seq)

        with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "wb") as f:
            pickle.dump(q_seqs, f)
        with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "wb") as f:
            pickle.dump(r_seqs, f)
        with open(os.path.join(self.dataset_dir, "q_list.pkl"), "wb") as f:
            pickle.dump(q_list, f)
        with open(os.path.join(self.dataset_dir, "u_list.pkl"), "wb") as f:
            pickle.dump(u_list, f)
        with open(os.path.join(self.dataset_dir, "q2idx.pkl"), "wb") as f:
            pickle.dump(q2idx, f)
        with open(os.path.join(self.dataset_dir, "u2idx.pkl"), "wb") as f:
            pickle.dump(u2idx, f)

        return q_seqs, r_seqs, q_list, u_list, q2idx, u2idx


class Assistmets2012Dataset(Dataset):
    """Dataset for Assistments 2012-2013 data.

    """

    def __init__(self, data_path: Union[str, PathLike], batch_size=16, maxlen=3):
        """Dataset constructor

        Args:
            data_path: Path to the csv file with sequences.
            batch_size: Batch size
            maxlen: max length of sequences in the batch.
        """
        self.data_df = pd.read_csv(data_path)
        self.data_df = self.data_df[~self.data_df['skill'].isna()]
        self.groups = self.data_df.groupby('user_id')
        self.users = self.data_df['user_id'].unique()

        self.maxlen = maxlen

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Gets the sequence with the maxlen limit.

        Args:
            idx: index of user (to extract user id).

        Returns:
            PyTorch sequence for specific user.
        """
        user = self.users[idx]
        indices = self.groups.groups[user]
        indices = indices[:self.maxlen]
        sequence_df = self.data_df[indices]

        timestamp = pd.to_datetime(sequence_df.end_time) - pd.to_datetime(sequence_df.start_time)
        timestamp = timestamp.astype(np.int64)

        skills = torch.Tensor(self.data_df['skill'].values).to(torch.int64)
        skills = torch.nn.functional.one_hot(skills)
