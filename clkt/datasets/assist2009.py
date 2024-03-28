import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from clkt.datasets.utils import match_seq_len


class ASSIST2009(Dataset):
    def __init__(self, seq_len, dataset_path) -> None:
        super().__init__()

        dataset_path = Path(dataset_path)
        self.dataset_path = dataset_path
        self.dataset_dir = dataset_path.parent
        filename = self.dataset_path.stem
        if os.path.exists(os.path.join(self.dataset_dir, f"q_seqs_{filename}.pkl")):
            with open(os.path.join(self.dataset_dir, f"q_seqs_{filename}.pkl"), "rb") as f:
                self.q_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, f"r_seqs_{filename}.pkl"), "rb") as f:
                self.r_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, f"q_list_{filename}.pkl"), "rb") as f:
                self.q_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, f"u_list_{filename}.pkl"), "rb") as f:
                self.u_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, f"q2idx_{filename}.pkl"), "rb") as f:
                self.q2idx = pickle.load(f)
            with open(os.path.join(self.dataset_dir, f"u2idx_{filename}.pkl"), "rb") as f:
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

    # def __getitem__(self, index):
    #     return self.q_seqs[index], self.r_seqs[index]
    def __getitem__(self, index):
        return self.q_seqs[index], np.array(self.r_seqs[index])

    def __len__(self):
        return self.len

    def preprocess(self):
        df = pd.read_csv(self.dataset_path, encoding='latin').dropna(subset=["skill_name"]) \
            .drop_duplicates(subset=["order_id", "skill_name"]) \
            .sort_values(by=["order_id"])

        u_list = np.unique(df["user_id"].values)
        q_list = np.unique(df["skill_name"].values)

        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}

        q_seqs = []
        r_seqs = []

        for u in u_list:
            df_u = df[df["user_id"] == u]

            q_seq = np.array([q2idx[q] for q in df_u["skill_name"]])
            r_seq = df_u["correct"].values

            q_seqs.append(q_seq)
            r_seqs.append(r_seq)
        filename = self.dataset_path.stem
        with open(os.path.join(self.dataset_dir, f"q_seqs_{filename}.pkl"), "wb") as f:
            pickle.dump(q_seqs, f)
        with open(os.path.join(self.dataset_dir, f"r_seqs_{filename}.pkl"), "wb") as f:
            pickle.dump(r_seqs, f)
        with open(os.path.join(self.dataset_dir, f"q_list_{filename}.pkl"), "wb") as f:
            pickle.dump(q_list, f)
        with open(os.path.join(self.dataset_dir, f"u_list_{filename}.pkl"), "wb") as f:
            pickle.dump(u_list, f)
        with open(os.path.join(self.dataset_dir, f"q2idx_{filename}.pkl"), "wb") as f:
            pickle.dump(q2idx, f)
        with open(os.path.join(self.dataset_dir, f"u2idx_{filename}.pkl"), "wb") as f:
            pickle.dump(u2idx, f)

        return q_seqs, r_seqs, q_list, u_list, q2idx, u2idx
