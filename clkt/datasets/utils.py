from typing import List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split, Dataset


def match_seq_len(q_seqs, r_seqs, seq_len, pad_val=-1):
    '''
        Args:
            q_seqs: the question(KC) sequences with the size of \
                [batch_size, some_sequence_length]
            r_seqs: the response sequences with the size of \
                [batch_size, some_sequence_length]

            Note that the "some_sequence_length" is not uniform over \
                the whole batch of q_seqs and r_seqs

            seq_len: the sequence length to match the q_seqs, r_seqs \
                to same length
            pad_val: the padding value for the sequence with the length \
                longer than seq_len

        Returns:
            proc_q_seqs: the processed q_seqs with the size of \
                [batch_size, seq_len + 1]
            proc_r_seqs: the processed r_seqs with the size of \
                [batch_size, seq_len + 1]
    '''
    proc_q_seqs = []
    proc_r_seqs = []

    for q_seq, r_seq in zip(q_seqs, r_seqs):
        i = 0
        while i + seq_len + 1 < len(q_seq):
            proc_q_seqs.append(q_seq[i:i + seq_len + 1])
            proc_r_seqs.append(r_seq[i:i + seq_len + 1])

            i += seq_len + 1

        proc_q_seqs.append(
            np.concatenate(
                [
                    q_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
                ]
            )
        )
        proc_r_seqs.append(
            tuple(np.concatenate(
                [
                    r_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
                ]
            ).tolist())
        )

    return proc_q_seqs, proc_r_seqs


def collate_fn(batch, pad_val=-1, return_task=True):
    """
        The collate function for torch.utils.data.DataLoader

        Returns:
            q_seqs: the question(KC) sequences with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            r_seqs: the response sequences with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            qshft_seqs: the question(KC) sequences which were shifted \
                one step to the right with ths size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            rshft_seqs: the response sequences which were shifted \
                one step to the right with ths size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            mask_seqs: the mask sequences indicating where \
                the padded entry is with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]

    """
    q_seqs = []
    r_seqs = []
    qshft_seqs = []
    rshft_seqs = []
    task_seqs = []
    for data in batch:
        if return_task:
            q_seq, r_seq, task_id = data
            task_seqs.append(task_id)
        else:
            q_seq, r_seq = data
        q_seqs.append(torch.Tensor(q_seq[:-1]))
        r_seqs.append(torch.Tensor(r_seq[:-1]))
        qshft_seqs.append(torch.Tensor(q_seq[1:]))
        rshft_seqs.append(torch.Tensor(r_seq[1:]))

    q_seqs = pad_sequence(
        q_seqs, batch_first=True, padding_value=pad_val
    )
    r_seqs = pad_sequence(
        r_seqs, batch_first=True, padding_value=pad_val
    )
    qshft_seqs = pad_sequence(
        qshft_seqs, batch_first=True, padding_value=pad_val
    )
    rshft_seqs = pad_sequence(
        rshft_seqs, batch_first=True, padding_value=pad_val
    )

    mask_seqs = (q_seqs != pad_val) * (qshft_seqs != pad_val)

    q_seqs, r_seqs, qshft_seqs, rshft_seqs = \
        q_seqs * mask_seqs, r_seqs * mask_seqs, qshft_seqs * mask_seqs, \
        rshft_seqs * mask_seqs

    y = torch.masked_select(rshft_seqs, mask_seqs)
    x = torch.stack((q_seqs.long(), r_seqs.long(), qshft_seqs.long(), mask_seqs))
    # x = torch.stack((q_seqs, r_seqs, qshft_seqs, mask_seqs))
    # x = torch.stack((q_seqs, r_seqs, qshft_seqs, mask_seqs))
    if return_task:
        return x, y, torch.Tensor(task_seqs)
    else:
        return x, y


def get_datasets(ds: Dataset, split_ratio: float, generator: torch.Generator) -> List[Dataset]:
    """Splits the dataset into the list of datasets with corresponding split_ratio.

    Args:
        ds: dataset for split.
        split_ratio: split value for split.
        generator: generator for random_split.
    """
    split_size = int(len(ds) * split_ratio)
    ds_list = []
    while len(ds) - split_size >= split_size:
        ds, split_ds = random_split(ds, [len(ds) - split_size, split_size], generator=generator)
        ds_list.append(split_ds)
    ds_list.append(ds)
    return ds_list


def random_datasets_split(datasets: List[Dataset], train_ratio: float, generator) -> Tuple[
    List[Dataset], List[Dataset]]:
    """Gets the list of train and test datasets corresponding to train datasets.

    Args:
        datasets: List of train datasets.
        train_ratio: percent that defines train dataset size.
        generator: Generator for random split.

    Returns:
        List of test datasets corresponding to train datasets.

    """
    train_datasets, test_datasets = [], []

    for ds in datasets:
        train_size = int(len(ds) * train_ratio)
        test_size = len(ds) - train_size
        train_dataset, test_dataset = random_split(ds, [train_size, test_size], generator=generator)
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
    return train_datasets, test_datasets
