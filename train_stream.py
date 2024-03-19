"""
The script is used to evaluate model performance on each test stream for each train dataset to create the stream
scenario.
"""
import json
import os
import random
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Tuple, List, Dict

import wandb
import numpy as np
from sklearn import metrics
import torch
from avalanche.benchmarks import AvalancheDataset, benchmark_from_datasets, task_incremental_benchmark
from avalanche.training import EWC, Naive, JointTraining, LearningWithoutForgetting, AMLCriterion, PNNStrategy, \
    Cumulative
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader

from clkt.datasets import ASSIST2009
from clkt.logging_processing import log
from clkt.models.sakt import SAKT
from clkt.utils import collate_fn


def manual_seed(seed: int) -> None:
    """Sets manual seed for maximum reproducibility.

    See https://pytorch.org/docs/stable/notes/randomness.html for other details.

    Args:
        seed: seed for reproducibility.

    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # torch.utils.deterministic.fill_uninitialized_memory(True)


def get_datasets(ds: Dataset, split_ratio: float, generator: torch.Generator) -> List[Dataset]:
    """Splits dataset corresponding split_ratio

    Args:
        ds: dataset for split.
        split_ratio: split value for split.
        generator: generator for random_split.
    """
    split_size = int(len(ds) * split_ratio)
    ds_list = []
    while len(ds) - split_size > split_size:
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


def get_eval_metrics(eval_model: torch.nn.Module, loader: DataLoader, threshold: float = 0.5) -> Dict[str, float]:
    """

    Args:
        eval_model: Model for evaluation.
        loader: Data loader.
        threshold: threshold for classification metrics.

    Returns:

    """
    assert 0 <= threshold <= 1
    with torch.no_grad():
        for data, labels in loader:
            eval_model.eval()

            p = eval_model(data).detach().cpu().to()
            t = labels.detach().cpu()

            acc = metrics.accuracy_score(y_true=t.numpy(), y_pred=p.numpy() > threshold)
            auc = metrics.roc_auc_score(y_true=t.numpy(), y_score=p.numpy())

            return {'acc': acc, 'auc': auc}


# Set device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set manual seed.
seed = 0
manual_seed(seed)
g = torch.Generator(device=device)
g.manual_seed(seed)
log.debug(g.device)

# Set datasets.
dataset_path = Path(r'...')
dataset = ASSIST2009(100, dataset_path)
datasets = get_datasets(dataset, 0.1, generator=g)

# Set Avalanche train datasets and benchmark.
train_datasets, test_datasets = random_datasets_split(datasets, 0.9, generator=g)
train_datasets = [AvalancheDataset(ds, collate_fn=collate_fn) for ds in train_datasets]
log.debug(f'Train datasets size: {[len(ds) for ds in train_datasets]}')
log.debug(f'Test datasets size: {[len(ds) for ds in test_datasets]}')
bm = benchmark_from_datasets(train=train_datasets)
bm = task_incremental_benchmark(bm)
train_stream = bm.streams['train']

# Set test loaders
test_loaders = []
for ds in test_datasets:
    test_loader = DataLoader(ds, batch_size=len(ds), collate_fn=partial(collate_fn, return_task=False))
    test_loaders.append(test_loader)

# Set model, criterion and optimizer.
model = SAKT(564, n=100, d=100, num_attn_heads=5, dropout=0.2)
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = binary_cross_entropy

# Define CL Strategy and init wandb run.
config_args = {
    'epochs': 100,
    'batch_size': 256,
    'model': str(model),
    'optimizer': str(optimizer),
    'criterion': criterion,
}

# cl_strategy = Naive(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
#                     train_epochs=config_args['epochs'],
#                     eval_mb_size=32, device=device)

# config_args['ewc_lambda'] = 0.1
# cl_strategy = EWC(model, optimizer, criterion, ewc_lambda=config_args['ewc_lambda'],
#                   train_mb_size=config_args['batch_size'],
#                   train_epochs=config_args['epochs'],
#                   eval_mb_size=32, device=device)


# cl_strategy = Replay(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
#                      train_epochs=config_args['epochs'],
#                      eval_mb_size=32, device=device)

# cl_strategy = JointTraining(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
#                             train_epochs=config_args['epochs'],
#                             eval_mb_size=32, device=device)

cl_strategy = Cumulative(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
                         train_epochs=config_args['epochs'], device=device)

# JointTraining, LearningWithoutForgetting, AMLCriterion, PNNStrategy

config_args['strategy'] = str(cl_strategy)

wandb.login(key='...')
wandb.init(project='clkt-experiments', config=config_args)

# Run CL tasks.
for train_task in train_stream:
    model.train()
    train_task.task_labels = list(train_task.task_labels)
    cl_strategy.train(train_task, pin_memory=False)
    # eval_metrics = [get_eval_metrics(model, test_loader) for i, test_loader in enumerate(test_loaders)]
    for i, test_loader in enumerate(test_loaders):
        eval_metrics = get_eval_metrics(model, test_loader)
        eval_metrics = {f'{key}-{i}': value for key, value in eval_metrics.items()}
        log.debug(eval_metrics)
        wandb.log(eval_metrics)

wandb.finish()
