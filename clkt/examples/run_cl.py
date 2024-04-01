"""
The script is used to show the example of Avalance continual learning process using Assistment2009 dataset and SAKT.
"""

import os
import random
from functools import partial
from pathlib import Path
from typing import Union, Iterable, Dict

import click
import numpy as np
import torch
import wandb
from avalanche.benchmarks import benchmark_from_datasets, task_incremental_benchmark, AvalancheDataset
from avalanche.benchmarks.scenarios.generic_scenario import TCLStream
from avalanche.training import EWC, Naive, Cumulative, Replay, FromScratchTraining
from avalanche.training.templates import SupervisedTemplate
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from clkt.config import ConfigFile, YAMLType
from clkt.datasets.assist2009 import ASSIST2009
from clkt.datasets.utils import collate_fn
from clkt.datasets.utils import get_datasets, random_datasets_split
from clkt.logging_processing import log
from clkt.models.sakt import SAKT
from clkt.models.utils import get_eval_metrics


def run_experiment(cl_strategy: SupervisedTemplate, model: torch.nn.Module, train_stream: TCLStream,
                   test_loaders: Iterable[DataLoader], test_loader: DataLoader,
                   config_args: Dict[str, YAMLType]) -> None:
    """Runs the experiment for the specific CL strategy.

    Args:
        cl_strategy: Continual Learning strategy
        model: Knowledge Tracing model for ASSISTMENT2009.
        train_stream: Train stream from benchmark object.
        test_loaders: List of test loaders.
        test_loader: Test loader for the overall ASSISTMENT2009 dataset.
        config_args: Config parameters for wandb logging.

    """
    config_args = config_args.copy()
    config_args['strategy'] = str(cl_strategy)
    wandb.init(project=config_args['project'], config=config_args)
    # Run CL tasks.
    for train_task in train_stream:
        model.train()
        train_task.task_labels = list(train_task.task_labels)
        cl_strategy.train(train_task)
        # eval_metrics = [get_eval_metrics(model, test_loader) for i, test_loader in enumerate(test_loaders)]
        eval_metrics = get_eval_metrics(model, test_loader)
        log.debug(eval_metrics)
        wandb.log(eval_metrics)
        for i, test_loader in enumerate(test_loaders):
            eval_metrics = get_eval_metrics(model, test_loader)
            eval_metrics = {f'{key}-{i}': value for key, value in eval_metrics.items()}
            log.debug(eval_metrics)
            wandb.log(eval_metrics)

    wandb.finish()


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


@click.command()
@click.option('--data-path', type=click.Path(exists=True), default='data/assist2009/skill_builder_data.csv',
              help='Assitments2009 data path.')
@click.option('--config', type=click.Path(exists=True),
              default=Path(__file__).parent / 'config' / 'run_cl.yaml',
              help='Config file with parameters.')
def run_experiments(data_path: Union[str, os.PathLike], config=Union[str, os.PathLike]):
    """Runs the Continual learning experiment with some Avalanche strategies and knowledge tracing model."""

    # Define config parameters.
    config_args = ConfigFile(config)

    # Reproducibility and device.
    device = torch.device(config_args.device)
    seed = config_args.seed
    manual_seed(seed)
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # Set datasets.
    dataset_path = Path(data_path)
    dataset = ASSIST2009(config_args.seq_len, dataset_path)

    train_ratio = config_args.train_ratio
    split_ratio = config_args.split_ratio
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=g)
    train_datasets = get_datasets(train_dataset, split_ratio=split_ratio, generator=g)
    train_datasets, test_datasets = random_datasets_split(train_datasets, train_ratio, generator=g)
    train_datasets = [AvalancheDataset(ds, collate_fn=collate_fn) for ds in train_datasets]
    bm = benchmark_from_datasets(train=train_datasets)
    bm = task_incremental_benchmark(bm)
    train_stream = bm.streams['train']

    # Set test loaders
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset),
                             collate_fn=partial(collate_fn, return_task=False), generator=g)
    test_loaders = []
    for ds in test_datasets:
        test_loader = DataLoader(ds, batch_size=len(ds), collate_fn=partial(collate_fn, return_task=False), generator=g)
        test_loaders.append(test_loader)

    # Set model, criterion and optimizer.
    batch_size = config_args.batch_size
    epochs = config_args.epochs
    seq_len = config_args.seq_len
    model = SAKT(564, n=seq_len, d=seq_len, num_attn_heads=5, dropout=0.2)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = binary_cross_entropy

    # CL strategies.
    from_scratch = FromScratchTraining(model, optimizer, criterion, train_mb_size=batch_size, train_epochs=epochs,
                                       device=device)

    naive = Naive(model, optimizer, criterion, train_mb_size=batch_size, train_epochs=epochs, device=device)

    ewc = EWC(model, optimizer, criterion, ewc_lambda=config_args.ewc_lambda, train_mb_size=batch_size,
              train_epochs=epochs, device=device)

    cumulative = Cumulative(model, optimizer, criterion, train_mb_size=batch_size, train_epochs=epochs, device=device)

    replay = Replay(model, optimizer, criterion, train_mb_size=batch_size, train_epochs=epochs, device=device)

    strategies = [from_scratch, naive, ewc, replay, cumulative]

    # Run experiments
    for strategy in strategies:
        run_experiment(strategy, model, train_stream, test_loaders, test_loader, config_args.config)
