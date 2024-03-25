"""
The script is used to show the example of Avalance continual learning process using Assistment2009 dataset and SAKT.
"""

import os
from functools import partial
from pathlib import Path
from typing import Union, List, Tuple

import click
import torch
import wandb
from avalanche.benchmarks import AvalancheDataset, benchmark_from_datasets, task_incremental_benchmark
from avalanche.training import EWC, Naive, Cumulative, Replay, FromScratchTraining
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split

from clkt.config import ConfigFile
from clkt.datasets.assist2009 import ASSIST2009
from clkt.datasets.utils import collate_fn
from clkt.logging_processing import log
from clkt.models.sakt import SAKT
from clkt.utils import manual_seed


def set_manual_seed(seed: int = 0) -> None:
    """Sets manual seed for the script.

    Args:
        seed:

    Returns:

    """
    manual_seed(seed)
    g = torch.Generator(device=device)
    g.manual_seed(seed)


def get_train_test(dataset_path: Union[str, os.PathLike]) -> Tuple[List[Dataset], List[Dataset]]:
    # Set datasets.
    dataset = ASSIST2009(100, dataset_path)

    train_ratio = config.train_ratio
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=g)

    # Set Avalanche train datasets and benchmark.
    train_datasets = get_datasets(train_dataset, 0.1, generator=g)
    train_datasets, test_datasets = random_datasets_split(train_datasets, 0.9, generator=g)
    train_datasets = [AvalancheDataset(ds, collate_fn=collate_fn) for ds in train_datasets]
    return train_datasets, test_datasets


def get_device(device_name):
    device = torch.device(device_name)


# Set device.
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

torch.set_default_device(device)

# Set manual seed.

# Set datasets.
dataset_path = Path(r'../../data/assist2009/skill_builder_data.csv')
dataset = ASSIST2009(100, dataset_path)

train_ratio = 0.1
train_size = int(len(dataset) * train_ratio)
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=g)

# Set Avalanche train datasets and benchmark.
train_datasets = get_datasets(train_dataset, 0.1, generator=g)
train_datasets, test_datasets = random_datasets_split(train_datasets, 0.9, generator=g)
train_datasets = [AvalancheDataset(ds, collate_fn=collate_fn) for ds in train_datasets]
log.debug(f'Train datasets size: {[len(ds) for ds in train_datasets]}')
bm = benchmark_from_datasets(train=train_datasets)
bm = task_incremental_benchmark(bm)
train_stream = bm.streams['train']

# Set test loader
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn=partial(collate_fn, return_task=False),
                         generator=g)

# Set test loaders
test_loaders = []
for ds in test_datasets:
    test_loader = DataLoader(ds, batch_size=len(ds), collate_fn=partial(collate_fn, return_task=False), generator=g)
    test_loaders.append(test_loader)

# Set model, criterion and optimizer.
model = SAKT(564, n=100, d=100, num_attn_heads=5, dropout=0.2)
# model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = binary_cross_entropy

# Define CL Strategy and init wandb run.
config_args = {
    'epochs': 10,
    'batch_size': 256,
    'model': str(model),
    'optimizer': str(optimizer),
    'criterion': criterion,
}

cl_strategy = Naive(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
                    train_epochs=config_args['epochs'],
                    eval_mb_size=32, device=device)

config_args['ewc_lambda'] = 0.1
cl_strategy = EWC(model, optimizer, criterion, ewc_lambda=config_args['ewc_lambda'],
                  train_mb_size=config_args['batch_size'],
                  train_epochs=config_args['epochs'],
                  eval_mb_size=32, device=device)

# config_args['ewc_lambda'] = 0.1
# config_args['ewc_mode'] = 'online'
# config_args['decay_factor'] = 0.5
# cl_strategy = EWC(model, optimizer, criterion, ewc_lambda=config_args['ewc_lambda'],
#                   mode=config_args['ewc_mode'], decay_factor=config_args['decay_factor'],
#                   train_mb_size=config_args['batch_size'],
#                   train_epochs=config_args['epochs'],
#                   eval_mb_size=32, device=device)


cl_strategy = Cumulative(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
                         train_epochs=config_args['epochs'], device=device)

cl_strategy = Replay(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
                     train_epochs=config_args['epochs'],
                     eval_mb_size=32, device=device)

# cl_strategy = JointTraining(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
#                             train_epochs=config_args['epochs'],
#                             eval_mb_size=32, device=device)


# # Not Working
# cl_strategy = GDumb(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
#                          train_epochs=config_args['epochs'], device=device)

# Not Working
# config_args['alpha'] = 0.1
# config_args['temperature'] = 0.1
# cl_strategy = LwF(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
#                   alpha=config_args['alpha'], temperature=config_args['temperature'],
#                   train_epochs=config_args['epochs'], device=device)


# It works!!!
# config_args['patterns_per_exp'] = 5
# cl_strategy = AGEM(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
#                    patterns_per_exp=config_args['patterns_per_exp'],
#                   train_epochs=config_args['epochs'], device=device)


# config_args['patterns_per_exp'] = 5
# cl_strategy = GEM(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
#                    patterns_per_exp=config_args['patterns_per_exp'],
#                   train_epochs=config_args['epochs'], device=device)


# config_args['si_lambda'] = 0.5
# cl_strategy = SynapticIntelligence(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
#                                    si_lambda=config_args['si_lambda'],
#                                    train_epochs=config_args['epochs'], device=device)

# Not working
# cl_strategy = CoPE(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
#                    train_epochs=config_args['epochs'], device=device)

# Not working
# config_args['lambda_e'] = 0.5
# cl_strategy = LFL(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
#                   lambda_e=config_args['lambda_e'],
#                   train_epochs=config_args['epochs'], device=device)

# # Not working
# # Cannot pin tensors
# cl_strategy = GenerativeReplay(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
#                                train_epochs=config_args['epochs'], device=device)

# Not working
# config_args['alpha'] = 0.9
# cl_strategy = MAS(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
#                   alpha=config_args['alpha'],
#                   train_epochs=config_args['epochs'], device=device)

# It needs targets
# cl_strategy = BiC(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
#                   train_epochs=config_args['epochs'], device=device)

# It needs targets
# cl_strategy = ER_ACE(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
#                   train_epochs=config_args['epochs'], device=device)

# # Not working
# cl_strategy = SCR(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
#                   train_epochs=config_args['epochs'], device=device)


cl_strategy = FromScratchTraining(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
                                  train_epochs=config_args['epochs'], device=device)

# # Not working
# cl_strategy = ExpertGateStrategy(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
#                                   train_epochs=config_args['epochs'], device=device)


# Not working
# cl_strategy = LearningToPrompt(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
#                   train_epochs=config_args['epochs'], device=device)

# cl_strategy = DER(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
#                                   train_epochs=config_args['epochs'], device=device)

# cl_strategy = LaMAML(model, optimizer, criterion, train_mb_size=config_args['batch_size'],
#                                   train_epochs=config_args['epochs'], device=device)


config_args['strategy'] = str(cl_strategy)

wandb.login(key='...')
# wandb.init(project='clkt-experiments', config=config_args, mode="disabled")
wandb.init(project='clkt-experiments', config=config_args)

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


@click.command()
@click.option('--dataset_path', type=click.Path(exists=True), default='data/assist2009/skill_builder_data.csv',
              description='Assitments2009 data path.')
@click.option('--epochs', type=int, default=10, description='Number of epochs.')
@click.option('--split-ratio', type=float, default=0.1, description='Test split ratio.')
@click.option('--config', type=click.Path(exists=True), default=Path(__file__).parent / 'config.yaml',
              description='Minimum sequence length for sequential data.')
def run_experiments(dataset_path: Union[str, os.PathLike], epochs: int, batch_size: int,
                    config=Union[str, os.PathLike]):
    """Runs the Continual learning experiment with some Avalanche strategies and knowledge tracing model."""
    run_config = ConfigFile(config)
