from pathlib import Path

import torch
from avalanche.benchmarks.utils import AvalancheDataset, DataAttribute
from avalanche.benchmarks.scenarios.dataset_scenario import DatasetExperience
from avalanche.benchmarks import benchmark_from_datasets, task_incremental_benchmark, data_incremental_benchmark, \
    nc_benchmark, ni_benchmark
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader
from avalanche.benchmarks.utils import DataAttribute
from torch.utils.data.dataloader import DataLoader
from avalanche.training import Naive
from avalanche.training.templates import SupervisedTemplate
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.utils.data import random_split

from clkt.datasets import ASSIST2015
from clkt.models.sakt import SAKT
from clkt.utils import collate_fn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_dir = Path(r'C:\Users\georg\Projects\clkt\data\assist2015')

train_ratio = 0.8
batch_size = 32
dataset = ASSIST2015(50, dataset_dir)
data_loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True,
    collate_fn=collate_fn
)

# dataset.targets = DataAttribute(dataset.r_seqs, name='assist2015')
generator = torch.Generator(device=device)
train_size = int(len(dataset) * train_ratio)
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset, [train_size, test_size],
    generator=generator
)

train_dataset = AvalancheDataset(train_dataset, collate_fn=collate_fn)
# train_dataset.targets = DataAttribute(dataset.r_seqs, name='assist2015')
# train_dataset.targets_task_labels = DataAttribute([0] * len(dataset), name='assist2015_task')
test_dataset = AvalancheDataset(test_dataset, collate_fn=collate_fn)

# for data in dl:
#     print(len(data))
# # for x in DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator):
# #     print(x)
# #     break
#

bm = benchmark_from_datasets(
    train=[train_dataset],
    test=[test_dataset]
)
bm = task_incremental_benchmark(bm, reset_task_labels=False)
# bm = ni_benchmark(train_dataset, test_dataset, 10)
# bm = data_incremental_benchmark(bm, 100)


train_stream, test_stream = bm.streams.values()
# train_stream = bm.streams['train']
# print(f"{bm.train_stream.name} - len {len(bm.train_stream)}")
# print(f"{bm.test_stream.name} - len {len(bm.test_stream)}")

model = SAKT(dataset.num_q, n=50, d=50, num_attn_heads=5, dropout=0.3)
# model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()

# for train_task in train_stream:
#     for x,y,t in train_task.dataset:
#         print(x)


# for x, y in data_loader:
#     print(model(x).shape, y.shape)
cl_strategy = Naive(
    model, optimizer, criterion, train_mb_size=256, train_epochs=2,
    eval_mb_size=32, device=device)
# cl_strategy = SupervisedTemplate(model, optimizer, criterion)
results = []
for train_task in train_stream:
    # cl_strategy.eval(train_task)
    # dl = TaskBalancedDataLoader(train_task.dataset, batch_size=32, collate_fn=collate_fn)
    train_task.task_labels = list(train_task.task_labels)
    # train_task.task_labels = list(train_task.task_labels)
    # train_task.dataloader = dl
    cl_strategy.train(train_task, pin_memory=False)
    # results.append(cl_strategy.eval(test_stream))