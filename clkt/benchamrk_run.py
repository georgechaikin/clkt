from functools import partial
from pathlib import Path

import numpy as np
import torch
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks import benchmark_from_datasets, task_incremental_benchmark, split_validation_random, \
    split_online_stream
from torch.utils.data.dataloader import DataLoader
from avalanche.training import Naive, EWC
from avalanche.training.supervised import JointTraining, AGEM, VAETraining, AETraining, Replay
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.utils.data import random_split
from torch.nn.functional import one_hot, binary_cross_entropy

from sklearn import metrics

from clkt.datasets import ASSIST2015, ASSIST2009
from clkt.models.sakt import SAKT
from clkt.utils import collate_fn


def eval(model, test_loader):
    with torch.no_grad():
        for data, labels in test_loader:
            model.eval()

            p = model(data).detach().cpu()
            t = labels.detach().cpu()

            auc = metrics.roc_auc_score(
                y_true=t.numpy(), y_score=p.numpy()
            )

            print(
                "Test AUC: {}"
                .format(auc)
            )


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dataset_path = Path(r'C:\Users\georg\Projects\clkt\data\assist2009\statistics.csv')

train_ratio = 0.9
split_ratio = 0.25
val_ratio = 0.5
batch_size = 256
dataset_dir = Path(r'C:/Users/georg/Projects/clkt/data/assist2009')
generator = torch.Generator(device=device)

# dataset = ASSIST2015(50, dataset_dir)
# dataset = ASSIST2009(100, dataset_path)
# data_loader = DataLoader(
#     dataset, batch_size=batch_size, shuffle=True,
#     collate_fn=collate_fn
# )

# dataset.targets = DataAttribute(dataset.r_seqs, name='assist2015')


# train_datasets = [ASSIST2009(100, dataset_dir / filename) for filename in
#                   ['algebra.csv', 'geometry.csv', 'statistics.csv', 'basic_maths.csv']]

train_datasets = [ASSIST2009(100, dataset_dir / f'train_{i}.csv') for i in range(10)]

dataset = ASSIST2009(100, dataset_dir / 'skill_builder_data.csv')

train_size = int(len(dataset) * train_ratio)
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
train_size = train_size - test_size


def get_train_datasets(train_ds, split_ratio, generator):
    split_size = int(len(train_ds) * split_ratio)
    train_datasets = []
    while len(train_ds) > split_size:
        train_ds, split_ds = random_split(train_ds, [len(train_ds)-split_size, split_size], generator=generator)
        train_datasets.append(split_ds)
    train_datasets.append(train_ds)
    return train_datasets


# train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
# test_dataset = ASSIST2009(100, dataset_dir / 'test.csv')
# train_dataset = AvalancheDataset(train_dataset, collate_fn=collate_fn)
split_ratio = 0.25
train_datasets = get_train_datasets(train_dataset, split_ratio, generator)
print(len(train_datasets))
train_datasets = [AvalancheDataset(ds, collate_fn=collate_fn) for ds in train_datasets]
# train_dataset.targets = DataAttribute(dataset.r_seqs, name='assist2015')
# train_dataset.targets_task_labels = DataAttribute([0] * len(dataset), name='assist2015_task')
# test_dataset = AvalancheDataset(test_dataset, collate_fn=collate_fn)
# train_datasets = split_validation_random(0.5, shuffle=True, dataset=train_dataset)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn=partial(collate_fn, return_task=False))
# for data in dl:
#     print(len(data))
# # for x in DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator):
# #     print(x)
# #     break
#

bm = benchmark_from_datasets(
    # train=[train_dataset, test_dataset],
    train=train_datasets,
    # test=[test_dataset]
)
bm = task_incremental_benchmark(bm)
# bm = ni_benchmark(train_dataset, test_dataset, 10)
# bm = data_incremental_benchmark(train_dataset, 100, )
# with_task_labels


# train_stream, test_stream = bm.streams.values()
# train_stream, test_stream = bm.streams['train'], bm.streams['test']
train_stream = bm.streams['train']
# print(f"{bm.train_stream.name} - len {len(bm.train_stream)}")
# print(f"{bm.test_stream.name} - len {len(bm.test_stream)}")

model = SAKT(564, n=100, d=100, num_attn_heads=5, dropout=0.2)
optimizer = Adam(model.parameters(), lr=0.001)
# criterion = CrossEntropyLoss()
criterion = binary_cross_entropy

# cl_strategy = Naive(
#     model, optimizer, criterion, train_mb_size=256, train_epochs=2,
#     eval_mb_size=32, device=device)

# cl_strategy = JointTraining(model, optimizer, criterion, train_mb_size=256, train_epochs=50,
#                             eval_mb_size=32, device=device)

cl_strategy = EWC(model, optimizer, criterion, ewc_lambda=0.1, train_mb_size=batch_size, train_epochs=10,
                  eval_mb_size=32, device=device)
# cl_strategy = Replay(model, optimizer, criterion, train_mb_size=batch_size, train_epochs=50,
#                      eval_mb_size=32, device=device)

results = []
for train_task in train_stream:
    model.train()
    train_task.task_labels = list(train_task.task_labels)
    cl_strategy.train(train_task, pin_memory=False)
    eval(model, test_loader)
    # print(test_stream.task_labels)
    # for test_task in train_stream:
    #     print(test_task.task_labels)
    #     test_task.task_labels = list(test_task.task_labels)
    # results.append(cl_strategy.eval(test_stream, pin_memory=False))

print(results)
