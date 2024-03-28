from typing import Dict

import torch
from sklearn import metrics
from torch.utils.data import DataLoader


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

            p = eval_model(data).detach().cpu()
            t = labels.detach().cpu()

            acc = metrics.accuracy_score(y_true=t.numpy(), y_pred=p.numpy() > threshold)
            auc = metrics.roc_auc_score(y_true=t.numpy(), y_score=p.numpy())

            return {'acc': acc, 'auc': auc}
