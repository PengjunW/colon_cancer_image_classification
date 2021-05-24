"""Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - AverageMeter: compute and store the average and current value.
    - EarlyStopping: check and save the best model state
"""

import torch
from torch.utils.data import DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_mean_and_std(dataset):
    """compute the mean and std value of dataset."""

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    mean = torch.zeros(3).to(device)
    std = torch.zeros(3).to(device)
    print("==> Computing mean and std..")
    for inputs in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean().to(device)
            std[i] += inputs[:, i, :, :].std().to(device)
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


class AverageMeter(object):
    """compute and store the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """check and save the best model state"""

    def __init__(self, patience=7, path="checkpoint.pth"):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, state):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(state)

        elif val_loss > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(state)
            self.counter = 0

    def save_checkpoint(self, state):
        torch.save(state, self.path)
