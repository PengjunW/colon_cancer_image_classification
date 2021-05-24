import pandas as pd
from tqdm.notebook import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from criterions import focal_loss
from utils import (
    AverageMeter,
    accuracy,
    get_transforms,
    TrianValDataset,
    EarlyStopping,
    make_model,
)

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
checkpoint_dir = "checkpoint/resnet34/checkpoint.path"


def main():

    # hyperparameters
    input_size = 224
    test_size = 224
    num_epoch = 40
    batch_size = 10
    lr = 0.0001
    weight_decay = 0.0005
    class_weight = [4, 2.5, 6, 12]
    alpha = [0.25, 0.33, 0.33, 0.09]
    gamma = 2

    # load train data
    train_data = pd.read_csv("dataset/train.csv")
    train_data["path"] = train_data["Id"].apply(
        lambda x: "dataset/train/" + str(x) + ".png"
    )

    # encoder label
    enc = LabelEncoder()
    enc.fit(["Connective", "Cancer", "Immune", "Normal"])
    train_data["Type"] = enc.transform(train_data["Type"].values)

    # custormed transform: resize image
    transform = get_transforms(input_size=input_size, test_size=test_size)

    # split the train data
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    ss.split(train_data["Id"].index.values, train_data["Type"].values)
    for train_index, test_index in ss.split(
        train_data["Id"].index.values, train_data["Type"].values
    ):
        train_idx = train_index
        val_idx = test_index

    # calculate weight of different class at trainset and use the WeightedRandomSampler to sovle the imblanced problem
    trainset = TrianValDataset(train_data.iloc[train_idx], transform=transform["train"])
    class_weights = torch.tensor(class_weight)
    sample_weights = [0] * len(trainset)
    for idx, (data, label) in enumerate(trainset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    # dataloader
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    train_loader = DataLoader(
        trainset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True
    )
    valset = TrianValDataset(train_data.iloc[val_idx], transform=transform["val"])
    val_loader = DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # make model (resnet34 as baseline)
    model = make_model("resnet34", num_classes=4)

    # define loss function and optimizer
    criterion = focal_loss(alpha=alpha, gamma=gamma, num_classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=3, verbose=False
    )

    # save checkpoint
    early_stopping = EarlyStopping(path=checkpoint_dir)

    for epoch in range(num_epoch):
        print(
            "\nEpoch: [%d | %d] LR: %f"
            % (epoch + 1, num_epoch, optimizer.param_groups[0]["lr"])
        )

        train_loss, train_acc, train_2 = train(
            train_loader, model, criterion, optimizer
        )
        val_loss, val_acc, val_2 = val(val_loader, model, criterion)
        scheduler.step(val_loss)

        print(
            "train_loss:%f, val_loss:%f, train_acc:%f, train_2:%f, val_acc:%f, val_2:%f"
            % (train_loss, val_loss, train_acc, train_2, val_acc, val_2)
        )

        state = {
            "model": model.state_dict(),
            "acc": val_acc,
            "epoch": num_epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_loss,
        }

        early_stopping(val_loss, state)
        if early_stopping.early_stop:
            print("Early Stopping")
            print(f"Best loss: {early_stopping.best_score}")
            break


def train(train_loader, model, criterion, optimizer):
    # switch to train1 mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    for inputs, targets in tqdm(train_loader, leave=False):

        inputs = inputs.to(device)
        targets = targets.to(device)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec2 = accuracy(outputs, targets, topk=(1, 2))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top2.update(prec2.item(), inputs.size(0))

        # compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (losses.avg, top1.avg, top2.avg)


def val(val_loader, model, criterion):

    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    for inputs, targets in tqdm(val_loader, leave=False):

        inputs = inputs.to(device)
        targets = targets.to(device)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec2 = accuracy(outputs, targets, topk=(1, 2))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top2.update(prec2.item(), inputs.size(0))

    return (losses.avg, top1.avg, top2.avg)


if __name__ == "__main__":
    main()
