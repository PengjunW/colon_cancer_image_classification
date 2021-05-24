import numpy as np
import pandas as pd

import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from utils import get_transforms, TestDataset, resnet34

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
checkpoint_dir = "checkpoint/resnet34/"
submission_dir = "checkpoint/"


def main():
    # hyperparameters
    input_size = 224
    test_size = 224
    batch_size = 10
    tta = 10

    test_label = pd.read_csv("dataset/example.csv")
    test_label["path"] = test_label["Id"].apply(
        lambda x: "dataset/test/" + str(x) + ".png"
    )

    transform = get_transforms(input_size=input_size, test_size=test_size)

    test_loader = DataLoader(
        TestDataset(test_label, transform=transform["pred"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = resnet34(4).to(device)
    model.load_state_dict(torch.load(checkpoint_dir)["model"])
    test_pred = predict(test_loader, model, tta=tta)
    test_csv = pd.DataFrame()
    test_csv[0] = list(range(1, 401))
    test_csv[1] = np.argmax(test_pred, 1)

    enc = LabelEncoder()
    enc.fit(["Connective", "Cancer", "Immune", "Normal"])
    prediction = enc.inverse_transform(test_csv[1])
    with open(submission_dir + ".csv", "w") as f:
        f.write("Id,Type\n")
        for i, y in enumerate(prediction):
            f.write("{},{}\n".format(10001 + i, y))


def predict(test_loader, model, tta=1):
    # test time augmentation
    # switch to evaluate mode
    model.eval()

    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                input = input.to(device)
                # compute output
                output = model(input)
                output = output.data.cpu().numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)

        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta


if __name__ == "__main__":
    main()
