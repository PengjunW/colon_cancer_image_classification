import torchvision.transforms as transforms


def get_train_transform(size):
    train_transform = transforms.Compose(
        [
            transforms.RandomGrayscale(),
            transforms.Resize((256, 256)),
            transforms.RandomAffine(5),
            transforms.ColorJitter(hue=0.05, saturation=0.05),
            transforms.RandomCrop(size),
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.7957, 0.6218, 0.7356], [0.1113, 0.1651, 0.1540]),
        ]
    )
    return train_transform


def get_val_transform(size):
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.8200, 0.6171, 0.7767], [0.0885, 0.1459, 0.1315]),
        ]
    )


def get_pred_transform(size):
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize([0.8264, 0.6205, 0.7782], [0.0914, 0.1500, 0.1345]),
        ]
    )


def get_transforms(
    input_size=224,
    test_size=224,
):

    transformations = {}
    transformations["train1"] = get_train_transform(input_size)
    transformations["val"] = get_val_transform(test_size)
    transformations["pred"] = get_pred_transform(test_size)
    return transformations
