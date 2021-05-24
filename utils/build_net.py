import torch
import torchvision.models as models

from models import resnet34, resnet50

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def make_model(model_name, num_classes):

    if model_name == "resnet34":
        model = resnet34(num_classes)
        resnet = models.resnet34(pretrained=True)
    elif model_name == "resnet50":
        model = resnet50(num_classes)
        resnet = models.resnet50(pretrained=True)
    else:
        assert "input erro"

    # load the pretrained model
    new_state_dict = resnet.state_dict()
    model_state = model.state_dict()
    for k in new_state_dict.keys():
        if k in model_state.keys() and not k.startswith("fc"):
            model_state[k] = new_state_dict[k]
    model = model.to(device)
    return model
