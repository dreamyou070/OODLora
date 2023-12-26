import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from torchvision.models.resnet import ResNet, Bottleneck
import torchvision
import pdb


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Decoder_Teacher(nn.Module):
    def __init__(self, base_model, base_quant_layer) -> None:
        super().__init__()
        self.quant_layer = base_quant_layer
        self.model = base_model

    def forward(self, x):
        x = self.quant_layer(x)
        x = self.model(x)
        return x

class Decoder_Student(nn.Module):
    def __init__(self, base_model, base_quant_layer) -> None:
        super().__init__()
        self.quant_layer = base_quant_layer
        self.model = base_model



    def forward(self, x):
        x = self.quant_layer(x)
        x = self.model(x)
        return x


class Encoder_Teacher(nn.Module):
    def __init__(self, base_model, base_quant_layer) -> None:
        super().__init__()
        self.quant_layer = base_quant_layer
        self.model = base_model

    def forward(self, x):
        x = self.model(x)
        x = self.quant_layer(x)
        return x


class Encoder_Student(nn.Module):
    def __init__(self, base_model, base_quant_layer) -> None:
        super().__init__()
        self.quant_layer = base_quant_layer
        self.model = base_model

    def forward(self, x):
        x = self.model(x)
        x = self.quant_layer(x)
        return x