# efficientnet_backbone.py
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet

class EfficientNetBackbone(nn.Module):
    def __init__(self, model_name='efficientnet-b0', pretrained=True):
        super(EfficientNetBackbone, self).__init__()
        self.model = EfficientNet.from_pretrained(model_name) if pretrained else EfficientNet.from_name(model_name)
        self.model = nn.Sequential(*list(self.model.children())[:-2])  # Remove the final layers

    def forward(self, x):
        return self.model(x)
