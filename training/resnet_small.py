import torch
import torchvision.models as models
import torch.nn as nn

class Resnet50SmallDim(nn.Module):
    def __init__(self, output_dim = 256):
        super().__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.reduce = nn.Linear(2048, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.reduce(x)
        return x