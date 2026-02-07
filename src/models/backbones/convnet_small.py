import torch.nn as nn


class ConvNetSmall(nn.Module):
    """
    ConvNet-small backbone (single feature map, C=256).
    Matches the simplified backbone used in the paper.
    """

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out_channels = 256

    def forward(self, x):
        return self.body(x)

