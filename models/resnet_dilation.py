"""
Dilated ResNet-18 used as the final feature extraction stage for style images.

The `conv5_x` block of a standard ResNet-18 is replaced with dilated
convolutions (dilation=2) so that the spatial resolution of the feature
map is preserved rather than halved. This allows richer style features
across the full width of a handwriting line.
"""

import torch.nn as nn


class BasicBlock(nn.Module):
    """ResNet basic block with optional dilation support."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 dilation: int = 1, first_dilation: bool = False):
        super().__init__()

        if first_dilation:
            # First conv in a dilated stage: no dilation yet, then dilation=2
            self.residual_function = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                          padding=1, bias=False, dilation=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels * BasicBlock.expansion,
                          kernel_size=3, padding=2, bias=False, dilation=2),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )
        else:
            padding = 1 if dilation == 1 else 2
            self.residual_function = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                          padding=padding, bias=False, dilation=dilation),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels * BasicBlock.expansion,
                          kernel_size=3, padding=padding, bias=False, dilation=dilation),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

    def forward(self, x):
        return nn.functional.relu(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.conv2_x = self._make_layer(block, 64, num_block[0], stride=1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], stride=1, dilation=1)
        self.conv4_x = self._make_layer(block, 256, num_block[2], stride=1, dilation=1)
        self.conv5_x = self._make_layer(block, 512, num_block[3], stride=1, dilation=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 100)

    def _make_layer(self, block, out_channels: int, num_blocks: int,
                    stride: int = 1, dilation: int = 1) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        if dilation == 1:
            for s in strides:
                layers.append(block(self.in_channels, out_channels, s, dilation=1))
                self.in_channels = out_channels * block.expansion
        else:
            # First block: first conv without dilation, second with dilation=2
            layers.append(block(self.in_channels, out_channels, strides[0],
                                dilation=2, first_dilation=True))
            self.in_channels = out_channels * block.expansion
            for s in strides[1:]:
                layers.append(block(self.in_channels, out_channels, s, dilation=2))
                self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def resnet18() -> ResNet:
    """Return a ResNet-18 with dilated conv5_x."""
    return ResNet(BasicBlock, [2, 2, 2, 2])
