import torch.nn as nn


class BasicBlock(nn.Module):
    """
    A basic residual block used in ResNet architectures.

    This block applies two 3×3 convolutional layers with batch normalization
    and ReLU activation, along with an optional shortcut connection to handle
    changes in input/output dimensions or spatial resolution.

    Structure:
        Input → Conv3×3 → BN → ReLU → Conv3×3 → BN → (Add Shortcut) → ReLU → Output

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        stride (int, optional): Stride for the first convolution layer.
            If stride > 1, downsampling is applied to reduce spatial dimensions.
            Default: 1.

    Components:
        conv1: 3×3 convolution with the specified stride, followed by batch normalization and ReLU.
        conv2: 3×3 convolution with stride 1, followed by batch normalization.
        shortcut: Identity mapping by default, or a 1×1 convolution (with BN)
                  to match spatial or channel dimensions when needed.

    Forward pass:
        1. Apply the first conv–BN–ReLU sequence.
        2. Apply the second conv–BN sequence.
        3. Add the input (possibly transformed by the shortcut) to the output (residual connection).
        4. Apply ReLU activation to the summed output.

    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet10(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, feature_dim=512):
        super(ResNet10, self).__init__()
        self.in_channels = 64
        # keep same stem as your ResNet18
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # one BasicBlock per stage
        self.layer1 = self._make_layer(BasicBlock, 64, 1, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 1, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 1, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        # projection used only for KD features
        self.kd_proj = nn.Identity()
        if feature_dim != 512:
            self.kd_proj = nn.Linear(512, feature_dim, bias=False)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x, return_features=True):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        feat = out.view(out.size(0), -1)  # N, 512
        kd_feat = self.kd_proj(feat)
        logits = self.fc(feat)
        return (logits, kd_feat) if return_features else logits
