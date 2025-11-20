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

class ResNet10mini(nn.Module):
    """
    A lighter ResNet-10 style network with ~2M parameters (for ~100 classes)
    by reducing the channel widths.

    base_channels=40  -> stages (40, 80, 160, 320)
    """

    def __init__(self, num_classes=100, base_channels=40, block=BasicBlock):
        super().__init__()

        self.in_channels = base_channels

        # Stem: 3x3 or 7x7 both fine – here 7x7 like standard ResNet
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=base_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Four stages, 1 block each (ResNet-10 style)
        self.layer1 = self._make_layer(block, base_channels,  blocks=1, stride=1)  # 40
        self.layer2 = self._make_layer(block, base_channels * 2, blocks=1, stride=2)  # 80
        self.layer3 = self._make_layer(block, base_channels * 4, blocks=1, stride=2)  # 160
        self.layer4 = self._make_layer(block, base_channels * 8, blocks=1, stride=2)  # 320

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        final_feat_dim = base_channels * 8  # 320 if base_channels=40
        self.fc = nn.Linear(final_feat_dim, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        """
        Make one ResNet stage.

        block: your BasicBlock(in_channels, out_channels, stride=1)
        blocks: number of BasicBlocks in this stage
        stride: stride of the first block in the stage
        """
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, stride=s))
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
        feat = out.view(out.size(0), -1)  # N, base_channels*8 (e.g. 320)
        logits = self.fc(feat)
        return (logits, feat) if return_features else logits