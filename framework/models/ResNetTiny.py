import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class ResNetTiny(nn.Module):
    """
    ~1M-parameter ResNet-10-ish model:
      - Same layout as your ResNet10mini (1 block per stage)
      - base_channels=27 -> ~0.99M params for num_classes=10, kd_dim=512
    """
    def __init__(self, input_channels=3, num_classes=10, kd_dim=512, base_channels=27):
        super().__init__()
        self.in_channels = base_channels

        # Keep your ImageNet-style stem (7x7 s2 + maxpool)
        self.conv1 = nn.Conv2d(input_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(base_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, base_channels,     blocks=1, stride=1)
        self.layer2 = self._make_layer(BasicBlock, base_channels * 2, blocks=1, stride=2)
        self.layer3 = self._make_layer(BasicBlock, base_channels * 4, blocks=1, stride=2)
        self.layer4 = self._make_layer(BasicBlock, base_channels * 8, blocks=1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        backbone_dim = base_channels * 8  # 27 * 8 = 216

        self.fc = nn.Linear(backbone_dim, num_classes)
        self.feat_proj = nn.Linear(backbone_dim, kd_dim)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        for s in [stride] + [1] * (blocks - 1):
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x, return_features=True):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        feat = out.flatten(1)              # (B, 216)
        logits = self.fc(feat)
        feat_kd = self.feat_proj(feat)     # (B, kd_dim)

        return (logits, feat_kd) if return_features else logits


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

