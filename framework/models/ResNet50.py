import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    # ResNet-50/101/152 bottleneck
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        width = out_channels  # base width before expansion

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, imagenet_stem=False, feature_dim=512):
        super().__init__()
        self.in_channels = 64

        if imagenet_stem:
            # ImageNet-style stem
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1   = nn.BatchNorm2d(64)
            self.relu  = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            # CIFAR-friendly stem
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(64)
            self.relu  = nn.ReLU(inplace=True)
            self.maxpool = nn.Identity()  # often no maxpool for small images

        # ResNet-50 layers: [3, 4, 6, 3]
        self.layer1 = self._make_layer(Bottleneck, 64,  3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        # projection used only for KD features
        out_feats = 512 * Bottleneck.expansion  # 2048
        self.kd_proj = nn.Identity()
        if feature_dim != out_feats:
            self.kd_proj = nn.Linear(out_feats, feature_dim, bias=False)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=True):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        feat = torch.flatten(out, 1)
        kd_feat = self.kd_proj(feat)  # feature_dim
        logits = self.fc(feat)
        return (logits, kd_feat) if return_features else logits