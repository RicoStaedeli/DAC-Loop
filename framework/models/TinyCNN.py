import torch.nn as nn
import torch.nn.functional as F


class TinyCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, width=16, kd_dim=512):
        super().__init__()
        c1, c2, c3 = width, width * 2, width * 4
        self.c3 = c3
        self.kd_dim = kd_dim

        self.conv1 = nn.Conv2d(input_channels, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # classification head uses the raw pooled feature
        self.fc = nn.Linear(c3, num_classes)
        # projection head for distillation feature, maps c3 -> 512 to match the teacher
        self.proj = nn.Linear(c3, kd_dim, bias=False)

    def forward(self, x, return_features=True):
        x = self.pool(F.relu(self.conv1(x)))  # 32 -> 16
        x = self.pool(F.relu(self.conv2(x)))  # 16 -> 8
        x = F.relu(self.conv3(x))  # spatial 8
        feat_raw = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)  # [N, c3]
        logits = self.fc(feat_raw)
        if return_features:
            feat_kd = self.proj(feat_raw)  # [N, 512] to match teacher
            return logits, feat_kd
        return logits
