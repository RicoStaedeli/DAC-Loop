import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFeatureRegressor(nn.Module):
    def __init__(self, in_ch, out_ch):
        """
        A convolutional feature regressor that maps features from in_ch to out_ch using a 1x1 convolution,
        followed by batch normalization and ReLU activation.
        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )

    def forward(self, x, target_hw=None):
        y = self.net(x)
        if target_hw is not None and (y.shape[-2:] != target_hw):
            y = F.adaptive_avg_pool2d(y, target_hw)
        return y


class LinearFeatureRegressor(nn.Module):
    def __init__(self, in_dim, out_dim):
        """
        A linear feature regressor that maps features from in_dim to out_dim using a linear layer,
        followed by ReLU activation.
        Args:
            in_dim (int): Input feature dimension.
            out_dim (int): Output feature dimension.
        """
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(True))

    def forward(self, x, target_hw=None):
        return self.net(x)


class FeatureRegressorFactory:
    @staticmethod
    def make(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> nn.Module:
        """
        Factory method to create an appropriate feature regressor based on the dimensions of student and teacher features.
        Args:
            student_feat (torch.Tensor): Feature tensor from the student model.
            teacher_feat (torch.Tensor): Feature tensor from the teacher model.
        Returns:
            nn.Module: An instance of ConvFeatureRegressor or LinearFeatureRegressor.
        Raises:
            RuntimeError: If the feature shapes are unsupported.
        """
        if student_feat.dim() == 4 and teacher_feat.dim() == 4:
            return ConvFeatureRegressor(student_feat.size(1), teacher_feat.size(1))

        if student_feat.dim() == 2 and teacher_feat.dim() == 2:
            return LinearFeatureRegressor(student_feat.size(1), teacher_feat.size(1))

        raise RuntimeError(f"Unsupported feature shapes: s={tuple(student_feat.shape)}, t={tuple(teacher_feat.shape)}")
