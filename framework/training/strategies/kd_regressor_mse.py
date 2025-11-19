import torch
import torch.nn.functional as F
from tqdm import tqdm

from framework.training.fitnets import FeatureRegressorFactory
from framework.training.strategies.base import EpochResult, top1_accuracy


def _ensure_regressor(student, device, s_feat, t_feat, optimizer):
    reg = getattr(student, "_fitnets_regressor", None)
    if reg is None:
        reg = FeatureRegressorFactory.make(s_feat, t_feat).to(device)
        optimizer.add_param_group({"params": reg.parameters()})
        student._fitnets_regressor = reg
    return reg


class RegressorMSEKDStrategy:
    def __init__(self, ce_loss, T: float, w_rep: float, w_ce: float, w_kd: float):
        """
        Regressor Mean Squared Error based Knowledge Distillation Strategy.
        Args:
            ce_loss: Cross-entropy loss function.
            T (float): Temperature parameter for softening logits.
            w_rep (float): Weight for the representation loss component.
            w_ce (float): Weight for the cross-entropy loss component.
            w_kd (float): Weight for the KD loss component.
        """
        self.cross_entropy_loss = ce_loss
        self.temperature = T
        self.weight_representation_loss = w_rep
        self.weight_cross_entropy_loss = w_ce
        self.weight_kd_loss = w_kd

    def train_one_epoch(self, *, teacher, student, loader, device, optimizer, epoch, total_epochs) -> EpochResult:
        teacher.eval();
        student.train()
        tot_loss = tot_acc = tot_n = 0.0

        for inputs, labels in tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                teacher_logits, teacher_features = teacher(inputs)

            student_logits, student_features = student(inputs)

            regressor = _ensure_regressor(student, device, student_features, teacher_features, optimizer)
            tgt_hw = teacher_features.shape[-2:] if teacher_features.dim() == 4 else None
            student_projected = regressor(student_features, target_hw=tgt_hw)

            mse_loss = F.mse_loss(student_projected, teacher_features)

            ce_loss = self.cross_entropy_loss(student_logits, labels)

            student_logp_temperature = F.log_softmax(student_logits / self.temperature, dim=1)
            teacher_logp_temperature = F.log_softmax(teacher_logits / self.temperature, dim=1)

            loss_kd = F.kl_div(student_logp_temperature, teacher_logp_temperature, reduction="batchmean",
                               log_target=True) * (self.temperature ** 2)

            loss = self.weight_representation_loss * mse_loss + self.weight_cross_entropy_loss * ce_loss + self.weight_kd_loss * loss_kd

            loss.backward()

            optimizer.step()

            with torch.no_grad():
                b = inputs.size(0)
                tot_n += b
                tot_loss += loss.item() * b
                tot_acc += top1_accuracy(student_logits, labels).item() * b

        return EpochResult(tot_loss / tot_n, 100.0 * tot_acc / tot_n)
