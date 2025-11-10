import torch
import torch.nn.functional as F
from tqdm import tqdm
from framework.training.strategies.base import EpochResult, top1_accuracy, kd_kl
from framework.training.fitnets import FeatureRegressorFactory


def _ensure_regressor(student, device, s_feat, t_feat, optimizer):
    reg = getattr(student, "_fitnets_regressor", None)
    if reg is None:
        reg = FeatureRegressorFactory.make(s_feat, t_feat).to(device)
        optimizer.add_param_group({"params": reg.parameters()})
        student._fitnets_regressor = reg
    return reg


class RegressorMSEKDStrategy:
    def __init__(self, ce_loss, T: float, w_rep: float, w_ce: float, w_kd: float):
        self.ce = ce_loss; self.T = T; self.w_rep = w_rep; self.w_ce = w_ce; self.w_kd = w_kd

    def train_one_epoch(self, *, teacher, student, loader, device, optimizer, epoch, total_epochs) -> EpochResult:
        teacher.eval(); student.train()
        tot_loss = tot_acc = tot_n = 0.0

        for x, y in tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                t_logits, t_feat = teacher(x)
            s_logits, s_feat = student(x)

            reg = _ensure_regressor(student, device, s_feat, t_feat, optimizer)
            tgt_hw = t_feat.shape[-2:] if t_feat.dim() == 4 else None
            s_proj = reg(s_feat, target_hw=tgt_hw)

            rep = F.mse_loss(s_proj, t_feat)
            ce = self.ce(s_logits, y)
            kd = kd_kl(s_logits, t_logits, self.T)
            loss = self.w_rep * rep + self.w_ce * ce + self.w_kd * kd

            loss.backward(); optimizer.step()

            with torch.no_grad():
                b = x.size(0)
                tot_n += b
                tot_loss += loss.item() * b
                tot_acc += top1_accuracy(s_logits, y).item() * b

        return EpochResult(tot_loss / tot_n, 100.0 * tot_acc / tot_n)