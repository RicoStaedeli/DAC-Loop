import torch
import torch.nn.functional as F
from tqdm import tqdm
from .base import EpochResult, top1_accuracy, kd_kl

class KLDivergenceKDStrategy:
    def __init__(self, ce_loss, T: float, alpha_kd: float, beta_feat: float, w_ce: float, use_mse_feat: bool):
        self.ce = ce_loss; self.T = T; self.alpha = alpha_kd; self.beta = beta_feat
        self.w_ce = w_ce; self.use_mse_feat = use_mse_feat

    def train_one_epoch(self, *, teacher, student, loader, device, optimizer, epoch, total_epochs) -> EpochResult:
        teacher.eval(); student.train()
        tot_loss = tot_acc = tot_n = 0.0

        for x, y in tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                t_logits, t_feat = teacher(x)
            s_logits, s_feat = student(x)

            loss_kd = kd_kl(s_logits, t_logits, self.T)
            loss_ce = self.ce(s_logits, y)

            loss_feat = torch.tensor(0.0, device=device)
            if self.beta > 0.0:
                s_n = F.normalize(s_feat, dim=1)
                t_n = F.normalize(t_feat, dim=1)
                loss_feat = F.mse_loss(s_n, t_n) if self.use_mse_feat else (1.0 - torch.cosine_similarity(s_n, t_n, dim=1).mean())

            loss = self.alpha * loss_kd + self.w_ce * loss_ce + self.beta * loss_feat
            loss.backward(); optimizer.step()

            with torch.no_grad():
                b = x.size(0)
                tot_n += b
                tot_loss += loss.item() * b
                tot_acc += top1_accuracy(s_logits, y).item() * b

        return EpochResult(tot_loss / tot_n, 100.0 * tot_acc / tot_n)