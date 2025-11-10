import torch
import torch.nn.functional as F
from tqdm import tqdm
from .base import EpochResult, top1_accuracy

class CrossEntropyKDStrategy:
    def __init__(self, ce_loss, T: float, w_soft: float, w_ce: float):
        self.ce = ce_loss; self.T = T; self.w_soft = w_soft; self.w_ce = w_ce

    def train_one_epoch(self, *, teacher, student, loader, device, optimizer, epoch, total_epochs) -> EpochResult:
        teacher.eval(); student.train()
        tot_loss = tot_acc = tot_n = 0.0

        for x, y in tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                t_logits, _ = teacher(x)
            s_logits, _ = student(x)

            s_logp_T = F.log_softmax(s_logits / self.T, dim=1)
            t_logp_T = F.log_softmax(t_logits / self.T, dim=1)
            soft = F.kl_div(s_logp_T, t_logp_T, reduction="batchmean", log_target=True) * (self.T ** 2)
            ce = self.ce(s_logits, y)
            loss = self.w_soft * soft + self.w_ce * ce

            loss.backward(); optimizer.step()

            with torch.no_grad():
                b = x.size(0)
                tot_n += b
                tot_loss += loss.item() * b
                tot_acc += top1_accuracy(s_logits, y).item() * b

        return EpochResult(tot_loss / tot_n, 100.0 * tot_acc / tot_n)