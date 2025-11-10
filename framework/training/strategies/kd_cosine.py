import torch
import torch.nn as nn
from tqdm import tqdm
from .base import EpochResult, top1_accuracy

class CosineEmbeddingKDStrategy:
    def __init__(self, ce_loss, w_rep: float, w_ce: float):
        self.ce = ce_loss
        self.cos = nn.CosineEmbeddingLoss()
        self.w_rep = w_rep; self.w_ce = w_ce

    def train_one_epoch(self, *, teacher, student, loader, device, optimizer, epoch, total_epochs) -> EpochResult:
        teacher.eval(); student.train()
        tot_loss = tot_acc = tot_n = 0.0

        for x, y in tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                _, t_feat = teacher(x)
            s_logits, s_feat = student(x)

            rep = self.cos(s_feat, t_feat, target=torch.ones(x.size(0), device=device))
            ce = self.ce(s_logits, y)
            loss = self.w_rep * rep + self.w_ce * ce

            loss.backward(); optimizer.step()

            with torch.no_grad():
                b = x.size(0)
                tot_n += b
                tot_loss += loss.item() * b
                tot_acc += top1_accuracy(s_logits, y).item() * b

        return EpochResult(tot_loss / tot_n, 100.0 * tot_acc / tot_n)