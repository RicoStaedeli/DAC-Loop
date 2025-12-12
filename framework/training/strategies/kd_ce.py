import torch
import torch.nn.functional as F
from tqdm import tqdm

from framework.training.strategies.base import EpochResult, top1_accuracy


class CrossEntropyKDStrategy:
    def __init__(self, ce_loss, T: float, w_soft: float, w_ce: float):
        """
        Cross-Entropy based Knowledge Distillation Strategy.
        Args:
            ce_loss: Cross-entropy loss function.
            T (float): Temperature parameter for softening logits.
            w_soft (float): Weight for the soft target loss component.
            w_ce (float): Weight for the cross-entropy loss component.
        """
        self.cross_entropy_loss = ce_loss
        self.temperature = T
        self.weight_loss_kl = w_soft
        self.weight_loss_ce = w_ce

    def train_one_epoch(self, *, teacher, student, loader, device, optimizer, epoch, total_epochs) -> EpochResult:
        teacher.eval()
        student.train()
        tot_loss = tot_acc = tot_n = 0.0

        for image, label in tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}"):
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                teacher_logits, _ = teacher(image)

            student_logits, _ = student(image)

            student_logp_temperature = F.log_softmax(student_logits / self.temperature, dim=1)
            teacher_logp_temperature = F.log_softmax(teacher_logits / self.temperature, dim=1)

            loss_kl = F.kl_div(student_logp_temperature, teacher_logp_temperature, reduction="batchmean",
                            log_target=True) * (self.temperature ** 2)

            loss_ce = self.cross_entropy_loss(student_logits, label)

            loss = self.weight_loss_kl * loss_kl + self.weight_loss_ce * loss_ce

            loss.backward()

            optimizer.step()

            with torch.no_grad():
                batch_size = image.size(0)
                tot_n += batch_size
                tot_loss += loss.item() * batch_size
                tot_acc += top1_accuracy(student_logits, label).item() * batch_size

        return EpochResult(tot_loss / tot_n, 100.0 * tot_acc / tot_n)
