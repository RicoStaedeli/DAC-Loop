import torch
import torch.nn as nn
from tqdm import tqdm

from framework.training.strategies.base import EpochResult, top1_accuracy


class CosineEmbeddingKDStrategy:
    def __init__(self, ce_loss, w_rep: float, w_ce: float):
        """
        Cosine Embedding based Knowledge Distillation Strategy.
        Args:
            ce_loss: Cross-entropy loss function.
            w_rep (float): Weight for the representation loss component.
            w_ce (float): Weight for the cross-entropy loss component.
        """
        self.cross_entropy_loss = ce_loss
        self.cosine_embedding_loss = nn.CosineEmbeddingLoss()
        self.weight_representation_loss = w_rep
        self.weight_cross_entropy_loss = w_ce

    def train_one_epoch(self, *, teacher, student, loader, device, optimizer, epoch, total_epochs) -> EpochResult:
        teacher.eval()
        student.train()
        tot_loss = tot_acc = tot_n = 0.0

        for inputs, labels in tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                _, teacher_features = teacher(inputs)

            student_logits, student_features = student(inputs)

            loss_cosine_embedding = self.cosine_embedding_loss(student_features, teacher_features,
                                                               target=torch.ones(inputs.size(0), device=device))

            loss_cross_entropy = self.cross_entropy_loss(student_logits, labels)

            loss = self.weight_representation_loss * loss_cosine_embedding + self.weight_cross_entropy_loss * loss_cross_entropy

            loss.backward()

            optimizer.step()

            with torch.no_grad():
                batch_size = inputs.size(0)
                tot_n += batch_size
                tot_loss += loss.item() * batch_size
                tot_acc += top1_accuracy(student_logits, labels).item() * batch_size

        return EpochResult(tot_loss / tot_n, 100.0 * tot_acc / tot_n)
