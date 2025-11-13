import torch
import torch.nn.functional as F
from tqdm import tqdm
from framework.training.strategies.base import EpochResult, top1_accuracy

class KLDivergenceKDStrategy:
    def __init__(self, ce_loss, T: float, alpha_kd: float, beta_feat: float, w_ce: float, use_mse_feat: bool = True):
        """
        Kullback-Leibler Divergence based Knowledge Distillation Strategy.
        Args:
            ce_loss: Cross-entropy loss function.
            T (float): Temperature parameter for softening logits.
            alpha_kd (float): Weight for the KD loss component.
            beta_feat (float): Weight for the feature matching loss component.
            w_ce (float): Weight for the cross-entropy loss component.
            use_mse_feat (bool): If True, use Mean Squared Error for feature loss; otherwise, use Cosine Similarity.
        """
        self.cross_entropy_loss = ce_loss
        self.temperature = T
        self.alpha = alpha_kd
        self.beta = beta_feat
        self.weight_cross_entropy_loss = w_ce
        self.use_mse_feat = use_mse_feat

    def train_one_epoch(self, *, teacher, student, loader, device, optimizer, epoch, total_epochs) -> EpochResult:
        teacher.eval(); student.train()
        tot_loss = tot_acc = tot_n = 0.0

        for inputs, labels in tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}"):

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                teacher_logits, teacher_features = teacher(inputs)

            student_logits, student_features = student(inputs)

            student_logp_temperature = F.log_softmax(student_logits / self.temperature, dim=1)
            teacher_logp_temperature = F.log_softmax(teacher_logits / self.temperature, dim=1)

            loss_kd = F.kl_div(student_logp_temperature, teacher_logp_temperature, reduction="batchmean",
                            log_target=True) * (self.temperature ** 2)

            loss_ce = self.cross_entropy_loss(student_logits, labels)

            loss_features = torch.tensor(0.0, device=device)

            if self.beta > 0.0:
                student_features_normalized = F.normalize(student_features, dim=1)
                teacher_features_normalized = F.normalize(teacher_features, dim=1)

                if self.use_mse_feat:
                    loss_features = F.mse_loss(student_features_normalized, teacher_features_normalized)
                else:
                    loss_features = (1.0 - torch.cosine_similarity(student_features_normalized, teacher_features_normalized, dim=1).mean())

            loss = self.alpha * loss_kd + self.weight_cross_entropy_loss * loss_ce + self.beta * loss_features

            loss.backward()

            optimizer.step()

            with torch.no_grad():
                batch_size = inputs.size(0)
                tot_n += batch_size
                tot_loss += loss.item() * batch_size
                tot_acc += top1_accuracy(student_logits, labels).item() * batch_size

        return EpochResult(tot_loss / tot_n, 100.0 * tot_acc / tot_n)