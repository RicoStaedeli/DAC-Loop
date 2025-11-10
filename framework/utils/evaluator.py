from typing import Dict
import torch
import torch.nn.functional as F
import torch.nn as nn

class Evaluator:
    def __init__(self, ce_loss: nn.Module):
        self.ce_loss = ce_loss

    @torch.inference_mode()
    def __call__(self, model, loader, device) -> Dict[str, float]:
        model.eval()
        total_loss = total_mse = total = correct = 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = self.ce_loss(logits, y)
            total_loss += loss.item() * x.size(0)

            probs = F.softmax(logits, dim=1)
            y_oh = F.one_hot(y, num_classes=probs.size(1)).float()
            total_mse += F.mse_loss(probs, y_oh, reduction="mean").item() * x.size(0)

            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        return {"loss": total_loss / total, "mse": total_mse / total, "acc": 100.0 * correct / total}