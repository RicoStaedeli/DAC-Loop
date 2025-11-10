from __future__ import annotations

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.optimizer import Optimizer

from framework.training.strategies.kd_ce import CrossEntropyKDStrategy
from framework.training.strategies.kd_cosine import CosineEmbeddingKDStrategy
from framework.training.strategies.kd_kldiv import KLDivergenceKDStrategy
from framework.training.strategies.kd_regressor_mse import RegressorMSEKDStrategy
from framework.training.strategies.plain import PlainSupervisedStrategy



class OptimizerFactory:
    @staticmethod
    def make(model, cfg) -> Optimizer:
        return AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)

class SchedulerFactory:
    @staticmethod
    def make(optimizer: Optimizer, cfg) -> CosineAnnealingLR | None:
        if cfg.training.scheduler == "cosine":
            return CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
        return None

# Strategy selection is imported late to avoid circulars
def build_strategy(cfg, ce_loss):

    ts = getattr(cfg.training, "teacher_student", None)
    if ts is None:
        return PlainSupervisedStrategy(ce_loss)

    kd_type = getattr(ts, "kd_type", "None")
    if kd_type == "CrossEntropyLoss":
        return CrossEntropyKDStrategy(ce_loss, T=ts.T_param, w_soft=ts.soft_target_loss_weight, w_ce=ts.ce_loss_weight)
    if kd_type == "CosineEmbeddingLoss":
        return CosineEmbeddingKDStrategy(ce_loss, w_rep=ts.hidden_rep_loss_weight, w_ce=ts.ce_loss_weight)
    if kd_type == "KLDivLoss":
        return KLDivergenceKDStrategy(ce_loss, T=ts.T_param, alpha_kd=ts.alpha_kd,
                                      beta_feat=getattr(ts, "beta_feat", 0.0), w_ce=ts.ce_loss_weight,
                                      use_mse_feat=True)
    if kd_type == "RegressorMSE":
        return RegressorMSEKDStrategy(ce_loss, T=ts.T_param, w_rep=ts.hidden_rep_loss_weight,
                                      w_ce=ts.ce_loss_weight, w_kd=ts.soft_target_loss_weight)

    return PlainSupervisedStrategy(ce_loss)