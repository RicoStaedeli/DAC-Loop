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
        """
        Creates an AdamW optimizer for the given model based on the provided configuration.
        Args:
            model: The model whose parameters will be optimized.
            cfg: Configuration object containing training parameters.
        Returns:
            Optimizer: An instance of AdamW optimizer.
        """
        return AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)


class SchedulerFactory:
    @staticmethod
    def make(optimizer: Optimizer, cfg) -> CosineAnnealingLR | None:
        """
        Creates a learning rate scheduler based on the provided configuration.
        Args:
            optimizer (Optimizer): The optimizer for which the scheduler will be created.
            cfg: Configuration object containing training parameters.
        Returns:
            CosineAnnealingLR | None: An instance of CosineAnnealingLR if specified in the configuration, otherwise None.
        """
        if cfg.training.scheduler == "cosine":
            return CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
        return None


# Strategy selection is imported late to avoid circulars
def build_strategy(cfg, ce_loss):
    """
    Builds a training strategy based on the provided configuration.
    Args:
        cfg: Configuration object containing training parameters.
        ce_loss: Cross-entropy loss function.
    Returns:
        An instance of a training strategy.
    """
    ts = getattr(cfg.training, "teacher_student", None)
    if ts is None:
        return PlainSupervisedStrategy(ce_loss)

    kd_type = getattr(ts, "kd_type", "None")

    if kd_type == "CrossEntropyLoss":
        return CrossEntropyKDStrategy(ce_loss=ce_loss,
                                      T=ts.T_param,
                                      w_soft=ts.soft_target_loss_weight,
                                      w_ce=ts.ce_loss_weight)

    if kd_type == "CosineEmbeddingLoss":
        return CosineEmbeddingKDStrategy(ce_loss=ce_loss,
                                         w_rep=ts.hidden_rep_loss_weight,
                                         w_ce=ts.ce_loss_weight)

    if kd_type == "KLDivLoss":
        return KLDivergenceKDStrategy(ce_loss=ce_loss,
                                      T=ts.T_param,
                                      w_soft=ts.soft_target_loss_weight,
                                      beta_feat=getattr(ts, "beta_feat", 0.0),
                                      w_ce=ts.ce_loss_weight,
                                      use_mse_feat=True)
    if kd_type == "RegressorMSE":
        return RegressorMSEKDStrategy(ce_loss=ce_loss,
                                      T=ts.T_param,
                                      w_rep=ts.hidden_rep_loss_weight,
                                      w_ce=ts.ce_loss_weight,
                                      w_kd=ts.soft_target_loss_weight)

    return PlainSupervisedStrategy(ce_loss)
