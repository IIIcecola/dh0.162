
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Union, Optional
from abc import ABC, abstractmethod


# Base Loss Class
class BaseLoss(ABC, nn.Module):
  def __init__(self):
    super().__init__()
    self.name = self.__class__.__name__

  @abstractmethod
  def forward(self, pred, target):
    """
    计算loss

    Args:
      pred: (B, T, D) or (B, T) predictions
      target: (B, T, D) or (B, T) targets
    Returns:
      loss: scalar tensor or unreduced tensor
    """
    pass

  def get_config(self) -> Dict:
    """返回当前配置的字典"""
    return {}


# Standard Losses
class MSELoss(BaseLoss):
  """
  Config: 
    loss_type: "mse"
    reduction: "mean" | "sum" | "none"
  """
  def __init__(self, reduction="mean"):
    super().__init__()
    self.reduction = reduction
    self.mse_loss = nn.MSELoss(reduction=reduction)

  def forward(self, pred, target):
    return self.mse_loss(pred, target)

  def get_config(self):
    return {"reduction": self.reduction}
  
class L1Loss(BaseLoss):
  """
  Config:
    loss_type: "l1"
    reduction: "mean" | "sum" | "none"
  """
  def __init__(self, reduction="mean"):
    super().__init__()
    self.reduction = reduction

  def forward(self, pred, target):
    if self.reduction == "mean":
      return torch.mean(torch.abs(pred-target))
    elif self.reduction == "sum":
      return torch.sum(torch.abs(pred-target))
    else: 
      return torch.abs(pred-target)

  def get_config(self):
    return {"reduction": self.reduction}

class RankLoss01Range(BaseLoss):
  """
  Rank Loss（排序损失）
  约束最后一个维度（dim）上的相对高低关系，使模型输出在时间维度上的排序与标签一致
  适用于 0-1 区间的输出，对每个时间步的 136 个特征之间的相对高低关系进行约束
  """
   def __init__(self, **kwargs):
     super().__init__(**kwargs)
     self.gamma_min = kwargs.get('gamma_min', 0.05)
     self.gamma_max = kwargs.get('gamma_max', 0.3)

  def forward(self, scores, y_ture):
    """
    :param scores: 模型输出值, shape = [batch, seq_len, dim]
    :param y_ture: 标签值, shape = [batch, seq_len, dim]
    :return: Rank Loss（constant）
    """
    batch, seq_len, dim = scores.shape
    device = scores.device

    gamma = torch.std(y_ture, dim=-1, keepdim=True)
    gamma = torch.clamp(gamma, self.gamma_min, self.gamma_max)

    # 对每个（batch, seq_len）在dim维度随机采样成对索引
    idx_i = torch.randint(0, dim, (batch, seq_len), device=device)
    idx_j = torch.randint(0, dim, (batch, seq_len), device=device)

    # 根据索引提取对应的值
    batch_idx = torch.arange(batch, device=device).view(-1, 1).expend(-1, seq_len)
    seq_idx = torch.arange(seq_len, device=device).view(-1, 1).expend(batch, -1)

    # 提取对应的值[batch, seq_len]
    s_i = scores[batch_idx, seq_idx, idx_i]
    s_j = scores[batch_idx, seq_idx, idx_j]
    y_i = y_ture[batch_idx, seq_idx, idx_i]
    y_j = y_ture[batch_idx, seq_idx, idx_j]

    y_ij = torch.where(
      y_i >= y_j,
      torch.tensor(1.0, device=device),
      torch.tensor(-1.0, device=device)
    )

    delta = (s_i - s_j) * y_ij
    loss_terms = F.softplus(gamma.squeeze(-1) * delta)

    rank_loss = loss_terms.mean()

    return rank_loss
    

# combinded Loss
class CombinedLoss(BaseLoss):
  """
  组合多个loss
  Config:
    loss_type: "combined"
    losses: List[dict]

  每个loss配置：
  {
    "type": "mse" | "" | "" | ... ,
    "weight": float, # 该loss权重
    "params": dict # loss特定参数（可选）
  }
  """
  LOSS_REGISTRY = {
    "mse": MSELoss,
    "l1": L1Loss
  }

  def __init__(self, losses_config: List[Dict]):
    super().__init__()

    self.losses_list = []
    self.weights = []

    for loss_config in losses_config:
      loss_type = loss_config["type"]
      weight = loss_config.get("weight", 1.0)
      params = loss_config.get("params", {})

      if loss_type not in self.LOSS_REGISTRY:
        raise ValueError()

      loss_class = self.LOSS_REGISTRY[loss_type]
      loss_instance = loss_class(**params)

      self.losses_list.append(loss_instance)
      self.weights.append(weight)

      if isinstance(loss_instance, nn.Module):
        self.add_module(f"loss_{len(self.losses_list)-1}", loss_instance)

    print()
    for i, (loss, w) in enumerate(zip(self.losses_list, self.weights)):
      print(f" [{i}] {loss.name}: weight={w}")

  def forward(self, pred, target):
    """
    计算组合loss
    """
    total_loss = 0.0
    loss_dict = {}

    for loss, weight in zip(self.losses_list, self.weights):
      loss_value = loss(pred, target)
      if loss_value.dim() > 0:
        loss_value = loss_value.mean()
      total_loss = total_loss + weight * loss_value
      loss_dict[loss.name] = {
        "value": loss_value.item(),
        "weighted": weight * loss_value.item()
      }
    loss_dict["total"]: total_loss.item()
    return total_loss, loss_dict

  def get_config(self):
    config = {
      "loss_type": "combined",
      "losses": []
    }
    for loss, weight in zip(self.losses_list, self.weights):
      loss_config = {
        "type": loss.name,
        "weight": weight,
        "params": loss.get_coinfig()
      }
      config["losses"].append(loss_config)
    return config














