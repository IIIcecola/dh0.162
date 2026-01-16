from torch.utils.tensorboard import SummaryWriter
import time
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import math
import argparse
from omegaconf import Omegaconf
import os

from ModelDecoder import TransformerStackedDecoder
from AudioDataset import AudioDataset
from losses import LossManager

from transformers import Wav2Vec2Processor, Wav2Vec2Model


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
  def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1):
    self.warmup_steps = warmup_steps
    self.total_steps = total_steps
    self.min_lr = min_lr
    super().__init__(optimizer, last_epoch)

  def get_lr(self):
    step = self.last_epoch + 1
    lrs = []

    for base_lr in self.base_lrs:
      if step < self.warmup_steps:
        # lr = base_lr * step / max(1, self.warmup_steps)
        # 余弦warmup：从0平滑增长到base_lr（替代原线性增长）
        progress = step / self.warmup_steps  # 0~1
        cosine_warmup = 0.5 * (1 - math.cos(math.pi * progress))  # 0~1
        lr = base_lr * cosine_warmup
      else:
        progress = (step-self.warmup_steps) / max(1, self.total_steps-self.warmup_steps)
        cosine = 0.5 * (1+math.cos(math.pi*progress))
        lr = self.min_lr + (base_lr-self.min_lr)*cosine
      lrs.append(lr)
    return lrs

def train(model, dataloader, optimizer, scheduler, loss_manager, device, writer, config):
  """
  :param loss_manager: 损失函数管理器
  :param writer: tensorboard的writer参数
  :param module_loss_weight: 多源数据loss权重
  """
  
  model.train()
  # criterion.reduction = "none" # 改为"none"以获取每个样本的原始loss
  dataset_module_names = config.dataset_module_names
  epochs = config.epochs
  
  for epoch in range(epoches):
    total_loss = 0.0
    module_raw_loss = {m: 0.0 for m in loss_weight_map.keys()}
    module_weighted_loss = {m: 0.0 for m in loss_weight_map.keys()}
    module_sample_count = {m: 0 for m in loss_weight_map.keys()}
    
    for step, (audio_feat, target, module_labels, loss_weights) in enumerate(dataloader):
      # audio_feat: (B, 249, 768)
      # target: (B, 125, 136)
      audio_feat = audio_feat.to(device)
      target = target.to(device)
      optimizer.zero_grad()
      output = model(audio_feat)
      # 计算每个样本的原始loss（保留batch维度）
      sample_raw_loss = criterion(output, target).mean(dim=(1, 2))
      # 应用模块loss权重
      sample_weighted_loss = []
      for idx, (loss, module, weight) in enumerate(zip(sample_raw_loss, module_labels, loss_weights)):
          weighted_loss = loss * weight
          sample_weighted_loss.append(weighted_loss)
          # 累计每个模块的原始/加权loss和样本数
          module_raw_loss[module] += loss.item()
          module_weighted_loss[module] += weighted_loss.item()
          module_sample_count[module] += 1
      # 计算batch总损失并反向传播
      batch_loss = torch.stack(sample_weighted_loss).mean()  # 也可使用sum()
      batch_loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()
      scheduler.step()
      # 日志记录（新增模块loss监控）
      total_loss += batch_loss.item()
      global_step = epoch * len(dataloader) + step
      # 总loss
      writer.add_scalar('Train/Step Loss', batch_loss.item(), global_step)
      writer.flush()
      # 各模块原始loss
      for module in dataset_module_names:
          if module_sample_count[module] > 0:
              avg_raw = module_raw_loss[module] / module_sample_count[module]
              writer.add_scalar(f'Train/Step_{module}_Raw_Loss', avg_raw, global_step)
              avg_weighted = module_weighted_loss[module] / module_sample_count[module]
              writer.add_scalar(f'Train/Step_{module}_Weighted_Loss', avg_weighted, global_step)
      writer.flush()
      # 学习率
      lr = scheduler.get_last_lr()[0]
      writer.add_scalar('Train/Learning Rate', lr, global_step)
      writer.flush()
      # 中间日志
      if step % 20 == 0:
        print(
          f"[Epoch {epoch+1}/{epoches}]"
          f"Step {step}/{len(dataloader)}"
          f"Loss: {loss.item():.6f}"
          f"LR: {lr:.8f}"
        )
    # Epoch级日志（解耦可视化）
    avg_total_loss = total_loss / len(dataloader)
    writer.add_scalar('Train/Epoch_Avg_Total_Loss', avg_total_loss, epoch)
    writer.flush()
    # 各模块Epoch级原始/加权loss
    for module in dataset_module_names:
        if module_sample_count[module] > 0:
            epoch_avg_raw = module_raw_loss[module] / module_sample_count[module]
            epoch_avg_weighted = module_weighted_loss[module] / module_sample_count[module]
            writer.add_scalar(f'Train/Epoch_{module}_Raw_Loss', epoch_avg_raw, epoch)
            writer.add_scalar(f'Train/Epoch_{module}_Weighted_Loss', epoch_avg_weighted, epoch)
            print(f"  - {module}: Raw Loss={epoch_avg_raw:.6f}, Weighted Loss={epoch_avg_weighted:.6f}")
    writer.flush()
    print(f"==== Epoch {epoch+1} Avg Loss: {avg_total_loss:.6f} ====")  
  writer.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, default="config.yaml", help="")
  parser.add_argument("--device", type=str, help="")
  parser.add_argument("--batch_size", type=int, help="")
  parser.add_argument("--epochs", type=int, help="")
  parser.add_argument("--save_path", type=str, help="")
  args = parser.parse_args()

  config = OmegaConf.load(args.config)

  config.device = args.device
  config.dataset.batch_size = args.batch_size
  config.training.epochs = args.epochs
  config.save_path = args.save_path

  device = torch.device(config.device if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")
  
  os.makedirs(os.path.dirname(config.save_path), exists_ok=True)

  processor = None
  wav2vec2_model = None
  if config.dataset.use_processor:
    processor = Wav2Vec2Peocessor.from_pretrained(config.wav2vec2.path)
    wav2vec2_model = Wav2Vec2Model.from_pretrained(config.wav2vec2.path)

  # 初始化多模块数据集（替换原单路径dataset）
  dataset = AudioDataset(
    processor=processor,
    model=wav2vec2_model,
    config=config.dataset
  )
  # 创建加权采样器
  sampler = WeightedRandomSampler(
      weights=dataset.sample_weights,
      num_samples=config.dataloader.sampler.get('oversample_factor', 1) * len(dataset),
      replacement=config.dataloader.sampler.replacement
  )
  # 创建DataLoader（使用采样器，关闭shuffle）
  dataloader = DataLoader(
      dataset,
      batch_size=config.dataset.batch_size,
      sampler=sampler,
      shuffle=config.dataloader.shuffle,
      num_workers=config.dataset.num_workers,
      pin_memory=config.dataset.pin_memory
  )

  decoderModel = TransformerStackedDecoder(
    input_dim=config.model.input_dim,
    output_dim=config.model.output_dim,
    num_layers=config.model.num_layers,
    num_heads=config.model.num_heads,
    ff_dim=config.model.ff_dim,
    dropout=config.model.dropout
  ).to(device)

  # 创建TensorBoard日志目录
  timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
  log_dir = os.path.join(os.path.dirname(config.save_path), "tensorboard_logs", timestamp)
  os.makedirs(log_dir, exist_ok=True)
  writer = SummaryWriter(log_dir=log_dir)
  
  optimizer = AdamW(
    decoderModel.parameters(),
    lr=config.training.optimizer.lr,
    weight_decay=config.training.optimizer.weight_decay
  )

  steps_per_epoch = len(dataloader)
  total_steps = steps_per_epoch * config.training.epochs
  if config.training.scheduler.get('warmup_steps', None) is not None:
    warmup_steps = config.training.scheduler.warmup_steps
  else:
    warmup_steps = int(config.training.scheduler.warmup_ratio * total_steps)

  scheduler = WarmupCosineScheduler(
    optimizer,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    min_lr=config.training.scheduler.min_lr
  )

  criterion = nn.MSELoss()
  
  try:
    train(
      model=decoderModel,
      dataloader=dataloader,
      optimizer=optimizer,
      scheduler=scheduler,
      criterion=criterion,
      device=device,
      writer=writer, # tensorboard
      config=config.training
    )
  except KeyboardInterrupt:
    print("\n训练被手动中断，正在保存日志...")
    writer.flush()  # 最后一次强制刷新
    writer.close()  # 关闭writer
  finally:
    if 'writer' in locals():
        writer.close()

  torch.save(decoderModel.state_dict(), config.save_path)
  print(f"Model saved to {config.save_path}")

  config_save_path = os.path.splitext(config.save_path)[0] + "_config.yaml"
  with open(config_save_path, "w") as f:
    OmegaConf.save(config, f)
  print(f"Configuration saved to {config_save_path}")
