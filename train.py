from torch.utils.tensorboard import SummaryWriter
import time
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data import DataLoader
import torch.optim import AdamW
import math
import argparse
from omegaconf import Omegaconf
import os

from ModelDecoder import TransformerStackedDecoder
from AudioDataset import AudioDataset

from transformers import Wav2Vec2Peocessor, Wav2Vec2Model


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
        lr = base_lr * step / max(1, self.warmup_steps)
      else:
        progress = (step-self.warmup_steps) / max(1, self.total_steps-self.warmup_steps)
        cosine = 0.5 * (1+math.cos(math.pi*progress))
        lr = self.min_lr + (base_lr-self.min_lr)*cosine
      lrs.append(lr)
    return lrs

# 新增参数writer
def train(model, dataloader, optimizer, scheduler, criterion, device, epochs, writer):
  model.train()

  for epoch in range(epoches):
    total_loss = 0.0
    for step, (audio_feat, target) in enumerate(dataloader):
      # audio_feat: (B, 249, 768)
      # target: (B, 125, 136)
  
      audio_feat = audio_feat.to(device)
      target = target.to(device)
  
      optimizer.zero_grad()
  
      output = model(audio_feat)
      loss = criterion(output, target)
  
      loss.backward()
      optimizer.step()
      scheduler.step()
  
      total_loss += loss.item()

      # 新增
      # 记录step级别的损失和学习率
      global_step = epoch * len(dataloader) + step
      writer.add_scalar('Train/Step Loss', loss.item(), global_step)
      writer.add_scalar('Train/Learning Rate', lr, global_step)
  
      if step % 20 == 0:
        lr = scheduler.get_last_lr()[0]
        print(
          f"[Epoch {epoch+1}/{epoches}]"
          f"Step {step}/{len(dataloader)}"
          f"Loss: {loss.item():.6f}"
          f"LR: {lr:.8f}"
        )
      avg_loss = total_loss / len(dataloader)
      # 新增
      # 记录epoch级别的平均损失
      writer.add_scalar('Train/Epoch Avg Loss', avg_loss, epoch)
      print(f"==== Epoch {epoch+1} Avg Loss: {avg_loss:.6f} ====")
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

  # 新增
  # 创建TensorBoard日志目录
  timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
  log_dir = os.path.join(os.path.dirname(config.save_path), "tensorboard_logs", timestamp)
  os.makedirs(log_dir, exist_ok=True)
  writer = SummaryWriter(log_dir=log_dir)

  processor = None
  wav2vec2_model = None
  if config.dataset.use_processor:
    processor = Wav2Vec2Peocessor.from_pretrained(config.wav2vec2.path)
    wav2vec2_model = Wav2Vec2Model.from_pretrained(config.wav2vec2.path)

  dataset = AudioDataset(
    processor=processor,
    model=wav2vec2_model,
    cache_path=config.dataset.cache_path
  )

  dataset.generateSample(
    seconds=config.dataset.seconds,
    load_flag=config.dataset.load_flag
  )

  dataloader = DataLoader(
    dataset,
    batch_size=config.dataset.batch_size,
    shuffle=config.dataset.shuffle,
    num_workers=config.dataset.num_workers,
    pin_memory=config.dataset.pin_memory
  )

  decoderModel = TransformerStackDecoder(
    input_dim=config.model.input_dim,
    output_dim=config.model.output_dim,
    num_layers=config.model.num_layers,
    num_heads=config.model.num_heads,
    ff_dim=config.model.ff_dim,
    dropout=config.model.dropout
  ).to(device)

  optimizer = AdamW(
    decoderModel.parameters(),
    lr=config.optimizer.lr,
    weight_decay=config.optimizer.weight_decay
  )

  epoches = config.training.epochs
  steps_per_epoch = len(dataloader)
  total_steps = steps_per_epoch * epochs
  warmup_steps = int(config.scheduler.warmup_ratio * total_steps)

  scheduler = WarmupCosineScheduler(
    optimizer,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    min_lr=config.scheduler.min_lr
  )

  criterion = nn.MSELoss()

  train(
    model=decoderModel,
    dataloader=dataloader,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    device=device,
    epochs=epochs,
    writer=writer # 新增参数
  )

  torch.save(decoderModel.state_dict(), config.save_path)
  print(f"Model saved to {config.save_path}")

  config_save_path = os.path.splitext(config.save_path)[0] + "_config.yaml"
  with open(config_save_path, "w") as f:
    OmegaConf.save(config, f)
  print(f"Configuration saved to {config_save_path}")
