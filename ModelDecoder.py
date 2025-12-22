import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=300):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, T, D)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]



class TransformerStackedDecoder(nn.Module):
    def __init__(
        self,
        input_dim=768,
        output_dim=136,
        num_layers=6,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1
    ):
        super().__init__()

        # 🔽 时间下采样：249 → 125
        self.downsample = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.pos_enc = PositionalEncoding(input_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        x: (B, 249, 768)
        """
        # Conv1d expects (B, C, T)
        x = x.transpose(1, 2)          # (B, 768, 249)
        x = self.downsample(x)         # (B, 768, 125)
        x = x.transpose(1, 2)          # (B, 125, 768)

        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.proj(x)               # (B, 125, 136)

        return x


if __name__ == "__main__":
    model = TransformerStackedDecoder()
    x = torch.randn(2, 249, 768)
    y = model(x)
    print(y.shape)
