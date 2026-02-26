from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, p: int | None = None, act: str = "relu"):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"act inválida: {act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation leve (ajuda em qualidade do embedding).
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s

class ResidualBlock(nn.Module):
    """
    Residual básico com SE opcional.
    """
    def __init__(self, c: int, act: str = "relu", use_se: bool = True):
        super().__init__()
        self.conv1 = ConvBNAct(c, c, 3, 1, act=act)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.se = SEBlock(c) if use_se else None
        self.act = nn.ReLU(inplace=True) if act == "relu" else nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.bn2(self.conv2(y))
        if self.se is not None:
            y = self.se(y)
        return self.act(x + y)

class DownBlock(nn.Module):
    """
    Reduz resolução e aumenta canais.
    """
    def __init__(self, c_in: int, c_out: int, act: str = "relu"):
        super().__init__()
        self.conv = ConvBNAct(c_in, c_out, 3, 2, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

@dataclass(frozen=True)
class EmbedConfig:
    input_hw: Tuple[int, int] = (112, 112)  # crops 112x112
    base_c: int = 64
    emb_dim: int = 256
    act: str = "relu"
    use_se: bool = True
    dropout: float = 0.0

class FaceEmbedNet(nn.Module):
    """
    CNN para verificação:
      entrada: (B,3,H,W) (ideal 112x112)
      saída: embedding (B,emb_dim) L2-normalizado

    - Sem "classificador final": é para métrica (cosine similarity).
    - Treino usa loss do verification/loss.py (triplet/contrastive).
    """
    def __init__(self, cfg: EmbedConfig = EmbedConfig()):
        super().__init__()
        self.cfg = cfg
        c = cfg.base_c

        self.stem = nn.Sequential(
            ConvBNAct(3, c, 3, 2, act=cfg.act),     # /2
            ConvBNAct(c, c, 3, 1, act=cfg.act),
        )

        self.stage1 = nn.Sequential(
            ResidualBlock(c, act=cfg.act, use_se=cfg.use_se),
            ResidualBlock(c, act=cfg.act, use_se=cfg.use_se),
        )

        self.down2 = DownBlock(c, 2 * c, act=cfg.act)  # /4
        self.stage2 = nn.Sequential(
            ResidualBlock(2 * c, act=cfg.act, use_se=cfg.use_se),
            ResidualBlock(2 * c, act=cfg.act, use_se=cfg.use_se),
        )

        self.down3 = DownBlock(2 * c, 4 * c, act=cfg.act)  # /8
        self.stage3 = nn.Sequential(
            ResidualBlock(4 * c, act=cfg.act, use_se=cfg.use_se),
            ResidualBlock(4 * c, act=cfg.act, use_se=cfg.use_se),
            ResidualBlock(4 * c, act=cfg.act, use_se=cfg.use_se),
        )

        self.down4 = DownBlock(4 * c, 6 * c, act=cfg.act)  # /16
        self.stage4 = nn.Sequential(
            ResidualBlock(6 * c, act=cfg.act, use_se=cfg.use_se),
            ResidualBlock(6 * c, act=cfg.act, use_se=cfg.use_se),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.fc = nn.Linear(6 * c, cfg.emb_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down2(x)
        x = self.stage2(x)
        x = self.down3(x)
        x = self.stage3(x)
        x = self.down4(x)
        x = self.stage4(x)

        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)

        # embedding L2 normalizado (cosine similarity vira dot product)
        x = F.normalize(x, p=2, dim=1)
        return x