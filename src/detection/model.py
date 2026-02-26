from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.boxes import AnchorSpec, generate_anchors_multi_scale

# Blocos básicos

class ConvBNAct(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int = 3,
        s: int = 1,
        p: Optional[int] = None,
        act: str = "relu",
    ):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"act inválida: {act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, c: int, act: str = "relu"):
        super().__init__()
        self.conv1 = ConvBNAct(c, c, k=3, s=1, act=act)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.act = nn.ReLU(inplace=True) if act == "relu" else nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.bn2(self.conv2(y))
        return self.act(x + y)

# Backbone (strides 8/16/32)

class TinyBackbone(nn.Module):
    """
    Entrada: (B,3,H,W)
    Saída: features [f8, f16, f32]
      f8  -> stride 8
      f16 -> stride 16
      f32 -> stride 32
    """

    def __init__(self, base_c: int = 32, act: str = "relu"):
        super().__init__()
        c = base_c

        # /2
        self.stem = nn.Sequential(
            ConvBNAct(3, c, k=3, s=2, act=act),
            ConvBNAct(c, c, k=3, s=1, act=act),
        )

        # /4
        self.stage4 = nn.Sequential(
            ConvBNAct(c, 2 * c, k=3, s=2, act=act),
            ResidualBlock(2 * c, act=act),
        )

        # /8
        self.stage8 = nn.Sequential(
            ConvBNAct(2 * c, 4 * c, k=3, s=2, act=act),
            ResidualBlock(4 * c, act=act),
            ResidualBlock(4 * c, act=act),
        )

        # /16
        self.stage16 = nn.Sequential(
            ConvBNAct(4 * c, 6 * c, k=3, s=2, act=act),
            ResidualBlock(6 * c, act=act),
            ResidualBlock(6 * c, act=act),
        )

        # /32
        self.stage32 = nn.Sequential(
            ConvBNAct(6 * c, 8 * c, k=3, s=2, act=act),
            ResidualBlock(8 * c, act=act),
            ResidualBlock(8 * c, act=act),
        )

        self.out_channels = (4 * c, 6 * c, 8 * c)  # (f8,f16,f32)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)     # /2
        x = self.stage4(x)   # /4
        f8 = self.stage8(x)  # /8
        f16 = self.stage16(f8)  # /16
        f32 = self.stage32(f16) # /32
        return [f8, f16, f32]

# Heads SSD

class SSDHead(nn.Module):
    """
    Para cada feature map:
      cls: (B, H*W*A, 2)
      reg: (B, H*W*A, 4)  -> deltas (tx,ty,tw,th)
    """

    def __init__(self, channels: List[int], anchors_per_cell: int):
        super().__init__()
        self.anchors_per_cell = anchors_per_cell

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for ch in channels:
            self.cls_convs.append(nn.Conv2d(ch, anchors_per_cell * 2, kernel_size=3, stride=1, padding=1))
            self.reg_convs.append(nn.Conv2d(ch, anchors_per_cell * 4, kernel_size=3, stride=1, padding=1))

        self._init_weights()

    def _init_weights(self) -> None:
        # Inicialização simples e estável para SSD
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_all = []
        reg_all = []

        for f, cls_conv, reg_conv in zip(feats, self.cls_convs, self.reg_convs):
            cls = cls_conv(f)  # (B, A*2, H, W)
            reg = reg_conv(f)  # (B, A*4, H, W)

            B, _, H, W = cls.shape
            A = self.anchors_per_cell

            cls = cls.permute(0, 2, 3, 1).contiguous().view(B, H * W * A, 2)
            reg = reg.permute(0, 2, 3, 1).contiguous().view(B, H * W * A, 4)

            cls_all.append(cls)
            reg_all.append(reg)

        return torch.cat(cls_all, dim=1), torch.cat(reg_all, dim=1)

@dataclass(frozen=True)
class DetectorConfig:
    input_hw: Tuple[int, int] = (320, 320)   # (H,W) padrão
    base_c: int = 32
    act: str = "relu"
    # specs p/ strides 8/16/32
    anchor_specs: Tuple[AnchorSpec, AnchorSpec, AnchorSpec] = (
        AnchorSpec(stride=8,  sizes=(0.06, 0.10), ratios=(1.0, 0.75, 1.25)),
        AnchorSpec(stride=16, sizes=(0.14, 0.20), ratios=(1.0, 0.75, 1.25)),
        AnchorSpec(stride=32, sizes=(0.28, 0.40), ratios=(1.0, 0.75, 1.25)),
    )

class TinySSD(nn.Module):
    """
    Forward:
      cls_logits: (B, A_total, 2)
      bbox_deltas: (B, A_total, 4)

    Anchors:
      use model.generate_anchors(device) -> (A_total,4) cxcywh normalizado
    """

    def __init__(self, cfg: DetectorConfig = DetectorConfig()):
        super().__init__()
        self.cfg = cfg
        self.backbone = TinyBackbone(base_c=cfg.base_c, act=cfg.act)

        # anchors_per_cell = len(sizes)*len(ratios) (assumindo mesmo p/ cada escala)
        apc0 = len(cfg.anchor_specs[0].sizes) * len(cfg.anchor_specs[0].ratios)
        apc1 = len(cfg.anchor_specs[1].sizes) * len(cfg.anchor_specs[1].ratios)
        apc2 = len(cfg.anchor_specs[2].sizes) * len(cfg.anchor_specs[2].ratios)
        if not (apc0 == apc1 == apc2):
            raise ValueError("Para simplificar, anchors_per_cell deve ser igual em todas as escalas.")
        self.anchors_per_cell = apc0

        ch = list(self.backbone.out_channels)
        self.head = SSDHead(channels=ch, anchors_per_cell=self.anchors_per_cell)

    @staticmethod
    def feature_map_sizes(input_hw: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Para um input (H,W), backbone gera:
          stride 8  -> (ceil(H/8),  ceil(W/8))  (na prática com conv stride, é floor-ish dependendo)
        Para manter consistência, usamos divisão inteira como resultado típico do Conv2d stride.
        Se você usar input múltiplo de 32 (ex: 320), fica exato.
        """
        H, W = input_hw
        # Com conv stride=2 repetido, para inputs múltiplos de 32, é exato:
        return ((H // 8, W // 8), (H // 16, W // 16), (H // 32, W // 32))

    def generate_anchors(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Retorna anchors cxcywh normalizados (A_total,4), coerentes com cfg.input_hw.
        """
        fm_sizes = self.feature_map_sizes(self.cfg.input_hw)
        img_h, img_w = self.cfg.input_hw
        return generate_anchors_multi_scale(
            feature_map_sizes=fm_sizes,
            specs=self.cfg.anchor_specs,
            img_h=img_h,
            img_w=img_w,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B,3,H,W) — idealmente H,W = cfg.input_hw
        """
        feats = self.backbone(x)
        cls_logits, bbox_deltas = self.head(feats)
        return cls_logits, bbox_deltas