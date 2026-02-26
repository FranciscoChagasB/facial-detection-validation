from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

# Conversões de bounding box

def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    boxes: (..., 4) em (cx, cy, w, h)
    return: (..., 4) em (x1, y1, x2, y2)
    """
    cx, cy, w, h = boxes.unbind(dim=-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)

def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    boxes: (..., 4) em (x1, y1, x2, y2)
    return: (..., 4) em (cx, cy, w, h)
    """
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    w = (x2 - x1).clamp(min=0)
    h = (y2 - y1).clamp(min=0)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return torch.stack([cx, cy, w, h], dim=-1)

def clip_xyxy(boxes: torch.Tensor, minv: float = 0.0, maxv: float = 1.0) -> torch.Tensor:
    """
    Clipa boxes xyxy para [minv, maxv].
    Útil quando boxes estão normalizadas.
    """
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    x1 = x1.clamp(minv, maxv)
    y1 = y1.clamp(minv, maxv)
    x2 = x2.clamp(minv, maxv)
    y2 = y2.clamp(minv, maxv)
    return torch.stack([x1, y1, x2, y2], dim=-1)

# IoU
def iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Calcula IoU entre:
      a: (N,4) xyxy
      b: (M,4) xyxy
    Retorna: (N,M)

    Observação: funciona em CPU/GPU.
    """
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.size(0), b.size(0)), device=a.device, dtype=a.dtype)

    N, M = a.size(0), b.size(0)
    a_exp = a[:, None, :].expand(N, M, 4)
    b_exp = b[None, :, :].expand(N, M, 4)

    x1 = torch.maximum(a_exp[..., 0], b_exp[..., 0])
    y1 = torch.maximum(a_exp[..., 1], b_exp[..., 1])
    x2 = torch.minimum(a_exp[..., 2], b_exp[..., 2])
    y2 = torch.minimum(a_exp[..., 3], b_exp[..., 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area_a = (a_exp[..., 2] - a_exp[..., 0]).clamp(min=0) * (a_exp[..., 3] - a_exp[..., 1]).clamp(min=0)
    area_b = (b_exp[..., 2] - b_exp[..., 0]).clamp(min=0) * (b_exp[..., 3] - b_exp[..., 1]).clamp(min=0)

    union = area_a + area_b - inter + 1e-6
    return inter / union


# Anchors (SSD-like)
@dataclass(frozen=True)
class AnchorSpec:
    """
    Definição de anchors para um feature map.
    - stride: passo em pixels no input (ex: 8, 16, 32)
    - sizes: tamanhos relativos (normalizados) ex: [0.06, 0.10, 0.16]
    - ratios: razões w/h ex: [1.0, 0.75, 1.25]
    """
    stride: int
    sizes: Tuple[float, ...]
    ratios: Tuple[float, ...]

def generate_anchors_for_fm(
    fm_h: int,
    fm_w: int,
    spec: AnchorSpec,
    img_h: int,
    img_w: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Gera anchors (cx,cy,w,h) NORMALIZADOS [0,1], no total fm_h*fm_w*A.

    - Centro em pixels: (x+0.5)*stride
    - Normaliza por img_w/img_h.
    - w/h também em escala normalizada (proporção do tamanho da imagem).
    """
    # centros em pixels
    ys = (torch.arange(fm_h, device=device, dtype=dtype) + 0.5) * spec.stride
    xs = (torch.arange(fm_w, device=device, dtype=dtype) + 0.5) * spec.stride
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)

    # normaliza centro
    cx = xx / float(img_w)
    cy = yy / float(img_h)

    anchors = []
    for s in spec.sizes:
        for r in spec.ratios:
            w = s * (r ** 0.5)
            h = s / (r ** 0.5)
            aw = torch.full_like(cx, w)
            ah = torch.full_like(cy, h)
            a = torch.stack([cx, cy, aw, ah], dim=-1)  # (H,W,4)
            anchors.append(a)

    anchors = torch.stack(anchors, dim=-2)          # (H,W,A,4)
    anchors = anchors.reshape(-1, 4).contiguous()   # (H*W*A,4)
    return anchors

def generate_anchors_multi_scale(
    feature_map_sizes: Tuple[Tuple[int, int], ...],
    specs: Tuple[AnchorSpec, ...],
    img_h: int,
    img_w: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Gera anchors para múltiplas escalas (f3,f4,f5).
    Retorna: (A_total, 4) em cxcywh normalizado.
    """
    assert len(feature_map_sizes) == len(specs), "feature_map_sizes e specs devem ter mesmo tamanho"

    all_anchors = []
    for (h, w), spec in zip(feature_map_sizes, specs):
        all_anchors.append(generate_anchors_for_fm(h, w, spec, img_h, img_w, device, dtype))

    return torch.cat(all_anchors, dim=0)

# Encode / Decode (SSD style)
def encode_deltas(
    anchors_cxcywh: torch.Tensor,
    gt_cxcywh: torch.Tensor,
    variances: Tuple[float, float] = (0.1, 0.2),
) -> torch.Tensor:
    """
    Codifica gt em deltas (tx,ty,tw,th) relativo aos anchors, estilo SSD.

    tx = (gcx - acx)/(aw*vx)
    ty = (gcy - acy)/(ah*vx)
    tw = log(gw/aw)/vy
    th = log(gh/ah)/vy
    """
    ax, ay, aw, ah = anchors_cxcywh.unbind(-1)
    gx, gy, gw, gh = gt_cxcywh.unbind(-1)
    vx, vy = variances

    tx = (gx - ax) / (aw * vx + 1e-9)
    ty = (gy - ay) / (ah * vx + 1e-9)
    tw = torch.log((gw / (aw + 1e-9)).clamp(min=1e-9)) / vy
    th = torch.log((gh / (ah + 1e-9)).clamp(min=1e-9)) / vy
    return torch.stack([tx, ty, tw, th], dim=-1)

def decode_deltas(
    anchors_cxcywh: torch.Tensor,
    deltas: torch.Tensor,
    variances: Tuple[float, float] = (0.1, 0.2),
) -> torch.Tensor:
    """
    Decodifica deltas (tx,ty,tw,th) para cxcywh.
    """
    ax, ay, aw, ah = anchors_cxcywh.unbind(-1)
    tx, ty, tw, th = deltas.unbind(-1)
    vx, vy = variances

    cx = ax + tx * aw * vx
    cy = ay + ty * ah * vx
    w = aw * torch.exp((tw * vy).clamp(-10, 10))
    h = ah * torch.exp((th * vy).clamp(-10, 10))
    return torch.stack([cx, cy, w, h], dim=-1)

# Matching GT -> Anchors
def match_anchors(
    anchors_xyxy: torch.Tensor,
    gt_xyxy: torch.Tensor,
    pos_iou: float = 0.5,
    neg_iou: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Faz matching por IoU:
      - positivos: IoU >= pos_iou
      - negativos: IoU < neg_iou
      - neutros: resto

    Retorna:
      best_gt_idx: (A,) índice do GT associado (mesmo para neutros/neg)
      pos_mask: (A,) bool
      neg_mask: (A,) bool
    """
    A = anchors_xyxy.size(0)
    if gt_xyxy.numel() == 0:
        best_gt_idx = torch.zeros((A,), device=anchors_xyxy.device, dtype=torch.long)
        pos_mask = torch.zeros((A,), device=anchors_xyxy.device, dtype=torch.bool)
        neg_mask = torch.ones((A,), device=anchors_xyxy.device, dtype=torch.bool)
        return best_gt_idx, pos_mask, neg_mask

    ious = iou_xyxy(anchors_xyxy, gt_xyxy)  # (A,G)
    best_iou, best_gt_idx = ious.max(dim=1)

    pos_mask = best_iou >= pos_iou
    neg_mask = best_iou < neg_iou
    return best_gt_idx, pos_mask, neg_mask