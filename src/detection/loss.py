from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F

from src.common.boxes import (
    iou_xyxy,
    xyxy_to_cxcywh,
    encode_deltas,
)

@dataclass(frozen=True)
class LossConfig:
    pos_iou: float = 0.5
    neg_iou: float = 0.4
    neg_pos_ratio: int = 3
    variances: Tuple[float, float] = (0.1, 0.2)
    cls_weight: float = 1.0
    reg_weight: float = 1.0
    smooth_l1_beta: float = 1.0  # beta=1.0 é estável e padrão

def _smooth_l1(input: torch.Tensor, target: torch.Tensor, beta: float = 1.0, reduction: str = "sum") -> torch.Tensor:
    """
    Smooth L1 compatível com versões antigas do PyTorch (sem parâmetro beta no F.smooth_l1_loss).
    """
    diff = torch.abs(input - target)
    loss = torch.where(diff < beta, 0.5 * (diff * diff) / beta, diff - 0.5 * beta)

    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    if reduction == "none":
        return loss
    raise ValueError(f"reduction inválido: {reduction}")

def _ssd_match(
    anchors_xyxy: torch.Tensor,
    gt_xyxy: torch.Tensor,
    pos_iou: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Matching SSD com bipartite matching:
      - associa cada anchor ao melhor gt (best_gt_idx)
      - define positivos por IoU >= pos_iou
      - garante que cada gt tenha pelo menos 1 anchor positivo (melhor anchor daquele gt)

    Retorna:
      assigned_gt_idx: (A,) long — índice do gt atribuído a cada anchor
      pos_mask: (A,) bool — anchors positivos
    """
    device = anchors_xyxy.device
    A = anchors_xyxy.size(0)
    G = gt_xyxy.size(0)

    ious = iou_xyxy(anchors_xyxy, gt_xyxy)  # (A,G)

    best_iou_per_anchor, best_gt_idx = ious.max(dim=1)      # (A,)
    best_anchor_per_gt = ious.argmax(dim=0)                 # (G,)

    pos_mask = best_iou_per_anchor >= pos_iou

    # bipartite: força 1 positivo por gt
    pos_mask[best_anchor_per_gt] = True

    # assigned_gt_idx começa pelo melhor gt por anchor
    assigned_gt_idx = best_gt_idx.clone()

    # e garante que os anchors "forçados" apontem exatamente para seu gt correspondente
    assigned_gt_idx[best_anchor_per_gt] = torch.arange(G, device=device, dtype=torch.long)

    return assigned_gt_idx, pos_mask

def multibox_loss(
    cls_logits: torch.Tensor,              # (B, A, 2)
    bbox_deltas: torch.Tensor,             # (B, A, 4)  deltas SSD (tx,ty,tw,th)
    anchors_cxcywh: torch.Tensor,          # (A, 4)  normalizado
    anchors_xyxy: torch.Tensor,            # (A, 4)  normalizado
    gt_boxes_xyxy_list: List[torch.Tensor],# lista len B, cada (G,4) normalizado
    cfg: LossConfig = LossConfig(),
) -> Dict[str, torch.Tensor]:
    """
    Loss SSD (classificação + regressão):
      - classificação: CE(bg/face) com hard negative mining
      - regressão: SmoothL1 nos deltas SSD (somente para positivos)

    Retorna dict com:
      - loss_total
      - loss_cls
      - loss_reg
      - num_pos (tensor)
    """
    device = cls_logits.device
    B, A, C = cls_logits.shape
    assert C == 2, "Detector está configurado para 2 classes (bg/face)."
    assert anchors_cxcywh.shape[0] == A and anchors_xyxy.shape[0] == A, "Anchors devem bater com A."

    total_cls_sum = torch.zeros((), device=device)
    total_reg_sum = torch.zeros((), device=device)
    total_pos = torch.zeros((), device=device)

    for b in range(B):
        gt_xyxy = gt_boxes_xyxy_list[b].to(device=device, dtype=anchors_xyxy.dtype)
        G = gt_xyxy.size(0)

        # targets de classificação
        target_cls = torch.zeros((A,), device=device, dtype=torch.long)

        if G == 0:
            # Sem GT: tudo background. Usa CE em todos anchors.
            cls_loss = F.cross_entropy(cls_logits[b], target_cls, reduction="sum")
            total_cls_sum += cls_loss
            # reg_loss = 0
            continue

        assigned_gt_idx, pos_mask = _ssd_match(
            anchors_xyxy=anchors_xyxy,
            gt_xyxy=gt_xyxy,
            pos_iou=cfg.pos_iou,
        )

        num_pos = int(pos_mask.sum().item())
        total_pos += float(num_pos)

        # Classe 1 para positivos
        target_cls[pos_mask] = 1

        # Classificação (com hard negative mining)
        cls_loss_all = F.cross_entropy(cls_logits[b], target_cls, reduction="none")  # (A,)

        # negativos: anchors não positivos e com IoU baixa
        # Para isso, recalculamos o best_iou por anchor (barato e garante consistência)
        ious = iou_xyxy(anchors_xyxy, gt_xyxy)  # (A,G)
        best_iou, _ = ious.max(dim=1)
        neg_mask = (best_iou < cfg.neg_iou) & (~pos_mask)

        # seleção de hard negatives
        neg_losses = cls_loss_all[neg_mask]
        if neg_losses.numel() > 0:
            k = min(cfg.neg_pos_ratio * max(1, num_pos), int(neg_losses.numel()))
            # topk em neg_losses
            topk_vals, topk_idx = torch.topk(neg_losses, k=k, largest=True, sorted=False)
            # criar máscara dos selecionados no espaço dos anchors
            neg_indices = neg_mask.nonzero(as_tuple=False).squeeze(1)
            hard_neg_mask = torch.zeros_like(neg_mask)
            hard_neg_mask[neg_indices[topk_idx]] = True
        else:
            hard_neg_mask = torch.zeros_like(neg_mask)

        cls_mask = pos_mask | hard_neg_mask
        cls_loss_sum = cls_loss_all[cls_mask].sum()
        total_cls_sum += cls_loss_sum

        # Regressão (apenas positivos)
        if num_pos > 0:
            # GT atribuído para cada anchor positivo
            gt_pos_xyxy = gt_xyxy[assigned_gt_idx[pos_mask]]           # (P,4)
            gt_pos_cxcywh = xyxy_to_cxcywh(gt_pos_xyxy)                # (P,4)

            anc_pos = anchors_cxcywh[pos_mask]                         # (P,4)
            target_deltas = encode_deltas(anc_pos, gt_pos_cxcywh, variances=cfg.variances)  # (P,4)

            pred_deltas = bbox_deltas[b, pos_mask]                     # (P,4)
            reg_loss_sum = _smooth_l1(pred_deltas, target_deltas, beta=cfg.smooth_l1_beta, reduction="sum")
            total_reg_sum += reg_loss_sum

    # normalização: padrão SSD divide pelo total de positivos (ou 1 se 0)
    denom = torch.clamp(total_pos, min=1.0)
    loss_cls = total_cls_sum / denom
    loss_reg = total_reg_sum / denom
    loss_total = cfg.cls_weight * loss_cls + cfg.reg_weight * loss_reg

    return {
        "loss_total": loss_total,
        "loss_cls": loss_cls,
        "loss_reg": loss_reg,
        "num_pos": total_pos,
    }