from __future__ import annotations

from typing import Tuple
import torch


def _iou_1vN(a: torch.Tensor, bs: torch.Tensor) -> torch.Tensor:
    """
    IoU entre 1 box e N boxes, todos em xyxy.
    a: (4,)
    bs: (N,4)
    """
    x1 = torch.maximum(a[0], bs[:, 0])
    y1 = torch.maximum(a[1], bs[:, 1])
    x2 = torch.minimum(a[2], bs[:, 2])
    y2 = torch.minimum(a[3], bs[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area_a = (a[2] - a[0]).clamp(min=0) * (a[3] - a[1]).clamp(min=0)
    area_b = (bs[:, 2] - bs[:, 0]).clamp(min=0) * (bs[:, 3] - bs[:, 1]).clamp(min=0)
    return inter / (area_a + area_b - inter + 1e-6)


def nms_xyxy(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_thr: float = 0.4,
    topk: int = 300,
) -> torch.Tensor:
    """
    NMS clássico.
    boxes: (N,4) xyxy
    scores: (N,)
    Retorna: indices (K,) referentes ao array original.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), device=boxes.device, dtype=torch.long)

    scores_sorted, order = scores.sort(descending=True)
    order = order[:topk]
    boxes_sel = boxes[order]

    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        ious = _iou_1vN(boxes_sel[0], boxes_sel[1:])
        mask = ious <= iou_thr

        order = order[1:][mask]
        boxes_sel = boxes_sel[1:][mask]

    return torch.tensor(keep, device=boxes.device, dtype=torch.long)


def batched_nms_xyxy(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_thr: float = 0.4,
    topk: int = 300,
) -> torch.Tensor:
    """
    NMS por classe (labels).
    boxes: (N,4)
    scores: (N,)
    labels: (N,) int
    Retorna indices keep no array original.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), device=boxes.device, dtype=torch.long)

    keep_all = []
    unique_labels = labels.unique()
    for lab in unique_labels.tolist():
        mask = labels == lab
        idx = mask.nonzero(as_tuple=False).squeeze(1)
        k = nms_xyxy(boxes[idx], scores[idx], iou_thr=iou_thr, topk=topk)
        # k contém indices em termos do idx (por causa da seleção)
        # mas nosso nms retorna indices do array original (via item())
        # então aqui, como passamos boxes[idx], scores[idx], ele retorna indices do "subarray"
        # -> precisamos adaptar para retornar relativo ao original.
        # Para manter consistência, vamos reimplementar a versão subarray:
        keep_sub = _nms_subarray(boxes[idx], scores[idx], iou_thr=iou_thr, topk=topk)
        keep_all.append(idx[keep_sub])

    return torch.cat(keep_all, dim=0) if len(keep_all) else torch.empty((0,), device=boxes.device, dtype=torch.long)


def _nms_subarray(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_thr: float = 0.4,
    topk: int = 300,
) -> torch.Tensor:
    """
    NMS retornando índices relativos ao subarray.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), device=boxes.device, dtype=torch.long)

    _, order = scores.sort(descending=True)
    order = order[:topk]
    boxes_sel = boxes[order]

    keep_rel = []
    while order.numel() > 0:
        keep_rel.append(order[0].item())

        if order.numel() == 1:
            break

        ious = _iou_1vN(boxes_sel[0], boxes_sel[1:])
        mask = ious <= iou_thr

        order = order[1:][mask]
        boxes_sel = boxes_sel[1:][mask]

    return torch.tensor(keep_rel, device=boxes.device, dtype=torch.long)