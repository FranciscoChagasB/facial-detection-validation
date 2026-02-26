from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

from src.common.boxes import (
    decode_deltas,
    cxcywh_to_xyxy,
    clip_xyxy,
)
from src.common.nms import nms_xyxy
from src.common.preprocess import letterbox, unletterbox_xyxy, normalize_image_tensor, LetterboxResult

@dataclass(frozen=True)
class Detection:
    """
    Box em pixels no frame ORIGINAL (sem letterbox).
    """
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

def _to_rgb(frame: np.ndarray, assume_bgr: bool = True) -> np.ndarray:
    """
    Converte frame para RGB uint8.
    - frame: (H,W,3) uint8
    """
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    if assume_bgr:
        # BGR -> RGB
        return frame[:, :, ::-1].copy()
    return frame.copy()

def _img_to_tensor(img_rgb_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    RGB uint8 (H,W,3) -> tensor float (1,3,H,W), normalizado.
    """
    img = img_rgb_uint8.astype(np.float32) / 255.0
    img = normalize_image_tensor(img)  # mean=0.5 std=0.5
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return x

@torch.no_grad()
def detect_faces(
    model,
    frame: np.ndarray,
    device: torch.device,
    score_thr: float = 0.60,
    iou_thr: float = 0.40,
    topk: int = 300,
    assume_bgr: bool = True,
    return_debug: bool = False,
) -> Tuple[List[Detection], Optional[Dict[str, Any]]]:
    """
    Detector completo:
      - faz preprocess (letterbox para input do modelo)
      - roda modelo
      - decodifica boxes e aplica NMS
      - retorna boxes em pixels no frame original

    model: instância do TinySSD
    frame: numpy (H,W,3)
    device: torch.device

    return:
      detections: List[Detection]
      debug (opcional): dict com info intermediária
    """
    model.eval()

    # 1) preprocess
    rgb = _to_rgb(frame, assume_bgr=assume_bgr)
    in_h, in_w = rgb.shape[:2]

    out_h, out_w = model.cfg.input_hw
    meta: LetterboxResult = letterbox(rgb, (out_h, out_w))
    x = _img_to_tensor(meta.image, device=device)

    # 2) forward
    cls_logits, bbox_deltas = model(x)  # (1,A,2), (1,A,4)
    cls_logits = cls_logits[0]
    bbox_deltas = bbox_deltas[0]

    # 3) anchors e decode
    anchors_cxcywh = model.generate_anchors(device=device, dtype=bbox_deltas.dtype)  # (A,4)
    pred_cxcywh = decode_deltas(anchors_cxcywh, bbox_deltas)                         # (A,4)
    pred_xyxy = cxcywh_to_xyxy(pred_cxcywh)                                          # (A,4) normalizado
    pred_xyxy = clip_xyxy(pred_xyxy, 0.0, 1.0)

    # 4) scores (classe 1 = face)
    probs = F.softmax(cls_logits, dim=1)[:, 1]  # (A,)

    # 5) threshold
    keep = probs >= score_thr
    if keep.sum().item() == 0:
        return [], ({"meta": meta} if return_debug else None)

    pred_xyxy = pred_xyxy[keep]
    probs = probs[keep]

    # 6) converter boxes normalizadas -> pixels no espaço letterbox (out_h,out_w)
    # pred_xyxy norm refere-se ao input do modelo (out_h,out_w).
    # *converter p/ pixels e depois desfazer letterbox.
    boxes_lb = pred_xyxy.detach().cpu().numpy().astype(np.float32)
    boxes_lb[:, [0, 2]] *= float(out_w)
    boxes_lb[:, [1, 3]] *= float(out_h)

    # 7) unletterbox p/ pixels no frame original
    boxes_orig = unletterbox_xyxy(boxes_lb, meta, clamp=True)  # (N,4) em pixels orig

    # 8) NMS (em torch)
    boxes_t = torch.from_numpy(boxes_orig).to(device=device, dtype=torch.float32)
    scores_t = probs.to(device=device, dtype=torch.float32)

    # Opcional: limitar topk pré-nms
    if scores_t.numel() > topk:
        s_sorted, idx = scores_t.sort(descending=True)
        idx = idx[:topk]
        boxes_t = boxes_t[idx]
        scores_t = s_sorted[:topk]

    keep_idx = nms_xyxy(boxes_t, scores_t, iou_thr=iou_thr, topk=topk)
    boxes_t = boxes_t[keep_idx]
    scores_t = scores_t[keep_idx]

    # 9) montar saída
    dets: List[Detection] = []
    boxes_out = boxes_t.detach().cpu().numpy()
    scores_out = scores_t.detach().cpu().numpy()

    for b, s in zip(boxes_out, scores_out):
        x1, y1, x2, y2 = b.tolist()
        dets.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=float(s)))

    debug = None
    if return_debug:
        debug = {
            "in_hw": (in_h, in_w),
            "model_hw": (out_h, out_w),
            "meta": meta,
            "num_candidates_after_thr": int(keep.sum().item()),
            "num_final": len(dets),
        }

    return dets, debug