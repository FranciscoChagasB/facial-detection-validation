from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np

@dataclass(frozen=True)
class LetterboxResult:
    image: np.ndarray               # imagem final (H',W',3)
    scale: float                    # escala aplicada no resize
    pad_xy: Tuple[int, int]         # padding (pad_x, pad_y) aplicado à esquerda/topo
    out_hw: Tuple[int, int]         # (out_h, out_w)
    in_hw: Tuple[int, int]          # (in_h, in_w)

def letterbox(
    img_rgb: np.ndarray,
    out_hw: Tuple[int, int],
    color: Tuple[int, int, int] = (114, 114, 114),
) -> LetterboxResult:
    """
    Redimensiona mantendo aspecto e aplica padding para caber em out_hw.
    Retorna metadados para reverter boxes depois (desletterbox).
    img_rgb: (H,W,3) RGB uint8
    """
    import cv2

    in_h, in_w = img_rgb.shape[:2]
    out_h, out_w = out_hw

    scale = min(out_w / in_w, out_h / in_h)
    new_w = int(round(in_w * scale))
    new_h = int(round(in_h * scale))

    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (out_w - new_w) // 2
    pad_y = (out_h - new_h) // 2

    canvas = np.full((out_h, out_w, 3), color, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return LetterboxResult(
        image=canvas,
        scale=scale,
        pad_xy=(pad_x, pad_y),
        out_hw=(out_h, out_w),
        in_hw=(in_h, in_w),
    )

def unletterbox_xyxy(
    boxes_xyxy: np.ndarray,
    meta: LetterboxResult,
    clamp: bool = True,
) -> np.ndarray:
    """
    Converte boxes do espaço letterbox (out_hw) para o espaço original (in_hw).
    boxes_xyxy em pixels relativos ao out_hw.
    """
    pad_x, pad_y = meta.pad_xy
    scale = meta.scale
    in_h, in_w = meta.in_hw

    b = boxes_xyxy.copy().astype(np.float32)
    b[:, [0, 2]] -= pad_x
    b[:, [1, 3]] -= pad_y
    b /= max(scale, 1e-9)

    if clamp:
        b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, in_w - 1)
        b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, in_h - 1)

    return b

def normalize_image_tensor(img_float: np.ndarray, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> np.ndarray:
    """
    img_float: RGB float32 em [0,1], shape (H,W,3)
    retorna img normalizada
    """
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return (img_float - mean) / (std + 1e-9)

def crop_faces(
    frame_rgb: np.ndarray,
    boxes_xyxy_px: np.ndarray,
    out_size: int = 112,
    min_side: int = 10,
) -> List[np.ndarray]:
    """
    Recorta faces do frame (RGB uint8) usando boxes em pixels (xyxy).
    Retorna lista de crops RGB uint8 (out_size,out_size,3).
    """
    import cv2

    H, W = frame_rgb.shape[:2]
    crops = []
    for x1, y1, x2, y2 in boxes_xyxy_px.astype(np.int32):
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) < min_side or (y2 - y1) < min_side:
            continue

        crop = frame_rgb[y1:y2, x1:x2]
        crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        crops.append(crop)

    return crops

def expand_and_square_boxes_xyxy_px(
    boxes_xyxy_px: np.ndarray,
    img_w: int,
    img_h: int,
    scale: float = 1.25,
) -> np.ndarray:
    """
    Recebe boxes xyxy em pixels e:
      - expande pelo fator 'scale' (ex: 1.25 = +25%)
      - transforma em box quadrada (mantém centro)
      - clampa no tamanho da imagem
    Retorna boxes xyxy em pixels (float32).
    """
    if boxes_xyxy_px.size == 0:
        return boxes_xyxy_px.reshape(0, 4).astype(np.float32)

    b = boxes_xyxy_px.astype(np.float32).copy()
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    w = (x2 - x1).clip(min=1.0)
    h = (y2 - y1).clip(min=1.0)

    cx = x1 + w / 2.0
    cy = y1 + h / 2.0

    side = np.maximum(w, h) * float(scale)

    nx1 = cx - side / 2.0
    ny1 = cy - side / 2.0
    nx2 = cx + side / 2.0
    ny2 = cy + side / 2.0

    nx1 = np.clip(nx1, 0, img_w - 1)
    ny1 = np.clip(ny1, 0, img_h - 1)
    nx2 = np.clip(nx2, 0, img_w - 1)
    ny2 = np.clip(ny2, 0, img_h - 1)

    return np.stack([nx1, ny1, nx2, ny2], axis=1).astype(np.float32)