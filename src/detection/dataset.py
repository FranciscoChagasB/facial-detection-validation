from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.common.preprocess import letterbox, normalize_image_tensor

# Formatos suportados
#
# (A) JSON simples (recomendado)
# {
#   "images": [
#     {
#       "file": "subpasta/img001.jpg",
#       "boxes": [
#         [x1, y1, x2, y2],
#         ...
#       ]
#     },
#     ...
#   ]
# }
#
# Boxes em PIXELS no espaço da imagem original.
#
# (B) WIDER FACE (formato original de anotação .txt)
# Exemplo típico:
# 0--Parade/0_Parade_marchingband_1_849.jpg
# 2
# 449 330 122 149 0 0 0 0 0 0
# 345  80  90 120 0 0 0 0 0 0
# Onde cada face: x y w h ...
#

@dataclass(frozen=True)
class AugConfig:
    hflip_prob: float = 0.5
    brightness_prob: float = 0.3
    contrast_prob: float = 0.3
    blur_prob: float = 0.15
    max_brightness_delta: float = 0.20  # em [0..1] no espaço float
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    blur_ksize_choices: Tuple[int, ...] = (3, 5)

@dataclass(frozen=True)
class DetectionSample:
    image: torch.Tensor        # (3,H,W) float32 normalizada
    gt_boxes: torch.Tensor     # (G,4) xyxy NORMALIZADO (0..1) no espaço do input do modelo
    gt_labels: torch.Tensor    # (G,) long (1 = face)
    meta: Dict[str, Any]

def _read_image_rgb(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Falha ao ler imagem: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def _clip_boxes_xyxy_px(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    boxes = boxes.astype(np.float32).copy()
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
    return boxes

def _filter_invalid_boxes(boxes: np.ndarray, min_side: float = 2.0) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = x2 - x1
    h = y2 - y1
    keep = (w >= min_side) & (h >= min_side)
    return boxes[keep].astype(np.float32)

def _apply_hflip(img: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # img: RGB (H,W,3)
    H, W = img.shape[:2]
    img2 = img[:, ::-1, :].copy()
    if boxes.size == 0:
        return img2, boxes
    b = boxes.copy().astype(np.float32)
    x1 = b[:, 0].copy()
    x2 = b[:, 2].copy()
    b[:, 0] = (W - 1) - x2
    b[:, 2] = (W - 1) - x1
    return img2, b

def _apply_brightness(img: np.ndarray, delta: float) -> np.ndarray:
    # delta em [-max, +max] no espaço [0..1]
    x = img.astype(np.float32) / 255.0
    x = np.clip(x + delta, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)

def _apply_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    mean = x.mean(axis=(0, 1), keepdims=True)
    x = np.clip((x - mean) * factor + mean, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)

def _apply_blur(img: np.ndarray, k: int) -> np.ndarray:
    return cv2.GaussianBlur(img, (k, k), 0)

def _augment(img: np.ndarray, boxes: np.ndarray, cfg: AugConfig) -> Tuple[np.ndarray, np.ndarray]:
    # img RGB uint8, boxes xyxy px
    if random.random() < cfg.hflip_prob:
        img, boxes = _apply_hflip(img, boxes)

    if random.random() < cfg.brightness_prob:
        delta = random.uniform(-cfg.max_brightness_delta, cfg.max_brightness_delta)
        img = _apply_brightness(img, delta)

    if random.random() < cfg.contrast_prob:
        factor = random.uniform(cfg.contrast_range[0], cfg.contrast_range[1])
        img = _apply_contrast(img, factor)

    if random.random() < cfg.blur_prob:
        k = random.choice(cfg.blur_ksize_choices)
        img = _apply_blur(img, k)

    return img, boxes

def _to_model_space_letterbox(
    img_rgb: np.ndarray,
    boxes_xyxy_px: np.ndarray,
    input_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Converte imagem para letterbox (input_hw) e transforma boxes para o espaço letterbox.
    Retorna:
      img_lb_rgb_uint8, boxes_lb_xyxy_norm, meta
    """
    out_h, out_w = input_hw
    meta = letterbox(img_rgb, (out_h, out_w))  # LetterboxResult

    # converte boxes orig px -> letterbox px
    # forward do letterbox:
    # x' = x*scale + pad_x; y' = y*scale + pad_y
    pad_x, pad_y = meta.pad_xy
    scale = meta.scale

    if boxes_xyxy_px.size == 0:
        boxes_lb_px = boxes_xyxy_px.reshape(0, 4).astype(np.float32)
    else:
        b = boxes_xyxy_px.astype(np.float32).copy()
        b[:, [0, 2]] = b[:, [0, 2]] * scale + pad_x
        b[:, [1, 3]] = b[:, [1, 3]] * scale + pad_y
        boxes_lb_px = b

    # normaliza no espaço input_hw
    boxes_lb_norm = boxes_lb_px.copy().astype(np.float32)
    if boxes_lb_norm.size != 0:
        boxes_lb_norm[:, [0, 2]] /= float(out_w)
        boxes_lb_norm[:, [1, 3]] /= float(out_h)
        boxes_lb_norm = np.clip(boxes_lb_norm, 0.0, 1.0)

    meta_dict = {
        "in_hw": meta.in_hw,
        "out_hw": meta.out_hw,
        "scale": meta.scale,
        "pad_xy": meta.pad_xy,
    }

    return meta.image, boxes_lb_norm, meta_dict

# Loaders de anotação

def load_simple_json_annotations(ann_path: str) -> List[Dict[str, Any]]:
    """
    Lê JSON simples e retorna lista de registros:
      { "file": str, "boxes": List[List[float]] }
    """
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "images" not in data or not isinstance(data["images"], list):
        raise ValueError("JSON inválido: esperado chave 'images' (lista).")

    records = []
    for item in data["images"]:
        if "file" not in item:
            raise ValueError("JSON inválido: item sem 'file'.")
        boxes = item.get("boxes", [])
        if boxes is None:
            boxes = []
        records.append({"file": item["file"], "boxes": boxes})
    return records

def load_widerface_annotations(wider_txt_path: str) -> List[Dict[str, Any]]:
    """
    Parser robusto do WIDER FACE gt:

    Esperado (padrão):
      <rel_path.jpg>
      <num_faces>
      <x y w h ...>  (repetido num_faces vezes)

    Esta versão:
      - usa utf-8-sig (remove BOM)
      - ignora linhas vazias
      - se a linha do num_faces não for int e parecer outro .jpg, assume 0 faces (tolerante)
      - se for inválido de verdade, lança erro com contexto
    """
    import re

    def is_image_line(s: str) -> bool:
        s = s.strip()
        return s.lower().endswith((".jpg", ".jpeg", ".png"))

    def next_non_empty(lines: List[str], start: int) -> Tuple[int, str]:
        i = start
        while i < len(lines):
            t = lines[i].strip()
            if t != "":
                return i, t
            i += 1
        return len(lines), ""

    def parse_int(s: str) -> Optional[int]:
        # aceita "2", "2 " etc
        s = s.strip()
        if re.fullmatch(r"[+-]?\d+", s):
            try:
                return int(s)
            except Exception:
                return None
        return None

    with open(wider_txt_path, "r", encoding="utf-8-sig") as f:
        raw_lines = [ln.rstrip("\n") for ln in f.readlines()]

    records: List[Dict[str, Any]] = []

    i = 0
    while True:
        i, file_rel = next_non_empty(raw_lines, i)
        if i >= len(raw_lines) or file_rel == "":
            break

        if not is_image_line(file_rel):
            # linha inesperada, tenta seguir
            i += 1
            continue

        # tenta ler num_faces
        j, num_line = next_non_empty(raw_lines, i + 1)
        if j >= len(raw_lines) or num_line == "":
            # fim inesperado: assume 0 faces
            records.append({"file": file_rel, "boxes": []})
            break

        num_faces = parse_int(num_line)

        if num_faces is None:
            # se em vez do número veio outra imagem, assume 0 faces e NÃO consome essa linha
            if is_image_line(num_line):
                records.append({"file": file_rel, "boxes": []})
                i = j  # não "pula" a linha (ela é o próximo file_rel)
                continue

            # caso realmente inválido: explode com contexto
            ctx_start = max(0, j - 3)
            ctx_end = min(len(raw_lines), j + 4)
            contexto = "\n".join([f"{k}: {raw_lines[k]}" for k in range(ctx_start, ctx_end)])
            raise ValueError(
                f"[WIDER PARSE] Esperava num_faces após '{file_rel}', mas veio: '{num_line}'.\n"
                f"Trecho do arquivo:\n{contexto}"
            )

        # ok, agora lê num_faces linhas de boxes
        boxes = []
        k = j + 1
        for _ in range(num_faces):
            k, box_line = next_non_empty(raw_lines, k)
            if k >= len(raw_lines) or box_line == "":
                break

            parts = box_line.split()
            # padrão tem pelo menos 4 valores: x y w h
            if len(parts) >= 4:
                x = float(parts[0])
                y = float(parts[1])
                w = float(parts[2])
                h = float(parts[3])
                x1, y1, x2, y2 = x, y, x + w, y + h
                boxes.append([x1, y1, x2, y2])

            k += 1

        records.append({"file": file_rel, "boxes": boxes})
        i = k

    if len(records) == 0:
        raise ValueError("Nenhum registro lido do arquivo WIDER. Verifique se o caminho do .txt está correto.")

    return records

# Dataset

class FaceDetectionDataset(Dataset):
    """
    Dataset de detecção de faces.

    Retorna dict:
      {
        "image": FloatTensor (3,H,W) normalizada,
        "gt_boxes": FloatTensor (G,4) xyxy normalizado no espaço do input do modelo,
        "gt_labels": LongTensor (G,) com 1s,
        "meta": {...}
      }
    """

    def __init__(
        self,
        root_dir: str,
        annotations: Union[str, List[Dict[str, Any]]],
        input_hw: Tuple[int, int] = (320, 320),
        augment: bool = True,
        aug_cfg: AugConfig = AugConfig(),
        annotation_format: str = "auto",  # "auto" | "simple_json" | "widerface"
    ):
        """
        root_dir: pasta base das imagens
        annotations:
          - caminho p/ arquivo de anotação (json simples ou wider txt),
            ou
          - lista já carregada (records)
        """
        self.root_dir = root_dir
        self.input_hw = input_hw
        self.augment = augment
        self.aug_cfg = aug_cfg

        if isinstance(annotations, str):
            ann_path = annotations
            fmt = annotation_format.lower()

            if fmt == "auto":
                if ann_path.lower().endswith(".json"):
                    fmt = "simple_json"
                else:
                    fmt = "widerface"

            if fmt == "simple_json":
                self.records = load_simple_json_annotations(ann_path)
            elif fmt == "widerface":
                self.records = load_widerface_annotations(ann_path)
            else:
                raise ValueError(f"annotation_format inválido: {annotation_format}")
        else:
            self.records = annotations

        if not isinstance(self.records, list) or len(self.records) == 0:
            raise ValueError("Nenhum registro de anotação carregado.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        rel_path = rec["file"]
        img_path = os.path.join(self.root_dir, rel_path)

        img_rgb = _read_image_rgb(img_path)
        H, W = img_rgb.shape[:2]

        boxes = rec.get("boxes", [])
        if boxes is None:
            boxes = []
        boxes_np = np.array(boxes, dtype=np.float32).reshape(-1, 4) if len(boxes) > 0 else np.zeros((0, 4), dtype=np.float32)

        # sanitização
        if boxes_np.size != 0:
            boxes_np = _clip_boxes_xyxy_px(boxes_np, w=W, h=H)
            boxes_np = _filter_invalid_boxes(boxes_np, min_side=2.0)

        # augment no espaço original
        if self.augment:
            img_rgb, boxes_np = _augment(img_rgb, boxes_np, self.aug_cfg)
            # depois de flip/ajustes, reclip por segurança
            H2, W2 = img_rgb.shape[:2]
            if boxes_np.size != 0:
                boxes_np = _clip_boxes_xyxy_px(boxes_np, w=W2, h=H2)
                boxes_np = _filter_invalid_boxes(boxes_np, min_side=2.0)

        # letterbox -> espaço do modelo
        img_lb, boxes_norm, meta_dict = _to_model_space_letterbox(img_rgb, boxes_np, self.input_hw)

        # tensor image
        x = img_lb.astype(np.float32) / 255.0
        x = normalize_image_tensor(x)  # mean/std 0.5
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()  # (3,H,W)

        gt_boxes = torch.from_numpy(boxes_norm.astype(np.float32)) if boxes_norm.size != 0 else torch.zeros((0, 4), dtype=torch.float32)
        gt_labels = torch.ones((gt_boxes.size(0),), dtype=torch.long)  # 1=face

        meta = {
            "path": img_path,
            "rel_path": rel_path,
            "orig_hw": (H, W),
            "model_hw": self.input_hw,
            **meta_dict,
        }

        return {
            "image": x,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "meta": meta,
        }

# Collate fn (batch com GT variável)

def detection_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Junta batch com número variável de faces por imagem.
    Retorna:
      {
        "images": (B,3,H,W),
        "gt_boxes": List[Tensor(G,4)] len B,
        "gt_labels": List[Tensor(G)] len B,
        "meta": List[dict] len B
      }
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    gt_boxes = [b["gt_boxes"] for b in batch]
    gt_labels = [b["gt_labels"] for b in batch]
    meta = [b["meta"] for b in batch]
    return {"images": images, "gt_boxes": gt_boxes, "gt_labels": gt_labels, "meta": meta}