from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass(frozen=True)
class DetectorRuntimeConfig:
    # checkpoint do detector (last.pt/best.pt)
    checkpoint_path: str

    # tamanho do input do detector (deve bater com o treino do detector)
    input_hw: Tuple[int, int] = (320, 320)

    # thresholds da detecção
    score_thr: float = 0.25
    iou_thr: float = 0.60
    topk: int = 300

    # OpenCV frames normalmente são BGR
    assume_bgr: bool = True

@dataclass(frozen=True)
class VerificationRuntimeConfig:
    # checkpoint do verificador (last.pt/best.pt)
    checkpoint_path: str

    # config do modelo de embedding (deve bater com o treino)
    input_hw: Tuple[int, int] = (112, 112)
    base_c: int = 64
    emb_dim: int = 256
    act: str = "relu"
    use_se: bool = True
    dropout: float = 0.0

    # métrica/threshold
    metric: str = "cosine"   # "cosine" | "euclidean" | "sqeuclidean"
    threshold: float = 0.60

@dataclass(frozen=True)
class CropRuntimeConfig:
    # expande e faz box quadrada antes do crop
    expand_square_scale: float = 1.25
    min_side_px: int = 20

    # se quiser forçar crop quadrado central na referência (quando a ref não for crop)
    center_crop_square_reference: bool = True

@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cuda"  # "cuda" | "cpu"
    amp: bool = True

    detector: Optional[DetectorRuntimeConfig] = None
    verification: Optional[VerificationRuntimeConfig] = None
    crop: CropRuntimeConfig = CropRuntimeConfig()

    # caminhos opcionais para galeria persistida
    gallery_pt_path: str = ""          # ex: "data/gallery.pt"
    gallery_root_dir: str = ""         # ex: "data/galeria" (CPF/pasta/fotos)
    gallery_one_image_per_id: bool = True