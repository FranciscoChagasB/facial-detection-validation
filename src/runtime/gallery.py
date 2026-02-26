from __future__ import annotations

import os
import numpy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from src.verification.infer import FaceGallery, build_gallery_from_folder
from src.verification.model import FaceEmbedNet

@dataclass
class GalleryConfig:
    metric: str = "cosine"  # "cosine" | "euclidean" | "sqeuclidean"
    pt_path: str = ""       # caminho para salvar/carregar .pt
    root_dir: str = ""      # pasta no formato root/ID/*.jpg
    one_image_per_identity: bool = True

class GalleryManager:
    """
    Gerencia uma galeria em memória, com opção de persistência em .pt e/ou rebuild a partir de pasta.
    """

    def __init__(self, cfg: GalleryConfig):
        self.cfg = cfg
        self.gallery: Optional[FaceGallery] = None

    def is_loaded(self) -> bool:
        return self.gallery is not None and self.gallery.size > 0

    def to(self, device: torch.device) -> None:
        if self.gallery is not None:
            self.gallery.to(device)

    def load_pt(self, map_location: str = "cpu") -> bool:
        """
        Carrega galeria de pt_path. Retorna True se carregou.
        """
        if not self.cfg.pt_path:
            return False
        if not os.path.isfile(self.cfg.pt_path):
            return False

        self.gallery = FaceGallery.load(self.cfg.pt_path, map_location=map_location)
        # garante métrica desejada (se quiser forçar)
        self.gallery.metric = self.cfg.metric.lower()
        return True

    def save_pt(self) -> None:
        if not self.cfg.pt_path:
            return
        if self.gallery is None:
            return
        os.makedirs(os.path.dirname(self.cfg.pt_path) or ".", exist_ok=True)
        self.gallery.save(self.cfg.pt_path)

    @torch.no_grad()
    def rebuild_from_folder(self, model: FaceEmbedNet, device: torch.device) -> None:
        """
        Recria a galeria a partir de root_dir (root/ID/*.jpg).
        """
        if not self.cfg.root_dir:
            raise ValueError("GalleryConfig.root_dir não definido")
        if not os.path.isdir(self.cfg.root_dir):
            raise FileNotFoundError(f"Pasta da galeria não encontrada: {self.cfg.root_dir}")

        g = build_gallery_from_folder(
            model=model,
            root_dir=self.cfg.root_dir,
            device=device,
            metric=self.cfg.metric,
            one_image_per_identity=self.cfg.one_image_per_identity,
        )
        self.gallery = g.to(device)

    def get(self) -> Optional[FaceGallery]:
        return self.gallery

    def search_topk(self, query_emb: Union[torch.Tensor, "numpy.ndarray"], top_k: int = 5):
        if self.gallery is None:
            return []
        return self.gallery.search(query_emb, top_k=top_k)

    def verify_top1(self, query_emb: Union[torch.Tensor, "numpy.ndarray"], threshold: float):
        if self.gallery is None:
            return None, False
        return self.gallery.verify_top1(query_emb, threshold=threshold)