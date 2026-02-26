from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.runtime.config import RuntimeConfig, DetectorRuntimeConfig, VerificationRuntimeConfig
from src.runtime.gallery import GalleryManager, GalleryConfig

from src.common.preprocess import crop_faces
from src.common.preprocess import expand_and_square_boxes_xyxy_px  # você adicionou essa função

from src.detection.model import TinySSD, DetectorConfig
from src.detection.infer import detect_faces, Detection

from src.verification.model import FaceEmbedNet, EmbedConfig
from src.verification.infer import preprocess_face_image

@dataclass(frozen=True)
class FaceMatch:
    identity: Optional[str]
    score: float
    is_match: bool
    metric: str
    threshold: float

@dataclass(frozen=True)
class FaceDetectionResult:
    box_xyxy_px: Tuple[float, float, float, float]
    det_score: float
    match: Optional[FaceMatch] = None

@dataclass(frozen=True)
class FrameResult:
    camera_id: str
    timestamp: float
    detections: List[FaceDetectionResult]

def _load_ckpt_state_dict(path: str) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict):
        # pode ser state_dict direto
        return ckpt
    raise ValueError(f"Checkpoint inválido: {path}")

class FaceRuntimeService:
    """
    Serviço de runtime para:
      - carregar modelos (detector + verificador)
      - carregar/atualizar galeria
      - processar frames
    """

    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.use_amp = bool(cfg.amp and self.device.type == "cuda")

        self.det_model: Optional[TinySSD] = None
        self.ver_model: Optional[FaceEmbedNet] = None

        self.det_cfg: Optional[DetectorRuntimeConfig] = cfg.detector
        self.ver_cfg: Optional[VerificationRuntimeConfig] = cfg.verification

        self.gallery_mgr = GalleryManager(
            GalleryConfig(
                metric=(self.ver_cfg.metric if self.ver_cfg else "cosine"),
                pt_path=cfg.gallery_pt_path,
                root_dir=cfg.gallery_root_dir,
                one_image_per_identity=cfg.gallery_one_image_per_id,
            )
        )

    # Loaders

    def load_detector(self) -> None:
        if self.det_cfg is None:
            return

        if not os.path.isfile(self.det_cfg.checkpoint_path):
            raise FileNotFoundError(f"Detector checkpoint não encontrado: {self.det_cfg.checkpoint_path}")

        model = TinySSD(DetectorConfig(input_hw=self.det_cfg.input_hw)).to(self.device)
        sd = _load_ckpt_state_dict(self.det_cfg.checkpoint_path)
        model.load_state_dict(sd, strict=True)
        model.eval()
        self.det_model = model

    def load_verifier(self) -> None:
        if self.ver_cfg is None:
            return

        if not os.path.isfile(self.ver_cfg.checkpoint_path):
            raise FileNotFoundError(f"Verification checkpoint não encontrado: {self.ver_cfg.checkpoint_path}")

        cfg = EmbedConfig(
            input_hw=self.ver_cfg.input_hw,
            base_c=self.ver_cfg.base_c,
            emb_dim=self.ver_cfg.emb_dim,
            act=self.ver_cfg.act,
            use_se=self.ver_cfg.use_se,
            dropout=self.ver_cfg.dropout,
        )
        model = FaceEmbedNet(cfg).to(self.device)
        sd = _load_ckpt_state_dict(self.ver_cfg.checkpoint_path)
        model.load_state_dict(sd, strict=True)
        model.eval()
        self.ver_model = model

    def load_gallery(self, prefer_pt: bool = True) -> None:
        """
        Carrega galeria:
          - preferencialmente do .pt (se existir)
          - senão rebuild da pasta
        """
        if self.ver_model is None:
            raise RuntimeError("Carregue o verificador antes de carregar a galeria.")

        loaded = False
        if prefer_pt:
            loaded = self.gallery_mgr.load_pt(map_location="cpu")

        if loaded:
            self.gallery_mgr.to(self.device)
            return

        if self.cfg.gallery_root_dir:
            self.gallery_mgr.rebuild_from_folder(self.ver_model, self.device)
            if self.cfg.gallery_pt_path:
                self.gallery_mgr.save_pt()

    def load_all(self) -> None:
        self.load_detector()
        self.load_verifier()
        if self.ver_model is not None and (self.cfg.gallery_pt_path or self.cfg.gallery_root_dir):
            self.load_gallery(prefer_pt=True)

    # Core processing

    @torch.no_grad()
    def _match_embeddings(self, emb_faces: torch.Tensor) -> List[FaceMatch]:
        """
        emb_faces: (N,D), normalizado
        retorna lista de FaceMatch len N (top1 por face)
        """
        assert self.ver_cfg is not None

        gallery = self.gallery_mgr.get()
        metric = self.ver_cfg.metric.lower()
        thr = float(self.ver_cfg.threshold)

        matches: List[FaceMatch] = []
        if gallery is None or gallery.size == 0:
            # sem galeria: devolve sem identidade (somente score=0)
            for _ in range(emb_faces.size(0)):
                matches.append(FaceMatch(identity=None, score=0.0, is_match=False, metric=metric, threshold=thr))
            return matches

        # Para eficiência: busca via dot product (cosine) em batch quando metric=cosine
        if metric == "cosine":
            # gallery.embeddings: (M,D) normalizado
            sims = gallery.embeddings @ emb_faces.t()  # (M,N)
            best_scores, best_idx = sims.max(dim=0)    # (N,)
            for j in range(emb_faces.size(0)):
                score = float(best_scores[j].item())
                idx = int(best_idx[j].item())
                identity = gallery.identities[idx]
                is_match = score >= thr
                matches.append(FaceMatch(identity=identity, score=score, is_match=bool(is_match), metric=metric, threshold=thr))
            return matches

        # Para distâncias, faz em loop simples (N pequeno na prática)
        for j in range(emb_faces.size(0)):
            q = emb_faces[j].detach().cpu().numpy()
            top1, ok = gallery.verify_top1(q, threshold=thr)
            if top1 is None:
                matches.append(FaceMatch(identity=None, score=0.0, is_match=False, metric=metric, threshold=thr))
            else:
                matches.append(FaceMatch(identity=top1.identity, score=float(top1.score), is_match=bool(ok), metric=metric, threshold=thr))
        return matches

    @torch.no_grad()
    def process_frame(
        self,
        frame_bgr: np.ndarray,
        camera_id: str,
        timestamp: float,
        return_only_matches: bool = False,
    ) -> FrameResult:
        """
        Processa um frame BGR:
          - detecção (se det_model carregado)
          - crop melhorado (expand + square)
          - embeddings (se ver_model carregado)
          - matching (se galeria carregada)

        return_only_matches:
          - se True, retorna apenas detections com match.is_match=True
        """
        detections_out: List[FaceDetectionResult] = []

        # 1) Detecção
        if self.det_model is None or self.det_cfg is None:
            return FrameResult(camera_id=camera_id, timestamp=timestamp, detections=[])

        dets, _ = detect_faces(
            model=self.det_model,
            frame=frame_bgr,
            device=self.device,
            score_thr=self.det_cfg.score_thr,
            iou_thr=self.det_cfg.iou_thr,
            topk=self.det_cfg.topk,
            assume_bgr=self.det_cfg.assume_bgr,
            return_debug=False,
        )

        if len(dets) == 0:
            return FrameResult(camera_id=camera_id, timestamp=timestamp, detections=[])

        # 2) Boxes -> expand + square (em pixels)
        frame_rgb = frame_bgr[:, :, ::-1].copy()  # BGR->RGB
        H, W = frame_rgb.shape[:2]
        boxes_px = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=np.float32)
        boxes_px = expand_and_square_boxes_xyxy_px(
            boxes_xyxy_px=boxes_px,
            img_w=W,
            img_h=H,
            scale=self.cfg.crop.expand_square_scale,
        )

        # 3) Crops (RGB 112x112)
        crops = crop_faces(
            frame_rgb=frame_rgb,
            boxes_xyxy_px=boxes_px,
            out_size=self.ver_cfg.input_hw[0] if self.ver_cfg else 112,
            min_side=self.cfg.crop.min_side_px,
        )

        # Como crop_faces pode descartar alguns boxes, precisamos alinhar índice
        # Estratégia: recortar novamente sem descartar, mas como já testou aceitável,
        # usamos aqui um alinhamento simples: refaz a lista de boxes válidos na mesma ordem.
        # Para garantir 1-para-1, vamos gerar "valid_mask" replicando a lógica do crop_faces:
        valid_boxes: List[Tuple[float, float, float, float]] = []
        valid_det_scores: List[float] = []
        kept_crops: List[np.ndarray] = []

        for i, (x1, y1, x2, y2) in enumerate(boxes_px.tolist()):
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            if (x2i - x1i) < self.cfg.crop.min_side_px or (y2i - y1i) < self.cfg.crop.min_side_px:
                continue
            if x2i <= x1i or y2i <= y1i:
                continue
            # crop e resize aqui para manter 1:1 com as boxes válidas
            crop = frame_rgb[y1i:y2i, x1i:x2i]
            if crop.size == 0:
                continue
            crop = torch.from_numpy(crop)  # dummy p/ check, mas vamos usar cv2 para resize
            crop_np = frame_rgb[y1i:y2i, x1i:x2i]
            import cv2
            out_size = self.ver_cfg.input_hw[0] if self.ver_cfg else 112
            crop_np = cv2.resize(crop_np, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

            valid_boxes.append((float(x1), float(y1), float(x2), float(y2)))
            valid_det_scores.append(float(dets[i].score))
            kept_crops.append(crop_np)

        if len(kept_crops) == 0:
            return FrameResult(camera_id=camera_id, timestamp=timestamp, detections=[])

        # 4) Embeddings + matching
        if self.ver_model is None or self.ver_cfg is None:
            # retorna só detecções sem match
            for box, ds in zip(valid_boxes, valid_det_scores):
                detections_out.append(FaceDetectionResult(box_xyxy_px=box, det_score=ds, match=None))
            return FrameResult(camera_id=camera_id, timestamp=timestamp, detections=detections_out)

        # preprocess batch para o verificador
        xs = []
        for c in kept_crops:
            x = preprocess_face_image(
                image=c,
                input_hw=self.ver_cfg.input_hw,
                device=self.device,
                assume_bgr=False,
                center_crop_square=False,
            )
            xs.append(x)
        xb = torch.cat(xs, dim=0)  # (N,3,H,W)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            emb_faces = self.ver_model(xb)  # (N,D) normalizado

        matches = self._match_embeddings(emb_faces)

        for box, ds, m in zip(valid_boxes, valid_det_scores, matches):
            if return_only_matches and (m is None or not m.is_match):
                continue
            detections_out.append(FaceDetectionResult(box_xyxy_px=box, det_score=ds, match=m))

        return FrameResult(camera_id=camera_id, timestamp=timestamp, detections=detections_out)