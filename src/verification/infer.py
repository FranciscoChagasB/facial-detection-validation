from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.common.preprocess import normalize_image_tensor
from src.verification.model import FaceEmbedNet, EmbedConfig

# Dataclasses de saída

@dataclass(frozen=True)
class VerificationResult:
    score: float
    is_match: bool
    threshold: float
    metric: str  # "cosine" | "euclidean" | "sqeuclidean"

@dataclass(frozen=True)
class GalleryMatch:
    identity: str
    score: float
    metric: str
    rank: int

# Pré-processamento

def _read_image_rgb(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Falha ao ler imagem: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _to_rgb(img: np.ndarray, assume_bgr: bool = False) -> np.ndarray:
    """
    Garante RGB uint8.
    """
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Imagem deve ter shape (H,W,3). Recebido: {img.shape}")

    if assume_bgr:
        return img[:, :, ::-1].copy()
    return img.copy()

def _center_crop_square(img_rgb: np.ndarray) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    side = min(h, w)
    y1 = (h - side) // 2
    x1 = (w - side) // 2
    return img_rgb[y1:y1 + side, x1:x1 + side]

def _resize_rgb(img_rgb: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_hw
    return cv2.resize(img_rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

def _img_to_tensor(img_rgb_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    RGB uint8 (H,W,3) -> FloatTensor (1,3,H,W), normalizado mean/std=(0.5,0.5,0.5)
    """
    x = img_rgb_uint8.astype(np.float32) / 255.0
    x = normalize_image_tensor(x)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    return x

def preprocess_face_image(
    image: Union[str, np.ndarray],
    input_hw: Tuple[int, int] = (112, 112),
    device: torch.device = torch.device("cpu"),
    assume_bgr: bool = False,
    center_crop_square: bool = False,
) -> torch.Tensor:
    """
    Converte imagem (path ou numpy) em tensor pronto para o modelo (1,3,H,W).
    """
    if isinstance(image, str):
        img_rgb = _read_image_rgb(image)
    else:
        img_rgb = _to_rgb(image, assume_bgr=assume_bgr)

    if center_crop_square:
        img_rgb = _center_crop_square(img_rgb)

    img_rgb = _resize_rgb(img_rgb, input_hw)
    return _img_to_tensor(img_rgb, device=device)

# Embedding extraction

@torch.no_grad()
def extract_embedding(
    model: FaceEmbedNet,
    image: Union[str, np.ndarray],
    device: torch.device,
    assume_bgr: bool = False,
    center_crop_square: bool = False,
    return_tensor: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Extrai embedding L2-normalizado de UMA imagem de face.

    Retorna:
      - np.ndarray (D,) por padrão
      - torch.Tensor (D,) se return_tensor=True
    """
    model.eval()
    x = preprocess_face_image(
        image=image,
        input_hw=model.cfg.input_hw,
        device=device,
        assume_bgr=assume_bgr,
        center_crop_square=center_crop_square,
    )
    emb = model(x)[0]  # (D,), já normalizado no forward

    if return_tensor:
        return emb
    return emb.detach().cpu().numpy().astype(np.float32)

@torch.no_grad()
def extract_embeddings_batch(
    model: FaceEmbedNet,
    images: Sequence[Union[str, np.ndarray]],
    device: torch.device,
    assume_bgr: bool = False,
    center_crop_square: bool = False,
    batch_size: int = 32,
    return_tensor: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Extrai embeddings de uma lista de imagens (paths ou numpy arrays).
    Retorna:
      - np.ndarray (N,D) por padrão
      - torch.Tensor (N,D) se return_tensor=True
    """
    model.eval()

    tensors: List[torch.Tensor] = []
    for img in images:
        x = preprocess_face_image(
            image=img,
            input_hw=model.cfg.input_hw,
            device=device,
            assume_bgr=assume_bgr,
            center_crop_square=center_crop_square,
        )
        tensors.append(x)

    all_embs: List[torch.Tensor] = []
    for i in range(0, len(tensors), batch_size):
        xb = torch.cat(tensors[i:i + batch_size], dim=0)  # (B,3,H,W)
        eb = model(xb)  # (B,D), já normalizado
        all_embs.append(eb)

    embs = torch.cat(all_embs, dim=0)

    if return_tensor:
        return embs
    return embs.detach().cpu().numpy().astype(np.float32)

# Métricas / comparação

def cosine_similarity_score(emb1: Union[np.ndarray, torch.Tensor], emb2: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Como embeddings já são L2-normalizados, cosine = dot product.
    Intervalo típico: [-1, 1]
    """
    if isinstance(emb1, np.ndarray):
        emb1 = torch.from_numpy(emb1)
    if isinstance(emb2, np.ndarray):
        emb2 = torch.from_numpy(emb2)

    emb1 = emb1.float().view(-1)
    emb2 = emb2.float().view(-1)
    score = torch.dot(emb1, emb2).item()
    return float(score)

def euclidean_distance(emb1: Union[np.ndarray, torch.Tensor], emb2: Union[np.ndarray, torch.Tensor]) -> float:
    if isinstance(emb1, np.ndarray):
        emb1 = torch.from_numpy(emb1)
    if isinstance(emb2, np.ndarray):
        emb2 = torch.from_numpy(emb2)

    emb1 = emb1.float().view(-1)
    emb2 = emb2.float().view(-1)
    d = torch.norm(emb1 - emb2, p=2).item()
    return float(d)

def squared_euclidean_distance(emb1: Union[np.ndarray, torch.Tensor], emb2: Union[np.ndarray, torch.Tensor]) -> float:
    if isinstance(emb1, np.ndarray):
        emb1 = torch.from_numpy(emb1)
    if isinstance(emb2, np.ndarray):
        emb2 = torch.from_numpy(emb2)

    emb1 = emb1.float().view(-1)
    emb2 = emb2.float().view(-1)
    d = ((emb1 - emb2) ** 2).sum().item()
    return float(d)

def verify_embeddings(
    emb1: Union[np.ndarray, torch.Tensor],
    emb2: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    metric: str = "cosine",
) -> VerificationResult:
    """
    Verifica se dois embeddings são da mesma pessoa.

    metric:
      - "cosine": match se score >= threshold
      - "euclidean": match se dist <= threshold
      - "sqeuclidean": match se dist <= threshold
    """
    metric = metric.lower()

    if metric == "cosine":
        score = cosine_similarity_score(emb1, emb2)
        is_match = score >= threshold
        return VerificationResult(score=score, is_match=bool(is_match), threshold=threshold, metric=metric)

    elif metric == "euclidean":
        dist = euclidean_distance(emb1, emb2)
        is_match = dist <= threshold
        return VerificationResult(score=dist, is_match=bool(is_match), threshold=threshold, metric=metric)

    elif metric == "sqeuclidean":
        dist = squared_euclidean_distance(emb1, emb2)
        is_match = dist <= threshold
        return VerificationResult(score=dist, is_match=bool(is_match), threshold=threshold, metric=metric)

    else:
        raise ValueError("metric deve ser: cosine | euclidean | sqeuclidean")

@torch.no_grad()
def verify_faces(
    model: FaceEmbedNet,
    image1: Union[str, np.ndarray],
    image2: Union[str, np.ndarray],
    device: torch.device,
    threshold: float = 0.5,
    metric: str = "cosine",
    assume_bgr_image1: bool = False,
    assume_bgr_image2: bool = False,
    center_crop_square: bool = False,
    return_embeddings: bool = False,
) -> Union[VerificationResult, Tuple[VerificationResult, np.ndarray, np.ndarray]]:
    """
    Pipeline completo de verificação a partir de duas imagens de face.
    """
    emb1 = extract_embedding(
        model=model,
        image=image1,
        device=device,
        assume_bgr=assume_bgr_image1,
        center_crop_square=center_crop_square,
        return_tensor=False,
    )
    emb2 = extract_embedding(
        model=model,
        image=image2,
        device=device,
        assume_bgr=assume_bgr_image2,
        center_crop_square=center_crop_square,
        return_tensor=False,
    )

    result = verify_embeddings(emb1, emb2, threshold=threshold, metric=metric)

    if return_embeddings:
        return result, emb1, emb2
    return result

# Galeria (base de embeddings)

@dataclass
class FaceGallery:
    """
    Galeria simples em memória.

    identities: lista de IDs (ex.: CPF)
    embeddings: Tensor (N,D) L2-normalizado
    """
    identities: List[str]
    embeddings: torch.Tensor  # (N,D), float32, normalizado
    metric: str = "cosine"

    def __post_init__(self):
        if not isinstance(self.identities, list):
            raise ValueError("identities deve ser list[str]")
        if not torch.is_tensor(self.embeddings):
            raise ValueError("embeddings deve ser torch.Tensor")
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings deve ter shape (N,D)")
        if len(self.identities) != self.embeddings.size(0):
            raise ValueError("len(identities) deve ser igual a embeddings.shape[0]")
        self.metric = self.metric.lower()
        if self.metric not in ("cosine", "euclidean", "sqeuclidean"):
            raise ValueError("metric inválida")

    @property
    def size(self) -> int:
        return self.embeddings.size(0)

    @property
    def dim(self) -> int:
        return self.embeddings.size(1)

    def to(self, device: torch.device) -> "FaceGallery":
        self.embeddings = self.embeddings.to(device)
        return self

    def search(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor],
        top_k: int = 5,
    ) -> List[GalleryMatch]:
        """
        Busca top-k identidades mais similares ao embedding de consulta.
        """
        if self.size == 0:
            return []

        if isinstance(query_embedding, np.ndarray):
            q = torch.from_numpy(query_embedding).float()
        else:
            q = query_embedding.float()

        q = q.view(1, -1)  # (1,D)

        if q.size(1) != self.dim:
            raise ValueError(f"Dimensão incompatível: query={q.size(1)} gallery={self.dim}")

        # garante normalização (por segurança)
        q = F.normalize(q, p=2, dim=1)

        if self.metric == "cosine":
            # maior é melhor
            scores = (self.embeddings @ q.t()).squeeze(1)  # (N,)
            k = min(top_k, self.size)
            vals, idx = torch.topk(scores, k=k, largest=True, sorted=True)

            out: List[GalleryMatch] = []
            for rank, (v, i) in enumerate(zip(vals.tolist(), idx.tolist()), start=1):
                out.append(GalleryMatch(identity=self.identities[i], score=float(v), metric="cosine", rank=rank))
            return out

        elif self.metric == "euclidean":
            d = torch.norm(self.embeddings - q, p=2, dim=1)  # (N,)
            k = min(top_k, self.size)
            vals, idx = torch.topk(d, k=k, largest=False, sorted=True)

            out = []
            for rank, (v, i) in enumerate(zip(vals.tolist(), idx.tolist()), start=1):
                out.append(GalleryMatch(identity=self.identities[i], score=float(v), metric="euclidean", rank=rank))
            return out

        else:  # sqeuclidean
            d = ((self.embeddings - q) ** 2).sum(dim=1)
            k = min(top_k, self.size)
            vals, idx = torch.topk(d, k=k, largest=False, sorted=True)

            out = []
            for rank, (v, i) in enumerate(zip(vals.tolist(), idx.tolist()), start=1):
                out.append(GalleryMatch(identity=self.identities[i], score=float(v), metric="sqeuclidean", rank=rank))
            return out

    def verify_top1(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor],
        threshold: float,
    ) -> Tuple[Optional[GalleryMatch], bool]:
        """
        Retorna (top1, aprovado_por_threshold).
        """
        matches = self.search(query_embedding, top_k=1)
        if not matches:
            return None, False

        top1 = matches[0]

        if self.metric == "cosine":
            ok = top1.score >= threshold
        else:
            ok = top1.score <= threshold

        return top1, bool(ok)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "identities": self.identities,
            "embeddings": self.embeddings.detach().cpu(),
            "metric": self.metric,
        }
        torch.save(payload, path)

    @staticmethod
    def load(path: str, map_location: str = "cpu") -> "FaceGallery":
        payload = torch.load(path, map_location=map_location)
        identities = payload["identities"]
        embeddings = payload["embeddings"].float()
        metric = payload.get("metric", "cosine")

        # normaliza por segurança
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return FaceGallery(
            identities=list(identities),
            embeddings=embeddings,
            metric=metric,
        )

# Construção de galeria a partir de arquivos

@torch.no_grad()
def build_gallery_from_folder(
    model: FaceEmbedNet,
    root_dir: str,
    device: torch.device,
    metric: str = "cosine",
    one_image_per_identity: bool = False,
    assume_bgr: bool = False,
    center_crop_square: bool = False,
    valid_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
) -> FaceGallery:
    """
    Espera estrutura:
      root_dir/
        IDENTIDADE_1/
          a.jpg
          b.jpg
        IDENTIDADE_2/
          x.jpg

    Por padrão, usa TODAS as imagens e cria uma entrada por imagem (mesmo identity repetido).
    Se one_image_per_identity=True, usa apenas a primeira imagem válida de cada pasta.

    Isso é útil para a sua base por CPF:
      root_dir/12345678900/foto1.jpg
    """
    model.eval()

    identities: List[str] = []
    embs: List[torch.Tensor] = []

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"root_dir não encontrado: {root_dir}")

    for identity_name in sorted(os.listdir(root_dir)):
        identity_dir = os.path.join(root_dir, identity_name)
        if not os.path.isdir(identity_dir):
            continue

        image_files = []
        for fn in sorted(os.listdir(identity_dir)):
            if fn.lower().endswith(valid_exts):
                image_files.append(os.path.join(identity_dir, fn))

        if len(image_files) == 0:
            continue

        if one_image_per_identity:
            image_files = image_files[:1]

        for img_path in image_files:
            emb = extract_embedding(
                model=model,
                image=img_path,
                device=device,
                assume_bgr=assume_bgr,
                center_crop_square=center_crop_square,
                return_tensor=True,
            ).detach().cpu()
            identities.append(identity_name)
            embs.append(emb)

    if len(embs) == 0:
        raise ValueError(f"Nenhuma imagem válida encontrada em {root_dir}")

    emb_tensor = torch.stack(embs, dim=0).float()
    emb_tensor = F.normalize(emb_tensor, p=2, dim=1)

    gallery = FaceGallery(identities=identities, embeddings=emb_tensor, metric=metric)
    return gallery

# Helpers de modelo/checkpoint

def load_verification_model(
    checkpoint_path: Optional[str],
    device: torch.device,
    cfg: Optional[EmbedConfig] = None,
    strict: bool = True,
) -> FaceEmbedNet:
    """
    Cria modelo e (opcionalmente) carrega checkpoint.
    Aceita:
      - checkpoint contendo {"model": state_dict, ...}
      - checkpoint sendo diretamente state_dict
    """
    if cfg is None:
        cfg = EmbedConfig()

    model = FaceEmbedNet(cfg).to(device)

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=strict)

    model.eval()
    return model