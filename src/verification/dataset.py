from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterator

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader

from src.common.preprocess import normalize_image_tensor

# Configs de augmentation

@dataclass(frozen=True)
class VerificationAugConfig:
    hflip_prob: float = 0.5
    brightness_prob: float = 0.25
    contrast_prob: float = 0.25
    blur_prob: float = 0.10
    max_brightness_delta: float = 0.15   # em [0..1]
    contrast_range: Tuple[float, float] = (0.85, 1.15)
    blur_ksize_choices: Tuple[int, ...] = (3, 5)

# Utilitários

VALID_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def _is_image_file(name: str) -> bool:
    return name.lower().endswith(VALID_IMAGE_EXTS)

def _read_image_rgb(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Falha ao ler imagem: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def _resize_face(img_rgb: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_hw
    return cv2.resize(img_rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

def _center_crop_square(img_rgb: np.ndarray) -> np.ndarray:
    """
    Opcionalmente útil se a imagem da face vier retangular.
    Faz crop quadrado central.
    """
    h, w = img_rgb.shape[:2]
    side = min(h, w)
    y1 = (h - side) // 2
    x1 = (w - side) // 2
    return img_rgb[y1:y1 + side, x1:x1 + side]

def _apply_hflip(img: np.ndarray) -> np.ndarray:
    return img[:, ::-1, :].copy()

def _apply_brightness(img: np.ndarray, delta: float) -> np.ndarray:
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

def _augment_face(img_rgb: np.ndarray, cfg: VerificationAugConfig) -> np.ndarray:
    if random.random() < cfg.hflip_prob:
        img_rgb = _apply_hflip(img_rgb)

    if random.random() < cfg.brightness_prob:
        delta = random.uniform(-cfg.max_brightness_delta, cfg.max_brightness_delta)
        img_rgb = _apply_brightness(img_rgb, delta)

    if random.random() < cfg.contrast_prob:
        factor = random.uniform(cfg.contrast_range[0], cfg.contrast_range[1])
        img_rgb = _apply_contrast(img_rgb, factor)

    if random.random() < cfg.blur_prob:
        k = random.choice(cfg.blur_ksize_choices)
        img_rgb = _apply_blur(img_rgb, k)

    return img_rgb

def _to_tensor_normalized(img_rgb_uint8: np.ndarray) -> torch.Tensor:
    """
    RGB uint8 (H,W,3) -> FloatTensor (3,H,W), normalizado com mean/std=(0.5,0.5,0.5)
    """
    x = img_rgb_uint8.astype(np.float32) / 255.0
    x = normalize_image_tensor(x)
    x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
    return x

# Indexação de dataset por pastas (1 pasta = 1 identidade)

def scan_identities_from_folder(
    root_dir: str,
    min_images_per_identity: int = 1,
    recursive_inside_identity: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[str], Dict[int, List[int]]]:
    """
    Espera estrutura:
      root_dir/
        pessoa_001/
          a.jpg
          b.jpg
        pessoa_002/
          x.jpg
          y.jpg

    Retorna:
      samples: lista de dicts {path, label, label_name}
      label_to_index: map nome -> int
      index_to_label: lista index -> nome
      indices_by_label: dict label_int -> [indices em samples]
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Diretório não encontrado: {root_dir}")

    identity_dirs = []
    for name in sorted(os.listdir(root_dir)):
        full = os.path.join(root_dir, name)
        if os.path.isdir(full):
            identity_dirs.append((name, full))

    if len(identity_dirs) == 0:
        raise ValueError(f"Nenhuma pasta de identidade encontrada em: {root_dir}")

    samples: List[Dict[str, Any]] = []
    label_to_index: Dict[str, int] = {}
    index_to_label: List[str] = []
    indices_by_label: Dict[int, List[int]] = {}

    next_label = 0

    for identity_name, identity_dir in identity_dirs:
        image_paths: List[str] = []

        if recursive_inside_identity:
            for dirpath, _, filenames in os.walk(identity_dir):
                for fn in filenames:
                    if _is_image_file(fn):
                        image_paths.append(os.path.join(dirpath, fn))
        else:
            for fn in sorted(os.listdir(identity_dir)):
                full = os.path.join(identity_dir, fn)
                if os.path.isfile(full) and _is_image_file(fn):
                    image_paths.append(full)

        image_paths = sorted(image_paths)

        if len(image_paths) < min_images_per_identity:
            continue

        label_to_index[identity_name] = next_label
        index_to_label.append(identity_name)
        indices_by_label[next_label] = []

        for p in image_paths:
            idx = len(samples)
            samples.append({
                "path": p,
                "label": next_label,
                "label_name": identity_name,
            })
            indices_by_label[next_label].append(idx)

        next_label += 1

    if len(samples) == 0:
        raise ValueError(
            f"Nenhuma imagem válida encontrada em {root_dir} com min_images_per_identity={min_images_per_identity}"
        )

    return samples, label_to_index, index_to_label, indices_by_label

# Dataset principal

class FaceVerificationDataset(Dataset):
    """
    Dataset de verificação por identidade (1 pasta = 1 pessoa).

    Retorna por item:
      {
        "image": Tensor (3,H,W),
        "label": LongTensor scalar,
        "label_name": str,
        "path": str
      }

    Ideal para usar com PKBatchSampler + BatchSemiHardTripletLoss.
    """
    def __init__(
        self,
        root_dir: str,
        input_hw: Tuple[int, int] = (112, 112),
        augment: bool = True,
        aug_cfg: VerificationAugConfig = VerificationAugConfig(),
        min_images_per_identity: int = 2,
        recursive_inside_identity: bool = True,
        center_crop_square: bool = False,
    ):
        self.root_dir = root_dir
        self.input_hw = input_hw
        self.augment = augment
        self.aug_cfg = aug_cfg
        self.center_crop_square = center_crop_square

        (
            self.samples,
            self.label_to_index,
            self.index_to_label,
            self.indices_by_label,
        ) = scan_identities_from_folder(
            root_dir=root_dir,
            min_images_per_identity=min_images_per_identity,
            recursive_inside_identity=recursive_inside_identity,
        )

        self.num_classes = len(self.index_to_label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.samples[idx]
        path = rec["path"]
        label = rec["label"]
        label_name = rec["label_name"]

        img_rgb = _read_image_rgb(path)

        if self.center_crop_square:
            img_rgb = _center_crop_square(img_rgb)

        img_rgb = _resize_face(img_rgb, self.input_hw)

        if self.augment:
            img_rgb = _augment_face(img_rgb, self.aug_cfg)

        x = _to_tensor_normalized(img_rgb)

        return {
            "image": x,
            "label": torch.tensor(label, dtype=torch.long),
            "label_name": label_name,
            "path": path,
        }

# Collate fn

def verification_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    label_names = [b["label_name"] for b in batch]
    paths = [b["path"] for b in batch]

    return {
        "images": images,        # (B,3,H,W)
        "labels": labels,        # (B,)
        "label_names": label_names,
        "paths": paths,
    }

# PKBatchSampler (P identidades x K imagens)

class PKBatchSampler(Sampler[List[int]]):
    """
    Amostra batches no formato:
      P identidades x K imagens por identidade
    batch_size = P*K

    Isso é altamente recomendado para BatchSemiHardTripletLoss.

    Exemplo:
      P=8, K=4 -> batch=32

    Comportamento:
      - embaralha identidades a cada época
      - se uma identidade tem < K imagens, faz reposição dentro da identidade
      - __len__ retorna número aproximado de batches por época
    """
    def __init__(
        self,
        indices_by_label: Dict[int, List[int]],
        p: int,
        k: int,
        batches_per_epoch: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        if p <= 0 or k <= 0:
            raise ValueError("p e k devem ser > 0")

        self.indices_by_label = indices_by_label
        self.labels = sorted(list(indices_by_label.keys()))
        self.p = p
        self.k = k
        self.batch_size = p * k
        self.shuffle = shuffle
        self.drop_last = drop_last

        if len(self.labels) == 0:
            raise ValueError("indices_by_label vazio")
        if len(self.labels) < p:
            raise ValueError(f"Número de identidades ({len(self.labels)}) menor que p={p}")

        # heurística de batches por época, se não informado
        if batches_per_epoch is None:
            total_samples = sum(len(v) for v in self.indices_by_label.values())
            approx = total_samples // self.batch_size
            self.batches_per_epoch = max(1, approx)
        else:
            self.batches_per_epoch = max(1, int(batches_per_epoch))

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        labels = self.labels.copy()
        if self.shuffle:
            random.shuffle(labels)

        # cursor circular de labels
        label_ptr = 0
        n_labels = len(labels)

        for _ in range(self.batches_per_epoch):
            # Seleciona P identidades (sem repetição dentro do batch)
            if label_ptr + self.p > n_labels:
                # reinicia e reembaralha para próxima janela
                if self.shuffle:
                    random.shuffle(labels)
                label_ptr = 0

            selected_labels = labels[label_ptr:label_ptr + self.p]
            label_ptr += self.p

            # Se por qualquer motivo faltou (ex: n_labels < p, já barrado), completa circularmente
            if len(selected_labels) < self.p:
                rem = self.p - len(selected_labels)
                extra = labels[:rem]
                selected_labels.extend(extra)
                label_ptr = rem

            batch_indices: List[int] = []

            for lab in selected_labels:
                pool = self.indices_by_label[lab]
                if len(pool) >= self.k:
                    chosen = random.sample(pool, self.k)
                else:
                    # amostragem com reposição
                    chosen = [random.choice(pool) for _ in range(self.k)]
                batch_indices.extend(chosen)

            if self.shuffle:
                random.shuffle(batch_indices)

            if len(batch_indices) == self.batch_size:
                yield batch_indices
            elif not self.drop_last and len(batch_indices) > 0:
                yield batch_indices

# Helper para DataLoader de treino

@dataclass(frozen=True)
class VerificationLoaderConfig:
    root_dir: str
    input_hw: Tuple[int, int] = (112, 112)
    augment: bool = True
    min_images_per_identity: int = 2
    recursive_inside_identity: bool = True
    center_crop_square: bool = False
    num_workers: int = 4

    # PK sampler
    p_identities: int = 8
    k_images_per_identity: int = 4
    batches_per_epoch: Optional[int] = None

    # fallback sem PK (não recomendado para triplet semihard)
    batch_size: Optional[int] = None
    shuffle: bool = True

def build_verification_dataloader(cfg: VerificationLoaderConfig) -> Tuple[FaceVerificationDataset, DataLoader]:
    dataset = FaceVerificationDataset(
        root_dir=cfg.root_dir,
        input_hw=cfg.input_hw,
        augment=cfg.augment,
        aug_cfg=VerificationAugConfig(),
        min_images_per_identity=cfg.min_images_per_identity,
        recursive_inside_identity=cfg.recursive_inside_identity,
        center_crop_square=cfg.center_crop_square,
    )

    if cfg.batch_size is not None:
        # Modo normal (não ideal para semihard)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=verification_collate_fn,
            drop_last=True,
        )
        return dataset, loader

    sampler = PKBatchSampler(
        indices_by_label=dataset.indices_by_label,
        p=cfg.p_identities,
        k=cfg.k_images_per_identity,
        batches_per_epoch=cfg.batches_per_epoch,
        shuffle=cfg.shuffle,
        drop_last=True,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=verification_collate_fn,
    )
    return dataset, loader