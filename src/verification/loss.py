from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Utilitários de distância / similaridade

def pairwise_cosine_similarity(embeddings: torch.Tensor) -> torch.Tensor:
    """
    embeddings: (B,D), idealmente já L2-normalizados
    retorna: (B,B) com cosine similarity
    """
    return embeddings @ embeddings.t()

def pairwise_squared_euclidean(embeddings: torch.Tensor) -> torch.Tensor:
    """
    embeddings: (B,D)
    retorna: (B,B) com distância euclidiana ao quadrado
    """
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b
    norms = (embeddings ** 2).sum(dim=1, keepdim=True)  # (B,1)
    dist2 = norms + norms.t() - 2.0 * (embeddings @ embeddings.t())
    return dist2.clamp(min=0.0)

def pairwise_euclidean(embeddings: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    embeddings: (B,D)
    retorna: (B,B) com distância euclidiana
    """
    dist2 = pairwise_squared_euclidean(embeddings)
    return torch.sqrt(dist2 + eps)

# Contrastive Loss (pares)

@dataclass(frozen=True)
class ContrastiveLossConfig:
    margin: float = 1.0
    positive_weight: float = 1.0
    negative_weight: float = 1.0
    distance: str = "euclidean"  # "euclidean" | "sqeuclidean" | "cosine"

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss para pares:
      y = 1 (mesma identidade):   L = d^2
      y = 0 (identidades diferentes): L = max(0, margin - d)^2

    Suporta:
      - euclidean
      - sqeuclidean
      - cosine (usa d = 1 - cos)
    """
    def __init__(self, cfg: ContrastiveLossConfig = ContrastiveLossConfig()):
        super().__init__()
        self.cfg = cfg

    def _distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.cfg.distance == "euclidean":
            return torch.norm(a - b, p=2, dim=1)
        elif self.cfg.distance == "sqeuclidean":
            d = a - b
            return (d * d).sum(dim=1)
        elif self.cfg.distance == "cosine":
            # embeddings já normalizados => cos = dot
            cos = (a * b).sum(dim=1).clamp(-1.0, 1.0)
            return 1.0 - cos
        else:
            raise ValueError(f"distance inválida: {self.cfg.distance}")

    def forward(
        self,
        emb1: torch.Tensor,     # (N,D)
        emb2: torch.Tensor,     # (N,D)
        target: torch.Tensor,   # (N,) 1=mesma pessoa, 0=diferente
    ) -> Dict[str, torch.Tensor]:
        target = target.float()
        d = self._distance(emb1, emb2)

        pos_mask = target == 1
        neg_mask = target == 0

        # positivos
        pos_loss = d[pos_mask] ** 2 if pos_mask.any() else torch.zeros((0,), device=d.device, dtype=d.dtype)

        # negativos
        neg_margin = (self.cfg.margin - d[neg_mask]).clamp(min=0.0)
        neg_loss = neg_margin ** 2 if neg_mask.any() else torch.zeros((0,), device=d.device, dtype=d.dtype)

        # médias separadas (evita viés se lote tiver desbalanceado)
        pos_mean = pos_loss.mean() if pos_loss.numel() > 0 else torch.zeros((), device=d.device, dtype=d.dtype)
        neg_mean = neg_loss.mean() if neg_loss.numel() > 0 else torch.zeros((), device=d.device, dtype=d.dtype)

        loss = self.cfg.positive_weight * pos_mean + self.cfg.negative_weight * neg_mean

        return {
            "loss_total": loss,
            "loss_pos": pos_mean,
            "loss_neg": neg_mean,
            "num_pos_pairs": torch.tensor(int(pos_mask.sum().item()), device=d.device, dtype=torch.float32),
            "num_neg_pairs": torch.tensor(int(neg_mask.sum().item()), device=d.device, dtype=torch.float32),
        }

# Triplet Loss (triplets já prontos)

@dataclass(frozen=True)
class TripletLossConfig:
    margin: float = 0.2
    distance: str = "euclidean"  # "euclidean" | "sqeuclidean" | "cosine"
    reduction: str = "mean"      # "mean" | "sum"


class TripletLoss(nn.Module):
    """
    Loss para triplets explícitos (anchor, positive, negative):
      L = max(0, d(a,p) - d(a,n) + margin)
    """
    def __init__(self, cfg: TripletLossConfig = TripletLossConfig()):
        super().__init__()
        self.cfg = cfg

    def _distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.cfg.distance == "euclidean":
            return torch.norm(a - b, p=2, dim=1)
        elif self.cfg.distance == "sqeuclidean":
            d = a - b
            return (d * d).sum(dim=1)
        elif self.cfg.distance == "cosine":
            cos = (a * b).sum(dim=1).clamp(-1.0, 1.0)
            return 1.0 - cos
        else:
            raise ValueError(f"distance inválida: {self.cfg.distance}")

    def forward(
        self,
        anchor: torch.Tensor,    # (N,D)
        positive: torch.Tensor,  # (N,D)
        negative: torch.Tensor,  # (N,D)
    ) -> Dict[str, torch.Tensor]:
        d_ap = self._distance(anchor, positive)
        d_an = self._distance(anchor, negative)

        losses = (d_ap - d_an + self.cfg.margin).clamp(min=0.0)

        if self.cfg.reduction == "mean":
            loss = losses.mean() if losses.numel() > 0 else torch.zeros((), device=anchor.device, dtype=anchor.dtype)
        elif self.cfg.reduction == "sum":
            loss = losses.sum()
        else:
            raise ValueError(f"reduction inválida: {self.cfg.reduction}")

        active = (losses > 0).float()
        return {
            "loss_total": loss,
            "mean_d_ap": d_ap.mean() if d_ap.numel() > 0 else torch.zeros((), device=anchor.device, dtype=anchor.dtype),
            "mean_d_an": d_an.mean() if d_an.numel() > 0 else torch.zeros((), device=anchor.device, dtype=anchor.dtype),
            "triplets_active_ratio": active.mean() if active.numel() > 0 else torch.zeros((), device=anchor.device, dtype=anchor.dtype),
            "num_triplets": torch.tensor(float(anchor.size(0)), device=anchor.device, dtype=anchor.dtype),
        }

# Semi-hard triplet mining em batch (a partir de labels)

@dataclass(frozen=True)
class BatchSemiHardTripletConfig:
    margin: float = 0.2
    distance: str = "euclidean"  # "euclidean" | "sqeuclidean" | "cosine"
    reduction: str = "mean"      # "mean" | "sum"

class BatchSemiHardTripletLoss(nn.Module):
    """
    Gera triplets dentro do batch usando labels e aplica triplet loss com semi-hard negatives.

    Definição (semi-hard):
      escolhe negativo n tal que:
        d(a,p) < d(a,n) < d(a,p) + margin
      se não existir, usa hard/closest negative disponível (fallback).

    Requisitos do batch:
      - labels com pelo menos 2 amostras para algumas identidades
      - idealmente batch organizado com múltiplas imagens por identidade
    """
    def __init__(self, cfg: BatchSemiHardTripletConfig = BatchSemiHardTripletConfig()):
        super().__init__()
        self.cfg = cfg

    def _pairwise_distance(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.cfg.distance == "euclidean":
            return pairwise_euclidean(embeddings)
        elif self.cfg.distance == "sqeuclidean":
            return pairwise_squared_euclidean(embeddings)
        elif self.cfg.distance == "cosine":
            # distância = 1 - cos
            sim = pairwise_cosine_similarity(embeddings).clamp(-1.0, 1.0)
            return (1.0 - sim).clamp(min=0.0)
        else:
            raise ValueError(f"distance inválida: {self.cfg.distance}")

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        embeddings: (B,D), idealmente L2-normalizados
        labels: (B,) int/long (id da pessoa)
        """
        device = embeddings.device
        B = embeddings.size(0)
        labels = labels.view(-1)

        if B < 3:
            z = torch.zeros((), device=device, dtype=embeddings.dtype)
            return {
                "loss_total": z,
                "mean_d_ap": z,
                "mean_d_an": z,
                "triplets_active_ratio": z,
                "num_triplets": z,
                "num_anchors_used": z,
            }

        dmat = self._pairwise_distance(embeddings)  # (B,B)

        triplet_losses = []
        d_ap_list = []
        d_an_list = []
        anchors_used = 0

        eye = torch.eye(B, device=device, dtype=torch.bool)

        for a in range(B):
            same = labels == labels[a]
            diff = ~same

            # remove o próprio anchor dos positivos
            pos_mask = same & (~eye[a])
            neg_mask = diff

            pos_idx = torch.where(pos_mask)[0]
            neg_idx = torch.where(neg_mask)[0]

            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue

            anchors_used += 1

            d_ap_all = dmat[a, pos_idx]  # distâncias anchor->positivos
            d_an_all = dmat[a, neg_idx]  # distâncias anchor->negativos

            for p_local, p in enumerate(pos_idx):
                d_ap = d_ap_all[p_local]

                # semi-hard: d_ap < d_an < d_ap + margin
                semihard_mask = (d_an_all > d_ap) & (d_an_all < (d_ap + self.cfg.margin))

                if semihard_mask.any():
                    # escolhe o semihard mais "difícil" (mais próximo de d_ap+margin), ou mais perto do anchor
                    # aqui escolhemos o mais próximo do anchor para aumentar dificuldade
                    cand = d_an_all[semihard_mask]
                    d_an = cand.min()
                else:
                    # fallback: hard negative (mais próximo do anchor)
                    d_an = d_an_all.min()

                loss_apn = (d_ap - d_an + self.cfg.margin).clamp(min=0.0)

                triplet_losses.append(loss_apn)
                d_ap_list.append(d_ap)
                d_an_list.append(d_an)

        if len(triplet_losses) == 0:
            z = torch.zeros((), device=device, dtype=embeddings.dtype)
            return {
                "loss_total": z,
                "mean_d_ap": z,
                "mean_d_an": z,
                "triplets_active_ratio": z,
                "num_triplets": z,
                "num_anchors_used": torch.tensor(float(anchors_used), device=device, dtype=embeddings.dtype),
            }

        losses_t = torch.stack(triplet_losses)
        d_ap_t = torch.stack(d_ap_list)
        d_an_t = torch.stack(d_an_list)

        if self.cfg.reduction == "mean":
            loss = losses_t.mean()
        elif self.cfg.reduction == "sum":
            loss = losses_t.sum()
        else:
            raise ValueError(f"reduction inválida: {self.cfg.reduction}")

        active = (losses_t > 0).float()

        return {
            "loss_total": loss,
            "mean_d_ap": d_ap_t.mean(),
            "mean_d_an": d_an_t.mean(),
            "triplets_active_ratio": active.mean(),
            "num_triplets": torch.tensor(float(losses_t.numel()), device=device, dtype=embeddings.dtype),
            "num_anchors_used": torch.tensor(float(anchors_used), device=device, dtype=embeddings.dtype),
        }

@dataclass(frozen=True)
class ArcFaceConfig:
    in_features: int
    out_classes: int
    s: float = 64.0
    m: float = 0.50

class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace).
    Trata o problema como classificação de N classes, otimizando a margem angular.
    """
    def __init__(self, cfg: ArcFaceConfig):
        super().__init__()
        self.cfg = cfg
        self.s = cfg.s
        self.m = cfg.m
        
        # Matriz de pesos que representa os "centros" de cada identidade (N classes x Dimensão do Embedding)
        self.weight = nn.Parameter(torch.FloatTensor(cfg.out_classes, cfg.in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. Força float32 para evitar underflow/overflow do AMP
        embeddings = embeddings.float()
        norm_weight = F.normalize(self.weight.float(), p=2, dim=1)
        
        # Similaridade por cosseno
        cosine = F.linear(embeddings, norm_weight)
        
        # 2. A CORREÇÃO DO NaN: clamp com 1e-7 em vez de 0 evita a divisão por zero no backward
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(1e-7, 1.0))

        # Adiciona a margem angular
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encoding dos labels garantindo o tipo e o device corretos
        one_hot = torch.zeros(cosine.size(), device=embeddings.device, dtype=embeddings.dtype)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        # Aplica a margem apenas na classe correta
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        # Cross Entropy Loss
        loss = F.cross_entropy(output, labels)

        # Retorna chaves dummy
        z = torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)
        return {
            "loss_total": loss,
            "mean_d_ap": z,
            "mean_d_an": z,
            "triplets_active_ratio": z,
            "num_triplets": torch.tensor(float(embeddings.size(0)), device=embeddings.device, dtype=embeddings.dtype),
        }

# Fábrica / wrapper opcional

@dataclass(frozen=True)
class VerificationLossFactoryConfig:
    mode: str = "batch_semihard_triplet"
    margin: float = 0.2
    distance: str = "euclidean"
    reduction: str = "mean"
    
    # Configs exclusivas para ArcFace
    emb_dim: int = 512
    num_classes: int = 0
    arcface_s: float = 64.0
    arcface_m: float = 0.50

def build_verification_loss(cfg: VerificationLossFactoryConfig) -> nn.Module:
    mode = cfg.mode.lower()

    if mode == "arcface":
        if cfg.num_classes <= 0:
            raise ValueError("Para usar arcface, num_classes deve ser > 0 (fornecido pelo dataset).")
        return ArcFaceLoss(
            ArcFaceConfig(
                in_features=cfg.emb_dim,
                out_classes=cfg.num_classes,
                s=cfg.arcface_s,
                m=cfg.arcface_m,
            )
        )
    elif mode == "batch_semihard_triplet":
        return BatchSemiHardTripletLoss(
            BatchSemiHardTripletConfig(margin=cfg.margin, distance=cfg.distance, reduction=cfg.reduction)
        )
    elif mode == "triplet":
        return TripletLoss(
            TripletLossConfig(margin=cfg.margin, distance=cfg.distance, reduction=cfg.reduction)
        )
    elif mode == "contrastive":
        return ContrastiveLoss(
            ContrastiveLossConfig(margin=cfg.margin, distance=cfg.distance)
        )
    else:
        raise ValueError(f"mode inválido: {cfg.mode}")