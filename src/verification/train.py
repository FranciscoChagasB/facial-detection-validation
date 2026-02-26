from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader

from src.common.utils import (
    set_seed,
    get_device,
    ensure_dir,
    SimpleLogger,
    torch_save,
    torch_load,
    AvgMeter,
)
from src.verification.model import FaceEmbedNet, EmbedConfig
from src.verification.loss import (
    build_verification_loss,
    VerificationLossFactoryConfig,
)
from src.verification.dataset import (
    FaceVerificationDataset,
    VerificationAugConfig,
    verification_collate_fn,
    PKBatchSampler,
    build_verification_dataloader,
    VerificationLoaderConfig,
)

def _save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    step: int,
    best_val: float,
    cfg: Dict[str, Any],
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "best_val": best_val,
        "cfg": cfg,
    }
    torch_save(path, state)

def _load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    state = torch_load(path, map_location=map_location)
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return state

def _build_train_loader(args: argparse.Namespace):
    loader_cfg = VerificationLoaderConfig(
        root_dir=args.train_root,
        input_hw=(args.input_h, args.input_w),
        augment=True,
        min_images_per_identity=args.min_images_per_identity,
        recursive_inside_identity=not args.no_recursive_identity_scan,
        center_crop_square=args.center_crop_square,
        num_workers=args.num_workers,
        p_identities=args.p_identities,
        k_images_per_identity=args.k_images_per_identity,
        batches_per_epoch=args.batches_per_epoch if args.batches_per_epoch > 0 else None,
        batch_size=None if not args.use_plain_batch else args.batch_size,
        shuffle=True,
    )
    return build_verification_dataloader(loader_cfg)

def _build_val_loader(args: argparse.Namespace):
    """
    Validação usando mesmo formato P x K (sem augment).
    """
    if not args.val_root:
        return None, None

    ds = FaceVerificationDataset(
        root_dir=args.val_root,
        input_hw=(args.input_h, args.input_w),
        augment=False,
        aug_cfg=VerificationAugConfig(),
        min_images_per_identity=args.min_images_per_identity_val,
        recursive_inside_identity=not args.no_recursive_identity_scan,
        center_crop_square=args.center_crop_square,
    )

    if args.use_plain_batch:
        loader = DataLoader(
            ds,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=verification_collate_fn,
            drop_last=True,
        )
        return ds, loader

    sampler = PKBatchSampler(
        indices_by_label=ds.indices_by_label,
        p=args.val_p_identities,
        k=args.val_k_images_per_identity,
        batches_per_epoch=args.val_batches_per_epoch if args.val_batches_per_epoch > 0 else None,
        shuffle=False,
        drop_last=True,
    )

    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=verification_collate_fn,
    )
    return ds, loader

@torch.no_grad()
def _validate(
    model: FaceEmbedNet,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    loss_mode: str,
    device: torch.device,
    amp: bool,
) -> Dict[str, float]:
    model.eval()

    m_total = AvgMeter()
    m_dap = AvgMeter()
    m_dan = AvgMeter()
    m_active = AvgMeter()
    m_triplets = AvgMeter()
    m_batch = AvgMeter()

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp):
            emb = model(images)

            if loss_mode == "batch_semihard_triplet":
                losses = loss_fn(emb, labels)
            else:
                raise ValueError(
                    "Este train.py está preparado para mode=batch_semihard_triplet. "
                    "Para contrastive/triplet explícito, é preciso dataset específico."
                )

        loss_total = float(losses["loss_total"].item())
        mean_d_ap = float(losses.get("mean_d_ap", torch.tensor(0.0, device=device)).item())
        mean_d_an = float(losses.get("mean_d_an", torch.tensor(0.0, device=device)).item())
        active_ratio = float(losses.get("triplets_active_ratio", torch.tensor(0.0, device=device)).item())
        num_triplets = float(losses.get("num_triplets", torch.tensor(0.0, device=device)).item())

        bsz = images.size(0)
        m_total.update(loss_total, bsz)
        m_dap.update(mean_d_ap, bsz)
        m_dan.update(mean_d_an, bsz)
        m_active.update(active_ratio, bsz)
        m_triplets.update(num_triplets, 1)
        m_batch.update(float(bsz), 1)

    return {
        "val_loss_total": m_total.avg,
        "val_mean_d_ap": m_dap.avg,
        "val_mean_d_an": m_dan.avg,
        "val_triplets_active_ratio": m_active.avg,
        "val_num_triplets_avg": m_triplets.avg,
        "val_batch_size_avg": m_batch.avg,
    }

def train(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)
    logger = SimpleLogger(os.path.join(args.out_dir, "train.log"))

    if args.seed is not None:
        set_seed(args.seed)

    device = get_device(args.device)

    # Model
    embed_cfg = EmbedConfig(
        input_hw=(args.input_h, args.input_w),
        base_c=args.base_c,
        emb_dim=args.emb_dim,
        act=args.act,
        use_se=not args.disable_se,
        dropout=args.dropout,
    )
    model = FaceEmbedNet(embed_cfg).to(device)

    # Data
    train_ds, train_loader = _build_train_loader(args)
    val_ds, val_loader = _build_val_loader(args)

    # Loss
    loss_cfg = VerificationLossFactoryConfig(
        mode=args.loss_mode,
        margin=args.margin,
        distance=args.distance,
        reduction=args.reduction,
    )
    loss_fn = build_verification_loss(loss_cfg).to(device)

    if args.loss_mode.lower() != "batch_semihard_triplet":
        raise ValueError(
            "Este train.py está focado em batch_semihard_triplet. "
            "Para contrastive/triplet explícito, adapte o dataset/loader."
        )

    # Optimizer
    if args.optim.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.wd,
            nesterov=True,
        )
    else:
        raise ValueError("optim deve ser: adamw | sgd")

    # LR scheduler (por epoch)
    def lr_at_epoch(epoch: int) -> float:
        sched = args.scheduler.lower()
        if sched == "cosine":
            import math
            t = min(max(epoch, 0), args.epochs)
            return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1.0 + math.cos(math.pi * t / args.epochs))
        elif sched == "step":
            factor = args.lr_gamma ** (epoch // args.lr_step)
            return args.lr * factor
        else:
            return args.lr

    # AMP
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Resume
    start_epoch = 0
    global_step = 0
    best_val = float("inf")

    last_ckpt = os.path.join(args.out_dir, "last.pt")
    best_ckpt = os.path.join(args.out_dir, "best.pt")

    if args.resume and os.path.isfile(args.resume):
        logger.log(f"Resumindo de checkpoint: {args.resume}")
        st = _load_checkpoint(args.resume, model, optimizer=optimizer, scaler=scaler, map_location="cpu")
        start_epoch = int(st.get("epoch", 0)) + 1
        global_step = int(st.get("step", 0))
        best_val = float(st.get("best_val", float("inf")))
    elif args.auto_resume and os.path.isfile(last_ckpt):
        logger.log(f"Auto-resume de checkpoint: {last_ckpt}")
        st = _load_checkpoint(last_ckpt, model, optimizer=optimizer, scaler=scaler, map_location="cpu")
        start_epoch = int(st.get("epoch", 0)) + 1
        global_step = int(st.get("step", 0))
        best_val = float(st.get("best_val", float("inf")))

    # Logs iniciais
    logger.log(f"device={device} amp={use_amp}")
    logger.log(f"embed_cfg={asdict(embed_cfg)}")
    logger.log(f"loss_cfg={asdict(loss_cfg)}")
    logger.log(
        f"train_root={args.train_root} | classes={train_ds.num_classes} | samples={len(train_ds)} | "
        f"batch_mode={'plain' if args.use_plain_batch else 'PK'}"
    )
    if not args.use_plain_batch:
        logger.log(
            f"PK train: P={args.p_identities} K={args.k_images_per_identity} "
            f"batch={args.p_identities * args.k_images_per_identity}"
        )
    if val_loader is not None and val_ds is not None:
        logger.log(f"val_root={args.val_root} | classes={val_ds.num_classes} | samples={len(val_ds)}")

    # Loop de treino
    for epoch in range(start_epoch, args.epochs):
        model.train()

        lr_epoch = lr_at_epoch(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_epoch

        m_total = AvgMeter()
        m_dap = AvgMeter()
        m_dan = AvgMeter()
        m_active = AvgMeter()
        m_triplets = AvgMeter()
        m_batch = AvgMeter()

        for it, batch in enumerate(train_loader):
            images = batch["images"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                emb = model(images)  # (B,D) L2-normalizado

                if args.loss_mode.lower() == "batch_semihard_triplet":
                    losses = loss_fn(emb, labels)
                else:
                    raise ValueError("Modo de loss incompatível com este train.py.")

                loss_total = losses["loss_total"]

            if use_amp:
                scaler.scale(loss_total).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_total.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            bsz = images.size(0)
            m_total.update(float(loss_total.item()), bsz)
            m_dap.update(float(losses.get("mean_d_ap", torch.tensor(0.0, device=device)).item()), bsz)
            m_dan.update(float(losses.get("mean_d_an", torch.tensor(0.0, device=device)).item()), bsz)
            m_active.update(float(losses.get("triplets_active_ratio", torch.tensor(0.0, device=device)).item()), bsz)
            m_triplets.update(float(losses.get("num_triplets", torch.tensor(0.0, device=device)).item()), 1)
            m_batch.update(float(bsz), 1)

            global_step += 1

            if global_step % args.log_every == 0:
                logger.log(
                    f"epoch={epoch}/{args.epochs-1} step={global_step} "
                    f"lr={lr_epoch:.6g} "
                    f"loss={m_total.avg:.4f} "
                    f"d_ap={m_dap.avg:.4f} d_an={m_dan.avg:.4f} "
                    f"active={m_active.avg:.4f} "
                    f"triplets(avg/batch)={m_triplets.avg:.1f} "
                    f"batch(avg)={m_batch.avg:.1f}"
                )

            if args.save_every_steps > 0 and (global_step % args.save_every_steps == 0):
                cfg_dump = vars(args).copy()
                _save_checkpoint(
                    last_ckpt,
                    model,
                    optimizer,
                    scaler if use_amp else None,
                    epoch,
                    global_step,
                    best_val,
                    cfg_dump,
                )

        # salva last no fim da epoch
        cfg_dump = vars(args).copy()
        _save_checkpoint(
            last_ckpt,
            model,
            optimizer,
            scaler if use_amp else None,
            epoch,
            global_step,
            best_val,
            cfg_dump,
        )

        # validação (opcional)
        if val_loader is not None and (epoch % args.val_every == 0):
            stats = _validate(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                loss_mode=args.loss_mode.lower(),
                device=device,
                amp=use_amp,
            )
            val_loss = stats["val_loss_total"]

            logger.log(
                f"[VAL] epoch={epoch} "
                f"loss={val_loss:.4f} "
                f"d_ap={stats['val_mean_d_ap']:.4f} d_an={stats['val_mean_d_an']:.4f} "
                f"active={stats['val_triplets_active_ratio']:.4f} "
                f"triplets(avg/batch)={stats['val_num_triplets_avg']:.1f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                _save_checkpoint(
                    best_ckpt,
                    model,
                    optimizer,
                    scaler if use_amp else None,
                    epoch,
                    global_step,
                    best_val,
                    cfg_dump,
                )
                logger.log(f"Novo melhor checkpoint salvo em: {best_ckpt} (best_val={best_val:.4f})")

    logger.log("Treinamento de verificação finalizado.")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("train_face_verification")

    # paths
    p.add_argument("--train_root", type=str, required=True, help="Pasta com identidades (1 pasta = 1 pessoa)")
    p.add_argument("--val_root", type=str, default="", help="Pasta de validação (mesmo formato) opcional")
    p.add_argument("--out_dir", type=str, required=True, help="Saída para checkpoints e logs")

    # model
    p.add_argument("--input_h", type=int, default=112)
    p.add_argument("--input_w", type=int, default=112)
    p.add_argument("--base_c", type=int, default=64)
    p.add_argument("--emb_dim", type=int, default=256)
    p.add_argument("--act", type=str, default="relu", help="relu | silu")
    p.add_argument("--disable_se", action="store_true", help="Desabilita blocos SE")
    p.add_argument("--dropout", type=float, default=0.0)

    # dataset scan / preprocess
    p.add_argument("--min_images_per_identity", type=int, default=2)
    p.add_argument("--min_images_per_identity_val", type=int, default=2)
    p.add_argument("--no_recursive_identity_scan", action="store_true")
    p.add_argument("--center_crop_square", action="store_true")

    # loader (recomendado: PK)
    p.add_argument("--use_plain_batch", action="store_true", help="Usa batch comum (não recomendado p/ semihard)")
    p.add_argument("--batch_size", type=int, default=32, help="Usado apenas com --use_plain_batch")
    p.add_argument("--val_batch_size", type=int, default=32, help="Usado apenas com --use_plain_batch")

    p.add_argument("--p_identities", type=int, default=8, help="P no PK sampler")
    p.add_argument("--k_images_per_identity", type=int, default=4, help="K no PK sampler")
    p.add_argument("--batches_per_epoch", type=int, default=0, help="0=auto")

    p.add_argument("--val_p_identities", type=int, default=8)
    p.add_argument("--val_k_images_per_identity", type=int, default=4)
    p.add_argument("--val_batches_per_epoch", type=int, default=0, help="0=auto")

    p.add_argument("--num_workers", type=int, default=4)

    # loss (foco: batch semihard triplet)
    p.add_argument("--loss_mode", type=str, default="batch_semihard_triplet",
                   help="batch_semihard_triplet (recomendado)")
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--distance", type=str, default="euclidean", help="euclidean | sqeuclidean | cosine")
    p.add_argument("--reduction", type=str, default="mean", help="mean | sum")

    # otimização
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--optim", type=str, default="adamw", help="adamw | sgd")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_min", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--scheduler", type=str, default="cosine", help="cosine | step | none")
    p.add_argument("--lr_step", type=int, default=10)
    p.add_argument("--lr_gamma", type=float, default=0.3)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # misc
    p.add_argument("--device", type=str, default="cuda", help="cuda | cpu")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every_steps", type=int, default=0, help="0 desativa")
    p.add_argument("--val_every", type=int, default=1)

    # resume
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--auto_resume", action="store_true")

    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)