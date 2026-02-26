from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader

from src.common.utils import set_seed, get_device, ensure_dir, SimpleLogger, torch_save, torch_load, AvgMeter
from src.common.boxes import cxcywh_to_xyxy
from src.detection.model import TinySSD, DetectorConfig
from src.detection.loss import multibox_loss, LossConfig
from src.detection.dataset import FaceDetectionDataset, detection_collate_fn, AugConfig

def _build_dataloader(
    root_dir: str,
    ann_path: str,
    input_hw: tuple[int, int],
    batch_size: int,
    num_workers: int,
    augment: bool,
    annotation_format: str,
    aug_cfg: AugConfig,
    shuffle: bool,
) -> DataLoader:
    ds = FaceDetectionDataset(
        root_dir=root_dir,
        annotations=ann_path,
        input_hw=input_hw,
        augment=augment,
        aug_cfg=aug_cfg,
        annotation_format=annotation_format,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=detection_collate_fn,
        drop_last=False,
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
    if optimizer is not None and "optimizer" in state and state["optimizer"] is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and "scaler" in state and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])
    return state

@torch.no_grad()
def _validate(
    model: TinySSD,
    loader: DataLoader,
    device: torch.device,
    anchors_cxcywh: torch.Tensor,
    anchors_xyxy: torch.Tensor,
    loss_cfg: LossConfig,
    amp: bool,
) -> Dict[str, float]:
    model.eval()
    m_total = AvgMeter()
    m_cls = AvgMeter()
    m_reg = AvgMeter()
    m_pos = AvgMeter()

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        gt_boxes_list = batch["gt_boxes"]

        with torch.cuda.amp.autocast(enabled=amp):
            cls_logits, bbox_deltas = model(images)
            losses = multibox_loss(
                cls_logits=cls_logits,
                bbox_deltas=bbox_deltas,
                anchors_cxcywh=anchors_cxcywh,
                anchors_xyxy=anchors_xyxy,
                gt_boxes_xyxy_list=gt_boxes_list,
                cfg=loss_cfg,
            )

        loss_total = float(losses["loss_total"].item())
        loss_cls = float(losses["loss_cls"].item())
        loss_reg = float(losses["loss_reg"].item())
        num_pos = float(losses["num_pos"].item())

        bsz = images.size(0)
        m_total.update(loss_total, bsz)
        m_cls.update(loss_cls, bsz)
        m_reg.update(loss_reg, bsz)
        m_pos.update(num_pos, bsz)

    return {
        "val_loss_total": m_total.avg,
        "val_loss_cls": m_cls.avg,
        "val_loss_reg": m_reg.avg,
        "val_num_pos": m_pos.avg,
    }

def train(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)
    logger = SimpleLogger(os.path.join(args.out_dir, "train.log"))

    if args.seed is not None:
        set_seed(args.seed)

    device = get_device(args.device)

    det_cfg = DetectorConfig(
        input_hw=(args.input_h, args.input_w),
        base_c=args.base_c,
        act=args.act,
    )
    model = TinySSD(det_cfg).to(device)

    # Anchors (fixos pro input size)
    anchors_cxcywh = model.generate_anchors(device=device, dtype=torch.float32)  # (A,4) normalizado
    anchors_xyxy = cxcywh_to_xyxy(anchors_cxcywh).clamp(0.0, 1.0)               # (A,4)

    # Loss config
    loss_cfg = LossConfig(
        pos_iou=args.pos_iou,
        neg_iou=args.neg_iou,
        neg_pos_ratio=args.neg_pos_ratio,
        variances=(args.var_xy, args.var_wh),
        cls_weight=args.cls_weight,
        reg_weight=args.reg_weight,
        smooth_l1_beta=args.smooth_l1_beta,
    )

    # Data
    aug_cfg = AugConfig(
        hflip_prob=args.aug_hflip,
        brightness_prob=args.aug_brightness_prob,
        contrast_prob=args.aug_contrast_prob,
        blur_prob=args.aug_blur_prob,
        max_brightness_delta=args.aug_brightness_delta,
        contrast_range=(args.aug_contrast_min, args.aug_contrast_max),
        blur_ksize_choices=(3, 5),
    )

    train_loader = _build_dataloader(
        root_dir=args.train_root,
        ann_path=args.train_ann,
        input_hw=det_cfg.input_hw,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
        annotation_format=args.ann_format,
        aug_cfg=aug_cfg,
        shuffle=True,
    )

    val_loader = None
    if args.val_root and args.val_ann:
        val_loader = _build_dataloader(
            root_dir=args.val_root,
            ann_path=args.val_ann,
            input_hw=det_cfg.input_hw,
            batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            augment=False,
            annotation_format=args.ann_format,
            aug_cfg=aug_cfg,  # não usado, mas mantém assinatura
            shuffle=False,
        )

    # Optimizer
    if args.optim.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    else:
        raise ValueError("optim deve ser adamw ou sgd")

    # Scheduler (cosine simples por epoch)
    def lr_at_epoch(epoch: int) -> float:
        if args.scheduler.lower() == "cosine":
            import math
            t = min(max(epoch, 0), args.epochs)
            return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1.0 + math.cos(math.pi * t / args.epochs))
        if args.scheduler.lower() == "step":
            # decai a cada step_size
            factor = args.lr_gamma ** (epoch // args.lr_step)
            return args.lr * factor
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
        logger.log(f"Resumindo de: {args.resume}")
        st = _load_checkpoint(args.resume, model, optimizer=optimizer, scaler=scaler, map_location="cpu")
        start_epoch = int(st.get("epoch", 0) + 1)
        global_step = int(st.get("step", 0))
        best_val = float(st.get("best_val", float("inf")))
    elif args.auto_resume and os.path.isfile(last_ckpt):
        logger.log(f"Auto-resume de: {last_ckpt}")
        st = _load_checkpoint(last_ckpt, model, optimizer=optimizer, scaler=scaler, map_location="cpu")
        start_epoch = int(st.get("epoch", 0) + 1)
        global_step = int(st.get("step", 0))
        best_val = float(st.get("best_val", float("inf")))

    # Log configs
    logger.log(f"device={device} amp={use_amp}")
    logger.log(f"det_cfg={asdict(det_cfg)}")
    logger.log(f"loss_cfg={asdict(loss_cfg)}")
    logger.log(f"train: root={args.train_root} ann={args.train_ann} batch={args.batch_size}")
    if val_loader is not None:
        logger.log(f"val: root={args.val_root} ann={args.val_ann} batch={args.val_batch_size}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()

        # set lr
        lr_epoch = lr_at_epoch(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_epoch

        m_total = AvgMeter()
        m_cls = AvgMeter()
        m_reg = AvgMeter()
        m_pos = AvgMeter()

        for it, batch in enumerate(train_loader):
            images = batch["images"].to(device, non_blocking=True)
            gt_boxes_list = batch["gt_boxes"]

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                cls_logits, bbox_deltas = model(images)

                losses = multibox_loss(
                    cls_logits=cls_logits,
                    bbox_deltas=bbox_deltas,
                    anchors_cxcywh=anchors_cxcywh,
                    anchors_xyxy=anchors_xyxy,
                    gt_boxes_xyxy_list=gt_boxes_list,
                    cfg=loss_cfg,
                )

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

            # metrics
            bsz = images.size(0)
            m_total.update(float(losses["loss_total"].item()), bsz)
            m_cls.update(float(losses["loss_cls"].item()), bsz)
            m_reg.update(float(losses["loss_reg"].item()), bsz)
            m_pos.update(float(losses["num_pos"].item()), bsz)

            global_step += 1

            if global_step % args.log_every == 0:
                logger.log(
                    f"epoch={epoch}/{args.epochs-1} step={global_step} "
                    f"lr={lr_epoch:.6g} "
                    f"loss={m_total.avg:.4f} cls={m_cls.avg:.4f} reg={m_reg.avg:.4f} "
                    f"pos(avg/batch)={m_pos.avg:.2f}"
                )

            if args.save_every_steps > 0 and (global_step % args.save_every_steps == 0):
                cfg_dump = vars(args).copy()
                _save_checkpoint(last_ckpt, model, optimizer, scaler if use_amp else None, epoch, global_step, best_val, cfg_dump)

        # end epoch -> save last
        cfg_dump = vars(args).copy()
        _save_checkpoint(last_ckpt, model, optimizer, scaler if use_amp else None, epoch, global_step, best_val, cfg_dump)

        # validate
        if val_loader is not None and (epoch % args.val_every == 0):
            stats = _validate(
                model=model,
                loader=val_loader,
                device=device,
                anchors_cxcywh=anchors_cxcywh,
                anchors_xyxy=anchors_xyxy,
                loss_cfg=loss_cfg,
                amp=use_amp,
            )
            val_loss = stats["val_loss_total"]
            logger.log(
                f"[VAL] epoch={epoch} val_loss={val_loss:.4f} "
                f"cls={stats['val_loss_cls']:.4f} reg={stats['val_loss_reg']:.4f} "
                f"pos(avg/batch)={stats['val_num_pos']:.2f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                _save_checkpoint(best_ckpt, model, optimizer, scaler if use_amp else None, epoch, global_step, best_val, cfg_dump)
                logger.log(f"Novo melhor checkpoint salvo em: {best_ckpt} (best_val={best_val:.4f})")

    logger.log("Treinamento finalizado.")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("train_tinyssd_faces")

    # paths
    p.add_argument("--train_root", type=str, required=True, help="Pasta base das imagens de treino")
    p.add_argument("--train_ann", type=str, required=True, help="Arquivo de anotação (json simples ou widerface txt)")
    p.add_argument("--val_root", type=str, default="", help="Pasta base das imagens de validação (opcional)")
    p.add_argument("--val_ann", type=str, default="", help="Arquivo de anotação de validação (opcional)")
    p.add_argument("--ann_format", type=str, default="auto", help="auto | simple_json | widerface")
    p.add_argument("--out_dir", type=str, required=True, help="Diretório de saída (checkpoints/logs)")

    # model
    p.add_argument("--input_h", type=int, default=320)
    p.add_argument("--input_w", type=int, default=320)
    p.add_argument("--base_c", type=int, default=32)
    p.add_argument("--act", type=str, default="relu", help="relu | silu")

    # training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--val_batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr_min", type=float, default=2e-6)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--optim", type=str, default="adamw", help="adamw | sgd")
    p.add_argument("--scheduler", type=str, default="cosine", help="cosine | step | none")
    p.add_argument("--lr_step", type=int, default=10)
    p.add_argument("--lr_gamma", type=float, default=0.3)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # loss cfg
    p.add_argument("--pos_iou", type=float, default=0.5)
    p.add_argument("--neg_iou", type=float, default=0.4)
    p.add_argument("--neg_pos_ratio", type=int, default=3)
    p.add_argument("--var_xy", type=float, default=0.1)
    p.add_argument("--var_wh", type=float, default=0.2)
    p.add_argument("--cls_weight", type=float, default=1.0)
    p.add_argument("--reg_weight", type=float, default=1.0)
    p.add_argument("--smooth_l1_beta", type=float, default=1.0)

    # aug
    p.add_argument("--aug_hflip", type=float, default=0.5)
    p.add_argument("--aug_brightness_prob", type=float, default=0.3)
    p.add_argument("--aug_contrast_prob", type=float, default=0.3)
    p.add_argument("--aug_blur_prob", type=float, default=0.15)
    p.add_argument("--aug_brightness_delta", type=float, default=0.2)
    p.add_argument("--aug_contrast_min", type=float, default=0.8)
    p.add_argument("--aug_contrast_max", type=float, default=1.2)

    # misc
    p.add_argument("--device", type=str, default="cuda", help="cuda | cpu")
    p.add_argument("--amp", action="store_true", help="Habilita mixed precision (cuda)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every_steps", type=int, default=0, help="0 desativa; >0 salva last.pt a cada N steps")

    # validation frequency
    p.add_argument("--val_every", type=int, default=1, help="validar a cada N epochs")

    # resume
    p.add_argument("--resume", type=str, default="", help="Caminho de checkpoint para resumir")
    p.add_argument("--auto_resume", action="store_true", help="Se existir out_dir/last.pt, resume automaticamente")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)