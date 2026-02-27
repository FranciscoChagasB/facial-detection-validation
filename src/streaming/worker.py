# src/streaming/worker.py
from __future__ import annotations

import os
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, List

import multiprocessing as mp
import numpy as np
import cv2

from src.runtime.config import RuntimeConfig
from src.runtime.service import FaceRuntimeService, FrameResult, FaceDetectionResult, FaceMatch
from src.streaming.ingest import FramePacket


@dataclass(frozen=True)
class WorkerConfig:
    worker_id: str = "worker-01"
    drain_max: int = 8
    per_camera_min_interval_s: float = 0.4
    drop_same_stream_when_backlog: bool = True
    match_debounce_s: float = 10.0

    # se True: só emite eventos quando match aprovado
    # se False: também emite "UNKNOWN" para debug (mesmo sem galeria/match)
    only_matches: bool = True

    save_images: bool = False
    images_dir: str = "runs/events_images"
    jpg_quality: int = 90
    include_crop_path: bool = True


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _hash_short(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def _save_jpg(path: str, bgr: np.ndarray, quality: int = 90) -> None:
    _safe_mkdir(os.path.dirname(path) or ".")
    cv2.imwrite(path, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)], bgr)


def _now() -> float:
    return time.time()


@dataclass
class MatchEvent:
    camera_id: str
    camera_url: str
    timestamp: float
    box_xyxy_px: Tuple[float, float, float, float]
    det_score: float
    identity: str
    similarity: float
    metric: str
    threshold: float
    frame_path: str = ""
    crop_path: str = ""


class StreamingWorker:
    def __init__(
        self,
        runtime_cfg: RuntimeConfig,
        worker_cfg: WorkerConfig,
        in_queue: mp.Queue,
        out_queue: mp.Queue,
        frames_proc: Optional[mp.Value] = None,
        events_out: Optional[mp.Value] = None,
    ):
        self.runtime_cfg = runtime_cfg
        self.worker_cfg = worker_cfg
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.frames_proc = frames_proc
        self.events_out = events_out

        self.svc = FaceRuntimeService(runtime_cfg)
        self.svc.load_all()

        self._last_processed_by_cam: Dict[str, float] = {}
        self._last_match_by_cam_id: Dict[Tuple[str, str], float] = {}

        if worker_cfg.save_images:
            _safe_mkdir(worker_cfg.images_dir)

    def _inc_frames_proc(self) -> None:
        if self.frames_proc is None:
            return
        try:
            with self.frames_proc.get_lock():
                self.frames_proc.value += 1
        except Exception:
            pass

    def _inc_events_out(self) -> None:
        if self.events_out is None:
            return
        try:
            with self.events_out.get_lock():
                self.events_out.value += 1
        except Exception:
            pass

    def _should_process_camera(self, camera_id: str, ts: float) -> bool:
        last = self._last_processed_by_cam.get(camera_id, 0.0)
        if (ts - last) < self.worker_cfg.per_camera_min_interval_s:
            return False
        self._last_processed_by_cam[camera_id] = ts
        return True

    def _debounce_match(self, camera_id: str, identity: str, ts: float) -> bool:
        key = (camera_id, identity)
        last = self._last_match_by_cam_id.get(key, 0.0)
        if (ts - last) < self.worker_cfg.match_debounce_s:
            return False
        self._last_match_by_cam_id[key] = ts
        return True

    def _emit_event(
        self,
        pkt: FramePacket,
        det: FaceDetectionResult,
        frame_bgr: np.ndarray,
    ) -> None:
        assert det.match is not None
        m: FaceMatch = det.match
        assert m.identity is not None

        frame_path = ""
        crop_path = ""

        if self.worker_cfg.save_images:
            cam_hash = _hash_short(pkt.stream_id)
            base = os.path.join(self.worker_cfg.images_dir, f"{cam_hash}")
            _safe_mkdir(base)

            ts_int = int(pkt.ts * 1000)
            frame_path = os.path.join(base, f"{ts_int}_frame.jpg")
            cv2.imwrite(frame_path, frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.worker_cfg.jpg_quality)])

            if self.worker_cfg.include_crop_path:
                x1, y1, x2, y2 = map(int, det.box_xyxy_px)
                h, w = frame_bgr.shape[:2]
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w - 1, x2))
                y2 = max(0, min(h - 1, y2))
                if x2 > x1 and y2 > y1:
                    crop = frame_bgr[y1:y2, x1:x2]
                    crop_path = os.path.join(base, f"{ts_int}_crop_{m.identity}.jpg")
                    cv2.imwrite(crop_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.worker_cfg.jpg_quality)])

        event = MatchEvent(
            camera_id=pkt.stream_id,
            camera_url=pkt.url,
            timestamp=pkt.ts,
            box_xyxy_px=det.box_xyxy_px,
            det_score=float(det.det_score),
            identity=m.identity,
            similarity=float(m.score),
            metric=m.metric,
            threshold=float(m.threshold),
            frame_path=frame_path,
            crop_path=crop_path,
        )

        try:
            self.out_queue.put_nowait(event)
            self._inc_events_out()
        except Exception:
            pass

    def process_packet(self, pkt: FramePacket) -> None:
        if not self._should_process_camera(pkt.stream_id, pkt.ts):
            return

        # conta "processado" quando realmente vamos rodar inferência
        self._inc_frames_proc()

        res: FrameResult = self.svc.process_frame(
            frame_bgr=pkt.frame_bgr,
            camera_id=pkt.stream_id,
            timestamp=pkt.ts,
            return_only_matches=self.worker_cfg.only_matches,
        )

        if not res.detections:
            return

        for det in res.detections:
            # se não tem match e only_matches=False, emite UNKNOWN (debug)
            if det.match is None or det.match.identity is None:
                if self.worker_cfg.only_matches:
                    continue
                pseudo = FaceMatch(identity="UNKNOWN", score=0.0, is_match=False, metric="cosine", threshold=0.0)
                det = FaceDetectionResult(box_xyxy_px=det.box_xyxy_px, det_score=det.det_score, match=pseudo)
                self._emit_event(pkt, det, pkt.frame_bgr)
                continue

            # se tem match real:
            if self.worker_cfg.only_matches and not det.match.is_match:
                continue

            if not self._debounce_match(pkt.stream_id, det.match.identity, pkt.ts):
                continue

            self._emit_event(pkt, det, pkt.frame_bgr)

    def run_forever(self) -> None:
        while True:
            pkt = self.in_queue.get()
            if pkt is None:
                break

            try:
                self.process_packet(pkt)
            except Exception:
                pass

            drained = 0
            while drained < self.worker_cfg.drain_max:
                try:
                    pkt2 = self.in_queue.get_nowait()
                except Exception:
                    break
                if pkt2 is None:
                    return

                try:
                    if self.worker_cfg.drop_same_stream_when_backlog:
                        if (pkt2.ts - self._last_processed_by_cam.get(pkt2.stream_id, 0.0)) < (self.worker_cfg.per_camera_min_interval_s * 0.5):
                            drained += 1
                            continue
                    self.process_packet(pkt2)
                except Exception:
                    pass
                drained += 1


def _worker_process_main(
    runtime_cfg_dict: Dict[str, Any],
    worker_cfg_dict: Dict[str, Any],
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    frames_proc: Optional[mp.Value],
    events_out: Optional[mp.Value],
):
    runtime_cfg = RuntimeConfig(**runtime_cfg_dict)
    worker_cfg = WorkerConfig(**worker_cfg_dict)

    worker = StreamingWorker(runtime_cfg, worker_cfg, in_queue, out_queue, frames_proc=frames_proc, events_out=events_out)
    worker.run_forever()


def start_worker_processes(
    runtime_cfg: RuntimeConfig,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    num_workers: int = 1,
    worker_cfg: Optional[WorkerConfig] = None,
    frames_proc: Optional[mp.Value] = None,
    events_out: Optional[mp.Value] = None,
    daemon: bool = True,
) -> List[mp.Process]:
    ctx = mp.get_context("spawn")
    worker_cfg = worker_cfg or WorkerConfig()

    procs: List[mp.Process] = []
    for i in range(num_workers):
        cfg_i = WorkerConfig(**{**worker_cfg.__dict__, "worker_id": f"{worker_cfg.worker_id}-{i+1:02d}"})

        p = ctx.Process(
            target=_worker_process_main,
            args=(
                runtime_cfg.__dict__,
                cfg_i.__dict__,
                in_queue,
                out_queue,
                frames_proc,
                events_out,
            ),
            daemon=daemon,
        )
        p.start()
        procs.append(p)

    return procs