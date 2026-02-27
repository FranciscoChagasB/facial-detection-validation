# src/streaming/orchestrator.py
from __future__ import annotations

import os
import time
import json
import signal
import base64
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Dict

import multiprocessing as mp

from src.streaming.ingest import StreamSpec, start_ingest_processes
from src.streaming.worker import start_worker_processes, WorkerConfig, MatchEvent

from src.runtime.config import (
    RuntimeConfig,
    DetectorRuntimeConfig,
    VerificationRuntimeConfig,
    CropRuntimeConfig,
)
from src.common.utils import ensure_dir, SimpleLogger


@dataclass(frozen=True)
class OrchestratorConfig:
    frame_queue_maxsize: int = 5000
    event_queue_maxsize: int = 5000

    max_streams_per_process: int = 30
    ingest_fps: float = 3.0
    ingest_out_w: int = 640
    ingest_out_h: int = 360
    ingest_timeout_s: int = 10
    ingest_reconnect_backoff_s: float = 2.0
    ingest_drop_if_full: bool = True

    num_workers: int = 1

    save_events_jsonl: bool = True
    events_out_path: str = "runs/events/events.jsonl"

    log_path: str = "runs/events/orchestrator.log"

    print_every: int = 1
    heartbeat_every_s: float = 5.0


class EventSink:
    def __init__(self, jsonl_path: Optional[str], logger: SimpleLogger):
        self.jsonl_path = jsonl_path
        self.logger = logger
        self._fp = None
        if jsonl_path:
            ensure_dir(os.path.dirname(jsonl_path) or ".")
            self._fp = open(jsonl_path, "a", encoding="utf-8")

    def close(self):
        if self._fp:
            try:
                self._fp.close()
            except Exception:
                pass
            self._fp = None

    def write(self, ev: MatchEvent):
        data = asdict(ev)
        if self._fp:
            self._fp.write(json.dumps(data, ensure_ascii=False) + "\n")
            self._fp.flush()

        self.logger.log(
            f"[EVENT] cam={ev.camera_id} id={ev.identity} sim={ev.similarity:.4f} "
            f"det={ev.det_score:.3f} box={tuple(int(x) for x in ev.box_xyxy_px)}"
        )


def _build_basic_auth_headers_from_env() -> Optional[Dict[str, str]]:
    """
    Lê HLS_USER/HLS_PASS do ambiente e retorna headers para Basic Auth.
    Se não existir, retorna None.
    """
    user = os.getenv("HLS_USER", "").strip()
    pwd = os.getenv("HLS_PASS", "").strip()
    if not user or not pwd:
        return None
    token = base64.b64encode(f"{user}:{pwd}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}"}


class StreamingOrchestrator:
    def __init__(
        self,
        streams: Sequence[StreamSpec],
        runtime_cfg: RuntimeConfig,
        orch_cfg: OrchestratorConfig = OrchestratorConfig(),
        worker_cfg: WorkerConfig = WorkerConfig(),
    ):
        self.streams = list(streams)
        self.runtime_cfg = runtime_cfg
        self.orch_cfg = orch_cfg
        self.worker_cfg = worker_cfg

        ctx = mp.get_context("spawn")
        self.frame_queue: mp.Queue = ctx.Queue(maxsize=orch_cfg.frame_queue_maxsize)
        self.event_queue: mp.Queue = ctx.Queue(maxsize=orch_cfg.event_queue_maxsize)

        # contadores compartilhados (funcionam no macOS)
        self.frames_in = ctx.Value("i", 0)
        self.frames_proc = ctx.Value("i", 0)
        self.events_out = ctx.Value("i", 0)

        ensure_dir(os.path.dirname(orch_cfg.log_path) or ".")
        self.logger = SimpleLogger(orch_cfg.log_path)
        self.sink = EventSink(orch_cfg.events_out_path if orch_cfg.save_events_jsonl else None, self.logger)

        self.ingest_procs: List[mp.Process] = []
        self.worker_procs: List[mp.Process] = []
        self._stop = False

    def start(self):
        self.logger.log(f"Iniciando orchestrator com {len(self.streams)} streams...")

        extra_headers = _build_basic_auth_headers_from_env()
        if extra_headers:
            self.logger.log("Basic Auth habilitado via HLS_USER/HLS_PASS.")
        else:
            self.logger.log("Basic Auth NÃO configurado (HLS_USER/HLS_PASS ausentes).")

        self.ingest_procs = start_ingest_processes(
            streams=self.streams,
            out_queue=self.frame_queue,
            max_streams_per_process=self.orch_cfg.max_streams_per_process,
            fps=self.orch_cfg.ingest_fps,
            out_hw=(self.orch_cfg.ingest_out_w, self.orch_cfg.ingest_out_h),
            read_timeout_s=self.orch_cfg.ingest_timeout_s,
            reconnect_backoff_s=self.orch_cfg.ingest_reconnect_backoff_s,
            max_queue=self.orch_cfg.frame_queue_maxsize,
            drop_if_full=self.orch_cfg.ingest_drop_if_full,
            extra_headers=extra_headers,
            frames_in=self.frames_in,
            daemon=True,
        )
        self.logger.log(f"Ingest processes: {len(self.ingest_procs)}")

        self.worker_procs = start_worker_processes(
            runtime_cfg=self.runtime_cfg,
            in_queue=self.frame_queue,
            out_queue=self.event_queue,
            num_workers=self.orch_cfg.num_workers,
            worker_cfg=self.worker_cfg,
            frames_proc=self.frames_proc,
            events_out=self.events_out,
            daemon=True,
        )
        self.logger.log(f"Worker processes: {len(self.worker_procs)}")

        try:
            signal.signal(signal.SIGINT, self._handle_stop)
            signal.signal(signal.SIGTERM, self._handle_stop)
        except Exception:
            pass

    def _handle_stop(self, *_):
        self._stop = True

    def _terminate_processes(self):
        for p in self.worker_procs:
            try:
                p.terminate()
            except Exception:
                pass
        for p in self.ingest_procs:
            try:
                p.terminate()
            except Exception:
                pass

    def _healthcheck(self):
        for i, p in enumerate(self.ingest_procs):
            if not p.is_alive():
                self.logger.log(f"[WARN] ingest proc {i} morreu (exitcode={p.exitcode})")

        for i, p in enumerate(self.worker_procs):
            if not p.is_alive():
                self.logger.log(f"[WARN] worker proc {i} morreu (exitcode={p.exitcode})")

    def _heartbeat(self):
        try:
            with self.frames_in.get_lock():
                fin = self.frames_in.value
            with self.frames_proc.get_lock():
                fpr = self.frames_proc.value
            with self.events_out.get_lock():
                eout = self.events_out.value
        except Exception:
            fin, fpr, eout = -1, -1, -1

        self.logger.log(
            f"[HB] frames_in={fin} frames_proc={fpr} events_out={eout} "
            f"ingest_alive={sum(p.is_alive() for p in self.ingest_procs)}/{len(self.ingest_procs)} "
            f"worker_alive={sum(p.is_alive() for p in self.worker_procs)}/{len(self.worker_procs)}"
        )

    def run_forever(self):
        self.start()
        self.logger.log("Orchestrator rodando. CTRL+C para parar.")

        count = 0
        last_hb = time.time()

        try:
            while not self._stop:
                now = time.time()
                if now - last_hb >= self.orch_cfg.heartbeat_every_s:
                    last_hb = now
                    self._healthcheck()
                    self._heartbeat()

                try:
                    ev = self.event_queue.get(timeout=1.0)
                except Exception:
                    continue

                if ev is None:
                    continue

                count += 1
                if self.orch_cfg.print_every > 0 and (count % self.orch_cfg.print_every == 0):
                    self.sink.write(ev)

        finally:
            self.logger.log("Encerrando orchestrator...")
            self.sink.close()
            self._terminate_processes()


def example_run():
    streams = [
        StreamSpec(
            stream_id="cam-172-25-127-209",
            url="http://172.25.132.167:8888/cam-172-25-127-209/index.m3u8",
        )
    ]

    runtime_cfg = RuntimeConfig(
        device="cuda",
        amp=True,
        detector=DetectorRuntimeConfig(
            checkpoint_path=r"runs/detector_smoke/last.pt",
            input_hw=(320, 320),
            score_thr=0.35,
            iou_thr=0.60,
            topk=300,
            assume_bgr=True,
        ),
        verification=VerificationRuntimeConfig(
            checkpoint_path=r"runs/verification_ms1m_v2/last.pt",
            input_hw=(112, 112),
            base_c=64,
            emb_dim=256,
            threshold=0.60,
            metric="cosine",
        ),
        crop=CropRuntimeConfig(
            expand_square_scale=1.35,
            min_side_px=40,
            center_crop_square_reference=True,
        ),
        gallery_root_dir=r"data/galeria",
        gallery_one_image_per_id=True,
        gallery_pt_path=r"data/gallery.pt",
    )

    worker_cfg = WorkerConfig(
        worker_id="gpu-worker",
        per_camera_min_interval_s=0.4,
        match_debounce_s=10.0,
        only_matches=False,   # debug
        save_images=True,
        images_dir="runs/events_images",
    )

    orch_cfg = OrchestratorConfig(
        max_streams_per_process=30,
        ingest_fps=3.0,
        ingest_out_w=640,
        ingest_out_h=360,
        num_workers=1,
        save_events_jsonl=True,
        events_out_path="runs/events/events.jsonl",
        log_path="runs/events/orchestrator.log",
        print_every=1,
        heartbeat_every_s=5.0,
    )

    orch = StreamingOrchestrator(streams, runtime_cfg, orch_cfg, worker_cfg)
    orch.run_forever()


if __name__ == "__main__":
    example_run()