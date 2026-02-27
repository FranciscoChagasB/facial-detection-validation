# src/streaming/ingest.py
from __future__ import annotations

import asyncio
import sys
import time
import signal
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import multiprocessing as mp


@dataclass(frozen=True)
class StreamSpec:
    stream_id: str
    url: str


@dataclass
class FramePacket:
    stream_id: str
    url: str
    ts: float
    frame_bgr: Any  # numpy.ndarray (H,W,3) BGR uint8


def _ffmpeg_cmd(
    url: str,
    out_w: int,
    out_h: int,
    fps: float,
    read_timeout_s: int,
    extra_headers: Optional[Dict[str, str]] = None,
) -> List[str]:
    headers = ""
    if extra_headers:
        headers = "".join([f"{k}: {v}\r\n" for k, v in extra_headers.items()])

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-rw_timeout", str(int(read_timeout_s * 1_000_000)),
    ]

    if headers:
        cmd += ["-headers", headers]

    cmd += [
        "-i", url,
        "-an",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-vf", f"fps={fps},scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
               f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    return cmd


async def _read_exactly(stream: asyncio.StreamReader, n: int) -> Optional[bytes]:
    try:
        data = await stream.readexactly(n)
        return data
    except asyncio.IncompleteReadError:
        return None


class AsyncIngestWorker:
    def __init__(
        self,
        specs: Sequence[StreamSpec],
        out_queue: mp.Queue,
        fps: float = 3.0,
        out_hw: Tuple[int, int] = (640, 360),
        read_timeout_s: int = 10,
        reconnect_backoff_s: float = 2.0,
        max_queue: int = 2000,
        drop_if_full: bool = True,
        extra_headers: Optional[Dict[str, str]] = None,
        frames_in: Optional[mp.Value] = None,
    ):
        self.specs = list(specs)
        self.out_queue = out_queue
        self.fps = float(fps)
        self.out_w, self.out_h = int(out_hw[0]), int(out_hw[1])
        self.read_timeout_s = int(read_timeout_s)
        self.reconnect_backoff_s = float(reconnect_backoff_s)
        self.max_queue = int(max_queue)
        self.drop_if_full = bool(drop_if_full)
        self.extra_headers = extra_headers or {}
        self.frames_in = frames_in
        self._stop = False

    def stop(self):
        self._stop = True

    def _inc_frames_in(self) -> None:
        if self.frames_in is None:
            return
        try:
            with self.frames_in.get_lock():
                self.frames_in.value += 1
        except Exception:
            pass

    async def _run_one(self, spec: StreamSpec):
        frame_size = self.out_w * self.out_h * 3

        while not self._stop:
            proc = None
            try:
                cmd = _ffmpeg_cmd(
                    url=spec.url,
                    out_w=self.out_w,
                    out_h=self.out_h,
                    fps=self.fps,
                    read_timeout_s=self.read_timeout_s,
                    extra_headers=self.extra_headers,
                )

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                assert proc.stdout is not None
                reader: asyncio.StreamReader = proc.stdout  # type: ignore

                while not self._stop:
                    raw = await _read_exactly(reader, frame_size)
                    if raw is None:
                        break

                    import numpy as np
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.out_h, self.out_w, 3))

                    pkt = FramePacket(
                        stream_id=spec.stream_id,
                        url=spec.url,
                        ts=time.time(),
                        frame_bgr=frame,
                    )

                    if self.drop_if_full:
                        try:
                            try:
                                if self.out_queue.qsize() >= self.max_queue:
                                    for _ in range(3):
                                        try:
                                            self.out_queue.get_nowait()
                                        except Exception:
                                            break
                            except Exception:
                                pass

                            self.out_queue.put_nowait(pkt)
                            self._inc_frames_in()
                        except Exception:
                            pass
                    else:
                        self.out_queue.put(pkt)
                        self._inc_frames_in()

            except Exception:
                await asyncio.sleep(self.reconnect_backoff_s)
            finally:
                if proc is not None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    try:
                        await proc.wait()
                    except Exception:
                        pass

            await asyncio.sleep(self.reconnect_backoff_s)

    async def run(self):
        tasks = [asyncio.create_task(self._run_one(s)) for s in self.specs]
        try:
            await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()


def _ingest_process_main(
    specs: List[StreamSpec],
    out_queue: mp.Queue,
    fps: float,
    out_hw: Tuple[int, int],
    read_timeout_s: int,
    reconnect_backoff_s: float,
    max_queue: int,
    drop_if_full: bool,
    extra_headers: Optional[Dict[str, str]],
    frames_in: Optional[mp.Value],
):
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore

    worker = AsyncIngestWorker(
        specs=specs,
        out_queue=out_queue,
        fps=fps,
        out_hw=out_hw,
        read_timeout_s=read_timeout_s,
        reconnect_backoff_s=reconnect_backoff_s,
        max_queue=max_queue,
        drop_if_full=drop_if_full,
        extra_headers=extra_headers,
        frames_in=frames_in,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _handle_sig(*_):
        worker.stop()

    try:
        signal.signal(signal.SIGINT, _handle_sig)
        signal.signal(signal.SIGTERM, _handle_sig)
    except Exception:
        pass

    try:
        loop.run_until_complete(worker.run())
    finally:
        try:
            loop.stop()
            loop.close()
        except Exception:
            pass


def shard_streams(streams: Sequence[StreamSpec], max_streams_per_process: int) -> List[List[StreamSpec]]:
    streams = list(streams)
    if max_streams_per_process <= 0:
        raise ValueError("max_streams_per_process deve ser > 0")
    return [streams[i:i + max_streams_per_process] for i in range(0, len(streams), max_streams_per_process)]


def start_ingest_processes(
    streams: Sequence[StreamSpec],
    out_queue: mp.Queue,
    max_streams_per_process: int = 30,
    fps: float = 3.0,
    out_hw: Tuple[int, int] = (640, 360),
    read_timeout_s: int = 10,
    reconnect_backoff_s: float = 2.0,
    max_queue: int = 2000,
    drop_if_full: bool = True,
    extra_headers: Optional[Dict[str, str]] = None,
    frames_in: Optional[mp.Value] = None,
    daemon: bool = True,
) -> List[mp.Process]:
    ctx = mp.get_context("spawn")
    shards = shard_streams(streams, max_streams_per_process=max_streams_per_process)

    procs: List[mp.Process] = []
    for shard in shards:
        p = ctx.Process(
            target=_ingest_process_main,
            args=(
                list(shard),
                out_queue,
                fps,
                out_hw,
                read_timeout_s,
                reconnect_backoff_s,
                max_queue,
                drop_if_full,
                extra_headers,
                frames_in,
            ),
            daemon=daemon,
        )
        p.start()
        procs.append(p)

    return procs