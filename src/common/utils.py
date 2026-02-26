from __future__ import annotations

import os
import time
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # determinístico (pode reduzir performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(path: str, data: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def now_ts() -> float:
    return time.time()

@dataclass
class AvgMeter:
    """
    Medidor simples de média para losses/metrics.
    """
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)

class SimpleLogger:
    """
    Logger simples (stdout + opcional arquivo).
    """
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        if log_file:
            ensure_dir(os.path.dirname(log_file) or ".")

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")

class Timer:
    """
    Uso:
      with Timer() as t:
         ...
      print(t.ms)
    """
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.t1 = time.perf_counter()
        self.s = self.t1 - self.t0
        self.ms = self.s * 1000.0
        return False

def torch_save(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    torch.save(obj, path)

def torch_load(path: str, map_location: Optional[str] = None) -> Any:
    if map_location is None:
        return torch.load(path)
    return torch.load(path, map_location=map_location)