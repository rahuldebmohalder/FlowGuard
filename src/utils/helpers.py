"""
FlowGuard Utilities — config loading, logging, timing, and shared helpers.
"""

import os
import sys
import time
import json
import yaml
import logging
import hashlib
import functools
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# Configuration

_CONFIG_CACHE: Optional[Dict] = None

def load_config(path: str = None) -> Dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and path is None:
        return _CONFIG_CACHE
    if path is None:
        path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _CONFIG_CACHE = cfg
    return cfg


def get_nested(cfg: Dict, dotpath: str, default=None):
    keys = dotpath.split(".")
    val = cfg
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val


#  Logging

def setup_logger(name: str, log_dir: str = "outputs/logs",
                 level: int = logging.INFO) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(name)s — %(levelname)s — %(message)s",
        datefmt="%H:%M:%S"
    )

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# Timing / Profiling

class Timer:

    def __init__(self, label: str = "", logger: logging.Logger = None):
        self.label = label
        self.logger = logger
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        msg = f"[{self.label}] {self.elapsed:.3f}s"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)


def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        print(f"  ⏱  {func.__name__}: {dt:.3f}s")
        return result
    return wrapper


#File / Hashing

def contract_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


# Dataframe helpers

def safe_normalize(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx - mn < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mn) / (mx - mn)


def split_dataset(df: pd.DataFrame, cfg: Dict = None,
                  seed: int = 42) -> tuple:
    if cfg is None:
        cfg = load_config()
    train_r = cfg["experiments"]["train_ratio"]
    val_r = cfg["experiments"]["val_ratio"]

    from sklearn.model_selection import train_test_split

    train, temp = train_test_split(df, train_size=train_r, random_state=seed)
    relative_val = val_r / (1 - train_r)
    val, test = train_test_split(temp, train_size=relative_val, random_state=seed)
    return train, val, test
