import os
import gc
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F


def setup_logger(filename: str = "attack", ensure_unique: bool = True) -> str:
    """Configure the root logger to write to a fresh file and stdout."""
    os.makedirs("log", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"log/{timestamp}_{filename}"
    log_filename = f"{base}.log"
    if ensure_unique:
        suffix = 1
        while os.path.exists(log_filename):
            log_filename = f"{base}_{suffix}.log"
            suffix += 1

    logger = logging.getLogger()
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logging.info("Logging to %s", log_filename)
    return log_filename


def switch_logger(stem: str) -> str:
    """Convenience wrapper to rotate log files mid-session."""
    return setup_logger(stem)


def configure_cuda_allocator(expandable_segments: bool = True) -> None:
    """Optionally enable CUDA expandable segments to reduce fragmentation."""
    if not torch.cuda.is_available() or not expandable_segments:
        return
    key = "PYTORCH_CUDA_ALLOC_CONF"
    if "expandable_segments" not in os.environ.get(key, ""):
        os.environ[key] = "expandable_segments:True"
        logging.info("Enabled CUDA expandable segments allocator.")


def clear_cuda_memory(
    device: str | int | torch.device | None = None,
    *,
    sync: bool = True,
    reset_stats: bool = True,
    aggressive: bool = True,
) -> None:
    """Free unused CUDA memory and attempt to reduce fragmentation."""
    if not torch.cuda.is_available():
        if aggressive:
            gc.collect()
        return
    dev = torch.device(device) if device is not None else torch.device("cuda")
    try:
        if sync:
            torch.cuda.synchronize(dev)
        if aggressive:
            gc.collect()
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
            if reset_stats:
                try:
                    torch.cuda.reset_peak_memory_stats(dev)
                except Exception:
                    pass
            if aggressive:
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        if sync:
            torch.cuda.synchronize(dev)
    except Exception as exc:
        logging.debug("clear_cuda_memory skipped: %s", exc)
