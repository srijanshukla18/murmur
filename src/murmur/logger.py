"""Logging for Murmur."""

import logging
from datetime import datetime
from pathlib import Path


def setup_logger() -> logging.Logger:
    """Setup logger that writes to ~/Library/Logs/Murmur/."""
    log_dir = Path.home() / "Library" / "Logs" / "Murmur"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"murmur-{datetime.now().strftime('%Y-%m-%d')}.log"

    logger = logging.getLogger("murmur")
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    )
    logger.addHandler(fh)

    # Console handler (errors only)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger


log = setup_logger()
