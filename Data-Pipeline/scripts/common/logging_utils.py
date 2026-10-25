"""Logging setup shared by every pipeline step.

Each logger writes to logs/pipeline.log and propagates to the root logger,
so the same messages also appear in the Airflow task log or the console.
"""

import logging
from logging.handlers import RotatingFileHandler

from scripts.common.config import PIPELINE_ROOT

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FILE = PIPELINE_ROOT / "logs" / "pipeline.log"

_file_handler = None


def _get_file_handler() -> logging.Handler:
    global _file_handler
    if _file_handler is None:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        _file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        _file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return _file_handler


def get_logger(name: str) -> logging.Logger:
    """Return a logger named ``skinsort.<name>`` that also writes to pipeline.log."""
    logger = logging.getLogger(f"skinsort.{name}")
    logger.setLevel(logging.INFO)
    handler = _get_file_handler()
    if handler not in logger.handlers:
        logger.addHandler(handler)
    return logger


def configure_console_logging() -> None:
    """Print logs to the console; used when a script is run from the command line."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
