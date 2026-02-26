"""Logging setup with rich formatting."""
import logging
import sys
from pathlib import Path
from datetime import datetime
from rich.logging import RichHandler


def setup_logger(
    name: str = "chromaguide",
    log_dir: str | None = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Configure logger with optional file and rich console output.
    
    Args:
        name: Logger name.
        log_dir: Directory for log files (None = console only).
        level: Logging level.
        console: Whether to enable rich console output.
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Console handler with rich formatting
    if console:
        console_handler = RichHandler(
            level=level,
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_handler)
    
    # File handler
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_path / f"{name}_{timestamp}.log"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
            )
        )
        logger.addHandler(file_handler)
    
    return logger
