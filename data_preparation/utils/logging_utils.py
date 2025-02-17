"""Logging utilities for data preparation pipeline."""

import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

def setup_logging(
    log_folder: Path,
    log_file: str,
    max_bytes: int,
    backup_count: int,
    log_level: str,
    log_format: str
) -> None:
    """
    Set up logging configuration for the data preparation pipeline.
    
    Args:
        log_folder (Path): Directory where log files will be stored
        log_file (str): Name of the log file
        max_bytes (int): Maximum size of each log file before rotation
        backup_count (int): Number of backup log files to keep
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format (str): Format string for log messages
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_folder, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Setup file handler with rotation
    log_path = log_folder / log_file
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    logging.info("Logging setup completed.") 