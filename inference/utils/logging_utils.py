import os
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

def setup_logging(
    log_folder: Path,
    log_file: str = "app.log",
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB
    backup_count: int = 3,
    log_level: str = "INFO",
    log_format: str = "%(asctime)s - %(levelname)s - %(message)s"
) -> None:
    """
    Set up logging with rotation for the application.
    
    Args:
        log_folder: Directory to store log files
        log_file: Name of the log file
        max_bytes: Maximum size of each log file
        backup_count: Number of backup files to keep
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format string for log messages
    """
    try:
        # Create log directory if it doesn't exist
        os.makedirs(log_folder, exist_ok=True)
        log_file_path = log_folder / log_file

        # Create rotating file handler
        rotating_handler = RotatingFileHandler(
            filename=log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )

        # Set up basic configuration
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                rotating_handler,
                logging.StreamHandler(),  # Also log to console
            ],
            force=True,
        )

        logging.info(f"Logging configured successfully in {log_file_path}")
        
    except Exception as e:
        print(f"Failed to configure logging: {e}")
        # Set up a basic console logger as fallback
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            force=True,
        )
        logging.error(f"Falling back to basic console logging due to error: {e}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Name for the logger, typically __name__ of the module
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name) 