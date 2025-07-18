"""
Logging utilities for the Any Video Maker project
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional


def setup_logging(output_folder: Path, log_level: str = "INFO") -> logging.Logger:
    """Setup logging to both console and file"""
    logger = logging.getLogger("any_video_maker")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = output_folder / "generation.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def log_api_call(logger: logging.Logger, provider: str, model: str, params: Dict[str, Any]) -> None:
    """Log API call details"""
    logger.debug(f"API Call - Provider: {provider}, Model: {model}, Params: {params}")


def log_media_generation(logger: logging.Logger, media_type: str, media_id: str, 
                        params: Dict[str, Any], output_path: Path, duration: float) -> None:
    """Log media generation details"""
    logger.info(f"Generated {media_type} for {media_id} in {duration:.2f}s -> {output_path}")


def log_file_operation(logger: logging.Logger, operation: str, file_path: Path, 
                      success: bool, error_msg: Optional[str] = None) -> None:
    """Log file operation details"""
    if success:
        logger.debug(f"File operation '{operation}' successful: {file_path}")
    else:
        logger.error(f"File operation '{operation}' failed: {file_path} - {error_msg}") 