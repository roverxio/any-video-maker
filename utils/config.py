"""
Configuration utilities for the Any Video Maker project
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional


def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables and defaults"""
    
    config = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "FAL_AI_API_KEY": os.getenv("FAL_AI_API_KEY"),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "MAX_CONCURRENT_REQUESTS": int(os.getenv("MAX_CONCURRENT_REQUESTS", "3")),
        "IMAGE_MODEL": os.getenv("IMAGE_MODEL", "fal"),
        "VIDEO_MODEL": os.getenv("VIDEO_MODEL", "fal"),
        "AUDIO_MODEL": os.getenv("AUDIO_MODEL", "fal"),
        "video_width": int(os.getenv("VIDEO_WIDTH", "704")),
        "video_height": int(os.getenv("VIDEO_HEIGHT", "1304")),
        "video_fps": int(os.getenv("VIDEO_FPS", "24")),
        "output_dir": os.getenv("OUTPUT_DIR", "./outputs")
    }
    
    return config


def validate_ffmpeg() -> bool:
    """Validate that FFmpeg is installed and accessible"""
    try:
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, text=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_ffmpeg_path() -> Optional[str]:
    """Get the path to FFmpeg executable"""
    try:
        result = subprocess.run(["which", "ffmpeg"], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Try common Windows paths
        windows_paths = [
            "C:\\ffmpeg\\bin\\ffmpeg.exe",
            "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
            "C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe"
        ]
        for path in windows_paths:
            if Path(path).exists():
                return path
        return None 