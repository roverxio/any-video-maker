"""
Folder management utilities for the Any Video Maker project
"""

import requests
from pathlib import Path
from typing import Optional
import logging


def create_run_folder(base_output_dir: str = "./outputs") -> Path:
    """Create timestamped run folder"""
    from datetime import datetime
    
    now = datetime.now()
    folder_name = f"video_gen_{now.strftime('%d%m_%H%M%S')}"
    
    # Create the full path
    outputs_dir = Path(base_output_dir)
    outputs_dir.mkdir(exist_ok=True)
    
    run_folder = outputs_dir / folder_name
    run_folder.mkdir(exist_ok=True)
    
    # Create subdirectories
    (run_folder / "media").mkdir(exist_ok=True)
    (run_folder / "media" / "images").mkdir(exist_ok=True)
    (run_folder / "media" / "videos").mkdir(exist_ok=True)
    (run_folder / "media" / "audio").mkdir(exist_ok=True)
    (run_folder / "temp").mkdir(exist_ok=True)
    
    return run_folder


def get_media_path(run_folder: Path, media_type: str, media_id: str, extension: str) -> Path:
    """Get the path for a media file"""
    
    media_type_map = {
        "image": "images",
        "video": "videos", 
        "audio": "audio",
        "voiceover": "audio"
    }
    
    subfolder = media_type_map.get(media_type, "temp")
    media_folder = run_folder / "media" / subfolder
    
    # Ensure the folder exists
    media_folder.mkdir(parents=True, exist_ok=True)
    
    return media_folder / f"{media_id}.{extension}"


def download_reference_media(url: str, run_folder: Path) -> Optional[Path]:
    """Download reference media from URL"""
    
    try:
        # Create reference folder
        reference_folder = run_folder / "reference"
        reference_folder.mkdir(exist_ok=True)
        
        # Determine file extension from URL
        if "." in url.split("/")[-1]:
            extension = url.split(".")[-1].split("?")[0]  # Remove query parameters
        else:
            extension = "jpg"  # Default to jpg
        
        # Create file path
        file_path = reference_folder / f"reference.{extension}"
        
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return file_path
        
    except Exception as e:
        logging.error(f"Failed to download reference media from {url}: {e}")
        return None 