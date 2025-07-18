"""
Media Generation Handler for CrewAI
Handles the actual execution of media generation using existing modules
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import sys

# Import existing modules
try:
    from media_generator import MediaGenerator
    from video_stitcher import VideoStitcher
except ImportError as e:
    print(f"Error importing media generation modules: {e}")
    print("Please ensure media_generator.py and video_stitcher.py are in the same directory")
    sys.exit(1)


class MediaGenerationHandler:
    """Handles media generation execution for CrewAI workflow"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, output_folder: Path):
        self.config = config
        self.logger = logger
        self.output_folder = output_folder
        
        # Create necessary subdirectories
        self.media_folder = output_folder / "media"
        self.temp_folder = output_folder / "temp"
        self.media_folder.mkdir(exist_ok=True)
        self.temp_folder.mkdir(exist_ok=True)
        
        # Initialize media generation components
        self.media_generator = MediaGenerator(config, logger, output_folder)
        self.video_stitcher = VideoStitcher(config, logger, output_folder)
    
    def generate_media(self, script_result: Dict[str, Any], voice_result: Dict[str, Any], 
                      user_prompt: str, reference_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate all media assets and create final video
        
        Args:
            script_result: The generated video script
            voice_result: The voice selection result
            user_prompt: Original user prompt
            reference_url: Optional reference image URL
            
        Returns:
            Dictionary containing media generation results
        """
        
        try:
            self.logger.info("Starting media generation process...")
            
            # Step 1: Update script with voice information
            updated_script = self._update_script_with_voice(script_result, voice_result)
            
            # Step 2: Generate all media assets
            self.logger.info("Generating media assets...")
            media_files = self.media_generator.generate_all_media(updated_script, reference_url)
            
            # Step 3: Stitch videos together
            self.logger.info("Stitching videos...")
            final_video_path = self.video_stitcher.stitch_videos(updated_script, media_files)
            
            # Step 4: Generate summary statistics
            generation_summary = self._generate_summary_stats(media_files, final_video_path)
            
            # Step 5: Prepare result
            result = {
                "media_generation_status": "success",
                "generated_assets": {
                    "images": {k: str(v) for k, v in media_files.get("images", {}).items()},
                    "videos": {k: str(v) for k, v in media_files.get("videos", {}).items()},
                    "audio": {
                        "voiceovers": {k: str(v) for k, v in media_files.get("audio", {}).get("voiceovers", {}).items()},
                        "background_music": str(media_files.get("audio", {}).get("background_music", "")) if media_files.get("audio", {}).get("background_music") else None
                    }
                },
                "final_video": str(final_video_path),
                "output_folder": str(self.output_folder),
                "generation_summary": generation_summary,
                "errors": []
            }
            
            self.logger.info("Media generation completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Media generation failed: {str(e)}")
            
            # Return failure result with any partial assets
            result = {
                "media_generation_status": "failed",
                "generated_assets": {
                    "images": {},
                    "videos": {},
                    "audio": {"voiceovers": {}, "background_music": None}
                },
                "final_video": None,
                "output_folder": str(self.output_folder),
                "generation_summary": {
                    "total_images": 0,
                    "total_videos": 0,
                    "total_voiceovers": 0,
                    "final_duration": 0,
                    "file_size": "0 MB"
                },
                "errors": [str(e)]
            }
            
            return result
    
    def _update_script_with_voice(self, script_result: Dict[str, Any], voice_result: Dict[str, Any]) -> Dict[str, Any]:
        """Update the script with voice selection information"""
        
        updated_script = script_result.copy()
        
        # Extract voice ID from voice result
        voice_id = voice_result.get("voice_selection", {}).get("voice_id")
        if voice_id:
            # Add voice configuration to video_config
            if "video_config" not in updated_script:
                updated_script["video_config"] = {}
            
            updated_script["video_config"]["voice_id"] = voice_id
            updated_script["video_config"]["voice_characteristics"] = voice_result.get("voice_selection", {}).get("voice_characteristics", {})
            
            self.logger.info(f"Updated script with voice ID: {voice_id}")
        else:
            self.logger.warning("No voice ID found in voice result")
        
        return updated_script
    
    def _generate_summary_stats(self, media_files: Dict[str, Any], final_video_path: Path) -> Dict[str, Any]:
        """Generate summary statistics for the media generation"""
        
        try:
            # Count assets
            total_images = len(media_files.get("images", {}))
            total_videos = len(media_files.get("videos", {}))
            total_voiceovers = len(media_files.get("audio", {}).get("voiceovers", {}))
            
            # Get final video duration and size
            final_duration = 0
            file_size = "0 MB"
            
            if final_video_path and final_video_path.exists():
                # Get duration using ffprobe
                try:
                    cmd = [
                        "ffprobe",
                        "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        str(final_video_path)
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    final_duration = float(result.stdout.strip())
                except (subprocess.CalledProcessError, ValueError):
                    self.logger.warning("Could not determine final video duration")
                
                # Get file size
                try:
                    size_bytes = final_video_path.stat().st_size
                    size_mb = size_bytes / (1024 * 1024)
                    file_size = f"{size_mb:.1f} MB"
                except OSError:
                    self.logger.warning("Could not determine final video file size")
            
            return {
                "total_images": total_images,
                "total_videos": total_videos,
                "total_voiceovers": total_voiceovers,
                "final_duration": final_duration,
                "file_size": file_size
            }
            
        except Exception as e:
            self.logger.error(f"Error generating summary stats: {e}")
            return {
                "total_images": 0,
                "total_videos": 0,
                "total_voiceovers": 0,
                "final_duration": 0,
                "file_size": "0 MB"
            } 