"""
Video Stitcher
Combines multiple video clips with audio into a single video.
"""

import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import ffmpeg
from utils.logger import log_file_operation
from utils.config import validate_ffmpeg


class VideoStitcher:
    """Handles the stitching of multiple video clips with audio."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, run_folder: Path):
        self.config = config
        self.logger = logger
        self.run_folder = run_folder
        
        # Ensure ffmpeg is available
        validate_ffmpeg()
        
    def _get_standard_properties_from_aspect_ratio(self, aspect_ratio: str) -> Dict[str, Any]:
        """Calculate standard video properties based on aspect ratio."""
        
        # Map aspect ratios to standard dimensions
        aspect_dimensions = {
            "9:16": {"width": 720, "height": 1280},  # Vertical (mobile)
            "16:9": {"width": 1280, "height": 720}, # Horizontal (landscape)
            "1:1": {"width": 1080, "height": 1080}, # Square (social)
            "4:3": {"width": 1024, "height": 768},  # Traditional
            "3:4": {"width": 768, "height": 1024}   # Vertical traditional
        }
        
        # Get dimensions for the specified aspect ratio, default to 9:16
        dimensions = aspect_dimensions.get(aspect_ratio, aspect_dimensions["9:16"])
        
        return {
            'width': dimensions["width"],
            'height': dimensions["height"],
            'frame_rate': self.config.get('video_fps', 24)
        }

    def stitch_videos(self, script_data: Dict[str, Any], media_files: Dict[str, Dict[str, Path]]) -> Path:
        """Stitch together all video clips with their respective audio."""
        
        self.logger.info("Starting video stitching process...")
        
        # Step 1: Trim and standardize videos to specified lengths
        self.logger.info("Step 1: Trimming and standardizing videos...")
        trimmed_videos = self._trim_videos(script_data, media_files["videos"])
        
        # Step 2: Add individual voiceovers to each trimmed scene
        self.logger.info("Step 2: Adding individual voiceovers to scenes...")
        videos_with_voiceovers = self._add_individual_voiceovers(
            script_data, trimmed_videos, media_files["audio"].get("voiceovers", {})
        )
        
        # Step 3: Ensure all clips have an audio stream for concatenation
        self.logger.info("Step 3: Ensuring all clips have an audio stream...")
        videos_with_audio = self._ensure_all_clips_have_audio(videos_with_voiceovers)

        # Step 4: Combine videos in sequence
        self.logger.info("Step 4: Combining videos in sequence...")
        combined_video = self._combine_videos_with_audio(script_data, videos_with_audio)
        
        # Step 5: Get the duration of the combined video
        video_duration = self._get_duration(combined_video)
        if video_duration == 0:
            self.logger.error("Combined video has zero duration. Aborting background music steps.")
            return combined_video
        self.logger.info(f"Combined video duration: {video_duration:.2f} seconds")

        # Step 6: Generate background music if needed
        background_music_path = media_files["audio"].get("background_music")
        if not background_music_path and script_data["video_config"].get("needs_background_music"):
            self.logger.info("Step 6: Generating background music...")
            try:
                from media_generator import MediaGenerator
                media_generator = MediaGenerator(self.config, self.logger, self.run_folder)
                background_music_path = media_generator._generate_background_music(script_data, int(video_duration))
            except Exception as e:
                self.logger.error(f"Failed to generate background music: {e}")
                background_music_path = None # Ensure it's None on failure
        
        # Step 7: Add background music to the final combined video
        self.logger.info("Step 7: Adding background music...")
        final_video = self._add_background_music(script_data, combined_video, background_music_path)
        
        self.logger.info(f"✅ Final video created: {final_video}")
        return final_video
    
    def _get_duration(self, file_path: Path) -> float:
        """Get file duration using ffprobe"""
        if not file_path.exists():
            self.logger.warning(f"File not found for duration check: {file_path}")
            return 0.0
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(file_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError) as e:
            self.logger.warning(f"Could not determine duration of {file_path}: {str(e)}")
            return 0.0

    def _get_video_properties(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Get video properties using ffprobe"""
        try:
            probe = ffmpeg.probe(str(file_path))
            video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            return {
                'width': video_stream.get('width'),
                'height': video_stream.get('height'),
                'frame_rate': eval(video_stream.get('r_frame_rate', '0/1'))
            }
        except (ffmpeg.Error, StopIteration, KeyError) as e:
            self.logger.warning(f"Could not get video properties for {file_path}: {str(e)}")
            return None

    def _trim_videos(self, script_data: Dict[str, Any], video_paths: Dict[str, Path]) -> Dict[str, Path]:
        """Trim each video to its specified duration and standardize format."""

        scenes = script_data["scenes"]
        video_config = script_data.get("video_config", {})
        aspect_ratio = video_config.get("aspect_ratio", "9:16")
        trimmed_paths = {}

        # Calculate standard video properties based on aspect ratio
        standard_props = self._get_standard_properties_from_aspect_ratio(aspect_ratio)
        
        first_valid_video_found = False
        for scene in scenes:
            scene_id = scene["scene_id"]
            if scene_id in video_paths:
                props = self._get_video_properties(video_paths[scene_id])
                if props and props.get('width'):  # Check if props are valid
                    # Use the first video's properties if they match the expected aspect ratio
                    video_aspect = props['width'] / props['height']
                    expected_aspect = standard_props['width'] / standard_props['height']
                    
                    # If the video aspect ratio is close to expected, use its properties
                    if abs(video_aspect - expected_aspect) < 0.1:
                        standard_props['width'] = props['width']
                        standard_props['height'] = props['height']
                        standard_props['frame_rate'] = props['frame_rate']
                        self.logger.info(f"Using video properties from {scene_id} as standard: {props['width']}x{props['height']} @ {props['frame_rate']:.2f}fps")
                        first_valid_video_found = True
                        break
        
        if not first_valid_video_found:
            self.logger.info(f"Using calculated standard properties for {aspect_ratio}: {standard_props['width']}x{standard_props['height']} @ {standard_props['frame_rate']}fps")

        for scene in scenes:
            scene_id = scene["scene_id"]
            target_duration = scene.get("duration", 3.0)

            if scene_id not in video_paths:
                self.logger.warning(f"No video found for scene {scene_id}")
                continue

            input_path = video_paths[scene_id]
            output_path = self.run_folder / "temp" / f"{scene_id}_trimmed.mp4"

            original_duration = self._get_duration(input_path)
            if original_duration == 0:
                self.logger.warning(f"Skipping {input_path} due to invalid duration.")
                continue
            
            trim_duration = min(original_duration, target_duration)

            try:
                # FFmpeg command to trim and standardize video
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(input_path),
                    "-t", str(trim_duration),
                    "-vf", f"scale={standard_props['width']}:{standard_props['height']}:force_original_aspect_ratio=0,pad={standard_props['width']}:{standard_props['height']}:(ow-iw)/2:(oh-ih)/2,fps={standard_props['frame_rate']}",
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-preset", "medium",
                    "-crf", "23",
                    str(output_path)
                ]

                self.logger.debug(f"Trimming command for {scene_id}: {' '.join(cmd)}")

                subprocess.run(cmd, capture_output=True, text=True, check=True)

                trimmed_paths[scene_id] = output_path
                log_file_operation(self.logger, "trim_standardize", output_path, True)

            except subprocess.CalledProcessError as e:
                error_msg = f"Failed to trim/standardize video {scene_id}: {e.stderr}"
                self.logger.error(error_msg)
                log_file_operation(self.logger, "trim_standardize", output_path, False, error_msg)
                raise

        self.logger.info(f"Trimmed and standardized {len(trimmed_paths)} videos")
        return trimmed_paths
    
    def _add_individual_voiceovers(self, script_data: Dict[str, Any], trimmed_videos: Dict[str, Path],
                                  voiceover_files: Dict[str, Path]) -> Dict[str, Path]:
        """
        Add individual voiceover to each scene.
        Extends video/audio to match the longest of the two.
        If no voiceover, the original audio track is removed.
        """
        videos_with_audio = {}
        scenes = script_data["scenes"]

        for scene in scenes:
            scene_id = scene["scene_id"]

            if scene_id not in trimmed_videos:
                self.logger.warning(f"No trimmed video for scene {scene_id}, skipping voiceover.")
                continue

            video_path = trimmed_videos[scene_id]
            output_path = self.run_folder / "temp" / f"{scene_id}_with_audio.mp4"

            # Case 1: No voiceover for the scene. Remove any existing audio.
            if scene_id not in voiceover_files:
                self.logger.info(f"No voiceover for {scene_id}. Removing existing audio from video.")
                try:
                    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-c:v", "copy", "-an", str(output_path)]
                    subprocess.run(cmd, capture_output=True, text=True, check=True)
                    videos_with_audio[scene_id] = output_path
                    log_file_operation(self.logger, "remove_audio", output_path, True)
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Failed to remove audio from {scene_id}: {e.stderr}")
                    videos_with_audio[scene_id] = video_path # Fallback
                continue

            # Case 2: Voiceover exists. Merge and sync duration.
            voiceover_path = voiceover_files[scene_id]
            self.logger.info(f"Adding voiceover for {scene_id} from {voiceover_path}")

            video_duration = self._get_duration(video_path)
            voiceover_duration = self._get_duration(voiceover_path)

            if video_duration == 0 or voiceover_duration == 0:
                self.logger.warning(f"Skipping voiceover for {scene_id} due to invalid media duration.")
                videos_with_audio[scene_id] = video_path  # Fallback
                continue
            
            target_duration = max(video_duration, voiceover_duration)
            
            # Pad video if it's shorter than the audio
            if video_duration < target_duration:
                video_filter = f"tpad=stop_mode=clone:stop_duration={target_duration - video_duration}"
                self.logger.info(f"Video for {scene_id} is short. Padding with last frame for {target_duration - video_duration:.2f}s.")
            else:
                video_filter = "null"

            # Pad audio if it's shorter than the video
            if voiceover_duration < target_duration:
                audio_filter = f"apad=pad_dur={target_duration - voiceover_duration}"
                self.logger.info(f"Audio for {scene_id} is short. Padding with silence for {target_duration - voiceover_duration:.2f}s.")
            else:
                audio_filter = "anull"

            try:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(video_path),
                    "-i", str(voiceover_path),
                    "-filter_complex", f"[0:v]{video_filter}[v];[1:a]{audio_filter},aresample=44100[a]",
                    "-map", "[v]",
                    "-map", "[a]",
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-shortest",
                    str(output_path)
                ]
                self.logger.debug(f"Add/sync voiceover command for {scene_id}: {' '.join(cmd)}")
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                videos_with_audio[scene_id] = output_path
                log_file_operation(self.logger, "add_voiceover_synced", output_path, True)

            except subprocess.CalledProcessError as e:
                error_msg = f"Failed to add/sync voiceover to {scene_id}: {e.stderr}"
                self.logger.error(error_msg)
                log_file_operation(self.logger, "add_voiceover_synced", output_path, False, error_msg)
                videos_with_audio[scene_id] = video_path

        self.logger.info(f"Processed voiceovers for {len(scenes)} scenes.")
        return videos_with_audio
    
    def _ensure_all_clips_have_audio(self, video_paths: Dict[str, Path]) -> Dict[str, Path]:
        """
        Ensures every video clip has an audio stream, adding silent audio if needed.
        This is crucial for the concatenation process.
        """
        self.logger.info("Ensuring all video clips have an audio stream...")
        processed_paths = {}

        for scene_id, video_path in video_paths.items():
            try:
                probe = ffmpeg.probe(str(video_path))
                has_audio = any(s['codec_type'] == 'audio' for s in probe.get('streams', []))

                if has_audio:
                    processed_paths[scene_id] = video_path
                    continue

                # No audio stream found, let's add a silent one
                self.logger.info(f"Adding silent audio track to {scene_id}...")
                output_path = self.run_folder / "temp" / f"{scene_id}_silent_audio.mp4"

                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(video_path),
                    "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-shortest",
                    str(output_path)
                ]

                subprocess.run(cmd, capture_output=True, text=True, check=True)
                processed_paths[scene_id] = output_path
                log_file_operation(self.logger, "add_silent_audio", output_path, True)

            except (ffmpeg.Error, subprocess.CalledProcessError) as e:
                error_msg = f"Failed to process or add silent audio for {scene_id}: {getattr(e, 'stderr', e)}"
                self.logger.error(error_msg)
                log_file_operation(self.logger, "add_silent_audio", video_path, False, error_msg)
                # Fallback to the original video path
                processed_paths[scene_id] = video_path

        self.logger.info(f"Audio stream check completed for {len(processed_paths)} clips.")
        return processed_paths
    
    def _combine_videos_with_audio(self, script_data: Dict[str, Any], videos_with_audio: Dict[str, Path]) -> Path:
        """Combine all videos using the ffmpeg concat filter for robustness."""
        
        scenes = script_data["scenes"]
        
        # Create list of videos in the correct order
        video_list = []
        for scene in scenes:
            scene_id = scene["scene_id"]
            if scene_id in videos_with_audio:
                video_list.append(videos_with_audio[scene_id])
        
        if not video_list:
            raise ValueError("No videos with audio found to combine")
        
        combined_path = self.run_folder / "combined_video.mp4"

        if len(video_list) == 1:
            # If only one video, just copy it to the combined path and return
            import shutil
            shutil.copy2(video_list[0], combined_path)
            self.logger.info("Only one scene found, skipping concatenation.")
            return combined_path

        self.logger.info(f"Combining {len(video_list)} video clips into one...")
        
        # Build the ffmpeg command with multiple inputs and a complex filter graph
        inputs = []
        video_streams = []
        audio_streams = []
        for i, video_path in enumerate(video_list):
            inputs.extend(["-i", str(video_path)])
            video_streams.append(f"[{i}:v]")
            audio_streams.append(f"[{i}:a]")
            
        filter_complex = (
            f"{''.join(video_streams)}concat=n={len(video_list)}:v=1:a=0[outv];"
            f"{''.join(audio_streams)}concat=n={len(audio_streams)}:v=0:a=1[outa]"
        )
        
        try:
            cmd = ["ffmpeg", "-y"] + inputs + [
                "-filter_complex", filter_complex,
                "-map", "[outv]",
                "-map", "[outa]",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-preset", "medium",
                "-crf", "23",
                str(combined_path)
            ]
            
            self.logger.debug(f"Combine with audio (filter_complex) command: {' '.join(cmd)}")
            
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            log_file_operation(self.logger, "combine_with_audio", combined_path, True)
            self.logger.info(f"Successfully combined {len(video_list)} videos with audio.")
            return combined_path
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to combine videos: {e.stderr}"
            self.logger.error(error_msg)
            log_file_operation(self.logger, "combine_with_audio", combined_path, False, error_msg)
            raise
    
    def _add_background_music(self, script_data: Dict[str, Any], video_path: Path, background_music_path: Optional[Path]) -> Path:
        """Add background music to the final video."""
        
        final_path = self.run_folder / "final_video.mp4"
        
        # If no background music, just copy the video
        if not background_music_path or not background_music_path.exists():
            import shutil
            shutil.copy2(video_path, final_path)
            self.logger.info("No background music provided or found, using video as-is.")
            return final_path
        
        try:
            video_duration = self._get_duration(video_path)
            if video_duration == 0:
                raise ValueError("Cannot add background music to a video with zero duration.")

            # Mix the BGM with the video's existing audio (voiceovers)
            self.logger.info("Mixing background music with video...")
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),           # Input video (with individual voiceovers)
                "-i", str(background_music_path), # Input background music
                "-filter_complex", 
                # Normalize voiceover volume, set BGM volume, mix them.
                # Use 'shortest' to trim the output to the main video's duration.
                f"[0:a]volume=1.0[voice];[1:a]volume=0.2,aloop=loop=-1:size=2e+09[bg_music];"
                f"[bg_music]atrim=duration={video_duration}[bg_trimmed];"
                f"[voice][bg_trimmed]amix=inputs=2:duration=first:dropout_transition=0[mixed_audio]",
                "-map", "0:v:0",                   # Map video from input 0
                "-map", "[mixed_audio]",         # Map mixed audio
                "-c:v", "copy",                  # Copy video stream without re-encoding
                "-c:a", "aac",                   # Audio codec
                "-b:a", "192k",                  # Audio bitrate
                "-shortest",                     # End when shortest input ends (the video)
                str(final_path)
            ]
            
            self.logger.debug(f"Final mixing command: {' '.join(cmd)}")
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            log_file_operation(self.logger, "add_background_music", final_path, True)
            self.logger.info("Successfully added background music to final video.")
            return final_path
            
        except (subprocess.CalledProcessError, ValueError) as e:
            error_msg = f"Failed to add background music: {getattr(e, 'stderr', str(e))}"
            self.logger.error(error_msg)
            log_file_operation(self.logger, "add_background_music", final_path, False, error_msg)
            # Fall back to video without background music
            self.logger.warning("Falling back to video without background music.")
            import shutil
            shutil.copy2(video_path, final_path)
            return final_path
    
    def _create_preview_thumbnail(self, video_path: Path) -> Optional[Path]:
        """Create a thumbnail preview of the final video"""
        
        thumbnail_path = self.run_folder / "preview_thumbnail.jpg"
        
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-ss", "00:00:01",  # Take screenshot at 1 second
                "-vframes", "1",    # Only one frame
                "-q:v", "2",        # High quality
                str(thumbnail_path)
            ]
            
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            log_file_operation(self.logger, "create_thumbnail", thumbnail_path, True)
            return thumbnail_path
            
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Failed to create thumbnail: {e.stderr}")
            return None