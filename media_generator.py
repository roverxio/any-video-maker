"""
Media Generator using fal.ai APIs
Generates images, videos, audio, and background music
"""

import time
import asyncio
import logging
import base64
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import fal_client
import requests
from moviepy import VideoFileClip, concatenate_videoclips
from utils.logger import log_api_call, log_media_generation
from utils.folder_manager import get_media_path, download_reference_media
from openai_color_correction import correct_image_tint_in_memory


class MediaGenerator:
    """Generates all media using fal.ai APIs"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, run_folder: Path):
        self.config = config
        self.logger = logger
        self.run_folder = run_folder
        self.max_concurrent = config["MAX_CONCURRENT_REQUESTS"]
        self.image_model = config.get("IMAGE_MODEL", "fal")  # Default to fal.ai
        
        # Configure fal client
        import os
        os.environ["FAL_KEY"] = config["FAL_AI_API_KEY"]
        
    def generate_all_media(self, script_data: Dict[str, Any], reference_url: Optional[str] = None) -> Dict[str, Any]:
        """Generate all media for the video script"""
        
        self.logger.info("Starting media generation...")
        
        # Download reference media if provided
        reference_path = None
        if reference_url:
            self.logger.info(f"Downloading reference media: {reference_url}")
            reference_path = download_reference_media(reference_url, self.run_folder)
            if reference_path:
                self.logger.info(f"Reference media saved: {reference_path}")
        
        media_files = {
            "images": {},
            "videos": {},
            "audio": {},
            "reference": reference_path
        }
        
        # Step 1: Generate all images first (needed for video generation)
        self.logger.info("Generating images...")
        media_files["images"] = self._generate_all_images(script_data, reference_path)
        
        # Step 2: Generate videos using the images
        self.logger.info("Generating videos...")
        media_files["videos"] = self._generate_all_videos(script_data, media_files["images"])
        
        # Step 3: Generate audio (voiceover and background music)
        self.logger.info("Generating audio...")
        media_files["audio"] = self._generate_all_audio(script_data)
        
        self.logger.info("Media generation completed successfully")
        return media_files
    
    def _load_image_description_data(self) -> Optional[Dict[str, Any]]:
        """Load image description data from the run folder"""
        try:
            image_desc_file = self.run_folder / "video_scripts" / "image_description.json"
            if not image_desc_file.exists():
                self.logger.debug(f"No image_description.json found at {image_desc_file}")
                return None
            
            with open(image_desc_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate structure
            if "image_description" not in data:
                self.logger.warning(f"Invalid image_description.json structure - missing 'image_description' key")
                return None
            
            image_desc = data["image_description"]
            if "variable_name" not in image_desc or "short_description" not in image_desc:
                self.logger.warning(f"Invalid image_description.json structure - missing required keys")
                return None
            
            self.logger.debug(f"Loaded image description: variable_name='{image_desc['variable_name']}', short_description='{image_desc['short_description']}'")
            return image_desc
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse image_description.json: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load image_description.json: {e}")
            return None
    
    def _replace_variable_in_prompt(self, prompt: str, image_desc_data: Dict[str, Any]) -> str:
        """Replace variable name with short description in the prompt"""
        if not image_desc_data:
            return prompt
        
        variable_name = image_desc_data["variable_name"]
        short_description = image_desc_data["short_description"]
        
        original_prompt = prompt
        
        # Replace variable name in backticks (e.g., `chanel_perfume_bottle`)
        prompt = prompt.replace(f"`{variable_name}`", short_description)
        
        # Replace variable name in single quotes (e.g., 'chanel_perfume_bottle')
        prompt = prompt.replace(f"'{variable_name}'", short_description)
        
        # Replace variable name as direct string (case-sensitive exact match)
        # Use word boundaries to avoid partial replacements
        import re
        pattern = r'\b' + re.escape(variable_name) + r'\b'
        prompt = re.sub(pattern, short_description, prompt)
        
        if prompt != original_prompt:
            self.logger.info(f"Replaced '{variable_name}' with '{short_description}' in image prompt")
            self.logger.debug(f"Original prompt: {original_prompt}")
            self.logger.debug(f"Modified prompt: {prompt}")
        
        return prompt

    def _generate_all_images(self, script_data: Dict[str, Any], reference_path: Optional[Path] = None) -> Dict[str, Path]:
        """Generate all images concurrently"""
        
        scenes = script_data["scenes"]
        video_config = script_data["video_config"]
        aspect_ratio = video_config.get("aspect_ratio", "9:16")
        
        image_tasks = []
        
        # Prepare tasks for concurrent execution
        for scene in scenes:
            scene_id = scene["scene_id"]
            composition = scene.get("scene_composition", "single_shot")
            
            if composition == "single_shot":
                prompt = scene.get("image_prompt")
                if not prompt:
                    self.logger.warning(f"Skipping image for scene {scene_id} due to missing prompt.")
                    continue
                
                # Extract has_reference from scene data
                scene_has_reference = scene.get("has_reference", False)
                
                task = {
                    "task_id": scene_id,
                    "prompt": prompt,
                    "has_reference": scene_has_reference,
                    "reference_path": reference_path,
                    "aspect_ratio": aspect_ratio,
                    "log_id": scene_id
                }
                image_tasks.append(task)
            
            elif composition == "montage":
                scene_prompt = scene.get("image_prompt")
                for shot in scene.get("shots", []):
                    shot_id = shot["shot_id"]
                    task_id = f"{scene_id}_{shot_id}"
                    shot_prompt = shot.get("image_prompt")

                    # Combine scene and shot prompts for better context.
                    # This helps maintain consistency across shots in a montage.
                    prompt_parts = []
                    if scene_prompt:
                        prompt_parts.append(scene_prompt)
                    if shot_prompt:
                        prompt_parts.append(shot_prompt)
                    
                    final_prompt = ", ".join(prompt_parts)

                    if not final_prompt:
                        self.logger.warning(f"No image prompt found for shot {task_id} in scene {scene_id}. Skipping.")
                        continue
                    
                    # Extract has_reference from shot data
                    shot_has_reference = shot.get("has_reference", False)
                    
                    task = {
                        "task_id": task_id,
                        "prompt": final_prompt,
                        "has_reference": shot_has_reference,
                        "reference_path": reference_path,
                        "aspect_ratio": aspect_ratio,
                        "log_id": task_id
                    }
                    image_tasks.append(task)
        
        # Execute tasks concurrently
        image_paths = {}
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_task = {
                executor.submit(self._generate_single_image, task): task 
                for task in image_tasks
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    image_path = future.result()
                    image_paths[task["task_id"]] = image_path
                    self.logger.info(f"✅ Generated image for {task['log_id']}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to generate image for {task['log_id']}: {str(e)}")
                    # Do not raise, allow other images to be generated
        
        return image_paths
    
    def _generate_single_image(self, task: Dict[str, Any]) -> Path:
        """Generate a single image using fal.ai"""
        
        task_id = task["task_id"]
        prompt = task["prompt"]
        has_reference = task["has_reference"]
        reference_path = task["reference_path"]
        aspect_ratio = task["aspect_ratio"]
        log_id = task["log_id"]

        # Load image description data and replace variable names in prompt
        image_desc_data = self._load_image_description_data()
        if image_desc_data:
            prompt = self._replace_variable_in_prompt(prompt, image_desc_data)

        aspect_ratio_map = {
            "16:9": "landscape_16_9",
            "9:16": "portrait_16_9",
            "1:1": "square_1_1",
            "4:3": "landscape_4_3"
        }
        image_size = aspect_ratio_map.get(aspect_ratio, "portrait_16_9")
        
        start_time = time.time()
        
        retries = 3
        backoff_factor = 2

        for attempt in range(retries):
            try:
                if has_reference and reference_path:
                    # Use image-to-image model with reference
                    self.logger.debug(f"Using reference image for {log_id}")
                    
                    # Upload reference image
                    ref_url = fal_client.upload_file(str(reference_path))
                    
                    log_api_call(self.logger, "fal.ai", "flux-pro/kontext/max", 
                               {"prompt": prompt, "reference_image": "provided"})
                    
                    result = fal_client.submit(
                        "fal-ai/flux-pro/kontext/max",
                        arguments={
                            "prompt": prompt,
                            "image_url": ref_url,
                            "aspect_ratio": aspect_ratio,
                            "num_inference_steps": 28,
                            "guidance_scale": 3.5,
                            "num_images": 1,
                            "enable_safety_checker": False
                        }
                    )
                else:
                    # Use text-to-image model
                    log_api_call(self.logger, "fal.ai", "flux-pro/v1.1", 
                               {"prompt": prompt})
                    
                    result = fal_client.submit(
                        "fal-ai/flux-pro/v1.1",
                        arguments={
                            "prompt": prompt,
                            "image_size": image_size,
                            "num_inference_steps": 28,
                            "guidance_scale": 3.5,
                            "num_images": 1,
                            "enable_safety_checker": False
                        }
                    )
                
                # Wait for completion
                result = result.get()
                
                # Download the generated image
                image_url = result["images"][0]["url"]
                image_path = get_media_path(self.run_folder, "image", task_id, "jpg")
                
                self._download_file(image_url, image_path)
                
                duration = time.time() - start_time
                log_media_generation(self.logger, "image", log_id, 
                                   {"prompt": prompt, "provider": "fal", "has_reference": has_reference},
                                   image_path, duration)
                
                return image_path
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1}/{retries} failed for image {log_id}: {str(e)}")
                if attempt + 1 == retries:
                    self.logger.error(f"Failed to generate image for {log_id} after {retries} attempts.")
                    raise

                sleep_time = backoff_factor ** attempt
                self.logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
    
    def _generate_image_with_openai(self, task: Dict[str, Any]) -> Path:
        """Generate a single image using OpenAI """
        if not self.openai_client:
            raise Exception("OpenAI API key not configured. Please set OPENAI_API_KEY in your config.")

        task_id = task["task_id"]
        prompt = task["prompt"]
        has_reference = task["has_reference"]
        reference_path = task["reference_path"]
        aspect_ratio = task["aspect_ratio"]
        log_id = task["log_id"]

        start_time = time.time()
        self.logger.info(f"🎨 Starting image generation for {log_id}")
        self.logger.info(f"📝 Prompt: {prompt}")
        self.logger.info(f"📐 Aspect ratio: {aspect_ratio}")
        if has_reference:
            self.logger.info(f"🖼️ Using reference image: {reference_path}")

        aspect_ratio_to_size = {
            "16:9": "1536x1024",
            "9:16": "1024x1536",
            "1:1": "1024x1024",
            "4:3": "1024x1024", # Fallback for 4:3
        }
        size = aspect_ratio_to_size.get(aspect_ratio, "1024x1536") # Default to portrait
        self.logger.debug(f"📏 Image size: {size}")

        retries = 3
        backoff_factor = 2

        for attempt in range(retries):
            try:
                if has_reference and reference_path:
                    response = self._generate_openai_image_with_reference(prompt, reference_path, size, log_id)
                else:
                    response = self._generate_openai_image_from_prompt(prompt, size, log_id)

                b64_json = response.data[0].b64_json
                if not b64_json:
                    raise Exception("API response did not contain a base64 image.")

                # Decode the base64 string
                image_data = base64.b64decode(b64_json)
                
                # Save original image
                original_image_path = get_media_path(self.run_folder, "image", task_id, "png")
                with open(original_image_path, "wb") as f:
                    f.write(image_data)
                self.logger.info(f"💾 Saved original image to: {original_image_path}")
                
                # Apply color correction
                self.logger.info(f"🎨 Applying color correction to image {log_id}")
                self.logger.debug(f"Color correction parameters: R={0.90}, G={1.0}, B={1.25}")
                corrected_image_data = correct_image_tint_in_memory(image_data)
                
                # Save the corrected image in the corrected_images directory
                corrected_image_path = self.run_folder / "corrected_images" / f"{task_id}.png"
                with open(corrected_image_path, "wb") as f:
                    f.write(corrected_image_data)
                self.logger.info(f"💾 Saved color-corrected image to: {corrected_image_path}")

                duration = time.time() - start_time
                self.logger.info(f"⏱️ Image generation and correction completed in {duration:.2f} seconds")
                
                log_media_generation(self.logger, "image", log_id,
                                   {
                                       "prompt": prompt,
                                       "provider": "openai",
                                       "has_reference": has_reference,
                                       "color_corrected": True,
                                       "original_path": str(original_image_path),
                                       "corrected_path": str(corrected_image_path),
                                       "duration": duration
                                   },
                                   corrected_image_path, duration)

                return corrected_image_path

            except Exception as e:
                self.logger.warning(f"⚠️ Attempt {attempt + 1}/{retries} failed for image {log_id} with OpenAI: {str(e)}")
                if attempt + 1 == retries:
                    self.logger.error(f"❌ Failed to generate image for {log_id} with OpenAI after {retries} attempts.")
                    raise

                sleep_time = backoff_factor ** attempt
                self.logger.info(f"⏳ Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)

    def _generate_openai_image_with_reference(self, prompt: str, reference_path: Path, size: str, log_id: str):
        """Generate an image using OpenAI's image edit endpoint with a reference image"""
        self.logger.info(f"🔄 Attempting OpenAI image edit for {log_id} using reference image")
        
        log_api_call(self.logger, "openai", "images.edit",
                    {"prompt": prompt, "model": "gpt-image-1", "size": size, "quality": "high", "reference": "provided"})

        with open(reference_path, "rb") as image_file:
            response = self.openai_client.images.edit(
                model="gpt-image-1",
                image=image_file,
                prompt=prompt,
                size=size,
                quality="high",
                n=1
            )
        return response

    def _generate_openai_image_from_prompt(self, prompt: str, size: str, log_id: str):
        """Generate an image using OpenAI's text-to-image endpoint"""
        self.logger.info(f"🔄 Attempting OpenAI image generation for {log_id}")
        
        log_api_call(self.logger, "openai", "images.generate",
                    {"prompt": prompt, "model": "gpt-image-1", "size": size, "quality": "high", "moderation": "low"})

        response = self.openai_client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=size,
            quality="high",
            moderation="low",
            n=1
        )
        return response
    
    def _generate_image_with_replicate(self, task: Dict[str, Any]) -> Path:
        """Generate a single image using Replicate's flux-kontext-max model"""
        if not self.replicate_client:
            raise Exception("Replicate API key not configured. Please set REPLICATE_API_KEY in your config.")

        task_id = task["task_id"]
        prompt = task["prompt"]
        reference_path = task["reference_path"]
        aspect_ratio = task["aspect_ratio"]
        log_id = task["log_id"]

        start_time = time.time()
        self.logger.info(f"🎨 Starting Replicate image generation for {log_id}")
        self.logger.info(f"📝 Prompt: {prompt}")
        self.logger.info(f"📐 Aspect ratio: {aspect_ratio}")
        self.logger.info(f"🖼️ Using reference image: {reference_path}")

        retries = 3
        backoff_factor = 2

        for attempt in range(retries):
            try:
                self.logger.info(f"🔄 Attempting Replicate image generation for {log_id}")
                
                log_api_call(self.logger, "replicate", "black-forest-labs/flux-kontext-max",
                           {"prompt": prompt, "reference_image": "provided"})

                # Run the model
                input = {
                    "prompt": prompt,
                    "input_image": reference_path,
                    "output_format": "png",
                    "aspect_ratio": "9:16"
                }

                output = self.replicate_client.run(
                    "black-forest-labs/flux-kontext-max",
                    input=input
                )

                if not output:
                    raise Exception("Empty response from Replicate API")

                # Save the generated image
                image_path = get_media_path(self.run_folder, "image", task_id, "png")
                with open(image_path, "wb") as f:
                    f.write(output.read())
                self.logger.info(f"💾 Saved image to: {image_path}")

                duration = time.time() - start_time
                self.logger.info(f"⏱️ Image generation completed in {duration:.2f} seconds")
                
                log_media_generation(self.logger, "image", log_id,
                                   {
                                       "prompt": prompt,
                                       "provider": "replicate",
                                       "has_reference": True,
                                       "color_corrected": False,
                                       "path": str(image_path),
                                       "duration": duration
                                   },
                                   image_path, duration)

                return image_path

            except Exception as e:
                self.logger.warning(f"⚠️ Attempt {attempt + 1}/{retries} failed for image {log_id} with Replicate: {str(e)}")
                if attempt + 1 == retries:
                    self.logger.error(f"❌ Failed to generate image for {log_id} with Replicate after {retries} attempts.")
                    raise

                sleep_time = backoff_factor ** attempt
                self.logger.info(f"⏳ Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
    
    def _generate_all_videos(self, script_data: Dict[str, Any], image_paths: Dict[str, Path]) -> Dict[str, Path]:
        """Generate all videos concurrently"""
        
        scenes = script_data["scenes"]
        video_config = script_data["video_config"]
        aspect_ratio = video_config.get("aspect_ratio", "9:16")
        
        video_tasks = []
        
        # Prepare tasks for concurrent execution
        for scene in scenes:
            scene_id = scene["scene_id"]
            composition = scene.get("scene_composition", "single_shot")
            
            if composition == "single_shot":
                image_path = image_paths.get(scene_id)
                if not image_path:
                    self.logger.error(f"No image found for scene {scene_id}")
                    continue
                    
                task = {
                    "task_type": "single_video",
                    "scene_id": scene_id,
                    "video_id": scene_id,
                    "action_prompt": scene["action_prompt"],
                    "image_path": image_path,
                    "duration": scene.get("duration", 5),
                    "aspect_ratio": aspect_ratio
                }
                video_tasks.append(task)
            
            elif composition == "montage":
                shots = scene.get("shots", [])
                if not shots:
                    self.logger.error(f"No shots found for montage scene {scene_id}")
                    continue
                
                # Verify all images exist before creating task
                all_images_exist = True
                for shot in shots:
                    shot_id = shot["shot_id"]
                    task_id = f"{scene_id}_{shot_id}"
                    if task_id not in image_paths:
                        self.logger.warning(f"Image not found for {task_id} in montage {scene_id}")
                        all_images_exist = False

                if not all_images_exist:
                    self.logger.error(f"Missing one or more images for montage scene {scene_id}, skipping.")
                    continue
                
                task = {
                    "task_type": "montage",
                    "scene_id": scene_id,
                    "shots": shots,
                    "image_paths": image_paths,
                    "total_duration": scene.get("duration"),
                    "aspect_ratio": aspect_ratio
                }
                video_tasks.append(task)

        # Execute tasks concurrently
        video_paths = {}
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_task = {}
            for task in video_tasks:
                if task["task_type"] == "single_video":
                    future = executor.submit(self._generate_single_video, task)
                elif task["task_type"] == "montage":
                    future = executor.submit(self._generate_montage_from_shots, task)
                future_to_task[future] = task
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    video_path = future.result()
                    video_paths[task["scene_id"]] = video_path
                    self.logger.info(f"✅ Generated video for {task['scene_id']}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to generate video for {task['scene_id']}: {str(e)}")
                    # Do not raise, allow other videos to be generated
        
        return video_paths
    
    def _generate_single_video(self, task: Dict[str, Any]) -> Path:
        """Generate a single video using fal.ai"""
        
        video_id = task["video_id"]
        action_prompt = task["action_prompt"]
        image_path = task["image_path"]
        duration = task["duration"]
        aspect_ratio = task["aspect_ratio"]
        
        # Ensure duration is valid for fal.ai API (only accepts '5' or '10')
        if duration <= 5:
            api_duration = "5"
        else:
            api_duration = "10"
        
        start_time = time.time()
        
        retries = 3
        backoff_factor = 2

        for attempt in range(retries):
            try:
                # Upload the input image
                image_url = fal_client.upload_file(str(image_path))
                
                log_api_call(self.logger, "fal.ai", "kling-video/v2.1/pro/image-to-video", 
                           {"prompt": action_prompt, "duration": api_duration})
                
                result = fal_client.submit(
                    "fal-ai/kling-video/v2.1/pro/image-to-video",
                    arguments={
                        "prompt": action_prompt,
                        "image_url": image_url,
                        "duration": api_duration,
                        "aspect_ratio": aspect_ratio
                    }
                )
                
                # Wait for completion
                result = result.get()
                
                # Download the generated video
                video_url = result["video"]["url"]
                video_path = get_media_path(self.run_folder, "video", video_id, "mp4")
                
                self._download_file(video_url, video_path)
                
                generation_duration = time.time() - start_time
                log_media_generation(self.logger, "video", video_id, 
                                   {"prompt": action_prompt, "duration": api_duration, "requested_duration": duration},
                                   video_path, generation_duration)
                
                return video_path
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1}/{retries} failed for video {video_id}: {str(e)}")
                if attempt + 1 == retries:
                    self.logger.error(f"Failed to generate video for {video_id} after {retries} attempts.")
                    raise
                
                sleep_time = backoff_factor ** attempt
                self.logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
    
    def _generate_montage_from_shots(self, task: Dict[str, Any]) -> Path:
        """Generates a video montage by creating a video for each shot and concatenating them."""
        
        scene_id = task["scene_id"]
        shots = task["shots"]
        image_paths = task["image_paths"]
        aspect_ratio = task["aspect_ratio"]
        total_duration = task["total_duration"]
        
        start_time = time.time()
        
        try:
            shot_video_tasks = []
            for shot in shots:
                shot_id = shot["shot_id"]
                task_id = f"{scene_id}_{shot_id}"
                image_path = image_paths.get(task_id)

                if not image_path:
                    self.logger.warning(f"Image not found for {task_id}, skipping this shot.")
                    continue
                
                shot_task = {
                    "video_id": task_id,
                    "action_prompt": shot["action_prompt"],
                    "image_path": image_path,
                    "duration": float(shot.get("frame_duration", 0.5)),  # Convert to float
                    "aspect_ratio": aspect_ratio
                }
                shot_video_tasks.append(shot_task)

            # Generate videos for each shot concurrently
            shot_video_paths = []
            with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                future_to_task = {
                    executor.submit(self._generate_single_video, shot_task): shot_task 
                    for shot_task in shot_video_tasks
                }
                for future in as_completed(future_to_task):
                    shot_task = future_to_task[future]
                    try:
                        video_path = future.result()
                        shot_video_paths.append(video_path)
                        self.logger.info(f"✅ Generated video for shot {shot_task['video_id']}")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to generate video for shot {shot_task['video_id']}: {str(e)}")
                        # Continue to try and build montage with successful clips
            
            if not shot_video_paths:
                raise Exception(f"No video clips were generated for montage scene {scene_id}")

            # Clip videos to their specified frame durations (last shot gets +1 second)
            self.logger.info(f"Clipping {len(shot_video_paths)} videos to frame durations for scene {scene_id}")
            clipped_clips = []
            source_clips_to_close = []
            
            for i, video_path in enumerate(shot_video_paths):
                # Validate video file exists
                if not video_path or not video_path.exists():
                    self.logger.error(f"Video file does not exist: {video_path}")
                    continue
                
                # Get the corresponding shot for frame_duration
                shot = shots[i] if i < len(shots) else shots[-1]  # Fallback to last shot
                frame_duration = float(shot.get("frame_duration", 0.5))  # Convert to float
                
                # Add 1 second to the last shot
                if i == len(shot_video_paths) - 1:
                    target_duration = frame_duration + 1
                    self.logger.debug(f"Last shot {i+1}: extending to {target_duration}s (frame_duration + 1)")
                else:
                    target_duration = frame_duration
                    self.logger.debug(f"Shot {i+1}: clipping to {target_duration}s")
                
                # Load the video clip with error handling
                try:
                    clip = VideoFileClip(str(video_path))
                    source_clips_to_close.append(clip)
                    
                    # Validate the clip was loaded successfully
                    if clip is None:
                        self.logger.error(f"Failed to load video clip: {video_path} (VideoFileClip returned None)")
                        continue
                    
                    # Check if clip has valid duration
                    if not hasattr(clip, 'duration') or clip.duration is None or clip.duration <= 0:
                        self.logger.error(f"Video clip has invalid duration: {video_path} (duration: {getattr(clip, 'duration', 'None')})")
                        if clip:
                            clip.close()
                        continue
                    
                except Exception as e:
                    self.logger.error(f"Failed to load video clip {video_path}: {str(e)}")
                    continue
                
                try:
                    # If video is shorter than target, loop it
                    if clip.duration < target_duration:
                        self.logger.debug(f"Video {i+1} duration ({clip.duration}s) < target ({target_duration}s), looping")
                        # Calculate how many loops we need
                        loops_needed = int(target_duration / clip.duration) + 1
                        looped_clip = concatenate_videoclips([clip] * loops_needed, method="compose")
                        
                        # Validate looped clip
                        if looped_clip is None:
                            self.logger.error(f"Failed to create looped clip for {video_path}")
                            continue
                            
                        clipped_clip = looped_clip.subclipped(0, target_duration)
                    else:
                        # Clip to target duration
                        clipped_clip = clip.subclipped(0, target_duration)
                    
                    # Validate the final clipped clip
                    if clipped_clip is None:
                        self.logger.error(f"Failed to create clipped clip for {video_path}")
                        continue
                    
                    clipped_clips.append(clipped_clip)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process video clip {video_path}: {str(e)}")
                    continue
            
            # Check if we have any valid clips to concatenate
            if not clipped_clips:
                raise Exception(f"No valid video clips were processed for montage scene {scene_id}")
            
            # Concatenate clipped video clips
            self.logger.info(f"Concatenating {len(clipped_clips)} clipped clips for scene {scene_id}")
            
            montage = None
            try:
                montage = concatenate_videoclips(clipped_clips, method="compose")
                
                # Validate the montage was created successfully
                if montage is None:
                    raise Exception(f"Failed to create montage (concatenate_videoclips returned None)")
                
            except Exception as e:
                # Clean up clips before re-raising
                if montage:
                    montage.close()
                for clip in source_clips_to_close:
                    if clip:
                        clip.close()
                raise Exception(f"Failed to concatenate video clips: {str(e)}")
            
            video_path = get_media_path(self.run_folder, "video", scene_id, "mp4")
            
            try:
                montage.write_videofile(
                    str(video_path), 
                    fps=24, 
                    codec="libx264", 
                    audio_codec="aac",
                    logger=None
                )
            except Exception as e:
                # Clean up clips before re-raising
                if montage:
                    montage.close()
                for clip in source_clips_to_close:
                    if clip:
                        clip.close()
                raise Exception(f"Failed to write video file: {str(e)}")
            
            # Close clips to release file handles
            if montage:
                montage.close()
            for clip in source_clips_to_close:
                if clip:
                    clip.close()

            generation_duration = time.time() - start_time
            log_media_generation(self.logger, "video_montage", scene_id, 
                               {"num_shots": len(shot_video_paths), "total_duration": total_duration},
                               video_path, generation_duration)
            
            return video_path
            
        except Exception as e:
            self.logger.error(f"Failed to create video montage for {scene_id}: {str(e)}")
            raise
    
    def _generate_all_audio(self, script_data: Dict[str, Any]) -> Dict[str, Path]:
        """Generate all audio (individual voiceovers and background music)"""
        
        audio_files = {}
        config = script_data["video_config"]
        
        # Generate individual voiceovers if needed
        if config.get("needs_voiceover", False):
            self.logger.info("Generating individual voiceovers...")
            audio_files["voiceovers"] = self._generate_individual_voiceovers(script_data)
        
        # Generate background music if needed
        if config.get("needs_background_music", False):
            self.logger.info("Generating background music...")
            # Calculate total duration for background music
            total_duration = sum(scene.get("duration", 5) for scene in script_data["scenes"])
            bg_music_path = self._generate_background_music(script_data, total_duration)
            if bg_music_path:
                audio_files["background_music"] = bg_music_path
        
        return audio_files
    
    def _generate_individual_voiceovers(self, script_data: Dict[str, Any]) -> Dict[str, Path]:
        """Generate individual voiceover files for each scene using ElevenLabs TTS"""
        
        config = script_data["video_config"]
        scenes = script_data["scenes"]
        voice_characteristics = config.get("voice_characteristics", "Professional, clear, engaging")
        
        voiceover_files = {}
        
        for scene in scenes:
            scene_id = scene["scene_id"]
            voiceover_text = scene.get("voiceover_text", "").strip()
            
            # Skip if no voiceover text for this scene
            if not voiceover_text:
                self.logger.info(f"Skipping voiceover for {scene_id} - no text provided")
                continue
            
            retries = 3
            backoff_factor = 2
            
            for attempt in range(retries):
                try:
                    start_time = time.time()
                    
                    log_api_call(self.logger, "fal.ai", "elevenlabs/tts/multilingual-v2", 
                               {"scene_id": scene_id, "text_length": len(voiceover_text), "voice": voice_characteristics})
                    
                    result = fal_client.submit(
                        "fal-ai/elevenlabs/tts/multilingual-v2",
                        arguments={
                            "text": voiceover_text,
                            "voice": "Rachel",  # Default professional voice
                            "model_id": "eleven_multilingual_v2",
                            "voice_settings": {
                                "stability": 0.5,
                                "similarity_boost": 0.75,
                                "style": 0,
                                "use_speaker_boost": True
                            }
                        }
                    )
                    
                    result = result.get()
                    
                    # Debug: print the result structure
                    self.logger.debug(f"Voiceover result structure for {scene_id}: {result}")
                    
                    # Try different possible keys for audio URL
                    audio_url = None
                    if "audio_url" in result:
                        audio_url = result["audio_url"]
                    elif "url" in result:
                        audio_url = result["url"]
                    elif "audio" in result and isinstance(result["audio"], dict) and "url" in result["audio"]:
                        audio_url = result["audio"]["url"]
                    elif "audio" in result and isinstance(result["audio"], str):
                        audio_url = result["audio"]
                    else:
                        # Log all available keys
                        self.logger.error(f"Could not find audio URL in result for {scene_id}. Available keys: {list(result.keys())}")
                        raise KeyError("Could not find audio URL in response")
                    
                    if not audio_url:
                        raise ValueError("Audio URL is empty")
                    
                    audio_path = get_media_path(self.run_folder, "voiceover", scene_id, "mp3")
                    
                    self._download_file(audio_url, audio_path)
                    
                    duration = time.time() - start_time
                    log_media_generation(self.logger, "voiceover", scene_id, 
                                       {"text_length": len(voiceover_text)}, audio_path, duration)
                    
                    voiceover_files[scene_id] = audio_path
                    break # Success, exit retry loop
                    
                except Exception as e:
                    self.logger.warning(f"Attempt {attempt + 1}/{retries} failed for voiceover {scene_id}: {str(e)}")
                    if attempt + 1 == retries:
                        self.logger.error(f"Failed to generate voiceover for {scene_id} after {retries} attempts. Continuing...")
                        break # break from retry loop and continue to next scene
                    
                    sleep_time = backoff_factor ** attempt
                    self.logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
        
        return voiceover_files
    
    def _generate_background_music(self, script_data: Dict[str, Any], total_duration: int) -> Optional[Path]:
        """Generate background music using ACE audio generation"""
        
        config = script_data["video_config"]
        music_description = config.get("music_description", "Upbeat, engaging background music")
        
        try:
            start_time = time.time()
            
            log_api_call(self.logger, "fal.ai", "ace-step/prompt-to-audio", 
                       {"description": music_description, "duration": total_duration})
            
            result = fal_client.submit(
                "fal-ai/ace-step/prompt-to-audio",
                arguments={
                    "prompt": f"Create instrumental background music: {music_description}. No lyrics or vocals. Duration: {total_duration} seconds",
                    "instrumental": True,  # Ensure no lyrics
                    "duration": total_duration,  # Duration in seconds
                    "number_of_steps": 27,
                    "scheduler": "euler",
                    "guidance_type": "apg",
                    "granularity_scale": 10,
                    "guidance_interval": 0.5,
                    "guidance_interval_decay": 0,
                    "guidance_scale": 15,
                    "minimum_guidance_scale": 3,
                    "tag_guidance_scale": 5,
                    "lyric_guidance_scale": 0  # Set to 0 to avoid lyric influence
                }
            )
            
            result = result.get()
            
            # Debug: print the result structure
            self.logger.debug(f"Background music result structure: {result}")
            
            # Try different possible keys for audio URL
            audio_url = None
            if "audio_file" in result and isinstance(result["audio_file"], dict) and "url" in result["audio_file"]:
                audio_url = result["audio_file"]["url"]
            elif "audio" in result and isinstance(result["audio"], dict) and "url" in result["audio"]:
                audio_url = result["audio"]["url"]
            elif "url" in result:
                audio_url = result["url"]
            elif "audio" in result and isinstance(result["audio"], str):
                audio_url = result["audio"]
            else:
                # Log all available keys
                self.logger.error(f"Could not find audio URL in background music result. Available keys: {list(result.keys())}")
                raise KeyError("Could not find audio URL in background music response")
            
            if not audio_url:
                raise ValueError("Background music audio URL is empty")
            music_path = get_media_path(self.run_folder, "background_music", "background", "wav")
            
            self._download_file(audio_url, music_path)
            
            duration = time.time() - start_time
            log_media_generation(self.logger, "background_music", "full", 
                               {"description": music_description}, music_path, duration)
            
            return music_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate background music: {str(e)}")
            raise
    
    def _download_file(self, url: str, output_path: Path):
        """Download file from URL to local path"""
        
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Validate the downloaded file
            if not output_path.exists():
                raise Exception(f"Downloaded file does not exist: {output_path}")
            
            file_size = output_path.stat().st_size
            if file_size == 0:
                raise Exception(f"Downloaded file is empty: {output_path}")
            
            # For video files, do additional validation
            if output_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                if file_size < 1024:  # Less than 1KB is likely not a valid video
                    raise Exception(f"Downloaded video file is too small ({file_size} bytes): {output_path}")
                
                # Try to validate the video file with MoviePy
                try:
                    test_clip = VideoFileClip(str(output_path))
                    if test_clip is None:
                        raise Exception("VideoFileClip returned None")
                    if not hasattr(test_clip, 'duration') or test_clip.duration is None or test_clip.duration <= 0:
                        raise Exception(f"Invalid video duration: {getattr(test_clip, 'duration', 'None')}")
                    duration = test_clip.duration
                    test_clip.close()
                    self.logger.debug(f"Video file validation passed: {output_path} (size: {file_size} bytes, duration: {duration}s)")
                except Exception as video_error:
                    # Clean up the invalid file
                    if output_path.exists():
                        output_path.unlink()
                    raise Exception(f"Downloaded video file is invalid: {video_error}")
                    
        except Exception as e:
            self.logger.error(f"Failed to download file from {url}: {str(e)}")
            raise 