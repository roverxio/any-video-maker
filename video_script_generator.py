"""
Video Script Generator
Converts video summary output into structured JSON format for video production using GPT-4o
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VideoScriptGenerator:
    """Generates structured JSON scripts from video summaries using GPT-4o"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = openai.OpenAI(api_key=api_key)
        
    def generate_script(self, 
                       visual_summary: str, 
                       voiceover_data: Dict[str, Any],
                       video_config: Dict[str, Any],
                       user_prompt: str,
                       reference_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a structured JSON script from video summary components using GPT-4o
        
        Args:
            visual_summary: The generated visual summary text
            voiceover_data: Dictionary containing voiceover text and scenes
            video_config: Dictionary with total_duration, aspect_ratio, video_style
            user_prompt: Original user prompt
            reference_url: Optional reference image URL
        
        Returns:
            Structured JSON script for video production
        """
        
        try:
            # Prepare the context for GPT-4o
            context = self._prepare_context(
                visual_summary, voiceover_data, video_config, user_prompt, reference_url
            )
            
            # Generate the structured script using GPT-4o
            script = self._call_gpt4o_for_script_generation(context)
            
            # Validate and enhance the generated script
            validated_script = self._validate_and_enhance_script(script, video_config)
            
            self.logger.info(f"Generated structured script with {len(validated_script.get('scenes', []))} scenes using GPT-4o")
            return validated_script
            
        except Exception as e:
            self.logger.error(f"Error generating script: {str(e)}")
            raise
    
    def _prepare_context(self, visual_summary: str, voiceover_data: Dict[str, Any], 
                        video_config: Dict[str, Any], user_prompt: str, 
                        reference_url: Optional[str] = None) -> str:
        """Prepare context string for GPT-4o"""
        
        context = f"""
USER REQUEST: {user_prompt}

VISUAL SUMMARY:
{visual_summary}

VOICEOVER DATA:
{json.dumps(voiceover_data, indent=2)}

VIDEO CONFIGURATION:
{json.dumps(video_config, indent=2)}

REFERENCE IMAGE URL: {reference_url or "None provided"}

TASK: Convert the above information into a structured JSON script for video production.
"""
        return context
    
    def _call_gpt4o_for_script_generation(self, context: str) -> Dict[str, Any]:
        """Call GPT-4o to generate the structured video script"""
        
        system_prompt = """You are an expert video production assistant. Your task is to convert video summary information into a comprehensive, structured JSON script for automated video production.

REQUIRED JSON STRUCTURE:
{
  "video_config": {
    "total_duration": number,
    "aspect_ratio": string,
    "style": string,
    "pacing": string (fast/medium/slow),
    "content_type": string (advertisement/narrative/educational/presentation),
    "needs_voiceover": boolean,
    "needs_background_music": boolean,
    "voice_characteristics": string (detailed description),
    "music_description": string (detailed description)
  },
  "scenes": [
    {
      "scene_id": string,
      "timestamp": string (format: "start-end"),
      "duration": number,
      "scene_composition": string (single_shot/montage),
      "voiceover_text": string,
      "has_reference": boolean,
      "image_prompt": string (for single_shot only),
      "action_prompt": string (for single_shot only),
      "shots": [
        {
          "shot_id": string,
          "frame_duration": number,
          "image_prompt": string,
          "action_prompt": string
        }
      ] (for montage only)
    }
  ]
}

INSTRUCTIONS:
1. Analyze the visual summary to extract individual scenes with their descriptions and durations
2. Match voiceover text to appropriate scenes based on content and timing
3. Set "has_reference" to true ONLY for scenes that actually feature/show the reference image/product itself (not related scenes)
4. Set "scene_composition" to "single_shot" for single continuous shots or "montage" for multiple quick cuts/transitions
5. For SINGLE_SHOT scenes: Include "image_prompt" and "action_prompt" at the scene level, do NOT include "shots" array
6. For MONTAGE scenes: Include "shots" array with individual shots, each having shot_id, frame_duration, image_prompt, and action_prompt. The sum of all frame_durations MUST equal the scene duration. Do NOT include scene-level image_prompt/action_prompt
7. Voiceover text always stays at the scene level, never at the shot level
8. Create detailed image prompts optimized for AI image generation (be specific about lighting, composition, style)
9. Generate action prompts that describe desired motion/animation for each shot/scene
10. Set video configuration parameters based on the content type and user request
11. Ensure timestamps are accurate and scenes flow logically
12. Make voice characteristics and music descriptions detailed and contextually appropriate

Be intelligent about scene analysis - consider narrative structure, emotional flow, and production requirements. Return ONLY the JSON, no additional text."""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        try:
            # Get the response content
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code block wrapper if present
            if content.startswith('```json'):
                content = content[7:]  # Remove ```json
            if content.startswith('```'):
                content = content[3:]   # Remove ```
            if content.endswith('```'):
                content = content[:-3]  # Remove closing ```
            
            # Parse the JSON response
            script_json = json.loads(content.strip())
            return script_json
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse GPT-4o response as JSON: {e}")
            self.logger.error(f"Raw response: {response.choices[0].message.content}")
            raise ValueError("GPT-4o did not return valid JSON")
    
    def _validate_and_enhance_script(self, script: Dict[str, Any], 
                                   original_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enhance the generated script"""
        
        # Ensure required fields exist
        if "video_config" not in script:
            script["video_config"] = {}
        
        if "scenes" not in script:
            script["scenes"] = []
        
        # Validate video config
        video_config = script["video_config"]
        video_config.setdefault("total_duration", original_config.get("total_duration", 30))
        video_config.setdefault("aspect_ratio", original_config.get("aspect_ratio", "9:16"))
        video_config.setdefault("style", original_config.get("video_style", "commercial"))
        
        # Validate scenes
        total_duration = 0
        for i, scene in enumerate(script["scenes"]):
            # Ensure required scene fields
            scene.setdefault("scene_id", f"scene_{i+1}")
            scene.setdefault("duration", 5)
            scene.setdefault("scene_composition", "single_shot")
            scene.setdefault("has_reference", False)
            
            # Handle scene composition structure
            composition = scene.get("scene_composition", "single_shot")
            if composition == "single_shot":
                # Ensure scene-level prompts exist, remove shots if present
                scene.setdefault("image_prompt", "")
                scene.setdefault("action_prompt", "")
                if "shots" in scene:
                    del scene["shots"]
            elif composition == "montage":
                # Ensure shots array exists, remove scene-level prompts
                if "shots" not in scene:
                    scene["shots"] = []
                if "image_prompt" in scene:
                    del scene["image_prompt"]
                if "action_prompt" in scene:
                    del scene["action_prompt"]
                
                # Validate shots duration sum equals scene duration
                if scene["shots"]:
                    total_shots_duration = sum(shot.get("frame_duration", 0) for shot in scene["shots"])
                    if abs(total_shots_duration - scene["duration"]) > 0.1:  # Allow small floating point differences
                        self.logger.warning(f"Scene {scene['scene_id']}: shots duration ({total_shots_duration}) doesn't match scene duration ({scene['duration']})")
            
            # Update timestamps
            start_time = total_duration
            duration = scene["duration"]
            end_time = start_time + duration
            scene["timestamp"] = f"{start_time}-{end_time}"
            total_duration = end_time
        
        # Update total duration based on actual scene durations
        script["video_config"]["total_duration"] = total_duration
        
        return script

def main():
    """Main function for testing the video script generator"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Test data
        sample_visual_summary = """VIDEO CONFIG:
- Total Duration: 15s
- Aspect Ratio: 9:16
- Video Style: commercial

Scene 1: A dynamic close-up of a tennis ball bouncing on a court in bright natural lighting. [3s]

Scene 2: A frustrated player misses a shot, looking down at their racquet with disappointed expression. [4s]

Scene 3: The new tennis racquet is spotlighted against a dark background, showcasing its sleek design and premium materials. [5s]

Scene 4: A confident player executes a perfect serve with the new racquet, celebrating with a victorious expression. [3s]"""
        
        sample_voiceover = {
            'voiceover_text': 'Scene 1: Ready for the perfect game?\nScene 2: We\'ve all been there with the wrong gear.\nScene 3: This changes everything.\nScene 4: Your game, elevated.',
            'scenes': [
                {'scene': 'Scene 1', 'voiceover': 'Ready for the perfect game?'},
                {'scene': 'Scene 2', 'voiceover': 'We\'ve all been there with the wrong gear.'},
                {'scene': 'Scene 3', 'voiceover': 'This changes everything.'},
                {'scene': 'Scene 4', 'voiceover': 'Your game, elevated.'}
            ]
        }
        
        sample_config = {
            'total_duration': 15,
            'aspect_ratio': '9:16',
            'video_style': 'commercial'
        }
        
        # Initialize generator
        generator = VideoScriptGenerator(logger)
        
        # Generate script using GPT-4o
        script = generator.generate_script(
            visual_summary=sample_visual_summary,
            voiceover_data=sample_voiceover,
            video_config=sample_config,
            user_prompt="Create a tennis racquet advertisement video",
            reference_url="https://example.com/racquet.jpg"
        )
        
        # Output the result
        print("\n" + "="*50)
        print("GENERATED VIDEO SCRIPT JSON (via GPT-4o):")
        print("="*50)
        print(json.dumps(script, indent=2))
        
        # Save to file
        with open('generated_script.json', 'w') as f:
            json.dump(script, f, indent=2)
        
        logger.info("Script saved to 'generated_script.json'")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 