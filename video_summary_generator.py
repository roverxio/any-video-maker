"""
Script Generator using OpenAI GPT-4.1
Generates structured video scripts based on user input
"""

import json
import logging
import os
import re
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

def log_api_call(logger, service, method, data=None, response_info=None):
    """Simple logging function for API calls"""
    if response_info:
        logger.info(f"{service} {method}: {response_info}")
    else:
        logger.info(f"{service} {method} called")

# Load environment variables
load_dotenv()

# Global variable to store the generated summary
GENERATED_SUMMARY = None

class VideoSummaryGenerator:
    """Generates video summaries using OpenAI GPT-4o"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.client = OpenAI(api_key=config["OPENAI_API_KEY"])
        
    def generate_summary(self, user_prompt: str, reference_url: Optional[str] = None) -> Dict[str, Any]:
        """Generate a complete video summary from user prompt"""
        
        global GENERATED_SUMMARY
        
        self.logger.info("Generating video summary with OpenAI GPT-4o...")
        
        system_prompt = """You are a professional video concept summarizer.

Your task is to generate a **short, structured summary for a video** that can be created from:
1. **One input image** (such as a product, logo, or character)
2. **One text instruction** (which could describe purpose, emotion, category, or scene ideas)

The goal is to produce a video summary that maintains **visual and contextual consistency** across scenes, with clearly distinguished moments and feasible production requirements.

## Rules

* **Analyze both inputs together** to understand the true user intent and context. Combine what's visible in the image with the instruction to infer the overall purpose and tone.

* **Do NOT fabricate specific details** that would typically come from external sources (dates, prices, URLs, brand names not shown, specific statistics). Generic elements like countdown numbers or basic transitions are acceptable.

* **NEVER add meta-content** such as credits, title cards, "subscribe" messages, copyright notices, production information, or any text overlays unless explicitly requested by the user. The video should contain only the actual content scenes.

* **Include necessary visual elements** for a coherent video, even if not explicitly mentioned. For example, if creating a sunglasses ad, people wearing sunglasses should be included even if not specified.

* **Keep scenes technically feasible** for AI video generation. Avoid overly complex transitions, multiple simultaneous actions, or intricate visual effects that would be difficult to render.

* **Do NOT reference the image file directly** (e.g., don't say "the uploaded logo"); instead, incorporate what's visually evident into the scene descriptions.

* **Structure the summary with separate scenes** that flow logically from one to another.

* **Include directorial elements** such as camera angles, lighting, and character expressions to create rich, descriptive visuals.

* **Keep the video genre flexible** – it can be a product ad, narrative, whimsical short, abstract animation, or single-speaker moment, depending on the combined input context.

* **Maintain a professional, cinematic tone** with clear scene progression.

* **Final summary length**: ideally **3–5 scenes**, each with 1–2 sentences. Every scene should contain actual video content, not meta-elements.

* **Write each scene as a flowing paragraph** that naturally incorporates visual elements, camera movements, and directorial details. Do NOT use bullet points, subsections, or formatted labels like "Visual:" or "Camera Angle:". Each scene should read as a cohesive description.

* **Include duration estimates** for each scene in square brackets at the end, based on content complexity:
  - Simple shots/transitions: [3-5s]
  - Standard scenes: [5-7s]
  - Complex multi-element scenes: [7-10s]

## Aspect Ratio Rules

* Default to 9:16 (vertical) unless specified otherwise
* Intelligently detect aspect ratio from platform mentions:
  - "Instagram", "Reels", "TikTok", "Shorts" → 9:16
  - "YouTube", "TV", "cinema", "landscape" → 16:9
  - "Square", "carousel" → 1:1
* If user specifies dimensions or aspect ratio, use their preference

## Input Format

* `image`: A single visual reference
* `instruction`: A text description of what the user wants to convey with the video

## Output Format

Return the structured video summary with scene transitions AND video configuration. Format EXACTLY as follows:

VIDEO CONFIG:
- Total Duration: [sum of all scene durations]s
- Aspect Ratio: [ratio]
- Video Style: [detected style based on content]

Scene 1: Brief scene description incorporating visuals and camera work naturally. [5s]

Scene 2: Another flowing description without subsections or labels. [7s]

(Continue for all scenes)"""

        
        # Create user message with image
        user_message = self._create_user_message_with_image(user_prompt, reference_url)
        
        try:
            # Make API call to OpenAI
            log_api_call(self.logger, "OpenAI", "chat.completions.create", 
                        {"model": "gpt-4.1", "messages": [{"role": "system"}, {"role": "user"}]})
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=3000
            )
            
            # Get plain text response
            summary_content = response.choices[0].message.content
            log_api_call(self.logger, "OpenAI", "Response", 
                        response_info=f"Tokens used: {response.usage.total_tokens}")
            
            # Parse video config from response
            video_config = self._parse_video_config(summary_content)
            
            # Store summary globally for access by other scripts
            GENERATED_SUMMARY = summary_content
            
            # Return structured response with plain text content
            result = {
                "summary": summary_content,
                "video_config": video_config,
                "user_prompt": user_prompt,
                "reference_url": reference_url,
                "tokens_used": response.usage.total_tokens
            }
            
            self.logger.info(f"Generated summary successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {str(e)}")
            raise
    
    def generate_voiceover(self, visual_summary: str, user_prompt: str, auto_detect_style: bool = True) -> Dict[str, Any]:
        """Generate voice-over script from visual summary"""
        
        self.logger.info("Generating voice-over with auto-detected style...")
        
        # Combined system prompt that includes all styles and auto-detection
        auto_detect_prompt = """You are a professional voice-over scriptwriter who can intelligently adapt your style based on video content.

First, analyze the video summary to determine the most appropriate voice-over style:
- If it's a product showcase or advertisement → use promotional style (excited but genuine)
- If it's educational or about places/processes → use documentary style (natural wisdom)
- If it has characters or emotional journey → use storytelling style (intimate, personal)
- If it's artistic/abstract → use minimal style (sparse, poetic)
- Otherwise → use narrator style (friendly, conversational)

Then create NATURAL SPOKEN DIALOGUE using that style:

NARRATOR STYLE - General friendly narration:
- "Whoa, look at that..." or "Okay, so here's what happens..."
- Conversational, warm, engaging

DOCUMENTARY STYLE - Natural documentary narration:
- "You see, what's happening here is..."
- "Now this... this is interesting."
- Wisdom and gentle insights

PROMOTIONAL STYLE - Genuine enthusiasm:
- "Okay, check this out..."
- "You're gonna love this part"
- Real excitement, not fake hype

STORYTELLING STYLE - Intimate narrative:
- "So there I was..."
- "And then, out of nowhere..."
- Personal and emotional

MINIMAL STYLE - Sparse and poetic:
- Single powerful phrases
- "Rain falls."
- Let silence speak

CRITICAL TIMING CONSTRAINTS:
- Check the [duration] at the end of each scene
- For [3s] = max 7-8 words
- For [5s] = max 12-13 words
- For [7s] = max 17-18 words
- For [10s] = max 25 words

Write like you're talking to a friend, not reading ad copy."""
        
        # Use either auto-detect prompt or specific style prompt
        if auto_detect_style:
            system_prompt = auto_detect_prompt
            # Add original user prompt for context
            user_message = f"""Original request: {user_prompt}

Generate voice-over text for each scene in this video summary:

{visual_summary}

Format your response as:
Scene 1: [voice-over text]
Scene 2: [voice-over text]
(etc.)

Only provide the voice-over text, no additional commentary."""
        
        try:
            # Make API call
            log_api_call(self.logger, "OpenAI", "chat.completions.create", 
                        {"model": "gpt-4", "purpose": "voiceover generation"})
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            voiceover_text = response.choices[0].message.content
            log_api_call(self.logger, "OpenAI", "Response", 
                        response_info=f"Tokens used: {response.usage.total_tokens}")
            
            # Parse voice-over into structured format
            scenes_voiceover = self._parse_voiceover_response(voiceover_text)
            
            result = {
                "voiceover_text": voiceover_text,
                "scenes": scenes_voiceover,
                "style": "auto-detected",
                "tone": None,
                "tokens_used": response.usage.total_tokens
            }
            
            self.logger.info(f"Generated voice-over successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate voice-over: {str(e)}")
            raise
    
    def _parse_voiceover_response(self, voiceover_text: str) -> list:
        """Parse voice-over response into structured scene list"""
        scenes = []
        lines = voiceover_text.strip().split('\n')
        
        for line in lines:
            if line.strip() and line.strip().startswith('Scene'):
                # Extract scene number and voice-over text
                parts = line.split(':', 1)
                if len(parts) == 2:
                    scene_num = parts[0].strip()
                    vo_text = parts[1].strip()
                    
                    # Calculate word count and estimated speaking duration
                    word_count = len(vo_text.split())
                    speaking_duration = word_count / 2.5
                    
                    scenes.append({
                        "scene": scene_num,
                        "voiceover": vo_text,
                        "word_count": word_count,
                        "speaking_duration": round(speaking_duration, 1)
                    })
        
        return scenes
    
    def generate_complete_video_content(self, user_prompt: str, reference_url: Optional[str] = None) -> Dict[str, Any]:
        """Generate both visual summary and voice-over in one workflow"""
        
        # Generate visual summary first
        visual_result = self.generate_summary(user_prompt, reference_url)
        
        # Generate voice-over with auto-detection (passing original prompt for context)
        voiceover_result = self.generate_voiceover(
            visual_result["summary"],
            user_prompt=user_prompt,
            auto_detect_style=True
        )
        
        # Combine results
        complete_result = {
            "visual_summary": visual_result["summary"],
            "voiceover": voiceover_result["voiceover_text"],
            "voiceover_scenes": voiceover_result["scenes"],
            "video_config": visual_result["video_config"],
            "metadata": {
                "user_prompt": user_prompt,
                "reference_url": reference_url,
                "total_tokens": visual_result["tokens_used"] + voiceover_result["tokens_used"]
            }
        }
        
        return complete_result
    
    def _create_user_message_with_image(self, user_prompt: str, reference_url: Optional[str] = None) -> list:
        """Create user message with image and text"""
        
        if reference_url:
            # Format message for vision API with image
            return [
                {
                    "type": "text",
                    "text": f"Instruction: {user_prompt}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": reference_url
                    }
                }
            ]
        else:
            # Text only
            return [
                {
                    "type": "text", 
                    "text": f"Instruction: {user_prompt}"
                }
            ]

    def _create_user_message(self, user_prompt: str, reference_url: Optional[str] = None) -> str:
        """Legacy method - keeping for compatibility"""
        if reference_url:
            return f"User instruction: {user_prompt}\nReference URL: {reference_url}"
        return f"User instruction: {user_prompt}"
    
    def _parse_video_config(self, summary_content: str) -> Dict[str, Any]:
        """Parse video configuration from summary content"""
        config = {
            "total_duration": None,
            "aspect_ratio": "9:16",  # default
            "video_style": None
        }
        
        lines = summary_content.split('\n')
        in_config_section = False
        
        for line in lines:
            if "VIDEO CONFIG:" in line:
                in_config_section = True
                continue
            
            if in_config_section and line.strip():
                if line.startswith("Scene"):
                    # End of config section
                    break
                    
                if "Total Duration:" in line:
                    # Extract duration value
                    duration_match = re.search(r'(\d+)s', line)
                    if duration_match:
                        config["total_duration"] = int(duration_match.group(1))
                
                elif "Aspect Ratio:" in line:
                    # Extract aspect ratio
                    ratio_match = re.search(r'(\d+:\d+)', line)
                    if ratio_match:
                        config["aspect_ratio"] = ratio_match.group(1)
                
                elif "Video Style:" in line:
                    # Extract style (everything after the colon)
                    style = line.split(":", 1)[1].strip()
                    config["video_style"] = style
        
        return config
    
    

def get_summary() -> Optional[str]:
    """Get the currently stored summary"""
    return GENERATED_SUMMARY

def set_summary(summary: str) -> None:
    """Set the summary manually (useful for testing or external updates)"""
    global GENERATED_SUMMARY
    GENERATED_SUMMARY = summary

def main():
    """Main function to run the video summary generator"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Configuration - load from environment variables
    config = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
    }
    
    # Validate API key
    if not config["OPENAI_API_KEY"]:
        logger.error("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
        return 1
    
    try:
        # Initialize the generator
        generator = VideoSummaryGenerator(config, logger)
        
        # Get user inputs
        user_prompt = input("Enter your video prompt: ")
        reference_url = input("Enter reference image URL (or press Enter to skip): ").strip()
        
        if not reference_url:
            reference_url = None
        
        # Generate complete content with auto-detected voice-over style
        result = generator.generate_complete_video_content(
            user_prompt, 
            reference_url
        )
        
        print("\n" + "="*50)
        print("GENERATED VIDEO SUMMARY:")
        print("="*50)
        print(result["visual_summary"])
        
        print("\n" + "="*50)
        print("VOICE-OVER SCRIPT:")
        print("="*50)
        print(result["voiceover"])
        
        print("\n" + "="*50)
        print(f"Total tokens used: {result['metadata']['total_tokens']}")
        
        # The summary is still stored globally
        print(f"\nSummary stored globally. Access with get_summary(): {get_summary() is not None}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
    
    
