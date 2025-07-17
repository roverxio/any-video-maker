"""
Script Generator using OpenAI GPT-4.1
Generates structured video scripts based on user input
"""

import json
import logging
from typing import Dict, Any, Optional
from openai import OpenAI
from utils.logger import log_api_call


class ScriptGenerator:
    """Generates video summaries using OpenAI GPT-4.1"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.client = OpenAI(api_key=config["OPENAI_API_KEY"])
        
    def generate_script(self, user_prompt: str, reference_url: Optional[str] = None) -> Dict[str, Any]:
        """Generate a complete video script from user prompt"""
        
        self.logger.info("Generating video summary with OpenAI GPT-4.1...")
        
        system_prompt = """You are a professional video concept summarizer.

Your task is to generate a **short, structured summary for a video** that can be created from:
1. **One input image** (such as a product, logo, or character)
2. **One text instruction** (which could describe purpose, emotion, category, or scene ideas)

The goal is to produce a video summary that maintains **visual and contextual consistency** across scenes, with clearly distinguished moments and feasible production requirements.

## Rules

* **Analyze both inputs together** to understand the true user intent and context. Combine what's visible in the image with the instruction to infer the overall purpose and tone.

* **Do NOT fabricate specific details** that would typically come from external sources (dates, prices, URLs, brand names not shown, specific statistics). Generic elements like countdown numbers or basic transitions are acceptable.

* **Include necessary visual elements** for a coherent video, even if not explicitly mentioned. For example, if creating a sunglasses ad, people wearing sunglasses should be included even if not specified.

* **Keep scenes technically feasible** for AI video generation. Avoid overly complex transitions, multiple simultaneous actions, or intricate visual effects that would be difficult to render.

* **Do NOT reference the image file directly** (e.g., don't say "the uploaded logo"); instead, incorporate what's visually evident into the scene descriptions.

* **Structure the summary with separate scenes** that flow logically from one to another.

* **Include directorial elements** such as camera angles, lighting, and character expressions to create rich, descriptive visuals.

* **Keep the video genre flexible** – it can be a product ad, narrative, whimsical short, abstract animation, or single-speaker moment, depending on the combined input context.

* **Maintain a professional, cinematic tone** with clear scene progression.

* **Final summary length**: ideally **3–5 scenes**, each with 1–2 sentences.

## Input Format

* `image`: A single visual reference
* `instruction`: A text description of what the user wants to convey with the video

## Output Format

Return only the structured video summary with scene transitions. Do not include any headings or commentary."""

        
        # Create user message
        user_message = self._create_user_message(user_prompt, reference_url)
        
        try:
            # Make API call to OpenAI
            log_api_call(self.logger, "OpenAI", "chat.completions.create", 
                        {"model": "gpt-4.1", "messages": [{"role": "system"}, {"role": "user"}]})
            
            response = self.client.chat.completions.create(
                model="gpt-4.1",  # Using gpt-4o as it's the latest available
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=3000,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            script_content = response.choices[0].message.content
            log_api_call(self.logger, "OpenAI", "Response", 
                        response_info=f"Tokens used: {response.usage.total_tokens}")
            
            # Parse JSON response
            script_data = json.loads(script_content)
            
            # Validate and structure the script
            validated_script = self._validate_script(script_data)
            
            self.logger.info(f"Generated script with {len(validated_script['scenes'])} scenes")
            return validated_script
            
        except Exception as e:
            self.logger.error(f"Failed to generate script: {str(e)}")
            raise
    
    
