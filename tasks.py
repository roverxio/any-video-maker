"""
CrewAI Tasks for Video Generation Pipeline
Defines the specific tasks that agents will perform
"""

from crewai import Task
from typing import Dict, Any, Optional
import logging
import json


class VideoTasks:
    """Container for all video generation tasks"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def create_summary_generation_task(self, agent, user_prompt: str, reference_url: Optional[str] = None) -> Task:
        """Create the video summary generation task"""
        
        return Task(
            description=f"""
            Analyze the user input and generate a comprehensive video summary.
            
            USER INPUT:
            - Text Prompt: {user_prompt}
            - Reference Image URL: {reference_url or "None provided"}
            
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
            
            OUTPUT FORMAT:
            Return a JSON object with the following structure:
            {{
                "visual_summary": "Detailed scene descriptions with camera work and timing",
                "voiceover": "Natural spoken dialogue script",
                "voiceover_scenes": ["Array of voiceover segments"],
                "video_config": {{
                    "total_duration": "sum of scene durations",
                    "aspect_ratio": "detected ratio (9:16, 16:9, 1:1)",
                    "video_style": "detected style"
                }}
            }}
            
            IMPORTANT GUIDELINES:
            - Maintain visual and contextual consistency across scenes
            - Include directorial elements (camera angles, lighting, expressions)
            - Keep scenes technically feasible for AI generation
            - Avoid meta-content like credits or title cards
            - Use natural, conversational voiceover language
            - Include duration estimates for each scene
            """,
            
            agent=agent,
            expected_output="JSON object containing visual summary, voiceover, and video configuration"
        )
    
    def create_script_generation_task(self, agent, summary_result: Dict[str, Any], 
                                    user_prompt: str, reference_url: Optional[str] = None) -> Task:
        """Create the video script generation task"""
        
        return Task(
            description=f"""
            Transform the video summary into a detailed, production-ready JSON script.
            
            INPUT DATA:
            - Original User Prompt: {user_prompt}
            - Reference Image URL: {reference_url or "None provided"}
            - Video Summary: {summary_result.get('visual_summary', 'N/A')}
            - Voiceover Data: {summary_result.get('voiceover_scenes', [])}
            - Video Config: {summary_result.get('video_config', {})}
            
            TASK REQUIREMENTS:
            1. Convert the visual summary into structured scene objects
            2. Match voiceover text to appropriate scenes based on content and timing
            3. Create detailed image prompts optimized for AI image generation
            4. Generate action prompts for motion and animation
            5. Determine scene composition (single_shot vs montage)
            6. Set "has_reference" flag only for scenes featuring the reference image
            7. Ensure accurate timestamps and logical scene flow
            8. Validate production feasibility
            Aspect ratios: 9:16 (vertical/mobile - DEFAULT), 16:9 (landscape), 1:1 (square/social)
DEFAULT ASPECT RATIO: Use 9:16 unless specifically requested otherwise or content clearly requires landscape
MANDATORY: If the ascept ratio is specified by user, give it a higher preference.
MANDATORY: Every video MUST have voiceover ("needs_voiceover": true)
MANDATORY: Every scene/shot MUST have voiceover_text (no empty strings)
MANDATORY: For multi-shot montages, every individual shot MUST have a simple 'action_prompt' to create motion.
MANDATORY: Sum of the duration of all scenes should be equal to the total duration of the video.
MANDATORY: The duration of each scene should be between 0 and 10 seconds.
MANDATORY: For multi_shot_montage scenes, duration of each shot is the length of total shot to be taken. Sum of the duration of all shots should be equal to the duration of the scene.
MANDATORY: For multi_shot_montage scenes, each individual shot must be 2.0-3.0 seconds minimum (no shorter shots to prevent video getting stuck)
MANDATORY: Do NOT include text overlays, logos, or URLs in the image prompts. These are added in post-production. Prompts must ONLY describe visual scenes.
MANDATORY: Script Title should be short and catchy less than 50 characters and greater than 20 characters.
MANDATORY: StoryLine should be less than 200 characters and greater than 100 characters.
MANDATORY: Generate a caption and call-to-action for the ad video. It should be short ( less than 75 characters), crisp and to the point.
MANDATORY: Make sure the camera angels are dynmaic across scenes, and the object in context is in movement

CRITICAL MULTI-SHOT MONTAGE REQUIREMENTS:

MANDATORY: For multi_shot_montage scenes, the voiceover_text applies to the ENTIRE montage duration
MANDATORY: Individual shots within montages do NOT have separate voiceover - only the main scene has voiceover
MANDATORY: Each shot in montage must be 2.0-3.0 seconds minimum for proper cutting
MANDATORY: Voiceover pacing must match the total montage duration, not individual shots
MANDATORY: Sum of all shot durations in a montage must equal the scene duration exactly

MONTAGE STRUCTURE EXAMPLE:
Scene 4 (6 seconds total): "This is how you achieve the results you've been chasing." (11 words for 6s)
- Shot 1 (2s): Close-up of product
- Shot 2 (2s): Person using product  
- Shot 3 (2s): Results transformation
- Voiceover covers ALL 6 seconds continuously, not individual 2-second segments

CRITICAL ORGANIC VOICEOVER REQUIREMENTS:

MANDATORY: Write like a REAL PERSON talking, not advertising copy
MANDATORY: Use casual, authentic language with natural speech patterns
MANDATORY: Include hesitations, emphasis, and conversational rhythm
MANDATORY: Avoid perfect grammar - use how people actually speak
MANDATORY: Include personal touches and relatable observations
MANDATORY: Sound like someone genuinely excited about sharing something cool

            
            OUTPUT FORMAT:
            Return a comprehensive JSON script with this structure:
            {{
                "video_config": {{
                    "total_duration": number,
                    "aspect_ratio": string,
                    "style": string,
                    "pacing": string,
                    "content_type": string,
                    "needs_voiceover": boolean,
                    "needs_background_music": boolean,
                    "voice_characteristics": string,
                    "music_description": string
                }},
                "scenes": [
                    {{
                        "scene_id": string,
                        "timestamp": string,
                        "duration": number,
                        "scene_composition": string,
                        "voiceover_text": string,
                        "has_reference": boolean,
                        "image_prompt": string,
                        "action_prompt": string,
                        "shots": [] (for montage scenes only)
                    }}
                ]
            }}
            
            TECHNICAL REQUIREMENTS:
            - For single_shot scenes: Include image_prompt and action_prompt at scene level
            - For montage scenes: Include shots array with individual shot details
            - Ensure frame durations sum to scene duration for montages
            - Create specific, detailed prompts for AI generation systems
            - Maintain narrative coherence and emotional flow
            """,
            
            agent=agent,
            expected_output="Structured JSON script optimized for automated video production"
        )
    
    def create_voice_selection_task(self, agent, script_result: Dict[str, Any], 
                                   user_prompt: str, reference_url: Optional[str] = None) -> Task:
        """Create the voice selection task"""
        
        return Task(
            description=f"""
            Select the most appropriate voice actor from ElevenLabs voice library for this video.
            
            INPUT DATA:
            - User Prompt: {user_prompt}
            - Video Script: {json.dumps(script_result, indent=2)}
            
            TASK:
            Analyze the video content and select a suitable voice from ElevenLabs library.
            
            AVAILABLE VOICES (ElevenLabs IDs):
            - Rachel (21m00Tcm4TlvDq8ikWAM) - Young female, American, friendly
            - Domi (AZnzlk1XvdvUeBnXmlld) - Young female, American, energetic  
            - Bella (EXAVITQu4vr4xnSDxMaL) - Young female, British, warm
            - Antoni (ErXwobaYiN019PkySvjV) - Young male, American, friendly
            - Josh (TxGEqnHWrfWFTfGW9XjX) - Young male, American, professional
            - Arnold (VR6AewLTigWG4xSOukaG) - Middle-aged male, American, authoritative
            - Adam (pNInz6obpgDQGcFmaJgB) - Young male, American, casual
            - Sam (yoZ06aMxZJJ28mfd3POQ) - Young male, American, enthusiastic
            
            SELECTION CRITERIA:
            - Content type (ad, educational, narrative, etc.)
            - Target audience (age, gender, cultural background)
            - Tone (professional, friendly, authoritative, warm)
            - Brand voice alignment
            
            OUTPUT FORMAT:
            Return ONLY a JSON object with this exact structure:
            {{
                "voice_selection": {{
                    "voice_id": "21m00Tcm4TlvDq8ikWAM",
                    "voice_name": "Rachel",
                    "voice_characteristics": {{
                        "age": "Young",
                        "gender": "Female", 
                        "accent": "American",
                        "tone": "Friendly",
                        "style": "Conversational"
                    }},
                    "selection_reasoning": "Brief explanation of choice",
                    "content_match_score": 8,
                    "target_audience_alignment": "General audience"
                }},
                "voiceover_optimization": {{
                    "recommended_pacing": "medium",
                    "emotional_emphasis": "Natural tone",
                    "technical_notes": "Standard settings"
                }}
            }}
            
            IMPORTANT: Return ONLY the JSON object, no additional text or explanations.
            """,
            
            agent=agent,
            expected_output="JSON object with voice selection and optimization data"
        )
    
    def create_media_generation_task(self, agent, script_result: Dict[str, Any], 
                                   voice_result: Dict[str, Any], user_prompt: str, 
                                   reference_url: Optional[str] = None) -> Task:
        """Create the media generation task"""
        
        return Task(
            description=f"""
            Generate all media assets for the video using fal.ai APIs and create the final video.
            
            INPUT DATA:
            - Original User Prompt: {user_prompt}
            - Reference Image URL: {reference_url or "None provided"}
            - Video Script: {json.dumps(script_result, indent=2)}
            - Voice Selection: {json.dumps(voice_result, indent=2)}
            
            TASK REQUIREMENTS:
            1. Generate all images using fal.ai Flux Pro models
            2. Create videos from images with motion and animation
            3. Generate voiceovers using ElevenLabs via fal.ai with the selected voice ID
            4. Create background music if needed
            5. Stitch all media together into final video using ffmpeg
            6. Save all intermediate files for debugging and review
            7. Handle errors gracefully and fail the entire workflow if critical errors occur


            
            TECHNICAL SPECIFICATIONS:
            - Use fal.ai for all image and video generation
            - Use fal.ai's ElevenLabs endpoint for voice generation
            - Use ffmpeg for video stitching and editing
            - Preserve all intermediate files in organized folder structure
            - Use the voice_id from voice_result for voiceover generation
            - Ensure all media assets are properly synchronized
            
            OUTPUT FORMAT:
            Return a JSON object with the following structure:
            {{
                "media_generation_status": "success/failed",
                "generated_assets": {{
                    "images": {{"scene_id": "file_path"}},
                    "videos": {{"scene_id": "file_path"}},
                    "audio": {{
                        "voiceovers": {{"scene_id": "file_path"}},
                        "background_music": "file_path"
                    }}
                }},
                "final_video": "path_to_final_video.mp4",
                "output_folder": "path_to_output_folder",
                "generation_summary": {{
                    "total_images": number,
                    "total_videos": number,
                    "total_voiceovers": number,
                    "final_duration": number,
                    "file_size": string
                }},
                "errors": [] (if any occurred)
            }}
            
            ERROR HANDLING:
            - If any critical media generation fails, return status "failed"
            - Log all errors with detailed information
            - Preserve any successfully generated assets
            - Provide clear error messages for debugging
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

Be intelligent about scene analysis - consider narrative structure, emotional flow, and production requirements. Return ONLY the JSON, no additional text.

            """,
            
            agent=agent,
            expected_output="JSON object containing media generation results and final video path"
        ) 