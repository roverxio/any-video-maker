"""
CrewAI Tasks for Video Generation Pipeline
Defines the specific tasks that agents will perform
"""

from crewai import Task
from typing import Dict, Any, Optional
import logging
import json
import re


class VideoTasks:
    """Container for all video generation tasks"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def create_summary_generation_task(self, agent, user_prompt: str, reference_url: Optional[str] = None) -> Task:
        """Create the video summary generation task"""
        
        return Task(
            description=f"""
            Analyze the user text prompt and reference image and generate a comprehensive video summary.
            
            USER INPUT:
            - Text Prompt: {user_prompt}
            - Reference Image URL: {reference_url or "None provided"}

            ## Rules
             * **Total video length should be 15-40 seconds across all scenes** *
            
            * **Total video length should be 15-40 seconds across all scenes** *

            * **Analyze both inputs together** to understand the true user intent and context. Combine the reference image with the instruction to infer the overall purpose and tone.

            * **Do NOT fabricate specific details** that would typically come from external sources (dates, prices, URLs, brand names not shown, specific statistics). Generic elements like countdown numbers or basic transitions are acceptable.

            * **NEVER add meta-content** such as credits, title cards, "subscribe" messages, copyright notices, production information, or any text overlays unless explicitly requested by the user. The video should contain only the actual content scenes.

            * **Include necessary visual elements** for a coherent video, even if not explicitly mentioned. For example, if creating a sunglasses ad, people wearing sunglasses should be included even if not specified.

            * **Keep scenes technically feasible** for AI video generation. Avoid overly complex transitions, multiple simultaneous actions, or intricate visual effects that would be difficult to render.

            * **CRITICAL: Explicitly mention the reference image (logo/product/brand element/person/location) in your scene descriptions** when it should appear. Use clear phrases like "the logo appears", "logo integrates into", "product is shown", "brand symbol visible", etc. This ensures proper detection in later processing steps.

            * **Structure the summary with 3-5 separate scenes** that flow logically from one to another. Each scene should focus on a different aspect or moment of the video.

            * **Include directorial elements** such as camera angles, lighting, and character expressions to create rich, descriptive visuals.

            * **Keep the video genre flexible** – it can be a product ad, social media post, tiktok video, brand video, tutorial, abstract animation, or single-speaker moment, depending on the combined input context.

            * **Final summary length**: ideally **5-9 scenes**, each with 1–2 sentences. Every scene should contain actual video content, not meta-elements.

            * **CRITICAL: Create SEPARATE scenes, not one long description**. Each scene is a distinct moment in the video with its own focus and duration.

            * **Write each scene as its own paragraph** that naturally incorporates visual elements, camera movements, and directorial details. Do NOT use bullet points, subsections, or formatted labels like "Visual:" or "Camera Angle:". Each scene should read as a cohesive description.

            
            * **Include scene type** for each scene in square brackets after the scene description
              - Standard scene: scene which do not have multiple shots
              - Montage scenes: scenes which have multiple shots and a nested shot structure
            
            
            * **Include duration estimates** for each scene in square brackets at the end, based on content complexity:
              - Standard scenes: [3-5s]
              - Montage scenes: [5-10s]



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

            Scene 1: Brief scene description incorporating visuals and camera work naturally. [Standard scene] [5s]

            Scene 2: Another flowing description without subsections or labels. [Montage scene] [7s]

            Scene 3: Yet another distinct scene with its own focus. [Standard scene] [3s]

            Scene 4: Final scene bringing closure to the video. [Montage scene][7s]

            IMPORTANT: Each scene must be labeled "Scene 1:", "Scene 2:", etc. and end with duration in brackets.
            
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
    
    def create_voiceover_generation_task(self, agent, summary_result: Dict[str, Any]) -> Task:
        """Create the voiceover generation task"""
        
        return Task(
            description=f"""
            Generate professional voiceover script from the visual summary with intelligent style detection.
            
            INPUT DATA:
            - Visual Summary: {summary_result.get('visual_summary', 'N/A')}
            - Video Config: {json.dumps(summary_result.get('video_config', {}), indent=2)}
            
            ## AUTO-DETECT VOICEOVER STYLE
            
            Analyze the video content and select the appropriate style:
            - **Product showcase/advertisement** → Promotional style
            - **Educational/places/processes** → Documentary style  
            - **Characters/emotional journey** → Storytelling style
            - **Artistic/abstract content** → Minimal style
            - **General content** → Narrator style
            
            ## STYLE EXAMPLES
            
            **Promotional**: "Okay, check this out..." | "You're gonna love this part"
            **Documentary**: "You see, what's happening here is..." | "Now this... this is interesting."  
            **Storytelling**: "So there I was..." | "And then, out of nowhere..."
            **Minimal**: "Rain falls." | Single powerful phrases
            **Narrator**: "Whoa, look at that..." | "Okay, so here's what happens..."
            
            ## REQUIREMENTS
            
            **Natural Speech**: Write like talking to a friend, not reading ad copy. Use authentic conversational language.
            
            **Timing**: Extract scene durations from summary and match word count:
            - [3s] scenes = max 7-8 words
            - [5s] scenes = max 12-13 words  
            - [7s] scenes = max 17-18 words
            - [10s] scenes = max 25 words
            
            **Scene Matching**: Match voiceover content to each scene's visual elements and emotional tone.
            
            ## OUTPUT FORMAT
            Return ONLY this JSON structure:
            {{
                "voiceover_generation": {{
                    "detected_style": "promotional/documentary/storytelling/minimal/narrator",
                    "overall_tone": "description of tone",
                    "full_voiceover_text": "complete script with scene labels",
                    "scenes": [
                        {{
                            "scene_number": "Scene 1",
                            "voiceover_text": "natural spoken dialogue",
                            "word_count": number,
                            "estimated_duration": number,
                            "scene_duration": number,
                            "timing_match": "good/tight/needs_adjustment"
                        }}
                    ],
                    "style_justification": "why this style was chosen",
                    "technical_notes": "pacing and emphasis recommendations"
                }}
            }}
            
            CRITICAL: Generate voiceover for EVERY scene. Ensure word count matches duration. Use natural language. Return ONLY valid JSON.
            """,
            
            agent=agent,
            expected_output="JSON object containing voiceover script with style detection and timing optimization"
        )
    
    def create_script_generation_task(self, agent, summary_result: Dict[str, Any], reference_url: Optional[str] = None) -> Task:
        """Create the video script generation task"""
        
        return Task(
            description=f"""
            INPUT DATA:
            - Video Summary: {summary_result.get('visual_summary', 'N/A')}
            - Reference Image URL: {reference_url or "None provided"}
            - Voiceover Data: {json.dumps(summary_result.get('voiceover_scenes', []), indent=2)}
            - Video Config: {json.dumps(summary_result.get('video_config', {}), indent=2)}

## CRITICAL INSTRUCTIONS (READ FIRST)

### 1. REFERENCE IMAGE DETECTION
The reference image can be ANYTHING: logo, product, person, location, artwork, object, brand element, etc.

**LOOK FOR THESE PHRASES in the summary:**
- "logo appears", "logo integrates", "logo visible"
- "product shown", "product displayed", "featuring the product"
- "brand symbol", "brand element", "company logo"
- "the [reference] appears", "shows the [reference]"
- Any direct mention of the reference image being visible

**CRITICAL**: Evaluate EACH shot independently. Just because one shot has the reference doesn't mean all shots do.

### 2. IMAGE PROMPT REQUIREMENTS

**COMPLETE FRAME DESCRIPTION RULE**: Your image prompt MUST include ALL visual elements mentioned in that part of the scene.

**CHECKLIST for each image prompt:**
- ✓ Is the main subject included?
- ✓ Is the background/setting described?
- ✓ Are ALL visual elements from the summary included?
- ✓ If reference is mentioned, is it in the prompt?
- ✓ Is it purely static (no motion words)?

**COMMON MISTAKES TO AVOID:**
❌ Missing the reference when it's mentioned in the summary
❌ Describing only part of the visual (e.g., just sound waves without the person)
❌ Using motion words in image prompts
❌ Not including the logo/product/person/location/subject/reference when the summary says it appears

### 3. SCENE COMPOSITION RULES

SCENE COMPOSITION EXAMPLES:
- Multiple distinct visual elements in one scene = MONTAGE
- Camera movement revealing new subjects = MONTAGE  
- Single continuous action = SINGLE_SHOT
- Scene mentioned as [Montage scenes] = MONTAGE

REFERENCE IMAGE DETECTION:
- Each shot independently evaluated
- Only mark has_reference: true if that specific shot mentions the reference image
- Don't propagate has_reference across all shots in a scene

REQUIRED JSON STRUCTURE:
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
      "scene_composition": "single_shot",
      "voiceover_text": string,
      "image_prompt": string,
      "action_prompt": string,
      "has_reference": boolean
    }},
    {{
      "scene_id": string,
      "timestamp": string,
      "duration": number,
      "scene_composition": "montage",
      "voiceover_text": string,
      "shots": [
        {{
          "shot_id": string,
          "frame_duration": number,
          "has_reference": boolean,
          "image_prompt": string,
          "action_prompt": string
        }}
      ]
    }}
  ]
}}

CRITICAL JSON REQUIREMENTS:
- Return ONLY valid JSON - no trailing commas anywhere
- No extra text before or after the JSON object
- Ensure all quotes, brackets, and braces are properly matched
- Validate JSON syntax before returning

## STEP-BY-STEP INSTRUCTIONS:

1. **Extract scenes** from the visual summary with descriptions and durations

2. **For EACH scene part, check for reference mentions:**
   - Does this specific part mention the logo/product/reference appearing?
   - If yes → has_reference: true for that shot
   - If no → has_reference: false

3. **Create COMPLETE image prompts:**
   - Start with the main subject
   - Add the setting/background
   - Include ALL visual elements mentioned
   - If reference mentioned, describe it clearly
   - Use ONLY static descriptions

4. **Match voiceover** text to scenes based on content and timing

5. **Determine scene composition:**
   - "single_shot": ONE continuous action/view with no transitions
   - "montage": Multiple shots, camera reveals, or transitions

6. For SINGLE_SHOT scenes: Include "image_prompt", "action_prompt", and "has_reference" at the scene level, do NOT include "shots" array

7. For MONTAGE scenes: Include "shots" array with individual shots, each having shot_id (starting from "shot_1" for each scene), has_reference, frame_duration, image_prompt, and action_prompt. The sum of all frame_durations MUST equal the scene duration. Do NOT include scene-level image_prompt/action_prompt

8. For MONTAGE scenes: Break down the scene description into distinct visual moments, each becoming a separate shot

9. For MONTAGE scenes: The ideal duration for each shot is 1-3 seconds (max 3 seconds per shot!)

10. Voiceover text always stays at the scene level, never at the shot level

11. **IMAGE PROMPTS - Create STATIC image descriptions:**
    - Describe ONLY what's visible in a single frame/moment
    - Include specific details: age, clothing, setting, lighting, camera angle
    - Example: "Medium shot of a young woman in her 20s wearing casual blue jeans and white t-shirt, sitting in a modern minimalist office, soft natural lighting from window, warm color tones"
    - NEVER include motion words like "dancing", "moving", "transitioning", "appearing"
    - NEVER include temporal descriptions like "begins to", "starts", "then"

12. **ACTION PROMPTS - Describe motion/animation:**
    - CRITICAL: Can ONLY animate what's already visible in the static image
    - Good: "The woman dances energetically" (if woman is in the image)
    - Bad: "Logo elements fade in around her" (if logo wasn't in the original image)
    - Include camera movements: "camera slowly pulls back"
    - Focus on subject motion and camera movement only

13. Set video configuration parameters based on content type

14. Ensure timestamps are accurate and scenes flow logically

15. Make voice characteristics and music descriptions detailed and contextually appropriate

## SELF-CHECK BEFORE RETURNING:
- [ ] Did I check EACH shot for reference mentions?
- [ ] Do ALL my image prompts include EVERY visual element from the summary?
- [ ] Are my image prompts 100% static (no motion words)?
- [ ] If the summary mentions logo/product appearing, is has_reference: true?
- [ ] Is my JSON valid with no trailing commas?

## EXAMPLE ANALYSIS:
- "Close-up of headphones, camera pulls back revealing people" = MONTAGE (2+ distinct views)
- "Person walks across a room" = SINGLE_SHOT (one continuous action)
- "Logo appears in the scene" = has_reference: true
- "The logo subtly integrates into the sound waves" = has_reference: true

## IMAGE PROMPT EXAMPLES:

**GOOD COMPLETE PROMPT** (includes all elements):
✅ "Wide shot of a young woman in her early 20s wearing bright yellow athletic wear, standing in a modern dance studio with wooden floors. Colorful sound wave patterns flow around her in vibrant blues and purples. The company logo is integrated into the flowing patterns in the upper right."

**BAD INCOMPLETE PROMPT** (missing elements):
❌ "Close-up of sound waves in vibrant colors"
(Missing: the person, the setting, the logo if mentioned)

**ACTION PROMPT EXAMPLES:**
❌ BAD: "She begins dancing energetically, the camera slowly pulls back as animated logo elements fade in around her"
✅ GOOD: "The woman dances energetically as the camera slowly pulls back"

CRITICAL VOICEOVER REQUIREMENT:
🚨 MANDATORY: Every scene MUST have voiceover_text. Never use empty strings ("") for voiceover_text.
🚨 MANDATORY: If a scene has no voiceover in the summary, create appropriate voiceover text that matches the scene.
🚨 MANDATORY: Each scene's voiceover should be natural and conversational, matching the duration.

VOICEOVER GUIDELINES:
- Scene 1-2: Product introduction and benefits
- Scene 3-4: Emotional connection or call-to-action
- Keep each voiceover concise and impactful
- Match word count to scene duration (2-3 words per second)

CRITICAL: Return ONLY valid JSON - no trailing commas, no extra text, no explanations. The response must be parseable JSON.
            """,
            
            agent=agent,
            expected_output="Structured JSON script optimized for automated video production",
            callback=lambda result: self._validate_and_fix_script(
                str(result), 
                summary_result.get('visual_summary', ''), 
                reference_url
            )
        )
    
    def _validate_and_fix_script(self, script_result: str, visual_summary: str, reference_url: str = None) -> str:
        """
        Post-process script to fix reference detection and image prompt issues
        """
        try:
            # Parse the script JSON
            script_data = json.loads(script_result)
            
            self.logger.info("Starting post-processing validation of generated script...")
            
            # Extract scene descriptions from visual summary
            scene_patterns = re.findall(r'Scene \d+:(.*?)(?=Scene \d+:|$)', visual_summary, re.DOTALL)
            
            # Reference detection keywords
            reference_keywords = [
                'logo appears', 'logo integrates', 'logo visible', 'logo subtly',
                'product shown', 'product displayed', 'featuring the product',
                'brand symbol', 'brand element', 'company logo',
                'reference appears', 'reference shown', 'reference image',
                'appears in', 'integrates into', 'becomes part of', 'visible in'
            ]
            
            fixed_count = 0
            
            # Process each scene
            for i, scene in enumerate(script_data.get('scenes', [])):
                scene_num = i + 1
                scene_description = scene_patterns[i] if i < len(scene_patterns) else ""
                
                self.logger.debug(f"Processing Scene {scene_num}: {scene_description[:100]}...")
                
                # Check if reference is mentioned in this scene description
                has_reference_in_description = any(keyword.lower() in scene_description.lower() for keyword in reference_keywords)
                
                if scene.get('scene_composition') == 'single_shot':
                    # Fix single shot scene
                    current_has_ref = scene.get('has_reference', False)
                    if has_reference_in_description and not current_has_ref:
                        scene['has_reference'] = True
                        self.logger.info(f"Fixed Scene {scene_num}: Set has_reference=True")
                        fixed_count += 1
                    
                    # Fix image prompt if reference mentioned but missing from prompt
                    if has_reference_in_description and reference_url:
                        image_prompt = scene.get('image_prompt', '')
                        if 'logo' not in image_prompt.lower() and 'brand' not in image_prompt.lower():
                            # Add reference to image prompt
                            scene['image_prompt'] = f"{image_prompt.rstrip('.')}. The company logo is visible in the scene."
                            self.logger.info(f"Fixed Scene {scene_num}: Added reference to image prompt")
                            fixed_count += 1
                
                elif scene.get('scene_composition') == 'montage':
                    # Fix montage shots
                    shots = scene.get('shots', [])
                    for j, shot in enumerate(shots):
                        shot_num = j + 1
                        
                        # For montage, we need to check which part of the scene this shot represents
                        # Simple heuristic: if reference mentioned in scene and this is the later shot, likely has reference
                        current_has_ref = shot.get('has_reference', False)
                        
                        if has_reference_in_description and not current_has_ref:
                            # Check if this shot should have the reference (heuristic: later shots more likely)
                            if j >= len(shots) // 2:  # Second half of shots more likely to have reference
                                shot['has_reference'] = True
                                self.logger.info(f"Fixed Scene {scene_num} Shot {shot_num}: Set has_reference=True")
                                fixed_count += 1
                        
                        # Fix image prompt if reference mentioned but missing from prompt
                        if has_reference_in_description and reference_url and shot.get('has_reference', False):
                            image_prompt = shot.get('image_prompt', '')
                            if 'logo' not in image_prompt.lower() and 'brand' not in image_prompt.lower():
                                # Add reference to image prompt
                                shot['image_prompt'] = f"{image_prompt.rstrip('.')}. The company logo is visible in the frame."
                                self.logger.info(f"Fixed Scene {scene_num} Shot {shot_num}: Added reference to image prompt")
                                fixed_count += 1
            
            self.logger.info(f"Post-processing completed. Fixed {fixed_count} issues.")
            return json.dumps(script_data, indent=2)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse script JSON for validation: {e}")
            return script_result  # Return original if parsing fails
        except Exception as e:
            self.logger.error(f"Error during script validation: {e}")
            return script_result  # Return original if validation fails
    
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
    
 