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
            Analyze the user text prompt and reference image and generate a comprehensive video summary.
            
            USER INPUT:
            - Text Prompt: {user_prompt}
            - Reference Image URL: {reference_url or "None provided"}

            ## Rules

            * **Analyze both inputs together** to understand the true user intent and context. Combine the reference image with the instruction to infer the overall purpose and tone.

            * **Do NOT fabricate specific details** that would typically come from external sources (dates, prices, URLs, brand names not shown, specific statistics). Generic elements like countdown numbers or basic transitions are acceptable.

            * **NEVER add meta-content** such as credits, title cards, "subscribe" messages, copyright notices, production information, or any text overlays unless explicitly requested by the user. The video should contain only the actual content scenes.

            * **Include necessary visual elements** for a coherent video, even if not explicitly mentioned. For example, if creating a sunglasses ad, people wearing sunglasses should be included even if not specified.

            * **Keep scenes technically feasible** for AI video generation. Avoid overly complex transitions, multiple simultaneous actions, or intricate visual effects that would be difficult to render.

            * **CRITICAL: Explicitly mention the reference image (logo/product/brand element/person/location) in your scene descriptions** when it should appear. Use clear phrases like "the logo appears", "logo integrates into", "product is shown", "brand symbol visible", etc. This ensures proper detection in later processing steps.

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

            (Continue for all scenes)
            
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
            
            TASK REQUIREMENTS:
            
            ## Style Auto-Detection
            Analyze the video summary to determine the most appropriate voiceover style:
            - Product showcase/advertisement → **Promotional style** (excited but genuine)
            - Educational/places/processes → **Documentary style** (natural wisdom)
            - Characters/emotional journey → **Storytelling style** (intimate, personal)
            - Artistic/abstract content → **Minimal style** (sparse, poetic)
            - General content → **Narrator style** (friendly, conversational)
            
            ## Voiceover Style Guidelines
            
            **NARRATOR STYLE** - General friendly narration:
            - "Whoa, look at that..." or "Okay, so here's what happens..."
            - Conversational, warm, engaging
            
            **DOCUMENTARY STYLE** - Natural documentary narration:
            - "You see, what's happening here is..."
            - "Now this... this is interesting."
            - Wisdom and gentle insights
            
            **PROMOTIONAL STYLE** - Genuine enthusiasm:
            - "Okay, check this out..."
            - "You're gonna love this part"
            - Real excitement, not fake hype
            
            **STORYTELLING STYLE** - Intimate narrative:
            - "So there I was..."
            - "And then, out of nowhere..."
            - Personal and emotional
            
            **MINIMAL STYLE** - Sparse and poetic:
            - Single powerful phrases
            - "Rain falls."
            - Let silence speak
            
            ## Critical Requirements
            
            **NATURAL SPEECH PATTERNS:**
            - Write like you're talking to a friend, not reading ad copy.
            - Use casual, authentic language with natural speech patterns
            - Include hesitations, emphasis, and conversational rhythm
            - Avoid perfect grammar - use how people actually speak
            - Include personal touches and relatable observations
            - Sound like someone genuinely excited about sharing something cool

            
            **TIMING CONSTRAINTS:**
            Extract scene durations from the visual summary and match word count:
            - For [3s] scenes = max 7-8 words
            - For [5s] scenes = max 12-13 words  
            - For [7s] scenes = max 17-18 words
            - For [10s] scenes = max 25 words
            
            **SCENE MATCHING:**
            - Match voiceover content to each scene's visual elements
            - Ensure emotional tone aligns with scene mood
            - Consider pacing and natural speech rhythm
            
            OUTPUT FORMAT:
            Return a JSON object with the following structure:
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
            
            IMPORTANT:
            - Generate voiceover for EVERY scene identified in the visual summary
            - Ensure word count matches scene duration constraints
            - Use natural, conversational language that sounds authentic
            - Avoid marketing jargon or overly polished copy
            - Make each scene's voiceover complement its visual content
            - Return ONLY the JSON object, no additional text
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

SCENE COMPOSITION EXAMPLES FROM YOUR SUMMARY:
Scene 1: "close-up of headphones" + "camera pulls back revealing people" + "sound waves emanate" + "logo integrates" = MONTAGE (4 distinct visual elements)
- Shot 1: Close-up of headphones
- Shot 2: Pull back to reveal people
- Shot 3: Sound waves visual effect
- Shot 4: Logo integration

REFERENCE IMAGE DETECTION EXAMPLES:
- If reference is a logo: "The company logo subtly integrates into the sound waves" → has_reference: true
- If reference is a person: "Sarah walks into frame" (where Sarah is the reference person) → has_reference: true
- If reference is a location: "The Golden Gate Bridge spans across the bay" (where bridge is reference) → has_reference: true
- "people wearing headphones" → has_reference: false (unless the headphones/logo are explicitly mentioned as appearing)

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

INSTRUCTIONS:
1. Analyze the visual summary to extract individual scenes with their descriptions and durations
2. Match voiceover text to appropriate scenes based on content and timing
3. Image prompts should feature detailed descriptions (eg: "a young woman wearing a dress..." instead of "a person...")
4. REFERENCE IMAGE DETECTION - Only set has_reference=true when the summary EXPLICITLY mentions the reference image:
   - The reference could be ANYTHING: logo, product, person, location, artwork, object, etc.
   - Look for direct mentions of the reference appearing/being shown in the scene
   - DO NOT assume the reference is shown just because related content appears
5. Set "has_reference" to true ONLY for scenes/shots where the reference image is EXPLICITLY mentioned in the summary text
6. Incorporate the reference image description into image prompts when has_reference is true
7. SCENE COMPOSITION ANALYSIS:
   - "single_shot": ONLY use when the scene is ONE continuous action/view with no transitions
   - "montage": Use when scene contains:
     * Multiple distinct visual elements or shots
     * Camera movements that reveal new subjects (e.g. "pulls back to reveal")
     * Transitions between different views or subjects
     * Multiple actions happening in sequence
     * Any mention of "transitions to", "shifts to", "cuts to"
8. For SINGLE_SHOT scenes: Include "image_prompt", "action_prompt", and "has_reference" at the scene level, do NOT include "shots" array
9. For MONTAGE scenes: Include "shots" array with individual shots, each having shot_id, has_reference, frame_duration, image_prompt, and action_prompt. The sum of all frame_durations MUST equal the scene duration. Do NOT include scene-level image_prompt/action_prompt
10. For MONTAGE scenes: Break down the scene description into distinct visual moments, each becoming a separate shot
11. For MONTAGE scenes: The ideal duration for each shot is 1-3 seconds (max 3 seconds per shot!)
12. Voiceover text always stays at the scene level, never at the shot level
13. IMAGE PROMPTS - Create STATIC image descriptions for AI generation:
    - Describe ONLY what's visible in a single frame/moment
    - Include specific details: age, clothing, setting, lighting, camera angle
    - Example: "Medium shot of a young woman in her 20s wearing casual blue jeans and white t-shirt, sitting in a modern minimalist office, soft natural lighting from window, warm color tones"
    - NEVER include motion words like "dancing", "moving", "transitioning", "appearing"
    - NEVER include temporal descriptions like "begins to", "starts", "then"
14. ACTION PROMPTS - Describe motion/animation for EXISTING elements in the image:
    - CRITICAL: Can ONLY animate what's already visible in the static image
    - Good: "The woman dances energetically" (if woman is in the image)
    - Bad: "Logo elements fade in around her" (if logo wasn't in the original image)
    - Include camera movements: "camera slowly pulls back"
    - Focus on subject motion and camera movement only
15. Set video configuration parameters based on the content type and user request
16. Ensure timestamps are accurate and scenes flow logically
17. Make voice characteristics and music descriptions detailed and contextually appropriate

EXAMPLE ANALYSIS:
- "Close-up of headphones, camera pulls back revealing people" = MONTAGE (2+ distinct views)
- "Person walks across a room" = SINGLE_SHOT (one continuous action)
- "Logo appears in the scene" = has_reference: true

IMAGE PROMPT EXAMPLES (static descriptions only):
❌ BAD: "A person dances with energy as the logo flashes"
✅ GOOD: "Wide shot of a young woman in her early 20s wearing bright yellow athletic wear and white sneakers, standing in a modern dance studio with wooden floors and mirrors, dramatic side lighting creating shadows"

ACTION PROMPT EXAMPLES (motion descriptions):
❌ BAD: "She begins dancing energetically, the camera slowly pulls back as animated logo elements fade in around her"
✅ GOOD: "The woman dances energetically as the camera slowly pulls back"

Be intelligent about scene analysis - consider narrative structure, emotional flow, and production requirements. 

CRITICAL: Return ONLY valid JSON - no trailing commas, no extra text, no explanations. The response must be parseable JSON.
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
    
 